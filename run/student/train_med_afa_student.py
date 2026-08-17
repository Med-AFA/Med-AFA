import argparse
import hashlib
import itertools
import json
import random
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchmetrics import Accuracy
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "datasets").exists() and (parent / "run").exists():
            return parent
    raise FileNotFoundError(f"Could not find project root from {start}.")


REPO_ROOT = find_project_root(SCRIPT_DIR)
RUN_ROOT = REPO_ROOT / "run"
STUDENT_ROOT = RUN_ROOT / "student"
DATASETS_ROOT = REPO_ROOT / "datasets"
RESULTS_ROOT = REPO_ROOT / "results"
MEDAFA_ROOT = REPO_ROOT
for _path in (STUDENT_ROOT, RUN_ROOT, DATASETS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from acquisition_model import CMIEstimator, DoubleHeadOracleQEstimator, MaskingPretrainer
from acquisition_model.utils import MaskLayerGrouped, get_entropy, get_mlp_network, ind_to_onehot

from action_constraints import (
    build_prerequisite_matrix,
    legal_action_indices_from_selected,
    load_action_feature_matrix,
    mask_illegal_action_logits,
    validate_action_sequence,
)
from teacher_state_utils import (
    build_state_vector,
    compute_catboost_soft_teacher,
    load_dataset_arrays,
    load_teacher_artifacts,
    predict_catboost_state,
)
from full_path_planner import FullPathPlannerConfig


SCHEDULE_PRESET_RATIOS = {
    "fixed_20_45_35": (0.20, 0.45),
    "fixed_25_40_35": (0.25, 0.40),
    "fixed_25_50_25": (0.25, 0.50),
    "fixed_30_40_30": (0.30, 0.40),
    "fixed_20_60_20": (0.20, 0.60),
}

INFERENCE_SCHEDULE_PRESET_RATIOS = {
    "same_as_training": None,
    **SCHEDULE_PRESET_RATIOS,
}


def resolve_split_path(default_split_path: Path, split_path: Optional[str], split_seed: Optional[int]) -> Path:
    if split_path:
        return Path(split_path).resolve()
    if split_seed is not None:
        return Path(default_split_path).with_name(f"split_seed{int(split_seed)}.json").resolve()
    return Path(default_split_path).resolve()


@dataclass
class TrainConfig:
    lr: float
    eps: float
    eps_decay: float
    patience: int
    hidden: int
    dropout: float
    cmi_scaling: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Med-AFA student med_afa_student: fixed Med-AFA with "
            "validation-constraint-selected predictor architecture."
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset key used only for output folders.")
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Use the default split directory with split_seed{N}.json. Ignored when --split_path is set.",
    )
    parser.add_argument("--dataset_csv", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--actions_path", type=str, required=True)

    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size_train", type=int, default=128)
    parser.add_argument("--batch_size_eval", type=int, default=1024)
    parser.add_argument("--normalize_mode", choices=["center", "zscore"], default="zscore")

    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--train_epochs", type=int, default=250)
    parser.add_argument("--max_features_train", type=int, default=None)
    parser.add_argument("--max_eval_features", type=int, default=None)
    parser.add_argument("--eps_steps", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    parser.add_argument("--use_class_weight", action="store_true")

    parser.add_argument("--do_grid_search", action="store_true")
    parser.add_argument("--grid_lr", type=str, default="0.0005,0.001,0.002")
    parser.add_argument("--grid_eps", type=str, default="0.05,0.1")
    parser.add_argument("--grid_eps_decay", type=str, default="0.2,0.5")
    parser.add_argument("--grid_patience", type=str, default="5")
    parser.add_argument("--grid_hidden", type=str, default="128")
    parser.add_argument("--grid_dropout", type=str, default="0.3")
    parser.add_argument("--grid_cmi_scaling", type=str, default="positive")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--eps_decay", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--cmi_scaling", choices=["none", "positive", "bounded"], default="positive")

    parser.add_argument(
        "--teacher_run_dir",
        type=str,
        required=True,
        help="Teacher run directory containing ckpts/teacher_best.pt.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default="", help="Optional explicit teacher_best.pt path.")
    parser.add_argument(
        "--q_schedule",
        choices=["three_phase_rerank"],
        default="three_phase_rerank",
        help="Hybrid target schedule. three_phase_rerank uses one-step prefix, proposal-rerank middle, one-step suffix.",
    )
    parser.add_argument(
        "--proposal_top_k",
        type=int,
        default=3,
        help="Number of one-step top candidate actions reranked by the full-path Q head.",
    )
    parser.add_argument(
        "--full_path_head_loss_weight",
        type=float,
        default=0.5,
        help="Loss weight for the full-path Q head in the double-head value network.",
    )
    parser.add_argument(
        "--enable_intervention_aux",
        action="store_true",
        help="Enable training-time intervention auxiliary CE. Off by default for med_afa_student.",
    )
    parser.add_argument(
        "--disable_intervention_aux",
        action="store_true",
        help="Disable training-time intervention auxiliary CE.",
    )
    parser.add_argument(
        "--intervention_aux_weight",
        type=float,
        default=0.10,
        help="Weight for optional training-time predictor auxiliary CE.",
    )
    parser.add_argument(
        "--intervention_aux_only_changed_actions",
        action="store_true",
        help="Apply auxiliary CE only when one-step top1 and full-path top1 differ.",
    )
    parser.add_argument(
        "--intervention_aux_mode",
        choices=["one_full_ce", "oracle_positive_full_only"],
        default="oracle_positive_full_only",
        help="Predictor auxiliary mode. med_afa_student keeps this off unless --enable_intervention_aux is set.",
    )
    parser.add_argument(
        "--intervention_aux_oracle_margin",
        type=float,
        default=0.0,
        help="Minimum oracle full-path target advantage required for optional full-action CE.",
    )
    parser.add_argument(
        "--enable_posthoc_predictor_adapt",
        dest="disable_posthoc_predictor_adapt",
        action="store_false",
        help="Enable optional post-hoc predictor adaptation. Off by default for med_afa_student.",
    )
    parser.add_argument(
        "--disable_posthoc_predictor_adapt",
        dest="disable_posthoc_predictor_adapt",
        action="store_true",
        help="Disable post-hoc predictor adaptation on final-policy states.",
    )
    parser.set_defaults(disable_posthoc_predictor_adapt=True)
    parser.add_argument(
        "--posthoc_predictor_adapt_epochs",
        type=int,
        default=20,
        help="Number of optional predictor-only post-hoc adaptation epochs.",
    )
    parser.add_argument(
        "--posthoc_predictor_adapt_lr",
        type=float,
        default=1.0e-4,
        help="Learning rate for optional predictor-only post-hoc adaptation.",
    )
    parser.add_argument(
        "--predictor_arch_preset",
        choices=["auto_by_dataset", "base", "pred256_d00", "pred256_d01", "pred256_d02", "pred256_d03", "pred384_d01"],
        default="auto_by_dataset",
        help=(
            "Predictor architecture preset. auto_by_dataset uses pred256/dropout0.1 "
            "for multiclass datasets and base pred128/dropout0.3 for binary datasets."
        ),
    )
    parser.add_argument(
        "--arch_selection_mode",
        choices=["fixed", "val_constraint"],
        default="val_constraint",
        help="Predictor architecture selection mode. fixed uses one preset; val_constraint selects by validation constraint mean_acc@all.",
    )
    parser.add_argument(
        "--predictor_arch_grid",
        type=str,
        default="base,pred256_d01,pred256_d02,pred384_d01,pred256_d00",
        help="Comma-separated predictor architecture presets used when --arch_selection_mode val_constraint.",
    )
    parser.add_argument(
        "--predictor_hidden",
        type=int,
        default=-1,
        help="Hidden width for predictor MLP. <=0 means use --predictor_arch_preset.",
    )
    parser.add_argument(
        "--predictor_dropout",
        type=float,
        default=-1.0,
        help="Dropout for predictor MLP. <0 means use --predictor_arch_preset.",
    )
    parser.add_argument(
        "--value_hidden",
        type=int,
        default=-1,
        help="Hidden width for value-network MLP. <=0 means use aligned baseline 128.",
    )
    parser.add_argument(
        "--value_dropout",
        type=float,
        default=-1.0,
        help="Dropout for value-network MLP. <0 means use aligned baseline 0.3.",
    )
    parser.add_argument(
        "--disable_rerank_diagnostics",
        action="store_true",
        help="Disable student-outcome rerank diagnostics during test rollout.",
    )
    parser.add_argument(
        "--rerank_diag_max_states",
        type=int,
        default=2000,
        help="Maximum middle-phase states used for full-path rerank diagnostics. <=0 means no cap.",
    )
    parser.add_argument(
        "--rerank_diag_include_records",
        action="store_true",
        help="Include per-state rerank diagnostic records in the diagnostics JSON.",
    )
    parser.add_argument(
        "--proposal_recall_top_ks",
        type=str,
        default="1,3,5,7",
        help="Comma-separated one-step top-k cutoffs retained for proposal recall diagnostics.",
    )
    parser.add_argument(
        "--disable_intervention_sensitivity_diagnostics",
        action="store_true",
        help="Disable action-intervention sensitivity diagnostics.",
    )
    parser.add_argument(
        "--intervention_top_transition_k",
        type=int,
        default=30,
        help="Number of frequent action transitions retained in diagnostics.",
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=0.7,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha_gap_scale",
        type=float,
        default=0.35,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha_gap_floor",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha_agree_bonus",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--alpha_disagree_penalty",
        type=float,
        default=0.20,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--horizon_advantage_threshold",
        type=float,
        default=0.15,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--horizon_penalty",
        type=float,
        default=0.15,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--schedule_preset",
        choices=sorted(SCHEDULE_PRESET_RATIOS),
        default="fixed_25_40_35",
        help="Training and validation phase proportions: prefix/middle/suffix.",
    )
    parser.add_argument(
        "--one_step_prefix_steps",
        type=int,
        default=-1,
        help="Number of early acquisition steps trained with one-step Q. -1 uses --schedule_preset.",
    )
    parser.add_argument(
        "--full_path_middle_steps",
        type=int,
        default=-1,
        help="Number of middle acquisition steps trained with proposal plus full-path reranking. -1 uses --schedule_preset.",
    )
    parser.add_argument(
        "--inference_schedule_preset",
        choices=sorted(INFERENCE_SCHEDULE_PRESET_RATIOS),
        default="same_as_training",
        help="Test-rollout phase proportions. same_as_training reuses the training schedule.",
    )
    parser.add_argument(
        "--inference_one_step_prefix_steps",
        type=int,
        default=-1,
        help="Number of early one-step-Q acquisition steps at inference. -1 uses --inference_schedule_preset.",
    )
    parser.add_argument(
        "--inference_full_path_middle_steps",
        type=int,
        default=-1,
        help="Number of middle reranking acquisition steps at inference. -1 uses --inference_schedule_preset.",
    )
    parser.add_argument(
        "--paired_inference_schedule_presets",
        type=str,
        default="",
        help=(
            "Comma-separated test-rollout schedule presets evaluated on the same trained checkpoint. "
            "The first preset is the paired reference. Empty disables paired ablation."
        ),
    )
    parser.add_argument(
        "--paired_reference_schedule_preset",
        type=str,
        default="",
        help=(
            "Optional paired reference preset. If empty, the first item in "
            "--paired_inference_schedule_presets is used."
        ),
    )
    parser.add_argument(
        "--one_step_target_transform",
        choices=["raw", "clip_zero"],
        default="clip_zero",
        help="Transform one-step utility targets before fitting the acquisition model value network.",
    )
    parser.add_argument(
        "--full_path_target_transform",
        choices=["raw", "clip_zero"],
        default="raw",
        help="Transform full-path Q targets before fitting the acquisition model value network.",
    )
    parser.add_argument("--full_path_top_k", type=int, default=2, help="Number of top forced-first full paths used when reducing target scores.")
    parser.add_argument("--full_path_beam_width", type=int, default=2, help="Beam width for forced-first full-path Q search.")
    parser.add_argument("--full_path_max_depth", type=int, default=4, help="Maximum full-path lookahead depth including the forced first action.")
    parser.add_argument(
        "--full_path_score",
        choices=["mean_true_prob", "negative_ce", "ce_reduction", "mean_hard_acc", "mixed_score"],
        default="mean_true_prob",
        help="Path-level score used for oracle full-path Q target search.",
    )
    parser.add_argument("--full_path_temperature", type=float, default=0.2, help="Softmax temperature for top-k weighted full-path target reduction.")
    parser.add_argument(
        "--full_path_mixed_hard_acc_alpha",
        type=float,
        default=0.2,
        help="Hard-accuracy coefficient used only when --full_path_score mixed_score.",
    )
    parser.add_argument(
        "--full_path_q_reduce",
        choices=["best", "topk_weighted"],
        default="best",
        help="Reduce forced-first candidate paths into the scalar Q target.",
    )

    parser.add_argument("--results_dir", type=str, default=str(SCRIPT_DIR / "results_action_group"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--keep_checkpoints", action="store_true")
    parser.add_argument(
        "--candidate_cache_dir",
        type=str,
        default="",
        help="Directory for completed candidate-model cache. Empty uses results/candidate_cache/med_afa_student.",
    )
    parser.add_argument(
        "--disable_candidate_cache",
        action="store_true",
        help="Disable completed candidate-model cache/resume.",
    )
    parser.add_argument(
        "--force_retrain_candidate_cache",
        action="store_true",
        help="Ignore existing candidate cache entries and overwrite them after training.",
    )
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--save_diagnostics", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str, gpu_id: int) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda", gpu_id)
    if torch.cuda.is_available():
        return torch.device("cuda", gpu_id)
    return torch.device("cpu")


def load_split_json(split_path: Path) -> Dict:
    if not split_path.exists():
        raise FileNotFoundError(f"split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        split_info = json.load(f)
    if "indices" not in split_info:
        raise ValueError(f"invalid split json (missing indices): {split_path}")
    for k in ("train", "val", "test"):
        if k not in split_info["indices"]:
            raise ValueError(f"invalid split json (missing indices['{k}']): {split_path}")
    return split_info


def validate_indices(indices: Dict[str, List[int]], n_rows: int) -> Dict[str, np.ndarray]:
    output = {}
    for split_name in ("train", "val", "test"):
        arr = np.array(indices[split_name], dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError(f"indices['{split_name}'] must be 1D")
        if len(arr) == 0:
            raise ValueError(f"indices['{split_name}'] is empty")
        if arr.min() < 0 or arr.max() >= n_rows:
            raise ValueError(
                f"indices['{split_name}'] out of range for n_rows={n_rows}: "
                f"min={arr.min()}, max={arr.max()}"
            )
        output[split_name] = arr
    return output


def load_tabular_csv(dataset_csv: Path, label_col: str) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset csv not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    if label_col not in df.columns:
        raise ValueError(f"label column '{label_col}' not found in {dataset_csv}")

    feature_cols = [c for c in df.columns if c != label_col]
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[label_col].to_numpy()

    unique_vals = np.unique(y_raw)
    value_to_class = {val: idx for idx, val in enumerate(unique_vals.tolist())}
    y = np.array([value_to_class[val] for val in y_raw], dtype=np.int64)

    label_mapping = {str(k): int(v) for k, v in value_to_class.items()}
    return x, y, feature_cols, label_mapping


def load_action_groups(actions_path: Path, feature_names: List[str]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    if not actions_path.exists():
        raise FileNotFoundError(f"actions json not found: {actions_path}")
    payload = json.loads(actions_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError("actions.json must be a list or a dict with key 'actions'.")

    feature_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}
    used_features: set[int] = set()
    groups: list[list[int]] = []
    meta: list[dict[str, Any]] = []

    for i, action in enumerate(payload):
        action_id = str(action.get("action_id", f"action_{i + 1}"))
        action_name = str(action.get("name", action_id))
        raw_features = action.get("feature", action.get("features", []))
        if not isinstance(raw_features, list):
            raise ValueError(f"{action_id}: field 'feature' must be a list.")
        indices: list[int] = []
        missing: list[str] = []
        for raw_feature in raw_features:
            feature = str(raw_feature)
            if feature not in feature_to_idx:
                missing.append(feature)
                continue
            idx = int(feature_to_idx[feature])
            if idx in used_features:
                raise ValueError(
                    f"Feature '{feature}' appears in more than one action; acquisition model grouped mask "
                    "requires a disjoint action partition."
                )
            indices.append(idx)
            used_features.add(idx)
        if missing:
            raise ValueError(f"{action_id}: features not found in dataset columns: {missing}")
        if not indices:
            raise ValueError(f"{action_id}: zero resolved features.")
        groups.append(indices)
        meta.append(
            {
                "action_id": action_id,
                "name": action_name,
                "feature_indices": indices,
                "feature_names": [feature_names[j] for j in indices],
            }
        )

    for feature_idx, feature_name in enumerate(feature_names):
        if feature_idx in used_features:
            continue
        groups.append([feature_idx])
        meta.append(
            {
                "action_id": f"__ungrouped_feature_{feature_idx}",
                "name": f"Ungrouped feature: {feature_name}",
                "feature_indices": [feature_idx],
                "feature_names": [feature_name],
                "synthetic": True,
            }
        )

    group_matrix = torch.zeros((len(groups), len(feature_names)), dtype=torch.float32)
    for group_idx, indices in enumerate(groups):
        group_matrix[group_idx, indices] = 1.0
    return group_matrix, meta


def normalize_inputs(x: np.ndarray, train_idx: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_x = x[train_idx]
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    if mode == "center":
        std = np.ones_like(mean, dtype=np.float32)
        x_out = x - mean
    else:
        std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.clip(std, 1e-3, None)
        x_out = (x - mean) / std
    return x_out.astype(np.float32), mean, std


def parse_float_list(raw: str) -> List[float]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    return [float(v) for v in vals]


def parse_int_list(raw: str) -> List[int]:
    vals = [s.strip() for s in raw.split(",") if s.strip()]
    return [int(v) for v in vals]


def parse_str_list(raw: str) -> List[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def make_train_configs(args: argparse.Namespace) -> List[TrainConfig]:
    if not args.do_grid_search:
        return [
            TrainConfig(
                lr=args.lr,
                eps=args.eps,
                eps_decay=args.eps_decay,
                patience=args.patience,
                hidden=args.hidden,
                dropout=args.dropout,
                cmi_scaling=args.cmi_scaling,
            )
        ]

    grid_lr = parse_float_list(args.grid_lr)
    grid_eps = parse_float_list(args.grid_eps)
    grid_eps_decay = parse_float_list(args.grid_eps_decay)
    grid_patience = parse_int_list(args.grid_patience)
    grid_hidden = parse_int_list(args.grid_hidden)
    grid_dropout = parse_float_list(args.grid_dropout)
    grid_cmi_scaling = parse_str_list(args.grid_cmi_scaling)

    configs = []
    for lr, eps, eps_decay, patience, hidden, dropout, cmi_scaling in itertools.product(
        grid_lr, grid_eps, grid_eps_decay, grid_patience, grid_hidden, grid_dropout, grid_cmi_scaling
    ):
        configs.append(
            TrainConfig(
                lr=lr,
                eps=eps,
                eps_decay=eps_decay,
                patience=patience,
                hidden=hidden,
                dropout=dropout,
                cmi_scaling=cmi_scaling,
            )
        )
    return configs


def make_class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class DTypeSafeCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.detach().clone().float())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.weight is not None:
            weight = self.weight.to(device=logits.device, dtype=logits.dtype)
        return torch.nn.functional.cross_entropy(logits, target, weight=weight, reduction=self.reduction)


def resolve_a7_architecture_preset(
    preset: str,
    *,
    num_classes: int,
    requested_value_hidden: int,
    requested_value_dropout: float,
) -> Dict[str, Any]:
    preset = str(preset).strip()
    if preset == "base":
        preset_hidden = 128
        preset_dropout = 0.3
        preset_reason = "explicit_base"
    elif preset == "pred256_d00":
        preset_hidden = 256
        preset_dropout = 0.0
        preset_reason = "explicit_pred256_d00"
    elif preset == "pred256_d01":
        preset_hidden = 256
        preset_dropout = 0.1
        preset_reason = "explicit_pred256_d01"
    elif preset == "pred256_d02":
        preset_hidden = 256
        preset_dropout = 0.2
        preset_reason = "explicit_pred256_d02"
    elif preset == "pred256_d03":
        preset_hidden = 256
        preset_dropout = 0.3
        preset_reason = "explicit_pred256_d03"
    elif preset == "pred384_d01":
        preset_hidden = 384
        preset_dropout = 0.1
        preset_reason = "explicit_pred384_d01"
    elif preset == "auto_by_dataset":
        if int(num_classes) > 2:
            preset_hidden = 256
            preset_dropout = 0.1
            preset_reason = "auto_multiclass_predictor_capacity"
        else:
            preset_hidden = 128
            preset_dropout = 0.3
            preset_reason = "auto_binary_keep_aligned_baseline"
    else:
        raise ValueError(f"Unknown predictor architecture preset: {preset}")

    value_hidden = int(requested_value_hidden if int(requested_value_hidden) > 0 else 128)
    value_dropout = float(requested_value_dropout if float(requested_value_dropout) >= 0.0 else 0.3)
    return {
        "preset": str(preset),
        "preset_reason": preset_reason,
        "num_classes_rule_input": int(num_classes),
        "predictor_hidden": int(preset_hidden),
        "predictor_dropout": float(preset_dropout),
        "value_hidden": value_hidden,
        "value_dropout": value_dropout,
        "requested_value_hidden": int(requested_value_hidden),
        "requested_value_dropout": float(requested_value_dropout),
    }


def resolve_a7_fixed_architecture(args: argparse.Namespace, num_classes: int) -> Dict[str, Any]:
    base = resolve_a7_architecture_preset(
        args.predictor_arch_preset,
        num_classes=num_classes,
        requested_value_hidden=int(args.value_hidden),
        requested_value_dropout=float(args.value_dropout),
    )
    predictor_hidden = int(args.predictor_hidden if int(args.predictor_hidden) > 0 else base["predictor_hidden"])
    predictor_dropout = float(args.predictor_dropout if float(args.predictor_dropout) >= 0.0 else base["predictor_dropout"])
    base.update(
        {
            "predictor_hidden": predictor_hidden,
            "predictor_dropout": predictor_dropout,
            "requested_predictor_hidden": int(args.predictor_hidden),
            "requested_predictor_dropout": float(args.predictor_dropout),
        }
    )
    return base


def build_a7_architecture_candidates(args: argparse.Namespace, num_classes: int) -> List[Dict[str, Any]]:
    if str(args.arch_selection_mode) == "fixed":
        candidate = resolve_a7_fixed_architecture(args, num_classes)
        candidate["candidate_id"] = 0
        candidate["selection_mode"] = "fixed"
        return [candidate]

    presets = [item.strip() for item in str(args.predictor_arch_grid).split(",") if item.strip()]
    if not presets:
        raise ValueError("--predictor_arch_grid must contain at least one preset when --arch_selection_mode val_constraint.")
    candidates = []
    seen = set()
    for preset in presets:
        if preset in seen:
            continue
        seen.add(preset)
        candidate = resolve_a7_architecture_preset(
            preset,
            num_classes=num_classes,
            requested_value_hidden=int(args.value_hidden),
            requested_value_dropout=float(args.value_dropout),
        )
        candidate["candidate_id"] = int(len(candidates))
        candidate["selection_mode"] = "val_constraint"
        candidate["requested_predictor_hidden"] = int(args.predictor_hidden)
        candidate["requested_predictor_dropout"] = float(args.predictor_dropout)
        candidates.append(candidate)
    return candidates


def _is_compatible_teacher_ckpt(ckpt_path: Path, *, dataset: str, actions_path: Path) -> Tuple[bool, str]:
    try:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_type = str(checkpoint.get("model_type", "")).strip().lower()
        if model_type != "catboost_mask":
            return False, f"unsupported model_type={model_type!r}; expected 'catboost_mask'"
        feature_columns = [str(x) for x in checkpoint.get("feature_columns", [])]
        if not feature_columns:
            return False, "missing feature_columns"
        action_ids = [str(x) for x in checkpoint.get("action_ids", [])]
        action_matrix = torch.tensor(np.asarray(checkpoint.get("action_feature_matrix", []), dtype=np.float32))
        loaded_matrix, action_groups = load_action_feature_matrix(str(actions_path), dataset, feature_columns)
        loaded_ids = [str(item["action_id"]) for item in action_groups]
        if loaded_ids == action_ids:
            return True, "same action_ids"
        if action_matrix.shape == loaded_matrix.shape and torch.equal(action_matrix.float(), loaded_matrix.float()):
            return True, "renamed action_ids with identical action-feature matrix"
        return False, f"action mapping differs: teacher_actions={len(action_ids)} actions_json={len(loaded_ids)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_teacher_run_dir(args: argparse.Namespace, actions_path: Path) -> Path:
    del actions_path
    teacher_dir = Path(args.teacher_run_dir).resolve()
    checkpoint_path = teacher_dir / "ckpts" / "teacher_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    return teacher_dir


def transform_oracle_q_value(raw: float, *, transform: str) -> float:
    raw = float(raw)
    if not np.isfinite(raw):
        return 0.0
    if transform == "raw":
        return raw
    if transform == "clip_zero":
        return float(max(raw, 0.0))
    raise ValueError(f"Unknown target transform: {transform}")


def _resolve_phase_schedule(
    *,
    num_actions: int,
    prefix_steps: int,
    middle_steps: int,
    preset: str,
    presets: Dict[str, Optional[Tuple[float, float]]],
    fallback: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int]:
    if preset not in presets:
        raise ValueError(f"Unknown schedule preset: {preset}")
    ratio = presets[preset]
    if ratio is None:
        if fallback is None:
            raise ValueError(f"Schedule preset '{preset}' requires a training schedule.")
        default_prefix, default_middle = fallback
    else:
        default_prefix = max(0, int(round(float(num_actions) * float(ratio[0]))))
        default_middle = max(1, int(round(float(num_actions) * float(ratio[1]))))
    prefix = int(default_prefix) if int(prefix_steps) < 0 else int(prefix_steps)
    middle = int(default_middle) if int(middle_steps) < 0 else int(middle_steps)
    prefix = max(0, min(prefix, int(num_actions)))
    middle = max(0, min(middle, max(0, int(num_actions) - prefix)))
    suffix = max(0, int(num_actions) - prefix - middle)
    return int(prefix), int(middle), int(suffix)


def resolve_hybrid_schedule(
    dataset: str,
    num_actions: int,
    prefix_steps: int,
    middle_steps: int,
    schedule_preset: str,
) -> Tuple[int, int, int]:
    del dataset
    return _resolve_phase_schedule(
        num_actions=num_actions,
        prefix_steps=prefix_steps,
        middle_steps=middle_steps,
        preset=str(schedule_preset),
        presets=SCHEDULE_PRESET_RATIOS,
    )


def resolve_inference_schedule(
    dataset: str,
    num_actions: int,
    train_prefix_steps: int,
    train_middle_steps: int,
    prefix_steps: int,
    middle_steps: int,
    inference_schedule_preset: str,
) -> Tuple[int, int, int]:
    del dataset
    return _resolve_phase_schedule(
        num_actions=num_actions,
        prefix_steps=prefix_steps,
        middle_steps=middle_steps,
        preset=str(inference_schedule_preset),
        presets=INFERENCE_SCHEDULE_PRESET_RATIOS,
        fallback=(int(train_prefix_steps), int(train_middle_steps)),
    )


@contextmanager
def temporary_policy_schedule(
    cmi_model: CMIEstimator,
    *,
    one_step_prefix_steps: int,
    full_path_middle_steps: int,
):
    old_prefix = getattr(cmi_model, "one_step_prefix_steps", None)
    old_middle = getattr(cmi_model, "full_path_middle_steps", None)
    cmi_model.one_step_prefix_steps = int(max(0, one_step_prefix_steps))
    cmi_model.full_path_middle_steps = int(max(0, full_path_middle_steps))
    try:
        yield
    finally:
        if old_prefix is not None:
            cmi_model.one_step_prefix_steps = old_prefix
        if old_middle is not None:
            cmi_model.full_path_middle_steps = old_middle


def build_inference_schedule_spec(
    *,
    dataset: str,
    num_actions: int,
    train_prefix_steps: int,
    train_middle_steps: int,
    train_suffix_steps: int,
    preset: str,
    prefix_override: int = -1,
    middle_override: int = -1,
) -> Dict[str, Any]:
    prefix, middle, suffix = resolve_inference_schedule(
        dataset,
        num_actions,
        train_prefix_steps,
        train_middle_steps,
        prefix_override,
        middle_override,
        preset,
    )
    return {
        "preset": str(preset),
        "one_step_prefix_steps": int(prefix),
        "proposal_rerank_middle_steps": int(middle),
        "one_step_suffix_steps": int(suffix),
        "schedule_aligned_inference": bool(
            int(prefix) == int(train_prefix_steps) and int(middle) == int(train_middle_steps)
        ),
        "training_one_step_prefix_steps": int(train_prefix_steps),
        "training_proposal_rerank_middle_steps": int(train_middle_steps),
        "training_one_step_suffix_steps": int(train_suffix_steps),
    }


def unique_presets(raw: str) -> List[str]:
    seen = set()
    presets: List[str] = []
    for item in parse_str_list(str(raw or "")):
        if item in seen:
            continue
        seen.add(item)
        presets.append(item)
    return presets


def schedule_metrics_summary(name: str, spec: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    curve = [
        None if item.get("test_acc") is None else float(item["test_acc"])
        for item in metrics.get("constraint_acc_by_num_features_integer", [])
    ]
    return {
        "name": str(name),
        "preset": str(spec.get("preset", name)),
        "one_step_prefix_steps": int(spec.get("one_step_prefix_steps", 0)),
        "proposal_rerank_middle_steps": int(spec.get("proposal_rerank_middle_steps", 0)),
        "one_step_suffix_steps": int(spec.get("one_step_suffix_steps", 0)),
        "schedule_aligned_inference": bool(spec.get("schedule_aligned_inference", False)),
        "mean_acc@all": metrics.get("constraint_mean_acc_at_all"),
        "final_acc": metrics.get("constraint_final_acc"),
        "constraint_valid_rate": metrics.get("constraint_valid_rate"),
        "constraint_valid_n": int(metrics.get("constraint_valid_n", 0)),
        "constraint_total_n": int(metrics.get("constraint_total_n", 0)),
        "constraint_acc_curve": curve,
    }


def _path_by_sample_index(metrics: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    paths = metrics.get("constraint_sample_paths", [])
    out: Dict[int, Dict[str, Any]] = {}
    for row in paths:
        if not isinstance(row, dict) or "sample_index" not in row:
            continue
        out[int(row["sample_index"])] = row
    return out


def _bool_at(values: List[Any], idx: int) -> Optional[bool]:
    if idx < 0 or idx >= len(values):
        return None
    return bool(values[idx])


def _action_at(values: List[Any], idx: int) -> Optional[int]:
    if idx < 0 or idx >= len(values):
        return None
    try:
        return int(values[idx])
    except Exception:
        return None


def summarize_paired_schedule_comparison(
    *,
    reference_name: str,
    variant_name: str,
    reference_metrics: Dict[str, Any],
    variant_metrics: Dict[str, Any],
    budget_list: List[int],
    max_records: int = 500,
) -> Dict[str, Any]:
    ref_paths = _path_by_sample_index(reference_metrics)
    var_paths = _path_by_sample_index(variant_metrics)
    common_indices = sorted(set(ref_paths) & set(var_paths))
    max_budget = int(max(budget_list)) if budget_list else 0
    per_step: List[Dict[str, Any]] = []
    changed_records: List[Dict[str, Any]] = []
    total_win = 0
    total_loss = 0
    total_both_correct = 0
    total_both_wrong = 0
    total_correct_pairs = 0
    total_action_pairs = 0
    total_action_changed = 0

    for step_idx in range(max_budget):
        step_win = 0
        step_loss = 0
        step_both_correct = 0
        step_both_wrong = 0
        step_correct_pairs = 0
        step_action_pairs = 0
        step_action_changed = 0
        for sample_idx in common_indices:
            ref_path = ref_paths[sample_idx]
            var_path = var_paths[sample_idx]
            ref_correct = _bool_at(list(ref_path.get("correct_by_step", [])), step_idx)
            var_correct = _bool_at(list(var_path.get("correct_by_step", [])), step_idx)
            ref_action = _action_at(list(ref_path.get("selected_action_indices", [])), step_idx)
            var_action = _action_at(list(var_path.get("selected_action_indices", [])), step_idx)

            action_changed = None
            if ref_action is not None and var_action is not None:
                step_action_pairs += 1
                total_action_pairs += 1
                action_changed = bool(ref_action != var_action)
                if action_changed:
                    step_action_changed += 1
                    total_action_changed += 1

            if ref_correct is None or var_correct is None:
                continue
            step_correct_pairs += 1
            total_correct_pairs += 1
            outcome = "tie_wrong"
            if (not ref_correct) and var_correct:
                step_win += 1
                total_win += 1
                outcome = "win"
            elif ref_correct and (not var_correct):
                step_loss += 1
                total_loss += 1
                outcome = "loss"
            elif ref_correct and var_correct:
                step_both_correct += 1
                total_both_correct += 1
                outcome = "tie_correct"
            else:
                step_both_wrong += 1
                total_both_wrong += 1

            if len(changed_records) < int(max_records) and (
                outcome in {"win", "loss"} or bool(action_changed)
            ):
                changed_records.append(
                    {
                        "sample_index": int(sample_idx),
                        "acquisition_step": int(step_idx + 1),
                        "outcome": outcome,
                        "reference_correct": bool(ref_correct),
                        "variant_correct": bool(var_correct),
                        "reference_action": ref_action,
                        "variant_action": var_action,
                        "action_changed": action_changed,
                    }
                )

        step_net = int(step_win - step_loss)
        per_step.append(
            {
                "acquisition_step": int(step_idx + 1),
                "paired_count": int(step_correct_pairs),
                "win_count": int(step_win),
                "loss_count": int(step_loss),
                "net_win": int(step_net),
                "both_correct_count": int(step_both_correct),
                "both_wrong_count": int(step_both_wrong),
                "delta_acc": None if step_correct_pairs == 0 else float(step_net / step_correct_pairs),
                "action_pair_count": int(step_action_pairs),
                "changed_action_count": int(step_action_changed),
                "changed_action_rate": None if step_action_pairs == 0 else float(step_action_changed / step_action_pairs),
            }
        )

    net_win = int(total_win - total_loss)
    ref_mean = reference_metrics.get("constraint_mean_acc_at_all")
    var_mean = variant_metrics.get("constraint_mean_acc_at_all")
    return {
        "reference_schedule": str(reference_name),
        "variant_schedule": str(variant_name),
        "common_sample_count": int(len(common_indices)),
        "paired_sample_step_count": int(total_correct_pairs),
        "win_count": int(total_win),
        "loss_count": int(total_loss),
        "net_win": int(net_win),
        "both_correct_count": int(total_both_correct),
        "both_wrong_count": int(total_both_wrong),
        "mean_acc_delta_from_pairs": None if total_correct_pairs == 0 else float(net_win / total_correct_pairs),
        "mean_acc_delta_reported": (
            None
            if ref_mean is None or var_mean is None
            else float(float(var_mean) - float(ref_mean))
        ),
        "action_pair_count": int(total_action_pairs),
        "changed_action_count": int(total_action_changed),
        "changed_action_rate": None if total_action_pairs == 0 else float(total_action_changed / total_action_pairs),
        "same_action_rate": None if total_action_pairs == 0 else float(1.0 - total_action_changed / total_action_pairs),
        "per_step": per_step,
        "changed_records_capped": bool(len(changed_records) >= int(max_records)),
        "changed_records": changed_records,
    }


def _selected_indices_from_mask(mask_tuple: Tuple[int, ...]) -> List[int]:
    return [int(i) for i, flag in enumerate(mask_tuple) if int(flag) > 0]


def _path_score(
    *,
    true_probs: List[float],
    log_probs: List[float],
    predictions: List[int],
    label: int,
    current_log_prob: float,
    mode: str,
    mixed_hard_acc_alpha: float,
) -> float:
    if not true_probs:
        return float(current_log_prob)
    mode = str(mode).strip().lower()
    if mode == "mean_true_prob":
        return float(np.mean(true_probs))
    if mode == "negative_ce":
        return float(np.mean(log_probs))
    if mode == "ce_reduction":
        return float(log_probs[-1] - float(current_log_prob))
    if mode == "mean_hard_acc":
        return float(np.mean([1.0 if int(p) == int(label) else 0.0 for p in predictions]))
    if mode == "mixed_score":
        mean_true_prob = float(np.mean(true_probs))
        mean_hard_acc = float(np.mean([1.0 if int(p) == int(label) else 0.0 for p in predictions]))
        return float(mean_true_prob + float(mixed_hard_acc_alpha) * mean_hard_acc)
    raise ValueError(f"Unsupported full-path score mode: {mode}")


def _softmax_scores(scores: np.ndarray, *, temperature: float) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 0:
        return scores.astype(np.float32)
    temp = max(float(temperature), 1.0e-8)
    logits = scores / temp
    logits = logits - float(np.max(logits))
    exp_logits = np.exp(logits)
    denom = float(exp_logits.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full(scores.shape, 1.0 / float(scores.size), dtype=np.float32)
    return (exp_logits / denom).astype(np.float32)


class OracleHorizonAgreementAdaptiveFusedProposalRerankQTargetProvider:
    def __init__(
        self,
        *,
        dataset: str,
        run_dir: Path,
        teacher_ckpt: str,
        dataset_csv: Path,
        label_col: str,
        labels: np.ndarray,
        group_matrix: torch.Tensor,
        action_groups: List[Dict[str, Any]],
        prerequisite_matrix: torch.Tensor,
        one_step_target_transform: str,
        full_path_target_transform: str,
        planner_config: FullPathPlannerConfig,
        target_reduce: str,
        one_step_prefix_steps: int,
        full_path_middle_steps: int,
        proposal_top_k: int,
        alpha_min: float,
        alpha_max: float,
        alpha_gap_scale: float,
        alpha_gap_floor: float,
        alpha_agree_bonus: float,
        alpha_disagree_penalty: float,
        horizon_advantage_threshold: float,
        horizon_penalty: float,
    ) -> None:
        self.dataset = str(dataset)
        self.run_dir = Path(run_dir).resolve()
        self.one_step_target_transform = str(one_step_target_transform)
        self.full_path_target_transform = str(full_path_target_transform)
        self.teacher_art = load_teacher_artifacts(
            self.run_dir,
            teacher_ckpt=str(teacher_ckpt or ""),
            device=torch.device("cpu"),
        )
        teacher_matrix = self.teacher_art.action_feature_matrix.detach().cpu().float()
        current_matrix = group_matrix.detach().cpu().float()
        if teacher_matrix.shape != current_matrix.shape or not torch.equal(teacher_matrix, current_matrix):
            raise ValueError(
                "Teacher checkpoint action-feature matrix does not match the current actions.json. "
                "Re-run teacher pipeline for the current action design or pass a compatible --teacher_run_dir."
            )
        self.arrays = load_dataset_arrays(
            dataset_csv=dataset_csv,
            feature_columns=self.teacher_art.feature_columns,
            label_col=label_col,
            missing_value=float(self.teacher_art.missing_value),
            mean=self.teacher_art.mean,
            std=self.teacher_art.std,
        )
        if len(labels) != len(self.arrays.x_norm):
            raise ValueError(
                f"Label length mismatch: runner labels={len(labels)} teacher arrays={len(self.arrays.x_norm)}"
            )
        self.labels = np.asarray(labels, dtype=np.int64)
        self.x_norm = torch.tensor(self.arrays.x_norm, dtype=torch.float32)
        self.present = torch.tensor(self.arrays.present, dtype=torch.float32)
        self.action_feature_matrix = teacher_matrix
        self.action_groups = action_groups
        self.prerequisite_matrix = prerequisite_matrix.detach().cpu().float()
        self.action_ids = [str(x) for x in self.teacher_art.action_ids]
        self.num_classes = int(self.teacher_art.num_classes)
        self.planner_config = planner_config
        self.target_reduce = str(target_reduce)
        self.one_step_prefix_steps = int(max(0, one_step_prefix_steps))
        self.full_path_middle_steps = int(max(0, full_path_middle_steps))
        self.proposal_top_k = int(max(1, proposal_top_k))
        alpha_min = float(np.clip(float(alpha_min), 0.0, 1.0))
        alpha_max = float(np.clip(float(alpha_max), 0.0, 1.0))
        if alpha_max < alpha_min:
            alpha_min, alpha_max = alpha_max, alpha_min
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_gap_scale = max(float(alpha_gap_scale), 1.0e-8)
        self.alpha_gap_floor = max(float(alpha_gap_floor), 1.0e-8)
        self.alpha_agree_bonus = max(float(alpha_agree_bonus), 0.0)
        self.alpha_disagree_penalty = max(float(alpha_disagree_penalty), 0.0)
        self.horizon_advantage_threshold = max(float(horizon_advantage_threshold), 0.0)
        self.horizon_penalty = max(float(horizon_penalty), 0.0)
        self.cache: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
        self.one_step_state_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]] = {}
        self.proposal_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[int, ...]] = {}
        self.alpha_cache: Dict[Tuple[int, Tuple[int, ...]], Dict[str, Any]] = {}
        self.full_path_raw_cache: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
        self.num_calls = 0
        self.num_cache_hits = 0
        self.one_step_calls = 0
        self.full_path_calls = 0
        self.rerank_middle_calls = 0
        self.fused_target_calls = 0
        self.proposal_full_path_calls = 0
        self.proposal_fallback_one_step_calls = 0
        self.one_step_state_cache_hits = 0
        self.proposal_cache_hits = 0
        self.alpha_cache_hits = 0
        self.full_path_raw_cache_hits = 0
        self.alpha_stat_count = 0
        self.alpha_stat_sum = 0.0
        self.alpha_stat_min = float("inf")
        self.alpha_stat_max = float("-inf")
        self.alpha_base_stat_sum = 0.0
        self.alpha_after_agreement_stat_sum = 0.0
        self.agreement_count = 0
        self.disagreement_count = 0
        self.horizon_advantage_sum = 0.0
        self.horizon_penalty_count = 0
        self.alpha_gap_sum = 0.0
        self.alpha_relative_gap_sum = 0.0

    def _target_mode_for_mask(self, mask_tuple: Tuple[int, ...]) -> str:
        step_idx = int(sum(int(v > 0) for v in mask_tuple))
        if step_idx < self.one_step_prefix_steps:
            return "one_step"
        if step_idx < self.one_step_prefix_steps + self.full_path_middle_steps:
            return "proposal_rerank"
        return "one_step"

    def _one_step_soft_for_state(
        self,
        *,
        sample_idx: int,
        mask_tuple: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        key = (int(sample_idx), mask_tuple)
        if key in self.one_step_state_cache:
            self.one_step_state_cache_hits += 1
            return self.one_step_state_cache[key]
        m_act = torch.tensor(mask_tuple, dtype=torch.float32)
        soft = compute_catboost_soft_teacher(
            teacher_model=self.teacher_art.teacher_model,
            x_norm_row=self.x_norm[int(sample_idx)],
            present_row=self.present[int(sample_idx)],
            m_act=m_act,
            label=int(self.labels[int(sample_idx)]),
            action_feature_matrix=self.action_feature_matrix,
            action_ids=self.action_ids,
            num_classes=self.num_classes,
            prerequisite_matrix=self.prerequisite_matrix,
        )
        candidate_mask = np.asarray(soft["candidate_mask"], dtype=bool)
        utility = np.asarray(soft["utility"], dtype=np.float64)
        self.one_step_state_cache[key] = (candidate_mask, utility)
        return candidate_mask, utility

    def _one_step_score(
        self,
        *,
        sample_idx: int,
        mask_tuple: Tuple[int, ...],
        action_idx: int,
    ) -> float:
        candidate_mask, utility = self._one_step_soft_for_state(sample_idx=sample_idx, mask_tuple=mask_tuple)
        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= utility.shape[0] or not bool(candidate_mask[action_idx]):
            return 0.0
        return float(utility[action_idx])

    def _proposal_indices(self, *, sample_idx: int, mask_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
        key = (int(sample_idx), mask_tuple)
        if key in self.proposal_cache:
            self.proposal_cache_hits += 1
            return self.proposal_cache[key]
        candidate_mask, utility = self._one_step_soft_for_state(sample_idx=sample_idx, mask_tuple=mask_tuple)
        candidate_indices = np.where(candidate_mask)[0].astype(np.int64).tolist()
        candidate_indices.sort(key=lambda idx: (-float(utility[int(idx)]), int(idx)))
        proposal = tuple(int(idx) for idx in candidate_indices[: self.proposal_top_k])
        self.proposal_cache[key] = proposal
        return proposal

    def _full_path_scores_for_proposal(
        self,
        *,
        sample_idx: int,
        mask_tuple: Tuple[int, ...],
        proposal: Tuple[int, ...],
    ) -> Dict[int, float]:
        selected_start = _selected_indices_from_mask(mask_tuple)
        scores: Dict[int, float] = {}
        for action_idx in proposal:
            scores[int(action_idx)] = float(
                self._forced_first_full_path_score(
                    sample_idx=int(sample_idx),
                    selected_start=selected_start,
                    action_idx=int(action_idx),
                )
            )
        return scores

    @staticmethod
    def _normalized_top_margin(scores: Dict[int, float]) -> float:
        values = np.asarray(
            [float(v) for v in scores.values() if np.isfinite(float(v))],
            dtype=np.float64,
        )
        if values.size <= 1:
            return 0.0
        sorted_values = np.sort(values)[::-1]
        spread = float(np.max(values) - np.min(values))
        if spread <= 1.0e-8:
            return 0.0
        return float(np.clip((float(sorted_values[0]) - float(sorted_values[1])) / spread, 0.0, 1.0))

    def _adaptive_alpha_for_state(
        self,
        *,
        sample_idx: int,
        mask_tuple: Tuple[int, ...],
        proposal: Tuple[int, ...],
    ) -> Dict[str, Any]:
        key = (int(sample_idx), mask_tuple)
        if key in self.alpha_cache:
            self.alpha_cache_hits += 1
            return self.alpha_cache[key]

        candidate_mask, utility = self._one_step_soft_for_state(sample_idx=sample_idx, mask_tuple=mask_tuple)
        candidate_indices = np.where(candidate_mask)[0].astype(np.int64).tolist()
        values = np.asarray(utility[candidate_mask], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size <= 1:
            gap = 0.0
            relative_gap = 1.0 if values.size == 1 else 0.0
            top1 = float(values[0]) if values.size == 1 else None
            top2 = None
            utility_scale = float(self.alpha_gap_floor)
        else:
            sorted_values = np.sort(values)[::-1]
            top1 = float(sorted_values[0])
            top2 = float(sorted_values[1])
            gap = max(0.0, top1 - top2)
            utility_scale = max(
                float(np.max(np.abs(values))),
                float(np.std(values)),
                float(self.alpha_gap_floor),
            )
            relative_gap = float(np.clip(gap / utility_scale, 0.0, 1.0))

        confidence = float(np.clip(relative_gap / self.alpha_gap_scale, 0.0, 1.0))
        alpha_base = float(self.alpha_min + (self.alpha_max - self.alpha_min) * confidence)
        one_step_top1_action = None
        if candidate_indices:
            one_step_top1_action = int(
                sorted(
                    candidate_indices,
                    key=lambda idx: (-float(utility[int(idx)]), int(idx)),
                )[0]
            )
        proposal = tuple(int(x) for x in proposal)
        full_path_scores = self._full_path_scores_for_proposal(
            sample_idx=int(sample_idx),
            mask_tuple=mask_tuple,
            proposal=proposal,
        )
        full_path_top1_action = None
        full_path_top1_score = None
        if full_path_scores:
            full_path_top1_action = int(
                sorted(
                    full_path_scores.keys(),
                    key=lambda idx: (-float(full_path_scores[int(idx)]), int(idx)),
                )[0]
            )
            full_path_top1_score = float(full_path_scores[full_path_top1_action])
        agreement = (
            one_step_top1_action is not None
            and full_path_top1_action is not None
            and int(one_step_top1_action) == int(full_path_top1_action)
        )
        if agreement:
            alpha = min(self.alpha_max, alpha_base + self.alpha_agree_bonus)
            self.agreement_count += 1
        else:
            alpha = max(self.alpha_min, alpha_base - self.alpha_disagree_penalty)
            self.disagreement_count += 1
        alpha_after_agreement = float(alpha)

        one_step_scores = {
            int(idx): float(utility[int(idx)])
            for idx in proposal
            if int(idx) >= 0 and int(idx) < int(utility.shape[0])
        }
        one_step_margin = self._normalized_top_margin(one_step_scores)
        full_path_margin = self._normalized_top_margin(full_path_scores)
        long_horizon_advantage = float(full_path_margin - one_step_margin)
        horizon_penalty_applied = bool(long_horizon_advantage > self.horizon_advantage_threshold)
        if horizon_penalty_applied:
            alpha = max(self.alpha_min, alpha - self.horizon_penalty)
            self.horizon_penalty_count += 1
        selected_actions = _selected_indices_from_mask(mask_tuple)
        record = {
            "sample_idx": int(sample_idx),
            "step_idx": int(len(selected_actions)),
            "selected_actions": [int(x) for x in selected_actions],
            "candidate_count": int(values.size),
            "proposal_actions": [int(x) for x in proposal],
            "one_step_scores": {str(int(k)): float(v) for k, v in sorted(one_step_scores.items())},
            "one_step_top1_action": one_step_top1_action,
            "full_path_top1_action": full_path_top1_action,
            "full_path_top1_score": full_path_top1_score,
            "full_path_scores": {str(int(k)): float(v) for k, v in sorted(full_path_scores.items())},
            "agreement": bool(agreement),
            "top1_utility": None if top1 is None else float(top1),
            "top2_utility": None if top2 is None else float(top2),
            "utility_scale": float(utility_scale),
            "alpha_base": float(alpha_base),
            "alpha_after_agreement": float(alpha_after_agreement),
            "alpha": alpha,
            "one_step_margin": float(one_step_margin),
            "full_path_margin": float(full_path_margin),
            "long_horizon_advantage": float(long_horizon_advantage),
            "horizon_penalty_applied": bool(horizon_penalty_applied),
            "gap": float(gap),
            "relative_gap": float(relative_gap),
            "confidence": float(confidence),
        }
        self.alpha_cache[key] = record
        self.alpha_stat_count += 1
        self.alpha_base_stat_sum += float(alpha_base)
        self.alpha_after_agreement_stat_sum += float(alpha_after_agreement)
        self.alpha_stat_sum += alpha
        self.alpha_stat_min = min(self.alpha_stat_min, alpha)
        self.alpha_stat_max = max(self.alpha_stat_max, alpha)
        self.horizon_advantage_sum += float(long_horizon_advantage)
        self.alpha_gap_sum += float(gap)
        self.alpha_relative_gap_sum += float(relative_gap)
        return record

    @staticmethod
    def _numeric_summary(values: List[float]) -> Dict[str, Any]:
        clean = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
        if clean.size == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "p10": None,
                "p25": None,
                "p50": None,
                "p75": None,
                "p90": None,
                "p95": None,
                "max": None,
            }
        percentiles = np.percentile(clean, [10, 25, 50, 75, 90, 95])
        return {
            "count": int(clean.size),
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "min": float(np.min(clean)),
            "p10": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p50": float(percentiles[2]),
            "p75": float(percentiles[3]),
            "p90": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "max": float(np.max(clean)),
        }

    def _diagnostic_group_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        alpha_margin = max((self.alpha_max - self.alpha_min) * 0.05, 1.0e-8)
        n = len(records)
        if n == 0:
            return {
                "state_count": 0,
                "alpha": self._numeric_summary([]),
                "alpha_base": self._numeric_summary([]),
                "alpha_after_agreement": self._numeric_summary([]),
                "gap": self._numeric_summary([]),
                "relative_gap": self._numeric_summary([]),
                "confidence": self._numeric_summary([]),
                "candidate_count": self._numeric_summary([]),
                "one_step_margin": self._numeric_summary([]),
                "full_path_margin": self._numeric_summary([]),
                "long_horizon_advantage": self._numeric_summary([]),
                "horizon_penalty_rate": None,
                "horizon_penalty_n": 0,
                "agreement_rate": None,
                "agreement_n": 0,
                "disagreement_n": 0,
                "agree_alpha": self._numeric_summary([]),
                "disagree_alpha": self._numeric_summary([]),
                "alpha_at_min_rate": None,
                "alpha_at_max_rate": None,
                "alpha_near_min_rate": None,
                "alpha_near_max_rate": None,
            }
        alphas = [float(row["alpha"]) for row in records]
        agree_records = [row for row in records if bool(row.get("agreement"))]
        disagree_records = [row for row in records if not bool(row.get("agreement"))]
        return {
            "state_count": int(n),
            "alpha": self._numeric_summary(alphas),
            "alpha_base": self._numeric_summary([float(row["alpha_base"]) for row in records]),
            "alpha_after_agreement": self._numeric_summary([float(row["alpha_after_agreement"]) for row in records]),
            "gap": self._numeric_summary([float(row["gap"]) for row in records]),
            "relative_gap": self._numeric_summary([float(row["relative_gap"]) for row in records]),
            "confidence": self._numeric_summary([float(row["confidence"]) for row in records]),
            "candidate_count": self._numeric_summary([float(row["candidate_count"]) for row in records]),
            "one_step_margin": self._numeric_summary([float(row["one_step_margin"]) for row in records]),
            "full_path_margin": self._numeric_summary([float(row["full_path_margin"]) for row in records]),
            "long_horizon_advantage": self._numeric_summary([float(row["long_horizon_advantage"]) for row in records]),
            "horizon_penalty_rate": float(sum(bool(row.get("horizon_penalty_applied")) for row in records) / n),
            "horizon_penalty_n": int(sum(bool(row.get("horizon_penalty_applied")) for row in records)),
            "agreement_rate": float(len(agree_records) / n),
            "agreement_n": int(len(agree_records)),
            "disagreement_n": int(len(disagree_records)),
            "agree_alpha": self._numeric_summary([float(row["alpha"]) for row in agree_records]),
            "disagree_alpha": self._numeric_summary([float(row["alpha"]) for row in disagree_records]),
            "alpha_at_min_rate": float(sum(a <= self.alpha_min + 1.0e-8 for a in alphas) / n),
            "alpha_at_max_rate": float(sum(a >= self.alpha_max - 1.0e-8 for a in alphas) / n),
            "alpha_near_min_rate": float(sum(a <= self.alpha_min + alpha_margin for a in alphas) / n),
            "alpha_near_max_rate": float(sum(a >= self.alpha_max - alpha_margin for a in alphas) / n),
        }

    def alpha_gap_diagnostics(self) -> Dict[str, Any]:
        records = sorted(
            (dict(row) for row in self.alpha_cache.values()),
            key=lambda row: (
                int(row.get("step_idx", -1)),
                int(row.get("sample_idx", -1)),
                tuple(int(x) for x in row.get("selected_actions", [])),
            ),
        )
        by_step: Dict[str, List[Dict[str, Any]]] = {}
        for row in records:
            by_step.setdefault(str(int(row["step_idx"])), []).append(row)
        return {
            "target": "oracle_horizon_agreement_adaptive_fused_proposal_rerank_q",
            "fusion_mode": "proposal_inner_horizon_agreement_adaptive_weighted_one_step_full_path",
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "alpha_gap_scale": float(self.alpha_gap_scale),
            "alpha_gap_floor": float(self.alpha_gap_floor),
            "alpha_agree_bonus": float(self.alpha_agree_bonus),
            "alpha_disagree_penalty": float(self.alpha_disagree_penalty),
            "horizon_advantage_threshold": float(self.horizon_advantage_threshold),
            "horizon_penalty": float(self.horizon_penalty),
            "record_granularity": "unique_sample_state",
            "summary": self._diagnostic_group_summary(records),
            "by_step": {
                step: self._diagnostic_group_summary(step_records)
                for step, step_records in sorted(by_step.items(), key=lambda item: int(item[0]))
            },
            "records": records,
        }

    def _state_prediction_for_selected(self, *, sample_idx: int, selected: List[int], label: int) -> Dict[str, float]:
        m_act = torch.zeros(len(self.action_ids), dtype=torch.float32)
        for action_idx in selected:
            if 0 <= int(action_idx) < int(m_act.numel()):
                m_act[int(action_idx)] = 1.0
        state = build_state_vector(
            x_norm_row=self.x_norm[int(sample_idx)],
            present_row=self.present[int(sample_idx)],
            m_act=m_act,
            action_feature_matrix=self.action_feature_matrix,
        )
        pred = predict_catboost_state(self.teacher_art.teacher_model, state, num_classes=self.num_classes)
        probs = pred["proba"][0]
        label_idx = int(label)
        true_prob = float(probs[label_idx])
        return {
            "true_prob": true_prob,
            "log_prob": float(np.log(max(true_prob, 1.0e-12))),
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
        }

    def _forced_first_full_path_score(
        self,
        *,
        sample_idx: int,
        selected_start: List[int],
        action_idx: int,
    ) -> float:
        action_idx = int(action_idx)
        selected_start = [int(x) for x in selected_start]
        selected_set = set(selected_start)
        mask_tuple = tuple(1 if idx in selected_set else 0 for idx in range(len(self.action_ids)))
        cache_key = (int(sample_idx), mask_tuple, int(action_idx))
        if cache_key in self.full_path_raw_cache:
            self.full_path_raw_cache_hits += 1
            return float(self.full_path_raw_cache[cache_key])
        if action_idx in set(selected_start):
            return 0.0
        legal = legal_action_indices_from_selected(selected_start, self.action_groups)
        if action_idx not in {int(x) for x in legal}:
            return 0.0

        label = int(self.labels[int(sample_idx)])
        current = self._state_prediction_for_selected(
            sample_idx=int(sample_idx),
            selected=selected_start,
            label=label,
        )
        current_log_prob = float(current["log_prob"])

        selected_after = list(selected_start) + [action_idx]
        first_pred = self._state_prediction_for_selected(
            sample_idx=int(sample_idx),
            selected=selected_after,
            label=label,
        )
        true_probs = [float(first_pred["true_prob"])]
        log_probs = [float(first_pred["log_prob"])]
        predictions = [int(first_pred["prediction"])]
        first_score = _path_score(
            true_probs=true_probs,
            log_probs=log_probs,
            predictions=predictions,
            label=label,
            current_log_prob=current_log_prob,
            mode=self.planner_config.score_mode,
            mixed_hard_acc_alpha=float(self.planner_config.mixed_hard_acc_alpha),
        )

        num_actions = int(len(self.action_ids))
        remaining = max(0, num_actions - len(set(selected_start)))
        depth = max(1, min(int(self.planner_config.max_depth), remaining))
        beam_width = max(1, int(self.planner_config.beam_width))
        top_k = max(1, int(self.planner_config.top_k_paths))

        beam: List[Dict[str, Any]] = [
            {
                "path": [action_idx],
                "selected": selected_after,
                "true_probs": true_probs,
                "log_probs": log_probs,
                "predictions": predictions,
                "score": float(first_score),
            }
        ]
        completed: List[Dict[str, Any]] = []

        for _step in range(max(0, depth - 1)):
            expanded: List[Dict[str, Any]] = []
            for item in beam:
                legal_next = legal_action_indices_from_selected(item["selected"], self.action_groups)
                if not legal_next:
                    completed.append(item)
                    continue
                for next_action in legal_next:
                    new_selected = list(item["selected"]) + [int(next_action)]
                    pred = self._state_prediction_for_selected(
                        sample_idx=int(sample_idx),
                        selected=new_selected,
                        label=label,
                    )
                    item_true_probs = list(item["true_probs"]) + [float(pred["true_prob"])]
                    item_log_probs = list(item["log_probs"]) + [float(pred["log_prob"])]
                    item_predictions = list(item["predictions"]) + [int(pred["prediction"])]
                    score = _path_score(
                        true_probs=item_true_probs,
                        log_probs=item_log_probs,
                        predictions=item_predictions,
                        label=label,
                        current_log_prob=current_log_prob,
                        mode=self.planner_config.score_mode,
                        mixed_hard_acc_alpha=float(self.planner_config.mixed_hard_acc_alpha),
                    )
                    expanded.append(
                        {
                            "path": list(item["path"]) + [int(next_action)],
                            "selected": new_selected,
                            "true_probs": item_true_probs,
                            "log_probs": item_log_probs,
                            "predictions": item_predictions,
                            "score": float(score),
                        }
                    )
            if not expanded:
                break
            expanded.sort(key=lambda row: (-float(row["score"]), tuple(int(x) for x in row["path"])))
            beam = expanded[:beam_width]

        final_paths = completed + beam
        if not final_paths:
            final_paths = beam
        final_paths.sort(key=lambda row: (-float(row["score"]), tuple(int(x) for x in row["path"])))
        top_paths = final_paths[:top_k]
        if self.target_reduce == "best":
            result = float(top_paths[0]["score"])
            self.full_path_raw_cache[cache_key] = result
            return result
        if self.target_reduce == "topk_weighted":
            scores = np.asarray([float(row["score"]) for row in top_paths], dtype=np.float64)
            weights = _softmax_scores(scores, temperature=float(self.planner_config.temperature))
            result = float(np.sum(scores * weights))
            self.full_path_raw_cache[cache_key] = result
            return result
        raise ValueError(f"Unknown full_path_q_reduce: {self.target_reduce}")

    def __call__(
        self,
        sample_indices: torch.Tensor,
        selected_action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> np.ndarray:
        sample_np = sample_indices.detach().cpu().numpy().astype(np.int64)
        mask_np = selected_action_mask.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy().astype(np.int64)
        targets = np.zeros(len(sample_np), dtype=np.float32)

        for row_idx, (sample_idx, action_idx) in enumerate(zip(sample_np.tolist(), actions_np.tolist())):
            mask_tuple = tuple(int(v > 0.5) for v in mask_np[row_idx].tolist())
            key = (int(sample_idx), mask_tuple, int(action_idx))
            self.num_calls += 1
            if key in self.cache:
                self.num_cache_hits += 1
                targets[row_idx] = float(self.cache[key])
                continue

            target_mode = self._target_mode_for_mask(mask_tuple)
            if target_mode == "proposal_rerank":
                self.rerank_middle_calls += 1
                proposal = self._proposal_indices(sample_idx=int(sample_idx), mask_tuple=mask_tuple)
                if int(action_idx) in set(proposal):
                    self.fused_target_calls += 1
                    self.full_path_calls += 1
                    self.proposal_full_path_calls += 1
                    one_step_raw = self._one_step_score(
                        sample_idx=int(sample_idx),
                        mask_tuple=mask_tuple,
                        action_idx=int(action_idx),
                    )
                    full_path_raw = self._forced_first_full_path_score(
                        sample_idx=int(sample_idx),
                        selected_start=_selected_indices_from_mask(mask_tuple),
                        action_idx=int(action_idx),
                    )
                    one_step_value = transform_oracle_q_value(
                        one_step_raw,
                        transform=self.one_step_target_transform,
                    )
                    full_path_value = transform_oracle_q_value(
                        full_path_raw,
                        transform=self.full_path_target_transform,
                    )
                    alpha_record = self._adaptive_alpha_for_state(
                        sample_idx=int(sample_idx),
                        mask_tuple=mask_tuple,
                        proposal=proposal,
                    )
                    alpha = float(alpha_record["alpha"])
                    value = alpha * float(one_step_value) + (1.0 - alpha) * float(full_path_value)
                    self.cache[key] = float(value)
                    targets[row_idx] = float(value)
                    continue
                else:
                    self.one_step_calls += 1
                    self.proposal_fallback_one_step_calls += 1
                    raw_value = self._one_step_score(
                        sample_idx=int(sample_idx),
                        mask_tuple=mask_tuple,
                        action_idx=int(action_idx),
                    )
                    transform = self.one_step_target_transform
            else:
                self.one_step_calls += 1
                raw_value = self._one_step_score(
                    sample_idx=int(sample_idx),
                    mask_tuple=mask_tuple,
                    action_idx=int(action_idx),
                )
                transform = self.one_step_target_transform
            value = transform_oracle_q_value(
                raw_value,
                transform=transform,
            )
            self.cache[key] = float(value)
            targets[row_idx] = float(value)
        return targets

    def summary(self) -> Dict[str, Any]:
        hit_rate = float(self.num_cache_hits / self.num_calls) if self.num_calls else 0.0
        alpha_count = int(self.alpha_stat_count)
        alpha_mean = float(self.alpha_stat_sum / alpha_count) if alpha_count else None
        alpha_base_mean = float(self.alpha_base_stat_sum / alpha_count) if alpha_count else None
        alpha_after_agreement_mean = float(self.alpha_after_agreement_stat_sum / alpha_count) if alpha_count else None
        alpha_min_seen = None if alpha_count == 0 else float(self.alpha_stat_min)
        alpha_max_seen = None if alpha_count == 0 else float(self.alpha_stat_max)
        alpha_gap_mean = float(self.alpha_gap_sum / alpha_count) if alpha_count else None
        alpha_relative_gap_mean = float(self.alpha_relative_gap_sum / alpha_count) if alpha_count else None
        horizon_advantage_mean = float(self.horizon_advantage_sum / alpha_count) if alpha_count else None
        horizon_penalty_rate = float(self.horizon_penalty_count / alpha_count) if alpha_count else None
        agreement_total = int(self.agreement_count + self.disagreement_count)
        agreement_rate = float(self.agreement_count / agreement_total) if agreement_total else None
        return {
            "teacher_run_dir": str(self.run_dir),
            "teacher_ckpt_path": str(self.teacher_art.teacher_ckpt_path),
            "target": "oracle_horizon_agreement_adaptive_fused_proposal_rerank_q",
            "schedule": "one_step_prefix_proposal_rerank_middle_one_step_suffix",
            "fusion_mode": "proposal_inner_horizon_agreement_adaptive_weighted_one_step_full_path",
            "adaptive_alpha_enabled": True,
            "agreement_adaptive_alpha_enabled": True,
            "horizon_advantage_alpha_enabled": True,
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "alpha_gap_scale": float(self.alpha_gap_scale),
            "alpha_gap_floor": float(self.alpha_gap_floor),
            "alpha_agree_bonus": float(self.alpha_agree_bonus),
            "alpha_disagree_penalty": float(self.alpha_disagree_penalty),
            "horizon_advantage_threshold": float(self.horizon_advantage_threshold),
            "horizon_penalty": float(self.horizon_penalty),
            "adaptive_alpha_mean": alpha_mean,
            "adaptive_alpha_base_mean": alpha_base_mean,
            "adaptive_alpha_after_agreement_mean": alpha_after_agreement_mean,
            "adaptive_alpha_min_seen": alpha_min_seen,
            "adaptive_alpha_max_seen": alpha_max_seen,
            "adaptive_alpha_gap_mean": alpha_gap_mean,
            "adaptive_alpha_relative_gap_mean": alpha_relative_gap_mean,
            "long_horizon_advantage_mean": horizon_advantage_mean,
            "horizon_penalty_rate": horizon_penalty_rate,
            "horizon_penalty_n": int(self.horizon_penalty_count),
            "agreement_rate": agreement_rate,
            "agreement_n": int(self.agreement_count),
            "disagreement_n": int(self.disagreement_count),
            "one_step_prefix_steps": int(self.one_step_prefix_steps),
            "proposal_rerank_middle_steps": int(self.full_path_middle_steps),
            "one_step_proposal_top_k": int(self.proposal_top_k),
            "one_step_target_transform": self.one_step_target_transform,
            "full_path_target_transform": self.full_path_target_transform,
            "full_path_top_k": int(self.planner_config.top_k_paths),
            "full_path_beam_width": int(self.planner_config.beam_width),
            "full_path_max_depth": int(self.planner_config.max_depth),
            "full_path_score": str(self.planner_config.score_mode),
            "full_path_temperature": float(self.planner_config.temperature),
            "full_path_mixed_hard_acc_alpha": float(self.planner_config.mixed_hard_acc_alpha),
            "full_path_q_reduce": self.target_reduce,
            "cache_size": int(len(self.cache)),
            "one_step_state_cache_size": int(len(self.one_step_state_cache)),
            "proposal_cache_size": int(len(self.proposal_cache)),
            "adaptive_alpha_cache_size": int(len(self.alpha_cache)),
            "full_path_raw_cache_size": int(len(self.full_path_raw_cache)),
            "num_calls": int(self.num_calls),
            "one_step_calls": int(self.one_step_calls),
            "full_path_calls": int(self.full_path_calls),
            "rerank_middle_calls": int(self.rerank_middle_calls),
            "fused_target_calls": int(self.fused_target_calls),
            "proposal_full_path_calls": int(self.proposal_full_path_calls),
            "proposal_fallback_one_step_calls": int(self.proposal_fallback_one_step_calls),
            "num_cache_hits": int(self.num_cache_hits),
            "one_step_state_cache_hits": int(self.one_step_state_cache_hits),
            "proposal_cache_hits": int(self.proposal_cache_hits),
            "adaptive_alpha_cache_hits": int(self.alpha_cache_hits),
            "full_path_raw_cache_hits": int(self.full_path_raw_cache_hits),
            "cache_hit_rate": hit_rate,
        }


class OracleDoubleHeadQTargetProvider(OracleHorizonAgreementAdaptiveFusedProposalRerankQTargetProvider):


    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.double_head_calls = 0
        self.double_head_cache_hits = 0
        self.double_one_step_calls = 0
        self.double_full_path_label_calls = 0
        self.double_full_path_skipped_calls = 0
        self.double_head_cache: Dict[Tuple[int, Tuple[int, ...], int], Tuple[float, float, float]] = {}

    def double_head_targets(
        self,
        sample_indices: torch.Tensor,
        selected_action_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        sample_np = sample_indices.detach().cpu().numpy().astype(np.int64)
        mask_np = selected_action_mask.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy().astype(np.int64)
        one_step_targets = np.zeros(len(sample_np), dtype=np.float32)
        full_path_targets = np.zeros(len(sample_np), dtype=np.float32)
        full_path_mask = np.zeros(len(sample_np), dtype=np.float32)

        for row_idx, (sample_idx, action_idx) in enumerate(zip(sample_np.tolist(), actions_np.tolist())):
            mask_tuple = tuple(int(v > 0.5) for v in mask_np[row_idx].tolist())
            key = (int(sample_idx), mask_tuple, int(action_idx))
            self.double_head_calls += 1
            if key in self.double_head_cache:
                self.double_head_cache_hits += 1
                one_value, full_value, full_mask = self.double_head_cache[key]
                one_step_targets[row_idx] = float(one_value)
                full_path_targets[row_idx] = float(full_value)
                full_path_mask[row_idx] = float(full_mask)
                continue

            one_step_raw = self._one_step_score(
                sample_idx=int(sample_idx),
                mask_tuple=mask_tuple,
                action_idx=int(action_idx),
            )
            one_step_value = transform_oracle_q_value(
                one_step_raw,
                transform=self.one_step_target_transform,
            )
            self.double_one_step_calls += 1

            target_mode = self._target_mode_for_mask(mask_tuple)
            proposal = self._proposal_indices(sample_idx=int(sample_idx), mask_tuple=mask_tuple)
            if target_mode == "proposal_rerank" and int(action_idx) in set(proposal):
                full_path_raw = self._forced_first_full_path_score(
                    sample_idx=int(sample_idx),
                    selected_start=_selected_indices_from_mask(mask_tuple),
                    action_idx=int(action_idx),
                )
                full_path_value = transform_oracle_q_value(
                    full_path_raw,
                    transform=self.full_path_target_transform,
                )
                full_mask = 1.0
                self.double_full_path_label_calls += 1
                self.full_path_calls += 1
                self.proposal_full_path_calls += 1
            else:
                full_path_value = one_step_value
                full_mask = 0.0
                self.double_full_path_skipped_calls += 1

            self.double_head_cache[key] = (float(one_step_value), float(full_path_value), float(full_mask))
            one_step_targets[row_idx] = float(one_step_value)
            full_path_targets[row_idx] = float(full_path_value)
            full_path_mask[row_idx] = float(full_mask)

        return {
            "one_step": one_step_targets,
            "full_path": full_path_targets,
            "full_path_mask": full_path_mask,
        }

    def summary(self) -> Dict[str, Any]:
        hit_rate = float(self.double_head_cache_hits / self.double_head_calls) if self.double_head_calls else 0.0
        full_label_rate = (
            float(self.double_full_path_label_calls / self.double_head_calls)
            if self.double_head_calls
            else 0.0
        )
        return {
            "teacher_run_dir": str(self.run_dir),
            "teacher_ckpt_path": str(self.teacher_art.teacher_ckpt_path),
            "target": "oracle_fixed_ratio_schedule_double_head_q",
            "schedule": "one_step_prefix_proposal_rerank_middle_one_step_suffix",
            "selection_mode": "prefix_suffix_one_step_middle_one_step_topk_full_path_head_rerank",
            "schedule_aligned_inference": True,
            "one_step_prefix_steps": int(self.one_step_prefix_steps),
            "proposal_rerank_middle_steps": int(self.full_path_middle_steps),
            "one_step_proposal_top_k": int(self.proposal_top_k),
            "one_step_target_transform": self.one_step_target_transform,
            "full_path_target_transform": self.full_path_target_transform,
            "full_path_top_k": int(self.planner_config.top_k_paths),
            "full_path_beam_width": int(self.planner_config.beam_width),
            "full_path_max_depth": int(self.planner_config.max_depth),
            "full_path_score": str(self.planner_config.score_mode),
            "full_path_temperature": float(self.planner_config.temperature),
            "full_path_mixed_hard_acc_alpha": float(self.planner_config.mixed_hard_acc_alpha),
            "full_path_q_reduce": self.target_reduce,
            "double_head_cache_size": int(len(self.double_head_cache)),
            "double_head_calls": int(self.double_head_calls),
            "double_head_cache_hits": int(self.double_head_cache_hits),
            "double_head_cache_hit_rate": hit_rate,
            "double_one_step_target_calls": int(self.double_one_step_calls),
            "double_full_path_label_calls": int(self.double_full_path_label_calls),
            "double_full_path_skipped_calls": int(self.double_full_path_skipped_calls),
            "double_full_path_label_rate": full_label_rate,
            "one_step_state_cache_size": int(len(self.one_step_state_cache)),
            "proposal_cache_size": int(len(self.proposal_cache)),
            "full_path_raw_cache_size": int(len(self.full_path_raw_cache)),
            "one_step_state_cache_hits": int(self.one_step_state_cache_hits),
            "proposal_cache_hits": int(self.proposal_cache_hits),
            "full_path_raw_cache_hits": int(self.full_path_raw_cache_hits),
        }

    def double_head_diagnostics(self) -> Dict[str, Any]:
        return {
            "target": "oracle_fixed_ratio_schedule_double_head_q",
            "summary": self.summary(),
        }


def trainer_kwargs(device: torch.device, gpu_id: int) -> Dict:
    if device.type == "cuda":
        return {
            "accelerator": "gpu",
            "devices": [gpu_id],
            "precision": 16,
            "num_sanity_val_steps": 0,
        }
    return {
        "accelerator": "cpu",
        "devices": 1,
        "precision": 32,
        "num_sanity_val_steps": 0,
    }


def load_best_checkpoint_if_available(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("state_dict", None)
        if state_dict is not None:
            model.load_state_dict(state_dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_jsonable)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _file_identity(path: Path, *, content_hash: bool) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    out: Dict[str, Any] = {"path": str(resolved), "exists": bool(resolved.exists())}
    if not resolved.exists():
        return out
    stat = resolved.stat()
    out.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    if content_hash:
        digest = hashlib.sha256()
        with resolved.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        out["sha256"] = digest.hexdigest()
    return out


def _torch_load(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def build_candidate_model(
    cfg: TrainConfig,
    d_in: int,
    d_out: int,
    group_matrix: torch.Tensor,
    device: torch.device,
    max_features_train: int,
    eps_steps: int,
    min_lr: float,
    class_weights: Optional[torch.Tensor],
    oracle_target_provider: OracleDoubleHeadQTargetProvider,
    prerequisite_matrix: torch.Tensor,
    full_path_head_loss_weight: float,
    proposal_top_k: int,
    intervention_aux_enabled: bool,
    intervention_aux_weight: float,
    intervention_aux_only_changed_actions: bool,
    intervention_aux_mode: str,
    intervention_aux_oracle_margin: float,
    predictor_hidden: int,
    predictor_dropout: float,
    value_hidden: int,
    value_dropout: float,
) -> CMIEstimator:
    num_groups = group_matrix.shape[0]
    pred_hidden = int(cfg.hidden if int(predictor_hidden) <= 0 else predictor_hidden)
    pred_dropout = float(cfg.dropout if float(predictor_dropout) < 0.0 else predictor_dropout)
    val_hidden = int(cfg.hidden if int(value_hidden) <= 0 else value_hidden)
    val_dropout = float(cfg.dropout if float(value_dropout) < 0.0 else value_dropout)
    predictor = get_mlp_network(d_in + num_groups, d_out, hidden=pred_hidden, dropout=pred_dropout)
    value_network = get_mlp_network(d_in + num_groups, num_groups * 2, hidden=val_hidden, dropout=val_dropout)
    mask_layer = MaskLayerGrouped(group_matrix=group_matrix, append=True)
    acc_metric = Accuracy(task="multiclass", num_classes=d_out)
    ce_loss_none = DTypeSafeCrossEntropyLoss(weight=class_weights, reduction="none")
    model = DoubleHeadOracleQEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        lr=cfg.lr,
        min_lr=min_lr,
        max_features=max_features_train,
        eps=cfg.eps,
        loss_fn=ce_loss_none,
        val_loss_fn=acc_metric,
        eps_decay=cfg.eps_decay,
        eps_steps=eps_steps,
        patience=cfg.patience,
        feature_costs=None,
        cmi_scaling=cfg.cmi_scaling,
        oracle_double_target_fn=oracle_target_provider.double_head_targets,
        full_path_loss_weight=full_path_head_loss_weight,
        proposal_top_k=proposal_top_k,
        one_step_prefix_steps=int(oracle_target_provider.one_step_prefix_steps),
        full_path_middle_steps=int(oracle_target_provider.full_path_middle_steps),
        prerequisite_matrix=prerequisite_matrix.detach().cpu(),
        intervention_aux_enabled=intervention_aux_enabled,
        intervention_aux_weight=intervention_aux_weight,
        intervention_aux_only_changed_actions=intervention_aux_only_changed_actions,
        intervention_aux_mode=intervention_aux_mode,
        intervention_aux_oracle_margin=intervention_aux_oracle_margin,
    )
    model.to(device)
    return model


def _candidate_cache_dir(cache_root: Path, candidate_meta: Dict[str, Any]) -> Tuple[Path, str]:
    cache_hash = _stable_hash(candidate_meta)
    return cache_root / cache_hash[:2] / cache_hash, cache_hash


def save_candidate_cache(
    cache_root: Path,
    candidate_meta: Dict[str, Any],
    model: torch.nn.Module,
    train_val_score: float,
    val_constraint_metrics: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    cache_dir, cache_hash = _candidate_cache_dir(cache_root, candidate_meta)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "candidate_model.pt"
    tmp_model_path = cache_dir / "candidate_model.pt.tmp"
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"state_dict": state_dict, "cache_hash": cache_hash}, tmp_model_path)
    tmp_model_path.replace(model_path)
    meta_payload = {
        "cache_hash": cache_hash,
        "cache_schema": "candidate_model_cache",
        "candidate_meta": candidate_meta,
        "train_val_score": float(train_val_score),
        "val_constraint_metrics": val_constraint_metrics,
        "source": str(source),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    _atomic_write_json(cache_dir / "candidate_meta.json", meta_payload)
    _atomic_write_text(cache_dir / "COMPLETE", cache_hash)
    return {"enabled": True, "hit": False, "saved": True, "source": source, "cache_hash": cache_hash, "cache_dir": str(cache_dir)}


def load_candidate_cache(
    cache_root: Path,
    candidate_meta: Dict[str, Any],
    *,
    cfg: TrainConfig,
    d_in: int,
    d_out: int,
    group_matrix: torch.Tensor,
    device: torch.device,
    max_features_train: int,
    eps_steps: int,
    min_lr: float,
    class_weights: Optional[torch.Tensor],
    oracle_target_provider: OracleDoubleHeadQTargetProvider,
    prerequisite_matrix: torch.Tensor,
    full_path_head_loss_weight: float,
    proposal_top_k: int,
    intervention_aux_enabled: bool,
    intervention_aux_weight: float,
    intervention_aux_only_changed_actions: bool,
    intervention_aux_mode: str,
    intervention_aux_oracle_margin: float,
    predictor_hidden: int,
    predictor_dropout: float,
    value_hidden: int,
    value_dropout: float,
) -> Optional[Tuple[CMIEstimator, float, Dict[str, Any], Dict[str, Any]]]:
    cache_dir, cache_hash = _candidate_cache_dir(cache_root, candidate_meta)
    complete_path = cache_dir / "COMPLETE"
    meta_path = cache_dir / "candidate_meta.json"
    model_path = cache_dir / "candidate_model.pt"
    if not (complete_path.exists() and meta_path.exists() and model_path.exists()):
        return None
    try:
        if complete_path.read_text(encoding="utf-8").strip() != cache_hash:
            return None
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        payload_candidate_meta = payload.get("candidate_meta")
        if payload.get("cache_hash") != cache_hash or _stable_hash(payload_candidate_meta) != cache_hash:
            return None
        ckpt = _torch_load(model_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        model = build_candidate_model(
            cfg,
            d_in,
            d_out,
            group_matrix,
            device,
            max_features_train,
            eps_steps,
            min_lr,
            class_weights,
            oracle_target_provider,
            prerequisite_matrix,
            full_path_head_loss_weight,
            proposal_top_k,
            intervention_aux_enabled,
            intervention_aux_weight,
            intervention_aux_only_changed_actions,
            intervention_aux_mode,
            intervention_aux_oracle_margin,
            predictor_hidden,
            predictor_dropout,
            value_hidden,
            value_dropout,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        cache_info = {
            "enabled": True,
            "hit": True,
            "saved": False,
            "source": "candidate_cache",
            "cache_hash": cache_hash,
            "cache_dir": str(cache_dir),
        }
        return model, float(payload.get("train_val_score", float("-inf"))), payload.get("val_constraint_metrics", {}), cache_info
    except Exception as exc:
        print(f"[WARN] candidate cache ignored: {cache_dir} ({exc})")
        return None


def _extract_lightning_best_score(ckpt_path: Path) -> float:
    try:
        ckpt = _torch_load(ckpt_path, map_location="cpu")
        callbacks = ckpt.get("callbacks", {})
        if isinstance(callbacks, dict):
            for item in callbacks.values():
                if isinstance(item, dict) and item.get("best_model_score") is not None:
                    score = item["best_model_score"]
                    return float(score.item() if hasattr(score, "item") else score)
    except Exception:
        pass
    return float("-inf")


def load_candidate_from_existing_ckpt(
    ckpt_dir: Path,
    *,
    cfg: TrainConfig,
    d_in: int,
    d_out: int,
    group_matrix: torch.Tensor,
    device: torch.device,
    max_features_train: int,
    eps_steps: int,
    min_lr: float,
    class_weights: Optional[torch.Tensor],
    oracle_target_provider: OracleDoubleHeadQTargetProvider,
    prerequisite_matrix: torch.Tensor,
    full_path_head_loss_weight: float,
    proposal_top_k: int,
    intervention_aux_enabled: bool,
    intervention_aux_weight: float,
    intervention_aux_only_changed_actions: bool,
    intervention_aux_mode: str,
    intervention_aux_oracle_margin: float,
    predictor_hidden: int,
    predictor_dropout: float,
    value_hidden: int,
    value_dropout: float,
) -> Optional[Tuple[CMIEstimator, float, Dict[str, Any]]]:
    if not ckpt_dir.exists():
        return None
    ckpt_candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
    if not ckpt_candidates:
        return None
    ckpt_path = ckpt_candidates[0]
    try:
        model = build_candidate_model(
            cfg,
            d_in,
            d_out,
            group_matrix,
            device,
            max_features_train,
            eps_steps,
            min_lr,
            class_weights,
            oracle_target_provider,
            prerequisite_matrix,
            full_path_head_loss_weight,
            proposal_top_k,
            intervention_aux_enabled,
            intervention_aux_weight,
            intervention_aux_only_changed_actions,
            intervention_aux_mode,
            intervention_aux_oracle_margin,
            predictor_hidden,
            predictor_dropout,
            value_hidden,
            value_dropout,
        )
        load_best_checkpoint_if_available(model, str(ckpt_path), device)
        model.to(device).eval()
        info = {"enabled": True, "hit": True, "saved": False, "source": "existing_tmp_checkpoint", "ckpt_path": str(ckpt_path)}
        return model, _extract_lightning_best_score(ckpt_path), info
    except Exception as exc:
        print(f"[WARN] existing checkpoint ignored: {ckpt_dir} ({exc})")
        return None


def predict_policy_scores(
    cmi_model: CMIEstimator,
    x: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    prerequisite_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hasattr(cmi_model, "predict_policy_scores"):
        return cmi_model.predict_policy_scores(
            x,
            mask,
            pred,
            prerequisite_matrix=prerequisite_matrix,
        )

    x_masked = cmi_model.mask_layer(x, mask)
    if cmi_model.cmi_scaling == "bounded":
        entropy = get_entropy(pred).unsqueeze(1)
        pred_cmi = cmi_model.value_network(x_masked).sigmoid() * entropy
    elif cmi_model.cmi_scaling == "positive":
        pred_cmi = torch.nn.functional.softplus(cmi_model.value_network(x_masked))
    else:
        pred_cmi = cmi_model.value_network(x_masked)

    scores = pred_cmi / cmi_model.feature_costs
    if prerequisite_matrix is None:
        scores = scores - 1e6 * mask
    else:
        scores = mask_illegal_action_logits(scores, mask, prerequisite_matrix)
    return scores


def select_next_feature(
    cmi_model: CMIEstimator,
    x: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    prerequisite_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    scores = predict_policy_scores(
        cmi_model,
        x,
        mask,
        pred,
        prerequisite_matrix=prerequisite_matrix,
    )
    best_feature_index = torch.argmax(scores, dim=1)
    return torch.max(mask, ind_to_onehot(best_feature_index, cmi_model.mask_size))


def select_next_feature_with_index(
    cmi_model: CMIEstimator,
    x: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    prerequisite_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scores = predict_policy_scores(
        cmi_model,
        x,
        mask,
        pred,
        prerequisite_matrix=prerequisite_matrix,
    )
    best_feature_index = torch.argmax(scores, dim=1)
    return torch.max(mask, ind_to_onehot(best_feature_index, cmi_model.mask_size)), best_feature_index


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    clean = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=np.float64)
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    percentiles = np.percentile(clean, [10, 25, 50, 75, 90, 95])
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "max": float(np.max(clean)),
    }


def _rankdata_average(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    ranks = np.zeros(arr.shape[0], dtype=np.float64)
    if arr.size == 0:
        return ranks
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    start = 0
    while start < arr.size:
        end = start + 1
        while end < arr.size and sorted_arr[end] == sorted_arr[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _safe_corr(left: List[float], right: List[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(left: List[float], right: List[float]) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or y.size < 2:
        return None
    return _safe_corr(_rankdata_average(x.tolist()).tolist(), _rankdata_average(y.tolist()).tolist())


def _top_action(scores: Dict[int, float]) -> Optional[int]:
    if not scores:
        return None
    return int(sorted(scores.keys(), key=lambda idx: (-float(scores[int(idx)]), int(idx)))[0])


def _top1_top2_gap(scores: Dict[int, float]) -> Optional[float]:
    values = [float(v) for v in scores.values() if np.isfinite(float(v))]
    if len(values) < 2:
        return None
    values.sort(reverse=True)
    return float(values[0] - values[1])


def _rate(records: List[Dict[str, Any]], key: str) -> Optional[float]:
    if not records:
        return None
    return float(sum(bool(row.get(key)) for row in records) / len(records))


STUDENT_OUTCOME_DECISIVE_CATEGORIES = (
    "A_oracle_full_student_full",
    "B_oracle_full_student_one",
    "C_oracle_one_student_full",
    "D_oracle_one_student_one",
)


def _category_counts(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in records:
        value = row.get(key)
        if value is None:
            continue
        category = str(value)
        counts[category] = int(counts.get(category, 0) + 1)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _counts_to_rates(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if int(total) <= 0:
        return {}
    return {
        str(category): float(count / int(total))
        for category, count in sorted(counts.items(), key=lambda item: item[0])
    }


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _numeric_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in records:
        value = _finite_float(row.get(key))
        if value is not None:
            values.append(float(value))
    return values


def _action_descriptor(action_groups: List[Dict[str, Any]], action_idx: int) -> Dict[str, Any]:
    idx = int(action_idx)
    if 0 <= idx < len(action_groups):
        item = action_groups[idx]
        return {
            "action_index": int(idx),
            "action_id": str(item.get("action_id", f"action_{idx}")),
            "action_name": str(item.get("name", item.get("action_name", f"action_{idx}"))),
            "features": [str(x) for x in item.get("feature", item.get("features", []))],
        }
    return {
        "action_index": int(idx),
        "action_id": f"action_{idx}",
        "action_name": f"action_{idx}",
        "features": [],
    }


def _student_hard_outcome(row: Dict[str, Any]) -> str:
    if bool(row.get("one_full_agreement")):
        return "same_action"
    one_correct = bool(int(row.get("student_hard_acc_after_one", 0)) == 1)
    full_correct = bool(int(row.get("student_hard_acc_after_full", 0)) == 1)
    if not one_correct and full_correct:
        return "win"
    if one_correct and not full_correct:
        return "loss"
    if one_correct and full_correct:
        return "both_correct"
    return "both_wrong"


def _student_outcome_category(
    *,
    oracle_delta: float,
    student_delta: float,
    same_action: bool,
    eps: float = 1.0e-8,
) -> str:
    if bool(same_action):
        return "same_action"
    oracle_delta = float(oracle_delta)
    student_delta = float(student_delta)
    if not np.isfinite(oracle_delta) or not np.isfinite(student_delta):
        return "tie_or_ambiguous"
    if abs(oracle_delta) <= eps or abs(student_delta) <= eps:
        return "tie_or_ambiguous"
    if oracle_delta > eps and student_delta > eps:
        return "A_oracle_full_student_full"
    if oracle_delta > eps and student_delta < -eps:
        return "B_oracle_full_student_one"
    if oracle_delta < -eps and student_delta > eps:
        return "C_oracle_one_student_full"
    return "D_oracle_one_student_one"


def _summarize_rerank_record_group(
    records: List[Dict[str, Any]],
    *,
    proposal_recall_top_ks: List[int],
) -> Dict[str, Any]:
    top_ks = sorted({int(k) for k in proposal_recall_top_ks if int(k) > 0})
    if not records:
        empty = {
            "state_count": 0,
            "proposal_candidate_count": _numeric_summary([]),
            "legal_candidate_count": _numeric_summary([]),
            "oracle_target": _numeric_summary([]),
            "oracle_target_std_per_state": _numeric_summary([]),
            "oracle_top1_top2_gap": _numeric_summary([]),
            "oracle_top1_top2_gap_all_legal": _numeric_summary([]),
            "full_model_top1_top2_gap": _numeric_summary([]),
            "full_vs_oracle_pearson": _numeric_summary([]),
            "full_vs_oracle_spearman": _numeric_summary([]),
            "full_top1_matches_oracle_rate": None,
            "one_step_top1_matches_oracle_rate": None,
            "one_full_agreement_rate": None,
            "one_step_rank_of_full_oracle_top1": _numeric_summary([]),
            "policy_proposal_contains_full_oracle_top1_rate": None,
            "proposal_recall_regret": _numeric_summary([]),
            "proposal_recall_regret_positive_rate": None,
            "full_oracle_best_score_all_legal": _numeric_summary([]),
            "full_oracle_best_score_inside_policy_proposal": _numeric_summary([]),
            "oracle_available_gain_vs_one_step": _numeric_summary([]),
            "oracle_available_gain_all_legal_vs_one_step": _numeric_summary([]),
            "full_rerank_gain_vs_one_step": _numeric_summary([]),
            "full_rerank_regret_vs_oracle": _numeric_summary([]),
            "full_rerank_win_rate": None,
            "full_rerank_tie_rate": None,
            "full_rerank_loss_rate": None,
            "student_outcome_category_counts": {},
            "student_outcome_category_rates": {},
            "student_outcome_decisive_count": 0,
            "student_outcome_decisive_rate": None,
            "student_outcome_hard_category_counts": {},
            "student_outcome_hard_category_rates": {},
            "student_true_prob_after_one": _numeric_summary([]),
            "student_true_prob_after_full": _numeric_summary([]),
            "student_true_prob_delta_full_minus_one": _numeric_summary([]),
            "student_confidence_after_one": _numeric_summary([]),
            "student_confidence_after_full": _numeric_summary([]),
            "student_confidence_delta_full_minus_one": _numeric_summary([]),
            "student_entropy_after_one": _numeric_summary([]),
            "student_entropy_after_full": _numeric_summary([]),
            "student_entropy_delta_full_minus_one": _numeric_summary([]),
            "student_hard_acc_after_one_rate": None,
            "student_hard_acc_after_full_rate": None,
            "student_hard_acc_delta_full_minus_one": _numeric_summary([]),
            "oracle_delta_full_minus_one": _numeric_summary([]),
            "pooled_full_vs_oracle_pearson": None,
            "pooled_full_vs_oracle_spearman": None,
        }
        for k in top_ks:
            empty[f"full_oracle_top1_in_one_step_top{k}_rate"] = None
            empty[f"full_oracle_best_inside_one_step_top{k}_regret"] = _numeric_summary([])
        return empty

    oracle_values_all: List[float] = []
    full_values_all: List[float] = []
    for row in records:
        oracle_scores = {int(k): float(v) for k, v in row.get("oracle_full_path_targets", {}).items()}
        full_scores = {int(k): float(v) for k, v in row.get("full_path_model_scores", {}).items()}
        for action_idx in sorted(set(oracle_scores.keys()) & set(full_scores.keys())):
            oracle_values_all.append(float(oracle_scores[int(action_idx)]))
            full_values_all.append(float(full_scores[int(action_idx)]))

    student_category_counts = _category_counts(records, "student_outcome_category")
    student_hard_category_counts = _category_counts(records, "student_outcome_hard_category")
    student_decisive_count = int(
        sum(row.get("student_outcome_category") in STUDENT_OUTCOME_DECISIVE_CATEGORIES for row in records)
    )

    result = {
        "state_count": int(len(records)),
        "proposal_candidate_count": _numeric_summary([float(row["proposal_candidate_count"]) for row in records]),
        "legal_candidate_count": _numeric_summary([float(row["legal_candidate_count"]) for row in records]),
        "oracle_target": _numeric_summary(oracle_values_all),
        "oracle_target_std_per_state": _numeric_summary([float(row["oracle_target_std"]) for row in records]),
        "oracle_top1_top2_gap": _numeric_summary(
            [float(row["oracle_top1_top2_gap"]) for row in records if row.get("oracle_top1_top2_gap") is not None]
        ),
        "oracle_top1_top2_gap_all_legal": _numeric_summary(
            [
                float(row["oracle_top1_top2_gap_all_legal"])
                for row in records
                if row.get("oracle_top1_top2_gap_all_legal") is not None
            ]
        ),
        "full_model_top1_top2_gap": _numeric_summary(
            [float(row["full_model_top1_top2_gap"]) for row in records if row.get("full_model_top1_top2_gap") is not None]
        ),
        "full_vs_oracle_pearson": _numeric_summary(
            [float(row["full_vs_oracle_pearson"]) for row in records if row.get("full_vs_oracle_pearson") is not None]
        ),
        "full_vs_oracle_spearman": _numeric_summary(
            [float(row["full_vs_oracle_spearman"]) for row in records if row.get("full_vs_oracle_spearman") is not None]
        ),
        "full_top1_matches_oracle_rate": _rate(records, "full_top1_matches_oracle"),
        "one_step_top1_matches_oracle_rate": _rate(records, "one_step_top1_matches_oracle"),
        "one_full_agreement_rate": _rate(records, "one_full_agreement"),
        "one_step_rank_of_full_oracle_top1": _numeric_summary(
            [float(row["one_step_rank_of_full_oracle_top1"]) for row in records]
        ),
        "policy_proposal_contains_full_oracle_top1_rate": _rate(
            records,
            "policy_proposal_contains_full_oracle_top1",
        ),
        "proposal_recall_regret": _numeric_summary([float(row["proposal_recall_regret"]) for row in records]),
        "proposal_recall_regret_positive_rate": float(
            sum(float(row["proposal_recall_regret"]) > 1.0e-8 for row in records) / len(records)
        ),
        "full_oracle_best_score_all_legal": _numeric_summary(
            [float(row["full_oracle_best_score_all_legal"]) for row in records]
        ),
        "full_oracle_best_score_inside_policy_proposal": _numeric_summary(
            [float(row["full_oracle_best_score_inside_policy_proposal"]) for row in records]
        ),
        "oracle_available_gain_vs_one_step": _numeric_summary(
            [float(row["oracle_available_gain_vs_one_step"]) for row in records]
        ),
        "oracle_available_gain_all_legal_vs_one_step": _numeric_summary(
            [float(row["oracle_available_gain_all_legal_vs_one_step"]) for row in records]
        ),
        "full_rerank_gain_vs_one_step": _numeric_summary(
            [float(row["full_rerank_gain_vs_one_step"]) for row in records]
        ),
        "full_rerank_regret_vs_oracle": _numeric_summary(
            [float(row["full_rerank_regret_vs_oracle"]) for row in records]
        ),
        "full_rerank_win_rate": float(sum(row.get("full_rerank_outcome") == "win" for row in records) / len(records)),
        "full_rerank_tie_rate": float(sum(row.get("full_rerank_outcome") == "tie" for row in records) / len(records)),
        "full_rerank_loss_rate": float(sum(row.get("full_rerank_outcome") == "loss" for row in records) / len(records)),
        "student_outcome_category_counts": student_category_counts,
        "student_outcome_category_rates": _counts_to_rates(student_category_counts, len(records)),
        "student_outcome_decisive_count": int(student_decisive_count),
        "student_outcome_decisive_rate": float(student_decisive_count / len(records)),
        "student_outcome_hard_category_counts": student_hard_category_counts,
        "student_outcome_hard_category_rates": _counts_to_rates(student_hard_category_counts, len(records)),
        "student_true_prob_after_one": _numeric_summary(
            [
                float(row["student_true_prob_after_one"])
                for row in records
                if row.get("student_true_prob_after_one") is not None
            ]
        ),
        "student_true_prob_after_full": _numeric_summary(
            [
                float(row["student_true_prob_after_full"])
                for row in records
                if row.get("student_true_prob_after_full") is not None
            ]
        ),
        "student_true_prob_delta_full_minus_one": _numeric_summary(
            [
                float(row["student_true_prob_delta_full_minus_one"])
                for row in records
                if row.get("student_true_prob_delta_full_minus_one") is not None
            ]
        ),
        "student_confidence_after_one": _numeric_summary(_numeric_values(records, "student_confidence_after_one")),
        "student_confidence_after_full": _numeric_summary(_numeric_values(records, "student_confidence_after_full")),
        "student_confidence_delta_full_minus_one": _numeric_summary(
            _numeric_values(records, "student_confidence_delta_full_minus_one")
        ),
        "student_entropy_after_one": _numeric_summary(_numeric_values(records, "student_entropy_after_one")),
        "student_entropy_after_full": _numeric_summary(_numeric_values(records, "student_entropy_after_full")),
        "student_entropy_delta_full_minus_one": _numeric_summary(
            _numeric_values(records, "student_entropy_delta_full_minus_one")
        ),
        "student_hard_acc_after_one_rate": float(
            sum(bool(row.get("student_hard_acc_after_one")) for row in records) / len(records)
        ),
        "student_hard_acc_after_full_rate": float(
            sum(bool(row.get("student_hard_acc_after_full")) for row in records) / len(records)
        ),
        "student_hard_acc_delta_full_minus_one": _numeric_summary(
            [
                float(row["student_hard_acc_delta_full_minus_one"])
                for row in records
                if row.get("student_hard_acc_delta_full_minus_one") is not None
            ]
        ),
        "oracle_delta_full_minus_one": _numeric_summary(
            [float(row["oracle_delta_full_minus_one"]) for row in records if row.get("oracle_delta_full_minus_one") is not None]
        ),
        "pooled_full_vs_oracle_pearson": _safe_corr(full_values_all, oracle_values_all),
        "pooled_full_vs_oracle_spearman": _safe_spearman(full_values_all, oracle_values_all),
    }
    for category in STUDENT_OUTCOME_DECISIVE_CATEGORIES:
        result[f"{category}_rate"] = float(student_category_counts.get(category, 0) / len(records))
        result[f"{category}_decisive_rate"] = (
            float(student_category_counts.get(category, 0) / student_decisive_count)
            if student_decisive_count > 0
            else None
        )
    for k in top_ks:
        result[f"full_oracle_top1_in_one_step_top{k}_rate"] = _rate(
            records,
            f"full_oracle_top1_in_one_step_top{k}",
        )
        result[f"full_oracle_best_inside_one_step_top{k}_regret"] = _numeric_summary(
            [float(row[f"full_oracle_best_inside_one_step_top{k}_regret"]) for row in records]
        )
    return result


def summarize_full_path_rerank_diagnostics(
    records: List[Dict[str, Any]],
    *,
    enabled: bool,
    max_states: int,
    include_records: bool,
    proposal_recall_top_ks: List[int],
) -> Dict[str, Any]:
    top_ks = sorted({int(k) for k in proposal_recall_top_ks if int(k) > 0})
    by_step_records: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        by_step_records.setdefault(str(int(row["step_idx"])), []).append(row)
    result = {
        "target": "diagnostic_intervention_outcome_rerank_diagnostics",
        "enabled": bool(enabled),
        "record_granularity": "test_rollout_middle_state",
        "proposal_recall_top_ks": [int(k) for k in top_ks],
        "max_states": None if int(max_states) <= 0 else int(max_states),
        "capped": bool(int(max_states) > 0 and len(records) >= int(max_states)),
        "summary": _summarize_rerank_record_group(records, proposal_recall_top_ks=top_ks),
        "by_step": {
            step: _summarize_rerank_record_group(step_records, proposal_recall_top_ks=top_ks)
            for step, step_records in sorted(by_step_records.items(), key=lambda item: int(item[0]))
        },
    }
    if include_records:
        result["records"] = records
    return result


def _summarize_intervention_record_group(
    records: List[Dict[str, Any]],
    *,
    action_groups: List[Dict[str, Any]],
    top_transition_k: int,
) -> Dict[str, Any]:
    total = int(len(records))
    if total <= 0:
        return {
            "state_count": 0,
            "same_action_count": 0,
            "changed_action_count": 0,
            "same_action_rate": None,
            "changed_action_rate": None,
            "changed_hard_outcome_counts": {},
            "changed_hard_outcome_rates": {},
            "changed_hard_flip_rate": None,
            "changed_same_hard_rate": None,
            "changed_net_win": 0,
            "changed_student_true_prob_delta": _numeric_summary([]),
            "changed_student_confidence_delta": _numeric_summary([]),
            "changed_student_entropy_delta": _numeric_summary([]),
            "changed_oracle_delta": _numeric_summary([]),
            "top_action_transitions": [],
        }

    same_action_records = [row for row in records if bool(row.get("one_full_agreement"))]
    changed_records = [row for row in records if not bool(row.get("one_full_agreement"))]
    changed_count = int(len(changed_records))
    same_count = int(len(same_action_records))
    hard_outcomes = [_student_hard_outcome(row) for row in changed_records]
    hard_counts: Dict[str, int] = {}
    for category in hard_outcomes:
        hard_counts[category] = int(hard_counts.get(category, 0) + 1)
    hard_counts = dict(sorted(hard_counts.items(), key=lambda item: item[0]))
    win_count = int(hard_counts.get("win", 0))
    loss_count = int(hard_counts.get("loss", 0))
    both_correct_count = int(hard_counts.get("both_correct", 0))
    both_wrong_count = int(hard_counts.get("both_wrong", 0))
    hard_flip_count = int(win_count + loss_count)
    same_hard_count = int(both_correct_count + both_wrong_count)

    outcome_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row, category in zip(changed_records, hard_outcomes):
        outcome_groups.setdefault(category, []).append(row)

    transition_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in changed_records:
        one_action = int(row.get("one_step_top1_action", -1))
        full_action = int(row.get("full_path_top1_action", -1))
        transition_groups.setdefault((one_action, full_action), []).append(row)

    transition_rows: List[Dict[str, Any]] = []
    for (one_action, full_action), group in transition_groups.items():
        group_outcomes = [_student_hard_outcome(row) for row in group]
        group_counts: Dict[str, int] = {}
        for category in group_outcomes:
            group_counts[category] = int(group_counts.get(category, 0) + 1)
        group_win = int(group_counts.get("win", 0))
        group_loss = int(group_counts.get("loss", 0))
        transition_rows.append(
            {
                "one_step_action": _action_descriptor(action_groups, one_action),
                "full_path_action": _action_descriptor(action_groups, full_action),
                "count": int(len(group)),
                "rate_among_changed": float(len(group) / changed_count) if changed_count else None,
                "hard_outcome_counts": dict(sorted(group_counts.items(), key=lambda item: item[0])),
                "win_count": int(group_win),
                "loss_count": int(group_loss),
                "net_win": int(group_win - group_loss),
                "student_true_prob_delta": _numeric_summary(
                    _numeric_values(group, "student_true_prob_delta_full_minus_one")
                ),
                "student_confidence_delta": _numeric_summary(
                    _numeric_values(group, "student_confidence_delta_full_minus_one")
                ),
                "student_entropy_delta": _numeric_summary(
                    _numeric_values(group, "student_entropy_delta_full_minus_one")
                ),
                "oracle_delta": _numeric_summary(_numeric_values(group, "oracle_delta_full_minus_one")),
            }
        )
    transition_rows.sort(key=lambda row: (-int(row["count"]), -abs(int(row["net_win"])), str(row["one_step_action"]["action_id"])))
    if int(top_transition_k) > 0:
        transition_rows = transition_rows[: int(top_transition_k)]

    return {
        "state_count": int(total),
        "same_action_count": int(same_count),
        "changed_action_count": int(changed_count),
        "same_action_rate": float(same_count / total),
        "changed_action_rate": float(changed_count / total),
        "changed_hard_outcome_counts": hard_counts,
        "changed_hard_outcome_rates": _counts_to_rates(hard_counts, changed_count),
        "changed_win_count": int(win_count),
        "changed_loss_count": int(loss_count),
        "changed_both_correct_count": int(both_correct_count),
        "changed_both_wrong_count": int(both_wrong_count),
        "changed_net_win": int(win_count - loss_count),
        "changed_hard_flip_count": int(hard_flip_count),
        "changed_hard_flip_rate": float(hard_flip_count / changed_count) if changed_count else None,
        "changed_same_hard_count": int(same_hard_count),
        "changed_same_hard_rate": float(same_hard_count / changed_count) if changed_count else None,
        "changed_both_correct_rate": float(both_correct_count / changed_count) if changed_count else None,
        "changed_both_wrong_rate": float(both_wrong_count / changed_count) if changed_count else None,
        "changed_student_true_prob_delta": _numeric_summary(
            _numeric_values(changed_records, "student_true_prob_delta_full_minus_one")
        ),
        "changed_student_confidence_delta": _numeric_summary(
            _numeric_values(changed_records, "student_confidence_delta_full_minus_one")
        ),
        "changed_student_entropy_delta": _numeric_summary(
            _numeric_values(changed_records, "student_entropy_delta_full_minus_one")
        ),
        "changed_oracle_delta": _numeric_summary(_numeric_values(changed_records, "oracle_delta_full_minus_one")),
        "same_action_student_true_prob_delta": _numeric_summary(
            _numeric_values(same_action_records, "student_true_prob_delta_full_minus_one")
        ),
        "outcome_groups": {
            category: {
                "count": int(len(group)),
                "rate_among_changed": float(len(group) / changed_count) if changed_count else None,
                "student_true_prob_delta": _numeric_summary(
                    _numeric_values(group, "student_true_prob_delta_full_minus_one")
                ),
                "student_confidence_delta": _numeric_summary(
                    _numeric_values(group, "student_confidence_delta_full_minus_one")
                ),
                "student_entropy_delta": _numeric_summary(
                    _numeric_values(group, "student_entropy_delta_full_minus_one")
                ),
                "oracle_delta": _numeric_summary(_numeric_values(group, "oracle_delta_full_minus_one")),
            }
            for category, group in sorted(outcome_groups.items(), key=lambda item: item[0])
        },
        "top_action_transitions": transition_rows,
    }


def summarize_intervention_outcome_sensitivity(
    records: List[Dict[str, Any]],
    *,
    action_groups: List[Dict[str, Any]],
    enabled: bool,
    top_transition_k: int,
) -> Dict[str, Any]:
    by_step_records: Dict[str, List[Dict[str, Any]]] = {}
    for row in records:
        by_step_records.setdefault(str(int(row["step_idx"])), []).append(row)
    return {
        "target": "diagnostic_intervention_outcome_sensitivity_diagnostics",
        "enabled": bool(enabled),
        "record_granularity": "test_rollout_middle_state",
        "top_transition_k": int(top_transition_k),
        "summary": _summarize_intervention_record_group(
            records,
            action_groups=action_groups,
            top_transition_k=top_transition_k,
        ),
        "by_step": {
            step: _summarize_intervention_record_group(
                step_records,
                action_groups=action_groups,
                top_transition_k=top_transition_k,
            )
            for step, step_records in sorted(by_step_records.items(), key=lambda item: int(item[0]))
        },
    }


def collect_full_path_rerank_diagnostics_for_batch(
    *,
    cmi_model: CMIEstimator,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    batch_sample_indices: np.ndarray,
    step_idx: int,
    prerequisite_matrix: Optional[torch.Tensor],
    oracle_target_provider: OracleDoubleHeadQTargetProvider,
    max_records: Optional[int],
    proposal_recall_top_ks: List[int],
) -> List[Dict[str, Any]]:
    if max_records is not None and int(max_records) <= 0:
        return []
    required_attrs = ["_predict_heads", "_legal_action_mask_with_matrix", "_use_full_path_for_step"]
    if not all(hasattr(cmi_model, name) for name in required_attrs):
        return []

    x_masked = cmi_model.mask_layer(x, mask)
    one_step_q, full_path_q = cmi_model._predict_heads(x_masked, pred)
    legal = cmi_model._legal_action_mask_with_matrix(mask, prerequisite_matrix)
    use_full_path = cmi_model._use_full_path_for_step(mask)
    one_step_scores = one_step_q / cmi_model.feature_costs
    full_path_scores = full_path_q / cmi_model.feature_costs

    mask_np = mask.detach().cpu().numpy()
    legal_np = legal.detach().cpu().numpy().astype(bool)
    use_full_np = use_full_path.detach().cpu().numpy().astype(bool)
    one_scores_np = one_step_scores.detach().cpu().numpy().astype(np.float64)
    full_scores_np = full_path_scores.detach().cpu().numpy().astype(np.float64)
    proposal_top_k = int(max(1, getattr(cmi_model, "proposal_top_k", oracle_target_provider.proposal_top_k)))
    recall_top_ks = sorted({int(k) for k in proposal_recall_top_ks if int(k) > 0})

    records: List[Dict[str, Any]] = []
    for row_idx in range(int(x.shape[0])):
        if max_records is not None and len(records) >= int(max_records):
            break
        if not bool(use_full_np[row_idx]):
            continue
        legal_actions = np.where(legal_np[row_idx])[0].astype(np.int64).tolist()
        if not legal_actions:
            continue

        one_step_model_scores = {int(idx): float(one_scores_np[row_idx, int(idx)]) for idx in legal_actions}
        one_step_sorted_actions = sorted(
            legal_actions,
            key=lambda idx: (-float(one_step_model_scores[int(idx)]), int(idx)),
        )
        proposal = [int(idx) for idx in one_step_sorted_actions[: min(proposal_top_k, len(one_step_sorted_actions))]]
        if not proposal:
            continue

        mask_tuple = tuple(int(v > 0.5) for v in mask_np[row_idx].tolist())
        selected_start = _selected_indices_from_mask(mask_tuple)
        sample_idx = int(batch_sample_indices[row_idx])

        full_path_model_scores = {int(idx): float(full_scores_np[row_idx, int(idx)]) for idx in proposal}
        one_step_proposal_scores = {int(idx): float(one_scores_np[row_idx, int(idx)]) for idx in proposal}
        oracle_scores_all_legal: Dict[int, float] = {}
        for action_idx in one_step_sorted_actions:
            raw_value = oracle_target_provider._forced_first_full_path_score(
                sample_idx=sample_idx,
                selected_start=selected_start,
                action_idx=int(action_idx),
            )
            oracle_scores_all_legal[int(action_idx)] = float(
                transform_oracle_q_value(
                    raw_value,
                    transform=oracle_target_provider.full_path_target_transform,
                )
            )
        oracle_scores: Dict[int, float] = {
            int(action_idx): float(oracle_scores_all_legal[int(action_idx)])
            for action_idx in proposal
            if int(action_idx) in oracle_scores_all_legal
        }

        one_top = _top_action(one_step_proposal_scores)
        full_top = _top_action(full_path_model_scores)
        oracle_top = _top_action(oracle_scores)
        oracle_top_all_legal = _top_action(oracle_scores_all_legal)
        if one_top is None or full_top is None or oracle_top is None or oracle_top_all_legal is None:
            continue

        oracle_values = [float(oracle_scores[int(idx)]) for idx in proposal]
        oracle_values_all_legal = [float(oracle_scores_all_legal[int(idx)]) for idx in one_step_sorted_actions]
        full_values = [float(full_path_model_scores[int(idx)]) for idx in proposal]
        one_target = float(oracle_scores[int(one_top)])
        full_target = float(oracle_scores[int(full_top)])
        oracle_best_target = float(oracle_scores[int(oracle_top)])
        full_oracle_best_score_all_legal = float(oracle_scores_all_legal[int(oracle_top_all_legal)])
        one_step_rank_map = {
            int(action_idx): int(rank_idx + 1)
            for rank_idx, action_idx in enumerate(one_step_sorted_actions)
        }
        oracle_top_all_rank = int(one_step_rank_map[int(oracle_top_all_legal)])
        proposal_best_target_all_basis = float(max(float(oracle_scores_all_legal[int(idx)]) for idx in proposal))
        proposal_recall_regret = float(full_oracle_best_score_all_legal - proposal_best_target_all_basis)
        full_gain = float(full_target - one_target)
        eps = 1.0e-8
        if full_gain > eps:
            outcome = "win"
        elif full_gain < -eps:
            outcome = "loss"
        else:
            outcome = "tie"
        same_action = bool(int(one_top) == int(full_top))
        y_true = int(y[row_idx].detach().cpu().item())
        row_x = x[row_idx:row_idx + 1]
        one_mask_tensor = mask[row_idx:row_idx + 1].clone()
        full_mask_tensor = mask[row_idx:row_idx + 1].clone()
        one_mask_tensor[0, int(one_top)] = 1.0
        full_mask_tensor[0, int(full_top)] = 1.0
        one_logits = cmi_model.predictor(cmi_model.mask_layer(row_x, one_mask_tensor))
        full_logits = cmi_model.predictor(cmi_model.mask_layer(row_x, full_mask_tensor))
        one_probs = torch.softmax(one_logits, dim=1)
        full_probs = torch.softmax(full_logits, dim=1)
        one_pred_label = int(one_logits.argmax(dim=1).detach().cpu().item())
        full_pred_label = int(full_logits.argmax(dim=1).detach().cpu().item())
        student_confidence_after_one = float(one_probs.max(dim=1).values[0].detach().cpu().item())
        student_confidence_after_full = float(full_probs.max(dim=1).values[0].detach().cpu().item())
        student_confidence_delta = float(student_confidence_after_full - student_confidence_after_one)
        student_entropy_after_one = float(
            (-one_probs * torch.log(torch.clamp(one_probs, min=1.0e-12))).sum(dim=1)[0].detach().cpu().item()
        )
        student_entropy_after_full = float(
            (-full_probs * torch.log(torch.clamp(full_probs, min=1.0e-12))).sum(dim=1)[0].detach().cpu().item()
        )
        student_entropy_delta = float(student_entropy_after_full - student_entropy_after_one)
        if 0 <= y_true < int(one_probs.shape[1]):
            student_true_prob_after_one = float(one_probs[0, y_true].detach().cpu().item())
            student_true_prob_after_full = float(full_probs[0, y_true].detach().cpu().item())
        else:
            student_true_prob_after_one = float("nan")
            student_true_prob_after_full = float("nan")
        student_prob_delta = float(student_true_prob_after_full - student_true_prob_after_one)
        student_hard_after_one = int(one_pred_label == y_true)
        student_hard_after_full = int(full_pred_label == y_true)
        student_hard_delta = int(student_hard_after_full - student_hard_after_one)
        oracle_delta = float(full_target - one_target)
        student_outcome_category = _student_outcome_category(
            oracle_delta=oracle_delta,
            student_delta=student_prob_delta,
            same_action=same_action,
            eps=eps,
        )
        student_outcome_hard_category = _student_outcome_category(
            oracle_delta=oracle_delta,
            student_delta=float(student_hard_delta),
            same_action=same_action,
            eps=eps,
        )
        topk_fields: Dict[str, Any] = {}
        for k in recall_top_ks:
            top_actions = [int(idx) for idx in one_step_sorted_actions[: min(int(k), len(one_step_sorted_actions))]]
            if top_actions:
                best_inside_topk = float(max(float(oracle_scores_all_legal[int(idx)]) for idx in top_actions))
            else:
                best_inside_topk = float("nan")
            topk_fields[f"full_oracle_top1_in_one_step_top{k}"] = bool(
                int(oracle_top_all_legal) in set(top_actions)
            )
            topk_fields[f"full_oracle_best_inside_one_step_top{k}"] = float(best_inside_topk)
            topk_fields[f"full_oracle_best_inside_one_step_top{k}_regret"] = float(
                full_oracle_best_score_all_legal - best_inside_topk
            )

        records.append(
            {
                "sample_idx": sample_idx,
                "step_idx": int(step_idx),
                "acquisition_step": int(step_idx) + 1,
                "selected_actions_before_step": [int(x) for x in selected_start],
                "proposal_candidate_count": int(len(proposal)),
                "legal_candidate_count": int(len(one_step_sorted_actions)),
                "proposal_top_k": int(proposal_top_k),
                "proposal_recall_top_ks": [int(k) for k in recall_top_ks],
                "proposal_actions": [int(x) for x in proposal],
                "one_step_ranked_legal_actions": [int(x) for x in one_step_sorted_actions],
                "one_step_model_scores": {str(int(k)): float(v) for k, v in sorted(one_step_proposal_scores.items())},
                "one_step_model_scores_all_legal": {
                    str(int(k)): float(v) for k, v in sorted(one_step_model_scores.items())
                },
                "full_path_model_scores": {str(int(k)): float(v) for k, v in sorted(full_path_model_scores.items())},
                "oracle_full_path_targets": {str(int(k)): float(v) for k, v in sorted(oracle_scores.items())},
                "oracle_full_path_targets_all_legal": {
                    str(int(k)): float(v) for k, v in sorted(oracle_scores_all_legal.items())
                },
                "one_step_top1_action": int(one_top),
                "full_path_top1_action": int(full_top),
                "oracle_top1_action": int(oracle_top),
                "full_oracle_top1_action_all_legal": int(oracle_top_all_legal),
                "oracle_top1_action_all_legal": int(oracle_top_all_legal),
                "one_step_rank_of_full_oracle_top1": int(oracle_top_all_rank),
                "policy_proposal_contains_full_oracle_top1": bool(int(oracle_top_all_legal) in set(proposal)),
                "one_full_agreement": same_action,
                "full_top1_matches_oracle": bool(int(full_top) == int(oracle_top)),
                "one_step_top1_matches_oracle": bool(int(one_top) == int(oracle_top)),
                "oracle_target_std": float(np.std(np.asarray(oracle_values, dtype=np.float64))),
                "oracle_target_std_all_legal": float(np.std(np.asarray(oracle_values_all_legal, dtype=np.float64))),
                "oracle_top1_top2_gap": _top1_top2_gap(oracle_scores),
                "oracle_top1_top2_gap_all_legal": _top1_top2_gap(oracle_scores_all_legal),
                "full_model_top1_top2_gap": _top1_top2_gap(full_path_model_scores),
                "full_vs_oracle_pearson": _safe_corr(full_values, oracle_values),
                "full_vs_oracle_spearman": _safe_spearman(full_values, oracle_values),
                "full_oracle_best_score_all_legal": float(full_oracle_best_score_all_legal),
                "full_oracle_best_score_inside_policy_proposal": float(proposal_best_target_all_basis),
                "proposal_recall_regret": float(proposal_recall_regret),
                "oracle_available_gain_vs_one_step": float(oracle_best_target - one_target),
                "oracle_available_gain_all_legal_vs_one_step": float(full_oracle_best_score_all_legal - one_target),
                "full_rerank_gain_vs_one_step": full_gain,
                "full_rerank_regret_vs_oracle": float(oracle_best_target - full_target),
                "full_rerank_outcome": outcome,
                "oracle_delta_full_minus_one": oracle_delta,
                "student_true_label": int(y_true),
                "student_pred_after_one": int(one_pred_label),
                "student_pred_after_full": int(full_pred_label),
                "student_true_prob_after_one": student_true_prob_after_one,
                "student_true_prob_after_full": student_true_prob_after_full,
                "student_true_prob_delta_full_minus_one": student_prob_delta,
                "student_confidence_after_one": student_confidence_after_one,
                "student_confidence_after_full": student_confidence_after_full,
                "student_confidence_delta_full_minus_one": student_confidence_delta,
                "student_entropy_after_one": student_entropy_after_one,
                "student_entropy_after_full": student_entropy_after_full,
                "student_entropy_delta_full_minus_one": student_entropy_delta,
                "student_hard_acc_after_one": int(student_hard_after_one),
                "student_hard_acc_after_full": int(student_hard_after_full),
                "student_hard_acc_delta_full_minus_one": int(student_hard_delta),
                "student_outcome_category": student_outcome_category,
                "student_outcome_hard_category": student_outcome_hard_category,
                "oracle_prefers_full": bool(oracle_delta > eps),
                "oracle_prefers_one": bool(oracle_delta < -eps),
                "student_prob_prefers_full": bool(student_prob_delta > eps),
                "student_prob_prefers_one": bool(student_prob_delta < -eps),
                **topk_fields,
            }
        )
    return records


@torch.no_grad()
def predict_with_exact_budget(
    cmi_model: CMIEstimator,
    x: torch.Tensor,
    budget: int,
    prerequisite_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = torch.zeros(len(x), cmi_model.mask_size, dtype=x.dtype, device=x.device)
    x_masked = cmi_model.mask_layer(x, mask)
    pred = cmi_model.predictor(x_masked)

    for _ in range(budget):
        mask = select_next_feature(cmi_model, x, mask, pred, prerequisite_matrix=prerequisite_matrix)
        x_masked = cmi_model.mask_layer(x, mask)
        pred = cmi_model.predictor(x_masked)
    return pred


@torch.no_grad()
def evaluate_acc_by_budget(
    cmi_model: CMIEstimator,
    loader: DataLoader,
    budget_list: List[int],
    device: torch.device,
    prerequisite_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    cmi_model.eval()
    acc_by_budget = {}
    for budget in tqdm(budget_list, desc="Feature budgets", dynamic_ncols=True, leave=False):
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = predict_with_exact_budget(
                cmi_model,
                x,
                budget,
                prerequisite_matrix=prerequisite_matrix,
            )
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == y).sum().item()
            total += y.numel()
        acc = float(correct / total) if total > 0 else 0.0
        acc_by_budget[str(budget)] = acc
    return acc_by_budget


@torch.no_grad()
def evaluate_action_group_constraint(
    cmi_model: CMIEstimator,
    loader: DataLoader,
    budget_list: List[int],
    device: torch.device,
    action_groups: List[Dict[str, Any]],
    sample_indices: np.ndarray,
    prerequisite_matrix: Optional[torch.Tensor] = None,
    oracle_target_provider: Optional[OracleDoubleHeadQTargetProvider] = None,
    enable_rerank_diagnostics: bool = False,
    rerank_diag_max_states: int = 0,
    rerank_diag_include_records: bool = False,
    proposal_recall_top_ks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    cmi_model.eval()
    max_budget = int(max(budget_list))
    total = 0
    correct_by_step = np.zeros(max_budget, dtype=np.float64)
    all_correct_rows: List[np.ndarray] = []
    sample_paths: List[Dict[str, Any]] = []
    unmasked_sample_paths: List[Dict[str, Any]] = []
    rerank_records: List[Dict[str, Any]] = []
    rerank_diag_enabled = bool(enable_rerank_diagnostics and oracle_target_provider is not None)
    rerank_diag_cap = None if int(rerank_diag_max_states) <= 0 else int(rerank_diag_max_states)
    recall_top_ks = [1, 3, 5, 7] if proposal_recall_top_ks is None else [
        int(k) for k in proposal_recall_top_ks if int(k) > 0
    ]

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="constraint rollout", dynamic_ncols=True, leave=False)):
        x = x.to(device)
        y = y.to(device)
        batch_size = int(x.shape[0])
        batch_start = int(batch_idx * loader.batch_size)
        batch_sample_indices = sample_indices[batch_start:batch_start + batch_size]

        mask = torch.zeros(batch_size, cmi_model.mask_size, dtype=x.dtype, device=device)
        x_masked = cmi_model.mask_layer(x, mask)
        pred = cmi_model.predictor(x_masked)
        batch_sequences = [[] for _ in range(batch_size)]
        batch_correct = np.zeros((batch_size, max_budget), dtype=np.float32)

        for step_idx in range(max_budget):
            if rerank_diag_enabled and (rerank_diag_cap is None or len(rerank_records) < rerank_diag_cap):
                max_new_records = None if rerank_diag_cap is None else int(rerank_diag_cap - len(rerank_records))
                rerank_records.extend(
                    collect_full_path_rerank_diagnostics_for_batch(
                        cmi_model=cmi_model,
                        x=x,
                        y=y,
                        mask=mask,
                        pred=pred,
                        batch_sample_indices=batch_sample_indices,
                        step_idx=int(step_idx),
                        prerequisite_matrix=prerequisite_matrix,
                        oracle_target_provider=oracle_target_provider,
                        max_records=max_new_records,
                        proposal_recall_top_ks=recall_top_ks,
                    )
                )
            mask, selected = select_next_feature_with_index(
                cmi_model,
                x,
                mask,
                pred,
                prerequisite_matrix=prerequisite_matrix,
            )
            selected_np = selected.detach().cpu().numpy().astype(np.int64)
            for row_idx, action_idx in enumerate(selected_np.tolist()):
                batch_sequences[row_idx].append(int(action_idx))
            x_masked = cmi_model.mask_layer(x, mask)
            pred = cmi_model.predictor(x_masked)
            pred_label = pred.argmax(dim=1)
            correct_np = (pred_label == y).detach().cpu().numpy().astype(np.float32)
            batch_correct[:, step_idx] = correct_np
            correct_by_step[step_idx] += float(correct_np.sum())

        total += batch_size
        all_correct_rows.append(batch_correct)
        for row_idx, sequence in enumerate(batch_sequences):
            ok, violation_step, reason = validate_action_sequence(sequence, action_groups)
            sample_index = int(batch_sample_indices[row_idx]) if row_idx < len(batch_sample_indices) else int(total - batch_size + row_idx)
            row_correct = batch_correct[row_idx].astype(np.float32).tolist()
            sample_paths.append(
                {
                    "local_index": int(total - batch_size + row_idx),
                    "sample_index": sample_index,
                    "selected_action_indices": [int(x) for x in sequence],
                    "selected_action_ids": [
                        str(action_groups[int(idx)].get("action_id", f"action_{int(idx)}"))
                        for idx in sequence
                        if 0 <= int(idx) < len(action_groups)
                    ],
                    "correct_by_step": [bool(float(x) > 0.5) for x in row_correct],
                    "constraint_valid": bool(ok),
                    "first_violation_step": None if violation_step is None else int(violation_step + 1),
                    "violation_reason": reason,
                }
            )

        unmasked_mask = torch.zeros(batch_size, cmi_model.mask_size, dtype=x.dtype, device=device)
        unmasked_x_masked = cmi_model.mask_layer(x, unmasked_mask)
        unmasked_pred = cmi_model.predictor(unmasked_x_masked)
        unmasked_sequences = [[] for _ in range(batch_size)]
        for _ in range(max_budget):
            unmasked_mask, unmasked_selected = select_next_feature_with_index(
                cmi_model,
                x,
                unmasked_mask,
                unmasked_pred,
                prerequisite_matrix=None,
            )
            selected_np = unmasked_selected.detach().cpu().numpy().astype(np.int64)
            for row_idx, action_idx in enumerate(selected_np.tolist()):
                unmasked_sequences[row_idx].append(int(action_idx))
            unmasked_x_masked = cmi_model.mask_layer(x, unmasked_mask)
            unmasked_pred = cmi_model.predictor(unmasked_x_masked)

        for row_idx, sequence in enumerate(unmasked_sequences):
            ok, violation_step, reason = validate_action_sequence(sequence, action_groups)
            sample_index = int(batch_sample_indices[row_idx]) if row_idx < len(batch_sample_indices) else int(total - batch_size + row_idx)
            unmasked_sample_paths.append(
                {
                    "local_index": int(total - batch_size + row_idx),
                    "sample_index": sample_index,
                    "selected_action_indices": [int(x) for x in sequence],
                    "selected_action_ids": [
                        str(action_groups[int(idx)].get("action_id", f"action_{int(idx)}"))
                        for idx in sequence
                        if 0 <= int(idx) < len(action_groups)
                    ],
                    "constraint_valid": bool(ok),
                    "first_violation_step": None if violation_step is None else int(violation_step + 1),
                    "violation_reason": reason,
                }
            )

    raw_curve = correct_by_step / max(total, 1)
    all_correct = np.concatenate(all_correct_rows, axis=0) if all_correct_rows else np.zeros((0, max_budget), dtype=np.float32)
    valid_mask = np.asarray([bool(item["constraint_valid"]) for item in sample_paths], dtype=bool)
    constraint_valid_n = int(valid_mask.sum())
    constraint_total_n = int(len(valid_mask))
    constraint_invalid_n = int(constraint_total_n - constraint_valid_n)
    constraint_valid_rate = float(constraint_valid_n / constraint_total_n) if constraint_total_n else None
    if constraint_valid_n > 0:
        constraint_curve_values = all_correct[valid_mask].mean(axis=0).astype(np.float64)
    else:
        constraint_curve_values = np.asarray([np.nan] * max_budget, dtype=np.float64)

    constraint_curve = []
    acc_by_budget: Dict[str, float] = {}
    for budget in budget_list:
        idx = int(budget) - 1
        acc_by_budget[str(budget)] = float(raw_curve[idx])
        constraint_acc = None if not np.isfinite(constraint_curve_values[idx]) else float(constraint_curve_values[idx])
        constraint_curve.append(
            {
                "num_features": int(budget),
                "num_actions": int(budget),
                "test_acc": constraint_acc,
                "constraint_test_acc": constraint_acc,
                "constraint_valid_rate": constraint_valid_rate,
                "constraint_valid_n": int(constraint_valid_n),
                "constraint_total_n": int(constraint_total_n),
                "constraint_invalid_n": int(constraint_invalid_n),
            }
        )

    valid_values = [float(item["test_acc"]) for item in constraint_curve if item["test_acc"] is not None]
    unmasked_summary = _summarize_policy_path_validity(unmasked_sample_paths)
    rerank_diagnostics = summarize_full_path_rerank_diagnostics(
        rerank_records,
        enabled=rerank_diag_enabled,
        max_states=rerank_diag_max_states,
        include_records=rerank_diag_include_records,
        proposal_recall_top_ks=recall_top_ks,
    )
    return {
        "acc_by_num_features": acc_by_budget,
        "acc_by_num_actions": acc_by_budget,
        "constraint_acc_by_num_features_integer": constraint_curve,
        "constraint_valid_rate": constraint_valid_rate,
        "constraint_valid_n": int(constraint_valid_n),
        "constraint_total_n": int(constraint_total_n),
        "constraint_invalid_n": int(constraint_invalid_n),
        "constraint_mean_acc_at_all": None if not valid_values else float(np.mean(valid_values)),
        "constraint_final_acc": None if not valid_values else constraint_curve[-1]["test_acc"],
        "constraint_sample_paths": sample_paths,
        "unmasked_policy_constraint_sample_paths": unmasked_sample_paths,
        "full_path_rerank_diagnostics": rerank_diagnostics,
        **unmasked_summary,
    }


def build_midpoint_metrics(acc_by_num_features: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(int(k) for k in acc_by_num_features.keys())
    out = {}
    for left, right in zip(keys, keys[1:]):
        out[f"{left + 0.5:.1f}"] = float((acc_by_num_features[str(left)] + acc_by_num_features[str(right)]) / 2.0)
    return out


def _summarize_policy_path_validity(sample_paths: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_n = int(len(sample_paths))
    valid_n = int(sum(1 for item in sample_paths if bool(item.get("constraint_valid"))))
    invalid_n = int(total_n - valid_n)
    violation_counts: Dict[str, int] = {}
    for item in sample_paths:
        if bool(item.get("constraint_valid")):
            continue
        reason = str(item.get("violation_reason") or "unknown")
        violation_counts[reason] = int(violation_counts.get(reason, 0) + 1)
    return {
        "unmasked_policy_constraint_valid_rate": float(valid_n / total_n) if total_n else None,
        "unmasked_policy_constraint_valid_n": int(valid_n),
        "unmasked_policy_constraint_total_n": int(total_n),
        "unmasked_policy_constraint_invalid_n": int(invalid_n),
        "unmasked_policy_constraint_violation_counts": violation_counts,
    }


def _merge_violation_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for record in records:
        counts = record.get("unmasked_policy_constraint_violation_counts", {})
        if not isinstance(counts, dict):
            continue
        for key, value in counts.items():
            merged[str(key)] = int(merged.get(str(key), 0) + int(value))
    return merged


def evaluate_predictor_on_policy_states(
    model: CMIEstimator,
    data_loader: DataLoader,
    *,
    device: torch.device,
    max_features: int,
) -> Dict[str, Any]:
    if not hasattr(model, "predict_policy_scores"):
        return {"accuracy": None, "correct": 0, "count": 0}

    was_training = bool(model.training)
    model.value_network.eval()
    model.predictor.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            mask = torch.zeros(len(x), model.mask_size, dtype=x.dtype, device=device)
            for step in range(int(max_features) + 1):
                x_masked = model.mask_layer(x, mask)
                logits = model.predictor(x_masked)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().detach().cpu())
                count += int(len(x))
                loss_vec = model.loss_fn(logits, y)
                loss = loss_vec.mean() if loss_vec.ndim > 0 else loss_vec
                loss_sum += float(loss.detach().cpu())
                batch_count += 1
                if step >= int(max_features):
                    break
                scores = model.predict_policy_scores(
                    x,
                    mask,
                    logits.detach(),
                    prerequisite_matrix=getattr(model, "prerequisite_matrix", None),
                )
                actions = torch.argmax(scores, dim=1)
                mask = torch.max(mask, ind_to_onehot(actions, model.mask_size).to(dtype=mask.dtype))
    if was_training:
        model.train()
    else:
        model.eval()
    return {
        "accuracy": None if count <= 0 else float(correct / count),
        "loss": None if batch_count <= 0 else float(loss_sum / batch_count),
        "correct": int(correct),
        "count": int(count),
    }


def posthoc_adapt_predictor_on_policy_states(
    model: CMIEstimator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    max_features: int,
) -> Dict[str, Any]:
    epochs = int(max(0, epochs))
    lr = float(max(0.0, lr))
    if epochs <= 0 or lr <= 0:
        return {
            "enabled": False,
            "reason": "non_positive_epochs_or_lr",
            "epochs": int(epochs),
            "lr": float(lr),
        }
    if not hasattr(model, "predict_policy_scores"):
        return {
            "enabled": False,
            "reason": "model_has_no_predict_policy_scores",
            "epochs": int(epochs),
            "lr": float(lr),
        }

    model.to(device)
    value_requires_grad = [p.requires_grad for p in model.value_network.parameters()]
    for p in model.value_network.parameters():
        p.requires_grad_(False)
    for p in model.predictor.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    epoch_losses: List[float] = []
    epoch_state_counts: List[int] = []
    val_scores: List[Optional[float]] = []
    val_losses: List[Optional[float]] = []
    previous_training = bool(model.training)
    best_epoch = 0
    initial_val = evaluate_predictor_on_policy_states(
        model,
        val_loader,
        device=device,
        max_features=max_features,
    )
    best_val_score = initial_val.get("accuracy")
    best_predictor_state = {
        key: value.detach().cpu().clone()
        for key, value in model.predictor.state_dict().items()
    }
    val_scores.append(best_val_score)
    val_losses.append(initial_val.get("loss"))

    try:
        for epoch in range(1, epochs + 1):
            model.value_network.eval()
            model.predictor.train()
            loss_sum = 0.0
            state_count = 0
            batch_count = 0
            for batch in train_loader:
                x = batch[0].to(device)
                y = batch[1].to(device)
                mask = torch.zeros(len(x), model.mask_size, dtype=x.dtype, device=device)
                losses = []

                for step in range(int(max_features) + 1):
                    x_masked = model.mask_layer(x, mask)
                    logits = model.predictor(x_masked)
                    loss_vec = model.loss_fn(logits, y)
                    losses.append(loss_vec.mean() if loss_vec.ndim > 0 else loss_vec)
                    state_count += int(len(x))
                    if step >= int(max_features):
                        break
                    with torch.no_grad():
                        scores = model.predict_policy_scores(
                            x,
                            mask,
                            logits.detach(),
                            prerequisite_matrix=getattr(model, "prerequisite_matrix", None),
                        )
                        actions = torch.argmax(scores, dim=1)
                        mask = torch.max(mask, ind_to_onehot(actions, model.mask_size).to(dtype=mask.dtype))

                loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += float(loss.detach().cpu())
                batch_count += 1

            epoch_losses.append(float(loss_sum / max(batch_count, 1)))
            epoch_state_counts.append(int(state_count))
            val_result = evaluate_predictor_on_policy_states(
                model,
                val_loader,
                device=device,
                max_features=max_features,
            )
            val_score = val_result.get("accuracy")
            val_scores.append(val_score)
            val_losses.append(val_result.get("loss"))
            if val_score is not None and (best_val_score is None or float(val_score) > float(best_val_score)):
                best_val_score = float(val_score)
                best_epoch = int(epoch)
                best_predictor_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.predictor.state_dict().items()
                }
        model.predictor.load_state_dict(best_predictor_state)
    finally:
        for p, requires_grad in zip(model.value_network.parameters(), value_requires_grad):
            p.requires_grad_(requires_grad)
        if previous_training:
            model.train()
        else:
            model.eval()

    return {
        "enabled": True,
        "epochs": int(epochs),
        "lr": float(lr),
        "max_features": int(max_features),
        "loss_first": None if not epoch_losses else float(epoch_losses[0]),
        "loss_last": None if not epoch_losses else float(epoch_losses[-1]),
        "loss_min": None if not epoch_losses else float(min(epoch_losses)),
        "state_count_last_epoch": 0 if not epoch_state_counts else int(epoch_state_counts[-1]),
        "selection_metric": "validation_policy_state_accuracy",
        "best_epoch": int(best_epoch),
        "val_accuracy_initial": initial_val.get("accuracy"),
        "val_accuracy_best": best_val_score,
        "val_accuracy_last": None if not val_scores else val_scores[-1],
        "val_loss_initial": initial_val.get("loss"),
        "val_loss_last": None if not val_losses else val_losses[-1],
        "val_scores": val_scores,
        "val_losses": val_losses,
        "epoch_losses": epoch_losses,
    }


def train_one_config(
    cfg: TrainConfig,
    trial_seed: int,
    d_in: int,
    d_out: int,
    group_matrix: torch.Tensor,
    pretrain_loader: DataLoader,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    gpu_id: int,
    max_features_train: int,
    pretrain_epochs: int,
    train_epochs: int,
    eps_steps: int,
    min_lr: float,
    class_weights: torch.Tensor,
    ckpt_dir: Path,
    oracle_target_provider: OracleDoubleHeadQTargetProvider,
    prerequisite_matrix: torch.Tensor,
    full_path_head_loss_weight: float,
    proposal_top_k: int,
    intervention_aux_enabled: bool,
    intervention_aux_weight: float,
    intervention_aux_only_changed_actions: bool,
    intervention_aux_mode: str,
    intervention_aux_oracle_margin: float,
    predictor_hidden: int,
    predictor_dropout: float,
    value_hidden: int,
    value_dropout: float,
) -> Tuple[CMIEstimator, float]:
    set_seed(trial_seed)

    num_groups = group_matrix.shape[0]
    pred_hidden = int(cfg.hidden if int(predictor_hidden) <= 0 else predictor_hidden)
    pred_dropout = float(cfg.dropout if float(predictor_dropout) < 0.0 else predictor_dropout)
    val_hidden = int(cfg.hidden if int(value_hidden) <= 0 else value_hidden)
    val_dropout = float(cfg.dropout if float(value_dropout) < 0.0 else value_dropout)
    predictor = get_mlp_network(d_in + num_groups, d_out, hidden=pred_hidden, dropout=pred_dropout)
    value_network = get_mlp_network(d_in + num_groups, num_groups * 2, hidden=val_hidden, dropout=val_dropout)
    mask_layer = MaskLayerGrouped(group_matrix=group_matrix, append=True)

    acc_metric = Accuracy(task="multiclass", num_classes=d_out)
    ce_loss = DTypeSafeCrossEntropyLoss(weight=class_weights)
    ce_loss_none = DTypeSafeCrossEntropyLoss(weight=class_weights, reduction="none")

    pretrain = MaskingPretrainer(
        predictor,
        mask_layer,
        lr=cfg.lr,
        loss_fn=ce_loss,
        val_loss_fn=acc_metric,
        patience=cfg.patience,
        min_lr=min_lr,
    )
    pre_trainer = Trainer(
        max_epochs=pretrain_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **trainer_kwargs(device, gpu_id),
    )
    pre_trainer.fit(pretrain, pretrain_loader, val_loader)

    cmi_model = DoubleHeadOracleQEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        lr=cfg.lr,
        min_lr=min_lr,
        max_features=max_features_train,
        eps=cfg.eps,
        loss_fn=ce_loss_none,
        val_loss_fn=acc_metric,
        eps_decay=cfg.eps_decay,
        eps_steps=eps_steps,
        patience=cfg.patience,
        feature_costs=None,
        cmi_scaling=cfg.cmi_scaling,
        oracle_double_target_fn=oracle_target_provider.double_head_targets,
        full_path_loss_weight=full_path_head_loss_weight,
        proposal_top_k=proposal_top_k,
        one_step_prefix_steps=int(oracle_target_provider.one_step_prefix_steps),
        full_path_middle_steps=int(oracle_target_provider.full_path_middle_steps),
        prerequisite_matrix=prerequisite_matrix.detach().cpu(),
        intervention_aux_enabled=intervention_aux_enabled,
        intervention_aux_weight=intervention_aux_weight,
        intervention_aux_only_changed_actions=intervention_aux_only_changed_actions,
        intervention_aux_mode=intervention_aux_mode,
        intervention_aux_oracle_margin=intervention_aux_oracle_margin,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_top_k=1,
        monitor="Perf Val/Mean",
        mode="max",
        filename="best",
        verbose=False,
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        max_epochs=train_epochs,
        logger=False,
        callbacks=[checkpoint_callback],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=10,
        **trainer_kwargs(device, gpu_id),
    )
    trainer.fit(cmi_model, train_loader, val_loader)

    best_score = float(checkpoint_callback.best_model_score.item()) if checkpoint_callback.best_model_score else float(
        "-inf"
    )
    load_best_checkpoint_if_available(cmi_model, checkpoint_callback.best_model_path, device)
    cmi_model.to(device).eval()
    return cmi_model, best_score


def main() -> None:
    args = parse_args()
    default_split_path = RESULTS_ROOT / args.dataset / "split" / f"split_seed{int(args.split_seed if args.split_seed is not None else args.seed)}.json"
    split_path = resolve_split_path(default_split_path, args.split_path, args.split_seed)
    split_info = load_split_json(split_path)

    dataset_csv = Path(args.dataset_csv).resolve()
    label_col = str(args.label_col).strip()
    actions_path = Path(args.actions_path).resolve()
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions JSON not found: {actions_path}")
    if not label_col:
        raise ValueError("--label_col must be non-empty.")

    device = resolve_device(args.device, args.gpu)
    pin_memory = device.type == "cuda"

    x, y, feature_names, label_mapping = load_tabular_csv(dataset_csv, label_col)
    group_matrix, action_groups = load_action_feature_matrix(str(actions_path), args.dataset, feature_names)
    prerequisite_matrix = build_prerequisite_matrix(action_groups).to(device)
    idx = validate_indices(split_info["indices"], len(x))
    x_norm, mean, std = normalize_inputs(x, idx["train"], args.normalize_mode)

    dataset_xy = TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(y))
    dataset_xyi = TensorDataset(
        torch.from_numpy(x_norm),
        torch.from_numpy(y),
        torch.arange(len(x_norm), dtype=torch.long),
    )
    pretrain_subset = Subset(dataset_xy, idx["train"].tolist())
    train_subset = Subset(dataset_xyi, idx["train"].tolist())
    val_subset = Subset(dataset_xy, idx["val"].tolist())
    test_subset = Subset(dataset_xy, idx["test"].tolist())

    d_in = x_norm.shape[1]
    d_out = int(np.unique(y).shape[0])
    num_actions = int(group_matrix.shape[0])
    a7_architecture_candidates = build_a7_architecture_candidates(args, d_out)
    budget_max = num_actions if args.max_eval_features is None else min(args.max_eval_features, num_actions)
    if budget_max <= 0:
        raise ValueError("--max_eval_features must be positive.")
    budget_list = list(range(1, budget_max + 1))

    max_features_train = budget_max if args.max_features_train is None else args.max_features_train
    max_features_train = min(int(max_features_train), num_actions)
    if max_features_train <= 0:
        raise ValueError("--max_features_train must be positive.")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / args.dataset / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt_root = out_dir / "_tmp" / "checkpoints"
    if tmp_ckpt_root.exists():
        shutil.rmtree(tmp_ckpt_root, ignore_errors=True)
    tmp_ckpt_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset={args.dataset}, csv={dataset_csv}")
    print(f"[INFO] split={split_path}")
    print(f"[INFO] actions={actions_path}")
    print(
        f"[INFO] rows={len(x_norm)}, features={d_in}, action_groups={num_actions}, classes={d_out}, "
        f"action budgets=1..{budget_max}, max_actions_train={max_features_train}"
    )
    print(f"[INFO] device={device}, class_weight={args.use_class_weight}")
    print(f"[INFO] output_dir={out_dir.resolve()}")

    class_weights = None
    if args.use_class_weight:
        class_weights = make_class_weights(y[idx["train"]], d_out)

    teacher_run_dir = resolve_teacher_run_dir(args, actions_path)
    planner_config = FullPathPlannerConfig(
        top_k_paths=max(1, int(args.full_path_top_k)),
        beam_width=max(1, int(args.full_path_beam_width)),
        max_depth=max(1, int(args.full_path_max_depth)),
        score_mode=str(args.full_path_score),
        temperature=max(float(args.full_path_temperature), 1.0e-8),
        mixed_hard_acc_alpha=float(args.full_path_mixed_hard_acc_alpha),
    )
    proposal_recall_top_ks = sorted({int(k) for k in parse_int_list(args.proposal_recall_top_ks) if int(k) > 0})
    if not proposal_recall_top_ks:
        proposal_recall_top_ks = [1, 3, 5, 7]
    intervention_sensitivity_enabled = bool(
        args.save_diagnostics
        and
        not args.disable_intervention_sensitivity_diagnostics
        and not args.disable_rerank_diagnostics
    )
    diagnostic_include_records = bool(
        args.save_diagnostics
        and (args.rerank_diag_include_records or intervention_sensitivity_enabled)
    )
    prefix_steps, middle_steps, suffix_steps = resolve_hybrid_schedule(
        args.dataset,
        num_actions,
        args.one_step_prefix_steps,
        args.full_path_middle_steps,
        args.schedule_preset,
    )
    inference_prefix_steps, inference_middle_steps, inference_suffix_steps = resolve_inference_schedule(
        args.dataset,
        num_actions,
        prefix_steps,
        middle_steps,
        args.inference_one_step_prefix_steps,
        args.inference_full_path_middle_steps,
        args.inference_schedule_preset,
    )
    schedule_aligned_inference = (
        int(prefix_steps) == int(inference_prefix_steps)
        and int(middle_steps) == int(inference_middle_steps)
    )
    primary_inference_spec = build_inference_schedule_spec(
        dataset=args.dataset,
        num_actions=num_actions,
        train_prefix_steps=prefix_steps,
        train_middle_steps=middle_steps,
        train_suffix_steps=suffix_steps,
        preset=args.inference_schedule_preset,
        prefix_override=args.inference_one_step_prefix_steps,
        middle_override=args.inference_full_path_middle_steps,
    )
    paired_presets = unique_presets(args.paired_inference_schedule_presets)
    if paired_presets:
        reference_preset = str(args.paired_reference_schedule_preset or paired_presets[0])
        if reference_preset not in paired_presets:
            paired_presets.insert(0, reference_preset)
        if str(args.inference_schedule_preset) not in paired_presets:
            paired_presets.append(str(args.inference_schedule_preset))
    else:
        reference_preset = ""
    oracle_target_provider = OracleDoubleHeadQTargetProvider(
        dataset=args.dataset,
        run_dir=teacher_run_dir,
        teacher_ckpt=args.teacher_ckpt,
        dataset_csv=dataset_csv,
        label_col=label_col,
        labels=y,
        group_matrix=group_matrix,
        action_groups=action_groups,
        prerequisite_matrix=prerequisite_matrix.detach().cpu(),
        one_step_target_transform=args.one_step_target_transform,
        full_path_target_transform=args.full_path_target_transform,
        planner_config=planner_config,
        target_reduce=args.full_path_q_reduce,
        one_step_prefix_steps=prefix_steps,
        full_path_middle_steps=middle_steps,
        proposal_top_k=args.proposal_top_k,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_gap_scale=args.alpha_gap_scale,
        alpha_gap_floor=args.alpha_gap_floor,
        alpha_agree_bonus=args.alpha_agree_bonus,
        alpha_disagree_penalty=args.alpha_disagree_penalty,
        horizon_advantage_threshold=args.horizon_advantage_threshold,
        horizon_penalty=args.horizon_penalty,
    )
    print(f"[INFO] teacher_run_dir={teacher_run_dir}")
    print("[INFO] teacher_source_policy=explicit_teacher_run_dir")
    print(
        f"[INFO] train_fixed_ratio_schedule_double_head=one_step_prefix:{prefix_steps}, "
        f"proposal_rerank_middle:{middle_steps}, one_step_suffix:{suffix_steps}, "
        f"schedule_preset:{args.schedule_preset}, "
        f"proposal_top_k:{args.proposal_top_k}, "
        f"full_path_head_loss_weight:{args.full_path_head_loss_weight}"
    )
    print(
        f"[INFO] inference_only_schedule_ablation=one_step_prefix:{inference_prefix_steps}, "
        f"proposal_rerank_middle:{inference_middle_steps}, one_step_suffix:{inference_suffix_steps}, "
        f"inference_schedule_preset:{args.inference_schedule_preset}, "
        f"schedule_aligned_inference:{schedule_aligned_inference}"
    )
    if paired_presets:
        print(
            f"[INFO] paired_inference_schedule_presets:{paired_presets}, "
            f"paired_reference_schedule_preset:{reference_preset}"
        )
    print(
        f"[INFO] diagnostic_intervention_sensitivity_diagnostics={intervention_sensitivity_enabled}, "
        f"diagnostic_include_records={diagnostic_include_records}, "
        f"top_transition_k={args.intervention_top_transition_k}"
    )
    intervention_aux_enabled = bool(args.enable_intervention_aux) and not bool(args.disable_intervention_aux)
    posthoc_predictor_adapt_enabled = not bool(args.disable_posthoc_predictor_adapt)
    print(
        f"[INFO] auto_research_med_afa_student_intervention_aux_enabled={intervention_aux_enabled}, "
        f"weight={args.intervention_aux_weight}, "
        f"only_changed_actions={args.intervention_aux_only_changed_actions}, "
        f"mode={args.intervention_aux_mode}, "
        f"oracle_margin={args.intervention_aux_oracle_margin}"
    )
    print(
        f"[INFO] auto_research_med_afa_student_posthoc_predictor_adapt_enabled={posthoc_predictor_adapt_enabled}, "
        f"epochs={args.posthoc_predictor_adapt_epochs}, "
        f"lr={args.posthoc_predictor_adapt_lr}"
    )
    print(
        f"[INFO] auto_research_med_afa_student_arch_selection_mode={args.arch_selection_mode}, "
        f"candidates={a7_architecture_candidates}"
    )
    print(f"[INFO] oracle_q_double_head={oracle_target_provider.summary()}")

    configs = make_train_configs(args)
    print(f"[INFO] training_configs={len(configs)}, architecture_candidates={len(a7_architecture_candidates)}")
    candidate_cache_enabled = not bool(args.disable_candidate_cache)
    candidate_cache_root = (
        Path(args.candidate_cache_dir).resolve()
        if str(args.candidate_cache_dir).strip()
        else (RESULTS_ROOT / "candidate_cache" / "med_afa_student" / args.dataset).resolve()
    )
    if candidate_cache_enabled:
        candidate_cache_root.mkdir(parents=True, exist_ok=True)
    teacher_art = getattr(oracle_target_provider, "teacher_art", None)
    teacher_ckpt_path = Path(getattr(teacher_art, "teacher_ckpt_path", "")) if teacher_art is not None else Path("")
    class_weight_values = None if class_weights is None else class_weights.detach().cpu().numpy().astype(float).tolist()
    candidate_cache_common_meta = {
        "cache_code_version": "candidate_model_cache",
        "method": "med_afa_student",
        "dataset": str(args.dataset),
        "data": {
            "dataset_csv": _file_identity(dataset_csv, content_hash=True),
            "split_path": _file_identity(split_path, content_hash=True),
            "actions_path": _file_identity(actions_path, content_hash=True),
            "label_col": str(label_col),
            "label_mapping": label_mapping,
            "split_seed": split_info.get("seed", None),
            "num_rows": int(len(x_norm)),
            "num_features": int(d_in),
            "num_actions": int(num_actions),
            "num_classes": int(d_out),
        },
        "teacher": {
            "teacher_run_dir": str(teacher_run_dir.resolve()),
            "teacher_ckpt_arg": str(args.teacher_ckpt or ""),
            "resolved_teacher_ckpt": _file_identity(teacher_ckpt_path, content_hash=False),
        },
        "training": {
            "seed_base": int(args.seed),
            "num_trials": int(args.num_trials),
            "normalize_mode": str(args.normalize_mode),
            "use_class_weight": bool(args.use_class_weight),
            "class_weights": class_weight_values,
            "pretrain_epochs": int(args.pretrain_epochs),
            "train_epochs": int(args.train_epochs),
            "max_features_train": int(max_features_train),
            "batch_size_train": int(args.batch_size_train),
            "batch_size_eval": int(args.batch_size_eval),
            "eps_steps": int(args.eps_steps),
            "min_lr": float(args.min_lr),
            "arch_selection_mode": str(args.arch_selection_mode),
            "predictor_arch_grid": str(args.predictor_arch_grid),
        },
        "schedule": {
            "schedule_preset": str(args.schedule_preset),
            "one_step_prefix_steps_arg": int(args.one_step_prefix_steps),
            "full_path_middle_steps_arg": int(args.full_path_middle_steps),
            "resolved_prefix_steps": int(prefix_steps),
            "resolved_middle_steps": int(middle_steps),
            "resolved_suffix_steps": int(suffix_steps),
            "inference_schedule_preset": str(args.inference_schedule_preset),
            "inference_one_step_prefix_steps_arg": int(args.inference_one_step_prefix_steps),
            "inference_full_path_middle_steps_arg": int(args.inference_full_path_middle_steps),
            "resolved_inference_prefix_steps": int(inference_prefix_steps),
            "resolved_inference_middle_steps": int(inference_middle_steps),
            "resolved_inference_suffix_steps": int(inference_suffix_steps),
        },
        "oracle_q": {
            "one_step_target_transform": str(args.one_step_target_transform),
            "full_path_target_transform": str(args.full_path_target_transform),
            "proposal_top_k": int(args.proposal_top_k),
            "full_path_head_loss_weight": float(args.full_path_head_loss_weight),
            "full_path_top_k": int(args.full_path_top_k),
            "full_path_beam_width": int(args.full_path_beam_width),
            "full_path_max_depth": int(args.full_path_max_depth),
            "full_path_score": str(args.full_path_score),
            "full_path_temperature": float(args.full_path_temperature),
            "full_path_mixed_hard_acc_alpha": float(args.full_path_mixed_hard_acc_alpha),
            "full_path_q_reduce": str(args.full_path_q_reduce),
            "intervention_aux_enabled": bool(intervention_aux_enabled),
            "intervention_aux_weight": float(args.intervention_aux_weight),
            "intervention_aux_only_changed_actions": bool(args.intervention_aux_only_changed_actions),
            "intervention_aux_mode": str(args.intervention_aux_mode),
            "intervention_aux_oracle_margin": float(args.intervention_aux_oracle_margin),
        },
    }
    print(
        f"[INFO] candidate_cache_enabled={candidate_cache_enabled}, "
        f"force_retrain={args.force_retrain_candidate_cache}, root={candidate_cache_root}"
    )

    trial_records = []
    for trial in tqdm(range(args.num_trials), desc="Trials", dynamic_ncols=True):
        trial_seed = args.seed + trial

        generator = torch.Generator()
        generator.manual_seed(trial_seed)

        pretrain_loader = DataLoader(
            pretrain_subset,
            batch_size=args.batch_size_train,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=False,
            num_workers=args.num_workers,
            generator=generator,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size_train,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=False,
            num_workers=args.num_workers,
            generator=generator,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.batch_size_eval,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            num_workers=args.num_workers,
        )

        grid_results = []
        best_selection_score = float("-inf")
        best_train_val_score = float("-inf")
        best_cfg = None
        best_architecture = None
        best_val_constraint_metrics = None
        best_model = None

        cfg_iter = tqdm(configs, desc=f"Grid(trial={trial})", dynamic_ncols=True, leave=False)
        for cfg_idx, cfg in enumerate(cfg_iter):
            cfg_iter.set_postfix(lr=cfg.lr, eps=cfg.eps, decay=cfg.eps_decay, hid=cfg.hidden)
            for arch in a7_architecture_candidates:
                arch_idx = int(arch["candidate_id"])
                ckpt_dir = tmp_ckpt_root / f"trial_{trial}" / f"cfg_{cfg_idx:03d}_arch_{arch_idx:03d}"
                candidate_meta = dict(candidate_cache_common_meta)
                candidate_meta.update(
                    {
                        "trial": int(trial),
                        "trial_seed": int(trial_seed),
                        "config_id": int(cfg_idx),
                        "architecture_id": int(arch_idx),
                        "config": asdict(cfg),
                        "architecture": dict(arch),
                    }
                )
                candidate_model_kwargs = {
                    "cfg": cfg,
                    "d_in": d_in,
                    "d_out": d_out,
                    "group_matrix": group_matrix,
                    "device": device,
                    "max_features_train": max_features_train,
                    "eps_steps": args.eps_steps,
                    "min_lr": args.min_lr,
                    "class_weights": class_weights,
                    "oracle_target_provider": oracle_target_provider,
                    "prerequisite_matrix": prerequisite_matrix,
                    "full_path_head_loss_weight": args.full_path_head_loss_weight,
                    "proposal_top_k": args.proposal_top_k,
                    "intervention_aux_enabled": intervention_aux_enabled,
                    "intervention_aux_weight": float(args.intervention_aux_weight),
                    "intervention_aux_only_changed_actions": bool(args.intervention_aux_only_changed_actions),
                    "intervention_aux_mode": str(args.intervention_aux_mode),
                    "intervention_aux_oracle_margin": float(args.intervention_aux_oracle_margin),
                    "predictor_hidden": int(arch["predictor_hidden"]),
                    "predictor_dropout": float(arch["predictor_dropout"]),
                    "value_hidden": int(arch["value_hidden"]),
                    "value_dropout": float(arch["value_dropout"]),
                }
                candidate_cache_info: Dict[str, Any] = {
                    "enabled": bool(candidate_cache_enabled),
                    "hit": False,
                    "saved": False,
                    "source": "disabled" if not candidate_cache_enabled else "train",
                }
                cached_candidate = None
                if candidate_cache_enabled and not bool(args.force_retrain_candidate_cache):
                    cached_candidate = load_candidate_cache(candidate_cache_root, candidate_meta, **candidate_model_kwargs)
                if cached_candidate is not None:
                    model, val_score, _cached_val_metrics, candidate_cache_info = cached_candidate
                    print(
                        f"[CACHE] hit trial={trial} cfg={cfg_idx} arch={arch['preset']} "
                        f"hash={candidate_cache_info.get('cache_hash')}"
                    )
                else:
                    existing_candidate = None
                    if candidate_cache_enabled and not bool(args.force_retrain_candidate_cache):
                        existing_candidate = load_candidate_from_existing_ckpt(ckpt_dir, **candidate_model_kwargs)
                    if existing_candidate is not None:
                        model, val_score, candidate_cache_info = existing_candidate
                        print(
                            f"[CACHE] imported existing checkpoint trial={trial} cfg={cfg_idx} "
                            f"arch={arch['preset']} ckpt={candidate_cache_info.get('ckpt_path')}"
                        )
                    else:
                        model, val_score = train_one_config(
                            cfg=cfg,
                            trial_seed=trial_seed,
                            d_in=d_in,
                            d_out=d_out,
                            group_matrix=group_matrix,
                            pretrain_loader=pretrain_loader,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            gpu_id=args.gpu,
                            max_features_train=max_features_train,
                            pretrain_epochs=args.pretrain_epochs,
                            train_epochs=args.train_epochs,
                            eps_steps=args.eps_steps,
                            min_lr=args.min_lr,
                            class_weights=class_weights,
                            ckpt_dir=ckpt_dir,
                            oracle_target_provider=oracle_target_provider,
                            prerequisite_matrix=prerequisite_matrix,
                            full_path_head_loss_weight=args.full_path_head_loss_weight,
                            proposal_top_k=args.proposal_top_k,
                            intervention_aux_enabled=intervention_aux_enabled,
                            intervention_aux_weight=float(args.intervention_aux_weight),
                            intervention_aux_only_changed_actions=bool(args.intervention_aux_only_changed_actions),
                            intervention_aux_mode=str(args.intervention_aux_mode),
                            intervention_aux_oracle_margin=float(args.intervention_aux_oracle_margin),
                            predictor_hidden=int(arch["predictor_hidden"]),
                            predictor_dropout=float(arch["predictor_dropout"]),
                            value_hidden=int(arch["value_hidden"]),
                            value_dropout=float(arch["value_dropout"]),
                        )
                model.to(device).eval()
                with temporary_policy_schedule(
                    model,
                    one_step_prefix_steps=inference_prefix_steps,
                    full_path_middle_steps=inference_middle_steps,
                ):
                    val_constraint_metrics = evaluate_action_group_constraint(
                        model,
                        val_loader,
                        budget_list,
                        device,
                        action_groups,
                        idx["val"],
                        prerequisite_matrix=prerequisite_matrix,
                        oracle_target_provider=None,
                        enable_rerank_diagnostics=False,
                        rerank_diag_max_states=0,
                        rerank_diag_include_records=False,
                        proposal_recall_top_ks=proposal_recall_top_ks,
                    )
                if candidate_cache_enabled and candidate_cache_info.get("source") != "candidate_cache":
                    saved_info = save_candidate_cache(
                        candidate_cache_root,
                        candidate_meta,
                        model,
                        float(val_score),
                        val_constraint_metrics,
                        str(candidate_cache_info.get("source", "train")),
                    )
                    candidate_cache_info.update(saved_info)
                val_constraint_mean = val_constraint_metrics.get("constraint_mean_acc_at_all")
                selection_score = float(val_score)
                if str(args.arch_selection_mode) == "val_constraint":
                    selection_score = float("-inf") if val_constraint_mean is None else float(val_constraint_mean)
                grid_results.append(
                    {
                        "config_id": cfg_idx,
                        "architecture_id": arch_idx,
                        "config": asdict(cfg),
                        "architecture": dict(arch),
                        "train_val_score": float(val_score),
                        "val_constraint_mean_acc@all": None if val_constraint_mean is None else float(val_constraint_mean),
                        "val_constraint_final_acc": val_constraint_metrics.get("constraint_final_acc"),
                        "selection_score": float(selection_score),
                        "candidate_cache": candidate_cache_info,
                    }
                )
                print(
                    f"[GRID] trial={trial} cfg={cfg_idx} arch={arch['preset']} "
                    f"train_val={val_score:.6f} val_constraint={selection_score:.6f} "
                    f"config={asdict(cfg)} arch={arch}"
                )

                if selection_score > best_selection_score:
                    old_best = best_model
                    best_selection_score = float(selection_score)
                    best_train_val_score = float(val_score)
                    best_cfg = cfg
                    best_architecture = dict(arch)
                    best_val_constraint_metrics = val_constraint_metrics
                    best_model = model
                    if old_best is not None:
                        old_best.to("cpu")
                        del old_best
                else:
                    model.to("cpu")
                    del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if not args.keep_checkpoints:
                    if ckpt_dir.exists():
                        shutil.rmtree(ckpt_dir, ignore_errors=True)

        if best_model is None or best_cfg is None or best_architecture is None:
            raise RuntimeError("grid search produced no valid model.")

        print(
            f"[BEST] trial={trial} selection_score={best_selection_score:.6f} "
            f"train_val_score={best_train_val_score:.6f} config={asdict(best_cfg)} arch={best_architecture}"
        )
        if posthoc_predictor_adapt_enabled:
            posthoc_predictor_adapt_summary = posthoc_adapt_predictor_on_policy_states(
                best_model,
                train_loader,
                val_loader,
                device=device,
                epochs=int(args.posthoc_predictor_adapt_epochs),
                lr=float(args.posthoc_predictor_adapt_lr),
                max_features=max_features_train,
            )
        else:
            posthoc_predictor_adapt_summary = {
                "enabled": False,
                "reason": "disabled_by_cli",
                "epochs": int(args.posthoc_predictor_adapt_epochs),
                "lr": float(args.posthoc_predictor_adapt_lr),
            }
        print(f"[INFO] med_afa_student_posthoc_predictor_adapt={posthoc_predictor_adapt_summary}")
        best_model.to(device).eval()

        with temporary_policy_schedule(
            best_model,
            one_step_prefix_steps=inference_prefix_steps,
            full_path_middle_steps=inference_middle_steps,
        ):
            eval_metrics = evaluate_action_group_constraint(
                best_model,
                test_loader,
                budget_list,
                device,
                action_groups,
                idx["test"],
                prerequisite_matrix=prerequisite_matrix,
                oracle_target_provider=oracle_target_provider,
                enable_rerank_diagnostics=not bool(args.disable_rerank_diagnostics),
                rerank_diag_max_states=int(args.rerank_diag_max_states),
                rerank_diag_include_records=bool(diagnostic_include_records),
                proposal_recall_top_ks=proposal_recall_top_ks,
            )

        paired_inference_ablation = None
        if paired_presets:
            paired_metrics: Dict[str, Dict[str, Any]] = {}
            paired_specs: Dict[str, Dict[str, Any]] = {}
            for preset in paired_presets:
                spec = build_inference_schedule_spec(
                    dataset=args.dataset,
                    num_actions=num_actions,
                    train_prefix_steps=prefix_steps,
                    train_middle_steps=middle_steps,
                    train_suffix_steps=suffix_steps,
                    preset=preset,
                )
                paired_specs[preset] = spec
                if (
                    str(preset) == str(args.inference_schedule_preset)
                    and int(spec["one_step_prefix_steps"]) == int(inference_prefix_steps)
                    and int(spec["proposal_rerank_middle_steps"]) == int(inference_middle_steps)
                ):
                    paired_metrics[preset] = eval_metrics
                    continue
                with temporary_policy_schedule(
                    best_model,
                    one_step_prefix_steps=int(spec["one_step_prefix_steps"]),
                    full_path_middle_steps=int(spec["proposal_rerank_middle_steps"]),
                ):
                    paired_metrics[preset] = evaluate_action_group_constraint(
                        best_model,
                        test_loader,
                        budget_list,
                        device,
                        action_groups,
                        idx["test"],
                        prerequisite_matrix=prerequisite_matrix,
                        oracle_target_provider=oracle_target_provider,
                        enable_rerank_diagnostics=False,
                        rerank_diag_max_states=0,
                        rerank_diag_include_records=False,
                        proposal_recall_top_ks=proposal_recall_top_ks,
                    )

            paired_schedule_metrics = {
                preset: schedule_metrics_summary(preset, paired_specs[preset], paired_metrics[preset])
                for preset in paired_presets
            }
            if reference_preset not in paired_metrics:
                raise ValueError(f"Paired reference preset was not evaluated: {reference_preset}")
            paired_comparisons = {
                preset: summarize_paired_schedule_comparison(
                    reference_name=reference_preset,
                    variant_name=preset,
                    reference_metrics=paired_metrics[reference_preset],
                    variant_metrics=metrics,
                    budget_list=budget_list,
                )
                for preset, metrics in paired_metrics.items()
                if preset != reference_preset
            }
            paired_inference_ablation = {
                "enabled": True,
                "reference_schedule": reference_preset,
                "evaluated_schedules": paired_presets,
                "schedule_metrics": paired_schedule_metrics,
                "comparisons": paired_comparisons,
            }
        acc_by_num_features = eval_metrics["acc_by_num_features"]
        acc_by_num_features_midpoint = build_midpoint_metrics(acc_by_num_features)

        model_path = None
        if args.save_model:
            model_path = out_dir / f"oracle_q_trial{trial}.pt"
            torch.save(best_model.state_dict(), model_path)

        trial_record = {
            "trial": trial,
            "seed": trial_seed,
            "best_selection_score": float(best_selection_score),
            "best_train_val_score": float(best_train_val_score),
            "best_val_score": float(best_train_val_score),
            "best_config": asdict(best_cfg),
            "best_architecture": best_architecture,
            "best_val_constraint_metrics": best_val_constraint_metrics,
            "grid_results": grid_results,
            "inference_only_ablation": bool(not schedule_aligned_inference),
            "intervention_aux_diagnostics": (
                best_model.intervention_aux_summary()
                if hasattr(best_model, "intervention_aux_summary")
                else {"enabled": False}
            ),
            "posthoc_predictor_adapt": posthoc_predictor_adapt_summary,
            "diagnostic_intervention_sensitivity_enabled": bool(intervention_sensitivity_enabled),
            "schedule_aligned_inference": bool(schedule_aligned_inference),
            "training_schedule_preset": str(args.schedule_preset),
            "inference_schedule_preset": str(args.inference_schedule_preset),
            "training_one_step_prefix_steps": int(prefix_steps),
            "training_proposal_rerank_middle_steps": int(middle_steps),
            "training_one_step_suffix_steps": int(suffix_steps),
            "inference_one_step_prefix_steps": int(inference_prefix_steps),
            "inference_proposal_rerank_middle_steps": int(inference_middle_steps),
            "inference_one_step_suffix_steps": int(inference_suffix_steps),
            "paired_inference_ablation": paired_inference_ablation,
            "acc_by_num_features": acc_by_num_features,
            "acc_by_num_features_midpoint": acc_by_num_features_midpoint,
            "constraint_acc_by_num_features_integer": eval_metrics["constraint_acc_by_num_features_integer"],
            "constraint_valid_rate": eval_metrics["constraint_valid_rate"],
            "constraint_valid_n": eval_metrics["constraint_valid_n"],
            "constraint_total_n": eval_metrics["constraint_total_n"],
            "constraint_invalid_n": eval_metrics["constraint_invalid_n"],
            "constraint_mean_acc_at_all": eval_metrics["constraint_mean_acc_at_all"],
            "constraint_final_acc": eval_metrics["constraint_final_acc"],
            "constraint_sample_paths": eval_metrics["constraint_sample_paths"],
            "unmasked_policy_constraint_valid_rate": eval_metrics["unmasked_policy_constraint_valid_rate"],
            "unmasked_policy_constraint_valid_n": eval_metrics["unmasked_policy_constraint_valid_n"],
            "unmasked_policy_constraint_total_n": eval_metrics["unmasked_policy_constraint_total_n"],
            "unmasked_policy_constraint_invalid_n": eval_metrics["unmasked_policy_constraint_invalid_n"],
            "unmasked_policy_constraint_violation_counts": eval_metrics["unmasked_policy_constraint_violation_counts"],
            "unmasked_policy_constraint_sample_paths": eval_metrics["unmasked_policy_constraint_sample_paths"],
            "full_path_rerank_diagnostics": eval_metrics["full_path_rerank_diagnostics"],
            "model_path": str(model_path.resolve()) if model_path else None,
        }
        trial_records.append(trial_record)
        trial_output = {
            "trial": int(trial),
            "seed": int(trial_seed),
            "selection": {
                "score": float(best_selection_score),
                "architecture": best_architecture,
                "validation_mean_acc@all": trial_record["best_val_constraint_metrics"].get("constraint_mean_acc_at_all"),
                "validation_final_acc": trial_record["best_val_constraint_metrics"].get("constraint_final_acc"),
            },
            "metrics": {
                "mean_acc@all": float(trial_record["constraint_mean_acc_at_all"]),
                "final_acc": float(trial_record["constraint_final_acc"]),
                "per_action_accuracy": trial_record["constraint_acc_by_num_features_integer"],
                "constraint_valid_rate": trial_record["constraint_valid_rate"],
            },
        }
        with (out_dir / f"trial_{trial}.json").open("w", encoding="utf-8") as f:
            json.dump(trial_output, f, ensure_ascii=False, indent=2)

    mean_acc_by_num_features = {}
    for k in budget_list:
        key = str(k)
        vals = [rec["acc_by_num_features"][key] for rec in trial_records]
        mean_acc_by_num_features[key] = float(np.mean(vals))
    mean_acc_by_num_features_midpoint = build_midpoint_metrics(mean_acc_by_num_features)
    constraint_valid_n = int(sum(int(rec.get("constraint_valid_n", 0)) for rec in trial_records))
    constraint_total_n = int(sum(int(rec.get("constraint_total_n", 0)) for rec in trial_records))
    constraint_invalid_n = int(sum(int(rec.get("constraint_invalid_n", 0)) for rec in trial_records))
    constraint_valid_rate = float(constraint_valid_n / constraint_total_n) if constraint_total_n else None
    unmasked_valid_n = int(sum(int(rec.get("unmasked_policy_constraint_valid_n", 0)) for rec in trial_records))
    unmasked_total_n = int(sum(int(rec.get("unmasked_policy_constraint_total_n", 0)) for rec in trial_records))
    unmasked_invalid_n = int(sum(int(rec.get("unmasked_policy_constraint_invalid_n", 0)) for rec in trial_records))
    unmasked_valid_rate = float(unmasked_valid_n / unmasked_total_n) if unmasked_total_n else None
    unmasked_violation_counts = _merge_violation_counts(trial_records)
    constraint_curve = []
    for k in budget_list:
        vals = []
        for rec in trial_records:
            for item in rec.get("constraint_acc_by_num_features_integer", []):
                if int(item.get("num_actions", item.get("num_features", -1))) == int(k) and item.get("test_acc") is not None:
                    vals.append(float(item["test_acc"]))
                    break
        constraint_acc = None if not vals else float(np.mean(vals))
        constraint_curve.append(
            {
                "num_features": int(k),
                "num_actions": int(k),
                "test_acc": constraint_acc,
                "constraint_test_acc": constraint_acc,
                "constraint_valid_rate": constraint_valid_rate,
                "constraint_valid_n": int(constraint_valid_n),
                "constraint_total_n": int(constraint_total_n),
                "constraint_invalid_n": int(constraint_invalid_n),
            }
        )
    constraint_values = [float(item["test_acc"]) for item in constraint_curve if item["test_acc"] is not None]

    double_head_diagnostics_path = out_dir / "fixed_ratio_schedule_double_head_diagnostics.json"
    full_path_rerank_diagnostics_path = out_dir / "full_path_rerank_diagnostics.json"
    intervention_sensitivity_diagnostics_path = out_dir / "intervention_outcome_sensitivity_diagnostics.json"
    trial_rerank_diagnostics = [rec.get("full_path_rerank_diagnostics") for rec in trial_records]
    aggregate_rerank_records: List[Dict[str, Any]] = []
    for diag in trial_rerank_diagnostics:
        if isinstance(diag, dict) and isinstance(diag.get("records"), list):
            aggregate_rerank_records.extend(diag["records"])
    aggregate_rerank_diagnostics = summarize_full_path_rerank_diagnostics(
        aggregate_rerank_records,
        enabled=not bool(args.disable_rerank_diagnostics),
        max_states=int(args.rerank_diag_max_states) * int(args.num_trials),
        include_records=bool(diagnostic_include_records),
        proposal_recall_top_ks=proposal_recall_top_ks,
    ) if aggregate_rerank_records else None
    if aggregate_rerank_diagnostics is None and len(trial_rerank_diagnostics) == 1:
        first_diag = trial_rerank_diagnostics[0]
        if isinstance(first_diag, dict):
            aggregate_rerank_diagnostics = {
                key: value
                for key, value in first_diag.items()
                if key != "records"
            }
    intervention_sensitivity_diagnostics = summarize_intervention_outcome_sensitivity(
        aggregate_rerank_records,
        action_groups=action_groups,
        enabled=bool(intervention_sensitivity_enabled),
        top_transition_k=int(args.intervention_top_transition_k),
    )
    paired_trial_results = [
        rec.get("paired_inference_ablation")
        for rec in trial_records
        if isinstance(rec.get("paired_inference_ablation"), dict)
    ]
    if len(paired_trial_results) == 1:
        paired_inference_ablation_summary = paired_trial_results[0]
    elif paired_trial_results:
        paired_inference_ablation_summary = {
            "enabled": True,
            "aggregation": "per_trial",
            "per_trial": paired_trial_results,
        }
    else:
        paired_inference_ablation_summary = {
            "enabled": False,
            "reference_schedule": None,
            "evaluated_schedules": [],
            "schedule_metrics": {},
            "comparisons": {},
        }
    intervention_aux_trial_summaries = [
        rec.get("intervention_aux_diagnostics", {"enabled": False})
        for rec in trial_records
    ]
    selected_architecture_trial_summaries = [
        rec.get("best_architecture", {})
        for rec in trial_records
    ]
    val_constraint_selection_trial_summaries = [
        {
            "trial": int(rec.get("trial", idx)),
            "best_selection_score": rec.get("best_selection_score"),
            "best_train_val_score": rec.get("best_train_val_score"),
            "best_architecture": rec.get("best_architecture", {}),
            "best_val_constraint_mean_acc@all": (
                rec.get("best_val_constraint_metrics", {}).get("constraint_mean_acc_at_all")
                if isinstance(rec.get("best_val_constraint_metrics"), dict)
                else None
            ),
            "best_val_constraint_final_acc": (
                rec.get("best_val_constraint_metrics", {}).get("constraint_final_acc")
                if isinstance(rec.get("best_val_constraint_metrics"), dict)
                else None
            ),
        }
        for idx, rec in enumerate(trial_records)
    ]
    summary = {
        "dataset": args.dataset,
        "method": "med_afa_student_action_group",
        "mean_acc@all": None if not constraint_values else float(np.mean(constraint_values)),
        "final_acc": None if not constraint_values else constraint_curve[-1]["test_acc"],
        "base_method": "student_acquisition_model_action_group",
        "method_summary": (
            "Clinical action-group acquisition with short-term proposal and long-term reranking."
        ),
        "med_afa_student_fixed_parameters": {
            "full_path_head_loss_weight": float(args.full_path_head_loss_weight),
            "schedule_preset": str(args.schedule_preset),
            "inference_schedule_preset": str(args.inference_schedule_preset),
            "arch_selection_mode": str(args.arch_selection_mode),
            "predictor_arch_grid": str(args.predictor_arch_grid),
            "posthoc_predictor_adapt": bool(posthoc_predictor_adapt_enabled),
            "intervention_aux": bool(intervention_aux_enabled),
            "weight_search": False,
        },
        "teacher_source_policy": "explicit_teacher_run_dir",
        "oracle_q": oracle_target_provider.summary(),
        "double_head_diagnostics_path": str(double_head_diagnostics_path.resolve()),
        "full_path_rerank_diagnostics_path": str(full_path_rerank_diagnostics_path.resolve()),
        "full_path_rerank_diagnostics_summary": aggregate_rerank_diagnostics,
        "intervention_outcome_sensitivity_diagnostics_path": str(intervention_sensitivity_diagnostics_path.resolve()),
        "intervention_outcome_sensitivity_summary": intervention_sensitivity_diagnostics,
        "inference_only_ablation": bool(not schedule_aligned_inference),
        "schedule_aligned_inference": bool(schedule_aligned_inference),
        "schedule_ablation": str(args.inference_schedule_preset),
        "training_schedule_preset": str(args.schedule_preset),
        "inference_schedule_preset": str(args.inference_schedule_preset),
        "diagnostic_student_outcome_diagnostics_enabled": not bool(args.disable_rerank_diagnostics),
        "diagnostic_intervention_sensitivity_enabled": bool(intervention_sensitivity_enabled),
        "diagnostic_intervention_top_transition_k": int(args.intervention_top_transition_k),
        "diagnostic_rerank_diag_max_states": None if int(args.rerank_diag_max_states) <= 0 else int(args.rerank_diag_max_states),
        "diagnostic_rerank_diag_include_records": bool(diagnostic_include_records),
        "diagnostic_proposal_recall_top_ks": [int(k) for k in proposal_recall_top_ks],
        "med_afa_student_architecture_selection": {
            "mode": str(args.arch_selection_mode),
            "predictor_arch_grid": str(args.predictor_arch_grid),
            "candidate_count": int(len(a7_architecture_candidates)),
            "candidates": a7_architecture_candidates,
            "selected_trial_architectures": selected_architecture_trial_summaries,
            "trial_selection_summaries": val_constraint_selection_trial_summaries,
            "effective_default_hidden_source": "validation_constraint_selected_predictor_architecture",
            "effective_default_dropout_source": "validation_constraint_selected_predictor_architecture",
            "posthoc_predictor_adapt_enabled": bool(posthoc_predictor_adapt_enabled),
        },
        "med_afa_student_intervention_aux": {
            "enabled": bool(intervention_aux_enabled),
            "weight": float(args.intervention_aux_weight),
            "only_changed_actions": bool(args.intervention_aux_only_changed_actions),
            "mode": str(args.intervention_aux_mode),
            "oracle_margin": float(args.intervention_aux_oracle_margin),
            "trial_diagnostics": intervention_aux_trial_summaries,
        },
        "med_afa_student_posthoc_predictor_adapt": {
            "enabled": bool(posthoc_predictor_adapt_enabled),
            "epochs": int(args.posthoc_predictor_adapt_epochs),
            "lr": float(args.posthoc_predictor_adapt_lr),
            "trial_diagnostics": [
                rec.get("posthoc_predictor_adapt", {"enabled": False})
                for rec in trial_records
            ],
        },
        "training_one_step_prefix_steps": int(prefix_steps),
        "training_proposal_rerank_middle_steps": int(middle_steps),
        "training_one_step_suffix_steps": int(suffix_steps),
        "inference_one_step_prefix_steps": int(inference_prefix_steps),
        "inference_proposal_rerank_middle_steps": int(inference_middle_steps),
        "inference_one_step_suffix_steps": int(inference_suffix_steps),
        "paired_inference_schedule_presets": paired_presets,
        "paired_reference_schedule_preset": reference_preset or None,
        "paired_inference_ablation": paired_inference_ablation_summary,
        "one_step_prefix_steps": int(inference_prefix_steps),
        "proposal_rerank_middle_steps": int(inference_middle_steps),
        "one_step_suffix_steps": int(inference_suffix_steps),
        "schedule_ratios_actual": {
            "early": float(inference_prefix_steps / num_actions) if num_actions else None,
            "middle": float(inference_middle_steps / num_actions) if num_actions else None,
            "late": float(inference_suffix_steps / num_actions) if num_actions else None,
        },
        "training_schedule_ratios_actual": {
            "early": float(prefix_steps / num_actions) if num_actions else None,
            "middle": float(middle_steps / num_actions) if num_actions else None,
            "late": float(suffix_steps / num_actions) if num_actions else None,
        },
        "inference_schedule_ratios_actual": {
            "early": float(inference_prefix_steps / num_actions) if num_actions else None,
            "middle": float(inference_middle_steps / num_actions) if num_actions else None,
            "late": float(inference_suffix_steps / num_actions) if num_actions else None,
        },
        "dataset_csv": str(dataset_csv.resolve()),
        "actions_path": str(actions_path.resolve()),
        "label_col": label_col,
        "label_mapping": label_mapping,
        "split_path": str(split_path.resolve()),
        "split_seed": split_info.get("seed", None),
        "num_rows": int(len(x_norm)),
        "num_features": int(d_in),
        "num_actions": int(num_actions),
        "num_classes": int(d_out),
        "feature_names": feature_names,
        "action_groups": action_groups,
        "num_trials": int(args.num_trials),
        "seed_base": int(args.seed),
        "device": str(device),
        "normalize_mode": args.normalize_mode,
        "class_weight_enabled": bool(args.use_class_weight),
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "grouping_strategy": "medafa_action_groups",
        "budget_axis": "action",
        "constant_feature_cost": 1.0,
        "pretrain_epochs": int(args.pretrain_epochs),
        "train_epochs": int(args.train_epochs),
        "max_features_train": int(max_features_train),
        "max_eval_features": int(budget_max),
        "batch_size_train": int(args.batch_size_train),
        "batch_size_eval": int(args.batch_size_eval),
        "num_workers": int(args.num_workers),
        "eps_steps": int(args.eps_steps),
        "min_lr": float(args.min_lr),
        "grid_search_enabled": bool(args.do_grid_search),
        "candidate_cache": {
            "enabled": bool(candidate_cache_enabled),
            "root": str(candidate_cache_root),
            "force_retrain": bool(args.force_retrain_candidate_cache),
            "disable_candidate_cache": bool(args.disable_candidate_cache),
            "mode": "completed_candidate_model_cache",
        },
        "full_path_head_loss_weight": float(args.full_path_head_loss_weight),
        "proposal_top_k": int(args.proposal_top_k),
        "acc_by_num_features": mean_acc_by_num_features,
        "acc_by_num_actions": mean_acc_by_num_features,
        "acc_by_num_features_midpoint": mean_acc_by_num_features_midpoint,
        "acc_by_num_actions_midpoint": mean_acc_by_num_features_midpoint,
        "constraint_order_constraints": {
            "enabled": any(item.get("prerequisite_indices") for item in action_groups),
            "rule": "hard_prerequisite_mask_at_action_selection",
            "valid_samples_only_for_constraint_acc": True,
            "actions_with_prerequisites": [
                {
                    "action_id": str(item.get("action_id")),
                    "prerequisites": [str(x) for x in item.get("prerequisites", [])],
                }
                for item in action_groups
                if item.get("prerequisite_indices")
            ],
        },
        "constraint_valid_rate": constraint_valid_rate,
        "constraint_valid_n": int(constraint_valid_n),
        "constraint_total_n": int(constraint_total_n),
        "constraint_invalid_n": int(constraint_invalid_n),
        "constraint_mean_acc_at_all": None if not constraint_values else float(np.mean(constraint_values)),
        "constraint_final_acc": None if not constraint_values else constraint_curve[-1]["test_acc"],
        "constraint_acc_by_num_features_integer": constraint_curve,
        "unmasked_policy_constraint_valid_rate": unmasked_valid_rate,
        "unmasked_policy_constraint_valid_n": int(unmasked_valid_n),
        "unmasked_policy_constraint_total_n": int(unmasked_total_n),
        "unmasked_policy_constraint_invalid_n": int(unmasked_invalid_n),
        "unmasked_policy_constraint_violation_counts": unmasked_violation_counts,
        "trial_records": trial_records,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
    }

    trials_payload = []
    for rec in trial_records:
        trials_payload.append(
            {
                "trial": int(rec["trial"]),
                "seed": int(rec["seed"]),
                "selection": {
                    "score": float(rec["best_selection_score"]),
                    "architecture": rec["best_architecture"],
                    "validation_mean_acc@all": rec["best_val_constraint_metrics"].get("constraint_mean_acc_at_all"),
                    "validation_final_acc": rec["best_val_constraint_metrics"].get("constraint_final_acc"),
                },
                "metrics": {
                    "mean_acc@all": float(rec["constraint_mean_acc_at_all"]),
                    "final_acc": float(rec["constraint_final_acc"]),
                    "per_action_accuracy": rec["constraint_acc_by_num_features_integer"],
                    "constraint_valid_rate": rec["constraint_valid_rate"],
                },
            }
        )
    summary = {
        "dataset": args.dataset,
        "mean_acc@all": None if not constraint_values else float(np.mean(constraint_values)),
        "final_acc": None if not constraint_values else float(constraint_curve[-1]["test_acc"]),
        "per_action_accuracy": constraint_curve,
        "constraint_valid_rate": constraint_valid_rate,
        "num_trials": int(args.num_trials),
        "seed_base": int(args.seed),
        "trials": trials_payload,
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "trials.json").open("w", encoding="utf-8") as f:
        json.dump({"trials": trials_payload}, f, ensure_ascii=False, indent=2)
    if args.save_diagnostics:
        with double_head_diagnostics_path.open("w", encoding="utf-8") as f:
            json.dump(oracle_target_provider.double_head_diagnostics(), f, ensure_ascii=False, indent=2)
        with full_path_rerank_diagnostics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "target": "rerank_diagnostics",
                    "max_states_per_trial": None if int(args.rerank_diag_max_states) <= 0 else int(args.rerank_diag_max_states),
                    "include_records": bool(diagnostic_include_records),
                    "proposal_recall_top_ks": [int(k) for k in proposal_recall_top_ks],
                    "aggregate": aggregate_rerank_diagnostics,
                    "per_trial": trial_rerank_diagnostics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with intervention_sensitivity_diagnostics_path.open("w", encoding="utf-8") as f:
            json.dump(intervention_sensitivity_diagnostics, f, ensure_ascii=False, indent=2)

    if not args.keep_checkpoints:
        tmp_root = out_dir / "_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"[DONE] summary written to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
