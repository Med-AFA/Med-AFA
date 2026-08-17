from __future__ import annotations

import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "datasets").exists() and (parent / "run").exists():
            return parent
    raise FileNotFoundError(f"Could not find project root from {start}.")


REPO_ROOT = find_project_root(Path(__file__).resolve().parent)
RUN_ROOT = REPO_ROOT / "run"
STUDENT_ROOT = RUN_ROOT / "student"
DATASETS_ROOT = REPO_ROOT / "datasets"
for _path in (STUDENT_ROOT, RUN_ROOT, DATASETS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from action_constraints import (
    build_prerequisite_matrix,
    legal_action_mask,
    load_action_feature_matrix,
    validate_action_sequence,
)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_splits(text: str, valid: set[str] | None = None) -> list[str]:
    if valid is None:
        valid = {"train", "val", "test"}
    splits = [x.strip() for x in text.split(",") if x.strip()]
    bad = [x for x in splits if x not in valid]
    if bad:
        raise ValueError(f"Invalid splits: {bad}. Valid splits: {sorted(valid)}")
    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stage1PolicyPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.predictor_head = nn.Linear(hidden_dim, num_classes)
        self.policy_head = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.predictor_head(h), self.policy_head(h)


class RPCPolicyPredictor(nn.Module):


    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.predictor_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor_net(x), self.policy_net(x)

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor_net(x)

    def policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_net(x)


@dataclass
class TeacherArtifacts:
    teacher_ckpt_path: Path
    teacher_model_path: Path
    teacher_model: Any
    checkpoint: dict[str, Any]
    split_path: Path
    dataset_csv: Path
    label_col: str
    feature_columns: list[str]
    action_ids: list[str]
    action_id_to_idx: dict[str, int]
    action_groups: list[dict[str, Any]]
    prerequisite_matrix: torch.Tensor
    action_feature_matrix: torch.Tensor
    mean: np.ndarray
    std: np.ndarray
    missing_value: float
    num_classes: int


def _checkpoint_action_groups(
    checkpoint: dict[str, Any],
    *,
    action_ids: list[str],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    raw_actions = checkpoint.get("actions_resolved", [])
    by_id = {
        str(item.get("action_id", "")): item
        for item in raw_actions
        if isinstance(item, dict) and str(item.get("action_id", ""))
    }
    groups: list[dict[str, Any]] = []
    for action_idx, action_id in enumerate(action_ids):
        raw = by_id.get(str(action_id), {})
        feature_indices = [int(x) for x in raw.get("feature_indices", [])]
        feature_names = [
            str(feature_columns[i])
            for i in feature_indices
            if 0 <= int(i) < len(feature_columns)
        ]
        groups.append(
            {
                "action_id": str(action_id),
                "name": str(raw.get("name", action_id)),
                "feature_indices": feature_indices,
                "feature_names": feature_names,
                "prerequisites": [str(x) for x in raw.get("prerequisites", [])],
            }
        )

    action_id_to_idx = {str(item["action_id"]): i for i, item in enumerate(groups)}
    for item in groups:
        item["prerequisite_indices"] = [
            int(action_id_to_idx[str(x)])
            for x in item.get("prerequisites", [])
            if str(x) in action_id_to_idx
        ]
    return groups


def _current_action_groups(
    checkpoint: dict[str, Any],
    *,
    action_ids: list[str],
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    dataset_name = str(checkpoint.get("dataset_name", "")).strip()
    if dataset_name:
        try:
            _matrix, groups = load_action_feature_matrix(None, dataset_name, feature_columns)
            loaded_ids = [str(item.get("action_id", "")) for item in groups]
            if loaded_ids == [str(x) for x in action_ids]:
                return groups
        except Exception:
            pass
    return _checkpoint_action_groups(
        checkpoint,
        action_ids=action_ids,
        feature_columns=feature_columns,
    )


def load_teacher_artifacts(
    run_dir: Path,
    *,
    teacher_ckpt: str = "",
    device: torch.device | None = None,
) -> TeacherArtifacts:
    ckpt_path = Path(teacher_ckpt).resolve() if teacher_ckpt else (run_dir / "ckpts" / "teacher_best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_type = str(checkpoint.get("model_type", "")).strip().lower()
    supported_model_types = {"catboost_mask", "xgboost_mask", "logistic_regression_mask", "mlp_mask"}
    if model_type not in supported_model_types:
        raise ValueError(
            f"teacher_state expects a mask-aware teacher checkpoint with model_type in {sorted(supported_model_types)}, "
            f"got '{model_type or '<missing>'}': {ckpt_path}"
        )

    model_cfg = checkpoint["model_config"]
    model_path = Path(str(checkpoint.get("model_path", "")))
    if not str(model_path):
        raise ValueError(f"Teacher checkpoint does not contain model_path: {ckpt_path}")
    if not model_path.is_absolute():
        model_path = (ckpt_path.parent / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {model_path}")
    if model_type == "catboost_mask":
        teacher_model = load_catboost_model(model_path)
    else:
        with model_path.open("rb") as f:
            teacher_model = pickle.load(f)

    action_ids = [str(x) for x in checkpoint["action_ids"]]
    action_id_to_idx = {a: i for i, a in enumerate(action_ids)}
    feature_columns = [str(x) for x in checkpoint["feature_columns"]]
    action_groups = _current_action_groups(
        checkpoint,
        action_ids=action_ids,
        feature_columns=feature_columns,
    )
    prerequisite_matrix = build_prerequisite_matrix(action_groups).to(
        device=device if device is not None else "cpu",
        dtype=torch.float32,
    )
    action_feature_matrix = torch.tensor(
        np.asarray(checkpoint["action_feature_matrix"], dtype=np.float32),
        dtype=torch.float32,
        device=device if device is not None else "cpu",
    )

    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    std = np.clip(std, 1e-6, None)

    split_path = Path(checkpoint["split_path"]).resolve()
    dataset_csv = Path(checkpoint["dataset_csv"]).resolve()

    return TeacherArtifacts(
        teacher_ckpt_path=ckpt_path,
        teacher_model_path=model_path,
        teacher_model=teacher_model,
        checkpoint=checkpoint,
        split_path=split_path,
        dataset_csv=dataset_csv,
        label_col=str(checkpoint["label_col"]),
        feature_columns=feature_columns,
        action_ids=action_ids,
        action_id_to_idx=action_id_to_idx,
        action_groups=action_groups,
        prerequisite_matrix=prerequisite_matrix,
        action_feature_matrix=action_feature_matrix,
        mean=mean,
        std=std,
        missing_value=float(checkpoint.get("missing_value", -1.0)),
        num_classes=int(model_cfg["num_classes"]),
    )


def load_catboost_model(model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model not found: {model_path}")
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        raise RuntimeError("catboost is required for teacher_state CatBoost teacher/predictor support.") from exc

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model


def predict_catboost_proba_2d(model: Any, state_2d: np.ndarray, *, num_classes: int) -> np.ndarray:
    x = np.asarray(state_2d, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    proba = np.asarray(model.predict_proba(x), dtype=np.float64)
    if proba.ndim == 1:
        if num_classes == 2:
            proba = np.stack([1.0 - proba, proba], axis=1)
        else:
            proba = proba.reshape(-1, 1)
    if proba.shape[1] == num_classes:
        return proba

    aligned = np.zeros((proba.shape[0], num_classes), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        for src_idx, cls in enumerate(list(classes)):
            cls_idx = int(cls)
            if 0 <= cls_idx < num_classes and src_idx < proba.shape[1]:
                aligned[:, cls_idx] = proba[:, src_idx]
    else:
        width = min(num_classes, proba.shape[1])
        aligned[:, :width] = proba[:, :width]
    row_sum = aligned.sum(axis=1, keepdims=True)
    empty = row_sum.squeeze(1) <= 0.0
    if np.any(empty):
        aligned[empty, :] = 1.0 / float(num_classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(row_sum, 1e-12, None)


def predict_catboost_state(model: Any, state: torch.Tensor | np.ndarray, *, num_classes: int) -> dict[str, Any]:
    if isinstance(state, torch.Tensor):
        state_np = state.detach().cpu().numpy()
    else:
        state_np = np.asarray(state, dtype=np.float32)
    proba = predict_catboost_proba_2d(model, state_np, num_classes=num_classes)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    return {
        "proba": proba,
        "prediction": pred.astype(np.int64),
        "confidence": conf.astype(np.float64),
    }


def catboost_true_logprob(model: Any, state: torch.Tensor | np.ndarray, label: int, *, num_classes: int) -> float:
    pred = predict_catboost_state(model, state, num_classes=num_classes)
    probs = pred["proba"][0]
    label_idx = int(label)
    if label_idx < 0 or label_idx >= num_classes:
        raise ValueError(f"Label {label_idx} is out of range for num_classes={num_classes}")
    return float(np.log(max(float(probs[label_idx]), 1e-12)))


def softmax_masked_np(values: np.ndarray, mask: np.ndarray, *, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(values, dtype=np.float64)
    if not np.any(mask):
        return out
    temp = max(float(temperature), 1e-8)
    logits = values[mask] / temp
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    denom = float(exp_logits.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        out[mask] = 1.0 / float(mask.sum())
    else:
        out[mask] = exp_logits / denom
    return out


def margin_aware_temperature(
    *,
    values: np.ndarray,
    mask: np.ndarray,
    base_temperature: float,
    mode: str = "margin",
    margin_ref: float = 0.05,
    min_temperature: float = 0.03,
    max_temperature: float = 0.5,
) -> tuple[float, float]:


    base = max(float(base_temperature), 1e-8)
    mode = str(mode).strip().lower()
    if mode == "fixed":
        return base, 0.0
    if mode != "margin":
        raise ValueError(f"Invalid utility temperature mode: {mode}. Expected 'fixed' or 'margin'.")

    valid_values = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if valid_values.size <= 1:
        return base, 0.0
    sorted_values = np.sort(valid_values)[::-1]
    margin = float(sorted_values[0] - sorted_values[1])
    ref = max(float(margin_ref), 1e-8)
    temp = base * (ref / max(margin, 1e-8))
    temp = min(max(temp, float(min_temperature)), float(max_temperature))
    return float(temp), margin


@dataclass
class DatasetArrays:
    x_raw: np.ndarray
    x_norm: np.ndarray
    present: np.ndarray
    y: np.ndarray


def load_dataset_arrays(
    *,
    dataset_csv: Path,
    feature_columns: list[str],
    label_col: str,
    missing_value: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> DatasetArrays:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    df = pd.read_csv(dataset_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in dataset.")

    x_raw = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(missing_value)
        .to_numpy(dtype=np.float32)
    )
    y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int64)
    present = (np.isfinite(x_raw) & (x_raw != missing_value)).astype(np.float32)
    x_norm = ((x_raw - mean) / std).astype(np.float32)
    return DatasetArrays(x_raw=x_raw, x_norm=x_norm, present=present, y=y)


def build_state_vector(
    *,
    x_norm_row: torch.Tensor,
    present_row: torch.Tensor,
    m_act: torch.Tensor,
    action_feature_matrix: torch.Tensor,
) -> torch.Tensor:
    m_feat_selected = (m_act @ action_feature_matrix > 0).float()
    m_feat_observed = m_feat_selected * present_row
    x_obs = x_norm_row * m_feat_observed
    return torch.cat([x_obs, m_feat_observed, m_act], dim=0)


def build_state_batch(
    *,
    x_norm: torch.Tensor,
    present: torch.Tensor,
    m_act: torch.Tensor,
    action_feature_matrix: torch.Tensor,
) -> torch.Tensor:
    m_feat_selected = torch.clamp(m_act @ action_feature_matrix, min=0.0, max=1.0)
    m_feat_observed = m_feat_selected * present
    x_obs = x_norm * m_feat_observed
    return torch.cat([x_obs, m_feat_observed, m_act], dim=1)


def apply_hard_action_batch(m_act: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
    out = m_act.clone()
    out.scatter_(1, action_idx.view(-1, 1), 1.0)
    return out


def soft_next_state_from_logits(
    *,
    x_norm: torch.Tensor,
    present: torch.Tensor,
    m_act: torch.Tensor,
    policy_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    action_feature_matrix: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    temp = max(float(temperature), 1e-6)
    masked_logits = masked_policy_logits(policy_logits, candidate_mask) / temp
    action_prob = torch.softmax(masked_logits, dim=1) * candidate_mask.float()
    action_prob = action_prob / action_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    soft_m_act = torch.clamp(m_act + action_prob, min=0.0, max=1.0)
    soft_state = build_state_batch(
        x_norm=x_norm,
        present=present,
        m_act=soft_m_act,
        action_feature_matrix=action_feature_matrix,
    )
    return soft_state, soft_m_act, action_prob


def candidate_mask_for_state(
    m_act: torch.Tensor,
    prerequisite_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    if prerequisite_matrix is None:
        return m_act < 0.5
    return legal_action_mask(
        m_act,
        prerequisite_matrix.to(device=m_act.device, dtype=torch.float32),
    ).bool()


def compute_catboost_teacher_best_action(
    *,
    teacher_model: Any,
    x_norm_row: torch.Tensor,
    present_row: torch.Tensor,
    m_act: torch.Tensor,
    label: int,
    action_feature_matrix: torch.Tensor,
    action_ids: list[str],
    num_classes: int,
    prerequisite_matrix: torch.Tensor | None = None,
) -> dict[str, Any]:
    candidate_mask_tensor = candidate_mask_for_state(m_act, prerequisite_matrix)
    candidate_indices = [int(i) for i in torch.where(candidate_mask_tensor)[0].detach().cpu().tolist()]
    if not candidate_indices:
        raise ValueError("No legal candidate action remains for CatBoost teacher action selection.")

    current_state = build_state_vector(
        x_norm_row=x_norm_row,
        present_row=present_row,
        m_act=m_act,
        action_feature_matrix=action_feature_matrix,
    )
    current_value = catboost_true_logprob(
        teacher_model,
        current_state,
        int(label),
        num_classes=num_classes,
    )

    best_idx = candidate_indices[0]
    best_value = -float("inf")
    best_proba: np.ndarray | None = None
    for action_idx in candidate_indices:
        next_m_act = m_act.clone()
        next_m_act[action_idx] = 1.0
        next_state = build_state_vector(
            x_norm_row=x_norm_row,
            present_row=present_row,
            m_act=next_m_act,
            action_feature_matrix=action_feature_matrix,
        )
        pred = predict_catboost_state(teacher_model, next_state, num_classes=num_classes)
        probs = pred["proba"][0]
        value = float(np.log(max(float(probs[int(label)]), 1e-12)))
        if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and action_idx < best_idx):
            best_idx = int(action_idx)
            best_value = value
            best_proba = probs

    if best_proba is None:
        raise RuntimeError("Failed to compute CatBoost teacher best action.")
    return {
        "action_idx": int(best_idx),
        "action_id": str(action_ids[best_idx]),
        "value": float(best_value),
        "current_value": float(current_value),
        "delta": float(best_value - current_value),
        "true_prob": float(best_proba[int(label)]),
        "prediction": int(np.argmax(best_proba)),
        "confidence": float(np.max(best_proba)),
    }


def compute_catboost_soft_teacher(
    *,
    teacher_model: Any,
    x_norm_row: torch.Tensor,
    present_row: torch.Tensor,
    m_act: torch.Tensor,
    label: int,
    action_feature_matrix: torch.Tensor,
    action_ids: list[str],
    num_classes: int,
    prerequisite_matrix: torch.Tensor | None = None,
    utility_temperature: float = 0.1,
    utility_temperature_mode: str = "margin",
    utility_margin_ref: float = 0.05,
    utility_min_temperature: float = 0.03,
    utility_max_temperature: float = 0.5,
) -> dict[str, Any]:


    num_actions = len(action_ids)
    candidate_mask_tensor = candidate_mask_for_state(m_act, prerequisite_matrix)
    candidate_indices = [int(i) for i in torch.where(candidate_mask_tensor)[0].detach().cpu().tolist()]
    if not candidate_indices:
        raise ValueError("No legal candidate action remains for CatBoost teacher action selection.")

    current_state = build_state_vector(
        x_norm_row=x_norm_row,
        present_row=present_row,
        m_act=m_act,
        action_feature_matrix=action_feature_matrix,
    )
    current_value = catboost_true_logprob(
        teacher_model,
        current_state,
        int(label),
        num_classes=num_classes,
    )

    candidate_mask = np.zeros(num_actions, dtype=bool)
    utility = np.full(num_actions, -np.inf, dtype=np.float64)
    value = np.full(num_actions, -np.inf, dtype=np.float64)
    true_prob = np.zeros(num_actions, dtype=np.float64)
    prediction = np.full(num_actions, -1, dtype=np.int64)
    confidence = np.zeros(num_actions, dtype=np.float64)

    for action_idx in candidate_indices:
        next_m_act = m_act.clone()
        next_m_act[action_idx] = 1.0
        next_state = build_state_vector(
            x_norm_row=x_norm_row,
            present_row=present_row,
            m_act=next_m_act,
            action_feature_matrix=action_feature_matrix,
        )
        pred = predict_catboost_state(teacher_model, next_state, num_classes=num_classes)
        probs = pred["proba"][0]
        action_value = float(np.log(max(float(probs[int(label)]), 1e-12)))
        candidate_mask[action_idx] = True
        value[action_idx] = action_value
        utility[action_idx] = action_value - current_value
        true_prob[action_idx] = float(probs[int(label)])
        prediction[action_idx] = int(np.argmax(probs))
        confidence[action_idx] = float(np.max(probs))

    best_idx = int(candidate_indices[0])
    best_utility = -float("inf")
    for action_idx in candidate_indices:
        action_utility = float(utility[action_idx])
        if action_utility > best_utility + 1e-12 or (
            abs(action_utility - best_utility) <= 1e-12 and action_idx < best_idx
        ):
            best_idx = int(action_idx)
            best_utility = action_utility

    finite_utility = np.where(candidate_mask, utility, 0.0)
    effective_temperature, top1_top2_margin = margin_aware_temperature(
        values=finite_utility,
        mask=candidate_mask,
        base_temperature=float(utility_temperature),
        mode=utility_temperature_mode,
        margin_ref=float(utility_margin_ref),
        min_temperature=float(utility_min_temperature),
        max_temperature=float(utility_max_temperature),
    )
    soft_distribution = softmax_masked_np(
        finite_utility,
        candidate_mask,
        temperature=float(effective_temperature),
    )
    return {
        "action_idx": int(best_idx),
        "action_id": str(action_ids[best_idx]),
        "value": float(value[best_idx]),
        "current_value": float(current_value),
        "delta": float(utility[best_idx]),
        "true_prob": float(true_prob[best_idx]),
        "prediction": int(prediction[best_idx]),
        "confidence": float(confidence[best_idx]),
        "candidate_mask": candidate_mask.tolist(),
        "utility": finite_utility.astype(np.float64).tolist(),
        "value_by_action": np.where(candidate_mask, value, 0.0).astype(np.float64).tolist(),
        "soft_distribution": soft_distribution.astype(np.float64).tolist(),
        "utility_temperature": float(utility_temperature),
        "effective_utility_temperature": float(effective_temperature),
        "utility_temperature_mode": str(utility_temperature_mode),
        "utility_top1_top2_margin": float(top1_top2_margin),
    }


def apply_action_indices_to_m_act(m_act: torch.Tensor, action_indices: list[int]) -> torch.Tensor:
    out = m_act.clone()
    for action_idx in action_indices:
        if 0 <= int(action_idx) < int(out.numel()):
            out[int(action_idx)] = 1.0
    return out


def load_global_init_config(teacher_root: Path) -> dict[str, Any]:
    path = teacher_root / "summary" / "teacher_state_global_init.json"
    if not path.exists():
        return {"global_init_k": 0, "action_indices": [], "action_ids": []}
    payload = load_json(path)
    return {
        "global_init_k": int(payload.get("global_init_k", 0)),
        "action_indices": [int(x) for x in payload.get("action_indices", [])],
        "action_ids": [str(x) for x in payload.get("action_ids", [])],
        "path": str(path),
    }


def evaluate_learned_rollout(
    *,
    policy_model: nn.Module,
    predictor_model: nn.Module,
    teacher_art: TeacherArtifacts,
    arrays: DatasetArrays,
    split_indices: dict[str, list[int]],
    split_name: str,
    max_samples: int,
    device: torch.device,
    global_init_action_indices: list[int] | None = None,
) -> dict[str, Any]:


    policy_was_training = policy_model.training
    predictor_was_training = predictor_model.training
    policy_model.eval()
    predictor_model.eval()

    num_actions = len(teacher_art.action_ids)
    sample_ids = list(split_indices[split_name])
    if max_samples > 0:
        sample_ids = sample_ids[:max_samples]

    per_step_correct = [0 for _ in range(num_actions)]
    per_step_total = [0 for _ in range(num_actions)]

    with torch.no_grad():
        for sample_idx in sample_ids:
            label = int(arrays.y[sample_idx])
            x_norm_row = torch.tensor(arrays.x_norm[sample_idx], dtype=torch.float32, device=device)
            present_row = torch.tensor(arrays.present[sample_idx], dtype=torch.float32, device=device)
            m_act = torch.zeros(num_actions, dtype=torch.float32, device=device)

            init_indices = list(global_init_action_indices or [])
            for init_pos, action_idx in enumerate(init_indices):
                if not bool((m_act < 0.5).any().item()):
                    break
                action_idx = int(action_idx)
                if action_idx < 0 or action_idx >= num_actions or m_act[action_idx].item() >= 0.5:
                    continue
                m_act[action_idx] = 1.0
                state_after = build_state_vector(
                    x_norm_row=x_norm_row,
                    present_row=present_row,
                    m_act=m_act,
                    action_feature_matrix=teacher_art.action_feature_matrix,
                )
                pred_logits = predictor_model.predict_logits(state_after.unsqueeze(0))
                pred_cls = int(pred_logits.argmax(dim=1).item())
                step_idx = int(init_pos)
                if step_idx < num_actions:
                    per_step_total[step_idx] += 1
                    if pred_cls == label:
                        per_step_correct[step_idx] += 1

            for step_idx in range(len(init_indices), num_actions):
                if not bool((m_act < 0.5).any().item()):
                    break
                state_before = build_state_vector(
                    x_norm_row=x_norm_row,
                    present_row=present_row,
                    m_act=m_act,
                    action_feature_matrix=teacher_art.action_feature_matrix,
                )
                policy_logits = policy_model.policy_logits(state_before.unsqueeze(0))
                candidate_mask = candidate_mask_for_state(
                    m_act,
                    teacher_art.prerequisite_matrix,
                ).unsqueeze(0)
                if not bool(candidate_mask.any().item()):
                    break
                action_idx = int(masked_policy_logits(policy_logits, candidate_mask).argmax(dim=1).item())
                m_act[action_idx] = 1.0

                state_after = build_state_vector(
                    x_norm_row=x_norm_row,
                    present_row=present_row,
                    m_act=m_act,
                    action_feature_matrix=teacher_art.action_feature_matrix,
                )
                pred_logits = predictor_model.predict_logits(state_after.unsqueeze(0))
                pred_cls = int(pred_logits.argmax(dim=1).item())
                per_step_total[step_idx] += 1
                if pred_cls == label:
                    per_step_correct[step_idx] += 1

    if policy_was_training:
        policy_model.train()
    if predictor_was_training:
        predictor_model.train()

    per_step_accuracy: list[dict[str, Any]] = []
    values: list[float] = []
    budget_mean_accuracy: dict[str, float] = {}
    for step_idx in range(num_actions):
        denom = max(int(per_step_total[step_idx]), 1)
        acc = float(per_step_correct[step_idx] / denom)
        values.append(acc)
        budget_mean_accuracy[str(step_idx + 1)] = float(sum(values) / float(len(values)))
        per_step_accuracy.append(
            {
                "step": int(step_idx + 1),
                "num_actions_selected": int(step_idx + 1),
                "accuracy": acc,
            }
        )

    mean_acc_at_all = float(sum(values) / float(len(values))) if values else 0.0
    budget_mean_accuracy["all"] = mean_acc_at_all
    return {
        "split": split_name,
        "num_samples": int(len(sample_ids)),
        "max_samples": int(max_samples),
        "predictor": "learned_predictor",
        "per_step_accuracy": per_step_accuracy,
        "budget_mean_accuracy": budget_mean_accuracy,
        "mean_acc_at_all": mean_acc_at_all,
        "final_accuracy": float(per_step_accuracy[-1]["accuracy"]) if per_step_accuracy else 0.0,
    }


def build_m_act_from_ids(action_ids: list[str], action_id_to_idx: dict[str, int], num_actions: int) -> torch.Tensor:
    m_act = torch.zeros(num_actions, dtype=torch.float32)
    for aid in action_ids:
        idx = action_id_to_idx.get(str(aid))
        if idx is not None:
            m_act[idx] = 1.0
    return m_act


def candidate_mask_from_ids(candidate_action_ids: list[str], action_id_to_idx: dict[str, int], num_actions: int) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.bool)
    for aid in candidate_action_ids:
        idx = action_id_to_idx.get(str(aid))
        if idx is not None:
            mask[idx] = True
    return mask


def masked_policy_logits(policy_logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    fill_value = torch.finfo(policy_logits.dtype).min
    return policy_logits.masked_fill(~candidate_mask, fill_value)


def masked_policy_ce_loss(policy_logits: torch.Tensor, teacher_action: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(masked_policy_logits(policy_logits, candidate_mask), teacher_action)


def load_policy_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = ckpt["model_config"]
    model_type = str(model_cfg.get("model_type", ckpt.get("model_type", "rpc_policy_predictor"))).strip().lower()
    model_cls: type[nn.Module]
    if model_type in {"stage1_policy_predictor", "stage1_policy_predictor"}:
        model_cls = Stage1PolicyPredictor
    else:
        model_cls = RPCPolicyPredictor
    model = model_cls(
        input_dim=int(model_cfg["input_dim"]),
        num_classes=int(model_cfg["num_classes"]),
        num_actions=int(model_cfg["num_actions"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_stage1_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    return load_policy_model_from_ckpt(ckpt_path, device=device)


def ensure_source_flag(payload: dict[str, torch.Tensor], source_flag: int = 0) -> dict[str, torch.Tensor]:
    if "source_flag" in payload:
        return payload
    if "state" not in payload:
        raise KeyError("Tensor payload missing required key: state")
    out = dict(payload)
    out["source_flag"] = torch.full((int(out["state"].shape[0]),), int(source_flag), dtype=torch.long)
    return out


def combine_tensor_dicts(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not parts:
        raise ValueError("No tensor parts to combine.")
    keys = set(parts[0].keys())
    for p in parts[1:]:
        if set(p.keys()) != keys:
            raise ValueError("Tensor dict keys mismatch while combining.")
    out: dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        out[k] = torch.cat([p[k] for p in parts], dim=0)
    return out


def load_split_indices(split_path: Path) -> dict[str, list[int]]:
    payload = load_json(split_path)
    indices = payload["indices"]
    return {
        "train": [int(x) for x in indices["train"]],
        "val": [int(x) for x in indices["val"]],
        "test": [int(x) for x in indices["test"]],
    }


def filter_rows_by_max_samples(rows: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return rows
    allowed_samples: set[int] = set()
    for row in rows:
        sample_idx = int(row.get("sample_index", -1))
        if sample_idx >= 0:
            allowed_samples.add(sample_idx)
        if len(allowed_samples) >= max_samples:
            break
    return [r for r in rows if int(r.get("sample_index", -1)) in allowed_samples]
