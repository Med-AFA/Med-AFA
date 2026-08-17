

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

RUN_SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT_DIR = RUN_SCRIPT_DIR.parent
if str(RUN_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT_DIR))

from action_dataset_utils import (
    build_action_feature_matrix,
    class_distribution,
    compute_feature_stats,
    create_stratified_split_indices,
    load_actions,
    resolve_actions,
    save_json,
    standardize_features,
    uncovered_features,
)
from masked_teacher_utils import build_state_batch_np, predict_proba_2d


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "datasets").exists() and (parent / "run").exists():
            return parent
    raise FileNotFoundError(f"Could not find project root from {start}.")


BACKBONE_ALIASES = {
    "cat": "catboost",
    "catboost": "catboost",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "logistic": "logistic_regression",
    "lr": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "mlp": "mlp",
}


def normalize_dataset_name(text: str) -> str:
    normalized = str(text).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("--dataset_name must be non-empty.")
    return normalized


def normalize_version_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = str(text).strip()
    if normalized.lower() in {"none", "null", "na", "-"}:
        return ""
    return normalized


def resolve_dataset_options(
    *,
    project_root: Path,
    dataset_name: str,
    dataset_csv_arg: str,
    actions_json_arg: str,
    label_col_arg: str,
    version_arg: str | None,
) -> tuple[str, Path, Path, str, str]:
    del project_root
    dataset_key = normalize_dataset_name(dataset_name)
    if not dataset_csv_arg.strip():
        raise ValueError("--dataset_csv is required.")
    if not actions_json_arg.strip():
        raise ValueError("--actions_json is required.")
    if not label_col_arg.strip():
        raise ValueError("--label_col is required.")
    return (
        dataset_key,
        Path(dataset_csv_arg).resolve(),
        Path(actions_json_arg).resolve(),
        label_col_arg.strip(),
        normalize_version_text(version_arg),
    )


def build_dataset_root(results_root: Path, dataset_name: str, version: str) -> Path:
    dataset_root = results_root / str(dataset_name).strip()
    version_text = str(version).strip()
    return dataset_root / version_text if version_text else dataset_root


def prepare_paths(args: argparse.Namespace) -> dict[str, Path | str]:
    results_root = Path(args.results_root).resolve()
    dataset_root = build_dataset_root(results_root, args.dataset_name, args.version)
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_root / "runs" / run_id
    paths = {
        "run_id": run_id,
        "dataset_root": dataset_root,
        "run_dir": run_dir,
        "split_path": dataset_root / "split" / f"split_seed{args.seed}.json",
        "ckpt_dir": run_dir / "ckpts",
        "metrics_dir": run_dir / "metrics",
        "meta_dir": run_dir / "meta",
    }
    for key in ("ckpt_dir", "metrics_dir", "meta_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    Path(paths["split_path"]).parent.mkdir(parents=True, exist_ok=True)
    return paths


def parse_int_list(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Invalid integer list: {text}")
    return vals


def parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"Invalid float list: {text}")
    return vals


def parse_backbones(text: str) -> list[str]:
    items = [x.strip().lower().replace("-", "_").replace(" ", "_") for x in str(text).split(",") if x.strip()]
    if not items or items == ["all"]:
        items = ["catboost", "xgboost", "logistic_regression", "mlp"]
    out: list[str] = []
    for item in items:
        if item not in BACKBONE_ALIASES:
            raise ValueError(f"Unsupported teacher backbone '{item}'. Use catboost,xgboost,logistic_regression,mlp.")
        model_type = BACKBONE_ALIASES[item]
        if model_type not in out:
            out.append(model_type)
    return out


def baseline_dataset_name(dataset_name: str) -> str:
    return normalize_dataset_name(dataset_name)


def baseline_report_path(project_root: Path, dataset_name: str, model_type: str, *, grid_results: bool) -> Path:
    baseline_key = baseline_dataset_name(dataset_name)
    method_dir = {
        "catboost": "CatBoost",
        "xgboost": "XGboost",
        "logistic_regression": "Logistic Regression",
        "mlp": "MLP",
    }[model_type]
    file_model = {
        "catboost": "catboost",
        "xgboost": "xgboost",
        "logistic_regression": "logistic_regression",
        "mlp": "mlp",
    }[model_type]
    suffix = "grid_results" if grid_results else "best_multi_seed_report"
    return (
        project_root
        / "results"
        / "baselines"
        / method_dir
        / "grid_search_outputs"
        / baseline_key
        / f"baseline_{baseline_key}_full_{file_model}_{suffix}.json"
    )


def load_baseline_best_params(project_root: Path, dataset_name: str, model_type: str) -> tuple[dict[str, Any], str]:
    for grid_results in (True, False):
        path = baseline_report_path(project_root, dataset_name, model_type, grid_results=grid_results)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        params = payload.get("best_params", {})
        if not isinstance(params, dict) or not params:
            params = (payload.get("grid") or {}).get("best_params", {})
        if isinstance(params, dict) and params:
            return dict(params), str(path)
    return {}, ""


def resolve_catboost_grid_values(args: argparse.Namespace, baseline_params: dict[str, Any]) -> dict[str, list[Any]]:
    def ints(raw: str, fallback: list[int]) -> list[int]:
        return fallback if str(raw).strip().lower() == "auto" else parse_int_list(raw)

    def floats(raw: str, fallback: list[float]) -> list[float]:
        return fallback if str(raw).strip().lower() == "auto" else parse_float_list(raw)

    return {
        "iterations": ints(args.iterations_list, [300, 500]),
        "depth": ints(args.depth_list, [4, 6]),
        "learning_rate": floats(args.learning_rate_list, sorted({0.03, 0.05, float(baseline_params.get("learning_rate", 0.03))})),
        "subsample": floats(args.subsample_list, sorted({0.8, 1.0, float(baseline_params.get("subsample", 1.0))})),
        "rsm": floats(args.rsm_list, sorted({0.8, 1.0, float(baseline_params.get("rsm", 1.0))})),
        "l2_leaf_reg": floats(args.l2_leaf_reg_list, sorted({3.0, 5.0, float(baseline_params.get("l2_leaf_reg", 5.0))})),
    }


def build_catboost_grid_combinations(
    grid_values: dict[str, list[Any]],
    n_jobs: int,
    max_grid_combinations: int,
) -> list[dict[str, Any]]:
    combos: list[dict[str, Any]] = []
    for vals in itertools.product(
        grid_values["iterations"],
        grid_values["depth"],
        grid_values["learning_rate"],
        grid_values["subsample"],
        grid_values["rsm"],
        grid_values["l2_leaf_reg"],
    ):
        combos.append(
            {
                "iterations": int(vals[0]),
                "depth": int(vals[1]),
                "learning_rate": float(vals[2]),
                "subsample": float(vals[3]),
                "rsm": float(vals[4]),
                "l2_leaf_reg": float(vals[5]),
                "n_jobs": int(n_jobs),
            }
        )
    if max_grid_combinations > 0:
        combos = combos[:max_grid_combinations]
    if not combos:
        raise ValueError("No CatBoost grid combinations generated.")
    return combos


def fallback_params(model_type: str, n_jobs: int) -> dict[str, Any]:
    if model_type == "catboost":
        return {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "subsample": 1.0,
            "rsm": 1.0,
            "l2_leaf_reg": 5.0,
            "n_jobs": int(n_jobs),
        }
    if model_type == "xgboost":
        return {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
            "n_jobs": int(n_jobs),
            "tree_method": "hist",
        }
    if model_type == "logistic_regression":
        return {
            "solver": "lbfgs",
            "c_value": 1.0,
            "max_iter": 2000,
            "tol": 1e-4,
            "class_weight": "none",
            "n_jobs": int(n_jobs),
        }
    if model_type == "mlp":
        return {
            "epochs": 500,
            "batch_size_train": "auto",
            "lr": 0.001,
            "weight_decay": 1e-4,
            "n_hidden": 128,
            "patience": 20,
        }
    raise ValueError(model_type)


def merge_params(model_type: str, baseline_params: dict[str, Any], n_jobs: int) -> dict[str, Any]:
    params = fallback_params(model_type, n_jobs)
    params.update({k: v for k, v in baseline_params.items() if v is not None})
    if model_type == "logistic_regression":
        if "C" in params and "c_value" not in params:
            params["c_value"] = params["C"]
    if model_type == "mlp":
        if "max_iter" in params and "epochs" not in params:
            params["epochs"] = params["max_iter"]
        if "learning_rate_init" in params and "lr" not in params:
            params["lr"] = params["learning_rate_init"]
    params["n_jobs"] = int(params.get("n_jobs", n_jobs))
    return params


def build_action_masks(*, num_samples: int, num_actions: int, masks_per_sample: int, seed: int) -> np.ndarray:
    if masks_per_sample <= 0:
        raise ValueError("--*_masks_per_sample must be >= 1.")
    rng = np.random.default_rng(seed)
    masks = np.zeros((num_samples, masks_per_sample, num_actions), dtype=np.float32)
    if masks_per_sample >= 2:
        masks[:, 1, :] = 1.0
    start = min(masks_per_sample, 2)
    if start < masks_per_sample:
        n = num_samples * (masks_per_sample - start)
        uniforms = rng.random((n, num_actions), dtype=np.float32)
        refs = rng.random((n, 1), dtype=np.float32)
        random_masks = (uniforms > refs).astype(np.float32)
        masks[:, start:, :] = random_masks.reshape(num_samples, masks_per_sample - start, num_actions)
    return masks.reshape(num_samples * masks_per_sample, num_actions)


def build_masked_state_dataset(
    *,
    x_norm: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    action_feature_matrix: np.ndarray,
    masks_per_sample: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_count = int(len(indices))
    masks = build_action_masks(
        num_samples=sample_count,
        num_actions=int(action_feature_matrix.shape[0]),
        masks_per_sample=masks_per_sample,
        seed=seed,
    )
    repeated_idx = np.repeat(indices.astype(np.int64), masks_per_sample)
    states = build_state_batch_np(
        x_norm=x_norm[repeated_idx],
        feature_present_mask=present[repeated_idx],
        m_act=masks,
        action_feature_matrix=action_feature_matrix,
    )
    labels = y[repeated_idx].astype(np.int64)
    return states, labels, masks, repeated_idx


def build_fixed_state_dataset(
    *,
    x_norm: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    action_feature_matrix: np.ndarray,
    value: float,
) -> tuple[np.ndarray, np.ndarray]:
    m_act = np.full((len(indices), action_feature_matrix.shape[0]), float(value), dtype=np.float32)
    states = build_state_batch_np(
        x_norm=x_norm[indices],
        feature_present_mask=present[indices],
        m_act=m_act,
        action_feature_matrix=action_feature_matrix,
    )
    return states, y[indices].astype(np.int64)


def build_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return (counts.sum() / (num_classes * counts)).astype(np.float32)


def init_model(*, model_type: str, params: dict[str, Any], num_classes: int, seed: int) -> Any:
    if model_type == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed.")
        common: dict[str, Any] = {
            "iterations": int(params["iterations"]),
            "depth": int(params["depth"]),
            "learning_rate": float(params["learning_rate"]),
            "subsample": float(params["subsample"]),
            "rsm": float(params["rsm"]),
            "l2_leaf_reg": float(params["l2_leaf_reg"]),
            "random_seed": int(seed),
            "thread_count": int(params.get("n_jobs", -1)),
            "bootstrap_type": "Bernoulli",
            "allow_writing_files": False,
            "verbose": False,
        }
        if num_classes <= 2:
            return CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **common)
        return CatBoostClassifier(loss_function="MultiClass", eval_metric="MultiClass", **common)
    if model_type == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed.")
        objective = "binary:logistic" if num_classes <= 2 else "multi:softprob"
        kwargs = {
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "subsample": float(params["subsample"]),
            "colsample_bytree": float(params["colsample_bytree"]),
            "min_child_weight": float(params["min_child_weight"]),
            "reg_lambda": float(params["reg_lambda"]),
            "objective": objective,
            "random_state": int(seed),
            "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
            "n_jobs": int(params.get("n_jobs", -1)),
            "tree_method": str(params.get("tree_method", "hist")),
        }
        if num_classes > 2:
            kwargs["num_class"] = int(num_classes)
        return XGBClassifier(**kwargs)
    if model_type == "logistic_regression":
        class_weight = params.get("class_weight", None)
        if str(class_weight).lower() in {"none", "", "null"}:
            class_weight = None
        return LogisticRegression(
            penalty="l2",
            C=float(params["c_value"]),
            solver=str(params["solver"]),
            max_iter=int(params["max_iter"]),
            tol=float(params["tol"]),
            class_weight=class_weight,
            random_state=int(seed),
            n_jobs=int(params.get("n_jobs", -1)),
        )
    if model_type == "mlp":
        n_hidden = int(params.get("n_hidden", 128))
        batch_size = params.get("batch_size_train", "auto")
        if isinstance(batch_size, str) and batch_size.strip().lower() == "auto":
            parsed_batch_size: int | str = "auto"
        else:
            parsed_batch_size = int(batch_size)
        return MLPClassifier(
            hidden_layer_sizes=(n_hidden, n_hidden),
            alpha=float(params.get("weight_decay", 1e-4)),
            learning_rate_init=float(params.get("lr", 0.001)),
            batch_size=parsed_batch_size,
            max_iter=int(params.get("epochs", 500)),
            random_state=int(seed),
            early_stopping=True,
            n_iter_no_change=int(params.get("patience", 20)),
        )
    raise ValueError(model_type)


def evaluate_proba(proba: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64)
    idx = np.arange(labels.shape[0], dtype=np.int64)
    clipped = np.clip(proba[idx, labels], 1e-12, 1.0)
    pred = np.argmax(proba, axis=1).astype(np.int64)
    return {
        "logloss": float(-np.mean(np.log(clipped))),
        "acc": float(np.mean(pred == labels)),
    }


def evaluate_model_splits(
    *,
    model: Any,
    num_classes: int,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, (states, labels) in datasets.items():
        proba = predict_proba_2d(model, states, num_classes=num_classes)
        out[name] = evaluate_proba(proba, labels)
    return out


def selection_score(metrics: dict[str, dict[str, float]], masked_weight: float, full_weight: float) -> float:
    masked_acc = float(metrics["val_masked"]["acc"])
    full_acc = float(metrics["val_full"]["acc"])
    return float(masked_weight * masked_acc + full_weight * full_acc)


def build_prerequisite_indices(resolved_actions) -> list[list[int]]:
    action_id_to_idx = {str(action.action_id): int(i) for i, action in enumerate(resolved_actions)}
    out: list[list[int]] = []
    for action in resolved_actions:
        prereqs: list[int] = []
        for raw in action.prerequisites:
            key = str(raw)
            if key in action_id_to_idx:
                prereqs.append(int(action_id_to_idx[key]))
        out.append(prereqs)
    return out


def legal_remaining_actions(m_act: np.ndarray, prerequisite_indices: list[list[int]]) -> list[int]:
    selected = {int(i) for i, flag in enumerate(m_act.tolist()) if float(flag) > 0.5}
    legal: list[int] = []
    for action_idx, prereqs in enumerate(prerequisite_indices):
        if float(m_act[action_idx]) > 0.5:
            continue
        if set(int(x) for x in prereqs).issubset(selected):
            legal.append(int(action_idx))
    if legal:
        return legal
    return [int(i) for i in range(m_act.shape[0]) if float(m_act[i]) < 0.5]


def rollout_teacher_on_indices(
    *,
    model: Any,
    num_classes: int,
    x_norm: np.ndarray,
    present: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    action_feature_matrix: np.ndarray,
    action_ids: list[str],
    prerequisite_indices: list[list[int]],
    max_steps: int,
) -> dict[str, Any]:
    per_step_correct = np.zeros(max_steps, dtype=np.int64)
    per_step_total = np.zeros(max_steps, dtype=np.int64)
    action_selection_counts = {str(action_id): 0 for action_id in action_ids}
    first_correct_count = 0
    first_correct_sum = 0
    path_rows: list[dict[str, Any]] = []

    for sample_idx in indices.astype(np.int64).tolist():
        x_norm_row = x_norm[int(sample_idx)].astype(np.float32)
        present_row = present[int(sample_idx)].astype(np.float32)
        label = int(y[int(sample_idx)])
        m_act = np.zeros(len(action_ids), dtype=np.float32)
        chosen_action_indices: list[int] = []
        chosen_action_ids: list[str] = []
        step_predictions: list[int] = []
        step_confidences: list[float] = []
        chosen_deltas: list[float] = []
        first_correct_step: int | None = None

        for step in range(max_steps):
            candidates = legal_remaining_actions(m_act, prerequisite_indices)
            if not candidates:
                break
            state_now = build_state_batch_np(
                x_norm=x_norm_row.reshape(1, -1),
                feature_present_mask=present_row.reshape(1, -1),
                m_act=m_act.reshape(1, -1),
                action_feature_matrix=action_feature_matrix,
            )
            probs_now = predict_proba_2d(model, state_now, num_classes=num_classes)[0]
            base_value = float(np.log(max(float(probs_now[label]), 1.0e-12)))

            plus_masks = np.repeat(m_act.reshape(1, -1), len(candidates), axis=0).astype(np.float32)
            for row_idx, action_idx in enumerate(candidates):
                plus_masks[row_idx, int(action_idx)] = 1.0
            states_plus = build_state_batch_np(
                x_norm=np.repeat(x_norm_row.reshape(1, -1), len(candidates), axis=0),
                feature_present_mask=np.repeat(present_row.reshape(1, -1), len(candidates), axis=0),
                m_act=plus_masks,
                action_feature_matrix=action_feature_matrix,
            )
            probs_plus = predict_proba_2d(model, states_plus, num_classes=num_classes)
            true_probs = np.clip(probs_plus[:, label], 1.0e-12, 1.0)
            values_plus = np.log(true_probs)
            best_row = int(np.argmax(values_plus))
            best_action_idx = int(candidates[best_row])
            best_delta = float(values_plus[best_row] - base_value)
            updated_probs = probs_plus[best_row]
            updated_pred = int(np.argmax(updated_probs))
            updated_conf = float(np.max(updated_probs))

            m_act[best_action_idx] = 1.0
            chosen_action_indices.append(best_action_idx)
            chosen_action_ids.append(str(action_ids[best_action_idx]))
            chosen_deltas.append(best_delta)
            step_predictions.append(updated_pred)
            step_confidences.append(updated_conf)
            action_selection_counts[str(action_ids[best_action_idx])] += 1

            per_step_total[step] += 1
            if updated_pred == label:
                per_step_correct[step] += 1
                if first_correct_step is None:
                    first_correct_step = int(step + 1)

        if first_correct_step is not None:
            first_correct_count += 1
            first_correct_sum += int(first_correct_step)
        path_rows.append(
            {
                "sample_index": int(sample_idx),
                "label": int(label),
                "path_action_indices": chosen_action_indices,
                "path_action_ids": chosen_action_ids,
                "chosen_deltas": chosen_deltas,
                "step_predictions": step_predictions,
                "step_confidences": step_confidences,
                "first_correct_step": int(first_correct_step) if first_correct_step is not None else -1,
            }
        )

    per_step_accuracy = []
    for step_idx in range(max_steps):
        total = int(per_step_total[step_idx])
        acc = None if total <= 0 else float(per_step_correct[step_idx] / total)
        per_step_accuracy.append(
            {
                "step": int(step_idx + 1),
                "num_actions_selected": int(step_idx + 1),
                "accuracy": acc,
                "total": total,
            }
        )
    valid_accs = [float(item["accuracy"]) for item in per_step_accuracy if item["accuracy"] is not None]
    final_acc = valid_accs[-1] if valid_accs else 0.0
    return {
        "num_samples": int(len(indices)),
        "max_steps": int(max_steps),
        "mean_acc@all": float(np.mean(valid_accs)) if valid_accs else 0.0,
        "final_acc": float(final_acc),
        "early_acc@1": valid_accs[0] if valid_accs else None,
        "early_acc@3": float(np.mean(valid_accs[: min(3, len(valid_accs))])) if valid_accs else None,
        "first_correct_rate": float(first_correct_count / max(len(indices), 1)),
        "avg_first_correct_step": (
            float(first_correct_sum / first_correct_count) if first_correct_count > 0 else None
        ),
        "per_step_accuracy": per_step_accuracy,
        "action_selection_counts": action_selection_counts,
        "path_rows_preview": path_rows[:20],
    }


def rollout_selection_score(rollout: dict[str, Any], mean_weight: float, final_weight: float) -> float:
    return float(mean_weight * float(rollout["mean_acc@all"]) + final_weight * float(rollout["final_acc"]))


def fit_candidate(
    *,
    model_type: str,
    params: dict[str, Any],
    seed: int,
    num_classes: int,
    x_train_state: np.ndarray,
    y_train_state: np.ndarray,
    x_val_state: np.ndarray,
    y_val_state: np.ndarray,
    sample_weight: np.ndarray,
) -> Any:
    model = init_model(model_type=model_type, params=params, num_classes=num_classes, seed=seed)
    if model_type == "catboost":
        model.fit(
            x_train_state,
            y_train_state,
            sample_weight=sample_weight,
            eval_set=(x_val_state, y_val_state),
            verbose=False,
        )
    elif model_type == "xgboost":
        try:
            model.fit(
                x_train_state,
                y_train_state,
                sample_weight=sample_weight,
                eval_set=[(x_val_state, y_val_state)],
                verbose=False,
            )
        except TypeError:
            model.fit(
                x_train_state,
                y_train_state,
                sample_weight=sample_weight,
                eval_set=[(x_val_state, y_val_state)],
            )
    else:
        model.fit(x_train_state, y_train_state)
    return model


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Train and select a mask-aware multi-backbone teacher.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--actions_json", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--version", nargs="?", const="", default=None)
    parser.add_argument("--results_root", default=str(project_root / "results"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--force_new_split", action="store_true")
    parser.add_argument("--missing_value", type=float, default=-1.0)
    parser.add_argument("--train_masks_per_sample", type=int, default=96)
    parser.add_argument("--val_masks_per_sample", type=int, default=32)
    parser.add_argument("--random_mask_seed_offset", type=int, default=100000)
    parser.add_argument("--teacher_backbones", default="catboost,xgboost,logistic_regression,mlp")
    parser.add_argument(
        "--teacher_select_strategy",
        choices=["static", "rollout"],
        default="rollout",
        help="Selection strategy.",
    )
    parser.add_argument("--teacher_select_masked_weight", type=float, default=0.5)
    parser.add_argument("--teacher_select_full_weight", type=float, default=0.5)
    parser.add_argument("--teacher_rollout_select_mean_weight", type=float, default=0.8)
    parser.add_argument("--teacher_rollout_select_final_weight", type=float, default=0.2)
    parser.add_argument(
        "--teacher_rollout_select_max_samples",
        type=int,
        default=0,
        help="0 uses all validation samples; positive values limit the sample count.",
    )
    parser.add_argument("--iterations_list", default="auto")
    parser.add_argument("--depth_list", default="auto")
    parser.add_argument("--learning_rate_list", default="auto")
    parser.add_argument("--subsample_list", default="auto")
    parser.add_argument("--rsm_list", default="auto")
    parser.add_argument("--l2_leaf_reg_list", default="auto")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--max_grid_combinations", type=int, default=96)
    parser.add_argument("--allow_uncovered_features", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    args.dataset_name, dataset_csv, actions_json, args.label_col, args.version = resolve_dataset_options(
        project_root=project_root,
        dataset_name=args.dataset_name,
        dataset_csv_arg=args.dataset_csv,
        actions_json_arg=args.actions_json,
        label_col_arg=args.label_col,
        version_arg=args.version,
    )
    backbones = parse_backbones(args.teacher_backbones)
    args.dataset_csv = str(dataset_csv)
    args.actions_json = str(actions_json)
    paths = prepare_paths(args)

    df = pd.read_csv(dataset_csv)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in dataset.")
    feature_columns = [c for c in df.columns if c != args.label_col]
    x_raw = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(args.missing_value)
        .to_numpy(dtype=np.float32)
    )
    y = pd.to_numeric(df[args.label_col], errors="raise").to_numpy(dtype=np.int64)
    present = (np.isfinite(x_raw) & (x_raw != args.missing_value)).astype(np.float32)

    actions = load_actions(actions_json)
    resolved_actions, unresolved = resolve_actions(actions, feature_columns, strict=True)
    action_feature_matrix = build_action_feature_matrix(resolved_actions, num_features=len(feature_columns))
    uncovered = uncovered_features(feature_columns, resolved_actions)
    allow_uncovered = bool(args.allow_uncovered_features)
    if uncovered and not allow_uncovered:
        raise ValueError(f"Some feature columns are not covered by any action: {uncovered}")
    if uncovered:
        print(
            "[teacher][WARN] CSV has feature columns not covered by actions.json; "
            f"they will never be acquired: {uncovered}"
        )

    split_path = Path(paths["split_path"])
    if split_path.exists() and not args.force_new_split:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        split_indices = create_stratified_split_indices(
            labels=y,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_payload = {
            "dataset_csv": str(dataset_csv),
            "label_col": args.label_col,
            "seed": int(args.seed),
            "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
            "indices": split_indices,
            "class_distribution": {
                "train": class_distribution(y, split_indices["train"]),
                "val": class_distribution(y, split_indices["val"]),
                "test": class_distribution(y, split_indices["test"]),
            },
        }
        save_json(split_path, split_payload)

    train_idx = np.asarray(split_payload["indices"]["train"], dtype=np.int64)
    val_idx = np.asarray(split_payload["indices"]["val"], dtype=np.int64)
    test_idx = np.asarray(split_payload["indices"]["test"], dtype=np.int64)
    y_train = y[train_idx]
    num_classes = int(np.max(y) + 1)
    num_actions = int(action_feature_matrix.shape[0])
    num_features = int(action_feature_matrix.shape[1])
    input_dim = num_features * 2 + num_actions

    mean, std = compute_feature_stats(x_raw[train_idx], missing_value=args.missing_value)
    x_norm = standardize_features(x_raw, mean, std)

    print("[teacher] building fixed masked state datasets...")
    x_train_state, y_train_state, _, _ = build_masked_state_dataset(
        x_norm=x_norm,
        present=present,
        y=y,
        indices=train_idx,
        action_feature_matrix=action_feature_matrix,
        masks_per_sample=args.train_masks_per_sample,
        seed=args.seed + args.random_mask_seed_offset + 1,
    )
    x_val_state, y_val_state, _, _ = build_masked_state_dataset(
        x_norm=x_norm,
        present=present,
        y=y,
        indices=val_idx,
        action_feature_matrix=action_feature_matrix,
        masks_per_sample=args.val_masks_per_sample,
        seed=args.seed + args.random_mask_seed_offset + 2,
    )
    eval_state_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train_full": build_fixed_state_dataset(
            x_norm=x_norm,
            present=present,
            y=y,
            indices=train_idx,
            action_feature_matrix=action_feature_matrix,
            value=1.0,
        ),
        "val_masked": (x_val_state, y_val_state),
        "val_empty": build_fixed_state_dataset(
            x_norm=x_norm,
            present=present,
            y=y,
            indices=val_idx,
            action_feature_matrix=action_feature_matrix,
            value=0.0,
        ),
        "val_full": build_fixed_state_dataset(
            x_norm=x_norm,
            present=present,
            y=y,
            indices=val_idx,
            action_feature_matrix=action_feature_matrix,
            value=1.0,
        ),
        "test_full": build_fixed_state_dataset(
            x_norm=x_norm,
            present=present,
            y=y,
            indices=test_idx,
            action_feature_matrix=action_feature_matrix,
            value=1.0,
        ),
    }

    class_weights = build_class_weights(y_train, num_classes=num_classes)
    sample_weight = class_weights[y_train_state]

    candidate_records: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_record: dict[str, Any] | None = None
    best_model_by_backbone: dict[str, Any] = {}
    best_record_by_backbone: dict[str, dict[str, Any]] = {}

    for model_type in backbones:
        baseline_params, baseline_source = load_baseline_best_params(project_root, args.dataset_name, model_type)
        if model_type == "catboost":
            cat_params = merge_params(model_type, baseline_params, args.n_jobs)
            grid_values = resolve_catboost_grid_values(args, cat_params)
            param_grid = build_catboost_grid_combinations(grid_values, int(args.n_jobs), int(args.max_grid_combinations))
            params_source = baseline_source
            print(f"[teacher] backbone=catboost grid_combinations={len(param_grid)}")
        else:
            param_grid = [merge_params(model_type, baseline_params, args.n_jobs)]
            params_source = baseline_source
            print(f"[teacher] backbone={model_type} single candidate params_source={params_source or 'fallback'}")

        for combo_idx, params in enumerate(param_grid, start=1):
            try:
                model = fit_candidate(
                    model_type=model_type,
                    params=params,
                    seed=args.seed,
                    num_classes=num_classes,
                    x_train_state=x_train_state,
                    y_train_state=y_train_state,
                    x_val_state=x_val_state,
                    y_val_state=y_val_state,
                    sample_weight=sample_weight,
                )
                metrics = evaluate_model_splits(model=model, num_classes=num_classes, datasets=eval_state_sets)
                score = selection_score(
                    metrics,
                    masked_weight=float(args.teacher_select_masked_weight),
                    full_weight=float(args.teacher_select_full_weight),
                )
                record = {
                    "model_type": model_type,
                    "checkpoint_model_type": f"{model_type}_mask",
                    "combo_index": int(combo_idx),
                    "params": params,
                    "params_source": str(params_source),
                    "selection_score": float(score),
                    "metrics": metrics,
                    "failed": False,
                }
                is_better = False
                if best_record is None:
                    is_better = True
                else:
                    cur = float(record["selection_score"])
                    prev = float(best_record["selection_score"])
                    if cur > prev + 1e-12:
                        is_better = True
                    elif abs(cur - prev) <= 1e-12:
                        cur_loss = float(metrics["val_masked"]["logloss"])
                        prev_loss = float(best_record["metrics"]["val_masked"]["logloss"])
                        if cur_loss < prev_loss - 1e-12:
                            is_better = True
                if is_better:
                    best_model = model
                    best_record = record
                backbone_best = best_record_by_backbone.get(model_type)
                backbone_is_better = False
                if backbone_best is None:
                    backbone_is_better = True
                else:
                    cur = float(record["selection_score"])
                    prev = float(backbone_best["selection_score"])
                    if cur > prev + 1e-12:
                        backbone_is_better = True
                    elif abs(cur - prev) <= 1e-12:
                        cur_loss = float(metrics["val_masked"]["logloss"])
                        prev_loss = float(backbone_best["metrics"]["val_masked"]["logloss"])
                        if cur_loss < prev_loss - 1e-12:
                            backbone_is_better = True
                if backbone_is_better:
                    best_model_by_backbone[model_type] = model
                    best_record_by_backbone[model_type] = record
                print(
                    f"[teacher {model_type} {combo_idx:03d}/{len(param_grid):03d}] "
                    f"score={score:.6f} val_masked_acc={metrics['val_masked']['acc']:.4f} "
                    f"val_full_acc={metrics['val_full']['acc']:.4f} test_full_acc={metrics['test_full']['acc']:.4f}"
                )
            except Exception as exc:
                record = {
                    "model_type": model_type,
                    "checkpoint_model_type": f"{model_type}_mask",
                    "combo_index": int(combo_idx),
                    "params": params,
                    "params_source": str(params_source),
                    "selection_score": None,
                    "metrics": {},
                    "failed": True,
                    "error": repr(exc),
                }
                print(f"[teacher {model_type} {combo_idx:03d}/{len(param_grid):03d}] FAILED: {exc!r}")
            candidate_records.append(record)

    if best_model is None or best_record is None:
        raise RuntimeError("No teacher backbone candidate finished successfully.")

    static_best_model = best_model
    static_best_record = best_record
    rollout_candidate_records: list[dict[str, Any]] = []
    if args.teacher_select_strategy == "rollout":
        prerequisite_indices = build_prerequisite_indices(resolved_actions)
        rollout_val_idx = val_idx
        if int(args.teacher_rollout_select_max_samples) > 0:
            rollout_val_idx = val_idx[: int(args.teacher_rollout_select_max_samples)]
        max_rollout_steps = int(num_actions)
        print(
            "[teacher] rollout selection: "
            f"backbones={list(best_record_by_backbone.keys())} val_samples={len(rollout_val_idx)}"
        )
        rollout_best_score = -1.0e18
        rollout_best_record: dict[str, Any] | None = None
        rollout_best_model: Any | None = None
        for model_type in backbones:
            record = best_record_by_backbone.get(model_type)
            model = best_model_by_backbone.get(model_type)
            if record is None or model is None:
                continue
            rollout_metrics = rollout_teacher_on_indices(
                model=model,
                num_classes=num_classes,
                x_norm=x_norm,
                present=present,
                y=y,
                indices=rollout_val_idx,
                action_feature_matrix=action_feature_matrix,
                action_ids=[a.action_id for a in resolved_actions],
                prerequisite_indices=prerequisite_indices,
                max_steps=max_rollout_steps,
            )
            roll_score = rollout_selection_score(
                rollout_metrics,
                mean_weight=float(args.teacher_rollout_select_mean_weight),
                final_weight=float(args.teacher_rollout_select_final_weight),
            )
            record["rollout_metrics"] = rollout_metrics
            record["rollout_selection_score"] = float(roll_score)
            rollout_candidate_records.append(record)
            print(
                f"[rollout {model_type}] score={roll_score:.6f} "
                f"mean_acc@all={rollout_metrics['mean_acc@all']:.4f} "
                f"final_acc={rollout_metrics['final_acc']:.4f} "
                f"static_score={record['selection_score']:.6f}"
            )
            is_rollout_better = False
            if roll_score > rollout_best_score + 1e-12:
                is_rollout_better = True
            elif abs(roll_score - rollout_best_score) <= 1e-12 and rollout_best_record is not None:
                cur_mean = float(rollout_metrics["mean_acc@all"])
                prev_mean = float(rollout_best_record["rollout_metrics"]["mean_acc@all"])
                if cur_mean > prev_mean + 1e-12:
                    is_rollout_better = True
                elif abs(cur_mean - prev_mean) <= 1e-12:
                    cur_static = float(record["selection_score"])
                    prev_static = float(rollout_best_record["selection_score"])
                    if cur_static > prev_static + 1e-12:
                        is_rollout_better = True
            if is_rollout_better:
                rollout_best_score = float(roll_score)
                rollout_best_record = record
                rollout_best_model = model
        if rollout_best_model is not None and rollout_best_record is not None:
            best_model = rollout_best_model
            best_record = rollout_best_record
        else:
            print("[teacher][WARN] no rollout candidate succeeded; using static selection.")
            best_model = static_best_model
            best_record = static_best_record

    ckpt_dir = Path(paths["ckpt_dir"])
    selected_model_type = str(best_record["model_type"])
    selected_static_score = float(best_record["selection_score"])
    selected_rollout_score = (
        float(best_record["rollout_selection_score"])
        if best_record.get("rollout_selection_score") is not None
        else None
    )
    selected_score = selected_rollout_score if selected_rollout_score is not None else selected_static_score
    selection_rule_text = (
        "rollout: validation rollout score among per-backbone candidates; "
        "tie-break mean_acc@all then static validation score"
        if args.teacher_select_strategy == "rollout"
        else "static: validation composite score with masked-logloss tie-break"
    )
    model_suffix = ".cbm" if selected_model_type == "catboost" else ".pkl"
    model_path = ckpt_dir / f"teacher_best{model_suffix}"
    ckpt_path = ckpt_dir / "teacher_best.pt"
    if selected_model_type == "catboost":
        best_model.save_model(str(model_path))
    else:
        with model_path.open("wb") as f:
            pickle.dump(best_model, f)

    checkpoint = {
        "teacher_version": "teacher_pipeline",
        "teacher_model_type": selected_model_type,
        "model_type": f"{selected_model_type}_mask",
        "model_path": str(model_path),
        "model_config": {
            "input_dim": int(input_dim),
            "num_classes": int(num_classes),
            "num_actions": int(num_actions),
            "num_features": int(num_features),
            "teacher_params": best_record["params"],
        },
        "feature_columns": feature_columns,
        "uncovered_feature_columns": uncovered,
        "allow_uncovered_features": bool(allow_uncovered),
        "label_col": args.label_col,
        "actions_resolved": [a.to_dict() for a in resolved_actions],
        "action_ids": [a.action_id for a in resolved_actions],
        "action_feature_matrix": action_feature_matrix.tolist(),
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "missing_value": float(args.missing_value),
        "state_definition": {
            "x_obs": "normalized feature value only when action selected and raw value is not missing",
            "m_feat": "1 if feature is both selected by chosen actions and actually present in raw sample",
            "m_act": "1 if action has been selected",
        },
        "teacher_selection": {
            "candidate_backbones": backbones,
            "selection_strategy": str(args.teacher_select_strategy),
            "selection_rule": selection_rule_text,
            "selection_score": float(selected_score),
            "selected_static_score": float(selected_static_score),
            "selected_rollout_score": selected_rollout_score,
            "static_selection_weights": {
                "val_masked_acc": float(args.teacher_select_masked_weight),
                "val_full_acc": float(args.teacher_select_full_weight),
            },
            "rollout_selection_weights": {
                "val_rollout_mean_acc@all": float(args.teacher_rollout_select_mean_weight),
                "val_rollout_final_acc": float(args.teacher_rollout_select_final_weight),
            },
            "static_best_record": static_best_record,
            "rollout_candidate_records": rollout_candidate_records,
            "best_record": best_record,
        },
        "split_path": str(split_path),
        "dataset_csv": str(dataset_csv),
        "dataset_name": args.dataset_name,
        "version": args.version,
        "run_id": str(paths["run_id"]),
        "seed": int(args.seed),
        "train_masks_per_sample": int(args.train_masks_per_sample),
        "val_masks_per_sample": int(args.val_masks_per_sample),
        "teacher_select_strategy": str(args.teacher_select_strategy),
    }
    torch.save(checkpoint, ckpt_path)

    resolved_actions_path = Path(paths["meta_dir"]) / "actions_resolved.json"
    save_json(
        resolved_actions_path,
        {
            "dataset_csv": str(dataset_csv),
            "label_col": args.label_col,
            "num_actions": len(resolved_actions),
            "actions": [a.to_dict() for a in resolved_actions],
        },
    )

    teacher_backbone_selection_path = Path(paths["metrics_dir"]) / "teacher_backbone_selection.json"
    selection_payload = {
        "run_id": str(paths["run_id"]),
        "teacher_version": "teacher_pipeline",
        "candidate_backbones": backbones,
        "selection_strategy": str(args.teacher_select_strategy),
        "selection_rule": selection_rule_text,
        "static_selection_weights": {
            "val_masked_acc": float(args.teacher_select_masked_weight),
            "val_full_acc": float(args.teacher_select_full_weight),
        },
        "rollout_selection_weights": {
            "val_rollout_mean_acc@all": float(args.teacher_rollout_select_mean_weight),
            "val_rollout_final_acc": float(args.teacher_rollout_select_final_weight),
        },
        "selected_model_type": selected_model_type,
        "selected_checkpoint_model_type": str(best_record["checkpoint_model_type"]),
        "selected_score": float(selected_score),
        "selected_static_score": float(selected_static_score),
        "selected_rollout_score": selected_rollout_score,
        "static_best_model_type": str(static_best_record["model_type"]),
        "static_best_score": float(static_best_record["selection_score"]),
        "rollout_candidate_records": rollout_candidate_records,
        "best_record": best_record,
        "candidate_records": candidate_records,
    }
    save_json(teacher_backbone_selection_path, selection_payload)

    metrics_path = Path(paths["metrics_dir"]) / "teacher_train_metrics.json"
    save_json(
        metrics_path,
        {
            "run_id": str(paths["run_id"]),
            "teacher_version": "teacher_pipeline",
            "model_type": f"{selected_model_type}_mask",
            "teacher_model_type": selected_model_type,
            "selection_strategy": str(args.teacher_select_strategy),
            "selection_rule": selection_payload["selection_rule"],
            "teacher_backbone_selection_path": str(teacher_backbone_selection_path),
            "selected_static_score": float(selected_static_score),
            "selected_rollout_score": selected_rollout_score,
            "static_best_record": static_best_record,
            "rollout_candidate_records": rollout_candidate_records,
            "best_record": best_record,
            "candidate_records": candidate_records,
            "class_weights": class_weights.tolist(),
            "split_path": str(split_path),
            "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "state_train_rows": int(len(y_train_state)),
            "state_val_rows": int(len(y_val_state)),
            "train_masks_per_sample": int(args.train_masks_per_sample),
            "val_masks_per_sample": int(args.val_masks_per_sample),
            "unresolved_features": unresolved,
        },
    )

    outputs_path = Path(paths["metrics_dir"]) / "teacher_training_outputs.json"
    save_json(
        outputs_path,
        {
            "run_id": str(paths["run_id"]),
            "teacher_ckpt": str(ckpt_path),
            "teacher_model": str(model_path),
            "teacher_model_type": selected_model_type,
            "teacher_select_strategy": str(args.teacher_select_strategy),
            "selected_score": float(selected_score),
            "selected_static_score": float(selected_static_score),
            "selected_rollout_score": selected_rollout_score,
            "split_path": str(split_path),
            "metrics_path": str(metrics_path),
            "teacher_backbone_selection_path": str(teacher_backbone_selection_path),
            "actions_resolved_path": str(resolved_actions_path),
            "dataset_csv": str(dataset_csv),
            "model_type": f"{selected_model_type}_mask",
        },
    )

    print("\nTeacher training finished.")
    print(f"Run ID: {paths['run_id']}")
    print(f"Selected teacher backbone: {selected_model_type}")
    print(f"Teacher checkpoint: {ckpt_path}")
    print(f"Teacher model: {model_path}")
    print(f"Selection metrics: {teacher_backbone_selection_path}")


if __name__ == "__main__":
    main()
