from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def build_state_batch_np(
    *,
    x_norm: np.ndarray,
    feature_present_mask: np.ndarray,
    m_act: np.ndarray,
    action_feature_matrix: np.ndarray,
) -> np.ndarray:
    m_feat_selected = (m_act @ action_feature_matrix > 0.0).astype(np.float32)
    m_feat_observed = (m_feat_selected * feature_present_mask).astype(np.float32)
    x_obs = (x_norm * m_feat_observed).astype(np.float32)
    return np.concatenate([x_obs, m_feat_observed, m_act.astype(np.float32)], axis=1).astype(np.float32)


def build_state_single_np(
    *,
    x_norm_row: np.ndarray,
    feature_present_row: np.ndarray,
    m_act: np.ndarray,
    action_feature_matrix: np.ndarray,
) -> np.ndarray:
    state = build_state_batch_np(
        x_norm=np.asarray(x_norm_row, dtype=np.float32).reshape(1, -1),
        feature_present_mask=np.asarray(feature_present_row, dtype=np.float32).reshape(1, -1),
        m_act=np.asarray(m_act, dtype=np.float32).reshape(1, -1),
        action_feature_matrix=np.asarray(action_feature_matrix, dtype=np.float32),
    )
    return state[0]


def predict_proba_2d(model: Any, state_2d: np.ndarray, num_classes: int) -> np.ndarray:
    proba = np.asarray(model.predict_proba(np.asarray(state_2d, dtype=np.float32)), dtype=np.float64)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    aligned = np.zeros((proba.shape[0], int(num_classes)), dtype=np.float64)
    classes = getattr(model, "classes_", None)
    if classes is not None:
        for src_idx, cls in enumerate(list(classes)):
            cls_idx = int(cls)
            if 0 <= cls_idx < int(num_classes) and src_idx < proba.shape[1]:
                aligned[:, cls_idx] = proba[:, src_idx]
    else:
        width = min(int(num_classes), proba.shape[1])
        aligned[:, :width] = proba[:, :width]
    empty = aligned.sum(axis=1) <= 0.0
    if np.any(empty):
        aligned[empty] = 1.0 / float(num_classes)
    proba = np.clip(aligned, 1e-12, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def state_value_confidence_catboost(
    *,
    model: Any,
    state: np.ndarray,
    label: int,
    num_classes: int,
) -> tuple[float, float, int]:
    probs = predict_proba_2d(model, np.asarray(state, dtype=np.float32).reshape(1, -1), num_classes=num_classes)[0]
    pred = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label_idx = int(label)
    if label_idx < 0 or label_idx >= num_classes:
        value = math.log(1e-12)
    else:
        value = float(math.log(max(float(probs[label_idx]), 1e-12)))
    return value, confidence, pred


def state_prediction_stats_catboost(
    *,
    model: Any,
    state: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, int, float, float]:
    probs = predict_proba_2d(model, np.asarray(state, dtype=np.float32).reshape(1, -1), num_classes=num_classes)[0]
    pred = int(np.argmax(probs))
    conf = float(np.max(probs))
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    if probs.shape[0] > 1:
        entropy = entropy / float(np.log(probs.shape[0]))
    return probs.astype(np.float64), pred, conf, entropy


@dataclass
class CatBoostTeacherBundle:
    checkpoint_path: Path
    model_path: Path
    checkpoint: dict[str, Any]
    model: Any

    @property
    def num_classes(self) -> int:
        return int(self.checkpoint["model_config"]["num_classes"])


def load_catboost_teacher_bundle(checkpoint_path: Path) -> CatBoostTeacherBundle:
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_torch_checkpoint(checkpoint_path)
    model_type = str(checkpoint.get("model_type", "")).strip().lower()
    supported = {"catboost_mask", "xgboost_mask", "logistic_regression_mask", "mlp_mask"}
    if model_type not in supported:
        raise ValueError(
            f"Expected model_type in {sorted(supported)} in teacher checkpoint, got '{model_type or '<missing>'}': "
            f"{checkpoint_path}"
        )

    model_path_raw = str(checkpoint.get("model_path", "")).strip()
    if not model_path_raw:
        raise ValueError(f"CatBoost teacher checkpoint missing model_path: {checkpoint_path}")
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = checkpoint_path.parent / model_path
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost model file not found: {model_path}")

    if model_type == "catboost_mask":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(model_path))
    else:
        with model_path.open("rb") as f:
            model = pickle.load(f)
    return CatBoostTeacherBundle(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        checkpoint=checkpoint,
        model=model,
    )
