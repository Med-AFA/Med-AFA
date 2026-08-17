from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def resolve_feature_name(feature: str, columns: list[str], cutoff: float = 0.88) -> tuple[str | None, str]:
    if feature in columns:
        return feature, "exact"

    lower_map: dict[str, str] = {}
    for col in columns:
        lower_map.setdefault(col.lower(), col)
    if feature.lower() in lower_map:
        return lower_map[feature.lower()], "case_insensitive"

    norm_map: dict[str, list[str]] = {}
    for col in columns:
        norm_map.setdefault(normalize_name(col), []).append(col)

    feature_norm = normalize_name(feature)
    if feature_norm in norm_map and len(norm_map[feature_norm]) == 1:
        return norm_map[feature_norm][0], "normalized"

    candidates = list(norm_map.keys())
    close = get_close_matches(feature_norm, candidates, n=1, cutoff=cutoff)
    if close:
        matched_cols = norm_map[close[0]]
        if len(matched_cols) == 1:
            return matched_cols[0], "fuzzy"

    return None, "unresolved"


@dataclass
class ResolvedAction:
    action_id: str
    name: str
    requested_features: list[str]
    resolved_features: list[str]
    feature_indices: list[int]
    match_methods: list[str]
    prerequisites: list[Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "name": self.name,
            "requested_features": self.requested_features,
            "resolved_features": self.resolved_features,
            "feature_indices": self.feature_indices,
            "match_methods": self.match_methods,
            "prerequisites": self.prerequisites,
        }


def load_actions(actions_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(actions_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError("actions.json must be a list or a dict with key 'actions'.")
    return payload


def resolve_actions(
    actions: list[dict[str, Any]],
    feature_columns: list[str],
    *,
    strict: bool = True,
    cutoff: float = 0.88,
) -> tuple[list[ResolvedAction], list[dict[str, str]]]:
    resolved_actions: list[ResolvedAction] = []
    unresolved_entries: list[dict[str, str]] = []

    for i, action in enumerate(actions):
        action_id = str(action.get("action_id", f"action_{i+1:03d}"))
        name = str(action.get("name", action_id))
        raw_features = action.get("feature", action.get("features", []))
        prerequisites = action.get("prerequisites", [])
        if not isinstance(raw_features, list):
            raise ValueError(f"{action_id}: field 'feature' must be a list.")

        resolved_features: list[str] = []
        feature_indices: list[int] = []
        match_methods: list[str] = []
        seen: set[str] = set()

        for raw_f in raw_features:
            feature = str(raw_f)
            resolved, method = resolve_feature_name(feature, feature_columns, cutoff=cutoff)
            if resolved is None:
                unresolved_entries.append({"action_id": action_id, "feature": feature})
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            resolved_features.append(resolved)
            feature_indices.append(feature_columns.index(resolved))
            match_methods.append(method)

        resolved_action = ResolvedAction(
            action_id=action_id,
            name=name,
            requested_features=[str(x) for x in raw_features],
            resolved_features=resolved_features,
            feature_indices=feature_indices,
            match_methods=match_methods,
            prerequisites=prerequisites if isinstance(prerequisites, list) else [prerequisites],
        )
        resolved_actions.append(resolved_action)

    if strict and unresolved_entries:
        lines = [f"{x['action_id']} -> {x['feature']}" for x in unresolved_entries]
        raise ValueError("Unresolved action features:\n" + "\n".join(lines))

    empty_actions = [a.action_id for a in resolved_actions if len(a.feature_indices) == 0]
    if empty_actions:
        raise ValueError(f"Actions with zero resolved features: {empty_actions}")

    return resolved_actions, unresolved_entries


def build_action_feature_matrix(resolved_actions: list[ResolvedAction], num_features: int) -> np.ndarray:
    matrix = np.zeros((len(resolved_actions), num_features), dtype=np.float32)
    for action_idx, action in enumerate(resolved_actions):
        matrix[action_idx, action.feature_indices] = 1.0
    return matrix


def uncovered_features(feature_columns: list[str], resolved_actions: list[ResolvedAction]) -> list[str]:
    covered: set[int] = set()
    for action in resolved_actions:
        covered.update(action.feature_indices)
    return [feature_columns[i] for i in range(len(feature_columns)) if i not in covered]


def compute_feature_stats(x_train: np.ndarray, missing_value: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    d = x_train.shape[1]
    means = np.zeros(d, dtype=np.float32)
    stds = np.ones(d, dtype=np.float32)

    for j in range(d):
        col = x_train[:, j]
        observed_mask = np.isfinite(col) & (col != missing_value)
        if np.any(observed_mask):
            obs = col[observed_mask].astype(np.float32)
            means[j] = float(obs.mean())
            std = float(obs.std())
            stds[j] = std if std > 1e-6 else 1.0
        else:
            means[j] = 0.0
            stds[j] = 1.0

    return means, stds


def standardize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def create_stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-8):
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    labels = labels.astype(int)
    unique_classes = np.unique(labels)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        if n >= 3:
            if n_train == 0:
                n_train = 1
                n_test -= 1
            if n_val == 0:
                n_val = 1
                n_test -= 1
            if n_test == 0:
                n_test = 1
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1

        n_train = max(0, n_train)
        n_val = max(0, n_val)
        n_test = max(0, n_test)

        used = n_train + n_val + n_test
        if used < n:
            n_test += (n - used)
        elif used > n:
            overflow = used - n
            reduce_test = min(overflow, n_test)
            n_test -= reduce_test
            overflow -= reduce_test
            reduce_val = min(overflow, n_val)
            n_val -= reduce_val
            overflow -= reduce_val
            n_train -= overflow

        cls_train = cls_idx[:n_train].tolist()
        cls_val = cls_idx[n_train:n_train + n_val].tolist()
        cls_test = cls_idx[n_train + n_val:n_train + n_val + n_test].tolist()

        train_idx.extend(cls_train)
        val_idx.extend(cls_val)
        test_idx.extend(cls_test)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return {
        "train": sorted(train_idx),
        "val": sorted(val_idx),
        "test": sorted(test_idx),
    }


def class_distribution(labels: np.ndarray, indices: list[int]) -> dict[str, int]:
    if len(indices) == 0:
        return {}
    selected = labels[np.asarray(indices, dtype=np.int64)]
    unique, counts = np.unique(selected, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(unique, counts)}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
