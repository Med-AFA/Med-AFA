from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def load_action_feature_matrix(
    actions_path: Optional[str],
    dataset_name: str,
    feature_names,
    include_uncovered_features_as_actions: bool = False,
) -> Tuple[torch.Tensor, List[Dict]]:
    if not actions_path:
        raise ValueError("--actions_path is required.")
    path = Path(actions_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]
    if not isinstance(payload, list):
        raise ValueError("actions.json must be a list or a dict with key 'actions'.")

    feature_names = [str(x) for x in list(feature_names)]
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    used = set()
    groups = []
    meta = []
    for i, action in enumerate(payload):
        action_id = str(action.get("action_id", f"action_{i + 1}"))
        raw_features = action.get("feature", action.get("features", []))
        if not isinstance(raw_features, list):
            raise ValueError(f"{action_id}: field 'feature' must be a list.")
        indices = []
        for raw in raw_features:
            feature = str(raw)
            if feature not in feature_to_idx:
                raise ValueError(f"{action_id}: feature not found in dataset columns: {feature}")
            idx = int(feature_to_idx[feature])
            if idx in used:
                raise ValueError(f"Feature '{feature}' appears in multiple actions.")
            used.add(idx)
            indices.append(idx)
        if not indices:
            raise ValueError(f"{action_id}: zero resolved features.")
        groups.append(indices)
        meta.append(
            {
                "action_id": action_id,
                "name": str(action.get("name", action_id)),
                "feature_indices": indices,
                "feature_names": [feature_names[j] for j in indices],
                "_raw_prerequisites": [str(x) for x in action.get("prerequisites", [])],
            }
        )

    if include_uncovered_features_as_actions:
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_idx in used:
                continue
            groups.append([feature_idx])
            meta.append(
                {
                    "action_id": f"__ungrouped_feature_{feature_idx}",
                    "name": f"Ungrouped feature: {feature_name}",
                    "feature_indices": [feature_idx],
                    "feature_names": [feature_name],
                    "synthetic": True,
                    "_raw_prerequisites": [],
                }
            )

    action_id_to_idx = {item["action_id"]: i for i, item in enumerate(meta)}
    for item in meta:
        raw_prerequisites = item.pop("_raw_prerequisites", [])
        prerequisite_indices = []
        for prereq in raw_prerequisites:
            prereq_id = str(prereq)
            if prereq_id not in action_id_to_idx:
                raise ValueError(
                    f"{item['action_id']}: prerequisite action not found in actions.json: {prereq_id}"
                )
            prerequisite_indices.append(int(action_id_to_idx[prereq_id]))
        item["prerequisites"] = [str(x) for x in raw_prerequisites]
        item["prerequisite_indices"] = prerequisite_indices

    matrix = torch.zeros((len(groups), len(feature_names)), dtype=torch.float32)
    for action_idx, indices in enumerate(groups):
        matrix[action_idx, indices] = 1.0
    return matrix, meta


def action_mask_to_feature_mask(action_mask: torch.Tensor, action_feature_matrix: torch.Tensor) -> torch.Tensor:
    matrix = action_feature_matrix.to(device=action_mask.device, dtype=action_mask.dtype)
    return torch.clamp(action_mask @ matrix, min=0.0, max=1.0)


def feature_values_to_action_values(feature_values: torch.Tensor, action_feature_matrix: torch.Tensor) -> torch.Tensor:
    matrix = action_feature_matrix.to(device=feature_values.device, dtype=feature_values.dtype)
    denom = torch.clamp(matrix.sum(dim=1), min=1.0)
    return (torch.abs(feature_values) @ matrix.t()) / denom


def build_prerequisite_matrix(action_groups: List[Dict]) -> torch.Tensor:

    n_actions = len(action_groups)
    matrix = torch.zeros((n_actions, n_actions), dtype=torch.float32)
    for action_idx, item in enumerate(action_groups):
        for prereq_idx in item.get("prerequisite_indices", []):
            prereq_idx = int(prereq_idx)
            if prereq_idx < 0 or prereq_idx >= n_actions:
                raise ValueError(
                    f"{item.get('action_id', action_idx)}: invalid prerequisite index {prereq_idx}"
                )
            if prereq_idx == action_idx:
                raise ValueError(f"{item.get('action_id', action_idx)} cannot depend on itself.")
            matrix[action_idx, prereq_idx] = 1.0
    return matrix


def legal_action_mask(selected_action_mask: torch.Tensor, prerequisite_matrix: torch.Tensor) -> torch.Tensor:

    selected = selected_action_mask > 0.5
    prereq = prerequisite_matrix.to(device=selected_action_mask.device, dtype=selected_action_mask.dtype)
    squeeze = False
    if selected_action_mask.dim() == 1:
        selected = selected.unsqueeze(0)
        selected_float = selected_action_mask.unsqueeze(0).to(dtype=prereq.dtype)
        squeeze = True
    else:
        selected_float = selected_action_mask.to(dtype=prereq.dtype)

    required_counts = prereq.sum(dim=1).unsqueeze(0)
    satisfied_counts = selected_float @ prereq.t()
    prereq_satisfied = satisfied_counts >= (required_counts - 1.0e-6)
    legal = (~selected) & prereq_satisfied
    return legal.squeeze(0) if squeeze else legal


def mask_illegal_action_logits(
    logits: torch.Tensor,
    selected_action_mask: torch.Tensor,
    prerequisite_matrix: torch.Tensor,
    fill_value: float = -1.0e6,
) -> torch.Tensor:
    legal = legal_action_mask(selected_action_mask, prerequisite_matrix).to(device=logits.device)
    if legal.dim() == 1 and logits.dim() == 2:
        legal = legal.unsqueeze(0).expand_as(logits)
    return logits.masked_fill(~legal.bool(), fill_value)


def legal_action_indices_from_selected(selected_indices, action_groups: List[Dict]) -> List[int]:
    selected = {int(x) for x in selected_indices}
    out = []
    for action_idx, item in enumerate(action_groups):
        if action_idx in selected:
            continue
        prereqs = {int(x) for x in item.get("prerequisite_indices", [])}
        if prereqs.issubset(selected):
            out.append(int(action_idx))
    return out


def validate_action_sequence(action_indices, action_groups: List[Dict]) -> Tuple[bool, Optional[int], Optional[str]]:
    selected = set()
    for step_idx, raw_idx in enumerate(action_indices):
        action_idx = int(raw_idx)
        if action_idx in selected:
            return False, step_idx, "repeated_action"
        if action_idx < 0 or action_idx >= len(action_groups):
            return False, step_idx, "unknown_action"
        prereqs = {int(x) for x in action_groups[action_idx].get("prerequisite_indices", [])}
        if not prereqs.issubset(selected):
            missing = sorted(prereqs - selected)
            missing_ids = [
                str(action_groups[i].get("action_id", f"action_{i}"))
                for i in missing
            ]
            return False, step_idx, "missing_prerequisites:" + ",".join(missing_ids)
        selected.add(action_idx)
    return True, None, None


def constrained_greedy_order_from_scores(scores, action_groups: List[Dict]) -> List[int]:
    values = [float(x) for x in list(scores)]
    selected: List[int] = []
    while len(selected) < len(values):
        candidates = legal_action_indices_from_selected(selected, action_groups)
        if not candidates:
            break
        best = max(candidates, key=lambda idx: (values[int(idx)], -int(idx)))
        selected.append(int(best))
    return selected
