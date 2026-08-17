from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from action_constraints import legal_action_indices_from_selected
from teacher_state_utils import build_state_vector, predict_catboost_state


@dataclass(frozen=True)
class FullPathPlannerConfig:
    top_k_paths: int = 3
    beam_width: int = 3
    max_depth: int = 6
    score_mode: str = "mean_true_prob"
    temperature: float = 0.2
    mixed_hard_acc_alpha: float = 0.2


def selected_indices_from_mask(mask: np.ndarray | torch.Tensor) -> list[int]:
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    return [int(i) for i, flag in enumerate(arr.tolist()) if float(flag) > 0.5]


def action_mask_from_selected(selected: list[int], num_actions: int) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.float32)
    for idx in selected:
        if 0 <= int(idx) < num_actions:
            mask[int(idx)] = 1.0
    return mask


def softmax_from_path_scores(scores: np.ndarray, *, temperature: float) -> np.ndarray:
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


def _path_score(
    *,
    true_probs: list[float],
    log_probs: list[float],
    predictions: list[int],
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
    raise ValueError(
        f"Unsupported full-path score mode: {mode}. "
        "Expected mean_true_prob, negative_ce, ce_reduction, mean_hard_acc, or mixed_score."
    )


def _state_prediction_for_selected(
    *,
    teacher_model: Any,
    x_norm_row: torch.Tensor,
    present_row: torch.Tensor,
    selected: list[int],
    label: int,
    action_feature_matrix: torch.Tensor,
    num_classes: int,
) -> dict[str, float | int]:
    m_act = action_mask_from_selected(selected, int(action_feature_matrix.shape[0]))
    state = build_state_vector(
        x_norm_row=x_norm_row,
        present_row=present_row,
        m_act=m_act,
        action_feature_matrix=action_feature_matrix,
    )
    pred = predict_catboost_state(teacher_model, state, num_classes=int(num_classes))
    probs = pred["proba"][0]
    label_idx = int(label)
    true_prob = float(probs[label_idx])
    return {
        "true_prob": true_prob,
        "log_prob": float(np.log(max(true_prob, 1.0e-12))),
        "prediction": int(np.argmax(probs)),
        "confidence": float(np.max(probs)),
    }


def plan_full_path_topk(
    *,
    teacher_model: Any,
    x_norm_row: torch.Tensor,
    present_row: torch.Tensor,
    selected_indices: list[int],
    label: int,
    action_feature_matrix: torch.Tensor,
    action_ids: list[str],
    action_groups: list[dict],
    num_classes: int,
    config: FullPathPlannerConfig,
) -> dict[str, Any]:

    num_actions = int(len(action_ids))
    selected_start = [int(x) for x in selected_indices]
    selected_set = set(selected_start)
    if len(selected_set) != len(selected_start):
        selected_start = sorted(selected_set)

    first_candidates = legal_action_indices_from_selected(selected_start, action_groups)
    candidate_mask = np.zeros(num_actions, dtype=bool)
    for idx in first_candidates:
        candidate_mask[int(idx)] = True
    if not first_candidates:
        raise RuntimeError(f"No legal full-path candidate from selected={selected_start}.")

    current = _state_prediction_for_selected(
        teacher_model=teacher_model,
        x_norm_row=x_norm_row,
        present_row=present_row,
        selected=selected_start,
        label=int(label),
        action_feature_matrix=action_feature_matrix,
        num_classes=int(num_classes),
    )
    current_log_prob = float(current["log_prob"])

    remaining = max(0, num_actions - len(set(selected_start)))
    depth = max(1, min(int(config.max_depth), remaining))
    beam_width = max(1, int(config.beam_width))
    top_k = max(1, int(config.top_k_paths))

    beam: list[dict[str, Any]] = [
        {
            "path": [],
            "selected": list(selected_start),
            "true_probs": [],
            "log_probs": [],
            "predictions": [],
            "score": float(current_log_prob),
        }
    ]
    completed: list[dict[str, Any]] = []

    for _step in range(depth):
        expanded: list[dict[str, Any]] = []
        for item in beam:
            legal = legal_action_indices_from_selected(item["selected"], action_groups)
            if not legal:
                completed.append(item)
                continue
            for action_idx in legal:
                new_selected = list(item["selected"]) + [int(action_idx)]
                pred = _state_prediction_for_selected(
                    teacher_model=teacher_model,
                    x_norm_row=x_norm_row,
                    present_row=present_row,
                    selected=new_selected,
                    label=int(label),
                    action_feature_matrix=action_feature_matrix,
                    num_classes=int(num_classes),
                )
                true_probs = list(item["true_probs"]) + [float(pred["true_prob"])]
                log_probs = list(item["log_probs"]) + [float(pred["log_prob"])]
                predictions = list(item["predictions"]) + [int(pred["prediction"])]
                score = _path_score(
                    true_probs=true_probs,
                    log_probs=log_probs,
                    predictions=predictions,
                    label=int(label),
                    current_log_prob=current_log_prob,
                    mode=config.score_mode,
                    mixed_hard_acc_alpha=float(config.mixed_hard_acc_alpha),
                )
                expanded.append(
                    {
                        "path": list(item["path"]) + [int(action_idx)],
                        "selected": new_selected,
                        "true_probs": true_probs,
                        "log_probs": log_probs,
                        "predictions": predictions,
                        "score": float(score),
                    }
                )
        if not expanded:
            break
        expanded.sort(key=lambda row: (-float(row["score"]), tuple(int(x) for x in row["path"])))
        beam = expanded[:beam_width]

    final_paths = completed + beam
    if not final_paths:
        raise RuntimeError(f"Full-path beam search produced no paths from selected={selected_start}.")
    final_paths.sort(key=lambda row: (-float(row["score"]), tuple(int(x) for x in row["path"])))
    top_paths = final_paths[:top_k]
    top_scores = np.asarray([float(row["score"]) for row in top_paths], dtype=np.float64)
    weights = softmax_from_path_scores(top_scores, temperature=float(config.temperature))

    soft_distribution = np.zeros(num_actions, dtype=np.float32)
    best_score_by_first = np.full(num_actions, -np.inf, dtype=np.float64)
    for row in final_paths:
        if not row["path"]:
            continue
        first = int(row["path"][0])
        best_score_by_first[first] = max(float(best_score_by_first[first]), float(row["score"]))
    for row, weight in zip(top_paths, weights):
        if not row["path"]:
            continue
        soft_distribution[int(row["path"][0])] += float(weight)
    soft_distribution = soft_distribution * candidate_mask.astype(np.float32)
    soft_sum = float(soft_distribution.sum())
    if soft_sum <= 0.0:

        best_idx = int(top_paths[0]["path"][0])
        soft_distribution[best_idx] = 1.0
        soft_sum = 1.0
    soft_distribution = soft_distribution / max(soft_sum, 1.0e-12)

    finite_scores = best_score_by_first.copy()
    finite_seen = np.isfinite(finite_scores)
    if np.any(finite_seen):
        floor = float(np.min(finite_scores[finite_seen]) - 1.0)
    else:
        floor = float(current_log_prob - 1.0)
    utility = np.full(num_actions, floor, dtype=np.float64)
    utility[candidate_mask] = floor
    utility[finite_seen] = finite_scores[finite_seen]
    utility[~candidate_mask] = 0.0

    best_action = int(np.argmax(np.where(candidate_mask, utility, -np.inf)))
    return {
        "action_idx": int(best_action),
        "action_id": str(action_ids[best_action]),
        "candidate_mask": candidate_mask.astype(np.float32),
        "soft_distribution": soft_distribution.astype(np.float32),
        "utility": np.where(candidate_mask, utility, 0.0).astype(np.float32),
        "value": float(utility[best_action]),
        "current_value": float(current_log_prob),
        "delta": float(utility[best_action] - current_log_prob),
        "current_true_prob": float(current["true_prob"]),
        "top_paths": [
            {
                "path": [int(x) for x in row["path"]],
                "action_ids": [str(action_ids[int(x)]) for x in row["path"]],
                "score": float(row["score"]),
                "weight": float(weights[idx]),
                "true_probs": [float(x) for x in row["true_probs"]],
                "predictions": [int(x) for x in row["predictions"]],
            }
            for idx, row in enumerate(top_paths)
        ],
        "num_final_paths": int(len(final_paths)),
        "depth": int(depth),
        "score_mode": str(config.score_mode),
        "temperature": float(config.temperature),
    }
