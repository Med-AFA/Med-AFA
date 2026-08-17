

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RUN_SCRIPT_DIR = Path(__file__).resolve().parent
RUN_ROOT_DIR = RUN_SCRIPT_DIR.parent
if str(RUN_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_ROOT_DIR))

from action_dataset_utils import save_json, write_jsonl
from masked_teacher_utils import (
    build_state_single_np,
    load_catboost_teacher_bundle,
    state_value_confidence_catboost,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_prerequisite_indices(checkpoint: dict[str, Any], action_ids: list[str]) -> list[list[int]]:
    action_id_to_idx = {str(action_id): int(i) for i, action_id in enumerate(action_ids)}
    by_id = {
        str(item.get("action_id", "")): item
        for item in checkpoint.get("actions_resolved", [])
        if isinstance(item, dict) and str(item.get("action_id", ""))
    }
    out: list[list[int]] = []
    for action_id in action_ids:
        raw = by_id.get(str(action_id), {})
        prereqs: list[int] = []
        for item in raw.get("prerequisites", []):
            key = str(item)
            if key in action_id_to_idx:
                prereqs.append(int(action_id_to_idx[key]))
        out.append(prereqs)
    return out


def remaining_actions(m_act: np.ndarray, prerequisite_indices: list[list[int]]) -> list[int]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate rollout paths from a selected mask-aware teacher.")
    parser.add_argument("--dataset_csv", default="")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--split_path", default="")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--missing_value", type=float, default=-1.0)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means number_of_actions")
    parser.add_argument("--confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--save_delta_history", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="")
    return parser


def rollout_split(
    *,
    split_name: str,
    indices: list[int],
    x_norm: np.ndarray,
    x_present: np.ndarray,
    y: np.ndarray,
    model: Any,
    action_feature_matrix: np.ndarray,
    action_ids: list[str],
    prerequisite_indices: list[list[int]],
    num_classes: int,
    max_steps: int,
    save_delta_history: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    selection_counts = {action_id: 0 for action_id in action_ids}
    stop_reason_counts: dict[str, int] = {}
    total_steps = 0
    total_conf = 0.0
    per_step_correct_counts = np.zeros(max_steps, dtype=np.int64)
    per_step_total_counts = np.zeros(max_steps, dtype=np.int64)
    first_correct_hist: dict[str, int] = {}
    first_correct_sum = 0
    first_correct_count = 0

    for sample_idx in indices:
        x_norm_row = x_norm[sample_idx].astype(np.float32)
        feature_present_row = x_present[sample_idx].astype(np.float32)
        label = int(y[sample_idx])
        m_act = np.zeros(len(action_ids), dtype=np.float32)

        chosen_action_indices: list[int] = []
        chosen_action_ids: list[str] = []
        chosen_deltas: list[float] = []
        step_logs: list[dict[str, Any]] = []
        step_predictions: list[int] = []
        step_confidences: list[float] = []
        first_correct_step: int | None = None

        for step in range(max_steps):
            candidates = remaining_actions(m_act, prerequisite_indices)
            if not candidates:
                break
            state_now = build_state_single_np(
                x_norm_row=x_norm_row,
                feature_present_row=feature_present_row,
                m_act=m_act,
                action_feature_matrix=action_feature_matrix,
            )
            base_value, _, _ = state_value_confidence_catboost(
                model=model,
                state=state_now,
                label=label,
                num_classes=num_classes,
            )
            best_action_idx = -1
            best_delta = -1e18
            candidate_delta_rows: list[dict[str, Any]] = []

            for action_idx in candidates:
                m_act_plus = m_act.copy()
                m_act_plus[action_idx] = 1.0
                state_plus = build_state_single_np(
                    x_norm_row=x_norm_row,
                    feature_present_row=feature_present_row,
                    m_act=m_act_plus,
                    action_feature_matrix=action_feature_matrix,
                )
                value_plus, conf_plus, pred_plus = state_value_confidence_catboost(
                    model=model,
                    state=state_plus,
                    label=label,
                    num_classes=num_classes,
                )
                delta = float(value_plus - base_value)
                candidate_delta_rows.append(
                    {
                        "action_index": int(action_idx),
                        "action_id": action_ids[action_idx],
                        "delta": delta,
                        "value_plus": float(value_plus),
                        "confidence_plus": float(conf_plus),
                        "pred_plus": int(pred_plus),
                    }
                )
                if delta > best_delta:
                    best_delta = delta
                    best_action_idx = int(action_idx)

            if best_action_idx < 0:
                best_action_idx = int(candidates[0])
                best_delta = float("nan")

            m_act[best_action_idx] = 1.0
            chosen_action_indices.append(best_action_idx)
            chosen_action_ids.append(action_ids[best_action_idx])
            chosen_deltas.append(float(best_delta))
            selection_counts[action_ids[best_action_idx]] += 1

            updated_state = build_state_single_np(
                x_norm_row=x_norm_row,
                feature_present_row=feature_present_row,
                m_act=m_act,
                action_feature_matrix=action_feature_matrix,
            )
            _, updated_confidence, updated_pred = state_value_confidence_catboost(
                model=model,
                state=updated_state,
                label=label,
                num_classes=num_classes,
            )
            step_predictions.append(int(updated_pred))
            step_confidences.append(float(updated_confidence))
            if first_correct_step is None and updated_pred == label:
                first_correct_step = step + 1

            if save_delta_history:
                step_logs.append(
                    {
                        "step": int(step),
                        "base_value": float(base_value),
                        "chosen_action_index": int(best_action_idx),
                        "chosen_action_id": action_ids[best_action_idx],
                        "chosen_delta": float(best_delta),
                        "candidate_deltas": sorted(candidate_delta_rows, key=lambda x: x["delta"], reverse=True),
                    }
                )

        stop_reason = "all_actions_ranked" if len(chosen_action_indices) == len(action_ids) else "max_steps_reached"
        final_confidence = float(step_confidences[-1]) if step_confidences else 0.0
        final_pred = int(step_predictions[-1]) if step_predictions else -1
        stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
        total_steps += len(chosen_action_indices)
        total_conf += final_confidence
        for step_idx, pred in enumerate(step_predictions):
            per_step_total_counts[step_idx] += 1
            if pred == label:
                per_step_correct_counts[step_idx] += 1

        first_key = str(first_correct_step if first_correct_step is not None else -1)
        first_correct_hist[first_key] = first_correct_hist.get(first_key, 0) + 1
        if first_correct_step is not None:
            first_correct_sum += int(first_correct_step)
            first_correct_count += 1

        path_rows.append(
            {
                "split": split_name,
                "sample_index": int(sample_idx),
                "label": int(label),
                "path_action_indices": chosen_action_indices,
                "path_action_ids": chosen_action_ids,
                "chosen_deltas": chosen_deltas,
                "steps": int(len(chosen_action_indices)),
                "stop_reason": stop_reason,
                "final_confidence": float(final_confidence),
                "final_prediction": int(final_pred),
                "step_predictions": step_predictions,
                "step_confidences": step_confidences,
                "first_correct_step": int(first_correct_step) if first_correct_step is not None else -1,
            }
        )
        if save_delta_history:
            delta_rows.append({"split": split_name, "sample_index": int(sample_idx), "label": label, "steps": step_logs})

    n = max(len(indices), 1)
    per_step_accuracy = [
        {
            "step": int(step_idx + 1),
            "num_actions_selected": int(step_idx + 1),
            "accuracy": float(per_step_correct_counts[step_idx] / max(per_step_total_counts[step_idx], 1)),
        }
        for step_idx in range(max_steps)
    ]
    first_correct_rate = float(first_correct_count / max(len(indices), 1))
    avg_first_correct_step = float(first_correct_sum / first_correct_count) if first_correct_count > 0 else None
    summary = {
        "split": split_name,
        "num_samples": int(len(indices)),
        "avg_steps": float(total_steps / n),
        "avg_final_confidence": float(total_conf / n),
        "final_accuracy": float(per_step_accuracy[-1]["accuracy"]) if per_step_accuracy else 0.0,
        "per_step_accuracy": per_step_accuracy,
        "per_step_starts_at": 1,
        "first_correct_rate": first_correct_rate,
        "avg_first_correct_step": avg_first_correct_step,
        "first_correct_hist": first_correct_hist,
        "stop_reason_counts": stop_reason_counts,
        "action_selection_counts": selection_counts,
    }
    return path_rows, delta_rows, summary


def main() -> None:
    args = build_arg_parser().parse_args()
    teacher_bundle = load_catboost_teacher_bundle(Path(args.teacher_ckpt))
    checkpoint = teacher_bundle.checkpoint
    model_type = str(checkpoint.get("model_type", "unknown"))
    split_path = Path(args.split_path).resolve() if args.split_path else Path(checkpoint["split_path"]).resolve()
    dataset_csv = Path(args.dataset_csv).resolve() if args.dataset_csv else Path(checkpoint["dataset_csv"]).resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    label_col = args.label_col.strip() or str(checkpoint.get("label_col", "")).strip()
    if not label_col:
        raise ValueError("Label column is required through --label_col or the teacher checkpoint.")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else teacher_bundle.checkpoint_path.parent.parent
    paths_dir = run_dir / "paths"
    summary_dir = run_dir / "summary"
    metrics_dir = run_dir / "metrics"
    paths_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [str(x) for x in checkpoint["feature_columns"]]
    action_ids = [str(x) for x in checkpoint["action_ids"]]
    prerequisite_indices = build_prerequisite_indices(checkpoint, action_ids)
    action_feature_matrix = np.asarray(checkpoint["action_feature_matrix"], dtype=np.float32)
    mean = np.asarray(checkpoint["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization"]["std"], dtype=np.float32)
    std = np.clip(std, 1e-6, None)
    num_classes = int(checkpoint["model_config"]["num_classes"])

    df = pd.read_csv(dataset_csv)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
    x_raw = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(args.missing_value)
        .to_numpy(dtype=np.float32)
    )
    x_norm = ((x_raw - mean) / std).astype(np.float32)
    x_present = (np.isfinite(x_raw) & (x_raw != args.missing_value)).astype(np.float32)
    y = pd.to_numeric(df[label_col], errors="raise").to_numpy(dtype=np.int64)

    split_payload = load_json(split_path)
    max_steps = len(action_ids) if args.max_steps <= 0 else min(int(args.max_steps), len(action_ids))
    all_summaries: list[dict[str, Any]] = []
    all_path_files: dict[str, str] = {}
    all_delta_files: dict[str, str] = {}

    for split_name in ("train", "val", "test"):
        idx = [int(i) for i in split_payload["indices"][split_name]]
        path_rows, delta_rows, summary = rollout_split(
            split_name=split_name,
            indices=idx,
            x_norm=x_norm,
            x_present=x_present,
            y=y,
            model=teacher_bundle.model,
            action_feature_matrix=action_feature_matrix,
            action_ids=action_ids,
            prerequisite_indices=prerequisite_indices,
            num_classes=num_classes,
            max_steps=max_steps,
            save_delta_history=bool(args.save_delta_history),
        )
        path_file = paths_dir / f"teacher_paths_{split_name}.jsonl"
        write_jsonl(path_file, path_rows)
        all_path_files[split_name] = str(path_file)
        if args.save_delta_history:
            delta_file = paths_dir / f"delta_history_{split_name}.jsonl"
            write_jsonl(delta_file, delta_rows)
            all_delta_files[split_name] = str(delta_file)
        all_summaries.append(summary)
        print(
            f"[{split_name}] samples={summary['num_samples']} avg_steps={summary['avg_steps']:.4f} "
            f"final_acc={summary['final_accuracy']:.4f}"
        )

    summary_file = summary_dir / "teacher_paths_summary.json"
    save_json(
        summary_file,
        {
            "teacher_ckpt": str(teacher_bundle.checkpoint_path),
            "teacher_model": str(teacher_bundle.model_path),
            "model_type": model_type,
            "dataset_csv": str(dataset_csv),
            "label_col": label_col,
            "split_path": str(split_path),
            "max_steps": int(max_steps),
            "confidence_threshold": float(args.confidence_threshold),
            "notes": {
                "confidence_threshold": "ignored in full-ranking rollout mode",
            "value": "log probability of true label under mask-aware teacher state",
            "constraint_order": "actions with prerequisites are only legal after prerequisite actions are selected",
            },
            "summaries": all_summaries,
        },
    )
    outputs_file = metrics_dir / "teacher_path_outputs.json"
    save_json(
        outputs_file,
        {
            "teacher_ckpt": str(teacher_bundle.checkpoint_path),
            "teacher_model": str(teacher_bundle.model_path),
            "model_type": model_type,
            "split_path": str(split_path),
            "dataset_csv": str(dataset_csv),
            "label_col": label_col,
            "path_files": all_path_files,
            "delta_files": all_delta_files,
            "summary_file": str(summary_file),
            "max_steps": int(max_steps),
            "confidence_threshold": float(args.confidence_threshold),
            "confidence_threshold_effective": False,
            "metrics_include_empty_state": False,
        },
    )
    print("\nTeacher path generation finished.")
    print(f"Run dir: {run_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
