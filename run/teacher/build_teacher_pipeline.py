

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_project_root(start: Path) -> Path:
    for parent in [start] + list(start.parents):
        if (parent / "datasets").exists() and (parent / "run").exists():
            return parent
    raise FileNotFoundError(f"Could not find project root from {start}.")


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


def build_arg_parser() -> argparse.ArgumentParser:
    project_root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Run teacher pipeline: select a mask-aware teacher backbone then rollout teacher path.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--actions_json", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument(
        "--version",
        nargs="?",
        const="",
        default=None,
        help="Result version folder. Use empty/none/null/- for datasets without version folders.",
    )
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
    parser.add_argument("--teacher_rollout_select_max_samples", type=int, default=0)
    parser.add_argument("--iterations_list", default="auto")
    parser.add_argument("--depth_list", default="auto")
    parser.add_argument("--learning_rate_list", default="auto")
    parser.add_argument("--subsample_list", default="auto")
    parser.add_argument("--rsm_list", default="auto")
    parser.add_argument("--l2_leaf_reg_list", default="auto")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--max_grid_combinations", type=int, default=96)
    parser.add_argument(
        "--allow_uncovered_features",
        action="store_true",
        help="Allow CSV feature columns that are intentionally not assigned to an action.",
    )
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--confidence_threshold", type=float, default=-1.0)
    parser.add_argument("--save_delta_history", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="")
    parser.add_argument("--epochs", type=int, default=0, help="Ignored.")
    return parser


def run_subprocess(cmd: list[str]) -> None:
    print("\n[CMD]")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    script_dir = Path(__file__).resolve().parent
    teacher_training_script = script_dir / "train_masked_teacher.py"
    teacher_path_script = script_dir / "generate_teacher_paths.py"
    dataset_name, dataset_csv, actions_json, label_col, version = resolve_dataset_options(
        project_root=project_root,
        dataset_name=args.dataset_name,
        dataset_csv_arg=args.dataset_csv,
        actions_json_arg=args.actions_json,
        label_col_arg=args.label_col,
        version_arg=args.version,
    )
    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    if not actions_json.exists():
        raise FileNotFoundError(f"Actions JSON not found: {actions_json}")

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path(args.results_root).resolve()
    dataset_root = build_dataset_root(results_root, dataset_name, version)
    split_path = dataset_root / "split" / f"split_seed{args.seed}.json"
    run_dir = dataset_root / "runs" / run_id
    teacher_ckpt = run_dir / "ckpts" / "teacher_best.pt"

    teacher_training_cmd = [
        sys.executable,
        str(teacher_training_script),
        "--dataset_csv",
        str(dataset_csv),
        "--actions_json",
        str(actions_json),
        "--label_col",
        label_col,
        "--dataset_name",
        dataset_name,
        "--version",
        version,
        "--results_root",
        str(results_root),
        "--run_id",
        run_id,
        "--seed",
        str(args.seed),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--test_ratio",
        str(args.test_ratio),
        "--missing_value",
        str(args.missing_value),
        "--train_masks_per_sample",
        str(args.train_masks_per_sample),
        "--val_masks_per_sample",
        str(args.val_masks_per_sample),
        "--random_mask_seed_offset",
        str(args.random_mask_seed_offset),
        "--teacher_backbones",
        args.teacher_backbones,
        "--teacher_select_strategy",
        args.teacher_select_strategy,
        "--teacher_select_masked_weight",
        str(args.teacher_select_masked_weight),
        "--teacher_select_full_weight",
        str(args.teacher_select_full_weight),
        "--teacher_rollout_select_mean_weight",
        str(args.teacher_rollout_select_mean_weight),
        "--teacher_rollout_select_final_weight",
        str(args.teacher_rollout_select_final_weight),
        "--teacher_rollout_select_max_samples",
        str(args.teacher_rollout_select_max_samples),
        "--iterations_list",
        args.iterations_list,
        "--depth_list",
        args.depth_list,
        "--learning_rate_list",
        args.learning_rate_list,
        "--subsample_list",
        args.subsample_list,
        "--rsm_list",
        args.rsm_list,
        "--l2_leaf_reg_list",
        args.l2_leaf_reg_list,
        "--n_jobs",
        str(args.n_jobs),
        "--max_grid_combinations",
        str(args.max_grid_combinations),
    ]
    if args.force_new_split:
        teacher_training_cmd.append("--force_new_split")
    if args.allow_uncovered_features:
        teacher_training_cmd.append("--allow_uncovered_features")
    run_subprocess(teacher_training_cmd)

    teacher_path_cmd = [
        sys.executable,
        str(teacher_path_script),
        "--dataset_csv",
        str(dataset_csv),
        "--label_col",
        label_col,
        "--teacher_ckpt",
        str(teacher_ckpt),
        "--split_path",
        str(split_path),
        "--run_dir",
        str(run_dir),
        "--missing_value",
        str(args.missing_value),
        "--max_steps",
        str(args.max_steps),
        "--confidence_threshold",
        str(args.confidence_threshold),
        "--device",
        args.device,
    ]
    if args.save_delta_history:
        teacher_path_cmd.append("--save_delta_history")
    run_subprocess(teacher_path_cmd)

    print("\nTeacher pipeline finished.")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(f"Teacher checkpoint: {teacher_ckpt}")
    print(f"Split file: {split_path}")


if __name__ == "__main__":
    main()
