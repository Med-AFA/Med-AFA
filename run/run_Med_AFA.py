from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "run"
DATASETS_ROOT = PROJECT_ROOT / "datasets"
RESULTS_ROOT = PROJECT_ROOT / "results"


def resolve_path(value: str | None, default: Path) -> Path:
    if value is None or not str(value).strip():
        return default.resolve()
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def run_command(command: list[str], cwd: Path) -> None:
    print("\n$ " + " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def latest_teacher_dir(results_root: Path, dataset_name: str, run_id: str) -> Path:
    direct = results_root / dataset_name / "runs" / run_id
    if direct.exists():
        return direct.resolve()
    candidates = sorted((results_root / dataset_name / "runs").glob(f"{run_id}*"))
    if not candidates:
        raise FileNotFoundError(f"Teacher run directory not found for run_id={run_id}: {direct}")
    return candidates[-1].resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def remove_owned_path(path: Path, results_root: Path) -> None:
    if not path.exists():
        return
    resolved_path = path.resolve()
    resolved_root = results_root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to remove path outside results root: {resolved_path}") from exc
    if resolved_path == resolved_root:
        raise ValueError("Refusing to remove the results root.")
    if resolved_path.is_dir():
        shutil.rmtree(resolved_path)
    else:
        resolved_path.unlink()


def remove_empty_parents(paths: list[Path], results_root: Path) -> None:
    resolved_root = results_root.resolve()
    for path in paths:
        current = path
        while current.exists() and current.resolve() != resolved_root:
            if any(current.iterdir()):
                break
            current.rmdir()
            current = current.parent


def copy_teacher_artifacts(teacher_dir: Path, destination: Path) -> list[str]:
    checkpoint_dir = teacher_dir / "ckpts"
    copied = []
    if not checkpoint_dir.exists():
        return copied
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(checkpoint_dir.glob("teacher_best.*")):
        if source.suffix == ".pt":
            continue
        target = destination / source.name
        shutil.copy2(source, target)
        copied.append(str(Path("artifacts") / "teacher" / source.name))
    return copied


def finalize_results(
    *,
    args: argparse.Namespace,
    results_root: Path,
    dataset_name: str,
    base_run_id: str,
    split_path: Path,
    teacher_dir: Path | None,
    teacher_will_run: bool,
    student_dir: Path | None,
    teacher_backbones: str,
) -> Path:
    final_dir = results_root / dataset_name / base_run_id
    final_dir.mkdir(parents=True, exist_ok=False)

    teacher_summary = {}
    if teacher_dir:
        teacher_summary_path = teacher_dir / "summary" / "teacher_paths_summary.json"
        if teacher_summary_path.exists():
            teacher_summary = read_json(teacher_summary_path)
    teacher_artifacts = copy_teacher_artifacts(teacher_dir, final_dir / "artifacts" / "teacher") if teacher_dir else []
    if teacher_artifacts:
        write_json(
            final_dir / "artifacts" / "teacher" / "teacher_metadata.json",
            {
                "model_type": teacher_summary.get("model_type"),
                "label_col": args.label_col,
                "model_files": [Path(item).name for item in teacher_artifacts],
            },
        )

    student_summary = {}
    trials_payload: list[dict[str, Any]] = []
    if student_dir:
        summary_path = student_dir / "summary.json"
        trials_path = student_dir / "trials.json"
        if summary_path.exists():
            student_summary = read_json(summary_path)
        if trials_path.exists():
            trials_payload = list(read_json(trials_path).get("trials", []))
        if args.save_diagnostics:
            diagnostics_dir = final_dir / "diagnostics"
            for source in sorted(student_dir.glob("*diagnostics.json")):
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, diagnostics_dir / source.name)

    metrics = {
        "dataset": dataset_name,
        "mean_acc@all": student_summary.get("mean_acc@all"),
        "final_acc": student_summary.get("final_acc"),
        "per_action_accuracy": student_summary.get("per_action_accuracy", []),
        "constraint_valid_rate": student_summary.get("constraint_valid_rate"),
        "num_trials": student_summary.get("num_trials", 0),
    }
    run_config = {
        "dataset": dataset_name,
        "inputs": {
            "dataset_csv": portable_path(resolve_path(args.dataset_csv, PROJECT_ROOT)),
            "actions_path": portable_path(resolve_path(args.actions_path, PROJECT_ROOT)),
            "label_col": args.label_col,
            "split": "split.json",
        },
        "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "teacher": {
            "candidate_backbones": [item.strip() for item in teacher_backbones.split(",") if item.strip()],
            "train_masks_per_sample": 2 if args.smoke else int(args.teacher_train_masks_per_sample),
            "validation_masks_per_sample": 2 if args.smoke else int(args.teacher_val_masks_per_sample),
            "max_grid_combinations": 1 if args.smoke else int(args.teacher_max_grid_combinations),
            "selected_model_type": teacher_summary.get("model_type"),
            "artifacts": teacher_artifacts,
        },
        "student": {
            "num_trials": 1 if args.smoke else int(args.student_num_trials),
            "pretrain_epochs": 1 if args.smoke else int(args.student_pretrain_epochs),
            "train_epochs": 1 if args.smoke else int(args.student_train_epochs),
            "architecture_selection": "fixed" if args.smoke else args.student_arch_selection_mode,
            "predictor_arch_grid": args.predictor_arch_grid,
        },
        "save_diagnostics": bool(args.save_diagnostics),
    }
    write_json(final_dir / "metrics.json", metrics)
    write_json(final_dir / "trials.json", {"trials": trials_payload})
    write_json(final_dir / "run_config.json", run_config)
    split_payload = read_json(split_path)
    split_payload["dataset_csv"] = portable_path(resolve_path(args.dataset_csv, PROJECT_ROOT))
    write_json(final_dir / "split.json", split_payload)

    if student_dir:
        remove_owned_path(student_dir, results_root)
    if teacher_will_run and teacher_dir:
        remove_owned_path(teacher_dir, results_root)
    if not args.split_path:
        remove_owned_path(split_path, results_root)
    remove_empty_parents(
        [
            results_root / "student" / dataset_name,
            results_root / "student",
            results_root / dataset_name / "runs",
            results_root / dataset_name / "split",
        ],
        results_root,
    )
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pipeline.")
    parser.add_argument("--dataset_name", required=True, help="Dataset identifier used only for output folders.")
    parser.add_argument("--dataset_csv", required=True, help="Path to the tabular CSV file.")
    parser.add_argument("--actions_path", required=True, help="Path to the action-definition JSON file.")
    parser.add_argument("--label_col", required=True, help="Name of the label column in the CSV file.")
    parser.add_argument("--split_path", default="", help="Optional split JSON. Empty uses results/<dataset_name>/split/split_seed<seed>.json and lets teacher create it.")
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--run_id", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable, help="Default Python interpreter for both teacher and student.")
    parser.add_argument("--teacher_python", default="", help="Optional Python interpreter used only for teacher construction.")
    parser.add_argument("--student_python", default="", help="Optional Python interpreter used only for student training.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use a tiny configuration for a quick end-to-end check.")
    parser.add_argument("--skip_teacher", action="store_true")
    parser.add_argument("--skip_student", action="store_true")
    parser.add_argument("--teacher_run_dir", default="", help="Existing teacher run directory. If set, teacher creation is skipped unless --force_teacher is also set.")
    parser.add_argument("--force_teacher", action="store_true")
    parser.add_argument("--teacher_backbones", default="catboost,xgboost,logistic_regression,mlp")
    parser.add_argument("--teacher_train_masks_per_sample", type=int, default=96)
    parser.add_argument("--teacher_val_masks_per_sample", type=int, default=32)
    parser.add_argument("--teacher_max_grid_combinations", type=int, default=96)
    parser.add_argument("--student_num_trials", type=int, default=1)
    parser.add_argument("--student_pretrain_epochs", type=int, default=15)
    parser.add_argument("--student_train_epochs", type=int, default=80)
    parser.add_argument("--student_arch_selection_mode", default="val_constraint", choices=["fixed", "val_constraint", "test_constraint"])
    parser.add_argument("--predictor_arch_grid", default="base")
    parser.add_argument("--save_diagnostics", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_name = args.dataset_name.strip().lower()
    dataset_csv = resolve_path(args.dataset_csv, PROJECT_ROOT)
    actions_path = resolve_path(args.actions_path, PROJECT_ROOT)
    results_root = resolve_path(args.results_root, RESULTS_ROOT)
    split_path = resolve_path(args.split_path, results_root / dataset_name / "split" / f"split_seed{args.seed}.json")
    teacher_python = args.teacher_python.strip() or args.python
    student_python = args.student_python.strip() or args.python
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_id = args.run_id.strip() or f"med_afa_{dataset_name}_seed{args.seed}_{timestamp}"
    teacher_run_id = f"{base_run_id}_teacher"
    student_run_id = f"{base_run_id}_student"
    final_dir = results_root / dataset_name / base_run_id
    backbones = "catboost" if args.smoke else args.teacher_backbones

    if not dataset_csv.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_csv}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions JSON not found: {actions_path}")
    if final_dir.exists():
        raise FileExistsError(f"Final results directory already exists: {final_dir}")
    teacher_dir = resolve_path(args.teacher_run_dir, results_root / dataset_name / "runs" / teacher_run_id) if args.teacher_run_dir else None
    teacher_will_run = (not args.skip_teacher) and (args.force_teacher or not teacher_dir or not teacher_dir.exists())
    if not split_path.exists() and not teacher_will_run:
        raise FileNotFoundError(f"Split JSON not found: {split_path}")

    if teacher_will_run:
        train_masks = 2 if args.smoke else args.teacher_train_masks_per_sample
        val_masks = 2 if args.smoke else args.teacher_val_masks_per_sample
        max_grid = 1 if args.smoke else args.teacher_max_grid_combinations
        teacher_cmd = [
            teacher_python,
            str(RUN_ROOT / "teacher" / "build_teacher_pipeline.py"),
            "--dataset_name", dataset_name,
            "--version", "",
            "--dataset_csv", str(dataset_csv),
            "--actions_json", str(actions_path),
            "--label_col", args.label_col,
            "--results_root", str(results_root),
            "--run_id", teacher_run_id,
            "--seed", str(args.seed),
            "--teacher_backbones", backbones,
            "--train_masks_per_sample", str(train_masks),
            "--val_masks_per_sample", str(val_masks),
            "--max_grid_combinations", str(max_grid),
            "--n_jobs", "1",
            "--max_steps", "1" if args.smoke else "0",
            "--device", args.device,
        ]
        if args.smoke:
            teacher_cmd.extend([
                "--iterations_list", "5",
                "--depth_list", "2",
                "--learning_rate_list", "0.03",
                "--subsample_list", "1.0",
                "--rsm_list", "1.0",
                "--l2_leaf_reg_list", "3.0",
            ])
        run_command(teacher_cmd, PROJECT_ROOT)
        teacher_dir = latest_teacher_dir(results_root, dataset_name, teacher_run_id)
    elif not teacher_dir:
        teacher_dir = latest_teacher_dir(results_root, dataset_name, teacher_run_id)

    if not split_path.exists():
        raise FileNotFoundError(f"Split JSON was not created by teacher pipeline: {split_path}")

    if not args.skip_student:
        pretrain_epochs = 1 if args.smoke else args.student_pretrain_epochs
        train_epochs = 1 if args.smoke else args.student_train_epochs
        num_trials = 1 if args.smoke else args.student_num_trials
        arch_selection_mode = "fixed" if args.smoke else args.student_arch_selection_mode
        student_cmd = [
            student_python,
            str(RUN_ROOT / "student" / "train_med_afa_student.py"),
            "--dataset", dataset_name,
            "--dataset_csv", str(dataset_csv),
            "--label_col", args.label_col,
            "--actions_path", str(actions_path),
            "--split_path", str(split_path),
            "--teacher_run_dir", str(teacher_dir),
            "--results_dir", str(results_root / "student"),
            "--run_id", student_run_id,
            "--seed", str(args.seed),
            "--num_trials", str(num_trials),
            "--device", args.device,
            "--gpu", str(args.gpu),
            "--pretrain_epochs", str(pretrain_epochs),
            "--train_epochs", str(train_epochs),
            "--arch_selection_mode", arch_selection_mode,
            "--predictor_arch_grid", args.predictor_arch_grid,
            "--disable_candidate_cache",
        ]
        if args.save_diagnostics:
            student_cmd.append("--save_diagnostics")
        else:
            student_cmd.extend([
                "--disable_rerank_diagnostics",
                "--disable_intervention_sensitivity_diagnostics",
            ])
        run_command(student_cmd, PROJECT_ROOT)

    student_dir = results_root / "student" / dataset_name / student_run_id if not args.skip_student else None
    output_dir = finalize_results(
        args=args,
        results_root=results_root,
        dataset_name=dataset_name,
        base_run_id=base_run_id,
        split_path=split_path,
        teacher_dir=teacher_dir,
        teacher_will_run=teacher_will_run,
        student_dir=student_dir,
        teacher_backbones=backbones,
    )
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
