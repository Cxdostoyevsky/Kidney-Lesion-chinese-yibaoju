#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_WORK_DIR = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "kidney_lesion_modellllllllll/autoseg/segresnet3D_5fold"
)
DEFAULT_INTERNAL_ROOT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "肾肿瘤CT/动脉图像_平扫图像_平扫mask_HU值加窗/cz1_cz2/cz1医院_CY"
)
DEFAULT_EXTERNAL_ROOT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "肾肿瘤CT/动脉图像_平扫图像_平扫mask_HU值加窗/cz1_cz2/cz2医院_CE"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SegResNet five-fold binary tumor predictions."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def load_binary(path: str) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(path)
    return np.asanyarray(image.dataobj) > 0, image.affine


def metric_counts(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int]:
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    return true_positive, false_positive, false_negative


def dice_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def evaluate_case(task: dict) -> dict:
    target, target_affine = load_binary(task["label"])
    if not target.any():
        raise ValueError(f"Expected a positive label, but it is empty: {task['label']}")

    result = {
        "case_id": task["case_id"],
        "label": task["label"],
        "label_voxels": int(target.sum()),
    }
    if "validation_fold" in task:
        result["validation_fold"] = task["validation_fold"]
    for key in ("original_study_id", "base_study_id", "mask_variant"):
        if key in task:
            result[key] = task[key]

    votes = np.zeros(target.shape, dtype=np.uint8)
    for fold, prediction_path in enumerate(task["predictions"]):
        prediction, prediction_affine = load_binary(prediction_path)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {task['case_id']} fold {fold}: "
                f"{prediction.shape} != {target.shape}"
            )
        if not np.allclose(prediction_affine, target_affine, atol=1e-4):
            raise ValueError(f"Affine mismatch for {task['case_id']} fold {fold}")

        true_positive, false_positive, false_negative = metric_counts(prediction, target)
        result[f"fold{fold}_dice"] = dice_from_counts(
            true_positive, false_positive, false_negative
        )
        result[f"fold{fold}_tp"] = true_positive
        result[f"fold{fold}_fp"] = false_positive
        result[f"fold{fold}_fn"] = false_negative
        result[f"fold{fold}_prediction_voxels"] = int(prediction.sum())
        votes += prediction

    ensemble = votes >= 3
    true_positive, false_positive, false_negative = metric_counts(ensemble, target)
    result["ensemble_dice"] = dice_from_counts(
        true_positive, false_positive, false_negative
    )
    result["ensemble_tp"] = true_positive
    result["ensemble_fp"] = false_positive
    result["ensemble_fn"] = false_negative
    result["ensemble_prediction_voxels"] = int(ensemble.sum())
    return result


def summarize(rows: list[dict], metric_prefix: str) -> dict:
    dice_values = np.asarray(
        [row[f"{metric_prefix}_dice"] for row in rows], dtype=np.float64
    )
    true_positive = sum(row[f"{metric_prefix}_tp"] for row in rows)
    false_positive = sum(row[f"{metric_prefix}_fp"] for row in rows)
    false_negative = sum(row[f"{metric_prefix}_fn"] for row in rows)
    prediction_key = f"{metric_prefix}_prediction_voxels"
    standard_error = (
        float(dice_values.std(ddof=1) / math.sqrt(len(dice_values)))
        if len(dice_values) > 1
        else 0.0
    )
    return {
        "cases": len(rows),
        "macro_mean_dice": float(dice_values.mean()),
        "macro_median_dice": float(np.median(dice_values)),
        "macro_std_dice": float(dice_values.std(ddof=1)) if len(rows) > 1 else 0.0,
        "macro_95ci_low": float(dice_values.mean() - 1.96 * standard_error),
        "macro_95ci_high": float(dice_values.mean() + 1.96 * standard_error),
        "micro_dice": dice_from_counts(true_positive, false_positive, false_negative),
        "minimum_dice": float(dice_values.min()),
        "maximum_dice": float(dice_values.max()),
        "empty_predictions": sum(row[prediction_key] == 0 for row in rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_tasks(tasks: list[dict], workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluate_case, tasks, chunksize=1))


def build_internal_tasks(args: argparse.Namespace) -> list[dict]:
    datalist_path = args.work_dir / "cz1_cy_auto3dseg_datalist_tumor_only_5fold.json"
    datalist = json.loads(datalist_path.read_text(encoding="utf-8"))
    tasks = []
    for case in datalist["training"]:
        image_path = Path(case["image"][0])
        case_id = image_path.name.removesuffix("_0000.nii.gz")
        tasks.append(
            {
                "case_id": case_id,
                "validation_fold": int(case["fold"]),
                "label": str(args.internal_root / case["label"]),
                "predictions": [
                    str(
                        args.work_dir
                        / "predictions_all_5fold"
                        / f"fold{fold}"
                        / image_path
                    )
                    for fold in range(5)
                ],
            }
        )
    return tasks


def build_external_tasks(args: argparse.Namespace) -> list[dict]:
    datalist_path = args.work_dir / "cz2_ce_positive_auto3dseg_datalist.json"
    datalist = json.loads(datalist_path.read_text(encoding="utf-8"))
    tasks = []
    for case in datalist["testing"]:
        image_path = Path(case["image"][0])
        tasks.append(
            {
                "case_id": case["case_id"],
                "original_study_id": case["original_study_id"],
                "base_study_id": case["base_study_id"],
                "mask_variant": case["mask_variant"],
                "label": str(args.external_root / "labelsTr" / f"{case['case_id']}.nii.gz"),
                "predictions": [
                    str(
                        args.work_dir
                        / "predictions_cz2_ce_positive_5fold"
                        / f"fold{fold}"
                        / image_path
                    )
                    for fold in range(5)
                ],
            }
        )
    return tasks


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    output_dir = args.work_dir / "dice_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Evaluating 936 internal cases...", flush=True)
    internal_rows = evaluate_tasks(build_internal_tasks(args), args.workers)
    write_csv(output_dir / "internal_all_models_per_case.csv", internal_rows)

    internal_all_summary = {
        f"fold{fold}": summarize(internal_rows, f"fold{fold}") for fold in range(5)
    }
    internal_all_summary["ensemble"] = summarize(internal_rows, "ensemble")

    internal_oof_summary = {}
    oof_rows = []
    for row in internal_rows:
        fold = int(row["validation_fold"])
        prefix = f"fold{fold}"
        oof_row = {
            "case_id": row["case_id"],
            "validation_fold": fold,
            "label": row["label"],
            "label_voxels": row["label_voxels"],
            "oof_dice": row[f"{prefix}_dice"],
            "oof_tp": row[f"{prefix}_tp"],
            "oof_fp": row[f"{prefix}_fp"],
            "oof_fn": row[f"{prefix}_fn"],
            "oof_prediction_voxels": row[f"{prefix}_prediction_voxels"],
        }
        oof_rows.append(oof_row)

    for fold in range(5):
        fold_rows = [row for row in oof_rows if row["validation_fold"] == fold]
        internal_oof_summary[f"fold{fold}"] = summarize(fold_rows, "oof")
    internal_oof_summary["overall"] = summarize(oof_rows, "oof")
    write_csv(output_dir / "internal_oof_per_case.csv", oof_rows)

    print("Evaluating 247 external positive cases...", flush=True)
    external_rows = evaluate_tasks(build_external_tasks(args), args.workers)
    write_csv(output_dir / "external_cz2_positive_per_case.csv", external_rows)
    external_summary = {
        f"fold{fold}": summarize(external_rows, f"fold{fold}") for fold in range(5)
    }
    external_summary["ensemble_majority_vote"] = summarize(
        external_rows, "ensemble"
    )

    summary = {
        "metric_definition": {
            "foreground": "voxel value > 0",
            "case_dice": "2TP / (2TP + FP + FN)",
            "macro_mean_dice": "unweighted mean of per-case Dice",
            "micro_dice": "Dice from TP/FP/FN pooled across all cases",
            "ensemble": "foreground predicted by at least 3 of 5 folds",
        },
        "internal_all_936_per_model": internal_all_summary,
        "internal_oof": internal_oof_summary,
        "external_cz2_positive_247": external_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved evaluation files to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
