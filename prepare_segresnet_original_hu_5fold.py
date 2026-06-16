#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np


AUTOSEG_ROOT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "kidney_lesion_modellllllllll/autoseg"
)
DATA_ROOT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "肾肿瘤CT/平扫图像_动脉图像_动脉mask_原始HU值/cz1医院_CY"
)
WORK_DIR = AUTOSEG_ROOT / "segresnet3D_5fold_原始HU值"
TEMPLATES = AUTOSEG_ROOT / "monai_algo_templates_21ed8e5/algorithm_templates"
DATALIST = WORK_DIR / "cz1_cy_original_hu_auto3dseg_datalist_5fold.json"
INPUT_YAML = WORK_DIR / "input_5fold_segresnet_original_hu.yaml"
NORMALIZED_DATASET = WORK_DIR / "normalized_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Auto3DSeg SegResNet 5-fold files for CZ1 original-HU data."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--normalized-root", type=Path, default=NORMALIZED_DATASET)
    parser.add_argument("--templates", type=Path, default=TEMPLATES)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--prepare-bundles",
        action="store_true",
        help="Run Auto3DSeg analyze and SegResNet bundle generation, without training.",
    )
    return parser.parse_args()


def first_image_component(array: np.ndarray) -> np.ndarray:
    if array.ndim <= 3:
        return np.asarray(array)
    index = (slice(None), slice(None), slice(None)) + tuple(0 for _ in array.shape[3:])
    return np.asarray(array[index])


def label_foreground_union(array: np.ndarray) -> np.ndarray:
    if array.ndim <= 3:
        return (np.asarray(array) > 0).astype(np.uint8)
    axes = tuple(range(3, array.ndim))
    return np.any(np.asarray(array) > 0, axis=axes).astype(np.uint8)


def normalize_or_link(source: Path, target: Path, *, is_label: bool) -> bool:
    image = nib.load(str(source))
    source_shape = image.shape
    needs_fix = len(source_shape) != 3

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()

    if not needs_fix:
        target.symlink_to(source)
        return False

    array = np.asanyarray(image.dataobj)
    fixed = label_foreground_union(array) if is_label else first_image_component(array)
    if fixed.ndim != 3:
        raise ValueError(f"Unable to normalize {source}: {source_shape} -> {fixed.shape}")

    header = image.header.copy()
    fixed_image = nib.Nifti1Image(fixed, image.affine, header)
    fixed_image.set_data_dtype(np.uint8 if is_label else image.get_data_dtype())
    nib.save(fixed_image, str(target))
    return True


def remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def collect_cases(data_root: Path, normalized_root: Path) -> tuple[list[dict], Counter]:
    cases = []
    seen = set()
    normalized_counts = Counter()
    if normalized_root.exists():
        remove_tree(normalized_root)

    for image_dir_name, label_dir_name, source_split in (
        ("imagesTr", "labelsTr", "train"),
        ("imagesTs", "labelsTs", "test"),
    ):
        image_dir = data_root / image_dir_name
        label_dir = data_root / label_dir_name
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing split directories: {image_dir}, {label_dir}")

        for label_path in sorted(label_dir.glob("*.nii.gz")):
            case_id = label_path.name.removesuffix(".nii.gz")
            if case_id in seen:
                raise ValueError(f"Duplicate case id found across splits: {case_id}")
            seen.add(case_id)

            arterial = image_dir / f"{case_id}_0000.nii.gz"
            plain = image_dir / f"{case_id}_0001.nii.gz"
            missing = [str(path) for path in (arterial, plain, label_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing case files:\n" + "\n".join(missing))

            normalized_arterial = normalized_root / image_dir_name / arterial.name
            normalized_plain = normalized_root / image_dir_name / plain.name
            normalized_label = normalized_root / label_dir_name / label_path.name
            if normalize_or_link(arterial, normalized_arterial, is_label=False):
                normalized_counts["fixed_images"] += 1
            else:
                normalized_counts["linked_images"] += 1
            if normalize_or_link(plain, normalized_plain, is_label=False):
                normalized_counts["fixed_images"] += 1
            else:
                normalized_counts["linked_images"] += 1
            if normalize_or_link(label_path, normalized_label, is_label=True):
                normalized_counts["fixed_labels"] += 1
            else:
                normalized_counts["linked_labels"] += 1

            cases.append(
                {
                    "case_id": case_id,
                    "image": [
                        str(normalized_arterial.relative_to(normalized_root)),
                        str(normalized_plain.relative_to(normalized_root)),
                    ],
                    "label": str(normalized_label.relative_to(normalized_root)),
                    "source_split": source_split,
                }
            )
    return cases, normalized_counts


def assign_folds(cases: list[dict], folds: int, seed: int) -> list[dict]:
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    for index, case in enumerate(shuffled):
        case["fold"] = index % folds
    return sorted(shuffled, key=lambda item: (item["fold"], item["case_id"]))


def write_datalist(
    cases: list[dict],
    data_root: Path,
    source_data_root: Path,
    work_dir: Path,
    folds: int,
    seed: int,
) -> Path:
    datalist_path = work_dir / DATALIST.name
    output_cases = []
    for case in cases:
        output_cases.append(
            {
                "image": case["image"],
                "label": case["label"],
                "fold": case["fold"],
                "case_id": case["case_id"],
                "source_split": case["source_split"],
            }
        )

    datalist = {
        "description": "CZ1_CY original-HU arterial-space tumor-only data, train+test merged for 5-fold Auto3DSeg.",
        "name": "kidney_lesion_cz1_cy_original_hu_5fold",
        "tensorImageSize": "3D",
        "modality": {
            "0": "arterial_CT",
            "1": "plain_scan_CT",
        },
        "labels": {
            "0": "background",
            "1": "tumor",
        },
        "numTraining": len(output_cases),
        "fold_assignment": {
            "num_fold": folds,
            "seed": seed,
            "source": "merged imagesTr/labelsTr and imagesTs/labelsTs",
        },
        "dataroot": str(data_root),
        "source_dataset_root": str(source_data_root),
        "training": output_cases,
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    datalist_path.write_text(
        json.dumps(datalist, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return datalist_path


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_input_yaml(
    data_root: Path,
    work_dir: Path,
    datalist_path: Path,
    templates: Path,
    epochs: int,
    folds: int,
) -> Path:
    input_path = work_dir / INPUT_YAML.name
    text = f"""# Auto3DSeg 5-fold SegResNet config for CZ1_CY original-HU arterial-space tumor data.

name: kidney_lesion_cz1_cy_original_hu_5fold
task: segmentation

modality: CT
dataroot: {yaml_scalar(str(data_root))}
datalist: {yaml_scalar(str(datalist_path))}
class_names: [tumor]

num_fold: {folds}
ensemble: true
num_epochs: {epochs}

algos: segresnet
work_dir: {yaml_scalar(str(work_dir))}
templates_path_or_url: {yaml_scalar(str(templates))}
"""
    input_path.write_text(text, encoding="utf-8")
    return input_path


def prepare_bundles(input_yaml: Path, templates: Path, gpus: str) -> None:
    from monai.apps.auto3dseg import AutoRunner

    runner = AutoRunner(
        input=str(input_yaml),
        algos=["segresnet"],
        analyze=True,
        algo_gen=True,
        train=False,
        ensemble=False,
        templates_path_or_url=str(templates),
        allow_skip=True,
    )
    runner.set_device_info(cuda_visible_devices=gpus, num_nodes=1)
    runner.run()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.folds <= 1:
        raise ValueError("Epoch count must be positive and folds must be greater than one.")

    normalized_root = args.normalized_root
    if not normalized_root.is_absolute():
        normalized_root = args.work_dir / normalized_root

    cases, normalized_counts = collect_cases(args.data_root, normalized_root)
    folded_cases = assign_folds(cases, args.folds, args.seed)
    datalist_path = write_datalist(
        folded_cases, normalized_root, args.data_root, args.work_dir, args.folds, args.seed
    )
    input_path = write_input_yaml(
        normalized_root,
        args.work_dir,
        datalist_path,
        args.templates,
        args.epochs,
        args.folds,
    )

    fold_counts = Counter(case["fold"] for case in folded_cases)
    source_counts = Counter(case["source_split"] for case in folded_cases)
    print(f"Wrote {len(folded_cases)} cases to {datalist_path}")
    print(f"Wrote Auto3DSeg input to {input_path}")
    print(f"Fold counts: {dict(sorted(fold_counts.items()))}")
    print(f"Source split counts: {dict(sorted(source_counts.items()))}")
    print(f"Normalized dataset root: {normalized_root}")
    print(f"Normalization counts: {dict(sorted(normalized_counts.items()))}")

    if args.prepare_bundles:
        prepare_bundles(input_path, args.templates, args.gpus)
        print(f"Prepared SegResNet bundles in {args.work_dir}")


if __name__ == "__main__":
    main()
