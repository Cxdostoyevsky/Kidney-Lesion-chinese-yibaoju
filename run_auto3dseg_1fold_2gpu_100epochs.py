#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_INPUT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "kidney_lesion_modellllllllll/autoseg/input_1fold_full_auto3dseg.yaml"
)
DEFAULT_TEMPLATES = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "kidney_lesion_modellllllllll/autoseg/monai_algo_templates_21ed8e5/"
    "algorithm_templates"
)
EXPECTED_BUNDLES = ("dints_0", "segresnet_0", "segresnet2d_0", "swinunetr_0")


def apply_dints_memory_overrides(work_dir: Path, config_parser: type) -> None:
    config_path = work_dir / "dints_0" / "configs" / "hyper_parameters.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"DiNTS config not found: {config_path}")

    config = config_parser.load_config_file(config_path)
    config["training"].update(
        {
            "auto_scale_allowed": False,
            "num_cache_workers": 2,
            "num_workers": 2,
            "num_workers_validation": 1,
            "num_images_per_batch": 1,
            "num_crops_per_image": 40,
            "num_patches_per_iter": 20,
        }
    )
    config_parser.export_config_file(
        config,
        config_path,
        fmt="yaml",
        default_flow_style=None,
        sort_keys=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or train the one-fold, four-model Auto3DSeg experiment."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run data analysis and bundle generation without training.",
    )
    mode.add_argument(
        "--train",
        action="store_true",
        help="Train all generated bundles for 100 epochs.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["SEGRESNET2D_ALWAYS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    env_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    from monai.apps.auto3dseg import AutoRunner
    from monai.bundle import ConfigParser

    if not args.templates.is_dir():
        raise FileNotFoundError(f"Auto3DSeg templates not found: {args.templates}")

    input_config = ConfigParser.load_config_file(args.input)
    work_dir = Path(input_config["work_dir"])

    runner = AutoRunner(input=str(args.input), allow_skip=False)
    runner.algos = ["dints", "segresnet", "segresnet2d", "swinunetr"]
    runner.templates_path_or_url = str(args.templates)
    runner.allow_skip = False
    runner.ensemble = False
    runner.set_device_info(cuda_visible_devices=args.gpus)
    runner.set_training_params(
        {
            "num_epochs": args.epochs,
            "num_epochs_per_validation": args.validation_interval,
        }
    )

    if args.prepare_only:
        runner.train = False
        runner.run()
        missing = [name for name in EXPECTED_BUNDLES if not (work_dir / name).is_dir()]
        if missing:
            raise RuntimeError(f"Bundle generation did not create: {', '.join(missing)}")
        apply_dints_memory_overrides(work_dir, ConfigParser)
        print(f"Prepared four bundles in {work_dir}")
        return

    apply_dints_memory_overrides(work_dir, ConfigParser)
    runner.train = None
    runner.run()


if __name__ == "__main__":
    main()
