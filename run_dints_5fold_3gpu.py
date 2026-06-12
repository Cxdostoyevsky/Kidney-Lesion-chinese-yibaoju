#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


AUTOSEG_ROOT = Path(
    "/hdd/common/datasets/medical-image-analysis/3D_kindey_lesion_changzhou/"
    "kidney_lesion_modellllllllll/autoseg"
)
DEFAULT_INPUT = AUTOSEG_ROOT / "input_5fold_dints.yaml"
DEFAULT_TEMPLATES = AUTOSEG_ROOT / "monai_algo_templates_21ed8e5/algorithm_templates"
DEFAULT_STATS_DIR = AUTOSEG_ROOT / "segresnet3D_5fold"
EXPECTED_BUNDLES = tuple(f"dints_{fold}" for fold in range(5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train five Auto3DSeg DiNTS folds."
    )
    parser.add_argument(
        "--mode",
        choices=("prepare", "train", "all"),
        default="all",
        help="Prepare bundles, train prepared bundles, or do both.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument(
        "--reuse-stats-from",
        type=Path,
        default=DEFAULT_STATS_DIR,
        help="Directory containing datastats.yaml for the same dataset.",
    )
    parser.add_argument("--gpus", default="3,5,7")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=4)
    return parser.parse_args()


def copy_reusable_stats(source_dir: Path, work_dir: Path) -> None:
    source_stats = source_dir / "datastats.yaml"
    target_stats = work_dir / "datastats.yaml"
    if not source_stats.is_file():
        raise FileNotFoundError(f"Reusable data statistics not found: {source_stats}")

    work_dir.mkdir(parents=True, exist_ok=True)
    if not target_stats.exists():
        shutil.copy2(source_stats, target_stats)

    source_by_case = source_dir / "datastats_by_case.yaml"
    target_by_case = work_dir / "datastats_by_case.yaml"
    if source_by_case.is_file() and not target_by_case.exists():
        shutil.copy2(source_by_case, target_by_case)


def validate_reusable_stats(
    input_config: dict, stats: dict, datalist: dict
) -> None:
    training = datalist.get("training", [])
    summary = stats.get("stats_summary", {})
    channels = summary.get("image_stats", {}).get("channels", {}).get("max")
    labels = summary.get("label_stats", {}).get("labels")

    if summary.get("n_cases") != len(training):
        raise ValueError("Reusable statistics case count does not match the datalist.")
    if channels != 2:
        raise ValueError(f"Expected two image channels, but statistics report {channels}.")
    if labels != [0, 1]:
        raise ValueError(f"Expected labels [0, 1], but statistics report {labels}.")
    if int(input_config.get("num_fold", 0)) != 5:
        raise ValueError("The input configuration is not configured for five folds.")


def configure_bundles(
    work_dir: Path, config_parser: type, epochs: int, validation_interval: int
) -> None:
    missing = [name for name in EXPECTED_BUNDLES if not (work_dir / name).is_dir()]
    if missing:
        raise RuntimeError(f"Bundle generation did not create: {', '.join(missing)}")

    for bundle_name in EXPECTED_BUNDLES:
        config_path = work_dir / bundle_name / "configs/hyper_parameters.yaml"
        config = config_parser.load_config_file(config_path)
        config["training"].update(
            {
                "num_epochs": epochs,
                "num_epochs_per_validation": validation_interval,
            }
        )
        config_parser.export_config_file(
            config,
            config_path,
            fmt="yaml",
            default_flow_style=None,
            sort_keys=False,
        )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.validation_interval <= 0:
        raise ValueError("Epoch counts and validation interval must be positive.")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MONAI_ALLOW_PICKLE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    env_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"

    from monai.apps.auto3dseg import AutoRunner
    from monai.bundle import ConfigParser

    input_config = ConfigParser.load_config_file(args.input)
    work_dir = Path(input_config["work_dir"])

    if args.mode in ("prepare", "all"):
        copy_reusable_stats(args.reuse_stats_from, work_dir)
        stats = ConfigParser.load_config_file(work_dir / "datastats.yaml")
        datalist = ConfigParser.load_config_file(input_config["datalist"])
        validate_reusable_stats(input_config, stats, datalist)

        prepare_runner = AutoRunner(
            input=str(args.input),
            algos=["dints"],
            analyze=False,
            algo_gen=True,
            train=False,
            ensemble=False,
            templates_path_or_url=str(args.templates),
            allow_skip=False,
        )
        prepare_runner.ensemble = False
        prepare_runner.set_device_info(cuda_visible_devices=args.gpus, num_nodes=1)
        prepare_runner.run()
        prepare_runner.export_cache(
            analyze=True,
            datastats=str(work_dir / "datastats.yaml"),
        )
        configure_bundles(
            work_dir,
            ConfigParser,
            args.epochs,
            args.validation_interval,
        )
        print(f"Prepared five DiNTS bundles in {work_dir}", flush=True)

    if args.mode in ("train", "all"):
        configure_bundles(
            work_dir,
            ConfigParser,
            args.epochs,
            args.validation_interval,
        )
        train_runner = AutoRunner(
            input=str(args.input),
            algos=["dints"],
            analyze=False,
            algo_gen=False,
            train=None,
            ensemble=False,
            templates_path_or_url=str(args.templates),
            allow_skip=True,
        )
        train_runner.ensemble = False
        train_runner.set_device_info(cuda_visible_devices=args.gpus, num_nodes=1)
        train_runner.set_training_params(
            {
                "num_epochs": args.epochs,
                "num_epochs_per_validation": args.validation_interval,
            }
        )
        train_runner.run()


if __name__ == "__main__":
    main()
