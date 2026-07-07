#!/usr/bin/env python3
"""Train an MLP surrogate for the lhs_1000 fixed-matrix campaign datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
PARENT_DIR = SCRIPT_DIR
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import train_mlp_hgo_fixed_matrix as base_trainer


DATASET_MAP = {
    "frobenius": REPO_DIR / "data" / "processed" / "ml_hgo_dataset_lhs_1000_fixed_matrix_public.csv",
    "single_component": REPO_DIR / "data" / "processed" / "ml_hgo_dataset_lhs_1000_single_component_pressure_public.csv",
}

OUTPUT_DIR_MAP = {
    "frobenius": REPO_DIR / "models" / "frobenius",
    "single_component": REPO_DIR / "models" / "single_component_retrained",
}

BASE_INPUT_COLUMNS = [
    "target_fiber_volume_fraction",
    "target_mean_diameter_um",
    "matrix_mu",
    "matrix_alpha",
    "fiber_mu",
    "fiber_alpha",
]

INTERACTION_FEATURE_NAME = "mu_interaction_feature"


def parse_hidden_layers(text: str) -> list[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("Hidden layer widths must contain at least one integer.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an MLP model for the lhs_1000 fixed-matrix campaign using either "
            "the Frobenius-pressure dataset or the single-component-pressure dataset."
        )
    )
    parser.add_argument(
        "--dataset-mode",
        choices=tuple(DATASET_MAP.keys()),
        default="single_component",
        help="Choose which campaign dataset to train on.",
    )
    parser.add_argument(
        "--dataset-csv",
        default=None,
        help="Optional explicit dataset CSV path. Overrides --dataset-mode if given.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Overrides the dataset-mode default.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--hidden-layers",
        default="64,64,32",
        help="Comma-separated hidden layer widths.",
    )
    parser.add_argument(
        "--disable-stiffness-ratio-feature",
        dest="use_stiffness_ratio_feature",
        action="store_false",
        help="Train using only the six base inputs and omit the extra interaction feature.",
    )
    parser.add_argument(
        "--no-standardize-inputs",
        dest="standardize_inputs",
        action="store_false",
        help="Disable z-score standardization for inputs.",
    )
    parser.add_argument(
        "--no-standardize-targets",
        dest="standardize_targets",
        action="store_false",
        help="Disable z-score standardization for targets.",
    )
    parser.set_defaults(
        standardize_inputs=True,
        standardize_targets=True,
        use_stiffness_ratio_feature=True,
    )
    return parser.parse_args()


def flatten_metrics(metrics: dict) -> pd.DataFrame:
    rows = []
    for split_name in ("train", "val", "test"):
        split_metrics = metrics.get(split_name, {})
        for target_name, target_metrics in split_metrics.items():
            if target_name == "overall":
                continue
            rows.append(
                {
                    "split": split_name,
                    "target": target_name,
                    "mae": target_metrics.get("mae"),
                    "rmse": target_metrics.get("rmse"),
                    "r2": target_metrics.get("r2"),
                }
            )
    overall = metrics.get("training", {})
    if overall:
        rows.append(
            {
                "split": "training",
                "target": "summary",
                "mae": None,
                "rmse": overall.get("best_val_loss"),
                "r2": None,
            }
        )
    return pd.DataFrame(rows)


def build_augmented_dataset_csv(source_csv: Path, use_stiffness_ratio_feature: bool, output_dir: Path) -> tuple[Path, list[str]]:
    if not use_stiffness_ratio_feature:
        return source_csv, BASE_INPUT_COLUMNS

    df = pd.read_csv(source_csv)
    required_cols = set(BASE_INPUT_COLUMNS)
    missing = sorted(col for col in required_cols if col not in df.columns)
    if missing:
        raise KeyError("Dataset is missing base input columns required for interaction feature: %s" % missing)

    matrix_mu = df["matrix_mu"].to_numpy(dtype=float)
    fiber_mu = df["fiber_mu"].to_numpy(dtype=float)
    safe_matrix_mu = np.where(np.abs(matrix_mu) > 1.0e-15, matrix_mu, np.nan)
    df[INTERACTION_FEATURE_NAME] = (fiber_mu + matrix_mu) / safe_matrix_mu

    augmented_csv = output_dir / (source_csv.stem + "_with_interaction.csv")
    df.to_csv(augmented_csv, index=False)
    return augmented_csv, [*BASE_INPUT_COLUMNS, INTERACTION_FEATURE_NAME]


def main() -> None:
    args = parse_args()

    dataset_csv = Path(args.dataset_csv).resolve() if args.dataset_csv else DATASET_MAP[args.dataset_mode]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR_MAP[args.dataset_mode]
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_dataset_csv, input_columns = build_augmented_dataset_csv(
        dataset_csv,
        args.use_stiffness_ratio_feature,
        output_dir,
    )

    base_trainer.USER_DATASET_CSV = trainer_dataset_csv
    base_trainer.USER_OUTPUT_DIR = output_dir
    base_trainer.USER_INPUT_COLUMNS = input_columns
    base_trainer.USER_RANDOM_SEED = args.random_seed
    base_trainer.USER_TRAIN_FRACTION = args.train_fraction
    base_trainer.USER_VAL_FRACTION = args.val_fraction
    base_trainer.USER_TEST_FRACTION = args.test_fraction
    base_trainer.USER_STANDARDIZE_INPUTS = args.standardize_inputs
    base_trainer.USER_STANDARDIZE_TARGETS = args.standardize_targets
    base_trainer.USER_HIDDEN_LAYER_WIDTHS = parse_hidden_layers(args.hidden_layers)
    base_trainer.USER_BATCH_SIZE = args.batch_size
    base_trainer.USER_MAX_EPOCHS = args.max_epochs
    base_trainer.USER_EARLY_STOPPING_PATIENCE = args.patience
    base_trainer.USER_LEARNING_RATE = args.learning_rate
    base_trainer.USER_WEIGHT_DECAY = args.weight_decay

    print("Dataset mode      : %s" % args.dataset_mode)
    print("Dataset CSV       : %s" % dataset_csv)
    print("Trainer CSV       : %s" % trainer_dataset_csv)
    print("Output directory  : %s" % output_dir)
    print("Input columns     : %s" % input_columns)
    print("Hidden layers     : %s" % base_trainer.USER_HIDDEN_LAYER_WIDTHS)
    print("")

    base_trainer.main()

    metrics_path = output_dir / "mlp_fixed_matrix_metrics.json"
    summary_csv_path = output_dir / "mlp_fixed_matrix_metrics_summary.csv"
    run_info_path = output_dir / "mlp_fixed_matrix_campaign_run_info.json"

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    flatten_metrics(metrics).to_csv(summary_csv_path, index=False)

    with run_info_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_mode": args.dataset_mode,
                "dataset_csv": str(dataset_csv),
                "trainer_dataset_csv": str(trainer_dataset_csv),
                "output_dir": str(output_dir),
                "base_input_columns": BASE_INPUT_COLUMNS,
                "input_columns": input_columns,
                "interaction_feature": INTERACTION_FEATURE_NAME if args.use_stiffness_ratio_feature else None,
                "hidden_layers": base_trainer.USER_HIDDEN_LAYER_WIDTHS,
                "random_seed": args.random_seed,
                "train_fraction": args.train_fraction,
                "val_fraction": args.val_fraction,
                "test_fraction": args.test_fraction,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "standardize_inputs": args.standardize_inputs,
                "standardize_targets": args.standardize_targets,
            },
            f,
            indent=2,
        )

    print("Metrics summary   : %s" % summary_csv_path)
    print("Run info JSON     : %s" % run_info_path)


if __name__ == "__main__":
    main()
