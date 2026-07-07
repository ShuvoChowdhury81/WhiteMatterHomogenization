#!/usr/bin/env python3
"""Run inference with the published single-component MLP surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = REPO_DIR / "models" / "single_component" / "mlp_fixed_matrix_run_config.json"
DEFAULT_WEIGHTS = REPO_DIR / "models" / "single_component" / "mlp_fixed_matrix_model_weights.npz"


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def load_model(weights_path: Path) -> dict:
    payload = np.load(weights_path)
    weights = []
    biases = []
    idx = 0
    while f"W_{idx}" in payload:
        weights.append(payload[f"W_{idx}"])
        biases.append(payload[f"b_{idx}"])
        idx += 1
    return {
        "x_mean": payload["x_mean"],
        "x_std": payload["x_std"],
        "y_mean": payload["y_mean"],
        "y_std": payload["y_std"],
        "weights": weights,
        "biases": biases,
    }


def forward(x: np.ndarray, model: dict) -> np.ndarray:
    current = x
    last_idx = len(model["weights"]) - 1
    for idx, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
        current = current @ weight + bias
        if idx != last_idx:
            current = relu(current)
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fitted k1 and k2 from an input CSV.")
    parser.add_argument("input_csv", help="CSV containing the required input columns.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the public run-config JSON.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained weights NPZ.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path for the prediction CSV. Defaults next to the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    weights_path = Path(args.weights).resolve()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_csv.with_name(f"{input_csv.stem}_predictions.csv")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = load_model(weights_path)

    df = pd.read_csv(input_csv)
    input_columns = config["input_columns"]
    missing = [col for col in input_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")

    x = df[input_columns].to_numpy(dtype=float)
    x_std = np.where(np.abs(model["x_std"]) > 1.0e-15, model["x_std"], 1.0)
    x_scaled = (x - model["x_mean"]) / x_std
    y_scaled = forward(x_scaled, model)
    y = y_scaled * model["y_std"] + model["y_mean"]

    out = df.copy()
    for idx, target in enumerate(config["target_columns"]):
        out[f"{target}_pred"] = y[:, idx]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


if __name__ == "__main__":
    main()
