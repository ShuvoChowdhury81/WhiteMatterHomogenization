#!/usr/bin/env python3
"""Train a first-pass MLP baseline for fixed-matrix HGO parameter prediction.

This version uses the fixed-matrix dataset exactly as generated:
- Inputs : target fiber volume fraction, target mean diameter,
           matrix mu/alpha, fiber mu/alpha
- Outputs: fitted k1, fitted k2

No log transforms or feature engineering are applied. Optional z-score
standardization is used only as numerical preprocessing for training and
predictions are always reported back in the original physical units.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent


def find_default_project_dir() -> Path:
    return REPO_DIR


PROJECT_DIR = find_default_project_dir()


# ============================================================
# USER SETTINGS
# Edit these directly in VS Code when running this script.
# ============================================================

USER_DATASET_CSV = PROJECT_DIR / "data" / "processed" / "ml_hgo_dataset_lhs_1000_fixed_matrix_public.csv"
USER_OUTPUT_DIR = PROJECT_DIR / "models" / "fixed_matrix_outputs"

USER_INPUT_COLUMNS = [
    "target_fiber_volume_fraction",
    "target_mean_diameter_um",
    "matrix_mu",
    "matrix_alpha",
    "fiber_mu",
    "fiber_alpha",
]

USER_TARGET_COLUMNS = [
    "fitted_k1",
    "fitted_k2",
]

USER_RANDOM_SEED = 42
USER_TRAIN_FRACTION = 0.70
USER_VAL_FRACTION = 0.15
USER_TEST_FRACTION = 0.15

USER_STANDARDIZE_INPUTS = True
USER_STANDARDIZE_TARGETS = True

USER_HIDDEN_LAYER_WIDTHS = [64, 64, 32]
USER_BATCH_SIZE = 64
USER_MAX_EPOCHS = 4000
USER_EARLY_STOPPING_PATIENCE = 400
USER_LEARNING_RATE = 1.0e-3
USER_WEIGHT_DECAY = 1.0e-5

USER_FIG_DPI = 600
USER_FONT_FAMILY = "Times New Roman"
USER_PARITY_LEGEND_FONTSIZE = 12


def set_plot_style():
    plt.rcParams.update(
        {
            "font.family": USER_FONT_FAMILY,
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_target_label(target: str) -> str:
    if target == "fitted_k1":
        return r"$k_1$"
    if target == "fitted_k2":
        return r"$k_2$"
    return target


def parity_axis_limits(target: str) -> tuple[float, float] | None:
    if target == "fitted_k1":
        return 0.0, 250.0
    if target == "fitted_k2":
        return 0.0, 6.0
    return None


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0.0).astype(x.dtype)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    safe = np.where(ss_tot > 1.0e-15, 1.0 - ss_res / ss_tot, 0.0)
    return safe


def compute_metrics(y_true, y_pred, target_names):
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    r2 = compute_r2(y_true, y_pred)
    metrics = {}
    for idx, name in enumerate(target_names):
        metrics[name] = {
            "mae": float(mae[idx]),
            "rmse": float(rmse[idx]),
            "r2": float(r2[idx]),
        }
    metrics["overall"] = {
        "mae_mean": float(np.mean(mae)),
        "rmse_mean": float(np.mean(rmse)),
        "r2_mean": float(np.mean(r2)),
    }
    return metrics


def initialize_parameters(layer_sizes, rng):
    weights = []
    biases = []
    for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
        weight = rng.normal(0.0, np.sqrt(2.0 / in_size), size=(in_size, out_size))
        bias = np.zeros((1, out_size), dtype=float)
        weights.append(weight)
        biases.append(bias)
    return {"weights": weights, "biases": biases}


def copy_parameters(params):
    return {
        "weights": [w.copy() for w in params["weights"]],
        "biases": [b.copy() for b in params["biases"]],
    }


def forward_pass(x, params):
    activations = [x]
    pre_activations = []
    current = x
    last_index = len(params["weights"]) - 1
    for layer_index, (weight, bias) in enumerate(zip(params["weights"], params["biases"])):
        z = current @ weight + bias
        pre_activations.append(z)
        if layer_index == last_index:
            current = z
        else:
            current = relu(z)
        activations.append(current)
    return activations, pre_activations


def compute_loss_and_gradients(x, y, params, weight_decay):
    activations, pre_activations = forward_pass(x, params)
    pred = activations[-1]
    batch_size = x.shape[0]

    data_loss = np.mean((pred - y) ** 2)
    reg_loss = 0.0
    if weight_decay > 0.0:
        reg_loss = 0.5 * weight_decay * sum(np.sum(weight ** 2) for weight in params["weights"])
    loss = data_loss + reg_loss

    grad_output = 2.0 * (pred - y) / batch_size
    grad_weights = [None] * len(params["weights"])
    grad_biases = [None] * len(params["biases"])

    grad = grad_output
    for layer_index in reversed(range(len(params["weights"]))):
        a_prev = activations[layer_index]
        grad_weights[layer_index] = a_prev.T @ grad
        if weight_decay > 0.0:
            grad_weights[layer_index] += weight_decay * params["weights"][layer_index]
        grad_biases[layer_index] = np.sum(grad, axis=0, keepdims=True)
        if layer_index > 0:
            grad = grad @ params["weights"][layer_index].T
            grad *= relu_grad(pre_activations[layer_index - 1])

    grads = {"weights": grad_weights, "biases": grad_biases}
    return loss, grads


def predict(x, params):
    activations, _ = forward_pass(x, params)
    return activations[-1]


def initialize_adam_state(params):
    state = {
        "m_w": [np.zeros_like(weight) for weight in params["weights"]],
        "v_w": [np.zeros_like(weight) for weight in params["weights"]],
        "m_b": [np.zeros_like(bias) for bias in params["biases"]],
        "v_b": [np.zeros_like(bias) for bias in params["biases"]],
        "t": 0,
    }
    return state


def adam_step(params, grads, state, learning_rate, beta1=0.9, beta2=0.999, eps=1.0e-8):
    state["t"] += 1
    t = state["t"]
    for idx in range(len(params["weights"])):
        state["m_w"][idx] = beta1 * state["m_w"][idx] + (1.0 - beta1) * grads["weights"][idx]
        state["v_w"][idx] = beta2 * state["v_w"][idx] + (1.0 - beta2) * (grads["weights"][idx] ** 2)
        state["m_b"][idx] = beta1 * state["m_b"][idx] + (1.0 - beta1) * grads["biases"][idx]
        state["v_b"][idx] = beta2 * state["v_b"][idx] + (1.0 - beta2) * (grads["biases"][idx] ** 2)

        m_w_hat = state["m_w"][idx] / (1.0 - beta1 ** t)
        v_w_hat = state["v_w"][idx] / (1.0 - beta2 ** t)
        m_b_hat = state["m_b"][idx] / (1.0 - beta1 ** t)
        v_b_hat = state["v_b"][idx] / (1.0 - beta2 ** t)

        params["weights"][idx] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
        params["biases"][idx] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)


def fit_standardizer(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std > 1.0e-12, std, 1.0)
    return mean, std


def apply_standardizer(x, mean, std):
    return (x - mean) / std


def inverse_standardizer(x_scaled, mean, std):
    return x_scaled * std + mean


def split_indices(n_samples, rng):
    if not np.isclose(USER_TRAIN_FRACTION + USER_VAL_FRACTION + USER_TEST_FRACTION, 1.0):
        raise ValueError("Train/val/test fractions must sum to 1.")
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_train = int(round(USER_TRAIN_FRACTION * n_samples))
    n_val = int(round(USER_VAL_FRACTION * n_samples))
    n_test = n_samples - n_train - n_val
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def create_batches(x, y, batch_size, rng):
    indices = np.arange(x.shape[0])
    rng.shuffle(indices)
    for start in range(0, x.shape[0], batch_size):
        batch_ids = indices[start:start + batch_size]
        yield x[batch_ids], y[batch_ids]


def save_loss_plot(history_df, output_path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="Train loss", linewidth=1.8)
    ax.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=USER_FIG_DPI)
    plt.close(fig)


def save_parity_plot(predictions_df, target_names, output_path: Path):
    fig, axes = plt.subplots(1, len(target_names), figsize=(4.4 * len(target_names), 4.4))
    if len(target_names) == 1:
        axes = [axes]

    split_colors = {
        "train": "#4C78A8",
        "val": "#F58518",
        "test": "#54A24B",
    }

    for ax, target in zip(axes, target_names):
        target_label = format_target_label(target)
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        limits = parity_axis_limits(target)
        if limits is None:
            min_val = min(predictions_df[true_col].min(), predictions_df[pred_col].min())
            max_val = max(predictions_df[true_col].max(), predictions_df[pred_col].max())
        else:
            min_val, max_val = limits
        for split_name, split_df in predictions_df.groupby("split"):
            ax.scatter(
                split_df[true_col],
                split_df[pred_col],
                s=20,
                alpha=0.78,
                label=split_name,
                color=split_colors.get(split_name, "#333333"),
                edgecolors="none",
            )
        ax.plot([min_val, max_val], [min_val, max_val], color="#222222", linewidth=1.0, linestyle="--")
        ax.set_xlabel(rf"True {target_label}")
        ax.set_ylabel(rf"Predicted {target_label}")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
    axes[0].legend(frameon=False, fontsize=USER_PARITY_LEGEND_FONTSIZE)
    fig.savefig(output_path, dpi=USER_FIG_DPI)
    plt.close(fig)


def save_split_plot(case_ids_by_split, output_path: Path):
    payload = {
        "train_case_ids": case_ids_by_split["train"],
        "val_case_ids": case_ids_by_split["val"],
        "test_case_ids": case_ids_by_split["test"],
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    set_plot_style()
    output_dir = Path(USER_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = Path(USER_DATASET_CSV).resolve()
    if not dataset_csv.exists():
        raise FileNotFoundError("Dataset CSV not found: %s" % dataset_csv)

    df = pd.read_csv(dataset_csv)
    required_cols = ["case_id"] + USER_INPUT_COLUMNS + USER_TARGET_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError("Missing required columns: %s" % missing)

    model_df = df[required_cols].copy()
    model_df = model_df.dropna()
    if len(model_df) != len(df):
        print("Dropped %d rows containing NaN values." % (len(df) - len(model_df)))

    x_all = model_df[USER_INPUT_COLUMNS].to_numpy(dtype=float)
    y_all = model_df[USER_TARGET_COLUMNS].to_numpy(dtype=float)
    case_ids = model_df["case_id"].astype(str).tolist()

    rng = np.random.default_rng(USER_RANDOM_SEED)
    train_idx, val_idx, test_idx = split_indices(len(model_df), rng)

    x_train_raw = x_all[train_idx]
    x_val_raw = x_all[val_idx]
    x_test_raw = x_all[test_idx]
    y_train_raw = y_all[train_idx]
    y_val_raw = y_all[val_idx]
    y_test_raw = y_all[test_idx]

    if USER_STANDARDIZE_INPUTS:
        x_mean, x_std = fit_standardizer(x_train_raw)
        x_train = apply_standardizer(x_train_raw, x_mean, x_std)
        x_val = apply_standardizer(x_val_raw, x_mean, x_std)
        x_test = apply_standardizer(x_test_raw, x_mean, x_std)
        x_all_scaled = apply_standardizer(x_all, x_mean, x_std)
    else:
        x_mean = np.zeros((1, x_train_raw.shape[1]), dtype=float)
        x_std = np.ones((1, x_train_raw.shape[1]), dtype=float)
        x_train = x_train_raw.copy()
        x_val = x_val_raw.copy()
        x_test = x_test_raw.copy()
        x_all_scaled = x_all.copy()

    if USER_STANDARDIZE_TARGETS:
        y_mean, y_std = fit_standardizer(y_train_raw)
        y_train = apply_standardizer(y_train_raw, y_mean, y_std)
        y_val = apply_standardizer(y_val_raw, y_mean, y_std)
        y_test = apply_standardizer(y_test_raw, y_mean, y_std)
    else:
        y_mean = np.zeros((1, y_train_raw.shape[1]), dtype=float)
        y_std = np.ones((1, y_train_raw.shape[1]), dtype=float)
        y_train = y_train_raw.copy()
        y_val = y_val_raw.copy()
        y_test = y_test_raw.copy()

    layer_sizes = [x_train.shape[1]] + list(USER_HIDDEN_LAYER_WIDTHS) + [y_train.shape[1]]
    params = initialize_parameters(layer_sizes, rng)
    adam_state = initialize_adam_state(params)
    best_params = copy_parameters(params)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history_rows = []

    for epoch in range(1, USER_MAX_EPOCHS + 1):
        batch_rng = np.random.default_rng(USER_RANDOM_SEED + epoch)
        for x_batch, y_batch in create_batches(x_train, y_train, USER_BATCH_SIZE, batch_rng):
            train_loss_batch, grads = compute_loss_and_gradients(
                x_batch,
                y_batch,
                params,
                USER_WEIGHT_DECAY,
            )
            adam_step(params, grads, adam_state, USER_LEARNING_RATE)

        train_loss_epoch, _ = compute_loss_and_gradients(x_train, y_train, params, USER_WEIGHT_DECAY)
        val_loss_epoch, _ = compute_loss_and_gradients(x_val, y_val, params, USER_WEIGHT_DECAY)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_epoch),
                "val_loss": float(val_loss_epoch),
            }
        )

        if val_loss_epoch < best_val_loss - 1.0e-12:
            best_val_loss = float(val_loss_epoch)
            best_epoch = epoch
            best_params = copy_parameters(params)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 200 == 0 or epoch == 1:
            print(
                "Epoch %4d | train loss %.6f | val loss %.6f"
                % (epoch, train_loss_epoch, val_loss_epoch)
            )

        if patience_counter >= USER_EARLY_STOPPING_PATIENCE:
            print("Early stopping at epoch %d." % epoch)
            break

    params = best_params

    y_train_pred_scaled = predict(x_train, params)
    y_val_pred_scaled = predict(x_val, params)
    y_test_pred_scaled = predict(x_test, params)
    y_all_pred_scaled = predict(x_all_scaled, params)

    if USER_STANDARDIZE_TARGETS:
        y_train_pred = inverse_standardizer(y_train_pred_scaled, y_mean, y_std)
        y_val_pred = inverse_standardizer(y_val_pred_scaled, y_mean, y_std)
        y_test_pred = inverse_standardizer(y_test_pred_scaled, y_mean, y_std)
        y_all_pred = inverse_standardizer(y_all_pred_scaled, y_mean, y_std)
    else:
        y_train_pred = y_train_pred_scaled
        y_val_pred = y_val_pred_scaled
        y_test_pred = y_test_pred_scaled
        y_all_pred = y_all_pred_scaled

    metrics = {
        "train": compute_metrics(y_train_raw, y_train_pred, USER_TARGET_COLUMNS),
        "val": compute_metrics(y_val_raw, y_val_pred, USER_TARGET_COLUMNS),
        "test": compute_metrics(y_test_raw, y_test_pred, USER_TARGET_COLUMNS),
        "training": {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "max_epochs": USER_MAX_EPOCHS,
            "batch_size": USER_BATCH_SIZE,
            "learning_rate": USER_LEARNING_RATE,
            "weight_decay": USER_WEIGHT_DECAY,
            "hidden_layer_widths": USER_HIDDEN_LAYER_WIDTHS,
            "standardize_inputs": USER_STANDARDIZE_INPUTS,
            "standardize_targets": USER_STANDARDIZE_TARGETS,
        },
    }

    predictions_df = model_df[["case_id"]].copy()
    predictions_df["split"] = "train"
    predictions_df.loc[val_idx, "split"] = "val"
    predictions_df.loc[test_idx, "split"] = "test"
    for idx, target in enumerate(USER_TARGET_COLUMNS):
        predictions_df[f"{target}_true"] = y_all[:, idx]
        predictions_df[f"{target}_pred"] = y_all_pred[:, idx]
        predictions_df[f"{target}_abs_error"] = np.abs(y_all[:, idx] - y_all_pred[:, idx])

    history_df = pd.DataFrame(history_rows)

    predictions_path = output_dir / "mlp_fixed_matrix_predictions.csv"
    history_path = output_dir / "mlp_fixed_matrix_training_history.csv"
    metrics_path = output_dir / "mlp_fixed_matrix_metrics.json"
    split_path = output_dir / "mlp_fixed_matrix_data_split.json"
    config_path = output_dir / "mlp_fixed_matrix_run_config.json"
    weights_path = output_dir / "mlp_fixed_matrix_model_weights.npz"

    predictions_df.to_csv(predictions_path, index=False)
    history_df.to_csv(history_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_split_plot(
        {
            "train": [case_ids[i] for i in train_idx],
            "val": [case_ids[i] for i in val_idx],
            "test": [case_ids[i] for i in test_idx],
        },
        split_path,
    )
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_csv": str(dataset_csv),
                "input_columns": USER_INPUT_COLUMNS,
                "target_columns": USER_TARGET_COLUMNS,
                "random_seed": USER_RANDOM_SEED,
                "train_fraction": USER_TRAIN_FRACTION,
                "val_fraction": USER_VAL_FRACTION,
                "test_fraction": USER_TEST_FRACTION,
                "standardize_inputs": USER_STANDARDIZE_INPUTS,
                "standardize_targets": USER_STANDARDIZE_TARGETS,
                "hidden_layer_widths": USER_HIDDEN_LAYER_WIDTHS,
                "batch_size": USER_BATCH_SIZE,
                "max_epochs": USER_MAX_EPOCHS,
                "early_stopping_patience": USER_EARLY_STOPPING_PATIENCE,
                "learning_rate": USER_LEARNING_RATE,
                "weight_decay": USER_WEIGHT_DECAY,
            },
            f,
            indent=2,
        )

    np.savez(
        weights_path,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        **{f"W_{idx}": weight for idx, weight in enumerate(params["weights"])},
        **{f"b_{idx}": bias for idx, bias in enumerate(params["biases"])},
    )

    save_loss_plot(history_df, output_dir / "mlp_fixed_matrix_loss_history.png")
    save_parity_plot(
        predictions_df,
        USER_TARGET_COLUMNS,
        output_dir / "mlp_fixed_matrix_parity_plots.png",
    )

    print("")
    print("Training complete.")
    print("Best epoch         : %d" % best_epoch)
    print("Best val loss      : %.6f" % best_val_loss)
    print("Predictions CSV    : %s" % predictions_path)
    print("Metrics JSON       : %s" % metrics_path)
    print("Parity plot        : %s" % (output_dir / "mlp_fixed_matrix_parity_plots.png"))


if __name__ == "__main__":
    main()
