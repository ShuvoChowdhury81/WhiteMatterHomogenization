#!/usr/bin/env python3
"""Refit ML-surrogate case folders with a fixed one-term Ogden matrix and free k1, k2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from run_hgo_ml_dataset_pipeline import read_case_rows, sanitize_name


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
USER_CASES_CSV = REPO_DIR / "data" / "case_tables" / "ml_hgo_case_table_lhs_1000.csv"
USER_RUNS_DIR = REPO_DIR / "runs" / "ml_dataset_runs_lhs_1000_dual_target"
DEFAULT_RESULT_NAME = "fixed_matrix_k1_k2_result.json"
DEFAULT_SUMMARY_NAME = "fixed_matrix_k1_k2_summary.txt"
DEFAULT_PREDICTIONS_NAME = "fixed_matrix_k1_k2_predictions.csv"

CASE_MAP = {
    "tx": {
        "pattern": "*_UNIAXIAL_X.csv",
        "stress": "P11",
        "x": "F11",
    },
    "ty": {
        "pattern": "*_UNIAXIAL_Y.csv",
        "stress": "P22",
        "x": "F22",
    },
    "sxy": {
        "pattern": "*_PURE_SHEAR_XY.csv",
        "stress": "P12",
        "x": "F12",
    },
    "syz": {
        "pattern": "*_PURE_SHEAR_YZ.csv",
        "stress": "P23",
        "x": "F23",
    },
    "szx": {
        "pattern": "*_PURE_SHEAR_XZ.csv",
        "stress": "P31",
        "x": "F31",
    },
}

NEEDED_COLUMNS = [
    "load_case",
    "time",
    "W_hom",
    "kappa",
    "I4_star",
    "F11",
    "F12",
    "F13",
    "F21",
    "F22",
    "F23",
    "F31",
    "F32",
    "F33",
    "P11",
    "P12",
    "P13",
    "P21",
    "P22",
    "P23",
    "P31",
    "P32",
    "P33",
    "H11",
    "H12",
    "H13",
    "H21",
    "H22",
    "H23",
    "H31",
    "H32",
    "H33",
]

P_NAMES = ["P11", "P12", "P13", "P21", "P22", "P23", "P31", "P32", "P33"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit finished ML-surrogate case folders with a fixed exact one-term Ogden "
            "matrix term and fit only k1, k2."
        )
    )
    parser.add_argument("--cases-csv", default=str(USER_CASES_CSV))
    parser.add_argument("--runs-dir", default=str(USER_RUNS_DIR))
    parser.add_argument("--case-start", type=int, default=1, help="1-based start row in the case table.")
    parser.add_argument(
        "--case-count",
        type=int,
        default=None,
        help="Optional number of case-table rows to process from --case-start.",
    )
    parser.add_argument(
        "--initial-guess",
        default="40.0,25.0",
        help="Comma-separated initial guess for k1,k2.",
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float, default=1.0)
    parser.add_argument(
        "--stress-mode",
        choices=("relevant", "full_tensor"),
        default="relevant",
        help="Use only the requested Pij components or all nine P components.",
    )
    parser.add_argument("--robust-loss", default="linear")
    parser.add_argument("--result-name", default=DEFAULT_RESULT_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--predictions-name", default=DEFAULT_PREDICTIONS_NAME)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Reuse an existing fixed-matrix result JSON if it is already present.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force the refit even if a fixed-matrix result JSON already exists.",
    )
    return parser.parse_args()


def parse_initial_guess(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",")]
    if len(values) != 2:
        raise ValueError("--initial-guess must contain exactly k1,k2")
    return np.array(values, dtype=float)


def load_case_rows_subset(cases_csv: Path, case_start: int, case_count: int | None) -> list[dict]:
    rows = read_case_rows(str(cases_csv))
    if case_start < 1:
        raise ValueError("--case-start must be >= 1")

    start_idx = case_start - 1
    if start_idx >= len(rows):
        raise IndexError("--case-start=%d is beyond the case table length %d" % (case_start, len(rows)))

    if case_count is None:
        end_idx = len(rows)
    else:
        if case_count < 1:
            raise ValueError("--case-count must be >= 1 when provided")
        end_idx = min(start_idx + case_count, len(rows))
    return rows[start_idx:end_idx]


def load_summary_json(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("rve_truss_generation_summary*.json"))
    if not matches:
        raise FileNotFoundError("Missing summary JSON in %s" % case_dir)
    if len(matches) > 1:
        raise RuntimeError("Expected one summary JSON in %s, found %d" % (case_dir, len(matches)))
    return matches[0]


def load_fea_data(case_dir: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for load_case, meta in CASE_MAP.items():
        matches = sorted(case_dir.glob(meta["pattern"]))
        if len(matches) == 0:
            missing.append(str(case_dir / meta["pattern"]))
            continue
        if len(matches) > 1:
            raise RuntimeError(
                "Expected one CSV for %s in %s, found %d" % (load_case, case_dir, len(matches))
            )
        df = pd.read_csv(matches[0])
        df["load_case"] = load_case
        frames.append(df)

    if missing:
        raise FileNotFoundError("Missing required FEA CSV files:\n" + "\n".join(missing))

    df_all = pd.concat(frames, ignore_index=True)
    return df_all[NEEDED_COLUMNS].dropna().reset_index(drop=True)


def macaulay(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def safe_exp_fiber_energy(k1: float, k2: float, x: np.ndarray) -> np.ndarray:
    if abs(k2) < 1.0e-12:
        return 0.5 * k1 * x**2
    arg = np.clip(k2 * x**2, None, 50.0)
    return (k1 / (2.0 * k2)) * (np.exp(arg) - 1.0)


def dwd_i4star(k1: float, k2: float, i4s: np.ndarray) -> np.ndarray:
    x = macaulay(i4s - 1.0)
    out = np.zeros_like(i4s, dtype=float)
    active = i4s > 1.0
    if not np.any(active):
        return out
    if abs(k2) < 1.0e-12:
        out[active] = k1 * x[active]
    else:
        arg = np.clip(k2 * x[active] ** 2, None, 50.0)
        out[active] = k1 * x[active] * np.exp(arg)
    return out


def frob_inner(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a * b))


def fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if abs(ss_tot) < 1.0e-16:
        return rmse, float("nan")
    return rmse, 1.0 - ss_res / ss_tot


def build_case_weight_array(df: pd.DataFrame) -> np.ndarray:
    case_counts = df["load_case"].value_counts().to_dict()
    return np.array([1.0 / case_counts[case] for case in df["load_case"]], dtype=float)


def build_case_scale_maps(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    energy_scale_map = {}
    stress_scale_map = {}
    tensor_scale_map = {}
    for case in df["load_case"].unique():
        df_case = df[df["load_case"] == case]
        stress_comp = CASE_MAP[case]["stress"]
        energy_scale_map[case] = max(float(np.max(np.abs(df_case["W_hom"].values))), 1.0)
        stress_scale_map[case] = max(float(np.max(np.abs(df_case[stress_comp].values))), 1.0)
        tensor_scale_map[case] = max(float(np.max(np.abs(df_case[P_NAMES].values))), 1.0)
    return energy_scale_map, stress_scale_map, tensor_scale_map


def exact_ogden_matrix_response(
    f: np.ndarray,
    mu: float,
    alpha: float,
) -> tuple[float, np.ndarray]:
    if abs(alpha) < 1.0e-12:
        raise ValueError("Matrix alpha is too close to zero for a one-term Ogden response.")

    j_det = float(np.linalg.det(f))
    if j_det <= 0.0:
        raise ValueError("Encountered non-positive Jacobian in matrix response: %g" % j_det)

    b = f @ f.T
    eigvals, eigvecs = np.linalg.eigh(b)
    eigvals = np.clip(eigvals, 1.0e-16, None)
    lambdas = np.sqrt(eigvals)
    lambdabar = (j_det ** (-1.0 / 3.0)) * lambdas

    power_vals = np.power(lambdabar, alpha)
    w_matrix = float((2.0 * mu / (alpha * alpha)) * (np.sum(power_vals) - 3.0))

    mean_power = float(np.mean(power_vals))
    tau_principal = (2.0 * mu / alpha) * (power_vals - mean_power)
    tau_matrix = eigvecs @ np.diag(tau_principal) @ eigvecs.T
    p_matrix = tau_matrix @ np.linalg.inv(f).T
    return w_matrix, p_matrix


def compute_predicted_response(
    df: pd.DataFrame,
    k_params: np.ndarray,
    mu_matrix: float,
    alpha_matrix: float,
) -> pd.DataFrame:
    k1, k2 = k_params
    i4s = df["I4_star"].values
    dwd4 = dwd_i4star(k1, k2, i4s)

    df_out = df.copy()
    df_out["W_matrix_fixed"] = np.zeros(len(df_out))
    df_out["W_fiber_pred"] = np.zeros(len(df_out))
    df_out["W_pred"] = np.zeros(len(df_out))
    df_out["J_model"] = np.zeros(len(df_out))
    df_out["p_lagrange"] = np.zeros(len(df_out))
    for name in P_NAMES:
        df_out[name + "_pred"] = np.zeros(len(df_out))

    for idx, row in df.iterrows():
        f = np.array(
            [
                [row["F11"], row["F12"], row["F13"]],
                [row["F21"], row["F22"], row["F23"]],
                [row["F31"], row["F32"], row["F33"]],
            ],
            dtype=float,
        )
        h = np.array(
            [
                [row["H11"], row["H12"], row["H13"]],
                [row["H21"], row["H22"], row["H23"]],
                [row["H31"], row["H32"], row["H33"]],
            ],
            dtype=float,
        )
        p_fe = np.array(
            [
                [row["P11"], row["P12"], row["P13"]],
                [row["P21"], row["P22"], row["P23"]],
                [row["P31"], row["P32"], row["P33"]],
            ],
            dtype=float,
        )

        w_matrix, p_matrix = exact_ogden_matrix_response(f=f, mu=mu_matrix, alpha=alpha_matrix)
        w_fiber = safe_exp_fiber_energy(k1, k2, np.array([max(row["I4_star"] - 1.0, 0.0)]))[0]

        s_fiber = 2.0 * dwd4[idx] * h
        p_fiber = f @ s_fiber
        p_material = p_matrix + p_fiber
        f_inv_t = np.linalg.inv(f).T
        denominator = frob_inner(f_inv_t, f_inv_t)
        if abs(denominator) < 1.0e-16:
            p_lagrange = 0.0
        else:
            p_lagrange = frob_inner(f_inv_t, (p_material - p_fe)) / denominator
        p_pred = p_material - p_lagrange * f_inv_t

        df_out.at[idx, "W_matrix_fixed"] = w_matrix
        df_out.at[idx, "W_fiber_pred"] = w_fiber
        df_out.at[idx, "W_pred"] = w_matrix + w_fiber
        df_out.at[idx, "J_model"] = float(np.linalg.det(f))
        df_out.at[idx, "p_lagrange"] = p_lagrange

        for comp_idx, comp in enumerate(P_NAMES):
            row_idx = comp_idx // 3
            col_idx = comp_idx % 3
            df_out.at[idx, comp + "_pred"] = p_pred[row_idx, col_idx]

    return df_out


def combined_residuals(
    k_params: np.ndarray,
    df: pd.DataFrame,
    mu_matrix: float,
    alpha_matrix: float,
    case_weights: np.ndarray,
    energy_scale_map: dict[str, float],
    stress_scale_map: dict[str, float],
    tensor_scale_map: dict[str, float],
    stress_mode: str,
    energy_weight: float,
    stress_weight: float,
) -> np.ndarray:
    df_pred = compute_predicted_response(df, k_params, mu_matrix, alpha_matrix)
    base_weights = np.sqrt(case_weights)

    energy_residual = df_pred["W_pred"].values - df_pred["W_hom"].values
    energy_scale = np.array([energy_scale_map[case] for case in df_pred["load_case"]], dtype=float)
    energy_residual = np.sqrt(energy_weight) * base_weights * energy_residual / energy_scale

    if stress_mode == "relevant":
        stress_true = np.zeros(len(df_pred))
        stress_pred = np.zeros(len(df_pred))
        stress_scale = np.ones(len(df_pred))
        for i, case in enumerate(df_pred["load_case"]):
            comp = CASE_MAP[case]["stress"]
            stress_true[i] = df_pred.iloc[i][comp]
            stress_pred[i] = df_pred.iloc[i][comp + "_pred"]
            stress_scale[i] = stress_scale_map[case]
        stress_residual = np.sqrt(stress_weight) * base_weights * (stress_pred - stress_true) / stress_scale
        return np.concatenate([energy_residual, stress_residual])

    if stress_mode == "full_tensor":
        stress_blocks = []
        for comp in P_NAMES:
            comp_scale = np.array([tensor_scale_map[case] for case in df_pred["load_case"]], dtype=float)
            comp_residual = (
                np.sqrt(stress_weight)
                * base_weights
                * (df_pred[comp + "_pred"].values - df_pred[comp].values)
                / comp_scale
            )
            stress_blocks.append(comp_residual)
        return np.concatenate([energy_residual] + stress_blocks)

    raise ValueError("Unsupported stress mode: %s" % stress_mode)


def build_result_dict(
    case_id: str,
    case_dir: Path,
    summary_json: Path,
    summary_data: dict,
    df_pred: pd.DataFrame,
    kfit: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    matrix_terms = summary_data.get("matrix_ogden_terms", [[None, None, None]])
    mu_matrix = matrix_terms[0][0]
    alpha_matrix = matrix_terms[0][1]
    d_matrix = matrix_terms[0][2]

    out = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "summary_json": str(summary_json),
        "stress_mode": args.stress_mode,
        "energy_weight": args.energy_weight,
        "stress_weight": args.stress_weight,
        "robust_loss": args.robust_loss,
        "matrix_term": {
            "mu": float(mu_matrix),
            "alpha": float(alpha_matrix),
            "D": float(d_matrix),
            "fit_representation": "exact_one_term_ogden",
        },
        "params": {
            "k1": float(kfit[0]),
            "k2": float(kfit[1]),
        },
        "metrics": {},
        "per_case_metrics": {},
        "realized_kappa_mean": (
            float(pd.to_numeric(df_pred["kappa"], errors="coerce").mean())
            if "kappa" in df_pred.columns
            else None
        ),
    }

    if abs(float(alpha_matrix) + 4.0) < 1.0e-12:
        out["matrix_term"]["c10_invariant_equiv"] = float(mu_matrix) / 4.0

    rmse_energy, r2_energy = fit_metrics(df_pred["W_hom"].values, df_pred["W_pred"].values)
    out["metrics"]["energy"] = {"rmse": rmse_energy, "r2": r2_energy}

    stress_true_all = []
    stress_pred_all = []
    for case in CASE_MAP:
        comp = CASE_MAP[case]["stress"]
        mask = df_pred["load_case"] == case
        rmse_case, r2_case = fit_metrics(
            df_pred.loc[mask, comp].values,
            df_pred.loc[mask, comp + "_pred"].values,
        )
        out["per_case_metrics"][case] = {
            "stress_component": comp,
            "rmse": rmse_case,
            "r2": r2_case,
        }
        stress_true_all.append(df_pred.loc[mask, comp].values)
        stress_pred_all.append(df_pred.loc[mask, comp + "_pred"].values)

    stress_true_all = np.concatenate(stress_true_all)
    stress_pred_all = np.concatenate(stress_pred_all)
    rmse_stress, r2_stress = fit_metrics(stress_true_all, stress_pred_all)
    out["metrics"]["relevant_stress"] = {"rmse": rmse_stress, "r2": r2_stress}
    return out


def format_summary_text(result_dict: dict) -> str:
    lines = []
    lines.append("Modified HGO fixed-matrix fit from existing ML case data")
    lines.append("case_id = %s" % result_dict["case_id"])
    lines.append("stress objective mode = %s" % result_dict["stress_mode"])
    lines.append("energy objective weight = %.6f" % result_dict["energy_weight"])
    lines.append("stress objective weight = %.6f" % result_dict["stress_weight"])
    lines.append("")
    lines.append("Fixed matrix term:")
    lines.append("mu = %.8f" % result_dict["matrix_term"]["mu"])
    lines.append("alpha = %.8f" % result_dict["matrix_term"]["alpha"])
    lines.append("D = %.8f" % result_dict["matrix_term"]["D"])
    lines.append("fit representation = %s" % result_dict["matrix_term"]["fit_representation"])
    if "c10_invariant_equiv" in result_dict["matrix_term"]:
        lines.append("Abaqus invariant C10_equiv = %.8f" % result_dict["matrix_term"]["c10_invariant_equiv"])
    lines.append("")
    lines.append("Fitted fiber parameters:")
    lines.append("k1 = %.8f" % result_dict["params"]["k1"])
    lines.append("k2 = %.8f" % result_dict["params"]["k2"])
    lines.append("")
    lines.append("Energy fit quality:")
    lines.append("RMSE = %.8e" % result_dict["metrics"]["energy"]["rmse"])
    lines.append("R^2  = %.8f" % result_dict["metrics"]["energy"]["r2"])
    lines.append("")
    lines.append("Relevant stress fit quality:")
    lines.append("RMSE = %.8e" % result_dict["metrics"]["relevant_stress"]["rmse"])
    lines.append("R^2  = %.8f" % result_dict["metrics"]["relevant_stress"]["r2"])
    lines.append("")
    for case in CASE_MAP:
        rec = result_dict["per_case_metrics"][case]
        lines.append(
            "%s: %s -> RMSE = %.8e, R^2 = %.8f"
            % (case, rec["stress_component"], rec["rmse"], rec["r2"])
        )
    return "\n".join(lines)


def fit_case_dir(case_id: str, case_dir: Path, args: argparse.Namespace) -> dict:
    summary_json = load_summary_json(case_dir)
    summary_data = json.loads(summary_json.read_text(encoding="utf-8-sig"))
    matrix_terms = summary_data.get("matrix_ogden_terms", [[None, None, None]])
    mu_matrix = float(matrix_terms[0][0])
    alpha_matrix = float(matrix_terms[0][1])

    df_all = load_fea_data(case_dir)
    case_weights = build_case_weight_array(df_all)
    energy_scale_map, stress_scale_map, tensor_scale_map = build_case_scale_maps(df_all)

    result = least_squares(
        combined_residuals,
        parse_initial_guess(args.initial_guess),
        bounds=(np.array([0.0, 1.0e-8]), np.array([1.0e6, 1.0e6])),
        loss=args.robust_loss,
        args=(
            df_all,
            mu_matrix,
            alpha_matrix,
            case_weights,
            energy_scale_map,
            stress_scale_map,
            tensor_scale_map,
            args.stress_mode,
            args.energy_weight,
            args.stress_weight,
        ),
    )

    kfit = result.x
    df_pred = compute_predicted_response(df_all, kfit, mu_matrix, alpha_matrix)
    result_dict = build_result_dict(case_id, case_dir, summary_json, summary_data, df_pred, kfit, args)
    result_dict["optimizer"] = {
        "cost": float(result.cost),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
    }
    return {
        "summary_json": summary_json,
        "result_dict": result_dict,
        "predictions_df": df_pred,
        "summary_text": format_summary_text(result_dict),
    }


def main() -> None:
    args = parse_args()
    cases_csv = Path(args.cases_csv).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    if not cases_csv.exists():
        raise FileNotFoundError("Cases CSV not found: %s" % cases_csv)
    if not runs_dir.exists():
        raise FileNotFoundError("Runs directory not found: %s" % runs_dir)

    selected_rows = load_case_rows_subset(cases_csv, args.case_start, args.case_count)
    processed = 0
    reused = 0

    for row in selected_rows:
        case_id = sanitize_name(row.get("case_id") or "case")
        case_dir = runs_dir / case_id
        if not case_dir.exists():
            raise FileNotFoundError("Case directory not found: %s" % case_dir)

        result_json = case_dir / args.result_name
        summary_txt = case_dir / args.summary_name
        predictions_csv = case_dir / args.predictions_name

        print("")
        print("=" * 80)
        print("Case:", case_id)
        print("Directory:", case_dir)

        if args.skip_existing and result_json.exists():
            reused += 1
            print("Existing fixed-matrix result found. Reusing:")
            print(result_json)
            continue

        fit_payload = fit_case_dir(case_id, case_dir, args)
        fit_payload["predictions_df"].to_csv(predictions_csv, index=False)
        summary_txt.write_text(fit_payload["summary_text"] + "\n", encoding="utf-8")
        result_json.write_text(json.dumps(fit_payload["result_dict"], indent=2), encoding="utf-8")

        print(fit_payload["summary_text"])
        print("")
        print("Saved predictions :", predictions_csv)
        print("Saved summary     :", summary_txt)
        print("Saved result JSON :", result_json)
        processed += 1

    print("")
    print("=" * 80)
    print("Finished fixed-matrix refit")
    print("Processed cases :", processed)
    print("Reused cases    :", reused)
    print("Selected rows   :", len(selected_rows))


if __name__ == "__main__":
    main()
