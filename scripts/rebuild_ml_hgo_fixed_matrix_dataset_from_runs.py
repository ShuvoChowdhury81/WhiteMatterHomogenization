#!/usr/bin/env python3
"""Rebuild an ML dataset from finished case folders using fixed-matrix k1,k2 refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_hgo_ml_dataset_pipeline import (
    DEFAULT_TARGET_MEAN_PS,
    DEFAULT_TARGET_VOLUME_FRACTION,
    FORTRAN_TEMPLATE,
    USER_CASES_CSV,
    USER_RUNS_DIR,
    parse_optional_float,
    read_case_rows,
    sanitize_name,
    write_dataset_csv,
    write_dataset_jsonl,
)


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_RESULT_NAME = "fixed_matrix_k1_k2_result.json"
USER_OUTPUT_CSV = REPO_DIR / "data" / "generated" / "ml_hgo_dataset_lhs_1000_fixed_matrix.csv"
USER_OUTPUT_JSONL = REPO_DIR / "data" / "generated" / "ml_hgo_dataset_lhs_1000_fixed_matrix.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild an ML dataset from finished case folders using fixed-matrix result JSON files."
        )
    )
    parser.add_argument("--cases-csv", default=str(USER_CASES_CSV))
    parser.add_argument("--runs-dir", default=str(USER_RUNS_DIR))
    parser.add_argument("--output-csv", default=str(USER_OUTPUT_CSV))
    parser.add_argument("--output-jsonl", default=str(USER_OUTPUT_JSONL))
    parser.add_argument("--result-name", default=DEFAULT_RESULT_NAME)
    parser.add_argument(
        "--skip-missing-results",
        action="store_true",
        default=True,
        help="Skip case folders that do not yet have the fixed-matrix result JSON.",
    )
    parser.add_argument(
        "--strict-missing-results",
        dest="skip_missing_results",
        action="store_false",
        help="Fail if a case folder is missing the fixed-matrix result JSON.",
    )
    return parser.parse_args()


def load_case_table_by_id(cases_csv: Path) -> dict[str, dict]:
    case_rows = read_case_rows(str(cases_csv))
    rows_by_case_id = {}
    for index, row in enumerate(case_rows, start=1):
        case_id = sanitize_name(row.get("case_id") or ("case_%04d" % index))
        rows_by_case_id[case_id] = row
    return rows_by_case_id


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_summary_json(case_dir: Path) -> Path:
    matches = sorted(case_dir.glob("rve_truss_generation_summary*.json"))
    if not matches:
        raise FileNotFoundError("Missing summary JSON in %s" % case_dir)
    if len(matches) > 1:
        raise RuntimeError("Expected one summary JSON in %s, found %d" % (case_dir, len(matches)))
    return matches[0]


def build_dataset_row(
    case_row: dict,
    case_id: str,
    case_dir: Path,
    summary_json: Path,
    summary_data: dict,
    fit_result_json: Path,
    fit_data: dict,
) -> dict:
    matrix_terms = summary_data.get("matrix_ogden_terms", [[None, None, None]])
    matrix_mu = matrix_terms[0][0]
    matrix_alpha = matrix_terms[0][1]
    matrix_d = matrix_terms[0][2]
    axon_props = summary_data.get("axon_user_properties", [None, None])
    params = fit_data.get("params", {})
    energy_metrics = fit_data.get("metrics", {}).get("energy", {})
    stress_metrics = fit_data.get("metrics", {}).get("relevant_stress", {})
    generator_variant = (case_row.get("generator_variant") or "straight").strip().lower()

    row = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "generator_variant": generator_variant,
        "summary_json": str(summary_json),
        "fit_result_json": str(fit_result_json),
        "user_subroutine_path": str(FORTRAN_TEMPLATE),
        "fiber_volume_fraction": summary_data.get("achieved_approx_volume_fraction"),
        "mean_diameter_um": summary_data.get("diameter_mean_um"),
        "kappa": fit_data.get("realized_kappa_mean"),
        "mean_ps": summary_data.get("straightness_mean"),
        "matrix_mu": matrix_mu,
        "matrix_alpha": matrix_alpha,
        "matrix_d": matrix_d,
        "fiber_mu": axon_props[0] if len(axon_props) >= 1 else None,
        "fiber_alpha": axon_props[1] if len(axon_props) >= 2 else None,
        "target_fiber_volume_fraction": parse_optional_float(
            case_row, "target_volume_fraction", DEFAULT_TARGET_VOLUME_FRACTION
        ),
        "target_mean_diameter_um": parse_optional_float(case_row, "target_mean_diameter_um", None),
        "target_kappa": parse_optional_float(case_row, "target_kappa", 0.0),
        "target_mean_ps": parse_optional_float(case_row, "target_mean_ps", DEFAULT_TARGET_MEAN_PS),
        "fitted_k1": params.get("k1"),
        "fitted_k2": params.get("k2"),
        "fit_energy_rmse": energy_metrics.get("rmse"),
        "fit_energy_r2": energy_metrics.get("r2"),
        "fit_relevant_stress_rmse": stress_metrics.get("rmse"),
        "fit_relevant_stress_r2": stress_metrics.get("r2"),
        "accepted_axons": summary_data.get("accepted_axons"),
        "attempts_total": summary_data.get("attempts_total"),
        "seed": summary_data.get("seed"),
        "matrix_element_type": summary_data.get("matrix_element_type"),
        "matrix_fit_representation": fit_data.get("matrix_term", {}).get("fit_representation"),
    }

    if "c10_invariant_equiv" in fit_data.get("matrix_term", {}):
        row["matrix_c10_invariant_equiv"] = fit_data["matrix_term"]["c10_invariant_equiv"]

    if generator_variant == "oriented":
        pref = summary_data.get("dominant_orientation", {}).get("preferred_direction_vector")
        if pref is not None and len(pref) == 3:
            row["preferred_direction_x"] = pref[0]
            row["preferred_direction_y"] = pref[1]
            row["preferred_direction_z"] = pref[2]

    return row


def rebuild_rows(
    cases_csv: Path,
    runs_dir: Path,
    result_name: str,
    skip_missing_results: bool,
) -> list[dict]:
    case_rows_by_id = load_case_table_by_id(cases_csv)
    rows = []
    case_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("case_")])
    if not case_dirs:
        raise RuntimeError("No case directories found in %s" % runs_dir)

    for case_dir in case_dirs:
        case_id = case_dir.name
        if case_id not in case_rows_by_id:
            raise KeyError("Case '%s' not found in case table %s" % (case_id, cases_csv))

        fit_result_json = case_dir / result_name
        if not fit_result_json.exists():
            if skip_missing_results:
                continue
            raise FileNotFoundError("Missing fixed-matrix result JSON for %s: %s" % (case_id, fit_result_json))

        summary_json = load_summary_json(case_dir)
        summary_data = load_json(summary_json)
        fit_data = load_json(fit_result_json)
        case_row = case_rows_by_id[case_id]

        rows.append(
            build_dataset_row(
                case_row=case_row,
                case_id=case_id,
                case_dir=case_dir,
                summary_json=summary_json,
                summary_data=summary_data,
                fit_result_json=fit_result_json,
                fit_data=fit_data,
            )
        )

    return rows


def main() -> None:
    args = parse_args()
    cases_csv = Path(args.cases_csv).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()

    if not cases_csv.exists():
        raise FileNotFoundError("Cases CSV not found: %s" % cases_csv)
    if not runs_dir.exists():
        raise FileNotFoundError("Runs directory not found: %s" % runs_dir)

    rows = rebuild_rows(cases_csv, runs_dir, args.result_name, args.skip_missing_results)
    if not rows:
        raise RuntimeError("No fixed-matrix result rows were found to write.")

    write_dataset_csv(rows, output_csv)
    write_dataset_jsonl(rows, output_jsonl)

    print("")
    print("Rebuilt fixed-matrix dataset rows :", len(rows))
    print("Dataset CSV                       :", output_csv)
    print("Dataset JSONL                     :", output_jsonl)


if __name__ == "__main__":
    main()
