#!/usr/bin/env python3
"""End-to-end ML pipeline for the LHS dual-target fixed-matrix campaign.

This campaign runner is intentionally separate from the older grid-sampling
pipeline. It is configured for:
- the 1000-case LHS table
- the dual-target axon generator
- fixed-matrix modified-HGO refits (fit k1, k2 only)
- the current production geometry / mesh choices
  - RVE edge length = 30 um
  - hex element length = 1.5 um
  - truss element length = 1.5 um
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent

STRAIGHT_GENERATOR = PROJECT_DIR / "generate_brain_wm_rve_truss_inp_histology_segmented_ps_dual_target_abaqus.py"
RUN_BATCH_SCRIPT = PROJECT_DIR / "run_abaqus_inp_batch.py"
EXTRACT_SCRIPT = PROJECT_DIR / "extract_3d_modified_hgo_data_3RP_incompressible.py"
REFIT_SCRIPT = PROJECT_DIR / "refit_fixed_matrix_k1_k2_from_runs.py"
REBUILD_SCRIPT = PROJECT_DIR / "rebuild_ml_hgo_fixed_matrix_dataset_from_runs.py"
FORTRAN_TEMPLATE = PROJECT_DIR / "truss_with_recruited_stretch_Ogden_hyperelastic_single_fiber.for"

CASE_COMPLETION_FILE = "case_completion.json"
CASE_FAILURE_FILE = "case_failure.json"

# ============================================================
# USER RUN SETTINGS
# Edit these directly in VS Code when running this script.
# Command-line arguments still override these values if provided.
# ============================================================
USER_CASES_CSV = ROOT_DIR / "data" / "case_tables" / "ml_hgo_case_table_lhs_1000.csv"
USER_RUNS_DIR = ROOT_DIR / "runs" / "ml_dataset_runs_lhs_1000_dual_target"
USER_DATASET_CSV = ROOT_DIR / "data" / "generated" / "ml_hgo_dataset_lhs_1000_fixed_matrix.csv"
USER_DATASET_JSONL = ROOT_DIR / "data" / "generated" / "ml_hgo_dataset_lhs_1000_fixed_matrix.jsonl"
USER_ABAQUS_CMD = "abaqus"
USER_DEFAULT_CPUS = 4
USER_CASE_START = 1000
USER_CASE_COUNT = 1
USER_SKIP_EXISTING = False
USER_CONTINUE_ON_ERROR = True
USER_DRY_RUN = False

USER_RVE_EDGE_UM = 30.0
USER_HEX_ELEMENT_LENGTH_UM = 3.0
USER_TRUSS_ELEMENT_LENGTH_UM = 3.0
USER_MATRIX_ELEMENT_TYPE = "C3D8RH"
USER_TARGET_STRETCH = 1.40
USER_TARGET_SHEAR_GAMMA = 0.20

DEFAULT_DIAMETER_SHAPE_K = 1.5492
DEFAULT_PS_BETA_CONCENTRATION = 9.155 + 1.275
DEFAULT_TARGET_MEAN_PS = 9.155 / (9.155 + 1.275)
DEFAULT_TARGET_VOLUME_FRACTION = 0.40
DEFAULT_MATRIX_MU = 353.5
DEFAULT_MATRIX_ALPHA = -21.5
DEFAULT_MATRIX_D = 0.0
DEFAULT_FIBER_MU = 80.8
DEFAULT_FIBER_ALPHA = 62.3
DEFAULT_SEED = 42
DEFAULT_CPUS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Abaqus cases, extract homogenized CSVs, refit fixed-matrix "
            "modified-HGO parameters, and rebuild the LHS campaign dataset."
        )
    )
    parser.add_argument("--cases-csv", default=str(USER_CASES_CSV))
    parser.add_argument("--runs-dir", default=str(USER_RUNS_DIR))
    parser.add_argument("--dataset-csv", default=str(USER_DATASET_CSV))
    parser.add_argument("--dataset-jsonl", default=str(USER_DATASET_JSONL))
    parser.add_argument("--abaqus-cmd", default=USER_ABAQUS_CMD)
    parser.add_argument("--default-cpus", type=int, default=USER_DEFAULT_CPUS)
    parser.add_argument("--case-start", type=int, default=USER_CASE_START)
    parser.add_argument("--case-count", type=int, default=USER_CASE_COUNT)
    parser.add_argument("--skip-existing", action="store_true", default=USER_SKIP_EXISTING)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--continue-on-error", action="store_true", default=USER_CONTINUE_ON_ERROR)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=USER_DRY_RUN)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")

    parser.add_argument("--rve-edge-um", type=float, default=USER_RVE_EDGE_UM)
    parser.add_argument("--hex-element-length-um", type=float, default=USER_HEX_ELEMENT_LENGTH_UM)
    parser.add_argument("--truss-element-length-um", type=float, default=USER_TRUSS_ELEMENT_LENGTH_UM)
    parser.add_argument("--matrix-element-type", default=USER_MATRIX_ELEMENT_TYPE)
    parser.add_argument("--target-stretch", type=float, default=USER_TARGET_STRETCH)
    parser.add_argument("--target-shear-gamma", type=float, default=USER_TARGET_SHEAR_GAMMA)
    return parser.parse_args()


def sanitize_name(value):
    safe = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in ("_", "-", "."):
            safe.append(ch)
        else:
            safe.append("_")
    out = "".join(safe).strip("._")
    return out or "case"


def parse_optional_float(row, key, default=None):
    raw = row.get(key, "")
    if raw is None:
        return default
    raw = str(raw).strip()
    if raw == "":
        return default
    return float(raw)


def parse_optional_int(row, key, default=None):
    raw = row.get(key, "")
    if raw is None:
        return default
    raw = str(raw).strip()
    if raw == "":
        return default
    return int(float(raw))


def parse_optional_bool(row, key, default=False):
    raw = row.get(key, "")
    if raw is None:
        return default
    raw = str(raw).strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "y", "on")


def read_case_rows(cases_csv):
    with open(cases_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        raise RuntimeError("No case rows found in: %s" % cases_csv)
    return rows


def write_dataset_csv(rows, path):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_dataset_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def resolve_abaqus_command(cli_value):
    candidates = [
        cli_value,
        os.environ.get("ABAQUS_COMMAND"),
        shutil.which("abaqus"),
        shutil.which("abq2024"),
        shutil.which("abq2023"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise FileNotFoundError(
        "Could not locate the Abaqus command. Pass --abaqus-cmd or set ABAQUS_COMMAND."
    )


def run_command(cmd, cwd=None, env=None, dry_run=False):
    print("")
    print("Command:")
    print("  " + " ".join(str(part) for part in cmd))
    if dry_run:
        return
    if os.name == "nt":
        subprocess.run(
            subprocess.list2cmdline([str(part) for part in cmd]),
            cwd=cwd,
            env=env,
            shell=True,
            check=True,
        )
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def kappa_to_fixed_theta_deg(kappa):
    kappa = float(kappa)
    if kappa < 0.0 or kappa > (1.0 / 3.0 + 1.0e-12):
        raise ValueError("target_kappa must lie in [0, 1/3], got %.6f" % kappa)
    theta_rad = math.asin(math.sqrt(max(0.0, min(1.0, 2.0 * kappa))))
    return math.degrees(theta_rad)


def build_generator_config(case_row: dict, case_dir: Path, case_id: str, args: argparse.Namespace) -> dict:
    generator_variant = (case_row.get("generator_variant") or "straight").strip().lower()
    if generator_variant != "straight":
        raise ValueError(
            "This campaign runner currently supports only generator_variant='straight'. "
            "Got '%s' for case '%s'." % (generator_variant, case_id)
        )

    target_vf = parse_optional_float(case_row, "target_volume_fraction", DEFAULT_TARGET_VOLUME_FRACTION)
    target_mean_diameter = parse_optional_float(case_row, "target_mean_diameter_um", None)
    if target_mean_diameter is None:
        raise ValueError("target_mean_diameter_um is required for case '%s'." % case_id)

    target_mean_ps = parse_optional_float(case_row, "target_mean_ps", DEFAULT_TARGET_MEAN_PS)
    if not (0.0 < target_mean_ps < 1.0):
        raise ValueError("target_mean_ps must lie in (0, 1).")

    target_kappa = parse_optional_float(case_row, "target_kappa", 0.0)
    theta_deg = kappa_to_fixed_theta_deg(target_kappa)

    gamma_shape_k = parse_optional_float(case_row, "gamma_shape_k", DEFAULT_DIAMETER_SHAPE_K)
    if gamma_shape_k <= 0.0:
        raise ValueError("gamma_shape_k must be positive.")

    ps_beta_concentration = parse_optional_float(
        case_row,
        "ps_beta_concentration",
        DEFAULT_PS_BETA_CONCENTRATION,
    )
    if ps_beta_concentration <= 0.0:
        raise ValueError("ps_beta_concentration must be positive.")

    matrix_mu = parse_optional_float(case_row, "matrix_mu", DEFAULT_MATRIX_MU)
    matrix_alpha = parse_optional_float(case_row, "matrix_alpha", DEFAULT_MATRIX_ALPHA)
    matrix_d = parse_optional_float(case_row, "matrix_d", DEFAULT_MATRIX_D)
    fiber_mu = parse_optional_float(case_row, "fiber_mu", DEFAULT_FIBER_MU)
    fiber_alpha = parse_optional_float(case_row, "fiber_alpha", DEFAULT_FIBER_ALPHA)

    edge = float(args.rve_edge_um)
    uniaxial_displacement = edge * (float(args.target_stretch) - 1.0)
    pure_shear_displacement = edge * float(args.target_shear_gamma)

    job_name = sanitize_name(case_id.upper())
    model_name = sanitize_name("MODEL_" + case_id.upper())

    return {
        "OUTPUT_DIR": str(case_dir),
        "RVE_INP": "%s.inp" % sanitize_name(case_id.lower()),
        "SUMMARY_JSON": "rve_truss_generation_summary_%s.json" % sanitize_name(case_id.lower()),
        "JOB_NAME": job_name,
        "MODEL_NAME": model_name,
        "CUBE_MIN": [0.0, 0.0, 0.0],
        "CUBE_MAX": [edge, edge, edge],
        "TARGET_VOLUME_FRACTION": target_vf,
        "TARGET_MEAN_DIAMETER_UM": target_mean_diameter,
        "SEED": parse_optional_int(case_row, "seed", DEFAULT_SEED),
        "GAMMA_SHAPE_K": gamma_shape_k,
        "GAMMA_SCALE_S": target_mean_diameter / gamma_shape_k,
        "STRAIGHTNESS_BETA_A": target_mean_ps * ps_beta_concentration,
        "STRAIGHTNESS_BETA_B": (1.0 - target_mean_ps) * ps_beta_concentration,
        "FIXED_THETA_DEG": theta_deg,
        "MATRIX_OGDEN_TERMS": [[matrix_mu, matrix_alpha, matrix_d]],
        "AXON_USER_PROPERTIES": [fiber_mu, fiber_alpha],
        "TRUSS_ELEMENT_LENGTH": float(args.truss_element_length_um),
        "HEX_ELEMENT_LENGTH": float(args.hex_element_length_um),
        "MATRIX_ELEMENT_TYPE": str(args.matrix_element_type),
        "UNIAXIAL_TENSION_DISPLACEMENT": uniaxial_displacement,
        "PURE_SHEAR_DISPLACEMENT": pure_shear_displacement,
        "WRITE_FIX_CENTER_BC": True,
        "WRITE_PBC_EQUATIONS": True,
        "MAX_PLANNED_FIBERS": 12000,
    }


def write_case_completion_marker(path: Path, case_id: str, summary_json: Path) -> None:
    payload = {
        "case_id": case_id,
        "status": "success",
        "summary_json": str(summary_json),
        "extraction_complete": True,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_case_failure_marker(path: Path, case_id: str, case_dir: Path, error_message: str) -> None:
    payload = {
        "case_id": case_id,
        "status": "failed",
        "case_dir": str(case_dir),
        "error": str(error_message),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_failed_cases_csv(records: List[dict], path: Path) -> None:
    fieldnames = ["table_row", "case_id", "case_dir", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def write_failed_cases_jsonl(records: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def process_case(
    case_row: dict,
    table_row_index: int,
    args: argparse.Namespace,
    abaqus_cmd: str,
) -> dict:
    case_id = sanitize_name(case_row.get("case_id") or ("case_%04d" % table_row_index))
    case_dir = Path(args.runs_dir).resolve() / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    completion_marker = case_dir / CASE_COMPLETION_FILE
    failure_marker = case_dir / CASE_FAILURE_FILE
    if args.skip_existing and completion_marker.exists():
        return {
            "case_id": case_id,
            "case_dir": str(case_dir),
            "table_row_index": table_row_index,
            "status": "skipped_existing",
        }

    if failure_marker.exists():
        failure_marker.unlink()
    if completion_marker.exists():
        completion_marker.unlink()

    generator_config = build_generator_config(case_row, case_dir, case_id, args)
    config_path = case_dir / "case_generator_config.json"
    config_path.write_text(json.dumps(generator_config, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["BRAIN_WM_RVE_CONFIG_JSON"] = str(config_path)

    run_command(
        [sys.executable, str(STRAIGHT_GENERATOR)],
        cwd=str(PROJECT_DIR),
        env=env,
        dry_run=args.dry_run,
    )

    summary_json = case_dir / generator_config["SUMMARY_JSON"]
    cpus = parse_optional_int(case_row, "cpus", args.default_cpus)
    batch_cmd = [
        sys.executable,
        str(RUN_BATCH_SCRIPT),
        "--inp-dir",
        str(case_dir),
        "--summary-json",
        str(summary_json),
        "--user-subroutine",
        str(FORTRAN_TEMPLATE),
        "--cpus",
        str(cpus),
    ]
    if args.skip_existing:
        batch_cmd.append("--skip-existing")
    if args.abaqus_cmd:
        batch_cmd.extend(["--abaqus-cmd", abaqus_cmd])
    if parse_optional_bool(case_row, "double_precision", False):
        batch_cmd.append("--double")
    if args.continue_on_error:
        batch_cmd.append("--continue-on-error")
    run_command(batch_cmd, cwd=str(PROJECT_DIR), dry_run=args.dry_run)

    extract_cmd = [
        abaqus_cmd,
        "python",
        str(EXTRACT_SCRIPT),
        "--data-dir",
        str(case_dir),
        "--summary-json",
        str(summary_json),
    ]
    run_command(extract_cmd, cwd=str(PROJECT_DIR), dry_run=args.dry_run)

    if not args.dry_run:
        write_case_completion_marker(completion_marker, case_id, summary_json)

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "table_row_index": table_row_index,
        "status": "success" if not args.dry_run else "dry_run",
    }


def run_refit_for_case(table_row_index: int, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REFIT_SCRIPT),
        "--cases-csv",
        str(Path(args.cases_csv).resolve()),
        "--runs-dir",
        str(Path(args.runs_dir).resolve()),
        "--case-start",
        str(table_row_index),
        "--case-count",
        "1",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    else:
        cmd.append("--no-skip-existing")
    run_command(cmd, cwd=str(PROJECT_DIR), dry_run=args.dry_run)


def rebuild_dataset(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REBUILD_SCRIPT),
        "--cases-csv",
        str(Path(args.cases_csv).resolve()),
        "--runs-dir",
        str(Path(args.runs_dir).resolve()),
        "--output-csv",
        str(Path(args.dataset_csv).resolve()),
        "--output-jsonl",
        str(Path(args.dataset_jsonl).resolve()),
    ]
    run_command(cmd, cwd=str(PROJECT_DIR), dry_run=args.dry_run)


def main():
    args = parse_args()
    cases_csv = Path(args.cases_csv).resolve()
    if not cases_csv.exists():
        raise FileNotFoundError("Cases CSV not found: %s" % cases_csv)
    if args.case_start < 1:
        raise ValueError("--case-start must be at least 1.")
    if args.case_count is not None and args.case_count < 1:
        raise ValueError("--case-count must be at least 1 when provided.")

    Path(args.runs_dir).resolve().mkdir(parents=True, exist_ok=True)
    abaqus_cmd = resolve_abaqus_command(args.abaqus_cmd)
    all_case_rows = read_case_rows(str(cases_csv))

    start_idx = args.case_start - 1
    if start_idx >= len(all_case_rows):
        raise ValueError(
            "--case-start=%d is beyond the available %d case rows."
            % (args.case_start, len(all_case_rows))
        )
    end_idx = len(all_case_rows)
    if args.case_count is not None:
        end_idx = min(len(all_case_rows), start_idx + args.case_count)
    selected_case_records = list(enumerate(all_case_rows, start=1))[start_idx:end_idx]

    failures = []
    successes = []
    failed_cases_csv = Path(args.runs_dir).resolve() / "failed_case_ids.csv"
    failed_cases_jsonl = Path(args.runs_dir).resolve() / "failed_case_details.jsonl"
    for stale_path in (failed_cases_csv, failed_cases_jsonl):
        if stale_path.exists():
            stale_path.unlink()

    print("")
    print(
        "Selected case rows %d-%d of %d total."
        % (selected_case_records[0][0], selected_case_records[-1][0], len(all_case_rows))
    )
    print(
        "Campaign geometry: RVE edge = %.3f um, hex = %.3f um, truss = %.3f um"
        % (args.rve_edge_um, args.hex_element_length_um, args.truss_element_length_um)
    )

    for batch_idx, (global_idx, case_row) in enumerate(selected_case_records, start=1):
        case_id = sanitize_name(case_row.get("case_id") or "case_%04d" % global_idx)
        case_dir = Path(args.runs_dir).resolve() / case_id
        print("")
        print("============================================================")
        print(
            "Processing case %d/%d (table row %d): %s"
            % (batch_idx, len(selected_case_records), global_idx, case_id)
        )
        print("============================================================")
        try:
            result = process_case(case_row, global_idx, args, abaqus_cmd)
            successes.append(result)
        except Exception as exc:
            failure_record = {
                "table_row": global_idx,
                "case_id": case_id,
                "case_dir": str(case_dir),
                "error": str(exc),
            }
            failures.append(failure_record)
            write_case_failure_marker(case_dir / CASE_FAILURE_FILE, case_id, case_dir, exc)
            write_failed_cases_csv(failures, failed_cases_csv)
            write_failed_cases_jsonl(failures, failed_cases_jsonl)
            print("")
            print("Case failed: %s" % case_id)
            print(str(exc))
            if not args.continue_on_error:
                raise

    if not args.dry_run:
        completed_successes = [item for item in successes if item["status"] in ("success", "skipped_existing")]
        for item in completed_successes:
            run_refit_for_case(item["table_row_index"], args)
        rebuild_dataset(args)

    if failures:
        print("")
        print("Completed with failures:")
        for record in failures:
            print("  row %d / %s: %s" % (record["table_row"], record["case_id"], record["error"]))
        print("Failed case CSV  : %s" % failed_cases_csv)
        print("Failed case JSONL: %s" % failed_cases_jsonl)
    else:
        print("")
        print("All selected cases completed successfully.")

    print("")
    print("Dataset CSV : %s" % Path(args.dataset_csv).resolve())
    print("Dataset JSONL: %s" % Path(args.dataset_jsonl).resolve())


if __name__ == "__main__":
    main()
