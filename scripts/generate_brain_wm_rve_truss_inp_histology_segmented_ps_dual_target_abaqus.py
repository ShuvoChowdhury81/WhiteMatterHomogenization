#!/usr/bin/env python3
"""Dual-target straight-fiber generator with legacy Abaqus INP export hooks.

This script bridges the verified dual-target planning logic into the original
RVE verification pipeline by:
- reusing the current dual-target morphology generator,
- converting the packed parallel fibers into legacy ``Axon`` records, and
- calling the original INP export / summary helpers so downstream Abaqus and
  extraction scripts continue to work unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_ENV_VAR = "BRAIN_WM_RVE_CONFIG_JSON"

CORE_DUAL_TARGET_PY = SCRIPT_DIR / "generate_brain_wm_rve_truss_inp_histology_segmented_ps_dual_target.py"
LEGACY_EXPORT_PY = SCRIPT_DIR / "generate_brain_wm_rve_truss_inp_histology_segmented_ps.py"


# Defaults; these are overridden through BRAIN_WM_RVE_CONFIG_JSON during runs.
CUBE_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
CUBE_MAX = np.array([20.0, 20.0, 20.0], dtype=float)
TARGET_VOLUME_FRACTION = 0.40
TARGET_MEAN_DIAMETER_UM = 0.50
SEED = 42
MIN_GAP_UM = 0.02

GAMMA_SHAPE_K = 1.5492
GAMMA_SCALE_S = 0.4561
DIAMETER_MIN_UM = 0.12
DIAMETER_MAX_UM = 1.8

STRAIGHTNESS_BETA_A = 9.155
STRAIGHTNESS_BETA_B = 1.275
FIXED_STRAIGHTNESS_PS = None
THETA_RATE_PER_DEG = 0.076
THETA_MAX_DEG = 0.0
PHI_MODE = "uniform"
FIXED_THETA_DEG = 0.0
FIXED_PHI_DEG = None
ORIENTATION_RETRY_MODE = "contract_theta"
ORIENTATION_RETRY_STAGES = [1.0, 0.75, 0.50, 0.30, 0.15]
ANCHOR_SAMPLING_MODE = "uniform_cube"
ANCHOR_TRIALS_PER_STAGE = 1
PLACEMENT_CANDIDATES_PER_ATTEMPT = 1
USE_SPATIAL_HASH = True
SPATIAL_HASH_CELL_SIZE = 2.0
MAX_ATTEMPTS_TOTAL = 300000
MAX_FAILED_TRIES_PER_AXON = 500
ALLOW_DIAMETER_RESCUE = False
DIAMETER_RESCUE_SCALE = 0.90

TRUSS_ELEMENT_LENGTH = 1.0
HEX_ELEMENT_LENGTH = 1.0
MATRIX_ELEMENT_TYPE = "C3D8RH"
MATRIX_OGDEN_TERMS = [(353.5, -21.5, 0.0)]
AXON_USER_PROPERTIES = [80.8, 62.3]

OUTPUT_DIR = str(SCRIPT_DIR / "outputs")
RVE_INP = "brain_wm_rve_truss_matrix_with_ps.inp"
SUMMARY_JSON = "rve_truss_generation_summary_dual_target.json"
JOB_NAME = "JOB-1"
MODEL_NAME = "BRAIN_WM_RVE_TRUSS_MATRIX_DUAL_TARGET"
UNIAXIAL_TENSION_DISPLACEMENT = 20.0
PURE_SHEAR_DISPLACEMENT = 10.0
WRITE_FIX_CENTER_BC = True
WRITE_PBC_EQUATIONS = True


def load_runtime_config() -> Dict[str, object]:
    config_path = os.environ.get(CONFIG_ENV_VAR)
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_runtime_overrides(config: Dict[str, object]) -> None:
    global CUBE_MIN, CUBE_MAX, TARGET_VOLUME_FRACTION, TARGET_MEAN_DIAMETER_UM
    global SEED, MIN_GAP_UM, GAMMA_SHAPE_K, GAMMA_SCALE_S
    global DIAMETER_MIN_UM, DIAMETER_MAX_UM
    global STRAIGHTNESS_BETA_A, STRAIGHTNESS_BETA_B, FIXED_STRAIGHTNESS_PS
    global THETA_RATE_PER_DEG, THETA_MAX_DEG, PHI_MODE, FIXED_THETA_DEG, FIXED_PHI_DEG
    global ORIENTATION_RETRY_MODE
    global ANCHOR_SAMPLING_MODE, ANCHOR_TRIALS_PER_STAGE
    global PLACEMENT_CANDIDATES_PER_ATTEMPT, USE_SPATIAL_HASH, SPATIAL_HASH_CELL_SIZE
    global ORIENTATION_RETRY_STAGES, MAX_ATTEMPTS_TOTAL, MAX_FAILED_TRIES_PER_AXON
    global ALLOW_DIAMETER_RESCUE, DIAMETER_RESCUE_SCALE
    global TRUSS_ELEMENT_LENGTH, HEX_ELEMENT_LENGTH, MATRIX_ELEMENT_TYPE
    global MATRIX_OGDEN_TERMS, AXON_USER_PROPERTIES
    global OUTPUT_DIR, RVE_INP, SUMMARY_JSON, JOB_NAME, MODEL_NAME
    global UNIAXIAL_TENSION_DISPLACEMENT, PURE_SHEAR_DISPLACEMENT
    global WRITE_FIX_CENTER_BC, WRITE_PBC_EQUATIONS

    if "CUBE_MIN" in config:
        CUBE_MIN = np.array(config["CUBE_MIN"], dtype=float)
    if "CUBE_MAX" in config:
        CUBE_MAX = np.array(config["CUBE_MAX"], dtype=float)
    if "TARGET_VOLUME_FRACTION" in config:
        TARGET_VOLUME_FRACTION = float(config["TARGET_VOLUME_FRACTION"])
    if "TARGET_MEAN_DIAMETER_UM" in config:
        TARGET_MEAN_DIAMETER_UM = float(config["TARGET_MEAN_DIAMETER_UM"])
    if "SEED" in config:
        SEED = int(config["SEED"])

    if "MIN_GAP_UM" in config:
        MIN_GAP_UM = float(config["MIN_GAP_UM"])
    elif "MIN_GAP" in config:
        MIN_GAP_UM = float(config["MIN_GAP"])

    if "GAMMA_SHAPE_K" in config:
        GAMMA_SHAPE_K = float(config["GAMMA_SHAPE_K"])
    if "GAMMA_SCALE_S" in config:
        GAMMA_SCALE_S = float(config["GAMMA_SCALE_S"])

    if "DIAMETER_MIN_UM" in config:
        DIAMETER_MIN_UM = float(config["DIAMETER_MIN_UM"])
    elif "DIAMETER_MIN" in config:
        DIAMETER_MIN_UM = float(config["DIAMETER_MIN"])

    if "DIAMETER_MAX_UM" in config:
        DIAMETER_MAX_UM = float(config["DIAMETER_MAX_UM"])
    elif "DIAMETER_MAX" in config:
        DIAMETER_MAX_UM = float(config["DIAMETER_MAX"])

    if "STRAIGHTNESS_BETA_A" in config:
        STRAIGHTNESS_BETA_A = float(config["STRAIGHTNESS_BETA_A"])
    if "STRAIGHTNESS_BETA_B" in config:
        STRAIGHTNESS_BETA_B = float(config["STRAIGHTNESS_BETA_B"])
    if "FIXED_STRAIGHTNESS_PS" in config:
        raw_value = config["FIXED_STRAIGHTNESS_PS"]
        FIXED_STRAIGHTNESS_PS = None if raw_value is None else float(raw_value)
    if "THETA_RATE_PER_DEG" in config:
        THETA_RATE_PER_DEG = float(config["THETA_RATE_PER_DEG"])
    if "THETA_MAX_DEG" in config:
        THETA_MAX_DEG = float(config["THETA_MAX_DEG"])
    if "PHI_MODE" in config:
        PHI_MODE = str(config["PHI_MODE"])
    if "FIXED_THETA_DEG" in config:
        raw_value = config["FIXED_THETA_DEG"]
        FIXED_THETA_DEG = None if raw_value is None else float(raw_value)
    if "FIXED_PHI_DEG" in config:
        raw_value = config["FIXED_PHI_DEG"]
        FIXED_PHI_DEG = None if raw_value is None else float(raw_value)
    if "ORIENTATION_RETRY_MODE" in config:
        ORIENTATION_RETRY_MODE = str(config["ORIENTATION_RETRY_MODE"])
    if "ANCHOR_SAMPLING_MODE" in config:
        ANCHOR_SAMPLING_MODE = str(config["ANCHOR_SAMPLING_MODE"])
    if "ANCHOR_TRIALS_PER_STAGE" in config:
        ANCHOR_TRIALS_PER_STAGE = int(config["ANCHOR_TRIALS_PER_STAGE"])
    if "PLACEMENT_CANDIDATES_PER_ATTEMPT" in config:
        PLACEMENT_CANDIDATES_PER_ATTEMPT = int(config["PLACEMENT_CANDIDATES_PER_ATTEMPT"])
    if "USE_SPATIAL_HASH" in config:
        USE_SPATIAL_HASH = bool(config["USE_SPATIAL_HASH"])
    if "SPATIAL_HASH_CELL_SIZE" in config:
        SPATIAL_HASH_CELL_SIZE = float(config["SPATIAL_HASH_CELL_SIZE"])
    if "ORIENTATION_RETRY_STAGES" in config:
        ORIENTATION_RETRY_STAGES = [float(v) for v in config["ORIENTATION_RETRY_STAGES"]]
    if "MAX_ATTEMPTS_TOTAL" in config:
        MAX_ATTEMPTS_TOTAL = int(config["MAX_ATTEMPTS_TOTAL"])
    if "MAX_FAILED_TRIES_PER_AXON" in config:
        MAX_FAILED_TRIES_PER_AXON = int(config["MAX_FAILED_TRIES_PER_AXON"])
    if "ALLOW_DIAMETER_RESCUE" in config:
        ALLOW_DIAMETER_RESCUE = bool(config["ALLOW_DIAMETER_RESCUE"])
    if "DIAMETER_RESCUE_SCALE" in config:
        DIAMETER_RESCUE_SCALE = float(config["DIAMETER_RESCUE_SCALE"])

    if "TRUSS_ELEMENT_LENGTH" in config:
        TRUSS_ELEMENT_LENGTH = float(config["TRUSS_ELEMENT_LENGTH"])
    if "HEX_ELEMENT_LENGTH" in config:
        HEX_ELEMENT_LENGTH = float(config["HEX_ELEMENT_LENGTH"])
    if "MATRIX_ELEMENT_TYPE" in config:
        MATRIX_ELEMENT_TYPE = str(config["MATRIX_ELEMENT_TYPE"])
    if "MATRIX_OGDEN_TERMS" in config:
        MATRIX_OGDEN_TERMS = [tuple(float(v) for v in term) for term in config["MATRIX_OGDEN_TERMS"]]
    if "AXON_USER_PROPERTIES" in config:
        AXON_USER_PROPERTIES = [float(v) for v in config["AXON_USER_PROPERTIES"]]

    if "OUTPUT_DIR" in config:
        OUTPUT_DIR = str(config["OUTPUT_DIR"])
    if "RVE_INP" in config:
        RVE_INP = str(config["RVE_INP"])
    if "SUMMARY_JSON" in config:
        SUMMARY_JSON = str(config["SUMMARY_JSON"])
    if "JOB_NAME" in config:
        JOB_NAME = str(config["JOB_NAME"])
    if "MODEL_NAME" in config:
        MODEL_NAME = str(config["MODEL_NAME"])

    if "UNIAXIAL_TENSION_DISPLACEMENT" in config:
        UNIAXIAL_TENSION_DISPLACEMENT = float(config["UNIAXIAL_TENSION_DISPLACEMENT"])
    if "PURE_SHEAR_DISPLACEMENT" in config:
        PURE_SHEAR_DISPLACEMENT = float(config["PURE_SHEAR_DISPLACEMENT"])
    if "WRITE_FIX_CENTER_BC" in config:
        WRITE_FIX_CENTER_BC = bool(config["WRITE_FIX_CENTER_BC"])
    if "WRITE_PBC_EQUATIONS" in config:
        WRITE_PBC_EQUATIONS = bool(config["WRITE_PBC_EQUATIONS"])


def load_module(module_name: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def sync_legacy_globals(legacy: ModuleType) -> None:
    legacy.CUBE_MIN = np.array(CUBE_MIN, dtype=float)
    legacy.CUBE_MAX = np.array(CUBE_MAX, dtype=float)
    legacy.CUBE_SIZE = legacy.CUBE_MAX - legacy.CUBE_MIN
    legacy.CUBE_CENTER = 0.5 * (legacy.CUBE_MIN + legacy.CUBE_MAX)

    legacy.TARGET_VOLUME_FRACTION = float(TARGET_VOLUME_FRACTION)
    legacy.SEED = int(SEED)
    legacy.MIN_GAP = float(MIN_GAP_UM)

    legacy.GAMMA_SHAPE_K = float(GAMMA_SHAPE_K)
    legacy.GAMMA_SCALE_S = float(GAMMA_SCALE_S)
    legacy.DIAMETER_MIN = float(DIAMETER_MIN_UM)
    legacy.DIAMETER_MAX = float(DIAMETER_MAX_UM)

    legacy.STRAIGHTNESS_BETA_A = float(STRAIGHTNESS_BETA_A)
    legacy.STRAIGHTNESS_BETA_B = float(STRAIGHTNESS_BETA_B)
    legacy.FIXED_STRAIGHTNESS_PS = (
        None if FIXED_STRAIGHTNESS_PS is None else float(FIXED_STRAIGHTNESS_PS)
    )
    legacy.THETA_RATE_PER_DEG = float(THETA_RATE_PER_DEG)
    legacy.THETA_MAX_DEG = float(THETA_MAX_DEG)
    legacy.PHI_MODE = str(PHI_MODE)
    legacy.FIXED_THETA_DEG = None if FIXED_THETA_DEG is None else float(FIXED_THETA_DEG)
    legacy.FIXED_PHI_DEG = None if FIXED_PHI_DEG is None else float(FIXED_PHI_DEG)
    legacy.ORIENTATION_RETRY_MODE = str(ORIENTATION_RETRY_MODE)
    legacy.ANCHOR_SAMPLING_MODE = str(ANCHOR_SAMPLING_MODE)
    legacy.ANCHOR_TRIALS_PER_STAGE = int(ANCHOR_TRIALS_PER_STAGE)
    legacy.PLACEMENT_CANDIDATES_PER_ATTEMPT = int(PLACEMENT_CANDIDATES_PER_ATTEMPT)
    legacy.USE_SPATIAL_HASH = bool(USE_SPATIAL_HASH)
    legacy.SPATIAL_HASH_CELL_SIZE = float(SPATIAL_HASH_CELL_SIZE)
    legacy.ORIENTATION_RETRY_STAGES = [float(v) for v in ORIENTATION_RETRY_STAGES]
    legacy.MAX_ATTEMPTS_TOTAL = int(MAX_ATTEMPTS_TOTAL)
    legacy.MAX_FAILED_TRIES_PER_AXON = int(MAX_FAILED_TRIES_PER_AXON)
    legacy.ALLOW_DIAMETER_RESCUE = bool(ALLOW_DIAMETER_RESCUE)
    legacy.DIAMETER_RESCUE_SCALE = float(DIAMETER_RESCUE_SCALE)

    legacy.TRUSS_ELEMENT_LENGTH = float(TRUSS_ELEMENT_LENGTH)
    legacy.HEX_ELEMENT_LENGTH = float(HEX_ELEMENT_LENGTH)
    legacy.MATRIX_ELEMENT_TYPE = str(MATRIX_ELEMENT_TYPE)
    legacy.MATRIX_OGDEN_TERMS = [tuple(term) for term in MATRIX_OGDEN_TERMS]
    legacy.AXON_USER_PROPERTIES = list(AXON_USER_PROPERTIES)

    legacy.OUTPUT_DIR = str(OUTPUT_DIR)
    legacy.RVE_INP = str(RVE_INP)
    legacy.RVE_INP_STEM = os.path.splitext(legacy.RVE_INP)[0]
    legacy.SUMMARY_JSON = str(SUMMARY_JSON)
    legacy.JOB_NAME = str(JOB_NAME)
    legacy.MODEL_NAME = str(MODEL_NAME)

    legacy.UNIAXIAL_TENSION_DISPLACEMENT = float(UNIAXIAL_TENSION_DISPLACEMENT)
    legacy.PURE_SHEAR_DISPLACEMENT = float(PURE_SHEAR_DISPLACEMENT)
    legacy.WRITE_FIX_CENTER_BC = bool(WRITE_FIX_CENTER_BC)
    legacy.WRITE_PBC_EQUATIONS = bool(WRITE_PBC_EQUATIONS)
    legacy.ACTIVE_CONFIG_PATH = os.environ.get(CONFIG_ENV_VAR)


def install_stabilized_pure_shear_loading_cases(legacy: ModuleType) -> None:
    original_builder = legacy.build_loading_case_definitions

    def build_loading_case_definitions():
        cases = list(original_builder())
        rp1 = f"{legacy.REFPOINT1_INSTANCE_NAME}.{legacy.REFPOINT1_SET_NAME}"
        rp2 = f"{legacy.REFPOINT2_INSTANCE_NAME}.{legacy.REFPOINT2_SET_NAME}"
        rp3 = f"{legacy.REFPOINT3_INSTANCE_NAME}.{legacy.REFPOINT3_SET_NAME}"

        stabilized_blocks = {
            "PURE_SHEAR_XY": [
                (
                    "BC-2",
                    [
                        legacy.boundary_entry(rp1, 2, legacy.PURE_SHEAR_DISPLACEMENT),
                        legacy.boundary_entry(rp1, 3),
                    ],
                ),
                (
                    "BC-3",
                    [
                        legacy.boundary_entry(rp2, 1, legacy.PURE_SHEAR_DISPLACEMENT),
                        legacy.boundary_entry(rp2, 3),
                    ],
                ),
                (
                    "BC-4",
                    [
                        legacy.boundary_entry(rp3, 1),
                        legacy.boundary_entry(rp3, 2),
                    ],
                ),
            ],
            "PURE_SHEAR_YZ": [
                (
                    "BC-2",
                    [
                        legacy.boundary_entry(rp1, 2),
                        legacy.boundary_entry(rp1, 3),
                    ],
                ),
                (
                    "BC-3",
                    [
                        legacy.boundary_entry(rp2, 1),
                        legacy.boundary_entry(rp2, 3, legacy.PURE_SHEAR_DISPLACEMENT),
                    ],
                ),
                (
                    "BC-4",
                    [
                        legacy.boundary_entry(rp3, 1),
                        legacy.boundary_entry(rp3, 2, legacy.PURE_SHEAR_DISPLACEMENT),
                    ],
                ),
            ],
            "PURE_SHEAR_XZ": [
                (
                    "BC-2",
                    [
                        legacy.boundary_entry(rp1, 2),
                        legacy.boundary_entry(rp1, 3, legacy.PURE_SHEAR_DISPLACEMENT),
                    ],
                ),
                (
                    "BC-3",
                    [
                        legacy.boundary_entry(rp2, 1),
                        legacy.boundary_entry(rp2, 3),
                    ],
                ),
                (
                    "BC-4",
                    [
                        legacy.boundary_entry(rp3, 1, legacy.PURE_SHEAR_DISPLACEMENT),
                        legacy.boundary_entry(rp3, 2),
                    ],
                ),
            ],
        }

        updated_cases = []
        for case in cases:
            case_key = case.get("key")
            if case_key == "UNIAXIAL_Z":
                continue
            if case_key in stabilized_blocks:
                case = dict(case)
                case["boundary_blocks"] = stabilized_blocks[case_key]
            updated_cases.append(case)
        return updated_cases

    legacy.build_loading_case_definitions = build_loading_case_definitions


def packed_to_legacy_axons(
    packed: List[object],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    legacy: ModuleType,
) -> List[object]:
    edge = float(bounds_max[0] - bounds_min[0])
    x_mid = float(0.5 * (bounds_min[0] + bounds_max[0]))
    direction = np.array([1.0, 0.0, 0.0], dtype=float)

    axons: List[object] = []
    for fiber in packed:
        y_um = float(fiber.y_um)
        z_um = float(fiber.z_um)
        p1 = np.array([float(bounds_min[0]), y_um, z_um], dtype=float)
        p2 = np.array([float(bounds_max[0]), y_um, z_um], dtype=float)
        anchor = np.array([x_mid, y_um, z_um], dtype=float)
        straightness_ps = float(fiber.straightness_ps)
        tortuosity_tau = 1.0 / max(straightness_ps, 1.0e-12)

        axons.append(
            legacy.Axon(
                anchor=anchor,
                direction=direction.copy(),
                radius=float(fiber.radius_um),
                diameter=float(fiber.diameter_um),
                theta_deg=0.0,
                phi_deg=0.0,
                straightness_ps=straightness_ps,
                tortuosity_tau=tortuosity_tau,
                p1=p1,
                p2=p2,
                length_inside=edge,
                arc_length=float(fiber.arc_length_um),
                approx_volume=float(fiber.approx_volume_um3),
            )
        )
    return axons


def export_case(
    legacy: ModuleType,
    axons: List[object],
    stats: dict,
) -> Tuple[Path, List[str]]:
    (
        generated_paths,
        total_matrix_nodes,
        total_matrix_elements,
        total_truss_nodes,
        total_truss_elements,
        matrix_shape,
        center_node_id,
        pbc_pair_count,
        generated_case_records,
    ) = legacy.export_loading_case_inps(
        np.array(CUBE_MIN, dtype=float),
        np.array(CUBE_MAX, dtype=float),
        axons,
        str(OUTPUT_DIR),
    )

    summary_path = legacy.save_summary(
        axons=axons,
        stats=stats,
        out_dir=str(OUTPUT_DIR),
        total_matrix_nodes=total_matrix_nodes,
        total_matrix_elements=total_matrix_elements,
        total_truss_nodes=total_truss_nodes,
        total_truss_elements=total_truss_elements,
        matrix_shape=matrix_shape,
        center_node_id=center_node_id,
        pbc_pair_count=pbc_pair_count,
        generated_paths=generated_paths,
        generated_case_records=generated_case_records,
    )
    return Path(summary_path), [str(path) for path in generated_paths]


def main() -> None:
    config = load_runtime_config()
    apply_runtime_overrides(config)

    core = load_module("dual_target_core_generation_module", CORE_DUAL_TARGET_PY)
    core.apply_runtime_overrides(config)

    legacy = load_module("dual_target_legacy_export_module", LEGACY_EXPORT_PY)
    sync_legacy_globals(legacy)
    install_stabilized_pure_shear_loading_cases(legacy)

    packed, stats = core.evaluate_case(
        bounds_min=np.array(CUBE_MIN, dtype=float),
        bounds_max=np.array(CUBE_MAX, dtype=float),
        target_vf=float(TARGET_VOLUME_FRACTION),
        target_mean_diameter_um=float(TARGET_MEAN_DIAMETER_UM),
        seed=int(SEED),
    )

    axons = packed_to_legacy_axons(
        packed=packed,
        bounds_min=np.array(CUBE_MIN, dtype=float),
        bounds_max=np.array(CUBE_MAX, dtype=float),
        legacy=legacy,
    )

    summary_path, generated_inp_paths = export_case(
        legacy=legacy,
        axons=axons,
        stats=stats,
    )

    print("\nSummary")
    print("-------")
    for key, value in stats.items():
        if key in ("diameter_distribution", "straightness_distribution"):
            continue
        print(f"{key}: {value}")
    print(f"\nSummary JSON : {summary_path}")
    print("Generated INP files:")
    for inp_path in generated_inp_paths:
        print(f"  {inp_path}")


if __name__ == "__main__":
    main()
