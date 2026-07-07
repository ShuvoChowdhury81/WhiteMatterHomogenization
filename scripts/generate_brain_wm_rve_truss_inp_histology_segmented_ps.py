import os
import math
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ============================================================
# USER PARAMETERS
# ============================================================

# Cube: 20 x 20 x 20 um, centered at (10,10,10) -> bounds [0,20] in each direction
CUBE_MIN = np.array([0.0, 0.0, 0.0], dtype=float)
CUBE_MAX = np.array([20.0, 20.0, 20.0], dtype=float)
CUBE_SIZE = CUBE_MAX - CUBE_MIN
CUBE_CENTER = 0.5 * (CUBE_MIN + CUBE_MAX)

TARGET_VOLUME_FRACTION = 0.40
MIN_GAP = 0.02              # extra clearance between axons, in um
SEED = 42
MAX_ATTEMPTS_TOTAL = 300000
MAX_FAILED_TRIES_PER_AXON = 500

# Diameter distribution from the paper (Fig. 3a / Eq. 2 parameters shown in the figure)
GAMMA_SHAPE_K = 1.5492      # shape parameter k
GAMMA_SCALE_S = 0.4561      # scale parameter s, in um
DIAMETER_MIN = 0.12         # lower truncation for robustness, um
DIAMETER_MAX = 1.8          # upper truncation for robustness, um

# Straightness distribution from the paper: Ps = L0 / Lf
STRAIGHTNESS_BETA_A = 9.155
STRAIGHTNESS_BETA_B = 1.275
FIXED_STRAIGHTNESS_PS = None

# Orientation dispersion around x-axis from the paper (Fig. 3b / exponential CDF)
THETA_RATE_PER_DEG = 0.076  # theta CDF: F(x)=1-exp(-k x), x in degrees
THETA_MAX_DEG = 0           # truncate to keep geometry practical
PHI_MODE = "uniform"       # azimuth about x-axis
FIXED_THETA_DEG = None      # optional direct cone-angle override for kappa-driven studies
FIXED_PHI_DEG = None        # optional fixed azimuth override for exact preferred direction studies

# Adaptive orientation bias: diameter has priority, orientation is softened if packing is hard
ORIENTATION_RETRY_STAGES = [1.0, 0.75, 0.50, 0.30, 0.15]
# Retry mode:
# - "contract_theta": historical behavior, retries pull fibers back toward x
# - "preserve_dispersion": keep sampled theta/phi and use retry stages to relax only extra clearance
ORIENTATION_RETRY_MODE = "contract_theta"

# Anchor-placement strategy:
# - "uniform_cube": historical behavior
# - "mean_axis_midplane": sample anchors on the mid-plane normal to the mean x-axis
#   so candidate fibers tend to span longer chords of the cube
ANCHOR_SAMPLING_MODE = "uniform_cube"
# Number of anchor attempts for a fixed diameter/orientation/straightness candidate
# within each retry stage. Values > 1 help dispersed fibers find longer valid chords.
ANCHOR_TRIALS_PER_STAGE = 1

# Candidate-pool strategy:
# - sample several diameter/orientation/straightness families per outer attempt
# - accept the valid candidate with the highest added volume
PLACEMENT_CANDIDATES_PER_ATTEMPT = 1

# Spatial hash for collision checks:
# - dramatically reduces overlap-check cost for dense dispersed packs
USE_SPATIAL_HASH = True
SPATIAL_HASH_CELL_SIZE = 2.0

# Optional mild diameter rescue rule for the very end of packing
ALLOW_DIAMETER_RESCUE = False
DIAMETER_RESCUE_SCALE = 0.90

# Discretization: keep both truss and solid element sizes at 1 um
TRUSS_ELEMENT_LENGTH = 1.0
HEX_ELEMENT_LENGTH = 1.0
MATRIX_ELEMENT_TYPE = "C3D8RH"

# Output
OUTPUT_DIR = os.path.join(os.getcwd(), "rve_truss_histology_output_ps")
RVE_INP = "brain_wm_rve_truss_matrix_with_ps.inp"
RVE_INP_STEM = os.path.splitext(RVE_INP)[0]
SUMMARY_JSON = "rve_truss_generation_summary_with_ps.json"
JOB_NAME = "JOB-1"
MODEL_NAME = "BRAIN_WM_RVE_TRUSS_MATRIX_WITH_PS"

# Step / loading-case controls
STEP_NAME = "Step-1"
STEP_NLGEOM = "YES"
STEP_MAX_INCREMENTS = 1000
STATIC_STEP_PARAMETERS = (0.05, 1.0, 1e-05, 0.1)
UNIAXIAL_TENSION_DISPLACEMENT = 4.0
PURE_SHEAR_DISPLACEMENT = 2
WRITE_FIX_CENTER_BC = True

# Abaqus names
TRUSS_PART_NAME = "AXONS_NETWORK"
TRUSS_INSTANCE_NAME = "AXONS_INST"
AXON_MATERIAL_NAME = "AXON_USER"
MATRIX_PART_NAME = "MATRIX_BLOCK"
MATRIX_INSTANCE_NAME = "MATRIX_INST"
MATRIX_MATERIAL_NAME = "MATRIX_MAT"
ASSEMBLY_NAME = "ASSEMBLY"
EMBEDDED_CONSTRAINT_NAME = "CONSTRAINT-1"
PS_FIELD_VARIABLE = 1
FIELD_NAME_PREFIX = "FIELD"
FIELD_VALUE_DECIMALS = 6

# Periodic-boundary-condition export controls
WRITE_PBC_EQUATIONS = True
REFERENCE_POINT_OFFSET_UM = 2.0
REFPOINT1_PART_NAME = "RefPoint1"
REFPOINT2_PART_NAME = "RefPoint2"
REFPOINT3_PART_NAME = "RefPoint3"
REFPOINT1_SET_NAME = "SetRefPoint1"
REFPOINT2_SET_NAME = "SetRefPoint2"
REFPOINT3_SET_NAME = "SetRefPoint3"
REFPOINT1_INSTANCE_NAME = "RefPoint1-1"
REFPOINT2_INSTANCE_NAME = "RefPoint2-1"
REFPOINT3_INSTANCE_NAME = "RefPoint3-1"
CENTER_NODE_SET_NAME = "Center_Node"
MATRIX_NODE_SET_PREFIX = "Node-"

# Matrix Ogden material parameters
MATRIX_OGDEN_TERMS = [
    # (MU_i, ALPHA_i, D_i)
    (353.5, -21.5, 0.0),
]
AXON_USER_PROPERTIES = [80.8, 62.3]  # [fiber_mu, fiber_alpha]


CONFIG_ENV_VAR = "BRAIN_WM_RVE_CONFIG_JSON"


def _coerce_runtime_override(name: str, value):
    if name in {"CUBE_MIN", "CUBE_MAX"}:
        return np.array(value, dtype=float)
    if name in {"MATRIX_OGDEN_TERMS"}:
        return [tuple(float(v) for v in term) for term in value]
    if name in {"AXON_USER_PROPERTIES"}:
        return [float(v) for v in value]
    if name in {"STATIC_STEP_PARAMETERS"}:
        return tuple(float(v) for v in value)
    if name in {"ORIENTATION_RETRY_STAGES"}:
        return [float(v) for v in value]
    return value


def apply_runtime_config_overrides() -> Optional[str]:
    config_path = os.environ.get(CONFIG_ENV_VAR)
    if not config_path:
        return None

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    for key, value in config.items():
        if key in globals():
            globals()[key] = _coerce_runtime_override(key, value)

    globals()["CUBE_SIZE"] = CUBE_MAX - CUBE_MIN
    globals()["CUBE_CENTER"] = 0.5 * (CUBE_MIN + CUBE_MAX)
    globals()["RVE_INP_STEM"] = os.path.splitext(RVE_INP)[0]
    return config_path


ACTIVE_CONFIG_PATH = apply_runtime_config_overrides()


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Axon:
    anchor: np.ndarray
    direction: np.ndarray
    radius: float
    diameter: float
    theta_deg: float
    phi_deg: float
    straightness_ps: float
    tortuosity_tau: float
    p1: np.ndarray
    p2: np.ndarray
    length_inside: float
    arc_length: float
    approx_volume: float


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def cube_volume(bounds_min: np.ndarray, bounds_max: np.ndarray) -> float:
    size = bounds_max - bounds_min
    return float(size[0] * size[1] * size[2])


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def append_set_block(lines: List[str], keyword: str, ids: List[int], ids_per_line: int = 16) -> None:
    lines.append(keyword)
    chunk: List[str] = []
    for value in ids:
        chunk.append(str(value))
        if len(chunk) == ids_per_line:
            lines.append(", ".join(chunk) + "\n")
            chunk = []
    if chunk:
        lines.append(", ".join(chunk) + "\n")


def append_generate_block(lines: List[str], keyword: str, start_id: int, end_id: int, increment: int = 1) -> None:
    if end_id < start_id:
        return
    lines.append(keyword)
    lines.append(f"{start_id}, {end_id}, {increment}\n")


def format_ogden_data_lines(terms: List[Tuple[float, float, float]]) -> List[str]:
    if not terms:
        raise ValueError("MATRIX_OGDEN_TERMS must contain at least one (MU, ALPHA, D) tuple.")

    values: List[float] = []
    for term in terms:
        if len(term) != 3:
            raise ValueError("Each MATRIX_OGDEN_TERMS entry must contain exactly three values: (MU, ALPHA, D).")
        values.extend(float(v) for v in term)

    lines: List[str] = []
    for i in range(0, len(values), 6):
        chunk = values[i:i + 6]
        lines.append(", ".join(f"{value:.10g}" for value in chunk) + "\n")
    return lines


def structured_node_id(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return 1 + i + (nx + 1) * (j + (ny + 1) * k)


def build_matrix_face_node_sets(nx: int, ny: int, nz: int) -> dict[str, List[int]]:
    return {
        "XP": [structured_node_id(nx, j, k, nx, ny) for k in range(nz + 1) for j in range(ny + 1)],
        "XN": [structured_node_id(0, j, k, nx, ny) for k in range(nz + 1) for j in range(ny + 1)],
        "YP": [structured_node_id(i, ny, k, nx, ny) for k in range(nz + 1) for i in range(nx + 1)],
        "YN": [structured_node_id(i, 0, k, nx, ny) for k in range(nz + 1) for i in range(nx + 1)],
        "ZP": [structured_node_id(i, j, nz, nx, ny) for j in range(ny + 1) for i in range(nx + 1)],
        "ZN": [structured_node_id(i, j, 0, nx, ny) for j in range(ny + 1) for i in range(nx + 1)],
    }


def center_node_id_for_structured_mesh(nx: int, ny: int, nz: int) -> int:
    return structured_node_id(nx // 2, ny // 2, nz // 2, nx, ny)


def append_single_node_set(lines: List[str], set_name: str, node_id: int) -> None:
    lines.append(f"*Nset, nset={set_name}\n")
    lines.append(f"{node_id},\n")


def build_pbc_node_pair_records(nx: int, ny: int, nz: int) -> List[Tuple[int, int, int, str, str]]:
    records: List[Tuple[int, int, int, str, str]] = []
    pair_suffix = 1

    for k in range(nz + 1):
        for j in range(ny + 1):
            pos_id = structured_node_id(nx, j, k, nx, ny)
            neg_id = structured_node_id(0, j, k, nx, ny)
            records.append((pair_suffix, pos_id, neg_id, REFPOINT1_INSTANCE_NAME, REFPOINT1_SET_NAME))
            pair_suffix += 2

    for k in range(nz + 1):
        for i in range(nx + 1):
            if i == nx:
                continue
            pos_id = structured_node_id(i, ny, k, nx, ny)
            neg_id = structured_node_id(i, 0, k, nx, ny)
            records.append((pair_suffix, pos_id, neg_id, REFPOINT2_INSTANCE_NAME, REFPOINT2_SET_NAME))
            pair_suffix += 2

    for j in range(ny + 1):
        for i in range(nx + 1):
            if i == nx or j == ny:
                continue
            pos_id = structured_node_id(i, j, nz, nx, ny)
            neg_id = structured_node_id(i, j, 0, nx, ny)
            records.append((pair_suffix, pos_id, neg_id, REFPOINT3_INSTANCE_NAME, REFPOINT3_SET_NAME))
            pair_suffix += 2

    return records


def build_matrix_pbc_export_data(nx: int, ny: int, nz: int) -> Tuple[List[str], List[str], int, int]:
    node_set_lines: List[str] = []
    x_equation_lines: List[str] = []
    y_equation_lines: List[str] = []
    z_equation_lines: List[str] = []
    records = build_pbc_node_pair_records(nx, ny, nz)

    for pair_suffix, pos_id, neg_id, ref_inst_name, ref_set_name in records:
        pos_set_name = f"{MATRIX_NODE_SET_PREFIX}{pair_suffix}"
        neg_set_name = f"{MATRIX_NODE_SET_PREFIX}{pair_suffix + 1}"
        append_single_node_set(node_set_lines, pos_set_name, pos_id)
        append_single_node_set(node_set_lines, neg_set_name, neg_id)

        for axis_name, dof, equation_lines in (
            ("x", 1, x_equation_lines),
            ("y", 2, y_equation_lines),
            ("z", 3, z_equation_lines),
        ):
            equation_lines.append(f"** Constraint: Constraint-{axis_name}-{pair_suffix}\n")
            equation_lines.append("*Equation\n")
            equation_lines.append("3\n")
            equation_lines.append(f"{MATRIX_INSTANCE_NAME}.{pos_set_name}, {dof}, 1.\n")
            equation_lines.append(f"{MATRIX_INSTANCE_NAME}.{neg_set_name}, {dof}, -1.\n")
            equation_lines.append(f"{ref_inst_name}.{ref_set_name}, {dof}, -1.\n")

    return (
        node_set_lines,
        x_equation_lines + y_equation_lines + z_equation_lines,
        center_node_id_for_structured_mesh(nx, ny, nz),
        len(records),
    )


def build_reference_point_part_lines() -> List[str]:
    part_lines: List[str] = []
    ref_parts = (
        (REFPOINT1_PART_NAME, REFPOINT1_SET_NAME, (REFERENCE_POINT_OFFSET_UM, 0.0, 0.0)),
        (REFPOINT2_PART_NAME, REFPOINT2_SET_NAME, (0.0, REFERENCE_POINT_OFFSET_UM, 0.0)),
        (REFPOINT3_PART_NAME, REFPOINT3_SET_NAME, (0.0, 0.0, REFERENCE_POINT_OFFSET_UM)),
    )

    for part_name, set_name, coords in ref_parts:
        internal_set_name = f"{part_name}-RefPt_"
        part_lines.append("**\n")
        part_lines.append(f"*Part, name={part_name}\n")
        part_lines.append("*Node\n")
        part_lines.append(f"1, {coords[0]:.10f}, {coords[1]:.10f}, {coords[2]:.10f}\n")
        part_lines.append(f"*Nset, nset={internal_set_name}, internal\n")
        part_lines.append("1,\n")
        part_lines.append(f"*Nset, nset={set_name}\n")
        part_lines.append("1,\n")
        part_lines.append("*End Part\n")

    return part_lines


def format_abaqus_float(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12):
        return f"{int(round(value))}."
    return f"{value:.10g}"


def boundary_entry(target: str, dof: int, value: Optional[float] = None) -> str:
    if value is None:
        return f"{target}, {dof}, {dof}\n"
    return f"{target}, {dof}, {dof}, {format_abaqus_float(value)}\n"


def build_loading_case_definitions() -> List[dict]:
    rp1 = f"{REFPOINT1_INSTANCE_NAME}.{REFPOINT1_SET_NAME}"
    rp2 = f"{REFPOINT2_INSTANCE_NAME}.{REFPOINT2_SET_NAME}"
    rp3 = f"{REFPOINT3_INSTANCE_NAME}.{REFPOINT3_SET_NAME}"

    def make_case(key: str, file_suffix: str, description: str, boundary_blocks: List[Tuple[str, List[str]]]) -> dict:
        return {
            "key": key,
            "file_suffix": file_suffix,
            "description": description,
            "job_name": f"{JOB_NAME}_{key}",
            "model_name": f"{MODEL_NAME}_{key}",
            "boundary_blocks": boundary_blocks,
        }

    return [
        make_case(
            "UNIAXIAL_X",
            "uniaxial_x",
            "Uniaxial tension along X",
            [
                ("BC-2", [
                    boundary_entry(rp1, 1, UNIAXIAL_TENSION_DISPLACEMENT),
                    boundary_entry(rp1, 2),
                    boundary_entry(rp1, 3),
                ]),
                ("BC-3", [
                    boundary_entry(rp2, 1),
                    boundary_entry(rp2, 3),
                ]),
                ("BC-4", [
                    boundary_entry(rp3, 1),
                    boundary_entry(rp3, 2),
                ]),
            ],
        ),
        make_case(
            "UNIAXIAL_Y",
            "uniaxial_y",
            "Uniaxial tension along Y",
            [
                ("BC-2", [
                    boundary_entry(rp1, 2),
                    boundary_entry(rp1, 3),
                ]),
                ("BC-3", [
                    boundary_entry(rp2, 1),
                    boundary_entry(rp2, 2, UNIAXIAL_TENSION_DISPLACEMENT),
                    boundary_entry(rp2, 3),
                ]),
                ("BC-4", [
                    boundary_entry(rp3, 1),
                    boundary_entry(rp3, 2),
                ]),
            ],
        ),
        make_case(
            "UNIAXIAL_Z",
            "uniaxial_z",
            "Uniaxial tension along Z",
            [
                ("BC-2", [
                    boundary_entry(rp1, 2),
                    boundary_entry(rp1, 3),
                ]),
                ("BC-3", [
                    boundary_entry(rp2, 1),
                    boundary_entry(rp2, 3),
                ]),
                ("BC-4", [
                    boundary_entry(rp3, 1),
                    boundary_entry(rp3, 2),
                    boundary_entry(rp3, 3, UNIAXIAL_TENSION_DISPLACEMENT),
                ]),
            ],
        ),
        make_case(
            "PURE_SHEAR_XY",
            "pure_shear_xy",
            "Pure shear in the XY plane",
            [
                ("BC-2", [boundary_entry(rp1, 2, PURE_SHEAR_DISPLACEMENT)]),
                ("BC-3", [boundary_entry(rp2, 1, PURE_SHEAR_DISPLACEMENT)]),
            ],
        ),
        make_case(
            "PURE_SHEAR_YZ",
            "pure_shear_yz",
            "Pure shear in the YZ plane",
            [
                ("BC-3", [boundary_entry(rp2, 3, PURE_SHEAR_DISPLACEMENT)]),
                ("BC-4", [boundary_entry(rp3, 2, PURE_SHEAR_DISPLACEMENT)]),
            ],
        ),
        make_case(
            "PURE_SHEAR_XZ",
            "pure_shear_xz",
            "Pure shear in the XZ plane",
            [
                ("BC-2", [boundary_entry(rp1, 3, PURE_SHEAR_DISPLACEMENT)]),
                ("BC-4", [boundary_entry(rp3, 1, PURE_SHEAR_DISPLACEMENT)]),
            ],
        ),
    ]


def build_fix_center_boundary_lines(center_node_id: Optional[int]) -> List[str]:
    if not WRITE_FIX_CENTER_BC or center_node_id is None:
        return []

    return [
        "** Name: Fix_Center Type: Displacement/Rotation\n",
        "*Boundary\n",
        f"{CENTER_NODE_SET_NAME}, 1, 1\n",
        f"{CENTER_NODE_SET_NAME}, 2, 2\n",
        f"{CENTER_NODE_SET_NAME}, 3, 3\n",
    ]


def build_step_block_lines(load_case: dict, center_node_id: Optional[int]) -> List[str]:
    initial_inc, total_time, min_inc, max_inc = STATIC_STEP_PARAMETERS
    rp1 = f"{REFPOINT1_INSTANCE_NAME}.{REFPOINT1_SET_NAME}"
    rp2 = f"{REFPOINT2_INSTANCE_NAME}.{REFPOINT2_SET_NAME}"
    rp3 = f"{REFPOINT3_INSTANCE_NAME}.{REFPOINT3_SET_NAME}"
    lines = [
        "** ----------------------------------------------------------------\n",
        "** \n",
        f"** STEP: {STEP_NAME}\n",
        "** \n",
        f"*Step, name={STEP_NAME}, nlgeom={STEP_NLGEOM}, inc={STEP_MAX_INCREMENTS}\n",
        "*Static\n",
        (
            f"{format_abaqus_float(initial_inc)}, {format_abaqus_float(total_time)}, "
            f"{format_abaqus_float(min_inc)}, {format_abaqus_float(max_inc)}\n"
        ),
        "** \n",
        "** BOUNDARY CONDITIONS\n",
        "** \n",
    ]

    for block_name, entries in load_case["boundary_blocks"]:
        lines.append(f"** Name: {block_name} Type: Displacement/Rotation\n")
        lines.append("*Boundary\n")
        lines.extend(entries)

    lines.extend(build_fix_center_boundary_lines(center_node_id))

    lines.extend([
        "** \n",
        "** OUTPUT REQUESTS\n",
        "** \n",
        "*Restart, write, frequency=0\n",
        "** \n",
        "** FIELD OUTPUT: F-Output-1\n",
        "** \n",
        "*Output, field, frequency=1\n",
        f"*Node Output, nset={rp1}\n",
        "U, RF\n",
        f"*Node Output, nset={rp2}\n",
        "U, RF\n",
        f"*Node Output, nset={rp3}\n",
        "U, RF\n",
        "** \n",
        "** HISTORY OUTPUT: H-Output-1\n",
        "** \n",
        "*Output, history, frequency=1\n",
        "*Energy Output\n",
        "ALLSE\n",
        "*End Step\n",
    ])
    return lines


# ============================================================
# DISTRIBUTION SAMPLING
# ============================================================

def sample_diameter(rng: np.random.Generator) -> float:
    """
    Diameter in um. Histology-informed gamma(k, s), truncated for robustness.
    """
    for _ in range(10000):
        d = rng.gamma(shape=GAMMA_SHAPE_K, scale=GAMMA_SCALE_S)
        if DIAMETER_MIN <= d <= DIAMETER_MAX:
            return float(d)
    raise RuntimeError("Could not sample a diameter within the truncation window.")


def sample_theta_deg_truncated(rng: np.random.Generator) -> float:
    """
    Sample theta from the paper's exponential CDF in degrees:
        F(theta) = 1 - exp(-k theta)
    truncated to [0, THETA_MAX_DEG].
    """
    if FIXED_THETA_DEG is not None:
        return float(FIXED_THETA_DEG)

    k = THETA_RATE_PER_DEG
    umax = 1.0 - math.exp(-k * THETA_MAX_DEG)
    u = rng.uniform(0.0, umax)
    theta = -math.log(max(1e-15, 1.0 - u)) / k
    return float(theta)


def sample_phi_deg(rng: np.random.Generator) -> float:
    if FIXED_PHI_DEG is not None:
        return float(FIXED_PHI_DEG)
    if PHI_MODE != "uniform":
        raise ValueError(f"Unsupported PHI_MODE: {PHI_MODE}")
    return float(rng.uniform(0.0, 360.0))


def sample_straightness_ps(rng: np.random.Generator) -> float:
    if FIXED_STRAIGHTNESS_PS is not None:
        return max(float(FIXED_STRAIGHTNESS_PS), 1e-6)
    return max(float(rng.beta(STRAIGHTNESS_BETA_A, STRAIGHTNESS_BETA_B)), 1e-6)


def direction_from_theta_phi(theta_deg: float, phi_deg: float) -> np.ndarray:
    """
    Preferred axis is x. theta = tilt away from x-axis. phi = azimuth around x-axis.

    Direction vector:
        [cos(theta), sin(theta) cos(phi), sin(theta) sin(phi)]
    """
    th = math.radians(theta_deg)
    ph = math.radians(phi_deg)
    v = np.array([
        math.cos(th),
        math.sin(th) * math.cos(ph),
        math.sin(th) * math.sin(ph),
    ], dtype=float)
    return normalize(v)


# ============================================================
# LINE / SEGMENT CLIPPING WITH THE CUBE
# ============================================================

def line_box_intersection(anchor: np.ndarray,
                          direction: np.ndarray,
                          bmin: np.ndarray,
                          bmax: np.ndarray,
                          eps: float = 1e-12) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Intersect infinite line x(t) = anchor + t * direction with axis-aligned box.
    Returns the clipped segment endpoints inside the cube, or None if no valid segment exists.
    """
    tmin = -np.inf
    tmax = np.inf

    for i in range(3):
        if abs(direction[i]) < eps:
            if anchor[i] < bmin[i] or anchor[i] > bmax[i]:
                return None
            continue

        t1 = (bmin[i] - anchor[i]) / direction[i]
        t2 = (bmax[i] - anchor[i]) / direction[i]
        ta = min(t1, t2)
        tb = max(t1, t2)
        tmin = max(tmin, ta)
        tmax = min(tmax, tb)

        if tmin > tmax:
            return None

    p1 = anchor + tmin * direction
    p2 = anchor + tmax * direction

    if np.linalg.norm(p2 - p1) < eps:
        return None
    return p1, p2


# ============================================================
# DISTANCE / NON-OVERLAP CHECK
# ============================================================

def segment_segment_distance(p1: np.ndarray,
                             q1: np.ndarray,
                             p2: np.ndarray,
                             q2: np.ndarray,
                             eps: float = 1e-12) -> float:
    """Minimum distance between 3D line segments p1-q1 and p2-q2."""
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b

    sN, sD = D, D
    tN, tD = D, D

    if D < eps:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)

        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD

    c1 = p1 + sc * u
    c2 = p2 + tc * v
    return float(np.linalg.norm(c1 - c2))


def candidate_is_valid(candidate: Axon, accepted: List[Axon], min_gap: float) -> bool:
    required_base = candidate.radius + min_gap
    for old in accepted:
        d = segment_segment_distance(candidate.p1, candidate.p2, old.p1, old.p2)
        required = required_base + old.radius
        if d < required:
            return False
    return True


def axon_padded_aabb(axon: Axon, pad: float) -> Tuple[np.ndarray, np.ndarray]:
    pmin = np.minimum(axon.p1, axon.p2) - pad
    pmax = np.maximum(axon.p1, axon.p2) + pad
    return pmin, pmax


def aabb_to_grid_range(pmin: np.ndarray,
                       pmax: np.ndarray,
                       bounds_min: np.ndarray,
                       cell_size: float) -> Tuple[np.ndarray, np.ndarray]:
    gmin = np.floor((pmin - bounds_min) / cell_size).astype(int)
    gmax = np.floor((pmax - bounds_min) / cell_size).astype(int)
    return gmin, gmax


def iter_grid_cells(gmin: np.ndarray, gmax: np.ndarray):
    for i in range(int(gmin[0]), int(gmax[0]) + 1):
        for j in range(int(gmin[1]), int(gmax[1]) + 1):
            for k in range(int(gmin[2]), int(gmax[2]) + 1):
                yield (i, j, k)


def add_axon_to_spatial_hash(axon: Axon,
                             axon_index: int,
                             spatial_hash: dict,
                             bounds_min: np.ndarray,
                             cell_size: float) -> None:
    pmin, pmax = axon_padded_aabb(axon, axon.radius)
    gmin, gmax = aabb_to_grid_range(pmin, pmax, bounds_min, cell_size)
    for cell in iter_grid_cells(gmin, gmax):
        spatial_hash.setdefault(cell, []).append(axon_index)


def candidate_is_valid_spatial(candidate: Axon,
                               accepted: List[Axon],
                               min_gap: float,
                               spatial_hash: dict,
                               bounds_min: np.ndarray,
                               cell_size: float) -> bool:
    required_base = candidate.radius + min_gap
    pmin, pmax = axon_padded_aabb(candidate, required_base)
    gmin, gmax = aabb_to_grid_range(pmin, pmax, bounds_min, cell_size)

    checked_indices = set()
    for cell in iter_grid_cells(gmin, gmax):
        for axon_index in spatial_hash.get(cell, []):
            if axon_index in checked_indices:
                continue
            checked_indices.add(axon_index)
            old = accepted[axon_index]
            d = segment_segment_distance(candidate.p1, candidate.p2, old.p1, old.p2)
            required = required_base + old.radius
            if d < required:
                return False
    return True


# ============================================================
# AXON GENERATION
# ============================================================

def sample_anchor_point(rng: np.random.Generator,
                        radius: float,
                        bounds_min: np.ndarray,
                        bounds_max: np.ndarray) -> np.ndarray:
    """
    Sample a point in the cube interior. We leave a small radius-based margin to reduce
    immediate boundary-only placements while still allowing cube clipping later.
    """
    margin = min(0.4 * radius, 0.2)
    lo = bounds_min + margin
    hi = bounds_max - margin
    if ANCHOR_SAMPLING_MODE == "uniform_cube":
        return rng.uniform(lo, hi, size=3)
    if ANCHOR_SAMPLING_MODE == "mean_axis_midplane":
        center = 0.5 * (bounds_min + bounds_max)
        return np.array([
            center[0],
            float(rng.uniform(lo[1], hi[1])),
            float(rng.uniform(lo[2], hi[2])),
        ], dtype=float)
    raise ValueError(f"Unsupported ANCHOR_SAMPLING_MODE: {ANCHOR_SAMPLING_MODE}")


def build_axon_candidate(anchor: np.ndarray,
                         diameter: float,
                         theta_deg: float,
                         phi_deg: float,
                         straightness_ps: float,
                         bounds_min: np.ndarray,
                         bounds_max: np.ndarray) -> Optional[Axon]:
    radius = 0.5 * diameter
    direction = direction_from_theta_phi(theta_deg, phi_deg)

    clipped = line_box_intersection(anchor, direction, bounds_min, bounds_max)
    if clipped is None:
        return None

    p1, p2 = clipped
    length_inside = float(np.linalg.norm(p2 - p1))
    if length_inside <= 1e-8:
        return None

    arc_length = length_inside / straightness_ps
    approx_volume = math.pi * radius * radius * arc_length
    return Axon(
        anchor=anchor,
        direction=direction,
        radius=radius,
        diameter=diameter,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        straightness_ps=straightness_ps,
        tortuosity_tau=1.0 - straightness_ps,
        p1=p1,
        p2=p2,
        length_inside=length_inside,
        arc_length=arc_length,
        approx_volume=approx_volume,
    )


def propose_axon(rng: np.random.Generator,
                 diameter: float,
                 theta_target: float,
                 phi_deg: float,
                 straightness_ps: float,
                 stage_factor: float,
                 bounds_min: np.ndarray,
                 bounds_max: np.ndarray) -> Optional[Axon]:
    radius = 0.5 * diameter

    # A fixed theta is an explicit orientation override and should not be
    # pulled back to the historical THETA_MAX_DEG cone used for x-aligned runs.
    if FIXED_THETA_DEG is not None:
        theta_deg = float(FIXED_THETA_DEG)
    elif ORIENTATION_RETRY_MODE == "preserve_dispersion":
        theta_deg = clamp(theta_target, 0.0, THETA_MAX_DEG)
    else:
        theta_deg = clamp(theta_target * stage_factor, 0.0, THETA_MAX_DEG)
    best_candidate: Optional[Axon] = None
    for _ in range(max(int(ANCHOR_TRIALS_PER_STAGE), 1)):
        anchor = sample_anchor_point(rng, radius, bounds_min, bounds_max)
        candidate = build_axon_candidate(
            anchor=anchor,
            diameter=diameter,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            straightness_ps=straightness_ps,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        if candidate is None:
            continue
        if best_candidate is None or candidate.length_inside > best_candidate.length_inside:
            best_candidate = candidate
    return best_candidate


def generate_axons(bounds_min: np.ndarray,
                   bounds_max: np.ndarray,
                   target_vf: float,
                   min_gap: float,
                   seed: int) -> Tuple[List[Axon], dict]:
    rng = np.random.default_rng(seed)
    accepted: List[Axon] = []
    spatial_hash: dict = {}
    cell_size = max(float(SPATIAL_HASH_CELL_SIZE), max(float(DIAMETER_MAX), 1e-6))

    Vcube = cube_volume(bounds_min, bounds_max)
    Vtarget = target_vf * Vcube
    Vacc = 0.0

    attempts_total = 0
    failed_since_accept = 0
    rescue_uses = 0
    candidate_families = max(int(PLACEMENT_CANDIDATES_PER_ATTEMPT), 1)

    while Vacc < Vtarget and attempts_total < MAX_ATTEMPTS_TOTAL:
        attempts_total += 1

        placed = False
        best_attempt_candidate: Optional[Axon] = None

        for _ in range(candidate_families):
            diameter = sample_diameter(rng)
            theta_target = sample_theta_deg_truncated(rng)
            phi_deg = sample_phi_deg(rng)
            straightness_ps = sample_straightness_ps(rng)

            if ALLOW_DIAMETER_RESCUE and failed_since_accept > 0.6 * MAX_FAILED_TRIES_PER_AXON:
                diameter *= DIAMETER_RESCUE_SCALE
                diameter = max(diameter, DIAMETER_MIN)
                rescue_uses += 1

            for stage_factor in ORIENTATION_RETRY_STAGES:
                effective_min_gap = min_gap
                if ORIENTATION_RETRY_MODE == "preserve_dispersion":
                    effective_min_gap = min_gap * stage_factor

                best_stage_candidate: Optional[Axon] = None
                for _ in range(max(int(ANCHOR_TRIALS_PER_STAGE), 1)):
                    candidate = propose_axon(
                        rng=rng,
                        diameter=diameter,
                        theta_target=theta_target,
                        phi_deg=phi_deg,
                        straightness_ps=straightness_ps,
                        stage_factor=stage_factor,
                        bounds_min=bounds_min,
                        bounds_max=bounds_max,
                    )
                    if candidate is None:
                        continue
                    is_valid = (
                        candidate_is_valid_spatial(
                            candidate,
                            accepted,
                            effective_min_gap,
                            spatial_hash,
                            bounds_min,
                            cell_size,
                        )
                        if USE_SPATIAL_HASH
                        else candidate_is_valid(candidate, accepted, effective_min_gap)
                    )
                    if not is_valid:
                        continue
                    if best_stage_candidate is None or candidate.length_inside > best_stage_candidate.length_inside:
                        best_stage_candidate = candidate

                if best_stage_candidate is None:
                    continue
                if (
                    best_attempt_candidate is None
                    or best_stage_candidate.approx_volume > best_attempt_candidate.approx_volume
                ):
                    best_attempt_candidate = best_stage_candidate

        if best_attempt_candidate is not None:
            accepted.append(best_attempt_candidate)
            if USE_SPATIAL_HASH:
                add_axon_to_spatial_hash(
                    best_attempt_candidate,
                    len(accepted) - 1,
                    spatial_hash,
                    bounds_min,
                    cell_size,
                )
            Vacc += best_attempt_candidate.approx_volume
            failed_since_accept = 0
            placed = True
            vf_now = Vacc / Vcube
            print(
                f"Accepted axon {len(accepted):4d} | d={best_attempt_candidate.diameter:6.3f} um | "
                f"theta={best_attempt_candidate.theta_deg:6.2f} deg | Ps={best_attempt_candidate.straightness_ps:6.3f} | "
                f"L_in={best_attempt_candidate.length_inside:6.3f} um | Lf={best_attempt_candidate.arc_length:6.3f} um | "
                f"VF~{vf_now:7.4f}"
            )

        if not placed:
            failed_since_accept += 1
            if failed_since_accept >= MAX_FAILED_TRIES_PER_AXON:
                print(
                    "Stopped because repeated placement failures indicate packing saturation. "
                    "Try smaller max diameter, smaller min_gap, or tighter theta truncation."
                )
                break

    theta_vals = [ax.theta_deg for ax in accepted]
    diam_vals = [ax.diameter for ax in accepted]
    ps_vals = [ax.straightness_ps for ax in accepted]

    stats = {
        "cube_min": bounds_min.tolist(),
        "cube_max": bounds_max.tolist(),
        "cube_center": (0.5 * (bounds_min + bounds_max)).tolist(),
        "target_volume_fraction": target_vf,
        "target_cube_volume_um3": Vcube,
        "target_axon_volume_um3": Vtarget,
        "accepted_axons": len(accepted),
        "achieved_approx_axon_volume_um3": Vacc,
        "achieved_approx_volume_fraction": Vacc / Vcube,
        "attempts_total": attempts_total,
        "rescue_uses": rescue_uses,
        "placement_candidates_per_attempt": candidate_families,
        "anchor_sampling_mode": ANCHOR_SAMPLING_MODE,
        "anchor_trials_per_stage": ANCHOR_TRIALS_PER_STAGE,
        "spatial_hash_enabled": USE_SPATIAL_HASH,
        "spatial_hash_cell_size_um": cell_size,
        "diameter_distribution": {
            "type": "truncated_gamma",
            "shape_k": GAMMA_SHAPE_K,
            "scale_s_um": GAMMA_SCALE_S,
            "diameter_min_um": DIAMETER_MIN,
            "diameter_max_um": DIAMETER_MAX,
        },
        "orientation_distribution": (
            {
                "type": "fixed_cone_about_x_axis",
                    "fixed_theta_deg": FIXED_THETA_DEG,
                    "fixed_phi_deg": FIXED_PHI_DEG,
                    "phi_mode": PHI_MODE,
                    "retry_stage_factors": ORIENTATION_RETRY_STAGES,
                    "retry_mode": ORIENTATION_RETRY_MODE,
                    "target_kappa_from_fixed_theta": (
                    0.5 * math.sin(math.radians(FIXED_THETA_DEG)) ** 2
                    if FIXED_THETA_DEG is not None
                    else None
                ),
                "target_direction_unit": (
                    direction_from_theta_phi(FIXED_THETA_DEG, FIXED_PHI_DEG if FIXED_PHI_DEG is not None else 0.0).tolist()
                    if FIXED_THETA_DEG is not None and FIXED_PHI_DEG is not None
                    else None
                ),
            }
            if FIXED_THETA_DEG is not None
            else {
                "type": "truncated_exponential_dispersion_about_x_axis",
                "theta_rate_per_degree": THETA_RATE_PER_DEG,
                "theta_max_deg": THETA_MAX_DEG,
                "phi_mode": PHI_MODE,
                "retry_stage_factors": ORIENTATION_RETRY_STAGES,
                "retry_mode": ORIENTATION_RETRY_MODE,
            }
        ),
        "straightness_distribution": {
            "type": "beta",
            "shape_a": STRAIGHTNESS_BETA_A,
            "shape_b": STRAIGHTNESS_BETA_B,
        },
        "anchor_sampling_mode": ANCHOR_SAMPLING_MODE,
        "anchor_trials_per_stage": ANCHOR_TRIALS_PER_STAGE,
        "theta_mean_deg": float(np.mean(theta_vals)) if theta_vals else None,
        "diameter_mean_um": float(np.mean(diam_vals)) if diam_vals else None,
        "diameter_std_um": float(np.std(diam_vals)) if diam_vals else None,
        "straightness_mean": float(np.mean(ps_vals)) if ps_vals else None,
        "straightness_std": float(np.std(ps_vals)) if ps_vals else None,
        "min_gap_um": min_gap,
        "seed": seed,
        "truss_element_length_um": TRUSS_ELEMENT_LENGTH,
        "hex_element_length_um": HEX_ELEMENT_LENGTH,
        "matrix_element_type": MATRIX_ELEMENT_TYPE,
    }
    return accepted, stats


# ============================================================
# INP EXPORT HELPERS
# ============================================================

def subdivide_segment(p1: np.ndarray, p2: np.ndarray, target_len: float) -> List[np.ndarray]:
    length = float(np.linalg.norm(p2 - p1))
    if length <= 1e-12:
        return [p1.copy(), p2.copy()]

    # Use nearly uniform spacing so clipped fibers do not end with tiny tail
    # elements, which can make the embedded truss network numerically ill-conditioned.
    n_elem = max(1, int(math.ceil((length - 1e-12) / target_len)))
    direction = p2 - p1
    pts = [p1 + direction * (i / n_elem) for i in range(n_elem)]
    pts.append(p2.copy())
    return pts


def build_structured_hex_mesh(bounds_min: np.ndarray,
                              bounds_max: np.ndarray,
                              element_length: float) -> Tuple[List[str], List[str], int, int, Tuple[int, int, int]]:
    size = bounds_max - bounds_min
    counts: List[int] = []
    for axis_length in size:
        raw_count = axis_length / element_length
        rounded_count = int(round(raw_count))
        if not math.isclose(raw_count, rounded_count, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "Each cube edge length must be an integer multiple of HEX_ELEMENT_LENGTH "
                f"({element_length} um)."
            )
        counts.append(rounded_count)

    nx, ny, nz = counts
    if min(nx, ny, nz) < 1:
        raise ValueError("Structured hex mesh requires at least one element in each direction.")

    node_lines: List[str] = []
    elem_lines: List[str] = []
    node_map = {}
    node_id = 1

    for k in range(nz + 1):
        z = bounds_min[2] + k * element_length
        for j in range(ny + 1):
            y = bounds_min[1] + j * element_length
            for i in range(nx + 1):
                x = bounds_min[0] + i * element_length
                node_map[(i, j, k)] = node_id
                node_lines.append(f"{node_id}, {x:.10f}, {y:.10f}, {z:.10f}\n")
                node_id += 1

    elem_id = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = node_map[(i, j, k)]
                n2 = node_map[(i + 1, j, k)]
                n3 = node_map[(i + 1, j + 1, k)]
                n4 = node_map[(i, j + 1, k)]
                n5 = node_map[(i, j, k + 1)]
                n6 = node_map[(i + 1, j, k + 1)]
                n7 = node_map[(i + 1, j + 1, k + 1)]
                n8 = node_map[(i, j + 1, k + 1)]
                elem_lines.append(
                    f"{elem_id}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n"
                )
                elem_id += 1

    return node_lines, elem_lines, node_id - 1, elem_id - 1, (nx, ny, nz)


def export_loading_case_inps(bounds_min: np.ndarray,
                           bounds_max: np.ndarray,
                           axons: List[Axon],
                           out_dir: str) -> Tuple[List[str], int, int, int, int, Tuple[int, int, int], Optional[int], int, List[dict]]:
    """
    Write one Abaqus input file per loading case, all sharing the same generated
    geometry, PBC equations, materials, and Ps field assignments.
    """
    os.makedirs(out_dir, exist_ok=True)

    (
        matrix_node_lines,
        matrix_elem_lines,
        total_matrix_nodes,
        total_matrix_elements,
        matrix_shape,
    ) = build_structured_hex_mesh(bounds_min, bounds_max, HEX_ELEMENT_LENGTH)
    matrix_face_sets = build_matrix_face_node_sets(*matrix_shape)
    matrix_face_nset_lines: List[str] = []
    for set_name, node_ids in matrix_face_sets.items():
        append_set_block(matrix_face_nset_lines, f"*Nset, nset={set_name}\n", node_ids)

    matrix_pbc_nset_lines: List[str] = []
    pbc_equation_lines: List[str] = []
    reference_point_part_lines: List[str] = []
    center_node_id: Optional[int] = None
    pbc_pair_count = 0
    if WRITE_PBC_EQUATIONS:
        (
            matrix_pbc_nset_lines,
            pbc_equation_lines,
            center_node_id,
            pbc_pair_count,
        ) = build_matrix_pbc_export_data(*matrix_shape)
        reference_point_part_lines = build_reference_point_part_lines()

    ogden_data_lines = format_ogden_data_lines(MATRIX_OGDEN_TERMS)

    truss_node_lines: List[str] = []
    truss_elem_lines: List[str] = []
    truss_nset_lines: List[str] = []
    truss_elset_lines: List[str] = []
    truss_section_lines: List[str] = []
    assembly_nset_lines: List[str] = []
    predefined_field_lines: List[str] = []

    truss_node_id = 1
    truss_elem_id = 1

    for i, ax in enumerate(axons, start=1):
        pts = subdivide_segment(ax.p1, ax.p2, TRUSS_ELEMENT_LENGTH)
        first_node_id = truss_node_id

        for p in pts:
            truss_node_lines.append(f"{truss_node_id}, {p[0]:.10f}, {p[1]:.10f}, {p[2]:.10f}\n")
            truss_node_id += 1

        last_node_id = truss_node_id - 1
        first_elem_id = truss_elem_id

        for local_index in range(first_node_id, last_node_id):
            truss_elem_lines.append(f"{truss_elem_id}, {local_index}, {local_index + 1}\n")
            truss_elem_id += 1

        last_elem_id = truss_elem_id - 1
        nset_name = f"NFIBER_{i:05d}"
        nset_asm_name = f"{nset_name}_ASM"
        elset_name = f"ESET_AXON_{i:05d}"
        section_name = f"SECTION-ESET_AXON_{i:05d}"
        field_name = f"{FIELD_NAME_PREFIX}-{i:05d}"
        area = math.pi * (ax.diameter ** 2) / 4.0

        append_generate_block(truss_nset_lines, f"*Nset, nset={nset_name}, generate\n", first_node_id, last_node_id)
        append_generate_block(truss_elset_lines, f"*Elset, elset={elset_name}, generate\n", first_elem_id, last_elem_id)
        append_generate_block(
            assembly_nset_lines,
            f"*Nset, nset={nset_asm_name}, instance={TRUSS_INSTANCE_NAME}, generate\n",
            first_node_id,
            last_node_id,
        )

        truss_section_lines.append(f"** Section: {section_name}\n")
        truss_section_lines.append(f"*Solid Section, elset={elset_name}, material={AXON_MATERIAL_NAME}\n")
        truss_section_lines.append(f"{area:.12f}\n")

        predefined_field_lines.append(f"** Name: {field_name}   Type: Field\n")
        predefined_field_lines.append(f"*Initial Conditions, type=FIELD, variable={PS_FIELD_VARIABLE}\n")
        predefined_field_lines.append(
            f"{nset_asm_name}, {ax.straightness_ps:.{FIELD_VALUE_DECIMALS}f}\n"
        )

    total_truss_nodes = truss_node_id - 1
    total_truss_elements = truss_elem_id - 1
    load_cases = build_loading_case_definitions()
    generated_paths: List[str] = []
    generated_case_records: List[dict] = []

    for load_case in load_cases:
        inp_filename = f"{RVE_INP_STEM}_{load_case['file_suffix']}.inp"
        inp_path = os.path.join(out_dir, inp_filename)
        step_block_lines = build_step_block_lines(load_case, center_node_id)

        with open(inp_path, "w", encoding="utf-8") as f:
            f.write("*Heading\n")
            f.write(f"** Job name: {load_case['job_name']} Model name: {load_case['model_name']}\n")
            f.write("** Generated by: PYTHON SCRIPT\n")
            f.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
            f.write("**\n")
            f.write("** PARTS\n")
            f.write("**\n")

            f.write(f"*Part, name={TRUSS_PART_NAME}\n")
            f.write("*Node\n")
            for line in truss_node_lines:
                f.write(line)
            f.write("*Element, type=T3D2H\n")
            for line in truss_elem_lines:
                f.write(line)
            if total_truss_nodes > 0:
                f.write("*Nset, nset=NALL_AXONS, generate\n")
                f.write(f"1, {total_truss_nodes}, 1\n")
            if total_truss_elements > 0:
                f.write("*Elset, elset=EALL_AXONS, generate\n")
                f.write(f"1, {total_truss_elements}, 1\n")
            for line in truss_nset_lines:
                f.write(line)
            for line in truss_elset_lines:
                f.write(line)
            for line in truss_section_lines:
                f.write(line)
            f.write("*End Part\n")
            f.write("**\n")

            f.write(f"*Part, name={MATRIX_PART_NAME}\n")
            f.write("*Node\n")
            for line in matrix_node_lines:
                f.write(line)
            f.write(f"*Element, type={MATRIX_ELEMENT_TYPE}\n")
            for line in matrix_elem_lines:
                f.write(line)
            f.write("*Nset, nset=NALL_MATRIX, generate\n")
            f.write(f"1, {total_matrix_nodes}, 1\n")
            for line in matrix_face_nset_lines:
                f.write(line)
            for line in matrix_pbc_nset_lines:
                f.write(line)
            f.write("*Elset, elset=EALL_MATRIX, generate\n")
            f.write(f"1, {total_matrix_elements}, 1\n")
            f.write("** Section: SECTION-1-EALL_MATRIX\n")
            f.write(f"*Solid Section, elset=EALL_MATRIX, material={MATRIX_MATERIAL_NAME}\n")
            f.write(",\n")
            f.write("*End Part\n")

            for line in reference_point_part_lines:
                f.write(line)

            f.write("**\n")
            f.write("** ASSEMBLY\n")
            f.write("**\n")
            f.write(f"*Assembly, name={ASSEMBLY_NAME}\n")
            f.write("**\n")
            f.write(f"*Instance, name={MATRIX_INSTANCE_NAME}, part={MATRIX_PART_NAME}\n")
            f.write("*End Instance\n")
            f.write("**\n")
            f.write(f"*Instance, name={TRUSS_INSTANCE_NAME}, part={TRUSS_PART_NAME}\n")
            f.write("*End Instance\n")
            f.write("**\n")

            if WRITE_PBC_EQUATIONS:
                for ref_part_name, ref_instance_name in (
                    (REFPOINT1_PART_NAME, REFPOINT1_INSTANCE_NAME),
                    (REFPOINT2_PART_NAME, REFPOINT2_INSTANCE_NAME),
                    (REFPOINT3_PART_NAME, REFPOINT3_INSTANCE_NAME),
                ):
                    f.write(f"*Instance, name={ref_instance_name}, part={ref_part_name}\n")
                    f.write("*End Instance\n")
                    f.write("**\n")

            if center_node_id is not None:
                f.write(f"*Nset, nset={CENTER_NODE_SET_NAME}, instance={MATRIX_INSTANCE_NAME}\n")
                f.write(f"{center_node_id},\n")

            for line in assembly_nset_lines:
                f.write(line)

            for line in pbc_equation_lines:
                f.write(line)

            if total_truss_elements > 0:
                f.write(f"** Constraint: {EMBEDDED_CONSTRAINT_NAME}\n")
                f.write(f"*Embedded Element, host elset={MATRIX_INSTANCE_NAME}.EALL_MATRIX\n")
                f.write(f"{TRUSS_INSTANCE_NAME}.EALL_AXONS\n")

            f.write("*End Assembly\n")
            f.write("**\n")
            f.write("** MATERIALS\n")
            f.write("**\n")
            f.write(f"*Material, name={MATRIX_MATERIAL_NAME}\n")
            f.write(f"*Hyperelastic, ogden, n={len(MATRIX_OGDEN_TERMS)}\n")
            for line in ogden_data_lines:
                f.write(line)
            f.write("**\n")
            f.write(f"*Material, name={AXON_MATERIAL_NAME}\n")
            f.write("*Hyperelastic, user, formulation=stretch, type=incompressible, properties=2\n")
            f.write(", ".join(f"{value:.10g}" for value in AXON_USER_PROPERTIES) + "\n")
            f.write("** \n")
            f.write("** PREDEFINED FIELDS\n")
            f.write("** \n")
            for line in predefined_field_lines:
                f.write(line)
            for line in step_block_lines:
                f.write(line)

        generated_paths.append(inp_path)
        generated_case_records.append({
            "key": load_case["key"],
            "description": load_case["description"],
            "file_name": inp_filename,
            "job_name": load_case["job_name"],
            "model_name": load_case["model_name"],
            "boundary_blocks": [
                {
                    "name": block_name,
                    "entries": [entry.strip() for entry in entries],
                }
                for block_name, entries in load_case["boundary_blocks"]
            ],
        })

    return (
        generated_paths,
        total_matrix_nodes,
        total_matrix_elements,
        total_truss_nodes,
        total_truss_elements,
        matrix_shape,
        center_node_id,
        pbc_pair_count,
        generated_case_records,
    )


# ============================================================
# SUMMARY
# ============================================================

def save_summary(axons: List[Axon], stats: dict, out_dir: str,
                 total_matrix_nodes: int,
                 total_matrix_elements: int,
                 total_truss_nodes: int,
                 total_truss_elements: int,
                 matrix_shape: Tuple[int, int, int],
                 center_node_id: Optional[int],
                 pbc_pair_count: int,
                 generated_paths: List[str],
                 generated_case_records: List[dict]):
    data = dict(stats)
    data["job_name"] = JOB_NAME
    data["model_name"] = MODEL_NAME
    data["load_case_inp_files"] = [os.path.basename(p) for p in generated_paths]
    data["load_case_count"] = len(generated_case_records)
    data["runtime_config_path"] = ACTIVE_CONFIG_PATH
    data["embedded_constraint_name"] = EMBEDDED_CONSTRAINT_NAME
    data["matrix_material_name"] = MATRIX_MATERIAL_NAME
    data["axon_material_name"] = AXON_MATERIAL_NAME
    data["matrix_ogden_terms"] = [list(term) for term in MATRIX_OGDEN_TERMS]
    data["axon_user_properties"] = list(AXON_USER_PROPERTIES)
    data["step_name"] = STEP_NAME
    data["step_nlgeom"] = STEP_NLGEOM
    data["step_max_increments"] = STEP_MAX_INCREMENTS
    data["static_step_parameters"] = list(STATIC_STEP_PARAMETERS)
    data["uniaxial_tension_displacement"] = UNIAXIAL_TENSION_DISPLACEMENT
    data["pure_shear_displacement"] = PURE_SHEAR_DISPLACEMENT
    data["write_fix_center_bc"] = WRITE_FIX_CENTER_BC
    data["total_matrix_nodes"] = total_matrix_nodes
    data["total_matrix_elements"] = total_matrix_elements
    data["total_truss_nodes"] = total_truss_nodes
    data["total_truss_elements"] = total_truss_elements
    data["matrix_mesh_divisions"] = {
        "nx": matrix_shape[0],
        "ny": matrix_shape[1],
        "nz": matrix_shape[2],
    }
    data["matrix_part_name"] = MATRIX_PART_NAME
    data["matrix_instance_name"] = MATRIX_INSTANCE_NAME
    data["matrix_element_type"] = MATRIX_ELEMENT_TYPE
    data["matrix_face_node_sets"] = ["XP", "XN", "YP", "YN", "ZP", "ZN"]
    data["write_pbc_equations"] = WRITE_PBC_EQUATIONS
    data["reference_point_offset_um"] = REFERENCE_POINT_OFFSET_UM
    data["reference_point_parts"] = [REFPOINT1_PART_NAME, REFPOINT2_PART_NAME, REFPOINT3_PART_NAME]
    data["reference_point_instances"] = [REFPOINT1_INSTANCE_NAME, REFPOINT2_INSTANCE_NAME, REFPOINT3_INSTANCE_NAME]
    data["center_node_set_name"] = CENTER_NODE_SET_NAME if center_node_id is not None else None
    data["center_node_id"] = center_node_id
    data["matrix_pbc_pair_count"] = pbc_pair_count
    data["matrix_pbc_single_node_set_count"] = 2 * pbc_pair_count
    data["matrix_pbc_equation_count"] = 3 * pbc_pair_count
    data["truss_part_name"] = TRUSS_PART_NAME
    data["truss_instance_name"] = TRUSS_INSTANCE_NAME
    data["field_variable_number"] = PS_FIELD_VARIABLE
    data["loading_cases"] = generated_case_records
    data["axons"] = []
    data["fixed_straightness_ps"] = FIXED_STRAIGHTNESS_PS

    for i, ax in enumerate(axons, start=1):
        rec = {
            "id": i,
            "field_name": f"{FIELD_NAME_PREFIX}-{i:05d}",
            "part_node_set": f"NFIBER_{i:05d}",
            "assembly_node_set": f"NFIBER_{i:05d}_ASM",
            "field_variable_number": PS_FIELD_VARIABLE,
            "anchor": ax.anchor.tolist(),
            "direction": ax.direction.tolist(),
            "diameter_um": ax.diameter,
            "radius_um": ax.radius,
            "theta_deg": ax.theta_deg,
            "phi_deg": ax.phi_deg,
            "straightness_ps": ax.straightness_ps,
            "tortuosity_tau": ax.tortuosity_tau,
            "p1": ax.p1.tolist(),
            "p2": ax.p2.tolist(),
            "length_inside_um": ax.length_inside,
            "arc_length_um": ax.arc_length,
            "approx_volume_um3": ax.approx_volume,
            "cross_section_area_um2": math.pi * (ax.diameter ** 2) / 4.0,
            "estimated_truss_elements": int(math.ceil(ax.length_inside / TRUSS_ELEMENT_LENGTH)),
        }
        data["axons"].append(rec)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SUMMARY_JSON)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


# ============================================================
# MAIN
# ============================================================

def main():
    axons, stats = generate_axons(
        bounds_min=CUBE_MIN,
        bounds_max=CUBE_MAX,
        target_vf=TARGET_VOLUME_FRACTION,
        min_gap=MIN_GAP,
        seed=SEED,
    )

    print("\nSummary")
    print("-------")
    for k, v in stats.items():
        if k in ("diameter_distribution", "orientation_distribution", "straightness_distribution"):
            continue
        print(f"{k}: {v}")

    (
        generated_inp_paths,
        total_matrix_nodes,
        total_matrix_elements,
        total_truss_nodes,
        total_truss_elements,
        matrix_shape,
        center_node_id,
        pbc_pair_count,
        generated_case_records,
    ) = export_loading_case_inps(CUBE_MIN, CUBE_MAX, axons, OUTPUT_DIR)

    summary_path = save_summary(
        axons,
        stats,
        OUTPUT_DIR,
        total_matrix_nodes,
        total_matrix_elements,
        total_truss_nodes,
        total_truss_elements,
        matrix_shape,
        center_node_id,
        pbc_pair_count,
        generated_inp_paths,
        generated_case_records,
    )

    print("\nExport complete")
    print("---------------")
    print(f"Summary JSON : {summary_path}")
    print("Generated INP files:")
    for inp_path in generated_inp_paths:
        print(f"  {inp_path}")
    print("Note         : VF uses Ps-based arc-length correction Lf = L0 / Ps")
    print(f"Materials    : MATRIX={MATRIX_MATERIAL_NAME}, AXON={AXON_MATERIAL_NAME}")
    print(f"Field var    : Ps written as *Initial Conditions, type=FIELD, variable={PS_FIELD_VARIABLE}")
    print(
        f"Matrix mesh  : {matrix_shape[0]} x {matrix_shape[1]} x {matrix_shape[2]} "
        f"{MATRIX_ELEMENT_TYPE} elements"
    )
    print(f"Matrix nodes : {total_matrix_nodes}")
    print(f"Matrix elems : {total_matrix_elements}")
    print(f"Truss nodes  : {total_truss_nodes}")
    print(f"Truss elems  : {total_truss_elements}")
    print(f"PBC export   : {WRITE_PBC_EQUATIONS}")
    print(f"Fix center   : {WRITE_FIX_CENTER_BC}")
    print(f"Load cases   : {len(generated_case_records)}")
    if WRITE_PBC_EQUATIONS:
        print(f"Center node  : {center_node_id}")
        print(f"PBC pairs    : {pbc_pair_count}")
        print(f"PBC eqns     : {3 * pbc_pair_count}")


if __name__ == "__main__":
    main()
