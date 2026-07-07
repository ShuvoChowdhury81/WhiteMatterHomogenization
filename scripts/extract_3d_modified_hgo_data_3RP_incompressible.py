# -*- coding: utf-8 -*-
"""
Extractor for 3D incompressible RVE data for modified HGO fitting
=================================================================

Matches the 3-RP periodic BC script in xyzPeriodic1.py:

RefPoint1 controls jumps across X faces:
    u(X+) - u(X-) = U1(RP1)   -> F11 = 1 + U1(RP1)/Lx
    v(X+) - v(X-) = U2(RP1)   -> F21 =     U2(RP1)/Lx
    w(X+) - w(X-) = U3(RP1)   -> F31 =     U3(RP1)/Lx

RefPoint2 controls jumps across Y faces:
    u(Y+) - u(Y-) = U1(RP2)   -> F12 =     U1(RP2)/Ly
    v(Y+) - v(Y-) = U2(RP2)   -> F22 = 1 + U2(RP2)/Ly
    w(Y+) - w(Y-) = U3(RP2)   -> F32 =     U3(RP2)/Ly

RefPoint3 controls jumps across Z faces:
    u(Z+) - u(Z-) = U1(RP3)   -> F13 =     U1(RP3)/Lz
    v(Z+) - v(Z-) = U2(RP3)   -> F23 =     U2(RP3)/Lz
    w(Z+) - w(Z-) = U3(RP3)   -> F33 = 1 + U3(RP3)/Lz

Incompressible fitting:
- Volumetric term omitted
- J is stored as a check only
- C is used directly (not J^(-2/3) C); for ideal incompressibility J = 1

Outputs:
- W_hom = ALLSE / V0
- Fbar, Cbar
- J_check
- I1
- Orientation tensor A
- Preferred direction a0
- Dispersion parameter kappa
- Structure tensor H
- I4_star = tr(C H)
- Classical I4 and I5
- Pbar from RP reactions
"""

from odbAccess import openOdb
from abaqusConstants import NODAL
import argparse
import csv
import json
import math
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# USER INPUTS
# ============================================================
DATA_DIR = SCRIPT_DIR
STEP_NAME = "Step-1"

# These are the assembly node-set names you created in the model.
# Based on xyzPeriodic1.py, the part-level sets are SetRefPoint1/2/3.
# In the ODB, CAE commonly uppercases them.
RP1_SET = "SETREFPOINT1"
RP2_SET = "SETREFPOINT2"
RP3_SET = "SETREFPOINT3"
RP1_INSTANCE = "REFPOINT1-1"
RP2_INSTANCE = "REFPOINT2-1"
RP3_INSTANCE = "REFPOINT3-1"

# RVE dimensions (reference configuration)
# These defaults are used only as a fallback. When a generator summary JSON is
# available, the extractor reads the actual cube size from that file so studies
# with different RVE sizes are normalized correctly.
LX = 20.0
LY = 20.0
LZ = 20.0
V0 = LX * LY * LZ

# JSON exported by the truss generator
SUMMARY_JSON = os.path.join(DATA_DIR, "rve_truss_generation_summary_with_ps_oriented.json")
SUMMARY_JSON_GLOB = "rve_truss_generation_summary"

# Face areas for nominal stress
AX = LY * LZ   # face normal to x
AY = LX * LZ   # face normal to y
AZ = LX * LY   # face normal to z


# ============================================================
# BASIC 3x3 HELPERS
# ============================================================
def identity3():
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

def transpose3(A):
    return (
        (A[0][0], A[1][0], A[2][0]),
        (A[0][1], A[1][1], A[2][1]),
        (A[0][2], A[1][2], A[2][2]),
    )

def matmul3(A, B):
    return (
        (
            A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0],
            A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1],
            A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2],
        ),
        (
            A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0],
            A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1],
            A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2],
        ),
        (
            A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0],
            A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1],
            A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2],
        ),
    )

def det3(A):
    return (
        A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
        - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
        + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0])
    )

def trace3(A):
    return A[0][0] + A[1][1] + A[2][2]

def scale3(A, s):
    return (
        (s*A[0][0], s*A[0][1], s*A[0][2]),
        (s*A[1][0], s*A[1][1], s*A[1][2]),
        (s*A[2][0], s*A[2][1], s*A[2][2]),
    )

def add3(A, B):
    return (
        (A[0][0]+B[0][0], A[0][1]+B[0][1], A[0][2]+B[0][2]),
        (A[1][0]+B[1][0], A[1][1]+B[1][1], A[1][2]+B[1][2]),
        (A[2][0]+B[2][0], A[2][1]+B[2][1], A[2][2]+B[2][2]),
    )

def outer3(a, b):
    return (
        (a[0]*b[0], a[0]*b[1], a[0]*b[2]),
        (a[1]*b[0], a[1]*b[1], a[1]*b[2]),
        (a[2]*b[0], a[2]*b[1], a[2]*b[2]),
    )

def dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def matvec3(A, v):
    return (
        A[0][0]*v[0] + A[0][1]*v[1] + A[0][2]*v[2],
        A[1][0]*v[0] + A[1][1]*v[1] + A[1][2]*v[2],
        A[2][0]*v[0] + A[2][1]*v[1] + A[2][2]*v[2],
    )

def tr_AB(A, B):
    return trace3(matmul3(A, B))


# ============================================================
# ODB HELPERS
# ============================================================
def find_allse_history(step):
    for _, region in step.historyRegions.items():
        if 'ALLSE' in region.historyOutputs:
            return region.historyOutputs['ALLSE'].data
    raise RuntimeError("ALLSE history output not found in step '%s'." % step.name)

def nearest_history_value(history_data, target_time):
    best_val = None
    best_dt = 1.0e99
    for t, val in history_data:
        dt = abs(t - target_time)
        if dt < best_dt:
            best_dt = dt
            best_val = val
    return best_val

def get_rp_region(assembly, set_name, instance_name=None):
    if set_name in assembly.nodeSets:
        return assembly.nodeSets[set_name]

    if instance_name is not None:
        if instance_name not in assembly.instances:
            raise RuntimeError(
                "Assembly instance '%s' not found while looking for node set '%s'."
                % (instance_name, set_name)
            )
        instance = assembly.instances[instance_name]
        if set_name in instance.nodeSets:
            return instance.nodeSets[set_name]

    matching_regions = []
    for candidate_instance_name, instance in assembly.instances.items():
        if set_name in instance.nodeSets:
            matching_regions.append((candidate_instance_name, instance.nodeSets[set_name]))

    if len(matching_regions) == 1:
        return matching_regions[0][1]

    if len(matching_regions) > 1:
        matching_instance_names = ", ".join(name for name, _ in matching_regions)
        raise RuntimeError(
            "Node set '%s' was found in multiple instances (%s). "
            "Set the matching RP instance name explicitly."
            % (set_name, matching_instance_names)
        )

    raise RuntimeError(
        "Node set '%s' not found in assembly-level sets or instance-level sets."
        % set_name
    )

def average_nodal_vector(frame, field_name, node_set):
    if field_name not in frame.fieldOutputs:
        raise RuntimeError("Field output '%s' not available in frame." % field_name)
    subset = frame.fieldOutputs[field_name].getSubset(region=node_set, position=NODAL)
    sx = sy = sz = 0.0
    count = 0
    for v in subset.values:
        sx += v.data[0]
        sy += v.data[1]
        sz += v.data[2]
        count += 1
    if count == 0:
        raise RuntimeError("No nodal values found for '%s' in set '%s'." % (field_name, node_set.name))
    return (sx / count, sy / count, sz / count)


def find_odb_paths(data_dir):
    if not os.path.isdir(data_dir):
        raise RuntimeError("Data directory not found: %s" % data_dir)

    odb_paths = []
    for name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, name)
        if os.path.isfile(full_path) and name.lower().endswith(".odb"):
            odb_paths.append(full_path)

    if not odb_paths:
        raise RuntimeError("No ODB files found in data directory: %s" % data_dir)

    return odb_paths


def discover_summary_json(data_dir):
    if not os.path.isdir(data_dir):
        return None

    candidates = []
    for name in sorted(os.listdir(data_dir)):
        lower = name.lower()
        if os.path.isfile(os.path.join(data_dir, name)) and lower.endswith(".json") and SUMMARY_JSON_GLOB in lower:
            candidates.append(os.path.join(data_dir, name))

    if len(candidates) == 1:
        return candidates[0]
    return None


# ============================================================
# ORIENTATION / DISPERSION FROM SUMMARY JSON
# ============================================================
def normalize_vec(v):
    n = math.sqrt(dot3(v, v))
    if n < 1e-15:
        raise RuntimeError("Zero-length direction vector encountered.")
    return (v[0]/n, v[1]/n, v[2]/n)

def choose_power_iteration_seed(A):
    candidates = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    ]

    for candidate in candidates:
        w = matvec3(A, candidate)
        if dot3(w, w) > 1e-15:
            return normalize_vec(candidate)

    raise RuntimeError("Orientation tensor is near zero; could not find a power-iteration seed.")

def power_iteration_symmetric(A, n_iter=100):
    v = choose_power_iteration_seed(A)
    for _ in range(n_iter):
        w = matvec3(A, v)
        v = normalize_vec(w)
    Av = matvec3(A, v)
    lam = dot3(v, Av)
    return lam, v

def read_orientation_tensor(summary_json_path):
    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    axons = data["axons"]
    if len(axons) == 0:
        raise RuntimeError("No axons found in summary JSON.")

    A = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )

    N = float(len(axons))
    for ax in axons:
        n = ax["direction"]
        n = normalize_vec((float(n[0]), float(n[1]), float(n[2])))
        A = add3(A, outer3(n, n))

    A = scale3(A, 1.0 / N)

    lam1, a0 = power_iteration_symmetric(A)

    # Transversely isotropic approximation:
    # lambda1 = 1 - 2*kappa, lambda2 = lambda3 = kappa
    kappa = 0.5 * (1.0 - lam1)
    kappa = max(0.0, min(1.0/3.0, kappa))

    I = identity3()
    a0a0 = outer3(a0, a0)
    H = add3(scale3(I, kappa), scale3(a0a0, (1.0 - 3.0*kappa)))

    return A, a0, lam1, kappa, H


def read_geometry_from_summary(summary_json_path):
    dims = {
        "LX": LX,
        "LY": LY,
        "LZ": LZ,
        "V0": V0,
        "AX": AX,
        "AY": AY,
        "AZ": AZ,
    }

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cube_min = data.get("cube_min")
    cube_max = data.get("cube_max")
    if cube_min is None or cube_max is None:
        return dims

    lx = float(cube_max[0]) - float(cube_min[0])
    ly = float(cube_max[1]) - float(cube_min[1])
    lz = float(cube_max[2]) - float(cube_min[2])
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise RuntimeError(
            "Invalid cube dimensions found in summary JSON: cube_min=%s cube_max=%s"
            % (cube_min, cube_max)
        )

    dims["LX"] = lx
    dims["LY"] = ly
    dims["LZ"] = lz
    dims["V0"] = lx * ly * lz
    dims["AX"] = ly * lz
    dims["AY"] = lx * lz
    dims["AZ"] = lx * ly
    return dims


# ============================================================
# MACRO KINEMATICS / STRESS FROM THE 3 RP SETUP
# ============================================================
def extract_macro_F_from_3RP(frame, assembly, dims):
    rp1 = average_nodal_vector(frame, 'U', get_rp_region(assembly, RP1_SET, RP1_INSTANCE))
    rp2 = average_nodal_vector(frame, 'U', get_rp_region(assembly, RP2_SET, RP2_INSTANCE))
    rp3 = average_nodal_vector(frame, 'U', get_rp_region(assembly, RP3_SET, RP3_INSTANCE))

    Fbar = (
        (1.0 + rp1[0]/dims["LX"],      rp2[0]/dims["LY"],      rp3[0]/dims["LZ"]),
        (      rp1[1]/dims["LX"], 1.0 + rp2[1]/dims["LY"],     rp3[1]/dims["LZ"]),
        (      rp1[2]/dims["LX"],      rp2[2]/dims["LY"], 1.0 + rp3[2]/dims["LZ"]),
    )
    return Fbar, rp1, rp2, rp3

def extract_macro_P_from_3RP(frame, assembly, dims):
    rf1 = average_nodal_vector(frame, 'RF', get_rp_region(assembly, RP1_SET, RP1_INSTANCE))
    rf2 = average_nodal_vector(frame, 'RF', get_rp_region(assembly, RP2_SET, RP2_INSTANCE))
    rf3 = average_nodal_vector(frame, 'RF', get_rp_region(assembly, RP3_SET, RP3_INSTANCE))

    Pbar = (
        (rf1[0]/dims["AX"], rf2[0]/dims["AY"], rf3[0]/dims["AZ"]),
        (rf1[1]/dims["AX"], rf2[1]/dims["AY"], rf3[1]/dims["AZ"]),
        (rf1[2]/dims["AX"], rf2[2]/dims["AY"], rf3[2]/dims["AZ"]),
    )
    return Pbar, rf1, rf2, rf3

def compute_macro_tensors(Fbar, H, a0):
    Ft = transpose3(Fbar)
    Cbar = matmul3(Ft, Fbar)
    J = det3(Fbar)

    # Incompressible formulation: use C directly
    I1 = trace3(Cbar)
    I4_star = tr_AB(Cbar, H)

    Ca0 = matvec3(Cbar, a0)
    I4_classic = dot3(a0, Ca0)
    C2a0 = matvec3(Cbar, Ca0)
    I5_classic = dot3(a0, C2a0)

    return Cbar, J, I1, I4_star, I4_classic, I5_classic


def extract_rows_from_odb(odb_path, step_name, A_orient, a0, lam1, kappa, H, dims):
    odb = openOdb(odb_path, readOnly=True)
    if step_name not in odb.steps:
        odb.close()
        raise RuntimeError("Step '%s' not found in ODB '%s'." % (step_name, odb_path))

    step = odb.steps[step_name]
    assembly = odb.rootAssembly
    allse_history = find_allse_history(step)

    rows = []

    for i, frame in enumerate(step.frames):
        t = frame.frameValue
        allse = nearest_history_value(allse_history, t)

        Fbar, rp1U, rp2U, rp3U = extract_macro_F_from_3RP(frame, assembly, dims)
        Pbar, rp1RF, rp2RF, rp3RF = extract_macro_P_from_3RP(frame, assembly, dims)
        Cbar, J, I1, I4_star, I4_classic, I5_classic = compute_macro_tensors(Fbar, H, a0)

        W_hom = allse / float(dims["V0"])

        row = {
            "frame_id": i,
            "time": t,
            "ALLSE": allse,
            "W_hom": W_hom,

            # Fbar
            "F11": Fbar[0][0], "F12": Fbar[0][1], "F13": Fbar[0][2],
            "F21": Fbar[1][0], "F22": Fbar[1][1], "F23": Fbar[1][2],
            "F31": Fbar[2][0], "F32": Fbar[2][1], "F33": Fbar[2][2],

            # Cbar
            "C11": Cbar[0][0], "C12": Cbar[0][1], "C13": Cbar[0][2],
            "C21": Cbar[1][0], "C22": Cbar[1][1], "C23": Cbar[1][2],
            "C31": Cbar[2][0], "C32": Cbar[2][1], "C33": Cbar[2][2],

            # incompressible fitting invariants
            "J_check": J,
            "I1": I1,
            "I4_star": I4_star,
            "I4_classic": I4_classic,
            "I5_classic": I5_classic,

            # Pbar
            "P11": Pbar[0][0], "P12": Pbar[0][1], "P13": Pbar[0][2],
            "P21": Pbar[1][0], "P22": Pbar[1][1], "P23": Pbar[1][2],
            "P31": Pbar[2][0], "P32": Pbar[2][1], "P33": Pbar[2][2],

            # orientation / dispersion
            "a0x": a0[0], "a0y": a0[1], "a0z": a0[2],
            "A11": A_orient[0][0], "A12": A_orient[0][1], "A13": A_orient[0][2],
            "A21": A_orient[1][0], "A22": A_orient[1][1], "A23": A_orient[1][2],
            "A31": A_orient[2][0], "A32": A_orient[2][1], "A33": A_orient[2][2],
            "lambda1_A": lam1,
            "kappa": kappa,

            # structure tensor
            "H11": H[0][0], "H12": H[0][1], "H13": H[0][2],
            "H21": H[1][0], "H22": H[1][1], "H23": H[1][2],
            "H31": H[2][0], "H32": H[2][1], "H33": H[2][2],

            # raw RP values for debugging
            "U_RP1_1": rp1U[0], "U_RP1_2": rp1U[1], "U_RP1_3": rp1U[2],
            "U_RP2_1": rp2U[0], "U_RP2_2": rp2U[1], "U_RP2_3": rp2U[2],
            "U_RP3_1": rp3U[0], "U_RP3_2": rp3U[1], "U_RP3_3": rp3U[2],

            "RF_RP1_1": rp1RF[0], "RF_RP1_2": rp1RF[1], "RF_RP1_3": rp1RF[2],
            "RF_RP2_1": rp2RF[0], "RF_RP2_2": rp2RF[1], "RF_RP2_3": rp2RF[2],
            "RF_RP3_1": rp3RF[0], "RF_RP3_2": rp3RF[1], "RF_RP3_3": rp3RF[2],
        }

        rows.append(row)

    odb.close()
    return rows


def csv_path_from_odb_path(odb_path):
    base, _ = os.path.splitext(odb_path)
    return base + ".csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract homogenized modified-HGO data from all ODB files in a directory."
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory containing the ODB files and generator summary JSON.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional explicit path to the generator summary JSON. Defaults to auto-detect in --data-dir.",
    )
    parser.add_argument(
        "--step-name",
        default=STEP_NAME,
        help="Abaqus step name to extract.",
    )
    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    data_dir = args.data_dir
    summary_json = args.summary_json
    if summary_json is None:
        summary_json = discover_summary_json(data_dir)
    if summary_json is None:
        summary_json = SUMMARY_JSON

    if not os.path.exists(summary_json):
        raise RuntimeError("Summary JSON not found: %s" % summary_json)

    A_orient, a0, lam1, kappa, H = read_orientation_tensor(summary_json)
    dims = read_geometry_from_summary(summary_json)
    odb_paths = find_odb_paths(data_dir)

    for odb_path in odb_paths:
        rows = extract_rows_from_odb(odb_path, args.step_name, A_orient, a0, lam1, kappa, H, dims)
        csv_out = csv_path_from_odb_path(odb_path)

        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print("Done with: %s" % os.path.basename(odb_path))
        print("CSV saved to: %s" % csv_out)

    print("Preferred direction a0 = (%.6f, %.6f, %.6f)" % a0)
    print("Estimated dispersion kappa = %.6f" % kappa)
    print(
        "Reference dimensions: Lx = %.6f, Ly = %.6f, Lz = %.6f"
        % (dims["LX"], dims["LY"], dims["LZ"])
    )


if __name__ == "__main__":
    main()
