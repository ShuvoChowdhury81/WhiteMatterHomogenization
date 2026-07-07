#!/usr/bin/env python3
"""Generation-only dual-target prototype for fixed-theta parallel fibers.

This prototype is intentionally narrower than the production generator:
- no Abaqus launch
- no INP export requirement for testing
- fixed-theta / parallel-fiber packing logic for the current case_0938 study

Goal:
- satisfy target volume fraction and target mean diameter simultaneously
- keep a gamma-inspired diameter ranking while solving an exact per-case plan
  before packing
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


CONFIG_ENV_VAR = "BRAIN_WM_RVE_CONFIG_JSON"

# Defaults mirror the current case_0938 geometry study assumptions.
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
FIXED_THETA_DEG = 0.0

MEAN_DIAMETER_REL_TOL = 0.03
VF_ABS_TOL = 0.005
MAX_PLANNED_FIBERS = 6000
MAX_PLACEMENT_TRIES_PER_FIBER = 2500
MAX_PACKING_RESTARTS = 6


@dataclass(frozen=True)
class PlannedFiber:
    diameter_um: float
    straightness_ps: float
    arc_length_um: float
    approx_volume_um3: float


@dataclass(frozen=True)
class PackedFiber:
    diameter_um: float
    straightness_ps: float
    y_um: float
    z_um: float
    radius_um: float
    arc_length_um: float
    approx_volume_um3: float


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
    global STRAIGHTNESS_BETA_A, STRAIGHTNESS_BETA_B, FIXED_STRAIGHTNESS_PS, FIXED_THETA_DEG
    global MAX_PLANNED_FIBERS, MAX_PLACEMENT_TRIES_PER_FIBER, MAX_PACKING_RESTARTS

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
    if "GAMMA_SHAPE_K" in config:
        GAMMA_SHAPE_K = float(config["GAMMA_SHAPE_K"])
    if "GAMMA_SCALE_S" in config:
        GAMMA_SCALE_S = float(config["GAMMA_SCALE_S"])
    if "DIAMETER_MIN_UM" in config:
        DIAMETER_MIN_UM = float(config["DIAMETER_MIN_UM"])
    if "DIAMETER_MAX_UM" in config:
        DIAMETER_MAX_UM = float(config["DIAMETER_MAX_UM"])
    if "STRAIGHTNESS_BETA_A" in config:
        STRAIGHTNESS_BETA_A = float(config["STRAIGHTNESS_BETA_A"])
    if "STRAIGHTNESS_BETA_B" in config:
        STRAIGHTNESS_BETA_B = float(config["STRAIGHTNESS_BETA_B"])
    if "FIXED_STRAIGHTNESS_PS" in config:
        raw_value = config["FIXED_STRAIGHTNESS_PS"]
        FIXED_STRAIGHTNESS_PS = None if raw_value is None else float(raw_value)
    if "FIXED_THETA_DEG" in config:
        FIXED_THETA_DEG = float(config["FIXED_THETA_DEG"])
    if "MAX_PLANNED_FIBERS" in config:
        MAX_PLANNED_FIBERS = int(config["MAX_PLANNED_FIBERS"])
    if "MAX_PLACEMENT_TRIES_PER_FIBER" in config:
        MAX_PLACEMENT_TRIES_PER_FIBER = int(config["MAX_PLACEMENT_TRIES_PER_FIBER"])
    if "MAX_PACKING_RESTARTS" in config:
        MAX_PACKING_RESTARTS = int(config["MAX_PACKING_RESTARTS"])


def cube_edge_um(bounds_min: np.ndarray, bounds_max: np.ndarray) -> float:
    return float(bounds_max[0] - bounds_min[0])


def cube_volume_um3(bounds_min: np.ndarray, bounds_max: np.ndarray) -> float:
    size = bounds_max - bounds_min
    return float(size[0] * size[1] * size[2])


def solve_scale_for_truncated_mean(
    target_mean_um: float,
    shape_k: float,
    lower_um: float,
    upper_um: float,
) -> float:
    def truncated_mean(scale_s: float) -> float:
        xs = np.linspace(lower_um, upper_um, 6000, dtype=float)
        pdf = (
            np.power(xs, shape_k - 1.0)
            * np.exp(-xs / scale_s)
            / (math.gamma(shape_k) * (scale_s ** shape_k))
        )
        norm = np.trapezoid(pdf, xs)
        return float(np.trapezoid(xs * pdf, xs) / norm)

    lo = 1.0e-4
    hi = max(target_mean_um / max(shape_k, 1.0e-12), 1.0e-3)
    while truncated_mean(hi) < target_mean_um:
        hi *= 1.5
        if hi > 20.0:
            raise RuntimeError("Could not bracket a gamma scale for the target mean diameter.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if truncated_mean(mid) < target_mean_um:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_truncated_gamma_pool(
    rng: np.random.Generator,
    count: int,
    shape_k: float,
    scale_s: float,
    lower_um: float,
    upper_um: float,
) -> np.ndarray:
    values: List[float] = []
    while len(values) < count:
        remaining = count - len(values)
        batch = rng.gamma(shape=shape_k, scale=scale_s, size=max(remaining * 4, 256))
        batch = batch[(batch >= lower_um) & (batch <= upper_um)]
        values.extend(float(v) for v in batch[:remaining])
    return np.array(values[:count], dtype=float)


def sample_ps_pool(rng: np.random.Generator, count: int, a: float, b: float) -> np.ndarray:
    if FIXED_STRAIGHTNESS_PS is not None:
        return np.full(count, max(float(FIXED_STRAIGHTNESS_PS), 1.0e-6), dtype=float)
    values = np.maximum(rng.beta(a, b, size=count), 1.0e-6)
    return np.array(values, dtype=float)


def mean_for_shift(raw: np.ndarray, scale_a: float, shift_b: float, lower: float, upper: float) -> float:
    transformed = np.clip(scale_a * raw + shift_b, lower, upper)
    return float(np.mean(transformed))


def solve_shift_for_mean(
    raw: np.ndarray,
    scale_a: float,
    target_mean: float,
    lower: float,
    upper: float,
) -> float:
    lo = lower - scale_a * float(np.max(raw)) - 2.0 * upper
    hi = upper - scale_a * float(np.min(raw)) + 2.0 * upper
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        current_mean = mean_for_shift(raw, scale_a, mid, lower, upper)
        if current_mean < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def transformed_diameters(
    raw: np.ndarray,
    scale_a: float,
    shift_b: float,
    lower: float,
    upper: float,
) -> np.ndarray:
    return np.clip(scale_a * raw + shift_b, lower, upper)


def target_volume_for_case(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
) -> float:
    return target_vf * cube_volume_um3(bounds_min, bounds_max)


def planned_volume_for_diameters(diameters: np.ndarray, arc_lengths: np.ndarray) -> float:
    return float(np.sum((math.pi / 4.0) * np.square(diameters) * arc_lengths))


def estimate_count_guess(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
    target_mean_diameter_um: float,
    mean_ps: float,
) -> int:
    edge = cube_edge_um(bounds_min, bounds_max)
    mean_arc_length = edge / max(mean_ps, 1.0e-12)
    mean_fiber_volume = (math.pi / 4.0) * (target_mean_diameter_um ** 2) * mean_arc_length
    return max(12, int(round(target_volume_for_case(bounds_min, bounds_max, target_vf) / mean_fiber_volume)))


def solve_plan_for_count(
    raw_diameters: np.ndarray,
    ps_values: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
    target_mean_diameter_um: float,
    lower_um: float,
    upper_um: float,
) -> Optional[dict]:
    edge = cube_edge_um(bounds_min, bounds_max)
    arc_lengths = edge / np.maximum(ps_values, 1.0e-12)
    target_volume = target_volume_for_case(bounds_min, bounds_max, target_vf)

    shift_zero = solve_shift_for_mean(raw_diameters, 0.0, target_mean_diameter_um, lower_um, upper_um)
    diam0 = transformed_diameters(raw_diameters, 0.0, shift_zero, lower_um, upper_um)
    vol0 = planned_volume_for_diameters(diam0, arc_lengths)
    if vol0 > target_volume + 1.0e-9:
        return None

    if abs(vol0 - target_volume) <= 1.0e-9:
        return {
            "diameters": diam0,
            "arc_lengths": arc_lengths,
            "ps_values": ps_values,
            "scale_a": 0.0,
            "shift_b": shift_zero,
            "planned_volume_um3": vol0,
        }

    def eval_for_scale(scale_a: float) -> Tuple[np.ndarray, float, float]:
        shift_b = solve_shift_for_mean(raw_diameters, scale_a, target_mean_diameter_um, lower_um, upper_um)
        diam = transformed_diameters(raw_diameters, scale_a, shift_b, lower_um, upper_um)
        volume = planned_volume_for_diameters(diam, arc_lengths)
        return diam, shift_b, volume

    scale_hi = 1.0
    prev_volume = vol0
    best_hi = None
    for _ in range(30):
        diam_hi, shift_hi, vol_hi = eval_for_scale(scale_hi)
        if vol_hi >= target_volume:
            best_hi = (diam_hi, shift_hi, vol_hi)
            break
        if vol_hi <= prev_volume + 1.0e-9:
            return None
        prev_volume = vol_hi
        scale_hi *= 1.6

    if best_hi is None:
        return None

    scale_lo = 0.0
    best = None
    for _ in range(60):
        scale_mid = 0.5 * (scale_lo + scale_hi)
        diam_mid, shift_mid, vol_mid = eval_for_scale(scale_mid)
        if best is None or abs(vol_mid - target_volume) < abs(best["planned_volume_um3"] - target_volume):
            best = {
                "diameters": diam_mid,
                "arc_lengths": arc_lengths,
                "ps_values": ps_values,
                "scale_a": scale_mid,
                "shift_b": shift_mid,
                "planned_volume_um3": vol_mid,
            }
        if vol_mid < target_volume:
            scale_lo = scale_mid
        else:
            scale_hi = scale_mid
    return best


def choose_feasible_plan(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
    target_mean_diameter_um: float,
    seed: int,
    shape_k: float,
    lower_um: float,
    upper_um: float,
    beta_a: float,
    beta_b: float,
) -> dict:
    mean_ps = beta_a / (beta_a + beta_b)
    count_guess = estimate_count_guess(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        target_vf=target_vf,
        target_mean_diameter_um=target_mean_diameter_um,
        mean_ps=mean_ps,
    )
    low_n = max(12, int(math.floor(0.70 * count_guess)))
    high_n = min(MAX_PLANNED_FIBERS, int(math.ceil(1.35 * count_guess)) + 24)

    raw_scale = solve_scale_for_truncated_mean(target_mean_diameter_um, shape_k, lower_um, upper_um)
    plan_rng = np.random.default_rng(int(seed) + 1_000_003)
    raw_pool = sample_truncated_gamma_pool(plan_rng, high_n, shape_k, raw_scale, lower_um, upper_um)
    ps_pool = sample_ps_pool(plan_rng, high_n, beta_a, beta_b)

    feasible: List[dict] = []
    for count_n in range(low_n, high_n + 1):
        raw_subset = np.array(raw_pool[:count_n], dtype=float)
        plan = solve_plan_for_count(
            raw_diameters=raw_subset,
            ps_values=ps_pool[:count_n],
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            target_vf=target_vf,
            target_mean_diameter_um=target_mean_diameter_um,
            lower_um=lower_um,
            upper_um=upper_um,
        )
        if plan is None:
            continue
        diameters = np.array(plan["diameters"], dtype=float)
        raw_std = float(np.std(raw_subset))
        transformed_std = float(np.std(diameters))
        scale_a = float(plan["scale_a"])
        shift_b = float(plan["shift_b"])
        feasible.append(
            {
                **plan,
                "count_n": count_n,
                "raw_scale_s_um": raw_scale,
                "raw_diameter_mean_um": float(np.mean(raw_subset)),
                "raw_diameter_std_um": raw_std,
                "diameter_max_um": float(np.max(diameters)),
                "diameter_std_um": transformed_std,
                "scale_preservation_error": abs(math.log(max(scale_a, 1.0e-12))),
                "shift_relative_error": abs(shift_b) / max(target_mean_diameter_um, 1.0e-12),
                "std_relative_error": abs(transformed_std - raw_std) / max(raw_std, 1.0e-12),
                "volume_error_um3": abs(float(plan["planned_volume_um3"]) - target_volume_for_case(bounds_min, bounds_max, target_vf)),
            }
        )

    if not feasible:
        raise RuntimeError("Could not find a feasible planned fiber set for the requested targets.")

    feasible.sort(
        key=lambda item: (
            float(item["scale_preservation_error"]),
            float(item["shift_relative_error"]),
            float(item["std_relative_error"]),
            float(item["volume_error_um3"]),
            abs(int(item["count_n"]) - count_guess),
        )
    )
    return feasible[0]


def center_margin(radius_um: float) -> float:
    return min(0.4 * radius_um, 0.2)


def try_pack_plan(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    planned_diameters: np.ndarray,
    planned_ps: np.ndarray,
    min_gap_um: float,
    seed: int,
) -> Tuple[List[PackedFiber], int]:
    edge = cube_edge_um(bounds_min, bounds_max)
    order = np.argsort(planned_diameters)[::-1]
    ordered_d = planned_diameters[order]
    ordered_ps = planned_ps[order]

    for restart in range(MAX_PACKING_RESTARTS):
        rng = np.random.default_rng(int(seed) + 10_000 * restart)
        packed: List[PackedFiber] = []
        centers: List[Tuple[float, float, float]] = []
        success = True

        for diameter_um, ps_val in zip(ordered_d, ordered_ps):
            radius_um = 0.5 * float(diameter_um)
            margin = center_margin(radius_um)
            y_lo = float(bounds_min[1] + margin)
            y_hi = float(bounds_max[1] - margin)
            z_lo = float(bounds_min[2] + margin)
            z_hi = float(bounds_max[2] - margin)

            placed = False
            for _ in range(MAX_PLACEMENT_TRIES_PER_FIBER):
                y_um = float(rng.uniform(y_lo, y_hi))
                z_um = float(rng.uniform(z_lo, z_hi))
                ok = True
                for old_y, old_z, old_radius in centers:
                    required = radius_um + old_radius + float(min_gap_um)
                    dist2 = (y_um - old_y) ** 2 + (z_um - old_z) ** 2
                    if dist2 < required ** 2:
                        ok = False
                        break
                if not ok:
                    continue
                arc_length_um = edge / max(float(ps_val), 1.0e-12)
                approx_volume_um3 = (math.pi / 4.0) * (float(diameter_um) ** 2) * arc_length_um
                packed.append(
                    PackedFiber(
                        diameter_um=float(diameter_um),
                        straightness_ps=float(ps_val),
                        y_um=y_um,
                        z_um=z_um,
                        radius_um=radius_um,
                        arc_length_um=arc_length_um,
                        approx_volume_um3=approx_volume_um3,
                    )
                )
                centers.append((y_um, z_um, radius_um))
                placed = True
                break

            if not placed:
                success = False
                break

        if success:
            return packed, restart

    raise RuntimeError("Packing failed after all restart attempts.")


def build_summary(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
    target_mean_diameter_um: float,
    seed: int,
    plan: dict,
    packed: List[PackedFiber],
    restarts_used: int,
) -> dict:
    cube_volume = cube_volume_um3(bounds_min, bounds_max)
    achieved_volume = float(sum(f.approx_volume_um3 for f in packed))
    achieved_vf = achieved_volume / cube_volume
    diameters = np.array([f.diameter_um for f in packed], dtype=float)
    ps_vals = np.array([f.straightness_ps for f in packed], dtype=float)

    return {
        "generator_strategy": "planned_set_affine_mean_volume_parallel_packing",
        "generator_status": "success",
        "cube_min": bounds_min.tolist(),
        "cube_max": bounds_max.tolist(),
        "cube_center": (0.5 * (bounds_min + bounds_max)).tolist(),
        "cube_edge_um": cube_edge_um(bounds_min, bounds_max),
        "target_volume_fraction": float(target_vf),
        "achieved_approx_volume_fraction": achieved_vf,
        "vf_abs_error": achieved_vf - float(target_vf),
        "target_mean_diameter_um": float(target_mean_diameter_um),
        "diameter_mean_um": float(np.mean(diameters)),
        "diameter_std_um": float(np.std(diameters)),
        "diameter_rel_error_pct": 100.0 * (float(np.mean(diameters)) - float(target_mean_diameter_um)) / float(target_mean_diameter_um),
        "accepted_axons": len(packed),
        "planned_axons": int(plan["count_n"]),
        "packing_restarts_used": int(restarts_used),
        "planned_raw_scale_s_um": float(plan["raw_scale_s_um"]),
        "planned_affine_scale_a": float(plan["scale_a"]),
        "planned_affine_shift_b": float(plan["shift_b"]),
        "planned_volume_um3": float(plan["planned_volume_um3"]),
        "achieved_approx_axon_volume_um3": achieved_volume,
        "target_axon_volume_um3": float(target_vf) * cube_volume,
        "straightness_mean": float(np.mean(ps_vals)),
        "straightness_std": float(np.std(ps_vals)),
        "seed": int(seed),
        "min_gap_um": float(MIN_GAP_UM),
        "diameter_distribution": {
            "type": "truncated_gamma_planned_affine",
            "shape_k": float(GAMMA_SHAPE_K),
            "raw_scale_s_um": float(plan["raw_scale_s_um"]),
            "diameter_min_um": float(DIAMETER_MIN_UM),
            "diameter_max_um": float(DIAMETER_MAX_UM),
        },
        "straightness_distribution": {
            "type": "beta",
            "shape_a": float(STRAIGHTNESS_BETA_A),
            "shape_b": float(STRAIGHTNESS_BETA_B),
        },
        "fixed_theta_deg": float(FIXED_THETA_DEG),
    }


def evaluate_case(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    target_vf: float,
    target_mean_diameter_um: float,
    seed: int,
) -> Tuple[List[PackedFiber], dict]:
    if not math.isclose(float(FIXED_THETA_DEG), 0.0, abs_tol=1.0e-12):
        raise NotImplementedError(
            "This prototype currently assumes FIXED_THETA_DEG = 0 for parallel-fiber packing."
        )

    plan = choose_feasible_plan(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        target_vf=float(target_vf),
        target_mean_diameter_um=float(target_mean_diameter_um),
        seed=int(seed),
        shape_k=float(GAMMA_SHAPE_K),
        lower_um=float(DIAMETER_MIN_UM),
        upper_um=float(DIAMETER_MAX_UM),
        beta_a=float(STRAIGHTNESS_BETA_A),
        beta_b=float(STRAIGHTNESS_BETA_B),
    )
    packed, restarts_used = try_pack_plan(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        planned_diameters=np.array(plan["diameters"], dtype=float),
        planned_ps=np.array(plan["ps_values"], dtype=float),
        min_gap_um=float(MIN_GAP_UM),
        seed=int(seed),
    )
    stats = build_summary(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        target_vf=float(target_vf),
        target_mean_diameter_um=float(target_mean_diameter_um),
        seed=int(seed),
        plan=plan,
        packed=packed,
        restarts_used=restarts_used,
    )
    return packed, stats


def write_summary_json(output_dir: Path, summary_name: str, stats: dict, packed: List[PackedFiber]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    payload["fibers"] = [
        {
            "diameter_um": f.diameter_um,
            "straightness_ps": f.straightness_ps,
            "y_um": f.y_um,
            "z_um": f.z_um,
            "radius_um": f.radius_um,
            "arc_length_um": f.arc_length_um,
            "approx_volume_um3": f.approx_volume_um3,
        }
        for f in packed
    ]
    summary_path = output_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return summary_path


def main() -> None:
    config = load_runtime_config()
    apply_runtime_overrides(config)

    packed, stats = evaluate_case(
        bounds_min=np.array(CUBE_MIN, dtype=float),
        bounds_max=np.array(CUBE_MAX, dtype=float),
        target_vf=float(TARGET_VOLUME_FRACTION),
        target_mean_diameter_um=float(TARGET_MEAN_DIAMETER_UM),
        seed=int(SEED),
    )

    summary_path = write_summary_json(
        output_dir=Path(config.get("OUTPUT_DIR", ".")),
        summary_name=str(config.get("SUMMARY_JSON", "dual_target_generation_summary.json")),
        stats=stats,
        packed=packed,
    )

    print("\nSummary")
    print("-------")
    for key, value in stats.items():
        if key in ("diameter_distribution", "straightness_distribution"):
            continue
        print(f"{key}: {value}")
    print(f"\nSummary JSON : {summary_path}")
    print("Note         : Generation only. No Abaqus jobs were run.")


if __name__ == "__main__":
    main()
