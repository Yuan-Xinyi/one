"""Geometric-clean line-task sampler.

Local replacement for `v18_smm_core.get_task_target_pose` that adds a
geometry rejection filter on top of the existing line-task sampler.
The original v18 sampler can produce tasks whose 1.5 m extended line
(used internally to force partial-failure rollouts) passes very close
to the FR3 base column, which makes the ONE viewer animations look
like the line cuts through the arm.

The filter rejects a sampled task if any of these is violated:
  1. min distance from the entire 1.5 m extended line to the world
     origin (arm base) is below `min_base_clearance` (default 0.35 m);
  2. minimum z over the same line is below `min_z` (default 0.20 m);
  3. the line at the start does NOT point outward from the base
     (p0 . direction < 0): rejected so rollouts never head toward the
     base column.

Determinism: the function consumes more rng draws than the original
sampler (because of the rejection loop), so the task at "seed N" here
is NOT identical to the task at "seed N" in v18_smm_core. It IS still
deterministic per seed.

This module is the only file in 25_spring_pre/ that overrides upstream
task sampling. v18_smm_core is untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    LINE_L_RANGE, TARGET_PATH_M,
    extend_task_path, sample_line_task,
)
from Yuan.flow_connectivity.v18_data_prep import _build_R_from_normal_direction


@dataclass(frozen=True)
class CleanFilter:
    """Geometric criteria the sampled task must satisfy."""
    min_base_clearance: float = 0.35   # min distance from extended line to origin
    min_z: float = 0.20                # min height anywhere on the extended line
    require_outward_dir: bool = True   # initial direction must satisfy p0·d > 0
    max_tries: int = 60                # rejection-loop budget


def _line_geometry(path: np.ndarray) -> dict:
    """Geometry stats for the (already extended) path: closest point to origin,
       min z on the line, segment direction, and the outward-dot signal."""
    p0 = path[0]
    d = path[1] - path[0]
    d_n = float(np.linalg.norm(d))
    if d_n < 1e-9:
        return dict(d_base=0.0, min_z=float(path[:, 2].min()),
                     outward=-1.0, p0=p0, dir=np.zeros(3, dtype=np.float32))
    d_unit = (d / d_n).astype(np.float32)
    # Closest point on the segment [path[0], path[-1]] to origin.
    seg = path[-1] - path[0]
    seg_len = float(np.linalg.norm(seg))
    t = float(-np.dot(p0, seg / max(seg_len, 1e-12)))
    t_clamped = max(0.0, min(seg_len, t))
    closest = p0 + t_clamped * (seg / max(seg_len, 1e-12))
    d_base = float(np.linalg.norm(closest))
    z_min = float(path[:, 2].min())
    outward = float(np.dot(p0, d_unit))   # > 0 ⇒ initial step moves outward
    return dict(d_base=d_base, min_z=z_min, outward=outward,
                 p0=p0.astype(np.float32), dir=d_unit)


def sample_clean_line_task(rng: np.random.Generator,
                            kin: BatchedFR3Kinematics,
                            cfilter: CleanFilter = CleanFilter(),
                            l_range: tuple[float, float] = LINE_L_RANGE,
                            target_L: float = TARGET_PATH_M,
                            verbose: bool = False) -> tuple[dict, dict]:
    """Sample a v18 line task and accept only if it passes the geometry filter.

    Returns (task_dict, geom_stats). Raises RuntimeError if no valid task is
    sampled within `cfilter.max_tries` attempts.
    """
    for trial in range(cfilter.max_tries):
        task = sample_line_task(rng, kin, l_range=l_range)
        task = extend_task_path(task, target_L)
        path = task['fine_path_pts']
        g = _line_geometry(path)
        ok_clear = g['d_base'] >= cfilter.min_base_clearance
        ok_z = g['min_z'] >= cfilter.min_z
        ok_dir = (g['outward'] > 0.0) if cfilter.require_outward_dir else True
        if verbose:
            print(f'    try {trial:2d}: d_base={g["d_base"]:.2f}  '
                  f'min_z={g["min_z"]:.2f}  outward={g["outward"]:+.2f}  '
                  f'-> {"OK" if (ok_clear and ok_z and ok_dir) else "reject"}')
        if ok_clear and ok_z and ok_dir:
            return task, g
    raise RuntimeError(
        f'sample_clean_line_task: {cfilter.max_tries} rejected '
        f'(min_clear={cfilter.min_base_clearance:.2f}, '
        f'min_z={cfilter.min_z:.2f}, require_outward={cfilter.require_outward_dir})')


def get_clean_task_target_pose(seed: int,
                                kin: BatchedFR3Kinematics,
                                rng: np.random.Generator,
                                cfilter: CleanFilter = CleanFilter(),
                                verbose: bool = False
                                ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop-in replacement for v18_smm_core.get_task_target_pose with the
    geometric clean filter applied. Returns (p_tgt, R_tgt, task_dict)."""
    task, _g = sample_clean_line_task(rng, kin, cfilter=cfilter, verbose=verbose)
    path = task['fine_path_pts']
    tangent = path[1] - path[0]
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    R = _build_R_from_normal_direction(task['plane_normal'], tangent)
    return path[0].astype(np.float32), R.astype(np.float32), task
