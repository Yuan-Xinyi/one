"""Scene / arm helpers shared by the four viewer scripts in this folder.

Kept separate from `_shared.py` so the matplotlib figure scripts do NOT
have to import the heavy ONE viewer libs.
"""
from __future__ import annotations

import builtins
from typing import Tuple

import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from Yuan.flow_connectivity.fr3_with_pen import attach_pen_visual, make_fr3_with_pen


# Default alpha values used across viewer scripts.
ARC_GHOST_ALPHA = 0.12     # transparent ghosts along the SMM arc
START_GHOST_ALPHA = 0.30   # slightly stronger ghost at q_start
ACTIVE_ALPHA = 0.95        # opaque animator arm
HIDDEN_ALPHA = 0.0

# Tasks are extended to 1.5 m so the rollout always partially fails,
# but rendering the whole extension makes the visible line shoot past
# the workspace. Cap what we draw so the line stays close to where the
# rollout actually went.
MAX_VISIBLE_TASK_LEN = 1.0   # m


def make_world(task_path: np.ndarray) -> ovw.World:
    """Build a ONE World, parked at a fixed camera looking at the task path."""
    cam_idx = max(0, min(len(task_path) // 4, len(task_path) - 1))
    cam_focus = tuple(float(x) for x in task_path[cam_idx])
    cam_pos = (cam_focus[0] + 1.4, cam_focus[1] - 1.7, cam_focus[2] + 0.9)
    base = ovw.World(cam_pos=cam_pos, cam_lookat_pos=cam_focus,
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)
    return base


def add_task_path(base: ovw.World, task_path: np.ndarray,
                  plane_normal: np.ndarray,
                  max_length: float | None = MAX_VISIBLE_TASK_LEN,
                  draw_plane: bool = True) -> None:
    """Draw the task line, the paper plane, start (green) and end (red) markers.

    If max_length is set, the visible line is truncated to that many meters
    from the start so the rendered geometry stays close to the rollout region
    instead of the full 1.5 m extension used internally.

    If draw_plane is False the gray paper plane is skipped (line + markers
    only).
    """
    if max_length is not None and len(task_path) > 1:
        seg = np.linalg.norm(np.diff(task_path, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        if cum[-1] > max_length:
            idx = int(np.searchsorted(cum, max_length, side='right'))
            idx = min(max(idx, 2), len(task_path))
            task_path = task_path[:idx]
    if draw_plane:
        plane_center = task_path.mean(axis=0).astype(np.float32)
        path_len = float(np.linalg.norm(task_path[-1] - task_path[0]))
        plane_size = max(0.6, path_len + 0.3)
        ossop.plane(pos=tuple(plane_center),
                    normal=tuple(plane_normal.astype(np.float32)),
                    size=(plane_size, plane_size), thickness=2e-3,
                    rgb=(0.82, 0.82, 0.86), alpha=0.25).attach_to(base.scene)
    segs = np.stack([task_path[:-1], task_path[1:]], axis=1)
    ossop.linsegs(segs=segs, radius=0.0015,
                  srgbs=np.array([0.08, 0.08, 0.08]),
                  alpha=0.75).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[0]), radius=0.012,
                 rgb=(0.05, 0.65, 0.20), alpha=0.95).attach_to(base.scene)


def set_arm_alpha(arm, pen_pair, value: float) -> None:
    """Toggle an arm and its pen shaft/tip primitives together."""
    arm.alpha = value
    if pen_pair is not None:
        shaft, tip = pen_pair
        if shaft is not None:
            shaft.alpha = value
        if tip is not None:
            tip.alpha = value


def make_ghost_arm(base: ovw.World, q: np.ndarray,
                   rgb: Tuple[float, float, float] | None,
                   alpha: float) -> tuple:
    """Spawn a static transparent arm posed at q. Returns (arm, pen_pair).

    If `rgb` is None the FR3 keeps its default Franka renderer color
    (only `alpha` is applied).
    """
    origin = np.zeros(3, dtype=np.float32)
    arm, _ = make_fr3_with_pen(pos=origin)
    arm.attach_to(base.scene)
    if rgb is not None:
        arm.rgb = rgb
    arm.alpha = alpha
    pen_kwargs = {'alpha': alpha}
    if rgb is not None:
        pen_kwargs['rgb'] = rgb
    pen_pair = attach_pen_visual(arm, **pen_kwargs)
    arm.fk(q.astype(np.float32))
    return arm, pen_pair


def make_animator_arm(base: ovw.World, q0: np.ndarray,
                      rgb: Tuple[float, float, float],
                      alpha: float = ACTIVE_ALPHA) -> tuple:
    """Spawn an opaque arm that will be advanced via fk() during animation."""
    origin = np.zeros(3, dtype=np.float32)
    arm, _ = make_fr3_with_pen(pos=origin)
    arm.attach_to(base.scene)
    arm.rgb = rgb
    pen_pair = attach_pen_visual(arm, rgb=rgb, alpha=alpha)
    set_arm_alpha(arm, pen_pair, alpha)
    arm.fk(q0.astype(np.float32))
    return arm, pen_pair


def sample_arc_indices(traj_len: int, n: int) -> np.ndarray:
    """Evenly spaced ints in [0, traj_len-1]; n is clamped to traj_len."""
    n = max(1, min(int(n), int(traj_len)))
    return np.linspace(0, traj_len - 1, n).astype(int)
