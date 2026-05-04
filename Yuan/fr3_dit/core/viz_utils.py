"""Animation helpers for fr3_dit visualizers, ported to ``one``'s pyglet viewer.

Original wrs-based versions detached/re-created mesh models every frame; with
``one`` we attach each robot's arm once and just update its joint config via
``arm.fk(q)`` per tick. Coordinate frames similarly stay attached and have
their pose updated in place.
"""
from __future__ import annotations

import numpy as np

import one.scene.scene_object_primitive as ossop


def _scene_of(target):
    return target.scene if hasattr(target, "scene") else target


def visualize_anime_path(world, robot, path, frame_delay: float = 0.2):
    """Animate a single ``PenFrankaResearch3`` along ``path`` (T, 7).

    Attaches the arm + a TCP coordinate frame to ``world.scene`` and drives
    them via ``world.schedule_interval``. Loops indefinitely; closing the
    window stops it.
    """
    scene = _scene_of(world)
    arm = robot.arm
    arm.attach_to(scene)
    frame_obj = ossop.frame()
    frame_obj.attach_to(scene)

    path = np.asarray(path, dtype=np.float32)
    state = {"counter": 0}

    def update(dt):
        if state["counter"] >= len(path):
            state["counter"] = 0
        conf = path[state["counter"]]
        arm.fk(conf)
        frame_obj.set_rotmat_pos(
            rotmat=arm.gl_tcp_tf[:3, :3], pos=arm.gl_tcp_tf[:3, 3]
        )
        state["counter"] += 1

    world.schedule_interval_after(update, delay=1.0, interval=float(frame_delay))
    world.run()


def visualize_anime_dual(world, entries, frame_delay: float = 0.05):
    """Animate multiple robots in lock-step, each with its own trajectory.

    ``entries`` is a list of dicts::

        {"robot": <PenFrankaResearch3>, "path": np.ndarray (T, 7),
         "rgb": (3,) or None, "alpha": float, "name": str (optional)}

    Trajectories are right-padded with their last frame so the animation runs
    for ``max(len(path))`` steps; the loop restarts when it reaches the end.
    """
    if not entries:
        raise ValueError("entries must be non-empty")

    scene = _scene_of(world)
    paths = []
    for e in entries:
        p = np.asarray(e["path"], dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 7:
            raise ValueError(f"path shape must be (T, 7), got {p.shape}")
        paths.append(p)
    n_frames = max(p.shape[0] for p in paths)

    arms = []
    for e in entries:
        arm = e["robot"].arm
        rgb = e.get("rgb")
        alpha = e.get("alpha")
        if rgb is not None:
            arm.rgb = list(np.asarray(rgb, dtype=np.float32))
        if alpha is not None:
            arm.alpha = float(alpha)
        arm.attach_to(scene)
        arms.append(arm)

    state = {"counter": 0}

    def update(dt):
        if state["counter"] >= n_frames:
            state["counter"] = 0
        for arm, p in zip(arms, paths):
            idx = min(state["counter"], p.shape[0] - 1)
            arm.fk(p[idx])
        state["counter"] += 1

    world.schedule_interval_after(update, delay=1.0, interval=float(frame_delay))
    world.run()
