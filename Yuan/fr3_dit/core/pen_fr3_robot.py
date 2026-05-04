"""FR3 + pen end-effector wrappers — compatibility layer over the ``one`` stack.

The original implementation here was a thin wrapper around the ``wrs`` Franka
Research 3 robot. This file now adapts ``one``'s FR3 + the shared
``one.robots.manipulators.franka.fr3_pen`` module to the legacy wrs-flavored
API the rest of ``fr3_dit`` was written against:

  - ``PenFrankaResearch3`` exposes ``goto_given_conf``, ``fk``, ``ik``, a
    ``manipulator.gl_tcp_pos/gl_tcp_rotmat`` view, and a chainable
    ``gen_meshmodel(rgb, alpha, toggle_tcp_frame).attach_to(scene_or_world)``.
  - ``PenFrankaResearch3GPU(device)`` exposes ``self.robot`` with batched
    ``fk_batch(q) -> (p_tcp, R_tcp)`` and a ``jnt_ranges`` ``(7, 2)`` tensor.

These wrappers preserve the existing call sites verbatim and add no
algorithmic behavior of their own — the kinematics / TCP definition all live in
the shared module.
"""
from __future__ import annotations

import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen import (
    HAND_TCP_OFFSET,
    PEN_LENGTH,
    BatchedFR3Kinematics,
    attach_pen_visual,
    make_fr3_with_pen,
)


# Re-exported for legacy ``from fr3_dit.core.pen_fr3_robot import PEN_LENGTH``.
__all__ = [
    "PEN_LENGTH",
    "HAND_TCP_OFFSET",
    "PenFrankaResearch3",
    "PenFrankaResearch3GPU",
]


# ---------------------------------------------------------------------------
# GPU-side wrapper: just expose BatchedFR3Kinematics as ``self.robot``
# ---------------------------------------------------------------------------
class PenFrankaResearch3GPU:
    """Backward-compatible holder for the torch batched FR3 chain.

    ``self.robot`` is a ``BatchedFR3Kinematics`` with the FR3 + Franka hand +
    pen TCP offset baked in. Provides ``fk_batch(q) -> (p_tcp, R_tcp)`` and
    ``jnt_ranges`` matching the legacy interface.
    """

    def __init__(self, device, dtype=torch.float32):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.robot = BatchedFR3Kinematics(
            device=self.device,
            dtype=dtype,
            tcp_offset=HAND_TCP_OFFSET + PEN_LENGTH,
        )


# ---------------------------------------------------------------------------
# CPU-side wrapper around one's FR3
# ---------------------------------------------------------------------------
class _ManipulatorView:
    """Read-only ``arm.manipulator``-style view into one's FR3 TCP state."""

    __slots__ = ("_arm",)

    def __init__(self, arm):
        self._arm = arm

    @property
    def gl_tcp_pos(self) -> np.ndarray:
        return np.asarray(self._arm.gl_tcp_tf[:3, 3], dtype=np.float32).copy()

    @property
    def gl_tcp_rotmat(self) -> np.ndarray:
        return np.asarray(self._arm.gl_tcp_tf[:3, :3], dtype=np.float32).copy()

    @property
    def gl_flange_pos(self) -> np.ndarray:
        return np.asarray(self._arm.gl_flange_tf[:3, 3], dtype=np.float32).copy()

    @property
    def gl_flange_rotmat(self) -> np.ndarray:
        return np.asarray(self._arm.gl_flange_tf[:3, :3], dtype=np.float32).copy()


class _PenSnapshot:
    """Lazy snapshot returned by ``gen_meshmodel`` / ``gen_stickmodel``.

    Defers arm construction to ``attach_to`` so each snapshot is an
    independent visual pose. Constructing the arm at attach time means
    subsequent ``goto_given_conf`` calls on the parent don't mutate already-
    attached snapshots — matching wrs's semantics.
    """

    def __init__(self, q, root_rotmat, root_pos,
                 rgb=None, alpha=None, toggle_tcp_frame=False):
        self._q = np.asarray(q, dtype=np.float32).copy()
        self._root_rotmat = None if root_rotmat is None else np.asarray(root_rotmat, dtype=np.float32).copy()
        self._root_pos = None if root_pos is None else np.asarray(root_pos, dtype=np.float32).copy()
        self._rgb = None if rgb is None else np.asarray(rgb, dtype=np.float32).copy()
        self._alpha = None if alpha is None else float(alpha)
        self._toggle_tcp_frame = bool(toggle_tcp_frame)

    def attach_to(self, target):
        # Accept either a viewer ``World`` (with ``.scene``) or a ``Scene`` directly.
        scene = target.scene if hasattr(target, "scene") else target
        snap_arm, _ = make_fr3_with_pen(rotmat=self._root_rotmat, pos=self._root_pos)
        snap_arm.fk(self._q)
        pen_rgb = tuple(self._rgb) if self._rgb is not None else (0.15, 0.15, 0.15)
        pen_alpha = self._alpha if self._alpha is not None else 0.95
        attach_pen_visual(snap_arm, rgb=pen_rgb, alpha=pen_alpha)
        if self._rgb is not None:
            snap_arm.rgb = list(self._rgb)
        if self._alpha is not None:
            snap_arm.alpha = self._alpha
        snap_arm.attach_to(scene)
        if self._toggle_tcp_frame:
            snap_arm.toggle_tcp()
        return snap_arm


class PenFrankaResearch3:
    """Drop-in stand-in for the legacy wrs FrankaResearch3 + pen.

    Wraps ``one``'s FR3 with the Franka Hand engaged and the TCP shifted to
    the pen tip. The constructor mirrors the wrs signature; ``name`` and
    ``enable_cc`` are accepted for compatibility but unused (use
    ``FR3SphereCollision`` for collision queries).
    """

    def __init__(self, pos=None, rotmat=None, name: str = "pen_franka_research_3",
                 enable_cc: bool = True):
        self.name = name
        self.enable_cc = bool(enable_cc)
        self._root_pos = None if pos is None else np.asarray(pos, dtype=np.float32)
        self._root_rotmat = None if rotmat is None else np.asarray(rotmat, dtype=np.float32)
        self.arm, self.hand = make_fr3_with_pen(rotmat=rotmat, pos=pos)
        attach_pen_visual(self.arm)
        self._manipulator_view = _ManipulatorView(self.arm)
        self._collision_checker = None  # lazy: built on first is_collided() call

    # --- forward kinematics -------------------------------------------------
    def goto_given_conf(self, jnt_values) -> None:
        self.arm.fk(np.asarray(jnt_values, dtype=np.float32))

    def fk(self, jnt_values=None):
        """Match wrs API: returns ``(tcp_pos, tcp_rotmat)`` after FK."""
        if jnt_values is not None:
            self.arm.fk(np.asarray(jnt_values, dtype=np.float32))
        return (
            np.asarray(self.arm.gl_tcp_tf[:3, 3], dtype=np.float32).copy(),
            np.asarray(self.arm.gl_tcp_tf[:3, :3], dtype=np.float32).copy(),
        )

    # --- inverse kinematics -------------------------------------------------
    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None,
           option: str = "single"):
        """Single-solution IK closest to ``seed_jnt_values`` (or current qs).

        Returns a ``(7,)`` ``np.float32`` array on success, ``None`` on failure.
        ``option`` is accepted for wrs compatibility — only ``"single"``
        (nearest-to-seed) is supported here; use ``ik_multi`` for multiple.
        """
        del option
        sol = self.arm.ik_tcp_nearest(
            tgt_rotmat=np.asarray(tgt_rotmat, dtype=np.float32),
            tgt_pos=np.asarray(tgt_pos, dtype=np.float32),
            ref_qs=None if seed_jnt_values is None else np.asarray(seed_jnt_values, dtype=np.float32),
        )
        if sol is None:
            return None
        return np.asarray(sol, dtype=np.float32)

    def ik_multi(self, tgt_pos, tgt_rotmat, max_solutions: int = 8):
        """Multi-solution IK — wrs ``ik`` with multiple candidates."""
        sols = self.arm.ik_tcp(
            tgt_rotmat=np.asarray(tgt_rotmat, dtype=np.float32),
            tgt_pos=np.asarray(tgt_pos, dtype=np.float32),
            max_solutions=max_solutions,
        )
        if sols is None:
            return []
        return [np.asarray(s, dtype=np.float32) for s in sols]

    # --- joint-space helpers (wrs compat) -----------------------------------
    @property
    def jnt_ranges(self) -> np.ndarray:
        """``(7, 2)`` joint limits ``[lower, upper]``."""
        mask = self.arm._compiled.active_jnt_ids_mask
        ids = [i for i, m in enumerate(mask) if m]
        return np.array(
            [[self.arm.structure.jnts[i].lmt_lo,
              self.arm.structure.jnts[i].lmt_up] for i in ids],
            dtype=np.float32,
        )

    def rand_conf(self) -> np.ndarray:
        """Uniform random joint configuration within limits."""
        ranges = self.jnt_ranges
        u = np.random.rand(ranges.shape[0]).astype(np.float32)
        return ranges[:, 0] + u * (ranges[:, 1] - ranges[:, 0])

    def are_jnts_in_ranges(self, jnt_values) -> bool:
        """True iff ``jnt_values`` lies within the joint limits."""
        q = np.asarray(jnt_values, dtype=np.float32)
        ranges = self.jnt_ranges
        return bool(((q >= ranges[:, 0]) & (q <= ranges[:, 1])).all())

    def is_collided(self) -> bool:
        """Sphere self-collision check at the current pose (FR3 arm only).

        Note: this approximates the wrs ``is_collided`` and only considers the
        bare-arm sphere set — pen + hand spheres are not included. Returns
        ``True`` when any sphere pair penetrates beyond the default 0 m margin.
        """
        if self._collision_checker is None:
            from one.robots.manipulators.franka.fr3.sphere_collision import (
                FR3SphereCollision,
            )
            self._collision_checker = FR3SphereCollision(device="cpu")
        link_tfs = torch.from_numpy(self.arm.gl_lnk_tfarr[:8]).float().unsqueeze(0)
        return bool(self._collision_checker.is_collided(link_tfs)[0].item())

    # --- visualization ------------------------------------------------------
    @property
    def manipulator(self):
        return self._manipulator_view

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame: bool = False,
                      toggle_jnt_frames: bool = False, toggle_cdprim: bool = False,
                      toggle_cdmesh: bool = False) -> _PenSnapshot:
        """Snapshot of the current pose, attachable to a scene/world."""
        del toggle_jnt_frames, toggle_cdprim, toggle_cdmesh  # legacy no-ops
        return _PenSnapshot(
            q=self.arm.qs.copy(),
            root_rotmat=self._root_rotmat,
            root_pos=self._root_pos,
            rgb=rgb,
            alpha=alpha,
            toggle_tcp_frame=toggle_tcp_frame,
        )

    def gen_stickmodel(self, toggle_tcp_frame: bool = False,
                       toggle_jnt_frames: bool = False) -> _PenSnapshot:
        """Compatibility alias — one renders mesh links by default."""
        del toggle_jnt_frames
        return _PenSnapshot(
            q=self.arm.qs.copy(),
            root_rotmat=self._root_rotmat,
            root_pos=self._root_pos,
            rgb=None,
            alpha=None,
            toggle_tcp_frame=toggle_tcp_frame,
        )


if __name__ == "__main__":
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=[2.0, -1.8, 1.2], cam_lookat_pos=[0.2, 0.0, 0.4])
    ossop.frame().attach_to(base.scene)

    robot = PenFrankaResearch3(name="pen", enable_cc=True)
    robot.gen_meshmodel(alpha=0.6, toggle_tcp_frame=True).attach_to(base)

    print(f"[pen-fr3] pen_length = {PEN_LENGTH:.3f} m")
    print(f"[pen-fr3] tcp_pos    = {np.array2string(robot.manipulator.gl_tcp_pos, precision=4, suppress_small=True)}")
    builtins.base = base
    base.run()
