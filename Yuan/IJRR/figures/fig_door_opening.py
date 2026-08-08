#!/usr/bin/env python3
"""Door opening: why a single-point reachability map cannot answer the question.

The task
--------
A door leaf hangs on a vertical hinge; the leaf plane is vertical (perpendicular
to the xy plane). The gripper grasps a vertical handle bar at radius
``handle_r`` from the hinge and height ``handle_z``, and then the door is pulled
open. Because the grasp is rigid, the door angle ``theta`` fixes the full 6-D
end-effector pose

    p(theta) = H + Rz(swing*theta) @ (handle_r * d0) + handle_z * z
    R(theta) = Rz(swing*theta) @ R0

so a 7-DoF FR3 has a 1-D self-motion manifold left at every point of the path.
The quantity we care about is the *achievable opening angle* ``theta_max``: how
far the door can be pulled before the arm runs into a joint limit, a singularity
(tracking failure), a self-collision, or the wall / the swinging leaf.

What this script demonstrates
----------------------------
``theta_max`` is a property of the whole trajectory, not of any single point:

    scenario "init"        same base, same controller, same start TCP pose,
                           different start joint angles (different solutions of
                           the same IK problem) -> different theta_max
    scenario "redundancy"  same base, same start joint angles, different
                           null-space resolution -> different theta_max
    scenario "base"        same door and controller, different base placement
                           -> different theta_max
    scenario "height"      same base and controller, different grasp height on
                           the leaf -> different theta_max

With the defaults in this file the door *can* be opened all the way: one start
posture, one base placement and several of the resolution laws reach the full
90 degrees. Every other choice stalls between 15 and 75 degrees, while a
point-wise reachability map reports the whole arc as reachable in all of them --
each pose on the arc does have a collision-free IK solution. The map cannot see
the difference; opening the door means optimizing base pose, grasp, start
configuration and the whole null-space trajectory jointly. (In the search that
produced this scene, 35 of 4986 combinations of those choices opened it fully.)

Rendering
---------
The panels show the real FR3 meshes, rendered with MuJoCo from the STLs in
``one/robots/...`` (see ``fr3_scene.py``); ``--render stick`` falls back to link
polylines if there is no GL backend. ``--viewer`` plays the same rollouts in a
native MuJoCo window instead, with a free camera and no plots.

``--overlay N`` replaces the animation with a single static figure: N poses of
each rollout (and of the leaf it is pulling) drawn on top of each other with
rising opacity, which is the print version of the same content. It works for
both renderers.

Usage
-----
    conda activate one
    cd /home/lqin/one
    python Yuan/IJRR/figures/fig_door_opening.py --scenario init
    python Yuan/IJRR/figures/fig_door_opening.py --scenario redundancy
    python Yuan/IJRR/figures/fig_door_opening.py --scenario base
    python Yuan/IJRR/figures/fig_door_opening.py --scenario height
    python Yuan/IJRR/figures/fig_door_opening.py --scenario init --viewer
    python Yuan/IJRR/figures/fig_door_opening.py --scenario init --save door.mp4
    python Yuan/IJRR/figures/fig_door_opening.py --scenario init --save door.png
    python Yuan/IJRR/figures/fig_door_opening.py --scenario init --overlay 5
    python Yuan/IJRR/figures/fig_door_opening.py --sweep            # static landscape

The script may also be run from any directory; it inserts the repo root on
``sys.path`` itself.
"""
from __future__ import annotations

# matplotlib MUST be imported before torch in this environment: torch pulls in
# the system libstdc++, after which matplotlib's C extension fails to find
# CXXABI_1.3.15. Keep this order.
import matplotlib                                                    # noqa: E402
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.animation import FuncAnimation                       # noqa: E402
import matplotlib.colors as mcolors                                  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection              # noqa: E402

import argparse                                                      # noqa: E402
import math                                                          # noqa: E402
import sys                                                           # noqa: E402
from dataclasses import dataclass, replace                           # noqa: E402
from pathlib import Path                                             # noqa: E402

import numpy as np                                                   # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch                                                         # noqa: E402

from one.robots.manipulators.franka.fr3.sphere_collision import (    # noqa: E402
    FR3SphereCollision,
)
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import ( # noqa: E402
    HAND_TCP_OFFSET,
    BatchedFR3Kinematics,
)

torch.set_grad_enabled(False)
DTYPE = torch.float64

# Franka Hand mount: the hand frame is the flange frame rotated -45 deg about z
# (panda_hand_joint), so flange = hand @ Rz(+45 deg).
_C45 = math.cos(math.pi / 4)
HAND_MOUNT_R = np.array([[_C45, -_C45, 0.0], [_C45, _C45, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# task definition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DoorSpec:
    """Vertical door leaf on a vertical hinge, grasped at a handle bar."""
    hinge_xy: tuple = (0.82, 0.57)      # hinge axis (world x, y) [m]
    leaf_dir_deg: float = -100.0        # closed-leaf direction in xy [deg from +x]
    width: float = 0.70                 # hinge -> free edge [m]
    height: float = 2.0                 # leaf height [m]
    z_bottom: float = 0.0               # leaf bottom [m]
    handle_r: float = 0.64              # grasp radius from the hinge [m]
    handle_z: float = 0.95              # grasp height [m]
    handle_offset: float = 0.06         # how far the bar stands off the leaf [m]
    grasp_roll_deg: float = -40.0       # approach direction, about the bar axis
    grasp_flip: float = -1.0            # +-1: the hand's 180-deg roll about z_ee
    swing: float = -1.0                 # -1: pulls toward the robot, +1: pushes away
    theta_end_deg: float = 90.0         # fully open

    @property
    def d0(self) -> np.ndarray:
        a = math.radians(self.leaf_dir_deg)
        return np.array([math.cos(a), math.sin(a), 0.0])

    @property
    def hinge(self) -> np.ndarray:
        return np.array([self.hinge_xy[0], self.hinge_xy[1], 0.0])

    @property
    def n0(self) -> np.ndarray:
        """Closed-leaf normal pointing away from the robot (robot side is -n0)."""
        n = np.cross(np.array([0.0, 0.0, 1.0]), self.d0)
        # keep the robot (world origin) on the negative side
        if np.dot(self.hinge - np.zeros(3), n) < 0.0:
            n = -n
        return n / np.linalg.norm(n)

    def leaf_frame(self, theta: np.ndarray):
        """Per-angle leaf axes. ``theta`` (T,) -> d (T,3), n (T,3)."""
        a = self.swing * np.asarray(theta, dtype=float)
        c, s = np.cos(a), np.sin(a)
        d0, n0 = self.d0, self.n0
        d = c[:, None] * d0[None] + s[:, None] * np.cross([0, 0, 1], d0)[None]
        n = c[:, None] * n0[None] + s[:, None] * np.cross([0, 0, 1], n0)[None]
        return d, n

    def grasp_path(self, theta: np.ndarray):
        """Rigid-grasp EE path. ``theta`` (T,) -> p (T,3), R (T,3,3).

        The TCP sits on the axis of the vertical handle bar, so the position is
        fixed by the door angle. The bar is a cylinder, so the *direction* the
        hand comes in from is free at grasp time: ``grasp_roll_deg`` turns the
        approach about the bar axis (0 = straight into the leaf face, i.e. along
        the leaf normal), and ``grasp_flip`` is the hand's 180-degree roll. Both
        are chosen once and then held rigidly, so the whole 6-D pose is still a
        function of theta alone; they are grasp choices exactly like the height.
        """
        d, n = self.leaf_frame(theta)
        p = self.hinge[None] + self.handle_r * d - self.handle_offset * n
        p[:, 2] = self.handle_z
        a = math.radians(self.grasp_roll_deg)
        z_hat = np.array([0.0, 0.0, 1.0])
        z_ee = math.cos(a) * n + math.sin(a) * np.cross(z_hat, n)
        y_ee = self.grasp_flip * np.cross(z_ee, z_hat)   # fingers close on the bar
        x_ee = np.cross(y_ee, z_ee)
        R_hand = np.stack([x_ee, y_ee, z_ee], axis=-1)   # columns are the axes
        # the frame above is the *hand* frame (fingers along y); the Franka Hand
        # is mounted 45 deg about z on the flange, and the kinematics here is
        # flange-based, so undo that mount rotation
        return p, R_hand @ HAND_MOUNT_R

    def leaf_corners(self, theta: float) -> np.ndarray:
        d, _ = self.leaf_frame(np.array([theta]))
        d = d[0]
        h = self.hinge
        z0, z1 = self.z_bottom, self.z_bottom + self.height
        return np.array([
            [h[0], h[1], z0],
            [h[0] + self.width * d[0], h[1] + self.width * d[1], z0],
            [h[0] + self.width * d[0], h[1] + self.width * d[1], z1],
            [h[0], h[1], z1],
        ])


@dataclass(frozen=True)
class BasePose:
    """FR3 base in the world (mobile platform: the base plate sits at z)."""
    x: float = -0.05
    y: float = 0.45
    z: float = 0.70
    yaw_deg: float = 0.0

    @property
    def R(self) -> np.ndarray:
        a = math.radians(self.yaw_deg)
        return np.array([[math.cos(a), -math.sin(a), 0.0],
                         [math.sin(a), math.cos(a), 0.0],
                         [0.0, 0.0, 1.0]])

    @property
    def t(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_base(self, p_w: np.ndarray, R_w: np.ndarray):
        """World poses (T,3)/(T,3,3) -> base frame."""
        Rb = self.R
        p_b = (p_w - self.t[None]) @ Rb
        R_b = np.einsum('ij,tjk->tik', Rb.T, R_w)
        return p_b, R_b


# --------------------------------------------------------------------------- #
# kinematics helpers
# --------------------------------------------------------------------------- #
class Arm:
    """FR3 with a Franka Hand TCP plus the sphere collision model."""

    def __init__(self):
        self.kin = BatchedFR3Kinematics(dtype=DTYPE, tcp_offset=HAND_TCP_OFFSET)
        self.coll = FR3SphereCollision(dtype=DTYPE)
        self.lo = self.kin.lmt_lo
        self.hi = self.kin.lmt_up
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.radii = self.coll.radii.reshape(-1)
        self.link_of_sphere = np.asarray(self.coll.link_indices).reshape(-1)

    # -- FK ---------------------------------------------------------------- #
    def fk(self, q: torch.Tensor):
        return self.kin.tcp_fk_jac(q)                 # p, R, J, T_last

    def spheres_world(self, q: torch.Tensor, base: BasePose):
        tfs = self.kin.link_transforms(q)
        c = self.coll.sphere_positions(tfs)           # (B,S,3) base frame
        Rb = torch.as_tensor(base.R, dtype=DTYPE)
        tb = torch.as_tensor(base.t, dtype=DTYPE)
        return c @ Rb.T + tb.view(1, 1, 3)

    def self_collision_margin(self, q: torch.Tensor) -> torch.Tensor:
        return self.coll.min_margin(self.kin.link_transforms(q))

    def limit_margin(self, q: torch.Tensor) -> torch.Tensor:
        """Smallest normalized distance to a joint limit, 1 = mid-range, 0 = limit."""
        return (1.0 - ((q - self.mid) / self.half).abs()).min(dim=-1).values

    def manipulability(self, q: torch.Tensor) -> torch.Tensor:
        _, _, J, _ = self.fk(q)
        JJt = J @ J.transpose(1, 2)
        return torch.sqrt(torch.linalg.det(JJt).clamp_min(1e-24))

    def joints_world(self, q: torch.Tensor, base: BasePose) -> np.ndarray:
        """Polyline through link0..link7 origins plus the TCP, in world (B,9,3)."""
        tfs = self.kin.link_transforms(q)
        pts = tfs[:, :, :3, 3]
        p_tcp, _, _, _ = self.fk(q)
        pts = torch.cat([pts, p_tcp.unsqueeze(1)], dim=1)
        Rb = torch.as_tensor(base.R, dtype=DTYPE)
        tb = torch.as_tensor(base.t, dtype=DTYPE)
        return (pts @ Rb.T + tb.view(1, 1, 3)).numpy()

    def tcp_world(self, q: torch.Tensor, base: BasePose):
        p, R, _, _ = self.fk(q)
        Rb = torch.as_tensor(base.R, dtype=DTYPE)
        tb = torch.as_tensor(base.t, dtype=DTYPE)
        return (p @ Rb.T + tb.view(1, 3)).numpy(), (Rb @ R).numpy()


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix -> rotation vector, (B,3,3) -> (B,3)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    axis = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                        R[:, 0, 2] - R[:, 2, 0],
                        R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    small = ang < 1e-6
    scale = torch.where(small,
                        torch.full_like(ang, 0.5),
                        ang / (2.0 * torch.sin(ang).clamp_min(1e-9)))
    out = axis * scale.unsqueeze(-1)
    # near pi the formula above degenerates; fall back to the symmetric part
    near_pi = ang > (math.pi - 1e-3)
    if bool(near_pi.any()):
        idx = torch.nonzero(near_pi).flatten()
        for i in idx.tolist():
            A = 0.5 * (R[i] + torch.eye(3, dtype=R.dtype))
            v = torch.sqrt(torch.diagonal(A).clamp_min(0.0))
            k = int(torch.argmax(v))
            v = v * torch.sign(A[k] / v[k].clamp_min(1e-9))
            out[i] = ang[i] * v / v.norm().clamp_min(1e-9)
    return out


def pose_error(p, R, p_d, R_d) -> torch.Tensor:
    """6-vector [dp; dw] taking (p,R) to (p_d,R_d), batched."""
    return torch.cat([p_d - p, so3_log(R_d @ R.transpose(1, 2))], dim=-1)


def dls_solve(J: torch.Tensor, e: torch.Tensor, lam: float,
              winv: torch.Tensor | None = None):
    """Damped (weighted) least squares step and the matching null-space projector.

    ``winv`` is the diagonal of W^-1 in the weighted least-norm solution
    ``dq = W^-1 J^T (J W^-1 J^T + lam^2 I)^-1 e``; ``None`` means W = I.
    """
    B = J.shape[0]
    Jw = J if winv is None else J * winv.unsqueeze(1)
    A = J @ Jw.transpose(1, 2) + (lam ** 2) * torch.eye(6, dtype=J.dtype).expand(B, 6, 6)
    Jsharp = Jw.transpose(1, 2) @ torch.linalg.inv(A)
    dq = (Jsharp @ e.unsqueeze(-1)).squeeze(-1)
    N = torch.eye(7, dtype=J.dtype).expand(B, 7, 7) - Jsharp @ J
    return dq, N


# --------------------------------------------------------------------------- #
# inverse kinematics (batched, random restarts)
# --------------------------------------------------------------------------- #
def solve_ik(arm: Arm, p_d: torch.Tensor, R_d: torch.Tensor, q_init: torch.Tensor,
             iters: int = 200, lam: float = 0.05, pos_tol: float = 1e-3,
             rot_tol: float = 1e-3, center_gain: float = 0.02):
    """Batched DLS IK with a mild joint-centering null-space bias."""
    q = q_init.clone()
    for _ in range(iters):
        p, R, J, _ = arm.fk(q)
        e = pose_error(p, R, p_d, R_d)
        dq, N = dls_solve(J, e, lam)
        g = -(q - arm.mid) / (arm.half ** 2)
        dq = dq + (N @ (center_gain * g).unsqueeze(-1)).squeeze(-1)
        dq = dq.clamp(-0.25, 0.25)
        q = (q + dq).clamp(arm.lo + 1e-3, arm.hi - 1e-3)
    p, R, _, _ = arm.fk(q)
    e = pose_error(p, R, p_d, R_d)
    ok = (e[:, :3].norm(dim=-1) < pos_tol) & (e[:, 3:].norm(dim=-1) < rot_tol)
    return q, ok


def ik_solutions(arm: Arm, p_d_w: np.ndarray, R_d_w: np.ndarray, base: BasePose,
                 door: DoorSpec, n_restart: int = 400, seed: int = 0,
                 dedup: float = 0.7, want: int = 8):
    """Distinct collision-free IK solutions for one world pose, spread out.

    Returns ``(q (k,7), swivel (k,))`` sorted by elbow swivel angle, which is the
    natural label for "the same TCP pose held with a different arm posture".
    """
    p_b, R_b = base.to_base(p_d_w[None], R_d_w[None])
    g = torch.Generator().manual_seed(seed)
    q0 = arm.lo + torch.rand((n_restart, 7), generator=g, dtype=DTYPE) * (arm.hi - arm.lo)
    q, ok = solve_ik(arm, torch.as_tensor(p_b, dtype=DTYPE).expand(n_restart, 3),
                     torch.as_tensor(R_b, dtype=DTYPE).expand(n_restart, 3, 3), q0)
    ok = ok & (arm.self_collision_margin(q) > 0.0)
    ok = ok & static_env_ok(arm, q, base, door, theta=0.0)
    q = q[ok]
    if q.numel() == 0:
        return q.reshape(0, 7).numpy(), np.zeros(0)
    swivel = elbow_swivel(arm, q, base)
    order = torch.argsort(torch.as_tensor(swivel))
    q, swivel = q[order], swivel[order.numpy()]
    keep_q, keep_s = [], []
    for i in range(q.shape[0]):
        if all((q[i] - k).abs().max() > dedup for k in keep_q):
            keep_q.append(q[i])
            keep_s.append(swivel[i])
    q = torch.stack(keep_q)
    swivel = np.asarray(keep_s)
    if q.shape[0] > want:                                  # spread over the range
        idx = np.linspace(0, q.shape[0] - 1, want).round().astype(int)
        q, swivel = q[idx], swivel[idx]
    return q.numpy(), swivel


def elbow_swivel(arm: Arm, q: torch.Tensor, base: BasePose) -> np.ndarray:
    """Angle of the elbow around the shoulder->TCP axis, in degrees."""
    tfs = arm.kin.link_transforms(q)
    shoulder = tfs[:, 1, :3, 3]
    elbow = tfs[:, 4, :3, 3]
    p_tcp, _, _, _ = arm.fk(q)
    axis = p_tcp - shoulder
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    v = elbow - shoulder
    v = v - axis * (v * axis).sum(-1, keepdim=True)
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE).expand_as(axis)
    ref = ref - axis * (ref * axis).sum(-1, keepdim=True)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    side = torch.cross(axis, ref, dim=-1)
    return torch.rad2deg(torch.atan2((v * side).sum(-1), (v * ref).sum(-1))).numpy()


# --------------------------------------------------------------------------- #
# environment collision (wall, swinging leaf, floor)
# --------------------------------------------------------------------------- #
EXEMPT_LINK = 6      # link6/link7 spheres sit at the handle: not checked vs door


def _point_rect_dist(P: torch.Tensor, origin: np.ndarray, u: np.ndarray,
                     v: np.ndarray, lu: float, lv: float) -> torch.Tensor:
    """Distance from points (B,S,3) to a rectangle given by origin + u,v spans."""
    o = torch.as_tensor(origin, dtype=P.dtype).view(1, 1, 3)
    ut = torch.as_tensor(u, dtype=P.dtype).view(1, 1, 3)
    vt = torch.as_tensor(v, dtype=P.dtype).view(1, 1, 3)
    nt = torch.as_tensor(np.cross(u, v), dtype=P.dtype).view(1, 1, 3)
    d = P - o
    a = (d * ut).sum(-1).clamp(0.0, lu) - (d * ut).sum(-1)
    b = (d * vt).sum(-1).clamp(0.0, lv) - (d * vt).sum(-1)
    c = (d * nt).sum(-1)
    return torch.sqrt(a ** 2 + b ** 2 + c ** 2)


def env_clearance(arm: Arm, q: torch.Tensor, base: BasePose, door: DoorSpec,
                  theta: float) -> torch.Tensor:
    """Min clearance of the arm spheres to wall / leaf / floor, per sample (B,)."""
    P = arm.spheres_world(q, base)                                  # (B,S,3)
    r = torch.as_tensor(arm.radii, dtype=P.dtype).view(1, -1)
    mask = torch.as_tensor(arm.link_of_sphere < EXEMPT_LINK).view(1, -1)

    # wall: the closed-leaf plane, robot side is the -n0 side (doorway ignored,
    # i.e. the wall is treated as solid -> conservative)
    n0 = torch.as_tensor(door.n0, dtype=P.dtype).view(1, 1, 3)
    h = torch.as_tensor(door.hinge, dtype=P.dtype).view(1, 1, 3)
    d_wall = -((P - h) * n0).sum(-1) - r

    # swinging leaf
    dvec, _ = door.leaf_frame(np.array([theta]))
    d_leaf = _point_rect_dist(
        P, door.hinge + np.array([0.0, 0.0, door.z_bottom]),
        dvec[0], np.array([0.0, 0.0, 1.0]), door.width, door.height) - r

    # floor
    d_floor = P[..., 2] - r

    c = torch.minimum(torch.minimum(d_wall, d_leaf), d_floor)
    c = torch.where(mask, c, torch.full_like(c, 1e3))
    return c.min(dim=-1).values


def static_env_ok(arm: Arm, q: torch.Tensor, base: BasePose, door: DoorSpec,
                  theta: float, clearance: float = 0.0) -> torch.Tensor:
    return env_clearance(arm, q, base, door, theta) > clearance


# --------------------------------------------------------------------------- #
# trajectory rollout: pull the door and see how far we get
# --------------------------------------------------------------------------- #
STOP_LABEL = {0: 'reached goal', 1: 'joint limit', 2: 'singularity / tracking failure',
              3: 'self-collision', 4: 'wall / leaf collision'}


@dataclass(frozen=True)
class Law:
    """How the 1-D redundancy left by the rigid grasp is spent.

    Every law tracks exactly the same 6-D task; they differ only in what they do
    with the null space and with joints that reach a bound.

    ``w_center`` / ``w_manip``  null-space gains [rad of null motion per rad of
                                door rotation] for joint centering and for the
                                manipulability gradient
    ``wln``                     weighted least norm (Chan & Dubey): joints that
                                move toward a bound get heavy, so the task
                                velocity is redistributed before they saturate
    ``sat``                     saturation clamping: a joint that would leave its
                                range is frozen and its Jacobian column dropped,
                                the rest re-solve for the same twist
    ``w_clear``                 clearance gain: push the arm off the wall and off
                                the leaf swinging toward it, active only inside
                                ``CLEAR_MARGIN``
    ``classical``               the paper's own null-space objective, see
                                ``classical_null_velocity``; the value scales it
    ``hybrid``                  the paper's switch: ``classical`` while the arm is
                                in the interior, ``wln + sat`` once a joint gets
                                within ``tau_enter`` of a bound, back at
                                ``tau_exit`` (hysteresis, same gate variable as
                                ``FrozenHybridController``)
    """
    name: str
    w_center: float = 0.0
    w_manip: float = 0.0
    w_clear: float = 0.0
    wln: bool = False
    sat: bool = False
    classical: float = 0.0
    hybrid: bool = False
    tau_enter: float = 0.985
    tau_exit: float = 0.96


# Yuan/IJRR/env/env.py NSRLConfig: the null-space velocity the paper's
# controllers emit is per second of a path travelled at V_PATH_REF m/s, and each
# null-space basis coordinate is clamped at A_MAX_REF rad/s.
V_PATH_REF = 0.05
A_MAX_REF = 0.5
MANIP_DAMPING = 1e-3          # cfg.manip_damping
CLASSICAL_MANIP_GAIN = 0.8    # ClassicalNullspaceController defaults, retuned
CLASSICAL_JL_GAIN = 0.4       # on the 10k eval set on 2026-05-30
CLEAR_MARGIN = 0.10           # clearance below which the obstacle term switches on [m]

LAW_MINNORM = Law('minimum norm (damped pseudo-inverse)')
LAW_CENTER = Law('null space: joint centering', w_center=3.0)
LAW_MANIP = Law('null space: manipulability', w_manip=3.0)
LAW_WLN = Law('weighted least norm + clamping', wln=True, sat=True)
LAW_CLEAR = Law('limit- and obstacle-aware', w_clear=6.0, wln=True, sat=True)
LAW_CLASSICAL = Law('classical null space (paper, 0.8/0.4)', classical=1.0)
LAW_HYBRID = Law('hybrid: classical + boundary law (paper switch)',
                 classical=1.0, hybrid=True)
LAW_DEFAULT = Law('joint centering + clamping', w_center=1.0, sat=True)

LAW_BY_KEY = {'minnorm': LAW_MINNORM, 'center': LAW_CENTER, 'manip': LAW_MANIP,
              'wln': LAW_WLN, 'clear': LAW_CLEAR, 'classical': LAW_CLASSICAL,
              'hybrid': LAW_HYBRID}
LAWS_REDUNDANCY = ['minnorm', 'classical', 'wln', 'hybrid']


def wln_weight(arm: Arm, q: torch.Tensor, dq_ref: torch.Tensor) -> torch.Tensor:
    """Chan & Dubey weights: heavy on joints moving toward their own limit."""
    lo, hi = arm.lo, arm.hi
    num = (hi - lo) ** 2 * (2.0 * q - hi - lo)
    den = 4.0 * ((hi - q) ** 2) * ((q - lo) ** 2)
    grad = num / den.clamp_min(1e-9)                    # dH/dq, blows up at a bound
    toward = (dq_ref * grad) > 0.0                      # moving up the gradient
    return torch.where(toward, 1.0 + grad.abs(), torch.ones_like(grad))


def classical_null_velocity(arm: Arm, q: torch.Tensor,
                            u_hat: torch.Tensor) -> torch.Tensor:
    """The paper's ``ClassicalNullspaceController`` objective, in rad/s.

    Ported from ``Yuan/IJRR/env/classical_nullspace.py``: the gradient of the
    manipulability *along the direction the TCP has to move*,

        mu(q, u) = ( u^T (Jp Jp^T + damping^2 I)^-1 u )^(-1/2)

    plus a linear pull toward mid-range, with the retuned gains 0.8 / 0.4. The
    third term of the original controller is the 30-degree cone attractor; a
    rigid grasp pins the whole orientation, so there is no cone here and the
    term is dropped (it is reported inert on the paper's own task anyway).
    """
    with torch.enable_grad():
        qe = q.detach().clone().requires_grad_(True)
        _, _, J, _ = arm.fk(qe)
        Jp = J[:, :3, :]
        eye3 = torch.eye(3, dtype=q.dtype).expand(q.shape[0], 3, 3)
        JJt = Jp @ Jp.transpose(1, 2) + (MANIP_DAMPING ** 2) * eye3
        u = u_hat.unsqueeze(-1)
        inv_quad = (u.transpose(1, 2) @ torch.linalg.inv(JJt) @ u).reshape(-1)
        mu = inv_quad.clamp_min(1e-12).pow(-0.5)
        grad_mu = torch.autograd.grad(mu.sum(), qe)[0].detach()
    span = (arm.hi - arm.lo).clamp_min(1e-6)
    return (CLASSICAL_MANIP_GAIN * grad_mu
            - CLASSICAL_JL_GAIN * (q - arm.mid) / span)


@dataclass
class Rollout:
    theta: np.ndarray          # (T,) door angle grid [rad]
    q: np.ndarray              # (T,7) joint history (frozen after the stop)
    n_ok: int                  # number of valid steps (index of the last good one + 1)
    reason: int
    limit_margin: np.ndarray   # (T,)
    manip: np.ndarray          # (T,)
    on_boundary: np.ndarray | None = None   # (T,) hybrid: boundary law active
    switches: int = 0                       # hybrid: interior <-> boundary count

    @property
    def theta_max(self) -> float:
        return float(self.theta[self.n_ok - 1])

    @property
    def theta_max_deg(self) -> float:
        return math.degrees(self.theta_max)

    @property
    def reason_str(self) -> str:
        return STOP_LABEL[self.reason]


def rollout_door(arm: Arm, door: DoorSpec, base: BasePose, q_start: np.ndarray,
                 law: Law = LAW_DEFAULT,
                 dtheta_deg: float = 0.5, lam: float = 0.02, iters: int = 3,
                 pos_tol: float = 3e-3, rot_tol: float = 2e-2,
                 limit_margin: float = 0.01, clearance: float = 0.005) -> Rollout:
    """Track the rigid-grasp door path with a resolved-rate controller.

    The door is stepped by ``dtheta_deg``; at each step the arm is asked for the
    exact grasp pose of the new door angle. The rollout stops at the first step
    the arm cannot serve: a joint bound that cannot be worked around, a tracking
    failure (the task twist is no longer in the range of J), a self-collision, or
    a collision with the wall or with the leaf it is pulling.
    """
    theta = np.arange(0.0, math.radians(door.theta_end_deg) + 1e-9,
                      math.radians(dtheta_deg))
    p_w, R_w = door.grasp_path(theta)
    p_b, R_b = base.to_base(p_w, R_w)
    p_b = torch.as_tensor(p_b, dtype=DTYPE)
    R_b = torch.as_tensor(R_b, dtype=DTYPE)

    # base-frame path tangent, the direction the paper's controller optimizes for
    u_hat = torch.zeros_like(p_b)
    u_hat[:-1] = p_b[1:] - p_b[:-1]
    u_hat[-1] = u_hat[-2]
    u_hat = u_hat / u_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # rad of null-space motion per rad of door rotation, at the paper's path speed
    arc_r = math.hypot(door.handle_r, door.handle_offset)
    null_scale = arc_r / V_PATH_REF

    T = theta.shape[0]
    q = torch.as_tensor(q_start, dtype=DTYPE).reshape(1, 7)
    q_hist = np.zeros((T, 7))
    marg = np.zeros(T)
    manip = np.zeros(T)
    on_bnd = np.zeros(T, dtype=bool)
    switches = 0
    lo, hi = arm.lo + limit_margin, arm.hi - limit_margin
    boundary = False        # hybrid state: the paper starts in the interior

    n_ok, reason = 0, 0
    for t in range(T):
        if t > 0:
            # hybrid gate: same variable and hysteresis as FrozenHybridController
            if law.hybrid:
                qn = float(1.0 - arm.limit_margin(q))
                nxt = qn >= (law.tau_exit if boundary else law.tau_enter)
                switches += int(nxt != boundary)
                boundary = nxt
            use_classical = law.classical != 0.0 and not (law.hybrid and boundary)
            use_wln = law.wln or (law.hybrid and boundary)
            use_sat = law.sat or (law.hybrid and boundary)

            hit_limit = False
            for it in range(iters):
                p, R, J, _ = arm.fk(q)
                e = pose_error(p, R, p_b[t:t + 1], R_b[t:t + 1])
                g = torch.zeros_like(q)
                if it == 0 and law.w_center != 0.0:
                    g = g + law.w_center * (-(q - arm.mid) / (arm.half ** 2))
                if it == 0 and law.w_manip != 0.0:
                    g = g + law.w_manip * manip_grad(arm, q)
                if it == 0 and law.w_clear != 0.0:
                    c, gc = clearance_grad(arm, q, base, door, float(theta[t]))
                    gate = ((CLEAR_MARGIN - c) / CLEAR_MARGIN).clamp(0.0, 1.0)
                    g = g + law.w_clear * gate.unsqueeze(-1) * gc
                g = g * math.radians(dtheta_deg)
                if it == 0 and use_classical:
                    g = g + law.classical * math.radians(dtheta_deg) * null_scale \
                        * _clamp_null(classical_null_velocity(arm, q, u_hat[t:t + 1]),
                                      J, lam, A_MAX_REF)

                winv = None
                if use_wln:
                    dq_ref, _ = dls_solve(J, e, lam)
                    winv = 1.0 / wln_weight(arm, q, dq_ref)

                active = torch.ones_like(q)
                for _ in range(5):
                    w_act = active if winv is None else active * winv
                    dq, N = dls_solve(J * active.unsqueeze(1), e, lam, w_act)
                    dq = dq + (N @ g.unsqueeze(-1)).squeeze(-1)
                    dq = (dq * active).clamp(-0.35, 0.35)
                    out = ((q + dq) < lo) | ((q + dq) > hi)
                    hit_limit = hit_limit or bool(out.any())
                    if not use_sat or not bool((out & (active > 0)).any()):
                        break
                    active = active * (~out)
                q = (q + dq).clamp(lo, hi)

            p, R, _, _ = arm.fk(q)
            e = pose_error(p, R, p_b[t:t + 1], R_b[t:t + 1])
            bad_track = bool(e[0, :3].norm() > pos_tol or e[0, 3:].norm() > rot_tol)
            if bad_track:
                reason = 1 if hit_limit else 2
                break
            if float(arm.self_collision_margin(q)) <= 0.0:
                reason = 3
                break
            if float(env_clearance(arm, q, base, door, float(theta[t]))) <= clearance:
                reason = 4
                break

        q_hist[t] = q.numpy()[0]
        marg[t] = float(arm.limit_margin(q))
        manip[t] = float(arm.manipulability(q))
        on_bnd[t] = boundary
        n_ok = t + 1

    if n_ok < T:                       # freeze the pose after the stop
        q_hist[n_ok:] = q_hist[n_ok - 1]
        marg[n_ok:] = marg[n_ok - 1]
        manip[n_ok:] = manip[n_ok - 1]
        on_bnd[n_ok:] = on_bnd[n_ok - 1]
    return Rollout(theta, q_hist, n_ok, reason, marg, manip,
                   on_bnd if law.hybrid else None, switches)


def _clamp_null(q_dot: torch.Tensor, J: torch.Tensor, lam: float,
                a_max: float) -> torch.Tensor:
    """Project a null-space velocity request and clamp it like the paper's env.

    The paper expresses the request in a task-aligned null-space basis and clamps
    every coordinate at ``a_max``. A rigid grasp leaves a 1-D null space, so that
    is a single coordinate: project, then clamp the magnitude.
    """
    _, N = dls_solve(J, torch.zeros((J.shape[0], 6), dtype=J.dtype), lam)
    g = (N @ q_dot.unsqueeze(-1)).squeeze(-1)
    n = g.norm(dim=-1, keepdim=True)
    return g * (n.clamp_max(a_max) / n.clamp_min(1e-12))


def clearance_grad(arm: Arm, q: torch.Tensor, base: BasePose, door: DoorSpec,
                   theta: float, eps: float = 1e-3):
    """Clearance to wall/leaf/floor and its finite-difference gradient, (B,7).

    The leaf swings toward the robot, so the arm has to get out of its own way;
    none of the textbook null-space objectives knows that, this one does.
    """
    B = q.shape[0]
    qs = q.unsqueeze(1).repeat(1, 8, 1)
    for i in range(7):
        qs[:, i + 1, i] += eps
    c = env_clearance(arm, qs.reshape(B * 8, 7), base, door, theta).reshape(B, 8)
    return c[:, 0], (c[:, 1:] - c[:, :1]) / eps


def manip_grad(arm: Arm, q: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Finite-difference gradient of log manipulability, (B,7)."""
    B = q.shape[0]
    qs = q.unsqueeze(1).repeat(1, 8, 1)
    for i in range(7):
        qs[:, i + 1, i] += eps
    w = arm.manipulability(qs.reshape(B * 8, 7)).reshape(B, 8)
    return (w[:, 1:] - w[:, :1]) / eps / w[:, :1].clamp_min(1e-12)


def pointwise_reachable(arm: Arm, door: DoorSpec, base: BasePose,
                        n_restart: int = 64, dtheta_deg: float = 2.5,
                        seed: int = 0) -> tuple:
    """What a single-point reachability map would report.

    For every sampled door angle, solve IK from ``n_restart`` random starts and
    keep the pose if any solution is collision-free. Returns
    ``(theta_grid, reachable_mask, theta_reach_continuous)`` where the last entry
    is the largest angle such that every sampled pose up to it is reachable.
    """
    theta = np.arange(0.0, math.radians(door.theta_end_deg) + 1e-9,
                      math.radians(dtheta_deg))
    p_w, R_w = door.grasp_path(theta)
    p_b, R_b = base.to_base(p_w, R_w)
    T = theta.shape[0]
    g = torch.Generator().manual_seed(seed)
    q0 = arm.lo + torch.rand((T * n_restart, 7), generator=g, dtype=DTYPE) * (arm.hi - arm.lo)
    pd = torch.as_tensor(p_b, dtype=DTYPE).repeat_interleave(n_restart, 0)
    Rd = torch.as_tensor(R_b, dtype=DTYPE).repeat_interleave(n_restart, 0)
    q, ok = solve_ik(arm, pd, Rd, q0)
    ok = ok & (arm.self_collision_margin(q) > 0.0)
    ok = ok.reshape(T, n_restart)
    q = q.reshape(T, n_restart, 7)
    reach = np.zeros(T, dtype=bool)
    for t in range(T):                       # env collision depends on theta
        sel = ok[t]
        if not bool(sel.any()):
            continue
        reach[t] = bool(static_env_ok(arm, q[t][sel], base, door,
                                      float(theta[t])).any())
    # the map only rules out angles it cannot solve, so the horizon it promises
    # is the first sampled angle without an IK solution (an upper bound on any
    # trajectory, up to the sampling resolution)
    bad = np.flatnonzero(~reach)
    cont = float(theta[bad[0]]) if bad.size else float(theta[-1])
    return theta, reach, cont


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
@dataclass
class Variant:
    label: str
    door: DoorSpec
    base: BasePose
    q0: np.ndarray
    law: Law
    color: str
    roll: Rollout | None = None
    reach_deg: float | None = None      # point-wise reachability horizon [deg]


PALETTE = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e']
PROBE_DTHETA = 1.0          # coarse step used while picking the variants


def _start_configs(arm: Arm, door: DoorSpec, base: BasePose, want: int = 24):
    """All distinct start postures for the closed-door grasp pose."""
    p_w, R_w = door.grasp_path(np.array([0.0]))
    return ik_solutions(arm, p_w[0], R_w[0], base, door, want=want)


def _rank_starts(arm: Arm, door: DoorSpec, base: BasePose, law: Law = LAW_DEFAULT):
    """theta_max of every start posture, best first: (q, swivel, theta_max_deg)."""
    q_all, sw = _start_configs(arm, door, base)
    out = [(q, s, rollout_door(arm, door, base, q, law=law,
                               dtheta_deg=PROBE_DTHETA).theta_max_deg)
           for q, s in zip(q_all, sw)]
    out.sort(key=lambda r: -r[2])
    return out


def _best_start(arm: Arm, door: DoorSpec, base: BasePose, law: Law = LAW_DEFAULT):
    """The start posture that gets furthest - i.e. the start is optimized away."""
    ranked = _rank_starts(arm, door, base, law)
    return (None, 0.0) if not ranked else (ranked[0][0], ranked[0][2])


def scenario_init(arm: Arm, door: DoorSpec, base: BasePose, n: int = 3):
    """Same base, same controller, same TCP pose - different start joint angles."""
    ranked = _rank_starts(arm, door, base)
    if len(ranked) < n:
        raise RuntimeError('not enough distinct IK branches at the closed-door pose')
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)   # best .. worst
    return [Variant(label=f'start posture {chr(65 + k)}   (elbow swivel {ranked[i][1]:+.0f}°)',
                    door=door, base=base, q0=ranked[i][0], law=LAW_DEFAULT,
                    color=PALETTE[k])
            for k, i in enumerate(idx)]


def scenario_redundancy(arm: Arm, door: DoorSpec, base: BasePose,
                        keys: list | None = None):
    """Same start joint angles - different redundancy resolution."""
    laws = [LAW_BY_KEY[k] for k in (keys or LAWS_REDUNDANCY)]
    q_all, sw = _start_configs(arm, door, base)
    # use the start posture that separates the laws the most, so the figure shows
    # the controller effect rather than the start effect
    best_q, best_spread = None, -1.0
    for q in q_all:
        th = [rollout_door(arm, door, base, q, law=l,
                           dtheta_deg=PROBE_DTHETA).theta_max_deg for l in laws]
        if max(th) - min(th) > best_spread:
            best_q, best_spread = q, max(th) - min(th)
    return [Variant(label=law.name, door=door, base=base, q0=best_q, law=law,
                    color=PALETTE[k]) for k, law in enumerate(laws)]


def scenario_base(arm: Arm, door: DoorSpec, base: BasePose, n: int = 3):
    """Same door, same controller, best start posture - different base placement."""
    cand = [replace(base, x=base.x + dx, y=base.y + dy)
            for dx, dy in [(0.0, 0.0), (0.15, 0.0), (-0.15, 0.0), (0.0, -0.15),
                           (0.15, -0.15), (-0.15, -0.30)]]
    found = []
    for b in cand:
        q0, th = _best_start(arm, door, b)
        print(f'  probing base ({b.x:+.2f}, {b.y:+.2f}) -> theta_max {th:5.1f}°')
        if q0 is not None:
            found.append((b, q0, th))
    found.sort(key=lambda r: -r[2])
    idx = np.linspace(0, len(found) - 1, min(n, len(found))).round().astype(int)
    return [Variant(label=f'base at ({found[i][0].x:+.2f}, {found[i][0].y:+.2f}) m',
                    door=door, base=found[i][0], q0=found[i][1], law=LAW_DEFAULT,
                    color=PALETTE[k])
            for k, i in enumerate(idx)]


def scenario_height(arm: Arm, door: DoorSpec, base: BasePose, n: int = 3):
    """Same base, same controller, best start posture - different grasp height."""
    out = []
    for k, hz in enumerate(np.linspace(0.70, 1.20, n)):
        d = replace(door, handle_z=float(hz))
        q0, th = _best_start(arm, d, base)
        print(f'  probing grasp height {hz:.2f} m -> theta_max {th:5.1f}°')
        if q0 is None:
            continue
        out.append(Variant(label=f'grasp height {hz:.2f} m', door=d, base=base,
                           q0=q0, law=LAW_DEFAULT, color=PALETTE[k]))
    return out


SCENARIOS = {'init': scenario_init, 'redundancy': scenario_redundancy,
             'base': scenario_base, 'height': scenario_height}

SCENARIO_TITLE = {
    'init': 'Same base, same controller, same TCP pose — different start joint angles',
    'redundancy': 'Same base, same start joint angles — different redundancy resolution',
    'base': 'Same door, same controller — different base placement',
    'height': 'Same base, same controller — different grasp height on the leaf',
}


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def scene_bounds(variants) -> tuple:
    """A common, tight bounding box for every panel of one figure."""
    pts = []
    for v in variants:
        d = v.door
        pts.append(v.base.t)
        for th in np.linspace(0.0, math.radians(d.theta_end_deg), 12):
            c = d.leaf_corners(float(th))
            pts.extend([c[0], c[1]])
        pts.append(d.hinge + 1.0 * d.d0)
        pts.append(d.hinge - 0.5 * d.d0)
    pts = np.asarray(pts)
    lo = pts.min(0) - np.array([0.35, 0.35, 0.0])
    hi = pts.max(0) + np.array([0.35, 0.35, 0.0])
    lo[2], hi[2] = 0.0, 2.05
    return lo, hi


def draw_static(ax, door: DoorSpec, base: BasePose, bounds):
    """Floor, wall, hinge, handle arc and the base platform (drawn once)."""
    lo, hi = bounds
    d0 = door.d0
    h = door.hinge
    z0, z1 = door.z_bottom, door.z_bottom + door.height
    # wall panels either side of the doorway (the leaf swings inside the opening)
    for s0, s1 in [(-0.75, 0.0), (door.width, door.width + 0.75)]:
        c = np.array([h + s0 * d0 + [0, 0, z0], h + s1 * d0 + [0, 0, z0],
                      h + s1 * d0 + [0, 0, z1], h + s0 * d0 + [0, 0, z1]])
        ax.add_collection3d(Poly3DCollection([c], facecolor='#c9ccd1', alpha=0.30,
                                             edgecolor='#8d9298', linewidths=0.6))
    # floor grid
    for v in np.arange(math.floor(lo[0] * 4) / 4, hi[0] + 1e-6, 0.25):
        ax.plot([v, v], [lo[1], hi[1]], [0, 0], color='#e6e6e6', lw=0.5, zorder=0)
    for v in np.arange(math.floor(lo[1] * 4) / 4, hi[1] + 1e-6, 0.25):
        ax.plot([lo[0], hi[0]], [v, v], [0, 0], color='#e6e6e6', lw=0.5, zorder=0)
    # hinge axis
    ax.plot([h[0], h[0]], [h[1], h[1]], [z0, z1], color='#555555', lw=2.0, ls=':')
    # handle arc: the commanded path, i.e. what a reachability map would test
    th = np.linspace(0.0, math.radians(door.theta_end_deg), 120)
    p, _ = door.grasp_path(th)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color='#9a9a9a', lw=1.2, ls='--', zorder=1)
    # base platform
    bx, by, bz = base.t
    for zz in (0.0, bz):
        sq = np.array([[bx - .13, by - .13, zz], [bx + .13, by - .13, zz],
                       [bx + .13, by + .13, zz], [bx - .13, by + .13, zz]])
        ax.add_collection3d(Poly3DCollection([sq], facecolor='#8a8a8a', alpha=0.55,
                                             edgecolor='#4d4d4d', lw=0.6))
    for sx in (-.13, .13):
        for sy in (-.13, .13):
            ax.plot([bx + sx, bx + sx], [by + sy, by + sy], [0, bz],
                    color='#4d4d4d', lw=1.2)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), zoom=1.2)
    ax.set_axis_off()


def draw_static_top(ax, door: DoorSpec, base: BasePose, bounds):
    """Static part of the small top view."""
    lo, hi = bounds
    d0, h = door.d0, door.hinge
    ax.plot(*np.stack([h - 0.75 * d0, h])[:, :2].T, color='#8d9298', lw=3.0)
    ax.plot(*np.stack([h + door.width * d0, h + (door.width + 0.75) * d0])[:, :2].T,
            color='#8d9298', lw=3.0)
    th = np.linspace(0.0, math.radians(door.theta_end_deg), 120)
    p, _ = door.grasp_path(th)
    ax.plot(p[:, 0], p[:, 1], color='#9a9a9a', lw=1.0, ls='--')
    ax.plot([base.x], [base.y], 's', color='#4d4d4d', ms=5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
    ax.patch.set_alpha(0.65)
    ax.text(0.03, 0.03, 'top view', transform=ax.transAxes, fontsize=7,
            color='#777777')


class RobotArtists:
    """Everything that moves in one 3D panel (plus its top-view inset)."""

    def __init__(self, ax, ax_top, arm: Arm, variant: Variant, show_spheres: bool,
                 ghosts: int = 1):
        self.ax, self.arm, self.v = ax, arm, variant
        self.ghosts = ghosts
        c = variant.color
        self.ghost_links, self.ghost_leaves = [], []
        for j in range(max(0, ghosts - 1)):     # faint copies behind the live one
            a = 0.16 + 0.5 * (j / max(1, ghosts - 2)) if ghosts > 2 else 0.3
            self.ghost_links.append(ax.plot([], [], [], color=c, lw=3.5, alpha=a,
                                            solid_capstyle='round', zorder=4)[0])
            gl = Poly3DCollection([np.zeros((4, 3))], facecolor='#b58a5a',
                                  alpha=0.22 * a / 0.4, edgecolor='#6d4f2c', lw=0.6)
            ax.add_collection3d(gl)
            self.ghost_leaves.append(gl)
        self.link, = ax.plot([], [], [], color=c, lw=5.0, solid_capstyle='round',
                             zorder=6)
        self.joints, = ax.plot([], [], [], 'o', color=c, ms=5.0, mfc='white',
                               mew=1.4, zorder=7)
        self.finger1, = ax.plot([], [], [], color=c, lw=2.5, zorder=7)
        self.finger2, = ax.plot([], [], [], color=c, lw=2.5, zorder=7)
        self.trace, = ax.plot([], [], [], color=c, lw=1.6, alpha=0.8, zorder=5)
        self.leaf = Poly3DCollection([np.zeros((4, 3))], facecolor='#b58a5a',
                                     alpha=0.55, edgecolor='#6d4f2c', lw=1.2)
        ax.add_collection3d(self.leaf)
        self.spheres = None
        if show_spheres:
            self.spheres = ax.scatter([], [], [], s=1, c=c, alpha=0.12,
                                      depthshade=False)
        self.txt = ax.text2D(0.03, 0.93, '', transform=ax.transAxes, fontsize=11,
                             color=c, fontweight='bold')
        self.top = TopViewArtists(ax_top, arm, variant)

    def update(self, t: int):
        v, arm = self.v, self.arm
        roll = v.roll
        t = min(t, roll.q.shape[0] - 1)
        q = torch.as_tensor(roll.q[t], dtype=DTYPE).reshape(1, 7)
        pts = arm.joints_world(q, v.base)[0]
        self.link.set_data(pts[:, 0], pts[:, 1])
        self.link.set_3d_properties(pts[:, 2])
        self.joints.set_data(pts[1:8, 0], pts[1:8, 1])
        self.joints.set_3d_properties(pts[1:8, 2])

        p_tcp, R_tcp = arm.tcp_world(q, v.base)
        p_tcp, R_tcp = p_tcp[0], R_tcp[0]
        y, z = R_tcp[:, 1], R_tcp[:, 2]
        for art, sgn in ((self.finger1, 1.0), (self.finger2, -1.0)):
            a = p_tcp - 0.06 * z + sgn * 0.035 * y
            b = p_tcp + 0.01 * z + sgn * 0.035 * y
            art.set_data([a[0], b[0]], [a[1], b[1]])
            art.set_3d_properties([a[2], b[2]])

        tt = min(t, roll.n_ok - 1)
        tr = arm.tcp_world(torch.as_tensor(roll.q[:tt + 1], dtype=DTYPE), v.base)[0]
        self.trace.set_data(tr[:, 0], tr[:, 1])
        self.trace.set_3d_properties(tr[:, 2])

        theta = roll.theta[tt]
        self.leaf.set_verts([v.door.leaf_corners(theta)])
        self.top.update(tt)
        if self.ghosts > 1:
            idx = np.linspace(0, tt, self.ghosts).round().astype(int)[:-1]
            for art, leaf, ti in zip(self.ghost_links, self.ghost_leaves, idx):
                gp = arm.joints_world(torch.as_tensor(roll.q[ti], dtype=DTYPE
                                                      ).reshape(1, 7), v.base)[0]
                art.set_data(gp[:, 0], gp[:, 1])
                art.set_3d_properties(gp[:, 2])
                leaf.set_verts([v.door.leaf_corners(float(roll.theta[ti]))])
        if self.spheres is not None:
            P = arm.spheres_world(q, v.base).numpy()[0]
            self.spheres._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
            self.spheres.set_sizes((arm.radii.numpy() * 700.0) ** 2 * 0.02)
        self.txt.set_text(f'$\\theta$ = {math.degrees(theta):5.1f}°')


class MeshPanels:
    """The same panels, but drawn with the real FR3 meshes through MuJoCo."""

    def __init__(self, fig, gs_top, arm: Arm, variants: list, elev: float = 26.0,
                 az_offset: float = -30.0, zoom: float = 1.0, ghosts: int = 1,
                 width: int = 780, height: int = 700):
        from Yuan.IJRR.figures.fr3_scene import DoorScene   # needs a GL backend

        self.arm, self.variants = arm, variants
        self.ghosts = ghosts
        self.scene = DoorScene.from_variants(variants, ghosts=ghosts)
        self.w, self.h = width, height
        lo, hi = scene_bounds(variants)
        look = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.80])
        # MuJoCo's azimuth is the viewing *direction*, so azimuth 0 here means
        # looking from the robot's side straight into the door; az_offset swings
        # around that (the camera must stay on the robot's side of the wall)
        n = variants[0].door.n0
        az = math.degrees(math.atan2(n[1], n[0])) + az_offset
        span = max(hi[2] - lo[2], (hi[1] - lo[1]) * height / width)
        dist = 0.70 * span / math.tan(math.radians(0.5 * self.scene.fovy)) / zoom
        self.cams = [self.scene.camera(k, dist, az, -elev, look)
                     for k in range(len(variants))]
        self.ims, self.txts = [], []
        for k, v in enumerate(variants):
            ax = fig.add_subplot(gs_top[0, k])
            ax.set_axis_off()
            self.ims.append(ax.imshow(np.zeros((height, width, 3), np.uint8),
                                      interpolation='bilinear'))
            ax_in = ax.inset_axes([0.0, 0.0, 0.29, 0.29])
            draw_static_top(ax_in, v.door, v.base, (lo, hi))
            _panel_title(ax, v)
            self.txts.append(ax.text(0.03, 0.94, '', transform=ax.transAxes,
                                     fontsize=11, color=v.color, fontweight='bold'))
            v._top = TopViewArtists(ax_in, arm, v)

    def update(self, t: int):
        g = self.ghosts
        for k, v in enumerate(self.variants):
            tt = min(t, v.roll.n_ok - 1)
            if g == 1:
                self.scene.set_state(k, v.roll.q[tt], float(v.roll.theta[tt]),
                                     v.door.swing)
                continue
            # static overlay: the ghosts sample this variant's whole rollout
            idx = np.linspace(0, tt, g).round().astype(int)
            for j, ti in enumerate(idx):
                self.scene.set_state(k * g + j, v.roll.q[ti],
                                     float(v.roll.theta[ti]), v.door.swing)
        self.scene.forward()
        for k, v in enumerate(self.variants):
            tt = min(t, v.roll.n_ok - 1)
            qs = torch.as_tensor(v.roll.q[:tt + 1], dtype=DTYPE)
            tr = self.arm.tcp_world(qs, v.base)[0]
            arc, _ = v.door.grasp_path(v.roll.theta)          # the commanded path
            traces = [(arc, (0.55, 0.55, 0.55, 0.55)),
                      (tr, mcolors.to_rgba(v.color))]
            self.ims[k].set_data(self.scene.render(k, self.cams[k], self.w,
                                                   self.h, traces=traces))
            self.txts[k].set_text(f'$\\theta$ = {math.degrees(v.roll.theta[tt]):5.1f}°')
            v._top.update(tt)


class TopViewArtists:
    """The little top view, shared by both renderers."""

    def __init__(self, ax, arm: Arm, v: Variant):
        self.arm, self.v = arm, v
        self.link, = ax.plot([], [], color=v.color, lw=2.0)
        self.leaf, = ax.plot([], [], color='#8a5a2b', lw=3.0)
        self.tcp, = ax.plot([], [], 'o', color=v.color, ms=3.5)

    def update(self, t: int):
        v = self.v
        q = torch.as_tensor(v.roll.q[t], dtype=DTYPE).reshape(1, 7)
        pts = self.arm.joints_world(q, v.base)[0]
        self.link.set_data(pts[:, 0], pts[:, 1])
        p_tcp = self.arm.tcp_world(q, v.base)[0][0]
        self.tcp.set_data([p_tcp[0]], [p_tcp[1]])
        corners = v.door.leaf_corners(float(v.roll.theta[t]))
        self.leaf.set_data(corners[:2, 0], corners[:2, 1])


def _panel_title(ax, v: Variant):
    ax.set_title(f'{v.label}\n' + r'$\theta_{\max}$ = '
                 + f'{v.roll.theta_max_deg:.1f}°   ({v.roll.reason_str})',
                 fontsize=10.5, color=v.color, pad=-4)


def animate(arm: Arm, variants: list, title: str,
            fps: int = 30, save: str | None = None, spheres: bool = False,
            elev: float = 40.0, azim: float = -75.0, render: str = 'mesh',
            cam_elev: float = 26.0, cam_azim: float = -30.0, zoom: float = 1.0,
            overlay: int = 1):
    n = len(variants)
    bounds = scene_bounds(variants)
    fig = plt.figure(figsize=(4.7 * n, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.0], hspace=0.16,
                          left=0.045, right=0.98, top=0.90, bottom=0.09)
    gs_top = gs[0].subgridspec(1, n, wspace=0.02)
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.20)

    if render == 'mesh':
        panels = MeshPanels(fig, gs_top, arm, variants, elev=cam_elev,
                            az_offset=cam_azim, zoom=zoom, ghosts=overlay)
        art = [panels]
    else:
        art = []
        for k, v in enumerate(variants):
            ax = fig.add_subplot(gs_top[0, k], projection='3d')
            ax.view_init(elev=elev, azim=azim)
            draw_static(ax, v.door, v.base, bounds)
            ax_in = ax.inset_axes([0.0, 0.0, 0.30, 0.30])
            draw_static_top(ax_in, v.door, v.base, bounds)
            _panel_title(ax, v)
            art.append(RobotArtists(ax, ax_in, arm, v, spheres, ghosts=overlay))

    # bottom left: why each rollout dies - the joint-limit margin along the way
    ax_m = fig.add_subplot(gs_bot[0, 0])
    T = variants[0].roll.theta.shape[0]
    for v in variants:
        th = np.degrees(v.roll.theta[:v.roll.n_ok])
        ax_m.plot(th, v.roll.limit_margin[:v.roll.n_ok], color=v.color, lw=1.0,
                  alpha=0.30)
        if v.roll.on_boundary is not None:       # hybrid: where the switch fired
            ax_m.fill_between(th, 0.0, 1.0, where=v.roll.on_boundary[:v.roll.n_ok],
                              color=v.color, alpha=0.10, lw=0)
    lines = [ax_m.plot([], [], color=v.color, lw=2.2, label=v.label)[0]
             for v in variants]
    stops = [ax_m.plot([], [], 'X', color=v.color, ms=9, mec='k', mew=0.6)[0]
             for v in variants]
    ax_m.axhline(0.0, color='k', lw=1.0)
    ax_m.set_xlim(0, variants[0].door.theta_end_deg)
    ax_m.set_ylim(-0.03, 1.0)
    ax_m.set_xlabel('door opening angle  θ  [deg]')
    ax_m.set_ylabel('closest joint-limit margin\n(1 = mid-range, 0 = at a limit)')
    ax_m.grid(alpha=0.25)
    ax_m.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # bottom right: the achievable angle against what a reachability map promises
    ax_b = fig.add_subplot(gs_bot[0, 1])
    ypos = np.arange(n)[::-1]
    bars = ax_b.barh(ypos, np.zeros(n), color=[v.color for v in variants], height=0.62)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([chr(65 + k) for k in range(n)], fontsize=9)
    ax_b.set_xlim(0, variants[0].door.theta_end_deg)
    ax_b.set_ylim(-0.7, n - 0.35)
    ax_b.set_xlabel(r'$\theta_{\max}$ [deg]')
    ax_b.grid(alpha=0.25, axis='x')
    reaches = [v.reach_deg for v in variants]
    if all(r is not None for r in reaches):
        for y, r in zip(ypos, reaches):        # each variant has its own geometry
            ax_b.plot([r, r], [y - 0.38, y + 0.38], color='#2ca02c', lw=2.6,
                      zorder=5)
        ax_b.set_title('green: the same arc under a point-wise reachability map',
                       fontsize=9.5, color='#2ca02c')
    txt_bars = [ax_b.text(0, y, '', va='center', fontsize=9, color=v.color)
                for y, v in zip(ypos, variants)]

    fig.suptitle(title, fontsize=13.5)

    def update(t):
        for a in art:
            a.update(t)
        for k, v in enumerate(variants):
            tt = min(t, v.roll.n_ok - 1, T - 1)
            th = np.degrees(v.roll.theta[:tt + 1])
            lines[k].set_data(th, v.roll.limit_margin[:tt + 1])
            bars[k].set_width(th[-1])
            txt_bars[k].set_position((th[-1] + 1.5, ypos[k]))
            txt_bars[k].set_text(f'{th[-1]:.0f}°')
            if t >= v.roll.n_ok - 1:
                stops[k].set_data([th[-1]], [v.roll.limit_margin[tt]])
        return []

    if overlay > 1 or (save and save.endswith('.png')):
        update(T - 1)                         # static: the whole rollout at once
        if save:
            fig.savefig(save, dpi=180)
            print(f'wrote {save}')
        else:
            plt.show()
        return None

    ani = FuncAnimation(fig, update, frames=T + int(1.5 * fps), interval=1000 / fps,
                        blit=False, repeat=True, cache_frame_data=False)
    if save:
        print(f'writing {save} ...')
        if save.endswith('.gif'):
            ani.save(save, writer='pillow', fps=fps, dpi=90)
        else:
            ani.save(save, fps=fps, dpi=110)
        print('done')
    else:
        plt.show()
    return ani


def play_in_viewer(arm: Arm, variants: list, fps: int = 30, loops: int = 1000):
    """Play the same rollouts in a native MuJoCo window, all variants at once.

    Free camera (drag to orbit), real meshes, real time. No plots — use the
    matplotlib window for those.
    """
    import time

    import mujoco.viewer

    from Yuan.IJRR.figures.fr3_scene import DoorScene, add_polyline

    scene = DoorScene(variants)
    T = variants[0].roll.theta.shape[0]
    traces = []
    for k, v in enumerate(variants):
        qs = torch.as_tensor(v.roll.q[:v.roll.n_ok], dtype=DTYPE)
        traces.append((arm.tcp_world(qs, v.base)[0] + scene.offsets[k],
                       mcolors.to_rgba(v.color)))
    print('MuJoCo viewer: drag to orbit, close the window to quit')
    with mujoco.viewer.launch_passive(scene.model, scene.data,
                                      show_left_ui=False, show_right_ui=False) as vw:
        for _ in range(loops):
            for t in range(T + fps):
                if not vw.is_running():
                    return
                for k, v in enumerate(variants):
                    tt = min(t, v.roll.n_ok - 1, T - 1)
                    scene.set_state(k, v.roll.q[tt], float(v.roll.theta[tt]),
                                    v.door.swing)
                scene.forward()
                vw.user_scn.ngeom = 0
                for k, v in enumerate(variants):
                    tt = min(t, v.roll.n_ok - 1, T - 1)
                    add_polyline(vw.user_scn, traces[k][0][:tt + 1], traces[k][1])
                vw.sync()
                time.sleep(1.0 / fps)


# --------------------------------------------------------------------------- #
# the optimization landscape (static figure)
# --------------------------------------------------------------------------- #
def sweep_figure(arm: Arm, door: DoorSpec, base: BasePose, save: str | None,
                 grid: int = 7):    # grid x grid base placements, ~2 s per cell
    """The landscape the trajectory problem lives on, and what a map would say."""
    fig, axs = plt.subplots(2, 2, figsize=(12.4, 9.0))

    # (a) theta_max over the start postures of one and the same TCP pose
    ranked = _rank_starts(arm, door, base)
    sw = np.array([r[1] for r in ranked])
    th = np.array([r[2] for r in ranked])
    order = np.argsort(sw)
    _, _, reach0 = pointwise_reachable(arm, door, base)
    axs[0, 0].plot(sw[order], th[order], 'o-', color='#d62728',
                   label=r'trajectory $\theta_{\max}$')
    axs[0, 0].axhline(math.degrees(reach0), color='#2ca02c', ls='--',
                      label='point-wise reachability horizon')
    axs[0, 0].set_xlabel('elbow swivel of the start posture [deg]')
    axs[0, 0].set_ylabel(r'$\theta_{\max}$ [deg]')
    axs[0, 0].set_title('(a) one TCP pose, all its IK solutions')
    axs[0, 0].legend(fontsize=9); axs[0, 0].grid(alpha=0.3)

    # (b) theta_max vs grasp height, best start posture per height
    hs = np.linspace(0.65, 1.30, 12)
    th_h, rh_h = [], []
    for hz in hs:
        d = replace(door, handle_z=float(hz))
        th_h.append(_best_start(arm, d, base)[1])
        rh_h.append(math.degrees(pointwise_reachable(arm, d, base)[2]))
        print(f'  height sweep {hz:.2f} m -> {th_h[-1]:5.1f}° (map says {rh_h[-1]:5.1f}°)')
    axs[0, 1].plot(hs, th_h, 'o-', color='#d62728', label=r'best $\theta_{\max}$')
    axs[0, 1].plot(hs, rh_h, 's--', color='#2ca02c', label='point-wise horizon')
    axs[0, 1].set_xlabel('grasp height on the leaf [m]')
    axs[0, 1].set_ylabel(r'$\theta_{\max}$ [deg]')
    axs[0, 1].set_title('(b) grasp height (start posture optimized)')
    axs[0, 1].legend(fontsize=9); axs[0, 1].grid(alpha=0.3)

    # (c)/(d) base placement: what is actually achievable vs what a map promises
    xs = np.linspace(base.x - 0.20, base.x + 0.30, grid)
    ys = np.linspace(base.y - 0.20, base.y + 0.40, grid)
    Zt = np.zeros((grid, grid))
    Zr = np.zeros((grid, grid))
    for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
            b = replace(base, x=float(xx), y=float(yy))
            Zt[i, j] = _best_start(arm, door, b)[1]
            Zr[i, j] = math.degrees(pointwise_reachable(arm, door, b)[2])
        print(f'  base sweep row {i + 1}/{grid}')
    vmax = max(Zt.max(), Zr.max())
    for ax, Z, ttl in ((axs[1, 0], Zt, '(c) achievable $\\theta_{\\max}$ over base placement'),
                       (axs[1, 1], Zr, '(d) point-wise reachability horizon (the map)')):
        im = ax.pcolormesh(xs, ys, Z, cmap='viridis', shading='nearest',
                           vmin=0.0, vmax=vmax)
        ax.plot([base.x], [base.y], 'r*', ms=13)
        ax.set_xlabel('base x [m]'); ax.set_ylabel('base y [m]')
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, label='[deg]')
    fig.suptitle('Opening angle is a property of the whole trajectory: the map in (d) '
                 'is smooth and permissive, what is achievable in (c) is neither',
                 fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=180)
        print(f'wrote {save}')
    else:
        plt.show()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--scenario', default='init', choices=list(SCENARIOS))
    ap.add_argument('--save', default=None, help='write mp4/gif (animation) or png (--sweep)')
    ap.add_argument('--sweep', action='store_true', help='static optimization landscape')
    ap.add_argument('--grid', type=int, default=7, help='base grid size for --sweep')
    ap.add_argument('--laws', nargs='+', default=None, choices=list(LAW_BY_KEY),
                    help='which resolution laws --scenario redundancy compares')
    ap.add_argument('--render', default='mesh', choices=['mesh', 'stick'],
                    help='mesh: real FR3 meshes through MuJoCo; stick: link polylines')
    ap.add_argument('--viewer', action='store_true',
                    help='play in a native MuJoCo window (free camera, no plots)')
    ap.add_argument('--overlay', type=int, default=1, metavar='N',
                    help='static figure: overlay N poses of each rollout, '
                         'transparent, instead of animating')
    ap.add_argument('--spheres', action='store_true',
                    help='draw the collision spheres (--render stick only)')
    ap.add_argument('--no-reach', action='store_true', help='skip the reachability-map line')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--dtheta', type=float, default=0.5, help='door step [deg]')
    ap.add_argument('--theta-end', type=float, default=90.0, help='goal opening [deg]')
    ap.add_argument('--handle-z', type=float, default=0.95)
    ap.add_argument('--base-xy', type=float, nargs=2, default=(-0.05, 0.45))
    ap.add_argument('--cam-elev', type=float, default=26.0,
                    help='mesh camera elevation above the floor [deg]')
    ap.add_argument('--cam-azim', type=float, default=-30.0,
                    help='mesh camera swing around the door normal [deg]')
    ap.add_argument('--zoom', type=float, default=1.0, help='mesh camera zoom')
    ap.add_argument('--elev', type=float, default=40.0, help='--render stick only')
    ap.add_argument('--azim', type=float, default=-75.0, help='--render stick only')
    args = ap.parse_args()

    if args.save:
        plt.switch_backend('Agg')

    door = DoorSpec(handle_z=args.handle_z, theta_end_deg=args.theta_end)
    base = BasePose(x=args.base_xy[0], y=args.base_xy[1])
    arm = Arm()

    if args.sweep:
        sweep_figure(arm, door, base, args.save, grid=args.grid)
        return

    if args.scenario == 'redundancy':
        variants = scenario_redundancy(arm, door, base, keys=args.laws)
    else:
        variants = SCENARIOS[args.scenario](arm, door, base)
    print(f'scenario "{args.scenario}": {len(variants)} variants')
    for v in variants:
        v.roll = rollout_door(arm, v.door, v.base, v.q0, law=v.law,
                              dtheta_deg=args.dtheta)
        extra = (f'   boundary switches: {v.roll.switches}'
                 if v.law.hybrid else '')
        print(f'  {v.label:46s} theta_max = {v.roll.theta_max_deg:6.1f}°   '
              f'stop: {v.roll.reason_str}{extra}')

    if not args.no_reach:
        for k, v in enumerate(variants):
            v.reach_deg = math.degrees(pointwise_reachable(arm, v.door, v.base)[2])
            print(f'  {chr(65 + k)}: a point-wise reachability map would promise '
                  f'{v.reach_deg:5.1f}° of this arc')

    if args.viewer:
        play_in_viewer(arm, variants, fps=args.fps)
        return

    animate(arm, variants, SCENARIO_TITLE[args.scenario], fps=args.fps,
            save=args.save, spheres=args.spheres, elev=args.elev, azim=args.azim,
            render=args.render, cam_elev=args.cam_elev, cam_azim=args.cam_azim,
            zoom=args.zoom, overlay=args.overlay)


if __name__ == '__main__':
    main()
