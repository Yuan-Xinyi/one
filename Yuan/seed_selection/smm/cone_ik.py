"""Module 3a: cone-constrained IK enumeration.

Given a task ``c = (p0, line_dir, n_target)``, enumerate joint configurations
``q`` that simultaneously satisfy:

    FK(q).pos        ≈ p0                      (position constraint)
    ∠(FK(q).z, n_target)  ≤ cone_angle_deg     (cone constraint)
    joint_margin away from all limits          (interior of joint box)
    no self-collision

The aim is BRANCH COVERAGE for downstream SMM walks — return as many distinct
q's as possible (different SMM connected components on the constraint set).

Strategy ("方案 A" from the design discussion):
    For each of n_orientations sampled ``n_test`` in cone(n_target, cone_angle):
        For each of n_ik_restarts mixed seeds (uniform + boundary-biased):
            solve DLS IK to (p0, R_tgt with z column = n_test)
    Validate, then dedup in joint space.

Implementation note: the underlying `_batched_ik_project` is z-axis-only
when `Yuan.flow_connectivity.config.INIT_IK_ORIENT_MODE = "z_axis"` (the
default) — meaning the x/y columns of R_tgt are ignored. We still build a
stable R_tgt (using line_dir as an orientation hint) so the function behaves
predictably if the global config is ever switched to "full_rot".
"""
from __future__ import annotations

import numpy as np
import torch

from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision


def _sample_in_cone(axis: torch.Tensor,
                    cone_angle_deg: float,
                    n: int,
                    rng: np.random.Generator) -> torch.Tensor:
    """Uniform-area sample on the spherical cap ``angle(v, axis) ≤ α``.

    axis: (3,) unit. Returns (n, 3) unit vectors.
    """
    if n <= 0:
        return torch.empty((0, 3), dtype=axis.dtype, device=axis.device)
    alpha = float(cone_angle_deg) * np.pi / 180.0
    # cos_theta uniform on [cos(α), 1] gives uniform area on the cap.
    cos_theta = rng.uniform(np.cos(alpha), 1.0, size=n).astype(np.float32)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, 1.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    # Local frame (z = axis).
    v_local = np.stack([sin_theta * np.cos(phi),
                        sin_theta * np.sin(phi),
                        cos_theta], axis=-1)  # (n, 3)
    # Build a frame whose z-axis is `axis`, then rotate v_local into world.
    axis_np = axis.detach().cpu().numpy().astype(np.float32)
    axis_np = axis_np / (np.linalg.norm(axis_np) + 1e-12)
    # Choose a stable orthogonal vector for x_local.
    hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(hint @ axis_np)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_local = hint - (hint @ axis_np) * axis_np
    x_local = x_local / (np.linalg.norm(x_local) + 1e-12)
    y_local = np.cross(axis_np, x_local)
    R = np.stack([x_local, y_local, axis_np], axis=-1)  # (3, 3)
    v_world = v_local @ R.T  # (n, 3)
    return torch.as_tensor(v_world, dtype=axis.dtype, device=axis.device)


def _build_R_with_z(z: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
    """Build (B, 3, 3) rotation matrices whose z column = ``z``.

    ``hint`` is a (3,) world vector used to define the x column via
    Gram-Schmidt (so the result is stable / deterministic). Only the z
    column is used by z-axis-mode IK, but we keep x/y well-defined for
    safety if the IK config is ever switched to full-rotation mode.

    z: (B, 3) unit. hint: (3,) (need not be unit). Returns (B, 3, 3).
    """
    B = z.shape[0]
    dtype, device = z.dtype, z.device
    hint = hint / hint.norm().clamp_min(1e-12)
    hint_b = hint.unsqueeze(0).expand(B, 3)
    x_raw = hint_b - (hint_b * z).sum(-1, keepdim=True) * z
    x_norm = x_raw.norm(dim=-1, keepdim=True)
    # Fallback for degenerate cases: pick world y if hint is parallel to z.
    fallback = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    fb_b = fallback.unsqueeze(0).expand(B, 3)
    fb_raw = fb_b - (fb_b * z).sum(-1, keepdim=True) * z
    use_fb = (x_norm.squeeze(-1) < 1e-6).unsqueeze(-1)
    x_raw = torch.where(use_fb, fb_raw, x_raw)
    x = x_raw / x_raw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)  # (B, 3, 3) with cols [x, y, z]


def _mixed_seeds(n_total: int,
                 kin: BatchedFR3Kinematics,
                 rng: np.random.Generator,
                 boundary_pct: float = 0.03) -> torch.Tensor:
    """Match `_dense_ik_at`'s 50/50 uniform/boundary seed sampler.

    Returns (n_total, 7) torch tensor on kin.device.
    """
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    span = hi - lo
    n_unif = n_total // 2
    n_edge = n_total - n_unif
    seeds_unif = rng.uniform(lo[None, :], hi[None, :],
                             size=(n_unif, 7)).astype(np.float32)
    seeds_edge = rng.uniform(lo[None, :], hi[None, :],
                             size=(n_edge, 7)).astype(np.float32)
    for i in range(n_edge):
        n_extreme = int(rng.integers(1, 3))
        joints = rng.choice(7, size=n_extreme, replace=False)
        for j in joints:
            if int(rng.integers(0, 2)) == 0:
                seeds_edge[i, j] = lo[j] + boundary_pct * span[j]
            else:
                seeds_edge[i, j] = hi[j] - boundary_pct * span[j]
    seeds = np.concatenate([seeds_unif, seeds_edge], axis=0)
    return torch.as_tensor(seeds, device=kin.device, dtype=kin.dtype)


def _angle_to_axis(z_actual: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Angle (radians) between rows of ``z_actual`` (B, 3) and ``axis`` (3,)."""
    cos_a = (z_actual * axis.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
    return torch.arccos(cos_a)


def _dedup_q(Q: torch.Tensor, dedup_rad: float) -> torch.Tensor:
    """Greedy first-fit dedup: keep ``q`` only if it is ≥ dedup_rad from all
    already-kept points (Euclidean in joint space).

    O(N²); fine for our ~50-candidate scale. Returns (M, 7).
    """
    if Q.shape[0] <= 1:
        return Q
    kept_idx: list[int] = [0]
    for i in range(1, Q.shape[0]):
        diff = Q[kept_idx] - Q[i].unsqueeze(0)
        if float(diff.norm(dim=-1).min()) >= dedup_rad:
            kept_idx.append(i)
    return Q[kept_idx]


def cone_constrained_ik_enumerate(
    *,
    p0: torch.Tensor,
    n_target: torch.Tensor,
    line_dir: torch.Tensor,
    kin: BatchedFR3Kinematics,
    collision: FR3SphereCollision,
    cone_angle_deg: float = 5.0,
    n_orientations: int = 10,
    n_ik_restarts: int = 5,
    include_center: bool = True,
    joint_margin: float = 0.05,
    dedup_rad: float | None = 0.08,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Enumerate IK solutions on the cone-constrained set for task ``c``.

    Args:
        p0: (3,) TCP target position.
        n_target: (3,) unit, cone center axis (TCP z should align with this).
        line_dir: (3,) unit, the task's motion direction. Used only as an
            orientation hint for building R_tgt's x column (stability;
            ignored by z-axis-mode IK).
        kin, collision: FR3 kinematics + sphere collision checker.
        cone_angle_deg: cone half-angle (default 5°, matches
            `LineDistribution.n_target_noise_deg`).
        n_orientations: how many ``n_test`` directions to sample within the
            cone (the first is ``n_target`` itself when include_center=True).
        n_ik_restarts: random seeds per ``n_test``.
        include_center: include ``n_target`` as the first ``n_test``.
        joint_margin: drop q's within this distance of any joint limit (rad).
        dedup_rad: pairwise Euclidean distance (rad) below which two q's are
            merged. Set None to skip dedup.
        rng: numpy RNG for reproducibility (seed sampling + cone sampling).
            If None, a fresh default_rng() is created.

    Returns:
        (N, 7) tensor of validated, deduped IK solutions. May be empty.
    """
    if rng is None:
        rng = np.random.default_rng()
    p0 = p0.to(device=kin.device, dtype=kin.dtype)
    n_target = n_target.to(device=kin.device, dtype=kin.dtype)
    n_target = n_target / n_target.norm().clamp_min(1e-12)
    line_dir = line_dir.to(device=kin.device, dtype=kin.dtype)

    # 1) sample n_orientations n_test inside the cone.
    if include_center:
        n_extra = n_orientations - 1
        n_extra_samples = _sample_in_cone(n_target, cone_angle_deg, n_extra, rng)
        n_tests = torch.cat([n_target.unsqueeze(0), n_extra_samples], dim=0)
    else:
        n_tests = _sample_in_cone(n_target, cone_angle_deg, n_orientations, rng)
    # 2) replicate each n_test across n_ik_restarts, then sample seeds.
    n_total = n_orientations * n_ik_restarts
    n_tests_rep = n_tests.repeat_interleave(n_ik_restarts, dim=0)  # (n_total, 3)
    R_tgts = _build_R_with_z(n_tests_rep, line_dir)                 # (n_total, 3, 3)
    seeds = _mixed_seeds(n_total, kin, rng)                         # (n_total, 7)
    p0_rep = p0.unsqueeze(0).expand(n_total, 3)

    # 3) one batched DLS IK call.
    q_out, ok, _ = _batched_ik_project(kin, seeds, p0_rep, R_tgts,
                                        branch_action=None)
    if not bool(ok.any()):
        return torch.empty((0, 7), device=kin.device, dtype=kin.dtype)

    q_ok = q_out[ok]
    # 4) validate joint-limit margin.
    lo, hi = kin.lmt_lo, kin.lmt_up
    margin = torch.minimum(q_ok - lo, hi - q_ok).min(dim=-1).values
    keep_jl = margin >= joint_margin
    if not bool(keep_jl.any()):
        return torch.empty((0, 7), device=kin.device, dtype=kin.dtype)
    q_ok = q_ok[keep_jl]

    # 5) validate cone constraint (FK on full q batch).
    _, R_actual, _, _ = kin.tcp_fk_jac(q_ok)
    z_actual = R_actual[:, :, 2]
    ang = _angle_to_axis(z_actual, n_target)
    keep_cone = ang <= float(cone_angle_deg) * np.pi / 180.0
    if not bool(keep_cone.any()):
        return torch.empty((0, 7), device=kin.device, dtype=kin.dtype)
    q_ok = q_ok[keep_cone]

    # 6) validate no self-collision.
    link_tfs = kin.link_transforms(q_ok)
    is_coll = collision.is_collided(link_tfs)
    q_ok = q_ok[~is_coll]
    if q_ok.shape[0] == 0:
        return q_ok

    # 7) dedup in joint space.
    if dedup_rad is not None and dedup_rad > 0.0:
        q_ok = _dedup_q(q_ok, dedup_rad)
    return q_ok
