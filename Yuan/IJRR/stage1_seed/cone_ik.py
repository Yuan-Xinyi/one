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
when `Yuan.IJRR.kinematics.config.INIT_IK_ORIENT_MODE = "z_axis"` (the
default) — meaning the x/y columns of R_tgt are ignored. We still build a
stable R_tgt (using line_dir as an orientation hint) so the function behaves
predictably if the global config is ever switched to "full_rot".
"""
from __future__ import annotations

import numpy as np
import torch



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

