"""Physical validity checks beyond a candidate generator's IK flag."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from Yuan.IJRR.stage1_seed.candidate_batch import SeedCandidateBatch


@dataclass
class CandidateValidity:
    valid: torch.Tensor
    finite: torch.Tensor
    joint_limits: torch.Tensor
    position: torch.Tensor
    cone: torch.Tensor
    collision_free: torch.Tensor
    position_error_m: torch.Tensor
    cone_cosine: torch.Tensor


@torch.no_grad()
def check_candidate_validity(
    kin,
    collision,
    candidates: SeedCandidateBatch,
    *,
    position_tol_m: float = 5e-3,
    cone_deg: float = 30.0,
    joint_tol: float = 0.0,
) -> CandidateValidity:
    """Combine cached IK validity with limits/FK/cone/collision checks."""
    if position_tol_m <= 0:
        raise ValueError('position_tol_m must be positive')
    if joint_tol < 0:
        raise ValueError('joint_tol must be non-negative')
    b, k = candidates.n_tasks, candidates.n_candidates
    q_raw = candidates.q0.reshape(-1, 7).to(
        device=kin.device, dtype=kin.dtype)
    p0 = candidates.p0[:, None, :].expand(-1, k, -1).reshape(-1, 3)
    n_target = candidates.n_target[:, None, :].expand(-1, k, -1).reshape(-1, 3)
    p0 = p0.to(device=kin.device, dtype=kin.dtype)
    n_target = n_target.to(device=kin.device, dtype=kin.dtype)

    finite = torch.isfinite(q_raw).all(dim=-1)
    in_limits = ((q_raw >= kin.lmt_lo - joint_tol)
                 & (q_raw <= kin.lmt_up + joint_tol)).all(dim=-1)
    cached_valid = candidates.valid.to(kin.device).reshape(-1)
    safe_fk = cached_valid & finite & in_limits
    # Failed IK candidates can be NaN or huge finite values. Batched FK and
    # collision kernels still see masked rows, so only evaluate cached-valid,
    # in-limit configurations and substitute a neutral pose everywhere else.
    q = torch.where(safe_fk.unsqueeze(-1), q_raw, kin.q_mid.expand_as(q_raw))
    p, rot, _, _ = kin.tcp_fk_jac(q)
    position_error = (p - p0).norm(dim=-1)
    position_error = torch.where(
        safe_fk, position_error,
        torch.full_like(position_error, float('inf')))
    position_ok = safe_fk & (position_error <= position_tol_m)
    cone_cosine = (rot[:, :, 2] * n_target).sum(dim=-1)
    cone_cosine = torch.where(
        safe_fk, cone_cosine,
        torch.full_like(cone_cosine, float('-inf')))
    cone_ok = safe_fk & (cone_cosine >= math.cos(math.radians(cone_deg)))
    if collision is None:
        collision_free = safe_fk.clone()
    else:
        collision_free = safe_fk & ~collision.is_collided(kin.link_transforms(q))
    valid = safe_fk & position_ok & cone_ok & collision_free

    def shape(value):
        return value.reshape(b, k)

    return CandidateValidity(
        valid=shape(valid),
        finite=shape(finite),
        joint_limits=shape(in_limits),
        position=shape(position_ok),
        cone=shape(cone_ok),
        collision_free=shape(collision_free),
        position_error_m=shape(position_error),
        cone_cosine=shape(cone_cosine),
    )

