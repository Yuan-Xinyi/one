"""Physical validity checks beyond a candidate generator's IK flag."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedCandidateBatch,
)


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


def assert_same_valid_mask(
    dataset: CachedSeedCandidateDataset,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    """Reject a derived candidate action set that changed across runs."""
    expected = torch.as_tensor(expected, device='cpu', dtype=torch.bool)
    current = dataset.batch.valid.cpu()
    if expected.shape != current.shape:
        raise ValueError(
            f'{label} candidate valid mask shape changed from '
            f'{tuple(expected.shape)} to {tuple(current.shape)}')
    if not torch.equal(expected, current):
        changed = int((expected != current).sum().item())
        raise ValueError(
            f'{label} candidate valid mask changed at {changed} slots; '
            'use the same physical-validation code and dependencies as the '
            'checkpointed run')


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


@torch.no_grad()
def validate_cached_dataset(
    dataset: CachedSeedCandidateDataset,
    kin,
    collision,
    *,
    chunk_size: int = 4096,
    position_tol_m: float = 5e-3,
    cone_deg: float = 30.0,
) -> tuple[CachedSeedCandidateDataset, dict[str, float | list[int]]]:
    """Precompute a strict mask and remove tasks with no feasible action.

    Rejected task indices refer to the source cache, even if ``dataset`` is
    already a subset. All task-conditioned tensors are filtered together.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if len(dataset) < 1:
        raise ValueError('cannot validate an empty candidate dataset')
    masks = []
    counts = {
        'cached': 0,
        'finite': 0,
        'joint_limits': 0,
        'position': 0,
        'cone': 0,
        'collision_free': 0,
        'valid': 0,
    }
    total = dataset.batch.n_tasks * dataset.batch.n_candidates
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        index = torch.arange(start, end)
        batch = dataset.batch.index_select(index).to(kin.device, dtype=kin.dtype)
        result = check_candidate_validity(
            kin, collision, batch,
            position_tol_m=position_tol_m, cone_deg=cone_deg)
        masks.append(result.valid.cpu())
        counts['cached'] += int(batch.valid.sum().item())
        for key in ('finite', 'joint_limits', 'position', 'cone',
                    'collision_free', 'valid'):
            counts[key] += int(getattr(result, key).sum().item())
    mask = torch.cat(masks)
    keep = mask.any(dim=1)
    rejected_local = torch.nonzero(~keep, as_tuple=False).flatten()
    rejected_task_indices = dataset.task_indices[rejected_local].tolist()
    if not bool(keep.any().item()):
        raise ValueError(
            'strict candidate validation rejected every task; original task '
            f'indices: {rejected_task_indices}')
    kept_local = torch.nonzero(keep, as_tuple=False).flatten()
    validated = dataset.index_select(kept_local).with_valid(mask[keep])
    stats: dict[str, float | list[int]] = {
        f'frac_{key}': value / max(total, 1)
        for key, value in counts.items()
    }
    stats['n_tasks'] = float(len(dataset))
    stats['n_tasks_retained'] = float(len(validated))
    stats['n_tasks_rejected'] = float(rejected_local.numel())
    stats['n_candidates'] = float(dataset.batch.n_candidates)
    stats['rejected_task_indices'] = rejected_task_indices
    return validated, stats
