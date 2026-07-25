"""Candidate features shared by the seed policy and feasibility critic."""
from __future__ import annotations

import torch

from Yuan.RL_controller.env.env import LATERAL_SAFETY_NET
from Yuan.unified_rl.candidate_batch import SeedCandidateBatch


@torch.no_grad()
def initial_observation_features(kin, candidates: SeedCandidateBatch,
                                 include_log_manip: bool = False,
                                 include_ray_error: bool = False,
                                 include_directional_dynamics: bool = False,
                                 ) -> torch.Tensor:
    """Build the controller's initial observation for every seed candidate.

    The first 31 dimensions exactly mirror ``NSRLBatchedEnv._compute_obs``
    with ``a_prev = 0``. ``include_ray_error`` appends the same
    normalized lateral ray-error term used by the 34-D unified controller. An
    optional scalar is log Yoshikawa positional manipulability, matching the
    strongest historical ranker. ``include_directional_dynamics`` appends ten
    task-aligned differential-kinematics features: unit-line joint velocity
    (7), its norm, a linearized joint-limit horizon, and directional
    manipulability.
    """
    b, k = candidates.n_tasks, candidates.n_candidates
    q_raw = candidates.q0.reshape(b * k, 7).to(
        device=kin.device, dtype=kin.dtype)
    valid = candidates.valid.reshape(-1).to(device=kin.device)
    finite_q = torch.isfinite(q_raw).all(dim=-1)
    in_limits = ((q_raw >= kin.lmt_lo) & (q_raw <= kin.lmt_up)).all(dim=-1)
    bad_valid = valid & (~finite_q | ~in_limits)
    if bool(bad_valid.any().item()):
        flat_index = torch.nonzero(bad_valid, as_tuple=False).flatten().cpu()
        locations = [(int(index) // k, int(index) % k)
                     for index in flat_index[:8]]
        suffix = ' ...' if flat_index.numel() > len(locations) else ''
        raise ValueError(
            'valid seed candidates contain non-finite or out-of-limit q at '
            f'(task, candidate) indices {locations}{suffix}')

    # Masked candidates are not actions and may contain failed-IK NaNs. FK
    # still evaluates every row, so replace those values before the batched
    # call and explicitly zero their features below.
    q_mid = kin.q_mid.expand_as(q_raw)
    q = torch.where((valid & finite_q & in_limits).unsqueeze(-1), q_raw, q_mid)
    line_dir = candidates.line_dir[:, None, :].expand(-1, k, -1).reshape(-1, 3)
    n_target = candidates.n_target[:, None, :].expand(-1, k, -1).reshape(-1, 3)
    line_dir = line_dir.to(device=kin.device, dtype=kin.dtype)
    n_target = n_target.to(device=kin.device, dtype=kin.dtype)

    position, rot, jac, _ = kin.tcp_fk_jac(q)
    z_tool = rot[:, :, 2]
    q_half = 0.5 * (kin.lmt_up - kin.lmt_lo)
    q_norm = (q - kin.q_mid) / q_half
    cos_angle = (z_tool * n_target).sum(-1, keepdim=True)
    z_cross_n = torch.linalg.cross(z_tool, n_target, dim=-1)
    features = torch.cat([
        q_norm,
        q_norm.square(),
        line_dir,
        z_tool,
        n_target,
        cos_angle,
        z_cross_n,
        torch.zeros((b * k, 4), device=kin.device, dtype=kin.dtype),
    ], dim=-1)

    if include_ray_error:
        p0 = candidates.p0[:, None, :].expand(-1, k, -1).reshape(-1, 3)
        p0 = p0.to(device=kin.device, dtype=kin.dtype)
        ray_delta = position - p0
        ray_along = (ray_delta * line_dir).sum(-1, keepdim=True)
        lateral_offset = ray_delta - ray_along * line_dir
        features = torch.cat(
            [features, lateral_offset / LATERAL_SAFETY_NET], dim=-1)

    if include_log_manip:
        j_pos = jac[:, :3, :]
        manip = torch.det(j_pos @ j_pos.transpose(-1, -2)).clamp_min(1e-12).sqrt()
        features = torch.cat([features, manip.log().unsqueeze(-1)], dim=-1)

    if include_directional_dynamics:
        j_pos = jac[:, :3, :]
        gram = j_pos @ j_pos.transpose(-1, -2)
        eye = torch.eye(
            3, device=gram.device, dtype=gram.dtype).expand_as(gram)
        # A fixed, small damping makes the descriptor continuous at rank loss
        # without tying seed-policy inputs to a particular controller config.
        damped = gram + 1e-5 * eye
        solved_line = torch.linalg.solve(
            damped, line_dir.unsqueeze(-1)).squeeze(-1)
        dq_line = (
            j_pos.transpose(-1, -2) @ solved_line.unsqueeze(-1)).squeeze(-1)
        dq_norm = dq_line.norm(dim=-1, keepdim=True)

        eps = 1e-5
        upper_horizon = (kin.lmt_up - q) / dq_line.clamp_min(eps)
        lower_horizon = (kin.lmt_lo - q) / dq_line.clamp_max(-eps)
        infinite = torch.full_like(dq_line, torch.inf)
        per_joint_horizon = torch.where(
            dq_line > eps, upper_horizon,
            torch.where(dq_line < -eps, lower_horizon, infinite))
        joint_horizon = torch.nan_to_num(
            per_joint_horizon.min(dim=-1, keepdim=True).values,
            nan=0.0, posinf=100.0, neginf=0.0).clamp(0.0, 100.0)
        inverse_directional_effort = (
            line_dir * solved_line).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        directional_manip = inverse_directional_effort.rsqrt()
        features = torch.cat([
            features, dq_line, dq_norm, joint_horizon, directional_manip,
        ], dim=-1)
    features = features.reshape(b, k, -1)
    valid_matrix = candidates.valid.to(device=features.device)
    nonfinite_valid = valid_matrix & ~torch.isfinite(features).all(dim=-1)
    if bool(nonfinite_valid.any().item()):
        locations = torch.nonzero(nonfinite_valid, as_tuple=False).cpu().tolist()
        suffix = ' ...' if len(locations) > 8 else ''
        raise ValueError(
            'valid seed candidates produced non-finite features at '
            f'(task, candidate) indices {locations[:8]}{suffix}')
    features = features.masked_fill(~valid_matrix.unsqueeze(-1), 0.0)
    return features.float()


@torch.no_grad()
def fit_candidate_feature_normalization(
    kin,
    dataset,
    *,
    include_log_manip: bool = False,
    include_ray_error: bool = False,
    include_directional_dynamics: bool = False,
    chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Valid-only per-dimension moments over a cached training split."""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if len(dataset) < 1:
        raise ValueError('cannot fit feature statistics on an empty dataset')
    total = 0
    feature_sum = None
    feature_square_sum = None
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        batch = dataset.batch.index_select(
            torch.arange(start, end)).to(kin.device, dtype=kin.dtype)
        features = initial_observation_features(
            kin, batch,
            include_log_manip=include_log_manip,
            include_ray_error=include_ray_error,
            include_directional_dynamics=include_directional_dynamics)
        values = features[batch.valid].double()
        if feature_sum is None:
            feature_sum = torch.zeros(
                values.shape[-1], device=values.device, dtype=torch.float64)
            feature_square_sum = torch.zeros_like(feature_sum)
        feature_sum += values.sum(dim=0)
        feature_square_sum += values.square().sum(dim=0)
        total += values.shape[0]
    if total == 0:
        raise ValueError('training split has no valid candidate features')
    mean = feature_sum / total
    variance = feature_square_sum / total - mean.square()
    std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6)
    return mean.float(), std.float()
