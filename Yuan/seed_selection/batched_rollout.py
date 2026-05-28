"""Batched rollout: ``B`` ``(q, c)`` pairs evaluated in one ``env.step`` loop.

Day-1/2/3 used a single-env rollout (~12 s each on this box). For Module 8's
batch dataset generation that scales to thousands of c's, we need batched
rollouts — n_envs=64 brings per-rollout cost to near-zero.

The trick: instead of looping over (q, c) pairs and calling ``rollout_one``
each time, we pack them into a single NSRLBatchedEnv of size n_envs and use
``ScriptedLineDistribution`` to feed each env its own (q, c). The env's
``auto_reset=False`` mode handles per-env termination: finished envs freeze
while active ones keep stepping.

The same ``env.p_start[:] = stacked_p0`` override from ``rollout.py`` is
applied here so perturbed-p0 tasks actually use the perturbed anchor.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.env.env import NSRLBatchedEnv, TERM_TRUNCATED
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution


@torch.no_grad()
def _rollout_one_chunk(
    qs_chunk: torch.Tensor,         # (n_envs, 7)
    p0s_chunk: torch.Tensor,        # (n_envs, 3)
    line_dirs_chunk: torch.Tensor,  # (n_envs, 3)
    n_targets_chunk: torch.Tensor,  # (n_envs, 3)
    *,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    pierce_collision=None,                  # FR3SphereCollision or None
    pierce_keep_mask: torch.Tensor | None = None,  # (S,) bool sphere mask
    pierce_plane_extent_m: float = 1.5,
) -> dict:
    """Run env to completion for exactly n_envs (q, c) pairs.

    Returns per-env progress (m), episode_len, term_reason as tensors.
    """
    assert qs_chunk.shape[0] == env.n_envs

    spec = {
        "q0":       qs_chunk,
        "line_dir": line_dirs_chunk,
        "n_target": n_targets_chunk,
    }
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    # Override the FK-derived p_start with the task-defined p0's.
    env.p_start[:] = p0s_chunk

    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    action_fn = cn_action_fn(controller)

    n = env.n_envs
    episode_progress = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    episode_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)

    # max_q_norm tracking: per env, max over time and joints of
    # |(q - center) / half|. 1.0 = at joint limit; 0 = perfectly centered.
    q_center = ((env.kin.lmt_lo + env.kin.lmt_up) * 0.5).to(dtype=env.kin.dtype)
    q_half   = ((env.kin.lmt_up - env.kin.lmt_lo) * 0.5).clamp_min(1e-6).to(dtype=env.kin.dtype)
    max_q_norm = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    # Account for initial q too (before any step).
    qn0 = ((env.q - q_center) / q_half).abs().amax(dim=-1)
    max_q_norm = torch.maximum(max_q_norm, qn0)

    # Plane-pierce tracking: cumulative "did arm ever straddle the bounded plane
    # at any point during the rollout". `ever_pierced[i] = True` means at some
    # step `t`, env i had spheres on both sides of the bounded plane.
    do_pierce = pierce_collision is not None
    if do_pierce:
        ever_pierced = torch.zeros(n, dtype=torch.bool, device=env.device)

        def _step_pierce(q_now):
            link_tfs = env.kin.link_transforms(q_now)
            sp = pierce_collision.sphere_positions(link_tfs)   # (n, S, 3)
            signed = ((sp - p_start[:, None, :]) * n_targets_chunk[:, None, :]).sum(dim=-1)
            proj_d = ((sp - p_start[:, None, :]) * line_dir[:, None, :]).sum(dim=-1)
            over = (proj_d >= 0.0) & (proj_d <= float(pierce_plane_extent_m))
            if pierce_keep_mask is not None:
                over = over & pierce_keep_mask.bool().unsqueeze(0)
            sm = signed.masked_fill(~over, float('nan'))
            has_pos = (sm > 0.0).any(dim=-1)
            has_neg = (sm < 0.0).any(dim=-1)
            return has_pos & has_neg

        # Initial config piercing
        ever_pierced = ever_pierced | _step_pierce(env.q)

    for _ in range(env.max_steps + 1):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        qn = ((env.q - q_center) / q_half).abs().amax(dim=-1)
        active = ~env.done_persistent
        max_q_norm = torch.where(active, torch.maximum(max_q_norm, qn), max_q_norm)
        if do_pierce:
            pierce_now = _step_pierce(env.q)
            ever_pierced = ever_pierced | (active & pierce_now)
        new_done = info["episode_done"]
        if bool(new_done.any().item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            progress = ((p_now - p_start) * line_dir).sum(-1)
            episode_progress[new_done] = progress[new_done]
            episode_len[new_done] = env.t[new_done]
            term[new_done] = info["term_reason"][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break
    # Anything still not finished hit max_steps.
    if (~finished).any():
        not_done = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        progress = ((p_now - p_start) * line_dir).sum(-1)
        episode_progress[not_done] = progress[not_done]
        episode_len[not_done] = env.t[not_done]
        term[not_done] = TERM_TRUNCATED

    out = {
        "episode_progress_m": episode_progress,
        "episode_len": episode_len,
        "term_reason": term,
        "max_q_norm": max_q_norm,
    }
    if do_pierce:
        out["ever_pierced"] = ever_pierced
    return out


@torch.no_grad()
def batched_rollout_many(
    qs: torch.Tensor,
    cs: list[dict],
    *,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    target_distance_m: float = 1.5,
    pierce_collision=None,
    pierce_keep_mask: torch.Tensor | None = None,
    pierce_plane_extent_m: float = 1.5,
) -> dict:
    """Rollout ``B`` (q, c) pairs through a single shared env.

    Args:
        qs: (B, 7) joint configurations on env.device.
        cs: list of length B, each ``{'p0': (3,), 'line_dir': (3,), 'n_target': (3,)}``.
        env: NSRLBatchedEnv; we chunk B into ``env.n_envs``-sized batches.
            For B not divisible by ``env.n_envs``, the last chunk is padded
            by repeating its final entry (padded results are discarded).
        controller: bound classical-nullspace controller.
        target_distance_m: normalizer for L (default 1.5 — matches v18
            convention).

    Returns:
        dict with NumPy arrays of length B:
            L                     float32, ``episode_progress / target_distance_m``
            episode_progress_m    float32
            episode_len           int64
            term_reason           int32
    """
    B = qs.shape[0]
    assert len(cs) == B, f"qs has {B} entries but cs has {len(cs)}"
    n_envs = env.n_envs
    dtype = env.kin.dtype
    device = env.device

    # Stack the c's into per-field tensors once (cheaper than re-stacking
    # per chunk inside the loop).
    p0_full = torch.stack([c["p0"].to(device=device, dtype=dtype) for c in cs], dim=0)
    d_full = torch.stack([c["line_dir"].to(device=device, dtype=dtype) for c in cs], dim=0)
    n_full = torch.stack([c["n_target"].to(device=device, dtype=dtype) for c in cs], dim=0)
    q_full = qs.to(device=device, dtype=dtype)

    L_out = np.zeros(B, dtype=np.float32)
    prog_out = np.zeros(B, dtype=np.float32)
    len_out = np.zeros(B, dtype=np.int64)
    term_out = np.zeros(B, dtype=np.int32)
    mqn_out = np.zeros(B, dtype=np.float32)
    pierce_out = np.zeros(B, dtype=bool) if pierce_collision is not None else None

    n_chunks = math.ceil(B / n_envs)
    for ci in range(n_chunks):
        start = ci * n_envs
        end = min(start + n_envs, B)
        real_n = end - start
        # Build a chunk of exactly n_envs (pad if real_n < n_envs).
        if real_n == n_envs:
            qs_c = q_full[start:end]
            p0_c = p0_full[start:end]
            d_c = d_full[start:end]
            n_c = n_full[start:end]
        else:
            pad = n_envs - real_n
            qs_c = torch.cat([q_full[start:end], q_full[end - 1:end].expand(pad, 7)], dim=0)
            p0_c = torch.cat([p0_full[start:end], p0_full[end - 1:end].expand(pad, 3)], dim=0)
            d_c = torch.cat([d_full[start:end], d_full[end - 1:end].expand(pad, 3)], dim=0)
            n_c = torch.cat([n_full[start:end], n_full[end - 1:end].expand(pad, 3)], dim=0)

        res = _rollout_one_chunk(
            qs_c, p0_c, d_c, n_c, env=env, controller=controller,
            pierce_collision=pierce_collision,
            pierce_keep_mask=pierce_keep_mask,
            pierce_plane_extent_m=pierce_plane_extent_m,
        )
        progress = res["episode_progress_m"][:real_n].detach().cpu().numpy()
        prog_out[start:end] = progress
        L_out[start:end] = progress / float(target_distance_m)
        len_out[start:end] = res["episode_len"][:real_n].detach().cpu().numpy()
        term_out[start:end] = res["term_reason"][:real_n].detach().cpu().numpy()
        mqn_out[start:end] = res["max_q_norm"][:real_n].detach().cpu().numpy()
        if pierce_out is not None and "ever_pierced" in res:
            pierce_out[start:end] = res["ever_pierced"][:real_n].detach().cpu().numpy()

    out = {
        "L": L_out,
        "episode_progress_m": prog_out,
        "episode_len": len_out,
        "term_reason": term_out,
        "max_q_norm": mqn_out,
    }
    if pierce_out is not None:
        out["ever_pierced"] = pierce_out
    return out
