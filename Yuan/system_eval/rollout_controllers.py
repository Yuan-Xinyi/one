"""Batched rollouts for each controller variant used by the 5-cell ablation.

This module provides one entry point — `rollout_seeds_batched` — which
takes a flat (B, 7) tensor of seeds + per-seed (p0, line_dir, n_target),
runs them through a shared `NSRLBatchedEnv`, and returns per-seed metrics.
Two underlying controllers are supported:

  controller='classical'         — Yoshikawa-style nullspace (current code-path
                                    in `seed_selection.batched_rollout`).
  controller='hybrid_variantB'   — step-level hysteresis switching between RL
                                    (agent.pt) and Classical, as in
                                    `RL_controller.eval_hybrid_steplevel`.

Common to both: env.p_start is overridden to the task-defined p0 so the
progress signal aligns with what the data labels assume (matches
seed_selection.batched_rollout).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_TRUNCATED, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.ppo import Agent


# ----------------------------------------------------------------------
# Env construction
# ----------------------------------------------------------------------

def build_env(env_yaml: str | Path, n_envs: int, device: torch.device) -> NSRLBatchedEnv:
    import dataclasses
    with open(env_yaml, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg_yaml['env'].items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, 'n_envs': int(n_envs)})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    return env


def load_rl_agent(ckpt_dir: str | Path, env: NSRLBatchedEnv,
                  device: torch.device) -> Agent:
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / 'config.yaml', 'r') as f:
        rl_cfg = yaml.safe_load(f)
    agent = Agent(
        env.obs_dim, env.act_dim,
        hidden_dim=rl_cfg['ppo']['hidden_dim'],
        init_log_std=rl_cfg['ppo']['init_log_std'],
    ).to(device)
    sd = torch.load(ckpt_dir / 'agent.pt', map_location=device)
    agent.load_state_dict(sd)
    agent.eval()
    return agent


# ----------------------------------------------------------------------
# Single-chunk rollout (per-controller)
# ----------------------------------------------------------------------

@torch.no_grad()
def _rollout_classical_chunk(
    qs_chunk: torch.Tensor,
    p0s_chunk: torch.Tensor,
    line_dirs_chunk: torch.Tensor,
    n_targets_chunk: torch.Tensor,
    *,
    env: NSRLBatchedEnv,
    classical: ClassicalNullspaceController,
) -> dict:
    """Pure classical rollout (mirrors seed_selection.batched_rollout)."""
    assert qs_chunk.shape[0] == env.n_envs
    spec = {'q0': qs_chunk, 'line_dir': line_dirs_chunk, 'n_target': n_targets_chunk}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = p0s_chunk

    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    action_fn = cn_action_fn(classical)

    n = env.n_envs
    progress = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    ep_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)

    for _ in range(env.max_steps + 1):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info['episode_done']
        if bool(new_done.any().item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            prog = ((p_now - p_start) * line_dir).sum(-1)
            progress[new_done] = prog[new_done]
            ep_len[new_done] = env.t[new_done]
            term[new_done] = info['term_reason'][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break

    if (~finished).any():
        nd = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p_now - p_start) * line_dir).sum(-1)
        progress[nd] = prog[nd]
        ep_len[nd] = env.t[nd]
        term[nd] = TERM_TRUNCATED

    return {'progress_m': progress, 'episode_len': ep_len, 'term_reason': term}


@torch.no_grad()
def _rollout_hybrid_variantB_chunk(
    qs_chunk: torch.Tensor,
    p0s_chunk: torch.Tensor,
    line_dirs_chunk: torch.Tensor,
    n_targets_chunk: torch.Tensor,
    *,
    env: NSRLBatchedEnv,
    classical: ClassicalNullspaceController,
    agent: Agent,
    tau_enter: float,
    tau_exit: float,
) -> dict:
    """Step-level hysteresis hybrid: at each step,
        if using_rl and  max|q_norm(q_t)| >= tau_enter -> switch to Cls (and snapshot q_ref := q_t)
        if using_cls and max|q_norm(q_t)| <  tau_exit  -> switch back to RL
    The classical branch uses the per-env q_ref attractor.
    """
    assert qs_chunk.shape[0] == env.n_envs
    spec = {'q0': qs_chunk, 'line_dir': line_dirs_chunk, 'n_target': n_targets_chunk}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = p0s_chunk

    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    n = env.n_envs

    q_mid = env.q_mid
    q_half = env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    init_qn = _max_abs_qn(env.q)
    using_rl = init_qn < tau_enter
    q_ref = env.q.clone()                           # per-env attractor

    progress = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    ep_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    switch_count = torch.zeros(n, dtype=torch.long, device=env.device)

    for _ in range(env.max_steps + 1):
        cur_qn = _max_abs_qn(env.q)
        new_using_rl = torch.where(
            using_rl,
            cur_qn < tau_enter,            # RL: stay if still under enter threshold
            cur_qn < tau_exit,             # Cls: come back only if dropped past exit
        )
        switched = new_using_rl != using_rl
        # On RL -> Cls, snapshot the q_ref attractor at the switch point.
        rl_to_cls = using_rl & (~new_using_rl)
        if bool(rl_to_cls.any().item()):
            q_ref = torch.where(rl_to_cls.unsqueeze(-1), env.q, q_ref)
        active = ~finished
        switch_count = switch_count + (switched & active).long()
        using_rl = new_using_rl

        obs = env.current_obs()
        rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)

        # Classical action: q_dot_null -> task-aligned basis projection.
        B_basis, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping,
        )
        q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target, q_ref)
        cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
        cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)

        a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info['episode_done']
        if bool(new_done.any().item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            prog = ((p_now - p_start) * line_dir).sum(-1)
            progress[new_done] = prog[new_done]
            ep_len[new_done] = env.t[new_done]
            term[new_done] = info['term_reason'][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break

    if (~finished).any():
        nd = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p_now - p_start) * line_dir).sum(-1)
        progress[nd] = prog[nd]
        ep_len[nd] = env.t[nd]
        term[nd] = TERM_TRUNCATED

    return {
        'progress_m': progress,
        'episode_len': ep_len,
        'term_reason': term,
        'switch_count': switch_count,
        'init_max_qn': init_qn,
    }


# ----------------------------------------------------------------------
# Batched chunked driver
# ----------------------------------------------------------------------

@torch.no_grad()
def rollout_seeds_batched(
    qs_np: np.ndarray,             # (B, 7)
    p0s_np: np.ndarray,            # (B, 3)
    line_dirs_np: np.ndarray,      # (B, 3)
    n_targets_np: np.ndarray,      # (B, 3)
    *,
    env: NSRLBatchedEnv,
    controller: str,                # 'classical' or 'hybrid_variantB'
    classical: ClassicalNullspaceController,
    agent: Agent | None = None,
    tau_enter: float = 0.98,
    tau_exit: float = 0.98,
    target_distance_m: float = 1.5,
    progress_every_chunks: int = 20,
    progress_prefix: str = '',
) -> dict:
    """Run `B` rollouts through `env` (chunked into env.n_envs at a time).

    Returns dict of numpy arrays of length B:
        L                  float32 (progress / target_distance_m)
        episode_progress_m float32
        episode_len        int64
        term_reason        int32
        init_max_qn        float32  — computed for all seeds (cheap, useful)
        switch_count       int32    — populated only for hybrid_variantB
    """
    B = qs_np.shape[0]
    assert p0s_np.shape == (B, 3) and line_dirs_np.shape == (B, 3) and n_targets_np.shape == (B, 3)
    n_envs = env.n_envs
    dtype = env.kin.dtype
    device = env.device

    q_full   = torch.as_tensor(qs_np,        device=device, dtype=dtype)
    p0_full  = torch.as_tensor(p0s_np,       device=device, dtype=dtype)
    d_full   = torch.as_tensor(line_dirs_np, device=device, dtype=dtype)
    n_full   = torch.as_tensor(n_targets_np, device=device, dtype=dtype)

    L_out         = np.zeros(B, dtype=np.float32)
    prog_out      = np.zeros(B, dtype=np.float32)
    len_out       = np.zeros(B, dtype=np.int64)
    term_out      = np.zeros(B, dtype=np.int32)
    init_qn_out   = np.zeros(B, dtype=np.float32)
    switch_out    = np.zeros(B, dtype=np.int32)

    # Precompute init_max_qn from input qs (cheap; independent of rollout).
    q_mid_np = env.q_mid.detach().cpu().numpy().astype(np.float64)
    q_half_np = env.q_half.detach().cpu().numpy().astype(np.float64)
    init_qn_out[:] = np.abs((qs_np.astype(np.float64) - q_mid_np) / q_half_np).max(axis=-1).astype(np.float32)

    n_chunks = math.ceil(B / n_envs)
    for ci in range(n_chunks):
        start = ci * n_envs
        end = min(start + n_envs, B)
        real_n = end - start
        if real_n == n_envs:
            qs_c = q_full[start:end]
            p0_c = p0_full[start:end]
            d_c  = d_full[start:end]
            n_c  = n_full[start:end]
        else:
            pad = n_envs - real_n
            qs_c = torch.cat([q_full[start:end], q_full[end - 1:end].expand(pad, 7)], dim=0)
            p0_c = torch.cat([p0_full[start:end], p0_full[end - 1:end].expand(pad, 3)], dim=0)
            d_c  = torch.cat([d_full[start:end],  d_full[end - 1:end].expand(pad, 3)], dim=0)
            n_c  = torch.cat([n_full[start:end],  n_full[end - 1:end].expand(pad, 3)], dim=0)

        if controller == 'classical':
            res = _rollout_classical_chunk(qs_c, p0_c, d_c, n_c,
                                           env=env, classical=classical)
        elif controller == 'hybrid_variantB':
            assert agent is not None, 'hybrid_variantB requires agent'
            res = _rollout_hybrid_variantB_chunk(qs_c, p0_c, d_c, n_c,
                                                  env=env, classical=classical,
                                                  agent=agent,
                                                  tau_enter=tau_enter,
                                                  tau_exit=tau_exit)
        else:
            raise ValueError(f'unknown controller: {controller!r}')

        prog_out[start:end] = res['progress_m'][:real_n].detach().cpu().numpy()
        L_out[start:end]    = prog_out[start:end] / float(target_distance_m)
        len_out[start:end]  = res['episode_len'][:real_n].detach().cpu().numpy()
        term_out[start:end] = res['term_reason'][:real_n].detach().cpu().numpy()
        if 'switch_count' in res:
            switch_out[start:end] = res['switch_count'][:real_n].detach().cpu().numpy()

        if progress_every_chunks and (ci % progress_every_chunks == 0 or ci == n_chunks - 1):
            done = min(end, B)
            print(f'  {progress_prefix}{done}/{B} seeds rolled ({controller})', flush=True)

    return {
        'L': L_out,
        'episode_progress_m': prog_out,
        'episode_len': len_out,
        'term_reason': term_out,
        'init_max_qn': init_qn_out,
        'switch_count': switch_out,
    }
