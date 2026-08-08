"""Generic single-episode rollout helper.

Runs `env` with `auto_reset=False` until every env's first episode ends (or
`max_steps` cap), records per-env episode_len, term_reason, and progress (EE
travel along u_hat). Used by train (periodic eval), line_distribution
(feasibility filter), and anywhere else that needs deterministic one-shot
rollouts under a given action_fn.
"""
from __future__ import annotations

import torch

from Yuan.IJRR.env.env import NSRLBatchedEnv, TERM_TRUNCATED


@torch.no_grad()
def rollout_first_episode(env: NSRLBatchedEnv, action_fn,
                          max_steps: int | None = None) -> dict:
    """Run env with auto_reset=False until every env's first episode ends.

    `action_fn(env)` returns (B, ACT_DIM) action ∈ [-1, 1] using env's current
    state. Finished envs freeze (env handles this internally); we still call
    step() each tick so the active envs advance.

    Returns per-env episode_len (steps), term_reason, and progress (m, the
    EE travel along u_hat = (p_final - p_start) · u_hat).
    """
    cfg_max = env.max_steps if max_steps is None else max_steps
    n = env.n_envs
    episode_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    episode_term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    episode_progress = torch.zeros((n,), dtype=env.kin.dtype, device=env.device)
    finished = torch.zeros((n,), dtype=torch.bool, device=env.device)

    env.reset()
    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    for step_i in range(cfg_max + 1):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info["episode_done"]
        if new_done.any():
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            progress = ((p_now - p_start) * line_dir).sum(-1)
            episode_progress[new_done] = progress[new_done]
            episode_len[new_done] = env.t[new_done]
            episode_term[new_done] = info["term_reason"][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break
    if (~finished).any():
        not_done = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        progress = ((p_now - p_start) * line_dir).sum(-1)
        episode_progress[not_done] = progress[not_done]
        episode_len[not_done] = env.t[not_done]
        episode_term[not_done] = TERM_TRUNCATED
    return {"episode_len": episode_len, "term_reason": episode_term,
            "episode_progress": episode_progress}
