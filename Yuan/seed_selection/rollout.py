"""Module 2 wrapper: single (q0, c) rollout under the classical-nullspace
controller, returning a normalized path length L.

The env normally anchors the task line at ``FK(q0)`` on reset. For perturbed
tasks (where ``c['p0']`` has been moved away from FK(q0) by Module 4), we
override ``env.p_start`` after reset so the line is anchored at the
TASK-defined p0. Without this override, perturbing p0 in c has no effect on
the rollout dynamics — half of Module 4's perturbation budget would be wasted.
"""
from __future__ import annotations

import torch

from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.env.env import NSRLBatchedEnv, TERM_TRUNCATED
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution


# Normalize episode_progress by this (meters) so L is roughly in [0, 1]
# for typical tasks. Matches v18_smm_core.TARGET_PATH_M — the "infinite
# ray, no success terminate" convention. Override per-call if needed.
DEFAULT_TARGET_DISTANCE_M = 1.5


@torch.no_grad()
def rollout_one(
    q0: torch.Tensor,
    c: dict[str, torch.Tensor],
    *,
    env: NSRLBatchedEnv,
    controller: ClassicalNullspaceController,
    target_distance_m: float = DEFAULT_TARGET_DISTANCE_M,
) -> dict:
    """Run one classical-nullspace rollout starting at ``q0`` on task ``c``.

    Args:
        q0: (7,) initial joint configuration on env.device.
        c: dict {'p0': (3,), 'line_dir': (3,), 'n_target': (3,)}.
        env: a singleton NSRLBatchedEnv (n_envs == 1).
        controller: classical-nullspace controller bound to env.kin.
        target_distance_m: meters used to normalize L; L = progress_m / this.

    Returns:
        dict with keys:
            'L'                  float, episode_progress / target_distance_m
            'episode_progress_m' float, EE travel along u_hat in meters
            'episode_len'        int, steps before terminate / truncate
            'term_reason'        int, env's TERM_* code
    """
    assert env.n_envs == 1, f"rollout_one expects n_envs=1, got {env.n_envs}"

    spec = {
        "q0":       q0.to(env.device, dtype=env.kin.dtype).unsqueeze(0),
        "line_dir": c["line_dir"].to(env.device, dtype=env.kin.dtype).unsqueeze(0),
        "n_target": c["n_target"].to(env.device, dtype=env.kin.dtype).unsqueeze(0),
    }
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    # Override the env's FK-derived p_start with the task-defined p0.
    # This is what makes perturb_p0_mm in Module 4 actually affect the rollout.
    env.p_start[:] = c["p0"].to(env.p_start).unsqueeze(0)

    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    action_fn = cn_action_fn(controller)

    episode_progress = torch.zeros(1, dtype=env.kin.dtype, device=env.device)
    episode_len = torch.full((1,), -1, dtype=torch.long, device=env.device)
    term = torch.full((1,), -1, dtype=torch.long, device=env.device)

    truncated = True
    for _ in range(env.max_steps + 1):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        if bool(info["episode_done"][0].item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            episode_progress[0] = ((p_now[0] - p_start[0]) * line_dir[0]).sum()
            episode_len[0] = env.t[0]
            term[0] = info["term_reason"][0]
            truncated = False
            break
    if truncated:
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        episode_progress[0] = ((p_now[0] - p_start[0]) * line_dir[0]).sum()
        episode_len[0] = env.t[0]
        term[0] = TERM_TRUNCATED

    prog = float(episode_progress[0].item())
    return {
        "L": prog / float(target_distance_m),
        "episode_progress_m": prog,
        "episode_len": int(episode_len[0].item()),
        "term_reason": int(term[0].item()),
    }
