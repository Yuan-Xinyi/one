"""GPM nullspace baseline controller (eval ratio denominator).

Implements
    q_dot = J_p^+ v u_hat + B(q) · k_JL · B(q)^T ∇H(q)
with the same termination conditions and dt as the RL env, so episode_len
returned by `rollout_first_episode` is directly comparable to the RL policy.

Uses `auto_reset=False` on env so each env runs exactly one episode and
finished envs freeze (avoids ScriptedLineDistribution exhaustion in eval).
"""
from __future__ import annotations

import torch

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, build_task_aligned_basis,
)


class GPMBaselineController:
    """Classical GPM-JL baseline, optionally augmented with directional
    manipulability gradient ascent.

    Control law:
        q_dot = J_p^+ v u_hat
              + B(q) · k_jl    · B^T ∇H_jl(q)        # pull q toward joint center
              + B(q) · k_dm    · B^T ∇w_u(q)         # ascend w_u(q, u_hat)

    With k_dm=0 (default), this is the original GPM-JL ("weak") baseline.
    With k_dm > 0, this is the "strong" baseline that uses the same
    directional-manipulability signal the RL agent now gets in its reward —
    a fair comparison for what reward shaping alone can buy.
    """
    def __init__(self, kin, k_jl: float = 1.0,
                 k_dm: float = 0.0, manip_damping: float = 1e-3):
        self.kin = kin
        self.k_jl = float(k_jl)
        self.k_dm = float(k_dm)
        self.manip_damping = float(manip_damping)
        q_half = 0.5 * (kin.lmt_up - kin.lmt_lo)
        self._grad_denom = q_half * q_half
        self._q_mid = kin.q_mid

    def _grad_w_u(self, q: torch.Tensor, u_hat: torch.Tensor) -> torch.Tensor:
        """∇_q w_u(q, u_hat) via autograd. Returns (B, 7)."""
        with torch.enable_grad():
            q_eval = q.detach().clone().requires_grad_(True)
            _, _, J, _ = self.kin.tcp_fk_jac(q_eval)
            J_p = J[:, :3, :]
            eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(
                q.shape[0], 3, 3)
            JJt_dmp = J_p @ J_p.transpose(-1, -2) + (self.manip_damping ** 2) * eye3
            u_col = u_hat.unsqueeze(-1)
            inv_quad = (u_col.transpose(-1, -2)
                        @ torch.linalg.inv(JJt_dmp) @ u_col
                        ).squeeze(-1).squeeze(-1).clamp_min(1e-12)
            w_u = inv_quad.pow(-0.5)
            grad = torch.autograd.grad(w_u.sum(), q_eval,
                                       retain_graph=False, allow_unused=False)[0]
        return grad.detach()

    def action(self, q: torch.Tensor, B_basis: torch.Tensor,
               u_hat: torch.Tensor | None = None) -> torch.Tensor:
        """Return baseline a ∈ ℝ^4 in raw rad/s; caller divides by a_max.

        `u_hat` is required iff `k_dm > 0`.
        """
        grad_H_jl = (q - self._q_mid) / self._grad_denom
        a = self.k_jl * (B_basis.transpose(-1, -2)
                         @ grad_H_jl.unsqueeze(-1)).squeeze(-1)
        if self.k_dm > 0.0:
            assert u_hat is not None, "k_dm > 0 requires u_hat for ∇w_u(q)"
            grad_w_u = self._grad_w_u(q, u_hat)
            a = a + self.k_dm * (B_basis.transpose(-1, -2)
                                 @ grad_w_u.unsqueeze(-1)).squeeze(-1)
        return a


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
    # Snapshot start positions (env.p_start was set by reset)
    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    for step_i in range(cfg_max + 1):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info["episode_done"]
        if new_done.any():
            # Compute progress at the moment of done: (current p_tcp - p_start)·u_hat
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
        episode_term[not_done] = 5  # TERM_TRUNCATED
    return {"episode_len": episode_len, "term_reason": episode_term,
            "episode_progress": episode_progress}


def baseline_action_fn(controller: GPMBaselineController):
    """Closure: state → normalized GPM baseline action ∈ [-1, 1]^4."""
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        B_basis, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping,
        )
        a_raw = controller.action(env.q, B_basis, u_hat=env.line_dir)
        return (a_raw / env.a_max).clamp(-1.0, 1.0)
    return _fn


def zero_nullspace_action_fn():
    """Closure: state → a ≡ 0 (pure task-space motion, no nullspace term).

    Diagnostic baseline for the trained-worse-than-random failure mode: if a
    trained PPO policy can't beat doing literally nothing in the nullspace,
    then the nullspace signal is at best useless and possibly harmful — which
    isolates whether the issue is reward shaping / sign convention / policy
    learning vs the underlying task itself.
    """
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        return torch.zeros((env.n_envs, env.act_dim),
                           device=env.device, dtype=env.kin.dtype)
    return _fn
