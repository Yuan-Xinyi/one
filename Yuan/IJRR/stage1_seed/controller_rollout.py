"""Execute the control stage after a seed macro-action has been selected."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from Yuan.IJRR.env.classical_nullspace import ClassicalNullspaceController
from Yuan.IJRR.env.env import NSRLBatchedEnv, TERM_TRUNCATED, build_task_aligned_basis
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage1_seed.candidate_batch import SeedCandidateBatch, SeedSelection


class Controller(Protocol):
    """Stateful control-stage interface used by seeded rollouts."""

    def reset(self, env: NSRLBatchedEnv) -> None: ...

    def action(self, env: NSRLBatchedEnv) -> torch.Tensor: ...


@dataclass
class ControllerRolloutResult:
    discounted_return: torch.Tensor
    undiscounted_return: torch.Tensor
    episode_len: torch.Tensor
    term_reason: torch.Tensor
    progress_m: torch.Tensor
    switch_count: torch.Tensor


@dataclass(frozen=True)
class NSRLEnvState:
    """Complete dynamic state needed to fork the current Torch environment."""

    q: torch.Tensor
    line_dir: torch.Tensor
    n_target: torch.Tensor
    t: torch.Tensor
    a_prev: torch.Tensor
    done_persistent: torch.Tensor
    episode_reward: torch.Tensor
    episode_steps: torch.Tensor
    p_start: torch.Tensor

    @property
    def n_envs(self) -> int:
        return int(self.q.shape[0])

    def index_select(self, index: torch.Tensor) -> 'NSRLEnvState':
        index = torch.as_tensor(
            index, device=self.q.device, dtype=torch.long).reshape(-1)
        return NSRLEnvState(**{
            name: getattr(self, name).index_select(0, index).clone()
            for name in self.__dataclass_fields__
        })












class FrozenHybridController:
    """Historical RL/classical hysteresis controller behind seed labels."""

    def __init__(self, agent, classical: ClassicalNullspaceController,
                 tau_enter: float = 0.985, tau_exit: float = 0.96):
        if tau_exit > tau_enter:
            raise ValueError('tau_exit must not exceed tau_enter')
        self.agent = agent
        self.classical = classical
        self.tau_enter = float(tau_enter)
        self.tau_exit = float(tau_exit)
        self.using_rl: torch.Tensor | None = None
        self.switch_count: torch.Tensor | None = None

    @staticmethod
    def _max_abs_qn(env: NSRLBatchedEnv) -> torch.Tensor:
        return ((env.q - env.q_mid).abs() / env.q_half).max(dim=-1).values

    def reset(self, env: NSRLBatchedEnv) -> None:
        self.using_rl = self._max_abs_qn(env) < self.tau_enter
        self.switch_count = torch.zeros(
            env.n_envs, device=env.device, dtype=torch.long)

    def action(self, env: NSRLBatchedEnv) -> torch.Tensor:
        if self.using_rl is None or self.switch_count is None:
            raise RuntimeError('reset() must be called before action()')
        qn = self._max_abs_qn(env)
        new_using_rl = torch.where(
            self.using_rl,
            qn < self.tau_enter,
            qn < self.tau_exit,
        )
        active = ~env.done_persistent
        self.switch_count += ((new_using_rl != self.using_rl) & active).long()
        self.using_rl = new_using_rl

        with torch.no_grad():
            obs = env.current_obs()
            rl_action = self.agent.actor_mean(obs).clamp(-1.0, 1.0)
            basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        q_dot = self.classical.q_dot_null(env.q, env.line_dir, env.n_target)
        with torch.no_grad():
            classical_action = (
                basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
            classical_action = torch.nan_to_num(
                classical_action / env.a_max, nan=0.0).clamp(-1.0, 1.0)
            return torch.where(
                self.using_rl.unsqueeze(-1), rl_action, classical_action)


@torch.no_grad()
def rollout_seed_selection(
    env: NSRLBatchedEnv,
    selection: SeedSelection,
    controller: Controller,
    gamma: float = 0.99,
) -> ControllerRolloutResult:
    """Run complete control episodes and preserve delayed seed-stage return."""
    if selection.n_tasks != env.n_envs:
        raise ValueError(
            f'seed selection has {selection.n_tasks} tasks but env has {env.n_envs}')
    if not 0.0 <= gamma <= 1.0:
        raise ValueError('gamma must be in [0, 1]')
    env.line_dist = ScriptedLineDistribution(selection.specs())
    env.reset()
    controller.reset(env)

    n = env.n_envs
    discounted = torch.zeros(n, device=env.device, dtype=env.kin.dtype)
    undiscounted = torch.zeros_like(discounted)
    episode_len = torch.full((n,), -1, device=env.device, dtype=torch.long)
    term_reason = torch.full((n,), -1, device=env.device, dtype=torch.long)
    finished = torch.zeros(n, device=env.device, dtype=torch.bool)
    discount = 1.0

    for _ in range(env.max_steps + 1):
        action = controller.action(env)
        _, reward, _, _, info = env.step(action, auto_reset=False)
        discounted += discount * reward
        undiscounted += reward
        discount *= gamma
        new_done = info['episode_done']
        if bool(new_done.any().item()):
            episode_len[new_done] = env.t[new_done]
            term_reason[new_done] = info['term_reason'][new_done]
            finished |= new_done
        if bool(env.done_persistent.all().item()):
            break

    if bool((~finished).any().item()):
        episode_len[~finished] = env.t[~finished]
        term_reason[~finished] = TERM_TRUNCATED
    p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
    progress = ((p_now - selection.p0) * selection.line_dir).sum(-1)
    switch_count = getattr(controller, 'switch_count', None)
    if switch_count is None:
        switch_count = torch.zeros(n, device=env.device, dtype=torch.long)
    return ControllerRolloutResult(
        discounted_return=discounted,
        undiscounted_return=undiscounted,
        episode_len=episode_len,
        term_reason=term_reason,
        progress_m=progress,
        switch_count=switch_count.clone(),
    )


@torch.no_grad()
def rollout_selected_seeds(
    env: NSRLBatchedEnv,
    candidates: SeedCandidateBatch,
    candidate_index: torch.Tensor,
    controller: Controller,
    *,
    gamma: float = 0.99,
) -> ControllerRolloutResult:
    """Select one candidate per task, then run complete control episodes."""
    if candidates.n_tasks != env.n_envs:
        raise ValueError(
            f'candidate batch has {candidates.n_tasks} tasks but env has {env.n_envs}')
    if not 0.0 <= gamma <= 1.0:
        raise ValueError('gamma must be in [0, 1]')
    candidates = candidates.to(env.device, dtype=env.kin.dtype)
    selection = candidates.select(candidate_index)
    return rollout_seed_selection(env, selection, controller, gamma)

