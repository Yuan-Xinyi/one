"""Execute the control stage after a seed macro-action has been selected."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv,
    TERM_ALIVE,
    TERM_TRUNCATED,
    build_task_aligned_basis,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.unified_rl.candidate_batch import SeedCandidateBatch, SeedSelection


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


@dataclass
class TopKPrefixLookaheadResult:
    """Selected rollout plus auditable probe decisions and controller cost."""

    rollout: ControllerRolloutResult
    shortlist_index: torch.Tensor
    shortlist_valid: torch.Tensor
    prefix_undiscounted_return: torch.Tensor
    prefix_discounted_return: torch.Tensor
    prefix_progress_m: torch.Tensor
    prefix_steps: torch.Tensor
    prefix_term_reason: torch.Tensor
    prefix_alive: torch.Tensor
    prefix_score: torch.Tensor
    selected_shortlist_position: torch.Tensor
    probe_active_steps: torch.Tensor
    continuation_steps: torch.Tensor

    @property
    def total_controller_steps(self) -> torch.Tensor:
        return self.probe_active_steps + self.continuation_steps


def snapshot_env_state(env: NSRLBatchedEnv) -> NSRLEnvState:
    """Clone every mutable tensor that affects future NSRL transitions."""
    return NSRLEnvState(**{
        name: getattr(env, name).detach().clone()
        for name in NSRLEnvState.__dataclass_fields__
    })


def restore_env_state(env: NSRLBatchedEnv, state: NSRLEnvState) -> None:
    """Restore a snapshot without sampling a line or resetting episode time."""
    if not isinstance(state, NSRLEnvState):
        raise TypeError('state must be an NSRLEnvState')
    if state.n_envs != env.n_envs:
        raise ValueError(
            f'snapshot contains {state.n_envs} envs, expected {env.n_envs}')
    for name in NSRLEnvState.__dataclass_fields__:
        target = getattr(env, name)
        source = getattr(state, name)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(
                f'snapshot field {name} has shape/dtype '
                f'{tuple(source.shape)}/{source.dtype}, expected '
                f'{tuple(target.shape)}/{target.dtype}')
        target.copy_(source.to(device=target.device))


def topk_union_first_valid(
    actor_logits: torch.Tensor,
    valid: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable actor top-k followed by first-valid when it is not already in-k.

    The returned fixed-width shortlist has ``top_k + 1`` columns. The final
    column is masked when first-valid already occurs in actor top-k; tasks with
    fewer than ``top_k`` valid candidates also carry masked padding columns.
    Stable descending sort makes prefix-score ties resolve in static actor
    order, with candidate index resolving equal actor logits.
    """
    if actor_logits.ndim != 2 or valid.shape != actor_logits.shape:
        raise ValueError('actor_logits and valid must match in shape (B,K)')
    if valid.dtype != torch.bool:
        raise TypeError('valid must have dtype bool')
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if top_k > actor_logits.shape[1]:
        raise ValueError('top_k cannot exceed the candidate count')
    if not bool(valid.any(dim=-1).all().item()):
        raise ValueError('every task must have at least one valid candidate')
    if not bool(torch.isfinite(actor_logits[valid]).all().item()):
        raise ValueError('valid actor logits must be finite')
    k = int(top_k)
    masked = actor_logits.masked_fill(~valid, -torch.inf)
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    actor_index = order[:, :k]
    actor_valid = valid.gather(1, actor_index)
    first = valid.to(torch.int64).argmax(dim=-1, keepdim=True)
    first_is_present = ((actor_index == first) & actor_valid).any(
        dim=-1, keepdim=True)
    shortlist = torch.cat([actor_index, first], dim=-1)
    shortlist_valid = torch.cat([actor_valid, ~first_is_present], dim=-1)
    # Never let a masked lane gather an invalid q0: historical caches may
    # preserve NaN in failed-IK slots even though the action mask is correct.
    shortlist = torch.where(
        shortlist_valid, shortlist, first.expand_as(shortlist))
    return shortlist, shortlist_valid


class FrozenRLController:
    """Deterministic mean action of an existing PPO ``Agent``."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self, env: NSRLBatchedEnv) -> None:
        del env

    @torch.no_grad()
    def action(self, env: NSRLBatchedEnv) -> torch.Tensor:
        return self.agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)


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


@torch.no_grad()
def rollout_topk_prefix_lookahead(
    probe_env: NSRLBatchedEnv,
    continuation_env: NSRLBatchedEnv,
    candidates: SeedCandidateBatch,
    actor_logits: torch.Tensor,
    controller: FrozenRLController,
    *,
    top_k: int,
    horizon_steps: int,
    alive_bonus: float = 100.0,
    gamma: float = 0.99,
    restart_selected: bool = False,
    score_objective: str = 'undiscounted_return',
) -> TopKPrefixLookaheadResult:
    """Probe actor top-k union first-valid and continue the best prefix branch.

    This is deterministic model-based lookahead. Every shortlist branch runs
    under the same frozen pure RL controller for at most ``horizon_steps``.
    Selection uses only the configured prefix objective plus an explicit
    survival bonus. By default the selected branch's complete environment state is
    gathered into ``continuation_env`` and resumed. With ``restart_selected``,
    probes are virtual planning rollouts and the selected seed is instead
    executed from q0 in ``continuation_env``. The latter is the deployable
    seed-selection protocol and keeps policy/baseline/oracle rollouts in the
    same batch environment. Rejected branch rewards never enter the selected
    trajectory return, but their active controller steps remain in the cost.
    """
    if not isinstance(controller, FrozenRLController):
        raise TypeError(
            'top-k prefix lookahead supports only FrozenRLController')
    if horizon_steps < 1:
        raise ValueError('horizon_steps must be positive')
    if horizon_steps > probe_env.max_steps:
        raise ValueError('horizon_steps cannot exceed env.max_steps')
    if not math.isfinite(alive_bonus) or alive_bonus < 0.0:
        raise ValueError('alive_bonus must be finite and non-negative')
    if not 0.0 <= gamma <= 1.0:
        raise ValueError('gamma must be in [0, 1]')
    if score_objective not in (
            'undiscounted_return', 'discounted_return', 'progress_m'):
        raise ValueError(
            'unknown prefix score_objective')
    if score_objective == 'progress_m' and probe_env.v * probe_env.dt <= 0.0:
        raise ValueError('progress probe scoring requires positive v * dt')
    if candidates.n_tasks != continuation_env.n_envs:
        raise ValueError(
            'candidate task count must match continuation_env.n_envs')
    shortlist_index, shortlist_valid = topk_union_first_valid(
        actor_logits, candidates.valid, top_k)
    n_tasks, width = shortlist_index.shape
    if probe_env.n_envs != n_tasks * width:
        raise ValueError(
            f'probe_env has {probe_env.n_envs} envs; expected '
            f'{n_tasks} * {width}')
    for name in (
            'dt', 'v', 'a_max', 'max_steps', 'cos_cone'):
        if getattr(probe_env, name) != getattr(continuation_env, name):
            raise ValueError(
                f'probe and continuation env differ in {name}')
    if probe_env.obs_dim != continuation_env.obs_dim:
        raise ValueError('probe and continuation observations differ')

    candidates = candidates.to(probe_env.device, dtype=probe_env.kin.dtype)
    actor_logits = actor_logits.to(probe_env.device)
    shortlist_index = shortlist_index.to(probe_env.device)
    shortlist_valid = shortlist_valid.to(probe_env.device)
    task_row = torch.arange(n_tasks, device=probe_env.device)[:, None]
    branch_q0 = candidates.q0[task_row, shortlist_index].reshape(-1, 7)
    selection = SeedSelection(
        q0=branch_q0,
        p0=candidates.p0.repeat_interleave(width, dim=0),
        line_dir=candidates.line_dir.repeat_interleave(width, dim=0),
        n_target=candidates.n_target.repeat_interleave(width, dim=0),
    )
    probe_env.line_dist = ScriptedLineDistribution(selection.specs())
    probe_env.reset()
    controller.reset(probe_env)
    branch_valid = shortlist_valid.reshape(-1)
    # Padded lanes contain a safe duplicated seed but are absorbing from time 0
    # and therefore consume no reported controller interaction.
    probe_env.done_persistent |= ~branch_valid

    n_branches = probe_env.n_envs
    discounted = torch.zeros(
        n_branches, device=probe_env.device, dtype=probe_env.kin.dtype)
    undiscounted = torch.zeros_like(discounted)
    discount_multiplier = torch.ones_like(discounted)
    episode_len = torch.full(
        (n_branches,), -1, device=probe_env.device, dtype=torch.long)
    term_reason = torch.full_like(episode_len, TERM_ALIVE)
    probe_steps = torch.zeros(
        n_tasks, device=probe_env.device, dtype=torch.long)

    for _ in range(horizon_steps):
        active = ~probe_env.done_persistent & branch_valid
        if not bool(active.any().item()):
            break
        probe_steps += active.view(n_tasks, width).sum(dim=-1)
        action = controller.action(probe_env)
        _, reward, _, _, info = probe_env.step(action, auto_reset=False)
        discounted += discount_multiplier * reward
        undiscounted += reward
        discount_multiplier = torch.where(
            active, discount_multiplier * gamma, discount_multiplier)
        new_done = info['episode_done'] & branch_valid
        if bool(new_done.any().item()):
            episode_len[new_done] = probe_env.t[new_done]
            term_reason[new_done] = info['term_reason'][new_done]

    prefix_alive = (~probe_env.done_persistent & branch_valid).view(
        n_tasks, width)
    prefix_undiscounted = undiscounted.view(n_tasks, width)
    prefix_discounted = discounted.view(n_tasks, width)
    prefix_position, _, _, _ = probe_env.kin.tcp_fk_jac(probe_env.q)
    prefix_progress = (
        (prefix_position - selection.p0) * selection.line_dir
    ).sum(dim=-1).view(n_tasks, width)
    if score_objective == 'progress_m':
        # Match the controller reward's order-one per-step scale while scoring
        # the selector's exact endpoint-progress objective.
        score_base = prefix_progress / (probe_env.v * probe_env.dt)
    elif score_objective == 'discounted_return':
        score_base = prefix_discounted
    else:
        score_base = prefix_undiscounted
    prefix_score = (
        score_base + float(alive_bonus) * prefix_alive.to(score_base.dtype))
    prefix_score = prefix_score.masked_fill(~shortlist_valid, -torch.inf)
    # argmax returns the first maximum, preserving static shortlist order.
    selected_position = prefix_score.argmax(dim=-1)
    selected_flat = (
        torch.arange(n_tasks, device=probe_env.device) * width
        + selected_position)
    selected_candidate = shortlist_index.gather(
        1, selected_position.unsqueeze(-1)).squeeze(-1)

    if restart_selected:
        # The probes are model rollouts only. Execute the chosen initial joint
        # state from scratch in the primary environment, as real deployment
        # cannot physically traverse several alternative branches first.
        rollout = rollout_selected_seeds(
            continuation_env, candidates, selected_candidate, controller,
            gamma=gamma)
        continuation_steps = rollout.episode_len.clone()
    else:
        selected_state = snapshot_env_state(probe_env).index_select(
            selected_flat)
        restore_env_state(continuation_env, selected_state)
        selected_discounted = discounted.index_select(0, selected_flat)
        selected_undiscounted = undiscounted.index_select(0, selected_flat)
        selected_multiplier = discount_multiplier.index_select(
            0, selected_flat)
        selected_episode_len = episode_len.index_select(0, selected_flat)
        selected_term_reason = term_reason.index_select(0, selected_flat)
        continuation_steps = torch.zeros(
            n_tasks, device=continuation_env.device, dtype=torch.long)

        # FrozenRLController is stateless; deliberately do not reset it here.
        for _ in range(continuation_env.max_steps + 1):
            active = ~continuation_env.done_persistent
            if not bool(active.any().item()):
                break
            continuation_steps += active.long()
            action = controller.action(continuation_env)
            _, reward, _, _, info = continuation_env.step(
                action, auto_reset=False)
            selected_discounted += selected_multiplier * reward
            selected_undiscounted += reward
            selected_multiplier = torch.where(
                active, selected_multiplier * gamma, selected_multiplier)
            new_done = info['episode_done']
            if bool(new_done.any().item()):
                selected_episode_len[new_done] = continuation_env.t[new_done]
                selected_term_reason[new_done] = info['term_reason'][new_done]

        unfinished = selected_episode_len < 0
        if bool(unfinished.any().item()):
            selected_episode_len[unfinished] = continuation_env.t[unfinished]
            selected_term_reason[unfinished] = TERM_TRUNCATED
        p_now, _, _, _ = continuation_env.kin.tcp_fk_jac(continuation_env.q)
        progress = ((p_now - candidates.p0) * candidates.line_dir).sum(dim=-1)
        rollout = ControllerRolloutResult(
            discounted_return=selected_discounted,
            undiscounted_return=selected_undiscounted,
            episode_len=selected_episode_len,
            term_reason=selected_term_reason,
            progress_m=progress,
            switch_count=torch.zeros(
                n_tasks, device=continuation_env.device, dtype=torch.long),
        )
    prefix_steps = snapshot_env_state(probe_env).t.view(n_tasks, width)
    prefix_term_reason = term_reason.view(n_tasks, width)
    nan = torch.full_like(prefix_undiscounted, torch.nan)
    return TopKPrefixLookaheadResult(
        rollout=rollout,
        shortlist_index=shortlist_index,
        shortlist_valid=shortlist_valid,
        prefix_undiscounted_return=torch.where(
            shortlist_valid, prefix_undiscounted, nan),
        prefix_discounted_return=torch.where(
            shortlist_valid, prefix_discounted, nan),
        prefix_progress_m=torch.where(
            shortlist_valid, prefix_progress, nan),
        prefix_steps=torch.where(
            shortlist_valid, prefix_steps,
            torch.full_like(prefix_steps, -1)),
        prefix_term_reason=torch.where(
            shortlist_valid, prefix_term_reason,
            torch.full_like(prefix_term_reason, -1)),
        prefix_alive=prefix_alive,
        prefix_score=torch.where(shortlist_valid, prefix_score, nan),
        selected_shortlist_position=selected_position,
        probe_active_steps=probe_steps,
        continuation_steps=continuation_steps,
    )
