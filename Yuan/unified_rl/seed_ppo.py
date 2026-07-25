"""Contextual-bandit PPO for the seed macro-action."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn as nn

from Yuan.unified_rl.candidate_batch import SeedCandidateBatch
from Yuan.unified_rl.seed_policy import CandidateSeedActorCritic


@dataclass
class SeedPPOConfig:
    learning_rate: float = 1e-4
    update_epochs: int = 4
    n_minibatches: int = 4
    clip_coef: float = 0.1
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    feasibility_coef: float = 0.5
    preference_coef: float = 0.25
    feasibility_rank_coef: float = 0.25
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    target_kl: float | None = 0.01
    return_scale: float = 100.0
    center_within_task: bool = True


@dataclass
class SeedRolloutBatch:
    features: torch.Tensor
    valid: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    old_logits: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    raw_returns: torch.Tensor
    n_tasks: int
    samples_per_task: int


RolloutFn = Callable[[SeedCandidateBatch, torch.Tensor], torch.Tensor]


def _preference_losses(
    policy: CandidateSeedActorCritic,
    rollout: SeedRolloutBatch,
    task_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Within-task pairwise policy improvement from sampled downstream return.

    This is the explicit transition-feasibility path: comparisons never mix
    intrinsically easy and hard tasks, and they remain valid under arbitrary
    task-wise reward offsets.
    """
    m = rollout.samples_per_task
    if m < 2 or task_index.numel() == 0:
        zero = rollout.features.sum() * 0.0
        return zero, zero, 0
    first_row = task_index * m
    dist, _, feasibility = policy.distribution_and_values(
        rollout.features[first_row], rollout.valid[first_row])
    logits = dist.logits
    actions = rollout.actions.view(rollout.n_tasks, m)[task_index]
    returns = rollout.returns.view(rollout.n_tasks, m)[task_index]
    row = torch.arange(task_index.numel(), device=task_index.device)
    actor_terms = []
    feasibility_terms = []
    n_pairs = 0
    for left in range(m):
        for right in range(left + 1, m):
            different_action = actions[:, left] != actions[:, right]
            return_delta = returns[:, left] - returns[:, right]
            informative = different_action & (return_delta.abs() > 1e-6)
            if not bool(informative.any().item()):
                continue
            sign = return_delta[informative].sign()
            selected_row = row[informative]
            left_action = actions[informative, left]
            right_action = actions[informative, right]
            actor_delta = (
                logits[selected_row, left_action]
                - logits[selected_row, right_action])
            feasibility_delta = (
                feasibility[selected_row, left_action]
                - feasibility[selected_row, right_action])
            actor_terms.append(torch.nn.functional.softplus(-sign * actor_delta))
            feasibility_terms.append(
                torch.nn.functional.softplus(-sign * feasibility_delta))
            n_pairs += int(informative.sum().item())
    if not actor_terms:
        zero = logits.sum() * 0.0
        return zero, zero, 0
    return (
        torch.cat(actor_terms).mean(),
        torch.cat(feasibility_terms).mean(),
        n_pairs,
    )


@torch.no_grad()
def collect_seed_rollout(
    policy: CandidateSeedActorCritic,
    candidates: SeedCandidateBatch,
    features: torch.Tensor,
    rollout_fn: RolloutFn,
    *,
    samples_per_task: int = 1,
    return_scale: float = 100.0,
    center_within_task: bool = True,
) -> SeedRolloutBatch:
    """Sample seed actions, execute their complete controller rollouts, and
    turn the delayed controller return into a seed-stage PPO transition.

    ``rollout_fn`` receives a task-major repeated candidate batch and one
    chosen candidate index per row. It returns raw discounted episode return.
    """
    if samples_per_task < 1:
        raise ValueError('samples_per_task must be positive')
    if not math.isfinite(return_scale) or return_scale <= 0:
        raise ValueError('return_scale must be positive')
    if features.shape[:2] != candidates.valid.shape:
        raise ValueError('features and candidate mask shapes do not match')

    dist, value, _ = policy.distribution_and_values(features, candidates.valid)
    actions_mb = dist.sample((samples_per_task,))
    logprobs_mb = dist.log_prob(actions_mb)
    actions = actions_mb.transpose(0, 1).contiguous().reshape(-1)
    old_logprobs = logprobs_mb.transpose(0, 1).contiguous().reshape(-1)

    repeated_candidates = candidates.repeat_interleave(samples_per_task)
    raw_returns = rollout_fn(repeated_candidates, actions)
    raw_returns = torch.as_tensor(raw_returns, device=features.device,
                                  dtype=torch.float32).reshape(-1)
    expected_n = candidates.n_tasks * samples_per_task
    if raw_returns.shape != (expected_n,):
        raise ValueError(f'rollout_fn must return shape ({expected_n},)')
    if not bool(torch.isfinite(raw_returns).all().item()):
        bad = torch.nonzero(~torch.isfinite(raw_returns), as_tuple=False)
        raise ValueError(
            'rollout_fn returned non-finite values at flat indices '
            f'{bad[:20].flatten().cpu().tolist()}')
    returns = raw_returns / return_scale
    old_values = value.repeat_interleave(samples_per_task)

    if center_within_task and samples_per_task > 1:
        grouped = returns.view(candidates.n_tasks, samples_per_task)
        advantages = (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)
    else:
        advantages = returns - old_values

    return SeedRolloutBatch(
        features=features.repeat_interleave(samples_per_task, dim=0),
        valid=candidates.valid.repeat_interleave(samples_per_task, dim=0),
        actions=actions,
        old_logprobs=old_logprobs,
        old_logits=dist.logits.repeat_interleave(samples_per_task, dim=0),
        old_values=old_values,
        returns=returns,
        advantages=advantages,
        raw_returns=raw_returns,
        n_tasks=candidates.n_tasks,
        samples_per_task=samples_per_task,
    )


def update_seed_policy(
    policy: CandidateSeedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: SeedRolloutBatch,
    cfg: SeedPPOConfig,
) -> dict[str, float]:
    """Run PPO updates over a batch of one-step seed transitions."""
    n = rollout.actions.shape[0]
    if n < 1:
        raise ValueError('seed rollout must contain at least one transition')
    if cfg.update_epochs < 1 or cfg.n_minibatches < 1:
        raise ValueError('update_epochs and n_minibatches must be positive')
    if (not math.isfinite(cfg.max_grad_norm) or cfg.max_grad_norm <= 0.0
            or not math.isfinite(cfg.clip_coef) or cfg.clip_coef < 0.0):
        raise ValueError('max_grad_norm must be positive and clip_coef non-negative')
    coefficients = (
        cfg.ent_coef,
        cfg.vf_coef,
        cfg.feasibility_coef,
        cfg.preference_coef,
        cfg.feasibility_rank_coef,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in coefficients):
        raise ValueError('seed PPO loss coefficients must be finite and non-negative')
    if (cfg.target_kl is not None
            and (not math.isfinite(cfg.target_kl) or cfg.target_kl <= 0.0)):
        raise ValueError('target_kl must be positive or None')
    n_minibatches = min(cfg.n_minibatches, n)
    minibatch_size = max(n // n_minibatches, 1)
    advantages = rollout.advantages
    if cfg.norm_adv:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8)

    last = {}
    approx_kl_value = 0.0
    max_kl_value = 0.0
    for _ in range(cfg.update_epochs):
        order = torch.randperm(n, device=rollout.actions.device)
        starts = list(range(0, n, minibatch_size))
        task_chunks = torch.tensor_split(
            torch.randperm(
                rollout.n_tasks, device=rollout.actions.device),
            len(starts))
        for minibatch_index, start in enumerate(starts):
            idx = order[start:start + minibatch_size]
            new_dist, new_value, feasibility = policy.distribution_and_values(
                rollout.features[idx], rollout.valid[idx])
            new_logprob = new_dist.log_prob(rollout.actions[idx])
            entropy = new_dist.entropy()
            row = torch.arange(idx.shape[0], device=idx.device)
            selected_feasibility = feasibility[row, rollout.actions[idx]]
            logratio = new_logprob - rollout.old_logprobs[idx]
            ratio = logratio.clamp(-20.0, 20.0).exp()

            mb_adv = advantages[idx]
            pg_loss_1 = -mb_adv * ratio
            pg_loss_2 = -mb_adv * ratio.clamp(
                1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
            policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

            value_unclipped = (new_value - rollout.returns[idx]).square()
            value_clipped = rollout.old_values[idx] + (
                new_value - rollout.old_values[idx]).clamp(
                    -cfg.clip_coef, cfg.clip_coef)
            value_loss = 0.5 * torch.max(
                value_unclipped,
                (value_clipped - rollout.returns[idx]).square(),
            ).mean()
            feasibility_loss = 0.5 * (
                selected_feasibility - rollout.returns[idx]).square().mean()
            # Each task's all-sample preference loss is evaluated exactly once
            # per epoch. Deriving task ids from transition minibatches used to
            # count a task 2-4 times depending on how its samples were split.
            task_index = task_chunks[minibatch_index]
            preference_loss, feasibility_rank_loss, n_preference_pairs = (
                _preference_losses(policy, rollout, task_index))
            entropy_mean = entropy.mean()
            loss = (policy_loss - cfg.ent_coef * entropy_mean
                    + cfg.vf_coef * value_loss
                    + cfg.feasibility_coef * feasibility_loss
                    + cfg.preference_coef * preference_loss
                    + cfg.feasibility_rank_coef * feasibility_rank_loss)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            if bool(torch.isfinite(grad_norm).item()):
                optimizer.step()

            last = {
                'loss/policy': float(policy_loss.item()),
                'loss/value': float(value_loss.item()),
                'loss/feasibility': float(feasibility_loss.item()),
                'loss/preference': float(preference_loss.item()),
                'loss/feasibility_rank': float(feasibility_rank_loss.item()),
                'seed/preference_pairs': float(n_preference_pairs),
                'seed/entropy': float(entropy_mean.item()),
                'seed/raw_return_mean': float(rollout.raw_returns.mean().item()),
                'seed/raw_return_std': float(
                    rollout.raw_returns.std(unbiased=False).item()),
            }
        # Preference and PPO losses both move the categorical distribution.
        # Measure their combined effect exactly against the rollout policy,
        # over the full batch rather than the final minibatch only.
        with torch.no_grad():
            new_dist, _, _ = policy.distribution_and_values(
                rollout.features, rollout.valid)
            old_dist = torch.distributions.Categorical(logits=rollout.old_logits)
            kl = torch.distributions.kl_divergence(old_dist, new_dist)
            approx_kl_value = float(kl.mean().item())
            max_kl_value = float(kl.max().item())
        last['seed/approx_kl'] = approx_kl_value
        last['seed/max_kl'] = max_kl_value
        if cfg.target_kl is not None and approx_kl_value > cfg.target_kl:
            break
    return last
