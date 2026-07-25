"""Full-action policy improvement for the discrete seed macro-policy."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from Yuan.unified_rl.candidate_batch import SeedCandidateBatch
from Yuan.unified_rl.seed_policy import CandidateSeedActorCritic


@dataclass
class DenseSeedConfig:
    """Backward Monte-Carlo generalized policy-iteration settings."""

    learning_rate: float = 3e-4
    update_epochs: int = 4
    n_minibatches: int = 4
    target_improvement_kl: float = 0.05
    max_update_kl: float = 0.1
    reference_uniform_mix: float = 0.05
    value_coef: float = 0.25
    feasibility_coef: float = 1.0
    feasibility_rank_coef: float = 0.25
    max_grad_norm: float = 0.5
    return_scale: float = 100.0
    rank_delta: float = 1e-4
    min_eta: float = 1e-5
    max_eta: float = 100.0
    eta_search_steps: int = 40


@dataclass
class DenseSeedRolloutBatch:
    features: torch.Tensor
    valid: torch.Tensor
    old_logits: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    raw_returns: torch.Tensor

    @property
    def n_tasks(self) -> int:
        return self.features.shape[0]


DenseRolloutFn = Callable[[SeedCandidateBatch, torch.Tensor], torch.Tensor]


@torch.no_grad()
def collect_dense_seed_rollout(
    policy: CandidateSeedActorCritic,
    candidates: SeedCandidateBatch,
    features: torch.Tensor,
    rollout_fn: DenseRolloutFn,
    *,
    return_scale: float,
) -> DenseSeedRolloutBatch:
    """Evaluate every valid seed action under one frozen controller.

    The rollout environment is fixed-size, so invalid slots replay the first
    valid action and are discarded by the mask. This keeps task/candidate
    layout dense and makes the backward target deterministic and listwise.
    """
    if not math.isfinite(return_scale) or return_scale <= 0:
        raise ValueError('return_scale must be positive')
    if features.shape[:2] != candidates.valid.shape:
        raise ValueError('features and candidate mask shapes do not match')

    b, k = candidates.valid.shape
    dist, value, _ = policy.distribution_and_values(features, candidates.valid)
    slots = torch.arange(k, device=candidates.device).repeat(b)
    flat_valid = candidates.valid.reshape(-1)
    first_valid = candidates.valid.float().argmax(dim=1)
    fallback = first_valid.repeat_interleave(k)
    safe_actions = torch.where(flat_valid, slots, fallback)
    repeated_candidates = candidates.repeat_interleave(k)
    raw_returns = torch.as_tensor(
        rollout_fn(repeated_candidates, safe_actions),
        device=features.device, dtype=torch.float32).reshape(-1)
    if raw_returns.shape != (b * k,):
        raise ValueError(f'rollout_fn must return shape ({b * k},)')
    raw_returns = raw_returns.view(b, k)
    if not bool(torch.isfinite(raw_returns[candidates.valid]).all().item()):
        raise ValueError('valid candidate rollouts must return finite values')
    # Invalid proposal slots replay a safe action only to keep the rollout
    # environment dense. Their labels are semantically padding: erase them so
    # a custom rollout_fn returning NaN for padding cannot leak through 0*NaN.
    raw_returns = torch.where(
        candidates.valid, raw_returns, torch.zeros_like(raw_returns))

    return DenseSeedRolloutBatch(
        features=features,
        valid=candidates.valid,
        old_logits=dist.logits,
        old_values=value,
        returns=raw_returns / return_scale,
        raw_returns=raw_returns,
    )


def _target_distribution(
    old_logits: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    cfg: DenseSeedConfig,
) -> tuple[torch.Tensor, float, float]:
    """KL-constrained exponential policy improvement target.

    q*(a|c) is proportional to pi_ref(a|c) exp(A(c,a) / eta), where pi_ref is
    pi_old mixed with a small valid-uniform floor. The floor lets full-action
    evidence revive an action whose float32 probability previously underflowed
    to zero. A scalar eta is found by bisection so mean KL(q* || pi_ref) stays
    within the configured trust region.
    """
    if (not math.isfinite(cfg.target_improvement_kl)
            or cfg.target_improvement_kl <= 0):
        raise ValueError('target_improvement_kl must be positive')
    if (not math.isfinite(cfg.reference_uniform_mix)
            or not 0.0 < cfg.reference_uniform_mix <= 1.0):
        raise ValueError('reference_uniform_mix must be in (0, 1]')
    if old_logits.shape != returns.shape or valid.shape != returns.shape:
        raise ValueError('old_logits, returns, and valid must have matching shapes')
    if valid.dtype != torch.bool or valid.ndim != 2:
        raise ValueError('valid must be a two-dimensional bool tensor')
    if not bool(valid.any(dim=-1).all().item()):
        raise ValueError('every dense seed task must have a valid action')
    if not bool(torch.isfinite(returns[valid]).all().item()):
        raise ValueError('valid dense seed returns must be finite')
    returns = torch.where(valid, returns, torch.zeros_like(returns))
    if not bool(torch.isfinite(old_logits[valid]).all().item()):
        raise ValueError('valid old seed logits must be finite')
    # Public callers may provide raw logits; the rollout path happens to store
    # Categorical.logits (already normalized). Normalize here so KL can never
    # become negative merely because exp(old_logits) did not sum to one.
    old_log_probs = torch.log_softmax(
        old_logits.masked_fill(~valid, -torch.inf), dim=-1)
    old_probs = old_log_probs.exp()
    uniform = valid.to(returns.dtype)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    reference_probs = ((1.0 - cfg.reference_uniform_mix) * old_probs
                       + cfg.reference_uniform_mix * uniform)
    reference_log_probs = torch.where(
        valid, reference_probs.log(), torch.full_like(reference_probs, -torch.inf))
    baseline = (reference_probs * returns).sum(dim=-1, keepdim=True)
    advantage = returns - baseline

    def distribution(eta: float) -> tuple[torch.Tensor, torch.Tensor]:
        logits = reference_log_probs + advantage / eta
        logits = logits.masked_fill(~valid, -torch.inf)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        kl_terms = torch.where(
            valid, probs * (log_probs - reference_log_probs),
            torch.zeros_like(probs))
        return probs, kl_terms.sum(dim=-1)

    low = float(cfg.min_eta)
    high = float(cfg.max_eta)
    if not math.isfinite(low) or not math.isfinite(high) or not 0 < low < high:
        raise ValueError('dense seed eta bounds must satisfy 0 < min < max')
    if cfg.eta_search_steps < 1:
        raise ValueError('eta_search_steps must be positive')
    active = valid.sum(dim=-1) > 1

    def mean_active_kl(kl: torch.Tensor) -> float:
        if not bool(active.any().item()):
            return 0.0
        return float(kl[active].mean().item())

    with torch.no_grad():
        target, kl = distribution(low)
        if mean_active_kl(kl) <= cfg.target_improvement_kl:
            return target, low, mean_active_kl(kl)
        target, kl = distribution(high)
        if mean_active_kl(kl) > cfg.target_improvement_kl:
            raise ValueError(
                'max_eta is too small to satisfy target_improvement_kl: '
                f'KL={mean_active_kl(kl):.6g} at eta={high:g}')
        for _ in range(cfg.eta_search_steps):
            mid = 0.5 * (low + high)
            candidate, candidate_kl = distribution(mid)
            if mean_active_kl(candidate_kl) > cfg.target_improvement_kl:
                low = mid
            else:
                high = mid
                target = candidate
                kl = candidate_kl
        return target, high, mean_active_kl(kl)


def _dense_rank_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    valid: torch.Tensor,
    min_delta: float,
) -> tuple[torch.Tensor, int]:
    safe_scores = torch.where(valid, scores, torch.zeros_like(scores))
    return_delta = returns.unsqueeze(2) - returns.unsqueeze(1)
    score_delta = safe_scores.unsqueeze(2) - safe_scores.unsqueeze(1)
    better = (valid.unsqueeze(2) & valid.unsqueeze(1)
              & (return_delta > min_delta))
    pairs_per_task = better.sum(dim=(1, 2))
    n_pairs = int(pairs_per_task.sum().item())
    if n_pairs == 0:
        return scores.sum() * 0.0, 0
    # Larger downstream gains carry more weight than near-ties, matching the
    # oracle-headroom metric rather than treating every sign equally. Reduce
    # within each task first: otherwise tasks with K valid actions receive
    # O(K^2) more ranking weight than difficult tasks with few valid actions.
    raw_weights = return_delta.clamp_min(0.0) * better
    weight_mean = raw_weights.sum(dim=(1, 2)) / pairs_per_task.clamp_min(1)
    weights = raw_weights / weight_mean.clamp_min(1e-8)[:, None, None]
    pair_loss = F.softplus(-score_delta) * weights.detach() * better
    task_loss = pair_loss.sum(dim=(1, 2)) / pairs_per_task.clamp_min(1)
    return task_loss[pairs_per_task > 0].mean(), n_pairs


def update_dense_seed_policy(
    policy: CandidateSeedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: DenseSeedRolloutBatch,
    cfg: DenseSeedConfig,
) -> dict[str, float]:
    """Fit actor/value/Q heads to one full-action Monte-Carlo batch."""
    if rollout.n_tasks < 1:
        raise ValueError('dense seed rollout must contain at least one task')
    if cfg.update_epochs < 1 or cfg.n_minibatches < 1:
        raise ValueError('update_epochs and n_minibatches must be positive')
    if (not math.isfinite(cfg.max_update_kl) or cfg.max_update_kl <= 0.0
            or not math.isfinite(cfg.max_grad_norm)
            or cfg.max_grad_norm <= 0.0):
        raise ValueError('max_update_kl and max_grad_norm must be positive')
    coefficients = (
        cfg.value_coef,
        cfg.feasibility_coef,
        cfg.feasibility_rank_coef,
        cfg.rank_delta,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in coefficients):
        raise ValueError('dense seed loss coefficients must be finite and non-negative')
    target_probs, eta, improvement_kl = _target_distribution(
        rollout.old_logits, rollout.returns, rollout.valid, cfg)
    safe_returns = torch.where(
        rollout.valid, rollout.returns, torch.zeros_like(rollout.returns))
    value_target = (target_probs * safe_returns).sum(dim=-1)
    n_tasks = rollout.n_tasks
    n_minibatches = min(cfg.n_minibatches, n_tasks)
    last: dict[str, float] = {}
    update_kl = 0.0
    max_update_kl = 0.0

    for _ in range(cfg.update_epochs):
        task_order = torch.randperm(n_tasks, device=rollout.features.device)
        for task_index in torch.tensor_split(task_order, n_minibatches):
            if task_index.numel() == 0:
                continue
            dist, value, feasibility = policy.distribution_and_values(
                rollout.features[task_index], rollout.valid[task_index])
            target = target_probs[task_index]
            actor_loss = -(target * dist.logits).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(value, value_target[task_index])
            valid = rollout.valid[task_index]
            q_error = F.smooth_l1_loss(
                feasibility, safe_returns[task_index], reduction='none')
            # Actor/value losses are task-balanced. Keep dense Q supervision
            # task-balanced too instead of weighting a task by its valid K.
            valid_float = valid.to(q_error.dtype)
            q_error = torch.where(valid, q_error, torch.zeros_like(q_error))
            q_loss = (
                (q_error * valid_float).sum(dim=-1)
                / valid_float.sum(dim=-1).clamp_min(1.0)
            ).mean()
            rank_loss, n_pairs = _dense_rank_loss(
                feasibility, safe_returns[task_index], valid,
                cfg.rank_delta)
            loss = (actor_loss
                    + cfg.value_coef * value_loss
                    + cfg.feasibility_coef * q_loss
                    + cfg.feasibility_rank_coef * rank_loss)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.max_grad_norm)
            if bool(torch.isfinite(grad_norm).item()):
                optimizer.step()

            last = {
                'loss/policy': float(actor_loss.item()),
                'loss/value': float(value_loss.item()),
                'loss/feasibility': float(q_loss.item()),
                'loss/feasibility_rank': float(rank_loss.item()),
                'seed/preference_pairs': float(n_pairs),
                'seed/entropy': float(dist.entropy().mean().item()),
            }

        with torch.no_grad():
            new_dist, _, _ = policy.distribution_and_values(
                rollout.features, rollout.valid)
            old_dist = torch.distributions.Categorical(
                logits=rollout.old_logits)
            per_task_kl = torch.distributions.kl_divergence(
                old_dist, new_dist)
            active = rollout.valid.sum(dim=-1) > 1
            update_kl = float(
                per_task_kl[active].mean().item()
                if bool(active.any().item()) else 0.0)
            max_update_kl = float(per_task_kl.max().item())
        if update_kl > cfg.max_update_kl:
            break

    valid_returns = rollout.raw_returns[rollout.valid]
    with torch.no_grad():
        final_dist, _, final_q = policy.distribution_and_values(
            rollout.features, rollout.valid)
        row = torch.arange(n_tasks, device=rollout.features.device)
        policy_action = final_dist.logits.argmax(dim=-1)
        q_action = final_q.masked_fill(~rollout.valid, -torch.inf).argmax(dim=-1)
        first_action = rollout.valid.float().argmax(dim=-1)
        oracle_return = rollout.raw_returns.masked_fill(
            ~rollout.valid, -torch.inf).max(dim=-1).values
    last.update({
        'seed/approx_kl': update_kl,
        'seed/max_kl': max_update_kl,
        'seed/improvement_target_kl': improvement_kl,
        'seed/improvement_eta': eta,
        'seed/raw_return_mean': float(valid_returns.mean().item()),
        'seed/raw_return_std': float(valid_returns.std(unbiased=False).item()),
        'seed/policy_return_mean': float(
            rollout.raw_returns[row, policy_action].mean().item()),
        'seed/feasibility_return_mean': float(
            rollout.raw_returns[row, q_action].mean().item()),
        'seed/first_return_mean': float(
            rollout.raw_returns[row, first_action].mean().item()),
        'seed/oracle_return_mean': float(oracle_return.mean().item()),
    })
    return last
