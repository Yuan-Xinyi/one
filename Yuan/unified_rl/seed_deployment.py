"""Canonical deployment rule for a learned seed macro-policy."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class SeedDeploymentConfig:
    """How a checkpoint turns actor/Q scores into one deployed candidate."""

    mode: str = 'actor'
    proposal_head: str = 'actor'
    threshold: float = 0.0
    comparison: str = 'ge'
    proposal_q_weight: float = 0.0
    proposal_q_scale_m: float = 0.01

    def __post_init__(self) -> None:
        if self.mode not in ('actor', 'conservative'):
            raise ValueError("seed deployment mode must be 'actor' or 'conservative'")
        if self.proposal_head not in ('actor', 'feasibility', 'actor-q'):
            raise ValueError(
                "seed deployment proposal_head must be 'actor', "
                "'feasibility', or 'actor-q'")
        if (not math.isfinite(self.threshold) or self.threshold < 0.0):
            raise ValueError('seed deployment threshold must be finite and non-negative')
        if self.comparison != 'ge':
            raise ValueError("seed deployment comparison must be 'ge'")
        if (isinstance(self.proposal_q_weight, bool)
                or not isinstance(self.proposal_q_weight, (int, float))
                or not math.isfinite(self.proposal_q_weight)
                or self.proposal_q_weight < 0.0):
            raise ValueError(
                'seed deployment proposal_q_weight must be finite and '
                'non-negative')
        if (isinstance(self.proposal_q_scale_m, bool)
                or not isinstance(self.proposal_q_scale_m, (int, float))
                or not math.isfinite(self.proposal_q_scale_m)
                or self.proposal_q_scale_m <= 0.0):
            raise ValueError(
                'seed deployment proposal_q_scale_m must be finite and '
                'positive')
        if self.mode == 'actor' and (
                self.proposal_head != 'actor' or self.threshold != 0.0):
            raise ValueError(
                'actor deployment must use the actor head and zero threshold')
        if (self.proposal_head != 'actor-q'
                and (self.proposal_q_weight != 0.0
                     or self.proposal_q_scale_m != 0.01)):
            raise ValueError(
                'proposal_q_weight and proposal_q_scale_m are only '
                'configurable for actor-q proposals')

    def to_dict(self) -> dict[str, str | float]:
        result = {
            'mode': self.mode,
            'proposal_head': self.proposal_head,
            'threshold': self.threshold,
            'comparison': self.comparison,
        }
        # Preserve the exact four-field serialization of every legacy
        # deployment.  Actor-Q checkpoints opt into their extra semantics
        # explicitly, so an old checkpoint can never change behavior merely
        # because this parser learned a new proposal rule.
        if self.proposal_head == 'actor-q':
            result.update({
                'proposal_q_weight': self.proposal_q_weight,
                'proposal_q_scale_m': self.proposal_q_scale_m,
            })
        return result


@dataclass(frozen=True)
class SeedDeploymentDecision:
    """Indices and gate diagnostics for a batch of candidate sets."""

    selected_index: torch.Tensor
    proposal_index: torch.Tensor
    first_valid_index: torch.Tensor
    predicted_gain: torch.Tensor
    accepted: torch.Tensor


def deployment_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> SeedDeploymentConfig:
    """Parse deployment metadata, retaining exact legacy actor semantics."""
    value = checkpoint.get('seed_deployment')
    if value is None:
        return SeedDeploymentConfig()
    if not isinstance(value, Mapping):
        raise ValueError('checkpoint seed_deployment must be a mapping')
    base_keys = {'mode', 'proposal_head', 'threshold', 'comparison'}
    missing = base_keys - set(value)
    if missing:
        raise ValueError(
            f'checkpoint seed_deployment is missing fields: {sorted(missing)}')
    actor_q_keys = {'proposal_q_weight', 'proposal_q_scale_m'}
    semantic_keys = base_keys | (
        actor_q_keys if value['proposal_head'] == 'actor-q' else set())
    missing = semantic_keys - set(value)
    if missing:
        raise ValueError(
            'checkpoint actor-q seed_deployment is missing fields: '
            f'{sorted(missing)}')
    # Calibration diagnostics may be stored beside the canonical semantics,
    # but no unrecognized key is allowed to silently change deployment.
    allowed = semantic_keys | {'calibration'}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f'checkpoint seed_deployment has unknown fields: {sorted(unknown)}')
    calibration = value.get('calibration')
    if calibration is not None and not isinstance(calibration, Mapping):
        raise ValueError('seed_deployment calibration must be a mapping')
    threshold = value['threshold']
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError('seed deployment threshold must be numeric')
    q_weight = value.get('proposal_q_weight', 0.0)
    q_scale_m = value.get('proposal_q_scale_m', 0.01)
    return SeedDeploymentConfig(
        mode=value['mode'], proposal_head=value['proposal_head'],
        threshold=float(threshold), comparison=value['comparison'],
        proposal_q_weight=q_weight, proposal_q_scale_m=q_scale_m)


def select_seed_deployment(
    actor_logits: torch.Tensor,
    feasibility: torch.Tensor,
    valid: torch.Tensor,
    config: SeedDeploymentConfig,
) -> SeedDeploymentDecision:
    """Apply actor deployment or a Q-calibrated first-valid abstention gate."""
    if not isinstance(config, SeedDeploymentConfig):
        raise TypeError('config must be a SeedDeploymentConfig')
    if (actor_logits.ndim != 2 or feasibility.shape != actor_logits.shape
            or valid.shape != actor_logits.shape):
        raise ValueError('actor_logits, feasibility, and valid must match in (B,K)')
    if valid.dtype != torch.bool:
        raise TypeError('valid must have dtype bool')
    if not bool(valid.any(dim=-1).all().item()):
        raise ValueError('every task must contain at least one valid candidate')
    if not bool(torch.isfinite(actor_logits[valid]).all().item()):
        raise ValueError('valid actor scores must be finite')

    masked_actor = actor_logits.masked_fill(~valid, -torch.inf)
    finite_feasibility = torch.isfinite(feasibility) & valid
    masked_feasibility = feasibility.masked_fill(
        ~finite_feasibility, -torch.inf)
    actor_index = masked_actor.argmax(dim=-1)
    first_index = valid.to(torch.int64).argmax(dim=-1)
    if config.mode == 'actor':
        zero = feasibility.new_zeros(first_index.shape)
        accepted = torch.ones_like(first_index, dtype=torch.bool)
        return SeedDeploymentDecision(
            selected_index=actor_index,
            proposal_index=actor_index,
            first_valid_index=first_index,
            predicted_gain=zero,
            accepted=accepted,
        )

    if config.proposal_head == 'actor':
        proposal_index = actor_index
    elif config.proposal_head == 'feasibility':
        proposal_index = masked_feasibility.argmax(dim=-1)
        has_finite_proposal = finite_feasibility.any(dim=-1)
        proposal_index = torch.where(
            has_finite_proposal, proposal_index, first_index)
    elif config.proposal_q_weight == 0.0:
        # Weight zero is the exact actor block-zero candidate used during
        # model selection.  Avoid even a ``0 * NaN`` so its proposal indices
        # are bit-for-bit identical to the legacy actor rule.
        proposal_index = actor_index
    else:
        # Both tensors already contain the ensemble aggregates: mean member
        # log-probability and mean member feasibility.  Their weighted sum is
        # therefore still one ordinary selector forward, with no controller
        # probe or model rollout.  Keep the arithmetic in the input dtype to
        # match the policy's normal proposal path.
        combined_score = (
            actor_logits
            + config.proposal_q_weight
            * feasibility / config.proposal_q_scale_m)
        finite_combined = torch.isfinite(combined_score) & valid
        masked_combined = combined_score.masked_fill(
            ~finite_combined, -torch.inf)
        proposal_index = masked_combined.argmax(dim=-1)
        proposal_index = torch.where(
            finite_combined.any(dim=-1), proposal_index, first_index)
    row = torch.arange(valid.shape[0], device=valid.device)
    predicted_gain = (
        feasibility[row, proposal_index] - feasibility[row, first_index])
    # Preserve the calibrated float64 threshold during comparison.  Otherwise
    # a reject-all ``nextafter(max_margin, +inf)`` sentinel can round back down
    # to the float32 maximum margin and accidentally accept one proposal.
    accepted = torch.isfinite(predicted_gain) & (
        predicted_gain.to(torch.float64) >= config.threshold)
    selected_index = torch.where(accepted, proposal_index, first_index)
    return SeedDeploymentDecision(
        selected_index=selected_index,
        proposal_index=proposal_index,
        first_valid_index=first_index,
        predicted_gain=predicted_gain,
        accepted=accepted,
    )


__all__ = [
    'SeedDeploymentConfig',
    'SeedDeploymentDecision',
    'deployment_config_from_checkpoint',
    'select_seed_deployment',
]
