"""Low-variance bandit objective for a shielded residual seed head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Bernoulli


@dataclass(frozen=True)
class ResidualBanditConfig:
    std: float = 0.25
    return_scale: float = 500.0
    reject_penalty: float = 0.02
    gate_entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        values = {
            'std': self.std,
            'return_scale': self.return_scale,
            'max_grad_norm': self.max_grad_norm,
        }
        if any(not torch.isfinite(torch.tensor(value)).item() or value <= 0
               for value in values.values()):
            raise ValueError(f'{tuple(values)} must be finite and positive')
        if (not torch.isfinite(torch.tensor(self.reject_penalty)).item()
                or self.reject_penalty < 0):
            raise ValueError('reject_penalty must be finite and non-negative')
        if (not torch.isfinite(torch.tensor(self.gate_entropy_coef)).item()
                or self.gate_entropy_coef < 0):
            raise ValueError('gate_entropy_coef must be finite and non-negative')


def geometry_groups(fingerprints: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Return row indices grouped by identical task geometry."""
    if not fingerprints:
        raise ValueError('fingerprints must be non-empty')
    grouped: dict[str, list[int]] = {}
    for row, fingerprint in enumerate(fingerprints):
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('every fingerprint must be a non-empty string')
        grouped.setdefault(fingerprint, []).append(row)
    return tuple(torch.tensor(rows, dtype=torch.long) for rows in grouped.values())


def sample_group_balanced_indices(
    groups: tuple[torch.Tensor, ...], n: int, generator: torch.Generator,
) -> torch.Tensor:
    """Sample geometry uniformly, then a row uniformly within that geometry."""
    if not groups or n < 1:
        raise ValueError('groups must be non-empty and n must be positive')
    chosen_groups = torch.randint(len(groups), (n,), generator=generator)
    rows = torch.empty(n, dtype=torch.long)
    for group_index in chosen_groups.unique(sorted=False).tolist():
        locations = torch.nonzero(
            chosen_groups == group_index, as_tuple=False).flatten()
        members = groups[group_index]
        if members.ndim != 1 or members.numel() < 1:
            raise ValueError('each geometry group must be a non-empty vector')
        draw = torch.randint(
            members.numel(), (locations.numel(),), generator=generator)
        rows[locations] = members[draw]
    return rows


def residual_bandit_loss(
    gate_logits: torch.Tensor,
    antithetic_log_prob: torch.Tensor,
    base_return: torch.Tensor,
    branch_return: torch.Tensor,
    accepted_alpha: torch.Tensor,
    config: ResidualBanditConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exact gate expectation and antithetic score-function loss.

    ``branch_return`` and ``accepted_alpha`` are ``(B,2)`` in plus/minus
    order.  Centering the two branch scores preserves the antithetic policy
    gradient while removing every task-level baseline exactly.
    """
    batch = gate_logits.shape[0]
    expected = {
        'gate_logits': (batch,),
        'antithetic_log_prob': (batch, 2),
        'base_return': (batch,),
        'branch_return': (batch, 2),
        'accepted_alpha': (batch, 2),
    }
    for name, shape in expected.items():
        value = locals()[name]
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f'{name} must be finite with shape {shape}')
    if bool(((accepted_alpha < 0) | (accepted_alpha > 1)).any().item()):
        raise ValueError('accepted_alpha must lie in [0, 1]')

    scaled_base = base_return.detach() / config.return_scale
    adjusted = (
        branch_return.detach() / config.return_scale
        - config.reject_penalty * (1.0 - accepted_alpha.detach())
    )
    centered = adjusted - adjusted.mean(dim=1, keepdim=True)
    latent_loss = -0.5 * (
        antithetic_log_prob * centered).sum(dim=1).mean()
    gate_advantage = adjusted.mean(dim=1) - scaled_base
    gate_probability = torch.sigmoid(gate_logits)
    gate_loss = -(gate_probability * gate_advantage.detach()).mean()
    entropy = Bernoulli(logits=gate_logits).entropy().mean()
    total = latent_loss + gate_loss - config.gate_entropy_coef * entropy
    return total, {
        'loss': total.detach(),
        'latent_loss': latent_loss.detach(),
        'gate_loss': gate_loss.detach(),
        'gate_entropy': entropy.detach(),
        'gate_probability': gate_probability.detach().mean(),
        'gate_advantage': gate_advantage.detach().mean(),
        'base_return': base_return.detach().mean(),
        'branch_return': branch_return.detach().mean(),
    }


__all__ = [
    'ResidualBanditConfig',
    'geometry_groups',
    'sample_group_balanced_indices',
    'residual_bandit_loss',
]
