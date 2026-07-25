"""Standalone residual head for the frozen seed-policy representation."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn as nn


LATENT_DIM = 4


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return value


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


@dataclass(frozen=True)
class ResidualSeedHeadConfig:
    """Serializable architecture for a residual head over a ``2H`` input.

    The default gate starts just below the deployment threshold. Together
    with the exactly zero-initialized mean this is an exact discrete-policy
    baseline, while avoiding a saturated Bernoulli gate during learning.
    """

    input_dim: int
    hidden_dim: int = 128
    gate_initial_logit: float = -0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, 'input_dim', _positive_int(
            'input_dim', self.input_dim))
        object.__setattr__(self, 'hidden_dim', _positive_int(
            'hidden_dim', self.hidden_dim))
        object.__setattr__(self, 'gate_initial_logit', _finite_float(
            'gate_initial_logit', self.gate_initial_logit))
        if self.input_dim % 2 != 0:
            raise ValueError('input_dim must be even because it represents 2H')

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualSeedHead(nn.Module):
    """Predict a residual gate and task-aligned four-dimensional latent.

    The head intentionally owns no seed selector. Its input is detached in
    :meth:`forward`, so optimizing this module cannot update the selector
    backbone that produced the selected ``(candidate, context)`` pair.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        gate_initial_logit: float = -0.01,
    ):
        super().__init__()
        config = ResidualSeedHeadConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            gate_initial_logit=gate_initial_logit,
        )
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.gate_initial_logit = config.gate_initial_logit
        self.trunk = nn.Sequential(
            _orthogonal_init(
                nn.Linear(self.input_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
            _orthogonal_init(
                nn.Linear(self.hidden_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
        )
        self.gate = nn.Linear(self.hidden_dim, 1)
        self.mean = nn.Linear(self.hidden_dim, LATENT_DIM)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.gate_initial_logit)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    @property
    def architecture(self) -> dict[str, int | float]:
        """Serializable architecture metadata for checkpoints."""
        return ResidualSeedHeadConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            gate_initial_logit=self.gate_initial_logit,
        ).to_dict()

    def _check_input(self, representation: torch.Tensor) -> None:
        if not torch.is_tensor(representation):
            raise ValueError('representation must be a tensor')
        if representation.ndim != 2:
            raise ValueError('representation must have shape (B,2H)')
        if representation.shape[0] < 1:
            raise ValueError('representation must contain at least one task')
        if representation.shape[1] != self.input_dim:
            raise ValueError(
                f'expected representation dim {self.input_dim}, '
                f'got {representation.shape[1]}')
        if not torch.is_floating_point(representation):
            raise ValueError('representation must have a floating-point dtype')
        reference = self.trunk[0].weight
        if representation.device != reference.device:
            raise ValueError('representation and residual head must be on the same device')
        if representation.dtype != reference.dtype:
            raise ValueError('representation and residual head must have the same dtype')
        if not bool(torch.isfinite(representation).all().item()):
            raise ValueError('representation must be finite')

    def forward(
        self, representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return gate logits ``(B,)`` and latent means ``(B,4)``."""
        self._check_input(representation)
        hidden = self.trunk(representation.detach())
        gate_logit = self.gate(hidden).squeeze(-1)
        mean = self.mean(hidden)
        if (not bool(torch.isfinite(gate_logit).all().item())
                or not bool(torch.isfinite(mean).all().item())):
            raise ValueError('residual head output must be finite')
        return gate_logit, mean

    @torch.no_grad()
    def deterministic_action(
        self, representation: torch.Tensor, gate_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a thresholded gate and bounded latent for deployment."""
        gate_threshold = _finite_float('gate_threshold', gate_threshold)
        if not 0.0 < gate_threshold < 1.0:
            raise ValueError('gate_threshold must be strictly between zero and one')
        gate_logit, mean = self(representation)
        gate = torch.sigmoid(gate_logit) >= gate_threshold
        latent = torch.tanh(mean)
        return gate, latent


def antithetic_gaussian_actions_and_log_prob(
    mean: torch.Tensor,
    noise: torch.Tensor,
    std: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build antithetic bounded actions and score-function log probabilities.

    ``noise`` is an externally sampled standard-normal tensor with the same
    ``(B,4)`` shape as ``mean``. The returned actions have shape ``(B,2,4)``
    in ``(+noise, -noise)`` order and are squashed with ``tanh``. They are
    detached from the graph, while the ``(B,2)`` Gaussian log probabilities
    retain their dependency on ``mean``. Consequently gradients are pure
    score-function gradients rather than pathwise gradients.

    The log probability is that of the underlying unsquashed Gaussian. The
    omitted tanh Jacobian depends only on the detached action, so it does not
    change score-function gradients or likelihood ratios for a fixed action.
    """
    if not torch.is_tensor(mean) or not torch.is_tensor(noise):
        raise ValueError('mean and noise must be tensors')
    if mean.ndim != 2 or mean.shape[0] < 1 or mean.shape[1] != LATENT_DIM:
        raise ValueError(f'mean must have shape (B,{LATENT_DIM})')
    if noise.shape != mean.shape:
        raise ValueError('noise must have the same shape as mean')
    if not torch.is_floating_point(mean) or not torch.is_floating_point(noise):
        raise ValueError('mean and noise must have floating-point dtypes')
    if noise.device != mean.device or noise.dtype != mean.dtype:
        raise ValueError('mean and noise must have the same device and dtype')
    if (not bool(torch.isfinite(mean).all().item())
            or not bool(torch.isfinite(noise).all().item())):
        raise ValueError('mean and noise must be finite')
    std = _finite_float('std', std)
    if std <= 0.0:
        raise ValueError('std must be positive')

    detached_mean = mean.detach()
    offset = std * noise.detach()
    raw_actions = torch.stack(
        [detached_mean + offset, detached_mean - offset], dim=1).detach()
    scaled_error = (raw_actions - mean.unsqueeze(1)) / std
    log_prob = (
        -0.5 * scaled_error.square()
        - math.log(std)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=-1)
    actions = torch.tanh(raw_actions).detach()
    return actions, log_prob


__all__ = [
    'LATENT_DIM',
    'ResidualSeedHead',
    'ResidualSeedHeadConfig',
    'antithetic_gaussian_actions_and_log_prob',
]
