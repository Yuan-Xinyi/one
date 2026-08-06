"""One-shot continuous seed generator.

The generator maps one task geometry ``(p0, line_dir, n_target)`` directly to
one FR3 joint configuration.  It deliberately has no candidate axis: a
deployment forward pass produces exactly one seed.  A separate safety module
decides whether that seed can be used directly, needs one IK projection, or
must fall back to the supplied classical seed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping

import math
import torch
import torch.nn as nn


@dataclass(frozen=True)
class DirectSeedConfig:
    """Architecture and joint-interior settings."""

    task_dim: int = 9
    q_dim: int = 7
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    limit_fraction: float = 0.98

    def __post_init__(self) -> None:
        if self.task_dim != 9:
            raise ValueError('task_dim must be 9 for (p0, line_dir, n_target)')
        if self.q_dim != 7:
            raise ValueError('q_dim must be 7 for FR3')
        if self.hidden_dim < 8:
            raise ValueError('hidden_dim must be at least 8')
        if self.n_hidden_layers < 1:
            raise ValueError('n_hidden_layers must be positive')
        if (not math.isfinite(self.limit_fraction)
                or not 0.0 < self.limit_fraction < 1.0):
            raise ValueError('limit_fraction must be finite and in (0, 1)')


def direct_seed_task(
    p0: torch.Tensor,
    line_dir: torch.Tensor,
    n_target: torch.Tensor,
) -> torch.Tensor:
    """Build the canonical ``(B, 9)`` direct-generator condition."""
    if not all(isinstance(value, torch.Tensor)
               for value in (p0, line_dir, n_target)):
        raise TypeError('p0, line_dir, and n_target must be tensors')
    if p0.ndim != 2 or p0.shape[-1] != 3:
        raise ValueError(f'p0 must have shape (B, 3), got {tuple(p0.shape)}')
    expected = p0.shape
    for name, value in (('line_dir', line_dir), ('n_target', n_target)):
        if value.shape != expected:
            raise ValueError(
                f'{name} must have shape {tuple(expected)}, '
                f'got {tuple(value.shape)}')
        if value.device != p0.device or value.dtype != p0.dtype:
            raise ValueError(f'{name} must match p0 device and dtype')
    return torch.cat([p0, line_dir, n_target], dim=-1)


class DirectSeedGenerator(nn.Module):
    """Deterministic one-task-to-one-seed MLP.

    ``tanh`` and ``limit_fraction`` keep every finite output strictly inside
    the mechanical joint box.  The safety layer still performs a full
    fail-closed validity check because limits alone do not imply IK validity.
    """

    def __init__(
        self,
        q_lower: torch.Tensor,
        q_upper: torch.Tensor,
        config: DirectSeedConfig | None = None,
        *,
        task_mean: torch.Tensor | None = None,
        task_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = DirectSeedConfig() if config is None else config
        if not isinstance(self.config, DirectSeedConfig):
            raise TypeError('config must be a DirectSeedConfig')

        q_lower = torch.as_tensor(q_lower, dtype=torch.float32)
        q_upper = torch.as_tensor(q_upper, dtype=torch.float32)
        if q_lower.shape != (self.config.q_dim,):
            raise ValueError(
                f'q_lower must have shape ({self.config.q_dim},)')
        if q_upper.shape != q_lower.shape:
            raise ValueError('q_upper must match q_lower shape')
        if (not bool(torch.isfinite(q_lower).all())
                or not bool(torch.isfinite(q_upper).all())
                or not bool((q_lower < q_upper).all())):
            raise ValueError('joint limits must be finite with lower < upper')

        if task_mean is None:
            task_mean = torch.zeros(self.config.task_dim, dtype=torch.float32)
        if task_std is None:
            task_std = torch.ones(self.config.task_dim, dtype=torch.float32)
        task_mean = torch.as_tensor(task_mean, dtype=torch.float32)
        task_std = torch.as_tensor(task_std, dtype=torch.float32)
        if task_mean.shape != (self.config.task_dim,):
            raise ValueError(
                f'task_mean must have shape ({self.config.task_dim},)')
        if task_std.shape != task_mean.shape:
            raise ValueError('task_std must match task_mean shape')
        if (not bool(torch.isfinite(task_mean).all())
                or not bool(torch.isfinite(task_std).all())
                or not bool((task_std > 0.0).all())):
            raise ValueError('task normalization must be finite with std > 0')

        self.register_buffer('q_lower', q_lower.clone())
        self.register_buffer('q_upper', q_upper.clone())
        self.register_buffer('q_mid', 0.5 * (q_lower + q_upper))
        self.register_buffer('q_half', 0.5 * (q_upper - q_lower))
        self.register_buffer('task_mean', task_mean.clone())
        self.register_buffer('task_std', task_std.clone())

        layers: list[nn.Module] = []
        in_dim = self.config.task_dim
        for _ in range(self.config.n_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, self.config.hidden_dim),
                nn.SiLU(),
            ])
            in_dim = self.config.hidden_dim
        layers.append(nn.Linear(in_dim, self.config.q_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        if task.ndim != 2 or task.shape[-1] != self.config.task_dim:
            raise ValueError(
                f'task must have shape (B, {self.config.task_dim}), '
                f'got {tuple(task.shape)}')
        if not torch.is_floating_point(task):
            raise TypeError('task must have a floating dtype')
        normalised = (
            task - self.task_mean.to(dtype=task.dtype)
        ) / self.task_std.to(dtype=task.dtype)
        raw = self.network(normalised)
        q_normalised = self.config.limit_fraction * torch.tanh(raw)
        return (
            self.q_mid.to(dtype=task.dtype)
            + self.q_half.to(dtype=task.dtype) * q_normalised
        )


def direct_seed_checkpoint(model: DirectSeedGenerator) -> dict:
    """Return a self-contained, CPU checkpoint payload."""
    if not isinstance(model, DirectSeedGenerator):
        raise TypeError('model must be a DirectSeedGenerator')
    return {
        'format': 'direct-seed-generator-v1',
        'config': asdict(model.config),
        'q_lower': model.q_lower.detach().cpu(),
        'q_upper': model.q_upper.detach().cpu(),
        'task_mean': model.task_mean.detach().cpu(),
        'task_std': model.task_std.detach().cpu(),
        'model': {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }


def load_direct_seed_generator(
    checkpoint: str | Path | Mapping,
    device: torch.device | str = 'cpu',
) -> tuple[DirectSeedGenerator, Mapping]:
    """Load a generator while retaining extra training metadata."""
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location='cpu', weights_only=False)
    elif isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        raise TypeError('checkpoint must be a path or mapping')
    required = {
        'format', 'config', 'q_lower', 'q_upper',
        'task_mean', 'task_std', 'model',
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f'direct-seed checkpoint is missing keys: {sorted(missing)}')
    if payload['format'] != 'direct-seed-generator-v1':
        raise ValueError(
            f"unsupported direct-seed format {payload['format']!r}")
    model = DirectSeedGenerator(
        payload['q_lower'], payload['q_upper'],
        DirectSeedConfig(**dict(payload['config'])),
        task_mean=payload['task_mean'], task_std=payload['task_std'])
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    return model, payload


def load_deployment_generator(
    checkpoint: str | Path | Mapping,
    device: torch.device | str = 'cpu',
) -> tuple[nn.Module, Mapping]:
    """Load a contextual-RL, hard-MoE, or supervised deployment generator.

    Every format implements ``forward(task) -> one deterministic q``.
    Contextual RL is the main method; ``direct-seed-generator-v1`` remains
    supported for the explicitly labelled IK-pool bootstrap/ablation.
    """
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location='cpu', weights_only=False)
    elif isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        raise TypeError('checkpoint must be a path or mapping')
    format_name = payload.get('format')
    if format_name == 'direct-seed-contextual-rl-v1':
        # Lazy import avoids coupling the lightweight ablation model to the RL
        # optimizer implementation at module import time.
        from Yuan.unified_rl.direct_seed_rl import (
            load_direct_seed_rl_checkpoint,
        )
        actor, _, _, _, loaded = load_direct_seed_rl_checkpoint(
            payload, device=device)
        return actor, loaded
    if format_name == 'direct-seed-hard-moe-v1':
        from Yuan.unified_rl.direct_seed_rl import (
            load_direct_seed_moe_checkpoint,
        )
        actor, _, loaded = load_direct_seed_moe_checkpoint(
            payload, device=device)
        return actor, loaded
    if format_name == 'direct-seed-generator-v1':
        return load_direct_seed_generator(payload, device=device)
    raise ValueError(
        f'unsupported deployment generator format {format_name!r}')


__all__ = [
    'DirectSeedConfig',
    'DirectSeedGenerator',
    'direct_seed_checkpoint',
    'direct_seed_task',
    'load_deployment_generator',
    'load_direct_seed_generator',
]
