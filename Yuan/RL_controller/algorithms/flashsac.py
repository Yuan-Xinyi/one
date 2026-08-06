"""Torch-native FlashSAC core for the Yuan batched controller environment.

This is a dependency-light adaptation of the official MIT-licensed FlashSAC
implementation:

    Holiday-Robot/FlashSAC
    commit 87edc9061150ae9e962dd84e6544e27a1554b3ab

The upstream license text is retained in ``FLASHSAC_LICENSE.md``.

The algorithmic components are kept faithful to the official implementation:
unit-normalized inverted-residual networks, pre-activation batch
normalization, post RMS normalization, a categorical clipped-double critic,
cross-batch Bellman prediction, adaptive reward scaling, automatic
temperature tuning, Zeta-distributed noise repetition, uniform n-step replay,
delayed actor updates, and an EMA target critic.

Only the integration boundary differs: dimensions are passed directly and all
environment data stays as Torch tensors.  Gymnasium, Hydra, NumPy transfers,
and the official environment registry are intentionally not required.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


OFFICIAL_REPOSITORY = 'https://github.com/Holiday-Robot/FlashSAC'
OFFICIAL_COMMIT = '87edc9061150ae9e962dd84e6544e27a1554b3ab'
CHECKPOINT_FORMAT = 'yuan-torch-flashsac-v2'
REPLAY_FORMAT = 'yuan-torch-uniform-nstep-replay-v2'


@dataclass
class FlashSACConfig:
    """FlashSAC hyperparameters.

    Defaults mirror the official GPU configuration where possible.  Schedule
    lengths are optimizer-update counts, not environment steps.
    """

    gamma: float = 0.99
    n_step: int = 3
    buffer_max_length: int = 10_000_000
    buffer_min_length: int = 100_000
    buffer_device: str = 'cuda'
    sample_batch_size: int = 2048
    normalize_reward: bool = True
    normalized_g_max: float = 5.0

    learning_rate_init: float = 3e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1.5e-4
    learning_rate_warmup_steps: int = 0
    learning_rate_decay_steps: int = 58_594

    actor_num_blocks: int = 2
    actor_hidden_dim: int = 128
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 2.0
    actor_noise_zeta_max: int = 16
    actor_update_period: int = 2

    critic_num_blocks: int = 2
    critic_hidden_dim: int = 256
    critic_num_bins: int = 101
    critic_min_v: float = -5.0
    critic_max_v: float = 5.0
    critic_target_update_tau: float = 0.01

    temperature_initial_value: float = 0.01
    temperature_target_sigma: float = 0.15

    use_compile: bool = True
    compile_mode: str = 'auto'
    use_amp: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> 'FlashSACConfig':
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f'unknown FlashSAC config keys: {unknown}')
        config = cls(**dict(values))
        config.validate()
        return config

    def validate(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if self.n_step <= 0:
            raise ValueError('n_step must be positive')
        if self.buffer_max_length <= 0:
            raise ValueError('buffer_max_length must be positive')
        if not 0 < self.buffer_min_length <= self.buffer_max_length:
            raise ValueError(
                'buffer_min_length must be in (0, buffer_max_length]')
        if not 0 < self.sample_batch_size <= self.buffer_max_length:
            raise ValueError(
                'sample_batch_size must be in (0, buffer_max_length]')
        if not math.isfinite(self.normalized_g_max) or self.normalized_g_max <= 0:
            raise ValueError('normalized_g_max must be finite and positive')
        for name in (
                'learning_rate_init', 'learning_rate_peak',
                'learning_rate_end'):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        if self.learning_rate_warmup_steps < 0:
            raise ValueError('learning_rate_warmup_steps must be non-negative')
        if self.learning_rate_decay_steps <= self.learning_rate_warmup_steps:
            raise ValueError(
                'learning_rate_decay_steps must exceed warmup steps')
        if self.actor_num_blocks < 0 or self.critic_num_blocks < 0:
            raise ValueError('network block counts must be non-negative')
        if self.actor_hidden_dim <= 0 or self.critic_hidden_dim <= 0:
            raise ValueError('network hidden dimensions must be positive')
        if not math.isfinite(self.actor_bc_alpha) or self.actor_bc_alpha < 0:
            raise ValueError('actor_bc_alpha must be finite and non-negative')
        if (not math.isfinite(self.actor_noise_zeta_mu)
                or self.actor_noise_zeta_mu <= 0):
            raise ValueError(
                'actor_noise_zeta_mu must be finite and positive')
        if self.actor_noise_zeta_max <= 0:
            raise ValueError('actor_noise_zeta_max must be positive')
        if self.actor_update_period <= 0:
            raise ValueError('actor_update_period must be positive')
        if self.critic_num_bins < 2:
            raise ValueError('critic_num_bins must be at least two')
        if not self.critic_min_v < self.critic_max_v:
            raise ValueError('critic_min_v must be below critic_max_v')
        if not 0.0 < self.critic_target_update_tau <= 1.0:
            raise ValueError('critic_target_update_tau must be in (0, 1]')
        if (not math.isfinite(self.temperature_initial_value)
                or self.temperature_initial_value <= 0):
            raise ValueError(
                'temperature_initial_value must be finite and positive')
        if (not math.isfinite(self.temperature_target_sigma)
                or self.temperature_target_sigma <= 0):
            raise ValueError(
                'temperature_target_sigma must be finite and positive')
        if self.compile_mode not in (
                'auto', 'default', 'reduce-overhead', 'max-autotune'):
            raise ValueError(f'unsupported compile_mode: {self.compile_mode}')


def _resolve_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == 'cuda' and result.index is None:
        result = torch.device('cuda:0')
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested but unavailable: {result}')
    return result


def _resolve_compile_mode(mode: str) -> str:
    if mode != 'auto':
        return mode
    version = torch.__version__.split('+', 1)[0].split('.')[:2]
    major, minor = (int(value) for value in version)
    if (major, minor) >= (2, 9):
        return 'max-autotune'
    return 'reduce-overhead'


def _safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - tanh(x)^2)``."""
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class UnitLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x, self.running_mean, self.running_var, self.weight, self.bias,
            training=training, momentum=self.momentum, eps=self.eps)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale, bias = self.weight, self.bias
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.copy_(scale * norm_factor)
        self.bias.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            x, self.weight.shape, self.weight, eps=self.eps)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale = self.weight
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return self.w(self.norm(x, training=training))


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = F.relu(self.norm1(x, training=training))
        x = self.w2(x)
        x = F.relu(self.norm2(x, training=training))
        return x + residual


class NormalTanhPolicy(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int,
                 log_std_min: float = -10.0, log_std_max: float = 2.0):
        super().__init__()
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))
        self.std_w = UnitLinear(hidden_dim, action_dim)
        self.std_bias = nn.Parameter(torch.zeros(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(
            self, x: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        del training
        mean = F.linear(x, self.mean_w.w.weight, self.mean_bias)
        raw_log_std = F.linear(x, self.std_w.w.weight, self.std_bias)
        log_std = self.log_std_min + (
            self.log_std_max - self.log_std_min) * 0.5 * (
                1.0 + torch.tanh(raw_log_std))
        return mean, torch.exp(log_std)

    def forward(
            self, x: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(x, training)
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = distribution.log_prob(raw_action)
        log_prob = log_prob - _safe_tanh_log_det_jacobian(raw_action)
        return action, {'log_prob': log_prob.sum(dim=-1)}


class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_ensemble, output_dim, input_dim))
        for index in range(num_ensemble):
            nn.init.orthogonal_(self.weight[index], gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum('nbi,noi->nbo', x, self.weight)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int,
                 momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer(
            'running_mean', torch.zeros(num_ensemble, input_dim))
        self.register_buffer(
            'running_var', torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            if x.shape[1] <= 1:
                raise ValueError(
                    'FlashSAC critic BatchNorm needs batch size > 1')
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                batch_size = x.shape[1]
                self.running_mean.lerp_(
                    mean.squeeze(1).float(), self.momentum)
                unbiased = var.squeeze(1) * (
                    batch_size / (batch_size - 1))
                self.running_var.lerp_(unbiased.float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(
                self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale, bias = self.weight, self.bias
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.copy_(scale * norm_factor)
        self.bias.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        scale = self.weight
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return self.w(self.norm(x, training=training))


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int,
                 expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(
            num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(
            num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(
            num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = F.relu(self.norm1(x, training=training))
        x = self.w2(x)
        x = F.relu(self.norm2(x, training=training))
        return x + residual


class EnsembleCategoricalValue(nn.Module):
    bin_values: torch.Tensor

    def __init__(self, num_ensemble: int, hidden_dim: int, num_bins: int,
                 min_v: float, max_v: float):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            'bin_values',
            torch.linspace(
                min_v, max_v, num_bins, dtype=torch.float32
            ).reshape(1, 1, -1))

    def forward(
            self, x: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del training
        logits = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(logits, dim=-1)
        value = torch.sum(log_prob.exp() * self.bin_values, dim=-1)
        return value, {'log_prob': log_prob}


class FlashSACActor(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int,
                 action_dim: int):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim, action_dim)

    def _features(self, observations: torch.Tensor,
                  training: bool) -> torch.Tensor:
        x = self.embedder(observations, training)
        for block in self.encoder:
            x = block(x, training)
        return self.post_norm(x)

    def get_mean_and_std(
            self, observations: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.get_mean_and_std(
            self._features(observations, training), training)

    def forward(
            self, observations: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predictor(
            self._features(observations, training), training)


class FlashSACDoubleCritic(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int,
                 num_bins: int, min_v: float, max_v: float,
                 num_qs: int = 2):
        super().__init__()
        self.num_qs = num_qs
        self.embedder = EnsembleFlashSACEmbedder(
            num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList(
            [EnsembleFlashSACBlock(num_qs, hidden_dim)
             for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_qs, hidden_dim, num_bins, min_v, max_v)

    def forward(
            self, observations: torch.Tensor, actions: torch.Tensor,
            training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return self.predictor(x, training)


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(
            [math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)


def _normalize_parameters(module: nn.Module) -> None:
    with torch.no_grad():
        for child in module.modules():
            normalize = getattr(child, 'normalize_parameters', None)
            if callable(normalize):
                normalize()


def _make_normalize_parameters_fn(
        module: nn.Module, use_compile: bool,
        compile_mode: str) -> Any:
    """Build the same post-step normalization boundary as upstream.

    Upstream captures the normalization-capable modules once and compiles the
    closure alongside the network.  Keeping that boundary avoids graph breaks
    from walking ``module.modules()`` after every optimizer step.
    """
    norm_modules = [
        child for child in module.modules()
        if callable(getattr(child, 'normalize_parameters', None))]

    @torch.no_grad()
    def normalize_fn() -> None:
        for child in norm_modules:
            child.normalize_parameters()

    if use_compile:
        return torch.compile(normalize_fn, mode=compile_mode)
    return normalize_fn


def _make_ema_fn(
        target: nn.Module, source: nn.Module, tau: float,
        use_compile: bool, compile_mode: str) -> Any:
    """Prepare the target update closure at the official update boundary."""
    target_parameters = list(target.parameters())
    source_parameters = list(source.parameters())

    @torch.no_grad()
    def ema_fn() -> None:
        torch._foreach_lerp_(target_parameters, source_parameters, tau)

    if use_compile:
        return torch.compile(ema_fn, mode=compile_mode)
    return ema_fn


def _warmup_cosine_value(
        step: int, init_value: float, peak_value: float, end_value: float,
        warmup_steps: int, decay_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return init_value + (peak_value - init_value) * (
            step / warmup_steps)
    if step < decay_steps:
        decay_step = step - warmup_steps
        progress = decay_step / (decay_steps - warmup_steps)
        return end_value + (peak_value - end_value) * 0.5 * (
            1.0 + math.cos(math.pi * progress))
    return end_value


def _make_scheduler(
        optimizer: torch.optim.Optimizer,
        config: FlashSACConfig) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_value(
            step, config.learning_rate_init, config.learning_rate_peak,
            config.learning_rate_end, config.learning_rate_warmup_steps,
            config.learning_rate_decay_steps) / config.learning_rate_peak)


class RunningMeanStd:
    def __init__(self, device: torch.device, epsilon: float = 1e-4):
        self.mean = torch.zeros(1, dtype=torch.float32, device=device)
        self.var = torch.ones(1, dtype=torch.float32, device=device)
        self.count = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.epsilon = epsilon

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> None:
        sample_mean = samples.mean(dim=0)
        sample_var = samples.var(dim=0, unbiased=False)
        sample_count = float(samples.shape[0])
        delta = sample_mean - self.mean
        total_count = self.count + sample_count
        ratio = sample_count / total_count
        new_mean = self.mean + delta * ratio
        m_a = self.var * (self.count + self.epsilon)
        m_b = sample_var * sample_count
        m2 = m_a + m_b + delta.square() * self.count * ratio
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def state_dict(self) -> dict[str, Any]:
        return {
            'mean': self.mean.detach().clone(),
            'var': self.var.detach().clone(),
            'count': self.count.detach().clone(),
            'epsilon': self.epsilon,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mean.copy_(torch.as_tensor(state['mean']).to(self.mean))
        self.var.copy_(torch.as_tensor(state['var']).to(self.var))
        self.count.copy_(torch.as_tensor(state['count']).to(self.count))
        self.epsilon = float(state.get('epsilon', self.epsilon))


class RewardNormalizer:
    """Official FlashSAC adaptive reward scaling."""

    def __init__(self, gamma: float, g_max: float, device: torch.device,
                 epsilon: float = 1e-8):
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.return_accumulator = torch.zeros(
            1, dtype=torch.float32, device=device)
        self.return_max = torch.zeros(
            1, dtype=torch.float32, device=device)
        self.return_rms = RunningMeanStd(device)

    @torch.no_grad()
    def update(self, reward: torch.Tensor, terminated: torch.Tensor,
               truncated: torch.Tensor) -> None:
        done = torch.logical_or(
            terminated.bool(), truncated.bool()).float()
        self.return_accumulator = (
            self.gamma * (1.0 - done) * self.return_accumulator + reward)
        self.return_max = torch.maximum(
            self.return_max, self.return_accumulator.abs().max())
        self.return_rms.update(self.return_accumulator)

    def normalize(self, reward: torch.Tensor) -> torch.Tensor:
        variance_denominator = torch.sqrt(
            self.return_rms.var + self.epsilon)
        maximum_denominator = self.return_max / self.g_max
        denominator = torch.maximum(
            variance_denominator, maximum_denominator)
        return reward / denominator

    def reset_episode_accumulator(self, n_envs: int | None = None) -> None:
        """Reset only unfinished-return state after an external env reset."""
        if n_envs is None:
            self.return_accumulator.zero_()
        else:
            self.return_accumulator = torch.zeros(
                n_envs, dtype=self.return_accumulator.dtype,
                device=self.return_accumulator.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            'gamma': self.gamma,
            'g_max': self.g_max,
            'epsilon': self.epsilon,
            'return_accumulator': self.return_accumulator.detach().clone(),
            'return_max': self.return_max.detach().clone(),
            'return_rms': self.return_rms.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not math.isclose(float(state['gamma']), self.gamma):
            raise ValueError('reward normalizer gamma mismatch')
        if not math.isclose(float(state['g_max']), self.g_max):
            raise ValueError('reward normalizer g_max mismatch')
        self.epsilon = float(state.get('epsilon', self.epsilon))
        self.return_accumulator = torch.as_tensor(
            state['return_accumulator']).to(self.return_accumulator)
        self.return_max.copy_(
            torch.as_tensor(state['return_max']).to(self.return_max))
        self.return_rms.load_state_dict(state['return_rms'])


class TorchUniformReplay:
    """Device-resident uniform n-step replay without Gym spaces."""

    def __init__(self, obs_dim: int, action_dim: int, n_step: int,
                 gamma: float, max_length: int, min_length: int,
                 sample_batch_size: int, device: torch.device | str):
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError('replay dimensions must be positive')
        if n_step <= 0:
            raise ValueError('replay n_step must be positive')
        if not 0 < min_length <= max_length:
            raise ValueError('invalid replay min/max lengths')
        if not 0 < sample_batch_size <= max_length:
            raise ValueError('invalid replay sample batch size')
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.gamma = gamma
        self.max_length = max_length
        self.min_length = min_length
        self.sample_batch_size = sample_batch_size
        self.device = _resolve_device(device)
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        pin_memory = (
            self.device.type == 'cpu' and torch.cuda.is_available())
        common = {'device': self.device, 'pin_memory': pin_memory}
        self.observations = torch.empty(
            (self.max_length, self.obs_dim), dtype=torch.float32, **common)
        self.next_observations = torch.empty(
            (self.max_length, self.obs_dim), dtype=torch.float32, **common)
        self.actions = torch.empty(
            (self.max_length, self.action_dim), dtype=torch.float32, **common)
        self.rewards = torch.empty(
            self.max_length, dtype=torch.float32, **common)
        self.terminated = torch.empty(
            self.max_length, dtype=torch.float32, **common)
        self.truncated = torch.empty(
            self.max_length, dtype=torch.float32, **common)
        self.bootstrap_discount = torch.empty(
            self.max_length, dtype=torch.float32, **common)
        self._n_step_transitions: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.n_step)
        self._num_in_buffer = 0
        self._current_index = 0

    def _copy_to_device(self, value: torch.Tensor) -> torch.Tensor:
        return value.detach().to(
            device=self.device, dtype=torch.float32, copy=True)

    def _validate_transition(
            self, transition: Mapping[str, torch.Tensor]) -> int:
        required = {
            'observation', 'action', 'reward', 'terminated', 'truncated',
            'next_observation'}
        missing = sorted(required - set(transition))
        if missing:
            raise ValueError(f'replay transition missing keys: {missing}')
        batch_size = transition['observation'].shape[0]
        expected = {
            'observation': (batch_size, self.obs_dim),
            'action': (batch_size, self.action_dim),
            'reward': (batch_size,),
            'terminated': (batch_size,),
            'truncated': (batch_size,),
            'next_observation': (batch_size, self.obs_dim),
        }
        for key, shape in expected.items():
            if tuple(transition[key].shape) != shape:
                raise ValueError(
                    f'{key} has shape {tuple(transition[key].shape)}, '
                    f'expected {shape}')
        if batch_size > self.max_length:
            raise ValueError(
                'one replay insertion cannot exceed buffer capacity')
        return batch_size

    def add(self, transition: Mapping[str, torch.Tensor]) -> None:
        self._validate_transition(transition)
        copied = {
            key: self._copy_to_device(value)
            for key, value in transition.items()}
        self._n_step_transitions.append(copied)
        if len(self._n_step_transitions) < self.n_step:
            return

        oldest = self._n_step_transitions[0]
        batch_size = oldest['observation'].shape[0]
        n_step_reward = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device)
        n_step_terminated = torch.zeros_like(n_step_reward)
        n_step_truncated = torch.zeros_like(n_step_reward)
        n_step_next_obs = oldest['next_observation'].clone()
        bootstrap_discount = torch.ones_like(n_step_reward)
        active = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)

        # Each vector lane may encounter a boundary at a different offset.
        # Accumulate only through its first boundary and retain gamma**k.  The
        # critic then bootstraps from a time-limit terminal observation with
        # gamma**k, while true termination still masks the bootstrap.
        for item in self._n_step_transitions:
            active_float = active.float()
            n_step_reward += (
                active_float * bootstrap_discount * item['reward'])
            n_step_next_obs[active] = item['next_observation'][active]
            boundary = active & torch.logical_or(
                item['terminated'].bool(), item['truncated'].bool())
            n_step_terminated[boundary] = item['terminated'][boundary]
            n_step_truncated[boundary] = item['truncated'][boundary]
            bootstrap_discount = torch.where(
                active, bootstrap_discount * self.gamma,
                bootstrap_discount)
            active = active & (~boundary)

        end_index = self._current_index + batch_size
        if end_index <= self.max_length:
            indices: slice | torch.Tensor = slice(
                self._current_index, end_index)
        else:
            indices = (
                torch.arange(batch_size, device=self.device)
                + self._current_index) % self.max_length

        self.observations[indices] = oldest['observation']
        self.actions[indices] = oldest['action']
        self.rewards[indices] = n_step_reward
        self.terminated[indices] = n_step_terminated
        self.truncated[indices] = n_step_truncated
        self.bootstrap_discount[indices] = bootstrap_discount
        self.next_observations[indices] = n_step_next_obs
        self._num_in_buffer = min(
            self._num_in_buffer + batch_size, self.max_length)
        self._current_index = end_index % self.max_length

    def can_sample(self) -> bool:
        # Sampling is with replacement, so the official implementation only
        # gates on the warm-up length (which is normally > batch size).
        return len(self) >= self.min_length

    def sample(
            self, indices: torch.Tensor | None = None
            ) -> dict[str, torch.Tensor]:
        if len(self) == 0:
            raise RuntimeError('cannot sample an empty replay buffer')
        if indices is None:
            indices = torch.randint(
                0, len(self), (self.sample_batch_size,),
                device=self.device)
        else:
            indices = indices.to(device=self.device, dtype=torch.long)
            if bool(((indices < 0) | (indices >= len(self))).any().item()):
                raise IndexError('replay sample index out of range')
        return {
            'observation': self.observations[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'terminated': self.terminated[indices],
            'truncated': self.truncated[indices],
            'bootstrap_discount': self.bootstrap_discount[indices],
            'next_observation': self.next_observations[indices],
        }

    def state_dict(self) -> dict[str, Any]:
        length = len(self)
        return {
            'format': REPLAY_FORMAT,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'n_step': self.n_step,
            'gamma': self.gamma,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'sample_batch_size': self.sample_batch_size,
            'num_in_buffer': length,
            'current_index': self._current_index,
            'observation': self.observations[:length].detach().cpu(),
            'action': self.actions[:length].detach().cpu(),
            'reward': self.rewards[:length].detach().cpu(),
            'terminated': self.terminated[:length].detach().cpu(),
            'truncated': self.truncated[:length].detach().cpu(),
            'bootstrap_discount': (
                self.bootstrap_discount[:length].detach().cpu()),
            'next_observation': self.next_observations[:length].detach().cpu(),
            'n_step_transitions': [{
                key: value.detach().cpu()
                for key, value in transition.items()
            } for transition in self._n_step_transitions],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get('format') != REPLAY_FORMAT:
            raise ValueError('unsupported replay format')
        expected = {
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'n_step': self.n_step,
            'max_length': self.max_length,
            'min_length': self.min_length,
            'sample_batch_size': self.sample_batch_size,
        }
        for key, value in expected.items():
            if int(state[key]) != value:
                raise ValueError(f'replay {key} mismatch')
        if not math.isclose(float(state['gamma']), self.gamma):
            raise ValueError('replay gamma mismatch')
        length = int(state['num_in_buffer'])
        if not 0 <= length <= self.max_length:
            raise ValueError('invalid replay stored length')
        stored = {
            'observation': (self.observations, (length, self.obs_dim)),
            'action': (self.actions, (length, self.action_dim)),
            'reward': (self.rewards, (length,)),
            'terminated': (self.terminated, (length,)),
            'truncated': (self.truncated, (length,)),
            'bootstrap_discount': (
                self.bootstrap_discount, (length,)),
            'next_observation': (
                self.next_observations, (length, self.obs_dim)),
        }
        for key, (destination, shape) in stored.items():
            source = torch.as_tensor(state[key])
            if tuple(source.shape) != shape:
                raise ValueError(f'invalid stored replay {key} shape')
            destination[:length].copy_(
                source.to(destination, non_blocking=True))
        self._num_in_buffer = length
        self._current_index = int(state['current_index'])
        if not 0 <= self._current_index < self.max_length:
            raise ValueError('invalid replay current index')
        pending = state.get('n_step_transitions')
        if not isinstance(pending, list) or len(pending) > self.n_step:
            raise ValueError('invalid stored replay n-step transitions')
        self._n_step_transitions.clear()
        for transition in pending:
            if not isinstance(transition, Mapping):
                raise ValueError('invalid stored pending transition')
            self._validate_transition(transition)
            self._n_step_transitions.append({
                key: self._copy_to_device(value)
                for key, value in transition.items()})


def _select_min_q_log_probs(
        next_qs: torch.Tensor,
        next_q_log_probs: torch.Tensor) -> torch.Tensor:
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)
    return torch.gather(
        next_q_log_probs, dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins))[0]


def categorical_td_target(
        target_log_probs: torch.Tensor, reward: torch.Tensor,
        done: torch.Tensor, actor_entropy: torch.Tensor,
        gamma: float | torch.Tensor,
        num_bins: int, min_v: float, max_v: float) -> torch.Tensor:
    """Project the soft Bellman target onto a fixed categorical support."""
    batch_size = reward.shape[0]
    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device,
        dtype=target_log_probs.dtype).reshape(1, -1)
    bootstrap_discount = torch.as_tensor(
        gamma, dtype=target_log_probs.dtype,
        device=target_log_probs.device).reshape(-1, 1)
    target_values = reward + bootstrap_discount * (
        bin_values - actor_entropy) * (1.0 - done)
    target_values = target_values.clamp(min_v, max_v)
    positions = (target_values - min_v) / bin_width
    lower = torch.floor(positions).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)
    fraction = positions - lower.float()
    source_prob = target_log_probs.exp()
    target_prob = torch.zeros(
        batch_size, num_bins, dtype=source_prob.dtype,
        device=source_prob.device)
    target_prob.scatter_add_(1, lower, source_prob * (1.0 - fraction))
    target_prob.scatter_add_(1, upper, source_prob * fraction)
    return target_prob


def _build_truncated_zeta_cdf(
        exponent: float, maximum: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(
        1, maximum + 1, dtype=torch.float32, device=device)
    probability = values.pow(-exponent)
    return torch.cumsum(probability / probability.sum(), dim=0)


def _sample_zeta_integer(cdf: torch.Tensor) -> torch.Tensor:
    sample = torch.rand((), device=cdf.device)
    index = torch.argmax((sample < cdf).to(torch.int32))
    return (index + 1).to(torch.int32)


class FlashSACAgent:
    """Official FlashSAC update logic with a Torch-native public interface."""

    def __init__(self, obs_dim: int, action_dim: int, n_envs: int,
                 config: FlashSACConfig, device: torch.device | str):
        if obs_dim <= 0 or action_dim <= 0 or n_envs <= 0:
            raise ValueError('obs_dim, action_dim, and n_envs must be positive')
        config.validate()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = _resolve_device(device)
        self.config = replace(
            config, compile_mode=_resolve_compile_mode(config.compile_mode))
        if self.config.use_amp and self.device.type != 'cuda':
            raise ValueError('FlashSAC AMP is supported only on CUDA')

        actor = FlashSACActor(
            self.config.actor_num_blocks, obs_dim,
            self.config.actor_hidden_dim, action_dim).to(self.device)
        critic = FlashSACDoubleCritic(
            self.config.critic_num_blocks, obs_dim + action_dim,
            self.config.critic_hidden_dim, self.config.critic_num_bins,
            self.config.critic_min_v,
            self.config.critic_max_v).to(self.device)
        target_critic = copy.deepcopy(critic).to(self.device)
        temperature = FlashSACTemperature(
            self.config.temperature_initial_value).to(self.device)

        _normalize_parameters(actor)
        _normalize_parameters(critic)
        _normalize_parameters(target_critic)

        use_fused = self.device.type == 'cuda'
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=self.config.learning_rate_peak,
            fused=use_fused)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=self.config.learning_rate_peak,
            fused=use_fused)
        self.temperature_optimizer = torch.optim.Adam(
            temperature.parameters(), lr=self.config.learning_rate_peak,
            fused=use_fused)
        self.actor_scheduler = _make_scheduler(
            self.actor_optimizer, self.config)
        self.critic_scheduler = _make_scheduler(
            self.critic_optimizer, self.config)
        self.temperature_scheduler = _make_scheduler(
            self.temperature_optimizer, self.config)

        self._actor_raw = actor
        self._critic_raw = critic
        self._target_critic_raw = target_critic
        self.temperature = temperature
        self.actor: nn.Module = actor
        self.critic: nn.Module = critic
        self.target_critic: nn.Module = target_critic
        self._actor_mean_and_std = actor.get_mean_and_std
        if self.config.use_compile:
            mode = self.config.compile_mode
            self.actor = torch.compile(actor, mode=mode)
            self.critic = torch.compile(critic, mode=mode)
            self.target_critic = torch.compile(target_critic, mode=mode)
            # The official implementation compiles this inference path
            # separately because data collection does not call forward().
            self._actor_mean_and_std = torch.compile(
                actor.get_mean_and_std, mode=mode)
        self._normalize_actor = _make_normalize_parameters_fn(
            actor, self.config.use_compile, self.config.compile_mode)
        self._normalize_critic = _make_normalize_parameters_fn(
            critic, self.config.use_compile, self.config.compile_mode)
        self._update_target_critic = _make_ema_fn(
            target_critic, critic, self.config.critic_target_update_tau,
            self.config.use_compile, self.config.compile_mode)

        self.grad_scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.config.use_amp)
        target_entropy = 0.5 * action_dim * math.log(
            2.0 * math.pi * math.e
            * self.config.temperature_target_sigma ** 2)
        self.target_entropy = float(target_entropy)
        self.update_step = 0
        self.interaction_step = 0

        self.zeta_cdf = _build_truncated_zeta_cdf(
            self.config.actor_noise_zeta_mu,
            self.config.actor_noise_zeta_max, self.device)
        self.noise_repeat_length = torch.tensor(
            1, dtype=torch.int32, device=self.device)
        self.noise_repeat_count = torch.tensor(
            0, dtype=torch.int32, device=self.device)
        self.cached_noise = torch.randn(
            n_envs, action_dim, device=self.device)

        self.reward_normalizer = (
            RewardNormalizer(
                self.config.gamma, self.config.normalized_g_max, self.device)
            if self.config.normalize_reward else None)
        replay_device = self.config.buffer_device
        parsed_replay_device = (
            None if replay_device == 'same' else torch.device(replay_device))
        if (replay_device == 'same'
                or (parsed_replay_device is not None
                    and parsed_replay_device.type == 'cuda'
                    and parsed_replay_device.index is None)):
            # A bare ``cuda`` means co-locate with the agent.  Resolving it
            # independently would silently put replay on cuda:0 even when the
            # training agent explicitly runs on cuda:1.
            replay_device = str(self.device)
        self.replay = TorchUniformReplay(
            obs_dim, action_dim, self.config.n_step, self.config.gamma,
            self.config.buffer_max_length, self.config.buffer_min_length,
            self.config.sample_batch_size, replay_device)

    @torch.no_grad()
    def actor_mean(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(
            device=self.device, dtype=torch.float32)
        mean, _ = self._actor_mean_and_std(
            observations, training=False)
        return torch.tanh(mean)

    @torch.no_grad()
    def sample_actions(
            self, observations: torch.Tensor,
            training: bool = True) -> torch.Tensor:
        observations = observations.to(
            device=self.device, dtype=torch.float32)
        if tuple(observations.shape) != (self.n_envs, self.obs_dim):
            raise ValueError(
                f'observations must have shape '
                f'({self.n_envs}, {self.obs_dim})')
        mean, std = self._actor_mean_and_std(
            observations, training=False)
        if not training:
            return torch.tanh(mean)
        reinitialize = (
            (self.noise_repeat_count == 0)
            | (self.noise_repeat_count >= self.noise_repeat_length))
        new_noise = torch.randn_like(mean)
        new_length = _sample_zeta_integer(self.zeta_cdf)
        self.cached_noise = torch.where(
            reinitialize, new_noise, self.cached_noise)
        self.noise_repeat_length = torch.where(
            reinitialize, new_length, self.noise_repeat_length)
        self.noise_repeat_count = torch.where(
            reinitialize, torch.zeros_like(self.noise_repeat_count),
            self.noise_repeat_count)
        self.noise_repeat_count += 1
        return torch.tanh(mean + std * self.cached_noise)

    def add_transition(
            self, observation: torch.Tensor, action: torch.Tensor,
            reward: torch.Tensor, terminated: torch.Tensor,
            truncated: torch.Tensor,
            next_observation: torch.Tensor) -> None:
        transition = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'next_observation': next_observation,
        }
        self.replay.add(transition)
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(
                reward.to(self.device, dtype=torch.float32),
                terminated.to(self.device), truncated.to(self.device))
        self.interaction_step += 1

    def can_update(self) -> bool:
        return self.replay.can_sample()

    def _actor_update(
            self, batch: Mapping[str, torch.Tensor]
            ) -> dict[str, torch.Tensor]:
        with torch.autocast(
                device_type=self.device.type, dtype=torch.float16,
                enabled=self.config.use_amp):
            actor_obs = torch.cat(
                [batch['observation'], batch['next_observation']], dim=0)
            actions_all, info = self.actor(
                observations=actor_obs, training=True)
            actions = actions_all.chunk(2, dim=0)[0]
            log_prob = info['log_prob'].chunk(2, dim=0)[0]

            self._critic_raw.requires_grad_(False)
            try:
                q_values, _ = self.critic(
                    observations=batch['observation'], actions=actions,
                    training=False)
            finally:
                self._critic_raw.requires_grad_(True)
            q_min = torch.minimum(q_values[0], q_values[1])
            temperature = self.temperature().detach()
            actor_loss = (temperature * log_prob - q_min).mean()
            if self.config.actor_bc_alpha > 0.0:
                q_abs = q_min.abs().mean().detach()
                bc_loss = (actions - batch['action']).square().mean()
                actor_loss = actor_loss + (
                    self.config.actor_bc_alpha * q_abs * bc_loss)
            entropy = -log_prob.mean()
            mean_action = actions.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.config.use_amp:
            self.grad_scaler.scale(actor_loss).backward()
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            actor_loss.backward()
            self.actor_optimizer.step()
        self.actor_scheduler.step()
        self._normalize_actor()

        temperature_value = self.temperature().clone()
        temperature_loss = temperature_value * (
            entropy.detach() - self.target_entropy)
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature_scheduler.step()
        return {
            'actor/loss': actor_loss.detach(),
            'actor/entropy': entropy.detach(),
            'actor/mean_action': mean_action.detach(),
            'temperature/value': temperature_value.detach(),
            'temperature/loss': temperature_loss.detach(),
        }

    def _critic_update(
            self, batch: Mapping[str, torch.Tensor]
            ) -> dict[str, torch.Tensor]:
        with torch.autocast(
                device_type=self.device.type, dtype=torch.float16,
                enabled=self.config.use_amp):
            with torch.no_grad():
                next_actions, info = self.actor(
                    observations=batch['next_observation'], training=False)
                next_actions = next_actions.clone()
                next_log_prob = info['log_prob'].clone()
                entropy_term = self.temperature() * next_log_prob
                observations_all = torch.cat(
                    [batch['observation'], batch['next_observation']], dim=0)
                actions_all = torch.cat(
                    [batch['action'], next_actions], dim=0)
                target_q_all, target_info = self.target_critic(
                    observations=observations_all, actions=actions_all,
                    training=True)
                next_q = target_q_all.chunk(2, dim=1)[1]
                next_log_prob_q = target_info['log_prob'].chunk(
                    2, dim=1)[1]
                selected_log_prob = _select_min_q_log_probs(
                    next_q, next_log_prob_q)
                target_probability = categorical_td_target(
                    selected_log_prob, batch['reward'],
                    batch['terminated'], entropy_term,
                    batch['bootstrap_discount'],
                    self.config.critic_num_bins,
                    self.config.critic_min_v,
                    self.config.critic_max_v)
                max_entropy_bonus = entropy_term.max()

            _, prediction_info = self.critic(
                observations=observations_all, actions=actions_all,
                training=True)
            prediction_log_prob = prediction_info['log_prob'].chunk(
                2, dim=1)[0]
            critic_loss = -(
                target_probability.unsqueeze(0)
                * prediction_log_prob).sum(dim=-1).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.config.use_amp:
            self.grad_scaler.scale(critic_loss).backward()
            self.grad_scaler.step(self.critic_optimizer)
            self.grad_scaler.update()
        else:
            critic_loss.backward()
            self.critic_optimizer.step()
        self.critic_scheduler.step()
        self._normalize_critic()
        self._update_target_critic()
        return {
            'critic/loss': critic_loss.detach(),
            'critic/max_entropy_bonus': max_entropy_bonus.detach(),
        }

    def update(self) -> dict[str, float]:
        if not self.can_update():
            raise RuntimeError('FlashSAC replay has not reached warmup size')
        batch = self.replay.sample()
        batch = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()}
        if self.reward_normalizer is not None:
            batch['reward'] = self.reward_normalizer.normalize(batch['reward'])
        metrics: dict[str, torch.Tensor] = {}
        if self.update_step % self.config.actor_update_period == 0:
            metrics.update(self._actor_update(batch))
        metrics.update(self._critic_update(batch))
        self.update_step += 1
        metrics.update({
            'updates': torch.tensor(float(self.update_step)),
            'replay/size': torch.tensor(float(len(self.replay))),
            'learning_rate/critic': torch.tensor(
                self.critic_optimizer.param_groups[0]['lr']),
        })
        return {
            key: float(value.detach().float().cpu().item())
            for key, value in metrics.items()}

    def reset_after_env_reset(self) -> None:
        """Clear per-episode state while retaining learned statistics."""
        if self.reward_normalizer is not None:
            self.reward_normalizer.reset_episode_accumulator(self.n_envs)
        self.noise_repeat_count.zero_()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            'format': CHECKPOINT_FORMAT,
            'checkpoint_kind': 'inference',
            'algorithm': 'flashsac',
            'official_repository': OFFICIAL_REPOSITORY,
            'official_commit': OFFICIAL_COMMIT,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'n_envs': self.n_envs,
            'config': asdict(self.config),
            'actor': self._actor_raw.state_dict(),
            'critic': self._critic_raw.state_dict(),
            'target_critic': self._target_critic_raw.state_dict(),
            'temperature': self.temperature.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'temperature_optimizer': self.temperature_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'temperature_scheduler': self.temperature_scheduler.state_dict(),
            'grad_scaler': self.grad_scaler.state_dict(),
            'update_step': self.update_step,
            'interaction_step': self.interaction_step,
            'target_entropy': self.target_entropy,
            'cached_noise': self.cached_noise.detach().cpu(),
            'noise_repeat_length': self.noise_repeat_length.detach().cpu(),
            'noise_repeat_count': self.noise_repeat_count.detach().cpu(),
            'reward_normalizer': (
                self.reward_normalizer.state_dict()
                if self.reward_normalizer is not None else None),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(self.checkpoint_state(), temporary)
        temporary.replace(path)

    def load_checkpoint(
            self, path: str | Path, load_optimizers: bool = True) -> None:
        state = torch.load(
            path, map_location=self.device, weights_only=False)
        self.load_checkpoint_state(state, load_optimizers=load_optimizers)

    def load_checkpoint_state(
            self, state: Mapping[str, Any],
            load_optimizers: bool = True) -> None:
        if state.get('format') != CHECKPOINT_FORMAT:
            raise ValueError('unsupported FlashSAC checkpoint format')
        if state.get('official_commit') != OFFICIAL_COMMIT:
            raise ValueError('FlashSAC source commit mismatch')
        for key, expected in (
                ('obs_dim', self.obs_dim), ('action_dim', self.action_dim),
                ('n_envs', self.n_envs)):
            if int(state[key]) != expected:
                raise ValueError(f'FlashSAC checkpoint {key} mismatch')
        stored_config = FlashSACConfig.from_mapping(state['config'])
        if asdict(stored_config) != asdict(self.config):
            raise ValueError('FlashSAC checkpoint config mismatch')
        self._actor_raw.load_state_dict(state['actor'])
        self._critic_raw.load_state_dict(state['critic'])
        self._target_critic_raw.load_state_dict(state['target_critic'])
        self.temperature.load_state_dict(state['temperature'])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(state['actor_optimizer'])
            self.critic_optimizer.load_state_dict(state['critic_optimizer'])
            self.temperature_optimizer.load_state_dict(
                state['temperature_optimizer'])
            self.actor_scheduler.load_state_dict(state['actor_scheduler'])
            self.critic_scheduler.load_state_dict(state['critic_scheduler'])
            self.temperature_scheduler.load_state_dict(
                state['temperature_scheduler'])
            self.grad_scaler.load_state_dict(state['grad_scaler'])
        self.update_step = int(state['update_step'])
        self.interaction_step = int(state['interaction_step'])
        if not math.isclose(
                float(state['target_entropy']), self.target_entropy):
            raise ValueError('FlashSAC target entropy mismatch')
        self.cached_noise.copy_(
            torch.as_tensor(state['cached_noise']).to(self.cached_noise))
        self.noise_repeat_length.copy_(
            torch.as_tensor(state['noise_repeat_length']).to(
                self.noise_repeat_length))
        self.noise_repeat_count.copy_(
            torch.as_tensor(state['noise_repeat_count']).to(
                self.noise_repeat_count))
        reward_state = state.get('reward_normalizer')
        if self.reward_normalizer is None:
            if reward_state is not None:
                raise ValueError(
                    'checkpoint has reward normalization but config does not')
        elif reward_state is None:
            raise ValueError('checkpoint lacks reward normalizer state')
        else:
            self.reward_normalizer.load_state_dict(reward_state)

    def save_replay(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(self.replay.state_dict(), temporary)
        temporary.replace(path)

    def load_replay(self, path: str | Path) -> None:
        state = torch.load(
            path, map_location='cpu', weights_only=False)
        self.replay.load_state_dict(state)
