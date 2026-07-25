"""Set-conditioned actor/critic for the one-step seed-selection policy."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


SEED_ENSEMBLE_FORMAT = 'candidate-seed-policy-ensemble-v1'
SEED_ENSEMBLE_AGGREGATION = 'mean-log-probability-v1'


def _orthogonal_init(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < 1:
        raise ValueError(f'{name} must be positive')
    return value


@dataclass(frozen=True)
class SeedPolicyConfig:
    """Architecture needed to reconstruct a seed-selection policy."""

    feature_dim: int
    hidden_dim: int = 256
    encoder_type: str = 'mean'
    heads: int = 4
    layers: int = 1
    ff_mult: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'feature_dim', _positive_int('feature_dim', self.feature_dim))
        object.__setattr__(
            self, 'hidden_dim', _positive_int('hidden_dim', self.hidden_dim))
        object.__setattr__(self, 'heads', _positive_int('heads', self.heads))
        object.__setattr__(self, 'layers', _positive_int('layers', self.layers))
        object.__setattr__(self, 'ff_mult', _positive_int('ff_mult', self.ff_mult))
        if self.encoder_type not in ('mean', 'attention'):
            raise ValueError(
                "encoder_type must be either 'mean' or 'attention'")
        if self.encoder_type == 'attention' and self.hidden_dim % self.heads != 0:
            raise ValueError(
                'hidden_dim must be divisible by heads for attention')

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _checkpoint_model_state(checkpoint: Mapping[str, Any]
                            ) -> Mapping[str, Any] | None:
    for key in ('seed_policy', 'model'):
        state = checkpoint.get(key)
        if state is not None:
            if not isinstance(state, Mapping):
                raise ValueError(f'checkpoint {key!r} must be a state dictionary')
            return state
    if 'encoder.0.weight' in checkpoint:
        return checkpoint
    return None


def _state_dimensions(state: Mapping[str, Any] | None
                      ) -> tuple[int, int] | None:
    if state is None:
        return None
    first_weight = state.get('encoder.0.weight')
    if first_weight is None:
        raise ValueError('seed policy state has no encoder.0.weight')
    if not torch.is_tensor(first_weight) or first_weight.ndim != 2:
        raise ValueError('encoder.0.weight must be a rank-2 tensor')
    hidden_dim, feature_dim = first_weight.shape
    return int(feature_dim), int(hidden_dim)


def infer_seed_policy_config(checkpoint: Mapping[str, Any]) -> SeedPolicyConfig:
    """Infer a policy architecture from a current or legacy checkpoint.

    Current checkpoints carry a complete ``seed_architecture`` dictionary.
    Older checkpoints are unambiguously treated as mean encoders, with missing
    dimensions recovered from the first encoder weight.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError('checkpoint must be a mapping')
    model_state = _checkpoint_model_state(checkpoint)
    state_dims = _state_dimensions(model_state)
    saved_architecture = checkpoint.get('seed_architecture')
    if saved_architecture is not None:
        if not isinstance(saved_architecture, Mapping):
            raise ValueError('seed_architecture must be a mapping')
        expected_keys = {
            'feature_dim', 'hidden_dim', 'encoder_type', 'heads', 'layers',
            'ff_mult',
        }
        actual_keys = set(saved_architecture)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                'seed_architecture has invalid keys: '
                f'missing={missing}, extra={extra}')
        config = SeedPolicyConfig(**dict(saved_architecture))
    else:
        if (model_state is not None
                and any(str(key).startswith('attention_layers.')
                        for key in model_state)):
            raise ValueError(
                'attention seed policy state requires seed_architecture metadata')
        feature_dim = checkpoint.get('feature_dim')
        hidden_dim = checkpoint.get('hidden_dim')
        if feature_dim is None:
            if state_dims is None:
                raise ValueError(
                    'legacy checkpoint has neither feature_dim nor model state')
            feature_dim = state_dims[0]
        if hidden_dim is None:
            if state_dims is None:
                hidden_dim = 256
            else:
                hidden_dim = state_dims[1]
        config = SeedPolicyConfig(
            feature_dim=feature_dim, hidden_dim=hidden_dim, encoder_type='mean')

    if 'feature_dim' in checkpoint:
        declared_feature_dim = _positive_int(
            'checkpoint feature_dim', checkpoint['feature_dim'])
        if declared_feature_dim != config.feature_dim:
            raise ValueError(
                'checkpoint feature_dim disagrees with seed_architecture')
    if 'hidden_dim' in checkpoint:
        declared_hidden_dim = _positive_int(
            'checkpoint hidden_dim', checkpoint['hidden_dim'])
        if declared_hidden_dim != config.hidden_dim:
            raise ValueError(
                'checkpoint hidden_dim disagrees with seed_architecture')
    if state_dims is not None:
        if state_dims != (config.feature_dim, config.hidden_dim):
            raise ValueError(
                'seed model dimensions disagree with its checkpoint metadata')
    return config


def seed_policy_ensemble_states(
    checkpoint: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]] | None:
    """Validate and return optional seed-policy ensemble checkpoint data."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError('checkpoint must be a mapping')
    has_states = 'seed_policy_ensemble' in checkpoint
    has_metadata = 'seed_ensemble' in checkpoint
    if has_states != has_metadata:
        raise ValueError(
            'seed_policy_ensemble and seed_ensemble metadata must appear together')
    if not has_states:
        return None

    states = checkpoint['seed_policy_ensemble']
    if not isinstance(states, (list, tuple)) or not states:
        raise ValueError('seed_policy_ensemble must be a non-empty list or tuple')
    metadata = checkpoint['seed_ensemble']
    if not isinstance(metadata, Mapping):
        raise ValueError('seed_ensemble must be a mapping')
    expected_keys = {'format', 'aggregation', 'size'}
    actual_keys = set(metadata)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            'seed_ensemble has invalid keys: '
            f'missing={missing}, extra={extra}')
    if metadata['format'] != SEED_ENSEMBLE_FORMAT:
        raise ValueError(
            f'seed_ensemble format must be {SEED_ENSEMBLE_FORMAT!r}')
    if metadata['aggregation'] != SEED_ENSEMBLE_AGGREGATION:
        raise ValueError(
            'seed_ensemble aggregation must be '
            f'{SEED_ENSEMBLE_AGGREGATION!r}')
    size = _positive_int('seed_ensemble size', metadata['size'])
    if size != len(states):
        raise ValueError(
            'seed_ensemble size disagrees with seed_policy_ensemble length')

    policy_config = infer_seed_policy_config(checkpoint)
    expected_dimensions = (
        policy_config.feature_dim, policy_config.hidden_dim)
    validated_states = []
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise ValueError(
                f'seed_policy_ensemble[{index}] must be a state dictionary')
        try:
            dimensions = _state_dimensions(state)
        except ValueError as exc:
            raise ValueError(
                f'invalid seed_policy_ensemble[{index}]: {exc}') from exc
        if dimensions != expected_dimensions:
            raise ValueError(
                f'seed_policy_ensemble[{index}] dimensions disagree with '
                'seed_architecture')
        validated_states.append(state)
    return tuple(validated_states), metadata


class _MaskedSelfAttentionBlock(nn.Module):
    """Pre-LN self-attention block for an unordered padded candidate set."""

    def __init__(self, hidden_dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            _orthogonal_init(
                nn.Linear(hidden_dim, hidden_dim * ff_mult), 2 ** 0.5),
            nn.GELU(),
            _orthogonal_init(
                nn.Linear(hidden_dim * ff_mult, hidden_dim), 1.0),
        )

    def forward(self, encoded: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        query_valid = valid.unsqueeze(-1)
        encoded = torch.where(
            query_valid, encoded, torch.zeros_like(encoded))
        normalized = self.attention_norm(encoded)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            key_padding_mask=~valid, need_weights=False)
        encoded = torch.where(
            query_valid, encoded + attended, torch.zeros_like(encoded))
        residual = self.feedforward(self.feedforward_norm(encoded))
        return torch.where(
            query_valid, encoded + residual, torch.zeros_like(encoded))


class CandidateSeedActorCritic(nn.Module):
    """Permutation-equivariant policy over a variable valid candidate set.

    Each candidate is encoded independently. A masked mean summarizes the
    whole set, after which actor and feasibility heads see both the candidate
    embedding and set context. The value head only sees set context and is
    therefore a valid action-independent PPO baseline.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        encoder_type: str = 'mean',
        heads: int = 4,
        layers: int = 1,
        ff_mult: int = 2,
    ):
        super().__init__()
        config = SeedPolicyConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            heads=heads,
            layers=layers,
            ff_mult=ff_mult,
        )
        self.feature_dim = config.feature_dim
        self.hidden_dim = config.hidden_dim
        self.encoder_type = config.encoder_type
        self.heads = config.heads
        self.layers = config.layers
        self.ff_mult = config.ff_mult
        self.register_buffer('feature_mean', torch.zeros(self.feature_dim))
        self.register_buffer('feature_std', torch.ones(self.feature_dim))
        self.encoder = nn.Sequential(
            _orthogonal_init(
                nn.Linear(self.feature_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
            _orthogonal_init(
                nn.Linear(self.hidden_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
        )
        self.attention_layers = nn.ModuleList()
        if self.encoder_type == 'attention':
            self.attention_layers.extend(
                _MaskedSelfAttentionBlock(
                    self.hidden_dim, self.heads, self.ff_mult)
                for _ in range(self.layers)
            )
        self.actor = nn.Sequential(
            _orthogonal_init(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(self.hidden_dim, 1), 0.01),
        )
        self.feasibility = nn.Sequential(
            _orthogonal_init(
                nn.Linear(2 * self.hidden_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(self.hidden_dim, 1), 1.0),
        )
        self.value = nn.Sequential(
            _orthogonal_init(
                nn.Linear(self.hidden_dim, self.hidden_dim), 2 ** 0.5),
            nn.ReLU(),
            _orthogonal_init(nn.Linear(self.hidden_dim, 1), 1.0),
        )

    @property
    def architecture(self) -> dict[str, int | str]:
        """Serializable architecture metadata for checkpoints."""
        return SeedPolicyConfig(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            encoder_type=self.encoder_type,
            heads=self.heads,
            layers=self.layers,
            ff_mult=self.ff_mult,
        ).to_dict()

    @torch.no_grad()
    def set_feature_normalization(
        self, mean: torch.Tensor, std: torch.Tensor,
    ) -> None:
        mean = torch.as_tensor(mean, device=self.feature_mean.device,
                               dtype=self.feature_mean.dtype)
        std = torch.as_tensor(std, device=self.feature_std.device,
                              dtype=self.feature_std.dtype)
        if mean.shape != (self.feature_dim,) or std.shape != (self.feature_dim,):
            raise ValueError(
                f'feature normalization must have shape ({self.feature_dim},)')
        if (not bool(torch.isfinite(mean).all().item())
                or not bool(torch.isfinite(std).all().item())
                or bool((std <= 0).any().item())):
            raise ValueError('feature normalization must be finite with std > 0')
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        # Early unified checkpoints predate feature normalization. Identity
        # buffers retain their exact behavior while allowing legacy analysis.
        state = dict(state_dict)
        state.setdefault('feature_mean', self.feature_mean)
        state.setdefault('feature_std', self.feature_std)
        result = super().load_state_dict(state, strict=strict, assign=assign)
        if (not bool(torch.isfinite(self.feature_mean).all().item())
                or not bool(torch.isfinite(self.feature_std).all().item())
                or bool((self.feature_std <= 0).any().item())):
            raise ValueError(
                'loaded feature normalization must be finite with std > 0')
        return result

    def _check_inputs(self, features: torch.Tensor,
                      valid: torch.Tensor) -> None:
        if not torch.is_tensor(features) or not torch.is_tensor(valid):
            raise ValueError('features and valid must be tensors')
        if features.ndim != 3:
            raise ValueError('features must have shape (B,K,D)')
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError('features must contain at least one task and candidate')
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f'expected feature dim {self.feature_dim}, got {features.shape[-1]}')
        if not torch.is_floating_point(features):
            raise ValueError('features must have a floating-point dtype')
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('valid must be bool with shape (B,K)')
        if valid.device != features.device:
            raise ValueError('features and valid must be on the same device')
        if features.device != self.feature_mean.device:
            raise ValueError('features and seed policy must be on the same device')
        if features.dtype != self.feature_mean.dtype:
            raise ValueError('features and seed policy must have the same dtype')
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError('every task must have at least one valid candidate')
        valid_features = features[valid]
        if not bool(torch.isfinite(valid_features).all().item()):
            raise ValueError('valid candidate features must be finite')

    def distribution_and_values(
        self, features: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        encoded, context = self.encode_candidates(features, valid)
        context_per_candidate = context.unsqueeze(1).expand_as(encoded)
        joint = torch.cat([encoded, context_per_candidate], dim=-1)
        logits = self.actor(joint).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        feasibility = self.feasibility(joint).squeeze(-1)
        state_value = self.value(context).squeeze(-1)
        return Categorical(logits=logits), state_value, feasibility

    def encode_candidates(
        self, features: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode candidates and return their masked set context.

        Returns an ``(B,K,H)`` candidate tensor and an ``(B,H)`` context.
        This method only exposes the existing encoder computation; it adds no
        parameters and preserves the policy's legacy numerical path.
        """
        self._check_inputs(features, valid)
        # Invalid proposal slots are semantically padding.  Zero them before
        # the encoder as well as masking their logits: IEEE NaN multiplied by
        # a zero mask remains NaN and would otherwise poison the set context.
        normalized = (features - self.feature_mean) / self.feature_std
        safe_features = torch.where(
            valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        encoded = self.encoder(safe_features)
        for attention_layer in self.attention_layers:
            encoded = attention_layer(encoded, valid)
        mask = valid.unsqueeze(-1).to(encoded.dtype)
        context = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return encoded, context

    def selected_representation(
        self, features: torch.Tensor, valid: torch.Tensor,
        index: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected candidate/context representation ``(B,2H)``."""
        encoded, context = self.encode_candidates(features, valid)
        if not torch.is_tensor(index):
            raise ValueError('index must be a tensor')
        if index.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError('index must have an integer dtype')
        index = index.to(device=features.device, dtype=torch.long)
        if index.shape != features.shape[:1]:
            raise ValueError(f'index must have shape ({features.shape[0]},)')
        if bool(((index < 0) | (index >= features.shape[1])).any().item()):
            raise ValueError('index is out of range')
        row = torch.arange(features.shape[0], device=features.device)
        if not bool(valid[row, index].all().item()):
            raise ValueError('index selects an invalid candidate')
        return torch.cat([encoded[row, index], context], dim=-1)

    def get_action_and_value(
        self, features: torch.Tensor, valid: torch.Tensor,
        action: torch.Tensor | None = None, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value, feasibility = self.distribution_and_values(features, valid)
        if action is None:
            action = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
        elif not torch.is_tensor(action):
            raise ValueError('action must be a tensor')
        elif action.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError('action must have an integer dtype')
        action = action.to(device=features.device, dtype=torch.long)
        if action.shape != features.shape[:1]:
            raise ValueError(f'action must have shape ({features.shape[0]},)')
        if bool(((action < 0) | (action >= features.shape[1])).any().item()):
            raise ValueError('action index is out of range')
        row = torch.arange(features.shape[0], device=features.device)
        if not bool(valid[row, action].all().item()):
            raise ValueError('action selects an invalid candidate')
        selected_feasibility = feasibility[row, action]
        return action, dist.log_prob(action), dist.entropy(), value, selected_feasibility

    @torch.no_grad()
    def select(self, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Deterministic deployment selection."""
        action, _, _, _, _ = self.get_action_and_value(
            features, valid, deterministic=True)
        return action


class CandidateSeedPolicyEnsemble(nn.Module):
    """Inference-only geometric ensemble of compatible seed policies."""

    aggregation = SEED_ENSEMBLE_AGGREGATION

    def __init__(self, members: list[CandidateSeedActorCritic]
                 | tuple[CandidateSeedActorCritic, ...]):
        super().__init__()
        if not isinstance(members, (list, tuple)) or not members:
            raise ValueError('members must be a non-empty list or tuple')
        if not all(isinstance(member, CandidateSeedActorCritic)
                   for member in members):
            raise ValueError('every ensemble member must be a seed policy')
        architecture = members[0].architecture
        for index, member in enumerate(members[1:], start=1):
            if member.architecture != architecture:
                raise ValueError(
                    f'ensemble member {index} architecture does not match '
                    'member 0')
        self.members = nn.ModuleList(members)
        self.feature_dim = members[0].feature_dim
        self.hidden_dim = members[0].hidden_dim
        self.encoder_type = members[0].encoder_type
        self.heads = members[0].heads
        self.layers = members[0].layers
        self.ff_mult = members[0].ff_mult

    @property
    def architecture(self) -> dict[str, int | str]:
        return SeedPolicyConfig(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            encoder_type=self.encoder_type,
            heads=self.heads,
            layers=self.layers,
            ff_mult=self.ff_mult,
        ).to_dict()

    @property
    def size(self) -> int:
        return len(self.members)

    def distribution_and_values(
        self, features: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        outputs = [
            member.distribution_and_values(features, valid)
            for member in self.members
        ]
        # Divide before summing so padded logits at finfo.min remain finite.
        # A direct float32 mean over two such logits overflows during reduction.
        divisor = float(len(outputs))
        mean_logits = torch.stack([
            distribution.logits / divisor
            for distribution, _, _ in outputs]).sum(dim=0)
        mean_value = torch.stack([
            value / divisor for _, value, _ in outputs]).sum(dim=0)
        mean_feasibility = torch.stack([
            feasibility / divisor
            for _, _, feasibility in outputs]).sum(dim=0)
        return Categorical(logits=mean_logits), mean_value, mean_feasibility
