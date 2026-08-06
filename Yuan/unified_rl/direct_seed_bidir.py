"""Alternating contextual-RL seed generation and continuous control.

The seed phase is deliberately *not* a diffusion or candidate-ranking stage.
For every sampled task, a continuous actor emits exactly one raw joint vector:

    task -> one q_raw -> DIRECT / one REFINE / safe FALLBACK -> one C rollout

The complete downstream controller progress is the Monte-Carlo target of the
direct-seed twin critics.  The controller phase then freezes the updated seed
actor and adapts PPO on the resulting reset distribution.  This implements the
backward/forward alternation without adding deployment-time model rollouts or
multiple-seed selection.

Only task geometry and the explicitly named ``q0_pilot`` safety fallback are
read from the historical cache.  All other candidate joints are discarded at
the input boundary.  Moreover, fallback configurations that fail the exact
deployment gate are filtered once, before training, and their task ids and
geometry fingerprints are recorded.

``Yuan.unified_rl.direct_seed_rl`` is imported lazily.  This keeps ``--dry-run``
usable while the contextual-RL implementation is being developed, while a
real run enforces a small, explicit API protocol.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import yaml

from Yuan.RL_controller.algorithms.ppo import RewardScaler, train as ppo_train
from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedSelection,
)
from Yuan.unified_rl.checkpoint import (
    adapt_controller_optimizer_observation_state,
    atomic_torch_save,
    build_env_from_run,
    load_controller_agent,
    load_controller_state_dict,
    load_run_config,
    ppo_config_from_run,
    resolve_controller_dir,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    rollout_seed_selection,
)
from Yuan.unified_rl.direct_seed_model import direct_seed_task
from Yuan.unified_rl.direct_seed_projection import (
    DirectSeedProjectionConfig,
    ROUTE_DIRECT,
    ROUTE_FALLBACK,
    ROUTE_INVALID,
    ROUTE_NAMES,
    ROUTE_REFINED,
    route_generated_seed,
    strict_seed_validity,
)
from Yuan.unified_rl.reproducibility import (
    global_rng_state,
    restore_global_rng,
    seed_global_rng,
)
from Yuan.unified_rl.provenance import (
    file_fingerprint,
    state_dict_fingerprint,
)


_DIRECT_SEED_RL_MODULE = 'Yuan.unified_rl.direct_seed_rl'
_DIRECT_SEED_RL_SYMBOLS = (
    'DirectSeedActor',
    'DirectSeedActorConfig',
    'DirectSeedCriticConfig',
    'DirectSeedEliteMemory',
    'DirectSeedMacroReplay',
    'DirectSeedPairedArchive',
    'DirectSeedRLBatch',
    'DirectSeedRLConfig',
    'TwinMacroQ',
    'direct_seed_rl_checkpoint',
    'load_direct_seed_rl_checkpoint',
    'update_direct_seed_projection',
    'update_direct_seed_precision',
    'update_direct_seed_rl',
)


@dataclass(frozen=True)
class DirectSeedRLAPI:
    """Validated delayed-import view of the contextual-RL module."""

    module: ModuleType
    DirectSeedActor: type
    DirectSeedActorConfig: type
    DirectSeedCriticConfig: type
    DirectSeedEliteMemory: type
    DirectSeedMacroReplay: type
    DirectSeedPairedArchive: type
    DirectSeedRLBatch: type
    DirectSeedRLConfig: type
    TwinMacroQ: type
    direct_seed_rl_checkpoint: Any
    load_direct_seed_rl_checkpoint: Any
    update_direct_seed_projection: Any
    update_direct_seed_precision: Any
    update_direct_seed_rl: Any


def _load_direct_seed_rl_api(*, required: bool) -> DirectSeedRLAPI | None:
    try:
        module = importlib.import_module(_DIRECT_SEED_RL_MODULE)
    except ModuleNotFoundError as error:
        if error.name != _DIRECT_SEED_RL_MODULE or required:
            raise RuntimeError(
                f'cannot import required direct-seed RL API '
                f'{_DIRECT_SEED_RL_MODULE!r}') from error
        return None

    missing = [
        name for name in _DIRECT_SEED_RL_SYMBOLS
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            f'{_DIRECT_SEED_RL_MODULE} is missing protocol symbols: {missing}')
    for name in (
            'direct_seed_rl_checkpoint',
            'load_direct_seed_rl_checkpoint',
            'update_direct_seed_projection',
            'update_direct_seed_precision',
            'update_direct_seed_rl'):
        if not callable(getattr(module, name)):
            raise RuntimeError(
                f'{_DIRECT_SEED_RL_MODULE}.{name} must be callable')
    actor_class = module.DirectSeedActor
    for name in ('sample', 'mean_q'):
        if not callable(getattr(actor_class, name, None)):
            raise RuntimeError(
                f'DirectSeedActor must implement callable {name}()')
    elite_memory_class = module.DirectSeedEliteMemory
    for name in ('update', 'sample', 'clear', 'state_dict',
                 'load_state_dict'):
        if not callable(getattr(elite_memory_class, name, None)):
            raise RuntimeError(
                'DirectSeedEliteMemory must implement callable '
                f'{name}()')
    paired_archive_class = module.DirectSeedPairedArchive
    for name in (
            'update', 'sample_targets', 'target_stats', 'clear',
            'state_dict', 'load_state_dict'):
        if not callable(getattr(paired_archive_class, name, None)):
            raise RuntimeError(
                'DirectSeedPairedArchive must implement callable '
                f'{name}()')
    if dataclasses.is_dataclass(module.DirectSeedRLBatch):
        fields = {
            field.name for field in dataclasses.fields(
                module.DirectSeedRLBatch)
        }
        expected = {
            'task', 'q_raw', 'q_projected',
            'fallback_q', 'progress_m', 'route',
        }
        if fields != expected:
            raise RuntimeError(
                'DirectSeedRLBatch fields must be exactly '
                f'{sorted(expected)}, got {sorted(fields)}')
    return DirectSeedRLAPI(
        module=module,
        **{
            name: getattr(module, name)
            for name in _DIRECT_SEED_RL_SYMBOLS
        },
    )


@dataclass(frozen=True)
class DirectTaskBatch:
    """Task geometry plus one fixed, externally supplied safety fallback."""

    p0: torch.Tensor
    line_dir: torch.Tensor
    n_target: torch.Tensor
    fallback_q: torch.Tensor
    task_indices: torch.Tensor

    def __post_init__(self) -> None:
        if self.p0.ndim != 2 or self.p0.shape[-1] != 3:
            raise ValueError('p0 must have shape (B, 3)')
        batch_size = self.p0.shape[0]
        shapes = {
            'line_dir': (batch_size, 3),
            'n_target': (batch_size, 3),
            'fallback_q': (batch_size, 7),
            'task_indices': (batch_size,),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(
                    f'{name} must have shape {shape}, got {tuple(value.shape)}')
            if value.device != self.p0.device:
                raise ValueError('direct task tensors must share one device')
        floating = (
            self.p0, self.line_dir, self.n_target, self.fallback_q)
        if not all(value.dtype == self.p0.dtype for value in floating):
            raise ValueError('direct task floating tensors must share dtype')
        if not all(bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError('direct task tensors must be finite')

    @property
    def n_tasks(self) -> int:
        return int(self.p0.shape[0])

    @property
    def task(self) -> torch.Tensor:
        return direct_seed_task(self.p0, self.line_dir, self.n_target)

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> 'DirectTaskBatch':
        target_dtype = self.p0.dtype if dtype is None else dtype
        return DirectTaskBatch(
            p0=self.p0.to(device=device, dtype=target_dtype),
            line_dir=self.line_dir.to(device=device, dtype=target_dtype),
            n_target=self.n_target.to(device=device, dtype=target_dtype),
            fallback_q=self.fallback_q.to(
                device=device, dtype=target_dtype),
            task_indices=self.task_indices.to(device=device),
        )

    def index_select(self, index: torch.Tensor) -> 'DirectTaskBatch':
        index = index.to(device=self.p0.device, dtype=torch.long)
        return DirectTaskBatch(
            p0=self.p0.index_select(0, index),
            line_dir=self.line_dir.index_select(0, index),
            n_target=self.n_target.index_select(0, index),
            fallback_q=self.fallback_q.index_select(0, index),
            task_indices=self.task_indices.index_select(0, index),
        )

    def specs(self, q0: torch.Tensor) -> dict[str, torch.Tensor]:
        return SeedSelection(
            q0=q0,
            p0=self.p0,
            line_dir=self.line_dir,
            n_target=self.n_target,
        ).specs()


class DirectTaskDataset:
    """CPU dataset that cannot expose historical IK candidates to the actor."""

    def __init__(
        self,
        batch: DirectTaskBatch,
        fingerprints: tuple[str, ...],
    ):
        if batch.p0.device.type != 'cpu':
            raise ValueError('direct task dataset must remain on CPU')
        if len(fingerprints) != batch.n_tasks:
            raise ValueError('fingerprints must align with direct tasks')
        if batch.task_indices.unique().numel() != batch.n_tasks:
            raise ValueError('task_indices must be unique')
        self.batch = batch
        self.fingerprints = tuple(fingerprints)

    def __len__(self) -> int:
        return self.batch.n_tasks

    @classmethod
    def from_candidate_cache(
        cls,
        cache: CachedSeedCandidateDataset,
    ) -> 'DirectTaskDataset':
        if cache.fallback_index is None:
            raise ValueError(
                'direct-seed RL requires an explicit q0_pilot fallback')
        # This is the sole point where the legacy candidate-shaped object is
        # touched.  Only geometry and q0_pilot cross the boundary.
        fallback_q = cache.batch.q0[:, cache.fallback_index].clone()
        batch = DirectTaskBatch(
            p0=cache.batch.p0.clone(),
            line_dir=cache.batch.line_dir.clone(),
            n_target=cache.batch.n_target.clone(),
            fallback_q=fallback_q,
            task_indices=cache.task_indices.clone(),
        )
        return cls(batch, cache.task_fingerprints)

    def index_select(self, index: torch.Tensor) -> 'DirectTaskDataset':
        index = index.to(device='cpu', dtype=torch.long)
        return DirectTaskDataset(
            self.batch.index_select(index),
            tuple(self.fingerprints[int(i)] for i in index.tolist()),
        )

    def sample(
        self,
        n: int,
        generator: torch.Generator | None = None,
    ) -> DirectTaskBatch:
        if n < 1:
            raise ValueError('sample size must be positive')
        index = torch.randint(len(self), (n,), generator=generator)
        return self.batch.index_select(index)

    def task_normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        task = self.batch.task
        mean = task.mean(dim=0)
        std = task.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std


class DirectTaskCycleSampler:
    """Checkpointable CPU sampler visiting every task once per epoch."""

    _FORMAT = 'direct-task-cycle-sampler-v1'

    def __init__(self, n_tasks: int, *, seed: int = 0):
        if isinstance(n_tasks, bool) or not isinstance(n_tasks, int):
            raise TypeError('n_tasks must be an integer')
        if n_tasks < 1:
            raise ValueError('n_tasks must be positive')
        self.n_tasks = int(n_tasks)
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(int(seed))
        self.order = torch.randperm(
            self.n_tasks, generator=self.generator, device='cpu')
        self.cursor = 0
        self.epochs_started = 1
        self.total_sampled = 0

    def sample(self, n: int) -> torch.Tensor:
        """Return CPU dataset-row indices, crossing epoch boundaries safely."""
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError('sample size must be an integer')
        if n < 1:
            raise ValueError('sample size must be positive')
        parts = []
        remaining = n
        while remaining > 0:
            if self.cursor == self.n_tasks:
                self.order = torch.randperm(
                    self.n_tasks, generator=self.generator, device='cpu')
                self.cursor = 0
                self.epochs_started += 1
            available = self.n_tasks - self.cursor
            take = min(remaining, available)
            parts.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
        self.total_sampled += n
        return torch.cat(parts)

    def state_dict(self) -> dict[str, Any]:
        return {
            'format': self._FORMAT,
            'n_tasks': self.n_tasks,
            'order': self.order.clone(),
            'cursor': self.cursor,
            'epochs_started': self.epochs_started,
            'total_sampled': self.total_sampled,
            'generator_state': self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get('format') != self._FORMAT:
            raise ValueError('unsupported direct task cycle sampler state')
        if int(state.get('n_tasks', -1)) != self.n_tasks:
            raise ValueError(
                'cycle sampler n_tasks differs from the current dataset')
        order = state.get('order')
        if not torch.is_tensor(order):
            raise TypeError('cycle sampler order must be a tensor')
        order = order.detach().to(device='cpu', dtype=torch.long).clone()
        if order.shape != (self.n_tasks,):
            raise ValueError(
                'cycle sampler order must have shape '
                f'({self.n_tasks},)')
        if not torch.equal(
                order.sort().values, torch.arange(self.n_tasks)):
            raise ValueError(
                'cycle sampler order must be a permutation of dataset rows')
        cursor = int(state.get('cursor', -1))
        epochs_started = int(state.get('epochs_started', -1))
        total_sampled = int(state.get('total_sampled', -1))
        if not 0 <= cursor <= self.n_tasks:
            raise ValueError('cycle sampler cursor is out of range')
        if epochs_started < 1:
            raise ValueError('cycle sampler epochs_started must be positive')
        expected_total = (
            (epochs_started - 1) * self.n_tasks + cursor)
        if total_sampled != expected_total:
            raise ValueError(
                'cycle sampler counters are internally inconsistent')
        generator_state = state.get('generator_state')
        if not torch.is_tensor(generator_state):
            raise TypeError(
                'cycle sampler generator_state must be a tensor')
        generator = torch.Generator(device='cpu')
        try:
            generator.set_state(
                generator_state.detach().to(device='cpu').clone())
        except RuntimeError as error:
            raise ValueError(
                'invalid cycle sampler generator_state') from error
        self.generator = generator
        self.order = order
        self.cursor = cursor
        self.epochs_started = epochs_started
        self.total_sampled = total_sampled


@torch.no_grad()
def _filter_strict_fallbacks(
    dataset: DirectTaskDataset,
    env,
    config: DirectSeedProjectionConfig,
    *,
    batch_size: int,
) -> tuple[DirectTaskDataset, dict[str, Any]]:
    """Apply the deployment gate once and permanently remove unsafe tasks."""
    if batch_size < 1:
        raise ValueError('fallback validation batch size must be positive')
    valid_parts = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        rows = torch.arange(start, end)
        batch = dataset.batch.index_select(rows).to(
            env.device, dtype=env.kin.dtype)
        validity = strict_seed_validity(
            env.kin, env.collision, batch.fallback_q,
            batch.p0, batch.line_dir, batch.n_target, config)
        valid_parts.append(validity.valid.cpu())
    valid = torch.cat(valid_parts)
    kept = torch.nonzero(valid, as_tuple=False).flatten()
    excluded = torch.nonzero(~valid, as_tuple=False).flatten()
    if kept.numel() < 1:
        raise RuntimeError('strict fallback filter excluded every task')
    filtered = dataset.index_select(kept)

    def records(index: torch.Tensor) -> list[dict[str, Any]]:
        return [
            {
                'task_index': int(dataset.batch.task_indices[int(row)]),
                'fingerprint': dataset.fingerprints[int(row)],
            }
            for row in index.tolist()
        ]

    kept_records = records(kept)
    excluded_records = records(excluded)

    def fingerprint_sha256(values: list[dict[str, Any]]) -> str:
        # Newline-delimited hex digests are unambiguous and can be reproduced
        # without serializing the surrounding JSON representation.
        payload = '\n'.join(
            str(value['fingerprint']) for value in values)
        return hashlib.sha256(payload.encode('ascii')).hexdigest()

    manifest = {
        'format': 'direct-seed-safe-fallback-filter-v1',
        'source_task_count': len(dataset),
        'kept_task_count': int(kept.numel()),
        'excluded_task_count': int(excluded.numel()),
        'projection_config': dataclasses.asdict(config),
        'kept_fingerprint_list_sha256': (
            fingerprint_sha256(kept_records)),
        'excluded_fingerprint_list_sha256': (
            fingerprint_sha256(excluded_records)),
        'kept': kept_records,
        'excluded': excluded_records,
    }
    return filtered, manifest


class DirectSeedLineDistribution:
    """Controller reset distribution from one frozen deterministic seed actor."""

    def __init__(
        self,
        dataset: DirectTaskDataset,
        actor,
        kin,
        collision,
        config: DirectSeedProjectionConfig,
        *,
        fallback_probability: float,
        seed: int,
    ):
        if not 0.0 <= fallback_probability <= 1.0:
            raise ValueError('fallback_probability must be in [0, 1]')
        self.dataset = dataset
        self.actor = actor.eval()
        self.kin = kin
        self.collision = collision
        self.config = config
        self.fallback_probability = float(fallback_probability)
        self.cpu_generator = torch.Generator().manual_seed(int(seed))
        self.mode_generator = torch.Generator(
            device=kin.device).manual_seed(int(seed) + 1)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        cpu_generator = (
            self.cpu_generator if generator is None else generator)
        batch = self.dataset.sample(n, cpu_generator).to(
            self.kin.device, dtype=self.kin.dtype)
        q_raw = self.actor.mean_q(batch.task)
        routed = route_generated_seed(
            self.kin, self.collision, q_raw,
            batch.p0, batch.line_dir, batch.n_target,
            batch.fallback_q, self.config)
        if not bool(routed.valid.all()):
            bad = torch.nonzero(
                ~routed.valid, as_tuple=False).flatten().cpu().tolist()
            raise RuntimeError(
                'fixed safe-fallback invariant failed during controller '
                f'reset, rows={bad[:20]}')
        q0 = routed.q
        if self.fallback_probability > 0.0:
            use_fallback = (
                torch.rand(
                    n, device=self.kin.device,
                    generator=self.mode_generator)
                < self.fallback_probability)
            q0 = torch.where(
                use_fallback.unsqueeze(-1), batch.fallback_q, q0)
        return batch.specs(q0)


def _synchronise(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _optimizer_to(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> list[float]:
    """Apply the requested resume LR after optimizer state restoration."""
    previous = [
        float(group.get('lr', learning_rate))
        for group in optimizer.param_groups
    ]
    for group in optimizer.param_groups:
        group['lr'] = float(learning_rate)
    return previous


_PAIRED_COLLECTION_CONTRACT_FORMAT = (
    'direct-seed-paired-collection-contract-v1')


def _paired_collection_contract(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return the immutable settings defining one paired baseline archive."""
    return {
        'format': _PAIRED_COLLECTION_CONTRACT_FORMAT,
        'actor_action': 'deterministic-mean',
        'actor_frozen_during_collection': bool(
            args.freeze_seed_actor_during_collection),
        'deterministic_backward': bool(args.deterministic_backward),
        'task_sampling': str(args.task_sampling),
        'archive_write_policy': 'first-task-outcome-immutable',
        'run_seed': int(args.seed),
        'seed_tasks_per_update': int(args.seed_tasks_per_update),
        'backward_updates_per_round': int(
            args.backward_updates_per_round),
    }


def _actor_state_sha256(actor: torch.nn.Module) -> str:
    """Content identity of the exact deterministic baseline actor."""
    return state_dict_fingerprint(dict(actor.state_dict()))


def _paired_archive_task_count(paired_archive: Any) -> int:
    task_ids = getattr(paired_archive, 'task_ids', None)
    if (not torch.is_tensor(task_ids)
            or task_ids.ndim != 1
            or task_ids.numel() < 1):
        raise ValueError(
            'paired archive must expose a non-empty task_ids tensor')
    return int(task_ids.numel())


def _require_full_paired_archive(paired_archive: Any) -> None:
    """Reject post-training until every configured task has one baseline."""
    task_count = _paired_archive_task_count(paired_archive)
    outcome_count = len(paired_archive)
    if outcome_count != task_count:
        raise RuntimeError(
            'paired post-updates require full baseline coverage, got '
            f'{outcome_count}/{task_count} tasks')


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in '0123456789abcdef' for character in value)
    )


def _restore_paired_collection_provenance(
    saved: Mapping[str, Any],
    *,
    requested_contract: Mapping[str, Any],
    paired_archive: Any,
    current_actor_state_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Strictly restore provenance, while allowing old non-paired resumes.

    A legacy checkpoint with no paired archive starts a new archive under the
    current contract.  A checkpoint that *does* contain a paired archive must
    carry the new contract and baseline actor identity; silently guessing these
    fields would make heterogeneous first-observation labels possible.
    """
    if not isinstance(saved, Mapping):
        raise TypeError('resume checkpoint must be a mapping')
    if not isinstance(requested_contract, Mapping):
        raise TypeError('requested paired collection contract must be a mapping')
    if not _is_sha256(current_actor_state_sha256):
        raise ValueError('current baseline actor state SHA-256 is invalid')

    requested = copy.deepcopy(dict(requested_contract))
    saved_archive = saved.get('paired_archive')
    if saved_archive is None:
        # Backward compatibility for checkpoints that never collected paired
        # outcomes.  The current actor becomes the new baseline.
        return requested, current_actor_state_sha256

    saved_contract = saved.get('paired_collection_contract')
    saved_actor_sha256 = saved.get(
        'paired_baseline_actor_state_sha256')
    direct_seed = saved.get('direct_seed')
    direct_metadata = (
        direct_seed.get('metadata')
        if isinstance(direct_seed, Mapping) else None)
    metadata_contract = (
        direct_metadata.get('paired_collection_contract')
        if isinstance(direct_metadata, Mapping) else None)
    metadata_actor_sha256 = (
        direct_metadata.get('paired_baseline_actor_state_sha256')
        if isinstance(direct_metadata, Mapping) else None)
    if not isinstance(saved_contract, Mapping) \
            or not _is_sha256(saved_actor_sha256) \
            or not isinstance(metadata_contract, Mapping) \
            or not _is_sha256(metadata_actor_sha256):
        raise ValueError(
            'resume contains a legacy paired archive without a valid '
            'collection contract and baseline actor SHA-256; refusing to '
            'guess paired provenance')
    restored_contract = copy.deepcopy(dict(saved_contract))
    if (dict(metadata_contract) != restored_contract
            or metadata_actor_sha256 != saved_actor_sha256):
        raise ValueError(
            'resume paired collection provenance differs between the runner '
            'checkpoint and direct-seed metadata')
    if restored_contract != requested:
        raise ValueError(
            'resume paired collection contract differs from the current '
            'request')

    outcome_count = len(paired_archive)
    task_count = _paired_archive_task_count(paired_archive)
    if outcome_count == 0:
        # Controller-version invalidation clears the whole archive.  Starting a
        # new immutable collection therefore establishes a new baseline actor.
        return restored_contract, current_actor_state_sha256
    if (outcome_count < task_count
            and current_actor_state_sha256 != saved_actor_sha256):
        raise ValueError(
            'resume has a partial paired archive but the current actor differs '
            'from its recorded baseline actor')
    return restored_contract, saved_actor_sha256


def _load_paired_explorer_checkpoint(
    path: str | Path,
    api: DirectSeedRLAPI,
    *,
    task_ids: torch.Tensor,
    task: torch.Tensor,
    safe_task_fingerprint_list_sha256: str,
    projection_config: DirectSeedProjectionConfig,
    controller_state: Mapping[str, torch.Tensor],
) -> tuple[Any, dict[str, Any]]:
    """Load explorer elites only after strict task/gate/controller matching."""
    resolved = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(
        resolved, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError('paired explorer checkpoint must be a mapping')
    if checkpoint.get('format') != 'direct-seed-bidirectional-v1':
        raise ValueError('unsupported paired explorer checkpoint format')
    expected_task_ids = task_ids.detach().to(
        device='cpu', dtype=torch.int64)
    expected_task = task.detach().to(
        device='cpu', dtype=torch.float32)
    if expected_task.shape != (int(expected_task_ids.numel()), 9):
        raise ValueError(
            'current paired explorer task tensor must have shape (N, 9)')
    if checkpoint.get('kept_task_indices') != expected_task_ids.tolist():
        raise ValueError(
            'paired explorer task ids differ from the current safe dataset')
    if checkpoint.get(
            'safe_task_fingerprint_list_sha256'
    ) != safe_task_fingerprint_list_sha256:
        raise ValueError(
            'paired explorer task fingerprint differs from the current '
            'safe dataset')
    expected_projection = dataclasses.asdict(projection_config)
    if checkpoint.get('projection_config') != expected_projection:
        raise ValueError(
            'paired explorer projection config differs from the current gate')

    explorer_controller = checkpoint.get('controller')
    if not isinstance(explorer_controller, Mapping) \
            or not explorer_controller:
        raise ValueError(
            'paired explorer checkpoint has no controller state')
    if any(not isinstance(name, str) or not torch.is_tensor(value)
           for name, value in explorer_controller.items()):
        raise ValueError(
            'paired explorer controller state must contain only tensors')
    if (not isinstance(controller_state, Mapping)
            or not controller_state
            or any(not isinstance(name, str) or not torch.is_tensor(value)
                   for name, value in controller_state.items())):
        raise ValueError('current controller state must contain only tensors')
    explorer_controller_sha256 = state_dict_fingerprint(
        dict(explorer_controller))
    current_controller_sha256 = state_dict_fingerprint(
        dict(controller_state))
    if explorer_controller_sha256 != current_controller_sha256:
        raise ValueError(
            'paired explorer controller state differs from the current '
            'controller')

    controller_version = checkpoint.get('controller_update_count')
    elite_version = checkpoint.get(
        'per_task_elite_controller_update_count')
    for name, value in {
            'controller_update_count': controller_version,
            'per_task_elite_controller_update_count': elite_version,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f'paired explorer {name} must be a non-negative integer')
    if elite_version != controller_version:
        raise ValueError(
            'paired explorer elite/controller versions do not match')
    elite_state = checkpoint.get('per_task_elite_memory')
    if elite_state is None:
        raise ValueError(
            'paired explorer checkpoint has no per-task elite memory')
    explorer_memory = api.DirectSeedEliteMemory(
        expected_task_ids, seed=0)
    explorer_memory.load_state_dict(elite_state)
    if bool(explorer_memory.valid.any()) and not torch.equal(
            explorer_memory.task[explorer_memory.valid],
            expected_task[explorer_memory.valid]):
        raise ValueError(
            'paired explorer elite task geometry differs from the current '
            'safe dataset')
    artifact = file_fingerprint(resolved)
    provenance = {
        'checkpoint': artifact,
        'safe_task_fingerprint_list_sha256': (
            safe_task_fingerprint_list_sha256),
        'projection_config': expected_projection,
        'controller_state_sha256': explorer_controller_sha256,
        'controller_update_count': controller_version,
        'per_task_elite_controller_update_count': elite_version,
        'explorer_elite_size': len(explorer_memory),
        'explorer_elite_coverage': explorer_memory.coverage,
    }
    return explorer_memory, provenance


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = set.intersection(*(set(value) for value in values))
    return {
        key: float(sum(value[key] for value in values) / len(values))
        for key in sorted(keys)
    }


def _route_rollout_metrics(
    routed,
    rollout,
) -> dict[str, Any]:
    route = routed.route
    n = int(route.numel())
    counts = {
        name: int((route == code).sum().item())
        for code, name in enumerate(ROUTE_NAMES)
    }
    attempted = routed.ik_attempted
    attempted_count = int(attempted.sum().item())
    return {
        'tasks': n,
        'route_count': counts,
        'route_fraction': {
            name: value / n for name, value in counts.items()
        },
        'direct_pass_fraction': float(routed.raw.valid.float().mean()),
        'ik_attempt_fraction': float(attempted.float().mean()),
        'ik_success_given_attempt': (
            float(routed.ik_ok[attempted].float().mean())
            if attempted_count else 0.0
        ),
        'valid_fraction': float(routed.valid.float().mean()),
        'progress_mean_m': float(rollout.progress_m.mean()),
        'progress_std_m': float(
            rollout.progress_m.std(unbiased=False)),
        'episode_len_mean': float(
            rollout.episode_len.float().mean()),
        'raw_position_error_mean_m': float(
            routed.raw.position_error_m.mean()),
        'controller_rollouts_per_task': 1.0,
        'generated_seeds_per_task': 1.0,
        'ik_attempts_per_task': float(attempted.float().mean()),
        'candidate_queries_per_task': 0.0,
    }


class _JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open('a', encoding='utf-8')

    def write(self, value: dict[str, Any]) -> None:
        self.file.write(json.dumps(value, sort_keys=True) + '\n')
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Bidirectional one-seed contextual RL and controller PPO')
    parser.add_argument(
        '--task-cache',
        default='Yuan/unified_rl/runs/ikpool_full_v1/ikpool_candidates.npz',
        help='only task geometry and q0_pilot are consumed')
    parser.add_argument(
        '--init-controller-dir',
        default='Yuan/unified_rl/runs/r2_grouped_best')
    parser.add_argument(
        '--out-dir',
        default='Yuan/unified_rl/runs/direct_seed_bidir')
    parser.add_argument('--resume', default=None)
    parser.add_argument(
        '--override-resume-rl-config', action='store_true',
        help='explicitly continue saved weights/replay with the current '
             'seed-loss CLI settings; otherwise resume requires an exact '
             'RL-config match')
    parser.add_argument(
        '--reset-actor-optimizer-on-resume', action='store_true',
        help='load actor weights but start a fresh Adam state at --actor-lr')
    parser.add_argument('--device', default=None)
    parser.add_argument('--outer-rounds', type=int, default=4)
    parser.add_argument('--backward-updates-per-round', type=int, default=100)
    parser.add_argument('--seed-tasks-per-update', type=int, default=128)
    parser.add_argument(
        '--task-sampling', choices=('random', 'cycle'), default='random',
        help='backward task batches use replacement sampling (random) or '
             'checkpointable shuffled full-dataset epochs (cycle)')
    parser.add_argument(
        '--gradient-updates-per-rollout', type=int, default=4,
        help='replay gradient steps after each real macro-rollout batch')
    parser.add_argument(
        '--precision-only-updates-per-rollout', type=int, default=4,
        help='cheap training-only deterministic FK/self-distillation steps '
             'after each real macro-rollout batch; adds no deployment cost')
    parser.add_argument(
        '--precision-only-projection-weight', type=float, default=0.25,
        help='one-IK self-distillation weight in precision-only updates')
    parser.add_argument(
        '--elite-projection-updates-per-rollout', type=int, default=0,
        help='actor-only self-distillation steps from high-return online '
             'ROUTE_REFINED replay samples')
    parser.add_argument(
        '--elite-projection-fraction', type=float, default=0.25,
        help='top return fraction eligible for elite projection updates')
    parser.add_argument(
        '--per-task-elite-updates-per-rollout', type=int, default=0,
        help='actor-only self-distillation steps from each task’s best '
             'successful online projection; adds no deployment cost')
    parser.add_argument(
        '--per-task-elite-post-updates-per-round', type=int, default=0,
        help='additional actor-only replay updates from the per-task online '
             'elite memory after collection; adds no environment or '
             'deployment cost')
    parser.add_argument(
        '--collect-paired-baseline-archive', action='store_true',
        help='record the first complete controller outcome for every sampled '
             'task for paired baseline/explorer post updates')
    parser.add_argument(
        '--paired-explorer-checkpoint', default=None,
        help='external direct-seed bidirectional checkpoint supplying '
             'strictly matched per-task explorer elites')
    parser.add_argument(
        '--paired-advantage-margin-m', type=float, default=0.0,
        help='minimum real-progress advantage required to replace a paired '
             'baseline target with the external explorer projection')
    parser.add_argument(
        '--paired-post-updates-per-round', type=int, default=0,
        help='actor-only post-collection updates from paired baseline versus '
             'external-explorer targets')
    parser.add_argument(
        '--per-task-elite-post-anchor-weight', type=float, default=0.0,
        help='frozen pre-post-update actor-mean anchor weight for either '
             'per-task or paired post updates')
    parser.add_argument(
        '--collect-per-task-elite-memory', action='store_true',
        help='collect per-task successful projection targets without '
             'requiring online or post replay updates')
    parser.add_argument(
        '--macro-replay-capacity', type=int, default=65_536,
        help='number of real task/seed/controller outcomes kept on device; '
             '0 restores the legacy current-batch-only path')
    parser.add_argument(
        '--macro-replay-batch-size', type=int, default=256)
    parser.add_argument(
        '--macro-replay-warmup', type=int, default=128,
        help='minimum stored real macro rollouts before replay updates')
    parser.add_argument(
        '--seed-actor-update-period', type=int, default=2,
        help='one actor update per this many macro-critic updates')
    parser.add_argument(
        '--critic-warmup-updates-after-controller-change',
        type=int, default=128,
        help='critic-only macro updates before actor updates after each '
             'controller change')
    parser.add_argument(
        '--actor-behavior-anchor-weight', type=float, default=0.0,
        help='small replay-action anchor limiting macro-Q OOD exploitation')
    parser.add_argument('--seed-entropy-coef', type=float, default=1e-3)
    parser.add_argument('--seed-precision-weight', type=float, default=0.01)
    parser.add_argument(
        '--seed-projection-distill-weight', type=float, default=0.25)
    parser.add_argument(
        '--seed-failure-precision-weight', type=float, default=0.05)
    parser.add_argument(
        '--seed-precision-cone-deg', type=float, default=24.5,
        help='training margin for deterministic mean; deployment gate is 30deg')
    parser.add_argument(
        '--refine-route-penalty-m', type=float, default=0.002,
        help='training-only macro reward cost for invoking the one-shot IK')
    parser.add_argument(
        '--fallback-route-penalty-m', type=float, default=0.01,
        help='training-only macro reward cost for missing the generated route')
    parser.add_argument(
        '--collection-noise-initial', type=float, default=1.0)
    parser.add_argument(
        '--collection-noise-final', type=float, default=0.05)
    parser.add_argument(
        '--collection-noise-anneal-rollouts', type=int, default=50_000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument(
        '--deterministic-backward', action='store_true',
        help='ablation: disable contextual actor exploration during collection')
    parser.add_argument(
        '--freeze-seed-actor-during-collection', action='store_true',
        help='collect a fixed actor snapshot; post-round elite replay may '
             'still update the actor afterwards')
    parser.add_argument(
        '--forward-mode', choices=('ppo', 'skip'), default='ppo')
    parser.add_argument('--controller-n-envs', type=int, default=128)
    parser.add_argument(
        '--controller-steps-per-round', type=int, default=1_000_000)
    parser.add_argument('--controller-lr', type=float, default=1e-5)
    parser.add_argument(
        '--controller-fallback-probability', type=float, default=0.0,
        help='training-only reset mixture; deployment always uses one actor seed')
    parser.add_argument(
        '--fallback-validation-batch-size', type=int, default=1024)
    parser.add_argument('--position-tol-m', type=float, default=5e-3)
    parser.add_argument('--cone-deg', type=float, default=30.0)
    parser.add_argument('--projection-cone-deg', type=float, default=24.5)
    parser.add_argument('--joint-margin-rad', type=float, default=0.02)
    parser.add_argument('--collision-margin-m', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=20260728)
    parser.add_argument(
        '--dry-run', action='store_true',
        help='validate schedule and API protocol without loading artifacts')
    parser.add_argument(
        '--api-smoke', action='store_true',
        help='with --dry-run, also run direct_seed_rl synthetic CPU update')
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    DirectSeedProjectionConfig(
        position_tol_m=args.position_tol_m,
        cone_deg=args.cone_deg,
        projection_cone_deg=args.projection_cone_deg,
        joint_margin_rad=args.joint_margin_rad,
        collision_margin_m=args.collision_margin_m,
    )
    if args.api_smoke and not args.dry_run:
        raise ValueError('--api-smoke requires --dry-run')
    positive = {
        '--outer-rounds': args.outer_rounds,
        '--seed-tasks-per-update': args.seed_tasks_per_update,
        '--gradient-updates-per-rollout': (
            args.gradient_updates_per_rollout),
        '--macro-replay-batch-size': args.macro_replay_batch_size,
        '--macro-replay-warmup': args.macro_replay_warmup,
        '--seed-actor-update-period': args.seed_actor_update_period,
        '--controller-n-envs': args.controller_n_envs,
        '--fallback-validation-batch-size': (
            args.fallback_validation_batch_size),
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f'{name} must be positive')
    nonnegative = {
        '--backward-updates-per-round': (
            args.backward_updates_per_round),
        '--controller-steps-per-round': (
            args.controller_steps_per_round),
        '--macro-replay-capacity': args.macro_replay_capacity,
        '--precision-only-updates-per-rollout': (
            args.precision_only_updates_per_rollout),
        '--elite-projection-updates-per-rollout': (
            args.elite_projection_updates_per_rollout),
        '--per-task-elite-updates-per-rollout': (
            args.per_task_elite_updates_per_rollout),
        '--per-task-elite-post-updates-per-round': (
            args.per_task_elite_post_updates_per_round),
        '--paired-post-updates-per-round': (
            args.paired_post_updates_per_round),
        '--critic-warmup-updates-after-controller-change': (
            args.critic_warmup_updates_after_controller_change),
        '--collection-noise-anneal-rollouts': (
            args.collection_noise_anneal_rollouts),
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError(f'{name} must be non-negative')
    for name, value in {
            '--actor-lr': args.actor_lr,
            '--critic-lr': args.critic_lr,
            '--controller-lr': args.controller_lr,
            '--actor-behavior-anchor-weight': (
                args.actor_behavior_anchor_weight),
            '--seed-entropy-coef': args.seed_entropy_coef,
            '--seed-precision-weight': args.seed_precision_weight,
            '--seed-projection-distill-weight': (
                args.seed_projection_distill_weight),
            '--seed-failure-precision-weight': (
                args.seed_failure_precision_weight),
            '--refine-route-penalty-m': args.refine_route_penalty_m,
            '--fallback-route-penalty-m': args.fallback_route_penalty_m,
            '--precision-only-projection-weight': (
                args.precision_only_projection_weight),
            '--per-task-elite-post-anchor-weight': (
                args.per_task_elite_post_anchor_weight),
            '--paired-advantage-margin-m': (
                args.paired_advantage_margin_m),
            '--collection-noise-initial': args.collection_noise_initial,
            '--collection-noise-final': args.collection_noise_final,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f'{name} must be finite and non-negative')
    for name, value in {
            '--actor-lr': args.actor_lr,
            '--critic-lr': args.critic_lr,
            '--controller-lr': args.controller_lr,
    }.items():
        if value == 0.0:
            raise ValueError(f'{name} must be positive')
    if (args.macro_replay_capacity > 0
            and args.macro_replay_warmup > args.macro_replay_capacity):
        raise ValueError(
            '--macro-replay-warmup cannot exceed replay capacity')
    if (args.elite_projection_updates_per_rollout > 0
            and args.macro_replay_capacity == 0):
        raise ValueError(
            'elite projection updates require a non-zero macro replay')
    if args.collect_paired_baseline_archive:
        if not args.deterministic_backward:
            raise ValueError(
                'paired baseline collection requires '
                '--deterministic-backward')
        if not args.freeze_seed_actor_during_collection:
            raise ValueError(
                'paired baseline collection requires '
                '--freeze-seed-actor-during-collection')
        if args.task_sampling != 'cycle':
            raise ValueError(
                'paired baseline collection requires '
                '--task-sampling cycle')
    if args.freeze_seed_actor_during_collection:
        incompatible = {
            '--precision-only-updates-per-rollout': (
                args.precision_only_updates_per_rollout),
            '--elite-projection-updates-per-rollout': (
                args.elite_projection_updates_per_rollout),
            '--per-task-elite-updates-per-rollout': (
                args.per_task_elite_updates_per_rollout),
        }
        enabled = [
            name for name, value in incompatible.items() if value > 0]
        if enabled:
            raise ValueError(
                '--freeze-seed-actor-during-collection requires zero '
                f'online actor-only updates, got {enabled}')
    if (args.per_task_elite_post_updates_per_round > 0
            and args.paired_post_updates_per_round > 0):
        raise ValueError(
            'paired and per-task elite post updates are mutually exclusive')
    if args.paired_post_updates_per_round > 0:
        if not args.collect_paired_baseline_archive:
            raise ValueError(
                'paired post updates require '
                '--collect-paired-baseline-archive')
        if args.paired_explorer_checkpoint is None:
            raise ValueError(
                'paired post updates require '
                '--paired-explorer-checkpoint')
        controller_will_update = (
            args.forward_mode == 'ppo'
            and args.controller_steps_per_round > 0)
        if args.outer_rounds > 1 and controller_will_update:
            raise ValueError(
                'paired post updates cannot span multiple outer rounds when '
                'forward PPO updates the controller; use --outer-rounds 1, '
                '--forward-mode skip, or --controller-steps-per-round 0')
    if (args.paired_explorer_checkpoint is not None
            and not args.collect_paired_baseline_archive):
        raise ValueError(
            '--paired-explorer-checkpoint requires '
            '--collect-paired-baseline-archive')
    if (not math.isfinite(args.elite_projection_fraction)
            or not 0.0 < args.elite_projection_fraction <= 1.0):
        raise ValueError(
            '--elite-projection-fraction must be finite and in (0, 1]')
    if not 0.0 <= args.controller_fallback_probability <= 1.0:
        raise ValueError(
            '--controller-fallback-probability must be in [0, 1]')
    if (not math.isfinite(args.seed_precision_cone_deg)
            or not 0.0 < args.seed_precision_cone_deg <= args.cone_deg):
        raise ValueError(
            '--seed-precision-cone-deg must be in (0, --cone-deg]')


def _dry_run(args: argparse.Namespace) -> None:
    api = _load_direct_seed_rl_api(required=False)
    api_status = 'pending' if api is None else 'validated'
    smoke = None
    if args.api_smoke:
        if api is None:
            raise RuntimeError('--api-smoke requires direct_seed_rl API')
        smoke_fn = getattr(
            api.module, 'synthetic_direct_seed_rl_smoke', None)
        if not callable(smoke_fn):
            raise RuntimeError(
                'direct_seed_rl has no synthetic_direct_seed_rl_smoke()')
        smoke = smoke_fn('cpu')
    print(json.dumps({
        'format': 'direct-seed-bidirectional-dry-run-v1',
        'api': {
            'module': _DIRECT_SEED_RL_MODULE,
            'status': api_status,
            'symbols': list(_DIRECT_SEED_RL_SYMBOLS),
            'smoke': smoke,
        },
        'invariants': {
            'diffusion_dependency': False,
            'historical_candidate_joint_queries': 0,
            'generated_seeds_per_task': 1,
            'controller_rollouts_per_task': 1,
            'max_ik_refinements_per_task': 1,
            'fallback_filter': 'strict-source-filter-before-training',
            'paired_baseline_collection': (
                'deterministic-frozen-actor-full-cycle'),
            'paired_post_requires_full_coverage': True,
        },
        'schedule': {
            'outer_rounds': args.outer_rounds,
            'order_per_round': ['backward_seed', 'forward_controller'],
            'backward_updates': args.backward_updates_per_round,
            'seed_tasks_per_update': args.seed_tasks_per_update,
            'task_sampling': args.task_sampling,
            'freeze_seed_actor_during_collection': (
                args.freeze_seed_actor_during_collection),
            'gradient_updates_per_rollout': (
                args.gradient_updates_per_rollout),
            'precision_only_updates_per_rollout': (
                args.precision_only_updates_per_rollout),
            'precision_only_projection_weight': (
                args.precision_only_projection_weight),
            'elite_projection_updates_per_rollout': (
                args.elite_projection_updates_per_rollout),
            'elite_projection_fraction': (
                args.elite_projection_fraction),
            'per_task_elite_updates_per_rollout': (
                args.per_task_elite_updates_per_rollout),
            'per_task_elite_post_updates_per_round': (
                args.per_task_elite_post_updates_per_round),
            'collect_paired_baseline_archive': (
                args.collect_paired_baseline_archive),
            'paired_collection_contract': (
                _paired_collection_contract(args)
                if args.collect_paired_baseline_archive else None),
            'paired_explorer_checkpoint': (
                args.paired_explorer_checkpoint),
            'paired_advantage_margin_m': (
                args.paired_advantage_margin_m),
            'paired_post_updates_per_round': (
                args.paired_post_updates_per_round),
            'per_task_elite_post_anchor_weight': (
                args.per_task_elite_post_anchor_weight),
            'collect_per_task_elite_memory': (
                args.collect_per_task_elite_memory),
            'macro_replay_capacity': args.macro_replay_capacity,
            'macro_replay_batch_size': args.macro_replay_batch_size,
            'macro_replay_warmup': args.macro_replay_warmup,
            'seed_actor_update_period': (
                args.seed_actor_update_period),
            'critic_warmup_updates_after_controller_change': (
                args.critic_warmup_updates_after_controller_change),
            'actor_behavior_anchor_weight': (
                args.actor_behavior_anchor_weight),
            'seed_entropy_coef': args.seed_entropy_coef,
            'seed_precision_weight': args.seed_precision_weight,
            'seed_projection_distill_weight': (
                args.seed_projection_distill_weight),
            'seed_failure_precision_weight': (
                args.seed_failure_precision_weight),
            'seed_precision_cone_deg': args.seed_precision_cone_deg,
            'refine_route_penalty_m': args.refine_route_penalty_m,
            'fallback_route_penalty_m': args.fallback_route_penalty_m,
            'collection_noise': {
                'initial': args.collection_noise_initial,
                'final': args.collection_noise_final,
                'anneal_real_rollouts': (
                    args.collection_noise_anneal_rollouts),
            },
            'forward_mode': args.forward_mode,
            'controller_steps': args.controller_steps_per_round,
            'override_resume_rl_config': (
                args.override_resume_rl_config),
            'reset_actor_optimizer_on_resume': (
                args.reset_actor_optimizer_on_resume),
        },
    }, indent=2, sort_keys=True))


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if args.dry_run:
        _dry_run(args)
        return

    api = _load_direct_seed_rl_api(required=True)
    assert api is not None
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    seed_global_rng(args.seed)
    projection_config = DirectSeedProjectionConfig(
        position_tol_m=args.position_tol_m,
        cone_deg=args.cone_deg,
        projection_cone_deg=args.projection_cone_deg,
        joint_margin_rad=args.joint_margin_rad,
        collision_margin_m=args.collision_margin_m,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / 'direct_seed_bidir.pt'
    if state_path.exists() and args.resume is None:
        raise FileExistsError(
            f'{state_path} already exists; pass --resume explicitly')
    init_controller_dir = resolve_controller_dir(
        args.init_controller_dir)

    # Build the exact physics gate before admitting any fallback into the run.
    seed_env = build_env_from_run(
        init_controller_dir, args.seed_tasks_per_update, device,
        env_overrides={'observe_ray_error': True})
    cache = CachedSeedCandidateDataset.from_npz(args.task_cache)
    source_dataset = DirectTaskDataset.from_candidate_cache(cache)
    dataset, filter_manifest = _filter_strict_fallbacks(
        source_dataset, seed_env, projection_config,
        batch_size=args.fallback_validation_batch_size)
    filter_path = out_dir / 'fallback_filter.json'
    filter_text = json.dumps(
        filter_manifest, indent=2, sort_keys=True)
    if filter_path.exists():
        if filter_path.read_text(encoding='utf-8') != filter_text:
            raise RuntimeError(
                'existing fallback_filter.json differs from current gate')
    else:
        filter_path.write_text(filter_text, encoding='utf-8')
    print(
        f'[direct-bidir] safe fallback tasks: {len(dataset)}/'
        f'{len(source_dataset)}; excluded='
        f'{filter_manifest["excluded_task_count"]}',
        flush=True)

    forward_env = build_env_from_run(
        init_controller_dir, args.controller_n_envs, device,
        env_overrides={'observe_ray_error': True})
    controller = load_controller_agent(
        init_controller_dir, forward_env, device)
    run_config = load_run_config(init_controller_dir)
    controller_cfg = ppo_config_from_run(
        run_config,
        total_timesteps=args.controller_steps_per_round,
        learning_rate=args.controller_lr,
        anneal_lr=False,
    )
    gamma = float(controller_cfg.gamma)
    controller_optimizer = torch.optim.Adam(
        controller.parameters(), lr=args.controller_lr, eps=1e-5)
    controller_scaler = (
        RewardScaler(
            args.controller_n_envs, controller_cfg.gamma, device)
        if controller_cfg.normalize_returns else None)

    task_mean, task_std = dataset.task_normalization()
    actor_config = api.DirectSeedActorConfig()
    critic_config = api.DirectSeedCriticConfig()
    requested_rl_config = api.DirectSeedRLConfig(
        entropy_coef=args.seed_entropy_coef,
        precision_weight=args.seed_precision_weight,
        cone_deg=args.seed_precision_cone_deg,
        projection_distill_weight=(
            args.seed_projection_distill_weight),
        failure_precision_weight=args.seed_failure_precision_weight,
        behavior_anchor_weight=args.actor_behavior_anchor_weight,
        refine_route_penalty_m=args.refine_route_penalty_m,
        fallback_route_penalty_m=args.fallback_route_penalty_m)
    rl_config = requested_rl_config
    actor = api.DirectSeedActor(
        seed_env.kin.lmt_lo, seed_env.kin.lmt_up, actor_config,
        task_mean=task_mean, task_std=task_std).to(device)
    critic = api.TwinMacroQ(
        seed_env.kin.lmt_lo, seed_env.kin.lmt_up, critic_config,
        task_mean=task_mean, task_std=task_std).to(device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr)
    task_generator = torch.Generator().manual_seed(args.seed + 17)
    task_cycle_sampler = (
        DirectTaskCycleSampler(len(dataset), seed=args.seed + 17)
        if args.task_sampling == 'cycle' else None
    )
    collection_generator = torch.Generator(device=device)
    collection_generator.manual_seed(args.seed + 18)
    actor_update_generator = torch.Generator(device=device)
    actor_update_generator.manual_seed(args.seed + 19)
    macro_replay = (
        api.DirectSeedMacroReplay(
            args.macro_replay_capacity, device,
            dtype=seed_env.kin.dtype, seed=args.seed + 20)
        if args.macro_replay_capacity > 0 else None
    )
    per_task_elite_memory = (
        api.DirectSeedEliteMemory(
            dataset.batch.task_indices, seed=args.seed + 21)
        if (args.per_task_elite_updates_per_rollout > 0
            or args.per_task_elite_post_updates_per_round > 0
            or args.collect_per_task_elite_memory)
        else None
    )
    paired_archive = (
        api.DirectSeedPairedArchive(
            dataset.batch.task_indices, seed=args.seed + 22)
        if args.collect_paired_baseline_archive else None
    )
    paired_collection_contract = (
        _paired_collection_contract(args)
        if paired_archive is not None else None)
    paired_baseline_actor_state_sha256 = (
        _actor_state_sha256(actor)
        if paired_archive is not None else None)
    paired_explorer_memory = None
    paired_explorer_provenance = None
    global_backward_update = 0
    global_actor_update = 0
    global_precision_update = 0
    global_elite_projection_update = 0
    global_per_task_elite_update = 0
    global_per_task_elite_post_update = 0
    global_paired_post_update = 0
    real_macro_rollouts = 0
    replay_samples = 0
    precision_replay_samples = 0
    elite_projection_replay_samples = 0
    per_task_elite_samples = 0
    per_task_elite_improvements = 0
    paired_target_samples = 0
    controller_update_count = 0
    controller_version_critic_updates = 0
    macro_replay_controller_update_count = 0
    per_task_elite_controller_update_count = 0
    paired_archive_controller_update_count = 0
    start_round = 1
    resume_after_backward = False
    resume_rl_config_overridden = False
    resume_discarded_stale_replay = 0
    resume_discarded_stale_per_task_elites = 0
    resume_discarded_stale_paired_outcomes = 0
    resume_reset_stale_critic = False
    resume_cycle_initialized_from_legacy = False
    resume_actor_optimizer_reset = False
    saved_paired_explorer_provenance = None
    resume_optimizer_lr_overrides: dict[str, dict[str, Any]] = {}

    resume_path = Path(args.resume).expanduser().resolve() \
        if args.resume is not None else None
    if resume_path is not None:
        saved = torch.load(
            resume_path, map_location='cpu', weights_only=False)
        if saved.get('format') != 'direct-seed-bidirectional-v1':
            raise ValueError('unsupported direct-seed bidirectional resume')
        saved_paired_explorer_provenance = saved.get(
            'paired_explorer_provenance')
        if saved.get('task_cache') != str(
                Path(args.task_cache).expanduser().resolve()):
            raise ValueError('resume task cache differs from current request')
        if saved.get('init_controller_dir') != str(init_controller_dir):
            raise ValueError(
                'resume initial controller differs from current request')
        if saved.get('kept_task_indices') != (
                dataset.batch.task_indices.tolist()):
            raise ValueError(
                'resume strict-safe task set differs from current gate')
        saved_args_payload = saved.get('args', {})
        saved_task_sampling = saved.get(
            'task_sampling', saved_args_payload.get('task_sampling'))
        if (saved_task_sampling is not None
                and saved_task_sampling != args.task_sampling):
            raise ValueError(
                'resume task-sampling mode differs from current request')
        saved_direct_metadata = (
            saved.get('direct_seed', {}).get('metadata', {}))
        saved_safe_fingerprint = saved.get(
            'safe_task_fingerprint_list_sha256',
            saved_direct_metadata.get(
                'safe_task_fingerprint_list_sha256'))
        current_safe_fingerprint = (
            filter_manifest['kept_fingerprint_list_sha256'])
        if saved_safe_fingerprint != current_safe_fingerprint:
            raise ValueError(
                'resume safe-task geometry fingerprint differs from '
                'the current cache')
        saved_projection_config = saved.get('projection_config')
        if saved_projection_config is None:
            saved_args = saved.get('args', {})
            projection_fields = (
                'position_tol_m', 'cone_deg', 'projection_cone_deg',
                'joint_margin_rad', 'collision_margin_m')
            if not all(name in saved_args for name in projection_fields):
                raise ValueError(
                    'resume checkpoint has no projection-gate provenance')
            saved_projection_config = {
                name: saved_args[name] for name in projection_fields
            }
        if saved_projection_config != dataclasses.asdict(
                projection_config):
            raise ValueError(
                'resume projection/gate config differs from current request')
        (
            actor,
            critic,
            actor_optimizer_state,
            critic_optimizer_state,
            direct_payload,
        ) = api.load_direct_seed_rl_checkpoint(
            saved['direct_seed'], device)
        saved_rl_config = api.DirectSeedRLConfig(
            **dict(direct_payload['rl_config']))
        saved_rl_fields = dataclasses.asdict(saved_rl_config)
        requested_rl_fields = dataclasses.asdict(requested_rl_config)
        if args.override_resume_rl_config:
            rl_config = requested_rl_config
            resume_rl_config_overridden = (
                saved_rl_fields != requested_rl_fields)
        else:
            changed_rl_fields = sorted(
                name for name, value in saved_rl_fields.items()
                if requested_rl_fields[name] != value)
            if changed_rl_fields:
                raise ValueError(
                    'resume seed RL config differs in '
                    f'{changed_rl_fields}; restate the saved settings for an '
                    'exact resume or pass '
                    '--override-resume-rl-config explicitly')
            rl_config = saved_rl_config
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=args.actor_lr)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=args.critic_lr)
        if (actor_optimizer_state is not None
                and not args.reset_actor_optimizer_on_resume):
            actor_optimizer.load_state_dict(actor_optimizer_state)
            _optimizer_to(actor_optimizer, device)
            previous_lrs = _set_optimizer_lr(
                actor_optimizer, args.actor_lr)
            if any(lr != args.actor_lr for lr in previous_lrs):
                resume_optimizer_lr_overrides['actor'] = {
                    'saved': previous_lrs,
                    'effective': args.actor_lr,
                }
        elif (actor_optimizer_state is not None
                and args.reset_actor_optimizer_on_resume):
            resume_actor_optimizer_reset = True
        if critic_optimizer_state is not None:
            critic_optimizer.load_state_dict(critic_optimizer_state)
            _optimizer_to(critic_optimizer, device)
            previous_lrs = _set_optimizer_lr(
                critic_optimizer, args.critic_lr)
            if any(lr != args.critic_lr for lr in previous_lrs):
                resume_optimizer_lr_overrides['critic'] = {
                    'saved': previous_lrs,
                    'effective': args.critic_lr,
                }
        load_controller_state_dict(controller, saved['controller'])
        controller_optimizer.load_state_dict(
            saved['controller_optimizer'])
        adapt_controller_optimizer_observation_state(
            controller_optimizer, controller)
        _optimizer_to(controller_optimizer, device)
        previous_lrs = _set_optimizer_lr(
            controller_optimizer, args.controller_lr)
        if any(lr != args.controller_lr for lr in previous_lrs):
            resume_optimizer_lr_overrides['controller'] = {
                'saved': previous_lrs,
                'effective': args.controller_lr,
            }
        if (controller_scaler is not None
                and saved.get('controller_scaler') is not None):
            controller_scaler.load_state_dict(saved['controller_scaler'])
        if args.task_sampling == 'random':
            task_generator.set_state(saved['task_generator_state'])
        else:
            saved_cycle_sampler = saved.get('task_cycle_sampler')
            if saved_cycle_sampler is not None:
                assert task_cycle_sampler is not None
                task_cycle_sampler.load_state_dict(saved_cycle_sampler)
            elif saved_task_sampling is None:
                # A legacy random checkpoint has no cycle history.  An
                # explicitly requested transition starts a fresh shuffled
                # epoch and is recorded in the resume audit.
                resume_cycle_initialized_from_legacy = True
            else:
                raise ValueError(
                    'cycle resume checkpoint has no cycle sampler state')
        if saved.get('collection_generator_state') is not None:
            collection_generator.set_state(
                saved['collection_generator_state'])
        if saved.get('actor_update_generator_state') is not None:
            actor_update_generator.set_state(
                saved['actor_update_generator_state'])
        saved_replay = saved.get('macro_replay')
        if saved_replay is not None:
            if macro_replay is None:
                raise ValueError(
                    'resume contains macro replay but replay is disabled')
            macro_replay.load_state_dict(saved_replay)
        controller_version_known = (
            saved.get('controller_update_count') is not None)
        controller_update_count = int(saved.get(
            'controller_update_count', 0))
        controller_version_critic_updates = int(saved.get(
            'controller_version_critic_updates',
            args.critic_warmup_updates_after_controller_change))
        saved_per_task_elite_memory = saved.get(
            'per_task_elite_memory')
        if saved_per_task_elite_memory is not None:
            if per_task_elite_memory is None:
                raise ValueError(
                    'resume contains per-task elite memory but per-task '
                    'elite updates are disabled')
            per_task_elite_memory.load_state_dict(
                saved_per_task_elite_memory)
        saved_paired_archive = saved.get('paired_archive')
        if saved_paired_archive is not None:
            if paired_archive is None:
                raise ValueError(
                    'resume contains a paired baseline archive but paired '
                    'collection is disabled')
            paired_archive.load_state_dict(saved_paired_archive)
        if paired_archive is not None:
            assert paired_collection_contract is not None
            (
                paired_collection_contract,
                paired_baseline_actor_state_sha256,
            ) = _restore_paired_collection_provenance(
                saved,
                requested_contract=paired_collection_contract,
                paired_archive=paired_archive,
                current_actor_state_sha256=_actor_state_sha256(actor))
        saved_replay_controller_version = saved.get(
            'macro_replay_controller_update_count')
        replay_controller_version_known = (
            saved_replay_controller_version is not None)
        macro_replay_controller_update_count = (
            int(saved_replay_controller_version)
            if replay_controller_version_known
            else controller_update_count)
        saved_per_task_elite_controller_version = saved.get(
            'per_task_elite_controller_update_count')
        per_task_elite_controller_version_known = (
            saved_per_task_elite_memory is None
            or saved_per_task_elite_controller_version is not None)
        per_task_elite_controller_update_count = (
            int(saved_per_task_elite_controller_version)
            if saved_per_task_elite_controller_version is not None
            else controller_update_count)
        saved_paired_archive_controller_version = saved.get(
            'paired_archive_controller_update_count')
        paired_archive_controller_version_known = (
            saved_paired_archive is None
            or saved_paired_archive_controller_version is not None)
        paired_archive_controller_update_count = (
            int(saved_paired_archive_controller_version)
            if saved_paired_archive_controller_version is not None
            else controller_update_count)
        restore_global_rng(saved, device)
        global_backward_update = int(saved['global_backward_update'])
        global_actor_update = int(saved.get(
            'global_actor_update', global_backward_update))
        global_precision_update = int(saved.get(
            'global_precision_update', 0))
        global_elite_projection_update = int(saved.get(
            'global_elite_projection_update', 0))
        global_per_task_elite_update = int(saved.get(
            'global_per_task_elite_update', 0))
        global_per_task_elite_post_update = int(saved.get(
            'global_per_task_elite_post_update', 0))
        global_paired_post_update = int(saved.get(
            'global_paired_post_update', 0))
        real_macro_rollouts = int(saved.get(
            'real_macro_rollouts', 0))
        replay_samples = int(saved.get('replay_samples', 0))
        precision_replay_samples = int(saved.get(
            'precision_replay_samples', 0))
        elite_projection_replay_samples = int(saved.get(
            'elite_projection_replay_samples', 0))
        per_task_elite_samples = int(saved.get(
            'per_task_elite_samples', 0))
        per_task_elite_improvements = int(saved.get(
            'per_task_elite_improvements', 0))
        paired_target_samples = int(saved.get(
            'paired_target_samples', 0))
        phase = saved['phase']
        completed_round = int(saved['outer_round'])
        if phase == 'backward_complete':
            if (replay_controller_version_known
                    and macro_replay_controller_update_count
                    != controller_update_count):
                raise ValueError(
                    'backward-complete checkpoint has mismatched '
                    'controller and macro-replay versions')
            if (per_task_elite_controller_version_known
                    and per_task_elite_controller_update_count
                    != controller_update_count):
                raise ValueError(
                    'backward-complete checkpoint has mismatched controller '
                    'and per-task-elite versions')
            if (not paired_archive_controller_version_known
                    or paired_archive_controller_update_count
                    != controller_update_count):
                if saved_paired_archive is not None:
                    raise ValueError(
                        'backward-complete checkpoint has mismatched '
                        'controller and paired-archive versions')
            # Legacy backward-complete returns were collected immediately
            # before this saved controller and are therefore safe to bind.
            macro_replay_controller_update_count = controller_update_count
            per_task_elite_controller_update_count = (
                controller_update_count)
            paired_archive_controller_update_count = (
                controller_update_count)
            start_round = completed_round
            resume_after_backward = True
        elif phase == 'round_complete':
            start_round = completed_round + 1
            replay_matches_controller = (
                replay_controller_version_known
                and macro_replay_controller_update_count
                == controller_update_count)
            if macro_replay is not None and not replay_matches_controller:
                # A round-complete checkpoint already contains the updated
                # controller. Versioned frozen-controller checkpoints retain
                # valid returns; unversioned legacy checkpoints fail closed.
                resume_discarded_stale_replay = len(macro_replay)
                macro_replay.clear()
                macro_replay_controller_update_count = (
                    controller_update_count)
            per_task_elites_match_controller = (
                per_task_elite_controller_version_known
                and per_task_elite_controller_update_count
                == controller_update_count)
            if (per_task_elite_memory is not None
                    and not per_task_elites_match_controller):
                resume_discarded_stale_per_task_elites = len(
                    per_task_elite_memory)
                per_task_elite_memory.clear()
                per_task_elite_controller_update_count = (
                    controller_update_count)
            paired_archive_matches_controller = (
                paired_archive_controller_version_known
                and paired_archive_controller_update_count
                == controller_update_count)
            if (paired_archive is not None
                    and not paired_archive_matches_controller):
                resume_discarded_stale_paired_outcomes = len(
                    paired_archive)
                paired_archive.clear()
                paired_archive_controller_update_count = (
                    controller_update_count)
        else:
            raise ValueError(f'unknown resume phase {phase!r}')
        legacy_round_controller_changed = (
            not controller_version_known
            and phase == 'round_complete'
            and saved.get('args', {}).get('forward_mode') != 'skip'
            and int(saved.get('args', {}).get(
                'controller_steps_per_round', 0)) > 0)
        if legacy_round_controller_changed:
            critic = api.TwinMacroQ(
                seed_env.kin.lmt_lo, seed_env.kin.lmt_up, critic_config,
                task_mean=task_mean, task_std=task_std).to(device)
            critic_optimizer = torch.optim.Adam(
                critic.parameters(), lr=args.critic_lr)
            controller_version_critic_updates = 0
            resume_reset_stale_critic = True
        print(
            f'[direct-bidir] resumed {phase} at outer round '
            f'{start_round}; rl_config_override='
            f'{resume_rl_config_overridden}; stale_replay_discarded='
            f'{resume_discarded_stale_replay}; optimizer_lr_overrides='
            f'{resume_optimizer_lr_overrides}; controller_version='
            f'{controller_update_count}; replay_version='
            f'{macro_replay_controller_update_count}; '
            f'per_task_elite_version='
            f'{per_task_elite_controller_update_count}; '
            f'paired_archive_version='
            f'{paired_archive_controller_update_count}; '
            f'stale_per_task_elites_discarded='
            f'{resume_discarded_stale_per_task_elites}; '
            f'stale_paired_outcomes_discarded='
            f'{resume_discarded_stale_paired_outcomes}; '
            f'task_sampling={args.task_sampling}; '
            f'cycle_from_legacy='
            f'{resume_cycle_initialized_from_legacy}; '
            f'actor_optimizer_reset='
            f'{resume_actor_optimizer_reset}; '
            f'stale_critic_reset={resume_reset_stale_critic}; '
            f'controller_version_critic_updates='
            f'{controller_version_critic_updates}', flush=True)

    if args.paired_explorer_checkpoint is not None:
        (
            paired_explorer_memory,
            paired_explorer_provenance,
        ) = _load_paired_explorer_checkpoint(
            args.paired_explorer_checkpoint, api,
            task_ids=dataset.batch.task_indices,
            task=dataset.batch.task,
            safe_task_fingerprint_list_sha256=(
                filter_manifest['kept_fingerprint_list_sha256']),
            projection_config=projection_config,
            controller_state=controller.state_dict())
        if (saved_paired_explorer_provenance is not None
                and saved_paired_explorer_provenance
                != paired_explorer_provenance):
            raise ValueError(
                'paired explorer provenance differs from the resume '
                'checkpoint')

    # Preserve a standard controller directory for existing evaluation tools.
    output_config = out_dir / 'config.yaml'
    effective_run_config = copy.deepcopy(run_config)
    effective_run_config.setdefault('env', {})[
        'observe_ray_error'] = True
    output_config.write_text(
        yaml.safe_dump(effective_run_config, sort_keys=False),
        encoding='utf-8')
    logger = _JsonlLogger(out_dir / 'metrics.jsonl')
    if resume_path is not None:
        logger.write({
            'phase': 'resume_audit',
            'resume_path': str(resume_path),
            'rl_config_overridden': resume_rl_config_overridden,
            'discarded_stale_returns': (
                resume_discarded_stale_replay),
            'discarded_stale_per_task_elites': (
                resume_discarded_stale_per_task_elites),
            'discarded_stale_paired_outcomes': (
                resume_discarded_stale_paired_outcomes),
            'optimizer_lr_overrides': resume_optimizer_lr_overrides,
            'reset_stale_critic': resume_reset_stale_critic,
            'actor_optimizer_reset': resume_actor_optimizer_reset,
            'task_sampling': args.task_sampling,
            'freeze_seed_actor_during_collection': (
                args.freeze_seed_actor_during_collection),
            'cycle_initialized_from_legacy': (
                resume_cycle_initialized_from_legacy),
            'controller_update_count': controller_update_count,
            'controller_version_critic_updates': (
                controller_version_critic_updates),
            'macro_replay_controller_update_count': (
                macro_replay_controller_update_count),
            'per_task_elite_controller_update_count': (
                per_task_elite_controller_update_count),
            'paired_archive_controller_update_count': (
                paired_archive_controller_update_count),
            'paired_collection_contract': (
                copy.deepcopy(paired_collection_contract)),
            'paired_baseline_actor_state_sha256': (
                paired_baseline_actor_state_sha256),
            'replay_size_after': (
                len(macro_replay)
                if macro_replay is not None else 0),
            'per_task_elite_size_after': (
                len(per_task_elite_memory)
                if per_task_elite_memory is not None else 0),
            'paired_archive_size_after': (
                len(paired_archive)
                if paired_archive is not None else 0),
            'paired_explorer_provenance': (
                copy.deepcopy(paired_explorer_provenance)),
        })

    def save(outer_round: int, phase: str) -> None:
        metadata = {
            'runner_format': 'direct-seed-bidirectional-v1',
            'outer_round': outer_round,
            'phase': phase,
            'task_cache': str(
                Path(args.task_cache).expanduser().resolve()),
            'projection_config': dataclasses.asdict(projection_config),
            'safe_task_count': len(dataset),
            'excluded_task_count': (
                filter_manifest['excluded_task_count']),
            'safe_task_fingerprint_list_sha256': (
                filter_manifest['kept_fingerprint_list_sha256']),
            'excluded_task_fingerprint_list_sha256': (
                filter_manifest['excluded_fingerprint_list_sha256']),
            'one_task_one_seed': True,
            'max_ik_refinements': 1,
            'diffusion_dependency': False,
            'task_sampling': args.task_sampling,
            'freeze_seed_actor_during_collection': (
                args.freeze_seed_actor_during_collection),
            'collect_per_task_elite_memory': (
                args.collect_per_task_elite_memory),
            'per_task_elite_post_updates_per_round': (
                args.per_task_elite_post_updates_per_round),
            'per_task_elite_post_anchor_weight': (
                args.per_task_elite_post_anchor_weight),
            'collect_paired_baseline_archive': (
                args.collect_paired_baseline_archive),
            'paired_collection_contract': (
                copy.deepcopy(paired_collection_contract)),
            'paired_baseline_actor_state_sha256': (
                paired_baseline_actor_state_sha256),
            'paired_explorer_provenance': (
                copy.deepcopy(paired_explorer_provenance)),
            'paired_advantage_margin_m': (
                args.paired_advantage_margin_m),
            'paired_post_updates_per_round': (
                args.paired_post_updates_per_round),
            'cycle_initialized_from_legacy': (
                resume_cycle_initialized_from_legacy),
            'resume_rl_config_overridden': (
                resume_rl_config_overridden),
            'resume_discarded_stale_replay': (
                resume_discarded_stale_replay),
            'resume_discarded_stale_per_task_elites': (
                resume_discarded_stale_per_task_elites),
            'resume_discarded_stale_paired_outcomes': (
                resume_discarded_stale_paired_outcomes),
            'resume_optimizer_lr_overrides': (
                copy.deepcopy(resume_optimizer_lr_overrides)),
            'resume_reset_stale_critic': (
                resume_reset_stale_critic),
            'resume_actor_optimizer_reset': (
                resume_actor_optimizer_reset),
            'controller_update_count': controller_update_count,
            'controller_version_critic_updates': (
                controller_version_critic_updates),
            'macro_replay_controller_update_count': (
                macro_replay_controller_update_count),
            'per_task_elite_controller_update_count': (
                per_task_elite_controller_update_count),
            'paired_archive_controller_update_count': (
                paired_archive_controller_update_count),
            'macro_replay_capacity': (
                macro_replay.capacity if macro_replay is not None else 0),
            'real_macro_rollouts': real_macro_rollouts,
            'replay_samples': replay_samples,
            'precision_replay_samples': precision_replay_samples,
            'elite_projection_replay_samples': (
                elite_projection_replay_samples),
            'per_task_elite_samples': per_task_elite_samples,
            'per_task_elite_improvements': (
                per_task_elite_improvements),
            'per_task_elite_size': (
                len(per_task_elite_memory)
                if per_task_elite_memory is not None else 0),
            'per_task_elite_coverage': (
                per_task_elite_memory.coverage
                if per_task_elite_memory is not None else 0.0),
            'critic_updates': global_backward_update,
            'actor_updates': global_actor_update,
            'precision_only_actor_updates': global_precision_update,
            'elite_projection_actor_updates': (
                global_elite_projection_update),
            'per_task_elite_actor_updates': (
                global_per_task_elite_update),
            'per_task_elite_post_actor_updates': (
                global_per_task_elite_post_update),
            'paired_post_actor_updates': global_paired_post_update,
            'paired_target_samples': paired_target_samples,
            'paired_baseline_outcome_count': (
                len(paired_archive)
                if paired_archive is not None else 0),
            'paired_baseline_coverage': (
                paired_archive.coverage
                if paired_archive is not None else 0.0),
        }
        direct_payload = api.direct_seed_rl_checkpoint(
            actor=actor,
            critic=critic,
            config=rl_config,
            update_step=global_backward_update,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            metadata=metadata,
        )
        state = {
            'format': 'direct-seed-bidirectional-v1',
            'outer_round': outer_round,
            'phase': phase,
            'task_sampling': args.task_sampling,
            'projection_config': dataclasses.asdict(projection_config),
            'safe_task_fingerprint_list_sha256': (
                filter_manifest['kept_fingerprint_list_sha256']),
            'global_backward_update': global_backward_update,
            'global_actor_update': global_actor_update,
            'global_precision_update': global_precision_update,
            'global_elite_projection_update': (
                global_elite_projection_update),
            'global_per_task_elite_update': (
                global_per_task_elite_update),
            'global_per_task_elite_post_update': (
                global_per_task_elite_post_update),
            'global_paired_post_update': global_paired_post_update,
            'real_macro_rollouts': real_macro_rollouts,
            'replay_samples': replay_samples,
            'precision_replay_samples': precision_replay_samples,
            'elite_projection_replay_samples': (
                elite_projection_replay_samples),
            'per_task_elite_samples': per_task_elite_samples,
            'per_task_elite_improvements': (
                per_task_elite_improvements),
            'paired_target_samples': paired_target_samples,
            'controller_update_count': controller_update_count,
            'controller_version_critic_updates': (
                controller_version_critic_updates),
            'macro_replay_controller_update_count': (
                macro_replay_controller_update_count),
            'per_task_elite_controller_update_count': (
                per_task_elite_controller_update_count),
            'paired_archive_controller_update_count': (
                paired_archive_controller_update_count),
            'paired_collection_contract': (
                copy.deepcopy(paired_collection_contract)),
            'paired_baseline_actor_state_sha256': (
                paired_baseline_actor_state_sha256),
            'paired_explorer_provenance': (
                copy.deepcopy(paired_explorer_provenance)),
            'direct_seed': direct_payload,
            'controller': {
                name: value.detach().cpu()
                for name, value in controller.state_dict().items()
            },
            'controller_optimizer': controller_optimizer.state_dict(),
            'controller_scaler': (
                controller_scaler.state_dict()
                if controller_scaler is not None else None),
            'task_generator_state': task_generator.get_state(),
            'task_cycle_sampler': (
                task_cycle_sampler.state_dict()
                if task_cycle_sampler is not None else None),
            'collection_generator_state': (
                collection_generator.get_state()),
            'actor_update_generator_state': (
                actor_update_generator.get_state()),
            'macro_replay': (
                macro_replay.state_dict()
                if macro_replay is not None else None),
            'per_task_elite_memory': (
                per_task_elite_memory.state_dict()
                if per_task_elite_memory is not None else None),
            'paired_archive': (
                paired_archive.state_dict()
                if paired_archive is not None else None),
            'task_cache': str(
                Path(args.task_cache).expanduser().resolve()),
            'init_controller_dir': str(init_controller_dir),
            'kept_task_indices': dataset.batch.task_indices.tolist(),
            'args': vars(args),
        }
        state.update(global_rng_state(device))
        atomic_torch_save(state, state_path)
        atomic_torch_save(direct_payload, out_dir / 'direct_seed.pt')
        atomic_torch_save(controller.state_dict(), out_dir / 'agent.pt')

    def backward_phase(outer_round: int) -> None:
        nonlocal global_backward_update, global_actor_update
        nonlocal global_precision_update
        nonlocal global_elite_projection_update
        nonlocal global_per_task_elite_update
        nonlocal global_per_task_elite_post_update
        nonlocal global_paired_post_update
        nonlocal controller_version_critic_updates
        nonlocal real_macro_rollouts, replay_samples
        nonlocal precision_replay_samples
        nonlocal elite_projection_replay_samples
        nonlocal per_task_elite_samples
        nonlocal per_task_elite_improvements
        nonlocal paired_target_samples
        nonlocal paired_baseline_actor_state_sha256
        frozen_controller = FrozenRLController(controller.eval())
        if (macro_replay is not None
                and macro_replay_controller_update_count
                != controller_update_count):
            raise RuntimeError(
                'macro replay contains returns from controller version '
                f'{macro_replay_controller_update_count}, but the current '
                f'controller version is {controller_update_count}')
        if (per_task_elite_memory is not None
                and per_task_elite_controller_update_count
                != controller_update_count):
            raise RuntimeError(
                'per-task elite memory contains returns from controller '
                f'version {per_task_elite_controller_update_count}, but '
                f'the current controller version is '
                f'{controller_update_count}')
        if (paired_archive is not None
                and paired_archive_controller_update_count
                != controller_update_count):
            raise RuntimeError(
                'paired archive contains returns from controller version '
                f'{paired_archive_controller_update_count}, but the current '
                f'controller version is {controller_update_count}')
        if paired_archive is not None:
            expected_contract = _paired_collection_contract(args)
            if paired_collection_contract != expected_contract:
                raise RuntimeError(
                    'paired collection contract changed after validation')
            current_actor_sha256 = _actor_state_sha256(actor)
            paired_task_count = _paired_archive_task_count(paired_archive)
            if len(paired_archive) == 0:
                paired_baseline_actor_state_sha256 = current_actor_sha256
            elif (len(paired_archive) < paired_task_count
                  and current_actor_sha256
                  != paired_baseline_actor_state_sha256):
                raise RuntimeError(
                    'partial paired archive cannot continue because the '
                    'current actor differs from its baseline actor')
        for local_update in range(
                1, args.backward_updates_per_round + 1):
            if task_cycle_sampler is None:
                cpu_batch = dataset.sample(
                    args.seed_tasks_per_update, task_generator)
            else:
                task_rows = task_cycle_sampler.sample(
                    args.seed_tasks_per_update)
                cpu_batch = dataset.batch.index_select(task_rows)
            task_batch = cpu_batch.to(
                device, dtype=seed_env.kin.dtype)
            actor.train()

            _synchronise(device)
            before = time.perf_counter()
            with torch.no_grad():
                noise_fraction = min(
                    real_macro_rollouts
                    / max(1, args.collection_noise_anneal_rollouts),
                    1.0)
                collection_noise_scale = (
                    args.collection_noise_initial
                    + noise_fraction
                    * (args.collection_noise_final
                       - args.collection_noise_initial))
                if args.deterministic_backward:
                    collection_noise_scale = 0.0
                action = actor.sample(
                    task_batch.task,
                    deterministic=args.deterministic_backward,
                    generator=collection_generator,
                    noise_scale=collection_noise_scale)
                q_raw = action.q
            _synchronise(device)
            generator_s = time.perf_counter() - before

            before = time.perf_counter()
            routed = route_generated_seed(
                seed_env.kin, seed_env.collision, q_raw,
                task_batch.p0, task_batch.line_dir,
                task_batch.n_target, task_batch.fallback_q,
                projection_config)
            _synchronise(device)
            router_s = time.perf_counter() - before
            if not bool(routed.valid.all()):
                bad = torch.nonzero(
                    ~routed.valid, as_tuple=False).flatten().cpu().tolist()
                raise RuntimeError(
                    'source-filtered fallback invariant failed during '
                    f'backward collection, rows={bad[:20]}')

            selection = SeedSelection(
                q0=routed.q,
                p0=task_batch.p0,
                line_dir=task_batch.line_dir,
                n_target=task_batch.n_target,
            )
            before = time.perf_counter()
            rollout = rollout_seed_selection(
                seed_env, selection, frozen_controller, gamma=gamma)
            _synchronise(device)
            controller_s = time.perf_counter() - before

            rl_batch = api.DirectSeedRLBatch(
                task=task_batch.task.detach(),
                q_raw=routed.q_raw.detach(),
                q_projected=routed.q_projected.detach(),
                fallback_q=task_batch.fallback_q.detach(),
                progress_m=rollout.progress_m.detach(),
                route=routed.route.detach(),
            )
            real_macro_rollouts += rl_batch.batch_size
            batch_per_task_elite_improvements = 0
            before = time.perf_counter()
            if per_task_elite_memory is not None:
                batch_per_task_elite_improvements = (
                    per_task_elite_memory.update(
                        task_batch.task_indices, rl_batch))
                per_task_elite_improvements += (
                    batch_per_task_elite_improvements)
            per_task_elite_memory_update_s = (
                time.perf_counter() - before)
            before = time.perf_counter()
            batch_paired_baseline_insertions = 0
            if paired_archive is not None:
                batch_paired_baseline_insertions = paired_archive.update(
                    task_batch.task_indices, rl_batch)
            paired_archive_update_s = time.perf_counter() - before
            if macro_replay is not None:
                macro_replay.add(rl_batch)
            before = time.perf_counter()
            update_values = []
            precision_update_values = []
            elite_projection_values = []
            per_task_elite_values = []
            batch_replay_samples = 0
            batch_precision_replay_samples = 0
            batch_elite_projection_samples = 0
            batch_per_task_elite_samples = 0
            batch_actor_updates = 0
            replay_ready = (
                macro_replay is None
                or len(macro_replay) >= args.macro_replay_warmup)
            n_gradient_updates = (
                args.gradient_updates_per_rollout
                if replay_ready else 0)
            for _ in range(n_gradient_updates):
                if macro_replay is None:
                    update_batch = rl_batch
                else:
                    update_batch = macro_replay.sample(
                        args.macro_replay_batch_size)
                    batch_replay_samples += update_batch.batch_size
                    replay_samples += update_batch.batch_size
                critic_warmup_complete = (
                    controller_version_critic_updates
                    >= args.critic_warmup_updates_after_controller_change)
                actor_due = (
                    not args.freeze_seed_actor_during_collection
                    and critic_warmup_complete
                    and (global_backward_update + 1)
                    % args.seed_actor_update_period == 0)
                values = api.update_direct_seed_rl(
                    actor, critic,
                    actor_optimizer, critic_optimizer,
                    update_batch, seed_env.kin, rl_config,
                    collision=seed_env.collision,
                    update_actor=actor_due,
                    generator=actor_update_generator)
                update_values.append(values)
                global_backward_update += 1
                controller_version_critic_updates += 1
                if actor_due:
                    global_actor_update += 1
                    batch_actor_updates += 1
            for _ in range(args.precision_only_updates_per_rollout):
                if macro_replay is None:
                    precision_batch = rl_batch
                else:
                    precision_batch = macro_replay.sample(
                        args.macro_replay_batch_size)
                    batch_precision_replay_samples += (
                        precision_batch.batch_size)
                    precision_replay_samples += (
                        precision_batch.batch_size)
                precision_update_values.append(
                    api.update_direct_seed_precision(
                        actor, actor_optimizer, precision_batch,
                        seed_env.kin, rl_config,
                        collision=seed_env.collision,
                        projection_weight=(
                            args.precision_only_projection_weight)))
                global_precision_update += 1
            elite_ready = (
                replay_ready
                and macro_replay is not None
                and bool((
                    macro_replay.route[:len(macro_replay)]
                    == ROUTE_REFINED).any())
            )
            n_elite_updates = (
                args.elite_projection_updates_per_rollout
                if elite_ready else 0)
            for _ in range(n_elite_updates):
                elite_batch = macro_replay.sample_elite(
                    args.macro_replay_batch_size,
                    args.elite_projection_fraction)
                batch_elite_projection_samples += elite_batch.batch_size
                elite_projection_replay_samples += elite_batch.batch_size
                elite_projection_values.append(
                    api.update_direct_seed_projection(
                        actor, actor_optimizer, elite_batch,
                        gradient_clip_norm=(
                            rl_config.gradient_clip_norm)))
                global_elite_projection_update += 1
            per_task_elite_ready = (
                replay_ready
                and per_task_elite_memory is not None
                and len(per_task_elite_memory) > 0)
            n_per_task_elite_updates = (
                args.per_task_elite_updates_per_rollout
                if per_task_elite_ready else 0)
            for _ in range(n_per_task_elite_updates):
                per_task_elite_batch = per_task_elite_memory.sample(
                    args.macro_replay_batch_size,
                    device=device, dtype=seed_env.kin.dtype)
                batch_per_task_elite_samples += (
                    per_task_elite_batch.batch_size)
                per_task_elite_samples += (
                    per_task_elite_batch.batch_size)
                per_task_elite_values.append(
                    api.update_direct_seed_projection(
                        actor, actor_optimizer, per_task_elite_batch,
                        gradient_clip_norm=(
                            rl_config.gradient_clip_norm)))
                global_per_task_elite_update += 1
            _synchronise(device)
            update_s = time.perf_counter() - before

            actor_update_values = [
                value for value in update_values
                if value.get('actor_updated') == 1.0
            ]
            metrics = {
                'phase': 'backward_seed',
                'outer_round': outer_round,
                'local_update': local_update,
                'global_backward_update': global_backward_update,
                'task_sampling': args.task_sampling,
                **_route_rollout_metrics(routed, rollout),
                'rl': _mean_dict(update_values),
                'actor_rl': _mean_dict(actor_update_values),
                'precision_only_rl': _mean_dict(
                    precision_update_values),
                'elite_projection_rl': _mean_dict(
                    elite_projection_values),
                'per_task_elite_rl': _mean_dict(
                    per_task_elite_values),
                'macro_training': {
                    'real_rollouts_this_batch': rl_batch.batch_size,
                    'real_rollouts_total': real_macro_rollouts,
                    'replay_enabled': macro_replay is not None,
                    'replay_ready': replay_ready,
                    'replay_size': (
                        len(macro_replay)
                        if macro_replay is not None else 0),
                    'replay_samples_this_batch': batch_replay_samples,
                    'replay_samples_total': replay_samples,
                    'precision_replay_samples_this_batch': (
                        batch_precision_replay_samples),
                    'precision_replay_samples_total': (
                        precision_replay_samples),
                    'elite_projection_replay_samples_this_batch': (
                        batch_elite_projection_samples),
                    'elite_projection_replay_samples_total': (
                        elite_projection_replay_samples),
                    'per_task_elite_improvements_this_batch': (
                        batch_per_task_elite_improvements),
                    'per_task_elite_improvements_total': (
                        per_task_elite_improvements),
                    'paired_baseline_insertions_this_batch': (
                        batch_paired_baseline_insertions),
                    'paired_baseline_outcome_count': (
                        len(paired_archive)
                        if paired_archive is not None else 0),
                    'paired_baseline_coverage': (
                        paired_archive.coverage
                        if paired_archive is not None else 0.0),
                    'per_task_elite_size': (
                        len(per_task_elite_memory)
                        if per_task_elite_memory is not None else 0),
                    'per_task_elite_coverage': (
                        per_task_elite_memory.coverage
                        if per_task_elite_memory is not None else 0.0),
                    'per_task_elite_samples_this_batch': (
                        batch_per_task_elite_samples),
                    'per_task_elite_samples_total': (
                        per_task_elite_samples),
                    'critic_updates_this_batch': len(update_values),
                    'critic_updates_total': global_backward_update,
                    'critic_updates_since_controller_change': (
                        controller_version_critic_updates),
                    'critic_warmup_updates_required': (
                        args.critic_warmup_updates_after_controller_change),
                    'critic_warmup_complete': (
                        controller_version_critic_updates
                        >= args.critic_warmup_updates_after_controller_change),
                    'actor_updates_this_batch': batch_actor_updates,
                    'actor_updates_total': global_actor_update,
                    'precision_only_updates_this_batch': (
                        len(precision_update_values)),
                    'precision_only_updates_total': (
                        global_precision_update),
                    'elite_projection_ready': elite_ready,
                    'elite_projection_updates_this_batch': (
                        len(elite_projection_values)),
                    'elite_projection_updates_total': (
                        global_elite_projection_update),
                    'per_task_elite_ready': (
                        per_task_elite_ready),
                    'per_task_elite_updates_this_batch': (
                        len(per_task_elite_values)),
                    'per_task_elite_updates_total': (
                        global_per_task_elite_update),
                    'replay_samples_per_real_rollout': (
                        batch_replay_samples / rl_batch.batch_size),
                    'precision_replay_samples_per_real_rollout': (
                        batch_precision_replay_samples
                        / rl_batch.batch_size),
                    'elite_projection_samples_per_real_rollout': (
                        batch_elite_projection_samples
                        / rl_batch.batch_size),
                    'per_task_elite_samples_per_real_rollout': (
                        batch_per_task_elite_samples
                        / rl_batch.batch_size),
                    'collection_noise_scale': collection_noise_scale,
                },
                'timing_s': {
                    'generator': generator_s,
                    'router_and_at_most_one_ik': router_s,
                    'controller_rollout': controller_s,
                    'per_task_elite_memory_update': (
                        per_task_elite_memory_update_s),
                    'paired_archive_update': paired_archive_update_s,
                    'gradient_updates': update_s,
                },
            }
            logger.write(metrics)
            if local_update == 1 or local_update % 10 == 0:
                route_fraction = metrics['route_fraction']
                print(
                    f'[seed-r{outer_round}] '
                    f'{local_update:>4}/'
                    f'{args.backward_updates_per_round}  '
                    f'progress={metrics["progress_mean_m"]:.4f}m  '
                    f'D/R/F='
                    f'{route_fraction["direct"]:.1%}/'
                    f'{route_fraction["refined"]:.1%}/'
                    f'{route_fraction["fallback"]:.1%}',
                    flush=True)

        n_paired_post_updates = args.paired_post_updates_per_round
        if n_paired_post_updates > 0:
            if paired_archive is None:
                raise RuntimeError(
                    'paired post-updates require a paired baseline archive')
            _require_full_paired_archive(paired_archive)
            if (paired_explorer_memory is None
                    or paired_explorer_provenance is None):
                raise RuntimeError(
                    'paired post-updates require a validated external '
                    'explorer checkpoint')
            current_controller_sha256 = state_dict_fingerprint(
                dict(controller.state_dict()))
            if current_controller_sha256 != paired_explorer_provenance[
                    'controller_state_sha256']:
                raise RuntimeError(
                    'current controller changed after paired explorer '
                    'validation')
            paired_stats = paired_archive.target_stats(
                paired_explorer_memory,
                advantage_margin_m=args.paired_advantage_margin_m)
            if paired_stats['target_count'] < 1:
                raise RuntimeError(
                    'paired baseline/explorer comparison produced no legal '
                    'projection targets')
            actor.train()
            paired_reference_actor = None
            if args.per_task_elite_post_anchor_weight > 0.0:
                paired_reference_actor = copy.deepcopy(actor).eval()
                for parameter in paired_reference_actor.parameters():
                    parameter.requires_grad_(False)
            _synchronise(device)
            paired_post_start = time.perf_counter()
            paired_window_values = []
            for local_post_update in range(
                    1, n_paired_post_updates + 1):
                post_batch = paired_archive.sample_targets(
                    paired_explorer_memory,
                    args.macro_replay_batch_size,
                    advantage_margin_m=args.paired_advantage_margin_m,
                    device=device, dtype=seed_env.kin.dtype)
                reference_q = None
                if paired_reference_actor is not None:
                    with torch.no_grad():
                        reference_q = paired_reference_actor.mean_q(
                            post_batch.task)
                values = api.update_direct_seed_projection(
                    actor, actor_optimizer, post_batch,
                    gradient_clip_norm=rl_config.gradient_clip_norm,
                    reference_q=reference_q,
                    anchor_weight=(
                        args.per_task_elite_post_anchor_weight))
                paired_window_values.append(values)
                paired_target_samples += post_batch.batch_size
                global_paired_post_update += 1
                should_log = (
                    local_post_update == 1
                    or local_post_update % 100 == 0
                    or local_post_update == n_paired_post_updates)
                if should_log:
                    mean_values = _mean_dict(paired_window_values)
                    logger.write({
                        'phase': 'backward_seed_post_paired',
                        'outer_round': outer_round,
                        'local_post_update': local_post_update,
                        'requested_post_updates': (
                            n_paired_post_updates),
                        'post_update_window': mean_values,
                        'paired_target_stats': paired_stats,
                        'paired_target_samples_total': (
                            paired_target_samples),
                        'paired_post_updates_total': (
                            global_paired_post_update),
                        'real_rollouts_added': 0,
                        'snapshot_anchor_weight': (
                            args.per_task_elite_post_anchor_weight),
                    })
                    print(
                        f'[paired-post-r{outer_round}] '
                        f'{local_post_update:>5}/'
                        f'{n_paired_post_updates}  '
                        f'loss='
                        f'{mean_values.get("projection_actor_loss", 0.0):.6f}',
                        flush=True)
                    paired_window_values = []
            _synchronise(device)
            logger.write({
                'phase': 'backward_seed_post_paired_complete',
                'outer_round': outer_round,
                'post_updates': n_paired_post_updates,
                'wall_s': time.perf_counter() - paired_post_start,
                'paired_target_stats': paired_stats,
                'paired_target_samples_total': paired_target_samples,
                'real_rollouts_added': 0,
                'snapshot_anchor_weight': (
                    args.per_task_elite_post_anchor_weight),
            })
            return

        n_post_updates = args.per_task_elite_post_updates_per_round
        if n_post_updates == 0:
            return
        if (per_task_elite_memory is None
                or len(per_task_elite_memory) == 0):
            raise RuntimeError(
                'per-task elite post-updates require at least one '
                'successful online refined target')
        actor.train()
        post_reference_actor = None
        if args.per_task_elite_post_anchor_weight > 0.0:
            post_reference_actor = copy.deepcopy(actor).eval()
            for parameter in post_reference_actor.parameters():
                parameter.requires_grad_(False)
        _synchronise(device)
        post_start = time.perf_counter()
        window_values = []
        for local_post_update in range(1, n_post_updates + 1):
            post_batch = per_task_elite_memory.sample(
                args.macro_replay_batch_size,
                device=device, dtype=seed_env.kin.dtype)
            reference_q = None
            if post_reference_actor is not None:
                with torch.no_grad():
                    reference_q = post_reference_actor.mean_q(
                        post_batch.task)
            values = api.update_direct_seed_projection(
                actor, actor_optimizer, post_batch,
                gradient_clip_norm=rl_config.gradient_clip_norm,
                reference_q=reference_q,
                anchor_weight=(
                    args.per_task_elite_post_anchor_weight))
            window_values.append(values)
            per_task_elite_samples += post_batch.batch_size
            global_per_task_elite_update += 1
            global_per_task_elite_post_update += 1
            should_log = (
                local_post_update == 1
                or local_post_update % 100 == 0
                or local_post_update == n_post_updates)
            if should_log:
                mean_values = _mean_dict(window_values)
                logger.write({
                    'phase': 'backward_seed_post_elite',
                    'outer_round': outer_round,
                    'local_post_update': local_post_update,
                    'requested_post_updates': n_post_updates,
                    'post_update_window': mean_values,
                    'per_task_elite_size': len(per_task_elite_memory),
                    'per_task_elite_coverage': (
                        per_task_elite_memory.coverage),
                    'per_task_elite_samples_total': (
                        per_task_elite_samples),
                    'per_task_elite_updates_total': (
                        global_per_task_elite_update),
                    'per_task_elite_post_updates_total': (
                        global_per_task_elite_post_update),
                    'real_rollouts_added': 0,
                    'snapshot_anchor_weight': (
                        args.per_task_elite_post_anchor_weight),
                })
                print(
                    f'[elite-post-r{outer_round}] '
                    f'{local_post_update:>5}/{n_post_updates}  '
                    f'loss='
                    f'{mean_values.get("projection_actor_loss", 0.0):.6f}',
                    flush=True)
                window_values = []
        _synchronise(device)
        logger.write({
            'phase': 'backward_seed_post_elite_complete',
            'outer_round': outer_round,
            'post_updates': n_post_updates,
            'wall_s': time.perf_counter() - post_start,
            'real_rollouts_added': 0,
            'snapshot_anchor_weight': (
                args.per_task_elite_post_anchor_weight),
        })

    def forward_phase(outer_round: int) -> bool:
        if (args.forward_mode == 'skip'
                or args.controller_steps_per_round == 0):
            logger.write({
                'phase': 'forward_controller',
                'outer_round': outer_round,
                'mode': 'skip',
                'controller_steps': 0,
            })
            return False
        batch_steps = controller_cfg.n_steps * args.controller_n_envs
        effective_steps = (
            args.controller_steps_per_round // batch_steps) * batch_steps
        if effective_steps < batch_steps:
            raise ValueError(
                '--controller-steps-per-round must contain at least one '
                f'PPO batch ({batch_steps} transitions)')
        frozen_actor = copy.deepcopy(actor).eval()
        for parameter in frozen_actor.parameters():
            parameter.requires_grad_(False)
        forward_env.line_dist = DirectSeedLineDistribution(
            dataset, frozen_actor,
            forward_env.kin, forward_env.collision,
            projection_config,
            fallback_probability=(
                args.controller_fallback_probability),
            seed=args.seed + 1000 * outer_round,
        )
        round_cfg = dataclasses.replace(
            controller_cfg,
            total_timesteps=args.controller_steps_per_round,
        )
        phase_start = time.perf_counter()

        def controller_log(stats: dict[str, Any]) -> None:
            if 'update' not in stats:
                return
            payload = {
                'phase': 'forward_controller',
                'outer_round': outer_round,
                **stats,
            }
            logger.write(payload)
            if stats['update'] == 1 or stats['update'] % 20 == 0:
                print(
                    f'[ctrl-r{outer_round}] '
                    f'upd={stats["update"]:>4}  '
                    f'progress='
                    f'{stats.get("reward/progress", 0.0):.3f}  '
                    f'kl={stats.get("train/approx_kl", 0.0):.4f}',
                    flush=True)

        controller.train()
        ppo_train(
            round_cfg, forward_env, device,
            log_fn=controller_log,
            agent=controller,
            optimizer=controller_optimizer,
            reward_scaler=controller_scaler,
        )
        logger.write({
            'phase': 'forward_controller_complete',
            'outer_round': outer_round,
            'requested_controller_steps': (
                args.controller_steps_per_round),
            'effective_controller_steps': effective_steps,
            'wall_s': time.perf_counter() - phase_start,
        })
        return True

    try:
        for outer_round in range(
                start_round, args.outer_rounds + 1):
            print(
                f'[direct-bidir] ===== round {outer_round}: '
                'backward seed =====',
                flush=True)
            if not (
                    resume_after_backward
                    and outer_round == start_round):
                backward_phase(outer_round)
                save(outer_round, 'backward_complete')
            else:
                print(
                    '[direct-bidir] backward phase already complete',
                    flush=True)

            print(
                f'[direct-bidir] ===== round {outer_round}: '
                'forward controller =====',
                flush=True)
            controller_changed = forward_phase(outer_round)
            if controller_changed:
                controller_update_count += 1
                macro_replay_controller_update_count = (
                    controller_update_count)
                per_task_elite_controller_update_count = (
                    controller_update_count)
                paired_archive_controller_update_count = (
                    controller_update_count)
                critic = api.TwinMacroQ(
                    seed_env.kin.lmt_lo, seed_env.kin.lmt_up,
                    critic_config,
                    task_mean=task_mean, task_std=task_std).to(device)
                critic_optimizer = torch.optim.Adam(
                    critic.parameters(), lr=args.critic_lr)
                controller_version_critic_updates = 0
                logger.write({
                    'phase': 'critic_reset_after_controller',
                    'outer_round': outer_round,
                    'reason': 'controller_return_function_changed',
                    'controller_update_count': controller_update_count,
                    'critic_warmup_updates_required': (
                        args.critic_warmup_updates_after_controller_change),
                })
            if controller_changed and macro_replay is not None:
                stale_count = len(macro_replay)
                macro_replay.clear()
                logger.write({
                    'phase': 'macro_replay_reset',
                    'outer_round': outer_round,
                    'reason': 'controller_policy_updated',
                    'discarded_stale_returns': stale_count,
                    'replay_size_after': len(macro_replay),
                    'controller_update_count': (
                        controller_update_count),
                    'macro_replay_controller_update_count': (
                        macro_replay_controller_update_count),
                })
            if controller_changed and per_task_elite_memory is not None:
                stale_count = len(per_task_elite_memory)
                per_task_elite_memory.clear()
                logger.write({
                    'phase': 'per_task_elite_reset',
                    'outer_round': outer_round,
                    'reason': 'controller_policy_updated',
                    'discarded_stale_returns': stale_count,
                    'elite_size_after': len(per_task_elite_memory),
                    'controller_update_count': (
                        controller_update_count),
                    'per_task_elite_controller_update_count': (
                        per_task_elite_controller_update_count),
                })
            if controller_changed and paired_archive is not None:
                stale_count = len(paired_archive)
                previous_baseline_actor_sha256 = (
                    paired_baseline_actor_state_sha256)
                paired_archive.clear()
                paired_baseline_actor_state_sha256 = (
                    _actor_state_sha256(actor))
                logger.write({
                    'phase': 'paired_archive_reset',
                    'outer_round': outer_round,
                    'reason': 'controller_policy_updated',
                    'discarded_stale_returns': stale_count,
                    'paired_archive_size_after': len(paired_archive),
                    'controller_update_count': (
                        controller_update_count),
                    'paired_archive_controller_update_count': (
                        paired_archive_controller_update_count),
                    'previous_baseline_actor_state_sha256': (
                        previous_baseline_actor_sha256),
                    'next_baseline_actor_state_sha256': (
                        paired_baseline_actor_state_sha256),
                })
            if controller_changed and paired_explorer_memory is not None:
                logger.write({
                    'phase': 'paired_explorer_invalidated',
                    'outer_round': outer_round,
                    'reason': 'controller_policy_updated',
                    'previous_provenance': paired_explorer_provenance,
                })
                paired_explorer_memory = None
                paired_explorer_provenance = None
            save(outer_round, 'round_complete')
            resume_after_backward = False
            print(
                f'[direct-bidir] round {outer_round} saved -> '
                f'{state_path}',
                flush=True)
    finally:
        logger.close()


if __name__ == '__main__':
    main()


__all__ = [
    'DirectSeedLineDistribution',
    'DirectTaskBatch',
    'DirectTaskCycleSampler',
    'DirectTaskDataset',
    'DirectSeedRLAPI',
    'main',
]
