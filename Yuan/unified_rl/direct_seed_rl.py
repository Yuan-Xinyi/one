"""One-step contextual RL for generating a single continuous seed.

This is the main learning path for direct seed generation.  A task-conditioned
tanh-Gaussian actor emits one raw joint vector.  The fixed deployment router
then either executes it directly, refines it once with IK, or uses the safe
fallback.  The resulting *real controller progress* is the Monte-Carlo target
for twin macro-Q critics.

There is no Bellman bootstrap: seed choice is one contextual macro-action
whose delayed return is the complete downstream controller episode.  Actor
updates maximize the conservative twin-Q estimate and add only physically
interpretable precision terms.  Successful IK projections self-distil back
into the raw actor, gradually moving inference from REFINE to DIRECT.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Yuan.unified_rl.direct_seed_projection import (
    ROUTE_DIRECT,
    ROUTE_FALLBACK,
    ROUTE_INVALID,
    ROUTE_REFINED,
)


@dataclass(frozen=True)
class DirectSeedActorConfig:
    task_dim: int = 9
    q_dim: int = 7
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    log_std_min: float = -5.0
    log_std_max: float = 0.5
    limit_fraction: float = 0.98

    def __post_init__(self) -> None:
        if self.task_dim != 9 or self.q_dim != 7:
            raise ValueError('direct seed actor requires task_dim=9, q_dim=7')
        if self.hidden_dim < 8 or self.n_hidden_layers < 1:
            raise ValueError('actor hidden dimensions must be positive')
        if (not math.isfinite(self.log_std_min)
                or not math.isfinite(self.log_std_max)
                or self.log_std_min >= self.log_std_max):
            raise ValueError('actor log-std bounds must be finite and ordered')
        if (not math.isfinite(self.limit_fraction)
                or not 0.0 < self.limit_fraction < 1.0):
            raise ValueError('limit_fraction must be finite and in (0, 1)')


@dataclass(frozen=True)
class DirectSeedMoEActorConfig:
    """Hard-gated deterministic mixture used only for direct seed output."""

    task_dim: int = 9
    q_dim: int = 7
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    n_experts: int = 4
    limit_fraction: float = 0.98
    # Converted single actors retain their original 14-output Gaussian head
    # as expert 0.  This matters beyond a nominally tiny numerical error:
    # changing the GEMM from 14 to 7 outputs can change a downstream IK
    # branch.  False is the backward-compatible layout used by v1 payloads
    # that predate this field.
    exact_baseline_head: bool = False
    # Zero preserves the original single Linear gate, including its state-dict
    # keys and floating-point path.  A positive width enables a small
    # task-only nonlinear gate without changing any expert or deployment
    # routing semantics.
    gate_hidden_dim: int = 0

    def __post_init__(self) -> None:
        if self.task_dim != 9 or self.q_dim != 7:
            raise ValueError(
                'direct seed MoE requires task_dim=9, q_dim=7')
        if self.hidden_dim < 8 or self.n_hidden_layers < 1:
            raise ValueError('MoE hidden dimensions must be positive')
        if self.n_experts < 1:
            raise ValueError('n_experts must be positive')
        if not isinstance(self.exact_baseline_head, bool):
            raise TypeError('exact_baseline_head must be a bool')
        if isinstance(self.gate_hidden_dim, bool) \
                or not isinstance(self.gate_hidden_dim, int):
            raise TypeError('gate_hidden_dim must be an integer')
        if self.gate_hidden_dim < 0:
            raise ValueError('gate_hidden_dim must be non-negative')
        if (not math.isfinite(self.limit_fraction)
                or not 0.0 < self.limit_fraction < 1.0):
            raise ValueError('limit_fraction must be finite and in (0, 1)')


@dataclass(frozen=True)
class DirectSeedCriticConfig:
    task_dim: int = 9
    q_dim: int = 7
    hidden_dim: int = 256
    n_hidden_layers: int = 3

    def __post_init__(self) -> None:
        if self.task_dim != 9 or self.q_dim != 7:
            raise ValueError('direct seed critic requires task_dim=9, q_dim=7')
        if self.hidden_dim < 8 or self.n_hidden_layers < 1:
            raise ValueError('critic hidden dimensions must be positive')


@dataclass(frozen=True)
class DirectSeedRLConfig:
    """Loss weights for one contextual macro-action update."""

    critic_huber_delta_m: float = 0.05
    entropy_coef: float = 1e-3
    precision_weight: float = 0.01
    position_scale_m: float = 0.01
    cone_deg: float = 30.0
    cone_scale: float = 0.05
    joint_margin_rad: float = 0.02
    joint_margin_scale_rad: float = 0.02
    collision_margin_m: float = 0.0
    collision_scale_m: float = 0.01
    collision_precision_weight: float = 1.0
    projection_distill_weight: float = 0.25
    # Kept as an explicit ablation only.  The main method must improve from
    # downstream RL return and the actor's own one-step IK projection, rather
    # than imitate the externally supplied fail-closed fallback.
    fallback_distill_weight: float = 0.0
    failure_precision_weight: float = 0.05
    behavior_anchor_weight: float = 0.0
    refine_route_penalty_m: float = 0.0
    fallback_route_penalty_m: float = 0.0
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        positive = {
            'critic_huber_delta_m': self.critic_huber_delta_m,
            'position_scale_m': self.position_scale_m,
            'cone_scale': self.cone_scale,
            'joint_margin_scale_rad': self.joint_margin_scale_rad,
            'collision_scale_m': self.collision_scale_m,
            'gradient_clip_norm': self.gradient_clip_norm,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
        nonnegative = {
            'entropy_coef': self.entropy_coef,
            'precision_weight': self.precision_weight,
            'joint_margin_rad': self.joint_margin_rad,
            'collision_margin_m': self.collision_margin_m,
            'collision_precision_weight': self.collision_precision_weight,
            'projection_distill_weight': self.projection_distill_weight,
            'fallback_distill_weight': self.fallback_distill_weight,
            'failure_precision_weight': self.failure_precision_weight,
            'behavior_anchor_weight': self.behavior_anchor_weight,
            'refine_route_penalty_m': self.refine_route_penalty_m,
            'fallback_route_penalty_m': self.fallback_route_penalty_m,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
        if (not math.isfinite(self.cone_deg)
                or not 0.0 < self.cone_deg <= 30.0):
            raise ValueError('cone_deg must be finite and in (0, 30]')


@dataclass(frozen=True)
class DirectSeedAction:
    q: torch.Tensor
    normalized_q: torch.Tensor
    pre_tanh: torch.Tensor
    log_prob: torch.Tensor


@dataclass(frozen=True)
class DirectSeedRLBatch:
    """One macro-transition per task.

    ``q_raw`` is the action presented to the fixed router and therefore the
    critic input. ``progress_m`` is the complete real-controller progress
    after DIRECT/REFINED/FALLBACK routing.  ``q_projected`` and ``fallback_q``
    provide supervised directions only for the corresponding route masks.
    """

    task: torch.Tensor
    q_raw: torch.Tensor
    q_projected: torch.Tensor
    fallback_q: torch.Tensor
    progress_m: torch.Tensor
    route: torch.Tensor

    def __post_init__(self) -> None:
        if self.task.ndim != 2 or self.task.shape[-1] != 9:
            raise ValueError('task must have shape (B, 9)')
        batch_size = self.task.shape[0]
        for name in ('q_raw', 'q_projected', 'fallback_q'):
            value = getattr(self, name)
            if value.shape != (batch_size, 7):
                raise ValueError(f'{name} must have shape (B, 7)')
        if self.progress_m.shape != (batch_size,):
            raise ValueError('progress_m must have shape (B,)')
        if self.route.shape != (batch_size,):
            raise ValueError('route must have shape (B,)')
        tensors = (
            self.task, self.q_raw, self.q_projected,
            self.fallback_q, self.progress_m)
        if not all(torch.is_floating_point(value) for value in tensors):
            raise TypeError('task, q, and progress tensors must be floating')
        if not all(value.device == self.task.device for value in tensors):
            raise ValueError('all batch tensors must share one device')
        if not all(value.dtype == self.task.dtype for value in tensors):
            raise ValueError('all floating batch tensors must share one dtype')
        if self.route.device != self.task.device:
            raise ValueError('route must share the batch device')
        if self.route.dtype == torch.bool \
                or self.route.dtype not in (
                    torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError('route must have an integer dtype')
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError('direct seed RL batch must be finite')
        known = torch.zeros_like(self.route, dtype=torch.bool)
        for code in (
                ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_INVALID):
            known |= self.route == code
        if not bool(known.all()):
            raise ValueError('route contains an unknown code')

    @property
    def batch_size(self) -> int:
        return int(self.task.shape[0])

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> 'DirectSeedRLBatch':
        target_dtype = self.task.dtype if dtype is None else dtype
        return DirectSeedRLBatch(
            task=self.task.to(device=device, dtype=target_dtype),
            q_raw=self.q_raw.to(device=device, dtype=target_dtype),
            q_projected=self.q_projected.to(
                device=device, dtype=target_dtype),
            fallback_q=self.fallback_q.to(
                device=device, dtype=target_dtype),
            progress_m=self.progress_m.to(
                device=device, dtype=target_dtype),
            route=self.route.to(device=device),
        )


class DirectSeedEliteMemory:
    """Per-task best successful online projection stored on CPU.

    Each configured task owns at most one target: the ``ROUTE_REFINED``
    projection with the highest observed *real* controller progress.  Sampling
    is uniform over tasks with a valid target (not over update frequency), so
    frequently visited tasks cannot dominate projection self-distillation.
    """

    _INTEGER_DTYPES = (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)

    def __init__(
        self,
        task_ids: torch.Tensor,
        seed: int = 0,
    ):
        ids = self._validate_task_ids(
            task_ids, require_unique=True, allow_empty=False)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('elite memory seed must be an integer')
        if not -(1 << 63) <= seed < (1 << 64):
            raise ValueError('elite memory seed is outside manual_seed range')

        self.task_ids = ids
        self._id_to_index = {
            int(task_id): index
            for index, task_id in enumerate(self.task_ids.tolist())
        }
        task_count = int(self.task_ids.numel())
        self.progress_m = torch.zeros(
            task_count, dtype=torch.float32, device='cpu')
        self.task = torch.zeros(
            (task_count, 9), dtype=torch.float32, device='cpu')
        self.q_projected = torch.zeros(
            (task_count, 7), dtype=torch.float32, device='cpu')
        self.valid = torch.zeros(
            task_count, dtype=torch.bool, device='cpu')
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(seed)

    @classmethod
    def _validate_task_ids(
        cls,
        task_ids: torch.Tensor,
        *,
        require_unique: bool,
        allow_empty: bool,
    ) -> torch.Tensor:
        if not torch.is_tensor(task_ids):
            try:
                task_ids = torch.as_tensor(task_ids)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    'task_ids must be an integer tensor or array') from error
        if task_ids.ndim != 1:
            raise ValueError('task_ids must be one-dimensional')
        if task_ids.dtype not in cls._INTEGER_DTYPES:
            raise TypeError('task_ids must have an integer dtype')
        ids = task_ids.detach().to(
            device='cpu', dtype=torch.int64).clone()
        if not allow_empty and ids.numel() < 1:
            raise ValueError('task_ids must not be empty')
        if require_unique and torch.unique(ids).numel() != ids.numel():
            raise ValueError('task_ids must be unique')
        return ids

    def __len__(self) -> int:
        return int(torch.count_nonzero(self.valid))

    @property
    def coverage(self) -> float:
        """Fraction of configured tasks that currently own an elite."""
        return len(self) / int(self.task_ids.numel())

    @torch.no_grad()
    def clear(self) -> None:
        """Remove all elites without perturbing the private sampling RNG."""
        self.progress_m.zero_()
        self.task.zero_()
        self.q_projected.zero_()
        self.valid.zero_()

    @torch.no_grad()
    def update(
        self,
        task_ids: torch.Tensor,
        batch: DirectSeedRLBatch,
    ) -> int:
        """Keep the highest-progress refined projection for every task.

        Non-refined routes never supply projection supervision and are
        ignored.  Duplicate task IDs within one call are reduced by real
        progress before comparing against the cross-call incumbent.

        Returns:
            Number of task slots inserted or improved.
        """
        if not isinstance(batch, DirectSeedRLBatch):
            raise TypeError('batch must be a DirectSeedRLBatch')
        ids = self._validate_task_ids(
            task_ids, require_unique=False, allow_empty=True)
        if ids.numel() != batch.batch_size:
            raise ValueError(
                'task_ids length must equal the DirectSeedRLBatch size')
        unknown = sorted({
            int(task_id) for task_id in ids.tolist()
            if int(task_id) not in self._id_to_index
        })
        if unknown:
            raise ValueError(
                f'update contains unknown task_ids: {unknown[:8]}')

        route = batch.route.detach().to(device='cpu', dtype=torch.int64)
        progress = batch.progress_m.detach().to(
            device='cpu', dtype=torch.float32)
        task = batch.task.detach().to(device='cpu', dtype=torch.float32)
        q_projected = batch.q_projected.detach().to(
            device='cpu', dtype=torch.float32)

        # slot -> source row.  Strict ``>`` makes ties deterministic: the
        # earliest refined observation wins and incumbents survive ties.
        best_rows: dict[int, int] = {}
        for row, task_id in enumerate(ids.tolist()):
            if int(route[row]) != ROUTE_REFINED:
                continue
            slot = self._id_to_index[int(task_id)]
            incumbent_row = best_rows.get(slot)
            if (incumbent_row is None
                    or bool(progress[row] > progress[incumbent_row])):
                best_rows[slot] = row

        improved = 0
        for slot, row in best_rows.items():
            if self.valid[slot] and not bool(
                    progress[row] > self.progress_m[slot]):
                continue
            self.progress_m[slot].copy_(progress[row])
            self.task[slot].copy_(task[row])
            self.q_projected[slot].copy_(q_projected[row])
            self.valid[slot] = True
            improved += 1
        return improved

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> DirectSeedRLBatch:
        """Uniformly sample valid task elites with replacement."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError('elite memory sample size must be an integer')
        if batch_size < 1:
            raise ValueError('elite memory sample size must be positive')
        if not isinstance(dtype, torch.dtype) \
                or not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError('elite memory sample dtype must be floating point')
        target_device = torch.device(device)
        valid_index = torch.nonzero(
            self.valid, as_tuple=False).flatten()
        valid_count = int(valid_index.numel())
        if valid_count < 1:
            raise RuntimeError('cannot sample an empty elite memory')
        choice = torch.randint(
            valid_count, (batch_size,), generator=self.generator,
            device='cpu')
        index = valid_index.index_select(0, choice)
        projected = self.q_projected.index_select(0, index)
        cpu_batch = DirectSeedRLBatch(
            task=self.task.index_select(0, index),
            # These fields are not used by projection self-distillation.  A
            # projected q is nevertheless a legal finite dummy raw action.
            q_raw=projected.clone(),
            q_projected=projected,
            fallback_q=torch.zeros_like(projected),
            progress_m=self.progress_m.index_select(0, index),
            route=torch.full(
                (batch_size,), ROUTE_REFINED,
                dtype=torch.int64, device='cpu'),
        )
        return cpu_batch.to(device=target_device, dtype=dtype)

    def state_dict(self) -> dict:
        """Return a portable checkpoint including the private RNG stream."""
        return {
            'format': 'direct-seed-elite-memory-v1',
            'task_ids': self.task_ids.clone(),
            'valid_count': len(self),
            'generator_state': self.generator.get_state().cpu().clone(),
            'storage': {
                'progress_m': self.progress_m.clone(),
                'task': self.task.clone(),
                'q_projected': self.q_projected.clone(),
                'valid': self.valid.clone(),
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Strictly restore data and RNG without partial mutation on failure."""
        if not isinstance(state, Mapping):
            raise TypeError('elite memory checkpoint must be a mapping')
        if state.get('format') != 'direct-seed-elite-memory-v1':
            raise ValueError('unsupported direct-seed elite memory format')
        checkpoint_ids = state.get('task_ids')
        if (not torch.is_tensor(checkpoint_ids)
                or checkpoint_ids.ndim != 1
                or checkpoint_ids.dtype != torch.int64
                or not torch.equal(
                    checkpoint_ids.detach().cpu(), self.task_ids)):
            raise ValueError('elite memory checkpoint task_ids differ')
        storage = state.get('storage')
        if not isinstance(storage, Mapping):
            raise ValueError(
                'elite memory checkpoint has no storage mapping')
        task_count = int(self.task_ids.numel())
        expected = {
            'progress_m': ((task_count,), torch.float32),
            'task': ((task_count, 9), torch.float32),
            'q_projected': ((task_count, 7), torch.float32),
            'valid': ((task_count,), torch.bool),
        }
        checked: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in expected.items():
            value = storage.get(name)
            if (not torch.is_tensor(value)
                    or tuple(value.shape) != shape
                    or value.dtype != dtype):
                raise ValueError(
                    f'invalid elite memory storage for {name!r}')
            value = value.detach().cpu().clone()
            if torch.is_floating_point(value) \
                    and not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f'elite memory storage {name!r} must be finite')
            checked[name] = value
        valid_count = state.get('valid_count')
        if (isinstance(valid_count, bool)
                or not isinstance(valid_count, int)
                or valid_count != int(torch.count_nonzero(checked['valid']))):
            raise ValueError(
                'elite memory checkpoint valid_count is inconsistent')
        generator_state = state.get('generator_state')
        if (not torch.is_tensor(generator_state)
                or generator_state.dtype != torch.uint8
                or generator_state.ndim != 1):
            raise ValueError(
                'elite memory checkpoint has invalid generator state')
        checked_generator = torch.Generator(device='cpu')
        try:
            checked_generator.set_state(
                generator_state.detach().cpu().clone())
        except RuntimeError as error:
            raise ValueError(
                'elite memory checkpoint has invalid generator state') \
                from error

        self.progress_m.copy_(checked['progress_m'])
        self.task.copy_(checked['task'])
        self.q_projected.copy_(checked['q_projected'])
        self.valid.copy_(checked['valid'])
        self.generator.set_state(checked_generator.get_state())


class DirectSeedPairedArchive:
    """Fixed-task CPU archive for deterministic baseline outcomes.

    The first observed outcome for each task is immutable by default.
    Replacing an existing outcome requires ``overwrite=True`` explicitly,
    which keeps full-coverage deterministic collection auditable.  Paired
    targets compare these baseline returns with
    :class:`DirectSeedEliteMemory` and expose only successful online
    projections; supplied fallback configurations never become imitation
    targets.
    """

    _FORMAT = 'direct-seed-paired-archive-v1'
    _KNOWN_ROUTES = (
        ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_INVALID)

    def __init__(
        self,
        task_ids: torch.Tensor,
        seed: int = 0,
    ):
        ids = DirectSeedEliteMemory._validate_task_ids(
            task_ids, require_unique=True, allow_empty=False)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('paired archive seed must be an integer')
        if not -(1 << 63) <= seed < (1 << 64):
            raise ValueError(
                'paired archive seed is outside manual_seed range')

        self.task_ids = ids
        self._id_to_index = {
            int(task_id): index
            for index, task_id in enumerate(self.task_ids.tolist())
        }
        task_count = int(self.task_ids.numel())
        self.task = torch.zeros(
            (task_count, 9), dtype=torch.float32, device='cpu')
        self.q_projected = torch.zeros(
            (task_count, 7), dtype=torch.float32, device='cpu')
        self.progress_m = torch.zeros(
            task_count, dtype=torch.float32, device='cpu')
        self.route = torch.zeros(
            task_count, dtype=torch.int64, device='cpu')
        self.valid = torch.zeros(
            task_count, dtype=torch.bool, device='cpu')
        self.generator = torch.Generator(device='cpu')
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        return int(torch.count_nonzero(self.valid))

    @property
    def coverage(self) -> float:
        """Fraction of configured tasks with a baseline outcome."""
        return len(self) / int(self.task_ids.numel())

    @torch.no_grad()
    def clear(self) -> None:
        """Remove outcomes without perturbing the private sampling RNG."""
        self.task.zero_()
        self.q_projected.zero_()
        self.progress_m.zero_()
        self.route.zero_()
        self.valid.zero_()

    @torch.no_grad()
    def update(
        self,
        task_ids: torch.Tensor,
        batch: DirectSeedRLBatch,
        *,
        overwrite: bool = False,
    ) -> int:
        """Store first outcomes, or explicitly replace the supplied tasks.

        Duplicate task IDs are deterministic: the first row wins for normal
        insertion and the last row wins for explicit overwrite.  Every row
        sharing a task ID must carry bit-identical task geometry.

        Returns:
            Number of distinct task slots written.
        """
        if not isinstance(batch, DirectSeedRLBatch):
            raise TypeError('batch must be a DirectSeedRLBatch')
        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be boolean')
        ids = DirectSeedEliteMemory._validate_task_ids(
            task_ids, require_unique=False, allow_empty=True)
        if ids.numel() != batch.batch_size:
            raise ValueError(
                'task_ids length must equal the DirectSeedRLBatch size')
        unknown = sorted({
            int(task_id) for task_id in ids.tolist()
            if int(task_id) not in self._id_to_index
        })
        if unknown:
            raise ValueError(
                f'update contains unknown task_ids: {unknown[:8]}')

        task = batch.task.detach().to(
            device='cpu', dtype=torch.float32)
        q_projected = batch.q_projected.detach().to(
            device='cpu', dtype=torch.float32)
        progress = batch.progress_m.detach().to(
            device='cpu', dtype=torch.float32)
        route = batch.route.detach().to(
            device='cpu', dtype=torch.int64)

        rows_by_slot: dict[int, list[int]] = {}
        for row, task_id in enumerate(ids.tolist()):
            slot = self._id_to_index[int(task_id)]
            rows_by_slot.setdefault(slot, []).append(row)
        for slot, rows in rows_by_slot.items():
            reference = task[rows[0]]
            if any(not torch.equal(reference, task[row])
                   for row in rows[1:]):
                raise ValueError(
                    'one task_id cannot carry different task geometry')
            if self.valid[slot] and not torch.equal(
                    self.task[slot], reference):
                raise ValueError(
                    'stored task geometry differs for an existing task_id')

        selected: dict[int, int] = {}
        for slot, rows in rows_by_slot.items():
            if self.valid[slot] and not overwrite:
                continue
            selected[slot] = rows[-1] if overwrite else rows[0]
        for slot, row in selected.items():
            self.task[slot].copy_(task[row])
            self.q_projected[slot].copy_(q_projected[row])
            self.progress_m[slot].copy_(progress[row])
            self.route[slot].copy_(route[row])
            self.valid[slot] = True
        return len(selected)

    def _validate_pairing(
        self,
        explorer_elite_memory: DirectSeedEliteMemory,
        advantage_margin_m: float,
    ) -> float:
        if not isinstance(
                explorer_elite_memory, DirectSeedEliteMemory):
            raise TypeError(
                'explorer_elite_memory must be a DirectSeedEliteMemory')
        if not torch.equal(
                explorer_elite_memory.task_ids, self.task_ids):
            raise ValueError(
                'explorer elite task_ids differ from paired archive')
        if (isinstance(advantage_margin_m, bool)
                or not isinstance(advantage_margin_m, (int, float))):
            raise TypeError('advantage_margin_m must be a real number')
        margin = float(advantage_margin_m)
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError(
                'advantage_margin_m must be finite and non-negative')
        paired = self.valid & explorer_elite_memory.valid
        if bool(paired.any()) and not torch.equal(
                self.task[paired],
                explorer_elite_memory.task[paired]):
            raise ValueError(
                'paired baseline and explorer task geometry differ')
        return margin

    def _target_masks(
        self,
        explorer_elite_memory: DirectSeedEliteMemory,
        advantage_margin_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        margin = self._validate_pairing(
            explorer_elite_memory, advantage_margin_m)
        explorer_selected = (
            self.valid
            & explorer_elite_memory.valid
            & (
                explorer_elite_memory.progress_m
                > self.progress_m + margin)
        )
        baseline_selected = (
            self.valid
            & (self.route == ROUTE_REFINED)
            & ~explorer_selected
        )
        return baseline_selected, explorer_selected, margin

    @torch.no_grad()
    def build_targets(
        self,
        explorer_elite_memory: DirectSeedEliteMemory,
        advantage_margin_m: float = 0.0,
    ) -> DirectSeedRLBatch:
        """Build all legal paired projection targets in fixed task order.

        A significantly better explorer projection replaces the baseline.
        Otherwise a refined baseline supplies its own projection.  Direct,
        fallback, invalid, and missing baselines have no target unless paired
        with a significantly better refined explorer.
        """
        baseline_selected, explorer_selected, _ = self._target_masks(
            explorer_elite_memory, advantage_margin_m)
        selected = baseline_selected | explorer_selected
        index = torch.nonzero(selected, as_tuple=False).flatten()
        task = self.task.index_select(0, index)
        baseline_q = self.q_projected.index_select(0, index)
        explorer_q = explorer_elite_memory.q_projected.index_select(
            0, index)
        choose_explorer = explorer_selected.index_select(0, index)
        projected = torch.where(
            choose_explorer.unsqueeze(-1), explorer_q, baseline_q)
        baseline_progress = self.progress_m.index_select(0, index)
        explorer_progress = explorer_elite_memory.progress_m.index_select(
            0, index)
        progress = torch.where(
            choose_explorer, explorer_progress, baseline_progress)
        return DirectSeedRLBatch(
            task=task,
            q_raw=projected.clone(),
            q_projected=projected,
            fallback_q=torch.zeros_like(projected),
            progress_m=progress,
            route=torch.full(
                (int(index.numel()),), ROUTE_REFINED,
                dtype=torch.int64, device='cpu'),
        )

    @torch.no_grad()
    def target_stats(
        self,
        explorer_elite_memory: DirectSeedEliteMemory,
        advantage_margin_m: float = 0.0,
    ) -> dict[str, int | float]:
        """Return paired-selection counts without consuming sampling RNG."""
        baseline_selected, explorer_selected, margin = self._target_masks(
            explorer_elite_memory, advantage_margin_m)
        selected = baseline_selected | explorer_selected
        task_count = int(self.task_ids.numel())
        baseline_count = len(self)
        explorer_count = len(explorer_elite_memory)
        target_count = int(torch.count_nonzero(selected))
        paired_count = int(torch.count_nonzero(
            self.valid & explorer_elite_memory.valid))
        return {
            'configured_task_count': task_count,
            'baseline_outcome_count': baseline_count,
            'baseline_coverage': baseline_count / task_count,
            'baseline_refined_count': int(torch.count_nonzero(
                self.valid & (self.route == ROUTE_REFINED))),
            'explorer_elite_count': explorer_count,
            'explorer_coverage': explorer_count / task_count,
            'paired_outcome_count': paired_count,
            'paired_coverage': paired_count / task_count,
            'explorer_selected_count': int(torch.count_nonzero(
                explorer_selected)),
            'baseline_selected_count': int(torch.count_nonzero(
                baseline_selected)),
            'target_count': target_count,
            'target_coverage': target_count / task_count,
            'advantage_margin_m': margin,
        }

    @torch.no_grad()
    def sample_targets(
        self,
        explorer_elite_memory: DirectSeedEliteMemory,
        batch_size: int,
        *,
        advantage_margin_m: float = 0.0,
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ) -> DirectSeedRLBatch:
        """Uniformly sample paired targets with replacement."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(
                'paired target sample size must be an integer')
        if batch_size < 1:
            raise ValueError(
                'paired target sample size must be positive')
        if not isinstance(dtype, torch.dtype) \
                or not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError(
                'paired target sample dtype must be floating point')
        targets = self.build_targets(
            explorer_elite_memory, advantage_margin_m)
        if targets.batch_size < 1:
            raise RuntimeError(
                'cannot sample empty paired targets')
        index = torch.randint(
            targets.batch_size, (batch_size,),
            generator=self.generator, device='cpu')
        cpu_batch = DirectSeedRLBatch(
            task=targets.task.index_select(0, index),
            q_raw=targets.q_raw.index_select(0, index),
            q_projected=targets.q_projected.index_select(0, index),
            fallback_q=targets.fallback_q.index_select(0, index),
            progress_m=targets.progress_m.index_select(0, index),
            route=targets.route.index_select(0, index),
        )
        return cpu_batch.to(device=device, dtype=dtype)

    def state_dict(self) -> dict:
        """Return a portable checkpoint including the private RNG stream."""
        return {
            'format': self._FORMAT,
            'task_ids': self.task_ids.clone(),
            'outcome_count': len(self),
            'generator_state': self.generator.get_state().cpu().clone(),
            'storage': {
                'task': self.task.clone(),
                'q_projected': self.q_projected.clone(),
                'progress_m': self.progress_m.clone(),
                'route': self.route.clone(),
                'valid': self.valid.clone(),
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Strictly restore outcomes and RNG without partial mutation."""
        if not isinstance(state, Mapping):
            raise TypeError(
                'paired archive checkpoint must be a mapping')
        if state.get('format') != self._FORMAT:
            raise ValueError(
                'unsupported direct-seed paired archive format')
        checkpoint_ids = state.get('task_ids')
        if (not torch.is_tensor(checkpoint_ids)
                or checkpoint_ids.ndim != 1
                or checkpoint_ids.dtype != torch.int64
                or not torch.equal(
                    checkpoint_ids.detach().cpu(), self.task_ids)):
            raise ValueError(
                'paired archive checkpoint task_ids differ')
        storage = state.get('storage')
        if not isinstance(storage, Mapping):
            raise ValueError(
                'paired archive checkpoint has no storage mapping')
        task_count = int(self.task_ids.numel())
        expected = {
            'task': ((task_count, 9), torch.float32),
            'q_projected': ((task_count, 7), torch.float32),
            'progress_m': ((task_count,), torch.float32),
            'route': ((task_count,), torch.int64),
            'valid': ((task_count,), torch.bool),
        }
        checked: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in expected.items():
            value = storage.get(name)
            if (not torch.is_tensor(value)
                    or tuple(value.shape) != shape
                    or value.dtype != dtype):
                raise ValueError(
                    f'invalid paired archive storage for {name!r}')
            value = value.detach().cpu().clone()
            if torch.is_floating_point(value) \
                    and not bool(torch.isfinite(value).all()):
                raise ValueError(
                    f'paired archive storage {name!r} must be finite')
            checked[name] = value
        outcome_count = state.get('outcome_count')
        if (isinstance(outcome_count, bool)
                or not isinstance(outcome_count, int)
                or outcome_count
                != int(torch.count_nonzero(checked['valid']))):
            raise ValueError(
                'paired archive outcome_count is inconsistent')
        known_route = torch.zeros(
            task_count, dtype=torch.bool, device='cpu')
        for code in self._KNOWN_ROUTES:
            known_route |= checked['route'] == code
        if not bool(known_route[checked['valid']].all()):
            raise ValueError(
                'paired archive contains an unknown recorded route')
        generator_state = state.get('generator_state')
        if (not torch.is_tensor(generator_state)
                or generator_state.dtype != torch.uint8
                or generator_state.ndim != 1):
            raise ValueError(
                'paired archive checkpoint has invalid generator state')
        checked_generator = torch.Generator(device='cpu')
        try:
            checked_generator.set_state(
                generator_state.detach().cpu().clone())
        except RuntimeError as error:
            raise ValueError(
                'paired archive checkpoint has invalid generator state') \
                from error

        self.task.copy_(checked['task'])
        self.q_projected.copy_(checked['q_projected'])
        self.progress_m.copy_(checked['progress_m'])
        self.route.copy_(checked['route'])
        self.valid.copy_(checked['valid'])
        self.generator.set_state(checked_generator.get_state())


class DirectSeedMacroReplay:
    """Device-resident ring buffer of real one-seed controller rollouts.

    Sampling uses a private generator so replay reuse neither consumes nor
    depends on the global RNG stream used by PPO or physics.  ``state_dict``
    moves storage to CPU for portable checkpoints while ``load_state_dict``
    restores it directly onto the buffer device.
    """

    _FLOAT_FIELDS = (
        'task', 'q_raw', 'q_projected', 'fallback_q', 'progress_m')

    def __init__(
        self,
        capacity: int,
        device: torch.device | str,
        *,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ):
        if capacity < 1:
            raise ValueError('replay capacity must be positive')
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError('replay dtype must be floating point')
        self.capacity = int(capacity)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype = dtype
        self.task = torch.zeros(
            (capacity, 9), device=self.device, dtype=dtype)
        self.q_raw = torch.zeros(
            (capacity, 7), device=self.device, dtype=dtype)
        self.q_projected = torch.zeros_like(self.q_raw)
        self.fallback_q = torch.zeros_like(self.q_raw)
        self.progress_m = torch.zeros(
            capacity, device=self.device, dtype=dtype)
        self.route = torch.zeros(
            capacity, device=self.device, dtype=torch.int64)
        self.size = 0
        self.write_index = 0
        self.total_added = 0
        self.total_sampled = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def clear(self, *, reset_counters: bool = False) -> None:
        """Drop stale samples while retaining the allocated device storage."""
        self.size = 0
        self.write_index = 0
        if reset_counters:
            self.total_added = 0
            self.total_sampled = 0

    def _copy_wrapped(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        start: int,
    ) -> None:
        first = min(source.shape[0], self.capacity - start)
        target[start:start + first].copy_(source[:first])
        if first < source.shape[0]:
            target[:source.shape[0] - first].copy_(source[first:])

    @torch.no_grad()
    def add(self, batch: DirectSeedRLBatch) -> None:
        if not isinstance(batch, DirectSeedRLBatch):
            raise TypeError('batch must be a DirectSeedRLBatch')
        if batch.task.device != self.device:
            raise ValueError('replay and batch must share one device')
        if batch.task.dtype != self.dtype:
            raise ValueError('replay and batch must share one dtype')
        original_count = batch.batch_size
        if original_count >= self.capacity:
            # Only the newest capacity entries survive, but advance the write
            # cursor as if every real transition had crossed the ring.
            skip = original_count - self.capacity
            start = (self.write_index + skip) % self.capacity
            source_slice = slice(skip, None)
            count = self.capacity
        else:
            start = self.write_index
            source_slice = slice(None)
            count = original_count
        for name in self._FLOAT_FIELDS:
            self._copy_wrapped(
                getattr(self, name),
                getattr(batch, name)[source_slice].detach(),
                start)
        self._copy_wrapped(
            self.route,
            batch.route[source_slice].to(dtype=torch.int64).detach(),
            start)
        self.write_index = (
            self.write_index + original_count) % self.capacity
        self.size = min(self.capacity, self.size + original_count)
        self.total_added += original_count

    @torch.no_grad()
    def sample(self, batch_size: int) -> DirectSeedRLBatch:
        if batch_size < 1:
            raise ValueError('replay sample size must be positive')
        if self.size < 1:
            raise RuntimeError('cannot sample an empty replay')
        index = torch.randint(
            self.size, (batch_size,), device=self.device,
            generator=self.generator)
        self.total_sampled += batch_size
        return DirectSeedRLBatch(
            task=self.task.index_select(0, index),
            q_raw=self.q_raw.index_select(0, index),
            q_projected=self.q_projected.index_select(0, index),
            fallback_q=self.fallback_q.index_select(0, index),
            progress_m=self.progress_m.index_select(0, index),
            route=self.route.index_select(0, index),
        )

    @torch.no_grad()
    def sample_elite(
        self,
        batch_size: int,
        elite_fraction: float,
    ) -> DirectSeedRLBatch:
        """Sample high-return successful projections with replacement.

        Eligibility is deliberately restricted to ``ROUTE_REFINED``: only a
        successful online IK projection supplies a trustworthy self-distillation
        target.  The top fraction is computed within that eligible set using
        real downstream controller progress, then sampled with the replay's
        private RNG.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError('elite replay sample size must be an integer')
        if batch_size < 1:
            raise ValueError('elite replay sample size must be positive')
        if (isinstance(elite_fraction, bool)
                or not isinstance(elite_fraction, (int, float))):
            raise TypeError('elite_fraction must be a real number')
        elite_fraction = float(elite_fraction)
        if (not math.isfinite(elite_fraction)
                or not 0.0 < elite_fraction <= 1.0):
            raise ValueError('elite_fraction must be finite and in (0, 1]')
        if self.size < 1:
            raise RuntimeError(
                'cannot sample elite projections from an empty replay')

        refined_index = torch.nonzero(
            self.route[:self.size] == ROUTE_REFINED,
            as_tuple=False).flatten()
        refined_count = int(refined_index.numel())
        if refined_count < 1:
            raise RuntimeError(
                'cannot sample elite projections without ROUTE_REFINED data')
        elite_count = max(
            1, int(math.ceil(refined_count * elite_fraction)))
        refined_progress = self.progress_m.index_select(0, refined_index)
        elite_local = torch.topk(
            refined_progress, elite_count, largest=True,
            sorted=False).indices
        elite_index = refined_index.index_select(0, elite_local)
        choice = torch.randint(
            elite_count, (batch_size,), device=self.device,
            generator=self.generator)
        index = elite_index.index_select(0, choice)
        self.total_sampled += batch_size
        return DirectSeedRLBatch(
            task=self.task.index_select(0, index),
            q_raw=self.q_raw.index_select(0, index),
            q_projected=self.q_projected.index_select(0, index),
            fallback_q=self.fallback_q.index_select(0, index),
            progress_m=self.progress_m.index_select(0, index),
            route=self.route.index_select(0, index),
        )

    def state_dict(self) -> dict:
        return {
            'format': 'direct-seed-macro-replay-v1',
            'capacity': self.capacity,
            'dtype': str(self.dtype),
            'size': self.size,
            'write_index': self.write_index,
            'total_added': self.total_added,
            'total_sampled': self.total_sampled,
            'generator_state': self.generator.get_state().cpu(),
            'storage': {
                name: getattr(self, name).detach().cpu()
                for name in (*self._FLOAT_FIELDS, 'route')
            },
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get('format') != 'direct-seed-macro-replay-v1':
            raise ValueError('unsupported direct-seed macro replay format')
        if int(state.get('capacity', -1)) != self.capacity:
            raise ValueError('replay checkpoint capacity differs')
        expected_dtype = str(self.dtype)
        if state.get('dtype') != expected_dtype:
            raise ValueError(
                f'replay checkpoint dtype differs from {expected_dtype}')
        size = int(state.get('size', -1))
        write_index = int(state.get('write_index', -1))
        if not 0 <= size <= self.capacity:
            raise ValueError('invalid replay checkpoint size')
        if not 0 <= write_index < self.capacity:
            raise ValueError('invalid replay checkpoint write index')
        storage = state.get('storage')
        if not isinstance(storage, Mapping):
            raise ValueError('replay checkpoint has no storage mapping')
        expected_shapes = {
            'task': (self.capacity, 9),
            'q_raw': (self.capacity, 7),
            'q_projected': (self.capacity, 7),
            'fallback_q': (self.capacity, 7),
            'progress_m': (self.capacity,),
            'route': (self.capacity,),
        }
        for name, shape in expected_shapes.items():
            value = storage.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(
                    f'invalid replay storage for {name!r}')
            target = getattr(self, name)
            target.copy_(value.to(device=self.device, dtype=target.dtype))
        self.size = size
        self.write_index = write_index
        total_added = int(state.get('total_added', size))
        total_sampled = int(state.get('total_sampled', 0))
        if total_added < size or total_sampled < 0:
            raise ValueError('invalid replay checkpoint counters')
        self.total_added = total_added
        self.total_sampled = total_sampled
        generator_state = state.get('generator_state')
        if not torch.is_tensor(generator_state):
            raise ValueError('replay checkpoint has no generator state')
        self.generator.set_state(generator_state.cpu())


def _validated_limits(
    q_lower: torch.Tensor,
    q_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_lower = torch.as_tensor(q_lower, dtype=torch.float32)
    q_upper = torch.as_tensor(q_upper, dtype=torch.float32)
    if q_lower.shape != (7,) or q_upper.shape != (7,):
        raise ValueError('q_lower and q_upper must have shape (7,)')
    if (not bool(torch.isfinite(q_lower).all())
            or not bool(torch.isfinite(q_upper).all())
            or not bool((q_lower < q_upper).all())):
        raise ValueError('joint limits must be finite with lower < upper')
    return q_lower, q_upper


def _validated_task_norm(
    task_mean: torch.Tensor | None,
    task_std: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.zeros(9) if task_mean is None \
        else torch.as_tensor(task_mean, dtype=torch.float32)
    std = torch.ones(9) if task_std is None \
        else torch.as_tensor(task_std, dtype=torch.float32)
    if mean.shape != (9,) or std.shape != (9,):
        raise ValueError('task_mean and task_std must have shape (9,)')
    if (not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool((std > 0.0).all())):
        raise ValueError('task normalization must be finite with std > 0')
    return mean, std


def _mlp(
    input_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    output_dim: int,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(n_hidden_layers):
        modules.extend([nn.Linear(width, hidden_dim), nn.SiLU()])
        width = hidden_dim
    modules.append(nn.Linear(width, output_dim))
    return nn.Sequential(*modules)


class DirectSeedActor(nn.Module):
    """Contextual tanh-Gaussian actor; deterministic mean is deployed."""

    def __init__(
        self,
        q_lower: torch.Tensor,
        q_upper: torch.Tensor,
        config: DirectSeedActorConfig | None = None,
        *,
        task_mean: torch.Tensor | None = None,
        task_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = DirectSeedActorConfig() if config is None else config
        if not isinstance(self.config, DirectSeedActorConfig):
            raise TypeError('config must be a DirectSeedActorConfig')
        lower, upper = _validated_limits(q_lower, q_upper)
        mean, std = _validated_task_norm(task_mean, task_std)
        self.register_buffer('q_lower', lower)
        self.register_buffer('q_upper', upper)
        self.register_buffer('q_mid', 0.5 * (lower + upper))
        self.register_buffer('q_half', 0.5 * (upper - lower))
        self.register_buffer('task_mean', mean)
        self.register_buffer('task_std', std)
        self.trunk = _mlp(
            self.config.task_dim,
            self.config.hidden_dim,
            self.config.n_hidden_layers,
            2 * self.config.q_dim)

    def _distribution(
        self, task: torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        if task.ndim != 2 or task.shape[-1] != self.config.task_dim:
            raise ValueError('task must have shape (B, 9)')
        normalised = (
            task - self.task_mean.to(dtype=task.dtype)
        ) / self.task_std.to(dtype=task.dtype)
        mean, raw_log_std = self.trunk(normalised).chunk(2, dim=-1)
        # Smoothly map raw values into the configured interval.
        unit = torch.tanh(raw_log_std)
        log_std = (
            self.config.log_std_min
            + 0.5 * (unit + 1.0)
            * (self.config.log_std_max - self.config.log_std_min))
        return torch.distributions.Normal(mean, log_std.exp()), mean

    def sample(
        self,
        task: torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        noise_scale: float = 1.0,
    ) -> DirectSeedAction:
        if not math.isfinite(noise_scale) or noise_scale < 0.0:
            raise ValueError('noise_scale must be finite and non-negative')
        distribution, mean = self._distribution(task)
        if deterministic or noise_scale == 0.0:
            pre_tanh = mean
        elif generator is None:
            pre_tanh = (
                mean + float(noise_scale) * (distribution.rsample() - mean))
        else:
            noise = torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device,
                generator=generator)
            pre_tanh = (
                mean
                + float(noise_scale) * distribution.scale * noise)
        unit = torch.tanh(pre_tanh)
        normalized_q = self.config.limit_fraction * unit
        q = (
            self.q_mid.to(dtype=task.dtype)
            + self.q_half.to(dtype=task.dtype) * normalized_q)
        eps = torch.finfo(task.dtype).eps
        scale = (
            self.config.limit_fraction
            * self.q_half.to(dtype=task.dtype)).clamp_min(eps)
        log_prob = (
            distribution.log_prob(pre_tanh)
            - torch.log((1.0 - unit.square()).clamp_min(eps))
            - torch.log(scale)
        ).sum(dim=-1)
        return DirectSeedAction(
            q=q,
            normalized_q=normalized_q,
            pre_tanh=pre_tanh,
            log_prob=log_prob)

    def mean_q(self, task: torch.Tensor) -> torch.Tensor:
        return self.sample(task, deterministic=True).q

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        """Alias used by the generic deployment generator interface."""
        return self.mean_q(task)


class DirectSeedMoEActor(nn.Module):
    """Task-conditioned hard mixture that deploys exactly one joint seed.

    A shared task trunk feeds ``K`` joint heads and one categorical gate.
    Deployment takes the deterministic gate argmax and gathers one expert
    output.  It never queries returns, candidates, critics, or controllers.
    """

    def __init__(
        self,
        q_lower: torch.Tensor,
        q_upper: torch.Tensor,
        config: DirectSeedMoEActorConfig | None = None,
        *,
        task_mean: torch.Tensor | None = None,
        task_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = (
            DirectSeedMoEActorConfig() if config is None else config)
        if not isinstance(self.config, DirectSeedMoEActorConfig):
            raise TypeError('config must be a DirectSeedMoEActorConfig')
        lower, upper = _validated_limits(q_lower, q_upper)
        mean, std = _validated_task_norm(task_mean, task_std)
        self.register_buffer('q_lower', lower)
        self.register_buffer('q_upper', upper)
        self.register_buffer('q_mid', 0.5 * (lower + upper))
        self.register_buffer('q_half', 0.5 * (upper - lower))
        self.register_buffer('task_mean', mean)
        self.register_buffer('task_std', std)

        modules: list[nn.Module] = []
        width = self.config.task_dim
        for _ in range(self.config.n_hidden_layers):
            modules.extend([
                nn.Linear(width, self.config.hidden_dim),
                nn.SiLU(),
            ])
            width = self.config.hidden_dim
        self.trunk = nn.Sequential(*modules)
        expert_output_dims = [self.config.q_dim] * self.config.n_experts
        if self.config.exact_baseline_head:
            # The first half is the deterministic mean and the second half
            # remains unused log-std output.  Keeping both preserves the exact
            # source GEMM rather than merely copying its first seven rows.
            expert_output_dims[0] = 2 * self.config.q_dim
        self.experts = nn.ModuleList([
            nn.Linear(width, output_dim)
            for output_dim in expert_output_dims
        ])
        if self.config.gate_hidden_dim == 0:
            # Keep the old module type and exact state-dict names
            # ``gate.weight``/``gate.bias`` for legacy checkpoints.
            self.gate = nn.Linear(width, self.config.n_experts)
            gate_output = self.gate
        else:
            self.gate = nn.Sequential(
                nn.Linear(width, self.config.gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    self.config.gate_hidden_dim,
                    self.config.n_experts),
            )
            gate_output = self.gate[-1]
        # Stable expert-0 tie-break before supervised gate training.  Argmax
        # resolves the all-zero tie to index zero.
        nn.init.zeros_(gate_output.weight)
        nn.init.zeros_(gate_output.bias)

    def _features(self, task: torch.Tensor) -> torch.Tensor:
        if task.ndim != 2 or task.shape[-1] != self.config.task_dim:
            raise ValueError('task must have shape (B, 9)')
        if not torch.is_floating_point(task):
            raise TypeError('task must have a floating dtype')
        if task.device != self.q_mid.device:
            raise ValueError('task and MoE actor must share one device')
        if task.dtype != self.q_mid.dtype:
            raise ValueError('task dtype must match MoE actor dtype')
        normalised = (
            task - self.task_mean.to(dtype=task.dtype)
        ) / self.task_std.to(dtype=task.dtype)
        return self.trunk(normalised)

    def _expert_q_from_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        raw = torch.stack([
            expert(features)[:, :self.config.q_dim]
            for expert in self.experts
        ], dim=1)
        normalised_q = self.config.limit_fraction * torch.tanh(raw)
        return (
            self.q_mid.to(dtype=features.dtype).view(1, 1, -1)
            + self.q_half.to(dtype=features.dtype).view(1, 1, -1)
            * normalised_q
        )

    def expert_q_and_gate(
        self,
        task: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return all training-only expert outputs and gate logits."""
        features = self._features(task)
        return (
            self._expert_q_from_features(features),
            self.gate(features),
        )

    def expert_q(self, task: torch.Tensor) -> torch.Tensor:
        """Return ``(B, K, 7)`` expert outputs for training diagnostics."""
        return self.expert_q_and_gate(task)[0]

    def gate_logits(self, task: torch.Tensor) -> torch.Tensor:
        return self.expert_q_and_gate(task)[1]

    def expert_index(self, task: torch.Tensor) -> torch.Tensor:
        """Return the deterministic hard-gate expert index per task."""
        return self.gate_logits(task).argmax(dim=-1)

    def mean_q(self, task: torch.Tensor) -> torch.Tensor:
        """Hard-route with one head multiply per row and no host sync."""
        features = self._features(task)
        index = self.gate(features).argmax(dim=-1)
        if self.config.exact_baseline_head:
            # Evaluate the intact baseline head on the complete batch.  Even
            # slicing the batch before this GEMM can select a different
            # floating-point kernel, so this also preserves baseline rows in
            # a mixed expert batch.
            baseline_raw = self.experts[0](features)[:, :self.config.q_dim]
            if self.config.n_experts == 1:
                raw = baseline_raw
            else:
                specialist_weight = torch.stack([
                    expert.weight for expert in self.experts[1:]
                ], dim=0)
                specialist_bias = torch.stack([
                    expert.bias for expert in self.experts[1:]
                ], dim=0)
                # Base rows may select specialist 0 here, but ``where`` drops
                # that value.  This keeps routing tensorized with no host
                # synchronization and only one specialist head per row.
                specialist_index = (index - 1).clamp_min(0)
                selected_weight = specialist_weight.index_select(
                    0, specialist_index)
                selected_bias = specialist_bias.index_select(
                    0, specialist_index)
                specialist_raw = torch.bmm(
                    selected_weight, features.unsqueeze(-1)
                ).squeeze(-1) + selected_bias
                raw = torch.where(
                    (index == 0).unsqueeze(-1),
                    baseline_raw,
                    specialist_raw)
        else:
            weight = torch.stack([
                expert.weight for expert in self.experts
            ], dim=0)
            bias = torch.stack([
                expert.bias for expert in self.experts
            ], dim=0)
            selected_weight = weight.index_select(0, index)
            selected_bias = bias.index_select(0, index)
            raw = torch.bmm(
                selected_weight, features.unsqueeze(-1)
            ).squeeze(-1) + selected_bias
        normalised_q = self.config.limit_fraction * torch.tanh(raw)
        return (
            self.q_mid.to(dtype=features.dtype)
            + self.q_half.to(dtype=features.dtype) * normalised_q
        )

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        return self.mean_q(task)


@torch.no_grad()
def direct_seed_moe_from_actor(
    actor: DirectSeedActor,
    *,
    n_experts: int = 4,
    expert_perturb_std: float = 0.0,
    seed: int = 0,
) -> DirectSeedMoEActor:
    """Convert a single actor while preserving expert-0 deployment numerically.

    The hidden trunk and deterministic mean head are copied into every expert.
    Optional private-RNG perturbations affect experts ``1..K-1`` only.  A
    one-unit gate bias selects untouched expert 0 until imitation trains the
    gate.  Expert 0 retains the complete original Gaussian output head so its
    deployed seed is bitwise identical, including baseline rows in a mixed
    expert batch.
    """
    if not isinstance(actor, DirectSeedActor):
        raise TypeError('actor must be a DirectSeedActor')
    if isinstance(n_experts, bool) or not isinstance(n_experts, int):
        raise TypeError('n_experts must be an integer')
    if n_experts < 1:
        raise ValueError('n_experts must be positive')
    if (isinstance(expert_perturb_std, bool)
            or not isinstance(expert_perturb_std, (int, float))):
        raise TypeError('expert_perturb_std must be a real number')
    perturb_std = float(expert_perturb_std)
    if not math.isfinite(perturb_std) or perturb_std < 0.0:
        raise ValueError(
            'expert_perturb_std must be finite and non-negative')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError('MoE conversion seed must be an integer')
    if not -(1 << 63) <= seed < (1 << 64):
        raise ValueError('MoE conversion seed is outside manual_seed range')

    config = DirectSeedMoEActorConfig(
        task_dim=actor.config.task_dim,
        q_dim=actor.config.q_dim,
        hidden_dim=actor.config.hidden_dim,
        n_hidden_layers=actor.config.n_hidden_layers,
        n_experts=n_experts,
        limit_fraction=actor.config.limit_fraction,
        exact_baseline_head=True,
    )
    device = actor.q_mid.device
    dtype = actor.q_mid.dtype
    moe = DirectSeedMoEActor(
        actor.q_lower.detach().cpu(),
        actor.q_upper.detach().cpu(),
        config,
        task_mean=actor.task_mean.detach().cpu(),
        task_std=actor.task_std.detach().cpu(),
    ).to(device=device, dtype=dtype)

    old_modules = list(actor.trunk.children())
    expected_modules = 2 * actor.config.n_hidden_layers + 1
    if len(old_modules) != expected_modules \
            or not isinstance(old_modules[-1], nn.Linear):
        raise ValueError(
            'single actor trunk does not match its serialized config')
    for layer in range(actor.config.n_hidden_layers):
        source = old_modules[2 * layer]
        target = moe.trunk[2 * layer]
        if not isinstance(source, nn.Linear) \
                or not isinstance(target, nn.Linear):
            raise ValueError(
                'single actor hidden trunk has an unexpected layout')
        target.weight.copy_(source.weight)
        target.bias.copy_(source.bias)

    mean_head = old_modules[-1]
    mean_weight = mean_head.weight[:actor.config.q_dim]
    mean_bias = mean_head.bias[:actor.config.q_dim]
    moe.experts[0].weight.copy_(mean_head.weight)
    moe.experts[0].bias.copy_(mean_head.bias)
    for expert in moe.experts[1:]:
        expert.weight.copy_(mean_weight)
        expert.bias.copy_(mean_bias)
    if perturb_std > 0.0 and n_experts > 1:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        for expert in moe.experts[1:]:
            expert.weight.add_(
                perturb_std * torch.randn(
                    expert.weight.shape,
                    device=device, dtype=dtype,
                    generator=generator))
            expert.bias.add_(
                perturb_std * torch.randn(
                    expert.bias.shape,
                    device=device, dtype=dtype,
                    generator=generator))
    moe.gate.weight.zero_()
    moe.gate.bias.zero_()
    moe.gate.bias[0] = 1.0
    moe.train(actor.training)
    return moe


class _MacroQ(nn.Module):
    def __init__(self, config: DirectSeedCriticConfig):
        super().__init__()
        self.network = _mlp(
            config.task_dim + config.q_dim,
            config.hidden_dim,
            config.n_hidden_layers,
            1)

    def forward(
        self,
        task_normalised: torch.Tensor,
        q_normalised: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            torch.cat([task_normalised, q_normalised], dim=-1)
        ).squeeze(-1)


class TwinMacroQ(nn.Module):
    """Twin estimates of complete controller progress from one raw seed."""

    def __init__(
        self,
        q_lower: torch.Tensor,
        q_upper: torch.Tensor,
        config: DirectSeedCriticConfig | None = None,
        *,
        task_mean: torch.Tensor | None = None,
        task_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = DirectSeedCriticConfig() if config is None else config
        if not isinstance(self.config, DirectSeedCriticConfig):
            raise TypeError('config must be a DirectSeedCriticConfig')
        lower, upper = _validated_limits(q_lower, q_upper)
        mean, std = _validated_task_norm(task_mean, task_std)
        self.register_buffer('q_lower', lower)
        self.register_buffer('q_upper', upper)
        self.register_buffer('q_mid', 0.5 * (lower + upper))
        self.register_buffer('q_half', 0.5 * (upper - lower))
        self.register_buffer('task_mean', mean)
        self.register_buffer('task_std', std)
        self.q1 = _MacroQ(self.config)
        self.q2 = _MacroQ(self.config)

    def forward(
        self,
        task: torch.Tensor,
        q_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if task.ndim != 2 or task.shape[-1] != self.config.task_dim:
            raise ValueError('task must have shape (B, 9)')
        if q_raw.shape != (task.shape[0], self.config.q_dim):
            raise ValueError('q_raw must have shape (B, 7)')
        task_normalised = (
            task - self.task_mean.to(dtype=task.dtype)
        ) / self.task_std.to(dtype=task.dtype)
        q_normalised = (
            q_raw - self.q_mid.to(dtype=q_raw.dtype)
        ) / self.q_half.to(dtype=q_raw.dtype)
        return (
            self.q1(task_normalised, q_normalised),
            self.q2(task_normalised, q_normalised),
        )


def _precision_per_task(
    actor: DirectSeedActor,
    kin,
    collision,
    task: torch.Tensor,
    q: torch.Tensor,
    config: DirectSeedRLConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p0 = task[:, :3]
    n_target = task[:, 6:9]
    p_tcp, rotation, _, _ = kin.tcp_fk_jac(q)
    position_error = (p_tcp - p0).norm(dim=-1)
    cone_cosine = (rotation[:, :, 2] * n_target).sum(dim=-1)
    joint_margin = torch.minimum(
        q - actor.q_lower.to(dtype=q.dtype),
        actor.q_upper.to(dtype=q.dtype) - q).amin(dim=-1)
    # log1p-quadratic is locally precise but prevents a very poor initial IK
    # guess from overwhelming the downstream-Q gradient by several orders of
    # magnitude.  It remains smooth and supplies a direction at large error.
    position = torch.log1p(
        (position_error / float(config.position_scale_m)).square())
    cone = torch.log1p((
        torch.relu(
            math.cos(math.radians(config.cone_deg)) - cone_cosine)
        / float(config.cone_scale)
    ).square())
    joint = torch.log1p((
        torch.relu(float(config.joint_margin_rad) - joint_margin)
        / float(config.joint_margin_scale_rad)
    ).square())
    collision_available = collision is not None
    if collision_available:
        if not hasattr(collision, 'min_margin'):
            raise ValueError(
                'optional collision precision requires collision.min_margin')
        collision_margin = torch.as_tensor(
            collision.min_margin(kin.link_transforms(q)),
            device=q.device, dtype=q.dtype)
        if collision_margin.shape != (q.shape[0],):
            raise ValueError(
                'collision.min_margin must return shape (B,)')
        collision_loss = torch.log1p((
            torch.relu(
                float(config.collision_margin_m) - collision_margin)
            / float(config.collision_scale_m)
        ).square())
    else:
        collision_margin = torch.zeros(
            q.shape[0], device=q.device, dtype=q.dtype)
        collision_loss = torch.zeros_like(collision_margin)
    total = (
        position + cone + joint
        + config.collision_precision_weight * collision_loss)
    return total, {
        'position_error_m': position_error,
        'cone_cosine': cone_cosine,
        'joint_margin_rad': joint_margin,
        'position_loss': position,
        'cone_loss': cone,
        'joint_loss': joint,
        'collision_margin_m': collision_margin,
        'collision_loss': collision_loss,
        'collision_available': torch.full(
            (q.shape[0],), float(collision_available),
            device=q.device, dtype=q.dtype),
    }


def _normalised_squared_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    q_half: torch.Tensor,
) -> torch.Tensor:
    return (
        (left - right) / q_half.to(dtype=left.dtype)
    ).square().mean(dim=-1)


def update_direct_seed_projection(
    actor: DirectSeedActor,
    actor_optimizer: torch.optim.Optimizer,
    batch: DirectSeedRLBatch,
    *,
    gradient_clip_norm: float = 5.0,
    reference_q: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
) -> dict[str, float]:
    """Self-distil elite online IK projections into the deployed actor mean.

    This actor-only update intentionally has no FK, critic, or controller
    dependency.  It minimizes joint-limit-normalized MSE to successful
    ``q_projected`` targets and therefore adds no deployment-time inference
    work.  An optional frozen-snapshot action anchor limits global drift while
    fitting discontinuous IK branches.
    """
    if not isinstance(actor, DirectSeedActor):
        raise TypeError('actor must be a DirectSeedActor')
    if not isinstance(actor_optimizer, torch.optim.Optimizer):
        raise TypeError('actor_optimizer must be a torch optimizer')
    if not isinstance(batch, DirectSeedRLBatch):
        raise TypeError('batch must be a DirectSeedRLBatch')
    if (not math.isfinite(gradient_clip_norm)
            or gradient_clip_norm <= 0.0):
        raise ValueError(
            'gradient_clip_norm must be finite and positive')
    if not math.isfinite(anchor_weight) or anchor_weight < 0.0:
        raise ValueError('anchor_weight must be finite and non-negative')
    if batch.task.device != actor.q_mid.device:
        raise ValueError('actor and batch must share one device')
    if batch.task.dtype != actor.q_mid.dtype:
        raise ValueError('batch dtype must match actor dtype')
    if reference_q is not None:
        if (reference_q.shape != (batch.batch_size, 7)
                or reference_q.device != batch.task.device
                or reference_q.dtype != batch.task.dtype
                or not bool(torch.isfinite(reference_q).all())):
            raise ValueError(
                'reference_q must be finite and match batch shape, '
                'device, and dtype')
    elif anchor_weight > 0.0:
        raise ValueError(
            'positive anchor_weight requires reference_q')
    refined = batch.route == ROUTE_REFINED
    refined_count = int(refined.sum())
    if refined_count < 1:
        raise ValueError(
            'projection update requires at least one ROUTE_REFINED sample')

    actor_parameters = {
        id(parameter)
        for parameter in actor.parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in actor_optimizer.param_groups
        for parameter in group['params']
    }
    if optimizer_parameters != actor_parameters:
        raise ValueError(
            'actor_optimizer must contain exactly the trainable actor '
            'parameters')

    actor_mean = actor.mean_q(batch.task[refined])
    projection_loss = _normalised_squared_distance(
        actor_mean,
        batch.q_projected[refined].detach(),
        actor.q_half).mean()
    anchor_loss = (
        _normalised_squared_distance(
            actor_mean, reference_q[refined].detach(),
            actor.q_half).mean()
        if reference_q is not None
        else actor_mean.sum() * 0.0)
    loss = projection_loss + float(anchor_weight) * anchor_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(
            'projection self-distillation loss is not finite')

    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), float(gradient_clip_norm))
    actor_optimizer.step()

    return {
        'projection_actor_updated': 1.0,
        'projection_actor_loss': float(loss.detach()),
        'projection_distill_loss': float(projection_loss.detach()),
        'projection_anchor_loss': float(anchor_loss.detach()),
        'projection_anchor_weight': float(anchor_weight),
        'projection_refined_count': float(refined_count),
        'projection_refined_fraction': float(
            refined.float().mean()),
        'actor_grad_norm': float(actor_grad),
    }


def update_direct_seed_moe_projection(
    actor: DirectSeedMoEActor,
    actor_optimizer: torch.optim.Optimizer,
    batch: DirectSeedRLBatch,
    *,
    gate_ce_weight: float = 0.1,
    load_balance_weight: float = 0.01,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    """Winner-take-all projection imitation for a hard-gated MoE actor.

    Every refined target is assigned to its nearest current expert in
    joint-range-normalized squared distance.  Only that expert receives the
    imitation gradient.  Cross entropy teaches the deployment gate to choose
    the same winner, while a soft gate-frequency penalty can discourage
    premature routing collapse.
    """
    if not isinstance(actor, DirectSeedMoEActor):
        raise TypeError('actor must be a DirectSeedMoEActor')
    if not isinstance(actor_optimizer, torch.optim.Optimizer):
        raise TypeError('actor_optimizer must be a torch optimizer')
    if not isinstance(batch, DirectSeedRLBatch):
        raise TypeError('batch must be a DirectSeedRLBatch')
    for name, value in (
            ('gate_ce_weight', gate_ce_weight),
            ('load_balance_weight', load_balance_weight)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0):
            raise ValueError(f'{name} must be finite and non-negative')
    if (isinstance(gradient_clip_norm, bool)
            or not isinstance(gradient_clip_norm, (int, float))
            or not math.isfinite(float(gradient_clip_norm))
            or float(gradient_clip_norm) <= 0.0):
        raise ValueError(
            'gradient_clip_norm must be finite and positive')
    if batch.task.device != actor.q_mid.device:
        raise ValueError('actor and batch must share one device')
    if batch.task.dtype != actor.q_mid.dtype:
        raise ValueError('batch dtype must match actor dtype')

    refined = batch.route == ROUTE_REFINED
    refined_count = int(torch.count_nonzero(refined))
    if refined_count < 1:
        raise ValueError(
            'MoE projection update requires at least one ROUTE_REFINED '
            'sample')
    actor_parameters = {
        id(parameter)
        for parameter in actor.parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in actor_optimizer.param_groups
        for parameter in group['params']
    }
    if optimizer_parameters != actor_parameters:
        raise ValueError(
            'actor_optimizer must contain exactly the trainable MoE actor '
            'parameters')

    task = batch.task[refined]
    target = batch.q_projected[refined].detach()
    expert_q, gate_logits = actor.expert_q_and_gate(task)
    distance = (
        (expert_q - target.unsqueeze(1))
        / actor.q_half.to(dtype=task.dtype).view(1, 1, -1)
    ).square().mean(dim=-1)
    winner = distance.detach().argmin(dim=-1)
    imitation_loss = distance.gather(
        1, winner.unsqueeze(-1)).mean()
    gate_ce_loss = F.cross_entropy(gate_logits, winner)
    mean_gate_probability = gate_logits.softmax(dim=-1).mean(dim=0)
    uniform_probability = 1.0 / actor.config.n_experts
    load_balance_loss = (
        actor.config.n_experts
        * (mean_gate_probability - uniform_probability).square().sum()
    )
    loss = (
        imitation_loss
        + float(gate_ce_weight) * gate_ce_loss
        + float(load_balance_weight) * load_balance_loss
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(
            'MoE projection imitation loss is not finite')

    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), float(gradient_clip_norm))
    actor_optimizer.step()

    with torch.no_grad():
        winner_count = torch.bincount(
            winner, minlength=actor.config.n_experts)
        gate_choice = gate_logits.argmax(dim=-1)
        metrics = {
            'moe_actor_updated': 1.0,
            'moe_actor_loss': float(loss),
            'moe_imitation_loss': float(imitation_loss),
            'moe_gate_ce_loss': float(gate_ce_loss),
            'moe_load_balance_loss': float(load_balance_loss),
            'moe_gate_ce_weight': float(gate_ce_weight),
            'moe_load_balance_weight': float(load_balance_weight),
            'moe_gate_winner_accuracy': float(
                (gate_choice == winner).float().mean()),
            'moe_refined_count': float(refined_count),
            'moe_refined_fraction': float(refined.float().mean()),
            'actor_grad_norm': float(actor_grad),
        }
        for expert_index, count in enumerate(winner_count.tolist()):
            metrics[
                f'moe_winner_expert_{expert_index}_fraction'
            ] = float(count / refined_count)
        return metrics


def update_direct_seed_moe_advantage(
    actor: DirectSeedMoEActor,
    actor_optimizer: torch.optim.Optimizer,
    batch: DirectSeedRLBatch,
    explorer_selected: torch.Tensor,
    *,
    gate_ce_weight: float = 1.0,
    positive_gate_weight: float = 1.0,
    specialist_load_balance_weight: float = 0.01,
    gradient_clip_norm: float = 5.0,
) -> dict[str, float]:
    """Fit advantage-gated specialists without moving the safe baseline.

    Expert 0 is the frozen deployed baseline.  A true ``explorer_selected``
    row is an externally established positive-advantage target: its projected
    joint vector is assigned to the nearest specialist (experts ``1..K-1``)
    with winner-take-all imitation, and the gate learns that specialist.
    Every other row trains only the gate to retain expert 0; its joint target
    is deliberately ignored.

    The shared trunk and expert 0 must already be frozen.  Requiring the
    optimizer to contain exactly the gate and specialists makes this safety
    property explicit and prevents optimizer momentum from drifting the
    baseline.  Batches with no positive rows remain valid gate updates.
    """
    if not isinstance(actor, DirectSeedMoEActor):
        raise TypeError('actor must be a DirectSeedMoEActor')
    if not isinstance(actor_optimizer, torch.optim.Optimizer):
        raise TypeError('actor_optimizer must be a torch optimizer')
    if not isinstance(batch, DirectSeedRLBatch):
        raise TypeError('batch must be a DirectSeedRLBatch')
    if not isinstance(explorer_selected, torch.Tensor):
        raise TypeError('explorer_selected must be a torch.Tensor')
    if actor.config.n_experts < 2:
        raise ValueError(
            'advantage-gated MoE requires at least two experts')
    for name, value in (
            ('gate_ce_weight', gate_ce_weight),
            ('positive_gate_weight', positive_gate_weight),
            ('specialist_load_balance_weight',
             specialist_load_balance_weight)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0):
            raise ValueError(f'{name} must be finite and non-negative')
    if (isinstance(gradient_clip_norm, bool)
            or not isinstance(gradient_clip_norm, (int, float))
            or not math.isfinite(float(gradient_clip_norm))
            or float(gradient_clip_norm) <= 0.0):
        raise ValueError(
            'gradient_clip_norm must be finite and positive')

    batch_size = batch.batch_size
    if batch_size < 1:
        raise ValueError('advantage-gated MoE batch must be non-empty')
    if explorer_selected.shape != (batch_size,):
        raise ValueError(
            'explorer_selected must have shape (B,)')
    if explorer_selected.dtype != torch.bool:
        raise TypeError('explorer_selected must have dtype torch.bool')
    if explorer_selected.device != batch.task.device:
        raise ValueError(
            'explorer_selected must share the batch device')
    if batch.task.device != actor.q_mid.device:
        raise ValueError('actor and batch must share one device')
    if batch.task.dtype != actor.q_mid.dtype:
        raise ValueError('batch dtype must match actor dtype')
    if (batch.task.shape != (batch_size, actor.config.task_dim)
            or batch.q_projected.shape
            != (batch_size, actor.config.q_dim)):
        raise ValueError(
            'batch task or q_projected shape does not match the actor')
    if (not bool(torch.isfinite(batch.task).all())
            or not bool(torch.isfinite(batch.q_projected).all())):
        raise ValueError(
            'advantage-gated MoE task and q_projected must be finite')

    trunk_parameters = list(actor.trunk.parameters())
    baseline_parameters = list(actor.experts[0].parameters())
    specialist_parameters = [
        parameter
        for expert in actor.experts[1:]
        for parameter in expert.parameters()
    ]
    gate_parameters = list(actor.gate.parameters())
    if any(parameter.requires_grad for parameter in trunk_parameters):
        raise ValueError(
            'advantage-gated MoE requires a frozen trunk')
    if any(parameter.requires_grad for parameter in baseline_parameters):
        raise ValueError(
            'advantage-gated MoE requires frozen expert 0')
    if not all(
            parameter.requires_grad
            for parameter in gate_parameters + specialist_parameters):
        raise ValueError(
            'advantage-gated MoE requires trainable gate and specialists')
    trainable_parameters = gate_parameters + specialist_parameters
    expected_parameter_ids = [id(parameter)
                              for parameter in trainable_parameters]
    optimizer_parameters = [
        parameter
        for group in actor_optimizer.param_groups
        for parameter in group['params']
    ]
    optimizer_parameter_ids = [
        id(parameter) for parameter in optimizer_parameters]
    if (len(set(optimizer_parameter_ids)) != len(optimizer_parameter_ids)
            or len(optimizer_parameter_ids) != len(expected_parameter_ids)
            or set(optimizer_parameter_ids)
            != set(expected_parameter_ids)):
        raise ValueError(
            'actor_optimizer must contain exactly the trainable gate and '
            'specialist parameters')
    actor_parameters = list(actor.parameters())
    if any(parameter.device != actor.q_mid.device
           for parameter in actor_parameters):
        raise ValueError('all MoE parameters must share the actor device')
    if any(parameter.dtype != actor.q_mid.dtype
           for parameter in actor_parameters):
        raise TypeError('all MoE parameters must share the actor dtype')
    if not all(bool(torch.isfinite(parameter).all())
               for parameter in actor_parameters):
        raise FloatingPointError('advantage-gated MoE parameters are not finite')

    features = actor._features(batch.task)
    gate_logits = actor.gate(features)
    if not bool(torch.isfinite(gate_logits).all()):
        raise FloatingPointError(
            'advantage-gated MoE gate logits are not finite')
    positive = explorer_selected
    positive_count = int(torch.count_nonzero(positive))
    negative_count = batch_size - positive_count
    gate_target = torch.zeros(
        batch_size, dtype=torch.long, device=batch.task.device)

    if positive_count > 0:
        positive_features = features[positive]
        specialist_raw = torch.stack([
            expert(positive_features) for expert in actor.experts[1:]
        ], dim=1)
        specialist_normalised_q = (
            actor.config.limit_fraction * torch.tanh(specialist_raw))
        specialist_q = (
            actor.q_mid.to(dtype=batch.task.dtype).view(1, 1, -1)
            + actor.q_half.to(dtype=batch.task.dtype).view(1, 1, -1)
            * specialist_normalised_q
        )
        target = batch.q_projected[positive].detach()
        specialist_distance = (
            (specialist_q - target.unsqueeze(1))
            / actor.q_half.to(dtype=batch.task.dtype).view(1, 1, -1)
        ).square().mean(dim=-1)
        if not bool(torch.isfinite(specialist_distance).all()):
            raise FloatingPointError(
                'advantage-gated specialist distance is not finite')
        specialist_winner = (
            specialist_distance.detach().argmin(dim=-1))
        absolute_winner = specialist_winner + 1
        gate_target[positive] = absolute_winner
        imitation_loss = specialist_distance.gather(
            1, specialist_winner.unsqueeze(-1)).mean()

        positive_specialist_probability = F.softmax(
            gate_logits[positive, 1:], dim=-1).mean(dim=0)
        uniform_probability = 1.0 / (actor.config.n_experts - 1)
        specialist_load_balance_loss = (
            (actor.config.n_experts - 1)
            * (
                positive_specialist_probability - uniform_probability
            ).square().sum()
        )
    else:
        specialist_winner = torch.empty(
            0, dtype=torch.long, device=batch.task.device)
        imitation_loss = gate_logits.sum() * 0.0
        specialist_load_balance_loss = gate_logits.sum() * 0.0

    gate_ce_per_row = F.cross_entropy(
        gate_logits, gate_target, reduction='none')
    gate_row_weight = torch.ones_like(gate_ce_per_row)
    gate_row_weight[positive] = float(positive_gate_weight)
    gate_weight_sum = gate_row_weight.sum()
    weighted_gate_ce_loss = (
        (gate_ce_per_row * gate_row_weight).sum()
        / gate_weight_sum.clamp_min(
            torch.finfo(gate_ce_per_row.dtype).tiny)
    )
    loss = (
        imitation_loss
        + float(gate_ce_weight) * weighted_gate_ce_loss
        + float(specialist_load_balance_weight)
        * specialist_load_balance_loss
    )
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(
            'advantage-gated MoE loss is not finite')

    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        trainable_parameters,
        float(gradient_clip_norm),
        error_if_nonfinite=True)
    actor_optimizer.step()
    if not all(bool(torch.isfinite(parameter).all())
               for parameter in trainable_parameters):
        raise FloatingPointError(
            'advantage-gated MoE optimizer produced non-finite parameters')

    with torch.no_grad():
        gate_choice = gate_logits.argmax(dim=-1)
        winner_count = torch.bincount(
            specialist_winner,
            minlength=actor.config.n_experts - 1)
        metrics = {
            'moe_advantage_actor_updated': 1.0,
            'moe_advantage_actor_loss': float(loss.detach()),
            'moe_advantage_imitation_loss': float(
                imitation_loss.detach()),
            'moe_advantage_gate_ce_loss': float(
                weighted_gate_ce_loss.detach()),
            'moe_advantage_load_balance_loss': float(
                specialist_load_balance_loss.detach()),
            'moe_advantage_gate_ce_weight': float(gate_ce_weight),
            'moe_advantage_positive_gate_weight': float(
                positive_gate_weight),
            'moe_advantage_load_balance_weight': float(
                specialist_load_balance_weight),
            'moe_advantage_positive_count': float(positive_count),
            'moe_advantage_negative_count': float(negative_count),
            'moe_advantage_positive_fraction': float(
                positive.float().mean()),
            'moe_advantage_gate_target_accuracy': float(
                (gate_choice == gate_target).float().mean()),
            'moe_advantage_positive_gate_accuracy': (
                float(
                    (gate_choice[positive] == gate_target[positive])
                    .float().mean())
                if positive_count > 0 else 0.0),
            'moe_advantage_negative_gate_accuracy': (
                float((gate_choice[~positive] == 0).float().mean())
                if negative_count > 0 else 0.0),
            'actor_grad_norm': float(actor_grad.detach()),
        }
        for specialist_index, count in enumerate(
                winner_count.tolist(), start=1):
            metrics[
                f'moe_advantage_winner_expert_'
                f'{specialist_index}_fraction'
            ] = (
                float(count / positive_count)
                if positive_count > 0 else 0.0)
        return metrics


def update_direct_seed_precision(
    actor: DirectSeedActor,
    actor_optimizer: torch.optim.Optimizer,
    batch: DirectSeedRLBatch,
    kin,
    config: DirectSeedRLConfig | None = None,
    *,
    collision=None,
    projection_weight: float = 0.25,
) -> dict[str, float]:
    """Apply one cheap geometry-only update to the deployed actor mean.

    This training-only auxiliary step needs neither a critic nor a controller
    rollout.  It evaluates differentiable FK, cone, joint-limit, and optional
    collision precision on every task.  Tasks whose single IK refinement
    succeeded additionally self-distil that projected joint vector.

    ``precision_weight`` in :class:`DirectSeedRLConfig` controls how geometry
    is mixed into the macro-Q actor objective; it is intentionally not applied
    here because geometry is the complete objective of this standalone step.
    """
    config = DirectSeedRLConfig() if config is None else config
    if not isinstance(actor, DirectSeedActor):
        raise TypeError('actor must be a DirectSeedActor')
    if not isinstance(actor_optimizer, torch.optim.Optimizer):
        raise TypeError('actor_optimizer must be a torch optimizer')
    if not isinstance(batch, DirectSeedRLBatch):
        raise TypeError('batch must be a DirectSeedRLBatch')
    if not isinstance(config, DirectSeedRLConfig):
        raise TypeError('config must be a DirectSeedRLConfig')
    if not math.isfinite(projection_weight) or projection_weight < 0.0:
        raise ValueError(
            'projection_weight must be finite and non-negative')
    if batch.task.device != actor.q_mid.device:
        raise ValueError('actor and batch must share one device')
    if batch.task.dtype != actor.q_mid.dtype:
        raise ValueError('batch dtype must match actor dtype')

    actor_parameters = {
        id(parameter)
        for parameter in actor.parameters()
        if parameter.requires_grad
    }
    optimizer_parameters = {
        id(parameter)
        for group in actor_optimizer.param_groups
        for parameter in group['params']
    }
    if optimizer_parameters != actor_parameters:
        raise ValueError(
            'actor_optimizer must contain exactly the trainable actor '
            'parameters')

    # Optimize the exact action used at deployment, never a stochastic sample.
    actor_mean = actor.mean_q(batch.task)
    precision, precision_stats = _precision_per_task(
        actor, kin, collision, batch.task, actor_mean, config)
    refined = batch.route == ROUTE_REFINED
    zero = actor_mean.sum() * 0.0
    projection_distill = (
        _normalised_squared_distance(
            actor_mean[refined],
            batch.q_projected[refined].detach(),
            actor.q_half).mean()
        if bool(refined.any()) else zero)
    loss = precision.mean() + float(projection_weight) * projection_distill

    actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), config.gradient_clip_norm)
    actor_optimizer.step()

    with torch.no_grad():
        return {
            'precision_actor_updated': 1.0,
            'precision_actor_loss': float(loss),
            'precision_loss': float(precision.mean()),
            'position_error_mean_m': float(
                precision_stats['position_error_m'].mean()),
            'position_loss': float(
                precision_stats['position_loss'].mean()),
            'cone_cosine_mean': float(
                precision_stats['cone_cosine'].mean()),
            'cone_loss': float(
                precision_stats['cone_loss'].mean()),
            'joint_margin_mean_rad': float(
                precision_stats['joint_margin_rad'].mean()),
            'joint_loss': float(
                precision_stats['joint_loss'].mean()),
            'collision_margin_mean_m': float(
                precision_stats['collision_margin_m'].mean()),
            'collision_precision_loss': float(
                precision_stats['collision_loss'].mean()),
            'collision_precision_available': float(
                precision_stats['collision_available'].mean()),
            'projection_distill_loss': float(projection_distill),
            'projection_refined_fraction': float(refined.float().mean()),
            'projection_weight': float(projection_weight),
            'actor_grad_norm': float(actor_grad),
        }


def update_direct_seed_rl(
    actor: DirectSeedActor,
    critic: TwinMacroQ,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    batch: DirectSeedRLBatch,
    kin,
    config: DirectSeedRLConfig | None = None,
    *,
    collision=None,
    update_actor: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Regress twin-Q and optionally apply one delayed actor update.

    ``update_actor=False`` is the inexpensive critic-only path used between
    delayed policy updates.  The default retains the original no-replay API.
    """
    config = DirectSeedRLConfig() if config is None else config
    if not isinstance(actor, DirectSeedActor):
        raise TypeError('actor must be a DirectSeedActor')
    if not isinstance(critic, TwinMacroQ):
        raise TypeError('critic must be a TwinMacroQ')
    if not isinstance(batch, DirectSeedRLBatch):
        raise TypeError('batch must be a DirectSeedRLBatch')
    if not isinstance(config, DirectSeedRLConfig):
        raise TypeError('config must be a DirectSeedRLConfig')
    actor_device = actor.q_mid.device
    if batch.task.device != actor_device or critic.q_mid.device != actor_device:
        raise ValueError('actor, critic, and batch must share one device')
    if batch.task.dtype != actor.q_mid.dtype:
        raise ValueError('batch dtype must match actor dtype')
    if not torch.equal(actor.q_lower, critic.q_lower) \
            or not torch.equal(actor.q_upper, critic.q_upper):
        raise ValueError('actor and critic joint limits must match')

    # Backward credit: regress both critics directly to the complete downstream
    # controller Monte-Carlo progress of the raw seed macro-action.
    q1, q2 = critic(batch.task, batch.q_raw)
    refine = batch.route == ROUTE_REFINED
    fallback_or_invalid = (
        (batch.route == ROUTE_FALLBACK)
        | (batch.route == ROUTE_INVALID))
    target = (
        batch.progress_m
        - config.refine_route_penalty_m * refine.to(batch.progress_m.dtype)
        - config.fallback_route_penalty_m
        * fallback_or_invalid.to(batch.progress_m.dtype)
    ).detach()
    critic_loss = (
        F.huber_loss(
            q1, target, delta=config.critic_huber_delta_m)
        + F.huber_loss(
            q2, target, delta=config.critic_huber_delta_m))
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), config.gradient_clip_norm)
    critic_optimizer.step()

    with torch.no_grad():
        route_count = {
            code: float((batch.route == code).float().mean())
            for code in (
                ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_INVALID)
        }
        common_stats = {
            'critic_loss': float(critic_loss),
            'critic_q1_mean_m': float(q1.mean()),
            'critic_q2_mean_m': float(q2.mean()),
            'target_progress_mean_m': float(batch.progress_m.mean()),
            'critic_shaped_target_mean_m': float(target.mean()),
            'critic_route_penalty_mean_m': float(
                (batch.progress_m - target).mean()),
            'route_direct_fraction': route_count[ROUTE_DIRECT],
            'route_refined_fraction': route_count[ROUTE_REFINED],
            'route_fallback_fraction': route_count[ROUTE_FALLBACK],
            'route_invalid_fraction': route_count[ROUTE_INVALID],
            'critic_grad_norm': float(critic_grad),
        }
    if not update_actor:
        actor_optimizer.zero_grad(set_to_none=True)
        return {
            **common_stats,
            'actor_updated': 0.0,
        }

    # Freeze critic weights while retaining dQ/dq for the reparameterized
    # contextual actor action.
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    # Deployment uses the deterministic mean.  Optimize Q and every geometric
    # precision term on that exact action; stochasticity is retained only for
    # the entropy estimate and for explicit data collection in the runner.
    mean_action = actor.sample(batch.task, deterministic=True)
    entropy_action = actor.sample(batch.task, generator=generator)
    new_q1, new_q2 = critic(batch.task, mean_action.q)
    conservative_q = torch.minimum(new_q1, new_q2)
    policy_per_task = (
        -conservative_q
        + config.entropy_coef * entropy_action.log_prob)
    precision, precision_stats = _precision_per_task(
        actor, kin, collision, batch.task, mean_action.q, config)
    actor_mean = mean_action.q

    refined = batch.route == ROUTE_REFINED
    fallback = batch.route == ROUTE_FALLBACK
    failed_route = fallback | (batch.route == ROUTE_INVALID)
    zero = actor_mean.sum() * 0.0
    projection_distill = (
        _normalised_squared_distance(
            actor_mean[refined],
            batch.q_projected[refined].detach(),
            actor.q_half).mean()
        if bool(refined.any()) else zero)
    fallback_distill = (
        _normalised_squared_distance(
            actor_mean[fallback],
            batch.fallback_q[fallback].detach(),
            actor.q_half).mean()
        if bool(fallback.any()) else zero)
    behavior_anchor = _normalised_squared_distance(
        actor_mean, batch.q_raw.detach(), actor.q_half).mean()
    # A failed route supplies no differentiable router gradient.  Re-evaluate
    # the current sampled action's FK precision on exactly those contexts,
    # giving the actor a physical direction out of the failure region.
    failure_precision = (
        precision[failed_route].mean()
        if bool(failed_route.any()) else zero)

    actor_loss = (
        policy_per_task.mean()
        + config.precision_weight * precision.mean()
        + config.projection_distill_weight * projection_distill
        + config.fallback_distill_weight * fallback_distill
        + config.failure_precision_weight * failure_precision
        + config.behavior_anchor_weight * behavior_anchor)
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(
        actor.parameters(), config.gradient_clip_norm)
    actor_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)

    with torch.no_grad():
        return {
            **common_stats,
            'actor_updated': 1.0,
            'actor_loss': float(actor_loss),
            'actor_q_mean_m': float(conservative_q.mean()),
            'actor_log_prob_mean': float(entropy_action.log_prob.mean()),
            'precision_loss': float(precision.mean()),
            'position_error_mean_m': float(
                precision_stats['position_error_m'].mean()),
            'cone_cosine_mean': float(
                precision_stats['cone_cosine'].mean()),
            'joint_margin_mean_rad': float(
                precision_stats['joint_margin_rad'].mean()),
            'collision_margin_mean_m': float(
                precision_stats['collision_margin_m'].mean()),
            'collision_precision_loss': float(
                precision_stats['collision_loss'].mean()),
            'collision_precision_available': float(
                precision_stats['collision_available'].mean()),
            'projection_distill_loss': float(projection_distill),
            'fallback_distill_loss': float(fallback_distill),
            'failure_precision_loss': float(failure_precision),
            'behavior_anchor_loss': float(behavior_anchor),
            'actor_grad_norm': float(actor_grad),
        }


def direct_seed_moe_checkpoint(
    actor: DirectSeedMoEActor,
    *,
    update_step: int = 0,
    actor_optimizer: torch.optim.Optimizer | None = None,
    metadata: Mapping | None = None,
) -> dict:
    """Return a self-contained hard-MoE deployment checkpoint."""
    if not isinstance(actor, DirectSeedMoEActor):
        raise TypeError('actor must be a DirectSeedMoEActor')
    if isinstance(update_step, bool) or not isinstance(update_step, int):
        raise TypeError('update_step must be an integer')
    if update_step < 0:
        raise ValueError('update_step must be non-negative')
    if actor_optimizer is not None \
            and not isinstance(
                actor_optimizer, torch.optim.Optimizer):
        raise TypeError('actor_optimizer must be a torch optimizer')
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError('metadata must be a mapping')
    payload = {
        'format': 'direct-seed-hard-moe-v1',
        'actor_config': asdict(actor.config),
        'q_lower': actor.q_lower.detach().cpu(),
        'q_upper': actor.q_upper.detach().cpu(),
        'task_mean': actor.task_mean.detach().cpu(),
        'task_std': actor.task_std.detach().cpu(),
        'actor': {
            name: value.detach().cpu()
            for name, value in actor.state_dict().items()
        },
        'update_step': int(update_step),
        'metadata': (
            {} if metadata is None else copy.deepcopy(dict(metadata))),
    }
    if actor_optimizer is not None:
        payload['actor_optimizer'] = copy.deepcopy(
            actor_optimizer.state_dict())
    return payload


def load_direct_seed_moe_checkpoint(
    checkpoint: str | Path | Mapping,
    device: torch.device | str = 'cpu',
) -> tuple[DirectSeedMoEActor, Mapping | None, Mapping]:
    """Load a hard-MoE actor without touching legacy actor schemas."""
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location='cpu', weights_only=False)
    elif isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        raise TypeError('checkpoint must be a path or mapping')
    required = {
        'format', 'actor_config', 'q_lower', 'q_upper',
        'task_mean', 'task_std', 'actor', 'update_step', 'metadata',
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f'direct-seed MoE checkpoint is missing keys: '
            f'{sorted(missing)}')
    if payload['format'] != 'direct-seed-hard-moe-v1':
        raise ValueError(
            f"unsupported direct-seed MoE format {payload['format']!r}")
    config = DirectSeedMoEActorConfig(
        **dict(payload['actor_config']))
    actor = DirectSeedMoEActor(
        payload['q_lower'], payload['q_upper'], config,
        task_mean=payload['task_mean'], task_std=payload['task_std'])
    actor.load_state_dict(payload['actor'], strict=True)
    actor.to(device).eval()
    update_step = payload['update_step']
    if isinstance(update_step, bool) or not isinstance(update_step, int) \
            or update_step < 0:
        raise ValueError(
            'direct-seed MoE update_step must be a non-negative integer')
    if not isinstance(payload['metadata'], Mapping):
        raise ValueError(
            'direct-seed MoE metadata must be a mapping')
    return actor, payload.get('actor_optimizer'), payload


def direct_seed_rl_checkpoint(
    actor: DirectSeedActor,
    critic: TwinMacroQ,
    config: DirectSeedRLConfig,
    *,
    update_step: int,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    metadata: Mapping | None = None,
) -> dict:
    """Build a self-contained contextual-RL checkpoint."""
    if update_step < 0:
        raise ValueError('update_step must be non-negative')
    if not torch.equal(actor.q_lower, critic.q_lower) \
            or not torch.equal(actor.q_upper, critic.q_upper):
        raise ValueError('actor and critic joint limits must match')
    if not torch.equal(actor.task_mean, critic.task_mean) \
            or not torch.equal(actor.task_std, critic.task_std):
        raise ValueError('actor and critic task normalization must match')
    payload = {
        'format': 'direct-seed-contextual-rl-v1',
        'actor_config': asdict(actor.config),
        'critic_config': asdict(critic.config),
        'rl_config': asdict(config),
        'q_lower': actor.q_lower.detach().cpu(),
        'q_upper': actor.q_upper.detach().cpu(),
        'task_mean': actor.task_mean.detach().cpu(),
        'task_std': actor.task_std.detach().cpu(),
        'actor': {
            name: value.detach().cpu()
            for name, value in actor.state_dict().items()
        },
        'critic': {
            name: value.detach().cpu()
            for name, value in critic.state_dict().items()
        },
        'update_step': int(update_step),
        'metadata': copy.deepcopy(dict(metadata or {})),
    }
    if actor_optimizer is not None:
        payload['actor_optimizer'] = copy.deepcopy(
            actor_optimizer.state_dict())
    if critic_optimizer is not None:
        payload['critic_optimizer'] = copy.deepcopy(
            critic_optimizer.state_dict())
    return payload


def load_direct_seed_rl_checkpoint(
    checkpoint: str | Path | Mapping,
    device: torch.device | str = 'cpu',
) -> tuple[
    DirectSeedActor,
    TwinMacroQ,
    Mapping | None,
    Mapping | None,
    Mapping,
]:
    """Load networks plus optional optimizer states and the full payload."""
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(
            Path(checkpoint), map_location='cpu', weights_only=False)
    elif isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        raise TypeError('checkpoint must be a path or mapping')
    required = {
        'format', 'actor_config', 'critic_config', 'rl_config',
        'q_lower', 'q_upper', 'task_mean', 'task_std',
        'actor', 'critic', 'update_step', 'metadata',
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f'direct-seed RL checkpoint is missing: {sorted(missing)}')
    if payload['format'] != 'direct-seed-contextual-rl-v1':
        raise ValueError(
            f"unsupported direct-seed RL format {payload['format']!r}")
    actor = DirectSeedActor(
        payload['q_lower'], payload['q_upper'],
        DirectSeedActorConfig(**dict(payload['actor_config'])),
        task_mean=payload['task_mean'], task_std=payload['task_std'])
    critic = TwinMacroQ(
        payload['q_lower'], payload['q_upper'],
        DirectSeedCriticConfig(**dict(payload['critic_config'])),
        task_mean=payload['task_mean'], task_std=payload['task_std'])
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    actor.to(device).eval()
    critic.to(device).eval()
    # Validate the serialized algorithm config even though callers reconstruct
    # it from the retained payload.  This fails early on schema drift.
    DirectSeedRLConfig(**dict(payload['rl_config']))
    return (
        actor,
        critic,
        payload.get('actor_optimizer'),
        payload.get('critic_optimizer'),
        payload,
    )


def synthetic_direct_seed_rl_smoke(
    device: torch.device | str = 'cpu',
    *,
    include_collision: bool = False,
) -> dict[str, float]:
    """Run one small synthetic update without a controller or filesystem."""
    device = torch.device(device)

    class _IdentityKin:
        lmt_lo = -torch.ones(7, device=device)
        lmt_up = torch.ones(7, device=device)
        q_mid = torch.zeros(7, device=device)

        @staticmethod
        def tcp_fk_jac(q):
            batch_size = q.shape[0]
            position = q[:, :3]
            rotation = torch.eye(
                3, device=q.device, dtype=q.dtype).expand(
                    batch_size, -1, -1).clone()
            jacobian = torch.zeros(
                (batch_size, 6, 7), device=q.device, dtype=q.dtype)
            transforms = torch.eye(
                4, device=q.device, dtype=q.dtype).expand(
                    batch_size, 1, -1, -1).clone()
            return position, rotation, jacobian, transforms

        @staticmethod
        def link_transforms(q):
            transforms = torch.eye(
                4, device=q.device, dtype=q.dtype).expand(
                    q.shape[0], 1, -1, -1).clone()
            transforms[:, 0, 0, 3] = q[:, 3]
            return transforms

    class _Collision:
        @staticmethod
        def min_margin(link_transforms):
            # Positive and differentiable around the synthetic samples.
            return 0.05 - link_transforms[:, 0, 0, 3].square()

    torch.manual_seed(20260728)
    actor_config = DirectSeedActorConfig(
        hidden_dim=32, n_hidden_layers=2)
    critic_config = DirectSeedCriticConfig(
        hidden_dim=32, n_hidden_layers=2)
    actor = DirectSeedActor(
        -torch.ones(7), torch.ones(7), actor_config).to(device)
    critic = TwinMacroQ(
        -torch.ones(7), torch.ones(7), critic_config).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    batch_size = 8
    task = torch.zeros((batch_size, 9), device=device)
    task[:, 3] = 1.0
    task[:, 8] = 1.0
    with torch.no_grad():
        q_raw = actor.sample(task).q
    q_projected = q_raw.clone()
    q_projected[:, :3] = task[:, :3]
    fallback_q = torch.zeros_like(q_raw)
    progress = 0.5 - 0.01 * q_raw.square().sum(dim=-1)
    route = torch.tensor([
        ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_INVALID,
        ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_DIRECT,
    ], device=device, dtype=torch.int8)
    batch = DirectSeedRLBatch(
        task=task,
        q_raw=q_raw,
        q_projected=q_projected,
        fallback_q=fallback_q,
        progress_m=progress,
        route=route)
    return update_direct_seed_rl(
        actor, critic, actor_optimizer, critic_optimizer,
        batch, _IdentityKin(), DirectSeedRLConfig(),
        collision=(_Collision() if include_collision else None))


if __name__ == '__main__':
    print(synthetic_direct_seed_rl_smoke())


__all__ = [
    'DirectSeedAction',
    'DirectSeedActor',
    'DirectSeedActorConfig',
    'DirectSeedCriticConfig',
    'DirectSeedEliteMemory',
    'DirectSeedMacroReplay',
    'DirectSeedMoEActor',
    'DirectSeedMoEActorConfig',
    'DirectSeedPairedArchive',
    'DirectSeedRLBatch',
    'DirectSeedRLConfig',
    'TwinMacroQ',
    'direct_seed_moe_checkpoint',
    'direct_seed_moe_from_actor',
    'direct_seed_rl_checkpoint',
    'load_direct_seed_moe_checkpoint',
    'load_direct_seed_rl_checkpoint',
    'synthetic_direct_seed_rl_smoke',
    'update_direct_seed_moe_advantage',
    'update_direct_seed_moe_projection',
    'update_direct_seed_projection',
    'update_direct_seed_precision',
    'update_direct_seed_rl',
]
