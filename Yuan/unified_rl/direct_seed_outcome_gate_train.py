"""Fit an auditable outcome gate between two frozen direct-seed branches.

The specialists are trained elsewhere from real controller outcomes.  This
script changes only the hard gate: expert 0 remains the exact contextual-RL
baseline and experts ``1..K-1`` remain exact specialist branches.  Labels are
task-paired return comparisons under one frozen controller version.

Deployment still performs one deterministic network forward, emits one joint
seed, uses at most one IK refinement, and executes one controller rollout.
No candidate, critic, return model, or controller probe is queried online.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    adapt_controller_observation_state_dict,
)
from Yuan.unified_rl.direct_seed_projection import (
    DirectSeedProjectionConfig,
    ROUTE_NAMES,
)
from Yuan.unified_rl.direct_seed_rl import (
    DirectSeedActor,
    DirectSeedMoEActor,
    DirectSeedPairedArchive,
    direct_seed_moe_checkpoint,
    load_direct_seed_moe_checkpoint,
    load_direct_seed_rl_checkpoint,
)
from Yuan.unified_rl.provenance import (
    file_fingerprint,
    state_dict_fingerprint,
)


_RUNNER_FORMAT = 'direct-seed-bidirectional-v1'
_MOE_FORMAT = 'direct-seed-hard-moe-v1'
_BASE_FORMAT = 'direct-seed-contextual-rl-v1'
_DEFAULT_QUOTAS = (
    0.01, 0.02, 0.03, 0.05, 0.075, 0.10,
    0.125, 0.15, 0.175, 0.20, 0.25, 0.30,
)
_DISABLED_GATE_LOGIT = -1e6


def _require_mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    return value


def _tensor_mapping_equal(
    actual: Mapping,
    expected: Mapping,
    *,
    label: str,
    ignored_prefixes: Sequence[str] = (),
) -> None:
    """Require bitwise equality for two tensor mappings."""
    actual = _require_mapping(actual, f'{label} actual state')
    expected = _require_mapping(expected, f'{label} expected state')

    def included(name: str) -> bool:
        return not any(name.startswith(prefix) for prefix in ignored_prefixes)

    actual_keys = {str(key) for key in actual if included(str(key))}
    expected_keys = {str(key) for key in expected if included(str(key))}
    if actual_keys != expected_keys:
        raise ValueError(
            f'{label} tensor keys differ: '
            f'actual_only={sorted(actual_keys - expected_keys)[:8]}, '
            f'expected_only={sorted(expected_keys - actual_keys)[:8]}')
    for name in sorted(actual_keys):
        left = actual[name]
        right = expected[name]
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f'{label} field {name!r} is not a tensor')
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ValueError(f'{label} tensor {name!r} differs')


def _validate_moe_baseline_branch(
    moe: DirectSeedMoEActor,
    base: DirectSeedActor,
) -> dict[str, Any]:
    """Prove that MoE trunk/expert 0 is the exact old actor."""
    if not isinstance(moe, DirectSeedMoEActor):
        raise TypeError('moe must be a DirectSeedMoEActor')
    if not isinstance(base, DirectSeedActor):
        raise TypeError('base must be a DirectSeedActor')
    if moe.config.n_experts < 2:
        raise ValueError('outcome gate requires a baseline and specialists')
    if not moe.config.exact_baseline_head:
        raise ValueError('outcome gate requires exact_baseline_head=True')

    shared = (
        'task_dim', 'q_dim', 'hidden_dim',
        'n_hidden_layers', 'limit_fraction',
    )
    for name in shared:
        if getattr(moe.config, name) != getattr(base.config, name):
            raise ValueError(
                f'MoE/base actor config differs for {name!r}')
    for name in ('q_lower', 'q_upper', 'task_mean', 'task_std'):
        if not torch.equal(
                getattr(moe, name).detach().cpu(),
                getattr(base, name).detach().cpu()):
            raise ValueError(f'MoE/base actor buffer {name!r} differs')

    source_modules = list(base.trunk.children())
    target_modules = list(moe.trunk.children())
    expected_source = 2 * base.config.n_hidden_layers + 1
    expected_target = 2 * base.config.n_hidden_layers
    if (len(source_modules) != expected_source
            or len(target_modules) != expected_target):
        raise ValueError('MoE/base actor has an unexpected trunk layout')
    for layer in range(base.config.n_hidden_layers):
        source = source_modules[2 * layer]
        target = target_modules[2 * layer]
        if not isinstance(source, torch.nn.Linear) \
                or not isinstance(target, torch.nn.Linear):
            raise ValueError('MoE/base hidden layer is not linear')
        if not torch.equal(
                target.weight.detach().cpu(),
                source.weight.detach().cpu()):
            raise ValueError(f'MoE/base hidden weight {layer} differs')
        if not torch.equal(
                target.bias.detach().cpu(),
                source.bias.detach().cpu()):
            raise ValueError(f'MoE/base hidden bias {layer} differs')

    source_head = source_modules[-1]
    if not isinstance(source_head, torch.nn.Linear):
        raise ValueError('base actor output head is not linear')
    expert_zero = moe.experts[0]
    if (expert_zero.out_features != 2 * base.config.q_dim
            or source_head.out_features != 2 * base.config.q_dim):
        raise ValueError('exact baseline heads must retain all 14 outputs')
    if not torch.equal(
            expert_zero.weight.detach().cpu(),
            source_head.weight.detach().cpu()):
        raise ValueError('MoE expert 0 weight differs from base output head')
    if not torch.equal(
            expert_zero.bias.detach().cpu(),
            source_head.bias.detach().cpu()):
        raise ValueError('MoE expert 0 bias differs from base output head')
    return {
        'n_experts': moe.config.n_experts,
        'exact_baseline_head': True,
        'hidden_linear_layers_checked': base.config.n_hidden_layers,
        'expert0_full_output_rows_checked': 2 * base.config.q_dim,
        'bitwise_equal': True,
    }


def _validate_forced_specialist(
    moe_payload: Mapping,
    forced_payload: Mapping,
    *,
    moe_sha256: str,
    expert_index: int = 1,
) -> dict[str, Any]:
    """Require a model-identical checkpoint forcing one specialist."""
    if moe_payload.get('format') != _MOE_FORMAT \
            or forced_payload.get('format') != _MOE_FORMAT:
        raise ValueError('both specialist checkpoints must be hard MoE v1')
    if dict(forced_payload.get('actor_config', {})) != dict(
            moe_payload.get('actor_config', {})):
        raise ValueError('forced-specialist actor config differs from MoE')
    for name in ('q_lower', 'q_upper', 'task_mean', 'task_std'):
        left = forced_payload.get(name)
        right = moe_payload.get(name)
        if not torch.is_tensor(left) or not torch.is_tensor(right) \
                or not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ValueError(
                f'forced-specialist top-level tensor {name!r} differs')
    _tensor_mapping_equal(
        forced_payload.get('actor', {}),
        moe_payload.get('actor', {}),
        label='forced-specialist/MoE non-gate state',
        ignored_prefixes=('gate.',))
    forced_actor = _require_mapping(
        forced_payload.get('actor'), 'forced-specialist actor')
    weight = forced_actor.get('gate.weight')
    bias = forced_actor.get('gate.bias')
    config = _require_mapping(
        moe_payload.get('actor_config'), 'MoE actor config')
    n_experts = config.get('n_experts')
    if (isinstance(n_experts, bool) or not isinstance(n_experts, int)
            or not 2 <= n_experts
            or not 1 <= expert_index < n_experts):
        raise ValueError('forced-specialist expert index is invalid')
    if (not torch.is_tensor(weight) or tuple(weight.shape)[0] != n_experts
            or not bool(torch.isfinite(weight).all())
            or not bool((weight == 0).all())):
        raise ValueError(
            'forced-specialist gate weight must be finite and exactly zero')
    if (not torch.is_tensor(bias)
            or tuple(bias.shape) != (n_experts,)
            or not bool(torch.isfinite(bias).all())
            or int(bias.argmax()) != expert_index
            or int(torch.count_nonzero(
                bias == bias[expert_index])) != 1):
        raise ValueError(
            'forced-specialist gate bias must uniquely select the requested '
            'expert')
    if forced_payload.get('update_step') != moe_payload.get('update_step'):
        raise ValueError('forced-specialist update_step differs from MoE')
    metadata = _require_mapping(
        forced_payload.get('metadata'), 'forced-specialist metadata')
    if metadata.get(
            'method'
    ) != 'forced-hard-moe-branch-training-collection-only':
        raise ValueError(
            'forced-specialist metadata has an unsupported method')
    if metadata.get('forced_expert_index') != expert_index:
        raise ValueError(
            'forced-specialist metadata identifies a different expert')
    recorded_source = metadata.get('source_checkpoint_sha256')
    if recorded_source != moe_sha256:
        raise ValueError(
            'forced-specialist source checkpoint hash differs from MoE')
    return {
        'non_gate_state_bitwise_equal': True,
        'gate_weight_exact_zero': True,
        'forced_expert_index': expert_index,
        'source_hash_verified': True,
    }


def _nested_exact(actual: Any, expected: Any, label: str = 'value') -> None:
    """Recursively require exact equality, including tensor bytes."""
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or not torch.is_tensor(expected)
                or not torch.equal(
                    actual.detach().cpu(), expected.detach().cpu())):
            raise ValueError(f'{label} differs')
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            raise ValueError(f'{label} type differs')
        if set(actual) != set(expected):
            raise ValueError(f'{label} keys differ')
        for key in actual:
            _nested_exact(actual[key], expected[key], f'{label}.{key}')
        return
    if isinstance(actual, (list, tuple)) \
            or isinstance(expected, (list, tuple)):
        if (not isinstance(actual, (list, tuple))
                or not isinstance(expected, (list, tuple))
                or len(actual) != len(expected)):
            raise ValueError(f'{label} sequence differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _nested_exact(left, right, f'{label}[{index}]')
        return
    if actual != expected:
        raise ValueError(f'{label} differs')


def _task_bytes(task: torch.Tensor) -> bytes:
    row = task.detach().to(
        device='cpu', dtype=torch.float32).contiguous()
    if row.shape != (9,) or not bool(torch.isfinite(row).all()):
        raise ValueError('task rows must be finite float32 vectors of length 9')
    return row.numpy().tobytes(order='C')


def _macro_replay_order(state: Mapping) -> list[int]:
    capacity = state.get('capacity')
    size = state.get('size')
    write_index = state.get('write_index')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (capacity, size, write_index)):
        raise ValueError('macro replay capacity/size/write_index are invalid')
    if capacity < 1 or not 0 <= size <= capacity \
            or not 0 <= write_index < capacity:
        raise ValueError('macro replay ring bounds are invalid')
    if size < capacity:
        if write_index != size % capacity:
            raise ValueError(
                'partially filled macro replay has inconsistent write_index')
        return list(range(size))
    return list(range(write_index, capacity)) + list(range(write_index))


def _validate_archive_against_runner(
    archive_state: Mapping,
    runner: Mapping,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Match a naked archive to embedded state or legacy first replay rows."""
    embedded = runner.get('paired_archive')
    if embedded is not None:
        _nested_exact(
            archive_state, embedded,
            'standalone/embedded paired archive')
        return {
            'method': 'embedded-paired-archive-bitwise-exact',
            'task_rows_checked': int(archive_state['outcome_count']),
            'self_certifying': True,
            'legacy_override_used': False,
        }

    if not allow_legacy:
        raise ValueError(
            'baseline runner has no embedded paired archive/provenance; '
            'pass --allow-legacy-runner-provenance only for the audited '
            'legacy P8 archive')
    replay = _require_mapping(
        runner.get('macro_replay'), 'legacy runner macro replay')
    storage = _require_mapping(
        replay.get('storage'), 'legacy runner macro replay storage')
    order = _macro_replay_order(replay)
    required = ('task', 'q_projected', 'progress_m', 'route')
    capacity = int(replay['capacity'])
    for name in required:
        value = storage.get(name)
        if not torch.is_tensor(value) or value.shape[0] != capacity:
            raise ValueError(
                f'legacy macro replay field {name!r} is invalid')

    first: dict[bytes, int] = {}
    for row in order:
        key = _task_bytes(storage['task'][row])
        first.setdefault(key, row)
    archive_storage = _require_mapping(
        archive_state.get('storage'), 'paired archive storage')
    valid = archive_storage.get('valid')
    task = archive_storage.get('task')
    if (not torch.is_tensor(valid) or valid.dtype != torch.bool
            or not bool(valid.all()) or not torch.is_tensor(task)):
        raise ValueError('paired archive must have full valid task storage')
    missing = []
    for row in range(int(task.shape[0])):
        key = _task_bytes(task[row])
        replay_row = first.get(key)
        if replay_row is None:
            missing.append(row)
            continue
        for name in ('q_projected', 'progress_m', 'route'):
            if not torch.equal(
                    archive_storage[name][row].detach().cpu(),
                    storage[name][replay_row].detach().cpu()):
                raise ValueError(
                    f'paired archive task row {row} {name!r} differs from '
                    'its first legacy macro-replay outcome')
    if missing:
        raise ValueError(
            'paired archive tasks are absent from legacy macro replay: '
            f'{missing[:20]}')
    return {
        'method': 'legacy-macro-replay-first-outcome-exact',
        'task_rows_checked': int(task.shape[0]),
        'unique_task_geometry_rows_in_replay': len(first),
        'chronological_replay_rows_checked': len(order),
        'self_certifying': False,
        'legacy_override_used': True,
    }


def _geometry_fingerprints(task: torch.Tensor) -> tuple[str, ...]:
    task = task.detach().to(
        device='cpu', dtype=torch.float32).contiguous()
    if task.ndim != 2 or task.shape[1] != 9:
        raise ValueError('task geometry must have shape (N, 9)')
    canonical = np.asarray(
        task.numpy(), dtype=np.dtype('<f4'), order='C').copy()
    if not np.isfinite(canonical).all():
        raise ValueError('task geometry must be finite')
    canonical[canonical == np.float32(0.0)] = np.float32(0.0)
    return tuple(
        hashlib.sha256(row.tobytes(order='C')).hexdigest()
        for row in canonical)


def _geometry_group_keys(task: torch.Tensor) -> tuple[str, ...]:
    """Exact float32 task bytes used as leakage-proof group identities."""
    task = task.detach().to(
        device='cpu', dtype=torch.float32).contiguous()
    if task.ndim != 2 or task.shape[1] != 9 \
            or not bool(torch.isfinite(task).all()):
        raise ValueError('task geometry must be finite with shape (N, 9)')
    return tuple(row.numpy().tobytes(order='C').hex() for row in task)


def _fingerprint_list_sha256(fingerprints: Sequence[str]) -> str:
    return hashlib.sha256(
        '\n'.join(fingerprints).encode('ascii')).hexdigest()


def _validate_contextual_payload_equal(
    actual: Mapping,
    expected: Mapping,
    *,
    label: str,
) -> None:
    if actual.get('format') != _BASE_FORMAT \
            or expected.get('format') != _BASE_FORMAT:
        raise ValueError(f'{label} must compare contextual-RL v1 payloads')
    for name in (
            'actor_config', 'q_lower', 'q_upper',
            'task_mean', 'task_std', 'actor'):
        _nested_exact(actual.get(name), expected.get(name), f'{label}.{name}')


def _validate_runner(
    runner: Mapping,
    archive: DirectSeedPairedArchive,
    archive_state: Mapping,
    base_payload: Mapping,
    *,
    allow_legacy: bool,
) -> dict[str, Any]:
    """Validate task, actor, controller, projection, and collection contract."""
    if runner.get('format') != _RUNNER_FORMAT:
        raise ValueError('baseline runner format is unsupported')
    if runner.get('phase') != 'round_complete':
        raise ValueError('baseline full runner must be round_complete')
    task_ids = archive.task_ids
    if runner.get('kept_task_indices') != task_ids.tolist():
        raise ValueError('baseline runner task order differs from archive')

    args = _require_mapping(runner.get('args'), 'baseline runner args')
    contract = {
        'deterministic_backward': True,
        'freeze_seed_actor_during_collection': True,
        'task_sampling': 'cycle',
        'forward_mode': 'skip',
        'controller_steps_per_round': 0,
    }
    for name, expected in contract.items():
        if args.get(name) != expected:
            raise ValueError(
                f'baseline runner violates frozen collection contract: '
                f'{name}={args.get(name)!r}, expected {expected!r}')
    if runner.get('task_sampling') != 'cycle':
        raise ValueError('baseline runner top-level task_sampling is not cycle')
    for name in (
            'precision_only_updates_per_rollout',
            'elite_projection_updates_per_rollout',
            'per_task_elite_updates_per_rollout',
            'per_task_elite_post_updates_per_round',
            'paired_post_updates_per_round',
    ):
        if int(args.get(name, 0)) != 0:
            raise ValueError(
                f'baseline runner changed the frozen actor via {name}')

    cycle = _require_mapping(
        runner.get('task_cycle_sampler'), 'baseline cycle sampler')
    if (cycle.get('format') != 'direct-task-cycle-sampler-v1'
            or cycle.get('n_tasks') != int(task_ids.numel())
            or int(cycle.get('total_sampled', -1)) < int(task_ids.numel())):
        raise ValueError(
            'baseline cycle sampler does not prove one full task traversal')

    expected_projection = dataclasses.asdict(DirectSeedProjectionConfig())
    if runner.get('projection_config') != expected_projection:
        raise ValueError(
            'baseline runner projection differs from deployment default')
    runner_direct = _require_mapping(
        runner.get('direct_seed'), 'baseline runner direct_seed')
    _validate_contextual_payload_equal(
        runner_direct, base_payload, label='runner/base direct actor')
    direct_metadata = _require_mapping(
        runner_direct.get('metadata'), 'runner direct_seed metadata')
    if direct_metadata.get('projection_config') != expected_projection:
        raise ValueError(
            'runner direct-seed projection metadata differs')
    deployment_metadata = {
        'one_task_one_seed': True,
        'max_ik_refinements': 1,
        'diffusion_dependency': False,
    }
    for name, expected in deployment_metadata.items():
        if direct_metadata.get(name) != expected:
            raise ValueError(
                f'runner direct-seed deployment metadata {name!r} differs')

    fingerprints = _geometry_fingerprints(archive.task)
    fingerprint_sha = _fingerprint_list_sha256(fingerprints)
    if runner.get('safe_task_fingerprint_list_sha256') != fingerprint_sha:
        raise ValueError(
            'runner safe task fingerprint differs from archive geometry')
    if direct_metadata.get(
            'safe_task_fingerprint_list_sha256') != fingerprint_sha:
        raise ValueError(
            'runner direct-seed safe task fingerprint differs')

    task_cache_path = Path(
        str(runner.get('task_cache'))).expanduser().resolve(strict=True)
    cache = CachedSeedCandidateDataset.from_npz(task_cache_path)
    selected = cache.select_source_tasks(task_ids)
    cache_task = torch.cat([
        selected.batch.p0,
        selected.batch.line_dir,
        selected.batch.n_target,
    ], dim=-1)
    if not torch.equal(cache_task, archive.task):
        raise ValueError(
            'paired archive geometry differs from runner task cache')
    if tuple(selected.task_fingerprints) != fingerprints:
        raise ValueError(
            'task cache and archive geometry fingerprints differ')

    controller = _require_mapping(
        runner.get('controller'), 'baseline runner controller')
    controller_version = runner.get('controller_update_count')
    replay_version = runner.get(
        'macro_replay_controller_update_count')
    if (isinstance(controller_version, bool)
            or not isinstance(controller_version, int)
            or controller_version < 0
            or replay_version != controller_version):
        raise ValueError(
            'baseline runner macro outcomes use a different controller version')
    if runner.get('paired_archive') is not None \
            and runner.get(
                'paired_archive_controller_update_count') != controller_version:
        raise ValueError(
            'embedded paired archive uses a different controller version')
    if runner.get('paired_archive') is not None:
        paired_contract = _require_mapping(
            runner.get('paired_collection_contract'),
            'embedded paired collection contract')
        required_paired_contract = {
            'format': 'direct-seed-paired-collection-contract-v1',
            'actor_action': 'deterministic-mean',
            'actor_frozen_during_collection': True,
            'deterministic_backward': True,
            'task_sampling': 'cycle',
            'archive_write_policy': 'first-task-outcome-immutable',
        }
        for name, expected in required_paired_contract.items():
            if paired_contract.get(name) != expected:
                raise ValueError(
                    f'embedded paired collection contract {name!r} differs')
        base_actor_sha = state_dict_fingerprint(
            dict(base_payload['actor']))
        if runner.get(
                'paired_baseline_actor_state_sha256') != base_actor_sha:
            raise ValueError(
                'embedded archive baseline actor hash differs from base')
    if controller_version == 0:
        controller_dir = Path(
            str(runner.get('init_controller_dir'))
        ).expanduser().resolve(strict=True)
        initial = torch.load(
            controller_dir / 'agent.pt',
            map_location='cpu', weights_only=False)
        initial = _require_mapping(initial, 'initial controller checkpoint')
        target_input = controller.get('_actor_trunk.0.weight')
        if not torch.is_tensor(target_input) or target_input.ndim != 2:
            raise ValueError('runner controller has no actor input weight')
        adapted = adapt_controller_observation_state_dict(
            dict(initial), int(target_input.shape[1]))
        _tensor_mapping_equal(
            controller, adapted, label='runner/initial frozen controller')

    archive_match = _validate_archive_against_runner(
        archive_state, runner, allow_legacy=allow_legacy)
    return {
        'runner_format': _RUNNER_FORMAT,
        'phase': 'round_complete',
        'task_count': int(task_ids.numel()),
        'task_order_exact': True,
        'task_cache': file_fingerprint(task_cache_path),
        'task_fingerprint_list_sha256': fingerprint_sha,
        'unique_geometry_fingerprint_count': len(set(fingerprints)),
        'projection_config': expected_projection,
        'direct_seed_actor_bitwise_equal_to_base': True,
        'deterministic_frozen_cycle_contract': contract,
        'controller_update_count': controller_version,
        'controller_state_sha256': state_dict_fingerprint(dict(controller)),
        'archive_match': archive_match,
    }


def _load_outcomes(
    path: Path,
    archive: DirectSeedPairedArchive,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ('task_indices', 'progress_m', 'route', 'valid')
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                f'specialist outcomes are missing keys: {missing}')
        result = {name: np.asarray(data[name]).copy() for name in required}
    count = int(archive.task_ids.numel())
    if result['task_indices'].shape != (count,) \
            or not np.issubdtype(
                result['task_indices'].dtype, np.integer):
        raise ValueError(
            'specialist task_indices must be an integer vector')
    if not np.array_equal(
            result['task_indices'].astype(np.int64, copy=False),
            archive.task_ids.numpy()):
        raise ValueError(
            'specialist task_indices/order differ from baseline archive')
    if result['progress_m'].shape != (count,) \
            or not np.isfinite(result['progress_m']).all():
        raise ValueError('specialist progress must be a finite vector')
    if (result['route'].shape != (count,)
            or not np.isin(
                result['route'], np.arange(len(ROUTE_NAMES))).all()):
        raise ValueError('specialist route is invalid')
    if result['valid'].shape != (count,) \
            or result['valid'].dtype != np.bool_ \
            or not result['valid'].all():
        raise ValueError(
            'specialist outcomes must mark every task valid')
    return result


def _validate_outcome_manifest(
    outcome_path: Path,
    outcomes: Mapping[str, np.ndarray],
    *,
    forced_path: Path,
    runner: Mapping,
    runner_audit: Mapping,
    allow_legacy: bool,
) -> dict[str, Any]:
    sidecar = outcome_path.with_suffix('.json')
    report = None
    if sidecar.exists():
        with sidecar.open('r', encoding='utf-8') as stream:
            report = json.load(stream)
        if not isinstance(report, Mapping):
            raise ValueError('specialist outcome JSON must be a mapping')
        if report.get('n_tasks') != len(outcomes['progress_m']):
            raise ValueError(
                'specialist outcome JSON task count differs from NPZ')
        reported_mean = report.get('progress_mean_m')
        actual_mean = float(np.mean(
            outcomes['progress_m'], dtype=np.float64))
        if (not isinstance(reported_mean, (int, float))
                or not math.isclose(
                    float(reported_mean), actual_mean,
                    rel_tol=0.0, abs_tol=1e-10)):
            raise ValueError(
                'specialist outcome JSON progress mean differs from NPZ')
        fallback_filter = report.get('fallback_strict_filter')
        if isinstance(fallback_filter, Mapping):
            if fallback_filter.get(
                    'kept_geometry_fingerprint_list_sha256'
            ) != runner_audit['task_fingerprint_list_sha256']:
                raise ValueError(
                    'specialist outcome task fingerprint differs from runner')

    artifacts = report.get('artifacts') \
        if isinstance(report, Mapping) else None
    if artifacts is None:
        if not allow_legacy:
            raise ValueError(
                'specialist outcomes have no self-certifying '
                'direct_seed_eval artifacts; pass '
                '--allow-legacy-outcome-manifest only for the audited old '
                'P11 artifact')
        return {
            'self_certifying': False,
            'legacy_override_used': True,
            'sidecar': (
                file_fingerprint(sidecar) if sidecar.exists() else None),
            'unverified_fields': [
                'executed_checkpoint_sha256',
                'executed_controller_state',
                'executed_projection_config',
            ],
        }

    artifacts = _require_mapping(
        artifacts, 'specialist outcome artifacts')
    required = (
        'checkpoint_sha256', 'checkpoint_format',
        'controller_dir', 'controller_agent_sha256',
        'projection_config',
    )
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(
            f'specialist outcome artifacts are incomplete: {missing}')
    forced_artifact = file_fingerprint(forced_path)
    if (artifacts['checkpoint_sha256'] != forced_artifact['sha256']
            or artifacts['checkpoint_format'] != _MOE_FORMAT):
        raise ValueError(
            'specialist outcomes were not executed with the supplied '
            'forced-specialist checkpoint')
    if artifacts['projection_config'] != runner_audit['projection_config']:
        raise ValueError(
            'specialist outcome projection differs from baseline runner')

    controller_dir = Path(
        str(artifacts['controller_dir'])).expanduser().resolve(strict=True)
    agent_path = controller_dir / 'agent.pt'
    agent_artifact = file_fingerprint(agent_path)
    if agent_artifact['sha256'] != artifacts['controller_agent_sha256']:
        raise ValueError(
            'specialist outcome controller file hash differs from manifest')
    executed_controller = torch.load(
        agent_path, map_location='cpu', weights_only=False)
    executed_controller = _require_mapping(
        executed_controller, 'specialist outcome controller')
    runner_controller = _require_mapping(
        runner.get('controller'), 'baseline runner controller')
    target_input = runner_controller.get('_actor_trunk.0.weight')
    if not torch.is_tensor(target_input) or target_input.ndim != 2:
        raise ValueError('runner controller has no actor input weight')
    executed_controller = adapt_controller_observation_state_dict(
        dict(executed_controller), int(target_input.shape[1]))
    _tensor_mapping_equal(
        executed_controller, runner_controller,
        label='specialist/baseline controller')
    return {
        'self_certifying': True,
        'legacy_override_used': False,
        'sidecar': file_fingerprint(sidecar),
        'executed_checkpoint': forced_artifact,
        'executed_controller_agent': agent_artifact,
        'executed_controller_state_sha256': state_dict_fingerprint(
            dict(executed_controller)),
        'executed_projection_config': artifacts['projection_config'],
    }


@torch.no_grad()
def _frozen_features(
    actor: DirectSeedMoEActor,
    task: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError('feature batch size must be positive')
    actor.eval()
    parts = []
    for start in range(0, task.shape[0], batch_size):
        chunk = task[start:start + batch_size].to(
            device=actor.q_mid.device, dtype=actor.q_mid.dtype)
        parts.append(actor._features(chunk).detach().cpu())
    features = torch.cat(parts).numpy()
    if not np.isfinite(features).all():
        raise ValueError('frozen MoE hidden features must be finite')
    return features


def _top_fraction_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    """Select an exact stable top quota for diagnostics."""
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or score.size < 1 or not np.isfinite(score).all():
        raise ValueError('quota scores must be a finite non-empty vector')
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError('quota fraction must be in (0, 1]')
    count = min(len(score), max(1, int(round(len(score) * fraction))))
    order = np.argsort(-score, kind='stable')
    selected = np.zeros(len(score), dtype=np.bool_)
    selected[order[:count]] = True
    return selected


def _gated_metrics(
    deployed_branch: np.ndarray,
    branch_progress_m: np.ndarray,
    *,
    positive_margin_m: float,
) -> dict[str, Any]:
    deployed_branch = np.asarray(deployed_branch, dtype=np.int64)
    progress = np.asarray(branch_progress_m, dtype=np.float64)
    if (progress.ndim != 2
            or deployed_branch.shape != (progress.shape[0],)
            or np.any(deployed_branch < 0)
            or np.any(deployed_branch >= progress.shape[1])
            or not np.isfinite(progress).all()):
        raise ValueError('deployed branches or progress matrix are invalid')
    rows = np.arange(len(progress))
    baseline = progress[:, 0]
    deployed = progress[rows, deployed_branch]
    delta = deployed - baseline
    selected = deployed_branch != 0
    oracle_gain = progress.max(axis=1) - baseline
    denominator = float(oracle_gain.sum())
    branch_count = np.bincount(
        deployed_branch, minlength=progress.shape[1])
    return {
        'realized_specialist_count': int(np.count_nonzero(selected)),
        'realized_specialist_fraction': float(selected.mean()),
        'mean_delta_mm': float(delta.mean() * 1e3),
        'selected_mean_delta_mm': (
            float(delta[selected].mean() * 1e3)
            if bool(selected.any()) else 0.0),
        'advantage_precision': (
            float((delta[selected] > positive_margin_m).mean())
            if bool(selected.any()) else 0.0),
        'overall_harm_gt_1mm_pct': float(
            (delta < -0.001).mean() * 100.0),
        'overall_win_gt_1mm_pct': float(
            (delta > 0.001).mean() * 100.0),
        'selected_harm_gt_1mm_pct': (
            float((delta[selected] < -0.001).mean() * 100.0)
            if bool(selected.any()) else 0.0),
        'selected_win_gt_1mm_pct': (
            float((delta[selected] > 0.001).mean() * 100.0)
            if bool(selected.any()) else 0.0),
        'pool_oracle_capture_pct': (
            float(delta.sum() / denominator * 100.0)
            if abs(denominator) > 1e-12 else None),
        'deployed_branch_count': {
            str(index): int(count)
            for index, count in enumerate(branch_count.tolist())
        },
    }


def _quota_metrics(
    score: np.ndarray,
    specialist_choice: np.ndarray,
    branch_progress_m: np.ndarray,
    target_branch: np.ndarray,
    quotas: Sequence[float],
    *,
    positive_margin_m: float,
) -> dict[str, dict[str, float | int]]:
    branch_progress_m = np.asarray(branch_progress_m, dtype=np.float64)
    target_branch = np.asarray(target_branch, dtype=np.int64)
    specialist_choice = np.asarray(specialist_choice, dtype=np.int64)
    if (branch_progress_m.ndim != 2
            or branch_progress_m.shape[0] != len(score)
            or specialist_choice.shape != (len(score),)
            or target_branch.shape != (len(score),)):
        raise ValueError('quota branch arrays have inconsistent shapes')
    if (not np.isfinite(branch_progress_m).all()
            or np.any(specialist_choice < 1)
            or np.any(specialist_choice >= branch_progress_m.shape[1])):
        raise ValueError('quota branch outcomes or choices are invalid')
    result = {}
    for quota in quotas:
        selected = _top_fraction_mask(score, quota)
        deployed = np.where(selected, specialist_choice, 0)
        metrics = _gated_metrics(
            deployed, branch_progress_m,
            positive_margin_m=positive_margin_m)
        result[f'{quota:g}'] = {
            **metrics,
            'target_branch_accuracy': float(
                (specialist_choice[selected]
                 == target_branch[selected]).mean()),
        }
    return result


def _fit_logistic(
    features: np.ndarray,
    target_branch: np.ndarray,
    *,
    n_experts: int,
    logistic_c: float,
    seed: int,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(target_branch)
    if (classes.size < 2 or 0 not in classes
            or np.any(classes < 0) or np.any(classes >= n_experts)):
        raise ValueError(
            'outcome labels must contain baseline and at least one valid '
            'specialist class')
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=logistic_c,
            class_weight='balanced',
            solver='lbfgs',
            max_iter=2000,
            tol=1e-4,
            random_state=seed))
    model.fit(features, target_branch.astype(np.int64))
    logistic = model.named_steps['logisticregression']
    if (logistic.n_iter_.size != 1
            or int(logistic.n_iter_[0]) >= logistic.max_iter):
        raise RuntimeError('logistic outcome gate did not converge')
    return model


def _raw_logistic_parameters(
    model,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold StandardScaler and logistic parameters into raw feature space."""
    steps = getattr(model, 'named_steps', None)
    if not isinstance(steps, Mapping) \
            or 'standardscaler' not in steps \
            or 'logisticregression' not in steps:
        raise ValueError(
            'gate model must be StandardScaler -> LogisticRegression')
    scaler = steps['standardscaler']
    logistic = steps['logisticregression']
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    coef = np.asarray(logistic.coef_, dtype=np.float64)
    intercept = np.asarray(logistic.intercept_, dtype=np.float64)
    if (scale.ndim != 1 or mean.shape != scale.shape
            or coef.ndim != 2 or coef.shape[1] != len(scale)
            or not np.isfinite(scale).all()
            or not np.isfinite(mean).all()
            or not np.isfinite(coef).all()
            or not np.isfinite(intercept).all()
            or np.any(scale <= 0.0)):
        raise ValueError('fitted StandardScaler/logistic parameters are invalid')
    raw_coef = coef / scale[None, :]
    raw_intercept = intercept - (
        coef * mean[None, :] / scale[None, :]).sum(axis=1)
    classes = np.asarray(logistic.classes_, dtype=np.int64)
    return classes, raw_coef, raw_intercept


def _model_gate_parameters(
    model,
    *,
    n_experts: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand sklearn binary/multiclass parameters to all MoE branches."""
    classes, coef, intercept = _raw_logistic_parameters(model)
    if (classes.ndim != 1 or classes.size < 2 or 0 not in classes
            or np.any(classes < 0) or np.any(classes >= n_experts)
            or coef.ndim != 2 or coef.shape[1] != feature_dim):
        raise ValueError('logistic classes or coefficient shape is invalid')
    weight = np.zeros((n_experts, feature_dim), dtype=np.float64)
    bias = np.full(n_experts, -1e6, dtype=np.float64)
    if len(classes) == 2 and coef.shape == (1, feature_dim) \
            and intercept.shape == (1,):
        # sklearn binary decision_function is class[1] versus class[0].
        weight[classes[0]] = 0.0
        bias[classes[0]] = 0.0
        weight[classes[1]] = coef[0]
        bias[classes[1]] = intercept[0]
    elif coef.shape == (len(classes), feature_dim) \
            and intercept.shape == (len(classes),):
        for row, branch in enumerate(classes.tolist()):
            weight[branch] = coef[row]
            bias[branch] = intercept[row]
    else:
        raise ValueError('unsupported sklearn logistic parameter layout')
    # A common affine logit is irrelevant.  Making baseline exactly zero is
    # convenient for the conservative specialist-margin calibration.
    baseline_weight = weight[0].copy()
    baseline_bias = float(bias[0])
    weight -= baseline_weight[None, :]
    bias -= baseline_bias
    weight[0] = 0.0
    bias[0] = 0.0
    return weight, bias


def _model_logits(
    model,
    features: np.ndarray,
    *,
    n_experts: int,
) -> np.ndarray:
    weight, bias = _model_gate_parameters(
        model, n_experts=n_experts,
        feature_dim=features.shape[1])
    logits = features @ weight.T + bias[None, :]
    if not np.isfinite(logits).all():
        raise ValueError('logistic branch logits must be finite')
    return logits


def _fit_gate_models(
    features: np.ndarray,
    target_branch: np.ndarray,
    branch_progress_m: np.ndarray,
    *,
    objective: str,
    n_experts: int,
    positive_margin_m: float,
    logistic_c: float,
    seed: int,
) -> list[Any]:
    if objective == 'multinomial-best':
        return [_fit_logistic(
            features, target_branch,
            n_experts=n_experts,
            logistic_c=logistic_c, seed=seed)]
    if objective != 'ovr-positive':
        raise ValueError(f'unsupported gate objective {objective!r}')
    baseline = branch_progress_m[:, 0]
    models = []
    for expert_index in range(1, n_experts):
        positive = (
            branch_progress_m[:, expert_index]
            > baseline + positive_margin_m
        ).astype(np.int64)
        models.append(_fit_logistic(
            features, positive,
            n_experts=2,
            logistic_c=logistic_c,
            seed=seed + expert_index))
    return models


def _gate_model_parameters(
    models: Sequence[Any],
    *,
    objective: str,
    n_experts: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if objective == 'multinomial-best':
        if len(models) != 1:
            raise ValueError('multinomial gate requires exactly one model')
        return _model_gate_parameters(
            models[0], n_experts=n_experts,
            feature_dim=feature_dim)
    if objective != 'ovr-positive' or len(models) != n_experts - 1:
        raise ValueError('one-vs-baseline gate model count is invalid')
    weight = np.zeros((n_experts, feature_dim), dtype=np.float64)
    bias = np.zeros(n_experts, dtype=np.float64)
    for expert_index, model in enumerate(models, start=1):
        binary_weight, binary_bias = _model_gate_parameters(
            model, n_experts=2, feature_dim=feature_dim)
        weight[expert_index] = binary_weight[1]
        bias[expert_index] = binary_bias[1]
    return weight, bias


def _gate_model_logits(
    models: Sequence[Any],
    features: np.ndarray,
    *,
    objective: str,
    n_experts: int,
) -> np.ndarray:
    weight, bias = _gate_model_parameters(
        models, objective=objective,
        n_experts=n_experts, feature_dim=features.shape[1])
    logits = features @ weight.T + bias[None, :]
    if not np.isfinite(logits).all():
        raise ValueError('gate model logits must be finite')
    return logits


def _gate_model_audit(
    models: Sequence[Any],
    *,
    objective: str,
) -> dict[str, Any]:
    logistics = [
        model.named_steps['logisticregression']
        for model in models
    ]
    return {
        'objective': objective,
        'model_count': len(models),
        'iterations': [
            int(model.n_iter_[0]) for model in logistics],
        'fitted_classes': [
            [int(value) for value in model.classes_.tolist()]
            for model in logistics
        ],
        'feature_scaler': (
            'StandardScaler fit independently inside each training fold'),
    }


def _parse_enabled_specialists(
    text: str,
    n_experts: int,
) -> tuple[int, ...]:
    if text.strip().lower() == 'all':
        return tuple(range(1, n_experts))
    try:
        values = tuple(
            int(value.strip()) for value in text.split(',')
            if value.strip())
    except ValueError as error:
        raise ValueError(
            '--enabled-specialists must be all or comma-separated integers'
        ) from error
    if not values or len(set(values)) != len(values):
        raise ValueError(
            '--enabled-specialists must be non-empty and unique')
    if any(value < 1 or value >= n_experts for value in values):
        raise ValueError(
            '--enabled-specialists entries must be in [1, K-1]')
    return tuple(sorted(values))


def _mask_disabled_logits(
    logits: np.ndarray,
    enabled_specialists: Sequence[int],
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    enabled = set(int(value) for value in enabled_specialists)
    if logits.ndim != 2 or not enabled \
            or any(value < 1 or value >= logits.shape[1]
                   for value in enabled):
        raise ValueError('enabled specialists do not match branch logits')
    result = logits.copy()
    for expert_index in range(1, logits.shape[1]):
        if expert_index not in enabled:
            result[:, expert_index] = _DISABLED_GATE_LOGIT
    return result


def _disable_gate_parameters(
    weight: np.ndarray,
    bias: np.ndarray,
    enabled_specialists: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float64).copy()
    bias = np.asarray(bias, dtype=np.float64).copy()
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError('gate parameter shapes are invalid')
    enabled = set(int(value) for value in enabled_specialists)
    if not enabled or any(
            value < 1 or value >= len(bias) for value in enabled):
        raise ValueError('enabled specialists do not match gate parameters')
    for expert_index in range(1, len(bias)):
        if expert_index not in enabled:
            weight[expert_index] = 0.0
            bias[expert_index] = _DISABLED_GATE_LOGIT
    return weight, bias


def _specialist_margin_and_choice(
    logits: np.ndarray,
    enabled_specialists: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2 \
            or not np.isfinite(logits).all():
        raise ValueError('branch logits must have shape (N, K>=2)')
    enabled = (
        tuple(range(1, logits.shape[1]))
        if enabled_specialists is None
        else tuple(int(value) for value in enabled_specialists)
    )
    if not enabled or len(set(enabled)) != len(enabled) \
            or any(value < 1 or value >= logits.shape[1]
                   for value in enabled):
        raise ValueError('enabled specialists are invalid')
    enabled_array = np.asarray(enabled, dtype=np.int64)
    local_choice = logits[:, enabled_array].argmax(axis=1)
    choice = enabled_array[local_choice]
    rows = np.arange(len(logits))
    margin = logits[rows, choice] - logits[:, 0]
    return margin, choice


def _grouped_oof(
    features: np.ndarray,
    target_branch: np.ndarray,
    branch_progress_m: np.ndarray,
    fingerprints: Sequence[str],
    group_keys: Sequence[str],
    *,
    objective: str,
    n_experts: int,
    enabled_specialists: Sequence[int],
    positive_margin_m: float,
    logistic_c: float,
    seed: int,
    quotas: Sequence[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn import __version__ as sklearn_version
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(group_keys)
    if len(groups) != len(features) \
            or len(fingerprints) != len(features):
        raise ValueError('geometry fingerprints and features differ in length')
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 5:
        raise ValueError('five-fold OOF requires at least five task groups')
    splitter = GroupKFold(n_splits=5)
    oof_logits = np.full(
        (len(features), n_experts), np.nan, dtype=np.float64)
    fold_reports = []
    for fold, (train, heldout) in enumerate(
            splitter.split(features, target_branch, groups), start=1):
        if set(groups[train]).intersection(groups[heldout]):
            raise RuntimeError('geometry group leaked across OOF folds')
        models = _fit_gate_models(
            features[train], target_branch[train],
            branch_progress_m[train],
            objective=objective,
            n_experts=n_experts,
            positive_margin_m=positive_margin_m,
            logistic_c=logistic_c, seed=seed + 100 * fold)
        logits = _gate_model_logits(
            models, features[heldout],
            objective=objective, n_experts=n_experts)
        oof_logits[heldout] = logits
        fold_reports.append({
            'fold': fold,
            'train_count': int(len(train)),
            'heldout_count': int(len(heldout)),
            'train_specialist_count': int(np.count_nonzero(
                target_branch[train])),
            'heldout_specialist_count': int(np.count_nonzero(
                target_branch[heldout])),
            'train_branch_count': {
                str(index): int(np.count_nonzero(
                    target_branch[train] == index))
                for index in range(n_experts)
            },
            'heldout_branch_count': {
                str(index): int(np.count_nonzero(
                    target_branch[heldout] == index))
                for index in range(n_experts)
            },
            'train_unique_geometry_groups': int(
                len(np.unique(groups[train]))),
            'heldout_unique_geometry_groups': int(
                len(np.unique(groups[heldout]))),
            'heldout_group_list_sha256': _fingerprint_list_sha256(
                sorted(set(groups[heldout].tolist()))),
            'gate_models': _gate_model_audit(
                models, objective=objective),
        })
    if not np.isfinite(oof_logits).all():
        raise RuntimeError('OOF logits do not cover every task')
    oof_logits = _mask_disabled_logits(
        oof_logits, enabled_specialists)
    oof_margin, oof_specialist_choice = (
        _specialist_margin_and_choice(
            oof_logits, enabled_specialists))
    shifted = oof_logits - oof_logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    any_specialist = target_branch != 0
    specialist_probability = 1.0 - probability[:, 0]
    predicted_branch = oof_logits.argmax(axis=1)
    one_hot = np.eye(n_experts, dtype=np.float64)[target_branch]
    present = np.unique(target_branch)
    multiclass_auc = None
    multiclass_ap = None
    if n_experts > 2 and len(present) == n_experts:
        multiclass_auc = float(roc_auc_score(
            target_branch, probability,
            multi_class='ovr', average='macro'))
        multiclass_ap = float(average_precision_score(
            one_hot, probability, average='macro'))
    report = {
        'protocol': 'exact-float32-task-bytes-grouped-5-fold',
        'gate_objective': objective,
        'enabled_specialists': [
            int(value) for value in enabled_specialists],
        'disabled_specialists': [
            value for value in range(1, n_experts)
            if value not in set(enabled_specialists)],
        'n_splits': 5,
        'shuffle': False,
        'seed': seed,
        'sklearn_version': sklearn_version,
        'unique_geometry_group_count': unique_group_count,
        'geometry_fingerprint_list_sha256': (
            _fingerprint_list_sha256(fingerprints)),
        'any_specialist_roc_auc': float(roc_auc_score(
            any_specialist, specialist_probability)),
        'any_specialist_average_precision': float(
            average_precision_score(
                any_specialist, specialist_probability)),
        'multiclass_macro_roc_auc': multiclass_auc,
        'multiclass_macro_average_precision': multiclass_ap,
        'uncalibrated_branch_accuracy': float(
            (predicted_branch == target_branch).mean()),
        'folds': fold_reports,
        'quota_grid': _quota_metrics(
            oof_margin, oof_specialist_choice,
            branch_progress_m, target_branch, quotas,
            positive_margin_m=positive_margin_m),
    }
    return oof_logits, report


def _parse_quotas(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(',')
                       if part.strip())
    except ValueError as error:
        raise ValueError('--quota-grid must contain comma-separated floats') \
            from error
    if not values or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in values):
        raise ValueError('--quota-grid values must be in (0, 1]')
    if len(set(values)) != len(values):
        raise ValueError('--quota-grid values must be unique')
    return values


def _exclusive_save(value: Any, path: Path, *, json_value: bool) -> None:
    """Atomically install a new file without ever replacing an old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if json_value:
            temporary.write_text(
                json.dumps(value, indent=2) + '\n', encoding='utf-8')
        else:
            torch.save(value, temporary)
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f'refusing to overwrite {path}') from error
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Fit a frozen multi-branch real-outcome direct-seed gate.')
    parser.add_argument('--moe-checkpoint', required=True)
    parser.add_argument(
        '--forced-specialist-checkpoint',
        required=True, action='append',
        help='repeat in expert-index order for experts 1..K-1')
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--baseline-archive', required=True)
    parser.add_argument('--baseline-runner-checkpoint', required=True)
    parser.add_argument(
        '--specialist-outcomes',
        required=True, action='append',
        help='repeat in expert-index order for experts 1..K-1')
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--selection-fraction', required=True, type=float,
        help='explicit train-quantile specialist deployment fraction')
    parser.add_argument('--positive-margin-m', type=float, default=0.01)
    parser.add_argument('--logistic-c', type=float, default=0.1)
    parser.add_argument(
        '--gate-objective',
        choices=('ovr-positive', 'multinomial-best'),
        default='ovr-positive',
        help='independent specialist-vs-baseline advantage classifiers '
             '(recommended) or one multinomial best-branch classifier')
    parser.add_argument(
        '--enabled-specialists', default='all',
        help='all (default) or comma-separated expert indices; disabled '
             'branches are fail-closed in both OOF and the saved hard gate')
    parser.add_argument(
        '--quota-grid',
        default=','.join(f'{value:g}' for value in _DEFAULT_QUOTAS))
    parser.add_argument('--feature-batch-size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20260728)
    parser.add_argument(
        '--allow-legacy-outcome-manifest', action='store_true',
        help='explicitly accept old P11 outcomes that predate execution '
             'checkpoint/controller hashes; recorded as not self-certifying')
    parser.add_argument(
        '--allow-legacy-runner-provenance', action='store_true',
        help='explicitly reconstruct an old P8 standalone archive from the '
             'first exact task-byte match in its macro replay')
    return parser


def _resolved_file(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=True)


def main() -> None:
    args = _parser().parse_args()
    if (not math.isfinite(args.selection_fraction)
            or not 0.0 < args.selection_fraction <= 1.0):
        raise ValueError('--selection-fraction must be in (0, 1]')
    if (not math.isfinite(args.positive_margin_m)
            or args.positive_margin_m < 0.0):
        raise ValueError('--positive-margin-m must be non-negative')
    if not math.isfinite(args.logistic_c) or args.logistic_c <= 0.0:
        raise ValueError('--logistic-c must be positive')
    if args.feature_batch_size < 1:
        raise ValueError('--feature-batch-size must be positive')
    quotas = _parse_quotas(args.quota_grid)

    paths = {
        'moe_checkpoint': _resolved_file(args.moe_checkpoint),
        'base_checkpoint': _resolved_file(args.base_checkpoint),
        'baseline_archive': _resolved_file(args.baseline_archive),
        'baseline_runner_checkpoint': _resolved_file(
            args.baseline_runner_checkpoint),
    }
    forced_paths = tuple(
        _resolved_file(value)
        for value in args.forced_specialist_checkpoint)
    outcome_paths = tuple(
        _resolved_file(value) for value in args.specialist_outcomes)
    output = Path(args.output).expanduser().resolve()
    if output.suffix != '.pt':
        raise ValueError('--output must use a .pt suffix')
    report_path = output.with_suffix('.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(
            f'refusing to overwrite {output} or {report_path}')

    source_files: dict[str, Any] = {
        name: file_fingerprint(path) for name, path in paths.items()
    }
    source_files['forced_specialist_checkpoints'] = [
        file_fingerprint(path) for path in forced_paths]
    source_files['specialist_outcomes'] = [
        file_fingerprint(path) for path in outcome_paths]
    moe, _, moe_payload = load_direct_seed_moe_checkpoint(
        paths['moe_checkpoint'], device='cpu')
    base, _, _, _, base_payload = load_direct_seed_rl_checkpoint(
        paths['base_checkpoint'], device='cpu')
    branch_audit = _validate_moe_baseline_branch(moe, base)
    n_experts = moe.config.n_experts
    enabled_specialists = _parse_enabled_specialists(
        args.enabled_specialists, n_experts)
    disabled_specialists = tuple(
        value for value in range(1, n_experts)
        if value not in set(enabled_specialists))
    if (len(forced_paths) != n_experts - 1
            or len(outcome_paths) != n_experts - 1):
        raise ValueError(
            f'K={n_experts} requires exactly {n_experts - 1} forced '
            'checkpoints and outcome files, ordered as experts 1..K-1')
    forced_audit = []
    for expert_index, forced_path in enumerate(forced_paths, start=1):
        forced, _, forced_payload = load_direct_seed_moe_checkpoint(
            forced_path, device='cpu')
        if (forced.config != moe.config
                or forced.config.n_experts != n_experts
                or not forced.config.exact_baseline_head):
            raise ValueError(
                f'forced expert {expert_index} MoE config differs')
        forced_audit.append(_validate_forced_specialist(
            moe_payload, forced_payload,
            moe_sha256=source_files['moe_checkpoint']['sha256'],
            expert_index=expert_index))

    archive_state = torch.load(
        paths['baseline_archive'],
        map_location='cpu', weights_only=False)
    archive_state = _require_mapping(
        archive_state, 'baseline paired archive')
    task_ids = archive_state.get('task_ids')
    if not torch.is_tensor(task_ids):
        raise ValueError('baseline paired archive has no task_ids')
    archive = DirectSeedPairedArchive(task_ids)
    archive.load_state_dict(archive_state)
    if len(archive) != int(archive.task_ids.numel()):
        raise ValueError('baseline paired archive is not full coverage')

    runner = torch.load(
        paths['baseline_runner_checkpoint'],
        map_location='cpu', weights_only=False)
    runner = _require_mapping(runner, 'baseline full runner')
    runner_audit = _validate_runner(
        runner, archive, archive_state, base_payload,
        allow_legacy=args.allow_legacy_runner_provenance)
    outcomes = [
        _load_outcomes(path, archive) for path in outcome_paths]
    outcome_manifest = [
        _validate_outcome_manifest(
            outcome_path, branch_outcome,
            forced_path=forced_path,
            runner=runner, runner_audit=runner_audit,
            allow_legacy=args.allow_legacy_outcome_manifest)
        for outcome_path, forced_path, branch_outcome in zip(
            outcome_paths, forced_paths, outcomes)
    ]

    baseline_progress = archive.progress_m.numpy().astype(
        np.float64, copy=False)
    branch_progress = np.column_stack([
        baseline_progress,
        *[
            outcome['progress_m'].astype(np.float64, copy=False)
            for outcome in outcomes
        ],
    ])
    rows = np.arange(len(baseline_progress))
    eligible_branches = np.asarray(
        (0, *enabled_specialists), dtype=np.int64)
    eligible_local = branch_progress[:, eligible_branches].argmax(axis=1)
    best_branch = eligible_branches[eligible_local]
    best_progress = branch_progress[rows, best_branch]
    target_branch = np.where(
        best_progress > baseline_progress + args.positive_margin_m,
        best_branch, 0).astype(np.int64)
    fingerprints = _geometry_fingerprints(archive.task)
    group_keys = _geometry_group_keys(archive.task)
    features = _frozen_features(
        moe, archive.task, args.feature_batch_size)

    oof_logits, oof_report = _grouped_oof(
        features, target_branch, branch_progress,
        fingerprints, group_keys,
        objective=args.gate_objective,
        n_experts=n_experts,
        enabled_specialists=enabled_specialists,
        positive_margin_m=args.positive_margin_m,
        logistic_c=args.logistic_c,
        seed=args.seed,
        quotas=quotas)
    oof_margin, oof_specialist_choice = (
        _specialist_margin_and_choice(
            oof_logits, enabled_specialists))
    if args.selection_fraction == 1.0:
        threshold = float(np.nextafter(
            oof_margin.min(), -np.inf))
    else:
        threshold = float(np.quantile(
            oof_margin, 1.0 - args.selection_fraction,
            method='linear'))
    oof_deployed = np.where(
        oof_margin > threshold, oof_specialist_choice, 0)
    oof_report['selection_calibration'] = {
        'source': (
            'explicit CLI fraction on grouped training OOF '
            'specialist-vs-baseline logit margins'),
        'validation_artifacts_read': False,
        'target_fraction': float(args.selection_fraction),
        'train_oof_quantile': float(1.0 - args.selection_fraction),
        'logit_margin_threshold': threshold,
        'strict_comparison': (
            'best specialist iff specialist logit - baseline logit '
            '> OOF threshold'),
        **_gated_metrics(
            oof_deployed, branch_progress,
            positive_margin_m=args.positive_margin_m),
    }
    models = _fit_gate_models(
        features, target_branch, branch_progress,
        objective=args.gate_objective,
        n_experts=n_experts,
        positive_margin_m=args.positive_margin_m,
        logistic_c=args.logistic_c, seed=args.seed)
    full_logits = _gate_model_logits(
        models, features,
        objective=args.gate_objective,
        n_experts=n_experts)
    full_logits = _mask_disabled_logits(
        full_logits, enabled_specialists)
    gate_weight, gate_bias = _gate_model_parameters(
        models, objective=args.gate_objective,
        n_experts=n_experts,
        feature_dim=features.shape[1])
    enabled_array = np.asarray(
        enabled_specialists, dtype=np.int64)
    gate_bias[enabled_array] -= threshold
    gate_weight, gate_bias = _disable_gate_parameters(
        gate_weight, gate_bias, enabled_specialists)

    non_gate_before = {
        name: value.detach().cpu().clone()
        for name, value in moe.state_dict().items()
        if not name.startswith('gate.')
    }
    with torch.no_grad():
        moe.gate.weight.copy_(torch.from_numpy(
            gate_weight).to(dtype=moe.gate.weight.dtype))
        moe.gate.bias.copy_(torch.from_numpy(
            gate_bias).to(dtype=moe.gate.bias.dtype))
    for name, expected in non_gate_before.items():
        if not torch.equal(moe.state_dict()[name].cpu(), expected):
            raise RuntimeError(
                f'frozen branch tensor {name!r} changed while fitting gate')
    deployed_branch = moe.expert_index(archive.task).numpy()
    if disabled_specialists and np.isin(
            deployed_branch, disabled_specialists).any():
        raise RuntimeError('saved gate selected a disabled specialist')
    expected_logits = full_logits.copy()
    expected_logits[:, enabled_array] -= threshold
    expected_branch = expected_logits.argmax(axis=1)
    if not np.array_equal(expected_branch, deployed_branch):
        raise RuntimeError(
            'serialized hard gate differs from calibrated logistic selection')
    # The baseline proof must remain true after replacing only the gate.
    _validate_moe_baseline_branch(moe, base)

    from sklearn import __version__ as sklearn_version
    from sklearn.metrics import average_precision_score, roc_auc_score

    shifted_logits = full_logits - full_logits.max(axis=1, keepdims=True)
    train_probability = np.exp(shifted_logits)
    train_probability /= train_probability.sum(axis=1, keepdims=True)
    any_specialist = target_branch != 0
    present_classes = np.unique(target_branch)
    multiclass_auc = None
    multiclass_ap = None
    if n_experts > 2 and len(present_classes) == n_experts:
        one_hot = np.eye(n_experts)[target_branch]
        multiclass_auc = float(roc_auc_score(
            target_branch, train_probability,
            multi_class='ovr', average='macro'))
        multiclass_ap = float(average_precision_score(
            one_hot, train_probability, average='macro'))
    train_gate_metrics = _gated_metrics(
        deployed_branch, branch_progress,
        positive_margin_m=args.positive_margin_m)
    gated_progress = branch_progress[rows, deployed_branch]
    if threshold >= 0.0:
        threshold_probability = 1.0 / (1.0 + math.exp(-threshold))
    else:
        exp_threshold = math.exp(threshold)
        threshold_probability = exp_threshold / (1.0 + exp_threshold)
    training = {
        'task_count': int(len(baseline_progress)),
        'n_experts': n_experts,
        'enabled_specialists': list(enabled_specialists),
        'disabled_specialists': list(disabled_specialists),
        'disabled_branch_gate_contract': {
            'weight': 'exact-zero',
            'bias': _DISABLED_GATE_LOGIT,
            'fail_closed': True,
        },
        'positive_margin_m': float(args.positive_margin_m),
        'target_branch_count': {
            str(index): int(np.count_nonzero(
                target_branch == index))
            for index in range(n_experts)
        },
        'target_specialist_count': int(np.count_nonzero(any_specialist)),
        'target_specialist_fraction': float(any_specialist.mean()),
        'per_specialist_positive_advantage_count': {
            str(index): int(np.count_nonzero(
                branch_progress[:, index]
                > baseline_progress + args.positive_margin_m))
            for index in range(1, n_experts)
        },
        'logistic_regression': {
            'implementation': (
                'sklearn StandardScaler -> LogisticRegression; scaler '
                'folded exactly into raw hidden-feature gate weights'),
            'sklearn_version': sklearn_version,
            'C': float(args.logistic_c),
            'class_weight': 'balanced',
            'solver': 'lbfgs',
            'max_iter': 2000,
            'tol': 1e-4,
            **_gate_model_audit(
                models, objective=args.gate_objective),
            'train_any_specialist_roc_auc': float(roc_auc_score(
                any_specialist, 1.0 - train_probability[:, 0])),
            'train_any_specialist_average_precision': float(
                average_precision_score(
                    any_specialist, 1.0 - train_probability[:, 0])),
            'train_multiclass_macro_roc_auc': multiclass_auc,
            'train_multiclass_macro_average_precision': multiclass_ap,
            'train_uncalibrated_branch_accuracy': float(
                (full_logits.argmax(axis=1) == target_branch).mean()),
        },
        'selection_calibration': {
            'source': (
                'grouped training OOF only; full-fit in-sample scores did '
                'not choose the threshold'),
            'validation_artifacts_read': False,
            'target_fraction': float(args.selection_fraction),
            'train_oof_quantile': float(
                1.0 - args.selection_fraction),
            'logit_margin_threshold': threshold,
            'binary_equivalent_probability_threshold': float(
                threshold_probability),
            'strict_comparison': (
                'best specialist iff specialist logit - baseline logit '
                '> OOF threshold'),
            'full_fit_train_realization': train_gate_metrics,
        },
        'development_status': {
            'specialist_subset_origin': (
                'predefined-all-specialists'
                if not disabled_specialists else
                'post-validation-viewed-development-candidate'),
            'threshold_and_linear_weights_source': (
                'training grouped OOF/full training only'),
            'validation_data_read_by_this_script': False,
            'independent_confirmation_claim': False,
            'claim_scope': (
                'development result; disabled-specialist candidate must not '
                'be described as independent confirmation'
                if disabled_specialists else
                'training-only model selection result'),
        },
        'paired_train_metrics': {
            'baseline_progress_mean_m': float(baseline_progress.mean()),
            'branch_progress_mean_m': {
                str(index): float(branch_progress[:, index].mean())
                for index in range(n_experts)
            },
            'paired_oracle_progress_mean_m': float(
                branch_progress.max(axis=1).mean()),
            'gated_progress_mean_m': float(gated_progress.mean()),
            **train_gate_metrics,
        },
    }
    deployment_protocol = {
        'one_deterministic_seed': True,
        'hard_selected_expert_heads_per_task': 1,
        'candidate_enumeration': 0,
        'return_model_queries': 0,
        'controller_probes': 0,
        'max_ik_refinements': 1,
        'controller_rollouts': 1,
    }
    audit = {
        'method': 'outcome-matched-frozen-multi-branch-gate-v2',
        'seed': int(args.seed),
        'source_files': source_files,
        'branch_audit': branch_audit,
        'forced_specialist_audit': forced_audit,
        'runner_audit': runner_audit,
        'outcome_manifest': outcome_manifest,
        'oof': oof_report,
        'training': training,
        'deployment_protocol': deployment_protocol,
    }
    metadata = {
        **audit,
        'source_moe_update_step': int(moe_payload['update_step']),
        'output_contract': (
            'self-contained direct-seed-hard-moe-v1; gate-only fit'),
    }
    checkpoint = direct_seed_moe_checkpoint(
        moe,
        update_step=int(moe_payload['update_step']),
        metadata=metadata)
    _exclusive_save(checkpoint, output, json_value=False)

    report = {
        **audit,
        'output': file_fingerprint(output),
        'output_checkpoint_format': checkpoint['format'],
        'output_update_step': int(checkpoint['update_step']),
    }
    _exclusive_save(report, report_path, json_value=True)
    print(json.dumps({
        'output': str(output),
        'output_sha256': report['output']['sha256'],
        'enabled_specialists': list(enabled_specialists),
        'selection_fraction': training[
            'selection_calibration'
        ]['full_fit_train_realization'][
            'realized_specialist_fraction'],
        'train_gated_delta_mm': training[
            'paired_train_metrics']['mean_delta_mm'],
        'oof_quota_grid': oof_report['quota_grid'],
        'outcome_manifests_self_certifying': [
            value['self_certifying'] for value in outcome_manifest],
        'runner_archive_self_certifying': runner_audit[
            'archive_match']['self_certifying'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()


__all__ = [
    '_disable_gate_parameters',
    '_geometry_fingerprints',
    '_geometry_group_keys',
    '_gated_metrics',
    '_grouped_oof',
    '_parse_enabled_specialists',
    '_quota_metrics',
    '_top_fraction_mask',
    '_specialist_margin_and_choice',
    '_validate_archive_against_runner',
    '_validate_forced_specialist',
    '_validate_moe_baseline_branch',
]
