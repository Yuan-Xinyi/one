"""Strict paired evaluation of a shielded residual-seed checkpoint.

The discrete selector and continuous controller are frozen.  For every task,
the evaluator runs the original selected seed and its deterministic residual
counterpart in one ``2B`` controller rollout.  This keeps the comparison
paired while making every artifact and split dependency explicit.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedSelection,
)
from Yuan.unified_rl.checkpoint import (
    build_env_from_run,
    load_controller_agent,
    load_run_config,
    ppo_config_from_run,
    require_checkpoint_format_version,
    require_checkpoint_keys,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenHybridController,
    FrozenRLController,
    rollout_seed_selection,
)
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.provenance import (
    controller_fingerprint,
    file_fingerprint,
    state_dict_fingerprint,
)
from Yuan.unified_rl.residual_policy import (
    ResidualSeedHead,
    ResidualSeedHeadConfig,
)
from Yuan.unified_rl.residual_seed import (
    ResidualSeedConfig,
    apply_residual_seed,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    infer_seed_policy_config,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    validate_cached_dataset,
)


RESIDUAL_FORMAT = 'unified-residual-seed-v2'
RESIDUAL_FORMAT_VERSION = 2
LEGACY_RESIDUAL_FORMAT = 'unified-residual-seed-v1'
LEGACY_RESIDUAL_FORMAT_VERSION = 1
SOURCE_FORMAT = 'unified-bidirectional-v4'
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_SAMPLES = 10_000


def _load_mapping(path: str | Path, device: torch.device,
                  label: str) -> dict[str, Any]:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f'{label} must contain a dictionary')
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be a mapping')
    return value


def _require_hex_digest(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{label} must be a lowercase SHA-256 digest')
    return value


def _artifact_identity(artifact: Mapping[str, Any], label: str
                       ) -> tuple[int, str]:
    artifact = _require_mapping(artifact, label)
    if 'size' not in artifact or 'sha256' not in artifact:
        raise ValueError(f'{label} is missing size or sha256')
    size = artifact['size']
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f'{label}.size must be a non-negative integer')
    digest = _require_hex_digest(artifact['sha256'], f'{label}.sha256')
    return size, digest


def _assert_artifact(saved: Mapping[str, Any], current: Mapping[str, Any],
                     label: str) -> None:
    if _artifact_identity(saved, label) != _artifact_identity(current, label):
        raise ValueError(f'{label} content fingerprint does not match')


def _assert_controller_artifacts(
    saved: Mapping[str, Any], current: Mapping[str, Any], label: str,
) -> None:
    saved = _require_mapping(saved, label)
    current = _require_mapping(current, f'current {label}')
    for name in ('agent', 'config'):
        if name not in saved or name not in current:
            raise ValueError(f'{label} must contain {name!r}')
        _assert_artifact(saved[name], current[name], f'{label}.{name}')


def _assert_tensor_equal(left: Any, right: Any, label: str) -> None:
    left_tensor = torch.as_tensor(left, device='cpu')
    right_tensor = torch.as_tensor(right, device='cpu')
    if left_tensor.dtype != right_tensor.dtype:
        # Index tensors from otherwise equivalent checkpoints may use an
        # integer subtype.  Masks, however, must stay genuinely boolean.
        if left_tensor.dtype == torch.bool or right_tensor.dtype == torch.bool:
            raise ValueError(f'{label} dtype differs')
        left_tensor = left_tensor.to(torch.int64)
        right_tensor = right_tensor.to(torch.int64)
    if (left_tensor.shape != right_tensor.shape
            or not torch.equal(left_tensor, right_tensor)):
        raise ValueError(f'{label} differs from the source checkpoint')


def _prepare_output_path(
    value: str | Path, input_paths: Sequence[str | Path],
) -> Path:
    """Resolve a new NPZ output and reject destructive path aliases."""
    requested = Path(value).expanduser()
    if requested.suffix.lower() != '.npz':
        raise ValueError('--out must use the .npz suffix')
    requested.parent.mkdir(parents=True, exist_ok=True)
    output = requested.resolve()
    inputs = {Path(path).expanduser().resolve(strict=True) for path in input_paths}
    if output in inputs:
        raise ValueError('--out must differ from every input artifact')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing output: {output}')
    return output


def _atomic_savez_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete NPZ atomically without replacing an existing file."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is atomic and fails if another process
        # has created the requested output since _prepare_output_path.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _source_model_state(source: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    provenance = _require_mapping(source.get('provenance'), 'source provenance')
    source_format = provenance.get('format')
    key = 'seed_policy'
    state = _require_mapping(source.get(key), f'source {key}')
    if not state or not all(isinstance(name, str) and torch.is_tensor(tensor)
                            for name, tensor in state.items()):
        raise ValueError(f'source {key} must be a non-empty tensor state dictionary')
    return state


def _validate_source_checkpoint(source: dict[str, Any]) -> str:
    provenance = _require_mapping(source.get('provenance'), 'source provenance')
    source_format = provenance.get('format')
    if source_format != SOURCE_FORMAT:
        raise ValueError(
            f'source unified checkpoint must use {SOURCE_FORMAT!r}, '
            f'got {source_format!r}')
    require_checkpoint_format_version(source, 4, kind=str(source_format))
    require_checkpoint_keys(
        source,
        (
            'format_version', 'feature_dim', 'seed_architecture', 'split_mode',
            'seed_include_log_manip', 'train_task_indices',
            'validation_task_indices', 'train_valid_mask',
            'validation_valid_mask', 'controller', 'controller_config',
            'controller_state_sha256', 'controller_run_config_sha256',
            'provenance',
        ),
        kind=str(source_format),
    )
    if source.get('split_mode') != 'task-geometry-grouped-v1':
        raise ValueError(
            'residual evaluation requires a leak-free '
            'task-geometry-grouped-v1 source split')
    if 'candidate_cache' not in provenance:
        raise ValueError('source provenance is missing candidate_cache')
    _artifact_identity(provenance['candidate_cache'],
                       'source provenance candidate_cache')
    _source_model_state(source)
    return str(source_format)


def _validate_residual_checkpoint(
    residual: dict[str, Any], source: dict[str, Any],
    source_artifact: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], ResidualSeedHeadConfig,
           ResidualSeedConfig, str, str]:
    required = (
        'format_version', 'source_checkpoint', 'candidate_cache',
        'controller_artifacts', 'source_seed_state_sha256',
        'controller_state_sha256', 'controller_run_config_sha256',
        'residual_head_state_sha256', 'seed_return', 'gate_threshold',
        'residual_head',
        'residual_architecture', 'shield_config', 'bandit_config',
        'train_task_indices',
        'validation_task_indices', 'train_valid_mask',
        'validation_valid_mask', 'split_mode', 'seed_include_ray_error',
        'seed_include_log_manip', 'source_seed_architecture', 'provenance',
    )
    provenance = _require_mapping(
        residual.get('provenance'), 'residual provenance')
    residual_format = provenance.get('format')
    supported_formats = {
        LEGACY_RESIDUAL_FORMAT: LEGACY_RESIDUAL_FORMAT_VERSION,
        RESIDUAL_FORMAT: RESIDUAL_FORMAT_VERSION,
    }
    if residual_format not in supported_formats:
        raise ValueError(
            'residual provenance format must be one of '
            f'{sorted(supported_formats)}, got {residual_format!r}')
    require_checkpoint_keys(
        residual, required, kind=str(residual_format))
    require_checkpoint_format_version(
        residual, supported_formats[str(residual_format)],
        kind=str(residual_format))
    is_legacy_v1 = residual_format == LEGACY_RESIDUAL_FORMAT
    for key in ('source_checkpoint', 'candidate_cache',
                'controller_artifacts', 'settings'):
        if key not in provenance:
            raise ValueError(f'residual provenance is missing {key!r}')
    _assert_artifact(
        residual['source_checkpoint'], source_artifact,
        'residual source_checkpoint')
    _assert_artifact(
        provenance['source_checkpoint'], source_artifact,
        'residual provenance source_checkpoint')
    _assert_artifact(
        residual['candidate_cache'], provenance['candidate_cache'],
        'residual candidate_cache provenance mirror')
    _assert_controller_artifacts(
        residual['controller_artifacts'], provenance['controller_artifacts'],
        'residual controller_artifacts provenance mirror')

    source_provenance = _require_mapping(
        source['provenance'], 'source provenance')
    _assert_artifact(
        residual['candidate_cache'], source_provenance['candidate_cache'],
        'residual candidate_cache versus source')
    source_state = _source_model_state(source)
    source_state_hash = state_dict_fingerprint(dict(source_state))
    expected_source_state_hash = _require_hex_digest(
        residual['source_seed_state_sha256'], 'source_seed_state_sha256')
    if source_state_hash != expected_source_state_hash:
        raise ValueError('source selector state hash does not match residual checkpoint')
    for key in (
            'source_seed_state_sha256', 'controller_state_sha256',
            'controller_run_config_sha256'):
        top_digest = _require_hex_digest(residual[key], key)
        provenance_digest = _require_hex_digest(
            provenance.get(key), f'residual provenance {key}')
        if top_digest != provenance_digest:
            raise ValueError(f'{key} differs from residual provenance')

    for key in ('train_task_indices', 'validation_task_indices',
                'train_valid_mask', 'validation_valid_mask'):
        _assert_tensor_equal(residual[key], source[key], key)
    if residual['split_mode'] != source['split_mode']:
        raise ValueError('residual split_mode differs from source checkpoint')
    if bool(residual['seed_include_log_manip']) != bool(
            source['seed_include_log_manip']):
        raise ValueError('residual log-manip feature setting differs from source')
    source_include_ray = bool(source.get(
        'seed_include_ray_error', int(source['feature_dim']) in (34, 35)))
    if bool(residual['seed_include_ray_error']) != source_include_ray:
        raise ValueError('residual ray-error feature setting differs from source')
    if dict(_require_mapping(
            residual['source_seed_architecture'],
            'source_seed_architecture')) != dict(_require_mapping(
                source['seed_architecture'], 'source seed_architecture')):
        raise ValueError('source seed architecture differs from residual provenance')

    seed_return = residual['seed_return']
    source_seed_return = source.get('seed_return')
    if source_seed_return is None:
        source_seed_return = _require_mapping(
            source.get('args'), 'source args').get('seed_return')
    if source_seed_return is None:
        source_seed_return = source_provenance.get('settings', {}).get(
            'seed_return')
    if seed_return not in ('discounted', 'undiscounted'):
        raise ValueError(f'unknown residual seed objective: {seed_return!r}')
    if seed_return != source_seed_return:
        raise ValueError(
            'residual seed objective differs from source unified checkpoint')
    gate_threshold = residual['gate_threshold']
    if (isinstance(gate_threshold, bool)
            or not isinstance(gate_threshold, (int, float))
            or float(gate_threshold) != 0.5):
        raise ValueError('residual gate_threshold must be exactly 0.5')
    settings = _require_mapping(
        provenance['settings'], 'residual provenance settings')
    if settings.get('gate_threshold') != gate_threshold:
        raise ValueError('gate_threshold differs from residual provenance')
    if settings.get('seed_return') != seed_return:
        raise ValueError('seed_return differs from residual provenance')

    architecture_mapping = _require_mapping(
        residual['residual_architecture'], 'residual_architecture')
    expected_architecture_keys = {
        'input_dim', 'hidden_dim', 'gate_initial_logit',
    }
    if set(architecture_mapping) != expected_architecture_keys:
        raise ValueError(
            'residual_architecture has invalid keys: '
            f'{sorted(architecture_mapping)}')
    head_config = ResidualSeedHeadConfig(**dict(architecture_mapping))
    if not is_legacy_v1:
        if dict(_require_mapping(
                settings.get('residual_architecture'),
                'residual provenance residual_architecture')) != dict(
                    architecture_mapping):
            raise ValueError(
                'residual architecture differs from residual provenance')
    head_state = _require_mapping(residual['residual_head'], 'residual_head')
    if not head_state or not all(isinstance(name, str) and torch.is_tensor(tensor)
                                 for name, tensor in head_state.items()):
        raise ValueError('residual_head must be a non-empty tensor state dictionary')
    head_state_hash = state_dict_fingerprint(dict(head_state))
    if head_state_hash != _require_hex_digest(
            residual['residual_head_state_sha256'],
            'residual_head_state_sha256'):
        raise ValueError('residual head state hash does not match checkpoint metadata')

    shield_mapping = _require_mapping(
        residual['shield_config'], 'shield_config')
    expected_shield_keys = set(ResidualSeedConfig.__dataclass_fields__)
    if set(shield_mapping) != expected_shield_keys:
        raise ValueError(
            'shield_config keys differ from the supported safety schema')
    shield_config = ResidualSeedConfig(**dict(shield_mapping))
    if not is_legacy_v1:
        if dict(_require_mapping(
                settings.get('shield_config'),
                'residual provenance shield_config')) != dict(shield_mapping):
            raise ValueError('shield config differs from residual provenance')
    bandit_mapping = _require_mapping(
        residual['bandit_config'], 'bandit_config')
    if not is_legacy_v1:
        if dict(_require_mapping(
                settings.get('bandit_config'),
                'residual provenance bandit_config')) != dict(bandit_mapping):
            raise ValueError('bandit config differs from residual provenance')
    if settings.get('rho') != shield_config.rho:
        raise ValueError('shield rho differs from residual provenance')
    return (head_state, head_config, shield_config, str(seed_return),
            str(residual_format))


def _load_models(
    source: dict[str, Any], residual_head_state: Mapping[str, torch.Tensor],
    head_config: ResidualSeedHeadConfig, device: torch.device,
) -> tuple[CandidateSeedActorCritic, ResidualSeedHead]:
    seed_config = infer_seed_policy_config(source)
    selector = CandidateSeedActorCritic(**seed_config.to_dict()).to(device)
    selector.load_state_dict(_source_model_state(source), strict=True)
    selector.eval()
    head = ResidualSeedHead(**head_config.to_dict()).to(device)
    head.load_state_dict(residual_head_state, strict=True)
    head.eval()
    if head.input_dim != 2 * selector.hidden_dim:
        raise ValueError(
            'residual head input_dim must equal twice the source selector hidden_dim')
    return selector, head


def _pad_indices(start: int, end: int, total: int,
                 batch_size: int) -> tuple[torch.Tensor, int]:
    n_real = end - start
    if n_real < 1 or end > total:
        raise ValueError('invalid evaluation slice')
    index = torch.arange(start, end, dtype=torch.long)
    if n_real < batch_size:
        index = torch.cat([
            index,
            index[-1:].expand(batch_size - n_real),
        ])
    return index, n_real


def _geometry_group_means(
    values: np.ndarray, task_fingerprints: Sequence[str],
) -> np.ndarray:
    """Validate rows and collapse duplicate task geometries to group means."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError('grouped values must be a non-empty vector')
    if len(task_fingerprints) != values.size:
        raise ValueError('task fingerprint count differs from grouped values')
    if not np.isfinite(values).all():
        raise ValueError('grouped values must be finite')
    groups: dict[str, list[float]] = {}
    for fingerprint, value in zip(task_fingerprints, values.tolist()):
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError('task fingerprints must be SHA-256 strings')
        groups.setdefault(fingerprint, []).append(value)
    return np.asarray(
        [np.mean(group) for group in groups.values()], dtype=np.float64)


def geometry_grouped_mean(
    values: np.ndarray, task_fingerprints: Sequence[str],
) -> tuple[float, int]:
    """Return a unique-task-geometry-weighted mean and group count."""
    group_means = _geometry_group_means(values, task_fingerprints)
    return float(group_means.mean()), int(group_means.size)


def geometry_grouped_bootstrap_ci(
    values: np.ndarray,
    task_fingerprints: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float, int]:
    """Return mean and percentile CI after collapsing duplicate geometries."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('bootstrap seed must be an integer')
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('bootstrap samples must be positive')
    group_means = _geometry_group_means(values, task_fingerprints)
    estimate = float(group_means.mean())
    if group_means.size == 1:
        return estimate, estimate, estimate, 1

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype=np.float64)
    # Bound temporary allocation for large validation sets.
    block_size = max(1, min(samples, 1_000_000 // group_means.size))
    for start in range(0, samples, block_size):
        end = min(start + block_size, samples)
        index = rng.integers(
            0, group_means.size, size=(end - start, group_means.size))
        bootstrap_means[start:end] = group_means[index].mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return estimate, float(low), float(high), int(group_means.size)


def _record_rollout(
    destination: dict[str, np.ndarray], result, source_slice: slice,
    target_slice: slice, n_real: int,
) -> None:
    for name, dtype in (
        ('discounted_return', np.float32),
        ('undiscounted_return', np.float32),
        ('progress_m', np.float32),
        ('episode_len', np.int64),
        ('term_reason', np.int32),
        ('switch_count', np.int64),
    ):
        value = getattr(result, name)[source_slice][:n_real].cpu().numpy()
        destination[name][target_slice] = value.astype(dtype, copy=False)


def _controller_factory(source: Mapping[str, Any], agent, env):
    provenance = _require_mapping(source['provenance'], 'source provenance')
    settings = _require_mapping(
        provenance.get('settings', {}), 'source provenance settings')
    controller_kind = source.get(
        'controller_kind', settings.get('controller_kind', 'pure'))
    if controller_kind == 'pure':
        return FrozenRLController(agent), 'pure', None, None
    if controller_kind != 'hybrid':
        raise ValueError(f'unknown source controller kind: {controller_kind!r}')
    tau_enter = settings.get('tau_enter')
    tau_exit = settings.get('tau_exit')
    if tau_enter is None or tau_exit is None:
        raise ValueError('hybrid source controller is missing hysteresis thresholds')
    controller = FrozenHybridController(
        agent, ClassicalNullspaceController(env.kin),
        float(tau_enter), float(tau_exit))
    return controller, 'hybrid', float(tau_enter), float(tau_exit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Strict paired evaluation of a residual-seed checkpoint')
    parser.add_argument('--residual-checkpoint', required=True)
    parser.add_argument('--source-checkpoint', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--controller-ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--max-tasks', type=int, default=None)
    parser.add_argument('--external-holdout', action='store_true')
    parser.add_argument(
        '--allow-task-overlap', action='store_true',
        help='allow geometry overlap only for an explicit transfer diagnostic')
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError('--chunk-size must be positive')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('--max-tasks must be positive')
    if args.allow_task_overlap and not args.external_holdout:
        raise ValueError('--allow-task-overlap requires --external-holdout')
    out = _prepare_output_path(
        args.out,
        (args.residual_checkpoint, args.source_checkpoint,
         args.candidates, args.controller_ckpt),
    )
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))

    residual_artifact = file_fingerprint(args.residual_checkpoint)
    source_artifact = file_fingerprint(args.source_checkpoint)
    candidate_artifact = file_fingerprint(args.candidates)
    controller_artifact = controller_fingerprint(args.controller_ckpt)
    source = _load_mapping(args.source_checkpoint, device, 'source checkpoint')
    source_format = _validate_source_checkpoint(source)
    residual = _load_mapping(
        args.residual_checkpoint, device, 'residual checkpoint')
    head_state, head_config, shield_config, seed_return, residual_format = (
        _validate_residual_checkpoint(
            residual, source, source_artifact))
    selector, residual_head = _load_models(
        source, head_state, head_config, device)

    _assert_controller_artifacts(
        residual['controller_artifacts'], controller_artifact,
        'controller checkpoint')
    if (controller_artifact['config']['sha256']
            != residual['controller_run_config_sha256']):
        raise ValueError('controller config state hash differs from residual checkpoint')

    # One environment executes both members of every pair.  Its kinematics is
    # also reused for validation, feature construction, and the safety shield.
    env = build_env_from_run(
        args.controller_ckpt, 2 * args.chunk_size, device)
    if shield_config.cone_deg != env.cfg.cone_deg:
        raise ValueError(
            'shield cone must exactly match the fingerprinted controller '
            'environment cone')
    controller_agent = load_controller_agent(
        args.controller_ckpt, env, device).eval()
    controller_state_hash = state_dict_fingerprint(
        controller_agent.state_dict())
    if controller_state_hash != residual['controller_state_sha256']:
        raise ValueError('controller state hash differs from residual checkpoint')
    if source.get('controller_state_sha256') != controller_state_hash:
        raise ValueError('controller state hash differs from source checkpoint')
    if (source.get('controller_run_config_sha256')
            != controller_artifact['config']['sha256']):
        raise ValueError('controller config hash differs from source checkpoint')
    embedded_controller = _require_mapping(
        source.get('controller'), 'source embedded controller')
    if state_dict_fingerprint(dict(embedded_controller)) != controller_state_hash:
        raise ValueError('source embedded controller state is internally inconsistent')

    raw_dataset = CachedSeedCandidateDataset.from_npz(args.candidates)
    raw_task_fingerprints = raw_dataset.task_fingerprints
    physical_n_tasks_input = len(raw_dataset)
    dataset, valid_stats = validate_cached_dataset(
        raw_dataset, env.kin, env.collision, cone_deg=env.cfg.cone_deg)
    physical_n_tasks_retained = len(dataset)
    physical_n_tasks_rejected = (
        physical_n_tasks_input - physical_n_tasks_retained)
    if args.external_holdout and physical_n_tasks_rejected:
        raise ValueError(
            'external evaluation must retain every task; physical validation '
            f'rejected {physical_n_tasks_rejected}/{physical_n_tasks_input} '
            'tasks')
    validated_dataset = dataset
    print(
        '[residual-eval] physical candidate validity '
        f'{valid_stats["frac_valid"]:.1%}', flush=True)

    source_candidate = source['provenance']['candidate_cache']
    same_candidate_cache = (
        _artifact_identity(source_candidate, 'source candidate_cache')
        == _artifact_identity(candidate_artifact, 'current candidate_cache'))
    overlap_rows = 0
    overlap_unique = 0
    overlap_audited = False
    if args.external_holdout:
        if same_candidate_cache:
            raise ValueError(
                '--external-holdout requires a cache different from the source cache')
        source_cache_path = Path(str(source_candidate.get('path', '')))
        if not source_cache_path.is_file():
            raise ValueError(
                'recorded source candidate cache is unavailable for overlap audit: '
                f'{source_cache_path}')
        current_source_artifact = file_fingerprint(source_cache_path)
        _assert_artifact(
            source_candidate, current_source_artifact,
            'recorded source candidate cache')
        source_dataset = CachedSeedCandidateDataset.from_npz(source_cache_path)
        source_fingerprints = set(source_dataset.task_fingerprints)
        overlaps = [fingerprint for fingerprint in raw_task_fingerprints
                    if fingerprint in source_fingerprints]
        overlap_rows = len(overlaps)
        overlap_unique = len(set(overlaps))
        overlap_audited = True
        if overlap_rows and not args.allow_task_overlap:
            raise ValueError(
                f'external cache contains {overlap_rows} rows '
                f'({overlap_unique} unique geometries) from the source training '
                'cache; --allow-task-overlap is diagnostic only')
        if overlap_rows:
            print(
                '[residual-eval] WARNING: overlap diagnostic includes '
                f'{overlap_rows} source-cache rows', flush=True)
        print(
            f'[residual-eval] external holdout ({len(dataset)} tasks)',
            flush=True)
    else:
        if not same_candidate_cache:
            raise ValueError(
                'candidate cache differs from source; pass --external-holdout '
                'for a separately audited cache')
        dataset = validated_dataset.select_source_tasks(
            source['validation_task_indices'].cpu())
        assert_same_valid_mask(
            dataset, source['validation_valid_mask'], label='source validation')
        train_dataset = validated_dataset.select_source_tasks(
            source['train_task_indices'].cpu())
        assert_same_valid_mask(
            train_dataset, source['train_valid_mask'], label='source train')
        train_fingerprints = set(train_dataset.task_fingerprints)
        validation_fingerprints = dataset.task_fingerprints
        overlaps = [fingerprint for fingerprint in validation_fingerprints
                    if fingerprint in train_fingerprints]
        overlap_rows = len(overlaps)
        overlap_unique = len(set(overlaps))
        overlap_audited = True
        if overlap_rows:
            raise ValueError(
                'source checkpoint claims a grouped split but validation '
                f'contains {overlap_rows} train-overlapping rows')
        print(
            '[residual-eval] using source validation split '
            f'({len(dataset)} tasks)', flush=True)

    n = len(dataset) if args.max_tasks is None else min(
        len(dataset), args.max_tasks)
    if n < 1:
        raise ValueError('evaluation dataset is empty')
    task_fingerprints = dataset.task_fingerprints[:n]
    run_config = load_run_config(args.controller_ckpt)
    controller_gamma = float(ppo_config_from_run(run_config).gamma)
    if (not np.isfinite(controller_gamma)
            or not 0.0 <= controller_gamma <= 1.0):
        raise ValueError('fingerprinted controller gamma must be in [0, 1]')
    source_controller_config = _require_mapping(
        source['controller_config'], 'source controller_config')
    source_gamma = source_controller_config.get('gamma')
    if (isinstance(source_gamma, bool)
            or not isinstance(source_gamma, (int, float))
            or not np.isfinite(float(source_gamma))
            or not 0.0 <= float(source_gamma) <= 1.0
            or float(source_gamma) != controller_gamma):
        raise ValueError(
            'source controller gamma differs from fingerprinted config')
    if residual_format == RESIDUAL_FORMAT:
        residual_settings = _require_mapping(
            residual['provenance']['settings'],
            'residual provenance settings')
        if residual_settings.get('controller_gamma') != controller_gamma:
            raise ValueError(
                'residual controller gamma differs from fingerprinted config')
    controller, controller_kind, tau_enter, tau_exit = _controller_factory(
        source, controller_agent, env)

    rollout_outputs = {
        variant: {
            'discounted_return': np.zeros(n, dtype=np.float32),
            'undiscounted_return': np.zeros(n, dtype=np.float32),
            'progress_m': np.zeros(n, dtype=np.float32),
            'episode_len': np.zeros(n, dtype=np.int64),
            'term_reason': np.zeros(n, dtype=np.int32),
            'switch_count': np.zeros(n, dtype=np.int64),
        }
        for variant in ('base', 'residual')
    }
    candidate_index = np.zeros(n, dtype=np.int64)
    gate_output = np.zeros(n, dtype=np.bool_)
    latent_output = np.zeros((n, 4), dtype=np.float32)
    alpha_output = np.zeros(n, dtype=np.float32)
    q_delta_norm = np.zeros(n, dtype=np.float32)
    shield_outputs = {
        'valid': np.zeros(n, dtype=np.bool_),
        'hard_violation': np.zeros(n, dtype=np.bool_),
        'position_error': np.zeros(n, dtype=np.float32),
        'cone_cosine': np.zeros(n, dtype=np.float32),
        'collision_margin': np.zeros(n, dtype=np.float32),
        'joint_margin': np.zeros(n, dtype=np.float32),
        'branch_distance': np.zeros(n, dtype=np.float32),
        'basis_fallback': np.zeros((n, 3), dtype=np.bool_),
        'selected_index': np.zeros(n, dtype=np.int64),
    }

    include_log_manip = bool(residual['seed_include_log_manip'])
    include_ray_error = bool(residual['seed_include_ray_error'])
    for start in range(0, n, args.chunk_size):
        end = min(start + args.chunk_size, n)
        index, n_real = _pad_indices(start, end, n, args.chunk_size)
        candidates = dataset.batch.index_select(index).to(
            device, dtype=env.kin.dtype)
        features = initial_observation_features(
            env.kin, candidates,
            include_log_manip=include_log_manip,
            include_ray_error=include_ray_error)
        with torch.no_grad():
            distribution, _, _ = selector.distribution_and_values(
                features, candidates.valid)
            selected_index = distribution.logits.argmax(dim=-1)
            representation = selector.selected_representation(
                features, candidates.valid, selected_index)
            gate, latent = residual_head.deterministic_action(
                representation,
                gate_threshold=float(residual['gate_threshold']))
        row = torch.arange(args.chunk_size, device=device)
        base_q = candidates.q0[row, selected_index]
        shield = apply_residual_seed(
            env.kin, env.collision, base_q, candidates.p0,
            candidates.line_dir, candidates.n_target, latent,
            enabled=gate, config=shield_config)
        hard_valid = (
            shield.diagnostics.finite
            & shield.diagnostics.joint_limits
            & shield.diagnostics.position
            & shield.diagnostics.cone
            & shield.diagnostics.collision_free
            & shield.diagnostics.branch
            & shield.diagnostics.projection_ok
            & shield.diagnostics.input_finite
        )
        hard_violation = ~hard_valid
        if (not bool(shield.valid.all().item())
                or bool(hard_violation.any().item())):
            failed = torch.nonzero(
                (~shield.valid | hard_violation)[:n_real],
                as_tuple=False).flatten().cpu().tolist()
            task_ids = dataset.task_indices[start:end][failed].tolist()
            raise RuntimeError(
                'safety shield rejected an evaluation seed; refusing rollout. '
                f'source task ids: {task_ids[:20]}')

        paired_selection = SeedSelection(
            q0=torch.cat([base_q, shield.q], dim=0),
            p0=torch.cat([candidates.p0, candidates.p0], dim=0),
            line_dir=torch.cat(
                [candidates.line_dir, candidates.line_dir], dim=0),
            n_target=torch.cat(
                [candidates.n_target, candidates.n_target], dim=0),
        )
        result = rollout_seed_selection(
            env, paired_selection, controller, gamma=controller_gamma)
        output_slice = slice(start, end)
        _record_rollout(
            rollout_outputs['base'], result,
            slice(0, args.chunk_size), output_slice, n_real)
        _record_rollout(
            rollout_outputs['residual'], result,
            slice(args.chunk_size, 2 * args.chunk_size), output_slice, n_real)

        candidate_index[output_slice] = (
            selected_index[:n_real].cpu().numpy())
        gate_output[output_slice] = gate[:n_real].cpu().numpy()
        latent_output[output_slice] = latent[:n_real].cpu().numpy()
        alpha_output[output_slice] = (
            shield.accepted_alpha[:n_real].cpu().numpy())
        q_delta_norm[output_slice] = (
            (shield.q - base_q).norm(dim=-1)[:n_real].cpu().numpy())
        shield_outputs['valid'][output_slice] = (
            shield.valid[:n_real].cpu().numpy())
        shield_outputs['hard_violation'][output_slice] = (
            hard_violation[:n_real].cpu().numpy())
        for name in (
                'position_error', 'cone_cosine', 'collision_margin',
                'joint_margin', 'branch_distance', 'basis_fallback',
                'selected_index'):
            shield_outputs[name][output_slice] = (
                getattr(shield.diagnostics, name)[:n_real].cpu().numpy())
        print(f'[residual-eval] {end}/{n}', flush=True)

    objective_key = f'{seed_return}_return'
    base_objective = rollout_outputs['base'][objective_key]
    residual_objective = rollout_outputs['residual'][objective_key]
    objective_delta = residual_objective - base_objective
    progress_delta = (
        rollout_outputs['residual']['progress_m']
        - rollout_outputs['base']['progress_m'])
    objective_mean, objective_low, objective_high, n_geometries = (
        geometry_grouped_bootstrap_ci(
            objective_delta, task_fingerprints))
    progress_mean, progress_low, progress_high, progress_geometries = (
        geometry_grouped_bootstrap_ci(
            progress_delta, task_fingerprints))
    if progress_geometries != n_geometries:
        raise RuntimeError('bootstrap geometry grouping is internally inconsistent')
    base_objective_mean, base_geometries = geometry_grouped_mean(
        base_objective, task_fingerprints)
    residual_objective_mean, residual_geometries = geometry_grouped_mean(
        residual_objective, task_fingerprints)
    base_progress_mean, base_progress_geometries = geometry_grouped_mean(
        rollout_outputs['base']['progress_m'], task_fingerprints)
    residual_progress_mean, residual_progress_geometries = geometry_grouped_mean(
        rollout_outputs['residual']['progress_m'], task_fingerprints)
    if {base_geometries, residual_geometries, base_progress_geometries,
            residual_progress_geometries} != {n_geometries}:
        raise RuntimeError('metric geometry grouping is internally inconsistent')

    active_count = int(gate_output.sum())
    active_alpha0 = gate_output & (alpha_output == 0.0)
    gate_rate = float(gate_output.mean())
    active_alpha0_rate = (
        float(active_alpha0.sum() / active_count) if active_count else 0.0)
    active_alpha_median = (
        float(np.median(alpha_output[gate_output])) if active_count else 0.0)
    hard_violation_count = int(shield_outputs['hard_violation'].sum())
    print(
        f'[residual-eval] {seed_return} paired delta '
        f'{objective_mean:+.4f}  95% CI '
        f'[{objective_low:+.4f}, {objective_high:+.4f}]', flush=True)
    print(
        f'[residual-eval] progress delta {progress_mean:+.4f} m  '
        f'95% CI [{progress_low:+.4f}, {progress_high:+.4f}] m',
        flush=True)
    print(
        f'[residual-eval] gate {gate_rate:.1%}  active alpha0 '
        f'{active_alpha0_rate:.1%}  active median alpha '
        f'{active_alpha_median:.3f}  hard violations '
        f'{hard_violation_count}', flush=True)

    payload: dict[str, Any] = {
        'task_indices': dataset.task_indices[:n].numpy(),
        'task_geometry_sha256': np.asarray(task_fingerprints),
        'residual_checkpoint_sha256': np.asarray(
            residual_artifact['sha256']),
        'source_checkpoint_sha256': np.asarray(source_artifact['sha256']),
        'candidate_cache_sha256': np.asarray(candidate_artifact['sha256']),
        'controller_agent_sha256': np.asarray(
            controller_artifact['agent']['sha256']),
        'controller_config_sha256': np.asarray(
            controller_artifact['config']['sha256']),
        'controller_state_sha256': np.asarray(controller_state_hash),
        'source_format': np.asarray(source_format),
        'residual_format': np.asarray(residual_format),
        'seed_return_objective': np.asarray(seed_return),
        'controller_kind': np.asarray(controller_kind),
        'tau_enter': np.float64(np.nan if tau_enter is None else tau_enter),
        'tau_exit': np.float64(np.nan if tau_exit is None else tau_exit),
        'task_overlap_audited': np.bool_(overlap_audited),
        'task_overlap_rows': np.int64(overlap_rows),
        'task_overlap_unique': np.int64(overlap_unique),
        'physical_n_tasks_input': np.int64(physical_n_tasks_input),
        'physical_n_tasks_retained': np.int64(physical_n_tasks_retained),
        'physical_n_tasks_rejected': np.int64(physical_n_tasks_rejected),
        'candidate_index': candidate_index,
        'residual_gate': gate_output,
        'residual_latent': latent_output,
        'residual_alpha': alpha_output,
        'residual_q_delta_norm': q_delta_norm,
        'paired_objective_delta': objective_delta,
        'paired_progress_delta_m': progress_delta,
        'metric_weighting': np.asarray('unique-task-geometry'),
        'metric_objective_base_mean': np.float64(base_objective_mean),
        'metric_objective_residual_mean': np.float64(
            residual_objective_mean),
        'metric_progress_base_mean_m': np.float64(base_progress_mean),
        'metric_progress_residual_mean_m': np.float64(
            residual_progress_mean),
        'diagnostic_row_objective_base_mean': np.float64(
            base_objective.mean()),
        'diagnostic_row_objective_residual_mean': np.float64(
            residual_objective.mean()),
        'diagnostic_row_paired_objective_delta': np.float64(
            objective_delta.mean()),
        'diagnostic_row_paired_progress_delta_m': np.float64(
            progress_delta.mean()),
        'metric_paired_objective_delta': np.float64(objective_mean),
        'metric_paired_objective_ci95_low': np.float64(objective_low),
        'metric_paired_objective_ci95_high': np.float64(objective_high),
        'metric_paired_progress_delta_m': np.float64(progress_mean),
        'metric_paired_progress_ci95_low_m': np.float64(progress_low),
        'metric_paired_progress_ci95_high_m': np.float64(progress_high),
        'metric_unique_task_geometries': np.int64(n_geometries),
        'metric_bootstrap_seed': np.int64(BOOTSTRAP_SEED),
        'metric_bootstrap_samples': np.int64(BOOTSTRAP_SAMPLES),
        'metric_gate_activation_rate': np.float64(gate_rate),
        'metric_active_alpha0_rate': np.float64(active_alpha0_rate),
        'metric_active_alpha_median': np.float64(active_alpha_median),
        'metric_hard_violation_count': np.int64(hard_violation_count),
    }
    for variant, metrics in rollout_outputs.items():
        for name, value in metrics.items():
            payload[f'{variant}_{name}'] = value
    for name, value in shield_outputs.items():
        payload[f'shield_{name}'] = value
    _atomic_savez_new(out, payload)
    print(f'[residual-eval] saved -> {out}', flush=True)


if __name__ == '__main__':
    main()


__all__ = [
    'BOOTSTRAP_SAMPLES',
    'BOOTSTRAP_SEED',
    'RESIDUAL_FORMAT',
    'geometry_grouped_bootstrap_ci',
    'geometry_grouped_mean',
    'main',
]
