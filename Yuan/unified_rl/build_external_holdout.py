"""Build a genuinely independent K=8 external candidate holdout.

The default random seeds are a development set only. A sealed final set must
use newly chosen, explicitly recorded seeds and must remain undisclosed until
all model choices are frozen. In both cases, pass every train/validation
candidate cache via repeated ``--exclude-candidates`` arguments: construction
fails unless task geometry is internally unique and has zero exact overlap.

The output uses the six-field ``rank_train/candidates_K8.npz`` schema:
``seeds``, ``ik_ok``, ``p0``, ``line_dir``, ``n_target``, and ``q0_pilot``.
The newly constructed and filtered line pool is published beside it as
``<output-stem>.pool.pt``. The metadata JSON is published last and is the
    valid commit marker: renaming each file is atomic, but the three-file group cannot
be committed as one filesystem transaction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.system_eval.seed_sources import diffusion_seeds
from Yuan.unified_rl.checkpoint import (
    env_config_from_run,
    resolve_controller_dir,
)
from Yuan.unified_rl.provenance import file_fingerprint


RANK_TRAIN_FIELDS = (
    'seeds',
    'ik_ok',
    'p0',
    'line_dir',
    'n_target',
    'q0_pilot',
)
N_CANDIDATES = 8
DEV_POOL_SEED = 20260721
DEV_TASK_SEED = 20260722
DEV_DIFFUSION_SEED = 20260723


def canonical_task_geometry(
    p0: np.ndarray,
    line_dir: np.ndarray,
    n_target: np.ndarray,
) -> np.ndarray:
    """Return canonical little-endian float32 task rows ``(p0, d, n)``.

    This function is pure and intentionally mirrors the geometry identity used
    by ``CachedSeedCandidateDataset.task_fingerprints``.
    """
    values = {
        'p0': np.asarray(p0),
        'line_dir': np.asarray(line_dir),
        'n_target': np.asarray(n_target),
    }
    if values['p0'].ndim != 2 or values['p0'].shape[1:] != (3,):
        raise ValueError(f"p0 must have shape (N,3), got {values['p0'].shape}")
    n_tasks = values['p0'].shape[0]
    if n_tasks < 1:
        raise ValueError('task geometry must contain at least one row')
    for name, value in values.items():
        if value.shape != (n_tasks, 3):
            raise ValueError(
                f'{name} must have shape ({n_tasks},3), got {value.shape}')
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f'{name} must contain numeric values')

    geometry = np.concatenate(
        [values['p0'], values['line_dir'], values['n_target']], axis=1)
    canonical = np.asarray(
        geometry, dtype=np.dtype('<f4'), order='C').copy()
    if not np.isfinite(canonical).all():
        raise ValueError('task geometry must be finite after float32 conversion')
    canonical[canonical == np.float32(0.0)] = np.float32(0.0)
    return canonical


def task_geometry_fingerprints(geometry: np.ndarray) -> tuple[str, ...]:
    """Hash canonical ``(N,9)`` geometry rows without filesystem access."""
    geometry = np.asarray(geometry)
    if geometry.ndim != 2 or geometry.shape[1:] != (9,):
        raise ValueError(
            f'canonical geometry must have shape (N,9), got {geometry.shape}')
    canonical = np.asarray(
        geometry, dtype=np.dtype('<f4'), order='C').copy()
    if not np.isfinite(canonical).all():
        raise ValueError('canonical task geometry must be finite')
    canonical[canonical == np.float32(0.0)] = np.float32(0.0)
    return tuple(
        hashlib.sha256(row.tobytes(order='C')).hexdigest()
        for row in canonical)


def audit_geometry_independence(
    geometry: np.ndarray,
    excluded_geometry: Mapping[str, np.ndarray] | None = None,
) -> dict:
    """Pure audit for within-set uniqueness and zero excluded-set overlap.

    Args:
        geometry: Canonical ``(N,9)`` task geometry to audit.
        excluded_geometry: Human-readable source name to canonical ``(M,9)``
            geometry. Excluded sets need not be mutually disjoint.

    Returns:
        JSON-serializable counts. Any duplicate or overlap raises ``ValueError``.
    """
    fingerprints = task_geometry_fingerprints(geometry)
    fingerprint_set = set(fingerprints)
    if len(fingerprint_set) != len(fingerprints):
        counts: dict[str, int] = {}
        for digest in fingerprints:
            counts[digest] = counts.get(digest, 0) + 1
        duplicate = next(key for key, count in counts.items() if count > 1)
        raise ValueError(
            'generated task geometry is not unique: '
            f'{len(fingerprints) - len(fingerprint_set)} duplicate rows; '
            f'first duplicate sha256={duplicate}')

    exclusion_stats: dict[str, dict[str, int]] = {}
    for name, excluded in (excluded_geometry or {}).items():
        excluded_fingerprints = task_geometry_fingerprints(excluded)
        overlap = fingerprint_set.intersection(excluded_fingerprints)
        if overlap:
            raise ValueError(
                f'task geometry overlaps excluded cache {name!r}: '
                f'{len(overlap)} rows; first sha256={sorted(overlap)[0]}')
        exclusion_stats[str(name)] = {
            'n_tasks': len(excluded_fingerprints),
            'n_unique_tasks': len(set(excluded_fingerprints)),
            'n_overlap': 0,
        }
    return {
        'n_tasks': len(fingerprints),
        'n_unique_tasks': len(fingerprint_set),
        'n_internal_duplicates': 0,
        'exclusions': exclusion_stats,
    }


def select_valid_indices_without_replacement(
    valid_indices: np.ndarray,
    n_tasks: int,
    seed: int,
) -> np.ndarray:
    """Pure deterministic draw from explicit valid pool indices."""
    valid_indices = np.asarray(valid_indices)
    if valid_indices.ndim != 1:
        raise ValueError(
            f'valid_indices must be one-dimensional, got {valid_indices.shape}')
    if not np.issubdtype(valid_indices.dtype, np.integer):
        raise ValueError('valid_indices must contain integers')
    valid_indices = valid_indices.astype(np.int64, copy=True)
    if valid_indices.size == 0:
        raise ValueError('valid_indices cannot be empty')
    if np.unique(valid_indices).size != valid_indices.size:
        raise ValueError('valid_indices must be unique')
    if (valid_indices < 0).any():
        raise ValueError('valid_indices cannot contain negative values')
    if n_tasks < 1:
        raise ValueError('n_tasks must be positive')
    if n_tasks > valid_indices.size:
        raise ValueError(
            f'cannot draw {n_tasks} tasks without replacement from '
            f'{valid_indices.size} valid pool entries')
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(valid_indices.size)
    return valid_indices[order[:n_tasks]].copy()


def validate_rank_train_payload(
    payload: Mapping[str, np.ndarray],
    *,
    n_tasks: int,
    n_candidates: int = N_CANDIDATES,
) -> None:
    """Pure fail-closed validation of the six-field candidate-cache payload."""
    if set(payload) != set(RANK_TRAIN_FIELDS):
        missing = sorted(set(RANK_TRAIN_FIELDS) - set(payload))
        extra = sorted(set(payload) - set(RANK_TRAIN_FIELDS))
        raise ValueError(
            f'candidate payload must contain exactly the rank_train fields; '
            f'missing={missing}, extra={extra}')
    expected = {
        'seeds': (n_tasks, n_candidates, 7),
        'ik_ok': (n_tasks, n_candidates),
        'p0': (n_tasks, 3),
        'line_dir': (n_tasks, 3),
        'n_target': (n_tasks, 3),
        'q0_pilot': (n_tasks, 7),
    }
    arrays = {name: np.asarray(payload[name]) for name in RANK_TRAIN_FIELDS}
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f'{name} must have shape {shape}, got {arrays[name].shape}')
    if arrays['ik_ok'].dtype != np.bool_:
        raise ValueError('ik_ok must have boolean dtype')
    for name in ('seeds', 'p0', 'line_dir', 'n_target', 'q0_pilot'):
        if not np.issubdtype(arrays[name].dtype, np.number):
            raise ValueError(f'{name} must contain numeric values')
    if not np.isfinite(arrays['seeds'][arrays['ik_ok']]).all():
        raise ValueError('IK-valid diffusion seeds must be finite')
    for name in ('p0', 'line_dir', 'n_target', 'q0_pilot'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'{name} must be finite')
    for name in ('line_dir', 'n_target'):
        norm = np.linalg.norm(arrays[name].astype(np.float64), axis=1)
        if not np.allclose(norm, 1.0, atol=1e-3, rtol=1e-3):
            raise ValueError(f'{name} must contain unit vectors')


def _array_sha256(value: np.ndarray, dtype: np.dtype) -> str:
    canonical = np.asarray(value, dtype=dtype, order='C')
    return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def _read_geometry(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        aliases = {
            'p0': ('p0', 'cs_p0'),
            'line_dir': ('line_dir', 'cs_line_dir'),
            'n_target': ('n_target', 'cs_n_target'),
        }
        values: dict[str, np.ndarray] = {}
        for label, names in aliases.items():
            present = [name for name in names if name in data]
            if not present:
                raise ValueError(
                    f'excluded cache {path} is missing {label}; '
                    f'expected one of {names}')
            candidates = [np.asarray(data[name]) for name in present]
            if any(not np.array_equal(candidates[0], value)
                   for value in candidates[1:]):
                raise ValueError(
                    f'excluded cache {path} contains conflicting aliases '
                    f'for {label}: {present}')
            values[label] = candidates[0]
    return canonical_task_geometry(
        values['p0'], values['line_dir'], values['n_target'])


def _resolve_config_reference(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=True)
    from_working_directory = path.resolve()
    if from_working_directory.exists():
        return from_working_directory.resolve(strict=True)
    return (config_path.parent / path).resolve(strict=True)


def _assert_new_seed_set(pool_seed: int, task_seed: int,
                         diffusion_seed: int, source_pool_seed: int | None) -> None:
    seeds = (int(pool_seed), int(task_seed), int(diffusion_seed))
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            'pool, task-selection, and diffusion seeds must be distinct')
    if source_pool_seed is not None and pool_seed == source_pool_seed:
        raise ValueError(
            '--pool-seed must differ from the controller run training-pool seed '
            f'({source_pool_seed})')


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix('.meta.json')


def _pool_path(output_path: Path) -> Path:
    return output_path.with_suffix('.pool.pt')


def _check_output_targets(output_path: Path) -> tuple[Path, Path]:
    if output_path.suffix != '.npz':
        raise ValueError('--out must end in .npz')
    metadata_path = _metadata_path(output_path)
    pool_path = _pool_path(output_path)
    existing = [path for path in (output_path, pool_path, metadata_path)
                if path.exists()]
    if existing:
        raise FileExistsError(
            'refusing to overwrite holdout artifacts: '
            + ', '.join(str(path) for path in existing))
    return metadata_path, pool_path


def _assert_inputs_unchanged(
    fingerprints: Mapping[str, Mapping[str, str | int]],
) -> None:
    """Re-hash every external input immediately before artifact publication."""
    changed = []
    for label, before in fingerprints.items():
        current = file_fingerprint(str(before['path']))
        if current != dict(before):
            changed.append(
                f'{label}: before={dict(before)!r}, current={current!r}')
    if changed:
        raise RuntimeError(
            'external inputs changed after first use; refusing publication:\n  - '
            + '\n  - '.join(changed))


def _reserve_output_targets(paths: Sequence[Path]) -> list[Path]:
    """Reserve final names with O_EXCL; caller owns returned placeholders."""
    reserved: list[Path] = []
    try:
        for path in paths:
            descriptor = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(descriptor)
            reserved.append(path)
    except BaseException:
        for path in reserved:
            path.unlink(missing_ok=True)
        raise
    return reserved


def _write_artifacts(
    output_path: Path,
    pool_path: Path,
    metadata_path: Path,
    payload: Mapping[str, np.ndarray],
    pool: LineDistribution,
    metadata: dict,
    input_fingerprints: Mapping[str, Mapping[str, str | int]],
) -> None:
    """Stage all artifacts, then reserve and publish them metadata-last.

    Each final rename is atomic. There is no atomic transaction spanning all
    three paths, so readers must require a valid metadata commit marker and verify
    its recorded hashes. A process failure is rolled back best-effort; a hard
    crash can leave a partial group without the commit marker.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_paths = (pool_path, output_path, metadata_path)
    with tempfile.TemporaryDirectory(
            prefix=f'.{output_path.stem}.stage-',
            dir=output_path.parent) as staging_directory:
        staging_path = Path(staging_directory)
        temporary_npz = staging_path / 'candidates.npz'
        temporary_pool = staging_path / 'line_pool.pt'
        temporary_json = staging_path / 'metadata.json'
        reserved: list[Path] = []
        try:
            np.savez_compressed(temporary_npz, **payload)
            pool.save(temporary_pool)
            output_fingerprint = file_fingerprint(temporary_npz)
            output_fingerprint['path'] = str(output_path.resolve())
            pool_fingerprint = file_fingerprint(temporary_pool)
            pool_fingerprint['path'] = str(pool_path.resolve())
            metadata['output'] = output_fingerprint
            metadata['pool_artifact'] = pool_fingerprint
            temporary_json.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + '\n',
                encoding='utf-8')

            # Last possible content check before acquiring final names. Inputs
            # were fingerprinted before first read/use in build_external_holdout.
            _assert_inputs_unchanged(input_fingerprints)
            reserved = _reserve_output_targets(final_paths)
            # Pool and cache first; valid metadata last is the commit marker.
            os.replace(temporary_pool, pool_path)
            os.replace(temporary_npz, output_path)
            os.replace(temporary_json, metadata_path)
            reserved.clear()
        finally:
            for path in reserved:
                path.unlink(missing_ok=True)


def build_external_holdout(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Build and audit the external cache described by parsed CLI arguments."""
    output_path = Path(args.out).expanduser()
    metadata_path, pool_path = _check_output_targets(output_path)
    if args.n_tasks < 1:
        raise ValueError('--n-tasks must be positive')
    if args.chunk_tasks < 1:
        raise ValueError('--chunk-tasks must be positive')

    controller_dir = resolve_controller_dir(args.controller_ckpt)
    controller_config_path = (controller_dir / 'config.yaml').resolve(strict=True)
    input_fingerprints: dict[str, dict[str, str | int]] = {
        'controller_config': file_fingerprint(controller_config_path),
    }
    with open(controller_config_path, 'r', encoding='utf-8') as stream:
        controller_config = yaml.safe_load(stream)
    if not isinstance(controller_config, dict):
        raise ValueError('controller config must contain a mapping')
    if 'line_distribution' not in controller_config:
        raise ValueError('controller config is missing line_distribution')
    line_config = controller_config['line_distribution']
    source_pool_seed = line_config.get('train_seed')
    _assert_new_seed_set(
        args.pool_seed, args.task_seed, args.diffusion_seed,
        None if source_pool_seed is None else int(source_pool_seed))

    diffusion_config_path = Path(
        args.diffusion_config).expanduser().resolve(strict=True)
    input_fingerprints['diffusion_config'] = file_fingerprint(
        diffusion_config_path)
    with open(diffusion_config_path, 'r', encoding='utf-8') as stream:
        full_diffusion_config = yaml.safe_load(stream)
    if not isinstance(full_diffusion_config, dict):
        raise ValueError('diffusion config must contain a mapping')
    diffusion_config = full_diffusion_config.get('diffusion')
    if not isinstance(diffusion_config, dict):
        raise ValueError('diffusion config is missing a diffusion mapping')
    configured_candidates = int(diffusion_config.get('n_samples', N_CANDIDATES))
    if configured_candidates != N_CANDIDATES:
        raise ValueError(
            f'diffusion config n_samples must be {N_CANDIDATES}, '
            f'got {configured_candidates}')
    required_diffusion = ('ckpt', 'ddim_steps', 'cfg_w')
    missing = [key for key in required_diffusion if key not in diffusion_config]
    if missing:
        raise ValueError(f'diffusion config is missing keys: {missing}')
    diffusion_checkpoint = _resolve_config_reference(
        diffusion_config['ckpt'], diffusion_config_path)
    input_fingerprints['diffusion_checkpoint'] = file_fingerprint(
        diffusion_checkpoint)

    exclude_paths = [Path(path).expanduser().resolve(strict=True)
                     for path in args.exclude_candidates]
    if len(set(exclude_paths)) != len(exclude_paths):
        raise ValueError('--exclude-candidates contains duplicate paths')
    if len(exclude_paths) < 3 and not args.allow_incomplete_exclusions:
        raise ValueError(
            'at least three distinct --exclude-candidates caches are required; '
            'use --allow-incomplete-exclusions only for an explicitly labeled '
            'diagnostic artifact')
    for index, path in enumerate(exclude_paths):
        input_fingerprints[f'excluded_candidates[{index}]'] = file_fingerprint(path)
    excluded_geometry = {
        str(path): _read_geometry(path) for path in exclude_paths
    }

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    env_config = env_config_from_run(controller_config, n_envs=1)
    proxy = NSRLBatchedEnv(env_config, line_dist=None, device=device)

    pool_size = (
        int(line_config['n_pool']) if args.pool_size is None
        else int(args.pool_size))
    if pool_size < args.n_tasks:
        raise ValueError(
            f'pool size {pool_size} cannot supply {args.n_tasks} unique tasks')
    noise_degrees = float(line_config['n_target_noise_deg'])
    threshold_m = (
        float(line_config['feasibility_threshold_m'])
        if line_config.get('feasibility_filter', False) else None)
    # Deliberately do not use LineDistribution.load_or_build: its historical
    # cache key does not cover all construction/filter implementation inputs.
    pool = LineDistribution(
        kin=proxy.kin,
        collision=proxy.collision,
        n_pool=pool_size,
        n_target_noise_deg=noise_degrees,
        seed=args.pool_seed,
    )
    if threshold_m is None:
        filter_stats = {
            'n_initial': int(pool.n_pool),
            'n_feasible': int(pool.n_valid),
            'threshold_m': None,
            'threshold_steps': None,
        }
    else:
        filter_stats = pool.filter_by_classical_controller(
            env_config, threshold_m=threshold_m)

    valid_indices = torch.nonzero(
        pool.valid_mask, as_tuple=False).squeeze(-1).cpu().numpy()
    selected_indices = select_valid_indices_without_replacement(
        valid_indices, args.n_tasks, args.task_seed)
    selected = torch.from_numpy(selected_indices).to(
        device=pool.q_pool.device, dtype=torch.long)
    q0_pilot_t = pool.q_pool[selected]
    line_dir_t = pool.line_dir_pool[selected]
    n_target_t = pool.n_target_pool[selected]
    with torch.no_grad():
        p0_t, _, _, _ = proxy.kin.tcp_fk_jac(
            q0_pilot_t.to(dtype=proxy.kin.dtype))

    q0_pilot = q0_pilot_t.float().cpu().numpy().astype(np.float32, copy=True)
    p0 = p0_t.float().cpu().numpy().astype(np.float32, copy=True)
    line_dir = line_dir_t.float().cpu().numpy().astype(np.float32, copy=True)
    n_target = n_target_t.float().cpu().numpy().astype(np.float32, copy=True)
    geometry = canonical_task_geometry(p0, line_dir, n_target)

    # Audit before the expensive diffusion pass.
    audit = audit_geometry_independence(geometry, excluded_geometry)
    eval_like = {
        'cs_p0': p0,
        'cs_line_dir': line_dir,
        'cs_n_target': n_target,
    }
    seeds, ik_ok = diffusion_seeds(
        eval_like,
        diffusion_checkpoint,
        n_samples=N_CANDIDATES,
        ddim_steps=int(diffusion_config['ddim_steps']),
        cfg_w=float(diffusion_config['cfg_w']),
        sample_seed=args.diffusion_seed,
        kin=proxy.kin,
        device=device,
        use_ema=bool(diffusion_config.get('use_ema', True)),
        chunk_tasks=args.chunk_tasks,
    )
    payload = {
        'seeds': np.asarray(seeds, dtype=np.float32),
        'ik_ok': np.asarray(ik_ok, dtype=bool),
        'p0': p0,
        'line_dir': line_dir,
        'n_target': n_target,
        'q0_pilot': q0_pilot,
    }
    validate_rank_train_payload(
        payload, n_tasks=args.n_tasks, n_candidates=N_CANDIDATES)
    # Re-run the pure audit on the exact float32 arrays being serialized.
    audit = audit_geometry_independence(
        canonical_task_geometry(
            payload['p0'], payload['line_dir'], payload['n_target']),
        excluded_geometry,
    )

    source_fingerprints = {
        'controller_config': input_fingerprints['controller_config'],
        'diffusion_config': input_fingerprints['diffusion_config'],
        'diffusion_checkpoint': input_fingerprints['diffusion_checkpoint'],
        'excluded_candidates': [
            input_fingerprints[f'excluded_candidates[{index}]']
            for index in range(len(exclude_paths))
        ],
    }
    uses_development_seeds = (
        int(args.pool_seed), int(args.task_seed), int(args.diffusion_seed)
    ) == (DEV_POOL_SEED, DEV_TASK_SEED, DEV_DIFFUSION_SEED)
    metadata = {
        'schema': 'unified-external-holdout-v1',
        'purpose': (
            'external-holdout' if len(exclude_paths) >= 3
            else 'diagnostic-incomplete-exclusions'),
        'parameters': {
            'seed_profile': (
                'development-defaults' if uses_development_seeds
                else 'explicit-custom'),
            'n_tasks': int(args.n_tasks),
            'n_candidates': N_CANDIDATES,
            'device': str(device),
            'pool_seed': int(args.pool_seed),
            'task_seed': int(args.task_seed),
            'diffusion_seed': int(args.diffusion_seed),
            'pool_size': pool_size,
            'n_pool_valid': int(valid_indices.size),
            'line_pool_filter': filter_stats,
            'n_target_noise_deg': noise_degrees,
            'feasibility_threshold_m': threshold_m,
            'ddim_steps': int(diffusion_config['ddim_steps']),
            'cfg_w': float(diffusion_config['cfg_w']),
            'use_ema': bool(diffusion_config.get('use_ema', True)),
            'chunk_tasks': int(args.chunk_tasks),
            'selected_pool_indices_sha256': _array_sha256(
                selected_indices, np.dtype('<i8')),
            'allow_incomplete_exclusions': bool(
                args.allow_incomplete_exclusions),
            'n_exclusion_caches': len(exclude_paths),
        },
        'audit': audit,
        'sources': source_fingerprints,
        'publication': {
            'protocol': (
                'stage all files; reserve all final names with O_EXCL; '
                'atomically replace pool and cache; publish metadata last'),
            'atomic_boundary': (
                'each rename is atomic, but the three-file group is not a '
                'single filesystem transaction'),
            'commit_marker': str(metadata_path.resolve()),
            'reader_rule': (
                'require parseable metadata and verify its output and '
                'pool_artifact hashes before use; an empty reservation is not '
                'a commit marker'),
        },
    }
    _write_artifacts(
        output_path, pool_path, metadata_path, payload, pool, metadata,
        input_fingerprints)
    return output_path, pool_path, metadata_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build an independently sampled K=8 external holdout cache.',
        epilog=(
            'The default seeds are development-only. For a sealed final test, '
            'choose fresh explicit seeds after freezing all model decisions and '
            'keep the resulting cache hidden until final evaluation.'),
    )
    parser.add_argument('--out', required=True,
                        help='new .npz cache path; cache/pool/JSON overwrite is refused')
    parser.add_argument(
        '--exclude-candidates', action='append', default=[],
        help=(
            'candidate cache whose task geometry must have zero overlap; '
            'repeat at least three times by default'))
    parser.add_argument(
        '--allow-incomplete-exclusions', action='store_true',
        help=(
            'allow fewer than three exclusion caches and label the result as '
            'diagnostic-incomplete-exclusions'))
    parser.add_argument(
        '--controller-ckpt',
        default='Yuan/RL_controller/runs/distill_r12m_b0.965_soup2',
        help='controller run supplying environment and line-pool configuration')
    parser.add_argument(
        '--diffusion-config', default='Yuan/system_eval/config.yaml',
        help='YAML supplying diffusion checkpoint, DDIM steps, CFG, and EMA choice')
    parser.add_argument('--n-tasks', type=int, default=2048)
    parser.add_argument('--pool-size', type=int, default=None,
                        help='override controller config n_pool (development only)')
    parser.add_argument('--pool-seed', type=int, default=DEV_POOL_SEED)
    parser.add_argument('--task-seed', type=int, default=DEV_TASK_SEED)
    parser.add_argument('--diffusion-seed', type=int, default=DEV_DIFFUSION_SEED)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--chunk-tasks', type=int, default=64)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    output_path, pool_path, metadata_path = build_external_holdout(args)
    output_fingerprint = file_fingerprint(output_path)
    print(
        f'[external-holdout] wrote {output_path}, {pool_path}, '
        f'and {metadata_path}  '
        f'sha256={output_fingerprint["sha256"]}')


if __name__ == '__main__':
    main()
