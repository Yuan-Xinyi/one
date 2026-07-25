"""Strict paired 2x2 analysis for joint seed/controller evaluations.

The four cells are ordered as ``S0C0, S0C1, S1C0, S1C1``.  The primary
decomposition uses ``S0C0`` as the reference::

    joint = controller_at_s0 + selector_at_c0 + interaction

All reported macro estimates weight unique task geometries equally.  The
bootstrap resamples geometry groups, so repeated rows of one geometry cannot
dominate the confidence interval.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CELL_NAMES = ('s0c0', 's0c1', 's1c0', 's1c1')
EFFECT_COEFFICIENTS = {
    'controller_at_s0': np.asarray([-1.0, 1.0, 0.0, 0.0]),
    'selector_at_c0': np.asarray([-1.0, 0.0, 1.0, 0.0]),
    'interaction': np.asarray([1.0, -1.0, -1.0, 1.0]),
    'joint': np.asarray([-1.0, 0.0, 0.0, 1.0]),
    'controller_at_s1': np.asarray([0.0, 0.0, -1.0, 1.0]),
    'selector_at_c1': np.asarray([0.0, -1.0, 0.0, 1.0]),
}


@dataclass(frozen=True)
class EvaluationCell:
    """Validated subset of one ``evaluate.py`` NPZ artifact."""

    name: str
    path: Path
    file_size: int
    file_sha256: str
    task_indices: np.ndarray
    task_fingerprints: np.ndarray
    valid: np.ndarray
    policy_progress_m: np.ndarray
    policy_candidate_index: np.ndarray
    first_valid_candidate_index: np.ndarray
    candidate_cache_sha256: str
    seed_checkpoint_sha256: str
    controller_agent_sha256: str
    controller_config_sha256: str
    controller_state_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sha256_mask(mask: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(mask).tobytes()).hexdigest()


def _scalar_string(data: Mapping[str, np.ndarray], key: str) -> str:
    if key not in data:
        raise ValueError(f'evaluation is missing {key!r}')
    value = data[key]
    if value.shape != () or value.dtype.kind != 'U':
        raise ValueError(f'{key} must be a scalar unicode string')
    result = str(value.item())
    if not result:
        raise ValueError(f'{key} must be non-empty')
    return result


def _sha256_scalar(data: Mapping[str, np.ndarray], key: str) -> str:
    result = _scalar_string(data, key)
    if (len(result) != 64
            or any(char not in '0123456789abcdef' for char in result)):
        raise ValueError(f'{key} must be a lowercase SHA-256 string')
    return result


def _array(
    data: Mapping[str, np.ndarray],
    key: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if key not in data:
        raise ValueError(f'evaluation is missing {key!r}')
    value = data[key]
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(
            f'{key} must have shape {shape} and dtype {dtype}, got '
            f'{value.shape} and {value.dtype}')
    return np.array(value, copy=True)


def _valid_mask(data: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    masks = []
    for key in ('evaluated_candidate_valid', 'candidate_valid'):
        if key not in data:
            continue
        value = data[key]
        if (value.ndim != 2 or value.shape[0] != n
                or value.shape[1] < 1 or value.dtype != np.dtype(np.bool_)):
            raise ValueError(
                f'{key} must be a boolean (n_tasks, n_candidates) array')
        masks.append((key, np.array(value, copy=True)))
    if not masks:
        raise ValueError(
            'evaluation contains neither evaluated_candidate_valid nor '
            'candidate_valid; a strict 2x2 audit requires the exact mask')
    mask = masks[0][1]
    for key, other in masks[1:]:
        if not np.array_equal(mask, other):
            raise ValueError(
                f'{masks[0][0]} and {key} disagree within one evaluation')
    if not mask.any(axis=1).all():
        raise ValueError('every evaluated task must have a valid candidate')
    if 'evaluated_candidate_valid_sha256' in data:
        recorded = _sha256_scalar(data, 'evaluated_candidate_valid_sha256')
        if recorded != _sha256_mask(mask):
            raise ValueError('evaluated candidate-valid mask SHA-256 is invalid')
    return mask


def load_evaluation(path: str | Path, name: str) -> EvaluationCell:
    """Load and fail-closed validate one paired-evaluation artifact."""
    if name not in CELL_NAMES:
        raise ValueError(f'unknown 2x2 cell name: {name!r}')
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f'evaluation path is not a file: {resolved}')
    file_size = resolved.stat().st_size
    file_sha256 = _sha256_file(resolved)
    with np.load(resolved, allow_pickle=False) as data:
        if 'task_indices' not in data:
            raise ValueError('evaluation is missing task_indices')
        task_indices_raw = data['task_indices']
        if task_indices_raw.ndim != 1:
            raise ValueError('task_indices must be one-dimensional')
        n = int(task_indices_raw.size)
        if n < 1:
            raise ValueError('evaluation must contain at least one task')
        task_indices = _array(
            data, 'task_indices', (n,), np.dtype(np.int64))
        if np.unique(task_indices).size != n:
            raise ValueError('task_indices must be unique')
        fingerprints = _array(
            data, 'task_geometry_sha256', (n,), np.dtype('<U64'))
        for fingerprint in fingerprints.tolist():
            if (len(fingerprint) != 64
                    or any(char not in '0123456789abcdef'
                           for char in fingerprint)):
                raise ValueError(
                    'task_geometry_sha256 contains an invalid fingerprint')
        valid = _valid_mask(data, n)
        progress = data.get('policy_progress_m')
        if (progress is None or progress.shape != (n,)
                or progress.dtype not in (np.dtype(np.float32),
                                          np.dtype(np.float64))):
            raise ValueError(
                'policy_progress_m must be a float32/float64 task vector')
        progress = np.asarray(progress, dtype=np.float64).copy()
        if not np.isfinite(progress).all():
            raise ValueError('policy_progress_m must be finite')
        selected = _array(
            data, 'policy_candidate_index', (n,), np.dtype(np.int64))
        first = _array(
            data, 'first_valid_candidate_index', (n,), np.dtype(np.int64))
        n_candidates = valid.shape[1]
        for label, indices in (('policy', selected), ('first-valid', first)):
            if ((indices < 0) | (indices >= n_candidates)).any():
                raise ValueError(f'{label} candidate index is out of range')
            if not valid[np.arange(n), indices].all():
                raise ValueError(f'{label} selected an invalid candidate')
        canonical_first = valid.argmax(axis=1)
        if not np.array_equal(first, canonical_first):
            raise ValueError(
                'first_valid_candidate_index is not the first valid slot')
        if 'seed_probe_enabled' in data:
            probe = data['seed_probe_enabled']
            if (probe.shape != () or probe.dtype != np.dtype(np.bool_)):
                raise ValueError('seed_probe_enabled must be a boolean scalar')
            if bool(probe.item()):
                raise ValueError(
                    '2x2 analysis requires static one-seed deployment; '
                    'controller-probe evaluation is not admissible')
        elif 'static_policy_progress_m' in data:
            raise ValueError(
                'evaluation contains probe outputs without an explicit '
                'disabled seed_probe_enabled flag')
        result = EvaluationCell(
            name=name,
            path=resolved,
            file_size=file_size,
            file_sha256=file_sha256,
            task_indices=task_indices,
            task_fingerprints=fingerprints,
            valid=valid,
            policy_progress_m=progress,
            policy_candidate_index=selected,
            first_valid_candidate_index=first,
            candidate_cache_sha256=_sha256_scalar(
                data, 'candidate_cache_sha256'),
            seed_checkpoint_sha256=_sha256_scalar(
                data, 'seed_checkpoint_sha256'),
            controller_agent_sha256=_sha256_scalar(
                data, 'controller_agent_sha256'),
            controller_config_sha256=_sha256_scalar(
                data, 'controller_config_sha256'),
            controller_state_sha256=_sha256_scalar(
                data, 'controller_state_sha256'),
        )
    # Recheck after reading to catch a concurrently replaced artifact.
    if (resolved.stat().st_size != file_size
            or _sha256_file(resolved) != file_sha256):
        raise RuntimeError(f'evaluation changed while loading: {resolved}')
    return result


def _require_equal(
    cells: Mapping[str, EvaluationCell],
    attribute: str,
    *,
    array: bool = False,
) -> Any:
    reference = getattr(cells[CELL_NAMES[0]], attribute)
    for name in CELL_NAMES[1:]:
        value = getattr(cells[name], attribute)
        same = np.array_equal(reference, value) if array else reference == value
        if not same:
            raise ValueError(
                f'2x2 cells disagree on {attribute}: '
                f'{CELL_NAMES[0]} versus {name}')
    return reference


def validate_2x2(cells: Mapping[str, EvaluationCell]) -> dict[str, Any]:
    """Validate paired rows, factor identities, masks and static selections."""
    if set(cells) != set(CELL_NAMES):
        raise ValueError(f'2x2 cells must be exactly {CELL_NAMES}')
    task_indices = _require_equal(cells, 'task_indices', array=True)
    fingerprints = _require_equal(cells, 'task_fingerprints', array=True)
    valid = _require_equal(cells, 'valid', array=True)
    candidate_cache_sha256 = _require_equal(
        cells, 'candidate_cache_sha256')
    first = _require_equal(cells, 'first_valid_candidate_index', array=True)

    if (cells['s0c0'].seed_checkpoint_sha256
            != cells['s0c1'].seed_checkpoint_sha256):
        raise ValueError('S0 checkpoint identity differs across C0/C1')
    if (cells['s1c0'].seed_checkpoint_sha256
            != cells['s1c1'].seed_checkpoint_sha256):
        raise ValueError('S1 checkpoint identity differs across C0/C1')
    if (cells['s0c0'].seed_checkpoint_sha256
            == cells['s1c0'].seed_checkpoint_sha256):
        raise ValueError('S0 and S1 checkpoint identities are identical')
    for attribute in (
            'controller_agent_sha256', 'controller_config_sha256',
            'controller_state_sha256'):
        if getattr(cells['s0c0'], attribute) != getattr(cells['s1c0'], attribute):
            raise ValueError(f'C0 {attribute} differs across S0/S1')
        if getattr(cells['s0c1'], attribute) != getattr(cells['s1c1'], attribute):
            raise ValueError(f'C1 {attribute} differs across S0/S1')
    if (cells['s0c0'].controller_state_sha256
            == cells['s0c1'].controller_state_sha256):
        raise ValueError('C0 and C1 controller state identities are identical')

    s0_selection = cells['s0c0'].policy_candidate_index
    s1_selection = cells['s1c0'].policy_candidate_index
    if not np.array_equal(s0_selection, cells['s0c1'].policy_candidate_index):
        raise ValueError(
            'S0 seed choices depend on controller cell; selection is not static')
    if not np.array_equal(s1_selection, cells['s1c1'].policy_candidate_index):
        raise ValueError(
            'S1 seed choices depend on controller cell; selection is not static')

    return {
        'n_tasks': int(task_indices.size),
        'n_geometry_groups': int(np.unique(fingerprints).size),
        'n_candidates': int(valid.shape[1]),
        'candidate_cache_sha256': candidate_cache_sha256,
        'valid_mask_sha256': _sha256_mask(valid),
        'first_valid_selection_sha256': hashlib.sha256(
            np.ascontiguousarray(first).tobytes()).hexdigest(),
        's0_seed_checkpoint_sha256': cells['s0c0'].seed_checkpoint_sha256,
        's1_seed_checkpoint_sha256': cells['s1c0'].seed_checkpoint_sha256,
        'c0_controller_state_sha256': cells['s0c0'].controller_state_sha256,
        'c1_controller_state_sha256': cells['s0c1'].controller_state_sha256,
    }


def _geometry_group_means(
    values: np.ndarray,
    fingerprints: Sequence[str],
) -> np.ndarray:
    """Reduce row values to one equally weighted mean per geometry."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] != len(fingerprints):
        raise ValueError('values and task fingerprints have inconsistent shapes')
    if not np.isfinite(matrix).all():
        raise ValueError('geometry-grouped values must be finite')
    _, inverse = np.unique(np.asarray(fingerprints), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    count = np.bincount(inverse, minlength=n_groups).astype(np.float64)
    grouped = np.empty((n_groups, matrix.shape[1]), dtype=np.float64)
    for column in range(matrix.shape[1]):
        grouped[:, column] = (
            np.bincount(
                inverse, weights=matrix[:, column], minlength=n_groups)
            / count)
    return grouped


def _bootstrap_ci(
    group_cells: np.ndarray,
    coefficients: Mapping[str, np.ndarray],
    *,
    seed: int,
    samples: int,
    batch_size: int = 512,
) -> dict[str, tuple[float, float]]:
    """Paired percentile intervals from shared geometry resamples."""
    if samples < 1:
        raise ValueError('bootstrap samples must be positive')
    if batch_size < 1:
        raise ValueError('bootstrap batch size must be positive')
    n_groups = group_cells.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, group_cells.shape[1]), dtype=np.float64)
    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        index = rng.integers(
            0, n_groups, size=(end - start, n_groups), endpoint=False)
        draws[start:end] = group_cells[index].mean(axis=1)
    intervals = {}
    for name, coefficient in coefficients.items():
        sampled = draws @ coefficient
        low, high = np.quantile(sampled, (0.025, 0.975))
        intervals[name] = (float(low), float(high))
    return intervals


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    if not 0.0 <= fraction < 0.5:
        raise ValueError('trim fraction must be in [0, 0.5)')
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(math.floor(fraction * ordered.size))
    if 2 * trim >= ordered.size:
        raise ValueError('trim fraction removes every geometry group')
    selected = ordered[trim:ordered.size - trim] if trim else ordered
    return float(selected.mean())


def analyze_joint_2x2(
    cells: Mapping[str, EvaluationCell],
    *,
    bootstrap_seed: int = 73_001,
    bootstrap_samples: int = 20_000,
    trim_fraction: float = 0.05,
    clip_m: float = 0.05,
    harm_threshold_m: float = 0.001,
) -> dict[str, Any]:
    """Compute strict geometry-balanced paired 2x2 statistics."""
    if not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError('bootstrap seed must be a non-negative integer')
    if bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be positive')
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError('trim fraction must be in [0, 0.5)')
    if not math.isfinite(clip_m) or clip_m <= 0.0:
        raise ValueError('clip_m must be finite and positive')
    if not math.isfinite(harm_threshold_m) or harm_threshold_m <= 0.0:
        raise ValueError('harm threshold must be finite and positive')

    audit = validate_2x2(cells)
    fingerprints = cells['s0c0'].task_fingerprints.tolist()
    row_cells = np.column_stack([
        cells[name].policy_progress_m for name in CELL_NAMES])
    group_cells = _geometry_group_means(row_cells, fingerprints)
    coefficients = {
        **{name: np.eye(4, dtype=np.float64)[index]
           for index, name in enumerate(CELL_NAMES)},
        **EFFECT_COEFFICIENTS,
    }
    intervals = _bootstrap_ci(
        group_cells, coefficients, seed=bootstrap_seed,
        samples=bootstrap_samples)

    cell_metrics = {}
    for index, name in enumerate(CELL_NAMES):
        low, high = intervals[name]
        cell_metrics[name] = {
            'row_mean_progress_m': float(row_cells[:, index].mean()),
            'geometry_macro_progress_m': float(group_cells[:, index].mean()),
            'ci95_low_m': low,
            'ci95_high_m': high,
        }

    effect_metrics = {}
    for name, coefficient in EFFECT_COEFFICIENTS.items():
        row_delta = row_cells @ coefficient
        group_delta = group_cells @ coefficient
        clipped_groups = _geometry_group_means(
            np.clip(row_delta, -clip_m, clip_m), fingerprints)[:, 0]
        harm = (row_delta < -harm_threshold_m).astype(np.float64)
        group_harm = _geometry_group_means(harm, fingerprints)[:, 0]
        low, high = intervals[name]
        effect_metrics[name] = {
            'row_mean_delta_m': float(row_delta.mean()),
            'geometry_macro_delta_m': float(group_delta.mean()),
            'ci95_low_m': low,
            'ci95_high_m': high,
            'row_harm_gt_threshold_count': int(harm.sum()),
            'row_harm_gt_threshold_rate': float(harm.mean()),
            'geometry_macro_harm_gt_threshold_rate': float(group_harm.mean()),
            'geometry_group_trimmed_delta_m': _trimmed_mean(
                group_delta, trim_fraction),
            'row_delta_clipped_geometry_macro_m': float(
                clipped_groups.mean()),
        }

    s0 = cells['s0c0'].policy_candidate_index
    s1 = cells['s1c0'].policy_candidate_index
    changed = (s0 != s1).astype(np.float64)
    changed_groups = _geometry_group_means(changed, fingerprints)[:, 0]
    decomposition_residual = (
        effect_metrics['joint']['geometry_macro_delta_m']
        - effect_metrics['controller_at_s0']['geometry_macro_delta_m']
        - effect_metrics['selector_at_c0']['geometry_macro_delta_m']
        - effect_metrics['interaction']['geometry_macro_delta_m'])

    return {
        'format': 'joint-static-seed-controller-2x2-analysis-v1',
        'units': {
            'progress_and_delta': 'metre',
            'rates': 'fraction',
        },
        'estimand': (
            'equal-weight mean over unique task geometries; rows are averaged '
            'inside geometry before macro averaging'),
        'decomposition': (
            'joint = controller_at_s0 + selector_at_c0 + interaction'),
        'audit': audit,
        'settings': {
            'bootstrap': 'paired-geometry-cluster-percentile-v1',
            'bootstrap_seed': bootstrap_seed,
            'bootstrap_samples': bootstrap_samples,
            'trim_fraction_each_tail': trim_fraction,
            'clip_absolute_delta_m': clip_m,
            'harm_threshold_m': harm_threshold_m,
        },
        'inputs': {
            name: {
                'path': str(cells[name].path),
                'size': cells[name].file_size,
                'sha256': cells[name].file_sha256,
            }
            for name in CELL_NAMES
        },
        'cells': cell_metrics,
        'effects': effect_metrics,
        'seed_selection': {
            'changed_rows': int(changed.sum()),
            'changed_row_rate': float(changed.mean()),
            'changed_geometry_groups_any': int((changed_groups > 0.0).sum()),
            'changed_geometry_group_rate_any': float(
                (changed_groups > 0.0).mean()),
            'changed_geometry_macro_rate': float(changed_groups.mean()),
            'unchanged_rows': int(changed.size - changed.sum()),
        },
        'checks': {
            'decomposition_residual_m': float(decomposition_residual),
            'static_single_seed_per_task': True,
            'factor_identities_validated': True,
            'paired_task_order_validated': True,
            'candidate_valid_masks_validated': True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Strict geometry-balanced paired 2x2 analysis of static one-seed '
            'joint selector/controller evaluations.'))
    parser.add_argument('--s0c0', required=True)
    parser.add_argument('--s0c1', required=True)
    parser.add_argument('--s1c0', required=True)
    parser.add_argument('--s1c1', required=True)
    parser.add_argument('--bootstrap-seed', type=int, default=73_001)
    parser.add_argument('--bootstrap-samples', type=int, default=20_000)
    parser.add_argument('--trim-fraction', type=float, default=0.05)
    parser.add_argument('--clip-mm', type=float, default=50.0)
    parser.add_argument('--harm-mm', type=float, default=1.0)
    parser.add_argument(
        '--json-out', default=None,
        help='optional new JSON file; the same JSON is always printed')
    return parser


def main() -> None:
    args = _parser().parse_args()
    cells = {
        name: load_evaluation(getattr(args, name), name)
        for name in CELL_NAMES
    }
    result = analyze_joint_2x2(
        cells,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        trim_fraction=args.trim_fraction,
        clip_m=args.clip_mm / 1000.0,
        harm_threshold_m=args.harm_mm / 1000.0,
    )
    encoded = json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if args.json_out is not None:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
