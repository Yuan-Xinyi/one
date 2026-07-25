"""Strict paired analysis of one static selector under two controllers.

The estimand is ``C1 - C0``.  Task geometries receive equal macro weight and
the confidence interval resamples complete geometry groups, preserving the
pairing between the two controller evaluations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from Yuan.unified_rl.analyze_joint_2x2 import (
    EvaluationCell,
    _bootstrap_ci,
    _geometry_group_means,
    _sha256_file,
    _sha256_mask,
    _trimmed_mean,
    load_evaluation,
)


PAIR_NAMES = ('c0', 'c1')


@dataclass(frozen=True)
class ControllerEvaluation:
    """Strict base evaluation plus optional full-oracle task vectors."""

    cell: EvaluationCell
    first_valid_progress_m: np.ndarray | None
    best_progress_m: np.ndarray | None
    policy_episode_len: np.ndarray | None

    @property
    def has_full_oracle(self) -> bool:
        return self.best_progress_m is not None


def _optional_vector(
    data: Mapping[str, np.ndarray],
    key: str,
    n: int,
    *,
    integer: bool = False,
) -> np.ndarray | None:
    if key not in data:
        return None
    value = data[key]
    expected = np.dtype(np.int64) if integer else None
    if (value.shape != (n,)
            or (integer and value.dtype != expected)
            or (not integer and value.dtype not in (
                np.dtype(np.float32), np.dtype(np.float64)))):
        kind = 'int64' if integer else 'float32/float64'
        raise ValueError(f'{key} must be a {kind} task vector')
    result = np.array(value, copy=True)
    if integer:
        if (result < 0).any():
            raise ValueError(f'{key} must be non-negative')
    elif not np.isfinite(result).all():
        raise ValueError(f'{key} must be finite')
    return result


def load_controller_evaluation(
    path: str | Path,
    name: str,
) -> ControllerEvaluation:
    """Load one controller cell through the strict 2x2 artifact loader."""
    if name not in PAIR_NAMES:
        raise ValueError(f'unknown controller-pair cell name: {name!r}')
    # Reusing the 2x2 loader keeps all provenance, mask, index and static-probe
    # checks identical.  Its names are semantic labels only.
    base = load_evaluation(path, 's0c0' if name == 'c0' else 's0c1')
    with np.load(base.path, allow_pickle=False) as data:
        n = int(base.task_indices.size)
        first = _optional_vector(data, 'first_valid_progress_m', n)
        best = _optional_vector(data, 'best_progress_m', n)
        episode_len = _optional_vector(
            data, 'policy_episode_len', n, integer=True)
    if _sha256_file(base.path) != base.file_sha256:
        raise RuntimeError(
            f'evaluation changed while loading optional fields: {base.path}')

    if best is not None:
        missing = [
            key for key, value in (
                ('first_valid_progress_m', first),
                ('policy_episode_len', episode_len),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                'full-oracle evaluation is missing ' + ', '.join(missing))
        assert first is not None and episode_len is not None
        tolerance = 1e-7
        if (best + tolerance < base.policy_progress_m).any():
            raise ValueError('best_progress_m is below policy_progress_m')
        if (best + tolerance < first).any():
            raise ValueError('best_progress_m is below first_valid_progress_m')

    return ControllerEvaluation(
        cell=base,
        first_valid_progress_m=first,
        best_progress_m=best,
        policy_episode_len=episode_len,
    )


def validate_controller_pair(
    evaluations: Mapping[str, ControllerEvaluation],
) -> dict[str, Any]:
    """Fail closed unless the files differ only in controller identity/output."""
    if set(evaluations) != set(PAIR_NAMES):
        raise ValueError(f'controller cells must be exactly {PAIR_NAMES}')
    c0 = evaluations['c0'].cell
    c1 = evaluations['c1'].cell
    for attribute in ('task_indices', 'task_fingerprints', 'valid',
                      'first_valid_candidate_index',
                      'policy_candidate_index'):
        if not np.array_equal(getattr(c0, attribute), getattr(c1, attribute)):
            raise ValueError(
                f'controller cells disagree on {attribute}; '
                'the paired/static-selector contract is invalid')
    if c0.candidate_cache_sha256 != c1.candidate_cache_sha256:
        raise ValueError('controller cells disagree on candidate_cache_sha256')
    if c0.seed_checkpoint_sha256 != c1.seed_checkpoint_sha256:
        raise ValueError('controller cells disagree on seed_checkpoint_sha256')
    if c0.controller_state_sha256 == c1.controller_state_sha256:
        raise ValueError('C0 and C1 controller state identities are identical')

    return {
        'n_tasks': int(c0.task_indices.size),
        'n_geometry_groups': int(np.unique(c0.task_fingerprints).size),
        'n_candidates': int(c0.valid.shape[1]),
        'candidate_cache_sha256': c0.candidate_cache_sha256,
        'valid_mask_sha256': _sha256_mask(c0.valid),
        'seed_checkpoint_sha256': c0.seed_checkpoint_sha256,
        'static_seed_selection_sha256': _sha256_mask(
            np.ascontiguousarray(c0.policy_candidate_index)),
        'c0_controller_agent_sha256': c0.controller_agent_sha256,
        'c0_controller_config_sha256': c0.controller_config_sha256,
        'c0_controller_state_sha256': c0.controller_state_sha256,
        'c1_controller_agent_sha256': c1.controller_agent_sha256,
        'c1_controller_config_sha256': c1.controller_config_sha256,
        'c1_controller_state_sha256': c1.controller_state_sha256,
        'full_oracle_available': {
            name: evaluations[name].has_full_oracle for name in PAIR_NAMES
        },
    }


def _full_oracle_metrics(
    evaluation: ControllerEvaluation,
    fingerprints: np.ndarray,
) -> dict[str, Any]:
    first = evaluation.first_valid_progress_m
    oracle = evaluation.best_progress_m
    episode_len = evaluation.policy_episode_len
    assert first is not None and oracle is not None and episode_len is not None
    policy = evaluation.cell.policy_progress_m
    row_gain = float((policy - first).mean())
    row_headroom = float((oracle - first).mean())
    grouped = _geometry_group_means(
        np.column_stack((policy, first, oracle, episode_len)), fingerprints)
    macro_gain = float((grouped[:, 0] - grouped[:, 1]).mean())
    macro_headroom = float((grouped[:, 2] - grouped[:, 1]).mean())
    return {
        'row_mean': {
            'policy_progress_m': float(policy.mean()),
            'first_valid_progress_m': float(first.mean()),
            'oracle_progress_m': float(oracle.mean()),
            'capture': row_gain / max(row_headroom, 1e-8),
            'policy_episode_len_steps': float(episode_len.mean()),
        },
        'geometry_macro': {
            'policy_progress_m': float(grouped[:, 0].mean()),
            'first_valid_progress_m': float(grouped[:, 1].mean()),
            'oracle_progress_m': float(grouped[:, 2].mean()),
            'capture': macro_gain / max(macro_headroom, 1e-8),
            'policy_episode_len_steps': float(grouped[:, 3].mean()),
        },
    }


def analyze_controller_pair(
    evaluations: Mapping[str, ControllerEvaluation],
    *,
    bootstrap_seed: int = 73_101,
    bootstrap_samples: int = 20_000,
    trim_fraction: float = 0.05,
    clip_m: float = 0.05,
    threshold_m: float = 0.001,
) -> dict[str, Any]:
    """Compute paired ``C1 - C0`` statistics for one static selector."""
    if not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError('bootstrap seed must be a non-negative integer')
    if bootstrap_samples < 1:
        raise ValueError('bootstrap samples must be positive')
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError('trim fraction must be in [0, 0.5)')
    if not math.isfinite(clip_m) or clip_m <= 0.0:
        raise ValueError('clip_m must be finite and positive')
    if not math.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError('threshold_m must be finite and positive')

    audit = validate_controller_pair(evaluations)
    fingerprints = evaluations['c0'].cell.task_fingerprints
    row_cells = np.column_stack([
        evaluations[name].cell.policy_progress_m for name in PAIR_NAMES
    ])
    group_cells = _geometry_group_means(row_cells, fingerprints)
    coefficient = np.asarray([-1.0, 1.0], dtype=np.float64)
    low, high = _bootstrap_ci(
        group_cells, {'controller_delta': coefficient},
        seed=bootstrap_seed, samples=bootstrap_samples)['controller_delta']
    row_delta = row_cells @ coefficient
    group_delta = group_cells @ coefficient
    row_win = row_delta > threshold_m
    row_harm = row_delta < -threshold_m
    grouped_rates = _geometry_group_means(
        np.column_stack((row_win, row_harm)), fingerprints)
    clipped_groups = _geometry_group_means(
        np.clip(row_delta, -clip_m, clip_m), fingerprints)[:, 0]

    result: dict[str, Any] = {
        'format': 'static-selector-controller-pair-analysis-v1',
        'comparison': 'c1_minus_c0',
        'units': {
            'progress_and_delta': 'metre',
            'episode_length': 'controller steps',
            'rates': 'fraction',
        },
        'estimand': (
            'equal-weight mean over unique task geometries; rows are averaged '
            'inside geometry before macro averaging'),
        'audit': audit,
        'settings': {
            'bootstrap': 'paired-geometry-cluster-percentile-v1',
            'bootstrap_seed': bootstrap_seed,
            'bootstrap_samples': bootstrap_samples,
            'trim_fraction_each_tail': trim_fraction,
            'clip_absolute_delta_m': clip_m,
            'win_harm_threshold_m': threshold_m,
        },
        'inputs': {
            name: {
                'path': str(evaluations[name].cell.path),
                'size': evaluations[name].cell.file_size,
                'sha256': evaluations[name].cell.file_sha256,
            }
            for name in PAIR_NAMES
        },
        'cells': {
            name: {
                'row_mean_policy_progress_m': float(
                    row_cells[:, index].mean()),
                'geometry_macro_policy_progress_m': float(
                    group_cells[:, index].mean()),
            }
            for index, name in enumerate(PAIR_NAMES)
        },
        'delta': {
            'row_mean_delta_m': float(row_delta.mean()),
            'geometry_macro_delta_m': float(group_delta.mean()),
            'geometry_macro_ci95_low_m': low,
            'geometry_macro_ci95_high_m': high,
            'row_win_gt_threshold_count': int(row_win.sum()),
            'row_win_gt_threshold_rate': float(row_win.mean()),
            'geometry_macro_win_gt_threshold_rate': float(
                grouped_rates[:, 0].mean()),
            'row_harm_gt_threshold_count': int(row_harm.sum()),
            'row_harm_gt_threshold_rate': float(row_harm.mean()),
            'geometry_macro_harm_gt_threshold_rate': float(
                grouped_rates[:, 1].mean()),
            'geometry_group_trimmed_delta_m': _trimmed_mean(
                group_delta, trim_fraction),
            'row_delta_clipped_geometry_macro_m': float(
                clipped_groups.mean()),
        },
        'checks': {
            'static_single_seed_per_task': True,
            'same_selector_checkpoint_validated': True,
            'different_controller_state_validated': True,
            'paired_task_order_validated': True,
            'candidate_valid_masks_validated': True,
        },
    }
    if all(evaluations[name].has_full_oracle for name in PAIR_NAMES):
        result['full_oracle'] = {
            name: _full_oracle_metrics(evaluations[name], fingerprints)
            for name in PAIR_NAMES
        }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Strict paired analysis of one static selector under C0 and C1.'))
    parser.add_argument('--c0', required=True)
    parser.add_argument('--c1', required=True)
    parser.add_argument('--bootstrap-seed', type=int, default=73_101)
    parser.add_argument('--bootstrap-samples', type=int, default=20_000)
    parser.add_argument('--trim-fraction', type=float, default=0.05)
    parser.add_argument('--clip-mm', type=float, default=50.0)
    parser.add_argument('--threshold-mm', type=float, default=1.0)
    parser.add_argument(
        '--json-out', default=None,
        help='optional new JSON file; the same JSON is always printed')
    return parser


def main() -> None:
    args = _parser().parse_args()
    evaluations = {
        name: load_controller_evaluation(getattr(args, name), name)
        for name in PAIR_NAMES
    }
    result = analyze_controller_pair(
        evaluations,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        trim_fraction=args.trim_fraction,
        clip_m=args.clip_mm / 1000.0,
        threshold_m=args.threshold_mm / 1000.0,
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
