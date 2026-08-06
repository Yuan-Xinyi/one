"""Compare fresh PPO and FlashSAC runs under the fixed-task protocol.

The script consumes ``eval/eval_step_<requested>.json`` artifacts written by
the fair runners.  It reports:

* transition-normalized AUC (sample efficiency);
* core/e2e time-normalized AUC and conservative time-to-threshold;
* paired final-task deltas with a deterministic paired bootstrap interval.

Example:
    python -m Yuan.RL_controller.algorithms.compare_controller_runs \
        Yuan/RL_controller/runs/ppo_fair_seed0 \
        Yuan/RL_controller/runs/flashsac_fair_seed0 \
        --out Yuan/RL_controller/runs/controller_comparison.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from Yuan.RL_controller.algorithms.controller_benchmark import (
    DEFAULT_MILESTONES,
    EVAL_SCHEMA,
    write_json,
)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite, got {value}')
    return result


def _timing(record: Mapping[str, Any], name: str) -> float:
    candidates = (
        name,
        f'time/{name}',
        name.replace('_s', '_seconds'),
    )
    for key in candidates:
        if key in record and record[key] is not None:
            return _finite_float(record[key], key)
    raise ValueError(f'evaluation artifact is missing timing field {name}')


def _load_eval(path: Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as stream:
        record = json.load(stream)
    if not isinstance(record, dict):
        raise ValueError(f'evaluation artifact is not an object: {path}')
    schema = record.get('schema')
    if schema is not None and schema != EVAL_SCHEMA:
        raise ValueError(f'unsupported evaluation schema {schema}: {path}')
    required = (
        'algorithm', 'run_seed', 'requested_step', 'global_step',
        'eval/mean_progress_m', 'per_task')
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f'{path} is missing fields {missing}')
    rows = record['per_task']
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path} has no per-task rows')
    task_indices = [int(row['task_index']) for row in rows]
    if len(set(task_indices)) != len(task_indices):
        raise ValueError(f'{path} contains duplicate task indices')
    progress = np.asarray(
        [_finite_float(row['progress_m'], 'progress_m') for row in rows],
        dtype=np.float64)
    reported_mean = _finite_float(
        record['eval/mean_progress_m'], 'eval/mean_progress_m')
    if not np.isclose(progress.mean(), reported_mean, atol=1e-6, rtol=1e-6):
        raise ValueError(
            f'{path} aggregate mean does not match its per-task rows')
    record['_path'] = str(path.resolve())
    return record


def load_run(
        path: str | Path,
        *,
        require_published_milestones: bool = True) -> dict[str, Any]:
    """Load and validate one run directory."""
    run_dir = Path(path)
    eval_dir = run_dir / 'eval'
    files = sorted(eval_dir.glob('eval_step_*.json'))
    if not files:
        raise FileNotFoundError(
            f'no eval/eval_step_*.json artifacts in {run_dir}')
    evaluations = [_load_eval(file) for file in files]
    evaluations.sort(key=lambda record: int(record['requested_step']))
    requested = tuple(int(record['requested_step']) for record in evaluations)
    if len(set(requested)) != len(requested):
        raise ValueError(f'duplicate requested milestones in {run_dir}')
    if require_published_milestones and requested != DEFAULT_MILESTONES:
        raise ValueError(
            f'{run_dir} milestones {requested} differ from published '
            f'{DEFAULT_MILESTONES}')
    if requested[0] != 0:
        raise ValueError(f'{run_dir} lacks step-0 evaluation')
    if any(right <= left for left, right in zip(requested, requested[1:])):
        raise ValueError(f'{run_dir} milestones are not strictly increasing')

    algorithms = {str(record['algorithm']).lower() for record in evaluations}
    seeds = {int(record['run_seed']) for record in evaluations}
    fingerprints = {
        str(record['task_fingerprint'])
        for record in evaluations
        if record.get('task_fingerprint')
    }
    if len(algorithms) != 1 or len(seeds) != 1:
        raise ValueError(f'algorithm/run seed changes within {run_dir}')
    if len(fingerprints) > 1:
        raise ValueError(f'holdout fingerprint changes within {run_dir}')
    config: dict[str, Any] = {}
    config_path = run_dir / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
    return {
        'run_dir': str(run_dir.resolve()),
        'algorithm': next(iter(algorithms)),
        'run_seed': next(iter(seeds)),
        'task_fingerprint': (
            next(iter(fingerprints)) if fingerprints else None),
        'evaluations': evaluations,
        'config': config,
    }


def _normalized_auc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError('AUC inputs must be same-length vectors')
    if x.size < 2 or x[-1] <= x[0]:
        return 0.0, float(y[-1])
    if np.any(np.diff(x) < 0.0):
        raise ValueError('AUC x axis must be monotonic')
    area = float(np.trapezoid(y, x))
    return area, area / float(x[-1] - x[0])


def summarize_run(
        run: Mapping[str, Any],
        thresholds_m: Sequence[float]) -> dict[str, Any]:
    evaluations = run['evaluations']
    requested = np.asarray(
        [record['requested_step'] for record in evaluations],
        dtype=np.float64)
    actual = np.asarray(
        [record['global_step'] for record in evaluations],
        dtype=np.float64)
    progress = np.asarray(
        [record['eval/mean_progress_m'] for record in evaluations],
        dtype=np.float64)
    core = np.asarray(
        [_timing(record, 'core_train_s') for record in evaluations],
        dtype=np.float64)
    e2e = np.asarray(
        [_timing(record, 'e2e_s') for record in evaluations],
        dtype=np.float64)
    transition_area, transition_auc = _normalized_auc(requested, progress)
    core_area, core_auc = _normalized_auc(core, progress)
    e2e_area, e2e_auc = _normalized_auc(e2e, progress)

    time_to_threshold: dict[str, Any] = {}
    for raw_threshold in thresholds_m:
        threshold = _finite_float(raw_threshold, 'threshold')
        reached = np.flatnonzero(progress >= threshold)
        key = f'{threshold:.6g}'
        if reached.size == 0:
            time_to_threshold[key] = None
            continue
        index = int(reached[0])
        time_to_threshold[key] = {
            'requested_step': int(requested[index]),
            'global_step': int(actual[index]),
            'core_train_s': float(core[index]),
            'e2e_s': float(e2e[index]),
            'observed_progress_m': float(progress[index]),
        }

    final = evaluations[-1]
    return {
        'run_dir': run['run_dir'],
        'algorithm': run['algorithm'],
        'run_seed': int(run['run_seed']),
        'task_fingerprint': run['task_fingerprint'],
        'n_tasks': len(final['per_task']),
        'final_requested_step': int(requested[-1]),
        'final_global_step': int(actual[-1]),
        'final_mean_progress_m': float(progress[-1]),
        'final_median_progress_m': _finite_float(
            final.get('eval/median_progress_m', np.median([
                row['progress_m'] for row in final['per_task']])),
            'eval/median_progress_m'),
        'transition_auc_m_steps': transition_area,
        'transition_auc_normalized_m': transition_auc,
        'core_time_auc_m_s': core_area,
        'core_time_auc_normalized_m': core_auc,
        'e2e_time_auc_m_s': e2e_area,
        'e2e_time_auc_normalized_m': e2e_auc,
        'final_core_train_s': float(core[-1]),
        'final_e2e_s': float(e2e[-1]),
        'time_to_threshold': time_to_threshold,
        'curve': [
            {
                'requested_step': int(requested[index]),
                'global_step': int(actual[index]),
                'mean_progress_m': float(progress[index]),
                'core_train_s': float(core[index]),
                'e2e_s': float(e2e[index]),
            }
            for index in range(requested.size)
        ],
    }


def _bootstrap_mean_interval(
        values: np.ndarray,
        *,
        samples: int,
        seed: int) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError('bootstrap sample count must be positive')
    rng = np.random.default_rng(seed)
    count = int(values.size)
    means = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(samples, 4096))
    offset = 0
    while offset < samples:
        size = min(chunk_size, samples - offset)
        indices = rng.integers(0, count, size=(size, count))
        means[offset:offset + size] = values[indices].mean(axis=1)
        offset += size
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _eval_at_step(run: Mapping[str, Any], requested_step: int) -> dict[str, Any]:
    for record in run['evaluations']:
        if int(record['requested_step']) == int(requested_step):
            return record
    raise KeyError(f'run has no requested step {requested_step}')


def paired_final_result(
        baseline: Mapping[str, Any],
        candidate: Mapping[str, Any],
        *,
        bootstrap_samples: int,
        bootstrap_seed: int) -> dict[str, Any]:
    """Task-paired final comparison for a matching run seed."""
    if int(baseline['run_seed']) != int(candidate['run_seed']):
        raise ValueError('paired runs must have the same run seed')
    baseline_fingerprint = baseline.get('task_fingerprint')
    candidate_fingerprint = candidate.get('task_fingerprint')
    if not baseline_fingerprint or not candidate_fingerprint:
        raise ValueError('paired comparison requires task fingerprints')
    if baseline_fingerprint != candidate_fingerprint:
        raise ValueError('paired runs use different fixed task lists')
    common_steps = sorted(
        {int(record['requested_step']) for record in baseline['evaluations']}
        & {int(record['requested_step']) for record in candidate['evaluations']})
    if not common_steps:
        raise ValueError('paired runs have no common evaluation milestone')
    final_step = common_steps[-1]
    base_eval = _eval_at_step(baseline, final_step)
    candidate_eval = _eval_at_step(candidate, final_step)
    base_rows = {
        int(row['task_index']): row for row in base_eval['per_task']}
    candidate_rows = {
        int(row['task_index']): row for row in candidate_eval['per_task']}
    if set(base_rows) != set(candidate_rows):
        raise ValueError('paired runs contain different task indices')
    task_indices = sorted(base_rows)
    base_progress = np.asarray(
        [base_rows[index]['progress_m'] for index in task_indices],
        dtype=np.float64)
    candidate_progress = np.asarray(
        [candidate_rows[index]['progress_m'] for index in task_indices],
        dtype=np.float64)
    delta = candidate_progress - base_progress
    low, high = _bootstrap_mean_interval(
        delta, samples=bootstrap_samples, seed=bootstrap_seed)
    tolerance = 1e-9
    return {
        'run_seed': int(baseline['run_seed']),
        'baseline_algorithm': baseline['algorithm'],
        'candidate_algorithm': candidate['algorithm'],
        'requested_step': int(final_step),
        'baseline_global_step': int(base_eval['global_step']),
        'candidate_global_step': int(candidate_eval['global_step']),
        'task_fingerprint': baseline_fingerprint,
        'n_tasks': len(task_indices),
        'baseline_mean_progress_m': float(base_progress.mean()),
        'candidate_mean_progress_m': float(candidate_progress.mean()),
        'paired_mean_delta_m': float(delta.mean()),
        'paired_median_delta_m': float(np.median(delta)),
        'paired_bootstrap_95ci_m': [low, high],
        'candidate_win_fraction': float((delta > tolerance).mean()),
        'tie_fraction': float((np.abs(delta) <= tolerance).mean()),
        'candidate_loss_fraction': float((delta < -tolerance).mean()),
    }


def compare_runs(
        runs: Sequence[Mapping[str, Any]],
        *,
        thresholds_m: Sequence[float],
        baseline_algorithm: str = 'ppo',
        candidate_algorithm: str = 'flashsac',
        bootstrap_samples: int = 10_000,
        bootstrap_seed: int = 20_260_729,
        require_pairs: bool = True) -> dict[str, Any]:
    summaries = [summarize_run(run, thresholds_m) for run in runs]
    baseline_name = baseline_algorithm.lower()
    candidate_name = candidate_algorithm.lower()
    baselines = {
        int(run['run_seed']): run for run in runs
        if run['algorithm'] == baseline_name}
    candidates = {
        int(run['run_seed']): run for run in runs
        if run['algorithm'] == candidate_name}
    common_seeds = sorted(set(baselines) & set(candidates))
    if require_pairs and not common_seeds:
        raise ValueError(
            f'no matching run_seed for {baseline_name} and {candidate_name}')
    paired = [
        paired_final_result(
            baselines[seed],
            candidates[seed],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + pair_index)
        for pair_index, seed in enumerate(common_seeds)
    ]
    aggregate: dict[str, Any] | None = None
    if paired:
        deltas = np.asarray(
            [record['paired_mean_delta_m'] for record in paired],
            dtype=np.float64)
        aggregate = {
            'n_seed_pairs': len(paired),
            'run_seeds': common_seeds,
            'mean_of_seed_paired_deltas_m': float(deltas.mean()),
            'min_seed_paired_delta_m': float(deltas.min()),
            'max_seed_paired_delta_m': float(deltas.max()),
        }
    return {
        'schema': 'controller-fair-comparison-v1',
        'baseline_algorithm': baseline_name,
        'candidate_algorithm': candidate_name,
        'progress_thresholds_m': [float(value) for value in thresholds_m],
        'bootstrap_samples': int(bootstrap_samples),
        'bootstrap_seed': int(bootstrap_seed),
        'runs': summaries,
        'paired_final': paired,
        'paired_seed_aggregate': aggregate,
    }


def _format_seconds(seconds: float) -> str:
    return f'{seconds / 3600.0:.3f} h'


def comparison_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        '# PPO vs FlashSAC 公平对比',
        '',
        'AUC 使用共同的 nominal transition milestones；time-to-threshold '
        '采用首次观测到达阈值的 checkpoint（不做乐观插值）。',
        '',
        '| 算法 | seed | 最终进度 (m) | transition AUC (m) | '
        'core 时间 | e2e 时间 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for run in result['runs']:
        lines.append(
            f"| {run['algorithm']} | {run['run_seed']} | "
            f"{run['final_mean_progress_m']:.6f} | "
            f"{run['transition_auc_normalized_m']:.6f} | "
            f"{_format_seconds(run['final_core_train_s'])} | "
            f"{_format_seconds(run['final_e2e_s'])} |")
    lines.extend([
        '',
        '## 最终逐任务配对结果',
        '',
        '| seed | baseline (m) | candidate (m) | Δ (mm) | '
        '95% CI (mm) | win rate |',
        '|---:|---:|---:|---:|---:|---:|',
    ])
    for pair in result['paired_final']:
        low, high = pair['paired_bootstrap_95ci_m']
        lines.append(
            f"| {pair['run_seed']} | "
            f"{pair['baseline_mean_progress_m']:.6f} | "
            f"{pair['candidate_mean_progress_m']:.6f} | "
            f"{1000.0 * pair['paired_mean_delta_m']:+.3f} | "
            f"[{1000.0 * low:+.3f}, {1000.0 * high:+.3f}] | "
            f"{100.0 * pair['candidate_win_fraction']:.1f}% |")
    lines.extend([
        '',
        '## Time-to-threshold',
        '',
        '| 算法 | seed | threshold (m) | transitions | core | e2e |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for run in result['runs']:
        for threshold, reached in run['time_to_threshold'].items():
            if reached is None:
                lines.append(
                    f"| {run['algorithm']} | {run['run_seed']} | "
                    f"{threshold} | 未达到 | — | — |")
            else:
                lines.append(
                    f"| {run['algorithm']} | {run['run_seed']} | "
                    f"{threshold} | {reached['requested_step']} | "
                    f"{_format_seconds(reached['core_train_s'])} | "
                    f"{_format_seconds(reached['e2e_s'])} |")
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', nargs='+')
    parser.add_argument('--baseline', default='ppo')
    parser.add_argument('--candidate', default='flashsac')
    parser.add_argument(
        '--thresholds', type=float, nargs='+',
        default=[0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument('--bootstrap-samples', type=int, default=10_000)
    parser.add_argument('--bootstrap-seed', type=int, default=20_260_729)
    parser.add_argument(
        '--allow-incomplete', action='store_true',
        help='allow debug runs without the published ten milestones')
    parser.add_argument(
        '--out', default='controller_comparison.json',
        help='comparison JSON path; Markdown is written beside it')
    args = parser.parse_args()

    runs = [
        load_run(
            path,
            require_published_milestones=not args.allow_incomplete)
        for path in args.run_dirs
    ]
    result = compare_runs(
        runs,
        thresholds_m=args.thresholds,
        baseline_algorithm=args.baseline,
        candidate_algorithm=args.candidate,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed)
    output = Path(args.out)
    write_json(output, result)
    markdown_path = output.with_suffix('.md')
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with open(markdown_path, 'w', encoding='utf-8') as stream:
        stream.write(comparison_markdown(result))
    print(f'[compare-controller] JSON: {output}')
    print(f'[compare-controller] Markdown: {markdown_path}')


if __name__ == '__main__':
    main()
