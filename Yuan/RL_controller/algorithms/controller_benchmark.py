"""Shared protocol helpers for fair controller-training comparisons.

The helpers in this module deliberately know nothing about PPO or FlashSAC.
They fix the evaluation task list once, run deterministic first-episode
rollouts, and write a small JSON artifact containing one row per task.  Both
trainers can therefore share the same evaluation and artifact schema.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


DEFAULT_MILESTONES = (
    0,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    20_000_000,
    30_000_000,
)
TASK_KEYS = ('q0', 'line_dir', 'n_target')
EVAL_SCHEMA = 'controller-fair-eval-v1'


def validate_milestones(
        values: Sequence[int],
        *,
        require_default: bool = False) -> tuple[int, ...]:
    """Return a strictly increasing tuple of non-negative transition budgets."""
    milestones = tuple(int(value) for value in values)
    if not milestones:
        raise ValueError('at least one evaluation milestone is required')
    if milestones[0] != 0:
        raise ValueError('evaluation milestones must start at step 0')
    if any(value < 0 for value in milestones):
        raise ValueError('evaluation milestones must be non-negative')
    if any(right <= left for left, right in zip(milestones, milestones[1:])):
        raise ValueError('evaluation milestones must be strictly increasing')
    if require_default and milestones != DEFAULT_MILESTONES:
        raise ValueError(
            f'published comparison requires milestones {list(DEFAULT_MILESTONES)}')
    return milestones


def synchronize_device(device: torch.device, enabled: bool = True) -> None:
    """Synchronize CUDA timing while remaining a no-op on CPU."""
    if enabled and device.type == 'cuda':
        torch.cuda.synchronize(device)


def canonical_task_specs(
        specs: Mapping[str, torch.Tensor],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Validate and clone fixed task tensors.

    All tensors must have shape ``(N, D)`` with the same ``N``.  The returned
    tensors are contiguous, which makes their SHA-256 fingerprint independent
    of the source tensor's stride.
    """
    missing = sorted(set(TASK_KEYS) - set(specs))
    extra = sorted(set(specs) - set(TASK_KEYS))
    if missing or extra:
        raise ValueError(
            f'fixed task specs need exactly {TASK_KEYS}; '
            f'missing={missing}, extra={extra}')
    count: int | None = None
    result: dict[str, torch.Tensor] = {}
    for key in TASK_KEYS:
        value = torch.as_tensor(specs[key])
        if value.ndim != 2:
            raise ValueError(f'{key} must be rank two, got {tuple(value.shape)}')
        if count is None:
            count = int(value.shape[0])
        elif int(value.shape[0]) != count:
            raise ValueError('fixed task tensors have inconsistent task counts')
        value = value.detach().to(
            device=device if device is not None else value.device,
            dtype=dtype).contiguous()
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f'{key} contains a non-finite value')
        result[key] = value.clone()
    if count is None or count <= 0:
        raise ValueError('fixed task list must be non-empty')
    return result


def task_specs_fingerprint(specs: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over task names, shapes, and little-endian float32 values."""
    canonical = canonical_task_specs(specs, device=torch.device('cpu'))
    digest = hashlib.sha256()
    digest.update(b'controller-fixed-task-specs-v1\0')
    for key in TASK_KEYS:
        array = canonical[key].numpy().astype('<f4', copy=False)
        digest.update(key.encode('utf-8') + b'\0')
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def save_fixed_task_specs(
        path: str | Path,
        specs: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any] | None = None,
        *,
        refuse_overwrite: bool = True) -> str:
    """Save the exact holdout tensors and return their fingerprint."""
    output = Path(path)
    if refuse_overwrite and output.exists():
        raise FileExistsError(f'refusing to overwrite fixed tasks: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical = canonical_task_specs(specs, device=torch.device('cpu'))
    fingerprint = task_specs_fingerprint(canonical)
    torch.save({
        'schema': 'controller-fixed-task-specs-v1',
        'fingerprint': fingerprint,
        'specs': canonical,
        'metadata': dict(metadata or {}),
    }, output)
    return fingerprint


def load_fixed_task_specs(
        path: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load fixed tasks, rejecting corrupt or fingerprint-mismatched files."""
    source = Path(path)
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or payload.get('schema') != (
            'controller-fixed-task-specs-v1'):
        raise ValueError(f'unsupported fixed-task artifact: {source}')
    raw_specs = payload.get('specs')
    if not isinstance(raw_specs, dict):
        raise ValueError(f'fixed-task artifact has no specs mapping: {source}')
    fingerprint = task_specs_fingerprint(raw_specs)
    if fingerprint != payload.get('fingerprint'):
        raise ValueError(f'fixed-task fingerprint mismatch: {source}')
    specs = canonical_task_specs(raw_specs, device=device, dtype=dtype)
    metadata = dict(payload.get('metadata') or {})
    metadata['fingerprint'] = fingerprint
    return specs, metadata


def build_fixed_task_specs(
        eval_env: Any,
        env_config: Any,
        line_config: Mapping[str, Any],
        eval_config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Sample the canonical holdout once from explicit pool/task seeds."""
    from Yuan.RL_controller.env.line_distribution import LineDistribution

    required = ('n_holdout', 'holdout_pool_seed', 'holdout_task_seed')
    missing = [key for key in required if key not in eval_config]
    if missing:
        raise ValueError(f'eval config is missing fixed-task keys: {missing}')
    threshold = (
        float(line_config['feasibility_threshold_m'])
        if line_config.get('feasibility_filter', False) else None)
    sampler = LineDistribution.load_or_build(
        kin=eval_env.kin,
        collision=eval_env.collision,
        n_pool=int(line_config['n_pool']),
        n_target_noise_deg=float(line_config['n_target_noise_deg']),
        seed=int(eval_config['holdout_pool_seed']),
        env_cfg=env_config,
        feasibility_threshold_m=threshold)
    generator = torch.Generator(device=eval_env.device)
    generator.manual_seed(int(eval_config['holdout_task_seed']))
    return canonical_task_specs(
        sampler.sample(int(eval_config['n_holdout']), generator=generator),
        device=eval_env.device,
        dtype=eval_env.kin.dtype)


@torch.no_grad()
def evaluate_fixed_tasks(
        eval_env: Any,
        fixed_specs: Mapping[str, torch.Tensor],
        actor_mean: Callable[[torch.Tensor], torch.Tensor],
        *,
        algorithm: str,
        run_seed: int,
        requested_step: int,
        global_step: int) -> dict[str, Any]:
    """Evaluate a deterministic actor and return aggregate + per-task rows."""
    from Yuan.RL_controller.env.env import TERM_NAMES
    from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
    from Yuan.RL_controller.env.rollout import rollout_first_episode

    specs = canonical_task_specs(
        fixed_specs, device=eval_env.device, dtype=eval_env.kin.dtype)
    if int(specs['q0'].shape[0]) != int(eval_env.n_envs):
        raise ValueError(
            f'eval env has {eval_env.n_envs} slots but fixed task list has '
            f'{specs["q0"].shape[0]} tasks')
    eval_env.line_dist = ScriptedLineDistribution({
        key: value.clone() for key, value in specs.items()})

    def action_fn(environment: Any) -> torch.Tensor:
        action = actor_mean(environment.current_obs())
        if tuple(action.shape) != (environment.n_envs, environment.act_dim):
            raise ValueError(
                f'actor returned {tuple(action.shape)}, expected '
                f'{(environment.n_envs, environment.act_dim)}')
        return action.clamp(-1.0, 1.0)

    stats = rollout_first_episode(eval_env, action_fn)
    progress = stats['episode_progress'].detach().float().cpu()
    episode_length = stats['episode_len'].detach().long().cpu()
    termination = stats['term_reason'].detach().long().cpu()
    if not bool(torch.isfinite(progress).all().item()):
        raise RuntimeError('evaluation produced non-finite progress')
    count = int(progress.numel())
    term_fraction = {
        f'eval_term/{name}': float((termination == code).sum().item()) / count
        for code, name in TERM_NAMES.items()
    }
    per_task = [
        {
            'task_index': index,
            'progress_m': float(progress[index].item()),
            'term_reason': int(termination[index].item()),
            'episode_length': int(episode_length[index].item()),
        }
        for index in range(count)
    ]
    return {
        'schema': EVAL_SCHEMA,
        'algorithm': str(algorithm),
        'run_seed': int(run_seed),
        'requested_step': int(requested_step),
        'global_step': int(global_step),
        'task_fingerprint': task_specs_fingerprint(specs),
        'n_tasks': count,
        'eval/mean_progress_m': float(progress.mean().item()),
        'eval/median_progress_m': float(progress.median().item()),
        'eval/max_progress_m': float(progress.max().item()),
        'eval/mean_episode_length': float(episode_length.float().mean().item()),
        **term_fraction,
        'per_task': per_task,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_ready(value.item())
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(
        path: str | Path,
        payload: Mapping[str, Any],
        *,
        refuse_overwrite: bool = False) -> None:
    """Write strict, deterministic JSON suitable for comparison scripts."""
    output = Path(path)
    if refuse_overwrite and output.exists():
        raise FileExistsError(f'refusing to overwrite artifact: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as stream:
        json.dump(
            _json_ready(payload), stream, ensure_ascii=False,
            indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(
            _json_ready(payload), ensure_ascii=False,
            sort_keys=True, allow_nan=False) + '\n')


def ensure_fresh_outdir(path: str | Path) -> Path:
    """Create an empty run directory and reject accidental result mixing."""
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'fair benchmark requires a fresh output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output
