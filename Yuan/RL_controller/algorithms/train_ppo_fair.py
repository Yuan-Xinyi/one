"""Fresh PPO runner for the PPO-versus-FlashSAC comparison.

This runner intentionally has no resume/checkpoint input.  It creates a new
PPO actor, critic, optimizer, and reward normalizer, then evaluates the same
fixed holdout at the common transition milestones:

    0, 0.1M, 0.25M, 0.5M, 1M, 2M, 5M, 10M, 20M, 30M.

PPO itself is not copied or forked.  Milestone evaluation is injected through
``ppo.train``'s existing ``log_fn`` callback, so there is one uninterrupted
optimizer and learning-rate schedule.  Since PPO only changes policy after a
complete ``n_envs * n_steps`` rollout, each artifact records both its requested
budget and the first attainable post-update ``global_step``.

Usage:
    python -m Yuan.RL_controller.algorithms.train_ppo_fair \
        --config Yuan/RL_controller/config_ppo_fair.yaml \
        --out-dir Yuan/RL_controller/runs/ppo_fair_seed0
"""
from __future__ import annotations

import os
import sys

import argparse
import copy
import dataclasses
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from Yuan.RL_controller.algorithms.controller_benchmark import (
    DEFAULT_MILESTONES,
    append_jsonl,
    build_fixed_task_specs,
    ensure_fresh_outdir,
    evaluate_fixed_tasks,
    load_fixed_task_specs,
    save_fixed_task_specs,
    synchronize_device,
    task_specs_fingerprint,
    validate_milestones,
    write_json,
)
from Yuan.RL_controller.algorithms.ppo import (
    Agent,
    PPOConfig,
    RewardScaler,
    train as ppo_train,
)


Evaluator = Callable[[Agent, int, int], dict[str, Any]]


def _relaunch_with_conda_lib() -> None:
    """Apply the controller entry points' libstdc++ workaround."""
    conda_lib = os.path.join(sys.prefix, 'lib')
    if conda_lib in os.environ.get('LD_LIBRARY_PATH', ''):
        return
    new_env = dict(os.environ)
    new_env['LD_LIBRARY_PATH'] = (
        conda_lib + ':' + new_env.get('LD_LIBRARY_PATH', ''))
    if __spec__ is not None and __spec__.name != '__main__':
        argv = [sys.executable, '-m', __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)


def _checkpoint_payload(
        agent: Agent,
        optimizer: torch.optim.Optimizer,
        reward_scaler: RewardScaler | None,
        *,
        run_seed: int,
        requested_step: int,
        global_step: int,
        effective_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'schema': 'controller-fair-ppo-checkpoint-v1',
        'algorithm': 'ppo',
        'initialization': 'fresh_random',
        'run_seed': int(run_seed),
        'requested_step': int(requested_step),
        'global_step': int(global_step),
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward_scaler_state_dict': (
            reward_scaler.state_dict() if reward_scaler is not None else None),
        'effective_config': copy.deepcopy(dict(effective_config)),
    }


def run_fair_ppo(
        *,
        env: Any,
        agent: Agent,
        optimizer: torch.optim.Optimizer,
        reward_scaler: RewardScaler | None,
        ppo_config: PPOConfig,
        evaluator: Evaluator,
        out_dir: str | Path,
        run_seed: int,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        effective_config: Mapping[str, Any] | None = None,
        setup_seconds: float = 0.0,
        initial_save_seconds: float = 0.0,
        process_start: float | None = None,
        synchronize_timing: bool = True,
        save_checkpoints: bool = True) -> dict[str, Any]:
    """Run one uninterrupted fresh PPO optimization with milestone callbacks.

    ``ppo_config.total_timesteps`` must be at least the final requested
    milestone.  Callers should round it up to a full PPO rollout when an exact
    budget is not divisible by ``n_envs * n_steps``.
    """
    milestones = validate_milestones(milestones)
    if int(ppo_config.total_timesteps) < milestones[-1]:
        raise ValueError(
            'PPO total_timesteps is below the final evaluation milestone')
    if process_start is None:
        process_start = time.perf_counter()
    output = Path(out_dir)
    eval_dir = output / 'eval'
    checkpoint_dir = output / 'checkpoints'
    eval_dir.mkdir(parents=True, exist_ok=True)
    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timing = {
        'setup_s': float(setup_seconds),
        'core_train_s': 0.0,
        'eval_s': 0.0,
        'save_s': float(initial_save_seconds),
    }
    pending = list(milestones)
    captured: list[dict[str, Any]] = []
    core_active = False
    core_segment_start = time.perf_counter()

    def capture(requested_step: int, global_step: int) -> None:
        nonlocal core_segment_start
        if core_active:
            synchronize_device(env.device, synchronize_timing)
            now = time.perf_counter()
            timing['core_train_s'] += now - core_segment_start

        synchronize_device(env.device, synchronize_timing)
        eval_start = time.perf_counter()
        record = evaluator(agent, requested_step, global_step)
        synchronize_device(env.device, synchronize_timing)
        timing['eval_s'] += time.perf_counter() - eval_start

        if save_checkpoints:
            save_start = time.perf_counter()
            torch.save(
                _checkpoint_payload(
                    agent, optimizer, reward_scaler,
                    run_seed=run_seed,
                    requested_step=requested_step,
                    global_step=global_step,
                    effective_config=effective_config or {}),
                checkpoint_dir / f'ppo_step_{requested_step}.pt')
            synchronize_device(env.device, synchronize_timing)
            timing['save_s'] += time.perf_counter() - save_start

        record.update({
            'algorithm': 'ppo',
            'run_seed': int(run_seed),
            'initialization': 'fresh_random',
            'requested_step': int(requested_step),
            'global_step': int(global_step),
            'setup_s': timing['setup_s'],
            'core_train_s': timing['core_train_s'],
            'eval_s': timing['eval_s'],
            'save_s': timing['save_s'],
            'e2e_s': time.perf_counter() - process_start,
        })
        artifact = eval_dir / f'eval_step_{requested_step}.json'
        write_json(artifact, record, refuse_overwrite=True)
        append_jsonl(eval_dir / 'eval_metrics.jsonl', {
            key: value for key, value in record.items() if key != 'per_task'
        })
        captured.append(record)
        if core_active:
            core_segment_start = time.perf_counter()

    if not pending or pending.pop(0) != 0:
        raise AssertionError('validated milestones unexpectedly lack step 0')
    capture(0, 0)

    core_active = True
    core_segment_start = time.perf_counter()

    def log_fn(payload: dict[str, Any]) -> None:
        nonlocal core_segment_start
        record = {
            'algorithm': 'ppo',
            'run_seed': int(run_seed),
            **payload,
        }
        append_jsonl(output / 'train_metrics.jsonl', record)
        if 'update' not in payload:
            return
        actual_step = int(payload['global_step'])
        while pending and actual_step >= pending[0]:
            requested_step = pending.pop(0)
            capture(requested_step, actual_step)

    train_error: BaseException | None = None
    try:
        ppo_train(
            ppo_config,
            env,
            device=env.device,
            eval_fn=None,
            log_fn=log_fn,
            ckpt_path=None,
            agent=agent,
            optimizer=optimizer,
            reward_scaler=reward_scaler)
    except BaseException as error:
        train_error = error
    finally:
        synchronize_device(env.device, synchronize_timing)
        timing['core_train_s'] += time.perf_counter() - core_segment_start

    if train_error is not None:
        raise train_error
    if pending:
        raise RuntimeError(
            f'PPO ended before evaluation milestones {pending}; '
            f'total_timesteps={ppo_config.total_timesteps}')

    final_global_step = int(captured[-1]['global_step'])
    save_start = time.perf_counter()
    torch.save(
        _checkpoint_payload(
            agent, optimizer, reward_scaler,
            run_seed=run_seed,
            requested_step=milestones[-1],
            global_step=final_global_step,
            effective_config=effective_config or {}),
        output / 'ppo_final.pt')
    synchronize_device(env.device, synchronize_timing)
    timing['save_s'] += time.perf_counter() - save_start

    summary = {
        'schema': 'controller-fair-run-summary-v1',
        'algorithm': 'ppo',
        'initialization': 'fresh_random',
        'run_seed': int(run_seed),
        'requested_total_steps': int(milestones[-1]),
        'global_step': final_global_step,
        'rollout_overshoot_steps': final_global_step - int(milestones[-1]),
        'n_evaluations': len(captured),
        **timing,
        'e2e_s': time.perf_counter() - process_start,
        'final_eval': {
            key: value for key, value in captured[-1].items()
            if key.startswith('eval/')
        },
    }
    write_json(output / 'summary.json', summary, refuse_overwrite=True)
    append_jsonl(output / 'train_metrics.jsonl', {
        'training_complete': True,
        **summary,
    })
    return summary


def _effective_env_config(values: Mapping[str, Any]) -> dict[str, Any]:
    from Yuan.RL_controller.env.env import EnvConfig

    known = {field.name for field in dataclasses.fields(EnvConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f'unknown environment config keys: {unknown}')
    result = dict(values)
    if not bool(result.get('observe_ray_error', False)):
        raise ValueError(
            'fair controller benchmark requires observe_ray_error=true (34-D)')
    return result


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')


def main() -> None:
    _relaunch_with_conda_lib()
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='Yuan/RL_controller/config_ppo_fair.yaml')
    parser.add_argument(
        '--out-dir', default='Yuan/RL_controller/runs/ppo_fair_seed0')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--fixed-task-specs', default=None,
        help='optional canonical fixed_tasks.pt shared with FlashSAC')
    parser.add_argument(
        '--max-env-steps', type=int, default=None,
        help='debug-only shortened budget; published milestones stay unchanged')
    parser.add_argument(
        '--disable-lr-anneal', action='store_true',
        help=(
            'pilot-only: keep the initial LR over a shortened run, matching '
            'the early segment of the 30M schedule more closely'))
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError('PPO fair YAML root must be a mapping')
    for section in (
            'env', 'ppo', 'line_distribution', 'eval', 'benchmark'):
        if not isinstance(values.get(section), dict):
            raise ValueError(f'PPO fair YAML needs mapping section {section}')

    output = ensure_fresh_outdir(args.out_dir)
    seed = int(values.get('seed', 0))
    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    _seed_everything(seed, device)

    benchmark_values = values['benchmark']
    configured_milestones = validate_milestones(
        benchmark_values['milestones'],
        require_default=bool(
            benchmark_values.get('require_published_milestones', True)))
    if args.max_env_steps is None:
        milestones = configured_milestones
        requested_total = configured_milestones[-1]
    else:
        if args.max_env_steps <= 0:
            raise ValueError('--max-env-steps must be positive')
        milestones = tuple(
            value for value in configured_milestones
            if value <= args.max_env_steps)
        if not milestones or milestones[0] != 0:
            milestones = (0,)
        if milestones[-1] != args.max_env_steps:
            milestones = (*milestones, int(args.max_env_steps))
        milestones = validate_milestones(milestones)
        requested_total = int(args.max_env_steps)

    from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv

    env_values = _effective_env_config(values['env'])
    env_config = EnvConfig(**env_values)
    eval_values = values['eval']
    eval_config = EnvConfig(**{
        **env_values,
        'n_envs': int(eval_values['n_holdout']),
    })

    line_values = values['line_distribution']
    threshold = (
        float(line_values['feasibility_threshold_m'])
        if line_values.get('feasibility_filter', False) else None)
    from Yuan.RL_controller.env.line_distribution import LineDistribution

    print(
        f'[ppo-fair] device={device} n_envs={env_config.n_envs} '
        f'fresh initialization seed={seed}')
    train_env = NSRLBatchedEnv(
        env_config, line_dist=None, device=device)
    train_env.line_dist = LineDistribution.load_or_build(
        kin=train_env.kin,
        collision=train_env.collision,
        n_pool=int(line_values['n_pool']),
        n_target_noise_deg=float(line_values['n_target_noise_deg']),
        seed=int(line_values['train_seed']),
        env_cfg=env_config,
        feasibility_threshold_m=threshold)
    if 'train_task_seed' not in line_values:
        raise ValueError(
            'fair benchmark requires line_distribution.train_task_seed')
    train_task_seed = int(line_values['train_task_seed']) + seed
    # load_or_build otherwise leaves a different sampler RNG state on cache
    # hit versus cache miss.  An explicit post-load seed makes the task stream
    # reproducible and gives PPO/FlashSAC the same distribution for each run.
    train_env.line_dist._gen.manual_seed(train_task_seed)
    eval_env = NSRLBatchedEnv(
        eval_config, line_dist=None, device=device)

    if args.fixed_task_specs is not None:
        fixed_specs, fixed_metadata = load_fixed_task_specs(
            args.fixed_task_specs, device=device, dtype=eval_env.kin.dtype)
        if int(fixed_specs['q0'].shape[0]) != eval_config.n_envs:
            raise ValueError(
                'fixed task artifact count differs from eval.n_holdout')
    else:
        fixed_specs = build_fixed_task_specs(
            eval_env, eval_config, line_values, eval_values)
        fixed_metadata = {}
    fingerprint = task_specs_fingerprint(fixed_specs)

    ppo_values = copy.deepcopy(values['ppo'])
    if args.disable_lr_anneal:
        ppo_values['anneal_lr'] = False
    target_batch = env_config.n_envs * int(ppo_values['n_steps'])
    internal_total = int(
        math.ceil(requested_total / target_batch) * target_batch)
    ppo_values['total_timesteps'] = internal_total
    ppo_config = PPOConfig(**ppo_values)
    agent = Agent(
        train_env.obs_dim,
        train_env.act_dim,
        hidden_dim=ppo_config.hidden_dim,
        init_log_std=ppo_config.init_log_std,
        squashed_entropy=ppo_config.squashed_entropy).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
    reward_scaler = (
        RewardScaler(train_env.n_envs, ppo_config.gamma, device)
        if ppo_config.normalize_returns else None)

    if train_env.obs_dim != 34:
        raise RuntimeError(
            f'fair benchmark expected 34-D observation, got {train_env.obs_dim}')

    effective = copy.deepcopy(values)
    effective['source_config'] = str(Path(args.config).resolve())
    effective['device'] = str(device)
    effective['initialization'] = 'fresh_random'
    effective['ppo'] = dataclasses.asdict(ppo_config)
    effective['benchmark']['effective_milestones'] = list(milestones)
    effective['benchmark']['requested_total_steps'] = requested_total
    effective['benchmark']['ppo_rollout_size'] = target_batch
    effective['benchmark']['ppo_internal_total_timesteps'] = internal_total
    effective['benchmark']['pilot_disable_lr_anneal'] = bool(
        args.disable_lr_anneal)
    effective['eval']['task_fingerprint'] = fingerprint
    effective['line_distribution'][
        'effective_train_task_seed'] = train_task_seed
    if args.fixed_task_specs is not None:
        effective['eval']['source_fixed_task_specs'] = str(
            Path(args.fixed_task_specs).resolve())
        effective['eval']['source_fixed_task_metadata'] = fixed_metadata

    save_seconds = 0.0
    save_start = time.perf_counter()
    with open(output / 'config.yaml', 'w', encoding='utf-8') as stream:
        yaml.safe_dump(effective, stream, sort_keys=False)
    save_fixed_task_specs(
        output / 'fixed_tasks.pt',
        fixed_specs,
        metadata={
            'n_holdout': int(eval_values['n_holdout']),
            'holdout_pool_seed': int(eval_values['holdout_pool_seed']),
            'holdout_task_seed': int(eval_values['holdout_task_seed']),
            'source_path': args.fixed_task_specs,
        })
    synchronize_device(device, bool(
        benchmark_values.get('synchronize_timing', True)))
    save_seconds += time.perf_counter() - save_start
    setup_seconds = time.perf_counter() - process_start - save_seconds

    def evaluator(
            current_agent: Agent,
            requested_step: int,
            global_step: int) -> dict[str, Any]:
        return evaluate_fixed_tasks(
            eval_env,
            fixed_specs,
            current_agent.actor_mean,
            algorithm='ppo',
            run_seed=seed,
            requested_step=requested_step,
            global_step=global_step)

    summary = run_fair_ppo(
        env=train_env,
        agent=agent,
        optimizer=optimizer,
        reward_scaler=reward_scaler,
        ppo_config=ppo_config,
        evaluator=evaluator,
        out_dir=output,
        run_seed=seed,
        milestones=milestones,
        effective_config=effective,
        setup_seconds=setup_seconds,
        initial_save_seconds=save_seconds,
        process_start=process_start,
        synchronize_timing=bool(
            benchmark_values.get('synchronize_timing', True)),
        save_checkpoints=bool(
            benchmark_values.get('save_checkpoints', True)))
    print(
        f"[ppo-fair] done step={summary['global_step']} "
        f"core={summary['core_train_s']:.1f}s "
        f"e2e={summary['e2e_s']:.1f}s "
        f"checkpoint={output / 'ppo_final.pt'}")


if __name__ == '__main__':
    main()
