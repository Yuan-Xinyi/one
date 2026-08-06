"""Train the Yuan null-space controller with Torch-native FlashSAC.

Usage:
    python -m Yuan.RL_controller.algorithms.train_flashsac \
        --config Yuan/RL_controller/config_flashsac.yaml \
        --out-dir Yuan/RL_controller/runs/flashsac_seed0

The environment remains the existing ``NSRLBatchedEnv``.  This entry point
only replaces PPO's data collection and optimizer with the FlashSAC core in
``flashsac.py``.
"""
from __future__ import annotations

import os
import sys

import argparse
import copy
import dataclasses
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from Yuan.RL_controller.algorithms.flashsac import (
    OFFICIAL_COMMIT,
    FlashSACAgent,
    FlashSACConfig,
)


LogFn = Callable[[dict[str, Any]], None]
EvalFn = Callable[[FlashSACAgent, int, int], dict[str, Any]]
TRAINING_STATE_FORMAT = 'yuan-flashsac-training-state-v1'


def _relaunch_with_conda_lib() -> None:
    """Apply the existing controller entry point's libstdc++ workaround."""
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


def _synchronize(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == 'cuda':
        torch.cuda.synchronize(device)


class _SectionTimer:
    """Accumulate CUDA kernel time asynchronously and CPU wall sections.

    CUDA events preserve production asynchrony.  ``flush`` synchronizes only
    at reporting/evaluation/checkpoint boundaries; end-to-end wall time is
    measured separately and therefore also includes Python and I/O overhead.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.totals: dict[str, float] = {
            name: 0.0 for name in ('action', 'environment', 'replay', 'update')}
        self.first: dict[str, float | None] = {
            'policy_action': None,
            'update': None,
        }
        self._pending: list[
            tuple[str, Any, Any, str | None]] = []

    def start(self) -> Any:
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(
            self, name: str, token: Any,
            first_key: str | None = None) -> None:
        if name not in self.totals:
            raise ValueError(f'unknown timing section: {name}')
        if self.device.type == 'cuda':
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((name, token, end, first_key))
            return
        duration = time.perf_counter() - float(token)
        self.totals[name] += duration
        if first_key is not None and self.first[first_key] is None:
            self.first[first_key] = duration

    def flush(self) -> None:
        if self.device.type != 'cuda' or not self._pending:
            return
        torch.cuda.synchronize(self.device)
        for name, start, end, first_key in self._pending:
            duration = float(start.elapsed_time(end)) / 1000.0
            self.totals[name] += duration
            if first_key is not None and self.first[first_key] is None:
                self.first[first_key] = duration
        self._pending.clear()


def _capture_global_rng_state() -> dict[str, Any]:
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available()
            else None),
    }


def _restore_global_rng_state(state: Mapping[str, Any]) -> None:
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    if set(state) != required:
        raise ValueError('training state has incomplete global RNG state')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(torch.as_tensor(state['torch_cpu']).cpu())
    cuda_state = state['torch_cuda']
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA RNG state cannot be restored without CUDA')
        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError(
                'CUDA device count differs from saved training state')
        torch.cuda.set_rng_state_all([
            torch.as_tensor(value).cpu() for value in cuda_state])


def _capture_environment_rng_state(environment: Any | None) -> dict[str, Any]:
    generator = None
    if environment is not None:
        line_distribution = getattr(environment, 'line_dist', None)
        generator = getattr(line_distribution, '_gen', None)
    return {
        'line_distribution_generator': (
            generator.get_state().cpu() if generator is not None else None),
    }


def _restore_environment_rng_state(
        environment: Any | None, state: Mapping[str, Any]) -> None:
    if set(state) != {'line_distribution_generator'}:
        raise ValueError('training state has invalid environment RNG state')
    stored = state['line_distribution_generator']
    if stored is None:
        return
    if environment is None:
        raise ValueError(
            'training state needs an environment to restore its task RNG')
    line_distribution = getattr(environment, 'line_dist', None)
    generator = getattr(line_distribution, '_gen', None)
    if generator is None:
        raise ValueError(
            'training environment lacks a line-distribution generator')
    generator.set_state(torch.as_tensor(stored).cpu())


def save_training_state(
        path: str | Path, agent: FlashSACAgent, global_step: int,
        update_credit: float, environment: Any | None = None) -> None:
    """Atomically save every state required by continuous CLI resume."""
    if not 0.0 <= update_credit < 1.0 + 1e-9:
        raise ValueError('update_credit must be in [0, 1)')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': TRAINING_STATE_FORMAT,
        'checkpoint_kind': 'continuous_resume',
        'agent': agent.checkpoint_state(),
        'replay': agent.replay.state_dict(),
        'trainer': {
            'global_step': int(global_step),
            'update_credit': float(update_credit),
        },
        'global_rng': _capture_global_rng_state(),
        'environment_rng': _capture_environment_rng_state(environment),
    }
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_state(
        path: str | Path, agent: FlashSACAgent,
        environment: Any | None = None) -> dict[str, float | int]:
    """Load a strict continuous-resume bundle; reject inference checkpoints."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError('FlashSAC training state must be a mapping')
    if state.get('format') != TRAINING_STATE_FORMAT:
        raise ValueError(
            'continuous resume requires a training-state bundle; '
            'an inference-only checkpoint is not sufficient')
    if state.get('checkpoint_kind') != 'continuous_resume':
        raise ValueError('invalid FlashSAC training-state kind')
    agent_state = state.get('agent')
    replay_state = state.get('replay')
    trainer = state.get('trainer')
    rng_state = state.get('global_rng')
    environment_rng = state.get('environment_rng')
    if not all(isinstance(item, Mapping) for item in (
            agent_state, replay_state, trainer, rng_state,
            environment_rng)):
        raise ValueError('FlashSAC training-state bundle is incomplete')
    agent.load_checkpoint_state(agent_state, load_optimizers=True)
    agent.replay.load_state_dict(replay_state)
    global_step = int(trainer['global_step'])
    update_credit = float(trainer['update_credit'])
    if global_step != agent.interaction_step * agent.n_envs:
        raise ValueError(
            'training global_step disagrees with agent interaction_step')
    if not 0.0 <= update_credit < 1.0 + 1e-9:
        raise ValueError('invalid stored update_credit')
    _restore_global_rng_state(rng_state)
    _restore_environment_rng_state(environment, environment_rng)
    return {
        'global_step': global_step,
        'update_credit': update_credit,
    }


def run_training_loop(
        env: Any,
        agent: FlashSACAgent,
        total_env_steps: int,
        updates_per_1024_env_steps: float = 2.0,
        eval_fn: EvalFn | None = None,
        eval_every_env_steps: int = 1_000_000,
        eval_milestones_env_steps: Sequence[int] | None = None,
        eval_artifact_dir: str | Path | None = None,
        run_seed: int = 0,
        log_fn: LogFn | None = None,
        log_every_env_steps: int = 100_000,
        checkpoint_dir: str | Path | None = None,
        checkpoint_every_env_steps: int = 5_000_000,
        save_replay_at_checkpoints: bool = False,
        synchronize_timing: bool = False,
        initial_global_step: int | None = None,
        initial_update_credit: float = 0.0,
        process_start_time: float | None = None,
        setup_seconds: float = 0.0,
        evaluate_at_start: bool = True) -> dict[str, Any]:
    """Run the tensor-native interaction/update loop.

    ``updates_per_1024_env_steps`` preserves the official GPU update-to-data
    ratio when ``env.n_envs`` differs from 1024.  For example, the official
    value 2.0 means two optimizer updates per 1024 newly collected
    transitions; with 128 environments it schedules one update every four
    vector steps.
    """
    if env.n_envs != agent.n_envs:
        raise ValueError('environment and FlashSAC n_envs differ')
    if total_env_steps <= 0:
        raise ValueError('total_env_steps must be positive')
    if (not np.isfinite(updates_per_1024_env_steps)
            or updates_per_1024_env_steps < 0.0):
        raise ValueError(
            'updates_per_1024_env_steps must be finite and non-negative')
    if eval_every_env_steps <= 0 or log_every_env_steps <= 0:
        raise ValueError('evaluation and logging intervals must be positive')
    if checkpoint_every_env_steps <= 0:
        raise ValueError('checkpoint interval must be positive')
    if not 0.0 <= initial_update_credit < 1.0 + 1e-9:
        raise ValueError('initial_update_credit must be in [0, 1)')
    if setup_seconds < 0.0 or not np.isfinite(setup_seconds):
        raise ValueError('setup_seconds must be finite and non-negative')

    checkpoint_path = (
        Path(checkpoint_dir) if checkpoint_dir is not None else None)
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    artifact_path = (
        Path(eval_artifact_dir)
        if eval_artifact_dir is not None else None)
    if artifact_path is not None:
        artifact_path.mkdir(parents=True, exist_ok=True)

    observation = env.reset()
    observation = observation.to(
        device=agent.device, dtype=torch.float32)
    agent.reset_after_env_reset()
    global_step = (
        int(initial_global_step) if initial_global_step is not None
        else agent.interaction_step * agent.n_envs)
    if global_step != agent.interaction_step * agent.n_envs:
        raise ValueError(
            'initial_global_step disagrees with agent interaction_step')
    if eval_milestones_env_steps is None:
        evaluation_milestones: list[int] = []
        if evaluate_at_start and global_step == 0:
            evaluation_milestones.append(0)
        milestone = (
            ((global_step // eval_every_env_steps) + 1)
            * eval_every_env_steps)
        while milestone <= total_env_steps:
            evaluation_milestones.append(milestone)
            milestone += eval_every_env_steps
    else:
        from Yuan.RL_controller.algorithms.controller_benchmark import (
            validate_milestones,
        )
        validated = validate_milestones(eval_milestones_env_steps)
        evaluation_milestones = [
            milestone for milestone in validated
            if milestone <= total_env_steps and (
                milestone > global_step
                or (milestone == 0 and global_step == 0
                    and evaluate_at_start))]
    next_eval_index = 0
    next_log = (
        ((global_step // log_every_env_steps) + 1)
        * log_every_env_steps)
    next_checkpoint = (
        ((global_step // checkpoint_every_env_steps) + 1)
        * checkpoint_every_env_steps)

    update_credit = float(initial_update_credit)
    latest_update: dict[str, float] = {}
    start_time = time.perf_counter()
    if process_start_time is None:
        process_start_time = start_time
    section_timer = _SectionTimer(agent.device)
    evaluation_seconds = 0.0
    checkpoint_seconds = 0.0
    n_updates_this_run = 0
    core_wall_seconds = 0.0
    core_segment_start = time.perf_counter()
    core_is_running = True

    def detailed_core_seconds() -> float:
        return sum(section_timer.totals.values())

    def current_core_wall_seconds() -> float:
        if core_is_running:
            return core_wall_seconds + (
                time.perf_counter() - core_segment_start)
        return core_wall_seconds

    def pause_core() -> None:
        nonlocal core_wall_seconds, core_is_running
        if not core_is_running:
            return
        section_timer.flush()
        _synchronize(agent.device, True)
        core_wall_seconds += time.perf_counter() - core_segment_start
        core_is_running = False

    def resume_core() -> None:
        nonlocal core_segment_start, core_is_running
        if core_is_running:
            return
        core_segment_start = time.perf_counter()
        core_is_running = True

    def evaluate(requested_step: int) -> None:
        nonlocal evaluation_seconds
        if eval_fn is None:
            return
        pause_core()
        evaluation_start = time.perf_counter()
        evaluation = eval_fn(agent, requested_step, global_step)
        _synchronize(agent.device, True)
        evaluation_duration = time.perf_counter() - evaluation_start
        evaluation_seconds += evaluation_duration
        elapsed = time.perf_counter() - process_start_time
        evaluation = {
            **evaluation,
            'eval_at_step': int(requested_step),
            'requested_step': int(requested_step),
            'global_step': int(global_step),
            'setup_s': setup_seconds,
            'core_train_s': current_core_wall_seconds(),
            'eval_s': evaluation_seconds,
            'eval_last_s': evaluation_duration,
            'save_s': checkpoint_seconds,
            'e2e_s': elapsed,
            'wall_s': elapsed,
            'time/e2e_wall_s': elapsed,
            'time/core_train_s': current_core_wall_seconds(),
            'time/section_detail_s': detailed_core_seconds(),
            'time/evaluation_s': evaluation_seconds,
            'time/checkpoint_s': checkpoint_seconds,
        }
        if artifact_path is not None:
            from Yuan.RL_controller.algorithms.controller_benchmark import (
                append_jsonl,
                write_json,
            )
            write_json(
                artifact_path / f'eval_step_{requested_step}.json',
                evaluation, refuse_overwrite=True)
            append_jsonl(artifact_path / 'eval_metrics.jsonl', evaluation)
        if log_fn is not None:
            # The full per-task rows live in the JSON artifact, not train.log.
            log_fn({
                key: value for key, value in evaluation.items()
                if key != 'per_task'})
        resume_core()

    while (next_eval_index < len(evaluation_milestones)
           and evaluation_milestones[next_eval_index] <= global_step):
        evaluate(evaluation_milestones[next_eval_index])
        next_eval_index += 1

    while global_step < total_env_steps:
        segment_start = section_timer.start()
        policy_action = agent.can_update()
        if policy_action:
            action = agent.sample_actions(observation, training=True)
        else:
            action = torch.empty(
                agent.n_envs, agent.action_dim, device=agent.device,
                dtype=torch.float32).uniform_(-1.0, 1.0)
        section_timer.stop(
            'action', segment_start,
            first_key='policy_action' if policy_action else None)
        if synchronize_timing:
            section_timer.flush()

        segment_start = section_timer.start()
        next_observation, reward, terminated, truncated, info = env.step(
            action)
        section_timer.stop('environment', segment_start)
        if synchronize_timing:
            section_timer.flush()

        done = terminated | truncated
        replay_next_observation = next_observation.clone()
        terminal_observation = info.get('terminal_obs')
        if terminal_observation is None:
            raise RuntimeError(
                'FlashSAC requires info["terminal_obs"] from the environment')
        replay_next_observation[done] = terminal_observation[done]

        segment_start = section_timer.start()
        agent.add_transition(
            observation, action, reward, terminated, truncated,
            replay_next_observation)
        section_timer.stop('replay', segment_start)
        if synchronize_timing:
            section_timer.flush()
        observation = next_observation
        global_step += agent.n_envs

        if agent.can_update():
            update_credit += (
                updates_per_1024_env_steps * agent.n_envs / 1024.0)
            while update_credit >= 1.0:
                segment_start = section_timer.start()
                latest_update = agent.update()
                section_timer.stop(
                    'update', segment_start, first_key='update')
                if synchronize_timing:
                    section_timer.flush()
                n_updates_this_run += 1
                update_credit -= 1.0

        if log_fn is not None and global_step >= next_log:
            section_timer.flush()
            elapsed = time.perf_counter() - start_time
            first_policy_action_seconds = section_timer.first['policy_action']
            first_update_seconds = section_timer.first['update']
            compile_cold_seconds = (
                first_policy_action_seconds + first_update_seconds
                if (agent.config.use_compile
                    and first_policy_action_seconds is not None
                    and first_update_seconds is not None)
                else float('nan'))
            payload: dict[str, Any] = {
                'global_step': global_step,
                'interaction_step': agent.interaction_step,
                'update_step': agent.update_step,
                'replay_size': len(agent.replay),
                'wall_s': elapsed,
                'time/e2e_wall_s': elapsed,
                'setup_s': setup_seconds,
                'core_train_s': current_core_wall_seconds(),
                'eval_s': evaluation_seconds,
                'save_s': checkpoint_seconds,
                'e2e_s': time.perf_counter() - process_start_time,
                'time/core_train_s': current_core_wall_seconds(),
                'time/section_detail_s': detailed_core_seconds(),
                'time/action_s': section_timer.totals['action'],
                'time/environment_s': section_timer.totals['environment'],
                'time/replay_s': section_timer.totals['replay'],
                'time/update_s': section_timer.totals['update'],
                'time/evaluation_s': evaluation_seconds,
                'time/checkpoint_s': checkpoint_seconds,
                'time/save_s': checkpoint_seconds,
                'time/core_semantics': (
                    'synchronized_wall_excluding_eval_and_save'),
                'time/detail_semantics': (
                    'cuda_event_kernel_time' if agent.device.type == 'cuda'
                    else 'cpu_section_wall_time'),
                'time/first_policy_action_s': (
                    first_policy_action_seconds
                    if first_policy_action_seconds is not None
                    else float('nan')),
                'time/first_update_s': (
                    first_update_seconds
                    if first_update_seconds is not None
                    else float('nan')),
                'time/compile_cold_s': compile_cold_seconds,
                'reward/progress': float(
                    info.get('r_progress_mean', float('nan'))),
                'episode/progress_mean_m': float(
                    info.get('ep_progress_mean', float('nan'))),
                'episode/length_mean': float(
                    info.get('ep_len_mean', float('nan'))),
                **latest_update,
            }
            if agent.device.type == 'cuda':
                payload['memory/max_allocated_bytes'] = (
                    torch.cuda.max_memory_allocated(agent.device))
                payload['memory/max_reserved_bytes'] = (
                    torch.cuda.max_memory_reserved(agent.device))
            log_fn(payload)
            while next_log <= global_step:
                next_log += log_every_env_steps

        while (next_eval_index < len(evaluation_milestones)
               and global_step >= evaluation_milestones[next_eval_index]):
            evaluate(evaluation_milestones[next_eval_index])
            next_eval_index += 1

        if checkpoint_path is not None and global_step >= next_checkpoint:
            pause_core()
            segment_start = time.perf_counter()
            if save_replay_at_checkpoints:
                save_training_state(
                    checkpoint_path
                    / f'flashsac_resume_step_{global_step}.pt',
                    agent, global_step, update_credit,
                    environment=env)
            else:
                # Explicitly named inference-only: this cannot be passed to
                # --resume-from because replay/RNG/update credit are absent.
                agent.save_checkpoint(
                    checkpoint_path
                    / f'flashsac_inference_step_{global_step}.pt')
            _synchronize(agent.device, True)
            checkpoint_seconds += time.perf_counter() - segment_start
            resume_core()
            while next_checkpoint <= global_step:
                next_checkpoint += checkpoint_every_env_steps

    pause_core()
    run_loop_seconds = time.perf_counter() - start_time
    e2e_seconds = time.perf_counter() - process_start_time
    first_policy_action_seconds = section_timer.first['policy_action']
    first_update_seconds = section_timer.first['update']
    compile_cold_seconds = (
        first_policy_action_seconds + first_update_seconds
        if (agent.config.use_compile
            and first_policy_action_seconds is not None
            and first_update_seconds is not None)
        else float('nan'))
    return {
        'global_step': global_step,
        'interaction_step': agent.interaction_step,
        'update_step': agent.update_step,
        'updates_this_run': n_updates_this_run,
        'replay_size': len(agent.replay),
        'update_credit': update_credit,
        'setup_s': setup_seconds,
        'core_train_s': current_core_wall_seconds(),
        'eval_s': evaluation_seconds,
        'save_s': checkpoint_seconds,
        'e2e_s': e2e_seconds,
        'wall_s': e2e_seconds,
        'time/e2e_wall_s': e2e_seconds,
        'time/run_loop_wall_s': run_loop_seconds,
        'time/core_train_s': current_core_wall_seconds(),
        'time/section_detail_s': detailed_core_seconds(),
        'time/action_s': section_timer.totals['action'],
        'time/environment_s': section_timer.totals['environment'],
        'time/replay_s': section_timer.totals['replay'],
        'time/update_s': section_timer.totals['update'],
        'time/evaluation_s': evaluation_seconds,
        'time/checkpoint_s': checkpoint_seconds,
        'time/save_s': checkpoint_seconds,
        'time/core_semantics': (
            'synchronized_wall_excluding_eval_and_save'),
        'time/detail_semantics': (
            'cuda_event_kernel_time' if agent.device.type == 'cuda'
            else 'cpu_section_wall_time'),
        'time/first_policy_action_s': (
            first_policy_action_seconds
            if first_policy_action_seconds is not None else float('nan')),
        'time/first_update_s': (
            first_update_seconds
            if first_update_seconds is not None else float('nan')),
        'time/compile_cold_s': compile_cold_seconds,
    }


def _make_fixed_eval(
        eval_env: Any, fixed_specs: dict[str, torch.Tensor],
        run_seed: int) -> EvalFn:
    from Yuan.RL_controller.algorithms.controller_benchmark import (
        evaluate_fixed_tasks,
    )

    @torch.no_grad()
    def evaluate(
            agent: FlashSACAgent, requested_step: int,
            global_step: int) -> dict[str, Any]:
        return evaluate_fixed_tasks(
            eval_env, fixed_specs, agent.actor_mean,
            algorithm='flashsac', run_seed=run_seed,
            requested_step=requested_step, global_step=global_step)

    return evaluate


def _effective_env_config(values: dict[str, Any]) -> dict[str, Any]:
    from Yuan.RL_controller.env.env import EnvConfig

    known = {field.name for field in dataclasses.fields(EnvConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f'unknown environment config keys: {unknown}')
    return dict(values)


def main() -> None:
    # Must happen before the lazy imports of modules that transitively import
    # ``one`` and matplotlib.
    _relaunch_with_conda_lib()
    process_start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='Yuan/RL_controller/config_flashsac.yaml')
    parser.add_argument(
        '--out-dir', default='Yuan/RL_controller/runs/flashsac_seed0')
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--resume-from', default=None,
        help='strict continuous-resume bundle (not an inference checkpoint)')
    parser.add_argument(
        '--resume-replay-from', default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        '--max-env-steps', type=int, default=None,
        help='optional short-run override; recorded in the effective config')
    args = parser.parse_args()
    if args.resume_replay_from is not None:
        raise ValueError(
            '--resume-replay-from is no longer accepted: use one atomic '
            '--resume-from training-state bundle')
    is_resume = args.resume_from is not None
    out_dir = Path(args.out_dir)
    if is_resume:
        if not out_dir.is_dir() or not any(out_dir.iterdir()):
            raise FileNotFoundError(
                'continuous resume requires the existing non-empty run '
                f'directory: {out_dir}')
        for required_name in ('config.yaml', 'train.log', 'fixed_tasks.pt'):
            if not (out_dir / required_name).is_file():
                raise FileNotFoundError(
                    f'resume run directory lacks {required_name}: {out_dir}')
    else:
        from Yuan.RL_controller.algorithms.controller_benchmark import (
            ensure_fresh_outdir,
        )
        ensure_fresh_outdir(out_dir)

    with open(args.config, 'r') as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError('FlashSAC YAML root must be a mapping')
    for section in (
            'env', 'flashsac', 'line_distribution', 'eval', 'train'):
        if not isinstance(values.get(section), dict):
            raise ValueError(f'FlashSAC YAML needs mapping section {section}')

    seed = int(values.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    device = torch.device(
        args.device if args.device is not None
        else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())

    effective = copy.deepcopy(values)
    if args.max_env_steps is not None:
        effective['train']['total_env_steps'] = args.max_env_steps
    if device.type == 'cpu':
        # The publication configuration allocates a multi-GB replay and builds
        # a 100k-task pool.  CPU mode is an integration/debug profile only.
        effective['env']['n_envs'] = min(
            int(effective['env']['n_envs']), 16)
        effective['env']['max_steps'] = min(
            int(effective['env']['max_steps']), 100)
        effective['flashsac']['buffer_max_length'] = min(
            int(effective['flashsac']['buffer_max_length']), 8192)
        effective['flashsac']['buffer_min_length'] = min(
            int(effective['flashsac']['buffer_min_length']), 512)
        effective['flashsac']['sample_batch_size'] = min(
            int(effective['flashsac']['sample_batch_size']), 256)
        effective['flashsac']['buffer_device'] = 'cpu'
        effective['flashsac']['use_amp'] = False
        effective['flashsac']['use_compile'] = False
        effective['line_distribution']['n_pool'] = min(
            int(effective['line_distribution']['n_pool']), 512)
        effective['line_distribution']['feasibility_filter'] = False
        effective['eval']['n_holdout'] = min(
            int(effective['eval']['n_holdout']), 16)
        effective['train']['total_env_steps'] = min(
            int(effective['train']['total_env_steps']), 4096)
        effective['train']['log_every_env_steps'] = min(
            int(effective['train']['log_every_env_steps']), 1024)
        effective['train']['checkpoint_every_env_steps'] = min(
            int(effective['train']['checkpoint_every_env_steps']), 4096)
        print(
            '[flashsac] CPU debug profile: <=16 envs, <=8192 replay, '
            '<=512 task pool, <=4096 transitions; AMP/compile off')
    flash_values = copy.deepcopy(effective['flashsac'])
    flash_config = FlashSACConfig.from_mapping(flash_values)

    from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
    from Yuan.RL_controller.env.line_distribution import LineDistribution

    env_values = _effective_env_config(effective['env'])
    env_config = EnvConfig(**env_values)
    line_values = effective['line_distribution']
    threshold = (
        float(line_values['feasibility_threshold_m'])
        if line_values.get('feasibility_filter', False) else None)

    print(
        f'[flashsac] official_commit={OFFICIAL_COMMIT} device={device} '
        f'n_envs={env_config.n_envs}')
    train_env = NSRLBatchedEnv(
        env_config, line_dist=None, device=device)
    train_env.line_dist = LineDistribution.load_or_build(
        kin=train_env.kin, collision=train_env.collision,
        n_pool=int(line_values['n_pool']),
        n_target_noise_deg=float(line_values['n_target_noise_deg']),
        seed=int(line_values['train_seed']),
        env_cfg=env_config,
        feasibility_threshold_m=threshold)
    if 'train_task_seed' not in line_values:
        raise ValueError(
            'line_distribution.train_task_seed is required for a '
            'cache-independent training task stream')
    # load_or_build otherwise leaves the sampling generator in a different
    # state on cache hit versus fresh pool construction.
    train_env.line_dist._gen.manual_seed(
        int(line_values['train_task_seed']) + seed)

    eval_values = effective['eval']
    eval_count = int(eval_values['n_holdout'])
    eval_config = EnvConfig(**{
        **env_values,
        'n_envs': eval_count,
    })
    eval_env = NSRLBatchedEnv(
        eval_config, line_dist=None, device=device)
    from Yuan.RL_controller.algorithms.controller_benchmark import (
        build_fixed_task_specs,
        load_fixed_task_specs,
        save_fixed_task_specs,
    )
    if is_resume:
        fixed_specs, fixed_metadata = load_fixed_task_specs(
            out_dir / 'fixed_tasks.pt', device=device,
            dtype=eval_env.kin.dtype)
        if int(fixed_specs['q0'].shape[0]) != eval_count:
            raise ValueError(
                'resume fixed-task count differs from eval.n_holdout')
    else:
        fixed_specs = build_fixed_task_specs(
            eval_env, eval_config, line_values, eval_values)
        save_fixed_task_specs(
            out_dir / 'fixed_tasks.pt', fixed_specs,
            metadata={
                'algorithm': 'shared_controller_holdout',
                'holdout_pool_seed': int(
                    eval_values['holdout_pool_seed']),
                'holdout_task_seed': int(
                    eval_values['holdout_task_seed']),
            })
    eval_fn = _make_fixed_eval(eval_env, fixed_specs, seed)

    agent = FlashSACAgent(
        train_env.obs_dim, train_env.act_dim, train_env.n_envs,
        flash_config, device)
    resume_state: dict[str, float | int] | None = None
    if is_resume:
        resume_state = load_training_state(
            args.resume_from, agent, environment=train_env)

    effective['flashsac'] = dataclasses.asdict(agent.config)
    effective['source_config'] = str(Path(args.config).resolve())
    effective['official_flashsac_commit'] = OFFICIAL_COMMIT
    if is_resume:
        with open(out_dir / 'config.yaml', 'r') as stream:
            stored_effective = yaml.safe_load(stream)
        for section in (
                'seed', 'env', 'flashsac', 'line_distribution', 'eval'):
            if stored_effective.get(section) != effective.get(section):
                raise ValueError(
                    f'resume configuration changed section {section}')
        stored_train = dict(stored_effective['train'])
        current_train = dict(effective['train'])
        stored_train.pop('total_env_steps', None)
        current_train.pop('total_env_steps', None)
        if stored_train != current_train:
            raise ValueError('resume configuration changed training protocol')
    else:
        with open(out_dir / 'config.yaml', 'x') as stream:
            yaml.safe_dump(effective, stream, sort_keys=False)

    log_stream = open(out_dir / 'train.log', 'a' if is_resume else 'x')
    setup_seconds = time.perf_counter() - process_start_time

    def log_fn(payload: dict[str, Any]) -> None:
        log_stream.write(repr(payload) + '\n')
        log_stream.flush()
        if 'eval_at_step' in payload:
            print(
                f"[flashsac] eval step={payload['eval_at_step']} "
                f"mean={payload.get('eval/mean_progress_m', float('nan')):.4f}m",
                flush=True)
        elif 'global_step' in payload:
            print(
                f"[flashsac] step={payload['global_step']} "
                f"updates={payload['update_step']} "
                f"replay={payload['replay_size']} "
                f"wall={payload['wall_s']:.1f}s",
                flush=True)

    train_values = effective['train']
    try:
        summary = run_training_loop(
            train_env, agent,
            total_env_steps=int(train_values['total_env_steps']),
            updates_per_1024_env_steps=float(
                train_values['updates_per_1024_env_steps']),
            eval_fn=eval_fn,
            eval_every_env_steps=int(
                train_values['eval_every_env_steps']),
            eval_milestones_env_steps=train_values.get(
                'eval_milestones_env_steps'),
            eval_artifact_dir=out_dir / 'eval',
            run_seed=seed,
            log_fn=log_fn,
            log_every_env_steps=int(
                train_values['log_every_env_steps']),
            checkpoint_dir=out_dir / 'checkpoints',
            checkpoint_every_env_steps=int(
                train_values['checkpoint_every_env_steps']),
            save_replay_at_checkpoints=bool(
                train_values.get(
                    'save_replay_at_checkpoints', False)),
            synchronize_timing=bool(
                train_values.get('synchronize_timing', False)),
            initial_global_step=(
                int(resume_state['global_step'])
                if resume_state is not None else None),
            initial_update_credit=(
                float(resume_state['update_credit'])
                if resume_state is not None else 0.0),
            process_start_time=process_start_time,
            setup_seconds=setup_seconds,
            evaluate_at_start=not is_resume)
        final_save_start = time.perf_counter()
        agent.save_checkpoint(out_dir / 'flashsac_inference.pt')
        if train_values.get('save_final_replay', False):
            save_training_state(
                out_dir / 'flashsac_resume.pt', agent,
                int(summary['global_step']),
                float(summary['update_credit']),
                environment=train_env)
        _synchronize(agent.device, True)
        final_save_seconds = time.perf_counter() - final_save_start
        final_e2e_seconds = time.perf_counter() - process_start_time
        summary['time/final_save_s'] = final_save_seconds
        summary['time/save_s'] += final_save_seconds
        summary['save_s'] += final_save_seconds
        summary['time/e2e_wall_s'] = final_e2e_seconds
        summary['e2e_s'] = final_e2e_seconds
        summary['wall_s'] = final_e2e_seconds
        log_fn({'training_complete': True, **summary})
    finally:
        log_stream.close()

    print(
        '[flashsac] done; inference checkpoint='
        f'{out_dir / "flashsac_inference.pt"}')


if __name__ == '__main__':
    main()
