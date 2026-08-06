from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Yuan.RL_controller.algorithms.compare_controller_runs import (
    compare_runs,
    load_run,
)
from Yuan.RL_controller.algorithms.controller_benchmark import (
    load_fixed_task_specs,
    save_fixed_task_specs,
    task_specs_fingerprint,
    validate_milestones,
)
from Yuan.RL_controller.algorithms.ppo import (
    Agent,
    PPOConfig,
    RewardScaler,
)
from Yuan.RL_controller.algorithms.train_ppo_fair import run_fair_ppo


class _FakeBatchedEnv:
    def __init__(self) -> None:
        self.n_envs = 2
        self.obs_dim = 4
        self.act_dim = 2
        self.device = torch.device('cpu')
        self._obs = torch.zeros(self.n_envs, self.obs_dim)
        self._age = torch.zeros(self.n_envs, dtype=torch.long)

    def reset(self) -> torch.Tensor:
        self._obs.zero_()
        self._age.zero_()
        return self._obs.clone()

    def step(self, action: torch.Tensor):
        self._age += 1
        self._obs[:, :2] += 0.05 * action
        self._obs[:, 2] = self._age.float() / 3.0
        terminal_obs = self._obs.clone()
        terminated = self._age >= 3
        truncated = torch.zeros_like(terminated)
        reward = 1.0 - 0.1 * action.square().sum(dim=-1)
        if bool(terminated.any().item()):
            self._obs[terminated] = 0.0
            self._age[terminated] = 0
        info = {
            'terminal_obs': terminal_obs,
            'episode_done': terminated.clone(),
            'n_episodes_done': int(terminated.sum().item()),
            'r_progress_mean': float(reward.mean().item()),
        }
        return (
            self._obs.clone(), reward, terminated, truncated, info)


def _write_fake_eval(
        run_dir: Path,
        algorithm: str,
        seed: int,
        requested_step: int,
        mean_values: list[float],
        *,
        fingerprint: str = 'same-task-fingerprint') -> None:
    rows = [
        {
            'task_index': index,
            'progress_m': value,
            'term_reason': index % 2,
            'episode_length': 10 + index,
        }
        for index, value in enumerate(mean_values)
    ]
    record = {
        'schema': 'controller-fair-eval-v1',
        'algorithm': algorithm,
        'run_seed': seed,
        'requested_step': requested_step,
        'global_step': requested_step,
        'task_fingerprint': fingerprint,
        'eval/mean_progress_m': float(np.mean(mean_values)),
        'eval/median_progress_m': float(np.median(mean_values)),
        'core_train_s': requested_step / 10.0,
        'eval_s': 1.0,
        'save_s': 0.1,
        'setup_s': 2.0,
        'e2e_s': 3.1 + requested_step / 10.0,
        'per_task': rows,
    }
    path = run_dir / 'eval' / f'eval_step_{requested_step}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(record, stream)


class ControllerBenchmarkHelperTest(unittest.TestCase):
    def test_fixed_task_round_trip_and_fingerprint(self):
        specs = {
            'q0': torch.arange(14, dtype=torch.float32).reshape(2, 7),
            'line_dir': torch.tensor([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0]]),
            'n_target': torch.tensor([[0.0, 0.0, 1.0],
                                      [0.0, 0.0, 1.0]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.pt'
            fingerprint = save_fixed_task_specs(path, specs, {'seed': 3})
            loaded, metadata = load_fixed_task_specs(
                path, device=torch.device('cpu'))
        self.assertEqual(fingerprint, task_specs_fingerprint(loaded))
        self.assertEqual(metadata['seed'], 3)
        self.assertEqual(metadata['fingerprint'], fingerprint)

    def test_milestones_must_start_at_zero_and_increase(self):
        self.assertEqual(validate_milestones([0, 4, 8]), (0, 4, 8))
        with self.assertRaises(ValueError):
            validate_milestones([4, 8])
        with self.assertRaises(ValueError):
            validate_milestones([0, 8, 8])


class FairPPOSmokeTest(unittest.TestCase):
    def test_cpu_fake_env_reaches_all_milestones(self):
        torch.manual_seed(5)
        env = _FakeBatchedEnv()
        config = PPOConfig(
            total_timesteps=8,
            learning_rate=1e-3,
            n_steps=2,
            anneal_lr=True,
            n_minibatches=2,
            update_epochs=1,
            hidden_dim=8,
            target_kl=None,
            normalize_returns=True)
        agent = Agent(
            env.obs_dim, env.act_dim, hidden_dim=config.hidden_dim)
        optimizer = torch.optim.Adam(
            agent.parameters(), lr=config.learning_rate, eps=1e-5)
        scaler = RewardScaler(
            env.n_envs, config.gamma, torch.device('cpu'))

        def evaluator(
                current_agent: Agent,
                requested_step: int,
                global_step: int):
            action = current_agent.actor_mean(
                torch.zeros(2, env.obs_dim))
            values = action[:, 0].detach().cpu().tolist()
            return {
                'schema': 'controller-fair-eval-v1',
                'algorithm': 'ppo',
                'run_seed': 5,
                'requested_step': requested_step,
                'global_step': global_step,
                'task_fingerprint': 'fake-tasks',
                'eval/mean_progress_m': float(np.mean(values)),
                'eval/median_progress_m': float(np.median(values)),
                'per_task': [
                    {
                        'task_index': index,
                        'progress_m': float(value),
                        'term_reason': 0,
                        'episode_length': 1,
                    }
                    for index, value in enumerate(values)
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = run_fair_ppo(
                env=env,
                agent=agent,
                optimizer=optimizer,
                reward_scaler=scaler,
                ppo_config=config,
                evaluator=evaluator,
                out_dir=output,
                run_seed=5,
                milestones=(0, 4, 8),
                effective_config={'test': True},
                synchronize_timing=True,
                save_checkpoints=True)
            artifacts = sorted((output / 'eval').glob('eval_step_*.json'))
            checkpoints = sorted(
                (output / 'checkpoints').glob('ppo_step_*.pt'))
            final_exists = (output / 'ppo_final.pt').exists()
        self.assertEqual(summary['global_step'], 8)
        self.assertEqual(summary['rollout_overshoot_steps'], 0)
        self.assertEqual(len(artifacts), 3)
        self.assertEqual(len(checkpoints), 3)
        self.assertTrue(final_exists)


class ControllerComparisonTest(unittest.TestCase):
    def test_auc_threshold_and_paired_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ppo_dir = root / 'ppo'
            flash_dir = root / 'flash'
            ppo_curve = ([0.05, 0.15], [0.15, 0.25], [0.25, 0.35])
            flash_curve = ([0.05, 0.15], [0.20, 0.30], [0.35, 0.45])
            for step, values in zip((0, 10, 20), ppo_curve):
                _write_fake_eval(ppo_dir, 'ppo', 0, step, list(values))
            for step, values in zip((0, 10, 20), flash_curve):
                _write_fake_eval(
                    flash_dir, 'flashsac', 0, step, list(values))
            runs = [
                load_run(ppo_dir, require_published_milestones=False),
                load_run(flash_dir, require_published_milestones=False),
            ]
            result = compare_runs(
                runs,
                thresholds_m=[0.2, 0.4],
                bootstrap_samples=500,
                bootstrap_seed=7)

        summaries = {
            record['algorithm']: record for record in result['runs']}
        self.assertAlmostEqual(
            summaries['ppo']['transition_auc_normalized_m'], 0.2)
        self.assertAlmostEqual(
            summaries['flashsac']['transition_auc_normalized_m'], 0.25)
        self.assertEqual(
            summaries['ppo']['time_to_threshold']['0.2'][
                'requested_step'], 10)
        self.assertIsNone(
            summaries['ppo']['time_to_threshold']['0.4'])
        pair = result['paired_final'][0]
        self.assertAlmostEqual(pair['paired_mean_delta_m'], 0.1)
        self.assertEqual(pair['candidate_win_fraction'], 1.0)

    def test_paired_comparison_rejects_different_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ppo_dir = root / 'ppo'
            flash_dir = root / 'flash'
            for step in (0, 10):
                _write_fake_eval(
                    ppo_dir, 'ppo', 0, step, [0.1, 0.2],
                    fingerprint='tasks-a')
                _write_fake_eval(
                    flash_dir, 'flashsac', 0, step, [0.1, 0.2],
                    fingerprint='tasks-b')
            runs = [
                load_run(ppo_dir, require_published_milestones=False),
                load_run(flash_dir, require_published_milestones=False),
            ]
            with self.assertRaisesRegex(ValueError, 'different fixed task'):
                compare_runs(
                    runs,
                    thresholds_m=[0.1],
                    bootstrap_samples=20)


if __name__ == '__main__':
    unittest.main()
