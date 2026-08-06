"""CPU tests for the Torch-native Yuan FlashSAC adaptation."""
from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

import numpy as np
import torch

from Yuan.RL_controller.algorithms.flashsac import (
    FlashSACAgent,
    FlashSACConfig,
    TorchUniformReplay,
    UnitLinear,
    categorical_td_target,
)
from Yuan.RL_controller.algorithms.train_flashsac import (
    load_training_state,
    run_training_loop,
    save_training_state,
)


def _tiny_config(**overrides) -> FlashSACConfig:
    values = {
        'gamma': 0.9,
        'n_step': 1,
        'buffer_max_length': 128,
        'buffer_min_length': 8,
        'buffer_device': 'cpu',
        'sample_batch_size': 8,
        'normalize_reward': True,
        'normalized_g_max': 5.0,
        'learning_rate_init': 3e-4,
        'learning_rate_peak': 3e-4,
        'learning_rate_end': 1.5e-4,
        'learning_rate_warmup_steps': 0,
        'learning_rate_decay_steps': 100,
        'actor_num_blocks': 1,
        'actor_hidden_dim': 8,
        'actor_bc_alpha': 0.0,
        'actor_noise_zeta_mu': 2.0,
        'actor_noise_zeta_max': 4,
        'actor_update_period': 2,
        'critic_num_blocks': 1,
        'critic_hidden_dim': 16,
        'critic_num_bins': 11,
        'critic_min_v': -5.0,
        'critic_max_v': 5.0,
        'critic_target_update_tau': 0.01,
        'temperature_initial_value': 0.01,
        'temperature_target_sigma': 0.15,
        'use_compile': False,
        'compile_mode': 'default',
        'use_amp': False,
    }
    values.update(overrides)
    return FlashSACConfig.from_mapping(values)


class _FakeBatchedEnv:
    def __init__(self, n_envs: int = 4, obs_dim: int = 6,
                 action_dim: int = 2):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device('cpu')
        self.observation = torch.zeros(n_envs, obs_dim)
        self.steps = torch.zeros(n_envs, dtype=torch.long)

    def reset(self) -> torch.Tensor:
        self.observation.zero_()
        self.steps.zero_()
        return self.observation.clone()

    def step(self, action: torch.Tensor):
        next_observation = self.observation.clone()
        next_observation[:, :self.action_dim] += 0.05 * action
        next_observation[:, -1] = self.steps.float() / 3.0
        reward = 1.0 - 0.1 * action.square().mean(dim=-1)
        self.steps += 1
        terminated = self.steps >= 3
        truncated = torch.zeros_like(terminated)
        terminal_observation = next_observation.clone()
        next_observation[terminated] = 0.0
        self.steps[terminated] = 0
        self.observation = next_observation
        return (
            next_observation.clone(), reward, terminated, truncated,
            {
                'terminal_obs': terminal_observation,
                'r_progress_mean': float(reward.mean().item()),
                'ep_progress_mean': float('nan'),
                'ep_len_mean': float('nan'),
            })


class FlashSACCoreTest(unittest.TestCase):
    def test_unit_linear_projection(self):
        layer = UnitLinear(7, 5)
        self.assertTrue(torch.allclose(
            layer.w.weight.norm(dim=-1), torch.ones(5), atol=1e-6))
        with torch.no_grad():
            layer.w.weight.mul_(3.7)
        layer.normalize_parameters()
        self.assertTrue(torch.allclose(
            layer.w.weight.norm(dim=-1), torch.ones(5), atol=1e-6))

    def test_categorical_projection_is_probability_distribution(self):
        batch_size, bins = 4, 11
        source = torch.full(
            (batch_size, bins), -torch.log(torch.tensor(float(bins))))
        target = categorical_td_target(
            source,
            reward=torch.tensor([0.0, 1.0, -2.0, 10.0]),
            done=torch.tensor([0.0, 0.0, 1.0, 1.0]),
            actor_entropy=torch.tensor([0.1, -0.2, 3.0, -1.0]),
            gamma=0.9, num_bins=bins, min_v=-5.0, max_v=5.0)
        self.assertEqual(target.shape, (batch_size, bins))
        self.assertTrue(bool((target >= 0.0).all().item()))
        self.assertTrue(torch.allclose(
            target.sum(dim=-1), torch.ones(batch_size), atol=1e-6))

    def test_n_step_replay_stops_at_episode_boundary(self):
        replay = TorchUniformReplay(
            obs_dim=1, action_dim=1, n_step=3, gamma=0.9,
            max_length=16, min_length=1, sample_batch_size=2,
            device='cpu')

        def add(step: int, rewards, terminated):
            replay.add({
                'observation': torch.full((2, 1), float(step)),
                'action': torch.zeros(2, 1),
                'reward': torch.tensor(rewards, dtype=torch.float32),
                'terminated': torch.tensor(terminated),
                'truncated': torch.zeros(2, dtype=torch.bool),
                'next_observation': torch.full(
                    (2, 1), float((step + 1) * 10)),
            })

        add(0, [1.0, 1.0], [False, False])
        add(1, [2.0, 2.0], [True, False])
        add(2, [100.0, 3.0], [False, False])
        self.assertEqual(len(replay), 2)
        batch = replay.sample(torch.tensor([0, 1]))
        self.assertAlmostEqual(float(batch['reward'][0]), 2.8, places=5)
        self.assertAlmostEqual(float(batch['reward'][1]), 5.23, places=5)
        self.assertTrue(bool(batch['terminated'][0].item()))
        self.assertFalse(bool(batch['terminated'][1].item()))
        self.assertEqual(float(batch['next_observation'][0, 0]), 20.0)
        self.assertEqual(float(batch['next_observation'][1, 0]), 30.0)
        self.assertAlmostEqual(
            float(batch['bootstrap_discount'][0]), 0.9 ** 2, places=6)
        self.assertAlmostEqual(
            float(batch['bootstrap_discount'][1]), 0.9 ** 3, places=6)

    def test_n_step_replay_early_truncation_bootstraps_with_actual_discount(
            self):
        replay = TorchUniformReplay(
            obs_dim=1, action_dim=1, n_step=3, gamma=0.5,
            max_length=16, min_length=1, sample_batch_size=1,
            device='cpu')
        for step, reward, truncated in (
                (0, 2.0, False),
                (1, 4.0, True),
                (2, 100.0, False)):
            replay.add({
                'observation': torch.tensor([[float(step)]]),
                'action': torch.zeros(1, 1),
                'reward': torch.tensor([reward]),
                'terminated': torch.tensor([False]),
                'truncated': torch.tensor([truncated]),
                'next_observation': torch.tensor([[float(step + 1)]]),
            })
        batch = replay.sample(torch.tensor([0]))
        self.assertAlmostEqual(float(batch['reward'][0]), 4.0, places=6)
        self.assertAlmostEqual(
            float(batch['bootstrap_discount'][0]), 0.25, places=6)
        self.assertFalse(bool(batch['terminated'][0].item()))
        self.assertTrue(bool(batch['truncated'][0].item()))
        self.assertEqual(float(batch['next_observation'][0, 0]), 2.0)

    def test_agent_update_delay_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        config = _tiny_config()
        agent = FlashSACAgent(6, 2, 4, config, 'cpu')
        for index in range(2):
            observation = torch.randn(4, 6)
            action = torch.tanh(torch.randn(4, 2))
            agent.add_transition(
                observation, action, torch.rand(4),
                torch.zeros(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool),
                torch.randn(4, 6))
        first = agent.update()
        second = agent.update()
        self.assertIn('actor/loss', first)
        self.assertNotIn('actor/loss', second)
        for metrics in (first, second):
            self.assertTrue(all(
                torch.isfinite(torch.tensor(value))
                for value in metrics.values()))

        probe = torch.randn(4, 6)
        expected = agent.actor_mean(probe)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'flashsac.pt'
            agent.save_checkpoint(path)
            restored = FlashSACAgent(6, 2, 4, config, 'cpu')
            restored.load_checkpoint(path)
            actual = restored.actor_mean(probe)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(restored.update_step, agent.update_step)
        self.assertEqual(restored.interaction_step, agent.interaction_step)

    def test_continuous_resume_bundle_restores_replay_rng_and_credit(self):
        random.seed(31)
        np.random.seed(31)
        torch.manual_seed(31)
        config = _tiny_config(n_step=3, buffer_min_length=4)
        agent = FlashSACAgent(6, 2, 4, config, 'cpu')
        environment = _FakeBatchedEnv()
        environment.line_dist = type('LineDist', (), {})()
        environment.line_dist._gen = torch.Generator()
        environment.line_dist._gen.manual_seed(314)
        for _ in range(3):
            agent.add_transition(
                torch.randn(4, 6), torch.randn(4, 2),
                torch.rand(4), torch.zeros(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool), torch.randn(4, 6))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'resume.pt'
            inference_path = Path(temp_dir) / 'inference.pt'
            save_training_state(
                path, agent, agent.interaction_step * agent.n_envs, 0.375,
                environment=environment)
            agent.save_checkpoint(inference_path)
            expected = (
                random.random(), float(np.random.random()),
                torch.rand(3),
                torch.rand(3, generator=environment.line_dist._gen))
            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)
            restored = FlashSACAgent(6, 2, 4, config, 'cpu')
            restored_environment = _FakeBatchedEnv()
            restored_environment.line_dist = type('LineDist', (), {})()
            restored_environment.line_dist._gen = torch.Generator()
            restored_environment.line_dist._gen.manual_seed(999)
            trainer = load_training_state(
                path, restored, environment=restored_environment)
            actual = (
                random.random(), float(np.random.random()),
                torch.rand(3),
                torch.rand(
                    3, generator=restored_environment.line_dist._gen))
            with self.assertRaisesRegex(ValueError, 'training-state bundle'):
                load_training_state(inference_path, restored)
        self.assertEqual(trainer['global_step'], 12)
        self.assertAlmostEqual(float(trainer['update_credit']), 0.375)
        self.assertEqual(len(restored.replay), len(agent.replay))
        self.assertEqual(
            len(restored.replay._n_step_transitions),
            len(agent.replay._n_step_transitions))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_fake_environment_training_smoke(self):
        torch.manual_seed(11)
        env = _FakeBatchedEnv()
        agent = FlashSACAgent(6, 2, 4, _tiny_config(), 'cpu')
        logs = []
        eval_steps = []

        def evaluate(
                _: FlashSACAgent, requested_step: int,
                global_step: int):
            eval_steps.append((requested_step, global_step))
            return {
                'algorithm': 'flashsac',
                'run_seed': 11,
                'eval/fake_score': 1.0,
                'per_task': [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = run_training_loop(
                env, agent, total_env_steps=64,
                updates_per_1024_env_steps=256.0,
                eval_fn=evaluate, eval_every_env_steps=1000,
                eval_milestones_env_steps=[0, 16, 64],
                eval_artifact_dir=Path(temp_dir) / 'eval',
                log_fn=logs.append,
                log_every_env_steps=16,
                checkpoint_dir=None,
                synchronize_timing=False)
            self.assertTrue(
                (Path(temp_dir) / 'eval' / 'eval_step_64.json').is_file())
        self.assertEqual(summary['global_step'], 64)
        self.assertEqual(agent.interaction_step, 16)
        self.assertGreater(summary['updates_this_run'], 0)
        self.assertGreaterEqual(len(agent.replay), 8)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(eval_steps, [(0, 0), (16, 16), (64, 64)])
        self.assertEqual(
            [entry['eval_at_step'] for entry in logs
             if 'eval_at_step' in entry], [0, 16, 64])
        for key in (
                'time/e2e_wall_s', 'time/core_train_s',
                'time/evaluation_s', 'time/save_s',
                'time/first_update_s'):
            self.assertIn(key, summary)
        actions = agent.sample_actions(
            torch.zeros(4, 6), training=False)
        self.assertEqual(actions.shape, (4, 2))
        self.assertTrue(bool((actions.abs() <= 1.0).all().item()))


if __name__ == '__main__':
    unittest.main()
