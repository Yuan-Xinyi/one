"""Small CPU tests for the auditable frozen outcome gate."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from Yuan.unified_rl.direct_seed_outcome_gate_train import (
    _disable_gate_parameters,
    _grouped_oof,
    _parse_enabled_specialists,
    _specialist_margin_and_choice,
    _validate_archive_against_runner,
    _validate_forced_specialist,
    _validate_moe_baseline_branch,
)
from Yuan.unified_rl.direct_seed_projection import ROUTE_REFINED
from Yuan.unified_rl.direct_seed_rl import (
    DirectSeedActor,
    DirectSeedActorConfig,
    DirectSeedPairedArchive,
    DirectSeedRLBatch,
    direct_seed_moe_checkpoint,
    direct_seed_moe_from_actor,
)


class DirectSeedOutcomeGateTest(unittest.TestCase):
    def test_exact_baseline_and_forced_specialist_branches(self):
        torch.manual_seed(91)
        base = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=2))
        moe = direct_seed_moe_from_actor(
            base, n_experts=4, expert_perturb_std=0.01, seed=92)
        audit = _validate_moe_baseline_branch(moe, base)
        self.assertTrue(audit['bitwise_equal'])
        self.assertEqual(audit['n_experts'], 4)

        source = direct_seed_moe_checkpoint(
            moe, update_step=7, metadata={})
        forced = copy.deepcopy(source)
        forced['metadata'] = {
            'method': 'forced-hard-moe-branch-training-collection-only',
            'forced_expert_index': 3,
            'source_checkpoint_sha256': 'a' * 64,
        }
        forced['actor']['gate.weight'].zero_()
        forced['actor']['gate.bias'].copy_(
            torch.tensor([-1.0, -1.0, -1.0, 1.0]))
        forced_audit = _validate_forced_specialist(
            source, forced, moe_sha256='a' * 64, expert_index=3)
        self.assertEqual(forced_audit['forced_expert_index'], 3)

        with torch.no_grad():
            moe.experts[0].bias[0] += 1e-4
        with self.assertRaisesRegex(
                ValueError, 'expert 0 bias differs'):
            _validate_moe_baseline_branch(moe, base)

        forced_bad = copy.deepcopy(forced)
        forced_bad['actor']['experts.1.bias'][0] += 1e-4
        with self.assertRaisesRegex(ValueError, 'tensor .* differs'):
            _validate_forced_specialist(
                source, forced_bad, moe_sha256='a' * 64,
                expert_index=3)

    def test_k4_grouped_oof_reports_single_branch_quota_metrics(self):
        rng = np.random.default_rng(93)
        features = rng.normal(size=(80, 6)).astype(np.float32)
        target = np.tile(np.arange(4, dtype=np.int64), 20)
        progress = np.full((80, 4), 0.2, dtype=np.float64)
        for row, branch in enumerate(target):
            if branch:
                progress[row, branch] = 0.35
        fingerprints = tuple(
            f'{index:064x}' for index in range(80))
        group_keys = tuple(f'bytes-{index}' for index in range(80))
        logits, report = _grouped_oof(
            features, target, progress, fingerprints, group_keys,
            objective='ovr-positive',
            n_experts=4, enabled_specialists=(1, 2, 3),
            positive_margin_m=0.01,
            logistic_c=0.1, seed=94, quotas=(0.1,))
        self.assertEqual(logits.shape, (80, 4))
        quota = report['quota_grid']['0.1']
        self.assertEqual(quota['realized_specialist_count'], 8)
        self.assertIn('pool_oracle_capture_pct', quota)
        self.assertEqual(
            sum(quota['deployed_branch_count'].values()), 80)
        self.assertEqual(
            report['protocol'],
            'exact-float32-task-bytes-grouped-5-fold')

    def test_pruned_specialists_are_fail_closed(self):
        self.assertEqual(
            _parse_enabled_specialists('all', 4), (1, 2, 3))
        self.assertEqual(
            _parse_enabled_specialists('3,1', 4), (1, 3))
        with self.assertRaisesRegex(ValueError, r'\[1, K-1\]'):
            _parse_enabled_specialists('0,3', 4)

        logits = np.asarray([
            [0.0, 0.4, 9.0, 0.7],
            [0.0, 0.8, 8.0, 0.2],
        ])
        margin, choice = _specialist_margin_and_choice(
            logits, enabled_specialists=(1, 3))
        np.testing.assert_array_equal(choice, np.asarray([3, 1]))
        np.testing.assert_allclose(margin, np.asarray([0.7, 0.8]))

        weight = np.ones((4, 5), dtype=np.float64)
        bias = np.arange(4, dtype=np.float64)
        pruned_weight, pruned_bias = _disable_gate_parameters(
            weight, bias, enabled_specialists=(1, 3))
        np.testing.assert_array_equal(
            pruned_weight[2], np.zeros(5))
        self.assertEqual(pruned_bias[2], -1e6)
        np.testing.assert_array_equal(pruned_weight[1], np.ones(5))

    def test_archive_embedded_and_legacy_replay_mismatch(self):
        task_ids = torch.tensor([10, 20], dtype=torch.int64)
        archive = DirectSeedPairedArchive(task_ids, seed=3)
        task = torch.zeros((2, 9), dtype=torch.float32)
        task[:, 0] = torch.tensor([0.1, 0.2])
        q = torch.arange(14, dtype=torch.float32).reshape(2, 7) / 20.0
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q.clone(),
            q_projected=q,
            fallback_q=torch.zeros_like(q),
            progress_m=torch.tensor([0.3, 0.4]),
            route=torch.full((2,), ROUTE_REFINED, dtype=torch.int64),
        )
        archive.update(task_ids, batch)
        state = archive.state_dict()
        embedded = {'paired_archive': copy.deepcopy(state)}
        result = _validate_archive_against_runner(state, embedded)
        self.assertIn('embedded', result['method'])

        changed = copy.deepcopy(embedded)
        changed['paired_archive']['storage']['progress_m'][1] += 0.01
        with self.assertRaisesRegex(ValueError, 'progress_m differs'):
            _validate_archive_against_runner(state, changed)

        replay_storage = {
            'task': task.clone(),
            'q_projected': q.clone(),
            'progress_m': torch.tensor([0.3, 0.4]),
            'route': torch.full((2,), ROUTE_REFINED, dtype=torch.int64),
        }
        legacy = {
            'paired_archive': None,
            'macro_replay': {
                'capacity': 2,
                'size': 2,
                'write_index': 0,
                'storage': replay_storage,
            },
        }
        with self.assertRaisesRegex(ValueError, 'allow-legacy'):
            _validate_archive_against_runner(state, legacy)
        legacy_result = _validate_archive_against_runner(
            state, legacy, allow_legacy=True)
        self.assertIn('legacy', legacy_result['method'])
        legacy['macro_replay']['storage']['q_projected'][0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, 'q_projected'):
            _validate_archive_against_runner(
                state, legacy, allow_legacy=True)


if __name__ == '__main__':
    unittest.main()
