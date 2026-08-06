"""CPU unit tests for the one-shot direct-seed path."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedCandidateBatch,
)
from Yuan.unified_rl.direct_seed_eval import (
    _load_fallback_filter_manifest,
    _load_reference_progress,
    summarize_direct_seed,
)
from Yuan.unified_rl.direct_seed_bidir import (
    DirectTaskCycleSampler,
    _paired_collection_contract,
    _require_full_paired_archive,
    _restore_paired_collection_provenance,
    _dry_run as direct_seed_bidir_dry_run,
    _load_direct_seed_rl_api,
    _load_paired_explorer_checkpoint,
    _parser as direct_seed_bidir_parser,
    _set_optimizer_lr,
    _validate_args as validate_direct_seed_bidir_args,
)
from Yuan.unified_rl.direct_seed_model import (
    DirectSeedConfig,
    DirectSeedGenerator,
    direct_seed_checkpoint,
    load_deployment_generator,
    load_direct_seed_generator,
)
from Yuan.unified_rl.direct_seed_outcome_mlp_gate_train import (
    _fit_advantage_mlp,
    _nonlinear_gate_actor,
    _predict_advantage,
)
from Yuan.unified_rl.direct_seed_projection import (
    DirectSeedProjectionConfig,
    ROUTE_DIRECT,
    ROUTE_FALLBACK,
    ROUTE_INVALID,
    ROUTE_REFINED,
    route_generated_seed,
    strict_seed_validity,
)
from Yuan.unified_rl.direct_seed_rl import (
    DirectSeedActor,
    DirectSeedActorConfig,
    DirectSeedCriticConfig,
    DirectSeedEliteMemory,
    DirectSeedMacroReplay,
    DirectSeedMoEActor,
    DirectSeedMoEActorConfig,
    DirectSeedPairedArchive,
    DirectSeedRLBatch,
    DirectSeedRLConfig,
    TwinMacroQ,
    direct_seed_moe_checkpoint,
    direct_seed_moe_from_actor,
    direct_seed_rl_checkpoint,
    load_direct_seed_moe_checkpoint,
    load_direct_seed_rl_checkpoint,
    synthetic_direct_seed_rl_smoke,
    update_direct_seed_moe_advantage,
    update_direct_seed_moe_projection,
    update_direct_seed_projection,
    update_direct_seed_precision,
    update_direct_seed_rl,
)
from Yuan.unified_rl.direct_seed_train import (
    geometry_grouped_three_way_split,
    return_weighted_soft_nearest_support_loss,
)


class _DirectKin:
    device = torch.device('cpu')
    dtype = torch.float32
    lmt_lo = -torch.ones(7)
    lmt_up = torch.ones(7)
    q_mid = torch.zeros(7)

    @staticmethod
    def tcp_fk_jac(q):
        batch_size = q.shape[0]
        position = q[:, :3]
        rotation = torch.eye(
            3, dtype=q.dtype, device=q.device).expand(
                batch_size, -1, -1).clone()
        jacobian = torch.zeros(
            (batch_size, 6, 7), dtype=q.dtype, device=q.device)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 2, 2] = 1.0
        transforms = _DirectKin.link_transforms(q)
        return position, rotation, jacobian, transforms

    @staticmethod
    def link_transforms(q):
        return torch.eye(
            4, dtype=q.dtype, device=q.device).expand(
                q.shape[0], 1, -1, -1).clone()


class _SafeCollision:
    @staticmethod
    def min_margin(link_transforms):
        return torch.ones(
            link_transforms.shape[0],
            dtype=link_transforms.dtype,
            device=link_transforms.device)


def _task(batch_size):
    return (
        torch.zeros((batch_size, 3), dtype=torch.float32),
        torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32
        ).expand(batch_size, -1).clone(),
        torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32
        ).expand(batch_size, -1).clone(),
    )


class DirectSeedModelTest(unittest.TestCase):
    def test_cycle_sampler_covers_each_epoch_across_batch_boundaries(self):
        sampler = DirectTaskCycleSampler(5, seed=29)
        sampled = torch.cat([
            sampler.sample(3),
            sampler.sample(4),
            sampler.sample(8),
        ])
        self.assertEqual(sampled.device.type, 'cpu')
        self.assertEqual(sampled.dtype, torch.int64)
        for epoch in sampled.reshape(3, 5):
            self.assertTrue(torch.equal(
                epoch.sort().values, torch.arange(5)))
        self.assertEqual(sampler.total_sampled, 15)
        self.assertEqual(sampler.epochs_started, 3)
        self.assertEqual(sampler.cursor, 5)

        same_seed = DirectTaskCycleSampler(5, seed=29)
        self.assertTrue(torch.equal(
            sampled, same_seed.sample(15)))

    def test_cycle_sampler_checkpoint_restores_order_and_private_rng(self):
        sampler = DirectTaskCycleSampler(7, seed=31)
        sampler.sample(9)
        state = sampler.state_dict()
        expected = sampler.sample(23)

        # Neither construction with another seed nor use of the global RNG
        # may perturb the checkpointed sampler's continuation.
        torch.manual_seed(999)
        torch.rand(100)
        restored = DirectTaskCycleSampler(7, seed=12345)
        restored.load_state_dict(state)
        actual = restored.sample(23)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(restored.total_sampled, 32)

        wrong_size = DirectTaskCycleSampler(8)
        with self.assertRaisesRegex(ValueError, 'n_tasks differs'):
            wrong_size.load_state_dict(state)

    def test_task_sampling_cli_defaults_to_random_and_accepts_cycle(self):
        parser = direct_seed_bidir_parser()
        self.assertEqual(parser.parse_args([]).task_sampling, 'random')
        self.assertEqual(
            parser.parse_args(
                ['--task-sampling', 'cycle']).task_sampling,
            'cycle')

    def test_paired_runner_dry_run_and_mutually_exclusive_post_modes(self):
        parser = direct_seed_bidir_parser()
        args = parser.parse_args([
            '--dry-run',
            '--collect-paired-baseline-archive',
            '--deterministic-backward',
            '--freeze-seed-actor-during-collection',
            '--task-sampling', 'cycle',
            '--precision-only-updates-per-rollout', '0',
            '--paired-explorer-checkpoint', 'explorer.pt',
            '--paired-advantage-margin-m', '0.01',
            '--paired-post-updates-per-round', '12',
            '--outer-rounds', '1',
        ])
        validate_direct_seed_bidir_args(args)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            direct_seed_bidir_dry_run(args)
        payload = json.loads(output.getvalue())
        schedule = payload['schedule']
        self.assertTrue(schedule['collect_paired_baseline_archive'])
        self.assertEqual(
            schedule['paired_explorer_checkpoint'], 'explorer.pt')
        self.assertEqual(schedule['paired_post_updates_per_round'], 12)
        self.assertAlmostEqual(
            schedule['paired_advantage_margin_m'], 0.01)
        self.assertEqual(
            schedule['paired_collection_contract']['actor_action'],
            'deterministic-mean')
        self.assertTrue(
            payload['invariants']['paired_post_requires_full_coverage'])

        incompatible = parser.parse_args([
            '--collect-paired-baseline-archive',
            '--deterministic-backward',
            '--freeze-seed-actor-during-collection',
            '--task-sampling', 'cycle',
            '--precision-only-updates-per-rollout', '0',
            '--paired-explorer-checkpoint', 'explorer.pt',
            '--paired-post-updates-per-round', '1',
            '--per-task-elite-post-updates-per-round', '1',
            '--outer-rounds', '1',
        ])
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            validate_direct_seed_bidir_args(incompatible)
        missing_collection = parser.parse_args([
            '--paired-explorer-checkpoint', 'explorer.pt',
            '--paired-post-updates-per-round', '1',
        ])
        with self.assertRaisesRegex(
                ValueError, 'collect-paired-baseline-archive'):
            validate_direct_seed_bidir_args(missing_collection)

    def test_paired_collection_requires_fixed_deterministic_cycle(self):
        parser = direct_seed_bidir_parser()
        common = [
            '--collect-paired-baseline-archive',
            '--precision-only-updates-per-rollout', '0',
        ]
        with self.assertRaisesRegex(ValueError, 'deterministic-backward'):
            validate_direct_seed_bidir_args(parser.parse_args(common))
        with self.assertRaisesRegex(
                ValueError, 'freeze-seed-actor-during-collection'):
            validate_direct_seed_bidir_args(parser.parse_args(
                common + ['--deterministic-backward']))
        with self.assertRaisesRegex(ValueError, 'task-sampling cycle'):
            validate_direct_seed_bidir_args(parser.parse_args(
                common + [
                    '--deterministic-backward',
                    '--freeze-seed-actor-during-collection',
                ]))

        valid = parser.parse_args(common + [
            '--deterministic-backward',
            '--freeze-seed-actor-during-collection',
            '--task-sampling', 'cycle',
        ])
        validate_direct_seed_bidir_args(valid)
        contract = _paired_collection_contract(valid)
        self.assertEqual(
            contract['format'],
            'direct-seed-paired-collection-contract-v1')
        self.assertEqual(contract['task_sampling'], 'cycle')
        self.assertTrue(contract['deterministic_backward'])
        self.assertTrue(contract['actor_frozen_during_collection'])

    def test_paired_post_rejects_multiple_controller_update_rounds(self):
        parser = direct_seed_bidir_parser()
        common = [
            '--collect-paired-baseline-archive',
            '--deterministic-backward',
            '--freeze-seed-actor-during-collection',
            '--task-sampling', 'cycle',
            '--precision-only-updates-per-rollout', '0',
            '--paired-explorer-checkpoint', 'explorer.pt',
            '--paired-post-updates-per-round', '1',
            '--outer-rounds', '2',
        ]
        with self.assertRaisesRegex(
                ValueError, 'cannot span multiple outer rounds'):
            validate_direct_seed_bidir_args(parser.parse_args(common))

        skip = parser.parse_args(common + ['--forward-mode', 'skip'])
        validate_direct_seed_bidir_args(skip)
        zero_steps = parser.parse_args(
            common + ['--controller-steps-per-round', '0'])
        validate_direct_seed_bidir_args(zero_steps)

    def test_paired_archive_requires_full_coverage_before_post(self):
        task_ids = torch.tensor([3, 7], dtype=torch.int64)
        archive = DirectSeedPairedArchive(task_ids)
        with self.assertRaisesRegex(RuntimeError, '0/2 tasks'):
            _require_full_paired_archive(archive)

        task = task_ids.float()[:, None].expand(-1, 9).clone()
        q = torch.zeros((2, 7), dtype=torch.float32)
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q,
            q_projected=q,
            fallback_q=q,
            progress_m=torch.tensor([0.2, 0.3]),
            route=torch.full(
                (2,), ROUTE_REFINED, dtype=torch.int64),
        )
        archive.update(task_ids[:1], DirectSeedRLBatch(
            task=batch.task[:1],
            q_raw=batch.q_raw[:1],
            q_projected=batch.q_projected[:1],
            fallback_q=batch.fallback_q[:1],
            progress_m=batch.progress_m[:1],
            route=batch.route[:1],
        ))
        with self.assertRaisesRegex(RuntimeError, '1/2 tasks'):
            _require_full_paired_archive(archive)
        archive.update(task_ids[1:], DirectSeedRLBatch(
            task=batch.task[1:],
            q_raw=batch.q_raw[1:],
            q_projected=batch.q_projected[1:],
            fallback_q=batch.fallback_q[1:],
            progress_m=batch.progress_m[1:],
            route=batch.route[1:],
        ))
        _require_full_paired_archive(archive)

    def test_paired_resume_provenance_is_strict_and_legacy_safe(self):
        parser = direct_seed_bidir_parser()
        args = parser.parse_args([
            '--collect-paired-baseline-archive',
            '--deterministic-backward',
            '--freeze-seed-actor-during-collection',
            '--task-sampling', 'cycle',
            '--precision-only-updates-per-rollout', '0',
        ])
        contract = _paired_collection_contract(args)
        task_ids = torch.tensor([2, 5], dtype=torch.int64)
        archive = DirectSeedPairedArchive(task_ids)
        task = torch.full((1, 9), 2.0)
        q = torch.zeros((1, 7))
        archive.update(
            task_ids[:1],
            DirectSeedRLBatch(
                task=task,
                q_raw=q,
                q_projected=q,
                fallback_q=q,
                progress_m=torch.tensor([0.2]),
                route=torch.tensor([ROUTE_REFINED]),
            ))
        actor_sha = 'a' * 64
        saved = {
            'paired_archive': archive.state_dict(),
            'paired_collection_contract': contract,
            'paired_baseline_actor_state_sha256': actor_sha,
            'direct_seed': {
                'metadata': {
                    'paired_collection_contract': contract,
                    'paired_baseline_actor_state_sha256': actor_sha,
                },
            },
        }
        restored_contract, restored_sha = (
            _restore_paired_collection_provenance(
                saved,
                requested_contract=contract,
                paired_archive=archive,
                current_actor_state_sha256=actor_sha))
        self.assertEqual(restored_contract, contract)
        self.assertEqual(restored_sha, actor_sha)

        changed_contract = dict(contract)
        changed_contract['seed_tasks_per_update'] += 1
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            _restore_paired_collection_provenance(
                saved,
                requested_contract=changed_contract,
                paired_archive=archive,
                current_actor_state_sha256=actor_sha)
        with self.assertRaisesRegex(ValueError, 'legacy paired archive'):
            _restore_paired_collection_provenance(
                {'paired_archive': archive.state_dict()},
                requested_contract=contract,
                paired_archive=archive,
                current_actor_state_sha256=actor_sha)
        inconsistent = dict(saved)
        inconsistent['direct_seed'] = {
            'metadata': {
                'paired_collection_contract': contract,
                'paired_baseline_actor_state_sha256': 'c' * 64,
            },
        }
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            _restore_paired_collection_provenance(
                inconsistent,
                requested_contract=contract,
                paired_archive=archive,
                current_actor_state_sha256=actor_sha)
        with self.assertRaisesRegex(ValueError, 'partial paired archive'):
            _restore_paired_collection_provenance(
                saved,
                requested_contract=contract,
                paired_archive=archive,
                current_actor_state_sha256='b' * 64)

        second_task = torch.full((1, 9), 5.0)
        archive.update(
            task_ids[1:],
            DirectSeedRLBatch(
                task=second_task,
                q_raw=q,
                q_projected=q,
                fallback_q=q,
                progress_m=torch.tensor([0.3]),
                route=torch.tensor([ROUTE_REFINED]),
            ))
        full_saved = dict(saved)
        full_saved['paired_archive'] = archive.state_dict()
        _, full_baseline_sha = _restore_paired_collection_provenance(
            full_saved,
            requested_contract=contract,
            paired_archive=archive,
            current_actor_state_sha256='b' * 64)
        # A full immutable archive remains tied to its original baseline even
        # after paired post-training has changed the deployed actor.
        self.assertEqual(full_baseline_sha, actor_sha)

        empty_archive = DirectSeedPairedArchive(task_ids)
        empty_saved = dict(saved)
        empty_saved['paired_archive'] = empty_archive.state_dict()
        _, empty_baseline_sha = _restore_paired_collection_provenance(
            empty_saved,
            requested_contract=contract,
            paired_archive=empty_archive,
            current_actor_state_sha256='b' * 64)
        self.assertEqual(empty_baseline_sha, 'b' * 64)

        # Old checkpoints that never contained paired data remain resumable and
        # establish the current actor as the new deterministic baseline.
        nonpaired_contract, nonpaired_sha = (
            _restore_paired_collection_provenance(
                {'paired_archive': None},
                requested_contract=contract,
                paired_archive=DirectSeedPairedArchive(task_ids),
                current_actor_state_sha256='b' * 64))
        self.assertEqual(nonpaired_contract, contract)
        self.assertEqual(nonpaired_sha, 'b' * 64)

    def test_external_paired_explorer_requires_exact_provenance(self):
        task_ids = torch.tensor([11, 22], dtype=torch.int64)
        task = task_ids.float()[:, None].expand(-1, 9).clone()
        q = torch.zeros((2, 7), dtype=torch.float32)
        explorer = DirectSeedEliteMemory(task_ids, seed=101)
        explorer.update(
            task_ids,
            DirectSeedRLBatch(
                task=task,
                q_raw=q,
                q_projected=q + 0.1,
                fallback_q=q,
                progress_m=torch.tensor([0.3, 0.4]),
                route=torch.full(
                    (2,), ROUTE_REFINED, dtype=torch.int64),
            ))
        projection_config = DirectSeedProjectionConfig()
        controller_state = {
            'weight': torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        }
        checkpoint = {
            'format': 'direct-seed-bidirectional-v1',
            'kept_task_indices': task_ids.tolist(),
            'safe_task_fingerprint_list_sha256': 'a' * 64,
            'projection_config': dataclasses.asdict(projection_config),
            'controller': controller_state,
            'controller_update_count': 3,
            'per_task_elite_controller_update_count': 3,
            'per_task_elite_memory': explorer.state_dict(),
        }
        api = _load_direct_seed_rl_api(required=True)
        self.assertIsNotNone(api)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'explorer.pt'
            torch.save(checkpoint, path)
            loaded, provenance = _load_paired_explorer_checkpoint(
                path, api,
                task_ids=task_ids,
                task=task,
                safe_task_fingerprint_list_sha256='a' * 64,
                projection_config=projection_config,
                controller_state=controller_state)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(
                provenance['controller_update_count'], 3)
            self.assertEqual(
                provenance['explorer_elite_size'], 2)

            with self.assertRaisesRegex(ValueError, 'task fingerprint'):
                _load_paired_explorer_checkpoint(
                    path, api,
                    task_ids=task_ids,
                    task=task,
                    safe_task_fingerprint_list_sha256='b' * 64,
                    projection_config=projection_config,
                    controller_state=controller_state)
            with self.assertRaisesRegex(ValueError, 'projection config'):
                _load_paired_explorer_checkpoint(
                    path, api,
                    task_ids=task_ids,
                    task=task,
                    safe_task_fingerprint_list_sha256='a' * 64,
                    projection_config=DirectSeedProjectionConfig(
                        position_tol_m=4e-3),
                    controller_state=controller_state)
            with self.assertRaisesRegex(ValueError, 'controller state'):
                _load_paired_explorer_checkpoint(
                    path, api,
                    task_ids=task_ids,
                    task=task,
                    safe_task_fingerprint_list_sha256='a' * 64,
                    projection_config=projection_config,
                    controller_state={
                        'weight': torch.zeros((1, 2)),
                    })

    def test_one_forward_produces_one_strictly_interior_q(self):
        torch.manual_seed(3)
        model = DirectSeedGenerator(
            -torch.ones(7), torch.ones(7),
            DirectSeedConfig(hidden_dim=16, n_hidden_layers=2))
        task = torch.randn((5, 9))
        q = model(task)
        self.assertEqual(tuple(q.shape), (5, 7))
        self.assertTrue(bool((q < 0.98).all()))
        self.assertTrue(bool((q > -0.98).all()))

        loaded, payload = load_direct_seed_generator(
            direct_seed_checkpoint(model))
        self.assertEqual(payload['format'], 'direct-seed-generator-v1')
        self.assertTrue(torch.equal(model(task), loaded(task)))

    def test_hard_moe_deploys_one_deterministic_expert_output(self):
        torch.manual_seed(4)
        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=2, n_experts=3))
        task = torch.randn((6, 9))
        expert_q, gate_logits = actor.expert_q_and_gate(task)
        first = actor(task)
        second = actor(task)
        expert_index = actor.expert_index(task)
        expected = expert_q.gather(
            1, expert_index[:, None, None].expand(-1, 1, 7)
        ).squeeze(1)
        self.assertEqual(tuple(expert_q.shape), (6, 3, 7))
        self.assertEqual(tuple(gate_logits.shape), (6, 3))
        self.assertEqual(tuple(first.shape), (6, 7))
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(
            first, expected, rtol=1e-6, atol=1e-7)
        self.assertTrue(bool((first < 0.98).all()))
        self.assertTrue(bool((first > -0.98).all()))
        # Zero-initialized gate ties have one deterministic expert-0 winner.
        self.assertTrue(bool((expert_index == 0).all()))
        # Hard routing backpropagates through only the selected head.  The
        # tensorized implementation performs one batched head multiply and
        # never needs a CUDA-synchronizing Python ``bool(mask.any())``.
        actor.zero_grad(set_to_none=True)
        first.sum().backward()
        self.assertGreater(
            float(actor.experts[0].weight.grad.abs().sum()), 0.0)
        for expert in actor.experts[1:]:
            self.assertIsNotNone(expert.weight.grad)
            self.assertEqual(
                float(expert.weight.grad.abs().sum()), 0.0)

    def test_single_actor_to_moe_conversion_preserves_deployment(self):
        torch.manual_seed(5)
        single = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=2))
        single.eval()
        task = torch.randn((7, 9))
        expected = single.mean_q(task)

        exact = direct_seed_moe_from_actor(
            single, n_experts=3, expert_perturb_std=0.0, seed=101)
        expert_q = exact.expert_q(task)
        self.assertTrue(exact.config.exact_baseline_head)
        self.assertEqual(exact.experts[0].out_features, 14)
        self.assertTrue(torch.equal(
            exact.experts[0].weight, single.trunk[-1].weight))
        self.assertTrue(torch.equal(
            exact.experts[0].bias, single.trunk[-1].bias))
        # Keeping the complete 14-output head makes the baseline path exactly
        # equal, rather than merely close after a changed-width GEMM.
        self.assertTrue(torch.equal(exact(task), expected))
        torch.testing.assert_close(
            expert_q[:, 0], expert_q[:, 1], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            expert_q[:, 1], expert_q[:, 2], rtol=1e-6, atol=1e-7)

        perturbed_a = direct_seed_moe_from_actor(
            single, n_experts=3, expert_perturb_std=1e-3, seed=103)
        torch.manual_seed(999)
        torch.rand(100)
        perturbed_b = direct_seed_moe_from_actor(
            single, n_experts=3, expert_perturb_std=1e-3, seed=103)
        self.assertTrue(torch.equal(perturbed_a(task), expected))
        self.assertTrue(torch.equal(
            perturbed_a.experts[0].weight, single.trunk[-1].weight))
        self.assertTrue(torch.equal(
            perturbed_a.experts[0].bias, single.trunk[-1].bias))
        self.assertTrue(bool(
            (perturbed_a.expert_index(task) == 0).all()))
        self.assertFalse(torch.equal(
            perturbed_a.expert_q(task)[:, 0],
            perturbed_a.expert_q(task)[:, 1]))
        for name, value in perturbed_a.state_dict().items():
            self.assertTrue(torch.equal(
                value, perturbed_b.state_dict()[name]))

    def test_exact_moe_preserves_baseline_rows_in_mixed_batch(self):
        torch.manual_seed(51)
        single = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=2))
        single.eval()
        actor = direct_seed_moe_from_actor(
            single, n_experts=3, expert_perturb_std=1e-2, seed=151)
        task = torch.randn((11, 9))
        expected = single.mean_q(task)

        # Make a deterministic linear separator between two rows, forcing a
        # genuinely mixed batch without changing the shared baseline trunk.
        with torch.no_grad():
            features = actor._features(task)
            direction = features[0] - features[1]
            midpoint = 0.5 * (features[0] + features[1])
            offset = torch.dot(direction, midpoint)
            actor.gate.weight.zero_()
            actor.gate.bias.fill_(-100.0)
            actor.gate.weight[0].copy_(direction)
            actor.gate.bias[0] = -offset
            actor.gate.weight[1].copy_(-direction)
            actor.gate.bias[1] = offset

        expert_index = actor.expert_index(task)
        baseline = expert_index == 0
        specialist = expert_index != 0
        self.assertTrue(bool(baseline.any()))
        self.assertTrue(bool(specialist.any()))
        actual = actor.mean_q(task)
        self.assertTrue(torch.equal(actual[baseline], expected[baseline]))

    def test_hard_moe_wta_projection_updates_experts_and_gate(self):
        torch.manual_seed(6)
        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=3))
        with torch.no_grad():
            for expert, q_value in zip(
                    actor.experts, (-0.5, 0.0, 0.5)):
                expert.weight.zero_()
                raw_value = np.arctanh(q_value / 0.98)
                expert.bias.fill_(float(raw_value))
            actor.gate.weight.zero_()
            actor.gate.bias.zero_()

        task = torch.zeros((12, 9))
        task[:, 0] = torch.tensor(
            [-1.0] * 4 + [0.0] * 4 + [1.0] * 4)
        target_value = torch.tensor(
            [-0.7] * 4 + [0.1] * 4 + [0.7] * 4)
        target = target_value[:, None].expand(-1, 7).clone()
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=torch.zeros_like(target),
            q_projected=target,
            fallback_q=torch.zeros_like(target),
            progress_m=torch.linspace(0.1, 0.4, 12),
            route=torch.full(
                (12,), ROUTE_REFINED, dtype=torch.int64),
        )
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
        expert_before = [
            {
                name: value.detach().clone()
                for name, value in expert.state_dict().items()
            }
            for expert in actor.experts
        ]
        gate_before = {
            name: value.detach().clone()
            for name, value in actor.gate.state_dict().items()
        }
        metrics = update_direct_seed_moe_projection(
            actor, optimizer, batch,
            gate_ce_weight=1.0, load_balance_weight=0.01)
        self.assertTrue(all(
            np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics['moe_actor_updated'], 1.0)
        for index in range(3):
            self.assertAlmostEqual(
                metrics[f'moe_winner_expert_{index}_fraction'],
                1 / 3)
            self.assertTrue(any(
                not torch.equal(
                    expert_before[index][name], value)
                for name, value in
                actor.experts[index].state_dict().items()))
        self.assertTrue(any(
            not torch.equal(gate_before[name], value)
            for name, value in actor.gate.state_dict().items()))

        no_refined = DirectSeedRLBatch(
            task=task,
            q_raw=torch.zeros_like(target),
            q_projected=target,
            fallback_q=torch.zeros_like(target),
            progress_m=torch.zeros(12),
            route=torch.full(
                (12,), ROUTE_FALLBACK, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, 'ROUTE_REFINED'):
            update_direct_seed_moe_projection(
                actor, optimizer, no_refined)

    def test_advantage_moe_freezes_baseline_and_updates_specialist_gate(self):
        torch.manual_seed(61)
        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=3))
        with torch.no_grad():
            for expert, q_value in zip(
                    actor.experts, (0.0, -0.45, 0.45)):
                expert.weight.zero_()
                expert.bias.fill_(
                    float(np.arctanh(q_value / 0.98)))
            actor.gate.weight.zero_()
            actor.gate.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
        for parameter in actor.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.experts[0].parameters():
            parameter.requires_grad_(False)

        task = torch.randn((10, 9))
        target = torch.full((10, 7), -0.9)
        target[4:] = 0.7
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=torch.zeros_like(target),
            q_projected=target,
            fallback_q=torch.zeros_like(target),
            progress_m=torch.linspace(0.1, 0.5, 10),
            route=torch.full(
                (10,), ROUTE_REFINED, dtype=torch.int64),
        )
        explorer_selected = torch.tensor(
            [False] * 4 + [True] * 6)
        trainable = (
            list(actor.gate.parameters())
            + [
                parameter
                for expert in actor.experts[1:]
                for parameter in expert.parameters()
            ])
        optimizer = torch.optim.Adam(trainable, lr=1e-2)
        trunk_before = {
            name: value.detach().clone()
            for name, value in actor.trunk.state_dict().items()
        }
        baseline_before = {
            name: value.detach().clone()
            for name, value in actor.experts[0].state_dict().items()
        }
        specialists_before = [
            {
                name: value.detach().clone()
                for name, value in expert.state_dict().items()
            }
            for expert in actor.experts[1:]
        ]
        gate_before = {
            name: value.detach().clone()
            for name, value in actor.gate.state_dict().items()
        }

        metrics = update_direct_seed_moe_advantage(
            actor, optimizer, batch, explorer_selected,
            gate_ce_weight=1.0,
            positive_gate_weight=2.0,
            specialist_load_balance_weight=0.01)
        self.assertTrue(all(
            np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics['moe_advantage_actor_updated'], 1.0)
        self.assertEqual(metrics['moe_advantage_positive_count'], 6.0)
        self.assertEqual(metrics['moe_advantage_negative_count'], 4.0)
        for name, value in actor.trunk.state_dict().items():
            self.assertTrue(torch.equal(value, trunk_before[name]))
        for name, value in actor.experts[0].state_dict().items():
            self.assertTrue(torch.equal(value, baseline_before[name]))
        self.assertTrue(any(
            not torch.equal(
                specialists_before[index][name], value)
            for index, expert in enumerate(actor.experts[1:])
            for name, value in expert.state_dict().items()))
        self.assertTrue(any(
            not torch.equal(gate_before[name], value)
            for name, value in actor.gate.state_dict().items()))

    def test_advantage_moe_all_negative_is_stable_gate_only_update(self):
        torch.manual_seed(62)
        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=2))
        for parameter in actor.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.experts[0].parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            actor.gate.weight.zero_()
            actor.gate.bias.copy_(torch.tensor([0.0, 1.0]))
        trainable = (
            list(actor.gate.parameters())
            + list(actor.experts[1].parameters()))
        optimizer = torch.optim.Adam(trainable, lr=1e-2)
        task = torch.randn((6, 9))
        target = torch.full((6, 7), 0.9)
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=torch.zeros_like(target),
            q_projected=target,
            fallback_q=torch.zeros_like(target),
            progress_m=torch.zeros(6),
            route=torch.full(
                (6,), ROUTE_FALLBACK, dtype=torch.int64),
        )
        specialist_before = {
            name: value.detach().clone()
            for name, value in actor.experts[1].state_dict().items()
        }
        gate_before = {
            name: value.detach().clone()
            for name, value in actor.gate.state_dict().items()
        }

        metrics = update_direct_seed_moe_advantage(
            actor, optimizer, batch,
            torch.zeros(6, dtype=torch.bool))
        self.assertTrue(all(
            np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics['moe_advantage_positive_count'], 0.0)
        self.assertEqual(metrics['moe_advantage_imitation_loss'], 0.0)
        self.assertEqual(metrics['moe_advantage_load_balance_loss'], 0.0)
        self.assertEqual(
            metrics['moe_advantage_winner_expert_1_fraction'], 0.0)
        for name, value in actor.experts[1].state_dict().items():
            self.assertTrue(torch.equal(value, specialist_before[name]))
        self.assertTrue(any(
            not torch.equal(gate_before[name], value)
            for name, value in actor.gate.state_dict().items()))

    def test_advantage_moe_rejects_unsafe_training_contracts(self):
        torch.manual_seed(63)
        task = torch.randn((3, 9))
        target = torch.zeros((3, 7))
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=torch.zeros_like(target),
            q_projected=target,
            fallback_q=torch.zeros_like(target),
            progress_m=torch.zeros(3),
            route=torch.full(
                (3,), ROUTE_REFINED, dtype=torch.int64),
        )
        selected = torch.tensor([False, True, False])

        one_expert = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=1))
        for parameter in one_expert.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in one_expert.experts[0].parameters():
            parameter.requires_grad_(False)
        one_optimizer = torch.optim.Adam(
            one_expert.gate.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, 'at least two experts'):
            update_direct_seed_moe_advantage(
                one_expert, one_optimizer, batch, selected)

        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=2))
        unsafe_optimizer = torch.optim.Adam(
            actor.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, 'frozen trunk'):
            update_direct_seed_moe_advantage(
                actor, unsafe_optimizer, batch, selected)

        for parameter in actor.trunk.parameters():
            parameter.requires_grad_(False)
        for parameter in actor.experts[0].parameters():
            parameter.requires_grad_(False)
        exact_trainable = (
            list(actor.gate.parameters())
            + list(actor.experts[1].parameters()))
        exact_optimizer = torch.optim.Adam(
            exact_trainable, lr=1e-3)
        with self.assertRaisesRegex(TypeError, 'dtype torch.bool'):
            update_direct_seed_moe_advantage(
                actor, exact_optimizer, batch, selected.to(torch.int64))
        with self.assertRaisesRegex(ValueError, r'shape \(B,\)'):
            update_direct_seed_moe_advantage(
                actor, exact_optimizer, batch, selected[:, None])
        gate_only_optimizer = torch.optim.Adam(
            actor.gate.parameters(), lr=1e-3)
        with self.assertRaisesRegex(ValueError, 'exactly'):
            update_direct_seed_moe_advantage(
                actor, gate_only_optimizer, batch, selected)
        with self.assertRaisesRegex(ValueError, 'batch dtype'):
            update_direct_seed_moe_advantage(
                actor, exact_optimizer,
                batch.to('cpu', torch.float64), selected)
        with self.assertRaisesRegex(
                ValueError, 'finite and non-negative'):
            update_direct_seed_moe_advantage(
                actor, exact_optimizer, batch, selected,
                positive_gate_weight=float('nan'))

    def test_hard_moe_checkpoint_and_generic_deployment_loader(self):
        torch.manual_seed(7)
        single = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        actor = direct_seed_moe_from_actor(
            single, n_experts=2, expert_perturb_std=1e-3, seed=107)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        task = torch.randn((5, 9))
        expected = actor(task)
        payload = direct_seed_moe_checkpoint(
            actor, update_step=11, actor_optimizer=optimizer,
            metadata={'kind': 'hard-moe-unit'})

        loaded, optimizer_state, retained = (
            load_direct_seed_moe_checkpoint(payload))
        self.assertTrue(torch.equal(expected, loaded(task)))
        self.assertIsNotNone(optimizer_state)
        self.assertEqual(retained['update_step'], 11)
        self.assertEqual(
            retained['metadata']['kind'], 'hard-moe-unit')
        deployed, deployed_payload = load_deployment_generator(payload)
        self.assertTrue(torch.equal(expected, deployed(task)))
        self.assertEqual(
            deployed_payload['format'], 'direct-seed-hard-moe-v1')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'moe.pt'
            torch.save(payload, path)
            from_path, _, _ = load_direct_seed_moe_checkpoint(path)
            self.assertTrue(torch.equal(expected, from_path(task)))

    def test_nonlinear_moe_gate_is_exactly_hard_and_checkpointable(self):
        torch.manual_seed(72)
        single = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=2))
        single.eval()
        source = direct_seed_moe_from_actor(
            single, n_experts=3, expert_perturb_std=0.02, seed=172)
        task = torch.randn((24, 9))
        features = source._features(task).detach().numpy()
        target = np.zeros((len(task), 2), dtype=np.float32)
        model, scaler, _ = _fit_advantage_mlp(
            features, target, hidden_dim=8, epochs=1,
            batch_size=16, model_seed=173, shuffle_seed=174)
        # Construct a deterministic nonlinear separator in standardized
        # hidden-feature space. SiLU(z) is positive iff z is positive, so
        # expert 0 and expert 1 are both selected while expert 2 is disabled.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.network[0].weight[0, 0] = 1.0
            model.network[2].weight[0, 0] = 1.0
            model.network[2].bias[1] = -100.0
        actor = _nonlinear_gate_actor(
            source, model, scaler,
            threshold_m=0.0, gate_hidden_dim=8)
        index = actor.expert_index(task)
        self.assertTrue(bool((index == 0).any()))
        self.assertTrue(bool((index == 1).any()))
        self.assertFalse(bool((index == 2).any()))
        logits = actor.gate_logits(task)
        self.assertTrue(bool((logits[:, 0] == 0.0).all()))
        expected_advantage = _predict_advantage(
            model, scaler, features)
        torch.testing.assert_close(
            logits[:, 1:],
            torch.from_numpy(expected_advantage).to(logits),
            rtol=2e-5, atol=2e-6)

        all_experts = actor.expert_q(task)
        expected_q = all_experts.gather(
            1, index[:, None, None].expand(-1, 1, 7)
        ).squeeze(1)
        torch.testing.assert_close(
            actor.mean_q(task), expected_q, rtol=1e-6, atol=1e-7)
        baseline_rows = index == 0
        self.assertTrue(torch.equal(
            actor.mean_q(task)[baseline_rows],
            single.mean_q(task)[baseline_rows]))

        payload = direct_seed_moe_checkpoint(actor, update_step=13)
        loaded, _, retained = load_direct_seed_moe_checkpoint(payload)
        self.assertEqual(loaded.config.gate_hidden_dim, 8)
        self.assertIsInstance(loaded.gate, torch.nn.Sequential)
        self.assertTrue(torch.equal(actor.mean_q(task), loaded.mean_q(task)))
        self.assertEqual(
            retained['actor_config']['gate_hidden_dim'], 8)

    def test_legacy_hard_moe_checkpoint_without_exact_field_loads(self):
        torch.manual_seed(71)
        actor = DirectSeedMoEActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedMoEActorConfig(
                hidden_dim=16, n_hidden_layers=1, n_experts=3))
        task = torch.randn((4, 9))
        expected = actor(task)
        payload = direct_seed_moe_checkpoint(actor, update_step=3)
        payload['actor_config'].pop('exact_baseline_head')
        payload['actor_config'].pop('gate_hidden_dim')

        loaded, optimizer_state, retained = (
            load_direct_seed_moe_checkpoint(payload))
        self.assertIsNone(optimizer_state)
        self.assertFalse(loaded.config.exact_baseline_head)
        self.assertEqual(loaded.config.gate_hidden_dim, 0)
        self.assertIsInstance(loaded.gate, torch.nn.Linear)
        self.assertTrue(all(
            expert.out_features == 7 for expert in loaded.experts))
        self.assertTrue(torch.equal(expected, loaded(task)))
        self.assertNotIn(
            'exact_baseline_head', retained['actor_config'])
        self.assertNotIn(
            'gate_hidden_dim', retained['actor_config'])

    def test_soft_nearest_support_is_mode_seeking_and_differentiable(self):
        candidates = torch.stack([
            torch.zeros(7),
            torch.full((7,), 0.5),
        ]).unsqueeze(0)
        returns = torch.tensor([[0.0, 0.10]])
        valid = torch.ones((1, 2), dtype=torch.bool)
        q_half = torch.ones(7)
        low = return_weighted_soft_nearest_support_loss(
            torch.zeros((1, 7)), candidates, returns, valid, q_half,
            return_temperature_m=0.02, support_temperature=0.05)
        predicted = torch.full((1, 7), 0.5, requires_grad=True)
        high = return_weighted_soft_nearest_support_loss(
            predicted, candidates, returns, valid, q_half,
            return_temperature_m=0.02, support_temperature=0.05)
        self.assertLess(float(high.detach()), float(low))
        high.backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_geometry_split_is_complete_deterministic_and_disjoint(self):
        # Three duplicated geometries: duplicates must remain in one partition.
        p0 = torch.tensor([
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0], [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0], [0.2, 0.0, 0.0],
        ])
        batch_size = len(p0)
        batch = SeedCandidateBatch(
            q0=torch.zeros((batch_size, 2, 7)),
            p0=p0,
            line_dir=torch.tensor(
                [[1.0, 0.0, 0.0]]).expand(batch_size, -1).clone(),
            n_target=torch.tensor(
                [[0.0, 0.0, 1.0]]).expand(batch_size, -1).clone(),
            valid=torch.ones((batch_size, 2), dtype=torch.bool))
        dataset = CachedSeedCandidateDataset(batch)
        split_a = geometry_grouped_three_way_split(
            dataset, model_fraction=0.25,
            calibration_fraction=0.25, seed=9)
        split_b = geometry_grouped_three_way_split(
            dataset, model_fraction=0.25,
            calibration_fraction=0.25, seed=9)
        for left, right in zip(split_a, split_b):
            self.assertTrue(torch.equal(left, right))
        self.assertEqual(torch.cat(split_a).unique().numel(), batch_size)
        fingerprints = dataset.task_fingerprints
        sets = [
            {fingerprints[int(row)] for row in index}
            for index in split_a
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_contextual_rl_synthetic_update_and_checkpoint(self):
        stats = synthetic_direct_seed_rl_smoke(include_collision=True)
        self.assertIn('target_progress_mean_m', stats)
        self.assertIn('projection_distill_loss', stats)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
        self.assertAlmostEqual(
            stats['route_direct_fraction'], 3 / 8)
        self.assertAlmostEqual(
            stats['route_refined_fraction'], 2 / 8)
        self.assertEqual(stats['collision_precision_available'], 1.0)

        torch.manual_seed(4)
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        critic = TwinMacroQ(
            -torch.ones(7), torch.ones(7),
            DirectSeedCriticConfig(hidden_dim=16, n_hidden_layers=1))
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        task = torch.randn((3, 9))
        expected = actor.mean_q(task)
        payload = direct_seed_rl_checkpoint(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            config=DirectSeedRLConfig(),
            update_step=7,
            metadata={'kind': 'unit'})
        loaded_actor, loaded_critic, actor_state, critic_state, loaded = (
            load_direct_seed_rl_checkpoint(payload))
        self.assertTrue(torch.equal(expected, loaded_actor.mean_q(task)))
        q1, q2 = loaded_critic(task, expected)
        self.assertEqual(tuple(q1.shape), (3,))
        self.assertEqual(tuple(q2.shape), (3,))
        self.assertIsNotNone(actor_state)
        self.assertIsNotNone(critic_state)
        self.assertEqual(loaded['update_step'], 7)
        self.assertEqual(loaded['metadata']['kind'], 'unit')
        deploy_actor, deploy_payload = load_deployment_generator(payload)
        self.assertTrue(torch.equal(expected, deploy_actor(task)))
        self.assertEqual(
            deploy_payload['format'], 'direct-seed-contextual-rl-v1')
        legacy_payload = dict(payload)
        legacy_payload['rl_config'] = {
            name: value
            for name, value in payload['rl_config'].items()
            if name != 'behavior_anchor_weight'
        }
        _, _, _, _, legacy_loaded = load_direct_seed_rl_checkpoint(
            legacy_payload)
        self.assertNotIn(
            'behavior_anchor_weight', legacy_loaded['rl_config'])

    def test_macro_replay_ring_checkpoint_and_private_rng(self):
        def batch(first, count):
            value = torch.arange(
                first, first + count, dtype=torch.float32)
            task = value[:, None].expand(-1, 9).clone()
            q_raw = value[:, None].expand(-1, 7).clone() / 100.0
            return DirectSeedRLBatch(
                task=task,
                q_raw=q_raw,
                q_projected=q_raw + 0.01,
                fallback_q=torch.zeros_like(q_raw),
                progress_m=value / 10.0,
                route=torch.full(
                    (count,), ROUTE_DIRECT, dtype=torch.int64),
            )

        replay = DirectSeedMacroReplay(5, 'cpu', seed=41)
        replay.add(batch(0, 3))
        replay.add(batch(10, 4))
        self.assertEqual(len(replay), 5)
        self.assertEqual(replay.write_index, 2)
        self.assertEqual(replay.total_added, 7)
        self.assertEqual(
            set((replay.progress_m * 10).tolist()),
            {2.0, 10.0, 11.0, 12.0, 13.0})
        state = replay.state_dict()
        expected = replay.sample(16)

        restored = DirectSeedMacroReplay(5, 'cpu', seed=999)
        restored.load_state_dict(state)
        actual = restored.sample(16)
        for name in (
                'task', 'q_raw', 'q_projected',
                'fallback_q', 'progress_m', 'route'):
            self.assertTrue(torch.equal(
                getattr(expected, name), getattr(actual, name)))
        self.assertEqual(restored.total_sampled, 16)
        self.assertEqual(restored.task.device.type, 'cpu')
        restored.clear()
        self.assertEqual(len(restored), 0)
        self.assertEqual(restored.write_index, 0)
        self.assertEqual(restored.total_added, 7)
        with self.assertRaisesRegex(RuntimeError, 'empty replay'):
            restored.sample(1)

    def test_macro_replay_elite_uses_only_top_refined_returns(self):
        replay = DirectSeedMacroReplay(6, 'cpu', seed=53)
        value = torch.arange(6, dtype=torch.float32)
        q = value[:, None].expand(-1, 7).clone() / 10.0
        replay.add(DirectSeedRLBatch(
            task=value[:, None].expand(-1, 9).clone(),
            q_raw=q,
            q_projected=q + 0.01,
            fallback_q=torch.zeros_like(q),
            progress_m=torch.tensor(
                [0.1, 0.4, 9.0, 0.3, 0.2, 8.0]),
            route=torch.tensor([
                ROUTE_REFINED, ROUTE_REFINED, ROUTE_DIRECT,
                ROUTE_REFINED, ROUTE_REFINED, ROUTE_FALLBACK,
            ], dtype=torch.int64),
        ))
        sampled = replay.sample_elite(128, elite_fraction=0.5)
        # Top half of four refined entries is task 1 (0.4 m) and task 3
        # (0.3 m); very high non-refined returns remain ineligible.
        self.assertEqual(
            set(sampled.task[:, 0].tolist()), {1.0, 3.0})
        self.assertTrue(bool(
            (sampled.route == ROUTE_REFINED).all()))
        self.assertEqual(replay.total_sampled, 128)

        no_refined = DirectSeedMacroReplay(2, 'cpu')
        no_refined.add(DirectSeedRLBatch(
            task=torch.zeros((2, 9)),
            q_raw=torch.zeros((2, 7)),
            q_projected=torch.zeros((2, 7)),
            fallback_q=torch.zeros((2, 7)),
            progress_m=torch.ones(2),
            route=torch.tensor(
                [ROUTE_DIRECT, ROUTE_FALLBACK], dtype=torch.int64),
        ))
        with self.assertRaisesRegex(RuntimeError, 'ROUTE_REFINED'):
            no_refined.sample_elite(1, 0.5)
        with self.assertRaisesRegex(TypeError, 'integer'):
            replay.sample_elite(1.5, 0.5)
        with self.assertRaisesRegex(ValueError, r'\(0, 1\]'):
            replay.sample_elite(1, 0.0)
        with self.assertRaisesRegex(ValueError, r'\(0, 1\]'):
            replay.sample_elite(1, float('nan'))

    def test_per_task_elite_memory_keeps_max_and_ignores_non_refined(self):
        memory = DirectSeedEliteMemory(
            torch.tensor([101, 202, 303]), seed=61)
        row = torch.arange(5, dtype=torch.float32)
        q = row[:, None].expand(-1, 7).clone()
        inserted = memory.update(
            torch.tensor([101, 101, 101, 202, 303]),
            DirectSeedRLBatch(
                task=row[:, None].expand(-1, 9).clone(),
                q_raw=torch.zeros_like(q),
                q_projected=q,
                fallback_q=torch.zeros_like(q),
                progress_m=torch.tensor([0.2, 0.7, 99.0, 0.4, 88.0]),
                route=torch.tensor([
                    ROUTE_REFINED, ROUTE_REFINED, ROUTE_DIRECT,
                    ROUTE_FALLBACK, ROUTE_INVALID,
                ], dtype=torch.int64),
            ))
        self.assertEqual(inserted, 1)
        self.assertEqual(len(memory), 1)
        self.assertAlmostEqual(memory.coverage, 1 / 3)
        self.assertAlmostEqual(float(memory.progress_m[0]), 0.7)
        self.assertTrue(torch.equal(
            memory.q_projected[0], torch.ones(7)))

        # A lower cross-call result cannot replace task 101; duplicate task
        # 202 rows are reduced to the highest real refined progress.
        row = torch.tensor([6.0, 7.0, 8.0])
        q = row[:, None].expand(-1, 7).clone()
        improved = memory.update(
            torch.tensor([101, 202, 202]),
            DirectSeedRLBatch(
                task=row[:, None].expand(-1, 9).clone(),
                q_raw=torch.zeros_like(q),
                q_projected=q,
                fallback_q=torch.zeros_like(q),
                progress_m=torch.tensor([0.6, 0.3, 0.5]),
                route=torch.full(
                    (3,), ROUTE_REFINED, dtype=torch.int64),
            ))
        self.assertEqual(improved, 1)
        self.assertEqual(len(memory), 2)
        self.assertAlmostEqual(memory.coverage, 2 / 3)
        self.assertAlmostEqual(float(memory.progress_m[0]), 0.7)
        self.assertAlmostEqual(float(memory.progress_m[1]), 0.5)
        self.assertTrue(torch.equal(
            memory.q_projected[0], torch.ones(7)))
        self.assertTrue(torch.equal(
            memory.q_projected[1], torch.full((7,), 8.0)))
        self.assertFalse(bool(memory.valid[2]))

    def test_per_task_elite_sampling_is_uniform_and_returns_legal_batch(self):
        memory = DirectSeedEliteMemory(
            torch.tensor([10, 20, 30, 40]), seed=67)
        task_marker = torch.tensor([10.0, 20.0, 30.0])
        task = task_marker[:, None].expand(-1, 9).clone()
        q = (task_marker / 100.0)[:, None].expand(-1, 7).clone()
        memory.update(
            torch.tensor([10, 20, 30]),
            DirectSeedRLBatch(
                task=task,
                q_raw=torch.zeros_like(q),
                q_projected=q,
                fallback_q=torch.zeros_like(q),
                progress_m=torch.tensor([0.1, 0.2, 0.3]),
                route=torch.full(
                    (3,), ROUTE_REFINED, dtype=torch.int64),
            ))
        sampled = memory.sample(6000)
        self.assertEqual(sampled.batch_size, 6000)
        self.assertEqual(sampled.task.device.type, 'cpu')
        self.assertEqual(sampled.task.dtype, torch.float32)
        self.assertTrue(bool(
            (sampled.route == ROUTE_REFINED).all()))
        self.assertTrue(torch.equal(sampled.q_raw, sampled.q_projected))
        self.assertTrue(bool((sampled.fallback_q == 0.0).all()))
        expected_progress = sampled.task[:, 0] / 100.0
        self.assertTrue(torch.allclose(
            sampled.progress_m, expected_progress))
        self.assertTrue(torch.allclose(
            sampled.q_projected[:, 0], expected_progress))

        # Uniformity is over valid task slots.  A 10% band around expectation
        # is far wider than normal seeded sampling variation and non-flaky.
        _, counts = torch.unique(
            sampled.task[:, 0], return_counts=True)
        self.assertEqual(counts.numel(), 3)
        self.assertTrue(bool(((counts - 2000).abs() < 200).all()))

        converted = memory.sample(3, dtype=torch.float64)
        self.assertEqual(converted.task.dtype, torch.float64)
        self.assertEqual(converted.route.dtype, torch.int64)

    def test_per_task_elite_checkpoint_preserves_private_rng(self):
        memory = DirectSeedEliteMemory(
            torch.tensor([2, 4, 8]), seed=71)
        value = torch.tensor([2.0, 4.0, 8.0])
        q = value[:, None].expand(-1, 7).clone()
        memory.update(
            torch.tensor([2, 4, 8]),
            DirectSeedRLBatch(
                task=value[:, None].expand(-1, 9).clone(),
                q_raw=torch.zeros_like(q),
                q_projected=q,
                fallback_q=torch.zeros_like(q),
                progress_m=value / 10.0,
                route=torch.full(
                    (3,), ROUTE_REFINED, dtype=torch.int64),
            ))
        memory.sample(17)
        state = memory.state_dict()
        expected = memory.sample(128)

        restored = DirectSeedEliteMemory(
            torch.tensor([2, 4, 8]), seed=999)
        restored.load_state_dict(state)
        actual = restored.sample(128)
        for name in (
                'task', 'q_raw', 'q_projected',
                'fallback_q', 'progress_m', 'route'):
            self.assertTrue(torch.equal(
                getattr(expected, name), getattr(actual, name)))
        self.assertEqual(len(restored), 3)
        self.assertEqual(restored.coverage, 1.0)

        wrong_ids = DirectSeedEliteMemory(torch.tensor([2, 4, 9]))
        with self.assertRaisesRegex(ValueError, 'task_ids differ'):
            wrong_ids.load_state_dict(state)

    def test_per_task_elite_clear_removes_targets(self):
        memory = DirectSeedEliteMemory(torch.tensor([5, 6]), seed=73)
        q = torch.ones((1, 7))
        memory.update(
            torch.tensor([5]),
            DirectSeedRLBatch(
                task=torch.ones((1, 9)),
                q_raw=torch.zeros_like(q),
                q_projected=q,
                fallback_q=torch.zeros_like(q),
                progress_m=torch.tensor([0.5]),
                route=torch.tensor([ROUTE_REFINED]),
            ))
        memory.clear()
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.coverage, 0.0)
        self.assertTrue(bool((memory.progress_m == 0.0).all()))
        self.assertTrue(bool((memory.task == 0.0).all()))
        self.assertTrue(bool((memory.q_projected == 0.0).all()))
        self.assertFalse(bool(memory.valid.any()))
        with self.assertRaisesRegex(RuntimeError, 'empty elite memory'):
            memory.sample(1)

    def test_paired_archive_keeps_first_outcome_unless_overwritten(self):
        archive = DirectSeedPairedArchive(
            torch.tensor([10, 20, 30]), seed=79)
        task_marker = torch.tensor([10.0, 10.0, 20.0])
        task = task_marker[:, None].expand(-1, 9).clone()
        projected_marker = torch.tensor([1.0, 9.0, 2.0])
        projected = projected_marker[:, None].expand(-1, 7).clone()
        inserted = archive.update(
            torch.tensor([10, 10, 20]),
            DirectSeedRLBatch(
                task=task,
                q_raw=torch.zeros_like(projected),
                q_projected=projected,
                fallback_q=torch.zeros_like(projected),
                progress_m=torch.tensor([0.1, 0.9, 0.2]),
                route=torch.tensor([
                    ROUTE_REFINED, ROUTE_FALLBACK, ROUTE_DIRECT,
                ], dtype=torch.int64),
            ))
        self.assertEqual(inserted, 2)
        self.assertEqual(len(archive), 2)
        self.assertAlmostEqual(archive.coverage, 2 / 3)
        self.assertAlmostEqual(float(archive.progress_m[0]), 0.1)
        self.assertEqual(int(archive.route[0]), ROUTE_REFINED)
        self.assertTrue(torch.equal(
            archive.q_projected[0], torch.ones(7)))

        replacement = DirectSeedRLBatch(
            task=torch.full((1, 9), 10.0),
            q_raw=torch.zeros((1, 7)),
            q_projected=torch.full((1, 7), 4.0),
            fallback_q=torch.zeros((1, 7)),
            progress_m=torch.tensor([0.4]),
            route=torch.tensor([ROUTE_FALLBACK]),
        )
        self.assertEqual(
            archive.update(torch.tensor([10]), replacement), 0)
        self.assertAlmostEqual(float(archive.progress_m[0]), 0.1)
        self.assertEqual(
            archive.update(
                torch.tensor([10]), replacement, overwrite=True),
            1)
        self.assertAlmostEqual(float(archive.progress_m[0]), 0.4)
        self.assertEqual(int(archive.route[0]), ROUTE_FALLBACK)
        self.assertTrue(torch.equal(
            archive.q_projected[0], torch.full((7,), 4.0)))

        inconsistent = DirectSeedRLBatch(
            task=torch.full((1, 9), -10.0),
            q_raw=torch.zeros((1, 7)),
            q_projected=torch.zeros((1, 7)),
            fallback_q=torch.zeros((1, 7)),
            progress_m=torch.zeros(1),
            route=torch.tensor([ROUTE_REFINED]),
        )
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            archive.update(
                torch.tensor([10]), inconsistent, overwrite=True)
        with self.assertRaisesRegex(ValueError, 'unknown task_ids'):
            archive.update(torch.tensor([99]), inconsistent)
        with self.assertRaisesRegex(TypeError, 'overwrite must'):
            archive.update(
                torch.tensor([10]), replacement, overwrite=1)

    def test_paired_archive_builds_only_return_improving_targets(self):
        task_ids = torch.tensor([10, 20, 30, 40, 50, 60])
        archive = DirectSeedPairedArchive(task_ids, seed=83)
        baseline_ids = task_ids[:5]
        task = baseline_ids.float()[:, None].expand(-1, 9).clone()
        baseline_q = (
            baseline_ids.float() / 100.0
        )[:, None].expand(-1, 7).clone()
        archive.update(
            baseline_ids,
            DirectSeedRLBatch(
                task=task,
                q_raw=torch.zeros_like(baseline_q),
                q_projected=baseline_q,
                fallback_q=torch.zeros_like(baseline_q),
                progress_m=torch.tensor(
                    [0.4, 0.5, 0.3, 0.4, 0.6]),
                route=torch.tensor([
                    ROUTE_REFINED, ROUTE_REFINED,
                    ROUTE_FALLBACK, ROUTE_FALLBACK,
                    ROUTE_DIRECT,
                ], dtype=torch.int64),
            ))

        explorer = DirectSeedEliteMemory(task_ids, seed=89)
        explorer_ids = torch.tensor([10, 20, 30, 40, 60])
        explorer_task = (
            explorer_ids.float()[:, None].expand(-1, 9).clone())
        explorer_q = (
            explorer_ids.float() / 10.0
        )[:, None].expand(-1, 7).clone()
        explorer.update(
            explorer_ids,
            DirectSeedRLBatch(
                task=explorer_task,
                q_raw=torch.zeros_like(explorer_q),
                q_projected=explorer_q,
                fallback_q=torch.zeros_like(explorer_q),
                progress_m=torch.tensor(
                    [0.45, 0.505, 0.32, 0.405, 0.99]),
                route=torch.full(
                    (5,), ROUTE_REFINED, dtype=torch.int64),
            ))

        targets = archive.build_targets(
            explorer, advantage_margin_m=0.01)
        self.assertEqual(targets.batch_size, 3)
        self.assertEqual(
            targets.task[:, 0].tolist(), [10.0, 20.0, 30.0])
        self.assertTrue(torch.allclose(
            targets.q_projected[:, 0],
            torch.tensor([1.0, 0.2, 3.0])))
        self.assertTrue(torch.allclose(
            targets.progress_m,
            torch.tensor([0.45, 0.5, 0.32])))
        self.assertTrue(torch.equal(
            targets.q_raw, targets.q_projected))
        self.assertTrue(bool(
            (targets.fallback_q == 0.0).all()))
        self.assertTrue(bool(
            (targets.route == ROUTE_REFINED).all()))

        stats = archive.target_stats(
            explorer, advantage_margin_m=0.01)
        self.assertEqual(stats['configured_task_count'], 6)
        self.assertEqual(stats['baseline_outcome_count'], 5)
        self.assertEqual(stats['baseline_refined_count'], 2)
        self.assertEqual(stats['explorer_elite_count'], 5)
        self.assertEqual(stats['paired_outcome_count'], 4)
        self.assertEqual(stats['explorer_selected_count'], 2)
        self.assertEqual(stats['baseline_selected_count'], 1)
        self.assertEqual(stats['target_count'], 3)
        self.assertAlmostEqual(stats['target_coverage'], 0.5)

        explorer.task[0, 0] = -1.0
        with self.assertRaisesRegex(ValueError, 'geometry differ'):
            archive.build_targets(explorer, 0.01)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            archive.target_stats(explorer, -0.01)

    def test_paired_archive_checkpoint_and_uniform_target_rng(self):
        task_ids = torch.tensor([1, 2, 3, 4])
        archive = DirectSeedPairedArchive(task_ids, seed=97)
        task = task_ids.float()[:, None].expand(-1, 9).clone()
        projected = (
            task_ids.float() / 10.0
        )[:, None].expand(-1, 7).clone()
        archive.update(
            task_ids,
            DirectSeedRLBatch(
                task=task,
                q_raw=torch.zeros_like(projected),
                q_projected=projected,
                fallback_q=torch.zeros_like(projected),
                progress_m=task_ids.float() / 10.0,
                route=torch.full(
                    (4,), ROUTE_REFINED, dtype=torch.int64),
            ))
        explorer = DirectSeedEliteMemory(task_ids)

        sampled = archive.sample_targets(explorer, 6000)
        _, counts = torch.unique(
            sampled.task[:, 0], return_counts=True)
        self.assertEqual(counts.numel(), 4)
        self.assertTrue(bool(
            ((counts - 1500).abs() < 200).all()))
        self.assertTrue(bool(
            (sampled.route == ROUTE_REFINED).all()))
        converted = archive.sample_targets(
            explorer, 3, dtype=torch.float64)
        self.assertEqual(converted.task.dtype, torch.float64)
        self.assertEqual(converted.route.dtype, torch.int64)

        archive.sample_targets(explorer, 17)
        state = archive.state_dict()
        expected = archive.sample_targets(explorer, 128)
        restored = DirectSeedPairedArchive(task_ids, seed=999)
        restored.load_state_dict(state)
        actual = restored.sample_targets(explorer, 128)
        for name in (
                'task', 'q_raw', 'q_projected',
                'fallback_q', 'progress_m', 'route'):
            self.assertTrue(torch.equal(
                getattr(expected, name), getattr(actual, name)))
        self.assertEqual(len(restored), 4)
        self.assertEqual(restored.coverage, 1.0)

        wrong_ids = DirectSeedPairedArchive(
            torch.tensor([1, 2, 3, 5]))
        with self.assertRaisesRegex(ValueError, 'task_ids differ'):
            wrong_ids.load_state_dict(state)

        fallback_only = DirectSeedPairedArchive(task_ids)
        fallback_only.update(
            task_ids,
            DirectSeedRLBatch(
                task=task,
                q_raw=torch.zeros_like(projected),
                q_projected=projected,
                fallback_q=torch.zeros_like(projected),
                progress_m=torch.zeros(4),
                route=torch.full(
                    (4,), ROUTE_FALLBACK, dtype=torch.int64),
            ))
        with self.assertRaisesRegex(RuntimeError, 'empty paired targets'):
            fallback_only.sample_targets(explorer, 1)

    def test_elite_projection_update_changes_actor_with_finite_loss(self):
        torch.manual_seed(71)
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
        task = torch.randn((5, 9))
        q_raw = actor.mean_q(task).detach()
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q_raw,
            q_projected=torch.full_like(q_raw, 0.6),
            fallback_q=torch.zeros_like(q_raw),
            progress_m=torch.linspace(0.1, 0.5, 5),
            route=torch.tensor([
                ROUTE_REFINED, ROUTE_DIRECT, ROUTE_REFINED,
                ROUTE_FALLBACK, ROUTE_REFINED,
            ], dtype=torch.int64),
        )
        before = [
            parameter.detach().clone() for parameter in actor.parameters()
        ]
        metrics = update_direct_seed_projection(
            actor, optimizer, batch, gradient_clip_norm=1.0)
        self.assertTrue(all(
            np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics['projection_actor_updated'], 1.0)
        self.assertEqual(metrics['projection_refined_count'], 3.0)
        self.assertAlmostEqual(
            metrics['projection_refined_fraction'], 3 / 5)
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(before, actor.parameters())))
        with self.assertRaisesRegex(ValueError, 'requires reference_q'):
            update_direct_seed_projection(
                actor, optimizer, batch, anchor_weight=0.1)
        with self.assertRaisesRegex(ValueError, 'reference_q must'):
            update_direct_seed_projection(
                actor, optimizer, batch,
                reference_q=torch.zeros((5, 6)),
                anchor_weight=0.1)

        no_refined = DirectSeedRLBatch(
            task=task,
            q_raw=q_raw,
            q_projected=torch.zeros_like(q_raw),
            fallback_q=torch.zeros_like(q_raw),
            progress_m=torch.zeros(5),
            route=torch.full(
                (5,), ROUTE_DIRECT, dtype=torch.int64),
        )
        with self.assertRaisesRegex(ValueError, 'ROUTE_REFINED'):
            update_direct_seed_projection(
                actor, optimizer, no_refined)

    def test_delayed_actor_and_behavior_anchor_are_explicit(self):
        torch.manual_seed(15)
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        critic = TwinMacroQ(
            -torch.ones(7), torch.ones(7),
            DirectSeedCriticConfig(hidden_dim=16, n_hidden_layers=1))
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
        task = torch.zeros((6, 9))
        task[:, 3] = 1.0
        task[:, 8] = 1.0
        q_raw = actor.mean_q(task).detach()
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q_raw,
            q_projected=torch.zeros_like(q_raw),
            fallback_q=torch.zeros_like(q_raw),
            progress_m=torch.linspace(0.1, 0.2, 6),
            route=torch.full(
                (6,), ROUTE_REFINED, dtype=torch.int64),
        )
        before = {
            name: value.detach().clone()
            for name, value in actor.state_dict().items()
        }
        critic_only = update_direct_seed_rl(
            actor, critic, actor_optimizer, critic_optimizer,
            batch, _DirectKin(), update_actor=False)
        self.assertEqual(critic_only['actor_updated'], 0.0)
        self.assertNotIn('actor_loss', critic_only)
        for name, value in actor.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))

        generator = torch.Generator().manual_seed(99)
        actor_stats = update_direct_seed_rl(
            actor, critic, actor_optimizer, critic_optimizer,
            batch, _DirectKin(),
            DirectSeedRLConfig(
                behavior_anchor_weight=0.01,
                refine_route_penalty_m=0.01),
            update_actor=True, generator=generator)
        self.assertEqual(actor_stats['actor_updated'], 1.0)
        self.assertIn('behavior_anchor_loss', actor_stats)
        self.assertAlmostEqual(
            actor_stats['critic_route_penalty_mean_m'], 0.01, places=6)
        self.assertTrue(all(
            np.isfinite(value) for value in actor_stats.values()))

    def test_zero_collection_noise_matches_deployment_mean(self):
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        task = torch.randn((5, 9))
        expected = actor.mean_q(task)
        actual = actor.sample(
            task, generator=torch.Generator().manual_seed(7),
            noise_scale=0.0).q
        self.assertTrue(torch.equal(expected, actual))
        with self.assertRaisesRegex(ValueError, 'noise_scale'):
            actor.sample(task, noise_scale=-0.1)

    def test_precision_only_update_improves_deployment_mean(self):
        torch.manual_seed(37)
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-2)
        task = torch.zeros((8, 9))
        task[:, :3] = torch.tensor([0.25, -0.15, 0.10])
        task[:, 3] = 1.0
        task[:, 8] = 1.0
        with torch.no_grad():
            q_raw = actor.mean_q(task)
            initial_error = (
                q_raw[:, :3] - task[:, :3]).norm(dim=-1).mean()
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q_raw,
            q_projected=torch.cat(
                [task[:, :3], torch.zeros((8, 4))], dim=-1),
            fallback_q=torch.zeros_like(q_raw),
            progress_m=torch.zeros(8),
            route=torch.full(
                (8,), ROUTE_REFINED, dtype=torch.int64),
        )
        before = [
            parameter.detach().clone() for parameter in actor.parameters()
        ]
        stats = None
        for _ in range(20):
            stats = update_direct_seed_precision(
                actor, optimizer, batch, _DirectKin(),
                DirectSeedRLConfig(), collision=_SafeCollision())
        with torch.no_grad():
            final_error = (
                actor.mean_q(task)[:, :3] - task[:, :3]
            ).norm(dim=-1).mean()
        self.assertLess(float(final_error), float(initial_error))
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(before, actor.parameters())))
        self.assertIsNotNone(stats)
        self.assertEqual(stats['precision_actor_updated'], 1.0)
        self.assertEqual(stats['collision_precision_available'], 1.0)
        self.assertEqual(stats['projection_refined_fraction'], 1.0)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_precision_only_update_without_refined_route_is_safe(self):
        torch.manual_seed(38)
        actor = DirectSeedActor(
            -torch.ones(7), torch.ones(7),
            DirectSeedActorConfig(hidden_dim=16, n_hidden_layers=1))
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        task = torch.zeros((4, 9))
        task[:, 3] = 1.0
        task[:, 8] = 1.0
        q_raw = actor.mean_q(task).detach()
        batch = DirectSeedRLBatch(
            task=task,
            q_raw=q_raw,
            q_projected=torch.zeros_like(q_raw),
            fallback_q=torch.zeros_like(q_raw),
            progress_m=torch.zeros(4),
            route=torch.tensor([
                ROUTE_DIRECT, ROUTE_FALLBACK,
                ROUTE_INVALID, ROUTE_DIRECT,
            ], dtype=torch.int64),
        )
        stats = update_direct_seed_precision(
            actor, optimizer, batch, _DirectKin(),
            projection_weight=1.0)
        self.assertEqual(stats['projection_distill_loss'], 0.0)
        self.assertEqual(stats['projection_refined_fraction'], 0.0)
        self.assertEqual(stats['collision_precision_available'], 0.0)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_resume_optimizer_lr_is_applied_after_state_load(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-5)
        state = optimizer.state_dict()
        restored = torch.optim.Adam([parameter], lr=1e-4)
        restored.load_state_dict(state)
        self.assertEqual(restored.param_groups[0]['lr'], 1e-5)
        previous = _set_optimizer_lr(restored, 3e-5)
        self.assertEqual(previous, [1e-5])
        self.assertEqual(restored.param_groups[0]['lr'], 3e-5)


class DirectSeedProjectionTest(unittest.TestCase):
    def test_direct_refined_and_bit_exact_fallback_routes(self):
        p0, line_dir, n_target = _task(3)
        p0[1, 0] = 0.1
        p0[2, 0] = 0.2
        q_raw = torch.zeros((3, 7), dtype=torch.float32)
        fallback = torch.zeros_like(q_raw)
        fallback[:, :3] = p0
        fallback[2, 1] = -0.0
        calls = []

        def projector(kin, q_seed, p_target, rotation_target, **kwargs):
            del kin, rotation_target
            calls.append((q_seed.clone(), dict(kwargs)))
            q = q_seed.clone()
            q[:, :3] = p_target
            ok = torch.tensor([True, False])
            return q, ok, ('ok', 'fail')

        result = route_generated_seed(
            _DirectKin(), _SafeCollision(), q_raw,
            p0, line_dir, n_target, fallback,
            projector=projector)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0].shape[0], 2)
        self.assertIsNone(calls[0][1]['branch_action'])
        self.assertTrue(calls[0][1]['preserve_seed'])
        self.assertEqual(
            result.route.tolist(),
            [ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK])
        self.assertTrue(bool(result.valid.all()))
        self.assertFalse(bool(result.ik_attempted[0]))
        self.assertTrue(bool(result.ik_attempted[1:].all()))
        self.assertTrue(torch.equal(
            result.q[2].view(torch.int32),
            fallback[2].view(torch.int32)))

    def test_joint_margin_is_shared_by_hard_gate(self):
        p0, line_dir, n_target = _task(1)
        q = torch.zeros((1, 7))
        q[0, 3] = 0.99
        validity = strict_seed_validity(
            _DirectKin(), _SafeCollision(), q,
            p0, line_dir, n_target)
        self.assertAlmostEqual(
            float(validity.joint_margin_rad[0]), 0.01, places=6)
        self.assertFalse(bool(validity.joint_limits[0]))
        self.assertFalse(bool(validity.valid[0]))

    def test_projection_target_keeps_solver_tolerance_inside_hard_cone(self):
        with self.assertRaisesRegex(ValueError, '0.5deg buffer'):
            DirectSeedProjectionConfig(projection_cone_deg=25.0)
        p0, line_dir, _ = _task(1)
        line_dir[:] = torch.tensor([0.0, 1.0, 0.0])
        n_target = torch.tensor([[1.0, 0.0, 0.0]])
        q_raw = torch.zeros((1, 7))
        captured = {}

        def projector(kin, q_seed, p_target, rotation_target, **kwargs):
            del kin, p_target, kwargs
            captured['z'] = rotation_target[:, :, 2].clone()
            return q_seed, torch.zeros(1, dtype=torch.bool), ('fail',)

        result = route_generated_seed(
            _DirectKin(), _SafeCollision(), q_raw,
            p0, line_dir, n_target, q_raw.clone(),
            projector=projector)
        cosine = float((captured['z'][0] * n_target[0]).sum())
        self.assertGreaterEqual(
            cosine,
            np.cos(np.deg2rad(24.5)) - 1e-6)
        self.assertEqual(int(result.route[0]), ROUTE_INVALID)

    def test_missing_collision_evidence_fails_closed_bit_exact(self):
        p0, line_dir, n_target = _task(1)
        fallback = torch.zeros((1, 7))
        fallback[0, 0] = -0.0
        result = route_generated_seed(
            _DirectKin(), None, torch.zeros_like(fallback),
            p0, line_dir, n_target, fallback)
        self.assertFalse(bool(result.valid[0]))
        self.assertEqual(int(result.route[0]), ROUTE_INVALID)
        self.assertTrue(torch.equal(
            result.q.view(torch.int32),
            fallback.view(torch.int32)))


class DirectSeedEvalTest(unittest.TestCase):
    @staticmethod
    def _filter_manifest_fixture(
        directory: str,
        *,
        row_dtype=np.int64,
        include_artifacts: bool = True,
    ):
        batch = SeedCandidateBatch(
            q0=torch.zeros((3, 2, 7), dtype=torch.float32),
            p0=torch.tensor([
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ], dtype=torch.float32),
            line_dir=torch.tensor(
                [[1.0, 0.0, 0.0]] * 3, dtype=torch.float32),
            n_target=torch.tensor(
                [[0.0, 1.0, 0.0]] * 3, dtype=torch.float32),
            valid=torch.ones((3, 2), dtype=torch.bool),
        )
        dataset = CachedSeedCandidateDataset(
            batch,
            task_indices=torch.tensor([11, 12, 13]),
            fallback_index=1)
        rows = np.asarray([0, 2], dtype=row_dtype)
        excluded = np.asarray([1], dtype=np.int64)
        kept_tasks = np.asarray([11, 13], dtype=np.int64)
        excluded_tasks = np.asarray([12], dtype=np.int64)
        manifest = Path(directory) / 'prior.npz'
        np.savez(
            manifest,
            source_row_index=rows,
            excluded_source_row_index=excluded,
            task_indices=kept_tasks,
            excluded_task_indices=excluded_tasks)
        candidates = Path(directory) / 'candidates.npz'
        candidates.write_bytes(b'exact candidate artifact')
        candidates_sha256 = hashlib.sha256(
            candidates.read_bytes()).hexdigest()
        fingerprints = dataset.task_fingerprints
        saved_filter = {
            'explicit_filter_enabled': True,
            'n_source_tasks': 3,
            'n_kept_tasks': 2,
            'n_excluded_tasks': 1,
            'kept_geometry_fingerprint_list_sha256': hashlib.sha256(
                '\n'.join([fingerprints[0], fingerprints[2]])
                .encode('ascii')).hexdigest(),
            'excluded_geometry_fingerprint_list_sha256': hashlib.sha256(
                fingerprints[1].encode('ascii')).hexdigest(),
        }
        saved = {
            'n_tasks': 2,
            'fallback_strict_filter': saved_filter,
        }
        if include_artifacts:
            saved['artifacts'] = {
                'candidates_sha256': candidates_sha256,
            }
        manifest.with_suffix('.json').write_text(
            json.dumps(saved), encoding='utf-8')
        return dataset, manifest, candidates_sha256

    def test_summary_records_one_seed_one_rollout_and_routes(self):
        report = summarize_direct_seed(
            np.asarray([0.2, 0.3, 0.4], np.float32),
            np.asarray(
                [ROUTE_DIRECT, ROUTE_REFINED, ROUTE_FALLBACK], np.int8),
            np.ones(3, dtype=np.bool_),
            fallback_progress_m=np.asarray([0.1, 0.3, 0.3], np.float32),
            pool_oracle_progress_m=np.asarray([0.4, 0.5, 0.5], np.float32))
        self.assertEqual(report['candidate_enumeration_per_task'], 0)
        self.assertEqual(report['controller_rollouts_per_task'], 1.0)
        self.assertAlmostEqual(report['mean_ik_attempts_per_task'], 2 / 3)
        self.assertEqual(report['route_count']['direct'], 1)
        self.assertAlmostEqual(
            report['pool_capture_pct'], 2 / 7 * 100, places=6)

    def test_native_direct_eval_can_be_used_as_paired_reference(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'direct_eval.npz'
            np.savez(
                path,
                task_indices=np.asarray([7, 3], dtype=np.int64),
                progress_m=np.asarray([0.7, 0.3], dtype=np.float32))
            progress = _load_reference_progress(
                path, np.asarray([3, 7], dtype=np.int64))
        np.testing.assert_array_equal(
            progress, np.asarray([0.3, 0.7], dtype=np.float32))

    def test_filter_manifest_requires_exact_candidate_identity(self):
        with TemporaryDirectory() as directory:
            dataset, manifest, candidates_sha256 = (
                self._filter_manifest_fixture(directory))

            class CountingDataset(CachedSeedCandidateDataset):
                def __init__(self, source):
                    super().__init__(
                        source.batch, source.task_indices,
                        source.fallback_index)
                    self.fingerprint_accesses = 0

                @property
                def task_fingerprints(self):
                    self.fingerprint_accesses += 1
                    return (
                        CachedSeedCandidateDataset
                        .task_fingerprints.fget(self))

            dataset = CountingDataset(dataset)
            rows, excluded, report = _load_fallback_filter_manifest(
                manifest, dataset, 3,
                candidates_sha256=candidates_sha256)
            self.assertEqual(rows.tolist(), [0, 2])
            self.assertEqual(excluded.tolist(), [1])
            self.assertTrue(report['candidate_identity_verified'])
            self.assertFalse(report['legacy_unverified'])
            self.assertEqual(dataset.fingerprint_accesses, 1)
            with self.assertRaisesRegex(ValueError, 'SHA256 differs'):
                _load_fallback_filter_manifest(
                    manifest, dataset, 3,
                    candidates_sha256='0' * 64)

    def test_filter_manifest_legacy_reuse_is_explicitly_labelled(self):
        with TemporaryDirectory() as directory:
            dataset, manifest, candidates_sha256 = (
                self._filter_manifest_fixture(
                    directory, include_artifacts=False))
            with self.assertRaisesRegex(ValueError, 'fails closed'):
                _load_fallback_filter_manifest(
                    manifest, dataset, 3,
                    candidates_sha256=candidates_sha256)
            _, _, report = _load_fallback_filter_manifest(
                manifest, dataset, 3,
                candidates_sha256=candidates_sha256,
                allow_legacy=True)
            self.assertFalse(report['candidate_identity_verified'])
            self.assertTrue(report['legacy_unverified'])

    def test_filter_manifest_rejects_lossy_integer_coercion(self):
        with TemporaryDirectory() as directory:
            dataset, manifest, candidates_sha256 = (
                self._filter_manifest_fixture(
                    directory, row_dtype=np.float64))
            with self.assertRaisesRegex(ValueError, 'integer dtype'):
                _load_fallback_filter_manifest(
                    manifest, dataset, 3,
                    candidates_sha256=candidates_sha256)


if __name__ == '__main__':
    unittest.main()
