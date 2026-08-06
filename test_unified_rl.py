"""Standalone unit and CPU integration tests for ``Yuan.unified_rl``."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Yuan.RL_controller.algorithms.ppo import (
    Agent,
    PPOConfig,
    RewardScaler,
    train as ppo_train,
)
from Yuan.RL_controller.env.env import (
    EnvConfig,
    LATERAL_SAFETY_NET,
    NSRLBatchedEnv,
    TERM_TRUNCATED,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.unified_rl.candidate_batch import (
    CachedSeedCandidateDataset,
    SeedCandidateBatch,
)
from Yuan.unified_rl.bidirectional_train import resume_position
from Yuan.unified_rl.build_external_holdout import (
    audit_geometry_independence,
    canonical_task_geometry,
    select_valid_indices_without_replacement,
    validate_rank_train_payload,
)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController,
    restore_env_state,
    rollout_seed_selection,
    rollout_selected_seeds,
    rollout_topk_prefix_lookahead,
    snapshot_env_state,
    topk_union_first_valid,
)
from Yuan.unified_rl.checkpoint import (
    adapt_controller_optimizer_observation_state,
    load_controller_state_dict,
    require_checkpoint_format_version,
    require_checkpoint_keys,
    resolve_controller_dir,
)
from Yuan.unified_rl.features import initial_observation_features
from Yuan.unified_rl.joint_controller_refine import select_promoted_block
from Yuan.unified_rl.joint_controller_search_distill import (
    _conservative_search_targets,
    _paired_action_candidates,
    _robust_delta_metrics,
)
from Yuan.unified_rl.materialize_seed_blend import (
    _blend_state_dicts,
)
from Yuan.unified_rl.materialize_actor_q_selector import (
    ACTOR_Q_WEIGHT_GRID,
    _actor_q_proposal,
    _choose_model_candidate,
    _fixed_rule_report,
    _promotion_reasons,
)
from Yuan.unified_rl.offline_seed_ensemble_train import (
    _assert_paired_candidate_datasets,
    _assert_three_way_geometry_disjoint,
    _controller_robust_feasibility_target,
    _make_training_table,
    _warm_refit_is_promotable,
    _warm_retention_losses,
)
from Yuan.unified_rl.evaluate_residual import (
    _atomic_savez_new,
    _prepare_output_path,
    geometry_grouped_bootstrap_ci,
)
from Yuan.unified_rl.evaluate import (
    _effective_probe_task_chunk,
    _geometry_macro_mean,
    _pad_indices,
    load_seed_policy,
)
from Yuan.unified_rl.analyze_joint_2x2 import (
    CELL_NAMES,
    analyze_joint_2x2,
    load_evaluation,
)
from Yuan.unified_rl.analyze_controller_pair import (
    analyze_controller_pair,
    load_controller_evaluation,
)
from Yuan.unified_rl.provenance import (
    assert_same_provenance,
    controller_fingerprint,
    file_fingerprint,
)
from Yuan.unified_rl.reproducibility import (
    global_rng_state,
    restore_global_rng,
    seed_global_rng,
)
from Yuan.unified_rl.residual_bandit import (
    ResidualBanditConfig,
    geometry_groups,
    residual_bandit_loss,
    sample_group_balanced_indices,
)
from Yuan.unified_rl.residual_seed import (
    ResidualSeedConfig,
    apply_residual_seed,
)
from Yuan.unified_rl.residual_policy import (
    ResidualSeedHead,
    antithetic_gaussian_actions_and_log_prob,
)
from Yuan.unified_rl.seed_policy import (
    CandidateSeedActorCritic,
    CandidateSeedPolicyEnsemble,
    SEED_ENSEMBLE_AGGREGATION,
    SEED_ENSEMBLE_FORMAT,
    SeedPolicyConfig,
    infer_seed_policy_config,
    seed_policy_ensemble_states,
)
from Yuan.unified_rl.seed_distribution import SeedPolicyLineDistribution
from Yuan.unified_rl.seed_deployment import (
    SeedDeploymentConfig,
    deployment_config_from_checkpoint,
    select_seed_deployment,
)
from Yuan.unified_rl.seed_gpi import (
    DenseSeedConfig,
    _dense_rank_loss,
    _target_distribution,
    collect_dense_seed_rollout,
    update_dense_seed_policy,
)
from Yuan.unified_rl.seed_ppo import (
    SeedPPOConfig,
    collect_seed_rollout,
    update_seed_policy,
)
from Yuan.unified_rl.validity import (
    assert_same_valid_mask,
    check_candidate_validity,
    validate_cached_dataset,
)


def _batch(b=4, k=3):
    q0 = torch.zeros((b, k, 7), dtype=torch.float32)
    q0[:, :, 0] = torch.arange(k, dtype=torch.float32)
    valid = torch.ones((b, k), dtype=torch.bool)
    return SeedCandidateBatch(
        q0=q0,
        p0=torch.zeros((b, 3)),
        line_dir=torch.tensor([[1.0, 0.0, 0.0]]).expand(b, -1).clone(),
        n_target=torch.tensor([[0.0, 0.0, 1.0]]).expand(b, -1).clone(),
        valid=valid,
    )


class _IdentityKin:
    """Small deterministic FK stub for candidate-validation unit tests."""

    device = torch.device('cpu')
    dtype = torch.float32
    lmt_lo = -torch.ones(7)
    lmt_up = torch.ones(7)
    q_mid = torch.zeros(7)

    @staticmethod
    def tcp_fk_jac(q):
        n = q.shape[0]
        position = q[:, :3]
        rotation = torch.eye(3).expand(n, -1, -1).clone()
        jacobian = torch.zeros((n, 6, 7), dtype=q.dtype, device=q.device)
        transforms = torch.eye(4).expand(n, 1, -1, -1).clone()
        return position, rotation, jacobian, transforms


class _ResidualKin(_IdentityKin):
    """Identity FK with joint 3 exposed to a collision-test stub."""

    @staticmethod
    def tcp_fk_jac(q):
        position, rotation, jacobian, transforms = _IdentityKin.tcp_fk_jac(q)
        jacobian[:, 0, 0] = 1.0
        jacobian[:, 1, 1] = 1.0
        jacobian[:, 2, 2] = 1.0
        return position, rotation, jacobian, transforms

    @staticmethod
    def link_transforms(q):
        transforms = torch.eye(
            4, dtype=q.dtype, device=q.device).expand(
                q.shape[0], 1, -1, -1).clone()
        transforms[:, 0, 0, 3] = q[:, 3]
        return transforms


class _ResidualCollision:
    @staticmethod
    def min_margin(link_transforms):
        return 0.05 - link_transforms[:, 0, 0, 3].abs()


class _InfiniteMarginCollision:
    @staticmethod
    def min_margin(link_transforms):
        return torch.full(
            (link_transforms.shape[0],), float('inf'),
            dtype=link_transforms.dtype, device=link_transforms.device)


def _residual_basis(kin, q, line_dir, n_target, q_mid, q_half, damping):
    del kin, line_dir, n_target, q_mid, q_half, damping
    basis = torch.zeros((q.shape[0], 7, 4), dtype=q.dtype, device=q.device)
    basis[:, 3:, :] = torch.eye(4, dtype=q.dtype, device=q.device)
    fallback = torch.zeros((q.shape[0], 3), dtype=torch.bool, device=q.device)
    return basis, fallback


class CandidateBatchTest(unittest.TestCase):
    def test_geometry_macro_mean_weights_geometries_equally(self):
        values = np.asarray([1.0, 3.0, 9.0], dtype=np.float32)
        self.assertAlmostEqual(
            _geometry_macro_mean(values, ('same', 'same', 'other')),
            5.5)

    def test_external_holdout_pure_audits_are_fail_closed(self):
        p0 = np.array([[0, 0, 0], [1, 0, 0]], np.float32)
        line_dir = np.tile([1, 0, 0], (2, 1)).astype(np.float32)
        n_target = np.tile([0, 0, 1], (2, 1)).astype(np.float32)
        geometry = canonical_task_geometry(p0, line_dir, n_target)
        stats = audit_geometry_independence(geometry)
        self.assertEqual(stats['n_unique_tasks'], 2)
        with self.assertRaisesRegex(ValueError, 'overlaps excluded'):
            audit_geometry_independence(
                geometry, {'training': geometry[1:]})
        with self.assertRaisesRegex(ValueError, 'not unique'):
            audit_geometry_independence(geometry[[0, 0]])

        selected_a = select_valid_indices_without_replacement(
            np.array([2, 5, 7, 11]), 3, 19)
        selected_b = select_valid_indices_without_replacement(
            np.array([2, 5, 7, 11]), 3, 19)
        self.assertTrue(np.array_equal(selected_a, selected_b))
        self.assertEqual(np.unique(selected_a).size, 3)

        payload = {
            'seeds': np.zeros((2, 8, 7), np.float32),
            'ik_ok': np.ones((2, 8), bool),
            'p0': p0,
            'line_dir': line_dir,
            'n_target': n_target,
            'q0_pilot': np.zeros((2, 7), np.float32),
        }
        validate_rank_train_payload(payload, n_tasks=2)
        payload['ik_ok'] = payload['ik_ok'].astype(np.uint8)
        with self.assertRaisesRegex(ValueError, 'boolean dtype'):
            validate_rank_train_payload(payload, n_tasks=2)

    def test_mask_and_selection(self):
        batch = _batch()
        valid = batch.valid.clone()
        valid[:, 1] = False
        batch = SeedCandidateBatch(
            batch.q0, batch.p0, batch.line_dir, batch.n_target, valid)
        selected = batch.select(torch.tensor([0, 2, 0, 2]))
        self.assertEqual(tuple(selected.q0.shape), (4, 7))
        with self.assertRaisesRegex(ValueError, 'invalid'):
            batch.select(torch.tensor([1, 0, 0, 0]))

        repeated = batch.repeat_interleave(2)
        self.assertTrue(torch.equal(repeated.q0[0], repeated.q0[1]))
        self.assertTrue(torch.equal(repeated.q0[2], batch.q0[1]))

    def test_cached_pilot_is_feasible_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'candidates.npz'
            np.savez_compressed(
                path,
                seeds=np.zeros((2, 2, 7), np.float32),
                ik_ok=np.array([[False, False], [True, False]]),
                q0_pilot=np.ones((2, 7), np.float32),
                p0=np.zeros((2, 3), np.float32),
                line_dir=np.tile([1, 0, 0], (2, 1)).astype(np.float32),
                n_target=np.tile([0, 0, 1], (2, 1)).astype(np.float32),
            )
            dataset = CachedSeedCandidateDataset.from_npz(path)
            self.assertEqual(dataset.batch.n_candidates, 3)
            self.assertEqual(dataset.fallback_index, 2)
            self.assertTrue(bool(dataset.batch.valid[:, -1].all()))
            self.assertTrue(bool(dataset.batch.valid.any(dim=1).all()))

            with self.assertRaisesRegex(ValueError, 'at least one valid'):
                CachedSeedCandidateDataset.from_npz(
                    path, include_fallback=False)

            bad_pilot_path = Path(tmp) / 'bad-pilot.npz'
            np.savez_compressed(
                bad_pilot_path,
                seeds=np.zeros((1, 1, 7), np.float32),
                ik_ok=np.ones((1, 1), bool),
                q0_pilot=np.full((1, 7), np.nan, np.float32),
                p0=np.zeros((1, 3), np.float32),
                line_dir=np.array([[1, 0, 0]], np.float32),
                n_target=np.array([[0, 0, 1]], np.float32),
            )
            with self.assertRaisesRegex(ValueError, 'finite numeric'):
                CachedSeedCandidateDataset.from_npz(bad_pilot_path)

    def test_cached_mask_alias_and_malformed_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fields = {
                'seeds': np.zeros((2, 2, 7), np.float32),
                'p0': np.zeros((2, 3), np.float32),
                'line_dir': np.tile([1, 0, 0], (2, 1)).astype(np.float32),
                'n_target': np.tile([0, 0, 1], (2, 1)).astype(np.float32),
            }
            expected = np.array([[1, 0], [0, 1]], np.uint8)
            ok_path = tmp / 'ok.npz'
            np.savez_compressed(
                ok_path, **fields, ok=expected,
                src_idx=np.array([11, 17], np.int64))
            dataset = CachedSeedCandidateDataset.from_npz(
                ok_path, include_fallback=False)
            self.assertIsNone(dataset.fallback_index)
            self.assertTrue(torch.equal(
                dataset.task_indices, torch.tensor([11, 17])))
            self.assertTrue(torch.equal(
                dataset.batch.valid, torch.from_numpy(expected.astype(bool))))

            missing_path = tmp / 'missing.npz'
            np.savez_compressed(missing_path, **fields)
            with self.assertRaisesRegex(ValueError, 'validity mask'):
                CachedSeedCandidateDataset.from_npz(
                    missing_path, include_fallback=False)

            conflicting_path = tmp / 'conflicting.npz'
            np.savez_compressed(
                conflicting_path, **fields, ok=expected,
                ik_ok=np.logical_not(expected))
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                CachedSeedCandidateDataset.from_npz(
                    conflicting_path, include_fallback=False)

            malformed_path = tmp / 'malformed.npz'
            np.savez_compressed(
                malformed_path, **fields, ok=np.ones((2, 1), bool))
            with self.assertRaisesRegex(ValueError, 'must have shape'):
                CachedSeedCandidateDataset.from_npz(
                    malformed_path, include_fallback=False)

    def test_train_validation_split_is_disjoint_and_reproducible(self):
        batch = _batch(b=20, k=2)
        unique_p0 = batch.p0.clone()
        unique_p0[:, 0] = torch.arange(20, dtype=torch.float32)
        dataset = CachedSeedCandidateDataset(SeedCandidateBatch(
            batch.q0, unique_p0, batch.line_dir, batch.n_target, batch.valid))
        train_a, val_a, train_idx_a, val_idx_a = (
            dataset.train_validation_split(0.2, 17))
        _, _, train_idx_b, val_idx_b = dataset.train_validation_split(0.2, 17)
        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(val_a), 4)
        self.assertTrue(torch.equal(train_idx_a, train_idx_b))
        self.assertTrue(torch.equal(val_idx_a, val_idx_b))
        self.assertEqual(len(set(train_idx_a.tolist()) & set(val_idx_a.tolist())), 0)
        source_order = torch.tensor([
            int(val_a.task_indices[1]), int(val_a.task_indices[0])])
        selected = dataset.select_source_tasks(source_order)
        self.assertTrue(torch.equal(selected.task_indices, source_order))
        tiny_batch = _batch(b=2, k=2)
        tiny_p0 = tiny_batch.p0.clone()
        tiny_p0[:, 0] = torch.arange(2, dtype=torch.float32)
        tiny_dataset = CachedSeedCandidateDataset(SeedCandidateBatch(
            tiny_batch.q0, tiny_p0, tiny_batch.line_dir,
            tiny_batch.n_target, tiny_batch.valid))
        tiny_train, tiny_val, _, _ = tiny_dataset.train_validation_split(
            0.99, 17)
        self.assertEqual(len(tiny_train), 1)
        self.assertEqual(len(tiny_val), 1)
        with self.assertRaisesRegex(ValueError, 'at least two tasks'):
            CachedSeedCandidateDataset(
                _batch(b=1, k=2)).train_validation_split(0.5, 17)

    def test_group_split_prevents_duplicate_task_leakage(self):
        batch = _batch(b=8, k=2)
        p0 = batch.p0.clone()
        p0[:, 0] = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3],
                                dtype=torch.float32)
        dataset = CachedSeedCandidateDataset(SeedCandidateBatch(
            batch.q0, p0, batch.line_dir, batch.n_target, batch.valid))
        train_a, val_a, train_idx_a, val_idx_a = (
            dataset.train_validation_split(0.25, 31))
        train_b, val_b, train_idx_b, val_idx_b = (
            dataset.train_validation_split(0.25, 31))
        self.assertTrue(torch.equal(train_idx_a, train_idx_b))
        self.assertTrue(torch.equal(val_idx_a, val_idx_b))
        self.assertFalse(
            set(train_a.task_fingerprints) & set(val_a.task_fingerprints))
        self.assertEqual(len(train_a) + len(val_a), len(dataset))
        self.assertEqual(len(val_a), 2)

        one_group = CachedSeedCandidateDataset(_batch(b=3, k=2))
        with self.assertRaisesRegex(ValueError, 'unique task-geometry'):
            one_group.train_validation_split(0.2, 31)

    def test_task_fingerprint_canonicalizes_signed_zero(self):
        batch = _batch(b=2, k=2)
        p0 = batch.p0.clone()
        p0[1] = -0.0
        dataset = CachedSeedCandidateDataset(SeedCandidateBatch(
            batch.q0, p0, batch.line_dir, batch.n_target, batch.valid))
        self.assertEqual(
            dataset.task_fingerprints[0], dataset.task_fingerprints[1])

    def test_masked_nan_features_and_validity_are_safe(self):
        q0 = torch.zeros((1, 2, 7))
        q0[0, 1] = float('nan')
        candidates = SeedCandidateBatch(
            q0=q0,
            p0=torch.zeros((1, 3)),
            line_dir=torch.tensor([[1.0, 0.0, 0.0]]),
            n_target=torch.tensor([[0.0, 0.0, 1.0]]),
            valid=torch.tensor([[True, False]]),
        )
        features = initial_observation_features(_IdentityKin(), candidates)
        self.assertTrue(bool(torch.isfinite(features).all()))
        self.assertTrue(torch.equal(features[0, 1], torch.zeros(31)))

        result = check_candidate_validity(_IdentityKin(), None, candidates)
        self.assertTrue(bool(result.valid[0, 0]))
        self.assertFalse(bool(result.finite[0, 1]))
        self.assertFalse(bool(result.position[0, 1]))
        self.assertFalse(bool(result.cone[0, 1]))
        self.assertTrue(bool(torch.isinf(result.position_error_m[0, 1])))

        wrongly_valid = SeedCandidateBatch(
            q0=q0,
            p0=candidates.p0,
            line_dir=candidates.line_dir,
            n_target=candidates.n_target,
            valid=torch.ones((1, 2), dtype=torch.bool),
        )
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            initial_observation_features(_IdentityKin(), wrongly_valid)

        huge_q0 = torch.zeros((1, 2, 7))
        huge_q0[0, 1] = 1e38
        huge_invalid = SeedCandidateBatch(
            q0=huge_q0,
            p0=candidates.p0,
            line_dir=candidates.line_dir,
            n_target=candidates.n_target,
            valid=torch.tensor([[True, False]]),
        )
        huge_features = initial_observation_features(
            _IdentityKin(), huge_invalid)
        self.assertTrue(bool(torch.isfinite(huge_features).all()))
        huge_validity = check_candidate_validity(
            _IdentityKin(), None, huge_invalid)
        self.assertFalse(bool(huge_validity.joint_limits[0, 1]))
        self.assertFalse(bool(huge_validity.position[0, 1]))
        self.assertFalse(bool(huge_validity.collision_free[0, 1]))

    def test_directional_features_match_linearized_identity_axis(self):
        candidates = SeedCandidateBatch(
            q0=torch.zeros((1, 1, 7)),
            p0=torch.zeros((1, 3)),
            line_dir=torch.tensor([[1.0, 0.0, 0.0]]),
            n_target=torch.tensor([[0.0, 0.0, 1.0]]),
            valid=torch.ones((1, 1), dtype=torch.bool),
        )
        features = initial_observation_features(
            _ResidualKin(), candidates,
            include_directional_dynamics=True)
        directional = features[0, 0, -10:]
        self.assertEqual(tuple(features.shape), (1, 1, 41))
        self.assertTrue(torch.allclose(
            directional[:7],
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            atol=2e-5, rtol=2e-5))
        self.assertAlmostEqual(float(directional[7]), 1.0, places=4)
        self.assertAlmostEqual(float(directional[8]), 1.0, places=4)
        self.assertAlmostEqual(float(directional[9]), 1.0, places=4)

    def test_strict_validation_filters_tasks_and_reports_source_indices(self):
        q0 = torch.zeros((3, 1, 7))
        q0[:, 0, 0] = torch.tensor([0.1, 0.2, 0.3])
        p0 = q0[:, 0, :3].clone()
        p0[1, 0] += 1.0
        dataset = CachedSeedCandidateDataset(
            SeedCandidateBatch(
                q0=q0,
                p0=p0,
                line_dir=torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1).clone(),
                n_target=torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1).clone(),
                valid=torch.ones((3, 1), dtype=torch.bool),
            ),
            task_indices=torch.tensor([7, 11, 19]),
        )
        validated, stats = validate_cached_dataset(
            dataset, _IdentityKin(), None, chunk_size=2,
            position_tol_m=1e-3)
        self.assertEqual(len(validated), 2)
        self.assertTrue(torch.equal(
            validated.task_indices, torch.tensor([7, 19])))
        self.assertTrue(torch.allclose(
            validated.batch.q0[:, 0, 0], torch.tensor([0.1, 0.3])))
        self.assertEqual(stats['n_tasks_rejected'], 1.0)
        self.assertEqual(stats['rejected_task_indices'], [11])

        all_bad_p0 = q0[:, 0, :3] + 1.0
        all_bad = CachedSeedCandidateDataset(
            SeedCandidateBatch(
                q0=q0,
                p0=all_bad_p0,
                line_dir=dataset.batch.line_dir,
                n_target=dataset.batch.n_target,
                valid=dataset.batch.valid,
            ),
            task_indices=dataset.task_indices,
        )
        with self.assertRaisesRegex(ValueError, r'7, 11, 19'):
            validate_cached_dataset(
                all_bad, _IdentityKin(), None, position_tol_m=1e-3)

    def test_derived_valid_mask_is_checkpoint_strict(self):
        dataset = CachedSeedCandidateDataset(_batch(b=2, k=2))
        expected = dataset.batch.valid.clone()
        assert_same_valid_mask(dataset, expected, label='validation')
        expected[0, 1] = False
        with self.assertRaisesRegex(ValueError, 'changed at 1 slots'):
            assert_same_valid_mask(dataset, expected, label='validation')


class SeedPolicyTest(unittest.TestCase):
    def test_reused_three_way_split_must_be_geometry_disjoint(self):
        fingerprints = ('a', 'a', 'b', 'c')
        _assert_three_way_geometry_disjoint(
            (torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([3])),
            fingerprints)
        with self.assertRaisesRegex(ValueError, 'geometry split overlaps'):
            _assert_three_way_geometry_disjoint(
                (torch.tensor([0]), torch.tensor([1, 2]), torch.tensor([3])),
                fingerprints)

    def test_controller_robust_feasibility_uses_worst_relative_gain(self):
        valid = torch.tensor([
            [True, True, True],
            [False, True, True],
        ])
        primary = torch.tensor([
            [0.10, 0.15, 0.08],
            [float('nan'), 0.20, 0.24],
        ])
        reference = torch.tensor([
            [0.10, 0.12, 0.16],
            [float('nan'), 0.20, 0.18],
        ])
        target = _controller_robust_feasibility_target(
            primary, reference, valid, valid, label='unit')
        expected = torch.tensor([
            [0.0, 0.02, -0.02],
            [0.0, 0.0, -0.02],
        ])
        self.assertTrue(torch.allclose(target, expected, atol=1e-7))

        changed_valid = valid.clone()
        changed_valid[0, 2] = False
        with self.assertRaisesRegex(ValueError, 'valid masks differ'):
            _controller_robust_feasibility_target(
                primary, reference, valid, changed_valid, label='unit')

    def test_training_table_keeps_actor_progress_and_feasibility_separate(self):
        valid = torch.tensor([[True, True], [True, False]])
        progress = torch.tensor([[0.2, 0.3], [0.4, float('nan')]])
        robust = torch.tensor([[0.0, -0.05], [0.0, 0.0]])
        features = torch.zeros((2, 2, 45))
        fingerprints = ('a' * 64, 'b' * 64)
        table = _make_training_table(
            features, valid, progress, fingerprints, torch.arange(2), [],
            source_feasibility_target=robust)
        self.assertTrue(torch.allclose(
            table.progress_m, progress, equal_nan=True))
        self.assertTrue(torch.equal(table.feasibility_target_m, robust))

        default = _make_training_table(
            features, valid, progress, fingerprints, torch.arange(2), [])
        self.assertTrue(torch.allclose(
            default.feasibility_target_m,
            torch.tensor([[0.0, 0.1], [0.0, 0.0]]), atol=1e-7))

    def test_paired_controller_candidates_require_exact_arrays(self):
        batch = _batch(b=2, k=2)
        primary = CachedSeedCandidateDataset(
            batch, task_indices=torch.tensor([5, 9]), fallback_index=0)
        reference = CachedSeedCandidateDataset(
            SeedCandidateBatch(
                q0=batch.q0.clone(), p0=batch.p0.clone(),
                line_dir=batch.line_dir.clone(),
                n_target=batch.n_target.clone(), valid=batch.valid.clone()),
            task_indices=torch.tensor([5, 9]), fallback_index=0)
        _assert_paired_candidate_datasets(
            primary, reference, label='unit')

        changed_q0 = batch.q0.clone()
        changed_q0[0, 1, 0] += 1e-6
        changed = CachedSeedCandidateDataset(
            SeedCandidateBatch(
                q0=changed_q0, p0=batch.p0.clone(),
                line_dir=batch.line_dir.clone(),
                n_target=batch.n_target.clone(), valid=batch.valid.clone()),
            task_indices=torch.tensor([5, 9]), fallback_index=0)
        with self.assertRaisesRegex(ValueError, 'q0 arrays differ'):
            _assert_paired_candidate_datasets(
                primary, changed, label='unit')

    def test_warm_refit_promotion_requires_mean_gain_and_safety(self):
        warm = {'mean_gain_m': 0.04, 'worse_rate': 0.15}
        self.assertTrue(_warm_refit_is_promotable(
            {'mean_gain_m': 0.041, 'worse_rate': 0.14}, warm,
            minimum_gain_m=0.001, worse_tolerance=0.0))
        self.assertFalse(_warm_refit_is_promotable(
            {'mean_gain_m': 0.0409, 'worse_rate': 0.14}, warm,
            minimum_gain_m=0.001, worse_tolerance=0.0))
        self.assertFalse(_warm_refit_is_promotable(
            {'mean_gain_m': 0.042, 'worse_rate': 0.151}, warm,
            minimum_gain_m=0.001, worse_tolerance=0.0))

    def test_warm_retention_matches_deployment_margin(self):
        valid = torch.tensor([
            [True, True, False],
            [True, False, True],
        ])
        first = torch.tensor([0, 0])
        teacher_logits = torch.distributions.Categorical(logits=torch.tensor([
            [0.2, -0.1, torch.finfo(torch.float32).min],
            [0.5, torch.finfo(torch.float32).min, -0.4],
        ])).logits
        teacher_q = torch.tensor([
            [0.01, 0.03, 9.0],
            [-0.02, 8.0, 0.04],
        ])
        student_q = teacher_q + torch.tensor([[3.0], [-5.0]])
        q_loss, actor_kl = _warm_retention_losses(
            teacher_logits, student_q, teacher_logits, teacher_q,
            valid, first, beta_m=0.001)
        self.assertAlmostEqual(float(q_loss), 0.0, places=7)
        self.assertAlmostEqual(float(actor_kl), 0.0, places=7)

        changed_logits = torch.distributions.Categorical(logits=torch.tensor([
            [1.2, -0.1, torch.finfo(torch.float32).min],
            [0.5, torch.finfo(torch.float32).min, -0.4],
        ])).logits
        changed_q = student_q.clone()
        changed_q[0, 1] += 0.01
        q_loss, actor_kl = _warm_retention_losses(
            changed_logits, changed_q, teacher_logits, teacher_q,
            valid, first, beta_m=0.001)
        self.assertGreater(float(q_loss), 0.0)
        self.assertGreater(float(actor_kl), 0.0)

    def test_seed_policy_ensemble_averages_member_outputs(self):
        torch.manual_seed(31)
        members = [
            CandidateSeedActorCritic(5, hidden_dim=16),
            CandidateSeedActorCritic(5, hidden_dim=16),
        ]
        features = torch.randn(4, 3, 5)
        valid = torch.tensor([
            [True, False, True],
            [True, True, False],
            [False, True, True],
            [True, True, True],
        ])
        member_outputs = [
            member.distribution_and_values(features, valid)
            for member in members
        ]
        policy = CandidateSeedPolicyEnsemble(members)
        distribution, value, feasibility = policy.distribution_and_values(
            features, valid)
        expected_logits = torch.stack([
            output[0].logits / len(member_outputs)
            for output in member_outputs]).sum(dim=0)
        expected_distribution = torch.distributions.Categorical(
            logits=expected_logits)
        self.assertTrue(torch.equal(
            distribution.logits, expected_distribution.logits))
        self.assertTrue(bool(torch.isfinite(distribution.logits).all()))
        self.assertTrue(torch.equal(
            value, torch.stack([output[1] for output in member_outputs]).mean(0)))
        self.assertTrue(torch.equal(
            feasibility,
            torch.stack([output[2] for output in member_outputs]).mean(0)))
        self.assertEqual(policy.feature_dim, 5)
        self.assertEqual(policy.hidden_dim, 16)
        self.assertEqual(policy.encoder_type, 'mean')
        self.assertEqual(policy.architecture, members[0].architecture)
        self.assertEqual(policy.size, 2)
        self.assertEqual(policy.aggregation, SEED_ENSEMBLE_AGGREGATION)
        policy.eval()
        self.assertFalse(policy.training)
        self.assertTrue(all(not member.training for member in policy.members))

    def test_load_seed_policy_supports_ensemble_and_preserves_legacy(self):
        torch.manual_seed(32)
        members = [
            CandidateSeedActorCritic(4, hidden_dim=8),
            CandidateSeedActorCritic(4, hidden_dim=8),
        ]
        features = torch.randn(3, 2, 4)
        valid = torch.ones(3, 2, dtype=torch.bool)
        checkpoint = {
            'seed_policy': members[0].state_dict(),
            'seed_policy_ensemble': [member.state_dict() for member in members],
            'seed_ensemble': {
                'format': SEED_ENSEMBLE_FORMAT,
                'aggregation': SEED_ENSEMBLE_AGGREGATION,
                'size': 2,
            },
            'seed_architecture': members[0].architecture,
            'feature_dim': 4,
            'hidden_dim': 8,
        }
        expected = CandidateSeedPolicyEnsemble(members)
        expected_outputs = expected.distribution_and_values(features, valid)
        legacy = {'model': members[0].state_dict()}
        legacy_outputs = members[0].distribution_and_values(features, valid)
        with tempfile.TemporaryDirectory() as tmp:
            ensemble_path = Path(tmp) / 'ensemble.pt'
            legacy_path = Path(tmp) / 'legacy.pt'
            torch.save(checkpoint, ensemble_path)
            torch.save(legacy, legacy_path)
            loaded, loaded_checkpoint = load_seed_policy(
                ensemble_path, torch.device('cpu'))
            loaded_legacy, _ = load_seed_policy(
                legacy_path, torch.device('cpu'))
        self.assertIsInstance(loaded, CandidateSeedPolicyEnsemble)
        self.assertEqual(loaded.size, 2)
        self.assertEqual(loaded_checkpoint['seed_ensemble']['size'], 2)
        actual_outputs = loaded.distribution_and_values(features, valid)
        for actual, wanted in zip(actual_outputs[1:], expected_outputs[1:]):
            self.assertTrue(torch.equal(actual, wanted))
        self.assertTrue(torch.equal(
            actual_outputs[0].logits, expected_outputs[0].logits))
        self.assertIsInstance(loaded_legacy, CandidateSeedActorCritic)
        actual_legacy = loaded_legacy.distribution_and_values(features, valid)
        self.assertTrue(torch.equal(
            actual_legacy[0].logits, legacy_outputs[0].logits))
        self.assertTrue(torch.equal(actual_legacy[1], legacy_outputs[1]))
        self.assertTrue(torch.equal(actual_legacy[2], legacy_outputs[2]))

    def test_seed_policy_ensemble_checkpoint_metadata_is_strict(self):
        policy = CandidateSeedActorCritic(4, hidden_dim=8)
        state = policy.state_dict()
        checkpoint = {
            'seed_policy': state,
            'seed_policy_ensemble': [state, state],
            'seed_ensemble': {
                'format': SEED_ENSEMBLE_FORMAT,
                'aggregation': SEED_ENSEMBLE_AGGREGATION,
                'size': 2,
            },
            'seed_architecture': policy.architecture,
            'feature_dim': 4,
            'hidden_dim': 8,
        }
        states, metadata = seed_policy_ensemble_states(checkpoint)
        self.assertEqual(len(states), 2)
        self.assertEqual(metadata['size'], 2)

        missing_metadata = dict(checkpoint)
        missing_metadata.pop('seed_ensemble')
        with self.assertRaisesRegex(ValueError, 'must appear together'):
            seed_policy_ensemble_states(missing_metadata)
        extra_metadata = dict(checkpoint)
        extra_metadata['seed_ensemble'] = dict(checkpoint['seed_ensemble'])
        extra_metadata['seed_ensemble']['extra'] = True
        with self.assertRaisesRegex(ValueError, 'invalid keys'):
            seed_policy_ensemble_states(extra_metadata)
        wrong_aggregation = dict(checkpoint)
        wrong_aggregation['seed_ensemble'] = dict(checkpoint['seed_ensemble'])
        wrong_aggregation['seed_ensemble']['aggregation'] = 'probability-mean'
        with self.assertRaisesRegex(ValueError, 'aggregation'):
            seed_policy_ensemble_states(wrong_aggregation)
        wrong_size = dict(checkpoint)
        wrong_size['seed_ensemble'] = dict(checkpoint['seed_ensemble'])
        wrong_size['seed_ensemble']['size'] = 1
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            seed_policy_ensemble_states(wrong_size)
        wrong_architecture = CandidateSeedActorCritic(
            5, hidden_dim=8).state_dict()
        wrong_state = dict(checkpoint)
        wrong_state['seed_policy_ensemble'] = [state, wrong_architecture]
        with self.assertRaisesRegex(ValueError, 'seed_architecture'):
            seed_policy_ensemble_states(wrong_state)

        incomplete_state = dict(state)
        incomplete_state.pop('actor.0.weight')
        incomplete = dict(checkpoint)
        incomplete['seed_policy_ensemble'] = [state, incomplete_state]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'incomplete.pt'
            torch.save(incomplete, path)
            with self.assertRaisesRegex(
                    ValueError, r'invalid seed_policy_ensemble\[1\] state'):
                load_seed_policy(path, torch.device('cpu'))

    def test_seed_policy_state_blend_is_strict_and_deterministic(self):
        base = {
            'weight': torch.tensor([0.0, 2.0], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
            'mask': torch.tensor([True, False]),
        }
        updated = {
            'weight': torch.tensor([2.0, 6.0], dtype=torch.float32),
            'counter': torch.tensor([3], dtype=torch.int64),
            'mask': torch.tensor([True, False]),
        }
        blended = _blend_state_dicts(base, updated, 0.25)
        self.assertTrue(torch.equal(
            blended['weight'], torch.tensor([0.5, 3.0])))
        self.assertEqual(blended['weight'].dtype, torch.float32)
        self.assertTrue(torch.equal(blended['counter'], base['counter']))
        self.assertTrue(torch.equal(blended['mask'], base['mask']))
        blended['counter'][0] = 9
        self.assertEqual(int(base['counter'][0]), 3)

        bad_buffer = copy.deepcopy(updated)
        bad_buffer['counter'][0] = 4
        with self.assertRaisesRegex(ValueError, 'non-floating state'):
            _blend_state_dicts(base, bad_buffer, 0.25)
        bad_shape = copy.deepcopy(updated)
        bad_shape['weight'] = torch.zeros(3)
        with self.assertRaisesRegex(ValueError, 'shape/dtype differs'):
            _blend_state_dicts(base, bad_shape, 0.25)
        with self.assertRaisesRegex(ValueError, r'in \[0,1\]'):
            _blend_state_dicts(base, updated, 1.1)

    def test_reset_fallback_without_pilot_is_first_valid(self):
        source = _batch(b=3, k=2)
        valid = torch.tensor([[False, True]]).expand(3, -1).clone()
        dataset = CachedSeedCandidateDataset(SeedCandidateBatch(
            source.q0, source.p0, source.line_dir, source.n_target, valid))
        policy = CandidateSeedActorCritic(31, hidden_dim=8)
        distribution = SeedPolicyLineDistribution(
            dataset, policy, _IdentityKin(),
            policy_prob=0.0, uniform_prob=0.0, fallback_prob=1.0)
        specs = distribution.sample(16)
        self.assertTrue(torch.equal(
            specs['q0'][:, 0], torch.ones(16)))

        invalid_pilot = CachedSeedCandidateDataset(
            SeedCandidateBatch(
                source.q0, source.p0, source.line_dir, source.n_target,
                torch.tensor([[True, False]]).expand(3, -1).clone()),
            fallback_index=1)
        distribution = SeedPolicyLineDistribution(
            invalid_pilot, policy, _IdentityKin(),
            policy_prob=0.0, uniform_prob=0.0, fallback_prob=1.0)
        specs = distribution.sample(16)
        self.assertTrue(torch.equal(
            specs['q0'][:, 0], torch.zeros(16)))

    def test_invalid_candidates_are_never_selected(self):
        torch.manual_seed(1)
        policy = CandidateSeedActorCritic(5, hidden_dim=32)
        features = torch.randn(4, 3, 5)
        valid = torch.tensor([
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ])
        dist, value, feasibility = policy.distribution_and_values(features, valid)
        samples = dist.sample((1000,))
        rows = torch.arange(4).expand(1000, -1)
        self.assertTrue(bool(valid[rows, samples].all()))
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool(torch.isfinite(feasibility).all()))
        self.assertEqual(int(samples[:, 1].unique().item()), 1)
        self.assertAlmostEqual(float(dist.entropy()[1].detach()), 0.0, places=6)

        padded_features = features.clone()
        padded_features[0, 1] = float('nan')
        padded_dist, padded_value, _ = policy.distribution_and_values(
            padded_features, valid)
        self.assertTrue(bool(torch.isfinite(padded_dist.logits).all()))
        self.assertTrue(bool(torch.isfinite(padded_value).all()))
        padded_features[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            policy.distribution_and_values(padded_features, valid)
        for bad_action in (-1, features.shape[1]):
            action = torch.zeros(features.shape[0], dtype=torch.long)
            action[0] = bad_action
            with self.assertRaisesRegex(ValueError, 'out of range'):
                policy.get_action_and_value(features, valid, action=action)

    def test_permutation_equivariance(self):
        torch.manual_seed(2)
        policy = CandidateSeedActorCritic(4, hidden_dim=16)
        features = torch.randn(3, 5, 4)
        valid = torch.ones(3, 5, dtype=torch.bool)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        dist_a, value_a, feas_a = policy.distribution_and_values(features, valid)
        dist_b, value_b, feas_b = policy.distribution_and_values(
            features[:, permutation], valid[:, permutation])
        self.assertTrue(torch.allclose(dist_a.logits[:, permutation], dist_b.logits))
        self.assertTrue(torch.allclose(feas_a[:, permutation], feas_b))
        self.assertTrue(torch.allclose(value_a, value_b))

    def test_attention_permutation_equivariance(self):
        torch.manual_seed(20)
        policy = CandidateSeedActorCritic(
            7, hidden_dim=16, encoder_type='attention', heads=4)
        policy.eval()
        features = torch.randn(3, 6, 7)
        valid = torch.tensor([
            [True, False, True, True, False, True],
            [False, True, True, False, True, True],
            [True, True, False, True, True, False],
        ])
        permutation = torch.tensor([4, 1, 5, 0, 3, 2])
        dist_a, value_a, feas_a = policy.distribution_and_values(features, valid)
        dist_b, value_b, feas_b = policy.distribution_and_values(
            features[:, permutation], valid[:, permutation])
        self.assertTrue(torch.allclose(
            dist_a.logits[:, permutation], dist_b.logits,
            rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            dist_a.probs[:, permutation], dist_b.probs,
            rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            feas_a[:, permutation], feas_b, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            value_a, value_b, rtol=1e-5, atol=1e-6))

    def test_attention_invalid_padding_does_not_change_valid_outputs(self):
        torch.manual_seed(21)
        policy = CandidateSeedActorCritic(
            6, hidden_dim=16, encoder_type='attention', heads=4)
        policy.eval()
        features = torch.randn(3, 4, 6)
        valid = torch.ones(3, 4, dtype=torch.bool)
        dist, value, feasibility = policy.distribution_and_values(
            features, valid)

        padding = torch.empty(3, 2, 6)
        padding[:, 0] = float('nan')
        padding[:, 1] = 1e38
        padded_features = torch.cat([features, padding], dim=1)
        padded_valid = torch.cat([
            valid, torch.zeros(3, 2, dtype=torch.bool)], dim=1)
        padded_dist, padded_value, padded_feasibility = (
            policy.distribution_and_values(padded_features, padded_valid))

        self.assertTrue(torch.allclose(
            dist.logits, padded_dist.logits[:, :4], rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            dist.probs, padded_dist.probs[:, :4], rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            value, padded_value, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            feasibility, padded_feasibility[:, :4],
            rtol=1e-5, atol=1e-6))
        self.assertTrue(bool(torch.isfinite(padded_value).all()))
        self.assertTrue(bool(torch.isfinite(padded_feasibility).all()))

    def test_attention_single_candidate_and_single_valid_candidate(self):
        torch.manual_seed(22)
        policy = CandidateSeedActorCritic(
            5, hidden_dim=16, encoder_type='attention', heads=4)
        policy.eval()
        features = torch.randn(3, 1, 5)
        valid = torch.ones(3, 1, dtype=torch.bool)
        dist, value, feasibility = policy.distribution_and_values(
            features, valid)
        self.assertTrue(torch.equal(dist.probs, torch.ones(3, 1)))
        self.assertTrue(torch.equal(dist.logits, torch.zeros(3, 1)))
        self.assertTrue(torch.equal(policy.select(features, valid), torch.zeros(
            3, dtype=torch.long)))
        self.assertTrue(bool(torch.isfinite(value).all()))
        self.assertTrue(bool(torch.isfinite(feasibility).all()))

        slots = torch.tensor([2, 0, 3])
        padded_features = torch.full((3, 4, 5), float('nan'))
        padded_features[torch.arange(3), slots] = features[:, 0]
        padded_valid = torch.zeros(3, 4, dtype=torch.bool)
        padded_valid[torch.arange(3), slots] = True
        padded_dist, padded_value, padded_feasibility = (
            policy.distribution_and_values(padded_features, padded_valid))
        rows = torch.arange(3)
        self.assertTrue(torch.equal(
            padded_dist.probs[rows, slots], torch.ones(3)))
        self.assertTrue(torch.equal(policy.select(
            padded_features, padded_valid), slots))
        self.assertTrue(torch.allclose(
            padded_value, value, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(
            padded_feasibility[rows, slots], feasibility[:, 0],
            rtol=1e-5, atol=1e-6))
        with self.assertRaisesRegex(ValueError, 'at least one valid'):
            policy.distribution_and_values(
                padded_features, torch.zeros_like(padded_valid))

    def test_seed_architecture_checkpoint_inference_is_strict(self):
        attention = CandidateSeedActorCritic(
            35, hidden_dim=168, encoder_type='attention', heads=4,
            layers=1, ff_mult=2)
        checkpoint = {
            'seed_policy': attention.state_dict(),
            'seed_architecture': attention.architecture,
            'feature_dim': 35,
            'hidden_dim': 168,
        }
        expected = SeedPolicyConfig(
            feature_dim=35, hidden_dim=168, encoder_type='attention',
            heads=4, layers=1, ff_mult=2)
        inferred = infer_seed_policy_config(checkpoint)
        self.assertEqual(inferred, expected)
        restored = CandidateSeedActorCritic(**inferred.to_dict())
        restored.load_state_dict(checkpoint['seed_policy'], strict=True)

        legacy_policy = CandidateSeedActorCritic(35, hidden_dim=256)
        legacy = {'model': legacy_policy.state_dict()}
        self.assertEqual(
            infer_seed_policy_config(legacy),
            SeedPolicyConfig(feature_dim=35, hidden_dim=256))

        malformed = dict(checkpoint)
        malformed['seed_architecture'] = dict(attention.architecture)
        malformed['seed_architecture'].pop('ff_mult')
        with self.assertRaisesRegex(ValueError, 'invalid keys'):
            infer_seed_policy_config(malformed)

        mismatched = dict(checkpoint)
        mismatched['hidden_dim'] = 160
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            infer_seed_policy_config(mismatched)

        missing_architecture = {'seed_policy': attention.state_dict()}
        with self.assertRaisesRegex(ValueError, 'seed_architecture'):
            infer_seed_policy_config(missing_architecture)

    def test_attention_parameter_budget_matches_mean_encoder(self):
        mean_policy = CandidateSeedActorCritic(35, hidden_dim=256)
        attention_policy = CandidateSeedActorCritic(
            35, hidden_dim=168, encoder_type='attention', heads=4,
            layers=1, ff_mult=2)
        mean_parameters = sum(
            parameter.numel() for parameter in mean_policy.parameters())
        attention_parameters = sum(
            parameter.numel() for parameter in attention_policy.parameters())
        self.assertLessEqual(
            abs(mean_parameters - attention_parameters), 32,
            (mean_parameters, attention_parameters))

    def test_seed_ppo_update_is_finite(self):
        torch.manual_seed(3)
        batch = _batch(b=16, k=3)
        features = torch.randn(16, 3, 6)
        policy = CandidateSeedActorCritic(6, hidden_dim=32)

        def rollout_fn(repeated, action):
            row = torch.arange(repeated.n_tasks)
            quality = repeated.q0[row, action, 0]
            task_offset = repeated.p0[:, 0]
            return quality + task_offset

        rollout = collect_seed_rollout(
            policy, batch, features, rollout_fn,
            samples_per_task=3, return_scale=1.0, center_within_task=True)
        grouped_adv = rollout.advantages.view(16, 3)
        self.assertTrue(torch.allclose(
            grouped_adv.mean(dim=1), torch.zeros(16), atol=1e-6))
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        cfg = SeedPPOConfig(
            update_epochs=2, n_minibatches=4, return_scale=1.0)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        stats = update_seed_policy(policy, optimizer, rollout, cfg)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))
        self.assertGreaterEqual(stats['seed/approx_kl'], 0.0)
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, policy.parameters())))
        with self.assertRaisesRegex(ValueError, 'return_scale'):
            collect_seed_rollout(
                policy, batch, features, rollout_fn,
                samples_per_task=1, return_scale=float('nan'))

        def nan_rollout(repeated, action):
            del action
            return torch.full((repeated.n_tasks,), float('nan'))

        with self.assertRaisesRegex(ValueError, 'non-finite'):
            collect_seed_rollout(
                policy, batch, features, nan_rollout,
                samples_per_task=1, return_scale=1.0)

    def test_dense_seed_policy_iteration_uses_every_action(self):
        torch.manual_seed(31)
        source = _batch(b=32, k=3)
        valid = source.valid.clone()
        valid[:, 1] = False
        batch = SeedCandidateBatch(
            source.q0, source.p0, source.line_dir, source.n_target, valid)
        features = batch.q0[..., :1].clone()
        policy = CandidateSeedActorCritic(1, hidden_dim=32)

        def rollout_fn(repeated, action):
            row = torch.arange(repeated.n_tasks)
            return repeated.q0[row, action, 0]

        rollout = collect_dense_seed_rollout(
            policy, batch, features, rollout_fn, return_scale=1.0)
        self.assertEqual(tuple(rollout.raw_returns.shape), (32, 3))
        self.assertTrue(torch.equal(
            rollout.raw_returns[0], torch.tensor([0.0, 0.0, 2.0])))
        cfg = DenseSeedConfig(
            learning_rate=1e-2, update_epochs=20, n_minibatches=2,
            target_improvement_kl=0.5, max_update_kl=10.0,
            return_scale=1.0)
        target, _, target_kl = _target_distribution(
            rollout.old_logits, rollout.returns, rollout.valid, cfg)
        self.assertTrue(torch.allclose(target.sum(dim=1), torch.ones(32)))
        self.assertTrue(torch.equal(target[:, 1], torch.zeros(32)))
        self.assertLessEqual(target_kl, cfg.target_improvement_kl + 1e-4)
        revived, _, _ = _target_distribution(
            torch.tensor([[0.0, -1000.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[True, True]]), cfg)
        self.assertGreater(float(revived[0, 1]), 0.0)
        raw_target, _, raw_kl = _target_distribution(
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[True, True]]), cfg)
        self.assertTrue(torch.allclose(raw_target.sum(dim=-1), torch.ones(1)))
        self.assertGreaterEqual(raw_kl, 0.0)
        self.assertLessEqual(raw_kl, cfg.target_improvement_kl + 1e-4)
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        stats = update_dense_seed_policy(policy, optimizer, rollout, cfg)
        with torch.no_grad():
            action = policy.select(features, batch.valid)
        self.assertGreater(float((action == 2).float().mean()), 0.9)
        self.assertLessEqual(
            stats['seed/improvement_target_kl'],
            cfg.target_improvement_kl + 1e-4)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_dense_padding_is_nan_safe_and_rank_loss_is_task_balanced(self):
        source = _batch(b=1, k=2)
        candidates = SeedCandidateBatch(
            source.q0, source.p0, source.line_dir, source.n_target,
            torch.tensor([[True, False]]))
        features = torch.zeros((1, 2, 1))
        policy = CandidateSeedActorCritic(1, hidden_dim=8)

        def rollout_fn(repeated, actions):
            del repeated, actions
            return torch.tensor([1.0, float('nan')])

        rollout = collect_dense_seed_rollout(
            policy, candidates, features, rollout_fn, return_scale=1.0)
        self.assertTrue(torch.equal(
            rollout.raw_returns, torch.tensor([[1.0, 0.0]])))
        target, _, target_kl = _target_distribution(
            rollout.old_logits, rollout.returns, rollout.valid,
            DenseSeedConfig())
        self.assertTrue(bool(torch.isfinite(target).all()))
        self.assertEqual(target_kl, 0.0)

        scores = torch.tensor([[0.0, 1.0, 2.0], [0.5, -0.5, 0.0]])
        returns = torch.tensor([[2.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        valid = torch.tensor([[True, True, True], [True, True, False]])
        combined, _ = _dense_rank_loss(scores, returns, valid, 1e-4)
        first, _ = _dense_rank_loss(
            scores[:1], returns[:1], valid[:1], 1e-4)
        second, _ = _dense_rank_loss(
            scores[1:], returns[1:], valid[1:], 1e-4)
        self.assertTrue(torch.allclose(combined, 0.5 * (first + second)))


class SeedDeploymentTest(unittest.TestCase):
    def test_legacy_actor_ignores_invalid_slots_and_q_nans(self):
        decision = select_seed_deployment(
            torch.tensor([[100.0, 1.0, 2.0]]),
            torch.tensor([[0.0, float('nan'), float('nan')]]),
            torch.tensor([[False, True, True]]),
            SeedDeploymentConfig())
        self.assertEqual(int(decision.selected_index[0]), 2)
        self.assertEqual(int(decision.proposal_index[0]), 2)

    def test_conservative_gate_uses_first_valid_and_inclusive_threshold(self):
        logits = torch.tensor([[9.0, 0.0, 2.0], [9.0, 0.0, 2.0]])
        feasibility = torch.tensor([[50.0, 1.0, 1.5],
                                    [50.0, 1.0, 1.49]])
        valid = torch.tensor([[False, True, True], [False, True, True]])
        decision = select_seed_deployment(
            logits, feasibility, valid,
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor', threshold=0.5))
        self.assertTrue(torch.equal(
            decision.first_valid_index, torch.tensor([1, 1])))
        self.assertTrue(torch.equal(
            decision.selected_index, torch.tensor([2, 1])))
        self.assertTrue(torch.equal(
            decision.accepted, torch.tensor([True, False])))

    def test_feasibility_proposal_fails_closed_on_nonfinite_scores(self):
        decision = select_seed_deployment(
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]),
            torch.tensor([[0.0, float('nan'), 2.0],
                          [float('nan'), float('nan'), float('nan')]]),
            torch.ones((2, 3), dtype=torch.bool),
            SeedDeploymentConfig(
                mode='conservative', proposal_head='feasibility',
                threshold=0.0))
        self.assertTrue(torch.equal(
            decision.selected_index, torch.tensor([2, 0])))
        self.assertTrue(torch.equal(
            decision.accepted, torch.tensor([True, False])))

    def test_reject_all_threshold_keeps_float64_precision(self):
        decision = select_seed_deployment(
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.ones((1, 2), dtype=torch.bool),
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor',
                threshold=float(np.nextafter(1.0, np.inf))))
        self.assertFalse(bool(decision.accepted[0]))
        self.assertEqual(int(decision.selected_index[0]), 0)

    def test_actor_q_proposal_combines_one_forward_actor_and_feasibility(self):
        logits = torch.tensor([[0.0, -0.2, -0.4]])
        feasibility = torch.tensor([[0.0, 0.02, 0.01]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        decision = select_seed_deployment(
            logits, feasibility, valid,
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor-q', threshold=0.01,
                proposal_q_weight=0.2, proposal_q_scale_m=0.01))
        self.assertEqual(int(logits.argmax(dim=-1)[0]), 0)
        self.assertEqual(int(decision.proposal_index[0]), 1)
        self.assertEqual(int(decision.selected_index[0]), 1)
        self.assertAlmostEqual(float(decision.predicted_gain[0]), 0.02)

    def test_actor_q_zero_weight_is_exact_actor_proposal(self):
        logits = torch.tensor([[0.1, 0.3, -0.2], [1.0, -1.0, 0.0]])
        feasibility = torch.tensor([
            [4.0, -5.0, 9.0],
            [float('nan'), float('nan'), float('nan')],
        ])
        valid = torch.tensor([[True, True, True], [True, False, True]])
        actor_q = select_seed_deployment(
            logits, feasibility, valid,
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor-q', threshold=0.0,
                proposal_q_weight=0.0, proposal_q_scale_m=0.01))
        actor = select_seed_deployment(
            logits, feasibility, valid,
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor', threshold=0.0))
        self.assertTrue(torch.equal(
            actor_q.proposal_index, actor.proposal_index))

    def test_actor_q_nonfinite_scores_fail_closed(self):
        decision = select_seed_deployment(
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            torch.tensor([[0.0, float('nan')],
                          [float('nan'), float('nan')]]),
            torch.ones((2, 2), dtype=torch.bool),
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor-q', threshold=0.0,
                proposal_q_weight=1.0, proposal_q_scale_m=0.01))
        self.assertTrue(torch.equal(
            decision.selected_index, torch.tensor([0, 0])))
        self.assertTrue(torch.equal(
            decision.accepted, torch.tensor([True, False])))

    def test_checkpoint_parser_is_strict_and_legacy_safe(self):
        self.assertEqual(
            deployment_config_from_checkpoint({}), SeedDeploymentConfig())
        legacy = SeedDeploymentConfig(
            mode='conservative', proposal_head='actor', threshold=0.5)
        self.assertEqual(legacy.to_dict(), {
            'mode': 'conservative', 'proposal_head': 'actor',
            'threshold': 0.5, 'comparison': 'ge',
        })
        with self.assertRaisesRegex(ValueError, 'missing'):
            deployment_config_from_checkpoint({
                'seed_deployment': {'mode': 'conservative'}})
        with self.assertRaisesRegex(ValueError, 'unknown'):
            deployment_config_from_checkpoint({
                'seed_deployment': {
                    'mode': 'actor', 'proposal_head': 'actor',
                    'threshold': 0.0, 'comparison': 'ge', 'mystery': 1,
                }})
        with self.assertRaisesRegex(ValueError, 'missing'):
            deployment_config_from_checkpoint({
                'seed_deployment': {
                    'mode': 'conservative', 'proposal_head': 'actor-q',
                    'threshold': 0.0, 'comparison': 'ge',
                }})
        actor_q = deployment_config_from_checkpoint({
            'seed_deployment': {
                'mode': 'conservative', 'proposal_head': 'actor-q',
                'threshold': 0.001, 'comparison': 'ge',
                'proposal_q_weight': 0.386812,
                'proposal_q_scale_m': 0.01,
            }})
        self.assertEqual(actor_q.proposal_head, 'actor-q')
        self.assertEqual(actor_q.to_dict(), {
            'mode': 'conservative', 'proposal_head': 'actor-q',
            'threshold': 0.001, 'comparison': 'ge',
            'proposal_q_weight': 0.386812,
            'proposal_q_scale_m': 0.01,
        })
        with self.assertRaisesRegex(ValueError, 'only configurable'):
            SeedDeploymentConfig(
                mode='conservative', proposal_head='actor', threshold=0.0,
                proposal_q_scale_m=0.02)


class ActorQMaterializerTest(unittest.TestCase):
    def test_weight_grid_is_fixed_sorted_unique_and_contains_block_zero(self):
        self.assertEqual(ACTOR_Q_WEIGHT_GRID, (
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50, 0.75, 1.0,
        ))
        self.assertEqual(tuple(sorted(set(ACTOR_Q_WEIGHT_GRID))),
                         ACTOR_Q_WEIGHT_GRID)
        self.assertEqual(ACTOR_Q_WEIGHT_GRID[0], 0.0)

    def test_actor_q_array_proposal_matches_single_forward_formula(self):
        actor = np.asarray([[0.0, -0.2, -0.4]], dtype=np.float32)
        feasibility = np.asarray([[0.0, 0.02, 0.01]], dtype=np.float32)
        valid = np.ones((1, 3), dtype=np.bool_)
        proposal, margin, first = _actor_q_proposal(
            actor, feasibility, valid, 0.2)
        expected = actor + np.float32(0.2) * feasibility / np.float32(0.01)
        self.assertTrue(np.array_equal(proposal, expected.argmax(axis=1)))
        self.assertTrue(np.array_equal(proposal, np.asarray([1])))
        self.assertTrue(np.array_equal(first, np.asarray([0])))
        self.assertAlmostEqual(float(margin[0]), 0.02)

    def test_geometry_macro_report_does_not_overweight_duplicate_rows(self):
        proposal = np.asarray([1, 1, 1], dtype=np.int64)
        margin = np.ones(3, dtype=np.float64)
        first = np.zeros(3, dtype=np.int64)
        baseline = np.zeros(3, dtype=np.int64)
        progress = np.asarray([
            [0.0, 0.01], [0.0, 0.01], [0.0, -0.001],
        ], dtype=np.float64)
        valid = np.ones((3, 2), dtype=np.bool_)
        report = _fixed_rule_report(
            proposal, margin, first, 0.0, progress, valid, baseline,
            ('same', 'same', 'other'))
        # Geometry means are +10 mm and exactly -1 mm. The repeated first
        # geometry receives one vote, and exactly -1 mm is not a >1 mm harm.
        self.assertAlmostEqual(report['paired_mean_delta_m'], 0.0045)
        self.assertEqual(report['n_geometry_groups'], 2)
        self.assertEqual(report['geometry_harm_rate_gt_1mm'], 0.0)
        self.assertEqual(report['row_harm_rate_gt_1mm'], 0.0)

        progress[2, 1] = -0.00101
        harmed = _fixed_rule_report(
            proposal, margin, first, 0.0, progress, valid, baseline,
            ('same', 'same', 'other'))
        self.assertAlmostEqual(harmed['geometry_harm_rate_gt_1mm'], 0.5)
        self.assertAlmostEqual(harmed['row_harm_rate_gt_1mm'], 1.0 / 3.0)

    def test_model_candidate_constraints_and_ties_are_deterministic(self):
        def candidate(weight, paired, harm, total_lcb, eligible=True):
            return {
                'weight': weight,
                'eligible': eligible,
                'model_report': {
                    'paired_mean_delta_m': paired,
                    'paired_lower_bound_m': paired - 0.001,
                    'geometry_harm_rate_gt_1mm': harm,
                    'total_gain_lower_bound_m': total_lcb,
                },
            }

        tied_small = candidate(0.10, 0.002, 0.01, 0.01)
        tied_large = candidate(0.20, 0.002, 0.01, 0.01)
        ineligible_better = candidate(0.40, 0.01, 0.0, 0.01, False)
        selected = _choose_model_candidate(
            (tied_large, ineligible_better, tied_small))
        self.assertIs(selected, tied_small)
        self.assertIsNone(_choose_model_candidate((ineligible_better,)))

        boundary = {
            'total_gain_lower_bound_m': 0.0,
            'paired_mean_delta_m': 1e-12,
            'geometry_harm_rate_gt_1mm': 0.06,
        }
        self.assertEqual(_promotion_reasons(boundary), [])
        for key, value in (
                ('total_gain_lower_bound_m', -1e-12),
                ('paired_mean_delta_m', 0.0),
                ('geometry_harm_rate_gt_1mm', 0.0600001)):
            failed = dict(boundary)
            failed[key] = value
            self.assertTrue(_promotion_reasons(failed))


class ProvenanceTest(unittest.TestCase):
    def test_versioned_checkpoint_schema_is_fail_closed(self):
        require_checkpoint_keys(
            {'model': {}}, ('model',), kind='seed')
        with self.assertRaisesRegex(ValueError, "'optimizer'"):
            require_checkpoint_keys(
                {'model': {}}, ('model', 'optimizer'), kind='seed')
        require_checkpoint_format_version(
            {'format_version': 4}, 4, kind='seed')
        for value in (None, 3, True, '4'):
            with self.assertRaisesRegex(ValueError, 'must be 4'):
                require_checkpoint_format_version(
                    {'format_version': value}, 4, kind='seed')

    def test_fingerprint_and_resume_settings_are_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / 'artifact.bin'
            artifact.write_bytes(b'candidate-cache-v1')
            saved = {
                'candidate_cache': file_fingerprint(artifact),
                'settings': {'samples_per_task': 4},
            }
            assert_same_provenance(saved, {
                'candidate_cache': file_fingerprint(artifact),
                'settings': {'samples_per_task': 4},
            })
            artifact.write_bytes(b'candidate-cache-v2')
            with self.assertRaisesRegex(ValueError, 'sha256'):
                assert_same_provenance(saved, {
                    'candidate_cache': file_fingerprint(artifact),
                    'settings': {'samples_per_task': 4},
                })
            with self.assertRaisesRegex(ValueError, 'samples_per_task'):
                assert_same_provenance(saved, {
                    'candidate_cache': saved['candidate_cache'],
                    'settings': {'samples_per_task': 8},
                })

    def test_bidirectional_phase_resume_does_not_repeat_controller(self):
        self.assertEqual(
            resume_position({'outer_round': 2, 'phase': 'round_complete'}),
            (3, False))
        self.assertEqual(
            resume_position({'outer_round': 3, 'phase': 'controller_complete'}),
            (3, True))
        self.assertEqual(
            resume_position({'outer_round': 0, 'phase': 'warmup_complete'}),
            (1, False))
        with self.assertRaisesRegex(ValueError, 'unknown checkpoint phase'):
            resume_position({'outer_round': 1, 'phase': 'half_written'})


class ResidualSeedTest(unittest.TestCase):
    def test_shield_backoff_and_exact_zero_fallback(self):
        base_q = torch.zeros((3, 7), dtype=torch.float32)
        base_q[1, 0] = -0.0
        p0 = base_q[:, :3].clone()
        line_dir = torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32).expand(3, -1).clone()
        n_target = torch.tensor(
            [[0.0, 0.0, 1.0]], dtype=torch.float32).expand(3, -1).clone()
        latent = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [float('nan'), 0.0, 0.0, 0.0],
        ])
        result = apply_residual_seed(
            _ResidualKin(), _ResidualCollision(), base_q, p0,
            line_dir, n_target, latent,
            enabled=torch.tensor([True, False, True]),
            basis_builder=_residual_basis,
        )

        self.assertEqual(float(result.alpha[0]), 0.5)
        self.assertAlmostEqual(float(result.q[0, 3]), 0.04, places=6)
        self.assertTrue(bool(result.valid.all()))
        self.assertTrue(bool(result.diagnostics.position.all()))
        self.assertTrue(bool(result.diagnostics.cone.all()))
        self.assertTrue(bool(result.diagnostics.collision_free.all()))
        self.assertTrue(bool(result.diagnostics.branch.all()))
        self.assertEqual(
            int(result.diagnostics.selected_index[0]), 1)

        # Disabled and invalid-latent rows both take the exact alpha-zero path.
        self.assertEqual(float(result.alpha[1]), 0.0)
        self.assertEqual(float(result.alpha[2]), 0.0)
        self.assertTrue(torch.equal(
            result.q[1].view(torch.int32), base_q[1].view(torch.int32)))
        self.assertTrue(torch.equal(
            result.q[2].view(torch.int32), base_q[2].view(torch.int32)))
        self.assertFalse(bool(result.diagnostics.input_finite[2]))

    def test_unsafe_base_fails_closed_without_changing_it(self):
        base_q = torch.zeros((1, 7), dtype=torch.float32)
        base_q[0, 0] = 2.0
        result = apply_residual_seed(
            _ResidualKin(), None, base_q, base_q[:, :3],
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros((1, 4)), basis_builder=_residual_basis)
        self.assertFalse(bool(result.valid[0]))
        self.assertEqual(float(result.alpha[0]), 0.0)
        self.assertTrue(torch.equal(
            result.q.view(torch.int32), base_q.view(torch.int32)))
        self.assertFalse(bool(result.diagnostics.joint_limits[0]))

    def test_zero_latent_is_identity_and_missing_collision_fails_closed(self):
        base_q = torch.zeros((1, 7), dtype=torch.float32)
        # A projector could reduce this small seed error, but a zero action
        # must not invoke it or alter the discrete baseline.
        p0 = torch.tensor([[0.001, 0.0, 0.0]], dtype=torch.float32)
        result = apply_residual_seed(
            _ResidualKin(), _ResidualCollision(), base_q, p0,
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros((1, 4)), basis_builder=_residual_basis)
        self.assertEqual(float(result.alpha[0]), 0.0)
        self.assertTrue(torch.equal(
            result.q.view(torch.int32), base_q.view(torch.int32)))

        no_collision = apply_residual_seed(
            _ResidualKin(), None, base_q, base_q[:, :3],
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.zeros((1, 4)), basis_builder=_residual_basis)
        self.assertFalse(bool(no_collision.valid[0]))
        self.assertFalse(bool(no_collision.diagnostics.collision_free[0]))

    def test_projection_cannot_amplify_requested_residual_norm(self):
        base_q = torch.zeros((1, 7), dtype=torch.float32)
        result = apply_residual_seed(
            _ResidualKin(), _ResidualCollision(), base_q,
            torch.tensor([[0.001, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[1e-4, 0.0, 0.0, 0.0]]),
            basis_builder=_residual_basis)
        self.assertTrue(bool(result.valid[0]))
        self.assertLessEqual(
            float(result.diagnostics.branch_distance[0]), 1.001e-4)

    def test_nonfinite_collision_margin_fails_closed(self):
        base_q = torch.zeros((1, 7), dtype=torch.float32)
        result = apply_residual_seed(
            _ResidualKin(), _InfiniteMarginCollision(), base_q,
            base_q[:, :3], torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros((1, 4)),
            basis_builder=_residual_basis)
        self.assertFalse(bool(result.valid[0]))
        self.assertFalse(bool(result.diagnostics.collision_free[0]))

    def test_nonunit_task_vectors_fail_closed(self):
        base_q = torch.zeros((1, 7), dtype=torch.float32)
        result = apply_residual_seed(
            _ResidualKin(), _ResidualCollision(), base_q, base_q[:, :3],
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 0.9995]]), torch.zeros((1, 4)),
            basis_builder=_residual_basis)
        self.assertFalse(bool(result.valid[0]))
        self.assertFalse(bool(result.diagnostics.input_finite[0]))

    def test_config_rejects_wider_or_variable_safety_schedule(self):
        self.assertEqual(ResidualSeedConfig().rho, 0.08)
        with self.assertRaisesRegex(ValueError, 'fixed safety schedule'):
            ResidualSeedConfig(alphas=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, 'position_tol'):
            ResidualSeedConfig(position_tol=0.006)
        with self.assertRaisesRegex(ValueError, 'collision_margin'):
            ResidualSeedConfig(collision_margin=-1e-3)
        with self.assertRaisesRegex(ValueError, 'rho'):
            ResidualSeedConfig(rho=0.081)
        with self.assertRaisesRegex(ValueError, 'cone_deg'):
            ResidualSeedConfig(cone_deg=31.0)


class ResidualPolicyTest(unittest.TestCase):
    def test_evaluation_output_is_new_atomic_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / 'candidates.npz'
            candidate.write_bytes(b'input-must-survive')
            output = _prepare_output_path(
                root / 'eval.npz', (candidate,))
            _atomic_savez_new(output, {'value': np.asarray([1, 2, 3])})
            with np.load(output, allow_pickle=False) as archive:
                self.assertTrue(np.array_equal(
                    archive['value'], np.asarray([1, 2, 3])))
            with self.assertRaises(FileExistsError):
                _atomic_savez_new(output, {'value': np.asarray([9])})
            with self.assertRaises(FileExistsError):
                _prepare_output_path(output, (candidate,))
            with self.assertRaisesRegex(ValueError, 'differ'):
                _prepare_output_path(candidate, (candidate,))
            self.assertEqual(candidate.read_bytes(), b'input-must-survive')

    def test_group_sampler_and_bootstrap_are_geometry_balanced(self):
        groups = geometry_groups(('a', 'a', 'b', 'c', 'c', 'c'))
        generator_a = torch.Generator().manual_seed(73)
        generator_b = torch.Generator().manual_seed(73)
        sampled_a = sample_group_balanced_indices(groups, 100, generator_a)
        sampled_b = sample_group_balanced_indices(groups, 100, generator_b)
        self.assertTrue(torch.equal(sampled_a, sampled_b))
        self.assertTrue(bool(((sampled_a >= 0) & (sampled_a < 6)).all()))

        estimate, low, high, n_groups = geometry_grouped_bootstrap_ci(
            np.array([1.0, 3.0, 10.0]),
            ('a' * 64, 'a' * 64, 'b' * 64),
            seed=79, samples=1000)
        self.assertEqual(n_groups, 2)
        self.assertEqual(estimate, 6.0)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)

    def test_bandit_loss_updates_gate_and_mean_toward_better_plus_branch(self):
        mean = torch.zeros((2, 4), requires_grad=True)
        noise = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ])
        _, log_prob = antithetic_gaussian_actions_and_log_prob(
            mean, noise, std=0.25)
        gate_logits = torch.zeros(2, requires_grad=True)
        loss, metrics = residual_bandit_loss(
            gate_logits, log_prob,
            base_return=torch.zeros(2),
            branch_return=torch.tensor([[2.0, 0.0], [1.0, 0.0]]),
            accepted_alpha=torch.ones((2, 2)),
            config=ResidualBanditConfig(
                return_scale=1.0, reject_penalty=0.0,
                gate_entropy_coef=0.0))
        loss.backward()
        self.assertLess(float(gate_logits.grad.mean()), 0.0)
        self.assertLess(float(mean.grad[:, 0].mean()), 0.0)
        self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))

    def test_zero_initialization_is_deployment_identity_and_freezes_selector(self):
        torch.manual_seed(71)
        selector = CandidateSeedActorCritic(5, hidden_dim=16)
        features = torch.randn((6, 3, 5))
        valid = torch.ones((6, 3), dtype=torch.bool)
        index = selector.select(features, valid)
        representation = selector.selected_representation(
            features, valid, index)
        head = ResidualSeedHead(32, hidden_dim=12)
        gate, latent = head.deterministic_action(representation)
        self.assertFalse(bool(gate.any()))
        self.assertTrue(torch.equal(latent, torch.zeros_like(latent)))

        gate_logit, mean = head(representation)
        (gate_logit.sum() + mean.sum()).backward()
        self.assertTrue(all(
            parameter.grad is None for parameter in selector.parameters()))
        self.assertTrue(any(
            parameter.grad is not None for parameter in head.parameters()))

    def test_antithetic_actions_have_pure_score_gradient(self):
        mean = torch.zeros((2, 4), requires_grad=True)
        noise = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.5, 0.0],
        ])
        actions, log_prob = antithetic_gaussian_actions_and_log_prob(
            mean, noise, std=0.25)
        self.assertEqual(tuple(actions.shape), (2, 2, 4))
        self.assertEqual(tuple(log_prob.shape), (2, 2))
        self.assertTrue(torch.allclose(actions[:, 0], -actions[:, 1]))
        self.assertFalse(actions.requires_grad)

        loss = -0.5 * (log_prob[:, 0] - log_prob[:, 1]).sum()
        loss.backward()
        self.assertLess(float(mean.grad[0, 0]), 0.0)
        self.assertGreater(float(mean.grad[1, 1]), 0.0)
        self.assertEqual(float(mean.grad[0, 1]), 0.0)


class TopKPrefixProbeTest(unittest.TestCase):
    def test_probe_chunk_cap_and_partial_padding(self):
        self.assertEqual(
            _effective_probe_task_chunk(4096, 5, 9, 2048), 341)
        self.assertEqual(
            _effective_probe_task_chunk(256, 5, 9, 2048), 256)
        index, n_real = _pad_indices(3, 5, 5, 4)
        self.assertEqual(n_real, 2)
        self.assertTrue(torch.equal(index, torch.tensor([3, 4, 4, 4])))

    def test_shortlist_is_stable_topk_union_first_without_duplicates(self):
        logits = torch.tensor([
            [1.0, 3.0, 2.0, 0.0],
            [2.0, 2.0, 1.0, -4.0],
        ])
        valid = torch.tensor([
            [True, True, True, True],
            [True, True, True, False],
        ])
        index, mask = topk_union_first_valid(logits, valid, top_k=2)
        self.assertTrue(torch.equal(index, torch.tensor([
            [1, 2, 0],
            [0, 1, 0],
        ])))
        self.assertTrue(torch.equal(mask, torch.tensor([
            [True, True, True],
            [True, True, False],
        ])))

    def test_shortlist_masks_padding_when_fewer_than_topk_are_valid(self):
        logits = torch.tensor([[9.0, 8.0, 7.0]])
        valid = torch.tensor([[False, False, True]])
        index, mask = topk_union_first_valid(logits, valid, top_k=3)
        self.assertEqual(tuple(index.shape), (1, 4))
        self.assertEqual(int(index[0, 0]), 2)
        self.assertTrue(torch.equal(index, torch.full((1, 4), 2)))
        self.assertTrue(bool(mask[0, 0]))
        self.assertEqual(int(mask.sum()), 1)

    def test_shortlist_rejects_topk_above_candidate_count(self):
        with self.assertRaisesRegex(ValueError, 'candidate count'):
            topk_union_first_valid(
                torch.zeros((1, 2)), torch.ones((1, 2), dtype=torch.bool),
                top_k=3)


class FR3CPUIntegrationTest(unittest.TestCase):
    def test_explicit_task_origin_survives_reset(self):
        env = NSRLBatchedEnv(
            EnvConfig(n_envs=1, max_steps=1, tcp_offset=0.2034,
                      observe_ray_error=True),
            line_dist=None, device='cpu')
        q = env.q_mid[None]
        p_fk, rot, _, _ = env.kin.tcp_fk_jac(q)
        n_target = rot[:, :, 2]
        line_dir = torch.linalg.cross(
            n_target, torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)
        line_dir /= line_dir.norm(dim=-1, keepdim=True)
        task_p0 = p_fk + torch.tensor([[0.001, 0.002, 0.003]])
        env.line_dist = ScriptedLineDistribution({
            'q0': q,
            'p0': task_p0,
            'line_dir': line_dir,
            'n_target': n_target,
        })
        obs = env.reset()
        self.assertTrue(torch.equal(env.p_start, task_p0))
        self.assertEqual(env.obs_dim, 34)
        self.assertEqual(tuple(obs.shape), (1, 34))
        ray_delta = p_fk - task_p0
        ray_along = (ray_delta * line_dir).sum(-1, keepdim=True)
        expected_lateral = (
            ray_delta - ray_along * line_dir) / LATERAL_SAFETY_NET
        self.assertTrue(torch.allclose(obs[:, -3:], expected_lateral))

    def test_task_anchor_features_and_two_step_rollout(self):
        cfg = EnvConfig(n_envs=2, dt=0.05, v=0.1, a_max=0.5,
                        cone_deg=30.0, max_steps=2, tcp_offset=0.2034)
        env = NSRLBatchedEnv(cfg, line_dist=None, device='cpu')
        q = env.q_mid.expand(2, 7).clone()
        p_fk, rot, _, _ = env.kin.tcp_fk_jac(q)
        n_target = rot[:, :, 2]
        axis = torch.tensor([1.0, 0.0, 0.0]).expand_as(n_target).clone()
        parallel = (n_target * axis).sum(-1).abs() > 0.9
        axis[parallel] = torch.tensor([0.0, 1.0, 0.0])
        line_dir = torch.linalg.cross(n_target, axis, dim=-1)
        line_dir /= line_dir.norm(dim=-1, keepdim=True)
        p0 = p_fk - 1e-4 * line_dir
        candidates = SeedCandidateBatch(
            q0=q[:, None, :].expand(-1, 2, -1).clone(),
            p0=p0,
            line_dir=line_dir,
            n_target=n_target,
            valid=torch.ones((2, 2), dtype=torch.bool),
        )
        features = initial_observation_features(env.kin, candidates)
        self.assertEqual(tuple(features.shape), (2, 2, 31))
        self.assertTrue(torch.allclose(features[:, 0, :7], torch.zeros((2, 7))))
        self.assertTrue(torch.allclose(features[:, 0, 14:17], line_dir.float()))
        self.assertTrue(torch.allclose(features[:, 0, 20:23], n_target.float()))
        self.assertTrue(torch.equal(features[:, 0, 27:31], torch.zeros((2, 4))))
        ray_features = initial_observation_features(
            env.kin, candidates, include_ray_error=True)
        self.assertEqual(tuple(ray_features.shape), (2, 2, 34))
        ray_delta = p_fk - p0
        ray_along = (ray_delta * line_dir).sum(-1, keepdim=True)
        expected_ray_error = (
            (ray_delta - ray_along * line_dir) / LATERAL_SAFETY_NET
        )[:, None, :].expand(-1, 2, -1)
        self.assertTrue(torch.allclose(
            ray_features[:, :, -3:], expected_ray_error.float()))

        directional_features = initial_observation_features(
            env.kin, candidates, include_ray_error=True,
            include_log_manip=True, include_directional_dynamics=True)
        self.assertEqual(tuple(directional_features.shape), (2, 2, 45))
        directional = directional_features[:, :, -10:]
        self.assertTrue(bool(torch.isfinite(directional).all()))
        self.assertTrue(bool((directional[:, :, 8] >= 0.0).all()))
        self.assertTrue(bool((directional[:, :, 8] <= 100.0).all()))
        self.assertTrue(bool((directional[:, :, 9] > 0.0).all()))

        bad_q0 = candidates.q0.clone()
        bad_q0[:, 1, 0] = env.lmt_up[0] + 0.1
        partly_invalid = SeedCandidateBatch(
            bad_q0, candidates.p0, candidates.line_dir,
            candidates.n_target, candidates.valid)
        validity = check_candidate_validity(
            env.kin, None, partly_invalid, position_tol_m=5e-3)
        self.assertTrue(bool(validity.valid[:, 0].all()))
        self.assertFalse(bool(validity.joint_limits[:, 1].any()))

        class ZeroAgent:
            @staticmethod
            def actor_mean(obs):
                return torch.zeros((obs.shape[0], 4), device=obs.device)

        result = rollout_selected_seeds(
            env, candidates, torch.zeros(2, dtype=torch.long),
            FrozenRLController(ZeroAgent()), gamma=0.99)
        self.assertTrue(torch.allclose(env.p_start, p0))
        self.assertTrue(torch.equal(result.episode_len, torch.full((2,), 2)))
        self.assertTrue(torch.equal(
            result.term_reason, torch.full((2,), TERM_TRUNCATED)))
        self.assertTrue(bool(torch.isfinite(result.discounted_return).all()))
        self.assertTrue(bool((result.progress_m > 0.008).all()))

        direct = rollout_seed_selection(
            env, candidates.select(torch.zeros(2, dtype=torch.long)),
            FrozenRLController(ZeroAgent()), gamma=0.99)
        for field in result.__dataclass_fields__:
            self.assertTrue(torch.equal(
                getattr(result, field), getattr(direct, field)), field)

    def test_env_snapshot_restore_preserves_next_transition(self):
        cfg = EnvConfig(
            n_envs=1, dt=0.02, v=0.05, max_steps=4,
            tcp_offset=0.2034, observe_ray_error=True)
        source = NSRLBatchedEnv(cfg, line_dist=None, device='cpu')
        restored = NSRLBatchedEnv(cfg, line_dist=None, device='cpu')
        q = source.q_mid[None]
        p0, rot, _, _ = source.kin.tcp_fk_jac(q)
        n_target = rot[:, :, 2]
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        if float((axis * n_target).sum().abs()) > 0.9:
            axis = torch.tensor([[0.0, 1.0, 0.0]])
        line_dir = torch.linalg.cross(n_target, axis, dim=-1)
        line_dir /= line_dir.norm(dim=-1, keepdim=True)
        source.line_dist = ScriptedLineDistribution({
            'q0': q, 'p0': p0, 'line_dir': line_dir,
            'n_target': n_target,
        })
        source.reset()
        first_action = torch.tensor([[0.1, -0.2, 0.05, 0.0]])
        source.step(first_action, auto_reset=False)
        state = snapshot_env_state(source)
        restore_env_state(restored, state)
        self.assertTrue(torch.equal(source.current_obs(), restored.current_obs()))

        next_action = torch.tensor([[-0.1, 0.05, 0.2, -0.15]])
        left = source.step(next_action, auto_reset=False)
        right = restored.step(next_action, auto_reset=False)
        for left_value, right_value in zip(left[:4], right[:4]):
            self.assertTrue(torch.equal(left_value, right_value))
        for name in state.__dataclass_fields__:
            self.assertTrue(torch.equal(
                getattr(source, name), getattr(restored, name)), name)

    def test_prefix_tie_uses_static_order_and_restarts_selected_seed(self):
        task_count = 1
        top_k = 1
        width = top_k + 1
        base_cfg = dict(
            dt=0.02, v=0.05, max_steps=3, tcp_offset=0.2034,
            observe_ray_error=True)
        probe_env = NSRLBatchedEnv(
            EnvConfig(n_envs=task_count * width, **base_cfg),
            line_dist=None, device='cpu')
        continuation_env = NSRLBatchedEnv(
            EnvConfig(n_envs=task_count, **base_cfg),
            line_dist=None, device='cpu')
        direct_env = NSRLBatchedEnv(
            EnvConfig(n_envs=task_count, **base_cfg),
            line_dist=None, device='cpu')
        q = continuation_env.q_mid[None]
        p0, rot, _, _ = continuation_env.kin.tcp_fk_jac(q)
        n_target = rot[:, :, 2]
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        if float((axis * n_target).sum().abs()) > 0.9:
            axis = torch.tensor([[0.0, 1.0, 0.0]])
        line_dir = torch.linalg.cross(n_target, axis, dim=-1)
        line_dir /= line_dir.norm(dim=-1, keepdim=True)
        candidates = SeedCandidateBatch(
            q0=q[:, None, :].expand(-1, 2, -1).clone(),
            p0=p0, line_dir=line_dir, n_target=n_target,
            valid=torch.ones((1, 2), dtype=torch.bool))

        class ZeroAgent:
            @staticmethod
            def actor_mean(obs):
                return torch.zeros((obs.shape[0], 4), device=obs.device)

        controller = FrozenRLController(ZeroAgent())
        lookahead = rollout_topk_prefix_lookahead(
            probe_env, continuation_env, candidates,
            actor_logits=torch.tensor([[0.0, 1.0]]),
            controller=controller, top_k=top_k, horizon_steps=1,
            alive_bonus=100.0, gamma=0.99, restart_selected=True)
        # Candidate 1 is actor-first and candidate 0 is appended first-valid.
        # Their identical prefix scores tie, so static shortlist position 0 wins.
        self.assertTrue(torch.equal(
            lookahead.shortlist_index, torch.tensor([[1, 0]])))
        self.assertEqual(int(lookahead.selected_shortlist_position[0]), 0)
        self.assertEqual(int(lookahead.probe_active_steps[0]), 2)
        self.assertEqual(int(lookahead.continuation_steps[0]), 3)
        self.assertEqual(int(lookahead.total_controller_steps[0]), 5)
        self.assertEqual(int(lookahead.rollout.episode_len[0]), 3)

        direct = rollout_selected_seeds(
            direct_env, candidates, torch.tensor([1]), controller, gamma=0.99)
        for field in lookahead.rollout.__dataclass_fields__:
            left = getattr(lookahead.rollout, field)
            right = getattr(direct, field)
            if torch.is_floating_point(left):
                self.assertTrue(torch.allclose(left, right, atol=1e-6), field)
            else:
                self.assertTrue(torch.equal(left, right), field)

    def test_probe_replaces_masked_nan_seed_with_first_valid(self):
        top_k = 2
        base_cfg = dict(
            dt=0.02, v=0.05, max_steps=1, tcp_offset=0.2034,
            observe_ray_error=True)
        probe_env = NSRLBatchedEnv(
            EnvConfig(n_envs=top_k + 1, **base_cfg),
            line_dist=None, device='cpu')
        continuation_env = NSRLBatchedEnv(
            EnvConfig(n_envs=1, **base_cfg),
            line_dist=None, device='cpu')
        q = continuation_env.q_mid[None]
        p0, rot, _, _ = continuation_env.kin.tcp_fk_jac(q)
        n_target = rot[:, :, 2]
        axis = torch.tensor([[1.0, 0.0, 0.0]])
        if float((axis * n_target).sum().abs()) > 0.9:
            axis = torch.tensor([[0.0, 1.0, 0.0]])
        line_dir = torch.linalg.cross(n_target, axis, dim=-1)
        line_dir /= line_dir.norm(dim=-1, keepdim=True)
        q0 = torch.stack([q[0], torch.full_like(q[0], torch.nan)])[None]
        candidates = SeedCandidateBatch(
            q0=q0, p0=p0, line_dir=line_dir, n_target=n_target,
            valid=torch.tensor([[True, False]]))

        class ZeroAgent:
            @staticmethod
            def actor_mean(obs):
                return torch.zeros((obs.shape[0], 4), device=obs.device)

        result = rollout_topk_prefix_lookahead(
            probe_env, continuation_env, candidates,
            actor_logits=torch.tensor([[0.0, 100.0]]),
            controller=FrozenRLController(ZeroAgent()), top_k=top_k,
            horizon_steps=1, restart_selected=True,
            score_objective='progress_m')
        self.assertTrue(torch.equal(
            result.shortlist_index, torch.zeros((1, 3), dtype=torch.long)))
        self.assertTrue(bool(torch.isfinite(result.rollout.progress_m).all()))


class ControllerCheckpointCompatibilityTest(unittest.TestCase):
    def test_controller_path_accepts_run_dir_or_agent_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            agent_path = run_dir / 'agent.pt'
            agent_path.write_bytes(b'checkpoint')
            (run_dir / 'config.yaml').write_text('env: {}\nppo: {}\n')
            self.assertEqual(resolve_controller_dir(run_dir), run_dir.resolve())
            self.assertEqual(resolve_controller_dir(agent_path), run_dir.resolve())
            self.assertEqual(
                controller_fingerprint(run_dir)['agent']['sha256'],
                controller_fingerprint(agent_path)['agent']['sha256'])
            other = run_dir / 'other.pt'
            other.write_bytes(b'checkpoint')
            with self.assertRaisesRegex(ValueError, 'named agent.pt'):
                resolve_controller_dir(other)

    def test_31d_controller_is_zero_expanded_to_34d(self):
        torch.manual_seed(21)
        old_agent = Agent(31, 4, hidden_dim=16)
        new_agent = Agent(34, 4, hidden_dim=16)
        load_controller_state_dict(new_agent, old_agent.state_dict())

        old_obs = torch.randn(8, 31)
        extended_obs = torch.cat([old_obs, torch.randn(8, 3)], dim=-1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(
                old_agent.actor_mean(old_obs),
                new_agent.actor_mean(extended_obs), atol=1e-7, rtol=1e-6))
            self.assertTrue(torch.allclose(
                old_agent.get_value(old_obs),
                new_agent.get_value(extended_obs), atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.equal(
            new_agent._actor_trunk[0].weight[:, 31:],
            torch.zeros((16, 3))))
        self.assertTrue(torch.equal(
            new_agent.critic[0].weight[:, 31:],
            torch.zeros((16, 3))))

        with self.assertRaisesRegex(ValueError, 'only 31-D -> 34-D'):
            load_controller_state_dict(Agent(35, 4, hidden_dim=16),
                                       old_agent.state_dict())

        old_optimizer = torch.optim.Adam(old_agent.parameters())
        old_agent.get_value(old_obs).sum().backward()
        old_optimizer.step()
        new_optimizer = torch.optim.Adam(new_agent.parameters())
        new_optimizer.load_state_dict(old_optimizer.state_dict())
        adapt_controller_optimizer_observation_state(new_optimizer, new_agent)
        critic_moments = new_optimizer.state[new_agent.critic[0].weight]
        self.assertEqual(
            tuple(critic_moments['exp_avg'].shape), (16, 34))
        self.assertTrue(torch.equal(
            critic_moments['exp_avg'][:, 31:], torch.zeros((16, 3))))


class PPOInjectionTest(unittest.TestCase):
    def test_search_teacher_keeps_current_action_and_antithetic_pairs(self):
        current = torch.zeros(3, 4)
        classical = torch.full((3, 4), 0.25)
        candidates = _paired_action_candidates(
            current, classical, sigma=0.08,
            generator=torch.Generator().manual_seed(12))
        self.assertEqual(tuple(candidates.shape), (3, 16, 4))
        self.assertTrue(torch.equal(candidates[:, 0], current))
        self.assertTrue(torch.equal(candidates[:, 1], classical))
        paired = candidates[:, 2:].reshape(3, 7, 2, 4)
        self.assertTrue(torch.allclose(
            paired.mean(dim=2), current[:, None], atol=1e-7))

    def test_conservative_search_label_is_local_supported_and_capped(self):
        progress = torch.zeros((2, 16))
        progress[:, 1] = 1.0  # A distant classical action must not be copied.
        progress[0, 2:4] = torch.tensor([0.020, 0.015])
        progress[1, 2] = 0.020  # Only one supporting local action.
        actions = torch.zeros((2, 16, 4))
        actions[:, 1] = 0.8
        actions[:, 2, 0] = 0.08
        actions[:, 3, 1] = 0.08
        search = {
            'slot_progress_m': progress,
            'candidate_action': actions,
            'current_action': torch.zeros((2, 4)),
        }
        label = _conservative_search_targets(
            search, minimum_gain_m=0.001,
            blend_gain_scale_m=0.020, maximum_blend=0.25,
            local_only=True, minimum_supporting_actions=2,
            maximum_target_action_delta=0.01)
        self.assertEqual(label['label_best_index'].tolist(), [2, 2])
        self.assertEqual(
            label['accepted_before_verification'].tolist(), [True, False])
        self.assertLessEqual(
            float(label['target_action_delta_norm'].max()), 0.0100001)

    def test_robust_controller_metrics_detect_tail_and_sign_regressions(self):
        baseline = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        policy = baseline + np.asarray(
            [0.002, 0.002, -0.0015, 0.001], dtype=np.float32)
        metrics = _robust_delta_metrics(
            policy, baseline, trim_fraction=0.0, clip_m=0.01,
            cvar_fraction=0.25, harm_threshold_m=0.001)
        self.assertGreater(metrics['paired_delta_clipped_mean_m'], 0.0)
        self.assertGreater(metrics['paired_win_minus_harm_rate'], 0.0)
        self.assertAlmostEqual(metrics['paired_win_rate'], 0.5)
        self.assertAlmostEqual(metrics['paired_harm_rate'], 0.25)

    def test_joint_controller_promotion_respects_coverage(self):
        evaluations = [
            {
                'policy_progress_mean_m': 0.50,
                'first_valid_progress_mean_m': 0.48,
            },
            {
                'policy_progress_mean_m': 0.52,
                'first_valid_progress_mean_m': 0.47,
            },
            {
                'policy_progress_mean_m': 0.51,
                'first_valid_progress_mean_m': 0.4795,
            },
        ]
        self.assertEqual(
            select_promoted_block(
                evaluations, first_valid_tolerance_m=0.001),
            2)
        self.assertEqual(
            select_promoted_block(
                evaluations, first_valid_tolerance_m=0.02),
            1)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            select_promoted_block([], first_valid_tolerance_m=0.0)

    def test_joint_controller_strict_promotion_uses_ci_and_harm(self):
        evaluations = [
            {
                'policy_progress_mean_m': 0.50,
                'first_valid_progress_mean_m': 0.48,
                'policy_harm_gt_1mm_rate': 0.10,
                'gain_vs_baseline_ci95_low_m': 0.0,
            },
            {
                'policy_progress_mean_m': 0.53,
                'first_valid_progress_mean_m': 0.48,
                'policy_harm_gt_1mm_rate': 0.10,
                'gain_vs_baseline_ci95_low_m': -0.001,
            },
            {
                'policy_progress_mean_m': 0.52,
                'first_valid_progress_mean_m': 0.48,
                'policy_harm_gt_1mm_rate': 0.12,
                'gain_vs_baseline_ci95_low_m': 0.001,
            },
            {
                'policy_progress_mean_m': 0.51,
                'first_valid_progress_mean_m': 0.48,
                'policy_harm_gt_1mm_rate': 0.10,
                'gain_vs_baseline_ci95_low_m': 0.0001,
            },
        ]
        self.assertEqual(select_promoted_block(
            evaluations, first_valid_tolerance_m=0.001,
            require_positive_ci=True, harm_rate_tolerance=0.0), 3)
        self.assertEqual(select_promoted_block(
            evaluations, first_valid_tolerance_m=0.001,
            require_positive_ci=True, harm_rate_tolerance=0.02), 2)

    def test_optimizer_and_reward_scaler_can_cross_training_phases(self):
        class FakeEnv:
            n_envs = 4
            obs_dim = 3
            act_dim = 2
            device = torch.device('cpu')

            def __init__(self):
                self.t = torch.zeros(self.n_envs, dtype=torch.long)

            def reset(self):
                self.t.zero_()
                return torch.zeros((self.n_envs, self.obs_dim))

            def step(self, action):
                self.t += 1
                obs = torch.zeros((self.n_envs, self.obs_dim))
                obs[:, :2] = action
                reward = torch.ones(self.n_envs)
                term = torch.zeros(self.n_envs, dtype=torch.bool)
                trunc = self.t >= 2
                done = trunc.clone()
                self.t[done] = 0
                return obs, reward, term, trunc, {
                    'terminal_obs': obs.clone(),
                    'episode_done': done,
                    'r_progress_mean': 1.0,
                    'n_episodes_done': int(done.sum()),
                }

        env = FakeEnv()
        cfg = PPOConfig(
            total_timesteps=16, n_steps=4, n_minibatches=1,
            update_epochs=1, hidden_dim=16, normalize_returns=True,
            anneal_lr=False)
        agent = Agent(3, 2, hidden_dim=16)
        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate)
        scaler = RewardScaler(env.n_envs, cfg.gamma, env.device)
        logged = []
        returned = ppo_train(
            cfg, env, env.device, agent=agent, optimizer=optimizer,
            reward_scaler=scaler, log_fn=logged.append)
        self.assertIs(returned, agent)
        self.assertGreater(len(optimizer.state), 0)
        state = scaler.state_dict()
        restored = RewardScaler(env.n_envs, cfg.gamma, env.device)
        restored.load_state_dict(state)
        self.assertTrue(torch.equal(restored.rms.mean, scaler.rms.mean))
        self.assertTrue(torch.equal(restored.rms.var, scaler.rms.var))
        self.assertGreater(logged[-1]['train/approx_kl'], 0.0)

    def test_phase_start_actor_anchor_is_training_only_and_logged(self):
        class FakeEnv:
            n_envs = 4
            obs_dim = 3
            act_dim = 2
            device = torch.device('cpu')

            def __init__(self):
                self.t = torch.zeros(self.n_envs, dtype=torch.long)

            def reset(self):
                self.t.zero_()
                return torch.zeros((self.n_envs, self.obs_dim))

            def step(self, action):
                self.t += 1
                obs = torch.zeros((self.n_envs, self.obs_dim))
                obs[:, :2] = action
                reward = torch.ones(self.n_envs)
                term = torch.zeros(self.n_envs, dtype=torch.bool)
                trunc = self.t >= 2
                self.t[trunc] = 0
                return obs, reward, term, trunc, {
                    'terminal_obs': obs.clone(),
                    'episode_done': trunc.clone(),
                    'r_progress_mean': 1.0,
                    'n_episodes_done': int(trunc.sum()),
                }

        env = FakeEnv()
        cfg = PPOConfig(
            total_timesteps=16, n_steps=4, n_minibatches=1,
            update_epochs=1, hidden_dim=16, normalize_returns=False,
            anneal_lr=False)
        agent = Agent(3, 2, hidden_dim=16)
        anchor = copy.deepcopy(agent)
        anchor_state = {
            key: value.detach().clone()
            for key, value in anchor.state_dict().items()
        }
        logged = []
        ppo_train(
            cfg, env, env.device, agent=agent,
            anchor_agent=anchor, actor_anchor_coef=0.5,
            log_fn=logged.append)
        self.assertIn('train/actor_anchor_loss', logged[-1])
        self.assertEqual(logged[-1]['train/actor_anchor_coef'], 0.5)
        for key, value in anchor.state_dict().items():
            self.assertTrue(torch.equal(value, anchor_state[key]))
        self.assertFalse(any(parameter.requires_grad
                             for parameter in anchor.parameters()))

        with self.assertRaisesRegex(ValueError, 'actor_anchor_coef'):
            ppo_train(
                cfg, env, env.device, agent=Agent(3, 2, hidden_dim=16),
                anchor_agent=Agent(3, 2, hidden_dim=16),
                actor_anchor_coef=float('nan'))

    def test_reward_scaler_restore_validates_full_state(self):
        scaler = RewardScaler(3, 0.97, 'cpu', epsilon=2e-4)
        state = scaler.state_dict()
        restored = RewardScaler(3, 0.5, 'cpu', epsilon=0.1)
        restored.load_state_dict(state)
        self.assertEqual(restored.gamma, 0.97)
        self.assertEqual(restored.epsilon, 2e-4)

        bad_shape = dict(state, return_acc=torch.zeros(4))
        with self.assertRaisesRegex(ValueError, 'return_acc shape'):
            restored.load_state_dict(bad_shape)
        with self.assertRaisesRegex(ValueError, 'gamma'):
            restored.load_state_dict(dict(state, gamma=float('nan')))
        with self.assertRaisesRegex(ValueError, 'epsilon'):
            restored.load_state_dict(dict(state, epsilon=0.0))
        bad_rms = dict(state['rms'], var=torch.tensor([float('nan')]))
        with self.assertRaisesRegex(ValueError, 'mean/variance'):
            restored.load_state_dict(dict(state, rms=bad_rms))
        bad_rms = dict(state['rms'], count=0.0)
        with self.assertRaisesRegex(ValueError, 'count'):
            restored.load_state_dict(dict(state, rms=bad_rms))

    def test_global_rng_round_trip_includes_numpy_and_torch(self):
        device = torch.device('cpu')
        seed_global_rng(123)
        state = global_rng_state(device)
        expected_numpy = np.random.random(4)
        expected_torch = torch.rand(4)
        np.random.random(7)
        torch.rand(7)
        restore_global_rng(state, device)
        self.assertTrue(np.array_equal(np.random.random(4), expected_numpy))
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))


class Joint2x2AnalysisTest(unittest.TestCase):
    def test_geometry_paired_decomposition_and_strict_mask_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = np.asarray([
                [True, True], [True, False], [True, True], [True, True],
            ], dtype=np.bool_)
            common = {
                'task_indices': np.arange(4, dtype=np.int64),
                'task_geometry_sha256': np.asarray(
                    ['a' * 64, 'a' * 64, 'b' * 64, 'c' * 64],
                    dtype='<U64'),
                'candidate_cache_sha256': np.asarray('c' * 64),
                'controller_config_sha256': np.asarray('d' * 64),
                'candidate_valid': valid,
                'first_valid_candidate_index': np.zeros(4, dtype=np.int64),
            }
            progress = (
                (0.0, 2.0, 4.0, 6.0),
                (1.0, 3.0, 5.0, 7.0),
                (2.0, 4.0, 6.0, 8.0),
                (4.0, 6.0, 8.0, 10.0),
            )
            s0_choice = np.zeros(4, dtype=np.int64)
            s1_choice = np.asarray([1, 0, 1, 1], dtype=np.int64)
            paths = {}
            for index, name in enumerate(CELL_NAMES):
                path = root / f'{name}.npz'
                paths[name] = path
                controller_sha = (
                    '1' * 64 if name.endswith('c0') else '2' * 64)
                np.savez_compressed(
                    path, **common,
                    seed_checkpoint_sha256=np.asarray(
                        'e' * 64 if name.startswith('s0') else 'f' * 64),
                    controller_agent_sha256=np.asarray(controller_sha),
                    controller_state_sha256=np.asarray(controller_sha),
                    policy_progress_m=np.asarray(
                        progress[index], dtype=np.float32),
                    policy_candidate_index=(
                        s0_choice if name.startswith('s0') else s1_choice),
                )
            cells = {
                name: load_evaluation(paths[name], name)
                for name in CELL_NAMES
            }
            result = analyze_joint_2x2(
                cells, bootstrap_seed=3, bootstrap_samples=200,
                clip_m=100.0)
            self.assertAlmostEqual(
                result['cells']['s0c0']['geometry_macro_progress_m'],
                11.0 / 3.0)
            self.assertAlmostEqual(
                result['effects']['joint']['geometry_macro_delta_m'], 4.0)
            self.assertEqual(result['seed_selection']['changed_rows'], 3)
            self.assertAlmostEqual(
                result['checks']['decomposition_residual_m'], 0.0)

            bad = root / 'bad.npz'
            with np.load(paths['s1c1'], allow_pickle=False) as source:
                payload = {key: source[key] for key in source.files}
            payload['candidate_valid'] = valid.copy()
            payload['candidate_valid'][1, 1] = True
            np.savez_compressed(bad, **payload)
            bad_cells = dict(cells)
            bad_cells['s1c1'] = load_evaluation(bad, 's1c1')
            with self.assertRaisesRegex(ValueError, 'valid'):
                analyze_joint_2x2(
                    bad_cells, bootstrap_seed=3, bootstrap_samples=10)


class ControllerPairAnalysisTest(unittest.TestCase):
    def test_strict_paired_statistics_and_full_oracle_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = np.asarray([
                [True, True], [True, False], [True, True], [True, True],
            ], dtype=np.bool_)
            common = {
                'task_indices': np.arange(4, dtype=np.int64),
                'task_geometry_sha256': np.asarray(
                    ['a' * 64, 'a' * 64, 'b' * 64, 'c' * 64],
                    dtype='<U64'),
                'candidate_cache_sha256': np.asarray('c' * 64),
                'seed_checkpoint_sha256': np.asarray('e' * 64),
                'candidate_valid': valid,
                'first_valid_candidate_index': np.zeros(4, dtype=np.int64),
                'policy_candidate_index': np.asarray(
                    [1, 0, 1, 1], dtype=np.int64),
                'seed_probe_enabled': np.bool_(False),
            }
            c0_progress = np.asarray(
                [0.10, 0.20, 0.30, 0.40], dtype=np.float32)
            delta = np.asarray(
                [0.002, -0.003, 0.004, 0.0], dtype=np.float32)
            paths = {}
            for index, name in enumerate(('c0', 'c1')):
                progress = c0_progress + index * delta
                path = root / f'{name}.npz'
                paths[name] = path
                np.savez_compressed(
                    path, **common,
                    controller_agent_sha256=np.asarray(str(index + 1) * 64),
                    controller_config_sha256=np.asarray(str(index + 3) * 64),
                    controller_state_sha256=np.asarray(str(index + 5) * 64),
                    policy_progress_m=progress,
                    first_valid_progress_m=progress - 0.02,
                    best_progress_m=progress + 0.05,
                    policy_episode_len=np.asarray(
                        [10, 20, 30, 40], dtype=np.int64) + index,
                )
            evaluations = {
                name: load_controller_evaluation(paths[name], name)
                for name in ('c0', 'c1')
            }
            result = analyze_controller_pair(
                evaluations, bootstrap_seed=3, bootstrap_samples=200)
            self.assertAlmostEqual(
                result['delta']['row_mean_delta_m'], 0.00075, places=7)
            self.assertAlmostEqual(
                result['delta']['geometry_macro_delta_m'],
                (-0.0005 + 0.004) / 3.0, places=7)
            self.assertEqual(
                result['delta']['row_win_gt_threshold_count'], 2)
            self.assertEqual(
                result['delta']['row_harm_gt_threshold_count'], 1)
            self.assertAlmostEqual(
                result['full_oracle']['c0']['row_mean']['capture'],
                0.02 / 0.07)
            self.assertAlmostEqual(
                result['full_oracle']['c1']['row_mean'][
                    'policy_episode_len_steps'], 26.0)

            with np.load(paths['c1'], allow_pickle=False) as source:
                payload = {key: source[key] for key in source.files}
            payload['policy_candidate_index'] = np.zeros(4, dtype=np.int64)
            bad = root / 'bad.npz'
            np.savez_compressed(bad, **payload)
            bad_pair = dict(evaluations)
            bad_pair['c1'] = load_controller_evaluation(bad, 'c1')
            with self.assertRaisesRegex(ValueError, 'policy_candidate_index'):
                analyze_controller_pair(
                    bad_pair, bootstrap_seed=3, bootstrap_samples=10)


if __name__ == '__main__':
    unittest.main()
