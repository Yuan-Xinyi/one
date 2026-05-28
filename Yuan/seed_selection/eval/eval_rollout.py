"""Operational eval: DP samples → Newton IK to exact (p0, R_target) →
rollout from that EXACT start → compare achieved L_clean against labels.

This is the metric that matches deployment: the diffusion model returns a
seed; you refine it onto the task's start manifold and roll out from there.
The joint-distance eval (`eval_q0.py`) is a fast model-internal sanity
check; this one is the operational headline.

Pipeline per task:
  1. Sample N q0_dp from the diffusion model (with optional CFG).
  2. For each q0_dp, run `newton_project(kin, q0_dp, p0, R_target_strict, ...)`
     to obtain q0_refined that lands TCP on p0 with z=n_target (full 6-DOF).
  3. Batched rollout `(q0_refined_b, c)` to get the actual achieved L_clean
     of each sample's trajectory.
  4. Aggregate per task:
       - IK convergence rate
       - L_rollout distribution (per-sample)
       - best_of_N_L, ratio = best_of_N_L / max(label_L_clean)
       - improvement over L_seed
       - joint-distance coverage on REFINED samples (apples-to-apples with
         labels which are already 6-DOF refined)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Yuan.fr3_dit.training.task_cond_dit_q0 import denormalize_q
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import newton_project
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.seed_selection.diffusion.dataset import SeedSelectionDataset
from Yuan.seed_selection.diffusion.sampling import ddim_sample_q0, load_ckpt
from Yuan.seed_selection.smm.label_builder import _build_R_target_strict
from Yuan.seed_selection.smm.rollout_batched import batched_rollout_many


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz'
DEFAULT_CKPT = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/q0_20k_cfg_mirror_ckpts/step_300000.pt'
DEFAULT_CONFIG = _REPO_ROOT / 'Yuan/RL_controller/config.yaml'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, default=DEFAULT_CKPT)
    p.add_argument('--data', type=Path, default=DEFAULT_NPZ)
    p.add_argument('--n-samples', type=int, default=16)
    p.add_argument('--max-tasks', type=int, default=None)
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--cfg-w', type=float, default=1.0)
    p.add_argument('--match-rad', type=float, default=0.5,
                   help='joint-space L2 (rad) for "branch covered" (refined samples).')
    p.add_argument('--L-success', type=float, default=0.20,
                   help='per-sample L_rollout >= this counts as a "successful" sample.')
    p.add_argument('--L-recover-frac', type=float, default=0.90,
                   help='best_of_N_L >= L_recover_frac * max(label_L) → fully recovered task.')
    p.add_argument('--drop-pierce', action='store_true',
                   help='post-process: drop samples whose refined q has arm piercing the '
                        'bounded plane (same criterion as the data filter). Pierced samples '
                        'are excluded from best_of_N statistics.')
    p.add_argument('--pierce-plane-extent-m', type=float, default=1.5,
                   help='plane forward extent for inference piercing check (m).')
    p.add_argument('--pierce-exclude-links', type=int, nargs='*', default=[0, 1],
                   help='links to exclude from piercing check (base/shoulder by default).')
    p.add_argument('--no-plane-filter', action='store_true',
                   help='disable the dataset plane-collision filter (use raw dataset). '
                        'Use this when the ckpt was trained on a non-filtered split.')
    p.add_argument('--which', choices=['train', 'val', 'all'], default='val')
    p.add_argument('--split-file', type=Path, default=None)
    p.add_argument('--use-model', action='store_true')
    p.add_argument('--n-envs-rollout', type=int, default=64,
                   help='env.n_envs for batched rollout of refined samples.')
    p.add_argument('--config-yaml', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--target-distance-m', type=float, default=1.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-prefix', default='eval_rollout')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model, schedule, model_cfg, step = load_ckpt(args.ckpt, device, use_ema=not args.use_model)
    print(f'[eval-rollout] ckpt={args.ckpt} step={step} cfg_w={args.cfg_w} '
          f'weights={"model" if args.use_model else "ema"}')

    ds = SeedSelectionDataset(args.data,
                              plane_collision_path=(None if args.no_plane_filter else 'auto'))
    print(f'[eval-rollout] dataset: {len(ds)} entries')

    # Apply train/val split filter (same logic as eval_q0).
    subset_mask = np.ones(len(ds), dtype=bool)
    if args.which != 'all':
        split_path = args.split_file or (args.ckpt.parent / 'split.json')
        if not split_path.exists():
            raise SystemExit(f'--which={args.which} but no split.json at {split_path}')
        s = json.loads(split_path.read_text())
        if int(s['n_total']) != len(ds):
            raise SystemExit(f'split n_total={s["n_total"]} != ds size {len(ds)}')
        subset_idx = np.array(s[f'{args.which}_idx'], dtype=np.int64)
        subset_mask = np.zeros(len(ds), dtype=bool)
        subset_mask[subset_idx] = True
        print(f'[eval-rollout] using {args.which} split: {int(subset_mask.sum())} entries')

    # n>=2 multi-label tasks only (so "label branch covered" is meaningful).
    multi_idx = np.where((ds.n_labels >= 2) & subset_mask)[0]
    if args.max_tasks is not None:
        multi_idx = multi_idx[:args.max_tasks]
    print(f'[eval-rollout] {len(multi_idx)} multi-label tasks, {args.n_samples} samples each')

    # Build env for rollout.
    import yaml
    with open(args.config_yaml, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = EnvConfig(**{**cfg_yaml['env'], 'n_envs': args.n_envs_rollout})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    lo_np = env.kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = env.kin.lmt_up.detach().cpu().numpy().astype(np.float32)

    # Plane-piercing post-processing setup.
    if args.drop_pierce:
        plane_coll = FR3SphereCollision(device=device, dtype=env.kin.dtype)
        pierce_link_idx = plane_coll.link_indices.detach().cpu().numpy().astype(np.int32)
        pierce_keep = ~np.isin(pierce_link_idx, args.pierce_exclude_links)
        pierce_keep_t = torch.from_numpy(pierce_keep).to(device)
        print(f'[eval-rollout] --drop-pierce ON  '
              f'(plane_extent_m={args.pierce_plane_extent_m}, '
              f'exclude_links={args.pierce_exclude_links})')

    @torch.no_grad()
    def check_arm_pierces_plane(refined_q_np, p0_np_b, n_np_b, line_dir_np_b):
        """Returns (B,) bool — does refined_q[b]'s arm straddle the bounded plane?"""
        qt = torch.as_tensor(refined_q_np, device=device, dtype=env.kin.dtype)
        link_tfs = env.kin.link_transforms(qt)
        sp = plane_coll.sphere_positions(link_tfs)  # (B, S, 3)
        p0_t = torch.as_tensor(p0_np_b, device=device, dtype=env.kin.dtype)
        n_t  = torch.as_tensor(n_np_b,  device=device, dtype=env.kin.dtype)
        d_t  = torch.as_tensor(line_dir_np_b, device=device, dtype=env.kin.dtype)
        signed = ((sp - p0_t[:, None, :]) * n_t[:, None, :]).sum(dim=-1)
        proj_d = ((sp - p0_t[:, None, :]) * d_t[:, None, :]).sum(dim=-1)
        over_plane = (proj_d >= 0.0) & (proj_d <= float(args.pierce_plane_extent_m))
        valid = over_plane & pierce_keep_t.bool().unsqueeze(0)
        signed_m = signed.masked_fill(~valid, float('nan'))
        has_pos = (signed_m > 0.0).any(dim=-1)
        has_neg = (signed_m < 0.0).any(dim=-1)
        return (has_pos & has_neg).cpu().numpy()

    # L_seed lookup (raw NPZ; align to ds indices)
    z = np.load(args.data, allow_pickle=False)
    keep_mask = np.isin(z['status'], ['kept', 'edge', 'edge_seed_fallback']) & (z['n_labels'] >= 1)
    src_idx = np.where(keep_mask)[0]
    L_seed_per_entry = z['L_seed'][src_idx]

    def bucket(L):
        if L < 0.15: return 'weak'
        if L < 0.20: return 'medium-weak'
        if L < 0.30: return 'medium'
        return 'strong'

    # Per-task accumulators
    per_task = {
        'idx': [],
        'n_labels': [],
        'L_seed': [],
        'max_label_L': [],
        'ik_success_count': [],
        'pierce_count': [],          # # samples with refined arm piercing plane
        'mean_rollout_L': [],
        'best_rollout_L': [],
        'recover_ratio': [],
        'success_count': [],
        'branches_covered': [],   # # labels matched (joint-dist on REFINED q0)
        'full_cov': [],
        'bucket': [],
    }
    all_min_dists_refined = []
    all_rollout_L = []
    all_ik_ok = []
    all_pierce = []

    BATCH_TASKS = 32  # tunable; each batch = BATCH_TASKS * n_samples rollouts
    for start in range(0, len(multi_idx), BATCH_TASKS):
        batch_t = multi_idx[start:start + BATCH_TASKS]
        Bt = len(batch_t)
        M = args.n_samples

        # Per-task c
        c_per_task_np = np.stack([
            np.concatenate([ds.cs_p0[i], ds.cs_line_dir[i], ds.cs_n_target[i]])
            for i in batch_t
        ], axis=0).astype(np.float32)
        c_t = torch.from_numpy(c_per_task_np).to(device)
        c_rep = c_t.repeat_interleave(M, dim=0)   # (Bt*M, 9)

        # Sample from diffusion
        q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                                num_steps=args.ddim_steps, cfg_w=args.cfg_w)
        sample_q0 = denormalize_q(q_norm).cpu().numpy().astype(np.float32)  # (Bt*M, 7)

        # Newton-refine each sample to its task's (p0, R_target_strict)
        # CPU loop; total ~Bt*M iterations of cheap newton.
        refined_q = np.zeros_like(sample_q0)
        ok_mask = np.zeros(Bt * M, dtype=bool)
        for bi in range(Bt):
            ti = int(batch_t[bi])
            p0_np = ds.cs_p0[ti].astype(np.float32)
            d_np  = ds.cs_line_dir[ti].astype(np.float32)
            n_np  = ds.cs_n_target[ti].astype(np.float32)
            R_tgt = _build_R_target_strict(n_np, d_np)
            for si in range(M):
                idx = bi * M + si
                q_seed = sample_q0[idx]
                q_ref, ok, _err = newton_project(env.kin, q_seed, p0_np, R_tgt, lo_np, hi_np)
                refined_q[idx] = q_ref
                ok_mask[idx] = bool(ok)

        # Batched rollout of refined samples. For non-converged samples we still
        # run the rollout (will likely terminate fast on jl/collision/cone) and
        # mark L_rollout as NaN so they don't pollute the achievement metrics.
        cs_list = []
        for bi in range(Bt):
            ti = int(batch_t[bi])
            c_dict = {'p0': torch.as_tensor(ds.cs_p0[ti], device=device, dtype=env.kin.dtype),
                      'line_dir': torch.as_tensor(ds.cs_line_dir[ti], device=device, dtype=env.kin.dtype),
                      'n_target': torch.as_tensor(ds.cs_n_target[ti], device=device, dtype=env.kin.dtype)}
            cs_list += [c_dict] * M
        qs_t = torch.as_tensor(refined_q, device=device, dtype=env.kin.dtype)
        res = batched_rollout_many(qs_t, cs_list, env=env, controller=controller,
                                    target_distance_m=args.target_distance_m)
        L_rollout = res['L'].astype(np.float32)
        L_rollout[~ok_mask] = np.nan

        # --drop-pierce: check refined q for plane piercing; pierced samples
        # have their L_rollout set to NaN so they are excluded from best_of_N.
        if args.drop_pierce:
            # Build batched (p0, n, d) for the Bt*M refined qs.
            p0_b = np.repeat(np.stack([ds.cs_p0[int(ti)] for ti in batch_t]), M, axis=0)
            n_b  = np.repeat(np.stack([ds.cs_n_target[int(ti)] / (np.linalg.norm(ds.cs_n_target[int(ti)]) + 1e-12) for ti in batch_t]), M, axis=0)
            d_b  = np.repeat(np.stack([ds.cs_line_dir[int(ti)] / (np.linalg.norm(ds.cs_line_dir[int(ti)]) + 1e-12) for ti in batch_t]), M, axis=0)
            pierces = check_arm_pierces_plane(refined_q, p0_b, n_b, d_b)
            L_rollout[pierces] = np.nan
        else:
            pierces = np.zeros(Bt * M, dtype=bool)

        # Per-task aggregation
        for bi in range(Bt):
            ti = int(batch_t[bi])
            n_lab = int(ds.n_labels[ti])
            labels = ds.labels_q0[ti, :n_lab]
            # Label L_clean is not kept by SeedSelectionDataset; fetch from the raw NPZ.
            label_Ls = np.asarray(z['labels_L_clean'][src_idx[ti], :n_lab], dtype=np.float32)
            max_label_L = float(np.nanmax(label_Ls)) if np.isfinite(label_Ls).any() else float('nan')

            sl = slice(bi * M, (bi + 1) * M)
            sample_refined = refined_q[sl]
            sample_Ls = L_rollout[sl]
            ok_count = int(ok_mask[sl].sum())

            # Joint-distance coverage on REFINED samples
            d = np.linalg.norm(sample_refined[:, None, :] - labels[None, :, :], axis=-1)  # (M, n)
            min_dist = d.min(axis=1)
            all_min_dists_refined.append(min_dist)
            covered = (d < args.match_rad).any(axis=0)   # (n,) bool
            n_branches_covered = int(covered.sum())
            full_cov = bool(covered.all())

            # Rollout metrics
            valid_Ls = sample_Ls[np.isfinite(sample_Ls)]
            mean_L = float(np.mean(valid_Ls)) if len(valid_Ls) else float('nan')
            best_L = float(np.max(valid_Ls)) if len(valid_Ls) else float('nan')
            ratio = best_L / max_label_L if (np.isfinite(best_L) and max_label_L > 0) else float('nan')
            success_count = int((sample_Ls >= args.L_success).sum())

            pierce_count = int(pierces[sl].sum()) if args.drop_pierce else 0
            per_task['idx'].append(int(src_idx[ti]))
            per_task['n_labels'].append(n_lab)
            per_task['L_seed'].append(float(L_seed_per_entry[ti]))
            per_task['max_label_L'].append(max_label_L)
            per_task['ik_success_count'].append(ok_count)
            per_task['pierce_count'].append(pierce_count)
            per_task['mean_rollout_L'].append(mean_L)
            per_task['best_rollout_L'].append(best_L)
            per_task['recover_ratio'].append(ratio)
            per_task['success_count'].append(success_count)
            per_task['branches_covered'].append(n_branches_covered)
            per_task['full_cov'].append(full_cov)
            per_task['bucket'].append(bucket(L_seed_per_entry[ti]))

            all_rollout_L.append(sample_Ls)
            all_ik_ok.append(ok_mask[sl])
            all_pierce.append(pierces[sl])

        if (start // BATCH_TASKS) % 5 == 0:
            done = start + Bt
            print(f'  ...{done}/{len(multi_idx)} tasks  '
                  f'(latest task full_cov={full_cov}, ok={ok_count}/{M}, best_L={best_L:.3f})',
                  flush=True)

    # Aggregate
    all_min_dists_refined = np.concatenate(all_min_dists_refined)
    all_rollout_L = np.concatenate(all_rollout_L)
    all_ik_ok = np.concatenate(all_ik_ok)
    all_pierce = np.concatenate(all_pierce) if len(all_pierce) else np.zeros(0, dtype=bool)
    N = len(per_task['idx'])
    A = lambda k: np.asarray(per_task[k])

    print('\n' + '=' * 78)
    print(f'Rollout-based eval — N={N} multi-label tasks, {args.n_samples} samples/task')
    print(f'cfg_w={args.cfg_w}, match_rad={args.match_rad}, L_success={args.L_success}'
          f'{", drop_pierce=ON" if args.drop_pierce else ""}')
    print('=' * 78)

    print(f'\nIK convergence rate (per-sample): {100*all_ik_ok.mean():.1f}%  '
          f'({int(all_ik_ok.sum())}/{len(all_ik_ok)})')
    if args.drop_pierce:
        n_pierced = int(all_pierce.sum())
        print(f'Plane-piercing samples (refined q): {100*all_pierce.mean():.1f}%  '
              f'({n_pierced}/{len(all_pierce)}) — excluded from best_of_N')
        # How many tasks lost all their samples to piercing?
        tasks_no_valid = int(((A("pierce_count") + (args.n_samples - A("ik_success_count"))) >= args.n_samples).sum())
        print(f'Tasks where all samples were pierced OR IK-failed: {tasks_no_valid}/{N} '
              f'({100*tasks_no_valid/max(N,1):.1f}%)')

    print(f'\n--- Per-sample L_rollout (ALL samples, NaN excluded) ---')
    valid_L = all_rollout_L[np.isfinite(all_rollout_L)]
    print(f'  total valid: {len(valid_L)} / {len(all_rollout_L)}')
    if len(valid_L):
        for q in [10, 25, 50, 75, 90]:
            print(f'  p{q}: {np.percentile(valid_L, q):.3f}')
        print(f'  L >= 0.10 (above min_abs): {100*(valid_L>=0.10).mean():.1f}%')
        print(f'  L >= 0.20 (acceptable):    {100*(valid_L>=0.20).mean():.1f}%')
        print(f'  L >= 0.30 (good):          {100*(valid_L>=0.30).mean():.1f}%')

    print(f'\n--- Per-task BEST-OF-N L_rollout ---')
    best_L = A('best_rollout_L')
    finite_best = best_L[np.isfinite(best_L)]
    if len(finite_best):
        print(f'  best_L median: {np.median(finite_best):.3f}, mean: {finite_best.mean():.3f}, p90: {np.percentile(finite_best,90):.3f}')
    ratio = A('recover_ratio')
    finite_r = ratio[np.isfinite(ratio)]
    if len(finite_r):
        print(f'  recover_ratio (best_L / max(label_L)) median: {np.median(finite_r):.3f}, mean: {finite_r.mean():.3f}')
        print(f'  >= {args.L_recover_frac} (recovered): {100*(finite_r >= args.L_recover_frac).mean():.1f}%')
        print(f'  >= 0.5: {100*(finite_r >= 0.5).mean():.1f}%')
        print(f'  >= 0.7: {100*(finite_r >= 0.7).mean():.1f}%')

    # Improvement over L_seed
    L_seed_arr = A('L_seed')
    delta = best_L - L_seed_arr
    finite_d = delta[np.isfinite(delta)]
    if len(finite_d):
        print(f'\n--- Improvement over L_seed (best_L − L_seed) ---')
        print(f'  median: {np.median(finite_d):+.3f}, mean: {finite_d.mean():+.3f}, '
              f'p10: {np.percentile(finite_d,10):+.3f}, p90: {np.percentile(finite_d,90):+.3f}')
        print(f'  fraction strictly better than seed: {100*(finite_d > 0).mean():.1f}%')

    # Joint-distance branch coverage on REFINED samples (vs eval_q0 on RAW samples)
    print(f'\n--- Joint-distance branch coverage on REFINED samples ---')
    fc = A('full_cov')
    bc = A('branches_covered'); nl = A('n_labels')
    cov_frac = bc / np.maximum(nl, 1)
    print(f'  full-coverage tasks: {int(fc.sum())}/{N} ({100*fc.mean():.1f}%)')
    print(f'  mean cov frac: {cov_frac.mean():.3f}')

    # By bucket
    print(f'\n--- By L_seed bucket ---')
    bk = A('bucket')
    print(f"{'bucket':<14} {'N':>5} {'ik_ok %':>9} {'best_L med':>11} {'ratio med':>10} "
          f"{'full_cov %':>11} {'recovered %':>13}")
    for b in ['weak', 'medium-weak', 'medium', 'strong']:
        m = (bk == b)
        if m.sum() == 0: continue
        ik_pct = 100 * A('ik_success_count')[m].sum() / (m.sum() * args.n_samples)
        bL_med = np.nanmedian(A('best_rollout_L')[m])
        r_med  = np.nanmedian(A('recover_ratio')[m])
        fc_pct = 100 * A('full_cov')[m].mean()
        recov_pct = 100 * (A('recover_ratio')[m] >= args.L_recover_frac).mean()
        print(f'{b:<14} {int(m.sum()):>5} {ik_pct:>8.1f}% {bL_med:>11.3f} {r_med:>10.3f} '
              f'{fc_pct:>10.1f}% {recov_pct:>12.1f}%')

    # Save NPZ
    out_dir = args.ckpt.parent
    suffix = f'step{step}_{args.which}'
    if args.cfg_w != 1.0:
        suffix = f'{suffix}_cfg{args.cfg_w}'
    if args.drop_pierce:
        suffix = f'{suffix}_drop_pierce'
    np.savez(
        out_dir / f'{args.out_prefix}_{suffix}.npz',
        ik_ok_pct=float(all_ik_ok.mean()),
        rollout_L=all_rollout_L,
        ik_ok=all_ik_ok,
        per_task_idx=A('idx'),
        per_task_n_labels=A('n_labels'),
        per_task_L_seed=A('L_seed'),
        per_task_max_label_L=A('max_label_L'),
        per_task_best_L=A('best_rollout_L'),
        per_task_mean_L=A('mean_rollout_L'),
        per_task_recover_ratio=A('recover_ratio'),
        per_task_ik_ok=A('ik_success_count'),
        per_task_pierce_count=A('pierce_count'),
        per_task_success=A('success_count'),
        per_task_branches_covered=A('branches_covered'),
        per_task_full_cov=A('full_cov'),
        per_task_bucket=A('bucket'),
        per_sample_min_dist_refined=all_min_dists_refined,
        per_sample_pierces=all_pierce,
        match_rad=args.match_rad, L_success=args.L_success,
        L_recover_frac=args.L_recover_frac, cfg_w=args.cfg_w,
        n_samples=args.n_samples,
        drop_pierce=bool(args.drop_pierce),
        pierce_plane_extent_m=float(args.pierce_plane_extent_m),
    )
    print(f'\nSaved: {out_dir / f"{args.out_prefix}_{suffix}.npz"}')


if __name__ == '__main__':
    main()
