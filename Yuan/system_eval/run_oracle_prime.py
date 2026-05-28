"""Controller-aware oracle (cell `oracle_hyb`): roll every SMM candidate
through the HYBRID controller, take per-task max.

The classical-label oracle (cell `oracle_cls`) uses
labels_q0[argmax(labels_L_clean)], but labels_L_clean was measured under
CLASSICAL during SMM data generation. Under the HYBRID deployment
controller, that seed is no longer guaranteed to be optimal — and indeed
the full method (`diff_hyb`) beats `oracle_cls` on ~84% of tasks.

`oracle_hyb` is the *controller-aware* oracle: for each task we look at
all valid candidates in `top_Kprime_q[t]` (the SMM top-K' pool stored in
pilot_20k), roll each one through the hybrid controller, and take the
max L. This gives the true upper bound for any q0 in the SMM pool under
the deployment controller.

Output schema matches cell_<name>_results.npz so aggregate.py picks it up
as cell 'oracle_hyb'.

Usage:
    python -m Yuan.system_eval.run_oracle_prime \\
        --eval-set Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz \\
        --pilot-npz Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz \\
        --out-dir  Yuan/system_eval/runs/eval_10k_systematic
"""
from __future__ import annotations

# Conda lib bootstrap (same as run_cell).
import os, sys
_conda_lib = os.path.join(sys.prefix, 'lib')
if _conda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    new_env = dict(os.environ)
    new_env['LD_LIBRARY_PATH'] = _conda_lib + ':' + new_env.get('LD_LIBRARY_PATH', '')
    if __spec__ is not None and __spec__.name != '__main__':
        argv = [sys.executable, '-m', __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import (
    build_env, load_rl_agent, rollout_seeds_batched,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', required=True)
    p.add_argument('--pilot-npz', default='Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--max-tasks', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg['runner']['device'] if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.out_dir or cfg['output']['root'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg['output']['cell_results_pattern'].format(cell='oracle_hyb')

    # ---- Load eval set + pilot ---------------------------------------
    es = np.load(Path(args.eval_set), allow_pickle=False)
    n_tasks_total = int(es['src_idx'].shape[0])
    if args.max_tasks is not None:
        n_tasks = min(args.max_tasks, n_tasks_total)
    else:
        n_tasks = n_tasks_total

    src_idx = es['src_idx'][:n_tasks].astype(np.int64)
    pilot = np.load(Path(args.pilot_npz), allow_pickle=False)
    top_q     = pilot['top_Kprime_q'][src_idx].astype(np.float32)            # (n, K', 7)
    top_valid = pilot['top_Kprime_valid_mask'][src_idx].astype(bool)         # (n, K')
    top_Lclean= pilot['top_Kprime_L_clean'][src_idx].astype(np.float32)      # (n, K')
    K_prime = int(top_q.shape[1])
    print(f'[oracle_hyb] n_tasks={n_tasks}  K_prime={K_prime}  '
          f'avg valid candidates/task = {top_valid.sum(axis=1).mean():.2f}  '
          f'(min={top_valid.sum(axis=1).min()}, max={top_valid.sum(axis=1).max()})')

    # ---- Flatten valid (task, candidate) pairs -----------------------
    p0_all = es['cs_p0'][:n_tasks].astype(np.float32)
    d_all  = es['cs_line_dir'][:n_tasks].astype(np.float32)
    n_all  = es['cs_n_target'][:n_tasks].astype(np.float32)

    # Per-task seeds with NaN for invalid slots (we won't roll the NaN ones).
    qs_per_task = np.full((n_tasks, K_prime, 7), np.nan, dtype=np.float32)
    qs_per_task[top_valid] = top_q[top_valid]

    # Build a flat list of only the valid ones for the rollout.
    flat_task_idx, flat_sample_idx = np.where(top_valid)
    flat_n = int(flat_task_idx.size)
    qs_flat = top_q[flat_task_idx, flat_sample_idx]              # (flat_n, 7)
    p0_flat = p0_all[flat_task_idx]                              # (flat_n, 3)
    d_flat  = d_all[flat_task_idx]                               # (flat_n, 3)
    n_flat  = n_all[flat_task_idx]                               # (flat_n, 3)
    print(f'[oracle_hyb] flat valid candidates: {flat_n}')

    # ---- Build env + controllers --------------------------------------
    env = build_env(cfg['env']['config_yaml'],
                    n_envs=int(cfg['env']['n_envs']),
                    device=device)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, device)
    tau_e = float(cfg['rl_controller']['tau_enter'])
    tau_x = float(cfg['rl_controller']['tau_exit'])
    target_distance_m = float(cfg['env']['target_distance_m'])

    # ---- Rollout ------------------------------------------------------
    t0 = time.time()
    res = rollout_seeds_batched(
        qs_flat, p0_flat, d_flat, n_flat,
        env=env, controller='hybrid_variantB',
        classical=classical, agent=agent,
        tau_enter=tau_e, tau_exit=tau_x,
        target_distance_m=target_distance_m,
        progress_every_chunks=max(1, (flat_n // env.n_envs) // 20),
        progress_prefix='[oracle_hyb] ',
    )
    print(f'[oracle_hyb] rollout done: {flat_n} envs in {time.time()-t0:.1f}s')

    # ---- Regroup ------------------------------------------------------
    L_per_sample  = np.full((n_tasks, K_prime), np.nan, dtype=np.float32)
    prog_per_sample = np.full((n_tasks, K_prime), np.nan, dtype=np.float32)
    term_per_sample = np.full((n_tasks, K_prime), -1, dtype=np.int32)
    init_qn_per   = np.full((n_tasks, K_prime), np.nan, dtype=np.float32)
    switch_per    = np.full((n_tasks, K_prime), 0, dtype=np.int32)
    L_per_sample[flat_task_idx, flat_sample_idx]    = res['L'].astype(np.float32)
    prog_per_sample[flat_task_idx, flat_sample_idx] = res['episode_progress_m'].astype(np.float32)
    term_per_sample[flat_task_idx, flat_sample_idx] = res['term_reason'].astype(np.int32)
    init_qn_per[flat_task_idx, flat_sample_idx]     = res['init_max_qn'].astype(np.float32)
    switch_per[flat_task_idx, flat_sample_idx]      = res['switch_count'].astype(np.int32)

    # Best-of-K' (max over samples; invalid slots are NaN so not chosen)
    L_for_best = np.where(np.isfinite(L_per_sample), L_per_sample, -np.inf)
    best_idx = np.argmax(L_for_best, axis=1).astype(np.int32)
    L_best = L_for_best.max(axis=1)
    any_valid = top_valid.any(axis=1)
    L_best[~any_valid] = np.nan
    best_idx[~any_valid] = -1
    L_best = L_best.astype(np.float32)

    # The "seed actually used" by E' = top_q[t, best_idx[t]]
    seeds_used = np.zeros((n_tasks, 1, 7), dtype=np.float32)
    for t in range(n_tasks):
        bi = int(best_idx[t])
        seeds_used[t, 0] = top_q[t, bi] if bi >= 0 else np.nan

    # ---- Save ---------------------------------------------------------
    snapshot = json.dumps({
        'cell': 'oracle_hyb',
        'controller': 'hybrid_variantB',
        'tau_enter': tau_e, 'tau_exit': tau_x,
        'K_prime': K_prime,
        'avg_valid_candidates': float(top_valid.sum(axis=1).mean()),
        'flat_rollouts': int(flat_n),
        'rl_ckpt_dir': cfg['rl_controller']['ckpt_dir'],
        'env_config_yaml': str(cfg['env']['config_yaml']),
        'target_distance_m': target_distance_m,
        'eval_set': str(args.eval_set),
        'n_tasks': int(n_tasks),
    })

    # Shape conventions: store as (n_tasks, K_prime, 7) since K_prime > 1.
    np.savez_compressed(
        out_path,
        cell=np.array('oracle_hyb'),
        n_tasks=np.int64(n_tasks),
        n_samples=np.int64(K_prime),
        src_idx=src_idx,
        bucket=es['bucket'][:n_tasks],
        L_seed=es['L_seed'][:n_tasks].astype(np.float32),
        max_label_L=es['max_label_L'][:n_tasks].astype(np.float32),
        seeds=qs_per_task,
        ik_ok=top_valid,                  # treat valid_mask as the "ik_ok" surrogate
        L_per_sample=L_per_sample,
        progress_per_sample=prog_per_sample,
        term_per_sample=term_per_sample,
        init_max_qn_per_sample=init_qn_per,
        switch_per_sample=switch_per,
        L_best=L_best,
        best_sample_idx=best_idx,
        # Extras specific to E':
        top_Kprime_L_clean=top_Lclean,    # the classical L of each candidate
        seeds_used=seeds_used,             # which top_K candidate won
        config_snapshot=np.array(snapshot),
    )
    print(f'[oracle\'] wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)')

    # ---- Quick comparison vs oracle_cls and diff_hyb ----------------
    fin = np.isfinite(L_best)
    print(f'\n[oracle_hyb] L_best: n_valid={fin.sum()}/{n_tasks}  '
          f'median={np.nanmedian(L_best):.3f}  mean={np.nanmean(L_best):.3f}  '
          f'p25={np.nanpercentile(L_best,25):.3f}  '
          f'p75={np.nanpercentile(L_best,75):.3f}')
    ocls_path = out_dir / cfg['output']['cell_results_pattern'].format(cell='oracle_cls')
    if ocls_path.exists():
        ocls = np.load(ocls_path, allow_pickle=False)
        delta = L_best - ocls['L_best']
        f = np.isfinite(delta)
        print(f'[oracle_hyb] vs oracle_cls: '
              f'median(oracle_hyb - oracle_cls) = {np.median(delta[f]):+.3f}  '
              f'oracle_hyb > oracle_cls on {100*(delta[f] > 0).mean():.1f}% of tasks')
        full_path = out_dir / cfg['output']['cell_results_pattern'].format(cell='diff_hyb')
        if full_path.exists():
            full = np.load(full_path, allow_pickle=False)
            delta_full = full['L_best'] - L_best
            f = np.isfinite(delta_full)
            print(f'[oracle_hyb] diff_hyb vs oracle_hyb: '
                  f'median(diff_hyb - oracle_hyb) = {np.median(delta_full[f]):+.3f}  '
                  f'diff_hyb > oracle_hyb on {100*(delta_full[f] > 0).mean():.1f}% of tasks')


if __name__ == '__main__':
    main()
