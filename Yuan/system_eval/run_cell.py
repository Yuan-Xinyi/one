"""Single-cell runner: produces cell_{A,B,C,D,E}_results.npz.

Each cell loads the same eval set, builds its seed source, runs the
corresponding controller through the shared batched env, and saves per-task
results + best-of-N reduction for diffusion cells.

Output NPZ schema:
    cell                str (e.g. 'D')
    n_tasks             int
    n_samples           int (1 for A/C/E, N for B/D)
    src_idx             (n_tasks,) int64
    bucket              (n_tasks,) <U16
    L_seed              (n_tasks,) f32
    max_label_L         (n_tasks,) f32
    seeds               (n_tasks, n_samples, 7) f32        — actual q0 used
    ik_ok               (n_tasks, n_samples) bool          — True for valid seeds
    L_per_sample        (n_tasks, n_samples) f32           — per-sample rollout L (NaN if IK failed)
    progress_per_sample (n_tasks, n_samples) f32
    term_per_sample     (n_tasks, n_samples) int32
    init_max_qn         (n_tasks, n_samples) f32
    L_best              (n_tasks,) f32                     — best L over samples (NaN/0 for empty)
    best_sample_idx     (n_tasks,) int32                   — argmax over samples (–1 if none valid)
    config_snapshot     str (json)

Usage:
    python -m Yuan.system_eval.run_cell --cell A \
        --eval-set Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz \
        --config Yuan/system_eval/config.yaml
"""
from __future__ import annotations

# Mirror RL_controller convention: ensure conda lib on LD path for CUDA torch.
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
from Yuan.system_eval.seed_sources import build_seeds_for_cell


CELL_CONTROLLER = {
    'A': 'classical',
    'B': 'classical',
    'C': 'hybrid_variantB',
    'D': 'hybrid_variantB',
    'E': 'hybrid_variantB',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', required=True,
                   help='eval_set_*.npz produced by build_eval_set.py')
    p.add_argument('--cell', required=True, choices=list(CELL_CONTROLLER.keys()))
    p.add_argument('--out-dir', default=None,
                   help='override config output.root')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='evaluate only the first N tasks (sanity / debug)')
    p.add_argument('--diffusion-cache', default=None,
                   help='reuse seeds.npz from a prior diffusion cell '
                        '(skips diffusion sampling + IK refine).')
    p.add_argument('--write-diffusion-cache', action='store_true',
                   help='write seeds/ik_ok next to results so the partner cell '
                        '(B<->D) can reuse them via --diffusion-cache.')
    return p.parse_args()


def load_eval_set(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    cell = args.cell.upper()
    controller_name = CELL_CONTROLLER[cell]

    device = torch.device(cfg['runner']['device']
                          if torch.cuda.is_available() else 'cpu')

    eval_set = load_eval_set(Path(args.eval_set))
    n_total = int(eval_set['src_idx'].shape[0])
    if args.max_tasks is not None and args.max_tasks < n_total:
        n_tasks = args.max_tasks
        eval_set = {k: (v[:n_tasks] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == n_total else v)
                    for k, v in eval_set.items()}
        print(f'[run_cell] truncated eval set to first {n_tasks} tasks')
    else:
        n_tasks = n_total

    out_dir = Path(args.out_dir or cfg['output']['root'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg['output']['cell_results_pattern'].format(cell=cell)

    print(f'[run_cell] cell={cell}  controller={controller_name}  '
          f'n_tasks={n_tasks}  device={device}')

    # ---- Build env + controllers ----------------------------------------
    env_yaml = cfg['env']['config_yaml']
    n_envs = int(cfg['env']['n_envs'])
    env = build_env(env_yaml, n_envs=n_envs, device=device)
    classical = ClassicalNullspaceController(env.kin)
    agent = None
    if controller_name == 'hybrid_variantB':
        agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, device)
        print(f'[run_cell] loaded RL agent from {cfg["rl_controller"]["ckpt_dir"]}')

    # ---- Build seeds for the cell --------------------------------------
    t0 = time.time()
    if args.diffusion_cache is not None and cell in ('B', 'D'):
        cache = np.load(args.diffusion_cache, allow_pickle=False)
        seeds = cache['seeds'][:n_tasks].astype(np.float32)
        ik_ok = cache['ik_ok'][:n_tasks].astype(bool)
        print(f'[run_cell] loaded cached diffusion seeds from {args.diffusion_cache} '
              f'(shape {seeds.shape}, IK ok rate {100*ik_ok.mean():.1f}%)')
    else:
        seeds, ik_ok = build_seeds_for_cell(
            cell, eval_set,
            diffusion_cfg=cfg['diffusion'],
            kin=env.kin, device=device,
        )

    if ik_ok is None:
        ik_ok = np.ones(seeds.shape[:2], dtype=bool)

    print(f'[run_cell] seeds ready: shape={seeds.shape} '
          f'IK ok rate={100*ik_ok.mean():.2f}% '
          f'({time.time()-t0:.1f}s)')

    # Optional cache writeout so the partner cell can skip re-sampling.
    if args.write_diffusion_cache and cell in ('B', 'D'):
        cache_path = out_dir / f'diffusion_seeds_{cell}.npz'
        np.savez(cache_path,
                 seeds=seeds, ik_ok=ik_ok,
                 src_idx=eval_set['src_idx'])
        print(f'[run_cell] wrote diffusion cache → {cache_path}')

    # ---- Flatten (n_tasks, n_samples, ·) for batched rollout ----------
    n_samples = seeds.shape[1]
    p0_per = np.broadcast_to(eval_set['cs_p0'][:, None, :],
                             (n_tasks, n_samples, 3)).reshape(-1, 3).copy()
    d_per  = np.broadcast_to(eval_set['cs_line_dir'][:, None, :],
                             (n_tasks, n_samples, 3)).reshape(-1, 3).copy()
    n_per  = np.broadcast_to(eval_set['cs_n_target'][:, None, :],
                             (n_tasks, n_samples, 3)).reshape(-1, 3).copy()
    qs_flat = seeds.reshape(-1, 7)
    ik_flat = ik_ok.reshape(-1)

    # ---- Roll out ------------------------------------------------------
    t0 = time.time()
    tau_e = float(cfg['rl_controller']['tau_enter'])
    tau_x = float(cfg['rl_controller']['tau_exit'])
    res = rollout_seeds_batched(
        qs_flat, p0_per, d_per, n_per,
        env=env, controller=controller_name,
        classical=classical, agent=agent,
        tau_enter=tau_e, tau_exit=tau_x,
        target_distance_m=float(cfg['env']['target_distance_m']),
        progress_every_chunks=max(1, (qs_flat.shape[0] // env.n_envs) // 50),
        progress_prefix=f'[cell {cell}] ',
    )
    print(f'[run_cell] rollouts done: {qs_flat.shape[0]} envs '
          f'({time.time()-t0:.1f}s)')

    # Reshape to (n_tasks, n_samples, ·)
    def _resh(a):
        return a.reshape(n_tasks, n_samples)
    L_per = _resh(res['L']).copy()
    prog_per = _resh(res['episode_progress_m']).copy()
    term_per = _resh(res['term_reason']).astype(np.int32).copy()
    init_qn_per = _resh(res['init_max_qn']).astype(np.float32).copy()
    switch_per  = _resh(res['switch_count']).astype(np.int32).copy()

    # Mask out IK-failed seeds (NaN so they're excluded from finite reductions).
    L_per[~ik_ok] = np.nan
    prog_per[~ik_ok] = np.nan
    term_per[~ik_ok] = -1

    # Best-of-N reduction: max over samples, with IK fails treated as L=0
    # (per the user's spec). NaN → 0 for the purpose of argmax/max.
    L_for_best = np.where(np.isfinite(L_per), L_per, 0.0)
    best_idx = np.argmax(L_for_best, axis=1).astype(np.int32)
    L_best = L_for_best.max(axis=1).astype(np.float32)
    # If every sample for a task is invalid, mark best_idx = -1
    any_valid = ik_ok.any(axis=1)
    best_idx[~any_valid] = -1
    L_best[~any_valid] = np.nan

    # ---- Save ----------------------------------------------------------
    config_snapshot = json.dumps({
        'cell': cell,
        'controller': controller_name,
        'tau_enter': tau_e, 'tau_exit': tau_x,
        'n_samples': int(n_samples),
        'diffusion': cfg['diffusion'] if cell in ('B', 'D') else None,
        'rl_ckpt_dir': cfg['rl_controller']['ckpt_dir'] if controller_name == 'hybrid_variantB' else None,
        'env_config_yaml': str(env_yaml),
        'target_distance_m': float(cfg['env']['target_distance_m']),
        'eval_set': str(args.eval_set),
        'n_tasks': int(n_tasks),
    })

    np.savez_compressed(
        out_path,
        cell=np.array(cell),
        n_tasks=np.int64(n_tasks),
        n_samples=np.int64(n_samples),
        src_idx=eval_set['src_idx'].astype(np.int64),
        bucket=eval_set['bucket'],
        L_seed=eval_set['L_seed'].astype(np.float32),
        max_label_L=eval_set['max_label_L'].astype(np.float32),
        seeds=seeds.astype(np.float32),
        ik_ok=ik_ok,
        L_per_sample=L_per.astype(np.float32),
        progress_per_sample=prog_per.astype(np.float32),
        term_per_sample=term_per,
        init_max_qn_per_sample=init_qn_per,
        switch_per_sample=switch_per,
        L_best=L_best,
        best_sample_idx=best_idx,
        config_snapshot=np.array(config_snapshot),
    )
    print(f'[run_cell] wrote {out_path} '
          f'({out_path.stat().st_size/1e6:.1f} MB)')

    # ---- Tiny on-the-fly summary --------------------------------------
    fin = np.isfinite(L_best)
    print(f'[run_cell] L_best: n_valid={fin.sum()}/{len(L_best)}  '
          f'median={np.nanmedian(L_best):.3f}  mean={np.nanmean(L_best):.3f}  '
          f'p25={np.nanpercentile(L_best,25):.3f}  '
          f'p75={np.nanpercentile(L_best,75):.3f}')


if __name__ == '__main__':
    main()
