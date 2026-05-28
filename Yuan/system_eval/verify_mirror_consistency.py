"""Mirror-augmentation physics consistency test.

Mirror aug (used in training the cfg_mirror diffusion ckpt) reflects each
sample across the FR3's xz-plane:
    c:  (p0.y, line_dir.y, n_target.y) -> negated
    q:  q * FLIP_MULT_Q  with  FLIP_MULT_Q = [-1, 1, -1, 1, -1, 1, -1]

For the augmentation to be physically valid, the rolled-out lifetime
metric L must be invariant under this reflection. If it's not, the model
was trained on bogus correspondences and the +mirror_aug numbers are
inflated artifacts.

Test plan (per the user's spec):
    pick 10 random tasks
    for each:
        roll out (q, c)          -> L_orig
        roll out (q_m, c_m)      -> L_mirror
        assert |L_orig - L_mirror| < tol

The rigorous test uses the CLASSICAL controller (this is the same
controller used during SMM data generation, so it's the source of truth
for the labels' L_clean). We also include the HYBRID controller for
completeness -- if the RL portion isn't strictly y-symmetric (NNs aren't
by construction), small deviations are expected there.

Usage:
    python -m Yuan.system_eval.verify_mirror_consistency \\
        --eval-set Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz \\
        --n 10 --seed 42
"""
from __future__ import annotations

# Inherit LD_LIBRARY_PATH workaround used by run_cell.
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
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import (
    build_env, load_rl_agent, rollout_seeds_batched,
)


# Same convention as Yuan/seed_selection/dataset.py and fr3_dit's StartQDataset.
FLIP_MULT_Q = np.array([-1, 1, -1, 1, -1, 1, -1], dtype=np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', required=True)
    p.add_argument('--n', type=int, default=10, help='number of tasks to test')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for task pick')
    p.add_argument('--tol-m', type=float, default=0.02,
                   help='per-task tolerance (meters of EE progress)')
    p.add_argument('--out', default=None,
                   help='write JSON report (default: <eval-set-dir>/mirror_consistency.json)')
    return p.parse_args()


def mirror(c_p0, c_d, c_n, q):
    c_p0 = c_p0.copy(); c_d = c_d.copy(); c_n = c_n.copy()
    c_p0[..., 1] *= -1
    c_d[..., 1]  *= -1
    c_n[..., 1]  *= -1
    return c_p0, c_d, c_n, (q.copy() * FLIP_MULT_Q)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(cfg['runner']['device'] if torch.cuda.is_available() else 'cpu')

    eval_set = np.load(Path(args.eval_set), allow_pickle=False)
    n_total = int(eval_set['src_idx'].shape[0])
    rng = np.random.default_rng(int(args.seed))
    idx = rng.choice(n_total, size=int(args.n), replace=False)
    idx = np.sort(idx)
    print(f'[mirror-test] picked tasks (eval-set rows): {idx.tolist()}')
    print(f'[mirror-test] tolerance: |Δprogress| < {args.tol_m*1000:.0f} mm')

    # Pull originals
    q_o   = eval_set['q0_seed'][idx].astype(np.float32)            # (n, 7)
    p0_o  = eval_set['cs_p0'][idx].astype(np.float32)
    d_o   = eval_set['cs_line_dir'][idx].astype(np.float32)
    n_o   = eval_set['cs_n_target'][idx].astype(np.float32)
    # Mirror counterparts
    p0_m, d_m, n_m, q_m = mirror(p0_o, d_o, n_o, q_o)

    # Pack into one batch: [orig | mirrored]
    qs   = np.concatenate([q_o,  q_m],  axis=0)
    p0s  = np.concatenate([p0_o, p0_m], axis=0)
    ds   = np.concatenate([d_o,  d_m],  axis=0)
    ns   = np.concatenate([n_o,  n_m],  axis=0)
    n2 = qs.shape[0]
    target_distance_m = float(cfg['env']['target_distance_m'])

    # ---- Build env + controllers ------------------------------------
    env = build_env(cfg['env']['config_yaml'], n_envs=max(64, n2),
                    device=device)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, device)
    tau_e = float(cfg['rl_controller']['tau_enter'])
    tau_x = float(cfg['rl_controller']['tau_exit'])

    def _run(controller):
        res = rollout_seeds_batched(
            qs, p0s, ds, ns,
            env=env, controller=controller,
            classical=classical, agent=agent,
            tau_enter=tau_e, tau_exit=tau_x,
            target_distance_m=target_distance_m,
            progress_every_chunks=0,
        )
        return res['L'].astype(np.float64), res['episode_progress_m'].astype(np.float64)

    print('\n[mirror-test] running CLASSICAL controller …')
    L_cls, prog_cls = _run('classical')
    print('[mirror-test] running HYBRID variantB controller …')
    L_hyb, prog_hyb = _run('hybrid_variantB')

    # Split orig vs mirrored
    n = int(args.n)
    def _split(arr):
        return arr[:n], arr[n:]
    L_cls_o, L_cls_m = _split(L_cls)
    L_hyb_o, L_hyb_m = _split(L_hyb)
    prog_cls_o, prog_cls_m = _split(prog_cls)
    prog_hyb_o, prog_hyb_m = _split(prog_hyb)

    # ---- Report ------------------------------------------------------
    def _report(name, orig, mir, prog_orig, prog_mir):
        diff = mir - orig
        max_abs = float(np.max(np.abs(diff)))
        med_abs = float(np.median(np.abs(diff)))
        all_ok = bool((np.abs(prog_mir - prog_orig) < args.tol_m).all())
        print(f'\n[mirror-test] === {name} ===')
        print(f'  per-task progress (m):')
        print(f'   {"row":>3} {"src_idx":>8} {"prog_orig":>10} {"prog_mirror":>11} {"|Δm|":>8} {"|ΔL|":>8}')
        for i, ti in enumerate(idx):
            print(f'   {i:>3} {int(ti):>8} {prog_orig[i]:>10.4f} {prog_mir[i]:>11.4f} '
                  f'{abs(prog_mir[i]-prog_orig[i]):>8.4f} {abs(diff[i]):>8.4f}')
        ok_count = int((np.abs(prog_mir - prog_orig) < args.tol_m).sum())
        print(f'  → {ok_count}/{n} tasks within {args.tol_m*1000:.0f} mm   '
              f'(max |Δprog| = {float(np.max(np.abs(prog_mir-prog_orig)))*1000:.2f} mm, '
              f'median {float(np.median(np.abs(prog_mir-prog_orig)))*1000:.2f} mm)')
        return {
            'controller': name,
            'n': n, 'tol_m': args.tol_m,
            'pass': all_ok,
            'max_abs_progress_diff_m': float(np.max(np.abs(prog_mir - prog_orig))),
            'median_abs_progress_diff_m': float(np.median(np.abs(prog_mir - prog_orig))),
            'max_abs_L_diff': max_abs,
            'median_abs_L_diff': med_abs,
            'per_task': [{
                'eval_row': int(idx[i]),
                'src_idx': int(eval_set['src_idx'][int(idx[i])]),
                'progress_orig_m': float(prog_orig[i]),
                'progress_mirror_m': float(prog_mir[i]),
                'abs_diff_m': float(abs(prog_mir[i] - prog_orig[i])),
            } for i in range(n)],
        }

    rep_cls = _report('CLASSICAL (rigorous: same controller as data gen)',
                       L_cls_o, L_cls_m, prog_cls_o, prog_cls_m)
    rep_hyb = _report('HYBRID variantB (deployment controller)',
                       L_hyb_o, L_hyb_m, prog_hyb_o, prog_hyb_m)

    # Headline
    print('\n' + '=' * 72)
    print('VERDICT')
    print('=' * 72)
    if rep_cls['pass']:
        print('[CLASSICAL]: PASS — mirror aug is physically consistent under the '
              f'controller that generated SMM labels. Max diff {rep_cls["max_abs_progress_diff_m"]*1000:.2f} mm.')
    else:
        print('[CLASSICAL]: FAIL — mirror aug does NOT preserve L under classical. '
              f'Max diff {rep_cls["max_abs_progress_diff_m"]*1000:.2f} mm > '
              f'tol {args.tol_m*1000:.0f} mm. **The +mirror_aug uplift is suspect.**')
    if rep_hyb['pass']:
        print('[HYBRID]:    PASS — hybrid controller is also y-symmetric in practice.')
    else:
        print('[HYBRID]:    SOFT-FAIL — hybrid (with stochastic-NN RL portion) drifts '
              f'by up to {rep_hyb["max_abs_progress_diff_m"]*1000:.2f} mm. '
              'This is expected if RL is non-strictly-symmetric and is not by itself '
              'a problem for mirror aug, but classical PASS is the key sign.')

    out = Path(args.out or (Path(args.eval_set).parent / 'mirror_consistency.json'))
    out.write_text(json.dumps({'classical': rep_cls, 'hybrid': rep_hyb}, indent=2))
    print(f'\n[mirror-test] full report → {out}')


if __name__ == '__main__':
    main()
