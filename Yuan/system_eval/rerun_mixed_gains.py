"""Re-run the 4 mixed-gain rows of the classical ablation table, saving
per-task L arrays so the % column can later be recomputed against any oracle.
"""
from __future__ import annotations
import numpy as np
import torch
import yaml
from pathlib import Path

from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController


COMBOS = [(0.2, 0.1, 0.4), (0.4, 0.2, 0.4), (0.6, 0.2, 0.4), (0.8, 0.2, 0.4)]


def main():
    cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    p0 = z['cs_p0'].astype(np.float32)
    d = z['cs_line_dir'].astype(np.float32)
    n = z['cs_n_target'].astype(np.float32)
    q0 = z['q0_seed'].astype(np.float32)
    tdm = float(cfg['env']['target_distance_m'])
    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']),
                    device=dev)
    out = Path('Yuan/system_eval/runs/eval_10k_systematic/sweeps')

    for km, kj, kt in COMBOS:
        ctrl = ClassicalNullspaceController(env.kin, manip_gain=km, jl_gain=kj,
                                            angle_boundary_gain=kt)
        res = rollout_seeds_batched(q0, p0, d, n, env=env, controller='classical',
                                    classical=ctrl, target_distance_m=tdm,
                                    progress_every_chunks=99999)
        L = res['L'].astype(np.float32)
        l = L * tdm
        tag = f'mixed_{km}_{kj}_{kt}'
        np.savez_compressed(out / f'gain_{tag}.npz', L=L)
        print(f'[gain mixed] mu={km} jl={kj} th={kt}: '
              f'l(m) mean/std/min/max = '
              f'{l.mean():.3f} / {l.std():.3f} / {l.min():.3f} / {l.max():.3f}  '
              f'-> saved gain_{tag}.npz', flush=True)


if __name__ == '__main__':
    main()
