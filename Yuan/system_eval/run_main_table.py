"""System Integration table (tab:lnetresult).

Rolls the same DP seed (cached at sweeps/cfg_only_w1.5.npz, generated under
the deployment policy of Section sec:dp_inference) through THREE controllers:
  - classical
  - RL (= hybrid with tau=1.0/1.0; the policy never enters the boundary shell)
  - hybrid (adopted tau, read from config.yaml -> 0.98/0.94)

Difficulty buckets are by oracle_hyb (the controller-aware upper bound):
  Easy:      oracle_hyb >= 0.80 m
  Medium:    0.45 <= oracle_hyb < 0.80 m
  Difficult: oracle_hyb < 0.45 m
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched, load_rl_agent
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController


SEED_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'
ORACLE_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/cell_oracle_hyb_results.npz'
EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
OUT_JSON = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/main_table_results.json'

EASY_THR = 0.80
DIFF_THR = 0.45


def stats(a):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def fmtl(t): return f'{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f} / {t[3]:.3f}'
def fmtp(t): return f'{t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} / {t[3]:.2f}'


def main():
    cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tdm = float(cfg['env']['target_distance_m'])

    # Cached DP seeds (w=1.5, deployment policy)
    seeds = np.load(SEED_NPZ)['seeds'].astype(np.float32)   # (T, 7)
    T = seeds.shape[0]
    z = np.load(EVAL_NPZ)
    p0 = z['cs_p0'][:T].astype(np.float32)
    d  = z['cs_line_dir'][:T].astype(np.float32)
    n  = z['cs_n_target'][:T].astype(np.float32)

    # Oracle (controller-aware) -- normaliser
    oh = np.load(ORACLE_NPZ)['L_best'][:T].astype(np.float32) * tdm

    # Difficulty buckets
    bucket = np.where(oh >= EASY_THR, 'Easy',
              np.where(oh >= DIFF_THR, 'Medium', 'Difficult'))
    counts = {b: int((bucket == b).sum()) for b in ('Easy', 'Medium', 'Difficult')}
    print(f'[main_table] bucket counts: All={T} Easy={counts["Easy"]} '
          f'Medium={counts["Medium"]} Difficult={counts["Difficult"]}', flush=True)

    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']),
                    device=dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    tau_adopted_e = float(cfg['rl_controller']['tau_enter'])
    tau_adopted_x = float(cfg['rl_controller']['tau_exit'])
    print(f'[main_table] gains mu={classical.manip_gain} jl={classical.jl_gain} '
          f'theta={classical.angle_boundary_gain}; '
          f'tau_adopted=({tau_adopted_e},{tau_adopted_x})', flush=True)

    configs = [
        ('Classical', 'classical', None, None),
        ('RL',        'hybrid_variantB', 1.0, 1.0),    # tau=1.0 -> pure RL
        ('Hybrid',    'hybrid_variantB', tau_adopted_e, tau_adopted_x),
    ]
    results = {'meta': {'T': T, 'tdm': tdm, 'easy_thr': EASY_THR,
                        'diff_thr': DIFF_THR, 'bucket_counts': counts,
                        'tau_adopted': [tau_adopted_e, tau_adopted_x],
                        'classical_gains': [classical.manip_gain,
                                            classical.jl_gain,
                                            classical.angle_boundary_gain]},
               'rows': {}}

    for name, ctrl, te, tx in configs:
        t0 = time.time()
        kwargs = {'env': env, 'controller': ctrl, 'classical': classical,
                  'agent': agent, 'target_distance_m': tdm,
                  'progress_every_chunks': max(1, (T // env.n_envs) // 8),
                  'progress_prefix': f'[{name}] '}
        if ctrl == 'hybrid_variantB':
            kwargs['tau_enter'] = te
            kwargs['tau_exit'] = tx
        res = rollout_seeds_batched(seeds, p0, d, n, **kwargs)
        L = res['L'].astype(np.float32)
        l_m = L * tdm
        with np.errstate(invalid='ignore', divide='ignore'):
            pct = 100.0 * l_m / np.maximum(oh, 1e-9)
        dur = time.time() - t0
        # Save per-task L for later use
        np.savez_compressed(
            f'Yuan/system_eval/runs/eval_10k_systematic/sweeps/main_{name}.npz',
            L=L, pct=pct, bucket=bucket)
        rows = {}
        for b in ('All', 'Easy', 'Medium', 'Difficult'):
            m = np.ones(T, dtype=bool) if b == 'All' else (bucket == b)
            rows[b] = {'l': stats(l_m[m]), 'pct': stats(pct[m]),
                       'n': int(m.sum())}
            print(f'[{name}] {b:9s} (n={rows[b]["n"]}):  '
                  f'l(m)={fmtl(rows[b]["l"])} | %={fmtp(rows[b]["pct"])}',
                  flush=True)
        results['rows'][name] = rows
        print(f'[{name}] done in {dur:.0f}s', flush=True)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f'\n[main_table] wrote {OUT_JSON}', flush=True)

    print('\n==== MAIN SYSTEM-INTEGRATION TABLE ====')
    for b in ('All', 'Easy', 'Medium', 'Difficult'):
        print(f'  --- {b} ---')
        for name in ('Classical', 'RL', 'Hybrid'):
            r = results['rows'][name][b]
            print(f'    DP × {name:9s}:  l(m)={fmtl(r["l"])} | %={fmtp(r["pct"])}')


if __name__ == '__main__':
    main()
