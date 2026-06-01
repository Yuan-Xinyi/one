"""Two sequential sweeps on the 10k eval set.

Phase 1 -- Classical gain sweep (seed = q0_seed, controller = classical):
    q0_seed is the task's generating config, always IK-valid, so no seed
    selection is needed. Sweep the three secondary-objective gains
    (k_mu, k_jl, k_theta) over a grid, rank by mean-% on the 10k set, and
    report the 3 single-objective rows + the winning combo (single value each).

Phase 2 -- CFG sweep (seed = DP best-of-1 with projection retry, controller =
    hybrid using Phase-1's best gains):
    For each guidance scale w we draw ONE diffusion candidate per task and
    Newton-project it; if projection fails we resample (N_valid = 1). This only
    guarantees a *legal* seed, not a high-quality one -- no best-of-N rollout
    selection. Report one value per w.

Metrics: l (m) = L * target_distance_m, and % = 100 * l / l_oracle with
l_oracle = max_label_L * target_distance_m (per-task SMM classical oracle;
hybrid rows may exceed 100%).

Usage:
    python -m Yuan.system_eval.sweep_gain_cfg            # full 10k
    python -m Yuan.system_eval.sweep_gain_cfg --smoke    # 256 tasks, quick
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched, load_rl_agent
from Yuan.system_eval.seed_sources import diffusion_seeds
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController


GAIN_GRID = {'k_mu': [0.2, 0.4, 0.6, 0.8], 'k_jl': [0.1, 0.2, 0.4], 'k_theta': [0.2, 0.4, 0.6]}
W_LIST = [0.0, 1.0, 1.5, 2.0, 3.0]
DDIM_STEPS = 50
SAMPLE_SEED = 42
TAU = 0.98
RETRY_BATCH = 32         # candidates drawn per round; ~1 round suffices for typical IK ok rate
RETRY_MAX_ROUNDS = 64    # safety cap; loop is "retry until valid"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set',
                   default='Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    p.add_argument('--out-dir',
                   default='Yuan/system_eval/runs/eval_10k_systematic/sweeps')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def subset(es, idx, T_all):
    return {k: (v[idx] if hasattr(v, 'shape') and v.shape and v.shape[0] == T_all else v)
            for k, v in es.items()}


def roll(seeds_T7, p0, d, n, *, env, controller, classical, agent, tdm,
         tau_enter=None, tau_exit=None, prefix=''):
    """Roll one seed per task (T,7); return L (T,).
    tau_enter/tau_exit default to the single-threshold TAU constant when omitted,
    so callers that target the *adopted* hybrid setting must pass (0.98, 0.94)
    explicitly."""
    T = seeds_T7.shape[0]
    te = TAU if tau_enter is None else float(tau_enter)
    tx = TAU if tau_exit  is None else float(tau_exit)
    res = rollout_seeds_batched(
        seeds_T7.astype(np.float32), p0, d, n, env=env, controller=controller,
        classical=classical, agent=agent, tau_enter=te, tau_exit=tx,
        target_distance_m=tdm,
        progress_every_chunks=max(1, (T // env.n_envs) // 8), progress_prefix=prefix)
    return res['L'].astype(np.float32)


def stats(l_arr, l_oracle):
    m = np.isfinite(l_arr)
    l = l_arr[m]
    pct = np.where(l_oracle > 1e-9, 100.0 * l_arr / np.maximum(l_oracle, 1e-9), np.nan)
    p = pct[np.isfinite(pct)]
    f = lambda a: (float(a.mean()), float(a.std()), float(a.min()), float(a.max())) if a.size else (0, 0, 0, 0)
    return {'l': f(l), 'pct': f(p)}


def fmt(t):
    return f'{t[0]:.3f} / {t[1]:.3f} / {t[2]:.3f} / {t[3]:.3f}'


def dp_best_of_1(es_T, w, *, ckpt, use_ema, kin, device, q0_fallback):
    """One IK-valid DP seed per task via projection retry until valid.

    Loops until every task has an IK-valid sample (the round count is uncapped
    in spirit; ``RETRY_MAX_ROUNDS`` is a runaway safety net). Returns the
    seeds (T,7), the number of rounds used, and the number of unresolved
    tasks (should be 0 unless the safety cap fires).
    """
    T = es_T['cs_p0'].shape[0]
    seeds = np.zeros((T, 7), dtype=np.float32)
    have = np.zeros(T, dtype=bool)
    rounds_used = 0
    for r in range(RETRY_MAX_ROUNDS):
        todo = np.where(~have)[0]
        if todo.size == 0:
            break
        rounds_used = r + 1
        cand, ok = diffusion_seeds(
            subset(es_T, todo, T), ckpt, n_samples=RETRY_BATCH, ddim_steps=DDIM_STEPS,
            cfg_w=w, sample_seed=SAMPLE_SEED + r, kin=kin, device=device,
            use_ema=use_ema, verbose=False)
        first = np.argmax(ok, axis=1)
        any_ok = ok.any(axis=1)
        for j, ti in enumerate(todo):
            if any_ok[j]:
                seeds[ti] = cand[j, first[j]]
                have[ti] = True
        if r >= 2 and (~have).sum() <= 20:
            print(f'  [dp_best_of_1 w={w}] round {r+1}: still {(~have).sum()} tasks needing IK', flush=True)
    n_unresolved = int((~have).sum())
    if n_unresolved:
        print(f'  [dp_best_of_1 w={w}] WARN safety cap hit after {RETRY_MAX_ROUNDS} rounds; '
              f'{n_unresolved} tasks fall back to q0_seed', flush=True)
        seeds[~have] = q0_fallback[~have]
    return seeds, rounds_used, n_unresolved


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device(cfg['runner']['device'] if torch.cuda.is_available() else 'cpu')

    grid = {k: list(v) for k, v in GAIN_GRID.items()}
    w_list = list(W_LIST)
    if args.smoke:
        grid = {'k_mu': [0.4, 0.6], 'k_jl': [0.2], 'k_theta': [0.4]}
        w_list = [1.5]

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(Path(args.eval_set), allow_pickle=False)
    es = {k: z[k] for k in z.files}
    T_all = es['cs_p0'].shape[0]
    T = 256 if args.smoke else T_all
    es = subset(es, np.arange(T), T_all)
    p0, d, n = es['cs_p0'].astype(np.float32), es['cs_line_dir'].astype(np.float32), es['cs_n_target'].astype(np.float32)
    q0 = es['q0_seed'].astype(np.float32)
    tdm = float(cfg['env']['target_distance_m'])
    l_oracle = es['max_label_L'].astype(np.float32) * tdm

    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']), device=dev)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    ckpt = cfg['diffusion']['ckpt']
    use_ema = bool(cfg['diffusion'].get('use_ema', True))

    def cls(km, kj, kt):
        return ClassicalNullspaceController(env.kin, manip_gain=km, jl_gain=kj, angle_boundary_gain=kt)

    results = {'meta': {'T': T, 'w_list': w_list, 'grid': grid, 'tdm': tdm,
                        'seed_policy': 'gain=q0_seed; cfg=DP best-of-1 retry'},
               'gain': {}, 'cfg': {}}

    # ===== Phase 1: gain sweep (q0_seed, classical) =====
    print(f'[sweep] Phase 1 gain sweep (q0): {len(grid["k_mu"])*len(grid["k_jl"])*len(grid["k_theta"])} combos, T={T}', flush=True)
    grid_rows = []
    for km in grid['k_mu']:
        for kj in grid['k_jl']:
            for kt in grid['k_theta']:
                L = roll(q0, p0, d, n, env=env, controller='classical', classical=cls(km, kj, kt),
                         agent=None, tdm=tdm)
                s = stats(L * tdm, l_oracle)
                grid_rows.append({'gains': [km, kj, kt], 'pct_mean': s['pct'][0], 'l_mean': s['l'][0]})
                print(f'[gain] mu={km} jl={kj} th={kt}: l%={s["pct"][0]:.2f} l={s["l"][0]:.3f}', flush=True)
    grid_rows.sort(key=lambda r: -r['pct_mean'])
    best = grid_rows[0]['gains']
    results['gain']['grid'] = grid_rows
    results['gain']['best_combo'] = best
    print(f'[gain] BEST {best} l%={grid_rows[0]["pct_mean"]:.2f}', flush=True)

    for name, g in {'mu_only': [1.0, 0., 0.], 'jl_only': [0., 1.0, 0.],
                    'theta_only': [0., 0., 1.0], 'best_combo': best}.items():
        L = roll(q0, p0, d, n, env=env, controller='classical', classical=cls(*g), agent=None, tdm=tdm)
        results['gain'][name] = {'gains': g, **stats(L * tdm, l_oracle)}
        np.savez_compressed(out_dir / f'gain_{name}.npz', L=L)
        r = results['gain'][name]
        print(f'[gain] {name} {g}: l(m)={fmt(r["l"])} %={fmt(r["pct"])}', flush=True)

    # ===== Phase 2: CFG sweep (DP best-of-1 retry, hybrid w/ best gains) =====
    classical_best = cls(*best)
    print(f'[sweep] Phase 2 CFG sweep (DP best-of-1), hybrid w/ gains={best}', flush=True)
    for w in w_list:
        t0 = time.time()
        seeds, rounds_used, n_fb = dp_best_of_1(es, w, ckpt=ckpt, use_ema=use_ema,
                                                kin=env.kin, device=dev, q0_fallback=q0)
        t_seed = time.time() - t0
        t0 = time.time()
        L = roll(seeds, p0, d, n, env=env, controller='hybrid_variantB', classical=classical_best,
                 agent=agent, tdm=tdm, prefix=f'[cfg w={w}] ')
        s = stats(L * tdm, l_oracle)
        results['cfg'][f'{w}'] = {**s, 't_seed_s': t_seed, 't_roll_s': time.time() - t0,
                                  'rounds_used': rounds_used, 'n_unresolved': n_fb}
        np.savez_compressed(out_dir / f'cfg_w{w}.npz', seeds=seeds, L=L)
        print(f'[cfg] w={w}: l(m)={fmt(s["l"])} %={fmt(s["pct"])} '
              f'(rounds={rounds_used}, unresolved={n_fb}, {t_seed:.0f}s seed)', flush=True)

    (out_dir / 'sweep_results.json').write_text(json.dumps(results, indent=2))
    print(f'\n[sweep] wrote {out_dir/"sweep_results.json"}', flush=True)
    print('\n==== GAIN TABLE (q0_seed, classical) ====')
    for nm in ['mu_only', 'jl_only', 'theta_only', 'best_combo']:
        r = results['gain'][nm]
        print(f'{nm:11s} {r["gains"]}: l(m)={fmt(r["l"])} | %={fmt(r["pct"])}')
    print('\n==== CFG TABLE (DP best-of-1, hybrid) ====')
    for w in w_list:
        r = results['cfg'][f'{w}']
        print(f'w={w}: l(m)={fmt(r["l"])} | %={fmt(r["pct"])}')


if __name__ == '__main__':
    main()
