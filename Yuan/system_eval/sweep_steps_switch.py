"""Ablations 3 and 4: DDIM steps sweep + switching thresholds sweep.

Both use the adopted defaults from earlier sweeps:
  - DP deployment policy (sweep_cfg_only.dp_lazy_newton): batch 4, retry up to
    8 rounds (32 attempts), q0 fallback.
  - cfg_w = 1.5; classical gains 0.8/0.4/0.2 (now the class defaults).
  - Controller: hybrid_variantB.

For the switching sweep we reuse the DP seeds cached at
sweeps/cfg_only_w1.5.npz (same DDIM=50, w=1.5) so all four threshold rows
see identical initial conditions.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, yaml

from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched, load_rl_agent
from Yuan.system_eval.sweep_gain_cfg import roll, stats, fmt, subset, TAU
from Yuan.system_eval.sweep_cfg_only import dp_lazy_newton
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController


DDIM_STEPS_LIST = [10, 20, 50, 100]
# Expanded grid: 6 single-threshold rows + 2 hysteresis rows.
TAU_PAIRS = [
    (0.85, 0.85), (0.90, 0.90), (0.95, 0.95),
    (0.97, 0.97), (0.98, 0.98), (0.99, 0.99),
    (0.98, 0.94), (0.99, 0.93),
]
CFG_W = 1.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set',
                   default='Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    p.add_argument('--out-dir',
                   default='Yuan/system_eval/runs/eval_10k_systematic/sweeps')
    p.add_argument('--cached-w1p5-seeds',
                   default='Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz',
                   help='Reused for the switching sweep (DDIM=50, w=1.5).')
    p.add_argument('--skip-steps', action='store_true', help='skip DDIM-steps sweep')
    p.add_argument('--skip-switch', action='store_true', help='skip switching sweep')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def roll_with_tau(seeds_T7, p0, d, n, *, env, classical, agent, tdm,
                  tau_enter, tau_exit, prefix=''):
    """rollout_seeds_batched with explicit tau pair (roll()'s helper hard-codes TAU).

    Returns (L, switch_count) both of shape (T,)."""
    T = seeds_T7.shape[0]
    res = rollout_seeds_batched(
        seeds_T7.astype(np.float32), p0, d, n, env=env, controller='hybrid_variantB',
        classical=classical, agent=agent, tau_enter=tau_enter, tau_exit=tau_exit,
        target_distance_m=tdm,
        progress_every_chunks=max(1, (T // env.n_envs) // 8), progress_prefix=prefix)
    return res['L'].astype(np.float32), res['switch_count'].astype(np.int32)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device(cfg['runner']['device'] if torch.cuda.is_available() else 'cpu')

    steps_list = [50] if args.smoke else list(DDIM_STEPS_LIST)
    tau_pairs = [(0.98, 0.98)] if args.smoke else list(TAU_PAIRS)

    z = np.load(Path(args.eval_set), allow_pickle=False)
    es = {k: z[k] for k in z.files}
    T_all = es['cs_p0'].shape[0]
    T = 256 if args.smoke else T_all
    es = subset(es, np.arange(T), T_all)
    p0 = es['cs_p0'].astype(np.float32); d = es['cs_line_dir'].astype(np.float32)
    n  = es['cs_n_target'].astype(np.float32); q0 = es['q0_seed'].astype(np.float32)
    tdm = float(cfg['env']['target_distance_m'])
    l_oracle = es['max_label_L'].astype(np.float32) * tdm

    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']), device=dev)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    classical = ClassicalNullspaceController(env.kin)
    print(f'[steps+switch] gains: mu={classical.manip_gain} jl={classical.jl_gain} '
          f'theta={classical.angle_boundary_gain}; cfg_w={CFG_W}', flush=True)
    ckpt = cfg['diffusion']['ckpt']; use_ema = bool(cfg['diffusion'].get('use_ema', True))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {'meta': {'T': T, 'cfg_w': CFG_W, 'tdm': tdm,
                        'gains': [classical.manip_gain, classical.jl_gain,
                                  classical.angle_boundary_gain],
                        'tau_enter_steps': float(cfg['rl_controller']['tau_enter']),
                        'tau_exit_steps':  float(cfg['rl_controller']['tau_exit'])},
               'steps': {}, 'switch': {}}

    # ===== DDIM steps sweep =====
    if not args.skip_steps:
        from Yuan.system_eval import sweep_cfg_only as scfg
        for s in steps_list:
            t0 = time.time()
            old = scfg.DDIM_STEPS
            scfg.DDIM_STEPS = int(s)
            try:
                seeds, n_unres, avg_n = dp_lazy_newton(es, CFG_W, ckpt_path=ckpt,
                                                       use_ema=use_ema, kin=env.kin,
                                                       device=dev, q0_fallback=q0)
            finally:
                scfg.DDIM_STEPS = old
            t_seed = time.time() - t0
            t_seed_ms = 1000.0 * t_seed / max(T, 1)
            te = float(cfg['rl_controller']['tau_enter'])
            tx = float(cfg['rl_controller']['tau_exit'])
            L = roll(seeds, p0, d, n, env=env, controller='hybrid_variantB',
                     classical=classical, agent=agent, tdm=tdm,
                     tau_enter=te, tau_exit=tx, prefix=f'[steps={s}] ')
            sst = stats(L * tdm, l_oracle)
            results['steps'][f'{s}'] = {**sst, 't_seed_per_task_ms': t_seed_ms,
                                        'avg_newton_calls_per_task': avg_n,
                                        'n_unresolved': n_unres}
            np.savez_compressed(out_dir / f'steps_{s}.npz', seeds=seeds, L=L)
            print(f'[steps] ddim={s}: l(m)={fmt(sst["l"])} %={fmt(sst["pct"])} '
                  f'(t/task={t_seed_ms:.1f}ms, newton={avg_n:.2f}, unres={n_unres})', flush=True)

    # ===== Switching thresholds sweep (reuse cached w=1.5 seeds) =====
    if not args.skip_switch:
        cache_path = Path(args.cached_w1p5_seeds)
        if not cache_path.exists():
            raise SystemExit(f'switching sweep needs {cache_path}; run sweep_cfg_only first.')
        cached = np.load(cache_path, allow_pickle=False)
        seeds_w15 = cached['seeds'].astype(np.float32)
        if args.smoke:
            seeds_w15 = seeds_w15[:T]
        print(f'[switch] reusing cached w=1.5 DP seeds: shape={seeds_w15.shape}', flush=True)
        for te, tx in tau_pairs:
            t0 = time.time()
            L, sw = roll_with_tau(seeds_w15, p0, d, n, env=env, classical=classical,
                                  agent=agent, tdm=tdm, tau_enter=float(te),
                                  tau_exit=float(tx), prefix=f'[tau=({te},{tx})] ')
            sst = stats(L * tdm, l_oracle)
            sw_mean = float(sw.mean()); sw_std = float(sw.std())
            sw_max = int(sw.max()); sw_zero_frac = float((sw == 0).mean())
            results['switch'][f'{te}_{tx}'] = {**sst,
                                               't_roll_s': time.time() - t0,
                                               'tau_enter': float(te), 'tau_exit': float(tx),
                                               'switches_mean': sw_mean,
                                               'switches_std': sw_std,
                                               'switches_max': sw_max,
                                               'switches_zero_frac': sw_zero_frac}
            np.savez_compressed(out_dir / f'switch_{te}_{tx}.npz', L=L, switches=sw)
            print(f'[switch] te={te} tx={tx}: switches={sw_mean:.2f}+/-{sw_std:.2f} '
                  f'(max {sw_max}, zero {100*sw_zero_frac:.1f}%) | '
                  f'l(m)={fmt(sst["l"])} | %={fmt(sst["pct"])}', flush=True)

    # Merge with any existing JSON so --skip-steps / --skip-switch reruns do
    # not clobber the half that wasn't recomputed.
    out_json = out_dir / 'steps_switch_results.json'
    if out_json.exists():
        try:
            prev = json.loads(out_json.read_text())
            if args.skip_steps and 'steps' in prev:
                results['steps'] = prev['steps']
            if args.skip_switch and 'switch' in prev:
                results['switch'] = prev['switch']
            if 'meta' in prev:
                merged_meta = dict(prev['meta']); merged_meta.update(results['meta'])
                results['meta'] = merged_meta
        except Exception as e:
            print(f'[steps+switch] WARN could not merge existing JSON: {e}', flush=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f'\n[steps+switch] wrote {out_json}', flush=True)

    if not args.skip_steps:
        print('\n==== DDIM STEPS TABLE ====')
        for s in steps_list:
            r = results['steps'][f'{s}']
            print(f'ddim={s}: t/task={r["t_seed_per_task_ms"]:.1f}ms  '
                  f'l(m)={fmt(r["l"])} | %={fmt(r["pct"])}')
    if not args.skip_switch:
        print('\n==== SWITCHING TABLE ====')
        for te, tx in tau_pairs:
            r = results['switch'][f'{te}_{tx}']
            print(f'({te},{tx}): switches={r["switches_mean"]:.2f} '
                  f'l(m)={fmt(r["l"])} | %={fmt(r["pct"])}')


if __name__ == '__main__':
    main()
