"""Phase 2 only: CFG sweep with DP best-of-1 (retry projection until valid).

Uses LAZY Newton projection: DDIM samples a batch of 32 candidates per task
(cheap, GPU-batched), then projects them one-by-one and stops at the first
IK-valid configuration. Total Newton calls per task = 1 / IK_ok_rate on
average, not 32. If the entire 32-batch fails projection, resample once
(seed offset) and try again, up to a small safety cap.

Reuses the new defaults of ClassicalNullspaceController (0.8/0.4/0.2) as the
hybrid's boundary controller. Writes to sweeps/cfg_only_results.json so the
prior gain-sweep results are not overwritten.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import yaml

from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched, load_rl_agent
from Yuan.system_eval.sweep_gain_cfg import roll, stats, fmt, subset, W_LIST, TAU
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.seed_selection.diffusion import ddim_sample_q0, denormalize_q, load_ckpt
from Yuan.seed_selection.smm import _build_R_target_strict, newton_project


DDIM_STEPS = 50
SAMPLE_SEED = 42
CANDIDATES_PER_ROUND = 4
MAX_RETRY_ROUNDS = 8            # 4 x 8 = 32 attempts max; if all fail -> q0 fallback
CHUNK_TASKS = 64


def dp_lazy_newton(es_T, w, *, ckpt_path, use_ema, kin, device, q0_fallback):
    """DDIM (batched) + LAZY Newton: stop at first IK-valid sample per task.

    Returns (seeds, n_unresolved, avg_newton_calls_per_task).
    """
    p0_all = es_T['cs_p0'].astype(np.float32)
    d_all  = es_T['cs_line_dir'].astype(np.float32)
    n_all  = es_T['cs_n_target'].astype(np.float32)
    T = p0_all.shape[0]

    model, schedule, _mc, _step = load_ckpt(Path(ckpt_path), device, use_ema=use_ema)
    lo_np = kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = kin.lmt_up.detach().cpu().numpy().astype(np.float32)

    seeds = np.zeros((T, 7), dtype=np.float32)
    have  = np.zeros(T, dtype=bool)
    newton_calls = np.zeros(T, dtype=np.int64)

    for r in range(MAX_RETRY_ROUNDS):
        todo = np.where(~have)[0]
        if todo.size == 0:
            break
        torch.manual_seed(SAMPLE_SEED + r)
        np.random.seed(SAMPLE_SEED + r)

        # DDIM sample CANDIDATES_PER_ROUND candidates for each remaining task,
        # in task-chunks so GPU memory stays sane.
        for cstart in range(0, todo.size, CHUNK_TASKS):
            cend = min(cstart + CHUNK_TASKS, todo.size)
            tasks = todo[cstart:cend]
            Bt = tasks.size
            c_np = np.concatenate([p0_all[tasks], d_all[tasks], n_all[tasks]], axis=1).astype(np.float32)
            c_t = torch.from_numpy(c_np).to(device)
            c_rep = c_t.repeat_interleave(CANDIDATES_PER_ROUND, dim=0)
            q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                                    num_steps=DDIM_STEPS, cfg_w=w)
            q_raw = denormalize_q(q_norm).cpu().numpy().astype(np.float32)  # (Bt*K, 7)

            # Lazy Newton: per task, project samples one-by-one until first ok.
            for bi in range(Bt):
                ti = int(tasks[bi])
                p0 = p0_all[ti]; d = d_all[ti]; n = n_all[ti]
                R_tgt = _build_R_target_strict(n, d)
                for si in range(CANDIDATES_PER_ROUND):
                    q_seed = q_raw[bi * CANDIDATES_PER_ROUND + si]
                    q_ref, ok, _err = newton_project(kin, q_seed, p0, R_tgt, lo_np, hi_np)
                    newton_calls[ti] += 1
                    if ok:
                        seeds[ti] = q_ref
                        have[ti] = True
                        break
        print(f'  [dp_lazy w={w}] round {r+1}: {have.sum()}/{T} have IK-valid '
              f'(avg newton/task so far {newton_calls.mean():.2f})', flush=True)

    n_unresolved = int((~have).sum())
    if n_unresolved:
        print(f'  [dp_lazy w={w}] WARN safety cap; {n_unresolved} -> q0 fallback', flush=True)
        seeds[~have] = q0_fallback[~have]
    return seeds, n_unresolved, float(newton_calls.mean())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set',
                   default='Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    p.add_argument('--out-dir',
                   default='Yuan/system_eval/runs/eval_10k_systematic/sweeps')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device(cfg['runner']['device'] if torch.cuda.is_available() else 'cpu')

    w_list = [1.5] if args.smoke else list(W_LIST)

    z = np.load(Path(args.eval_set), allow_pickle=False)
    es = {k: z[k] for k in z.files}
    T_all = es['cs_p0'].shape[0]
    T = 256 if args.smoke else T_all
    es = subset(es, np.arange(T), T_all)
    p0 = es['cs_p0'].astype(np.float32)
    d  = es['cs_line_dir'].astype(np.float32)
    n  = es['cs_n_target'].astype(np.float32)
    q0 = es['q0_seed'].astype(np.float32)
    tdm = float(cfg['env']['target_distance_m'])
    l_oracle = es['max_label_L'].astype(np.float32) * tdm

    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']), device=dev)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    classical_best = ClassicalNullspaceController(env.kin)   # new defaults 0.8/0.4/0.2
    print(f'[cfg-only] gains in use: mu={classical_best.manip_gain} '
          f'jl={classical_best.jl_gain} theta={classical_best.angle_boundary_gain}', flush=True)
    ckpt = cfg['diffusion']['ckpt']; use_ema = bool(cfg['diffusion'].get('use_ema', True))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {'meta': {'T': T, 'w_list': w_list, 'tdm': tdm,
                        'gains': [classical_best.manip_gain, classical_best.jl_gain,
                                  classical_best.angle_boundary_gain],
                        'tau_enter': float(cfg['rl_controller']['tau_enter']),
                        'tau_exit':  float(cfg['rl_controller']['tau_exit']),
                        'seed_policy': 'DP lazy newton; retry until IK valid; batch 32'},
               'cfg': {}}
    for w in w_list:
        t0 = time.time()
        seeds, n_unres, avg_newton = dp_lazy_newton(
            es, w, ckpt_path=ckpt, use_ema=use_ema, kin=env.kin, device=dev, q0_fallback=q0)
        t_seed = time.time() - t0
        t0 = time.time()
        te = float(cfg['rl_controller']['tau_enter'])
        tx = float(cfg['rl_controller']['tau_exit'])
        L = roll(seeds, p0, d, n, env=env, controller='hybrid_variantB',
                 classical=classical_best, agent=agent, tdm=tdm,
                 tau_enter=te, tau_exit=tx, prefix=f'[cfg w={w}] ')
        s = stats(L * tdm, l_oracle)
        t_seed_per_task_ms = 1000.0 * t_seed / max(T, 1)
        results['cfg'][f'{w}'] = {**s, 't_seed_s': t_seed, 't_roll_s': time.time() - t0,
                                  't_seed_per_task_ms': t_seed_per_task_ms,
                                  'avg_newton_calls_per_task': avg_newton,
                                  'n_unresolved': n_unres}
        np.savez_compressed(out_dir / f'cfg_only_w{w}.npz', seeds=seeds, L=L)
        print(f'[cfg] w={w}: l(m)={fmt(s["l"])} %={fmt(s["pct"])} '
              f'(t/task={t_seed_per_task_ms:.1f}ms, avg_newton={avg_newton:.2f}, '
              f'unresolved={n_unres}, {t_seed:.0f}s seed total)', flush=True)

    (out_dir / 'cfg_only_results.json').write_text(json.dumps(results, indent=2))
    print(f'\n[cfg-only] wrote {out_dir/"cfg_only_results.json"}', flush=True)
    print('\n==== CFG TABLE (DP lazy-newton retry, hybrid w/ 0.8/0.4/0.2) ====')
    for w in w_list:
        r = results['cfg'][f'{w}']
        print(f'w={w}: l(m)={fmt(r["l"])} | %={fmt(r["pct"])} t/task={r["t_seed_per_task_ms"]:.1f}ms '
              f'newton/task={r["avg_newton_calls_per_task"]:.2f}')


if __name__ == '__main__':
    main()
