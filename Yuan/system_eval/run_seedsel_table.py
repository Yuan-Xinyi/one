"""Heuristic seed comparison table (tab:seedsel).

Generates two heuristic seeds per task by null-space gradient ascent starting
from q0_seed (which is on the constraint manifold by construction):
  - Manipulability seed : maximize w_d (directional manipulability along d)
  - Joint-Limit seed    : minimize ||(q-q_mid)/q_half||^2 (centering)

The DP rows reuse the cached seeds + rollouts from the main-table run.

Rolls each seed through Classical, RL (tau=1.0), Hybrid (adopted tau), and
reports % vs the controller-aware oracle oracle_hyb.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.seed_selection.smm import _build_R_target_strict, newton_project


SEED_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'
ORACLE_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/cell_oracle_hyb_results.npz'
EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
MAIN_PATTERN = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/main_{name}.npz'
OUT_JSON = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/seedsel_results.json'

N_ITER = 20            # null-space ascent iterations
ALPHA = 0.05           # step size
CHUNK = 256            # GPU chunk for grad
REPROJ_EVERY = 5       # newton_project cadence (also enforced at final step)


def directional_manip(kin, q_g, d_b, damping=1e-3):
    """Directional manipulability w_d at q along d. q_g: (B,7) requires_grad; d_b: (B,3)."""
    _, _, J, _ = kin.tcp_fk_jac(q_g)
    J_pos = J[:, :3, :]
    eye3 = torch.eye(3, device=q_g.device, dtype=q_g.dtype).expand(q_g.shape[0], 3, 3)
    JJt = J_pos @ J_pos.transpose(-1, -2) + (damping ** 2) * eye3
    d_col = d_b.unsqueeze(-1)
    inv_quad = (d_col.transpose(-1, -2)
                @ torch.linalg.solve(JJt, d_col)).squeeze(-1).squeeze(-1).clamp_min(1e-12)
    return inv_quad.pow(-0.5)


def heuristic_grad_batched(kin, q_b, d_b, kind):
    """Return null-space-projected heuristic gradient. q_b: (B,7); d_b: (B,3)."""
    with torch.enable_grad():
        q_g = q_b.detach().clone().requires_grad_(True)
        _, _, J, _ = kin.tcp_fk_jac(q_g)
        if kind == 'manip':
            obj = directional_manip(kin, q_g, d_b).sum()
        elif kind == 'jl':
            half = (kin.lmt_up - kin.lmt_lo).clamp_min(1e-6) / 2.0
            qn = (q_g - kin.q_mid) / half
            obj = -(qn * qn).sum()        # maximise -||qn||^2
        else:
            raise ValueError(kind)
        grad = torch.autograd.grad(obj, q_g)[0].detach()
        J_pos_d = J[:, :3, :].detach()
    with torch.no_grad():
        # null-space projector via SVD (fp64 for stability, then back)
        J_d = J_pos_d.double()
        _, _, Vh = torch.linalg.svd(J_d, full_matrices=True)
        N = Vh.transpose(-1, -2)[..., -4:]              # (B,7,4) span(N(J))
        coeffs = (N.transpose(-1, -2) @ grad.double().unsqueeze(-1)).squeeze(-1)
        proj = (N @ coeffs.unsqueeze(-1)).squeeze(-1)
        return proj.to(q_b.dtype)


def heuristic_seed_chunk(kin, q_init_t, p0_np, d_t, n_np, kind, lo_np, hi_np,
                         n_iter=N_ITER, alpha=ALPHA, reproj_every=REPROJ_EVERY):
    """One chunk: null-space ascent with periodic Newton reprojection + final."""
    q = q_init_t.clone()
    B = q.shape[0]
    d_np = d_t.cpu().numpy()
    for it in range(n_iter):
        step = heuristic_grad_batched(kin, q, d_t, kind)
        q = q + alpha * step
        if (it + 1) % reproj_every == 0 or it == n_iter - 1:
            q_cpu = q.detach().cpu().numpy()
            for bi in range(B):
                R_tgt = _build_R_target_strict(n_np[bi], d_np[bi])
                q_ref, ok, _ = newton_project(kin, q_cpu[bi], p0_np[bi], R_tgt,
                                              lo_np, hi_np)
                if ok:
                    q_cpu[bi] = q_ref
            q = torch.from_numpy(q_cpu).to(q.device, q.dtype)
    return q.cpu().numpy()


def gen_heuristic_seeds(kind, eval_set, q0_seed, kin, device, label):
    p0 = eval_set['cs_p0'].astype(np.float32)
    d  = eval_set['cs_line_dir'].astype(np.float32)
    n  = eval_set['cs_n_target'].astype(np.float32)
    T = q0_seed.shape[0]
    seeds = np.zeros((T, 7), dtype=np.float32)
    lo_np = kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = kin.lmt_up.detach().cpu().numpy().astype(np.float32)
    t0 = time.time()
    for start in range(0, T, CHUNK):
        end = min(start + CHUNK, T)
        q_init_t = torch.from_numpy(q0_seed[start:end]).to(device)
        d_t      = torch.from_numpy(d[start:end]).to(device)
        seeds[start:end] = heuristic_seed_chunk(
            kin, q_init_t, p0[start:end], d_t, n[start:end], kind, lo_np, hi_np)
        if (start // CHUNK) % 5 == 0:
            elapsed = time.time() - t0
            print(f'  [{label}] {end}/{T} tasks ({elapsed:.0f}s, '
                  f'{elapsed/max(1,end)*1000:.1f} ms/task)', flush=True)
    dt = time.time() - t0
    print(f'[{label}] done in {dt:.0f}s ({dt/T*1000:.1f} ms/task)', flush=True)
    return seeds, dt / T * 1000  # t per task in ms


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
    z = np.load(EVAL_NPZ)
    eval_set = {k: z[k] for k in z.files}
    T = eval_set['cs_p0'].shape[0]
    q0_seed = eval_set['q0_seed'].astype(np.float32)
    oh = np.load(ORACLE_NPZ)['L_best'].astype(np.float32) * tdm

    env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']),
                    device=dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    tau_e = float(cfg['rl_controller']['tau_enter'])
    tau_x = float(cfg['rl_controller']['tau_exit'])
    print(f'[seedsel] gains mu={classical.manip_gain} jl={classical.jl_gain} '
          f'theta={classical.angle_boundary_gain}; tau=({tau_e},{tau_x})',
          flush=True)

    seed_sources = {}
    print('\n[seedsel] generating Manipulability seeds...', flush=True)
    seeds_m, t_m_ms = gen_heuristic_seeds('manip', eval_set, q0_seed, env.kin, dev,
                                          'Manipulability')
    np.savez_compressed('Yuan/system_eval/runs/eval_10k_systematic/sweeps/seedsel_manip.npz',
                        seeds=seeds_m, t_per_task_ms=t_m_ms)
    seed_sources['Manipulability'] = (seeds_m, t_m_ms)

    print('\n[seedsel] generating Joint-Limit seeds...', flush=True)
    seeds_j, t_j_ms = gen_heuristic_seeds('jl', eval_set, q0_seed, env.kin, dev,
                                          'Joint-Limit')
    np.savez_compressed('Yuan/system_eval/runs/eval_10k_systematic/sweeps/seedsel_jl.npz',
                        seeds=seeds_j, t_per_task_ms=t_j_ms)
    seed_sources['Joint-Limit'] = (seeds_j, t_j_ms)

    # DP: reuse cached deployment seeds (cfg_only_w1.5)
    seeds_d = np.load(SEED_NPZ)['seeds'].astype(np.float32)
    # DP t/task is in cfg_only_results.json -- read directly
    cfg_only = json.load(open('Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_results.json'))
    t_d_ms = float(cfg_only['cfg']['1.5']['t_seed_per_task_ms'])
    seed_sources['DP'] = (seeds_d, t_d_ms)

    p0 = eval_set['cs_p0'].astype(np.float32)
    d  = eval_set['cs_line_dir'].astype(np.float32)
    n  = eval_set['cs_n_target'].astype(np.float32)

    controllers = [
        ('Classical', 'classical', None, None),
        ('RL',        'hybrid_variantB', 1.0, 1.0),
        ('Hybrid',    'hybrid_variantB', tau_e, tau_x),
    ]

    results = {'meta': {'T': T, 'tdm': tdm,
                        't_per_task_ms': {k: v[1] for k, v in seed_sources.items()},
                        'tau_adopted': [tau_e, tau_x]},
               'rows': {}}

    for cname, ctrl, te, tx in controllers:
        results['rows'][cname] = {}
        # For DP × {Classical,RL,Hybrid} we can reuse main_table results
        for sname, (seeds, t_ms) in seed_sources.items():
            if sname == 'DP':
                main_npz = MAIN_PATTERN.format(name=cname)
                if Path(main_npz).exists():
                    cache = np.load(main_npz)
                    L = cache['L'].astype(np.float32)
                    print(f'[seedsel] reused main_{cname}.npz for DP × {cname}',
                          flush=True)
                else:
                    L = _rollout(seeds, p0, d, n, env, ctrl, classical, agent,
                                 te, tx, tdm, prefix=f'[{sname} × {cname}] ')
            else:
                L = _rollout(seeds, p0, d, n, env, ctrl, classical, agent,
                             te, tx, tdm, prefix=f'[{sname} × {cname}] ')
            l_m = L * tdm
            with np.errstate(invalid='ignore', divide='ignore'):
                pct = 100.0 * l_m / np.maximum(oh, 1e-9)
            row = {'t_ms': t_ms, 'l': stats(l_m), 'pct': stats(pct)}
            results['rows'][cname][sname] = row
            np.savez_compressed(
                f'Yuan/system_eval/runs/eval_10k_systematic/sweeps/seedsel_{cname}_{sname}.npz',
                L=L, pct=pct)
            print(f'[seedsel] {sname:>14s} × {cname:9s}:  t={t_ms:6.1f}ms  '
                  f'l(m)={fmtl(row["l"])} | %={fmtp(row["pct"])}', flush=True)

    Path(OUT_JSON).write_text(json.dumps(results, indent=2))
    print(f'\n[seedsel] wrote {OUT_JSON}', flush=True)


def _rollout(seeds, p0, d, n, env, ctrl, classical, agent, te, tx, tdm, prefix=''):
    kwargs = {'env': env, 'controller': ctrl, 'classical': classical,
              'agent': agent, 'target_distance_m': tdm,
              'progress_every_chunks': max(1, (seeds.shape[0] // env.n_envs) // 8),
              'progress_prefix': prefix}
    if ctrl == 'hybrid_variantB':
        kwargs['tau_enter'] = te
        kwargs['tau_exit'] = tx
    res = rollout_seeds_batched(seeds.astype(np.float32), p0, d, n, **kwargs)
    return res['L'].astype(np.float32)


if __name__ == '__main__':
    main()
