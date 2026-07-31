"""Heuristic seed baselines for the EMG journal paper (conference Table-2
pattern): null-space gradient ascent from the task's generating
configuration with periodic Newton re-projection onto the exact pose.

Both heuristics share one procedure and differ only in objective:
  q_mu : maximize directional manipulability along the motion direction
  q_jl : maximize joint-limit centering
Because the ascent stays within the (1-D) null space of the fixed pose,
both remain confined to the SMM branch of q0_seed and cannot cross to
another branch -- the structural handicap the proposed enumeration
removes.

Stages: gen --set validation|external|sealed   -> heur_{set}.npz
        roll --set S --controller hybrid|classical|rl
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenHybridController, FrozenRLController, rollout_selected_seeds)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.emg_problem_analysis import (
    FrozenClassicalController, _project_full_pose, DEFAULT_GAINS)
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
G = Path('Yuan/unified_rl/runs/emg_analysis')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
TAU_ENTER, TAU_EXIT = 0.985, 0.96
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '128'))
STEP = 0.03           # ascent step along the 1-D null direction [rad]
MAX_ITERS = 120
REPROJECT_EVERY = 5


def _src(which):
    return np.load(D / ('ikpool_candidates.npz' if which == 'train'
                        else f'ikpool_{which}_candidates.npz'))


@torch.no_grad()
def _null_vec(kin, q):
    _, _, jac, _ = kin.tcp_fk_jac(q)
    _, _, Vh = torch.linalg.svd(jac)
    return Vh[:, -1, :]


@torch.no_grad()
def _objective(kin, q, line_dir, kind):
    if kind == 'jl':
        mid = kin.q_mid; half = (kin.lmt_up - kin.lmt_lo) / 2
        return -(((q - mid) / half) ** 2).sum(-1)
    p, R, jac, _ = kin.tcp_fk_jac(q)
    J = jac[:, :3, :]
    JJt = J @ J.transpose(-1, -2) + 1e-6 * torch.eye(
        3, device=q.device, dtype=q.dtype)
    d = line_dir.unsqueeze(-1)
    inv_quad = (d.transpose(-1, -2) @ torch.linalg.inv(JJt) @ d).squeeze(-1).squeeze(-1)
    return inv_quad.clamp_min(1e-12).rsqrt()


@torch.no_grad()
def _ascend(kin, q0, p_t, R_t, line_dir, kind):
    """1-D manifold hill-climb from q0; returns the local optimum."""
    lo, hi = kin.lmt_lo, kin.lmt_up
    q = q0.clone()
    best = _objective(kin, q, line_dir, kind)
    stall = 0
    for it in range(MAX_ITERS):
        v = _null_vec(kin, q)
        for s in (STEP, -STEP):
            q_try = q + s * v
            if ((q_try <= lo + 0.01) | (q_try >= hi - 0.01)).any(-1).item():
                continue
            if (it + 1) % REPROJECT_EVERY == 0:
                q_try = _project_full_pose(kin, q_try, p_t, R_t, iters=2)
            f = _objective(kin, q_try, line_dir, kind)
            if (f > best).item():
                q, best, stall = q_try, f, 0
                break
        else:
            stall += 1
            if stall >= 2:
                break
    return _project_full_pose(kin, q, p_t, R_t, iters=4)


def stage_gen(args, device):
    which = args.set
    out = G / f'heur_{which}.npz'
    if out.exists():
        print(f'[heur gen {which}] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    kin = env.kin
    c = _src(which)
    m = len(c['p0'])
    qs = {'mu': np.zeros((m, 7), np.float32), 'jl': np.zeros((m, 7), np.float32)}
    times = {'mu': 0.0, 'jl': 0.0}
    for i in range(m):
        p_t = torch.as_tensor(c['p0'][i], device=device, dtype=kin.dtype).unsqueeze(0)
        R_t = _build_R_with_z(
            torch.as_tensor(c['n_target'][i], device=device,
                            dtype=kin.dtype).unsqueeze(0),
            torch.as_tensor(c['line_dir'][i], device=device, dtype=kin.dtype))
        ld = torch.as_tensor(c['line_dir'][i], device=device,
                             dtype=kin.dtype).unsqueeze(0)
        q0 = _project_full_pose(
            kin, torch.as_tensor(c['q0_pilot'][i], device=device,
                                 dtype=kin.dtype).unsqueeze(0), p_t, R_t, iters=5)
        for kind in ('mu', 'jl'):
            t0 = time.time()
            qs[kind][i] = _ascend(kin, q0, p_t, R_t, ld, kind)[0].cpu().numpy()
            times[kind] += time.time() - t0
        if (i + 1) % 400 == 0:
            print(f'[heur gen {which}] {i+1}/{m}', flush=True)
    np.savez(out, q_mu=qs['mu'], q_jl=qs['jl'],
             p0=c['p0'], line_dir=c['line_dir'], n_target=c['n_target'],
             q0_pilot=c['q0_pilot'],
             t_ms_mu=np.float64(times['mu'] / m * 1e3),
             t_ms_jl=np.float64(times['jl'] / m * 1e3))
    print(f'[heur gen {which}] done  t_mu={times["mu"]/m*1e3:.1f}ms '
          f't_jl={times["jl"]/m*1e3:.1f}ms', flush=True)


def stage_roll(args, device):
    which, ctl_kind = args.set, args.controller
    out = G / f'heur_{which}_{ctl_kind}_returns.npz'
    if out.exists():
        print(f'[heur roll] {out.name} exists, skip'); return
    h = np.load(G / f'heur_{which}.npz')
    m = len(h['p0'])
    tmp = G / f'_heur_{which}_seeds.npz'
    seeds = np.stack([h['q_mu'], h['q_jl']], 1)      # (m, 2, 7)
    np.savez(tmp, seeds=seeds.astype(np.float32),
             ik_ok=np.ones((m, 2), bool), p0=h['p0'], line_dir=h['line_dir'],
             n_target=h['n_target'], q0_pilot=h['q0_pilot'],
             task_indices=np.arange(m, dtype=np.int64))
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    gamma = float(ppo_config_from_run(load_run_config(
        resolve_controller_dir(C0_DIR))).gamma)
    if ctl_kind == 'hybrid':
        agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
        ctl = FrozenHybridController(agent, ClassicalNullspaceController(env.kin),
                                     TAU_ENTER, TAU_EXIT)
    elif ctl_kind == 'rl':
        agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
        ctl = FrozenRLController(agent)
    else:
        ctl = FrozenClassicalController(env.kin, DEFAULT_GAINS)
    ds = CachedSeedCandidateDataset.from_npz(tmp, include_fallback=False)
    prog = np.zeros((m, 2), np.float32)
    for col in (0, 1):
        for s in range(0, m, ROLL_CHUNK):
            rows = torch.arange(s, min(s + ROLL_CHUNK, m))
            nr = len(rows)
            if nr < ROLL_CHUNK:
                rows = torch.cat([rows, rows[-1:].expand(ROLL_CHUNK - nr)])
            cb = ds.batch.index_select(rows).to(device=device, dtype=env.kin.dtype)
            res = rollout_selected_seeds(
                env, cb, torch.full((ROLL_CHUNK,), col, dtype=torch.long,
                                    device=device), ctl, gamma=gamma)
            prog[s:s + nr, col] = res.progress_m[:nr].cpu().numpy()
        print(f'[heur roll {which} {ctl_kind}] col {col} done', flush=True)
    np.savez(out, progress_mu=prog[:, 0], progress_jl=prog[:, 1])
    print(f'[heur roll] saved {out.name}  mu={prog[:,0].mean():.4f} '
          f'jl={prog[:,1].mean():.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('gen', 'roll'))
    ap.add_argument('--set', default='validation',
                    choices=('validation', 'external', 'sealed', 'train'))
    ap.add_argument('--controller', default='hybrid',
                    choices=('hybrid', 'classical', 'rl'))
    args = ap.parse_args()
    globals()[f'stage_{args.stage}'](args, torch.device('cuda:0'))


if __name__ == '__main__':
    main()
