"""Part-A problem analysis for the EMG journal paper (100 small-scale tasks).

A0  Attribution: two-factor grid (branch-representative seeds x classical
    gain settings) -> variance decomposition: how much of the achievable
    length is decided by the initial configuration vs the controller.
A1  Seed side, rigorous: retreat to the 1-D SMM (tool axis fixed at the
    nominal n). Trace the manifold by integrating the null vector of the
    full 6x7 Jacobian with full-pose Newton re-projection; segment into
    joint-limit-cut branches; roll sampled points under the default
    classical controller -> progress-vs-arclength curves; between-branch
    vs within-branch variance.
A2  Controller side: per-task optimal gains scatter (slice of A0) +
    two-phase gain switching on a subset, showing that no single gain set
    -- indeed no single *static* gain set -- suffices.

Stages: sample | a0 | a1 | a2seg | analyze
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_run_config, ppo_config_from_run,
    resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    build_task_aligned_basis, rollout_selected_seeds)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
F = Path('Yuan/unified_rl/runs/ikpool_final')
A = Path('Yuan/unified_rl/runs/emg_analysis')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '128'))
N_TASKS = 100
SEED = 20260801
GAIN_GRID = [(km, kj, kt)
             for km in (0.2, 0.8, 1.4)
             for kj in (0.1, 0.4, 1.0)
             for kt in (0.2, 0.6, 1.0)]
DEFAULT_GAINS = (0.8, 0.4, 0.2)
MAX_REPS = 8
BRANCH_LINK = 1.0          # single-linkage threshold [rad] for branch clusters


class FrozenClassicalController:
    """Classical-only controller in the shared action parameterization."""

    def __init__(self, kin, gains):
        km, kj, kt = gains
        self.classical = ClassicalNullspaceController(
            kin, manip_gain=km, jl_gain=kj, angle_boundary_gain=kt)
        self.switch_at = None          # optional (step, second-gains ctrl)
        self._t = 0

    def reset(self, env):
        self._t = 0

    def action(self, env):
        self._t += 1
        cls = self.classical
        if self.switch_at is not None and self._t > self.switch_at[0]:
            cls = self.switch_at[1]
        with torch.no_grad():
            basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping)
            q_dot = cls.q_dot_null(env.q, env.line_dir, env.n_target)
            act = (basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
            return torch.nan_to_num(act / env.a_max, nan=0.0).clamp(-1.0, 1.0)


def _load_val():
    cand = np.load(D / 'ikpool_validation_candidates.npz')
    enum_ret = np.load(F / 'enumeration_validation_returns_hybrid.npz')
    P = np.nan_to_num(enum_ret['progress_m'])
    V = enum_ret['valid']
    lref = np.where(V, P, -np.inf).max(1)
    return cand, P, V, lref


def stage_sample(args, device):
    A.mkdir(parents=True, exist_ok=True)
    _, _, _, lref = _load_val()
    rng = np.random.default_rng(SEED)
    easy = np.nonzero(lref >= 0.80)[0]
    med = np.nonzero((lref >= 0.45) & (lref < 0.80))[0]
    hard = np.nonzero((lref < 0.45) & (lref > 0))[0]
    pick = np.concatenate([
        rng.choice(easy, 26, replace=False),
        rng.choice(med, 48, replace=False),
        rng.choice(hard, 26, replace=False)])
    (A / 'tasks.json').write_text(json.dumps(
        {'rows': sorted(int(x) for x in pick),
         'buckets': {'easy': 26, 'medium': 48, 'difficult': 26},
         'seed': SEED}, indent=1))
    print(f'[sample] 100 tasks (E26/M48/D26), lref range '
          f'{lref[pick].min():.2f}-{lref[pick].max():.2f}', flush=True)


def _branch_reps(seeds, valid, progress):
    """Single-linkage clusters in joint space; return medoid reps (<=MAX_REPS)."""
    q = seeds[valid]
    if len(q) == 0:
        return np.zeros((0, 7), np.float32), np.zeros(0, np.int64)
    d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    n = len(q)
    lab = np.arange(n)
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < BRANCH_LINK and lab[j] != lab[i]:
                lab[lab == lab[j]] = lab[i]
    uniq, counts = np.unique(lab, return_counts=True)
    order = uniq[np.argsort(-counts)][:MAX_REPS]
    reps, labels = [], []
    for u in order:
        idx = np.nonzero(lab == u)[0]
        sub = d[np.ix_(idx, idx)]
        reps.append(q[idx[sub.sum(1).argmin()]])
        labels.append(u)
    return np.asarray(reps, np.float32), np.asarray(labels)


def _make_seed_npz(path, cand, rows, seed_lists):
    """One row per (task, rep) with K=1 candidate for rollout convenience."""
    p0, ld, nt, fb = [], [], [], []
    seeds = []
    task_of, rep_of = [], []
    for r, qs in zip(rows, seed_lists):
        for k, q in enumerate(qs):
            seeds.append(q[None, :])
            p0.append(cand['p0'][r]); ld.append(cand['line_dir'][r])
            nt.append(cand['n_target'][r]); fb.append(cand['q0_pilot'][r])
            task_of.append(r); rep_of.append(k)
    np.savez(path, seeds=np.asarray(seeds, np.float32),
             ik_ok=np.ones((len(seeds), 1), bool),
             p0=np.asarray(p0, np.float32), line_dir=np.asarray(ld, np.float32),
             n_target=np.asarray(nt, np.float32),
             q0_pilot=np.asarray(fb, np.float32),
             task_indices=np.arange(len(seeds), dtype=np.int64),
             orig_task=np.asarray(task_of, np.int64),
             rep_index=np.asarray(rep_of, np.int64))
    return len(seeds)


def _roll(ds_path, controller, device, note=''):
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    gamma = float(ppo_config_from_run(load_run_config(
        resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(ds_path, include_fallback=False)
    n = len(ds)
    out = np.zeros(n, np.float32)
    ctl = controller(env.kin) if callable(controller) else controller
    for s in range(0, n, ROLL_CHUNK):
        rows = torch.arange(s, min(s + ROLL_CHUNK, n))
        nr = len(rows)
        if nr < ROLL_CHUNK:
            rows = torch.cat([rows, rows[-1:].expand(ROLL_CHUNK - nr)])
        cb = ds.batch.index_select(rows).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(
            env, cb, torch.zeros(ROLL_CHUNK, dtype=torch.long, device=device),
            ctl, gamma=gamma)
        out[s:s + nr] = res.progress_m[:nr].cpu().numpy()
    if note:
        print(f'[roll] {note}: {n} episodes done', flush=True)
    return out


def stage_a0(args, device):
    out = A / 'a0_grid.npz'
    if out.exists():
        print('[a0] exists, skip'); return
    cand, P, V, _ = _load_val()
    rows = json.loads((A / 'tasks.json').read_text())['rows']
    seed_lists = []
    for r in rows:
        reps, _ = _branch_reps(cand['seeds'][r][:32], V[r][:32], P[r][:32])
        seed_lists.append(reps)
    n_rows = _make_seed_npz(A / 'a0_seeds.npz', cand, rows, seed_lists)
    print(f'[a0] {n_rows} (task,rep) rows; grid {len(GAIN_GRID)} gains', flush=True)
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    grid = np.zeros((n_rows, len(GAIN_GRID)), np.float32)
    for gi, gains in enumerate(GAIN_GRID):
        ctl = FrozenClassicalController(env1.kin, gains)
        grid[:, gi] = _roll(A / 'a0_seeds.npz', ctl, device,
                            note=f'gains {gi+1}/{len(GAIN_GRID)} {gains}')
    meta = np.load(A / 'a0_seeds.npz')
    np.savez(out, grid=grid, task_indices=meta['orig_task'],
             rep_index=meta['rep_index'],
             gain_grid=np.asarray(GAIN_GRID, np.float32))
    print('[a0] saved', flush=True)


# ---------------- A1: 1-D SMM tracing ----------------
@torch.no_grad()
def _project_full_pose(kin, q, p_t, R_t, iters=3):
    """Newton projection onto the FULL 6-DoF pose (position + rotation)."""
    for _ in range(iters):
        p, R, jac, _ = kin.tcp_fk_jac(q)
        e_p = p_t - p
        R_err = R_t @ R.transpose(-1, -2)
        # rotation vector of R_err (batched, safe small-angle form)
        tr = R_err[:, 0, 0] + R_err[:, 1, 1] + R_err[:, 2, 2]
        ang = torch.arccos(((tr - 1) / 2).clamp(-1, 1))
        w = torch.stack([R_err[:, 2, 1] - R_err[:, 1, 2],
                         R_err[:, 0, 2] - R_err[:, 2, 0],
                         R_err[:, 1, 0] - R_err[:, 0, 1]], -1)
        s = (2 * torch.sin(ang)).clamp_min(1e-8).unsqueeze(-1)
        e_r = w / s * ang.unsqueeze(-1)
        e = torch.cat([e_p, e_r], -1)
        dq = torch.linalg.lstsq(jac, e.unsqueeze(-1)).solution.squeeze(-1)
        q = q + dq
    return q


@torch.no_grad()
def _trace_smm(kin, q0, p_t, R_t, lo, hi, h=0.02, max_steps=600):
    """Walk the 1-D SMM both directions from q0; return polyline (arc, q)."""
    pts = []
    for direction in (1.0, -1.0):
        q = q0.clone().unsqueeze(0)
        v_prev = None
        arc = 0.0
        for _ in range(max_steps):
            _, _, jac, _ = kin.tcp_fk_jac(q)
            _, _, Vh = torch.linalg.svd(jac[0])
            v = Vh[-1]
            # NOTE: 1-D @ 1-D CUDA dot SIGFPEs on this machine; use (a*b).sum()
            if v_prev is not None and (v * v_prev).sum() < 0:
                v = -v
            elif v_prev is None:
                v = v * direction
            v_prev = v
            q_new = q + h * v.unsqueeze(0)
            q_new = _project_full_pose(kin, q_new, p_t.unsqueeze(0),
                                       R_t.unsqueeze(0), iters=2)
            if ((q_new[0] <= lo + 0.01) | (q_new[0] >= hi - 0.01)).any():
                break
            arc += h
            q = q_new
            pts.append((direction * arc, q[0].cpu().numpy()))
            if arc > 8.0:
                break
    pts.sort(key=lambda t: t[0])
    return pts


def stage_a1(args, device):
    out = A / 'a1_smm.npz'
    if out.exists():
        print('[a1] exists, skip'); return
    from Yuan.seed_selection.smm.cone_ik import _build_R_with_z
    cand, P, V, _ = _load_val()
    rows = json.loads((A / 'tasks.json').read_text())['rows']
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    kin = env1.kin
    lo, hi = kin.lmt_lo, kin.lmt_up
    all_task, all_branch, all_arc, all_q = [], [], [], []
    for ti, r in enumerate(rows):
        p_t = torch.as_tensor(cand['p0'][r], device=device, dtype=kin.dtype)
        R_t = _build_R_with_z(
            torch.as_tensor(cand['n_target'][r], device=device,
                            dtype=kin.dtype).unsqueeze(0),
            torch.as_tensor(cand['line_dir'][r], device=device,
                            dtype=kin.dtype))[0]
        # germs: enumeration candidates re-projected onto the exact full pose
        germs = torch.as_tensor(
            cand['seeds'][r][:32][V[r][:32]], device=device, dtype=kin.dtype)
        germs = _project_full_pose(kin, germs, p_t.expand(len(germs), 3),
                                   R_t.expand(len(germs), 3, 3), iters=5)
        ok = ((germs > lo + 0.01) & (germs < hi - 0.01)).all(-1)
        p_chk, R_chk, _, _ = kin.tcp_fk_jac(germs)
        ok &= (p_chk - p_t).norm(dim=-1) < 5e-3
        germs = germs[ok]
        # dedup germs onto distinct branches
        kept = []
        for g in germs:
            if all((g - k).norm() > 0.3 for k in kept):
                kept.append(g)
        for bi, g in enumerate(kept[:6]):
            pts = _trace_smm(kin, g, p_t, R_t, lo, hi)
            step = max(1, len(pts) // 15)
            for arc, qq in pts[::step]:
                all_task.append(r); all_branch.append(bi)
                all_arc.append(arc); all_q.append(qq)
        if (ti + 1) % 20 == 0:
            print(f'[a1] traced {ti+1}/100 tasks, {len(all_q)} pts', flush=True)
    # roll all sampled SMM points under default classical gains
    seed_lists, row_list = [], []
    by_task = {}
    for t, q in zip(all_task, all_q):
        by_task.setdefault(t, []).append(q)
    for t, qs in by_task.items():
        row_list.append(t); seed_lists.append(np.asarray(qs, np.float32))
    _make_seed_npz(A / 'a1_seeds.npz', cand, row_list, seed_lists)
    env1b = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    prog = _roll(A / 'a1_seeds.npz',
                 FrozenClassicalController(env1b.kin, DEFAULT_GAINS),
                 device, note='a1 smm points')
    np.savez(out, task=np.asarray(all_task), branch=np.asarray(all_branch),
             arc=np.asarray(all_arc, np.float32),
             q=np.asarray(all_q, np.float32), progress=prog)
    print(f'[a1] saved {len(prog)} SMM point rollouts', flush=True)


def stage_a2seg(args, device):
    out = A / 'a2_segmented.npz'
    if out.exists():
        print('[a2seg] exists, skip'); return
    a0 = np.load(A / 'a0_grid.npz')
    grid, tidx, ridx = a0['grid'], a0['task_indices'], a0['rep_index']
    cand, _, _, _ = _load_val()
    # 20 tasks; best rep under default; top-5 gain sets globally
    default_idx = GAIN_GRID.index(DEFAULT_GAINS) if DEFAULT_GAINS in GAIN_GRID else 0
    mean_by_gain = grid.mean(0)
    top5 = np.argsort(-mean_by_gain)[:5]
    tasks = sorted(set(tidx.tolist()))[:20]
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    rows, seeds = [], []
    for t in tasks:
        m = tidx == t
        best_rep_row = np.nonzero(m)[0][grid[m, default_idx].argmax()]
        s = np.load(A / 'a0_seeds.npz')['seeds'][best_rep_row, 0]
        rows.append(t); seeds.append(s[None, :])
    _make_seed_npz(A / 'a2_seeds.npz', cand, rows, seeds)
    res = np.zeros((len(tasks), 5, 5), np.float32)
    for i, gi in enumerate(top5):
        for j, gj in enumerate(top5):
            ctl = FrozenClassicalController(env1.kin, GAIN_GRID[gi])
            if gi != gj:
                ctl.switch_at = (27, ClassicalNullspaceController(
                    env1.kin, *GAIN_GRID[gj]))
            res[:, i, j] = _roll(A / 'a2_seeds.npz', ctl, device,
                                 note=f'seg {i},{j}')
    np.savez(out, res=res, tasks=np.asarray(tasks), top5=top5,
             gain_grid=np.asarray(GAIN_GRID, np.float32))
    print('[a2seg] saved', flush=True)


def stage_analyze(args, device):
    rep = {}
    # ---- A0 variance decomposition ----
    a0 = np.load(A / 'a0_grid.npz')
    grid, tidx = a0['grid'], a0['task_indices']
    shares = []
    for t in sorted(set(tidx.tolist())):
        g = grid[tidx == t]                    # (reps, gains)
        if g.shape[0] < 2:
            continue
        gm = g.mean()
        ss_tot = ((g - gm) ** 2).sum()
        if ss_tot < 1e-9:
            continue
        ss_seed = (g.shape[1] * ((g.mean(1) - gm) ** 2).sum())
        ss_gain = (g.shape[0] * ((g.mean(0) - gm) ** 2).sum())
        ss_int = ss_tot - ss_seed - ss_gain
        shares.append((ss_seed / ss_tot, ss_gain / ss_tot,
                       max(ss_int, 0) / ss_tot))
    sh = np.asarray(shares)
    rep['a0_variance_shares_pct'] = {
        'seed': float(sh[:, 0].mean() * 100),
        'controller_gains': float(sh[:, 1].mean() * 100),
        'interaction': float(sh[:, 2].mean() * 100),
        'n_tasks': int(len(sh))}
    # ---- A1 branch structure ----
    a1 = np.load(A / 'a1_smm.npz')
    bshares, nbr = [], []
    for t in sorted(set(a1['task'].tolist())):
        m = a1['task'] == t
        br, pr = a1['branch'][m], a1['progress'][m]
        if len(set(br.tolist())) < 2:
            continue
        gm = pr.mean(); ss_tot = ((pr - gm) ** 2).sum()
        if ss_tot < 1e-9:
            continue
        ss_b = sum(len(pr[br == b]) * (pr[br == b].mean() - gm) ** 2
                   for b in set(br.tolist()))
        bshares.append(ss_b / ss_tot)
        nbr.append(len(set(br.tolist())))
    rep['a1_smm'] = {
        'between_branch_variance_pct': float(np.mean(bshares) * 100),
        'mean_branches_per_task': float(np.mean(nbr)),
        'n_tasks_analyzed': int(len(bshares))}
    # ---- A2 controller flexibility ----
    fixed_best = grid.mean(0).max()
    per_task_best = []
    for t in sorted(set(tidx.tolist())):
        g = grid[tidx == t]
        per_task_best.append(g.max(1).max())   # best rep + best gains
    default_i = GAIN_GRID.index(DEFAULT_GAINS) if DEFAULT_GAINS in GAIN_GRID else None
    per_task_best_gain_of_best_rep = []
    best_gain_ids = []
    for t in sorted(set(tidx.tolist())):
        g = grid[tidx == t]
        r = g.max(1).argmax()
        per_task_best_gain_of_best_rep.append(g[r].max())
        best_gain_ids.append(int(g[r].argmax()))
    rep['a2_gains'] = {
        'global_fixed_best_gain_mean_m': float(fixed_best),
        'per_task_oracle_gain_mean_m': float(np.mean(per_task_best_gain_of_best_rep)),
        'flexibility_gap_mm': float(
            (np.mean(per_task_best_gain_of_best_rep) - fixed_best) * 1e3),
        'n_distinct_optimal_gain_sets': int(len(set(best_gain_ids))),
    }
    seg = A / 'a2_segmented.npz'
    if seg.exists():
        s = np.load(seg)
        res = s['res']
        static_best = max(res[:, i, i].mean() for i in range(res.shape[1]))
        switch_best = res.reshape(len(res), -1).max(1).mean()
        rep['a2_segmented'] = {
            'best_static_mean_m': float(static_best),
            'best_switched_mean_m': float(switch_best),
            'within_trajectory_gain_gap_mm': float(
                (switch_best - static_best) * 1e3)}
    (A / 'partA_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('sample', 'a0', 'a1', 'a2seg', 'analyze'))
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    globals()[f'stage_{args.stage}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
