"""Within-component value symmetry test (user-designed, zero training).

Question: within the same feasible self-motion connected component at s=0,
is the performance difference between start configurations mainly due to
the policy never performing zero-space reconfiguration?

Exp 1  For screened tasks, collect a dense admissible IK cloud at s=0,
       build an eps-connectivity graph, attach the K=32 candidates to
       components, and decompose the fresh rollout-label variance into
       within- vs across-component parts.
Exp 2  For 5-10 same-component pairs (A worse, B better, |dL| large),
       execute a v=0 null-space migration A->B along the certified graph
       path (every substep: joint limits, collision, 2 cm tube, 30 deg
       cone; no task progress, no reward), then roll the SAME deterministic
       mainline policy from the arrival state under the exact protocol of
       condition B. Report L_A, L_B, L_A->B and the recovery ratio
       (L_A->B - L_A) / (L_B - L_A).

No training, no shaping, no architecture change. Env/ckpt via env vars:
  CKPT=.../agent.pt  CFG=.../config.yaml  (default = the _rm retrain)
  SMOKE=1 limits to 3 tasks / 2 pairs.
"""
import os, sys, math, json, time, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, damped_pinv,
                               LATERAL_SAFETY_NET)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
SMOKE = bool(int(os.environ.get('SMOKE', '0')))
CFG = os.environ.get('CFG', str(
    REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_rh2048XXL_rm.yaml'))
CKPT = os.environ.get('CKPT', str(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL_rm/agent.pt'))
EPS = float(os.environ.get('EPS', '0.4'))       # rad, connectivity radius
SUBSTEP = 0.04                                   # rad, migration substep
DL_MIN = 0.15                                    # m, min pair gap
N_TASKS = 3 if SMOKE else 40
N_PAIRS = 2 if SMOKE else 10
K = 32

# ---------------- envs, agent, warm-start table --------------------------
env = lb.build_env(dev, 'stock', 512)            # admissibility machinery
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
dt = env.kin.dtype
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)

y = yaml.safe_load(open(CFG))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2500
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(CKPT, map_location=dev))
ag.eval()
rdt = renv.kin.dtype
print(f'[ms] ckpt {CKPT}', flush=True)

tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)['benchmark']
L_old = np.load(FU / 'dirfrac_labels_benchmark.npz')['L']
NF = cands['n_found'].numpy()[:L_old.shape[0]]
CC = cands['cands'][:L_old.shape[0]].numpy()

# ---------------- screening: tasks with large candidate spread -----------
Mv = np.arange(K)[None, :] < NF[:, None]
spread = np.where(Mv, L_old, np.nan)
spread = np.nanmax(spread, 1) - np.nanmin(spread, 1)
order = np.argsort(-np.nan_to_num(spread))
sel_tasks = [i for i in order if NF[i] >= 8][:N_TASKS]
print(f'[ms] screened {len(sel_tasks)} tasks, '
      f'old-label spread {spread[sel_tasks[0]]:.2f}..'
      f'{spread[sel_tasks[-1]]:.2f} m', flush=True)


def rollout_batch(q0s, task_ids):
    """Deterministic mainline rollouts; identical protocol for every
    condition (scripted reset -> actor_mean until done)."""
    n = len(q0s)
    out = np.zeros(n, np.float32)
    for lo in range(0, n, B):
        hi = min(lo + B, n)
        pad = B - (hi - lo)
        idx = task_ids[lo:hi]
        sub = {'q0': torch.tensor(np.stack(q0s[lo:hi]), dtype=rdt),
               'p0': torch.tensor(tb['cs_p0'][idx], dtype=rdt),
               'line_dir': torch.tensor(tb['cs_line_dir'][idx], dtype=rdt),
               'n_target': torch.tensor(tb['cs_n_target'][idx], dtype=rdt)}
        if pad:
            sub = {k2: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                   for k2, v in sub.items()}
        sub = {k2: v.to(dev) for k2, v in sub.items()}
        renv.line_dist = ScriptedLineDistribution(sub)
        renv.reset()
        with torch.no_grad():
            for _ in range(renv.cfg.max_steps // 2):
                a = ag.actor_mean(renv.current_obs())
                for _ in range(2):
                    renv.step(a, auto_reset=False)
                if bool(renv.done_persistent.all()):
                    break
        out[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


def admissible(q, p0, nt):
    """Full constraint check of configurations q against the task ray
    anchor p0 / cone axis nt. Returns bool mask (+ margins for logging)."""
    p0t = torch.as_tensor(p0, device=dev, dtype=dt).expand(q.shape[0], 3)
    ntt = torch.as_tensor(nt, device=dev, dtype=dt).expand(q.shape[0], 3)
    coll = env.collision.is_collided(env.kin.link_transforms(q))
    p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q)
    in_lmt = ((q >= env.kin.lmt_lo - 1e-5)
              & (q <= env.kin.lmt_up + 1e-5)).all(dim=-1)
    tip_err = (p_fk - p0t).norm(dim=-1)
    cosd = (R_fk[:, :, 2] * ntt).sum(-1)
    fine = (~coll) & in_lmt & (tip_err <= tube) & (cosd >= cos_lim)
    return fine, tip_err, cosd


def _hold_tip(q, p0t, iters=2):
    for _ in range(iters):
        p_fk, _, J, _ = env.kin.tcp_fk_jac(q)
        Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                            env.cfg.sigma_thr)
        q = q + (Jp @ (p0t - p_fk).unsqueeze(-1)).squeeze(-1)
    return q


def expand_cloud(Q, p0, nt, rng, n_dirs=2, step=0.12):
    """Grow the cloud ALONG the manifold: random null-space perturbations
    of every point, tip-held and constraint-checked. Bridges the sampling
    gaps that fragment the eps graph."""
    qt = torch.as_tensor(Q, device=dev, dtype=dt)
    p0t = torch.as_tensor(p0, device=dev, dtype=dt).expand(len(Q), 3)
    _, _, J, _ = env.kin.tcp_fk_jac(qt)
    Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
    PN = (torch.eye(qt.shape[1], device=dev, dtype=dt)[None]
          - Jp @ J[:, :3, :])
    new = []
    for _ in range(n_dirs):
        g = torch.as_tensor(rng.standard_normal(qt.shape), device=dev,
                            dtype=dt)
        d = (PN @ g.unsqueeze(-1)).squeeze(-1)
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-9)
        for sgn in (1.0, -1.0):
            q2 = _hold_tip(qt + sgn * step * d, p0t)
            fine, _, _ = admissible(q2, p0, nt)
            new.append(q2[fine].cpu().numpy())
    return np.concatenate(new, 0)


def collect_cloud(ti, m_dirs=32, k_nn=400, n_expand=2):
    """Dense admissible IK cloud at the task start (candidates included)."""
    p0 = tb['cs_p0'][ti].astype(np.float32)
    nt = tb['cs_n_target'][ti].astype(np.float32)
    nt = nt / np.linalg.norm(nt)
    pool = _sample_in_cone(torch.as_tensor(nt), lb.CONE_DEG, 2 * m_dirs,
                           np.random.default_rng(97 + ti)).numpy()
    dirs = np.concatenate([nt[None], pool[:m_dirs - 1]], 0)
    sols = []
    for m in range(len(dirs)):
        feat = np.concatenate([p0 * POS_SCALE, dirs[m]], 0)[None]
        _, ids = tree.query(feat.astype(np.float32), k=k_nn, workers=-1)
        fq = torch.as_tensor(T['q'][ids[0]], device=dev, dtype=dt)
        fp = torch.as_tensor(p0, device=dev, dtype=dt).expand(k_nn, 3)
        fz = torch.as_tensor(dirs[m], device=dev, dtype=dt).expand(k_nn, 3)
        q_o, _, _ = _batched_ik_project(env.kin, fq, fp,
                                        _build_R_with_z(fz, hint),
                                        branch_action=None)
        fine, _, _ = admissible(q_o, p0, nt)
        sols.append(q_o[fine].cpu().numpy())
    Q = np.unique(np.round(np.concatenate(sols, 0), 3), axis=0)
    rng = np.random.default_rng(1234 + ti)
    for _ in range(n_expand):
        Q = np.unique(np.round(np.concatenate(
            [Q, expand_cloud(Q, p0, nt, rng)], 0), 3), axis=0)
        if len(Q) > 12000:
            Q = Q[rng.choice(len(Q), 12000, replace=False)]
    # candidates join the cloud as first nf nodes
    cq = CC[ti, :NF[ti]]
    Q = np.concatenate([cq, Q], 0)
    return Q, p0, nt


def components(Q):
    kt = cKDTree(Q)
    prs = kt.query_pairs(EPS, output_type='ndarray')
    n = len(Q)
    g = coo_matrix((np.ones(len(prs)), (prs[:, 0], prs[:, 1])),
                   shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    return nc, lab, g


def migrate(Q, g, a_i, b_i, p0, nt):
    """Certified v=0 null-space migration along the graph path a->b.
    Every substep is projected back to the tip anchor and checked against
    the full constraint set. Returns (ok, q_arrival, n_steps, log)."""
    n = len(Q)
    w = coo_matrix((np.linalg.norm(Q[g.row] - Q[g.col], axis=1),
                    (g.row, g.col)), shape=(n, n))
    dist, pred = dijkstra(w, directed=False, indices=a_i,
                          return_predecessors=True)
    if not np.isfinite(dist[b_i]):
        return False, None, 0, 'no graph path'
    path = [b_i]
    while path[-1] != a_i:
        path.append(pred[path[-1]])
    path = path[::-1]
    q = torch.as_tensor(Q[a_i], device=dev, dtype=dt)[None]
    p0t = torch.as_tensor(p0, device=dev, dtype=dt)[None]
    n_sub = 0
    for node in path[1:]:
        tgt = torch.as_tensor(Q[node], device=dev, dtype=dt)[None]
        edge_cap = int(float((tgt - q).norm()) / SUBSTEP) * 3 + 12
        for _ in range(edge_cap):
            d = tgt - q
            nrm = float(d.norm())
            if nrm < 0.03:
                break
            q = q + d * min(1.0, SUBSTEP / nrm)
            q = _hold_tip(q, p0t)
            fine, tip_err, cosd = admissible(q, p0, nt)
            n_sub += 1
            if not bool(fine.item()):
                return False, None, n_sub, (
                    f'violated at substep {n_sub} '
                    f'(tip {float(tip_err):.4f}, cos {float(cosd):.3f})')
            if n_sub > 20000:
                return False, None, n_sub, 'substep budget'
        else:
            return False, None, n_sub, 'edge stalled (projection fights)'
    # settle exactly onto B
    q = _hold_tip(torch.as_tensor(Q[b_i], device=dev, dtype=dt)[None], p0t)
    fine, _, _ = admissible(q, p0, nt)
    if not bool(fine.item()):
        return False, None, n_sub, 'final settle inadmissible'
    arr = q[0].cpu().numpy()
    return True, arr, n_sub, f'{len(path)} nodes'


# ---------------- fresh labels under the chosen policy -------------------
flat_q, flat_t = [], []
for ti in sel_tasks:
    for k in range(NF[ti]):
        flat_q.append(CC[ti, k])
        flat_t.append(ti)
Lf = rollout_batch(flat_q, np.array(flat_t))
FRESH = {}
ptr = 0
for ti in sel_tasks:
    FRESH[ti] = Lf[ptr:ptr + NF[ti]]
    ptr += NF[ti]
print(f'[ms] fresh labels for {len(Lf)} candidates done', flush=True)

# ---------------- per-task: cloud, components, exp1 stats, pair ----------
exp1, pairs = [], []
for ti in sel_tasks:
    if len(pairs) >= N_PAIRS:
        break
    t0 = time.time()
    Q, p0, nt = collect_cloud(ti)
    nc, lab, g = components(Q)
    cl = lab[:NF[ti]]                      # candidate component labels
    Lc = FRESH[ti]
    # variance decomposition over valid candidates
    tot = float(np.var(Lc))
    comp_means = {c: Lc[cl == c].mean() for c in np.unique(cl)}
    within = float(np.mean([np.var(Lc[cl == c]) for c in np.unique(cl)]
                           if len(np.unique(cl)) else [0.0]))
    across = float(np.var([comp_means[c] for c in cl]))
    exp1.append(dict(task=int(ti), n_cloud=len(Q), n_comp_cloud=int(nc),
                     n_comp_cand=len(np.unique(cl)), var_total=tot,
                     var_within=within, var_across=across))
    print(f'[ms] task {ti}: cloud {len(Q)} ({nc} comps), '
          f'cands in {len(np.unique(cl))} comps, var w/a/t '
          f'{within:.4f}/{across:.4f}/{tot:.4f} ({time.time()-t0:.0f}s)',
          flush=True)
    # exp2 pair: largest fresh-label gap within one component
    best = None
    for c in np.unique(cl):
        idx = np.nonzero(cl == c)[0]
        if len(idx) < 2:
            continue
        a_k, b_k = idx[Lc[idx].argmin()], idx[Lc[idx].argmax()]
        gap = Lc[b_k] - Lc[a_k]
        if gap >= DL_MIN and (best is None or gap > best[0]):
            best = (gap, a_k, b_k)
    if best is None:
        continue
    gap, a_k, b_k = best
    ok, arr, n_sub, note = migrate(Q, g, a_k, b_k, p0, nt)
    if not ok:
        pairs.append(dict(task=int(ti), gap=float(gap), status=note))
        print(f'[ms] task {ti}: migration FAILED ({note})', flush=True)
        continue
    la, lb_, larr = rollout_batch(
        [CC[ti, a_k], CC[ti, b_k], arr], np.array([ti, ti, ti]))
    ratio = float((larr - la) / (lb_ - la)) if abs(lb_ - la) > 1e-6 else 1.0
    pairs.append(dict(
        task=int(ti), status='ok', L_A=float(la), L_B=float(lb_),
        L_AB=float(larr), ratio=ratio,
        arr_dist=float(np.abs(arr - CC[ti, b_k]).max()),
        n_sub=int(n_sub), dur_s=round(n_sub * float(kw['dt']), 1),
        note=note))
    print(f'[ms] task {ti}: L_A {la:.3f}  L_B {lb_:.3f}  '
          f'L_A->B {larr:.3f}  ratio {ratio:.2f}  '
          f'({n_sub} substeps = {n_sub * float(kw["dt"]):.0f}s hold, '
          f'arr-dist {np.abs(arr - CC[ti, b_k]).max():.3f})', flush=True)

# ---------------- summary ------------------------------------------------
okp = [p for p in pairs if p.get('status') == 'ok']
rs = np.array([p['ratio'] for p in okp])
w_share = np.array([e['var_within'] / max(e['var_total'], 1e-9)
                    for e in exp1])
print('\n===== Exp 1: variance decomposition over',
      len(exp1), 'tasks =====', flush=True)
print(f'within-component share of label variance: '
      f'mean {w_share.mean():.2f}  median {np.median(w_share):.2f}',
      flush=True)
print(f'tasks whose candidates span >1 component: '
      f'{sum(e["n_comp_cand"] > 1 for e in exp1)}/{len(exp1)}', flush=True)
print('\n===== Exp 2:', len(okp), 'certified pairs,',
      len(pairs) - len(okp), 'failed =====', flush=True)
if len(okp):
    print(f'recovery ratio mean {rs.mean():.2f}  median '
          f'{np.median(rs):.2f}  min {rs.min():.2f}  max {rs.max():.2f}',
          flush=True)
json.dump({'exp1': exp1, 'pairs': pairs},
          open(FU / 'manifold_symmetry.json', 'w'), indent=1)
print('saved -> manifold_symmetry.json', flush=True)
