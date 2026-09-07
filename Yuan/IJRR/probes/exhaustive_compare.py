"""Three-way per-task comparison: classical gradient vs DirFrac flagship
vs high-budget layered search (K_CAP 512, 12 kicks, 3 restarts, backtracked
joint path). argv: task ids...  Saves per-task npz + prints strokes."""
import sys, math, time, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET,
                               damped_pinv)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent

TASKS = [int(x) for x in sys.argv[1:]]
DS, V = 0.02, 0.2
K_CAP, N_PERT, PERT_MAX, N_SUB, RESTARTS = 1024, 24, 0.18, 3, 5
QD_LIM = np.array([2.175]*4 + [2.61]*3, np.float32)
DQ_BOX = QD_LIM * DS / V
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
OUT = MAIN/'runs/paper_fill/search_compare'; OUT.mkdir(exist_ok=True)
env = lb.build_env(dev, 'stock', 512)
dt = env.kin.dtype
tz = np.load(A/'tasks_pool_fr3.npz')
b = np.load(A/'bound_pool_fr3.npz'); w = np.load(A/'witness_pool_fr3.npz')
ref_all = np.maximum(b['L_hi'], w['prog'])
base = np.load(FU/'pool_fr3_straight.npz')
for k in base.files:
    if k.endswith('_progress'):
        ref_all = np.maximum(ref_all, base[k])
cos_lim = math.cos(math.radians(lb.CONE_DEG)); tube = LATERAL_SAFETY_NET


def admissible(q, pt, ntt):
    coll = env.collision.is_collided(env.kin.link_transforms(q))
    p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q)
    in_lmt = ((q >= env.kin.lmt_lo - 1e-5)
              & (q <= env.kin.lmt_up + 1e-5)).all(dim=-1)
    return ((~coll) & in_lmt & ((p_fk - pt).norm(dim=-1) <= tube)
            & ((R_fk[:, :, 2] * ntt).sum(-1) >= cos_lim))


def advance(q, ptt):
    for _ in range(3):
        p_fk, _, J, _ = env.kin.tcp_fk_jac(q)
        Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
        q = q + (Jp @ (ptt - p_fk).unsqueeze(-1)).squeeze(-1)
    return q


def search_one(ti, seed):
    p0 = tz['cs_p0'][ti].astype(np.float32)
    d = tz['cs_line_dir'][ti].astype(np.float32); d /= np.linalg.norm(d)
    nt = tz['cs_n_target'][ti].astype(np.float32); nt /= np.linalg.norm(nt)
    ntt = torch.as_tensor(nt, device=dev, dtype=dt)
    s_cap = min(float(ref_all[ti]) + 0.06, 1.80)
    rng_l = np.random.default_rng(seed)
    R = tz['q0_seed'][ti].astype(np.float32)[None]
    layers, parents = [R], [np.array([-1])]
    s = 0.0
    while s + DS <= s_cap:
        pt = (p0 + (s + DS) * d).astype(np.float32)
        ptt = torch.as_tensor(pt, device=dev, dtype=dt)
        qr = torch.as_tensor(R, device=dev, dtype=dt)
        B = len(qr)
        _, _, J, _ = env.kin.tcp_fk_jac(qr)
        Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
        PN = torch.eye(7, device=dev, dtype=dt)[None] - Jp @ J[:, :3, :]
        g = torch.as_tensor(rng_l.standard_normal((N_PERT, B, 7)),
                            device=dev, dtype=dt)
        u = (PN[None] @ g.unsqueeze(-1)).squeeze(-1)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        mag = torch.as_tensor(rng_l.uniform(0, PERT_MAX, (N_PERT, B, 1)),
                              device=dev, dtype=dt)
        q_all = advance(torch.cat([qr, (qr[None] + mag*u).reshape(-1, 7)]), ptt)
        fine = admissible(q_all, ptt, ntt)
        if not bool(fine.any()):
            break
        Q = q_all[fine].cpu().numpy().astype(np.float32)
        _, ui = np.unique(np.round(Q, 3), axis=0, return_index=True)
        Q = Q[np.sort(ui)]
        if len(Q) > K_CAP:
            pick = [0]; dmin = np.linalg.norm(Q - Q[0], axis=1)
            for _ in range(K_CAP - 1):
                j = int(dmin.argmax()); pick.append(j)
                dmin = np.minimum(dmin, np.linalg.norm(Q - Q[j], axis=1))
            Q = Q[pick]
        # certified edges + first certified parent per candidate
        dq = np.abs(Q[None] - R[:, None]); box = (dq <= DQ_BOX).all(-1)
        ii, jj = np.nonzero(box)
        if len(ii) == 0:
            break
        ok = np.ones(len(ii), bool)
        qa = torch.as_tensor(R[ii], device=dev, dtype=dt)
        qb = torch.as_tensor(Q[jj], device=dev, dtype=dt)
        pa = p0 + s * d
        for k in range(1, N_SUB + 1):
            tau = k / (N_SUB + 1)
            qm = qa + tau * (qb - qa)
            pm = torch.as_tensor((pa + tau*DS*d).astype(np.float32),
                                 device=dev, dtype=dt)
            for lo in range(0, len(qm), 8192):
                ok[lo:lo+8192] &= admissible(qm[lo:lo+8192], pm, ntt
                                             ).cpu().numpy()
        par = np.full(len(Q), -1)
        for e in np.nonzero(ok)[0]:
            if par[jj[e]] < 0:
                par[jj[e]] = ii[e]
        keep = par >= 0
        if not keep.any():
            break
        R = Q[keep]
        parents.append(par[keep]); layers.append(R)
        s += DS
    # backtrack from node 0 of the last layer
    path = []
    j = 0
    for k in range(len(layers) - 1, -1, -1):
        path.append(layers[k][j])
        j = parents[k][j] if k > 0 else 0
    return s, np.array(path[::-1], np.float32)


@torch.no_grad()
def roll(cfgfile, ckpt, tis, classical=False):
    y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj'/cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': len(tis)}), None, dev)
    rdt = renv.kin.dtype
    if classical:
        fn = cn_action_fn(ClassicalNullspaceController(renv.kin))
    else:
        ag = Agent(renv.obs_dim, renv.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO/ckpt, map_location=dev))
        ag.eval()
    renv.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(tz['q0_seed'][tis], dtype=rdt, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][tis], dtype=rdt, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][tis], dtype=rdt, device=dev)})
    renv.reset()
    QT, ST = [renv.q.cpu().numpy().copy()], [np.zeros(len(tis), np.float32)]
    for _ in range(renv.cfg.max_steps // 2):
        a = fn(renv) if classical else ag.actor_mean(renv.current_obs())
        for _ in range(2):
            renv.step(a, auto_reset=False)
            QT.append(renv.q.cpu().numpy().copy())
            ST.append(renv.arc_progress.float().cpu().numpy().copy())
        if bool(renv.done_persistent.all()):
            break
    del renv
    torch.cuda.empty_cache()
    return np.array(QT), np.array(ST)


tis = np.array(TASKS)
Qc, Sc = roll(str(Path(hl.ROBOTS['fr3'][0]).name), None, tis, classical=True)
Qr, Sr = roll('config_line_cont_dirfrac_e8kXXL_rm.yaml',
              'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', tis)
print('classical strokes:', np.round(Sc[-1], 3).tolist())
print('cache  classical :', np.round(base['classical_progress'][tis], 3).tolist())
print('rl strokes       :', np.round(Sr[-1], 3).tolist())
print('cache  rl        :', np.round(np.load(FU/'e8kXXL_10k.npz')['prog'][tis], 3).tolist())
t0 = time.time()
for ti in TASKS:
    best_s, best_path = -1.0, None
    for r in range(RESTARTS):
        s, path = search_one(ti, 9000 + 97*r + ti)
        if s > best_s:
            best_s, best_path = s, path
    i = TASKS.index(ti)
    np.savez_compressed(OUT/f't{ti}_three_way.npz',
                        cls_q=Qc[:, i], cls_s=Sc[:, i],
                        rl_q=Qr[:, i], rl_s=Sr[:, i],
                        search_q=best_path,
                        search_s=np.arange(len(best_path))*DS,
                        ref=ref_all[ti])
    print(f't{ti}: classical {Sc[-1, i]:.2f}  rl {Sr[-1, i]:.2f}  '
          f'search {best_s:.2f}  lpw {ref_all[ti]:.2f} '
          f'({(time.time()-t0)/60:.1f} min)', flush=True)
print('ALL DONE', flush=True)
