"""Deep-valley pipeline: detect 5 hidden-reach tasks in the 10k tail,
build a per-task ray-start curriculum pool (t27 recipe), train a 30M
specialist per task on the 8192-env infrastructure, eval canonical.

Stages print PIPE| lines; everything cached under runs/paper_fill/valley/.
"""
import matplotlib; matplotlib.use('Agg')
import sys, os, math, time, subprocess, dataclasses
from pathlib import Path
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
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
A = MAIN / 'runs/paper_fill/ratio_assets'
OUT = MAIN / 'runs/paper_fill/valley'
OUT.mkdir(exist_ok=True)
tz = np.load(A / 'tasks_pool_fr3.npz')
bnd = np.load(MAIN / 'runs/paper_fill/bound_pool_fr3.npz') if (
    MAIN / 'runs/paper_fill/bound_pool_fr3.npz').exists() else np.load(
    A / 'bound_pool_fr3.npz')
wit = np.load(A / 'witness_pool_fr3.npz')
ref = np.maximum(bnd['L_hi'], wit['prog'])
prog_main = np.load(FU / 'dirfrac_fr3rm_10k.npz')['prog']

env = lb.build_env(dev, 'stock', 512)
dt = env.kin.dtype
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)


def ik_starts(ti, s_vals, n_dirs, k_nn=48):
    """Certified mid-line starts for eval-set task ti at arc offsets."""
    p0 = tz['cs_p0'][ti].astype(np.float32)
    d = tz['cs_line_dir'][ti].astype(np.float32); d /= np.linalg.norm(d)
    nt = tz['cs_n_target'][ti].astype(np.float32); nt /= np.linalg.norm(nt)
    rng = np.random.default_rng(1000 + ti)
    dirs = np.concatenate([nt[None], _sample_in_cone(
        torch.as_tensor(nt), lb.CONE_DEG, max(n_dirs * 2, 4),
        rng).numpy()[:n_dirs - 1]], 0) if n_dirs > 1 else nt[None]
    qs, ss = [], []
    for s0 in s_vals:
        pt = (p0 + s0 * d).astype(np.float32)
        for m in range(len(dirs)):
            feat = np.concatenate([pt * POS_SCALE, dirs[m]], 0)[None]
            _, ids = tree.query(feat.astype(np.float32), k=k_nn, workers=-1)
            fq = torch.as_tensor(T['q'][ids[0]], device=dev, dtype=dt)
            fp = torch.as_tensor(pt, device=dev, dtype=dt).expand(k_nn, 3)
            fz = torch.as_tensor(dirs[m], device=dev, dtype=dt).expand(k_nn, 3)
            q_o, _, _ = _batched_ik_project(env.kin, fq, fp,
                                            _build_R_with_z(fz, hint),
                                            branch_action=None)
            coll = env.collision.is_collided(env.kin.link_transforms(q_o))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
            ntt = torch.as_tensor(nt, device=dev, dtype=dt)
            in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                      & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
            fine = ((~coll) & in_lmt
                    & ((p_fk - fp).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * ntt).sum(-1) >= cos_lim))
            k = q_o[fine].cpu().numpy()
            qs.append(np.unique(np.round(k, 2), axis=0))
            ss.append(np.full(len(qs[-1]), s0, np.float32))
    return (np.concatenate(qs) if qs else np.zeros((0, 7), np.float32),
            np.concatenate(ss) if ss else np.zeros(0, np.float32))


# ---------------- stage 1: detector -------------------------------------
DET_F = OUT / 'detector.npz'
if not DET_F.exists():
    rt = prog_main / np.maximum(ref, 1e-9)
    cand = np.argsort(rt)[:300]
    cand = cand[cand != 27]
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                            'config_line_cont_dirfrac_rh2048XXL_rm.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 2048}), None, dev)
    ag = Agent(renv.obs_dim, renv.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(
        REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL_rm/agent.pt',
        map_location=dev))
    ag.eval()
    rdt = renv.kin.dtype
    hidden = np.zeros(len(cand), np.float32)
    QS, SS, TIS = [], [], []
    for i, ti in enumerate(cand):
        smax = min(float(ref[ti]) - 0.05, 1.5)
        if smax < 0.25:
            continue
        q, s = ik_starts(int(ti), np.arange(0.15, smax, 0.15), 2)
        if len(q) == 0:
            continue
        QS.append(q); SS.append(s); TIS.append(np.full(len(q), ti))
    QS = np.concatenate(QS); SS = np.concatenate(SS)
    TIS = np.concatenate(TIS)
    print(f'PIPE| detector rollouts: {len(QS)}', flush=True)
    R = np.zeros(len(QS), np.float32)
    B = 2048
    with torch.no_grad():
        for lo in range(0, len(QS), B):
            hi = min(lo + B, len(QS))
            pad = B - (hi - lo)
            ids = TIS[lo:hi]
            sub = {'q0': torch.tensor(QS[lo:hi], dtype=rdt),
                   'p0': torch.tensor(tz['cs_p0'][ids] + SS[lo:hi, None]
                                      * tz['cs_line_dir'][ids], dtype=rdt),
                   'line_dir': torch.tensor(tz['cs_line_dir'][ids], dtype=rdt),
                   'n_target': torch.tensor(tz['cs_n_target'][ids], dtype=rdt)}
            if pad:
                sub = {k2: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                       for k2, v in sub.items()}
            sub = {k2: v.to(dev) for k2, v in sub.items()}
            renv.line_dist = ScriptedLineDistribution(sub)
            renv.reset()
            for _ in range(renv.cfg.max_steps // 2):
                a = ag.actor_mean(renv.current_obs())
                for _ in range(2):
                    renv.step(a, auto_reset=False)
                if bool(renv.done_persistent.all()):
                    break
            R[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
    reach = SS + R
    hid = {}
    for ti in np.unique(TIS):
        m = TIS == ti
        hid[int(ti)] = float(reach[m].max()) - float(prog_main[ti])
    np.savez(DET_F, tis=np.array(list(hid.keys())),
             hidden=np.array(list(hid.values())),
             canon=prog_main[np.array(list(hid.keys()))],
             ref=ref[np.array(list(hid.keys()))])
    del renv, ag; torch.cuda.empty_cache()
d = np.load(DET_F)
order = np.argsort(-d['hidden'])
sel = [int(d['tis'][i]) for i in order[:5]]
print('PIPE| selected', [(t, round(float(d['hidden'][order[k]]), 2))
                         for k, t in enumerate(sel)], flush=True)

# ---------------- stage 2+3: per-task pool + train + eval ---------------
summary = []
for ti in sel:
    tag = f't{ti}'
    pool_f = OUT / f'{tag}_pool.npz'
    if not pool_f.exists():
        smax = min(float(ref[ti]) + 0.1, 1.7)
        q, s = ik_starts(ti, np.arange(0.0, smax, 0.02), 8, k_nn=96)
        # canonical neighbourhood via certified null walk
        p0 = tz['cs_p0'][ti].astype(np.float32)
        nt = tz['cs_n_target'][ti].astype(np.float32); nt /= np.linalg.norm(nt)
        qc = torch.tensor(tz['q0_seed'][ti], dtype=dt, device=dev)[None]
        frontier = qc.expand(64, 7).clone()
        keep = [qc.cpu().numpy()[0][None]]
        rng = np.random.default_rng(ti)
        for _ in range(10):
            _, _, J, _ = env.kin.tcp_fk_jac(frontier)
            Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
            PN = torch.eye(7, device=dev, dtype=dt)[None] - Jp @ J[:, :3, :]
            g = torch.as_tensor(rng.standard_normal((64, 7)), device=dev, dtype=dt)
            u = (PN @ g.unsqueeze(-1)).squeeze(-1)
            u = u / (u.norm(dim=-1, keepdim=True) + 1e-9)
            q2 = frontier + 0.12 * u
            p0t = torch.as_tensor(p0, device=dev, dtype=dt).expand(64, 3)
            for _ in range(2):
                p_fk, _, J2, _ = env.kin.tcp_fk_jac(q2)
                Jp2, _ = damped_pinv(J2[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
                q2 = q2 + (Jp2 @ (p0t - p_fk).unsqueeze(-1)).squeeze(-1)
            coll = env.collision.is_collided(env.kin.link_transforms(q2))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q2)
            ntt = torch.as_tensor(nt, device=dev, dtype=dt)
            fine = ((~coll)
                    & ((q2 >= env.kin.lmt_lo - 1e-5) & (q2 <= env.kin.lmt_up + 1e-5)).all(-1)
                    & ((p_fk - p0t).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * ntt).sum(-1) >= cos_lim))
            q2 = q2[fine]
            if len(q2):
                keep.append(q2.cpu().numpy())
                frontier = q2[torch.randint(0, len(q2), (64,), device=dev)]
        canon = np.unique(np.round(np.concatenate(keep), 3), axis=0)
        Q = np.concatenate([q, canon]).astype(np.float32)
        S = np.concatenate([s, np.zeros(len(canon))]).astype(np.float32)
        dvec = tz['cs_line_dir'][ti] / np.linalg.norm(tz['cs_line_dir'][ti])
        P = (tz['cs_p0'][ti][None] + S[:, None] * dvec[None]).astype(np.float32)
        n_ray = len(q)
        near = (S < 0.25) & (np.arange(len(S)) < n_ray)
        can_m = np.arange(len(S)) >= n_ray
        rest = ~near & ~can_m
        w = np.zeros(len(S))
        if near.any(): w[near] = 0.5 / near.sum()
        if can_m.any(): w[can_m] = 0.2 / can_m.sum()
        if rest.any(): w[rest] = 0.3 / rest.sum()
        np.savez_compressed(pool_f, q=Q, p0=P, s=S, weight=w,
                            line_dir=dvec.astype(np.float32),
                            n_target=(tz['cs_n_target'][ti]
                                      / np.linalg.norm(tz['cs_n_target'][ti])
                                      ).astype(np.float32))
        print(f'PIPE| {tag} pool {len(Q)} starts ({len(canon)} canonical-nbhd)',
              flush=True)
    # config + train
    ckpt_dir = MAIN / f'runs/rl_valley_{tag}'
    if not (ckpt_dir / 'agent.pt').exists() or True:
        pass
    cfgf = REPO / f'Yuan/IJRR/stage2_traj/config_valley_{tag}.yaml'
    b = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                            'config_line_cont_dirfrac_e8k_b.yaml'))
    b['line_distribution'] = {'ray_start_npz': str(pool_f), 'train_seed': 0}
    yaml.safe_dump(b, open(cfgf, 'w'), sort_keys=False)
    log_f = MAIN / f'runs/rl_valley_{tag}.launch.log'
    if not (str(ckpt_dir / 'agent.pt') and (ckpt_dir / 'agent.pt').exists()
            and '[train] done' in open(log_f).read() if log_f.exists() else False):
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, '-m', 'Yuan.IJRR.stage2_traj.train',
             '--config', str(cfgf), '--out-dir', str(ckpt_dir)],
            cwd=str(REPO), capture_output=True, text=True,
            env={**os.environ, 'MKL_THREADING_LAYER': 'GNU'})
        open(log_f, 'w').write(r.stdout[-5000:] + r.stderr[-5000:])
        print(f'PIPE| {tag} trained ({(time.time()-t0)/60:.0f} min)', flush=True)
    # canonical eval
    y = yaml.safe_load(open(cfgf))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
    ag = Agent(renv.obs_dim, renv.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(ckpt_dir / 'agent.pt', map_location=dev))
    ag.eval()
    rdt = renv.kin.dtype
    renv.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(tz['q0_seed'][ti][None], dtype=rdt, device=dev),
         'p0': torch.tensor(tz['cs_p0'][ti][None], dtype=rdt, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][ti][None], dtype=rdt, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][ti][None], dtype=rdt, device=dev)})
    renv.reset()
    smax_true = 0.0
    dvec = tz['cs_line_dir'][ti] / np.linalg.norm(tz['cs_line_dir'][ti])
    with torch.no_grad():
        for _ in range(renv.cfg.max_steps // 2):
            a = ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            p, _, _, _ = renv.kin.tcp_fk_jac(renv.q)
            smax_true = max(smax_true, float(
                ((p[0].cpu().numpy() - tz['cs_p0'][ti]) * dvec).sum()))
            if bool(renv.done_persistent.all()):
                break
    row = dict(task=ti, before=float(prog_main[ti]), after=smax_true,
               hidden=float(d['hidden'][list(d['tis']).index(ti)]),
               ref=float(ref[ti]))
    summary.append(row)
    print(f'PIPE| RESULT {tag}: before {row["before"]:.3f} -> after '
          f'{row["after"]:.3f} (hidden was +{row["hidden"]:.2f}, '
          f'lpw {row["ref"]:.2f})', flush=True)
    del renv, ag; torch.cuda.empty_cache()

import json
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
print('PIPE| ALL DONE', flush=True)
