"""Dense candidate pools + zero-step critic selection on the full 10k.

Per task: cone-IK at s=0 (8 dirs x 96 warm starts, full admissibility,
dedup) -> critic scores every candidate at its reset observation ->
roll ONLY the picked one with the flagship policy. A 500-task subsample
additionally rolls ALL candidates for the pool-oracle and capture rate.
Zero training anywhere.
"""
import matplotlib; matplotlib.use('Agg')
import sys, math, time, dataclasses
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import (NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/'
MAIN = '/home/lqin/one/Yuan/IJRR/'
A = MAIN + 'runs/paper_fill/ratio_assets/'
FU = MAIN + 'runs/paper_fill/fam_unify/'
OUT = MAIN + 'runs/paper_fill/valley/'
dev = torch.device('cuda')
tz = np.load(A + 'tasks_pool_fr3.npz')
N = len(tz['cs_p0'])

env = lb.build_env(dev, 'stock', 512)
dt = env.kin.dtype
T = np.load(REPO + lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)

# ---------------- stage 1: dense s=0 pools for all 10k ------------------
import os
POOL_F = OUT + 'dense_s0_pools.npz'
if not os.path.exists(POOL_F):
    p0 = tz['cs_p0'].astype(np.float32)
    nt = tz['cs_n_target'].astype(np.float32)
    nt = nt / np.linalg.norm(nt, axis=1, keepdims=True)
    M_DIRS, K_NN = 8, 96
    all_q, all_tid = [], []
    rng = np.random.default_rng(0)
    t0 = time.time()
    for m in range(M_DIRS):
        if m == 0:
            zs = nt.copy()
        else:
            zs = np.stack([_sample_in_cone(torch.as_tensor(nt[i]),
                                           lb.CONE_DEG, 1,
                                           np.random.default_rng(
                                               m * 100000 + i)).numpy()[0]
                           for i in range(N)])
        feat = np.concatenate([p0 * POS_SCALE, zs], 1).astype(np.float32)
        _, ids = tree.query(feat, k=K_NN, workers=-1)
        CH = 96          # tasks per chunk -> 96*96 = 9216 rows per batch
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, 7),
                                 device=dev, dtype=dt)
            fp = torch.as_tensor(np.repeat(p0[lo:hi], K_NN, 0), device=dev, dtype=dt)
            fz = torch.as_tensor(np.repeat(zs[lo:hi], K_NN, 0), device=dev, dtype=dt)
            fn = torch.as_tensor(np.repeat(nt[lo:hi], K_NN, 0), device=dev, dtype=dt)
            q_o, _, _ = _batched_ik_project(env.kin, fq, fp,
                                            _build_R_with_z(fz, hint),
                                            branch_action=None)
            coll = env.collision.is_collided(env.kin.link_transforms(q_o))
            p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
            in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                      & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
            fine = ((~coll) & in_lmt
                    & ((p_fk - fp).norm(dim=-1) <= tube)
                    & ((R_fk[:, :, 2] * fn).sum(-1) >= cos_lim))
            f = fine.cpu().numpy()
            rows = (np.arange(len(f)) // K_NN) + lo
            all_q.append(q_o[fine].cpu().numpy())
            all_tid.append(rows[f])
        print(f'PIPE| dense dir {m+1}/{M_DIRS} '
              f'({sum(len(x) for x in all_q)} sols, {time.time()-t0:.0f}s)',
              flush=True)
    Q = np.concatenate(all_q).astype(np.float32)
    TID = np.concatenate(all_tid)
    order = np.argsort(TID, kind='stable')
    Q, TID = Q[order], TID[order]
    # dedup within task
    key = np.concatenate([TID[:, None], np.round(Q, 2)], 1)
    _, ui = np.unique(key, axis=0, return_index=True)
    Q, TID = Q[np.sort(ui)], TID[np.sort(ui)]
    np.savez_compressed(POOL_F, q=Q, tid=TID)
    cnt = np.bincount(TID, minlength=N)
    print(f'PIPE| dense pools: {len(Q)} candidates, per-task median '
          f'{int(np.median(cnt))}, zero-cand tasks {(cnt==0).sum()}',
          flush=True)
d = np.load(POOL_F)
Q, TID = d['q'], d['tid']
cnt = np.bincount(TID, minlength=N)
del env
torch.cuda.empty_cache()

# ---------------- stage 2: critic scores + pick --------------------------
y = yaml.safe_load(open(REPO + 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 4096
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO + 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
rdt = renv.kin.dtype
SC_F = OUT + 'dense_s0_scores.npy'
if not os.path.exists(SC_F):
    Vsc = np.zeros(len(Q), np.float32)
    t0 = time.time()
    for lo in range(0, len(Q), B):
        hi = min(lo + B, len(Q))
        pad = B - (hi - lo)
        ids = TID[lo:hi]
        sub = {'q0': torch.tensor(Q[lo:hi], dtype=rdt),
               'p0': torch.tensor(tz['cs_p0'][ids], dtype=rdt),
               'line_dir': torch.tensor(tz['cs_line_dir'][ids], dtype=rdt),
               'n_target': torch.tensor(tz['cs_n_target'][ids], dtype=rdt)}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                   for k, v in sub.items()}
        sub = {k: v.to(dev) for k, v in sub.items()}
        renv.line_dist = ScriptedLineDistribution(sub)
        renv.reset()
        with torch.no_grad():
            Vsc[lo:hi] = ag.get_value(
                renv.current_obs()).float().cpu().numpy()[:hi - lo]
        if lo % (B * 50) == 0:
            print(f'PIPE| scored {hi}/{len(Q)} ({time.time()-t0:.0f}s)',
                  flush=True)
    np.save(SC_F, Vsc)
Vsc = np.load(SC_F)

# per-task pick (fallback canonical when no candidates)
pick_q = tz['q0_seed'].astype(np.float32).copy()
have = np.zeros(N, bool)
lo = 0
while lo < len(TID):
    hi = lo
    t = TID[lo]
    while hi < len(TID) and TID[hi] == t:
        hi += 1
    j = lo + int(np.argmax(Vsc[lo:hi]))
    pick_q[t] = Q[j]
    have[t] = True
    lo = hi
print(f'PIPE| picks ready ({int(have.sum())} tasks with candidates)', flush=True)

# ---------------- stage 3: roll the picked start per task ---------------
prog = np.zeros(N, np.float32)
t0 = time.time()
for lo in range(0, N, B):
    hi = min(lo + B, N)
    pad = B - (hi - lo)
    sub = {'q0': torch.tensor(pick_q[lo:hi], dtype=rdt),
           'p0': torch.tensor(tz['cs_p0'][lo:hi], dtype=rdt),
           'line_dir': torch.tensor(tz['cs_line_dir'][lo:hi], dtype=rdt),
           'n_target': torch.tensor(tz['cs_n_target'][lo:hi], dtype=rdt)}
    if pad:
        sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
               for k, v in sub.items()}
    sub = {k: v.to(dev) for k, v in sub.items()}
    renv.line_dist = ScriptedLineDistribution(sub)
    renv.reset()
    with torch.no_grad():
        for _ in range(renv.cfg.max_steps // 2):
            a = ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            if bool(renv.done_persistent.all()):
                break
    prog[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
    print(f'PIPE| rolled picks {hi}/{N} ({(time.time()-t0)/60:.1f} min)',
          flush=True)
np.savez(FU + 'dense_sel_10k.npz', prog=prog, have=have,
         n_cand=cnt)
print('PIPE| picked-start rollouts done', flush=True)

# ---------------- stage 4: 500-task subsample oracle ---------------------
rng = np.random.default_rng(1)
subN = rng.choice(np.nonzero(have)[0], 500, replace=False)
subN.sort()
orc = np.zeros(500, np.float32)
cap_pick = np.zeros(500, np.float32)
t0 = time.time()
for k, t in enumerate(subN):
    m = TID == t
    qs = Q[m]
    L = np.zeros(len(qs), np.float32)
    for lo in range(0, len(qs), B):
        hi = min(lo + B, len(qs))
        pad = B - (hi - lo)
        sub = {'q0': torch.tensor(qs[lo:hi], dtype=rdt),
               'p0': torch.tensor(np.repeat(tz['cs_p0'][t][None], hi - lo, 0), dtype=rdt),
               'line_dir': torch.tensor(np.repeat(tz['cs_line_dir'][t][None], hi - lo, 0), dtype=rdt),
               'n_target': torch.tensor(np.repeat(tz['cs_n_target'][t][None], hi - lo, 0), dtype=rdt)}
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
        L[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
    orc[k] = L.max()
    cap_pick[k] = prog[t]
    if k % 50 == 0:
        print(f'PIPE| oracle subsample {k}/500 ({(time.time()-t0)/60:.0f} min)',
              flush=True)
np.savez(FU + 'dense_sel_oracle500.npz', tasks=subN, oracle=orc,
         picked=cap_pick)
print('PIPE| ALL DONE', flush=True)
