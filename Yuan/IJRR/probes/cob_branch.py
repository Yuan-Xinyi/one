"""Cobotta 6-DOF x tight-cone branch-selection experiment.

Per task: admissible starts at p0 (5-deg cone, 4 dirs x 96 kNN from the
cobotta FK table) -> cluster into IK branches (0.5 rad joint metric) ->
first / random / critic-pick rolled on all 1000 tasks; ALL candidates
rolled on a 300-task subsample (pool oracle + branch spread).
Denominator: 5-deg pointwise march per task."""
import sys, math, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
CONE = 5.0
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
env0 = lb.build_env(dev, 'stock', 512, robot='cobotta')
dt0 = env0.kin.dtype; NJ = 6
T = np.load(A/'fk_table_cobotta.npz')
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
cos5 = math.cos(math.radians(CONE)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)
tz = np.load(A/'tasks_pool_cobotta.npz')
rng = np.random.default_rng(7)
sub = rng.choice(len(tz['q0_seed']), 1000, replace=False); sub.sort()
p0 = tz['cs_p0'][sub].astype(np.float32)
dd = tz['cs_line_dir'][sub].astype(np.float32); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
N = 1000

# ---- stage 1: admissible starts ----
CQ, CT = [], []
for m in range(4):
    zs = nt.copy() if m == 0 else np.stack(
        [_sample_in_cone(torch.as_tensor(nt[i]), CONE, 1,
         np.random.default_rng(m*555+i)).numpy()[0] for i in range(N)])
    feat = np.concatenate([p0*POS_SCALE, zs], 1).astype(np.float32)
    _, ids = tree.query(feat, k=96, workers=-1)
    CH = 128
    for lo in range(0, N, CH):
        hi = min(lo+CH, N)
        fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, NJ), device=dev, dtype=dt0)
        fp = torch.as_tensor(np.repeat(p0[lo:hi], 96, 0), device=dev, dtype=dt0)
        fz = torch.as_tensor(np.repeat(zs[lo:hi], 96, 0), device=dev, dtype=dt0)
        fn = torch.as_tensor(np.repeat(nt[lo:hi], 96, 0), device=dev, dtype=dt0)
        q_o, _, _ = _batched_ik_project(env0.kin, fq, fp, _build_R_with_z(fz, hint),
                                        branch_action=None)
        coll = env0.collision.is_collided(env0.kin.link_transforms(q_o))
        p_fk, R_fk, _, _ = env0.kin.tcp_fk_jac(q_o)
        in_lmt = ((q_o >= env0.kin.lmt_lo-1e-5) & (q_o <= env0.kin.lmt_up+1e-5)).all(-1)
        fine = ((~coll) & in_lmt & ((p_fk-fp).norm(dim=-1) <= tube)
                & ((R_fk[:, :, 2]*fn).sum(-1) >= cos5))
        f = fine.cpu().numpy()
        rows = (np.arange(len(f))//96) + lo
        CQ.append(q_o[fine].cpu().numpy()); CT.append(rows[f])
    print(f'dir {m+1}/4', flush=True)
CQ = np.concatenate(CQ).astype(np.float32); CT = np.concatenate(CT)
order = np.argsort(CT, kind='stable'); CQ, CT = CQ[order], CT[order]
key = np.concatenate([CT[:, None], np.round(CQ, 2)], 1)
_, ui = np.unique(key, axis=0, return_index=True)
CQ, CT = CQ[np.sort(ui)], CT[np.sort(ui)]
cnt = np.bincount(CT, minlength=N)
has = cnt > 0

# branch clustering per task (greedy, 0.5 rad max-joint metric)
BR = np.zeros(len(CQ), np.int32)
n_br = np.zeros(N, np.int32)
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    reps = []
    for j in range(lo, hi):
        for bi, r in enumerate(reps):
            if np.abs(CQ[j]-r).max() < 0.5:
                BR[j] = bi
                break
        else:
            BR[j] = len(reps); reps.append(CQ[j])
    n_br[CT[lo]] = len(reps)
    lo = hi
print(f'starts: {has.mean()*100:.1f}% tasks; cands median {int(np.median(cnt[has]))}; '
      f'branches median {int(np.median(n_br[has]))} p90 {int(np.percentile(n_br[has],90))}',
      flush=True)

# ---- stage 2: 5-deg march denominator ----
b30 = np.load(A/'bound_pool_cobotta.npz')
w30 = np.load(A/'witness_pool_cobotta.npz')
ref30 = np.maximum(b30['L_hi'], w30['prog'])
pts_all, zs_all, nr_all, seg = [], [], [], []
for i in range(N):
    smax = min(float(ref30[sub[i]]) + 0.06, 0.9)
    ss = np.arange(0.02, max(smax, 0.06), 0.02, dtype=np.float32)
    P = p0[i][None] + ss[:, None]*dd[i][None]
    dirs = np.concatenate([nt[i][None], _sample_in_cone(
        torch.as_tensor(nt[i]), CONE, 8, np.random.default_rng(40+i)
        ).numpy()[:3]], 0).astype(np.float32)
    pts_all.append(np.repeat(P, 4, 0)); zs_all.append(np.tile(dirs, (len(ss), 1)))
    nr_all.append(np.repeat(nt[i][None], len(ss)*4, 0)); seg.append(len(ss))
okr, _ = lb.feasible_rows(env0, tree, T, np.concatenate(pts_all),
                          np.concatenate(zs_all), np.concatenate(nr_all),
                          cos5, tube, k_nn=100, n_try=8)
lpw5 = np.zeros(N, np.float32); lo = 0
for i, npts in enumerate(seg):
    o = okr[lo:lo+npts*4].reshape(npts, 4).any(1)
    bad = np.nonzero(~o)[0]
    lpw5[i] = (bad[0]+1)*0.02 if len(bad) else npts*0.02 + 0.02
    lo += npts*4
print(f'lpw5: mean {lpw5[has].mean():.3f}', flush=True)
del env0; torch.cuda.empty_cache()

# ---- stage 3: rolls with the cone5-retrained cobotta policy ----
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_cobotta_e8k_cone5.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_cobotta_e8k_cone5/agent.pt',
                              map_location=dev))
ag.eval()
rdt = env.kin.dtype

def roll_many(qs, task_rows):
    out = np.zeros(len(qs), np.float32)
    with torch.no_grad():
        for lo in range(0, len(qs), B):
            hi = min(lo+B, len(qs)); pad = B-(hi-lo)
            ids = task_rows[lo:hi]
            s2 = {'q0': torch.tensor(qs[lo:hi], dtype=rdt),
                  'line_dir': torch.tensor(dd[ids], dtype=rdt),
                  'n_target': torch.tensor(nt[ids], dtype=rdt)}
            if pad:
                s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
            s2 = {k: v.to(dev) for k, v in s2.items()}
            env.line_dist = ScriptedLineDistribution(s2)
            env.reset()
            for _ in range(env.cfg.max_steps//2):
                a = ag.actor_mean(env.current_obs())
                for _ in range(2):
                    env.step(a, auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
            out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
    return out

# critic scores for all candidates
V = np.zeros(len(CQ), np.float32)
with torch.no_grad():
    for lo in range(0, len(CQ), B):
        hi = min(lo+B, len(CQ)); pad = B-(hi-lo)
        ids = CT[lo:hi]
        s2 = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt),
              'line_dir': torch.tensor(dd[ids], dtype=rdt),
              'n_target': torch.tensor(nt[ids], dtype=rdt)}
        if pad:
            s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
        s2 = {k: v.to(dev) for k, v in s2.items()}
        env.line_dist = ScriptedLineDistribution(s2)
        env.reset()
        V[lo:hi] = ag.get_value(env.current_obs()).float().cpu().numpy()[:hi-lo]

q_first = np.zeros((N, NJ), np.float32)
q_rand = np.zeros((N, NJ), np.float32)
q_crit = np.zeros((N, NJ), np.float32)
rr = np.random.default_rng(4)
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    t = CT[lo]
    q_first[t] = CQ[lo]
    q_rand[t] = CQ[rr.integers(lo, hi)]
    q_crit[t] = CQ[lo + int(np.argmax(V[lo:hi]))]
    lo = hi
tr = np.arange(N)
p_first = roll_many(q_first, tr); print('first rolled', flush=True)
p_rand = roll_many(q_rand, tr); print('random rolled', flush=True)
p_crit = roll_many(q_crit, tr); print('critic rolled', flush=True)

# oracle on 300-task subsample: roll ALL candidates
osub = np.sort(np.random.default_rng(8).choice(np.nonzero(has)[0], 300, replace=False))
mask = np.isin(CT, osub)
L_all = roll_many(CQ[mask], CT[mask])
orc = np.zeros(N, np.float32)
br_spread = []
CTm, BRm = CT[mask], BR[mask]
for t in osub:
    m2 = CTm == t
    orc[t] = L_all[m2].max()
    per_br = [L_all[m2][BRm[m2] == b].max() for b in np.unique(BRm[m2])]
    if len(per_br) > 1:
        br_spread.append(max(per_br) - min(per_br))
np.savez(FU/'cob_branch_v1.npz', sub=sub, cnt=cnt, n_br=n_br, lpw5=lpw5,
         p_first=p_first, p_rand=p_rand, p_crit=p_crit, orc=orc, osub=osub,
         CQ=CQ, CT=CT, BR=BR, V=V, L_all=L_all, CTm_mask=mask)
ref = np.maximum.reduce([lpw5, p_first, p_rand, p_crit, orc])
for tag, v in (('first', p_first), ('random', p_rand), ('critic', p_crit)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:8s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
m3 = np.zeros(N, bool); m3[osub] = True
rto = orc[m3]/np.maximum(ref[m3], 1e-9)
rtc = p_crit[m3]/np.maximum(ref[m3], 1e-9)
print(f'oracle(300) ratio {rto.mean()*100:.1f} / {np.percentile(rto,10)*100:.1f}; '
      f'critic on same 300: {rtc.mean()*100:.1f}; '
      f'branch spread (max-min per task) median {np.median(br_spread):.3f} m', flush=True)
