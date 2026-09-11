"""Circle start selection: dense candidate pool at each scenario's p0
(shared across radii), flagship-critic zero-step pick, radius sweep with
the picked start. Compare against the canonical-start sweep."""
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
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
d = np.load(FU/'circle_rpw_v1.npz')
sc, r_pw = d['scenario'], d['r_pw']
N = len(sc)
tz = np.load(A/'tasks_sel_arc.npz')
env0 = lb.build_env(dev, 'stock', 512)
dt0 = env0.kin.dtype
T = np.load(REPO/lb.TABLE)
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)
p0 = tz['cs_p0'][sc].astype(np.float32)
nt = tz['cs_n_target'][sc].astype(np.float32)
nt /= np.linalg.norm(nt, axis=1, keepdims=True)

# stage 1: dense start pools at p0 (8 dirs x 96 kNN)
CQ, CT = [], []
for m in range(8):
    if m == 0:
        zs = nt.copy()
    else:
        zs = np.stack([_sample_in_cone(torch.as_tensor(nt[i]), lb.CONE_DEG, 1,
                       np.random.default_rng(m*1000+i)).numpy()[0] for i in range(N)])
    feat = np.concatenate([p0*POS_SCALE, zs], 1).astype(np.float32)
    _, ids = tree.query(feat, k=96, workers=-1)
    CH = 128
    for lo in range(0, N, CH):
        hi = min(lo+CH, N)
        fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, 7), device=dev, dtype=dt0)
        fp = torch.as_tensor(np.repeat(p0[lo:hi], 96, 0), device=dev, dtype=dt0)
        fz = torch.as_tensor(np.repeat(zs[lo:hi], 96, 0), device=dev, dtype=dt0)
        fn = torch.as_tensor(np.repeat(nt[lo:hi], 96, 0), device=dev, dtype=dt0)
        q_o, _, _ = _batched_ik_project(env0.kin, fq, fp, _build_R_with_z(fz, hint),
                                        branch_action=None)
        coll = env0.collision.is_collided(env0.kin.link_transforms(q_o))
        p_fk, R_fk, _, _ = env0.kin.tcp_fk_jac(q_o)
        in_lmt = ((q_o >= env0.kin.lmt_lo-1e-5) & (q_o <= env0.kin.lmt_up+1e-5)).all(-1)
        fine = ((~coll) & in_lmt & ((p_fk-fp).norm(dim=-1) <= tube)
                & ((R_fk[:, :, 2]*fn).sum(-1) >= cos_lim))
        f = fine.cpu().numpy()
        rows = (np.arange(len(f))//96) + lo
        CQ.append(q_o[fine].cpu().numpy()); CT.append(rows[f])
    print(f'dir {m+1}/8', flush=True)
CQ = np.concatenate(CQ).astype(np.float32); CT = np.concatenate(CT)
order = np.argsort(CT, kind='stable'); CQ, CT = CQ[order], CT[order]
key = np.concatenate([CT[:, None], np.round(CQ, 2)], 1)
_, ui = np.unique(key, axis=0, return_index=True)
CQ, CT = CQ[np.sort(ui)], CT[np.sort(ui)]
cnt = np.bincount(CT, minlength=N)
print(f'pools: median {int(np.median(cnt))} cands/scenario, zero-cand {(cnt==0).sum()}', flush=True)
del env0; torch.cuda.empty_cache()

# stage 2: critic zero-step scores (kappa irrelevant at reset; use r=0.2)
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 2000; kw['k_lateral'] = 5.0
B = 2048
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
rdt = renv.kin.dtype
V = np.zeros(len(CQ), np.float32)
with torch.no_grad():
    for lo in range(0, len(CQ), B):
        hi = min(lo+B, len(CQ)); pad = B-(hi-lo)
        ids = CT[lo:hi]
        sub = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt),
               'line_dir': torch.tensor(tz['cs_line_dir'][sc[ids]], dtype=rdt),
               'n_target': torch.tensor(nt[ids], dtype=rdt),
               'kappa': torch.tensor(np.full(hi-lo, 5.0), dtype=rdt)}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in sub.items()}
        sub = {k: v.to(dev) for k, v in sub.items()}
        renv.line_dist = ScriptedLineDistribution(sub)
        renv.reset()
        V[lo:hi] = ag.get_value(renv.current_obs()).float().cpu().numpy()[:hi-lo]
pick_q = tz['q0_seed'][sc].astype(np.float32).copy()
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    pick_q[CT[lo]] = CQ[lo + int(np.argmax(V[lo:hi]))]
    lo = hi
print('picks ready', flush=True)

# stage 3: radius sweep with picked starts
GRID = np.round(np.arange(0.05, 0.551, 0.025), 3)
jobs = [(i, r) for i in range(N) for r in GRID if r <= r_pw[i] + 0.026]
jobs = np.array(jobs, np.float64)
prog = np.zeros(len(jobs), np.float32)
with torch.no_grad():
    for lo in range(0, len(jobs), B):
        hi = min(lo+B, len(jobs)); pad = B-(hi-lo)
        ii = jobs[lo:hi, 0].astype(int); rr = jobs[lo:hi, 1]
        sub = {'q0': torch.tensor(pick_q[ii], dtype=rdt),
               'line_dir': torch.tensor(tz['cs_line_dir'][sc[ii]], dtype=rdt),
               'n_target': torch.tensor(nt[ii], dtype=rdt),
               'kappa': torch.tensor(1.0/rr, dtype=rdt)}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in sub.items()}
        sub = {k: v.to(dev) for k, v in sub.items()}
        renv.line_dist = ScriptedLineDistribution(sub)
        renv.reset()
        for _ in range(renv.cfg.max_steps//2):
            a = ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            if bool(renv.done_persistent.all()):
                break
        prog[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi-lo]
        print(f'{hi}/{len(jobs)}', flush=True)
closed = prog >= 2*np.pi*jobs[:, 1] - 1e-3
r_sel = np.zeros(N, np.float32)
for j, (i, r) in enumerate(jobs):
    if closed[j]:
        r_sel[int(i)] = max(r_sel[int(i)], r)
old = np.load(FU/'circle_policy_sweep_v1.npz')
r_can = old['r_best']
np.savez(FU/'circle_sel_v1.npz', scenario=sc, r_sel=r_sel, r_can=r_can,
         r_pw=r_pw, pick_q=pick_q, n_cand=cnt)
ok = r_pw > 0
for tag, rb in (('canonical', r_can), ('critic-pick', r_sel)):
    rt = rb[ok]/np.maximum(r_pw[ok], 1e-9)
    print(f'{tag:12s} closed-any {(rb[ok]>0).mean()*100:.1f}%  r/r_pw '
          f'{rt.mean()*100:.1f} / {np.percentile(rt,10)*100:.1f}', flush=True)
imp = (r_sel - r_can)[ok]
print(f'paired: better {(imp>0.01).mean()*100:.1f}%  worse {(imp<-0.01).mean()*100:.1f}%', flush=True)
