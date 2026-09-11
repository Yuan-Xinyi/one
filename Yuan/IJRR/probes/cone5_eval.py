"""Tight-cone (5 deg) straight-line analysis, eval side:
2000-task subsample, fresh 5-deg-admissible starts (z = n_target),
pointwise march at 5 deg, classical + flagship-zero-shot rollouts.
Saves candidates for the later selection stage."""
import sys, math, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent
CONE = 5.0
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
env0 = lb.build_env(dev, 'stock', 512)
dt0 = env0.kin.dtype
T = np.load(REPO/lb.TABLE)
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
cos5 = math.cos(math.radians(CONE)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)
tz = np.load(A/'tasks_pool_fr3.npz')
b30 = np.load(A/'bound_pool_fr3.npz'); w30 = np.load(A/'witness_pool_fr3.npz')
ref30 = np.maximum(b30['L_hi'], w30['prog'])
rng = np.random.default_rng(3)
sub = rng.choice(len(tz['q0_seed']), 2000, replace=False); sub.sort()
p0 = tz['cs_p0'][sub].astype(np.float32)
dd = tz['cs_line_dir'][sub].astype(np.float32)
dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32)
nt /= np.linalg.norm(nt, axis=1, keepdims=True)
N = 2000

# --- fresh 5-deg starts: 4 near-axis dirs x 96 kNN, keep all admissible ---
CQ, CT = [], []
for m in range(4):
    if m == 0:
        zs = nt.copy()
    else:
        zs = np.stack([_sample_in_cone(torch.as_tensor(nt[i]), CONE, 1,
                       np.random.default_rng(m*777+i)).numpy()[0] for i in range(N)])
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
                & ((R_fk[:, :, 2]*fn).sum(-1) >= cos5))
        f = fine.cpu().numpy()
        rows = (np.arange(len(f))//96) + lo
        CQ.append(q_o[fine].cpu().numpy()); CT.append(rows[f])
    print(f'starts dir {m+1}/4', flush=True)
CQ = np.concatenate(CQ).astype(np.float32); CT = np.concatenate(CT)
order = np.argsort(CT, kind='stable'); CQ, CT = CQ[order], CT[order]
key = np.concatenate([CT[:, None], np.round(CQ, 2)], 1)
_, ui = np.unique(key, axis=0, return_index=True)
CQ, CT = CQ[np.sort(ui)], CT[np.sort(ui)]
cnt = np.bincount(CT, minlength=N)
q0_first = np.zeros((N, 7), np.float32); has = cnt > 0
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    q0_first[CT[lo]] = CQ[lo]
    lo = hi
print(f'5-deg starts: {has.mean()*100:.1f}% tasks, median cands {int(np.median(cnt[has]))}', flush=True)

# --- pointwise march at 5 deg ---
pts_all, zs_all, nr_all, seg = [], [], [], []
for i in range(N):
    smax = min(float(ref30[sub[i]]) + 0.06, 1.8)
    ss = np.arange(0.02, smax, 0.02, dtype=np.float32)
    P = p0[i][None] + ss[:, None]*dd[i][None]
    dirs = np.concatenate([nt[i][None], _sample_in_cone(
        torch.as_tensor(nt[i]), CONE, 8, np.random.default_rng(50+i)
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
print(f'lpw5: mean {lpw5[has].mean():.3f} (30-deg ref mean {ref30[sub][has].mean():.3f})', flush=True)
del env0; torch.cuda.empty_cache()

# --- rollouts at cone 5: classical + flagship zero-shot ---
def roll(cfgfile, ckpt, classical=False):
    y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj'/cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    kw['cone_deg'] = CONE
    B = 2000
    renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    rdt = renv.kin.dtype
    if classical:
        fn = cn_action_fn(ClassicalNullspaceController(renv.kin))
    else:
        ag = Agent(renv.obs_dim, renv.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO/ckpt, map_location=dev))
        ag.eval()
    renv.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(q0_first, dtype=rdt, device=dev),
         'line_dir': torch.tensor(dd, dtype=rdt, device=dev),
         'n_target': torch.tensor(nt, dtype=rdt, device=dev)})
    renv.reset()
    with torch.no_grad():
        for _ in range(renv.cfg.max_steps//2):
            a = fn(renv) if classical else ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            if bool(renv.done_persistent.all()):
                break
    out = renv.arc_progress.float().cpu().numpy().copy()
    del renv; torch.cuda.empty_cache()
    return out

p_cls = roll(str(Path(hl.ROBOTS['fr3'][0]).name), None, classical=True)
print('classical rolled', flush=True)
p_rl = roll('config_line_cont_dirfrac_e8kXXL_rm.yaml',
            'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt')
print('flagship rolled', flush=True)
np.savez(FU/'cone5_eval_v1.npz', sub=sub, q0_first=q0_first, has=has,
         lpw5=lpw5, p_cls=p_cls, p_rl=p_rl, cands_q=CQ, cands_t=CT)
ref = np.maximum.reduce([lpw5, p_cls, p_rl])
for tag, v in (('classical', p_cls), ('flagship-0shot', p_rl)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:15s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
