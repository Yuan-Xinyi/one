"""Big-spline generalization probe: N random Catmull-Rom splines through
the workspace, straight-trained flagship rolled zero-shot, pointwise
march as the denominator. argv: n_splines [tag]"""
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

N_SP = int(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else 'smoke'
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
rng = np.random.default_rng(11)

def catmull(P, n_dense=4000):
    """Centripetal Catmull-Rom through control points P, arc resampled 5mm."""
    P = np.asarray(P, np.float64)
    pts = []
    for i in range(len(P)-3):
        p0, p1, p2, p3 = P[i], P[i+1], P[i+2], P[i+3]
        t = np.linspace(0, 1, n_dense//(len(P)-3))[:, None]
        a = 2*p1; b_ = p2-p0; c = 2*p0-5*p1+4*p2-p3; d_ = -p0+3*p1-3*p2+p3
        pts.append(0.5*(a + b_*t + c*t*t + d_*t**3))
    pts = np.concatenate(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    grid = np.arange(0, s[-1], 0.005)
    out = np.stack([np.interp(grid, s, pts[:, j]) for j in range(3)], 1)
    tan = np.gradient(out, 0.005, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True).clip(1e-9)
    return out.astype(np.float32), tan.astype(np.float32), float(grid[-1])

def sample_spline(rg):
    while True:
        K = 8
        ang = np.sort(rg.uniform(0, 2*np.pi, K))
        rad = rg.uniform(0.35, 0.62, K)
        z = rg.uniform(0.15, 0.75, K)
        P = np.stack([rad*np.cos(ang), rad*np.sin(ang), z], 1)
        P = np.concatenate([P[:1], P, P[-1:]])       # end clamps
        pts, tan, L = catmull(P)
        if 2.0 <= L <= 6.0:
            return pts, tan, L

SPL = [sample_spline(rng) for _ in range(N_SP)]
n_t = np.array([[0.0, 0.0, 1.0]]*N_SP, np.float32)   # cone axis: vertical
L_all = np.array([s[2] for s in SPL], np.float32)
print(f'{N_SP} splines, length {L_all.min():.2f}-{L_all.max():.2f} m '
      f'(mean {L_all.mean():.2f})', flush=True)

env0 = lb.build_env(dev, 'stock', 512)
dt0 = env0.kin.dtype
T = np.load(REPO/lb.TABLE)
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG)); tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt0, device=dev)

# start configs: cone-IK at spline start, tool along cone axis
q0s = np.zeros((N_SP, 7), np.float32); ok0 = np.zeros(N_SP, bool)
for i, (pts, tan, L) in enumerate(SPL):
    feat = np.concatenate([pts[0]*POS_SCALE, n_t[i]], 0)[None]
    _, ids = tree.query(feat.astype(np.float32), k=96, workers=-1)
    fq = torch.as_tensor(T['q'][ids[0]], device=dev, dtype=dt0)
    fp = torch.as_tensor(pts[0], device=dev, dtype=dt0).expand(96, 3)
    fz = torch.as_tensor(n_t[i], device=dev, dtype=dt0).expand(96, 3)
    q_o, _, _ = _batched_ik_project(env0.kin, fq, fp, _build_R_with_z(fz, hint),
                                    branch_action=None)
    coll = env0.collision.is_collided(env0.kin.link_transforms(q_o))
    p_fk, R_fk, _, _ = env0.kin.tcp_fk_jac(q_o)
    in_lmt = ((q_o >= env0.kin.lmt_lo-1e-5) & (q_o <= env0.kin.lmt_up+1e-5)).all(-1)
    fine = ((~coll) & in_lmt & ((p_fk-fp).norm(dim=-1) <= tube)
            & ((R_fk[:, :, 2]*fz).sum(-1) >= cos_lim))
    j = int(fine.float().argmax())
    if bool(fine[j]):
        q0s[i] = q_o[j].cpu().numpy(); ok0[i] = True
print(f'starts found: {ok0.mean()*100:.0f}%', flush=True)

# pointwise march along each spline (2cm, 4 cone dirs)
pts_all, zs_all, nr_all, seg = [], [], [], []
for i, (pts, tan, L) in enumerate(SPL):
    step = max(1, int(0.02/0.005))
    Pm = pts[::step]
    dirs = np.concatenate([n_t[i][None], _sample_in_cone(
        torch.as_tensor(n_t[i]), lb.CONE_DEG, 8,
        np.random.default_rng(70+i)).numpy()[:3]], 0).astype(np.float32)
    pts_all.append(np.repeat(Pm, 4, 0)); zs_all.append(np.tile(dirs, (len(Pm), 1)))
    nr_all.append(np.repeat(n_t[i][None], len(Pm)*4, 0)); seg.append(len(Pm))
okr, _ = lb.feasible_rows(env0, tree, T, np.concatenate(pts_all),
                          np.concatenate(zs_all), np.concatenate(nr_all),
                          cos_lim, tube, k_nn=100, n_try=8)
lpw = np.zeros(N_SP, np.float32); lo = 0
for i, npts in enumerate(seg):
    o = okr[lo:lo+npts*4].reshape(npts, 4).any(1)
    bad = np.nonzero(~o)[0]
    lpw[i] = (bad[0] if len(bad) else npts)*0.02
    lo += npts*4
print(f'lpw: mean {lpw.mean():.2f}  med {np.median(lpw):.2f}  '
      f'(spline len mean {L_all.mean():.2f})', flush=True)
del env0; torch.cuda.empty_cache()

# spline env: dense-table path frame
class SplineEnv(NSRLBatchedEnv):
    def set_tables(self, pts, tan):
        self._sp = pts; self._st = tan          # (B, M, 3) padded
    def _path_frame(self, p):
        d2 = ((p.unsqueeze(1) - self._sp)**2).sum(-1)
        j = d2.argmin(1)
        ar = torch.arange(len(j), device=p.device)
        closest = self._sp[ar, j]; tangent = self._st[ar, j]
        lat = closest - p
        return tangent, lat, lat.norm(dim=-1)

y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 3000; kw['k_lateral'] = 5.0
env = SplineEnv(EnvConfig(**{**kw, 'n_envs': N_SP}), None, dev)
dt_t = env.kin.dtype
M = max(len(s[0]) for s in SPL)
sp = torch.zeros(N_SP, M, 3, dtype=dt_t, device=dev)
st = torch.zeros(N_SP, M, 3, dtype=dt_t, device=dev)
for i, (pts, tan, L) in enumerate(SPL):
    sp[i, :len(pts)] = torch.tensor(pts, dtype=dt_t)
    sp[i, len(pts):] = torch.tensor(pts[-1], dtype=dt_t)   # pad with endpoint
    st[i, :len(tan)] = torch.tensor(tan, dtype=dt_t)
    st[i, len(tan):] = torch.tensor(tan[-1], dtype=dt_t)
env.set_tables(sp, st)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(q0s, dtype=dt_t, device=dev),
     'line_dir': torch.tensor(np.stack([s[1][0] for s in SPL]), dtype=dt_t, device=dev),
     'n_target': torch.tensor(n_t, dtype=dt_t, device=dev)})
env.reset()
QS = [env.q.cpu().numpy().copy()]
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
        QS.append(env.q.cpu().numpy().copy())
        if bool(env.done_persistent.all()):
            break
prog = env.arc_progress.float().cpu().numpy()
lpw2 = np.maximum(lpw, prog)
rt = np.where(ok0, prog, 0)/np.maximum(lpw2, 1e-9)
np.savez(FU/f'spline_probe_{TAG}.npz', prog=prog, lpw=lpw2, ok0=ok0,
         length=L_all, q_traj=np.array(QS)[:, :min(N_SP, 4)], all_q_last=np.array(QS)[-1],
         spl0=SPL[0][0], spl0_tan=SPL[0][1])
print(f'[spline] prog mean {prog[ok0].mean():.2f} m  ratio '
      f'{rt[ok0].mean()*100:.1f} / {np.percentile(rt[ok0],10)*100:.1f}  '
      f'(finished full spline: {(prog[ok0] >= L_all[ok0]-0.03).mean()*100:.0f}%)',
      flush=True)
