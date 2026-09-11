"""r^pw for circle scenarios: largest radius whose full circle is pointwise
admissible. Scenarios reuse the arc pool geometry (p0, d0, n_target);
binary search on r, each round marches all scenarios' candidate circles
in one batched feasible_rows call (2cm spacing, n-dir cone sampling).
argv: n_scenarios out_tag"""
import sys, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.env.env import LATERAL_SAFETY_NET

N_SC = int(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else 'pilot'
R_LO, R_HI, N_IT = 0.05, 0.55, 7
DS, M_DIRS = 0.02, 4
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'
env = lb.build_env(dev, 'stock', 512)
T = np.load(REPO/lb.TABLE)
tree = cKDTree(np.concatenate([T['pos']*POS_SCALE, T['zax']], 1).astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG)); tube = LATERAL_SAFETY_NET

tz = np.load(A/'tasks_sel_arc.npz')
rng = np.random.default_rng(5)
sc = rng.choice(len(tz['q0_seed']), N_SC, replace=False); sc.sort()
p0 = tz['cs_p0'][sc].astype(np.float32)
d0 = tz['cs_line_dir'][sc].astype(np.float32)
d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
nt = tz['cs_n_target'][sc].astype(np.float32)
nt /= np.linalg.norm(nt, axis=1, keepdims=True)
# circle centre for radius r: c = p0 + r * (n x d0)  (kappa>0 convention)
m0 = np.cross(nt, d0); m0 /= np.linalg.norm(m0, axis=1, keepdims=True)

# per-scenario cone directions (shared across rounds)
dirs = np.empty((N_SC, M_DIRS, 3), np.float32)
for i in range(N_SC):
    pool = _sample_in_cone(torch.as_tensor(nt[i]), lb.CONE_DEG, 8,
                           np.random.default_rng(60 + i)).numpy()
    dirs[i, 0] = nt[i]; dirs[i, 1:] = pool[:M_DIRS-1]

def circle_ok(r_vec):
    """For each scenario, is the full circle of radius r_vec[i] closable?"""
    ok_out = np.zeros(N_SC, bool)
    pts_all, zs_all, nr_all, seg = [], [], [], []
    for i in range(N_SC):
        r = float(r_vec[i])
        npts = max(8, int(np.ceil(2*np.pi*r/DS)))
        th = np.linspace(0, 2*np.pi, npts, endpoint=False, dtype=np.float32)
        c = p0[i] + r*m0[i]
        ring = (c[None] - r*np.cos(th)[:, None]*m0[i][None]
                + r*np.sin(th)[:, None]*np.cross(nt[i], m0[i])[None])
        P = np.repeat(ring, M_DIRS, 0)
        Z = np.tile(dirs[i], (npts, 1))
        pts_all.append(P); zs_all.append(Z)
        nr_all.append(np.repeat(nt[i][None], npts*M_DIRS, 0))
        seg.append((npts, M_DIRS))
    pts = np.concatenate(pts_all); zs = np.concatenate(zs_all)
    nr = np.concatenate(nr_all)
    ok, _ = lb.feasible_rows(env, tree, T, pts, zs, nr, cos_lim, tube,
                             k_nn=100, n_try=8)
    lo = 0
    for i, (npts, md) in enumerate(seg):
        o = ok[lo:lo+npts*md].reshape(npts, md).any(1)
        ok_out[i] = bool(o.all())
        lo += npts*md
    return ok_out

r_lo = np.full(N_SC, R_LO, np.float32)
r_hi = np.full(N_SC, R_HI, np.float32)
base_ok = circle_ok(r_lo)
print(f'r={R_LO}: closable {base_ok.mean()*100:.0f}%', flush=True)
r_hi[~base_ok] = R_LO  # scenarios where even the smallest circle fails
for it in range(N_IT):
    mid = (r_lo + r_hi)/2
    ok = circle_ok(mid)
    r_lo = np.where(ok & base_ok, mid, r_lo)
    r_hi = np.where(ok & base_ok, r_hi, np.where(base_ok, mid, r_hi))
    print(f'round {it+1}/{N_IT}: median r_pw so far {np.median(r_lo):.3f}', flush=True)
r_pw = np.where(base_ok, r_lo, 0.0)
np.savez(MAIN/'runs/paper_fill/fam_unify'/f'circle_rpw_{TAG}.npz',
         scenario=sc, r_pw=r_pw, base_ok=base_ok)
print(f'r_pw: median {np.median(r_pw[base_ok]):.3f}  p90 '
      f'{np.percentile(r_pw[base_ok],90):.3f}  max {r_pw.max():.3f}  '
      f'infeasible-at-{R_LO} {(~base_ok).mean()*100:.0f}%', flush=True)
