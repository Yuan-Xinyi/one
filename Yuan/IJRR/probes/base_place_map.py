"""Evidence for the base-placement criterion (sec:place).

A canonical seam (length 0.5 m, horizontal cone axis, representative task
height) is swept over a grid of positions relative to the manipulator
base -- equivalent to sweeping the base pose on the floor plane. Each
placement is judged twice:
  (a) pointwise: every 2 cm sample of the seam admits a cone-constrained
      IK solution (line_bound machinery, 8 cone directions);
  (b) continuous: the DirFrac mainline executes the full seam as ONE
      stroke from the best of the admissible start configurations.
Outputs the map, both feasible regions, and base-placement
recommendations: the max-clearance pose under each criterion, with the
positioning tolerance radius, and the actual continuous shortfall of the
pointwise-recommended pose."""
import sys, math, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, yaml
from scipy.spatial import cKDTree
from scipy import ndimage
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'

# ---- canonical task + placement grid -----------------------------------
L_REQ = 0.50
STEP_S = 0.02
NS = int(round(L_REQ / STEP_S)) + 1          # 26 samples
D = np.array([1.0, 0.0, 0.0], np.float32)
NT = np.array([0.0, -1.0, 0.0], np.float32)  # horizontal cone axis
Z0 = 0.526                                   # median task height
GRID = np.arange(-0.85, 0.851, 0.05, dtype=np.float32)
G = len(GRID)
xs, ys = np.meshgrid(GRID, GRID, indexing='ij')
P0 = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, Z0, np.float32)], 1)
NP_ = P0.shape[0]
print(f'[place] {NP_} placements x {NS} samples', flush=True)

# ---- pointwise map (line_bound machinery) ------------------------------
env = lb.build_env(dev, 'stock', 512)
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET

s_grid = np.arange(NS, dtype=np.float32) * STEP_S
pts = (P0[:, None, :] + s_grid[:, None] * D[None, None, :]).reshape(-1, 3)
n_refs = np.broadcast_to(NT, (pts.shape[0], 3)).copy()
M_DIRS = 8
pool = _sample_in_cone(torch.as_tensor(NT), lb.CONE_DEG, 32,
                       np.random.default_rng(7)).numpy().astype(np.float32)
dirs = np.concatenate([NT[None], pool[:M_DIRS - 1]], 0)

ok = np.zeros(pts.shape[0], bool)
q_start = [[] for _ in range(NP_)]           # admissible starts per placement
t0 = time.time()
for m in range(M_DIRS):
    pend = np.nonzero(~ok)[0]
    s0_rows = np.arange(0, pts.shape[0], NS)  # starts probed every round
    rows = np.union1d(pend, s0_rows)
    zs = np.broadcast_to(dirs[m], (rows.shape[0], 3)).copy()
    okm, qm = lb.feasible_rows(env, tree, T, pts[rows], zs, n_refs[rows],
                               cos_lim, tube, k_nn=200, n_try=8)
    ok[rows[okm]] = True
    for r, o, q in zip(rows, okm, qm):
        if o and r % NS == 0:
            q_start[r // NS].append(q)
    print(f'[place] dir {m + 1}/{M_DIRS}: pointwise-ok '
          f'{ok.mean() * 100:.1f}%  ({time.time() - t0:.0f}s)', flush=True)

ok = ok.reshape(NP_, NS)
first_bad = np.where(ok.all(1), NS, ok.argmin(1))
lpw = first_bad * STEP_S                     # L_hi convention
pw_complete = ok.all(1)
print(f'[place] pointwise-complete: {pw_complete.sum()}/{NP_}', flush=True)

# ---- continuous map (DirFrac rollouts from admissible starts) ----------
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_rh2048XXL.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
flat_q, flat_pid = [], []
for i, qs in enumerate(q_start):
    if not qs:
        continue
    Q = np.unique(np.round(np.stack(qs), 3), axis=0)
    for q in Q:
        flat_q.append(q)
        flat_pid.append(i)
flat_q = np.stack(flat_q) if flat_q else np.zeros((0, 7), np.float32)
flat_pid = np.array(flat_pid)
print(f'[place] {len(flat_q)} rollout starts over '
      f'{len(set(flat_pid.tolist()))} placements', flush=True)

B = 2500
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL/agent.pt',
    map_location=dev))
ag.eval()
dt = renv.kin.dtype
stroke_flat = np.zeros(len(flat_q), np.float32)
with torch.no_grad():
    for lo in range(0, len(flat_q), B):
        hi = min(lo + B, len(flat_q))
        pad = B - (hi - lo)
        q0 = np.concatenate([flat_q[lo:hi]] + ([flat_q[lo:lo + 1]] * 0
                            if not pad else [np.repeat(flat_q[hi - 1:hi],
                                                       pad, 0)]))
        sub = {'q0': torch.tensor(q0, dtype=dt, device=dev),
               'line_dir': torch.tensor(np.broadcast_to(D, (B, 3)).copy(),
                                        dtype=dt, device=dev),
               'n_target': torch.tensor(np.broadcast_to(NT, (B, 3)).copy(),
                                        dtype=dt, device=dev)}
        renv.line_dist = ScriptedLineDistribution(sub)
        renv.reset()
        for _ in range(renv.cfg.max_steps // 2):
            a = ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            if bool(renv.done_persistent.all()):
                break
        stroke_flat[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'[place] rollouts {hi}/{len(flat_q)}', flush=True)

stroke = np.zeros(NP_, np.float32)
np.maximum.at(stroke, flat_pid, stroke_flat)
cont_complete = stroke >= L_REQ - 0.005
# Executed strokes are feasibility witnesses: a placement whose rollout
# reached arc length s certifies every sample up to s (same max(search,
# executed) convention as the paper's ell^pw).
lpw = np.maximum(lpw, np.minimum(stroke, L_REQ))
pw_complete = pw_complete | cont_complete
print(f'[place] continuous-complete: {cont_complete.sum()}/{NP_}  '
      f'(gap {(pw_complete & ~cont_complete).sum()} placements)', flush=True)

# ---- recommendations ---------------------------------------------------
CELL = 0.05


def recommend(mask):
    m2 = mask.reshape(G, G)
    dist = ndimage.distance_transform_edt(m2)
    i, j = np.unravel_index(dist.argmax(), dist.shape)
    return (GRID[i], GRID[j]), dist[i, j] * CELL


rec_cont, tol_cont = recommend(cont_complete)
rec_pw, tol_pw = recommend(pw_complete)
i_pw = (np.abs(GRID - rec_pw[0]).argmin() * G
        + np.abs(GRID - rec_pw[1]).argmin())
print(f'[place] RECOMMEND (continuous criterion): seam start at '
      f'({rec_cont[0]:+.2f}, {rec_cont[1]:+.2f}) m in the base frame, '
      f'positioning tolerance {tol_cont * 100:.0f} cm', flush=True)
print(f'[place] pointwise criterion would pick ({rec_pw[0]:+.2f}, '
      f'{rec_pw[1]:+.2f}) tol {tol_pw * 100:.0f} cm; its continuous stroke '
      f'is {stroke[i_pw]:.3f} m of {L_REQ} m required', flush=True)

np.savez_compressed(FU / 'base_place_map.npz', grid=GRID, lpw=lpw,
                    stroke=stroke, pw=pw_complete, cont=cont_complete,
                    rec_cont=np.array(rec_cont), tol_cont=tol_cont,
                    rec_pw=np.array(rec_pw), tol_pw=tol_pw,
                    L_req=L_REQ, z0=Z0, d=D, n=NT)

# ---- figure ------------------------------------------------------------
S = np.minimum(stroke, L_REQ).reshape(G, G)
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(S.T, origin='lower', cmap='viridis',
               extent=[GRID[0] - CELL / 2, GRID[-1] + CELL / 2,
                       GRID[0] - CELL / 2, GRID[-1] + CELL / 2],
               vmin=0, vmax=L_REQ)
cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label('executed stroke (m, cap = required 0.5 m)')
X, Y = np.meshgrid(GRID, GRID, indexing='ij')
ax.contour(X, Y, pw_complete.reshape(G, G).astype(float), [0.5],
           colors='white', linestyles='--', linewidths=1.6)
ax.contour(X, Y, cont_complete.reshape(G, G).astype(float), [0.5],
           colors='red', linewidths=1.8)
ax.plot(*rec_cont, marker='*', ms=18, mec='k', mfc='red', ls='none',
        label=f'recommended (continuous, tol {tol_cont * 100:.0f} cm)')
ax.plot(*rec_pw, marker='P', ms=12, mec='k', mfc='white', ls='none',
        label=f'pointwise pick ({stroke[i_pw]:.2f} m realized)')
ax.plot(0, 0, marker='s', ms=10, color='k', ls='none', label='robot base')
ax.annotate('', xy=(rec_cont[0] + L_REQ, rec_cont[1]),
            xytext=rec_cont,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.6))
ax.set_xlabel('seam start x in base frame (m)')
ax.set_ylabel('seam start y in base frame (m)')
ax.set_title('Continuous vs pointwise feasibility of a 0.5 m seam')
ax.legend(loc='upper right', fontsize=8, framealpha=0.92)
fig.tight_layout()
fig.savefig(FU / 'base_place_map.png', dpi=220)
fig.savefig(FU / 'base_place_map.pdf')
print('[place] figure written', flush=True)
