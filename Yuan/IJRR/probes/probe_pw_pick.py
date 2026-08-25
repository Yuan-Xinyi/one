"""Anatomy of the pointwise-pick failure: for the seam placed at the
pointwise criterion's recommended base pose, collect ALL admissible IK
solutions at every 1 cm sample (many warm starts x cone directions) and
overlay the executed DirFrac trajectory. If the admissible set is
disconnected along the path, the ridden component dies at ~0.32 m while
the surviving solutions live on another branch."""
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
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE, _minimal_rotvec
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'

# the pointwise pick's task (seam start in the base frame)
P0 = np.array([-0.25, -0.25, 0.526], np.float32)
D = np.array([1.0, 0.0, 0.0], np.float32)
NT = np.array([0.0, -1.0, 0.0], np.float32)
L_REQ, STEP = 0.50, 0.01
NS = int(round(L_REQ / STEP)) + 1
s_grid = np.arange(NS, dtype=np.float32) * STEP
pts = P0[None] + s_grid[:, None] * D[None]

env = lb.build_env(dev, 'stock', 512)
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
dt = env.kin.dtype

pool = _sample_in_cone(torch.as_tensor(NT), lb.CONE_DEG, 32,
                       np.random.default_rng(7)).numpy().astype(np.float32)
dirs = np.concatenate([NT[None], pool[:7]], 0)

# ---- all admissible IK solutions per sample ----------------------------
K_NN = 200
sols_s, sols_q = [], []
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
for m in range(len(dirs)):
    zs = np.broadcast_to(dirs[m], (NS, 3)).astype(np.float32)
    feat = np.concatenate([pts * POS_SCALE, zs], 1).astype(np.float32)
    _, ids = tree.query(feat, k=K_NN, workers=-1)
    cand = T['q'][ids]                          # (NS, K, 7)
    fq = torch.as_tensor(cand.reshape(-1, 7), device=dev, dtype=dt)
    fp = torch.as_tensor(np.repeat(pts, K_NN, 0), device=dev, dtype=dt)
    fz = torch.as_tensor(np.repeat(zs, K_NN, 0), device=dev, dtype=dt)
    CH = 8192
    for lo in range(0, fq.shape[0], CH):
        q0 = fq[lo:lo + CH]
        p_t = fp[lo:lo + CH]
        R_t = _build_R_with_z(fz[lo:lo + CH], hint)
        q_o, _, _ = _batched_ik_project(env.kin, q0, p_t, R_t,
                                        branch_action=None)
        coll = env.collision.is_collided(env.kin.link_transforms(q_o))
        p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
        nt = torch.as_tensor(NT, device=dev, dtype=dt)
        in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                  & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
        fine = ((~coll) & in_lmt
                & ((p_fk - p_t).norm(dim=-1) <= tube)
                & ((R_fk[:, :, 2] * nt).sum(-1) >= cos_lim))
        f = fine.cpu().numpy()
        rows = (np.arange(lo, lo + q0.shape[0]) // K_NN)[f]
        sols_s.append(s_grid[rows])
        sols_q.append(q_o[fine].cpu().numpy())
    print(f'[probe] dir {m + 1}/{len(dirs)}: '
          f'{sum(len(x) for x in sols_s)} admissible so far', flush=True)
S = np.concatenate(sols_s)
Q = np.concatenate(sols_q)
# dedupe per (sample, rounded q)
key = np.concatenate([S[:, None], np.round(Q, 2)], 1)
_, ui = np.unique(key, axis=0, return_index=True)
S, Q = S[ui], Q[ui]
print(f'[probe] {len(S)} distinct admissible solutions over {NS} samples',
      flush=True)

# ---- best-start DirFrac rollout with logging ---------------------------
okm, qm = lb.feasible_rows(env, tree, T,
                           np.repeat(pts[:1], len(dirs), 0), dirs,
                           np.broadcast_to(NT, (len(dirs), 3)).copy(),
                           cos_lim, tube, k_nn=200, n_try=8)
starts = np.unique(np.round(qm[okm], 3), axis=0)
print(f'[probe] {len(starts)} candidate starts', flush=True)

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_rh2048XXL.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = len(starts)
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_rh2048XXL/agent.pt',
    map_location=dev))
ag.eval()
rdt = renv.kin.dtype
renv.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(starts, dtype=rdt, device=dev),
     'line_dir': torch.tensor(np.broadcast_to(D, (B, 3)).copy(),
                              dtype=rdt, device=dev),
     'n_target': torch.tensor(np.broadcast_to(NT, (B, 3)).copy(),
                              dtype=rdt, device=dev)})
renv.reset()
traj_q, traj_s = [renv.q.cpu().numpy().copy()], \
                 [renv.arc_progress.cpu().numpy().copy()]
term = None
with torch.no_grad():
    for _ in range(renv.cfg.max_steps // 2):
        a = ag.actor_mean(renv.current_obs())
        for _ in range(2):
            _, _, _, _, info = renv.step(a, auto_reset=False)
        traj_q.append(renv.q.cpu().numpy().copy())
        traj_s.append(renv.arc_progress.float().cpu().numpy().copy())
        if bool(renv.done_persistent.all()):
            break
final = renv.arc_progress.float().cpu().numpy()
best = int(final.argmax())
print(f'[probe] best start stroke {final[best]:.3f} m', flush=True)
TQ = np.stack([t[best] for t in traj_q])
TS = np.stack([t[best] for t in traj_s])
# terminating margins of the best rollout's final state
qf = torch.tensor(TQ[-1:], dtype=rdt, device=dev)
p_fk, R_fk, _, _ = renv.kin.tcp_fk_jac(qf)
m_jl_per = ((renv.q_half - (qf - renv.q_mid).abs()) / renv.q_half)[0]
cosf = float((R_fk[0, :, 2] * torch.tensor(NT, dtype=rdt, device=dev))
             .sum())
print('[probe] final per-joint jl margins:',
      m_jl_per.cpu().numpy().round(3))
print(f'[probe] final cone cos {cosf:.3f} (limit {cos_lim:.3f})',
      flush=True)

np.savez_compressed(FU / 'pwpick_anatomy.npz', S=S, Q=Q, TQ=TQ, TS=TS,
                    lmt_lo=env.kin.lmt_lo.cpu().numpy(),
                    lmt_up=env.kin.lmt_up.cpu().numpy(),
                    stroke=float(final[best]))
print('saved', flush=True)
