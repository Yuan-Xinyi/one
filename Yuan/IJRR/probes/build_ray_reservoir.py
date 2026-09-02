"""Ray-start reservoir for the generalized start curriculum.

For the mainline 100k training pool (feasibility-filtered, cached), draw
mid-line start states along each task's own ray: s0 ~ U(0.02, 1.4), one
cone direction per attempt, CVT warm starts, IK projection, full
admissibility. Saves (task_idx, s0, q, p_anchor) plus per-task p0 so a
mixed batch can carry an explicit p0 for every row.
"""
import matplotlib; matplotlib.use('Agg')
import sys, math, time, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import LineDistribution

dev = torch.device('cuda')
env = lb.build_env(dev, 'stock', 512)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=100000,
    n_target_noise_deg=5.0, seed=0, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=True)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
print(f'[res] pool valid tasks: {len(valid)}', flush=True)

dt = env.kin.dtype
T = np.load('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/'
            + lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)

# per-task ray anchor p0 = FK(q0)
p0_all = torch.zeros(len(pool.q_pool), 3, dtype=dt, device=dev)
with torch.no_grad():
    for lo in range(0, len(valid), 8192):
        ids = valid[lo:lo + 8192]
        p, _, _, _ = env.kin.tcp_fk_jac(pool.q_pool[ids])
        p0_all[ids] = p
print('[res] task anchors done', flush=True)

rng = np.random.default_rng(0)
TARGET = 300_000
K_NN = 48
CH = 4096
out_task, out_s, out_q, out_p = [], [], [], []
t0 = time.time()
rounds = 0
while sum(len(x) for x in out_task) < TARGET and rounds < 12:
    rounds += 1
    pick = valid[torch.as_tensor(
        rng.integers(0, len(valid), 60_000), device=dev)]
    s0 = rng.uniform(0.02, 1.4, len(pick)).astype(np.float32)
    p0t = p0_all[pick].cpu().numpy()
    d = pool.line_dir_pool[pick].cpu().numpy()
    nt = pool.n_target_pool[pick].cpu().numpy()
    pts = (p0t + s0[:, None] * d).astype(np.float32)
    # one cone direction per row (axis 30%, sampled 70%)
    dirs = np.empty_like(nt)
    use_axis = rng.random(len(pick)) < 0.3
    dirs[use_axis] = nt[use_axis]
    for i in np.nonzero(~use_axis)[0]:
        dirs[i] = _sample_in_cone(torch.as_tensor(nt[i]), lb.CONE_DEG, 1,
                                  np.random.default_rng(
                                      rng.integers(1 << 30))).numpy()[0]
    feat = np.concatenate([pts * POS_SCALE, dirs], 1).astype(np.float32)
    _, ids = tree.query(feat, k=K_NN, workers=-1)
    got = np.zeros(len(pick), bool)
    with torch.no_grad():
        for lo in range(0, len(pick), CH):
            hi = min(lo + CH, len(pick))
            rows = np.arange(lo, hi)
            fq = torch.as_tensor(T['q'][ids[lo:hi]].reshape(-1, 7),
                                 device=dev, dtype=dt)
            fp = torch.as_tensor(np.repeat(pts[lo:hi], K_NN, 0),
                                 device=dev, dtype=dt)
            fz = torch.as_tensor(np.repeat(dirs[lo:hi], K_NN, 0),
                                 device=dev, dtype=dt)
            fn = torch.as_tensor(np.repeat(nt[lo:hi], K_NN, 0),
                                 device=dev, dtype=dt)
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
            fine = fine.reshape(hi - lo, K_NN)
            first = fine.float().argmax(dim=1)
            ok = fine.any(dim=1).cpu().numpy()
            qs = q_o.reshape(hi - lo, K_NN, 7)[
                torch.arange(hi - lo, device=dev), first]
            sel = np.nonzero(ok)[0]
            out_task.append(pick.cpu().numpy()[rows[sel]])
            out_s.append(s0[rows[sel]])
            out_q.append(qs.cpu().numpy()[sel])
            out_p.append(pts[rows[sel]])
    n_now = sum(len(x) for x in out_task)
    print(f'[res] round {rounds}: total {n_now} '
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)

task_i = np.concatenate(out_task)[:TARGET]
np.savez_compressed(
    '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/ray_reservoir_fr3.npz',
    task_idx=task_i.astype(np.int64),
    s0=np.concatenate(out_s)[:TARGET].astype(np.float32),
    q=np.concatenate(out_q)[:TARGET].astype(np.float32),
    p_anchor=np.concatenate(out_p)[:TARGET].astype(np.float32),
    p0_task=p0_all.float().cpu().numpy().astype(np.float32))
print(f'[res] saved {len(task_i)} entries', flush=True)
