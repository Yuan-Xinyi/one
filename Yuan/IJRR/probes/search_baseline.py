"""Layered-graph search baseline (pathwise redundancy resolution).

Per task, from the canonical start q0 (main-table protocol):
  layer k = admissible IK configs at arc length s_k = k*DS
            (cone-IK: M_DIRS cone directions x K_NN CVT warm starts,
             full admissibility, dedupe, FPS-capped at K_CAP);
  edge (q -> q') between adjacent layers iff
    (a) velocity consistency: |q' - q|_i <= qd_lim_i * DS / V  per joint
        (the plan must be executable at the mainline task speed), and
    (b) substep certification: at interpolation fractions tau the config
        (1-tau) q + tau q' is collision-free, TCP within the lateral
        tube of p(s_k + tau*DS), and tool axis inside the cone
        (joint limits hold by convexity of the box).
  reachable set propagated layer by layer; result = farthest s reached.

argv: n_tasks [out_tag]   (smoke: small n_tasks, prints per-task lines)
"""
import sys, math, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch
from scipy.spatial import cKDTree
from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _sample_in_cone, _build_R_with_z
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.env.env import LATERAL_SAFETY_NET, damped_pinv

N_TASKS = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else 'smoke'
DS = 0.02                 # layer spacing (m)
V = 0.2                   # mainline task speed -> edge time budget DS/V
K_CAP = 64                # reachable-set width cap per layer
N_PERT = 6                # null-space kicks per config per layer
PERT_MAX = 0.18           # max kick magnitude (rad), inside the edge box
N_SUB = 3                 # interior interpolation points per edge
QD_LIM = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
                  np.float32)
DQ_BOX = QD_LIM * DS / V  # per-joint budget per layer step (rad)

dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'
FU = MAIN / 'runs/paper_fill/fam_unify'
env = lb.build_env(dev, 'stock', 512)
dt = env.kin.dtype
T = np.load(REPO / lb.TABLE)
tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
               .astype(np.float32))
cos_lim = math.cos(math.radians(lb.CONE_DEG))
tube = LATERAL_SAFETY_NET
hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)

tz = np.load(A / 'tasks_pool_fr3.npz')
b = np.load(A / 'bound_pool_fr3.npz')
w = np.load(A / 'witness_pool_fr3.npz')
ref_all = np.maximum(b['L_hi'], w['prog'])
base = np.load(FU / 'pool_fr3_straight.npz')
for k in base.files:
    if k.endswith('_progress'):
        ref_all = np.maximum(ref_all, base[k])

rng = np.random.default_rng(2)
tasks = rng.choice(len(tz['q0_seed']), N_TASKS, replace=False)
if 27 not in tasks:
    tasks[0] = 27
tasks.sort()


def admissible(q, pt, ntt):
    """Full admissibility of configs q at path point pt (n_target ntt)."""
    coll = env.collision.is_collided(env.kin.link_transforms(q))
    p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q)
    in_lmt = ((q >= env.kin.lmt_lo - 1e-5)
              & (q <= env.kin.lmt_up + 1e-5)).all(dim=-1)
    return ((~coll) & in_lmt
            & ((p_fk - pt).norm(dim=-1) <= tube)
            & ((R_fk[:, :, 2] * ntt).sum(-1) >= cos_lim))


def advance(q, ptt):
    """Minimum-norm advance to path point ptt: damped-pinv position
    steps only (the env's own task motion), orientation left free."""
    for _ in range(3):
        p_fk, _, J, _ = env.kin.tcp_fk_jac(q)
        Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
        q = q + (Jp @ (ptt - p_fk).unsqueeze(-1)).squeeze(-1)
    return q


def build_layer(p0, d, nt, s, rng_l, R_prev):
    """Candidates at arc length s: minimum-norm continuation of the
    reachable set + certified null-space perturbations of it (the same
    action space the controllers use, discretized). FPS-capped."""
    pt = (p0 + s * d).astype(np.float32)
    ptt = torch.as_tensor(pt, device=dev, dtype=dt)
    ntt = torch.as_tensor(nt, device=dev, dtype=dt)
    qr = torch.as_tensor(R_prev, device=dev, dtype=dt)
    B = len(qr)
    # branch: 1 pure continuation + N_PERT random null-space kicks
    _, _, J, _ = env.kin.tcp_fk_jac(qr)
    Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0, env.cfg.sigma_thr)
    PN = (torch.eye(7, device=dev, dtype=dt)[None] - Jp @ J[:, :3, :])
    g = torch.as_tensor(rng_l.standard_normal((N_PERT, B, 7)),
                        device=dev, dtype=dt)
    u = (PN[None] @ g.unsqueeze(-1)).squeeze(-1)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    mag = torch.as_tensor(rng_l.uniform(0.0, PERT_MAX, (N_PERT, B, 1)),
                          device=dev, dtype=dt)
    q_all = torch.cat([qr, (qr[None] + mag * u).reshape(-1, 7)], 0)
    q_all = advance(q_all, ptt)
    fine = admissible(q_all, ptt, ntt)
    if not bool(fine.any()):
        return np.zeros((0, 7), np.float32)
    Q = np.unique(np.round(q_all[fine].cpu().numpy(), 3), axis=0
                  ).astype(np.float32)
    if len(Q) > K_CAP:                       # farthest-point subsample
        pick = [0]
        dmin = np.linalg.norm(Q - Q[0], axis=1)
        for _ in range(K_CAP - 1):
            j = int(dmin.argmax())
            pick.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(Q - Q[j], axis=1))
        Q = Q[pick]
    return Q


def edges_ok(Ra, Lb, pt_a, d, nt):
    """Certified-edge existence: bool mask over Lb (any edge from Ra)."""
    dq = np.abs(Lb[None, :, :] - Ra[:, None, :])          # (|R|,|L|,7)
    box = (dq <= DQ_BOX[None, None, :]).all(-1)
    ii, jj = np.nonzero(box)
    if len(ii) == 0:
        return np.zeros(len(Lb), bool)
    ok_pair = np.ones(len(ii), bool)
    ntt = torch.as_tensor(nt, device=dev, dtype=dt)
    qa = torch.as_tensor(Ra[ii], device=dev, dtype=dt)
    qb = torch.as_tensor(Lb[jj], device=dev, dtype=dt)
    for k in range(1, N_SUB + 1):
        tau = k / (N_SUB + 1)
        qm = qa + tau * (qb - qa)
        pm = torch.as_tensor((pt_a + tau * DS * d).astype(np.float32),
                             device=dev, dtype=dt)
        CH = 8192
        for lo in range(0, len(qm), CH):
            fine = admissible(qm[lo:lo + CH], pm, ntt)
            ok_pair[lo:lo + CH] &= fine.cpu().numpy()
    hit = np.zeros(len(Lb), bool)
    hit[jj[ok_pair]] = True
    return hit


S_out = np.zeros(N_TASKS, np.float32)
NL = np.zeros(N_TASKS, np.int32)
t0 = time.time()
for i, ti in enumerate(tasks):
    p0 = tz['cs_p0'][ti].astype(np.float32)
    d = tz['cs_line_dir'][ti].astype(np.float32); d /= np.linalg.norm(d)
    nt = tz['cs_n_target'][ti].astype(np.float32); nt /= np.linalg.norm(nt)
    s_cap = min(float(ref_all[ti]) + 0.06, 1.80)
    rng_l = np.random.default_rng(7000 + int(ti))
    R = tz['q0_seed'][ti].astype(np.float32)[None]
    s = 0.0
    while s + DS <= s_cap:
        L = build_layer(p0, d, nt, s + DS, rng_l, R)
        if len(L) == 0:
            break
        hit = edges_ok(R, L, p0 + s * d, d, nt)
        if not hit.any():
            break
        R = L[hit]
        s += DS
    S_out[i] = s
    NL[i] = int(round(s / DS))
    if N_TASKS <= 20 or i % 50 == 0:
        print(f'[{i + 1}/{N_TASKS}] t{ti}: search {s:.2f}  '
              f'ref {ref_all[ti]:.2f}  ({(time.time() - t0) / 60:.1f} min)',
              flush=True)

np.savez_compressed(FU / f'search_baseline_fr3_{TAG}.npz',
                    tasks=tasks, s=S_out, ref=ref_all[tasks])
dfp = np.load(FU / 'e8kXXL_10k.npz')['prog'][tasks]
ref = np.maximum.reduce([ref_all[tasks], S_out, dfp])
for tag2, v in (('layered search', S_out), ('dirfrac flagship', dfp)):
    rt = v / np.maximum(ref, 1e-9)
    print(f'{tag2}: stroke {v.mean():.4f}  ratio {rt.mean() * 100:.2f} / '
          f'{np.percentile(rt, 10) * 100:.2f}')
i27 = int(np.nonzero(tasks == 27)[0][0]) if 27 in tasks else -1
if i27 >= 0:
    print(f't27: search {S_out[i27]:.3f}  dirfrac {dfp[i27]:.3f}  '
          f'ref {ref_all[27]:.3f}')
print(f'total {(time.time() - t0) / 60:.1f} min', flush=True)
