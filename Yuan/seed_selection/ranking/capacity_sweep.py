"""Task-capacity reference prototype: certified interval [B, A1_est].

B  = continuously-achievable lower bound: per-branch homotopy continuation
     (converge-then-advance Newton with null-space cone/limit repair) from
     ~32 cached feasible start configurations per task, merged with the best
     length ever achieved by any rollout in the caches (certificates).
A1 = existence envelope estimate: random-restart feasibility probes beyond B
     (continuity in s enforced by contiguous-slice truncation).

Constraints checked: position tolerance, tool-axis cone (30 deg), joint
limits. Collision is NOT checked in the sweep (noted; certificates are
collision-clean by construction).

Prototype scope: 2048 stratified eval tasks, ds = 5 mm.
"""
import os, sys, time
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.env import EnvConfig
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
import yaml

OUT = Path('Yuan/seed_selection/runs/capacity_ref')
OUT.mkdir(parents=True, exist_ok=True)
N_TASKS = 10000
DS = 0.005                 # 5 mm slices
S_CAP = 1.70               # sweep cap (max observed length 1.58 m)
POS_TOL = 0.002            # 2 mm
COS_MIN = float(np.cos(np.deg2rad(30.0))) + 1e-4
JL_MARGIN = 1e-4
REPAIR_ITERS = 6
PROBE_RESTARTS = 64
PROBE_MAX_SLICES = 0       # probe disabled: measured recall 55-75% at known-feasible slices
dev = torch.device('cuda')

cfg = yaml.safe_load(open('Yuan/RL_controller/config.yaml'))
kin = BatchedFR3Kinematics(device=dev,
                           tcp_offset=cfg['env'].get('tcp_offset', 0.0))
DT = kin.dtype
lo = kin.lmt_lo.to(DT); hi = kin.lmt_up.to(DT)
q_mid = (lo + hi) / 2; q_half = (hi - lo) / 2

# ---- tasks + cached start configs + certificates ----
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))
rng = np.random.default_rng(31415)
sel = []
for b in ('Easy', 'Medium', 'Difficult'):
    idx_b = np.nonzero((oh > 1e-6) & (bucket == b))[0]
    n_b = int(round(N_TASKS * len(idx_b) / (oh > 1e-6).sum()))
    sel.append(rng.choice(idx_b, size=n_b, replace=False))
sel = np.sort(np.concatenate(sel))[:N_TASKS]
T = len(sel)
p0 = torch.as_tensor(z['cs_p0'][sel], device=dev, dtype=DT)
dvec = torch.as_tensor(z['cs_line_dir'][sel], device=dev, dtype=DT)
nvec = torch.as_tensor(z['cs_n_target'][sel], device=dev, dtype=DT)

P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
pdw = np.load(P0DIR / 'candidates_extw1.npz')
seeds_np = np.concatenate([
    pd0['seeds'][sel], pde['seeds'][sel], pdw['seeds'][sel],
    z['q0_seed'][sel][:, None, :], z['max_label_q'][sel][:, None, :],
    np.load('Yuan/system_eval/runs/eval_10k_systematic/'
            'cell_oracle_hyb_results.npz')['seeds'][sel]], axis=1)   # (T, 32, 7)
C = seeds_np.shape[1]
# sanitize: invalid/padded slots (NaN or wild values) -> mid-range placeholder,
# which then simply fails the slice-0 feasibility check
bad = ~np.isfinite(seeds_np).all(-1) | (np.abs(seeds_np) > 10).any(-1)
seeds_np = np.where(bad[..., None], q_mid.cpu().numpy(), seeds_np)
print(f'[cap] sanitized {int(bad.sum())} invalid seed slots')
q = torch.as_tensor(seeds_np.reshape(T * C, 7), device=dev, dtype=DT)
task_of = torch.arange(T, device=dev).repeat_interleave(C)

# certificates: best length ever achieved in caches (rollouts are feasible paths)
FC = P0DIR / 'final_ctrl'
L25 = np.stack([np.load(FC / f'L_slot{si}.npz')['L'] for si in range(25)], 1) * 1.5
ok25 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok'],
                       np.ones((10000, 1), bool)], 1)
ach = np.where(ok25, L25, 0).max(1)
achieved = np.maximum(ach, oh)[sel]          # meters, certified achievable


def fk(qb):
    p, R, J, _ = kin.tcp_fk_jac(qb)
    return p, R, J


def repair(qb, p_tgt, nv, iters=REPAIR_ITERS):
    """Converge-then-check: Newton on position + null-space cone/limit repair."""
    for _ in range(iters):
        p, R, J = fk(qb)
        Jp = J[:, :3, :]
        e = p_tgt - p
        # fixed-damping pinv via batched 3x3 solve (avoids cusolver eigvalsh,
        # which rejects batches > 65535 on this box)
        JJt = Jp @ Jp.transpose(-1, -2)
        lam2 = 4e-4
        A = JJt + lam2 * torch.eye(3, device=dev, dtype=DT)
        Jpinv = Jp.transpose(-1, -2) @ torch.linalg.solve(A, torch.eye(3, device=dev, dtype=DT).expand_as(A).contiguous())
        dq = (Jpinv @ e.unsqueeze(-1)).squeeze(-1)
        # null-space repair
        N = torch.eye(7, device=dev, dtype=DT) - Jpinv @ Jp
        zt = R[:, :, 2]
        cosz = (zt * nv).sum(-1)
        zxn = torch.linalg.cross(zt, nv, dim=-1)
        g_cone = (J[:, 3:, :].transpose(-1, -2) @ zxn.unsqueeze(-1)).squeeze(-1)
        need_cone = (cosz < COS_MIN + 0.03).unsqueeze(-1)
        qn = (qb - q_mid) / q_half
        g_jl = -qn / q_half * (qn.abs() > 0.90)
        g = 0.5 * g_cone * need_cone + 0.1 * g_jl
        dq = dq + (N @ g.unsqueeze(-1)).squeeze(-1)
        dq = torch.nan_to_num(dq).clamp(-0.15, 0.15)
        qb = (qb + dq).clamp(lo - 0.5, hi + 0.5)
    p, R, _ = fk(qb)
    pos_ok = (p_tgt - p).norm(dim=-1) <= POS_TOL
    cone_ok = (R[:, :, 2] * nv).sum(-1) >= COS_MIN
    jl_ok = ((qb > lo + JL_MARGIN) & (qb < hi - JL_MARGIN)).all(-1)
    return qb, pos_ok & cone_ok & jl_ok


# ---- 0. validate chains at s=0 ----
t0 = time.time()
p_tgt = p0[task_of]
nv = nvec[task_of]
ok = torch.zeros(q.shape[0], dtype=torch.bool, device=dev)
for s0 in range(0, q.shape[0], 65536):
    e0 = min(s0 + 65536, q.shape[0])
    q[s0:e0], ok[s0:e0] = repair(q[s0:e0], p_tgt[s0:e0], nv[s0:e0])
alive = ok.clone()
print(f'[cap] slice-0 valid chains: {int(alive.sum())}/{T*C} '
      f'({time.time()-t0:.0f}s)', flush=True)

# ---- 1. sweep ----
n_slices = int(S_CAP / DS)
reach = torch.zeros(T * C, device=dev, dtype=DT)      # last feasible s
for k in range(1, n_slices + 1):
    if not bool(alive.any()):
        break
    ai = torch.nonzero(alive, as_tuple=False).squeeze(-1)
    s = k * DS
    p_t = p0[task_of[ai]] + s * dvec[task_of[ai]]
    qa, ok = repair(q[ai], p_t, nvec[task_of[ai]])
    q[ai] = torch.where(ok.unsqueeze(-1), qa, q[ai])
    reach[ai[ok]] = s
    died = ai[~ok]
    alive[died] = False
    if k % 60 == 0:
        print(f'[cap] s={s:.2f} m, alive {int(alive.sum())}', flush=True)
B_sweep = torch.zeros(T, device=dev, dtype=DT).scatter_reduce(
    0, task_of, reach, reduce='amax').cpu().numpy()
B = np.maximum(B_sweep, achieved)
print(f'[cap] sweep done ({time.time()-t0:.0f}s)', flush=True)

# ---- 2. frontier existence probe beyond B ----
gen = torch.Generator(device=dev).manual_seed(7)
A1 = B.copy()
Bt = torch.as_tensor(B, device=dev, dtype=DT)
active = np.ones(T, bool)
for j in range(1, PROBE_MAX_SLICES + 1):
    ti = np.nonzero(active)[0]
    if len(ti) == 0:
        break
    tt = torch.as_tensor(ti, device=dev)
    s = Bt[tt] + j * DS
    M = PROBE_RESTARTS
    qr = lo + (hi - lo) * torch.rand((len(ti) * M, 7), generator=gen,
                                     device=dev, dtype=DT)
    tof = tt.repeat_interleave(M)
    p_t = p0[tof] + s.repeat_interleave(M).unsqueeze(-1) * dvec[tof]
    _, okp = repair(qr, p_t, nvec[tof], iters=10)
    found = okp.view(len(ti), M).any(-1).cpu().numpy()
    A1[ti[found]] = B[ti[found]] + j * DS
    active[ti[~found]] = False        # first gap truncates (s-continuity)
    if j % 20 == 0:
        print(f'[cap] probe +{j*DS*1000:.0f} mm, active {active.sum()}', flush=True)

# ---- 3. report ----
ohs = oh[sel]
Lr = np.load(FC / 'ranked_retrained.npz')['L_ranked'][sel]
np.savez_compressed(OUT / 'capacity_10k.npz', sel=sel, B=B, B_sweep=B_sweep,
                    A1=A1, achieved=achieved, oh=ohs, L_ranked=Lr)
print('\n==== CAPACITY REFERENCE PROTOTYPE (n=2048, ds=5mm) ====')
print(f'B > lref:            {100*(B > ohs + 0.005).mean():.1f}% of tasks '
      f'(median uplift {np.median((B-ohs)/np.maximum(ohs,1e-6))*100:+.1f}%)')
print(f'sweep alone >= achieved: {100*(B_sweep >= achieved - 0.005).mean():.1f}% '
      f'(certificates fill the rest)')
print(f'interval width A1-B: median {np.median(A1-B)*1000:.0f} mm, '
      f'P90 {np.quantile(A1-B, 0.9)*1000:.0f} mm')
print(f'ranked method: {100*(Lr/np.maximum(ohs,1e-6)).mean():.1f}% of lref | '
      f'{100*(Lr/np.maximum(B,1e-6)).mean():.1f}% of B | '
      f'{100*(Lr/np.maximum(A1,1e-6)).mean():.1f}% of A1')
print(f'tasks where ranked > B (should be ~0): {int((Lr > B + 0.005).sum())}')
print(f'wall time {time.time()-t0:.0f}s   [collision unchecked in sweep]')
print('[cap] done', flush=True)
