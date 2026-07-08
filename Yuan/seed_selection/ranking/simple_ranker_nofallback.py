"""Two changes (user 0708):
1. Replace the 10-MLP ensemble ranker with a SINGLE model. Compare a plain
   listwise-contrastive loss (InfoNCE over the candidate group) against the
   pair+list combo, single model each.
2. Drop the fallback seed candidate (q0_seed is invisible at deployment).
   Fix the over-constrained-IK artifact first: for eval tasks whose 24
   diffusion candidates all failed the 6-DoF Newton projection, recover them
   with cone-IK (position + 30deg cone + free spin), re-roll under the
   deployed controller, so the no-fallback number is honest.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import obs_and_manip
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)

P0 = Path('Yuan/seed_selection/runs/rank_phase0')
TRDIRS = [Path('Yuan/seed_selection/runs/rank_train'),
          Path('Yuan/seed_selection/runs/rank_train_b'),
          Path('Yuan/seed_selection/runs/rank_train_c')]
FC = P0 / 'final_ctrl'
CKPT = Path('Yuan/RL_controller/runs/exit_rounds7plus/final_avg')
TAU = (0.985, 0.96)
K = 8
dev = torch.device('cuda')

oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
fin = oh > 1e-6
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')

env = build_env(CKPT / 'config.yaml', 4096, dev)
kin = env.kin
DT = kin.dtype
lo, hi = kin.lmt_lo.to(DT), kin.lmt_up.to(DT)
q_mid, q_half = (lo + hi) / 2, (hi - lo) / 2
COS_MIN = float(np.cos(np.deg2rad(30.0)))
POS_TOL = 0.002
classical_ctrl = __import__('Yuan.RL_controller.env.classical_nullspace',
                            fromlist=['ClassicalNullspaceController']
                            ).ClassicalNullspaceController(kin)
agent = load_rl_agent(CKPT, env, dev)


def cone_ik(qb, p_tgt, nv, iters=30):
    for _ in range(iters):
        p, R, J, _ = kin.tcp_fk_jac(qb)
        Jp = J[:, :3, :]
        e = p_tgt - p
        A = Jp @ Jp.transpose(-1, -2) + 4e-4 * torch.eye(3, device=dev, dtype=DT)
        Jpinv = Jp.transpose(-1, -2) @ torch.linalg.solve(
            A, torch.eye(3, device=dev, dtype=DT).expand_as(A).contiguous())
        dq = (Jpinv @ e.unsqueeze(-1)).squeeze(-1)
        N = torch.eye(7, device=dev, dtype=DT) - Jpinv @ Jp
        zt = R[:, :, 2]
        cosz = (zt * nv).sum(-1)
        zxn = torch.linalg.cross(zt, nv, dim=-1)
        g_cone = (J[:, 3:, :].transpose(-1, -2) @ zxn.unsqueeze(-1)).squeeze(-1)
        qn = (qb - q_mid) / q_half
        g = 0.8 * g_cone * (cosz < COS_MIN + 0.05).unsqueeze(-1) \
            + 0.1 * (-qn / q_half * (qn.abs() > 0.9))
        dq = torch.nan_to_num(dq + (N @ g.unsqueeze(-1)).squeeze(-1)).clamp(-0.15, 0.15)
        qb = (qb + dq).clamp(lo - 0.5, hi + 0.5)
    p, R, _, _ = kin.tcp_fk_jac(qb)
    ok = ((p_tgt - p).norm(dim=-1) <= POS_TOL) \
        & ((R[:, :, 2] * nv).sum(-1) >= COS_MIN) \
        & ((qb > lo + 1e-4) & (qb < hi - 1e-4)).all(-1)
    return qb, ok


# ================= 1. train single-model rankers =================
def load_train(d):
    cd = np.load(d / 'candidates_K8.npz')
    fv = np.load(d / 'feat_v2.npz')
    sd = d / 'final'
    L = np.stack([np.load(sd / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
    Lp = np.load(sd / 'L_pilot.npz')['L']
    obs = np.concatenate([fv['obs_slots'], fv['obs_pilot'][:, None, :]], 1)
    mu = np.concatenate([fv['mu_slots'], fv['mu_pilot'][:, None]], 1)
    X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1)
    y = np.concatenate([L, Lp[:, None]], 1) * 1.5
    ok = np.concatenate([cd['ik_ok'], np.ones((len(L), 1), bool)], 1)
    return X.astype(np.float32), y.astype(np.float32), ok

parts = [load_train(d) for d in TRDIRS]
X = np.concatenate([p[0] for p in parts])
y = np.concatenate([p[1] for p in parts])
ok = np.concatenate([p[2] for p in parts])
# drop the pilot/fallback column from TRAINING too (no privileged candidate)
X, y, ok = X[:, :K], y[:, :K], ok[:, :K]
N = X.shape[0]
rng = np.random.default_rng(0)
perm = rng.permutation(N)
nval = N // 10
va, tr = perm[:nval], perm[nval:]
mean = X[tr][ok[tr]].mean(0)
std = X[tr][ok[tr]].std(0) + 1e-6
Xn = (X - mean) / std
ysd = float(y[tr][ok[tr]].std())
Xd = torch.from_numpy(Xn[tr]).to(dev)
yd = torch.from_numpy(y[tr] / ysd).to(dev)
okd = torch.from_numpy(ok[tr]).to(dev)


class Rank(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(loss_kind, seed, epochs=60):  # loss_kind kept for signature; always contrastive
    torch.manual_seed(seed)
    net = Rank().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Ntr = Xd.shape[0]
    for ep in range(epochs):
        order = torch.randperm(Ntr, device=dev)
        for s in range(0, Ntr, 1024):
            ti = order[s:s + 1024]
            pred = net(Xd[ti].reshape(-1, 32)).view(len(ti), K)
            yy, okk = yd[ti], okd[ti]
            # Multi-positive contrastive (supervised InfoNCE over the
            # candidate group). Positives = valid candidates whose length is
            # within BAND of the group's best (a cluster of near-equivalent
            # good starts, not a single argmax); negatives = the rest, which
            # includes many distinct joint configs at similar-bad length. The
            # loss pulls up ALL positives equally against the shared partition
            # function, so the ranker need not order within the good cluster.
            BAND = 0.03            # relative: within 3% of the best length
            best = yy.masked_fill(~okk, -1e9).max(-1, keepdim=True).values
            pos = okk & (yy >= best * (1 - BAND))
            logp = torch.log_softmax(pred.masked_fill(~okk, -1e9), -1)
            npos = pos.sum(-1).clamp(min=1)
            loss = -((logp * pos).sum(-1) / npos).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net


# ================= 2. fix IK for zero-valid eval tasks =================
pd0 = np.load(P0 / 'candidates_K8.npz')
pde = np.load(P0 / 'candidates_ext8.npz')
pdw = np.load(P0 / 'candidates_extw1.npz')
seeds24 = np.concatenate([pd0['seeds'], pde['seeds'], pdw['seeds']], 1)  # (10000,24,7)
ok24 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok']], 1)
L24 = np.stack([np.load(FC / f'L_slot{si}.npz')['L'] for si in range(24)], 1) * 1.5
fv0 = np.load(P0 / 'feat_v2.npz')
fve = np.load(P0 / 'feat_v2_ext8.npz')
fvw = np.load(P0 / 'feat_v2_extw1.npz')
obs24 = np.concatenate([fv0['obs_slots'], fve['obs_slots'], fvw['obs_slots']], 1)
mu24 = np.concatenate([fv0['mu_slots'], fve['mu_slots'], fvw['mu_slots']], 1)

FIXCACHE = P0 / 'ikfix_nofallback.npz'
if FIXCACHE.exists():
    _c = np.load(FIXCACHE)
    ok24, L24, obs24, mu24 = _c['ok24'], _c['L24'], _c['obs24'], _c['mu24']
    print(f'[fix] loaded cached IK-fix; zero-valid tasks now '
          f'{int((~ok24.any(1) & fin).sum())}', flush=True)
    idx = np.array([], dtype=int)
else:
    no_valid = (~ok24.any(1)) & fin
    idx = np.nonzero(no_valid)[0]
    print(f'[fix] recovering {len(idx)} zero-valid tasks with cone-IK', flush=True)
if len(idx) > 0:
    p0i = torch.as_tensor(z['cs_p0'][idx], device=dev, dtype=DT)
    nti = torch.as_tensor(z['cs_n_target'][idx], device=dev, dtype=DT)
    qcand = torch.as_tensor(seeds24[idx], device=dev, dtype=DT)
    n = len(idx)
    tof = torch.arange(n, device=dev).repeat_interleave(24)
    qrec, okrec = cone_ik(qcand.reshape(n * 24, 7), p0i[tof], nti[tof], iters=40)
    okrec = okrec.view(n, 24).cpu().numpy()
    qrec = qrec.view(n, 24, 7).cpu().numpy()
    print(f'[fix] recovered candidates/task: mean {okrec.sum(1).mean():.1f}, '
          f'tasks with >=1 valid: {100*(okrec.any(1)).mean():.1f}%', flush=True)
    recov_L = np.zeros((n, 24), np.float32)
    recov_obs = np.zeros((n, 24, 31), np.float32)
    recov_mu = np.zeros((n, 24), np.float32)
    flat_valid = np.nonzero(okrec.reshape(-1))[0]
    if len(flat_valid) > 0:
        tofn = tof.cpu().numpy()
        qv = qrec.reshape(n * 24, 7)[flat_valid].astype(np.float32)
        p0v = z['cs_p0'][idx][tofn][flat_valid].astype(np.float32)
        ldv = z['cs_line_dir'][idx][tofn][flat_valid].astype(np.float32)
        ntv = z['cs_n_target'][idx][tofn][flat_valid].astype(np.float32)
        r = rollout_seeds_batched(qv, p0v, ldv, ntv, env=env,
                                  controller='hybrid_variantB', classical=classical_ctrl,
                                  agent=agent, tau_enter=TAU[0], tau_exit=TAU[1],
                                  progress_prefix='recov ')
        ob, m = obs_and_manip(env, qv, p0v, ldv, ntv)
        recov_L.reshape(-1)[flat_valid] = r['L'] * 1.5
        recov_obs.reshape(-1, 31)[flat_valid] = ob
        recov_mu.reshape(-1)[flat_valid] = m
    for a in range(n):
        t = idx[a]
        ok24[t], L24[t], obs24[t], mu24[t] = okrec[a], recov_L[a], recov_obs[a], recov_mu[a]
    print(f'[fix] after recovery, tasks with zero valid: '
          f'{int((~ok24.any(1) & fin).sum())}', flush=True)
    np.savez_compressed(FIXCACHE, ok24=ok24, L24=L24, obs24=obs24, mu24=mu24)

# ================= 3. no-fallback scoring =================
X10 = np.concatenate([obs24, np.log(mu24[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()


@torch.no_grad()
def score(net, shape):
    flat = X10n.reshape(-1, 32)
    out = [net(flat[s:s + 65536].to(dev)).cpu() for s in range(0, len(flat), 65536)]
    return torch.cat(out).view(shape).numpy()


def evalrow(name, sc):
    s = np.where(ok24, sc, -np.inf)
    pick = s.argmax(1)
    Lp = L24[np.arange(len(L24)), pick]
    Lp = np.where(ok24.any(1), Lp, 0.0)
    pct = lambda m: 100 * (Lp[m] / oh[m]).mean()
    line = f'  {name:26s} All {pct(fin):.1f}'
    for b in ('Easy', 'Medium', 'Difficult'):
        line += f'   {b[0]} {pct(fin & (bucket == b)):.1f}'
    print(line, flush=True)


print('\n==== SINGLE-MODEL RANKER, NO FALLBACK, IK-FIXED (% of lref) ====', flush=True)
# held-out capture for model selection
ok_v = ok[va]; y_v = y[va]
first = y_v[np.arange(len(va)), np.argmax(ok_v, 1)]
best = np.where(ok_v, y_v, -np.inf).max(1)
Xva = torch.from_numpy(Xn[va]).float().to(dev)
for kind in ('contrastive',):
    net = fit(kind, 0)
    with torch.no_grad():
        scv = net(Xva.reshape(-1, 32)).view(len(va), K).cpu().numpy()
    Lp = y_v[np.arange(len(va)), np.where(ok_v, scv, -np.inf).argmax(1)]
    cap = 100 * (Lp - first).sum() / (best - first).sum()
    sc10 = score(net, (len(L24), 24))
    print(f'[{kind}] held-out capture {cap:.1f}%', flush=True)
    evalrow(f'single-{kind}', sc10)

# reference: ens10 (old, WITH fallback, over-constrained IK) = 98.2 headline
print('  ens10 + fallback (old headline)   All 98.2  (for reference)')
print('[done]', flush=True)
