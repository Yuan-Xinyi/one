"""Minimal supervised value field: W(q, c) ~ observed remaining stroke.

The deliberately plain first version. Labels are Monte-Carlo remaining
strokes from mixed behavior policies (the analytic margin-gradient law, the
one-step margin lookahead, and the trained vertex policy), the loss is plain
MSE, and nothing else is added: no expectile, no gradient supervision, no
PDE residual. If this field's gradient is useful, it has to become useful on
its own; supervising it with the handcrafted margin gradient would smuggle
the answer in.

Input is the 27-D state-task encoding: the policy observation without the
a_prev channels, which belong to the policy's history, not to the state.
Truncated episodes are dropped (their labels are censored from below).

Outputs runs/w_field/w_field.pt with normalization stats, plus test-split
metrics: R^2, Spearman rho, and pairwise ranking accuracy.

Usage:
    python -m Yuan.IJRR.stage2_traj.wfield --tasks-per-policy 8192
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent
from Yuan.IJRR.eval.horizon_ladder import (
    StraightModel, make_myopic, make_sgngrad)

REPO = Path(__file__).resolve().parents[3]
CFG = 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'
VTX = 'Yuan/IJRR/runs/rl_vertex_line_30M'
OUT = REPO / 'Yuan/IJRR/runs/w_field'


def obs27(q, line_dir, n_target, q_mid, q_half, kin):
    """State-task encoding: the policy observation minus a_prev."""
    p, R, _, _ = kin.tcp_fk_jac(q)
    z = R[:, :, 2]
    qn = (q - q_mid) / q_half
    cos = (z * n_target).sum(-1, keepdim=True)
    crs = torch.linalg.cross(z, n_target, dim=-1)
    return torch.cat([qn, qn * qn, line_dir, z, n_target, cos, crs], dim=-1)


class WNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


@torch.no_grad()
def collect(env, action_fn, needs_done_arg: bool):
    """One scripted batch: per-step obs27, progress, alive; MC labels."""
    env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    obs_l, prog_l, live_l = [], [], []
    for _ in range(env.max_steps):
        o = obs27(env.q, env.line_dir, env.n_target,
                  env.kin.q_mid, env.q_half, env.kin)
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p - p0) * u).sum(-1)
        obs_l.append(o.cpu()); prog_l.append(prog.cpu())
        live_l.append((~done).cpu())
        a = action_fn(env, done) if needs_done_arg else action_fn(env)
        env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    final = ((p - p0) * u).sum(-1).cpu()
    truncated = (~done).cpu()          # episodes still alive at the step cap
    O = torch.stack(obs_l)             # (T, N, 27)
    P = torch.stack(prog_l)            # (T, N)
    L = torch.stack(live_l)            # (T, N)
    y = final.unsqueeze(0) - P         # remaining stroke
    keep = L & ~truncated.unsqueeze(0)
    return O[keep], y[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-per-policy', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y_cfg = yaml.safe_load(open(REPO / CFG))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y_cfg['env'].items() if k in keys}
    N = a.batch

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    # Training pool (seed 0), disjoint from every evaluation pool (seed 4242).
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=y_cfg['line_distribution']['n_pool'],
        n_target_noise_deg=y_cfg['line_distribution']['n_target_noise_deg'],
        seed=y_cfg['line_distribution']['train_seed'], env_cfg=env.cfg,
        feasibility_threshold_m=y_cfg['line_distribution']
        ['feasibility_threshold_m'], verbose=False)
    valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)

    model = StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y_cfg['env']['dt'])
    model.terms = None
    vtx = _agent(REPO / VTX, env.obs_dim, dev, act_dim=env.act_dim)
    policies = {
        'sgngrad': (make_sgngrad(model), True),
        'myopic': (make_myopic(model), True),
        'vertex': (lambda e: vtx.actor_mean(e.current_obs()), False),
    }

    X_l, Y_l, T_l = [], [], []
    g = torch.Generator().manual_seed(a.seed)
    for name, (fn, needs_done) in policies.items():
        n_chunks = a.tasks_per_policy // N
        for c in range(n_chunks):
            sel = valid[torch.randperm(valid.numel(), generator=g)[:N]]
            env.line_dist = ScriptedLineDistribution(
                {'q0': pool.q_pool[sel], 'line_dir': pool.line_dir_pool[sel],
                 'n_target': pool.n_target_pool[sel]})
            O, y = collect(env, fn, needs_done)
            X_l.append(O); Y_l.append(y)
            T_l.append(torch.full((len(y),), c + 100 * hash(name) % 997))
        print(f'[data] {name}: {sum(x.shape[0] for x in X_l)} states so far')
    X = torch.cat(X_l); Y = torch.cat(Y_l)
    print(f'[data] total {X.shape[0]} state-label pairs')

    # split 95/5 at random (tasks are freshly drawn per chunk; state-level
    # split is acceptable for the metrics reported here)
    idx = torch.randperm(X.shape[0], generator=g)
    n_te = X.shape[0] // 20
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
    ym, ys = Y[tr].mean(), Y[tr].std().clamp_min(1e-6)

    net = WNet(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    Xtr = ((X[tr] - mu) / sd).to(dev)
    Ytr = ((Y[tr] - ym) / ys).to(dev)
    for ep in range(a.epochs):
        perm = torch.randperm(Xtr.shape[0], device=dev)
        tot = 0.0
        for i in range(0, Xtr.shape[0], 8192):
            b = perm[i:i + 8192]
            loss = ((net(Xtr[b]) - Ytr[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f'[train] epoch {ep + 1}: mse {tot / Xtr.shape[0]:.4f}')

    net.eval()
    with torch.no_grad():
        Xte = ((X[te] - mu) / sd).to(dev)
        pred = (net(Xte).cpu() * ys + ym).numpy()
    yte = Y[te].numpy()
    ss_res = ((pred - yte) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    from scipy.stats import spearmanr
    rho = spearmanr(pred[:20000], yte[:20000]).statistic
    rng = np.random.default_rng(0)
    i1 = rng.integers(0, len(yte), 200000)
    i2 = rng.integers(0, len(yte), 200000)
    keep = np.abs(yte[i1] - yte[i2]) > 0.01
    pair = ((pred[i1] > pred[i2]) == (yte[i1] > yte[i2]))[keep].mean()
    print(f'\n[test1] R^2 {r2:.4f}   Spearman {rho:.4f}   '
          f'pairwise ranking acc {pair:.4f}')

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'mu': mu, 'sd': sd,
                'ym': ym, 'ys': ys, 'in_dim': X.shape[1]},
               OUT / 'w_field.pt')
    print(f'wrote {OUT / "w_field.pt"}')


if __name__ == '__main__':
    main()
