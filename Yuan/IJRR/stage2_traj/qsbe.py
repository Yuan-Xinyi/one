"""Safety-Bellman-Equation fitted Q over the vertex set.

The margin ladder is a family of truncations of one quantity -- the worst
future margin under optimal play. myopic is its one-step truncation, beam-H
its H-step truncation; this learns the H = infinity member by the discounted
safety Bellman backup (Fisac et al., 2019):

    y(s, v) = (1 - g) l(s') + g min( l(s'), max_v' Qt(s', v') )      alive
    y(s, v) = min(l(s'), 0)                                          dead

with l the analytic softmin(joint-limit, cone) margin and s' the exact
one-step model. The analytic parts live inside the target; learning supplies
only the continuation. No behavior is imitated: rollouts of the mixed
behaviors provide a state distribution and nothing else, and every state is
backed up through all 2^m actions at once, which is the paired comparison
policy-gradient methods never get. Deployment is argmax_v Q, one forward
pass.

Successor observations, margins and death flags are precomputed once per
state and cached, so the training loop touches no kinematics.

Usage:
    python -m Yuan.IJRR.stage2_traj.qsbe --collect --train
    python -m Yuan.IJRR.stage2_traj.qsbe --train --rounds 2   # add Q-greedy data
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
from Yuan.IJRR.eval.horizon_ladder import StraightModel, make_myopic
from Yuan.IJRR.stage2_traj.wfield import obs27

REPO = Path(__file__).resolve().parents[3]
CFG = 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'
VTX = 'Yuan/IJRR/runs/rl_vertex_line_30M'
OUT = REPO / 'Yuan/IJRR/runs/qsbe'
CHUNK = 24576


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_act: int, hidden: int = 512,
                 heads: int = 1):
        super().__init__()
        self.n_act, self.heads = n_act, heads
        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act * heads))

    def forward(self, x, head: int | None = None):
        out = self.f(x)
        if self.heads == 1:
            return out
        out = out.view(*out.shape[:-1], self.heads, self.n_act)
        return out if head is None else out[..., head, :]


class DuelingQNet(nn.Module):
    """Q = V(s) + A(s,v) - mean_v A: the common mode flows through the
    value stream, the advantage stream is zero-mean by construction, so the
    action-ranking signal is the advantage head's primary job instead of a
    residual between nearly equal outputs."""

    def __init__(self, in_dim: int, n_act: int, hidden: int = 512,
                 heads: int = 1):
        super().__init__()
        self.n_act, self.heads = n_act, heads
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.v = nn.Linear(hidden, heads)
        self.a = nn.Linear(hidden, heads * n_act)

    def forward(self, x, head: int | None = None):
        z = self.trunk(x)
        V = self.v(z).view(*x.shape[:-1], self.heads, 1)
        A = self.a(z).view(*x.shape[:-1], self.heads, self.n_act)
        Q = V + A - A.mean(-1, keepdim=True)
        if self.heads == 1:
            return Q.squeeze(-2)
        return Q if head is None else Q[..., head, :]


def build_env(n_envs, dev):
    y = yaml.safe_load(open(REPO / CFG))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': n_envs}), None, dev)
    model = StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    model.terms = [0, 1]          # softmin(jl, cone): the validated field
    return env, model, y


def _verts(m, dev):
    grid = np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                    -1).reshape(-1, m)
    return torch.as_tensor(grid, dtype=torch.float32, device=dev)


@torch.no_grad()
def successors(model, env, q, d, n, p0, verts):
    """For states q: obs27 of all K successors, their l, and death flags."""
    N, K = q.shape[0], verts.shape[0]
    qe = q.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)
    de = d.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
    ne = n.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
    pe = p0.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
    ae = verts.unsqueeze(0).expand(N, -1, -1).reshape(N * K, -1)
    o_l, l_l, dd_l = [], [], []
    for i in range(0, N * K, CHUNK):
        sl = slice(i, i + CHUNK)
        qn = model.step(qe[sl], de[sl], ne[sl], ae[sl])
        o_l.append(obs27(qn, de[sl], ne[sl], env.kin.q_mid, env.q_half,
                         env.kin).half().cpu())
        mg = model.margins(qn, pe[sl], de[sl], ne[sl])
        l_l.append(mg[:, [0, 1]].amin(-1).half().cpu())   # softmin terms hard-min is fine for labels
        dd_l.append((mg.amin(-1) < 0).cpu())
    return (torch.cat(o_l).reshape(N, K, -1), torch.cat(l_l).reshape(N, K),
            torch.cat(dd_l).reshape(N, K))


@torch.no_grad()
def collect(env, model, verts, behaviors, tasks_per, seed, qnet=None):
    """Roll behaviors, caching (obs27, successor obs/l/dead) per live state."""
    dev = env.device
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=100000,
        n_target_noise_deg=5.0, seed=0, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
    g = torch.Generator().manual_seed(seed)
    myopic = make_myopic(model)
    vtx = _agent(REPO / VTX, env.obs_dim, dev, act_dim=env.act_dim)

    S, SO, SL, SD = [], [], [], []
    N = env.n_envs
    for name in behaviors:
        for _ in range(tasks_per // N):
            sel = valid[torch.randperm(valid.numel(), generator=g)[:N]]
            env.line_dist = ScriptedLineDistribution(
                {'q0': pool.q_pool[sel], 'line_dir': pool.line_dir_pool[sel],
                 'n_target': pool.n_target_pool[sel]})
            env.reset()
            done = torch.zeros(N, dtype=torch.bool, device=dev)
            for _ in range(env.max_steps):
                live = ~done
                if live.any():
                    q = env.q[live]
                    d = env.line_dir[live]
                    n = env.n_target[live]
                    p0 = env.p_start[live]
                    S.append(obs27(q, d, n, env.kin.q_mid, env.q_half,
                                   env.kin).half().cpu())
                    so, sl_, sd = successors(model, env, q, d, n, p0, verts)
                    SO.append(so); SL.append(sl_); SD.append(sd)
                if name == 'myopic':
                    a = myopic(env, done)
                elif name == 'vertex':
                    a = vtx.actor_mean(env.current_obs())
                elif name == 'random':
                    a = verts[torch.randint(0, verts.shape[0], (N,),
                                            device=dev)]
                elif name == 'eps_myopic':
                    a = myopic(env, done)
                    flip = torch.rand(N, device=dev) < 0.3
                    a = torch.where(
                        flip.unsqueeze(-1),
                        verts[torch.randint(0, verts.shape[0], (N,),
                                            device=dev)], a)
                elif name == 'qgreedy':
                    o = obs27(env.q, env.line_dir, env.n_target,
                              env.kin.q_mid, env.q_half, env.kin)
                    a = verts[qnet(o.float()).argmax(-1)]
                env.step(a, auto_reset=False)
                done = env.done_persistent.clone()
                if bool(done.all()):
                    break
        print(f'[collect] {name}: {sum(x.shape[0] for x in S)} states total')
    return (torch.cat(S), torch.cat(SO), torch.cat(SL), torch.cat(SD))


def train_cls1(data, n_act, dev, epochs=10, lr=3e-4):
    """16-way classification of the exact one-step argmax: can the network
    even tell WHICH successor has the best margin, with the numeric
    regression removed entirely. Held-out top-1 accuracy is the resolution
    diagnostic."""
    S, SO, SL, SD = data
    lab = SL.float().argmax(-1)
    Nst = S.shape[0]
    g = torch.Generator().manual_seed(0)
    perm0 = torch.randperm(Nst, generator=g)
    n_te = Nst // 20
    te, tr = perm0[:n_te], perm0[n_te:]
    net = QNet(S.shape[1], n_act).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        perm = tr[torch.randperm(tr.numel())]
        tot, nb = 0.0, 0
        for i in range(0, tr.numel(), 4096):
            b = perm[i:i + 4096]
            loss = ce(net(S[b].float().to(dev)), lab[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        with torch.no_grad():
            accs = []
            for i in range(0, te.numel(), 8192):
                b = te[i:i + 8192]
                accs.append((net(S[b].float().to(dev)).argmax(-1)
                             == lab[b].to(dev)).float().mean().item())
            acc = float(np.mean(accs))
        print(f'[cls] epoch {ep + 1}: ce {tot / nb:.4f}  '
              f'held-out top-1 {acc:.3f}')
    return net


def train_finite_h(data, n_act, dev, H=16, epochs_per=3, lr=3e-4,
                   dueling=False):
    """Stage-wise fitted DP for the finite-horizon worst-margin recursion:
    Q_1 = l(s'); Q_h = min(l(s'), max_v' Q_{h-1}(s', v')). Targets for every
    stage are precomputed with the net as it stood when the stage began and
    cached, so training a later head cannot corrupt an earlier one's
    supervision; all cached targets stay in the loss to pin the shared
    trunk."""
    S, SO, SL, SD = data
    Nst, K = S.shape[0], n_act
    net = (DuelingQNet(S.shape[1], K, heads=H) if dueling
           else QNet(S.shape[1], K, heads=H)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Y = torch.zeros((Nst, H, K), dtype=torch.float16)
    dead_pin = torch.minimum(SL.float(), torch.zeros(1)).half()
    for h in range(H):
        if h == 0:
            y = SL.clone()
        else:
            with torch.no_grad():
                outs = []
                for i in range(0, Nst, 8192):
                    so = SO[i:i + 8192].float().to(dev)
                    qn = net(so.reshape(-1, so.shape[-1]),
                             head=h - 1).reshape(so.shape[0], K, K)
                    outs.append(qn.max(-1).values.cpu())
                cont = torch.cat(outs)
                y = torch.minimum(SL.float(), cont).half()
        y = torch.where(SD, dead_pin, y)
        Y[:, h] = y
        for ep in range(epochs_per):
            perm = torch.randperm(Nst)
            tot, nb = 0.0, 0
            for i in range(0, Nst, 4096):
                b = perm[i:i + 4096]
                pred = net(S[b].float().to(dev))          # (B, H, K)
                yb = Y[b, :h + 1].float().to(dev)
                loss = ((pred[:, :h + 1] - yb) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item(); nb += 1
        print(f'[fh] head {h + 1}/{H}: mse {tot / nb:.6f}')
    return net


def train(data, n_act, dev, epochs=8, gbar=0.99, lr=3e-4, sync=2000):
    S, SO, SL, SD = data
    net = QNet(S.shape[1], n_act).to(dev)
    tgt = QNet(S.shape[1], n_act).to(dev)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Nst = S.shape[0]
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(Nst)
        tot, nb = 0.0, 0
        for i in range(0, Nst, 4096):
            b = perm[i:i + 4096]
            s = S[b].float().to(dev)
            so = SO[b].float().to(dev)              # (B, K, 27)
            l = SL[b].float().to(dev)               # (B, K)
            dead = SD[b].to(dev)
            with torch.no_grad():
                qn = tgt(so.reshape(-1, so.shape[-1])).reshape(
                    so.shape[0], so.shape[1], -1).max(-1).values
                y = (1 - gbar) * l + gbar * torch.minimum(l, qn)
                y = torch.where(dead, torch.minimum(l, torch.zeros_like(l)),
                                y)
            loss = ((net(s) - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            step += 1
            if step % sync == 0:
                tgt.load_state_dict(net.state_dict())
        print(f'[train] epoch {ep + 1}: bellman mse {tot / nb:.5f}')
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks-per', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--gbar', type=float, default=0.99)
    ap.add_argument('--finite-h', type=int, default=0,
                    help='train the finite-horizon recursion with this H')
    ap.add_argument('--mode', default='q', choices=['q', 'dq', 'cls'],
                    help='q: plain heads; dq: dueling decomposition; '
                         'cls: 16-way argmax classification at H=1')
    ap.add_argument('--data', default=str(OUT / 'dataset.pt'))
    ap.add_argument('--collect-only', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    OUT.mkdir(parents=True, exist_ok=True)
    data_path = Path(a.data)
    if data_path.exists() and not a.collect_only:
        S, SO, SL, SD = torch.load(data_path, weights_only=False)
        print(f'[data] loaded {S.shape[0]} states from {data_path}')
        data = (S, SO, SL, SD)
        env, model, _ = build_env(a.batch, dev)
        verts = _verts(env.act_dim, dev)
    else:
        env, model, _ = build_env(a.batch, dev)
        verts = _verts(env.act_dim, dev)
        behaviors = ['myopic', 'eps_myopic', 'vertex', 'random']
        data = collect(env, model, verts, behaviors, a.tasks_per, a.seed)
        torch.save(data, data_path)
        print(f'[data] saved {data[0].shape[0]} states to {data_path}')
        if a.collect_only:
            return

    if a.mode == 'cls':
        net = train_cls1(data, verts.shape[0], dev)
        dst = OUT / f'q_cls1{a.tag}.pt'
        torch.save({'state_dict': net.state_dict(),
                    'in_dim': data[0].shape[1], 'n_act': verts.shape[0]},
                   dst)
    elif a.finite_h > 0:
        net = train_finite_h(data, verts.shape[0], dev, H=a.finite_h,
                             dueling=(a.mode == 'dq'))
        dst = OUT / f'q_fh{a.finite_h}_{a.mode}{a.tag}.pt'
        torch.save({'state_dict': net.state_dict(),
                    'in_dim': data[0].shape[1], 'n_act': verts.shape[0],
                    'heads': a.finite_h, 'mode': a.mode}, dst)
    else:
        net = train(data, verts.shape[0], dev, epochs=a.epochs, gbar=a.gbar)
        dst = OUT / f'q_net_g{str(a.gbar).replace("0.", "")}{a.tag}.pt'
        torch.save({'state_dict': net.state_dict(),
                    'in_dim': data[0].shape[1], 'n_act': verts.shape[0],
                    'gbar': a.gbar}, dst)
    print(f'wrote {dst}  ({data[0].shape[0]} states)')


if __name__ == '__main__':
    main()
