"""Path-conditioned selector and its out-of-family generalization.

One experiment, four path families, one question: does a selector that
reads the task as a sampled "road table" ahead of the start point rank
initial-configuration candidates on families it never trained on?

  families    straight (ID), constant-curvature arc (ID),
              serpentine (OOD), rotating-cone-axis non-planar (OOD)
  candidates  K per task from the CVT table: cone-sampled tool directions,
              warm-started projections, farthest-point selected in joint
              space, plus the task-generating configuration
  labels      arc progress of each candidate rolled to termination under
              the analytic one-step margin controller -- training-free
  scorer      permutation-equivariant listwise ranker; each candidate is
              embedded with the task's road table (T samples of relative
              position, tangent, cone axis in the base frame); an
              unconditioned twin isolates what the conditioning buys

Usage:
    python -m Yuan.IJRR.stage1_seed.selector_ood --stage tasks
    python -m Yuan.IJRR.stage1_seed.selector_ood --stage candidates
    python -m Yuan.IJRR.stage1_seed.selector_ood --stage labels
    python -m Yuan.IJRR.stage1_seed.selector_ood --stage train --report
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.spatial import cKDTree

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.eval.horizon_ladder import StraightModel, make_myopic, SUB
from Yuan.IJRR.eval.line_bound import feasible_rows
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE
from Yuan.IJRR.stage2_traj.wfield import obs27

REPO = Path(__file__).resolve().parents[3]
CFG = 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'
TABLE = 'Yuan/IJRR/runs/iksel_clean_v1/cvt_table_201600.npz'
K_CAND = 8
N_DIRS = 10               # cone-sampled tool directions tried per task
T_SAMP = 12               # road-table samples
DS = 0.1                  # every 10 cm, covering 1.2 m ahead
TUBE = 0.005              # candidate tip must sit within 5 mm of p0
FAMS = ('straight', 'arc', 'serpentine', 'nonplanar')
OOD = ('serpentine', 'nonplanar')


# ---------------------------------------------------------------- families

def make_specs(pool, idx, fam, rng):
    """Per-task spec dict for one family, built on straight base tasks."""
    n = idx.numel()
    spec = {'q0': pool.q_pool[idx].cpu(),
            'line_dir': pool.line_dir_pool[idx].cpu(),
            'n_target': pool.n_target_pool[idx].cpu()}
    if fam == 'arc':
        kap = rng.uniform(0.5, 3.0, n) * rng.choice([-1, 1], n)
        spec['kappa'] = torch.tensor(kap, dtype=torch.float32)
    elif fam == 'serpentine':
        sw = np.radians(rng.uniform(5.0, 30.0, n))
        lam = rng.uniform(0.4, 1.2, n)
        spec['amp'] = torch.tensor(lam * np.tan(sw) / (2 * np.pi),
                                   dtype=torch.float32)
        spec['wavelen'] = torch.tensor(lam, dtype=torch.float32)
    elif fam == 'nonplanar':
        n0 = spec['n_target'].numpy()
        r = rng.standard_normal((n, 3)).astype(np.float32)
        r -= (r * n0).sum(-1, keepdims=True) * n0
        r /= np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9
        spec['n_rot_axis'] = torch.tensor(r)
        spec['n_rot_rate'] = torch.tensor(
            rng.uniform(0.1, 0.6, n).astype(np.float32))
    return spec


def road_table(spec):
    """(N, T_SAMP, 9): relative position, tangent, cone axis at path
    parameter DS..T_SAMP*DS, analytic per family, in the base frame.
    Parametrizations mirror path_geometry.arc_point / serpentine_point."""
    d0 = spec['line_dir'].numpy().astype(np.float64)
    n0 = spec['n_target'].numpy().astype(np.float64)
    s = (np.arange(1, T_SAMP + 1) * DS)[None, :, None]      # (1, T, 1)
    m0 = np.cross(n0, d0)
    m0 /= np.linalg.norm(m0, axis=-1, keepdims=True) + 1e-9
    if 'kappa' in spec:
        kap = spec['kappa'].numpy().astype(np.float64)[:, None, None]
        th = kap * s
        pos = (np.sin(th) / kap) * d0[:, None] \
            + ((1 - np.cos(th)) / kap) * m0[:, None]
        tan = np.cos(th) * d0[:, None] + np.sin(th) * m0[:, None]
    elif 'amp' in spec:
        A = spec['amp'].numpy().astype(np.float64)[:, None, None]
        k = (2 * np.pi / spec['wavelen'].numpy().astype(np.float64)
             )[:, None, None]
        pos = s * d0[:, None] + A * np.sin(k * s) * m0[:, None]
        tv = d0[:, None] + A * k * np.cos(k * s) * m0[:, None]
        tan = tv / (np.linalg.norm(tv, axis=-1, keepdims=True) + 1e-9)
    else:
        pos = s * d0[:, None]
        tan = np.repeat(d0[:, None], T_SAMP, 1)
    if 'n_rot_rate' in spec:
        ax = spec['n_rot_axis'].numpy().astype(np.float64)[:, None]
        th = spec['n_rot_rate'].numpy().astype(np.float64)[:, None, None] * s
        c, sn = np.cos(th), np.sin(th)
        n0e = np.repeat(n0[:, None], T_SAMP, 1)
        axe = np.repeat(ax, T_SAMP, 1)
        nT = n0e * c + np.cross(axe, n0e) * sn \
            + axe * (axe * n0e).sum(-1, keepdims=True) * (1 - c)
    else:
        nT = np.repeat(n0[:, None], T_SAMP, 1)
    return np.concatenate([pos, tan, nT], -1).astype(np.float32)


# ------------------------------------------------------------- candidates

@torch.no_grad()
def gen_candidates(env, p0, n0, q_gen, rng):
    """Up to K_CAND diverse feasible configurations at p0 per task."""
    T = np.load(REPO / TABLE)
    tree = cKDTree(np.concatenate(
        [T['pos'] * POS_SCALE, T['zax']], 1).astype(np.float32))
    N = p0.shape[0]
    all_q = [[] for _ in range(N)]
    cos_lim = math.cos(math.radians(env.cfg.cone_deg))
    for j in range(N_DIRS):
        if j == 0:
            zs = n0.copy()
        else:
            r = rng.standard_normal((N, 3)).astype(np.float32)
            r -= (r * n0).sum(-1, keepdims=True) * n0
            r /= np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9
            ang = rng.uniform(0, math.radians(env.cfg.cone_deg * 0.8), N
                              ).astype(np.float32)[:, None]
            zs = np.cos(ang) * n0 + np.sin(ang) * r
        ok, q = feasible_rows(env, tree, T, p0, zs, n0, cos_lim, TUBE,
                              k_nn=200, n_try=6)
        for i in np.nonzero(ok)[0]:
            all_q[i].append(q[i])
        print(f'  [cand] dir {j + 1}/{N_DIRS}: feasible {ok.mean():.2f}',
              flush=True)
    cands = np.zeros((N, K_CAND, env.n_joints), np.float32)
    n_found = np.zeros(N, np.int64)
    for i in range(N):
        cs = np.stack([q_gen[i]] + all_q[i])
        # farthest-point selection in joint space, seeded on q_gen;
        # stop early when the remaining pool is duplicates
        keep = [0]
        while len(keep) < min(K_CAND, len(cs)):
            dmin = np.min(
                np.linalg.norm(cs[:, None] - cs[keep][None], axis=-1), 1)
            dmin[keep] = -1.0
            j = int(dmin.argmax())
            if dmin[j] < 1e-4:
                break
            keep.append(j)
        sel = cs[keep]
        n_found[i] = len(sel)
        cands[i, :len(sel)] = sel
        if len(sel) < K_CAND:                     # pad; masked out later
            cands[i, len(sel):] = sel[0]
    return cands, n_found


# ------------------------------------------------------------------ labels

@torch.no_grad()
def label_family(env, model, spec, cands, batch):
    """Arc progress of every candidate under the analytic controller,
    on the task's own path anchored at the shared p0 (all K candidates of
    a task race the identical path)."""
    myo = make_myopic(model)
    N, K = cands.shape[:2]
    dev, dt = env.device, env.kin.dtype
    rep = lambda t: t.repeat_interleave(K, 0)
    flat = {'q0': torch.tensor(cands.reshape(N * K, -1), dtype=dt),
            'p0': rep(spec['p0'])}
    for key in ('line_dir', 'n_target', 'kappa', 'amp', 'wavelen',
                'n_rot_axis', 'n_rot_rate'):
        if key in spec:
            flat[key] = rep(spec[key])
    L = np.zeros(N * K, np.float32)
    blocks = env.max_steps // SUB
    for lo in range(0, N * K, batch):
        hi = min(lo + batch, N * K)
        n_b = hi - lo
        sub = {k: v[lo:hi] for k, v in flat.items()}
        if n_b < env.n_envs:                       # pad the last batch
            pad = env.n_envs - n_b
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                   for k, v in sub.items()}
        for k in ('q0', 'line_dir', 'n_target'):   # consumed w/o .to()
            sub[k] = sub[k].to(device=dev, dtype=dt)
        env.line_dist = ScriptedLineDistribution(sub)
        env.reset()
        done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
        for _ in range(blocks):
            a = myo(env, done)
            for _ in range(SUB):
                env.step(a, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        L[lo:hi] = env.arc_progress[:n_b].float().cpu().numpy()
        print(f'  [label] {hi}/{N * K}', flush=True)
    return L.reshape(N, K)


# ------------------------------------------------------------------ scorer

class Ranker(nn.Module):
    def __init__(self, cand_dim, road_dim, hidden=256, conditioned=True):
        super().__init__()
        self.conditioned = conditioned
        in_dim = cand_dim + (road_dim if conditioned else 0)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, cand, road):
        # cand: (B, K, Dc); road: (B, Droad)
        if self.conditioned:
            r = road.unsqueeze(1).expand(-1, cand.shape[1], -1)
            x = torch.cat([cand, r], -1)
        else:
            x = cand
        e = self.embed(x)
        ctx = e.mean(1, keepdim=True).expand_as(e)
        return self.head(torch.cat([e, ctx], -1)).squeeze(-1)


@torch.no_grad()
def cand_features(env, cands, spec):
    N, K = cands.shape[:2]
    dt = env.kin.dtype
    q = torch.tensor(cands.reshape(N * K, -1), dtype=dt, device=env.device)
    d = spec['line_dir'].to(device=env.device, dtype=dt
                            ).repeat_interleave(K, 0)
    n = spec['n_target'].to(device=env.device, dtype=dt
                            ).repeat_interleave(K, 0)
    f = obs27(q, d, n, env.q_mid, env.q_half, env.kin)
    return f.reshape(N, K, -1).float().cpu()


def train_ranker(Xc, Xr, Y, M, conditioned, dev, epochs=30, lr=1e-3,
                 seed=0):
    torch.manual_seed(seed)
    net = Ranker(Xc.shape[-1], Xr.shape[-1], conditioned=conditioned).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = Xc.shape[0]
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, N, 512):
            b = perm[i:i + 512]
            s = net(Xc[b].to(dev), Xr[b].to(dev))
            y = Y[b].to(dev)
            m = M[b].to(dev)
            diff_s = s.unsqueeze(2) - s.unsqueeze(1)
            diff_y = y.unsqueeze(2) - y.unsqueeze(1)
            valid = (m.unsqueeze(2) & m.unsqueeze(1)
                     & (diff_y.abs() > 0.01))
            if not bool(valid.any()):
                continue
            loss = (torch.nn.functional.softplus(
                -diff_s * diff_y.sign())[valid]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 10 == 0:
            print(f'  [rank {"cond" if conditioned else "nocond"}] '
                  f'epoch {ep + 1}: pairwise loss {tot / max(nb, 1):.4f}',
                  flush=True)
    return net


def _selection_metrics(sel, Y, M):
    """Capture / regret / informative-fraction for one picked-label vector."""
    best = torch.where(M, Y, torch.full_like(Y, -1e9)).max(1).values
    rand = (torch.where(M, Y, torch.zeros_like(Y)).sum(1)
            / M.sum(1).clamp_min(1))
    nz = (best - rand) > 0.02          # tasks where selection matters
    cap = ((sel - rand) / (best - rand).clamp_min(1e-6))[nz]
    reg = ((best - sel) / best.clamp_min(1e-6))[nz]
    return float(cap.mean()), float(reg.mean()), float(nz.float().mean())


def evaluate(net, Xc, Xr, Y, M, dev):
    with torch.no_grad():
        s = net(Xc.to(dev), Xr.to(dev)).cpu()
    s = torch.where(M, s, torch.full_like(s, -1e9))
    sel = Y.gather(1, s.argmax(1, keepdim=True)).squeeze(1)
    cap, reg, frac = _selection_metrics(sel, Y, M)
    # mean within-task Spearman over informative tasks
    rhos = []
    for i in range(Y.shape[0]):
        m = M[i]
        if int(m.sum()) < 3 or float(Y[i][m].max() - Y[i][m].min()) < 0.02:
            continue
        a = s[i][m].numpy().argsort().argsort()
        b = Y[i][m].numpy().argsort().argsort()
        k = len(a)
        rhos.append(1 - 6 * ((a - b) ** 2).sum() / (k * (k * k - 1)))
    return cap, reg, frac, float(np.mean(rhos)) if rhos else float('nan')


# -------------------------------------------------------------------- main

def build_base_env(batch, dev):
    """Ladder-protocol env: 50 ms commands, 25 ms integration."""
    y = yaml.safe_load(open(REPO / CFG))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] = kw['dt'] / SUB
    kw['max_steps'] = int(y['env']['max_steps'] * SUB)
    kw['k_lateral'] = 5.0          # required on curves, inert on lines
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)
    model = StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    model.terms = [0, 1]           # jl + cone: the winning margin law
    return env, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['tasks', 'candidates', 'labels', 'train'])
    ap.add_argument('--n-train', type=int, default=20000)
    ap.add_argument('--n-test', type=int, default=10000)
    ap.add_argument('--pool-train', type=int, default=30000)
    ap.add_argument('--pool-test', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = REPO / 'Yuan/IJRR/runs/selector_ood' / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env, model = build_base_env(a.batch, dev)

    if a.stage == 'tasks':
        rng = np.random.default_rng(0)
        store = {}
        for split, seed, n_total, n_pool in (
                ('train', 0, a.n_train, a.pool_train),
                ('test', 4242, a.n_test, a.pool_test)):
            pool = LineDistribution.load_or_build(
                kin=env.kin, collision=env.collision, n_pool=n_pool,
                n_target_noise_deg=5.0, seed=seed, env_cfg=env.cfg,
                feasibility_threshold_m=0.1, verbose=True)
            valid = torch.nonzero(pool.valid_mask).squeeze(-1)
            fams = ('straight', 'arc') if split == 'train' else FAMS
            per = n_total // len(fams)
            assert valid.numel() >= per * len(fams), \
                f'{split}: pool has {valid.numel()} valid < {per * len(fams)}'
            perm = torch.randperm(valid.numel(),
                                  generator=torch.Generator()
                                  .manual_seed(seed)).to(valid.device)
            perm = valid[perm]
            for fi, fam in enumerate(fams):
                idx = perm[fi * per:(fi + 1) * per]
                spec = make_specs(pool, idx, fam, rng)
                p0, _, _, _ = env.kin.tcp_fk_jac(spec['q0'].to(
                    device=dev, dtype=env.kin.dtype))
                spec['p0'] = p0.float().cpu()
                store[f'{split}_{fam}'] = spec
        torch.save(store, out / 'tasks.pt')
        for k, v in store.items():
            print(f'[tasks] {k}: {v["q0"].shape[0]}')

    elif a.stage == 'candidates':
        tasks = torch.load(out / 'tasks.pt', weights_only=False)
        rng = np.random.default_rng(1)
        store = {}
        for key, spec in tasks.items():
            print(f'[cand] {key} ...', flush=True)
            cands, nf = gen_candidates(
                env, spec['p0'].numpy().astype(np.float32),
                spec['n_target'].numpy().astype(np.float32),
                spec['q0'].numpy().astype(np.float32), rng)
            store[key] = {'cands': torch.tensor(cands),
                          'n_found': torch.tensor(nf)}
            print(f'[cand] {key}: mean found '
                  f'{nf.astype(float).mean():.2f} of {K_CAND}', flush=True)
            torch.save(store, out / 'cands.pt')    # checkpoint per family
        torch.save(store, out / 'cands.pt')

    elif a.stage == 'labels':
        tasks = torch.load(out / 'tasks.pt', weights_only=False)
        cands = torch.load(out / 'cands.pt', weights_only=False)
        store = {}
        for key, spec in tasks.items():
            print(f'[label] {key} ...', flush=True)
            L = label_family(env, model, spec,
                             cands[key]['cands'].numpy(), a.batch)
            store[key] = torch.tensor(L)
            print(f'[label] {key}: mean best {L.max(1).mean():.3f} m  '
                  f'mean spread {(L.max(1) - L.min(1)).mean():.3f} m',
                  flush=True)
            torch.save(store, out / 'labels.pt')   # checkpoint per family

    elif a.stage == 'train':
        tasks = torch.load(out / 'tasks.pt', weights_only=False)
        cands = torch.load(out / 'cands.pt', weights_only=False)
        labels = torch.load(out / 'labels.pt', weights_only=False)

        def pack(keys):
            Xc, Xr, Y, M = [], [], [], []
            for key in keys:
                spec = tasks[key]
                c = cands[key]['cands']
                Xc.append(cand_features(env, c.numpy(), spec))
                rt = road_table(spec)
                Xr.append(torch.tensor(rt.reshape(rt.shape[0], -1)))
                Y.append(labels[key].float())
                nf = cands[key]['n_found']
                M.append(torch.arange(K_CAND)[None, :] < nf[:, None])
            return (torch.cat(Xc), torch.cat(Xr), torch.cat(Y),
                    torch.cat(M))

        tr = pack(['train_straight', 'train_arc'])
        print(f'[train] {tr[0].shape[0]} tasks, cand dim {tr[0].shape[-1]}, '
              f'road dim {tr[1].shape[-1]}')
        nets = {}
        for cond in (True, False):
            nets[cond] = train_ranker(*tr, conditioned=cond, dev=dev)
        torch.save({'cond': nets[True].state_dict(),
                    'nocond': nets[False].state_dict()},
                   out / 'rankers.pt')

        if a.report:
            lines = ['# Selector OOD report', '',
                     '| family | split | scorer | capture | top-1 regret '
                     '| Spearman | informative | oracle mean [m] '
                     '| generator mean [m] |',
                     '|---|---|---|---|---|---|---|---|---|']
            for fam in FAMS:
                key = f'test_{fam}'
                Xc, Xr, Y, M = pack([key])
                split = 'OOD' if fam in OOD else 'ID'
                best = torch.where(M, Y, torch.full_like(Y, -1e9)
                                   ).max(1).values
                gen_cap, gen_reg, _ = _selection_metrics(Y[:, 0], Y, M)
                for name, net in (('conditioned', nets[True]),
                                  ('unconditioned', nets[False])):
                    cap, reg, frac, rho = evaluate(net, Xc, Xr, Y, M, dev)
                    lines.append(
                        f'| {fam} | {split} | {name} | {cap:.3f} '
                        f'| {reg:.3f} | {rho:.3f} | {frac:.2f} '
                        f'| {best.mean():.3f} | {Y[:, 0].mean():.3f} |')
                lines.append(
                    f'| {fam} | {split} | generator-config | {gen_cap:.3f} '
                    f'| {gen_reg:.3f} | - | - | {best.mean():.3f} '
                    f'| {Y[:, 0].mean():.3f} |')
            rep = '\n'.join(lines) + '\n'
            (out / 'report.md').write_text(rep)
            print(rep)


if __name__ == '__main__':
    main()
