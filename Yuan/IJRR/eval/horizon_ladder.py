"""The horizon ladder: a=0, classical, myopic one-step, receding CEM, policy.

Every arm runs closed loop in the environment at a 50 ms control period with
the integration refined to 25 ms (two substeps per held command), the setting
the convergence check certified; the optimizing arms plan on a standalone
model that reproduces the environment's kinematic update exactly (validated
here by replay), so the optimizer cannot exploit anything the environment
does not have.

Arms:
  zero        a = 0: no redundancy allocation at all.
  classical   the tuned null-space law.
  myopic      argmax over the 2^m vertices of the soft-min constraint margin
              one control period ahead.
  cem-H       receding-horizon cross-entropy method over continuous command
              sequences of H control periods; executes the first command and
              replans. Per-increment wall time is recorded: the comparison
              against one policy forward pass is the amortization argument.
  vertex      the trained categorical policy.

Usage:
    python -m Yuan.IJRR.eval.horizon_ladder --validate
    python -m Yuan.IJRR.eval.horizon_ladder --arms zero,classical,myopic,vertex --n-tasks 1024
    python -m Yuan.IJRR.eval.horizon_ladder --arms cem4,cem8,cem16 --n-tasks 512
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis, damped_pinv,
    LATERAL_SAFETY_NET)
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

REPO = Path(__file__).resolve().parents[3]
ROBOTS = {
    'fr3': ('Yuan/IJRR/stage2_traj/config_vertex_line.yaml',
            'Yuan/IJRR/runs/rl_vertex_line_30M'),
    'xarm7': ('Yuan/IJRR/stage2_traj/config_vertex_line_xarm7.yaml',
              'Yuan/IJRR/runs/rl_vertex_line_xarm7_30M'),
    'cobotta': ('Yuan/IJRR/stage2_traj/config_vertex_line_cobotta.yaml',
                'Yuan/IJRR/runs/rl_vertex_line_cobotta_30M'),
}
SUB = 2          # integration substeps per held command (2 -> 25 ms)


class StraightModel:
    """Standalone copy of the environment's kinematic update on straight rays.

    step() advances a batch of configurations under a held command with the
    same damped task solution, the same exact null-space basis and the same
    integration as the environment; margins() reports the four normalized
    constraint margins whose first zero ends a stroke.
    """

    def __init__(self, env):
        self.kin = env.kin
        self.collision = env.collision
        self.cfg = env.cfg
        self.q_mid = env.q_mid
        self.q_half = env.q_half
        self.cos_cone = env.cos_cone
        self.act_dim = env.act_dim

    def step(self, q, d, n, a, substeps=None):
        substeps = SUB if substeps is None else substeps
        dt = self.cfg.dt / substeps
        for _ in range(substeps):
            _, _, J, _ = self.kin.tcp_fk_jac(q)
            J_p = J[:, :3, :]
            J_plus, _ = damped_pinv(J_p, self.cfg.lambda_0,
                                    self.cfg.sigma_thr)
            B, _ = build_task_aligned_basis(
                self.kin, q, d, n, self.kin.q_mid, self.q_half,
                self.cfg.manip_damping)
            x_dot = (self.cfg.v * d).unsqueeze(-1)
            qdot = (J_plus @ x_dot).squeeze(-1) \
                + (B @ (self.cfg.a_max * a).unsqueeze(-1)).squeeze(-1)
            q = q + qdot * dt
        return q

    def margins(self, q, p0, d, n):
        p, R, _, _ = self.kin.tcp_fk_jac(q)
        m_jl = ((self.q_half - (q - self.q_mid).abs()) / self.q_half) \
            .amin(dim=-1)
        cosv = (R[:, :, 2] * n).sum(-1)
        m_cone = (cosv - self.cos_cone) / (1.0 - self.cos_cone)
        rel = p - p0
        lat = (rel - (rel * d).sum(-1, keepdim=True) * d).norm(dim=-1)
        m_lat = (LATERAL_SAFETY_NET - lat) / LATERAL_SAFETY_NET
        tfs = self.kin.link_transforms(q)
        m_coll = self.collision.min_margin(tfs) / 0.05
        return torch.stack([m_jl, m_cone, m_lat, m_coll], dim=-1)

    def softmin_margin(self, q, p0, d, n, tau=0.1):
        m = self.margins(q, p0, d, n)
        if getattr(self, 'terms', None) is not None:
            m = m[:, self.terms]
        return -tau * torch.logsumexp(-m / tau, dim=-1)


@torch.no_grad()
def rollout_env(env, action_fn, timer=None):
    """Closed-loop episode in the env: 50 ms commands, 25 ms integration."""
    env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    prog = torch.zeros(env.n_envs, device=env.device)
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    term = torch.full((env.n_envs,), -1, dtype=torch.long, device=env.device)
    blocks = env.max_steps // SUB
    for _ in range(blocks):
        t0 = time.time()
        a = action_fn(env, done)
        if timer is not None and not bool(done.all()):
            timer.append(time.time() - t0)
        for _ in range(SUB):
            _, _, _, _, info = env.step(a, auto_reset=False)
            new = info['episode_done']
            if new.any():
                term[new] = info['term_reason'][new]
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = torch.where(done, prog, ((p - p0) * u).sum(-1))
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return prog.cpu().numpy().copy(), term.cpu().numpy().copy()


def make_myopic(model):
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * model.act_dim, indexing='ij'),
                 -1).reshape(-1, model.act_dim), dtype=torch.float32,
        device=model.q_mid.device)

    @torch.no_grad()
    def fn(env, done):
        N, K = env.n_envs, verts.shape[0]
        q = env.q.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)
        d = env.line_dir.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
        n = env.n_target.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
        p0 = env.p_start.unsqueeze(1).expand(-1, K, -1).reshape(N * K, 3)
        a = verts.unsqueeze(0).expand(N, -1, -1).reshape(N * K, -1)
        qn = model.step(q, d, n, a)
        M = model.softmin_margin(qn, p0, d, n).reshape(N, K)
        return verts[M.argmax(dim=-1)]
    return fn


def make_sgngrad(model):
    """a = sgn(B^T grad Phi(q)) at the CURRENT state: no lookahead, no
    model rollout. Same information class as the classical law (a gradient of
    a scalar field of the current configuration); only the field differs."""
    @torch.no_grad()
    def fn(env, done):
        q = env.q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            _, R, _, _ = env.kin.tcp_fk_jac(q)
            m_jl = ((model.q_half - (q - model.q_mid).abs())
                    / model.q_half).amin(dim=-1)
            cosv = (R[:, :, 2] * env.n_target).sum(-1)
            m_cone = (cosv - model.cos_cone) / (1.0 - model.cos_cone)
            tau = 0.1
            phi = -tau * torch.logsumexp(
                -torch.stack([m_jl, m_cone], dim=-1) / tau, dim=-1)
            g, = torch.autograd.grad(phi.sum(), q)
        B, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        sigma = torch.einsum('bij,bi->bj', B, g)
        s = torch.sign(sigma)
        return torch.where(s == 0, torch.ones_like(s), s)
    return fn


def make_sgnclassical(classical):
    """The classical law's direction with the magnitude discarded: separates
    'wrong field' from 'wrong magnitude'."""
    base = cn_action_fn(classical)

    @torch.no_grad()
    def fn(env, done):
        s = torch.sign(base(env))
        return torch.where(s == 0, torch.ones_like(s), s)
    return fn


def make_margin_tree2(model):
    """Exhaustive two-step vertex tree on the margin objective: maximize the
    worst softmin margin along the two-step path. Same objective as myopic,
    one more step of horizon."""
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]
    pairs_a = verts.repeat_interleave(K, dim=0)     # (K*K, m) first step
    pairs_b = verts.repeat(K, 1)                    # (K*K, m) second step

    @torch.no_grad()
    def fn(env, done):
        N = env.n_envs
        P = K * K
        q = env.q.unsqueeze(1).expand(-1, P, -1).reshape(N * P, -1)
        d = env.line_dir.unsqueeze(1).expand(-1, P, -1).reshape(N * P, 3)
        n = env.n_target.unsqueeze(1).expand(-1, P, -1).reshape(N * P, 3)
        p0 = env.p_start.unsqueeze(1).expand(-1, P, -1).reshape(N * P, 3)
        a1 = pairs_a.unsqueeze(0).expand(N, -1, -1).reshape(N * P, -1)
        a2 = pairs_b.unsqueeze(0).expand(N, -1, -1).reshape(N * P, -1)
        q1 = model.step(q, d, n, a1)
        M1 = model.softmin_margin(q1, p0, d, n)
        q2 = model.step(q1, d, n, a2)
        M2 = model.softmin_margin(q2, p0, d, n)
        score = torch.minimum(M1, M2).reshape(N, P)
        best = score.argmax(dim=-1)
        return pairs_a.reshape(K * K, -1)[best % (K * K)].reshape(N, -1)             if False else verts[(best // K)]
    return fn


def make_cem(model, H, pop=64, iters=3, elite=8, objective='progress'):
    @torch.no_grad()
    def fn(env, done):
        live = (~done).nonzero(as_tuple=False).squeeze(-1)
        N_all = env.n_envs
        out = torch.zeros((N_all, model.act_dim), device=env.device)
        if live.numel() == 0:
            return out
        N = live.numel()
        d = env.line_dir[live]
        n = env.n_target[live]
        p0 = env.p_start[live]
        q0 = env.q[live]
        mu = torch.zeros((N, H, model.act_dim), device=env.device)
        sd = torch.full((N, H, model.act_dim), 0.6, device=env.device)
        for _ in range(iters):
            samp = (mu.unsqueeze(1)
                    + sd.unsqueeze(1) * torch.randn(
                        (N, pop, H, model.act_dim), device=env.device)
                    ).clamp(-1.0, 1.0)                     # (N,P,H,m)
            q = q0.unsqueeze(1).expand(-1, pop, -1).reshape(N * pop, -1)
            dR = d.unsqueeze(1).expand(-1, pop, -1).reshape(N * pop, 3)
            nR = n.unsqueeze(1).expand(-1, pop, -1).reshape(N * pop, 3)
            pR = p0.unsqueeze(1).expand(-1, pop, -1).reshape(N * pop, 3)
            alive = torch.ones(N * pop, device=env.device)
            score = torch.zeros(N * pop, device=env.device)
            worst = torch.full((N * pop,), 1e6, device=env.device)
            p_prev, _, _, _ = model.kin.tcp_fk_jac(q)
            for h in range(H):
                a_h = samp[:, :, h, :].reshape(N * pop, -1)
                q = model.step(q, dR, nR, a_h)
                m = model.margins(q, pR, dR, nR).amin(dim=-1)
                alive = alive * (m > 0).float()
                sm = model.softmin_margin(q, pR, dR, nR)
                worst = torch.minimum(worst, sm)
                p_now, _, _, _ = model.kin.tcp_fk_jac(q)
                score = score + alive * ((p_now - p_prev) * dR).sum(-1)
                p_prev = p_now
            if objective == 'margin':
                score = worst
            score = score.reshape(N, pop)
            top = score.topk(elite, dim=-1).indices          # (N,E)
            el = torch.gather(
                samp, 1, top.view(N, elite, 1, 1).expand(-1, -1, H,
                                                         model.act_dim))
            mu = el.mean(dim=1)
            sd = el.std(dim=1).clamp_min(0.05)
        out[live] = mu[:, 0, :].clamp(-1.0, 1.0)
        return out
    return fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='zero,classical,myopic,vertex')
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--robot', default='fr3', choices=list(ROBOTS))
    ap.add_argument('--sub', type=int, default=2,
                    help='integration substeps per 50 ms command (1 or 2)')
    ap.add_argument('--margin-terms', default='jl,cone,lat,coll',
                    help='margin components the myopic objective may see')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    global SUB
    SUB = a.sub
    CFG, CKPT = ROBOTS[a.robot]
    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / CFG))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] = kw['dt'] / SUB
    kw['max_steps'] = int(y['env']['max_steps'] * SUB)
    N = a.n_tasks

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    # NOTE: env.cfg.dt is now 25 ms; the model must plan on the 50 ms command
    # period, so hand it a cfg view with the command-period dt.
    model = StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    names = ['jl', 'cone', 'lat', 'coll']
    sel = [names.index(t) for t in a.margin_terms.split(',')]
    model.terms = sel if len(sel) < 4 else None

    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}
    fresh = lambda: setattr(env, 'line_dist', ScriptedLineDistribution(
        {k: v.clone() for k, v in spec.items()}))

    if a.validate:
        # Replay identical random vertex commands through model and env; the
        # configurations must match to float tolerance while both live.
        fresh(); env.reset()
        g = torch.Generator(device='cpu').manual_seed(0)
        q_model = env.q.clone()
        errs = []
        for _ in range(40):
            av = torch.randint(0, 2, (N, env.act_dim), generator=g,
                               dtype=torch.float32).to(dev) * 2 - 1
            q_model = model.step(q_model, env.line_dir, env.n_target, av)
            for _ in range(SUB):
                env.step(av, auto_reset=False)
            live = ~env.done_persistent
            if live.any():
                errs.append((q_model[live] - env.q[live]).abs().max().item())
        print(f'model-vs-env replay: max |dq| over 40 blocks = {max(errs):.2e}')
        return

    classical = ClassicalNullspaceController(env.kin)
    arms = {}
    for name in a.arms.split(','):
        if name == 'zero':
            arms[name] = lambda e, dn: torch.zeros(
                (e.n_envs, e.act_dim), device=e.device)
        elif name == 'classical':
            fcl = cn_action_fn(classical)
            arms[name] = lambda e, dn, f=fcl: f(e)
        elif name == 'myopic':
            arms[name] = make_myopic(model)
        elif name.startswith('mcem'):
            arms[name] = make_cem(model, H=int(name[4:]), objective='margin')
        elif name.startswith('cem'):
            arms[name] = make_cem(model, H=int(name[3:]))
        elif name == 'sgngrad':
            arms[name] = make_sgngrad(model)
        elif name == 'sgnclassical':
            arms[name] = make_sgnclassical(classical)
        elif name == 'mtree2':
            arms[name] = make_margin_tree2(model)
        elif name == 'vertex':
            ag = _agent(REPO / CKPT, env.obs_dim, dev, act_dim=env.act_dim)
            arms[name] = lambda e, dn, g_=ag: g_.actor_mean(e.current_obs())
        else:
            raise ValueError(name)

    fresh(); env.reset()
    base = None
    results, terms = {}, {}
    for name, fn in arms.items():
        fresh()
        timer = []
        prog, term = rollout_env(env, fn, timer)
        results[name] = prog
        terms[name] = term
        ms = 1000 * float(np.mean(timer)) if timer else 0.0
        if base is None and name == 'classical':
            base = prog
        print(f'{name:<10s} mean progress {prog.mean():.4f} m   '
              f'per-increment compute {ms:7.1f} ms')

    if 'classical' in results:
        b = results['classical']; ok = b > 1e-6
        print()
        for name, p in results.items():
            print(f'{name:<10s} ratio to classical '
                  f'{(p[ok] / b[ok]).mean():.4f}')
    if a.out:
        np.savez_compressed(
            a.out, **{f'{k}_progress': v for k, v in results.items()},
            **{f'{k}_term': v for k, v in terms.items()})
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
