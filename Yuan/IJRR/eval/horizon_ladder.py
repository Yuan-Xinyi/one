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
        CH = 32768                     # batched-eigvalsh limit
        qn = torch.cat([model.step(q[i:i + CH], d[i:i + CH], n[i:i + CH],
                                   a[i:i + CH])
                        for i in range(0, N * K, CH)])
        M = torch.cat([model.softmin_margin(qn[i:i + CH], p0[i:i + CH],
                                            d[i:i + CH], n[i:i + CH])
                       for i in range(0, N * K, CH)]).reshape(N, K)
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


def make_qsbe(env):
    """argmax_v Q(s, v) of a fitted margin Q. The checkpoint path comes
    from $QSBE_CKPT (default the discounted-SBE net), the head to execute
    from $QSBE_HEAD (finite-horizon checkpoints; 1-indexed, default the
    deepest). One forward pass either way."""
    import os
    from Yuan.IJRR.stage2_traj.qsbe import QNet, DuelingQNet
    from Yuan.IJRR.stage2_traj.wfield import obs27
    path = os.environ.get('QSBE_CKPT', 'Yuan/IJRR/runs/qsbe/q_net.pt')
    ck = torch.load(REPO / path, map_location=env.device, weights_only=False)
    heads = ck.get('heads', 1)
    cls_ = DuelingQNet if ck.get('mode') == 'dq' else QNet
    net = cls_(ck['in_dim'], ck['n_act'], heads=heads).to(env.device)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    head = int(os.environ.get('QSBE_HEAD', heads)) - 1 if heads > 1 else None
    m = env.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32, device=env.device)

    @torch.no_grad()
    def fn(env_, done):
        o = obs27(env_.q, env_.line_dir, env_.n_target,
                  env_.kin.q_mid, env_.q_half, env_.kin)
        q = net(o, head=head) if heads > 1 else net(o)
        return verts[q.argmax(-1)]
    return fn


def make_beam(model, width, H, chunk=32768):
    """Beam search over vertex sequences, scored by the worst softmin
    margin along the path (dead branches score -inf). width = 2^m at depth
    1 reproduces the myopic arm; unlimited width equals exhaustive search.
    A deterministic, sampling-free long-horizon optimizer on the margin
    objective, chunked to stay under the batched-eigvalsh limit."""
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]

    def _chunk_step(q, d, n, a):
        return torch.cat([model.step(q[i:i + chunk], d[i:i + chunk],
                                     n[i:i + chunk], a[i:i + chunk])
                          for i in range(0, q.shape[0], chunk)])

    def _chunk_margin(q, p0, d, n):
        return torch.cat([model.softmin_margin(q[i:i + chunk],
                                               p0[i:i + chunk],
                                               d[i:i + chunk],
                                               n[i:i + chunk])
                          for i in range(0, q.shape[0], chunk)])

    @torch.no_grad()
    def fn(env, done):
        N = env.n_envs
        dev = env.device
        q = env.q.unsqueeze(1)                       # (N, W, 7)
        worst = torch.full((N, 1), 1e6, device=dev)
        first = torch.zeros((N, 1), dtype=torch.long, device=dev)
        for h in range(H):
            Wc = q.shape[1]
            qe = q.unsqueeze(2).expand(-1, -1, K, -1).reshape(N * Wc * K, -1)
            ae = verts.view(1, 1, K, m).expand(N, Wc, -1, -1).reshape(
                N * Wc * K, m)
            de = env.line_dir.view(N, 1, 1, 3).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            ne = env.n_target.view(N, 1, 1, 3).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            pe = env.p_start.view(N, 1, 1, 3).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            qn = _chunk_step(qe, de, ne, ae)
            mg = _chunk_margin(qn, pe, de, ne).reshape(N, Wc, K)
            wn = torch.minimum(worst.unsqueeze(-1).expand(-1, -1, K), mg)
            wn = torch.where(mg > 0, wn, torch.full_like(wn, -1e9))
            if h == 0:
                fn_new = torch.arange(K, device=dev).view(1, 1, K).expand(
                    N, Wc, -1)
            else:
                fn_new = first.unsqueeze(-1).expand(-1, -1, K)
            wn = wn.reshape(N, Wc * K)
            fn_new = fn_new.reshape(N, Wc * K)
            qn = qn.reshape(N, Wc * K, -1)
            Wk = min(width, Wc * K)
            top = wn.topk(Wk, dim=-1).indices
            worst = torch.gather(wn, 1, top)
            first = torch.gather(fn_new, 1, top)
            q = torch.gather(qn, 1, top.unsqueeze(-1).expand(
                -1, -1, qn.shape[-1]))
        best = worst.argmax(dim=1)
        return verts[first[torch.arange(N, device=dev), best]]
    return fn


def _obs_of(env, q, d, n, a_prev):
    """The environment's observation, built analytically for arbitrary states.

    Mirrors NSRLBatchedEnv._compute_obs for the straight-ray configs (no
    curvature / ray-error / prior-logit channels); verified against
    env.current_obs() at startup whenever the vlook arm is used.
    """
    p_tcp, R, _, _ = env.kin.tcp_fk_jac(q)
    z_tool = R[:, :, 2]
    q_norm = (q - env.q_mid) / env.q_half
    cos_angle = (z_tool * n).sum(-1, keepdim=True)
    return torch.cat([q_norm, q_norm * q_norm, d, z_tool, n, cos_angle,
                      torch.linalg.cross(z_tool, n, dim=-1), a_prev], -1)


def make_vlook(model, env, agent, chunk=32768):
    """One-step lookahead ranked by the LEARNED VALUE: enumerate all 2^m
    vertex commands with the exact model, drop the ones whose successor
    violates a constraint, and take the survivor the critic scores highest.

    Same model budget as the myopic margin law (2^m successor evaluations
    per command); the only difference is that the successor is scored by the
    policy's value head instead of a handcrafted margin potential.
    """
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]

    checked = []

    @torch.no_grad()
    def fn(e, done):
        if not checked:          # the analytic obs must equal the env's
            err = float((e.current_obs()
                         - _obs_of(e, e.q, e.line_dir, e.n_target,
                                   e.a_prev)).abs().max())
            print(f'[vlook] analytic-obs check: max |dobs| = {err:.2e}',
                  flush=True)
            assert err < 1e-4, 'analytic observation does not match the env'
            checked.append(True)
        B = e.n_envs
        qe = e.q.repeat_interleave(K, 0)
        ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(B * K, m)
        de = e.line_dir.repeat_interleave(K, 0)
        ne = e.n_target.repeat_interleave(K, 0)
        pe = e.p_start.repeat_interleave(K, 0)
        qn = torch.cat([model.step(qe[i:i + chunk], de[i:i + chunk],
                                   ne[i:i + chunk], ae[i:i + chunk])
                        for i in range(0, B * K, chunk)])
        mg = torch.cat([model.margins(qn[i:i + chunk], pe[i:i + chunk],
                                      de[i:i + chunk], ne[i:i + chunk])
                        for i in range(0, B * K, chunk)])
        alive = (mg.amin(-1) > 0).reshape(B, K)
        v = torch.cat([agent.critic(
            _obs_of(e, qn[i:i + chunk], de[i:i + chunk], ne[i:i + chunk],
                    ae[i:i + chunk])).squeeze(-1)
            for i in range(0, B * K, chunk)]).reshape(B, K)
        v = torch.where(alive, v, torch.full_like(v, -1e9))
        return verts[v.argmax(-1)]
    return fn


def make_vbeam(model, env, agent, width, H, chunk=32768):
    """Multi-step version of make_vlook: a beam over vertex sequences kept
    alive by the exact model and pruned by the learned value, returning the
    first action of the best surviving leaf. H=1, width=1 reduces to vlook.

    Tests whether horizon buys anything on the LEARNED scalar — on the
    handcrafted margin it does not (the exact two-step tree is worth +0.003).
    """
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]

    def _value(e, q, d, n, a):
        return torch.cat([agent.critic(
            _obs_of(e, q[i:i + chunk], d[i:i + chunk], n[i:i + chunk],
                    a[i:i + chunk])).squeeze(-1)
            for i in range(0, q.shape[0], chunk)])

    @torch.no_grad()
    def fn(e, done):
        B = e.n_envs
        dev_ = e.q.device
        q = e.q.unsqueeze(1)                       # (B, W, dof)
        first = torch.zeros((B, 1), dtype=torch.long, device=dev_)
        val = torch.zeros((B, 1), device=dev_)
        for h in range(H):
            Wc = q.shape[1]
            qe = q.unsqueeze(2).expand(-1, -1, K, -1).reshape(B * Wc * K, -1)
            ae = verts.view(1, 1, K, m).expand(B, Wc, -1, -1).reshape(-1, m)
            de = e.line_dir.view(B, 1, 1, -1).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            ne = e.n_target.view(B, 1, 1, -1).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            pe = e.p_start.view(B, 1, 1, -1).expand(
                -1, Wc, K, -1).reshape(-1, 3)
            qn = torch.cat([model.step(qe[i:i + chunk], de[i:i + chunk],
                                       ne[i:i + chunk], ae[i:i + chunk])
                            for i in range(0, qe.shape[0], chunk)])
            mg = torch.cat([model.margins(qn[i:i + chunk], pe[i:i + chunk],
                                          de[i:i + chunk], ne[i:i + chunk])
                            for i in range(0, qe.shape[0], chunk)])
            alive = (mg.amin(-1) > 0)
            v = _value(e, qn, de, ne, ae)
            v = torch.where(alive, v, torch.full_like(v, -1e9))
            v = v.reshape(B, Wc * K)
            fn_new = (torch.arange(K, device=dev_).view(1, 1, K)
                      .expand(B, Wc, -1) if h == 0
                      else first.unsqueeze(-1).expand(-1, -1, K))
            fn_new = fn_new.reshape(B, Wc * K)
            qn = qn.reshape(B, Wc * K, -1)
            Wk = min(width, Wc * K)
            top = v.topk(Wk, dim=-1).indices
            val = torch.gather(v, 1, top)
            first = torch.gather(fn_new, 1, top)
            q = torch.gather(qn, 1, top.unsqueeze(-1).expand(
                -1, -1, qn.shape[-1]))
        best = val.argmax(dim=1)
        return verts[first[torch.arange(B, device=dev_), best]]
    return fn


def make_hybrid(env, agent, classical, tau_enter, tau_exit):
    """The deployed RL/classical switching law (variant B): hand over to the
    classical controller once any joint passes tau_enter of its range, hand
    back below tau_exit."""
    base = cn_action_fn(classical)
    state = {}

    @torch.no_grad()
    def fn(e, done):
        qn = ((e.q - e.q_mid) / e.q_half).abs().max(dim=-1).values
        using_rl = state.get('using_rl')
        if using_rl is None or using_rl.shape[0] != qn.shape[0]:
            using_rl = qn < tau_enter
        stay = torch.where(using_rl, qn < tau_enter, qn < tau_exit)
        state['using_rl'] = stay
        return torch.where(stay.unsqueeze(-1),
                           agent.actor_mean(e.current_obs()).clamp(-1.0, 1.0),
                           base(e))
    return fn


def make_learnedW(env):
    """a = sgn(B^T grad W_theta(q, c)): the learned value field consumed
    exactly like the handcrafted margin field in make_sgngrad. The gradient
    is taken through the same state-task encoding the field was trained on;
    nothing else differs from the analytic law, so the comparison isolates
    the field."""
    from Yuan.IJRR.stage2_traj.wfield import WNet, obs27
    ck = torch.load(REPO / 'Yuan/IJRR/runs/w_field/w_field.pt',
                    map_location=env.device, weights_only=False)
    net = WNet(ck['in_dim']).to(env.device)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    mu = ck['mu'].to(env.device)
    sd = ck['sd'].to(env.device)

    @torch.no_grad()
    def fn(env_, done):
        q = env_.q.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            o = obs27(q, env_.line_dir, env_.n_target,
                      env_.kin.q_mid, env_.q_half, env_.kin)
            w = net((o - mu) / sd)
            g, = torch.autograd.grad(w.sum(), q)
        B, _ = build_task_aligned_basis(
            env_.kin, env_.q, env_.line_dir, env_.n_target,
            env_.kin.q_mid, env_.q_half, env_.cfg.manip_damping)
        s = torch.sign(torch.einsum('bij,bi->bj', B, g))
        return torch.where(s == 0, torch.ones_like(s), s)
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


def _value_agent(a, env, dev):
    """The network whose critic scores successors for the vlook/vbeam arms.

    Default: the deployed policy checkpoint of this robot. --vlook-ckpt
    swaps in another agent directory (training-recipe ablation), 'random'
    an untrained network (floor), and --vlook-value a standalone value MLP
    fitted outside PPO (does the gain need PPO, or only a good value?).
    """
    import torch.nn as nn
    if a.vlook_value:
        net = nn.Sequential(nn.Linear(env.obs_dim, 512), nn.ReLU(),
                            nn.Linear(512, 512), nn.ReLU(),
                            nn.Linear(512, 512), nn.ReLU(),
                            nn.Linear(512, 1)).to(dev)
        net.load_state_dict(torch.load(REPO / a.vlook_value,
                                       map_location=dev))
        net.eval()

        class _W:
            critic = net
        print(f'[vlook] standalone value net {a.vlook_value}')
        return _W()
    if a.vlook_ckpt == 'random':
        from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
        print('[vlook] UNTRAINED critic (floor control)')
        return VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                           hidden_dim=512).to(dev).eval()
    ck = REPO / (a.vlook_ckpt or ROBOTS[a.robot][1])
    if a.vlook_ckpt:
        print(f'[vlook] critic from {a.vlook_ckpt}')
    return _agent(ck, env.obs_dim, dev, act_dim=env.act_dim)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg-override', default=None,
                    help='env yaml overriding the robot default')
    ap.add_argument('--ckpt-override', default=None,
                    help='agent dir overriding the robot default')
    ap.add_argument('--vlook-ckpt', default=None,
                    help="agent dir whose critic scores successors, or "
                         "'random' for an untrained floor")
    ap.add_argument('--vlook-value', default=None,
                    help='standalone value MLP state dict (non-PPO ablation)')
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
    if a.cfg_override:
        CFG = a.cfg_override
    if a.ckpt_override:
        CKPT = a.ckpt_override
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
        elif name == 'learnedW':
            arms[name] = make_learnedW(env)
        elif name == 'qsbe':
            arms[name] = make_qsbe(env)
        elif name.startswith('beam'):
            w_, h_ = name[4:].split('x')
            arms[name] = make_beam(model, int(w_), int(h_))
        elif name == 'vertex':
            ag = _agent(REPO / CKPT, env.obs_dim, dev, act_dim=env.act_dim)
            arms[name] = lambda e, dn, g_=ag: g_.actor_mean(e.current_obs())
        elif name == 'vlook':
            arms[name] = make_vlook(model, env, _value_agent(a, env, dev))
        elif name.startswith('vbeam'):
            w_, h_ = name[5:].split('x')
            arms[name] = make_vbeam(model, env, _value_agent(a, env, dev),
                                    int(w_), int(h_))
        elif name.startswith('hybrid'):
            ag = _agent(REPO / CKPT, env.obs_dim, dev, act_dim=env.act_dim)
            te, tx = ((float(x) for x in name[6:].split('_'))
                      if '_' in name[6:] else (0.98, 0.94))
            arms[name] = make_hybrid(env, ag, classical, te, tx)
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
