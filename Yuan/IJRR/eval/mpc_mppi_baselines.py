"""Receding-horizon MPC and MPPI baselines on the DirFrac action interface.

Both planners act through exactly the interface the learned controller uses
(dir_frac_action = 2, rho_from_norm: a joint-space direction whose exact
null-space projection is executed at the largest feasible fraction of the
residual joint-velocity budget) and are evaluated with the fam_unify
protocol (50 ms command period held over two 25 ms integration substeps,
the same per-task initial configurations, the same references
ell^pw). The only thing that differs between the rows of the table is the
decision rule:

  mpc-H    nonlinear MPC by direct single shooting: H commands are optimised
           with Adam through the differentiable model on the smoothed
           exit-time objective  sum_h A_h ds_h + w_T A_H m(q_H), where A_h is
           the soft alive probability prod_{j<=h} sigmoid(m_j / eps) of the
           softmin constraint margin m; the first command is executed and
           the shifted solution warm-starts the next period.
  mppi-H   model predictive path integral control: K perturbed command
           sequences are rolled out on the same model, scored by the exact
           truncated exit-time objective (progress until the first violation
           within H, plus the same terminal margin bonus), and combined with
           softmax weights exp(-cost / lambda); shifted mean warm-starts.

The model (`DirFracModel`) is a standalone copy of the environment's
kinematic update for this action interface; `--validate` replays random
command sequences through model and environment and reports the
discrepancy, so the planners cannot exploit anything the environment does
not have.

Usage:
    python -m Yuan.IJRR.eval.mpc_mppi_baselines --validate --robot fr3 --family serpentine
    python -m Yuan.IJRR.eval.mpc_mppi_baselines --robot fr3 --family straight \
        --method mppi --H 16,32,64 --n-tasks 10000
    python -m Yuan.IJRR.eval.mpc_mppi_baselines --robot xarm7 --family nonplanar \
        --method mpc --H 10,20,30
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import matplotlib  # noqa: E402  (must precede torch in the `one` env)
matplotlib.use('Agg')
import matplotlib.pyplot  # noqa: F401,E402
import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import (
    NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.path_geometry import path_frame

REPO = Path(__file__).resolve().parents[3]
MAIN = REPO / 'Yuan/IJRR'
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
OUT = MAIN / 'runs/paper_fill/horizon'
CFG = {'fr3': 'config_line_cont_dirfrac_e8kXXL_rm.yaml',
       'xarm7': 'config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml',
       'cobotta': 'config_line_cont_dirfrac_cobotta_e8kXXL_v005.yaml'}
SUB = 2                    # integration substeps per held 50 ms command
CHUNK = 32768              # rollout states per model call
TAU = 0.1                  # softmin temperature of the margin objective


# --------------------------------------------------------------------- tasks

def load_tasks(robot: str, family: str):
    """Task specs (q0, line_dir, n_target, [p0], family params) exactly as
    the DirFrac table rows were produced, plus the per-task reference."""
    spec = {}
    if family == 'straight':
        tz = np.load(A / f'tasks_pool_{robot}.npz')
        spec = {'q0': tz['q0_seed'], 'line_dir': tz['cs_line_dir'],
                'n_target': tz['cs_n_target']}
    elif robot == 'fr3':
        t = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)[f'test_{family}']
        spec = {k: v.numpy() for k, v in t.items()}
    else:
        tz = np.load(A / f'tasks_selx_{family}_{robot}.npz')
        spec = {'q0': tz['q0_seed'], 'line_dir': tz['cs_line_dir'],
                'n_target': tz['cs_n_target']}
        for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
            if k in tz.files:
                spec[k] = tz[k]
    ref = np.load(FU / f'ref_{robot}_{family}_mpc.npz')['ref']
    assert ref.shape[0] == spec['q0'].shape[0], (ref.shape, spec['q0'].shape)
    return spec, ref


def build_env(robot: str, family: str, n_envs: int, dev):
    y = yaml.safe_load(open(MAIN / 'stage2_traj' / CFG[robot]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= SUB
    kw['max_steps'] = int(y['env']['max_steps'] * SUB)
    if family != 'straight':
        kw['k_lateral'] = 5.0
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': n_envs}), None, dev)


def scripted(env, spec, lo, hi):
    """ScriptedLineDistribution for rows lo:hi, padded to env.n_envs."""
    B, pad, dt, dev = env.n_envs, env.n_envs - (hi - lo), env.kin.dtype, \
        env.device
    sub = {}
    for k, v in spec.items():
        t = torch.as_tensor(v[lo:hi])
        if pad:
            t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
        sub[k] = t.to(device=dev, dtype=dt)
    return ScriptedLineDistribution(sub)


# --------------------------------------------------------------------- model

def _sym3_eigmin(M: torch.Tensor) -> torch.Tensor:
    """Smallest eigenvalue of a batch of symmetric 3x3 matrices, closed form
    (no cusolver: the batched syev has a batch-size limit and a cold-start
    failure mode)."""
    a, b, c = M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]
    d, e, f = M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]
    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    p = torch.sqrt((p2 / 6.0).clamp_min(1e-30))
    Bm = (M - q.view(-1, 1, 1) * torch.eye(3, device=M.device,
                                           dtype=M.dtype)) / p.view(-1, 1, 1)
    r = (torch.linalg.det(Bm) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    phi = torch.acos(r) / 3.0
    return q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)


class DirFracModel:
    """The environment's kinematic update for dir_frac_action = 2 with
    rho_from_norm, as a pure function of (q, arc_progress) and the task.

    Static task tensors (all (B, ...)): p0, d0, n0, kappa, amp, wavelen,
    rot_axis, rot_rate. State: q (B, n), arc (B,). A command is held for
    SUB substeps of env.cfg.dt (25 ms)."""

    def __init__(self, env: NSRLBatchedEnv):
        self.env = env
        self.kin = env.kin
        self.collision = env.collision
        self.cfg = env.cfg
        self.dt = env.cfg.dt
        self.v = env.cfg.v
        self.k_lat = float(getattr(env.cfg, 'k_lateral', 0.0) or 0.0)
        self.qd_limit = env.qd_limit
        self.q_mid, self.q_half = env.q_mid, env.q_half
        self.lmt_lo, self.lmt_up = env.lmt_lo, env.lmt_up
        self.cos_cone = env.cos_cone
        self.coll_thr = float(env.collision.margin)
        self.n_joints = env.n_joints
        # Self-collision as a pair list (the env evaluates the full S x S
        # matrix and masks it; identical minimum, 4-5x less memory, which
        # matters for the 155-sphere xArm7 model inside autograd graphs).
        pi, pj = torch.nonzero(env.collision.mask, as_tuple=True)
        self._pi, self._pj = pi, pj
        self._rsum = env.collision.radii[pi] + env.collision.radii[pj]
        assert int(getattr(env.cfg, 'dir_frac_action', 0)) == 2
        assert getattr(env.cfg, 'rho_from_norm', False)
        assert not getattr(env.cfg, 'dv_metric', 0)
        assert not getattr(env.cfg, 'alpha_joint', False)
        assert not env.cfg.speed_levels and not getattr(env.cfg, 'task_gate',
                                                        False)

    @staticmethod
    def task_of(env, idx=None):
        """Static task tensors read off the environment (rows idx)."""
        f = (lambda t: t) if idx is None else (lambda t: t[idx])
        return {'p0': f(env.p_start), 'd0': f(env.path_d0),
                'n0': f(env.n0_target), 'kappa': f(env.path_kappa),
                'amp': f(env.path_amp), 'wavelen': f(env.path_wavelen),
                'rot_axis': f(env.n_rot_axis), 'rot_rate': f(env.n_rot_rate)}

    @staticmethod
    def expand(T, K):
        return {k: v.repeat_interleave(K, 0) for k, v in T.items()}

    @staticmethod
    def index(T, idx):
        return {k: v[idx] for k, v in T.items()}

    def n_at(self, T, arc):
        th = (T['rot_rate'] * arc).unsqueeze(-1)
        k, n0 = T['rot_axis'], T['n0']
        c, s = torch.cos(th), torch.sin(th)
        n = n0 * c + torch.linalg.cross(k, n0, dim=-1) * s \
            + k * (k * n0).sum(-1, keepdim=True) * (1 - c)
        return torch.where((T['rot_rate'] != 0).unsqueeze(-1), n, n0)

    def frame(self, p, T, n):
        # On a wave task path_frame's (unselected) arc branch evaluates
        # 1/kappa = inf; the forward value is masked out, but autograd
        # propagates NaN through the masked branch. A wave never reads
        # kappa, so substituting 1 there is bit-identical in the forward.
        kappa = torch.where(T['amp'].abs() > 1e-6, torch.ones_like(T['kappa']),
                            T['kappa'])
        return path_frame(p, T['p0'], T['d0'], n, kappa, T['amp'],
                          T['wavelen'])

    def margins(self, q, p, R, T, n):
        """Normalized margins (jl, cone, lat, coll); a stroke is alive iff
        all four are > 0 (identical to the environment's termination)."""
        m_jl = ((self.q_half - (q - self.q_mid).abs()) / self.q_half
                ).amin(dim=-1)
        cosv = (R[:, :, 2] * n).sum(-1).clamp(-1.0, 1.0)
        m_cone = (cosv - self.cos_cone) / (1.0 - self.cos_cone)
        _, _, lat = self.frame(p, T, n)
        m_lat = (LATERAL_SAFETY_NET - lat) / LATERAL_SAFETY_NET
        cen = self.collision.sphere_positions(self.kin.link_transforms(q))
        mm = ((cen[:, self._pi] - cen[:, self._pj]).norm(dim=-1)
              - self._rsum).amin(dim=-1)
        m_coll = (mm - self.coll_thr) / 0.05
        return torch.stack([m_jl, m_cone, m_lat, m_coll], dim=-1)

    def substep(self, q, arc, a, T):
        """One 25 ms integration step under command a (B, n) in [-1, 1].
        Returns q_new, arc_new, margins at q_new (B, 4), p_new."""
        n = self.n_at(T, arc)
        p, R, J, _ = self.kin.tcp_fk_jac(q)
        Jp = J[:, :3, :]
        tangent, lat_vec, _ = self.frame(p, T, n)
        # damped pseudo-inverse (Nakamura-Hanafusa), eigmin in closed form
        JJt = Jp @ Jp.transpose(-1, -2)
        sig = torch.sqrt(_sym3_eigmin(JJt).clamp_min(1e-12))
        ratio = (sig / self.cfg.sigma_thr).clamp(max=1.0)
        lam = torch.where(sig < self.cfg.sigma_thr,
                          self.cfg.lambda_0 * torch.sqrt(
                              (1.0 - ratio * ratio).clamp_min(1e-12)),
                          torch.zeros_like(sig))
        I3 = torch.eye(3, device=q.device, dtype=q.dtype)
        Ainv = torch.linalg.inv(JJt + (lam * lam).view(-1, 1, 1) * I3)
        J_plus = Jp.transpose(-1, -2) @ Ainv
        x_dot = (self.v * tangent + self.k_lat * lat_vec).unsqueeze(-1)
        qdot_task = (J_plus @ x_dot).squeeze(-1)
        # exact null-space projection (double, like the env's SVD)
        Jd = Jp.double()
        G = Jd @ Jd.transpose(-1, -2)
        ad = a.double().unsqueeze(-1)
        xi = ad - Jd.transpose(-1, -2) @ torch.linalg.solve(G, Jd @ ad)
        xi = xi.squeeze(-1)
        nrm = xi.norm(dim=-1, keepdim=True)
        xi_dir = (xi / nrm.clamp_min(1e-6)).to(q.dtype)
        # rho * xi_dir == xi * min(1, 1/|xi|): the same executed direction
        # as the env (rho = min(|xi|, 1)), written so that its gradient at
        # a = 0 is the identity instead of zero (the env's form is only
        # evaluated, never differentiated).
        xi_exec = (xi * (1.0 / nrm.clamp_min(1e-6)).clamp(max=1.0)
                   ).to(q.dtype)
        head = (self.qd_limit - qdot_task) / xi_dir.clamp_min(1e-9)
        room = (self.qd_limit + qdot_task) / (-xi_dir).clamp_min(1e-9)
        bound = torch.where(xi_dir >= 0, head, room)
        alpha = bound.amin(dim=-1).clamp_min(0.0)
        q_new = q + (qdot_task + alpha.unsqueeze(-1) * xi_exec) * self.dt
        p_new, R_new, _, _ = self.kin.tcp_fk_jac(q_new)
        mg = self.margins(q_new, p_new, R_new, T, n)
        is_wave = T['amp'].abs() > 1e-6
        delta = torch.where(is_wave, ((p_new - p) * T['d0']).sum(-1),
                            ((p_new - p) * tangent).sum(-1))
        return q_new, arc + delta.clamp_min(0.0), mg, p_new

    def command(self, q, arc, alive, a, T):
        """Hold a for SUB substeps; dead rollouts are frozen. Returns
        q, arc, alive, progress gained (B,), margins after the command."""
        gained = torch.zeros_like(arc)
        mg = None
        for _ in range(SUB):
            q_n, arc_n, mg_n, _ = self.substep(q, arc, a, T)
            ok = mg_n.amin(-1) > 0
            step_alive = alive & ok
            gained = gained + torch.where(step_alive, arc_n - arc,
                                          torch.zeros_like(arc))
            q = torch.where(alive.unsqueeze(-1), q_n, q)
            arc = torch.where(step_alive, arc_n, arc)
            mg = mg_n if mg is None else torch.where(
                alive.unsqueeze(-1), mg_n, mg)
            alive = step_alive
        return q, arc, alive, gained, mg


def softmin(mg, tau=TAU):
    return -tau * torch.logsumexp(-mg / tau, dim=-1)


# ------------------------------------------------------------------ planners

class MPPI:
    def __init__(self, model, H, K=64, sigma=0.5, lam=0.01, w_T=0.05,
                 seed=0):
        self.m, self.H, self.K = model, H, K
        self.sigma, self.lam, self.w_T = sigma, lam, w_T
        self.gen = torch.Generator(device=model.q_mid.device).manual_seed(seed)
        self.U = None
        self.name = f'mppi{H}'

    def reset(self, N):
        self.U = torch.zeros((N, self.H, self.m.n_joints),
                             device=self.m.q_mid.device,
                             dtype=self.m.q_mid.dtype)

    @torch.no_grad()
    def plan(self, env, live):
        """Command for every live task (idx tensor). Returns (n_live, n)."""
        m, H, K = self.m, self.H, self.K
        n = m.n_joints
        T_all = m.task_of(env, live)
        q_all, arc_all = env.q[live], env.arc_progress[live]
        U_all = self.U[live]
        out = torch.zeros((live.numel(), n), device=env.device,
                          dtype=env.q.dtype)
        C = max(1, CHUNK // K)
        for lo in range(0, live.numel(), C):
            hi = min(lo + C, live.numel())
            c = hi - lo
            U = U_all[lo:hi]                                   # (c,H,n)
            eps = self.sigma * torch.randn((c, K, H, n), generator=self.gen,
                                           device=U.device, dtype=U.dtype)
            eps[:, 0] = 0.0                                    # nominal
            Us = (U.unsqueeze(1) + eps).clamp(-1.0, 1.0)      # (c,K,H,n)
            T = m.expand(m.index(T_all, slice(lo, hi)), K)
            q = q_all[lo:hi].repeat_interleave(K, 0)
            arc = arc_all[lo:hi].repeat_interleave(K, 0)
            alive = torch.ones(c * K, dtype=torch.bool, device=q.device)
            prog = torch.zeros(c * K, device=q.device, dtype=q.dtype)
            for h in range(H):
                q, arc, alive, g, mg = m.command(
                    q, arc, alive, Us[:, :, h].reshape(c * K, n), T)
                prog = prog + g
            term = torch.where(alive, softmin(mg), torch.zeros_like(prog))
            cost = -(prog + self.w_T * term).reshape(c, K)
            w = torch.softmax(-(cost - cost.min(dim=1, keepdim=True).values)
                              / self.lam, dim=1)
            U_new = (w.view(c, K, 1, 1) * Us).sum(1)
            out[lo:hi] = U_new[:, 0]
            # shifted warm start for the next period
            self.U[live[lo:hi]] = torch.cat([U_new[:, 1:], U_new[:, -1:]], 1)
        return out


class MPC:
    def __init__(self, model, H, iters=8, lr=0.15, eps=0.05, w_T=0.05,
                 chunk=2048):
        self.m, self.H, self.iters, self.lr = model, H, iters, lr
        self.eps, self.w_T, self.chunk = eps, w_T, chunk
        self.Z = None
        self.name = f'mpc{H}'
        self.n_grad, self.n_nan = 0, 0

    def reset(self, N):
        self.Z = torch.zeros((N, self.H, self.m.n_joints),
                             device=self.m.q_mid.device,
                             dtype=self.m.q_mid.dtype)

    def objective(self, z, q0, arc0, T):
        m, H, n = self.m, self.H, self.m.n_joints
        U = torch.tanh(z)
        c = q0.shape[0]
        q, arc = q0, arc0
        A = torch.ones(c, device=q.device, dtype=q.dtype)
        J = torch.zeros(c, device=q.device, dtype=q.dtype)
        for h in range(H):
            for _ in range(SUB):
                q_n, arc_n, mg, _ = m.substep(q, arc, U[:, h], T)
                A = A * torch.sigmoid(softmin(mg) / self.eps)
                J = J + A * (arc_n - arc)
                q, arc = q_n, arc_n
        J = J + self.w_T * A * softmin(mg)
        return J

    def plan(self, env, live):
        m, n = self.m, self.m.n_joints
        T_all = m.task_of(env, live)
        q_all, arc_all = env.q[live], env.arc_progress[live]
        out = torch.zeros((live.numel(), n), device=env.device,
                          dtype=env.q.dtype)
        for lo in range(0, live.numel(), self.chunk):
            hi = min(lo + self.chunk, live.numel())
            T = m.index(T_all, slice(lo, hi))
            z = self.Z[live[lo:hi]].clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=self.lr)
            best_J, best_z = None, z.detach().clone()
            for it in range(self.iters + 1):
                with torch.enable_grad():
                    J = self.objective(z, q_all[lo:hi], arc_all[lo:hi], T)
                    Jd = J.detach()
                    if best_J is None:
                        best_J, best_z = Jd, z.detach().clone()
                    else:
                        better = Jd > best_J
                        best_J = torch.where(better, Jd, best_J)
                        best_z = torch.where(better.view(-1, 1, 1),
                                             z.detach(), best_z)
                    if it == self.iters:
                        break
                    opt.zero_grad()
                    (-J.sum()).backward()
                    g = z.grad
                    bad = ~torch.isfinite(g).flatten(1).all(1)
                    self.n_grad += g.shape[0]
                    self.n_nan += int(bad.sum())
                    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    gn = g.flatten(1).norm(dim=1).clamp_min(1e-12)
                    z.grad = g / gn.view(-1, 1, 1) * torch.minimum(
                        gn, torch.full_like(gn, 10.0)).view(-1, 1, 1)
                    opt.step()
                    with torch.no_grad():
                        z.clamp_(-3.0, 3.0)
            out[lo:hi] = torch.tanh(best_z[:, 0])
            self.Z[live[lo:hi]] = torch.cat([best_z[:, 1:], best_z[:, -1:]],
                                            1)
        return out


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def _env_state(env):
    return env.q.clone(), env.arc_progress.clone(), env.n_target.clone()


def rollout(env, planner, spec, N, log=None, max_periods=0):
    """Closed loop in the environment; returns progress (N,), wall-time
    statistics of the planning calls."""
    B = env.n_envs
    out = np.zeros(N, np.float32)
    calls, secs, tasks = 0, 0.0, 0
    t_start = time.time()
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = scripted(env, spec, lo, hi)
        env.reset()
        planner.reset(B)
        for period in range(env.cfg.max_steps // SUB):
            if max_periods and period >= max_periods:
                break
            env.current_obs()               # refresh tangent / n_target
            done = env.done_persistent.clone()
            done[hi - lo:] = True            # padded rows
            live = (~done).nonzero(as_tuple=False).squeeze(-1)
            if live.numel() == 0:
                break
            torch.cuda.synchronize()
            t0 = time.time()
            a_live = planner.plan(env, live)
            torch.cuda.synchronize()
            secs += time.time() - t0
            calls += 1
            tasks += live.numel()
            a = torch.zeros((B, env.act_dim_policy), device=env.device,
                            dtype=env.q.dtype)
            a[live] = a_live
            for _ in range(SUB):
                env.step(a, auto_reset=False)
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        if log:
            log(f'  {planner.name} {hi}/{N} mean {out[:hi].mean():.4f} '
                f'{time.time() - t_start:.0f}s')
    return out, {'calls': calls, 'plan_seconds': secs,
                 'task_periods': tasks,
                 'ms_per_task_period': 1e3 * secs / max(tasks, 1),
                 'nan_grad_frac': (planner.n_nan / max(planner.n_grad, 1)
                                   if hasattr(planner, 'n_nan') else 0.0),
                 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9}


def stat(v, ref):
    ref = np.maximum(ref, v)
    rt = v / np.maximum(ref, 1e-9)
    return float(v.mean()), float(rt.mean() * 100), \
        float(np.percentile(rt, 10) * 100)


# ---------------------------------------------------------------- validation

@torch.no_grad()
def validate(robot, family, n_tasks, dev, n_cmd=80, seed=0):
    spec, _ = load_tasks(robot, family)
    N = min(n_tasks, spec['q0'].shape[0])
    env = build_env(robot, family, N, dev)
    env.line_dist = scripted(env, spec, 0, N)
    env.reset()
    env.current_obs()
    model = DirFracModel(env)
    T = model.task_of(env)
    q, arc = env.q.clone(), env.arc_progress.clone()
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    worst_q, worst_arc, flag_mismatch, n_alive_pairs = 0.0, 0.0, 0, 0
    for t in range(n_cmd):
        scale = torch.rand((N, 1), generator=g, device=dev)
        a = (torch.rand((N, env.act_dim_policy), generator=g, device=dev)
             * 2 - 1) * scale * 1.5
        a = a.clamp(-1, 1)
        for _ in range(SUB):
            env.step(a, auto_reset=False)
        q, arc, alive, _, _ = model.command(q, arc, alive, a, T)
        env_alive = ~env.done_persistent
        both = env_alive & alive
        n_alive_pairs += int(both.sum())
        flag_mismatch += int((env_alive != alive).sum())
        if both.any():
            worst_q = max(worst_q, float((env.q[both] - q[both]).abs().max()))
            worst_arc = max(worst_arc, float(
                (env.arc_progress[both] - arc[both]).abs().max()))
        if not bool(env_alive.any()):
            break
    print(f'[validate] {robot}/{family} N={N} commands={t + 1}: '
          f'max|dq| {worst_q:.2e} rad, max|d arc| {worst_arc:.2e} m, '
          f'alive-flag mismatches {flag_mismatch} '
          f'(alive pairs {n_alive_pairs}), '
          f'final env alive {int((~env.done_persistent).sum())}', flush=True)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='fr3', choices=list(CFG))
    ap.add_argument('--family', default='straight',
                    choices=['straight', 'serpentine', 'nonplanar'])
    ap.add_argument('--method', default='mppi', choices=['mpc', 'mppi'])
    ap.add_argument('--H', default='16,32,64')
    ap.add_argument('--n-tasks', type=int, default=0,
                    help='0 = all tasks of the family')
    ap.add_argument('--batch', type=int, default=2500)
    ap.add_argument('--K', type=int, default=64, help='MPPI samples')
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--lam', type=float, default=0.01)
    ap.add_argument('--iters', type=int, default=8, help='MPC Adam steps')
    ap.add_argument('--lr', type=float, default=0.15)
    ap.add_argument('--w-T', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--chunk', type=int, default=2048, help='MPC tasks per graph')
    ap.add_argument('--max-periods', type=int, default=0, help='timing smoke only')
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    if a.validate:
        validate(a.robot, a.family, a.n_tasks or 2048, dev)
        return
    spec, ref = load_tasks(a.robot, a.family)
    N = spec['q0'].shape[0] if a.n_tasks <= 0 else min(a.n_tasks,
                                                       spec['q0'].shape[0])
    ref = ref[:N]
    OUT.mkdir(parents=True, exist_ok=True)
    B = min(a.batch, N)
    env = build_env(a.robot, a.family, B, dev)
    model = DirFracModel(env)
    log = lambda s: print(s, flush=True)
    for H in [int(h) for h in a.H.split(',')]:
        # MPC graph chunk: memory grows with H * SUB substeps and with the
        # number of collision pairs (FR3 896, xArm7 4670).
        mpc_chunk = min(a.chunk, max(256, int(
            122880 // (H * SUB) * min(1.0, 800 / model._pi.numel()))))
        planner = (MPPI(model, H, K=a.K, sigma=a.sigma, lam=a.lam,
                        w_T=a.w_T, seed=a.seed) if a.method == 'mppi'
                   else MPC(model, H, iters=a.iters, lr=a.lr, w_T=a.w_T,
                            chunk=mpc_chunk))
        name = f'{a.robot}_{a.family}_{planner.name}{a.tag}'
        if N < spec['q0'].shape[0]:
            name += f'_n{N}'
        if (OUT / f'{name}.json').exists():
            log(f'[{a.robot}/{a.family}] {planner.name}: cached, skipped')
            continue
        log(f'[{a.robot}/{a.family}] {planner.name} N={N} B={B}')
        t0 = time.time()
        prog, tm = rollout(env, planner, spec, N, log, a.max_periods)
        mean, rm, p10 = stat(prog, ref)
        np.savez_compressed(OUT / f'{name}.npz', prog=prog)
        rec = {'robot': a.robot, 'family': a.family, 'method': a.method,
               'H': H, 'N': N, 'stroke': mean, 'ratio_mean': rm,
               'ratio_p10': p10, 'wall_s': time.time() - t0, **tm,
               'args': vars(a)}
        json.dump(rec, open(OUT / f'{name}.json', 'w'), indent=1)
        log(f'[{a.robot}/{a.family}] {planner.name}: stroke {mean:.3f}  '
            f'ratio {rm:.1f} / {p10:.1f}   '
            f'{tm["ms_per_task_period"]:.2f} ms per task-period, '
            f'nan-grad {tm["nan_grad_frac"]*100:.1f}%, '
            f'peak {tm["peak_mem_gb"]:.1f} GB, {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
