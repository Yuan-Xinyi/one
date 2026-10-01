"""One control cycle of the DirFrac controller for a single configuration,
written for latency instead of batch throughput.

``NSRLBatchedEnv.step`` is built for 8192 parallel environments: each of its
dozens of small tensor operations is cheap per environment but costs tens of
microseconds of dispatch for a batch of one, and the three projected-gradient
scales in the observation are obtained by autograd through the forward
kinematics.  On one configuration that is ~9 ms on the CPU and ~22 ms on the
GPU.  This module recomputes exactly the same quantities in NumPy float64 with
closed-form derivatives (the Hessian of a serial chain's position Jacobian is
a few cross products) and shares one forward-kinematics pass between the
observation, the command and the collision check; the policy itself stays a
PyTorch forward pass.  Agreement with the environment is verified by
``probes/fast_cycle_verify.py``.

Supported configuration: dir_frac_action = 2, rho_from_norm, a_prev_executed,
observe_margins, observe_proj_scales, true_reset_obs, straight-line tasks,
any flange-frame tool point (tool_xyz / tcp_offset) with the tool axis along
the flange z, chain (xArm7 / Cobotta) or FR3 kinematics.  Anything else
raises at construction.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from Yuan.IJRR.env.env import (LATERAL_SAFETY_NET, TERM_NAMES, TERM_COLLISION,
                               TERM_CONE, TERM_JL, TERM_LATERAL)

_COLL_SCALE = 0.05      # env: m_coll = min_margin / 0.05


def _skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


class FastCycle:
    def __init__(self, env, agent, policy_device=None):
        cfg = env.cfg
        bad = [k for k in ('observe_curvature', 'observe_headroom', 'observe_ray_error',
                           'observe_prior_logits', 'observe_force', 'task_gate',
                           'observe_cone', 'observe_preview', 'observe_tool', 'alpha_joint')
               if getattr(cfg, k, False)]
        if (int(cfg.dir_frac_action) != 2 or not cfg.rho_from_norm or not cfg.a_prev_executed
                or not cfg.observe_margins or not cfg.observe_proj_scales
                or not cfg.true_reset_obs or cfg.speed_levels or cfg.obs_drop
                or int(getattr(cfg, 'a_prev_stack', 1) or 1) != 1 or cfg.basis_raw_scale
                or int(getattr(cfg, 'dv_metric', 0) or 0) or bad
                or getattr(env, '_force_on', False)):
            raise ValueError(f'FastCycle does not cover this configuration (flags {bad})')
        kin = env.kin
        f64 = lambda t: np.asarray(t.detach().cpu().numpy(), np.float64)
        self.n = int(getattr(kin, 'n_joints', 7))
        self.zero_tfs = f64(kin.zero_tfs)                      # (n, 4, 4)
        if hasattr(kin, 'axes'):
            self.axes = f64(kin.axes)                          # (n, 3) local joint axes
        else:                                                  # FR3: every joint about local z
            self.axes = np.tile(np.array([0.0, 0.0, 1.0]), (self.n, 1))
        self.K = np.stack([_skew(a) for a in self.axes])
        self.K2 = self.K @ self.K
        self.flange_p = f64(kin.flange_p)
        self.flange_R = f64(kin.flange_R)
        self.lo, self.up = f64(kin.lmt_lo), f64(kin.lmt_up)
        self.q_mid = f64(kin.q_mid)
        self.q_half = 0.5 * (self.up - self.lo)
        self.qd_lim = f64(env.qd_limit)
        self.v, self.k_lat, self.dt = float(cfg.v), float(cfg.k_lateral), float(env.dt)
        self.lambda_0, self.sigma_thr = float(cfg.lambda_0), float(cfg.sigma_thr)
        self.lam_w = float(cfg.manip_damping)
        self.cos_cone = float(env.cos_cone)
        self.tube = float(LATERAL_SAFETY_NET)
        # spheres
        col = env.collision
        self.sph_c = f64(col.centers)
        self.sph_r = f64(col.radii)
        self.sph_l = col.link_indices.cpu().numpy()
        pairs = np.argwhere(col.mask.cpu().numpy())
        self.pi, self.pj = pairs[:, 0], pairs[:, 1]
        self.pair_r = self.sph_r[self.pi] + self.sph_r[self.pj]
        self.coll_thr = float(getattr(col, 'margin', 0.0))
        # policy
        self.dev = (torch.device(policy_device) if policy_device is not None
                    else next(agent.parameters()).device)
        # own copy: the caller's agent may live on another device (batched critic)
        self.agent = copy.deepcopy(agent).to(self.dev).eval()
        self.obs_dim = env.obs_dim
        self.task = None

    # ------------------------------------------------------------ kinematics
    def fk(self, q):
        """Chain frames, tool point, tool rotation, position/rotation Jacobian,
        joint axes and joint points in the world frame."""
        n = self.n
        T = np.eye(4)
        links = np.empty((n + 1, 4, 4)); links[0] = T
        W = np.empty((n, 3)); P = np.empty((n, 3))
        for i in range(n):
            Tj = T @ self.zero_tfs[i]
            W[i] = Tj[:3, :3] @ self.axes[i]
            P[i] = Tj[:3, 3]
            c, s = math.cos(q[i]), math.sin(q[i])
            Rm = np.eye(3) + s * self.K[i] + (1.0 - c) * self.K2[i]
            T = Tj.copy()
            T[:3, :3] = Tj[:3, :3] @ Rm
            links[i + 1] = T
        p = T[:3, :3] @ self.flange_p + T[:3, 3]
        R = T[:3, :3] @ self.flange_R
        Jp = np.cross(W, p[None, :] - P).T                      # (3, n)
        return dict(links=links, p=p, R=R, Jp=Jp, Jr=W.T, W=W, P=P)

    def collision_margin(self, links):
        tf = links[self.sph_l]                                  # (S, 4, 4)
        pos = np.einsum('sij,sj->si', tf[:, :3, :3], self.sph_c) + tf[:, :3, 3]
        d = np.linalg.norm(pos[self.pi] - pos[self.pj], axis=1)
        return float((d - self.pair_r).min())

    def _dJp(self, fk):
        """Hessian of the position Jacobian: dJ[i, :, j] = d J_p[:, j] / d q_i.
        Column j is w_j x (p - p_j); joint i rotates everything after it."""
        n = self.n
        W, P, p, Jp = fk['W'], fk['P'], fk['p'], fk['Jp']
        r = p[None, :] - P                                      # (n, 3)  p - p_j
        Wi = W[:, None, :]; Wj = W[None, :, :]; rj = r[None, :, :]
        Ji = Jp.T[:, None, :]                                   # (n, 1, 3) column i
        below = np.cross(np.cross(Wi, Wj), rj) + np.cross(Wj, np.cross(Wi, rj))   # i < j
        above = np.cross(Wj, Ji)                                # i > j
        diag = np.cross(Wj, np.cross(Wj, rj))                   # i == j
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        dJ = np.where((ii < jj)[..., None], np.broadcast_to(below, (n, n, 3)),
                      np.where((ii > jj)[..., None], np.broadcast_to(above, (n, n, 3)),
                               np.broadcast_to(diag, (n, n, 3))))   # (n_i, n_j, 3)
        return np.transpose(dJ, (0, 2, 1))                      # (n_i, 3, n_j)

    def proj_scales(self, fk, q, d, n_axis, Nn):
        """|P_N g_k| for the three basis objectives (directional manipulability
        along d, tool-axis alignment, joint centring), as build_task_aligned_basis
        returns them with return_scales (order-free projected norms)."""
        Jp = fk['Jp']
        A = Jp @ Jp.T + (self.lam_w ** 2) * np.eye(3)
        y = np.linalg.solve(A, d)
        s = float(d @ y)
        b = Jp.T @ y
        dJ = self._dJp(fk)
        ds = -2.0 * np.einsum('a,iac,c->i', y, dJ, b)
        g1 = -0.5 * s ** (-1.5) * ds
        z = fk['R'][:, 2]
        g2 = fk['Jr'].T @ np.cross(z, n_axis)
        qn = (q - self.q_mid) / self.q_half
        g3 = -(2.0 / self.n) * qn / self.q_half
        G = np.stack([g1, g2, g3], 1)                           # (n, 3)
        return np.linalg.norm(Nn @ (Nn.T @ G), axis=0)

    # ------------------------------------------------------------------ task
    def reset(self, q0, d, n_axis, p0=None):
        d = np.asarray(d, np.float64); d = d / np.linalg.norm(d)
        n_axis = np.asarray(n_axis, np.float64); n_axis = n_axis / np.linalg.norm(n_axis)
        q0 = np.asarray(q0, np.float64)
        fk = self.fk(q0)
        self.task = dict(d=d, n=n_axis, p0=(fk['p'] if p0 is None else np.asarray(p0, np.float64)))
        self.a_prev = np.zeros(self.n)
        self._scales_next = self.proj_scales(fk, q0, d, n_axis, self._null(fk['Jp']))

    @staticmethod
    def _null(Jp):
        _, _, Vt = np.linalg.svd(Jp, full_matrices=True)
        return Vt[3:].T                                          # (n, n-3)

    def _frame(self, p):
        t = self.task
        delta = p - t['p0']
        along = float(delta @ t['d'])
        lat_vec = along * t['d'] - delta
        return along, lat_vec, float(np.linalg.norm(lat_vec))

    def state(self, q, fk=None):
        """Margins, progress and violations of a configuration."""
        q = np.asarray(q, np.float64)
        fk = self.fk(q) if fk is None else fk
        along, lat_vec, lat = self._frame(fk['p'])
        cos = float(fk['R'][:, 2] @ self.task['n'])
        mm = self.collision_margin(fk['links'])
        m = np.array([((self.q_half - np.abs(q - self.q_mid)) / self.q_half).min(),
                      (cos - self.cos_cone) / (1.0 - self.cos_cone),
                      (self.tube - lat) / self.tube, mm / _COLL_SCALE])
        viol = dict(collision=mm < self.coll_thr, cone=cos < self.cos_cone,
                    jl=bool(((q < self.lo) | (q > self.up)).any()), lateral=lat > self.tube)
        return dict(fk=fk, p=fk['p'], progress=along, lat_vec=lat_vec, lateral=lat, cos=cos,
                    tilt_deg=math.degrees(math.acos(max(-1.0, min(1.0, cos)))),
                    margins=m, min_margin=mm, viol=viol,
                    violations=[k for k, v in viol.items() if v])

    def observation(self, q, st):
        fk = st['fk']
        qn = (q - self.q_mid) / self.q_half
        z = fk['R'][:, 2]
        t = self.task
        return np.concatenate([qn, qn * qn, t['d'], z, t['n'], [st['cos']], np.cross(z, t['n']),
                               self.a_prev, st['margins'], self._scales_next]).astype(np.float32)

    @torch.no_grad()
    def policy(self, obs):
        o = torch.as_tensor(obs, device=self.dev)[None]
        return self.agent.actor_mean(o)[0].float().cpu().numpy().astype(np.float64)

    def step(self, q):
        """One cycle from the measured configuration q: observation, policy,
        joint velocity, and the one-step predictive check at q + qdot*dt."""
        q = np.asarray(q, np.float64)
        st = self.state(q)
        obs = self.observation(q, st)
        u = np.clip(self.policy(obs), -1.0, 1.0)
        fk = st['fk']; Jp = fk['Jp']
        # damped pseudo-inverse (Nakamura-Hanafusa adaptive damping)
        JJt = Jp @ Jp.T
        ev = np.linalg.eigvalsh(JJt)
        sig = math.sqrt(max(float(ev[0]), 0.0))
        lam = (self.lambda_0 * math.sqrt(max(1.0 - min(sig / self.sigma_thr, 1.0) ** 2, 0.0))
               if sig < self.sigma_thr else 0.0)
        Jplus = Jp.T @ np.linalg.inv(JJt + lam * lam * np.eye(3))
        x_dot = self.v * self.task['d'] + self.k_lat * st['lat_vec']
        qdot_task = Jplus @ x_dot
        # exact null projection, amplitude from the joint-velocity limits
        Nn = self._null(Jp)
        xi = Nn @ (Nn.T @ u)
        nrm = float(np.linalg.norm(xi))
        xi_dir = xi / max(nrm, 1e-6)
        rho = min(nrm, 1.0)
        head = (self.qd_lim - qdot_task) / np.maximum(xi_dir, 1e-9)
        room = (self.qd_lim + qdot_task) / np.maximum(-xi_dir, 1e-9)
        alpha = max(float(np.where(xi_dir >= 0, head, room).min()), 0.0)
        qdot_null = rho * alpha * xi_dir
        qdot = qdot_task + qdot_null
        # the scales in the NEXT observation are those of this configuration
        self._scales_next = self.proj_scales(fk, q, self.task['d'], self.task['n'], Nn)
        self.a_prev = qdot_null / self.qd_lim
        # predictive check
        q_new = q + qdot * self.dt
        st_new = self.state(q_new)
        v = st_new['viol']
        reason = (TERM_COLLISION if v['collision'] else TERM_CONE if v['cone'] else
                  TERM_JL if v['jl'] else TERM_LATERAL if v['lateral'] else 0)
        return dict(obs=obs, u=u, qdot=qdot, q_new=q_new, done=reason != 0,
                    reason=TERM_NAMES[reason], state=st, state_new=st_new,
                    rho=rho, alpha=alpha, sigma_min=sig)
