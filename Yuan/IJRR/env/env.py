"""Torch-batched Gymnasium-style env for FR3 position-only NSRL path-following.

The task path is an unbounded Cartesian path given by an extension rule:
origin p_0, initial tangent d, plane normal n_target, and a signed curvature
kappa [1/m]. kappa = 0 is the infinite ray of the original experiments — no
length, no success terminate; the agent's only objective is to maximize the
arc length travelled before a safety constraint is violated. kappa != 0 is a
constant-curvature arc in the plane through p_0 spanned by (d, n_target x d);
see env/path_geometry.py.

Reward (P0 progress-only):
    r_t = w_progress · clip(Δp·T / (v·dt), 0, 1)   ∈ [0, w_progress]
with T the instantaneous path tangent (= u_hat when kappa = 0).

Per env state:
    q              (7,)  current joint config
    line_dir       (3,)  INSTANTANEOUS path tangent (unit, world frame);
                         refreshed every step, constant when kappa = 0
    path_d0        (3,)  reset-time tangent, i.e. the task descriptor d
    path_kappa     ()    signed curvature of the path [1/m]
    n_target       (3,)  target normal for 30° cone (unit, world frame)
    arc_progress   ()    arc length travelled this episode [m]
    t              ()    step counter
    a_prev         (4,)  last policy action ∈ [-1,1]^4
    done_persistent ()   True once env terminated/truncated (auto_reset=False mode)
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import torch

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

from Yuan.IJRR.kinematics.batched_chain_kin import BatchedChainKinematics
from Yuan.IJRR.kinematics.chain_sphere_collision import ChainSphereCollision

from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.env.path_geometry import path_frame, serpentine_curvature


OBS_DIM = 31
RAY_ERROR_DIM = 3
ACT_DIM = 4

# Framing B: lateral deviation is controlled by a task-space PD term in the
# controller (see step()), not by reward or a tight termination cap. This
# bound is a safety net — only trips if PD feedback fails entirely.
LATERAL_SAFETY_NET = 0.02
# Curvature is order 4 1/m on the tightest sampled task; scaling keeps the
# observation order-one like every other channel.
CURV_SCALE = 0.25


@dataclass
class EnvConfig:
    # Which arm: 'fr3' keeps the original FR3 classes; 'xarm7' / 'cobotta'
    # use the generic chain kinematics and the generated sphere sets. The
    # action and observation dimensions follow from the arm (m = n - 3).
    robot: str = "fr3"
    n_envs: int = 128
    dt: float = 0.01
    v: float = 0.05
    a_max: float = 0.5
    lambda_0: float = 0.05
    sigma_thr: float = 0.05
    cone_deg: float = 30.0
    max_steps: int = 10000
    tcp_offset: float = 0.0
    # Terminal penalties (unified to 0; lifetime alone reflects performance)
    r_collision: float = 0.0
    r_cone: float = 0.0
    r_jl: float = 0.0
    # Progress-only shaping reward.
    w_progress: float = 1.0
    # Non-empty: the policy also chooses a tangential speed each step, as a
    # trailing action channel holding one of these fractions of cfg.v. The
    # progress reward stays normalized by the FULL v, so a half-speed step
    # earns half the reward and the return still equals arc length: slowing
    # down is priced, surviving longer pays for it.
    speed_levels: tuple = ()
    # Potential-based margin shaping: r += w_margin*(margin_gamma*phi' - phi)
    # with phi = -tau*logsumexp(-m/tau) over the normalized joint-limit and
    # cone margins -- the two components the myopic one-step ablation showed
    # carry the whole effect. Potential-based, so the optimal policy of the
    # exit-time objective is unchanged (the sum telescopes); what changes is
    # the learning signal, which under progress-only reward is informative
    # only at the moment of death. margin_gamma must equal the PPO gamma.
    w_margin: float = 0.0
    margin_gamma: float = 0.99
    margin_tau: float = 0.1
    # Append the 2^m analytic prior scores sigma_margin^T v to the
    # observation, one per vertex, so a policy can treat the margin-gradient
    # law as a prior and learn deviations from it.
    observe_prior_logits: bool = False
    manip_damping: float = 1e-3
    # Task-space proportional feedback on the distance to the path, in 1/s.
    # On a straight ray the damped pseudo-inverse alone keeps the realized TCP
    # velocity on-axis, so the submitted results were produced with pure
    # feed-forward (k_lateral = 0). On a curved path pure feed-forward along
    # the instantaneous tangent cuts the corner and the error accumulates with
    # arc length, so this term must be enabled. Default 0 keeps every existing
    # checkpoint/pipeline bit-identical.
    k_lateral: float = 0.0
    # Append normalized lateral(p_tcp - p_start) to the historical 31-D
    # observation. Disabled by default so existing checkpoints remain compatible.
    observe_ray_error: bool = False
    # Local signed curvature of the path at the closest point. On a wave it
    # changes sign within an episode, so unlike a constant-curvature arc it
    # cannot be inferred from a few steps of experience and has to be observed.
    observe_curvature: bool = False


def damped_pinv(J_p: torch.Tensor, lambda_0: float, sigma_thr: float):
    """Position-only damped pseudo-inverse with Nakamura-Hanafusa adaptive λ.

    J_p: (B, 3, 7). Returns J_p^+ (B, 7, 3) and σ_min(J_p) (B,).
    """
    JJt = J_p @ J_p.transpose(-1, -2)
    eig = torch.linalg.eigvalsh(JJt)
    sigma_min = torch.sqrt(eig[..., 0].clamp(min=0.0))
    below = sigma_min < sigma_thr
    ratio = (sigma_min / sigma_thr).clamp(max=1.0)
    lam_below = lambda_0 * torch.sqrt((1.0 - ratio * ratio).clamp(min=0.0))
    lam = torch.where(below, lam_below, torch.zeros_like(sigma_min))
    I = torch.eye(3, device=J_p.device, dtype=J_p.dtype).expand_as(JJt)
    A = JJt + (lam * lam).view(-1, 1, 1) * I
    Ainv = torch.linalg.inv(A)
    return J_p.transpose(-1, -2) @ Ainv, sigma_min


def build_task_aligned_basis(
    kin,
    q: torch.Tensor,
    u_hat: torch.Tensor,
    n_target: torch.Tensor,
    q_mid: torch.Tensor,
    q_half: torch.Tensor,
    lam_w_u: float,
    eps_abs: float = 1e-8,
    eps_rel: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Task-aligned modified Gram-Schmidt nullspace basis.

        e_0 ∝ N N^T ∇w_u
        e_1 ⊥ e_0     in  N N^T ∇cos(z, n)  residual
        e_2 ⊥ {e_0,e_1} in N N^T ∇(-mean qn²) residual
        e_3 ⊥ {e_0,e_1,e_2}  in span(N)

    Fallback (when MGS step yields ‖v‖ < max(eps_abs, eps_rel·‖g_i‖)): take
    smallest unused SVD nullspace column, orthogonalize against {e_<k}, then
    sign-anchor on the residual to keep <e_k, g_k> ≥ 0 (Option B).

    Returns:
        B_basis: (B, 7, 4)
        fb_mask: (B, 3) bool — fallback triggered for e_0, e_1, e_2.
    """
    B_size, d = q.shape[0], q.shape[-1]
    device, dtype = q.device, q.dtype

    with torch.enable_grad():
        q_g = q.detach().requires_grad_(True)
        _, R, J, _ = kin.tcp_fk_jac(q_g)
        J_p = J[:, :3, :]
        eye3 = torch.eye(3, device=device, dtype=dtype).expand(B_size, 3, 3)
        JJt = J_p @ J_p.transpose(-1, -2) + (lam_w_u ** 2) * eye3
        u_col = u_hat.unsqueeze(-1)
        inv_quad = (u_col.transpose(-1, -2)
                    @ torch.linalg.solve(JJt, u_col)
                    ).squeeze(-1).squeeze(-1).clamp_min(1e-12)
        w_u = inv_quad.pow(-0.5)
        cos_val = (R[:, :, 2] * n_target).sum(-1)
        g1 = torch.autograd.grad(w_u.sum(), q_g, retain_graph=True)[0].detach()
        g2 = torch.autograd.grad(cos_val.sum(), q_g)[0].detach()
        J_p_det = J_p.detach()

    qn = (q - q_mid) / q_half
    g3 = -(2.0 / d) * qn / q_half  # ∇_q[-mean(qn²)] analytic

    # Promote to fp64 for SVD null-space identification + MGS. Fp32 SVD
    # leaks span(N)-orthogonality at ~1e-6, which compounds with ‖g_i‖ to
    # give ~1e-5 cross-coupling in the basis. Fp64 drops that to ~1e-14.
    J_p_d = J_p_det.double()
    g1d, g2d, g3d = g1.double(), g2.double(), g3.double()

    m = d - 3   # null-space dimension of the 3-DoF position task
    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(J_p_d, full_matrices=True)
        N = Vh.transpose(-1, -2)[..., -m:]
        NNt = N @ N.transpose(-1, -2)

    p1 = (NNt @ g1d.unsqueeze(-1)).squeeze(-1)
    p2 = (NNt @ g2d.unsqueeze(-1)).squeeze(-1)
    p3 = (NNt @ g3d.unsqueeze(-1)).squeeze(-1)
    gnorm = torch.stack(
        [g1d.norm(dim=-1), g2d.norm(dim=-1), g3d.norm(dim=-1)],
        dim=-1).clamp_min(1e-20)

    used = torch.zeros((B_size, m), dtype=torch.bool, device=device)
    arange_m = torch.arange(m, device=device).view(1, -1)
    fb_flags: list[torch.Tensor] = []

    def _gs(p, prev, gi, g_raw):
        nonlocal used
        # Main path: twice-is-enough reorthogonalization.
        v = p
        for e in prev:
            v = v - (v * e).sum(-1, keepdim=True) * e
        for e in prev:
            v = v - (v * e).sum(-1, keepdim=True) * e
        norm_v = v.norm(dim=-1)
        ok = (norm_v > eps_abs) & (norm_v > eps_rel * gnorm[:, gi])
        e_main = v / norm_v.clamp_min(eps_abs).unsqueeze(-1)
        # Fallback path: single pass (SVD col is already orthogonal to span(N)).
        fb_idx = (~used).float().argmax(dim=-1)
        n_col = torch.gather(
            N, -1, fb_idx.view(-1, 1, 1).expand(-1, d, 1)).squeeze(-1)
        v_fb = n_col
        for e in prev:
            v_fb = v_fb - (v_fb * e).sum(-1, keepdim=True) * e
        sgn = torch.sign((v_fb * g_raw).sum(-1, keepdim=True))
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
        v_fb = sgn * v_fb
        e_fb = v_fb / v_fb.norm(dim=-1, keepdim=True).clamp_min(eps_abs)
        e = torch.where(ok.unsqueeze(-1), e_main, e_fb)
        used = used | ((~ok).unsqueeze(-1) & (arange_m == fb_idx.view(-1, 1)))
        fb_flags.append(~ok)
        return e

    e0 = _gs(p1, [], 0, g1d)
    e1 = _gs(p2, [e0], 1, g2d)
    e2 = _gs(p3, [e0, e1], 2, g3d)

    # Complete the basis with the m - 3 remaining null directions (one for
    # the 7-DoF arms, none for Cobotta, whose three objective gradients
    # already fill the null space).
    cols = [e0, e1, e2]
    for _ in range(m - 3):
        fb_idx = (~used).float().argmax(dim=-1)
        n_col = torch.gather(
            N, -1, fb_idx.view(-1, 1, 1).expand(-1, d, 1)).squeeze(-1)
        v3 = n_col
        for e in cols:
            v3 = v3 - (v3 * e).sum(-1, keepdim=True) * e
        for e in cols:
            v3 = v3 - (v3 * e).sum(-1, keepdim=True) * e
        e3 = v3 / v3.norm(dim=-1, keepdim=True).clamp_min(eps_abs)
        used = used | (arange_m == fb_idx.view(-1, 1))
        cols.append(e3)

    B_basis = torch.stack(cols, dim=-1).to(dtype)
    fb_mask = torch.stack(fb_flags, dim=-1)
    return B_basis, fb_mask


# term_reason codes (success=1 removed: no success terminate in infinite-ray task)
TERM_ALIVE = 0
TERM_COLLISION = 2
TERM_CONE = 3
TERM_JL = 4
TERM_TRUNCATED = 5
TERM_LATERAL = 6

TERM_NAMES = {
    TERM_ALIVE: "alive",
    TERM_COLLISION: "collision",
    TERM_CONE: "cone",
    TERM_JL: "jl",
    TERM_TRUNCATED: "truncated",
    TERM_LATERAL: "lateral",
}


class NSRLBatchedEnv:
    obs_dim = OBS_DIM
    act_dim = ACT_DIM

    def __init__(self, cfg: EnvConfig, line_dist: LineDistribution | None,
                 device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_envs = cfg.n_envs
        self.dt = cfg.dt
        self.v = cfg.v
        self.a_max = cfg.a_max
        self.max_steps = cfg.max_steps
        self.cos_cone = math.cos(cfg.cone_deg * math.pi / 180.0)
        robot = getattr(cfg, 'robot', 'fr3')
        if robot == 'fr3':
            self.kin = BatchedFR3Kinematics(device=self.device,
                                            tcp_offset=cfg.tcp_offset)
            self.collision = FR3SphereCollision(device=self.device)
            self.n_joints = 7
        else:
            self.kin = BatchedChainKinematics(robot, device=self.device,
                                              tcp_offset=cfg.tcp_offset)
            self.collision = ChainSphereCollision(
                robot, self.kin.n_joints + 1, device=self.device)
            self.n_joints = self.kin.n_joints
        self.act_dim = self.n_joints - 3
        # 2n (q_norm, q_norm^2) + 13 task channels + a_prev (m); 31 for FR3.
        self.obs_dim = (2 * self.n_joints + 13 + self.act_dim
                        + (1 if getattr(cfg, 'observe_curvature', False) else 0)
                        + (RAY_ERROR_DIM if cfg.observe_ray_error else 0)
                        + (2 ** self.act_dim
                           if getattr(cfg, 'observe_prior_logits', False)
                           else 0))
        if getattr(cfg, 'observe_prior_logits', False):
            import numpy as _np
            grid = _np.stack(_np.meshgrid(
                *[[-1.0, 1.0]] * self.act_dim, indexing='ij'),
                -1).reshape(-1, self.act_dim)
            self._prior_verts = torch.as_tensor(
                grid, device=self.device, dtype=torch.float32)
        self.line_dist = line_dist

        self.lmt_lo = self.kin.lmt_lo
        self.lmt_up = self.kin.lmt_up
        self.q_mid = self.kin.q_mid
        self.q_half = 0.5 * (self.lmt_up - self.lmt_lo)

        B = self.n_envs
        d = self.kin.dtype
        self.q = torch.zeros((B, self.n_joints), device=self.device, dtype=d)
        # line_dir is the INSTANTANEOUS tangent of the path at the point
        # closest to the current TCP; it is refreshed every step. On a straight
        # ray it never leaves its reset value, so this is a no-op there.
        self.line_dir = torch.zeros((B, 3), device=self.device, dtype=d)
        # path_d0 is the reset-time tangent, i.e. the task descriptor d.
        self.path_d0 = torch.zeros((B, 3), device=self.device, dtype=d)
        # Signed curvature of the path [1/m]; 0 = the straight ray of the
        # submitted experiments.
        self.path_kappa = torch.zeros((B,), device=self.device, dtype=d)
        # Serpentine: lateral offset amp*sin(2*pi*x/wavelen) about the straight
        # axis path_d0. amp = 0 is not a wave. Unlike a constant-curvature arc,
        # this path never closes, so "how far" stays a finite distance along the
        # axis rather than turning into a lap count.
        self.path_amp = torch.zeros((B,), device=self.device, dtype=d)
        self.path_wavelen = torch.full((B,), 0.8, device=self.device, dtype=d)
        # Arc length travelled along the path. On a curved path the chord
        # projection (p - p_0)·d is no longer the travelled distance, so the
        # objective has to be accumulated step by step.
        self.arc_progress = torch.zeros((B,), device=self.device, dtype=d)
        self.n_target = torch.zeros((B, 3), device=self.device, dtype=d)
        # Non-planar surfaces: the cone axis rotates along the path,
        # n(s) = R(axis, rate*s) n0. rate = 0 keeps the historical fixed axis.
        self.n0_target = torch.zeros((B, 3), device=self.device, dtype=d)
        self.n_rot_axis = torch.zeros((B, 3), device=self.device, dtype=d)
        self.n_rot_rate = torch.zeros((B,), device=self.device, dtype=d)
        self.t = torch.zeros((B,), device=self.device, dtype=torch.long)
        self.a_prev = torch.zeros((B, self.act_dim), device=self.device,
                                  dtype=d)
        # Potential of the margin-shaping term at the previous step.
        self.phi_prev = torch.zeros((B,), device=self.device, dtype=d)
        self.done_persistent = torch.zeros((B,), device=self.device, dtype=torch.bool)
        # Cumulative per-episode reward (logged on done)
        self.episode_reward = torch.zeros((B,), device=self.device, dtype=d)
        self.episode_steps = torch.zeros((B,), device=self.device, dtype=torch.long)
        # EE position at episode start; used to compute progress = (p_now - p_start)·u_hat
        self.p_start = torch.full((B, 3), float("nan"), device=self.device, dtype=d)

    # ---------------------------------------------------------------- helpers

    def _compute_obs(self, p_tcp: torch.Tensor, R_tcp: torch.Tensor,
                     q: torch.Tensor | None = None,
                     a_prev: torch.Tensor | None = None) -> torch.Tensor:
        # q / a_prev default to self.q / self.a_prev, but step() must pass
        # post-step values when building terminal_obs so all 31 dims are
        # snapshotted at the same time (line_dir, n_target are reset-only).
        q = self.q if q is None else q
        a_prev = self.a_prev if a_prev is None else a_prev
        z_tool = R_tcp[:, :, 2]
        q_norm = (q - self.q_mid) / self.q_half
        cos_angle = (z_tool * self.n_target).sum(-1, keepdim=True)
        z_cross_n = torch.linalg.cross(z_tool, self.n_target, dim=-1)
        q_norm_sq = q_norm * q_norm
        obs_parts = [
            q_norm,         # 7
            q_norm_sq,      # 7
            self.line_dir,  # 3, u_hat
            z_tool,         # 3
            self.n_target,  # 3
            cos_angle,      # 1
            z_cross_n,      # 3
            a_prev,         # 4
        ]
        if getattr(self.cfg, "observe_curvature", False):
            kap = torch.where(
                self.path_amp.abs() > 1e-6,
                serpentine_curvature(p_tcp, self.p_start, self.path_d0,
                                     self.n_target, self.path_amp,
                                     self.path_wavelen),
                self.path_kappa)
            obs_parts.append(kap.unsqueeze(-1) * CURV_SCALE)   # 1
        if getattr(self.cfg, 'observe_prior_logits', False):
            obs_parts.append(self._prior_scores(q))
        if self.cfg.observe_ray_error:
            # The path origin can differ slightly from FK(q0) when a task and
            # its IK seed are decoupled. Exposing this offset makes the lateral
            # safety-net state Markov for the continuous controller. Only the
            # component perpendicular to the path matters; scaling by the
            # termination radius keeps the observation order-one even after
            # long travel.
            _, lateral_vec, _ = self._path_frame(p_tcp)
            obs_parts.append(-lateral_vec / LATERAL_SAFETY_NET)  # 3
        return torch.cat(obs_parts, dim=-1)

    def _prior_scores(self, q: torch.Tensor) -> torch.Tensor:
        """Analytic sigma_margin^T v for every vertex v, shape (B, 2^m).

        grad m_cone = J_rot^T (z x n) / (1 - cos_cone); grad m_jl is the
        subgradient at the binding joint. Softmin weights combine them, the
        task-aligned basis projects to null-space coordinates, and the score
        of a vertex is the inner product. Purely analytic: no autograd."""
        _, R, J, _ = self.kin.tcp_fk_jac(q)
        z = R[:, :, 2]
        m_jl_per = (self.q_half - (q - self.q_mid).abs()) / self.q_half
        m_jl, j_star = m_jl_per.min(dim=-1)
        cos = (z * self.n_target).sum(-1).clamp(-1.0, 1.0)
        m_cone = (cos - self.cos_cone) / (1.0 - self.cos_cone)
        tau = self.cfg.margin_tau
        w = torch.softmax(
            -torch.stack([m_jl, m_cone], dim=-1) / tau, dim=-1)
        g_jl = torch.zeros_like(q)
        slope = -torch.sign(q - self.q_mid) / self.q_half
        g_jl.scatter_(1, j_star.unsqueeze(-1),
                      torch.gather(slope, 1, j_star.unsqueeze(-1)))
        zxn = torch.linalg.cross(z, self.n_target, dim=-1)
        g_cone = torch.einsum('bij,bi->bj', J[:, 3:, :], zxn) \
            / (1.0 - self.cos_cone)
        g_phi = w[:, :1] * g_jl + w[:, 1:] * g_cone
        B_basis, _ = build_task_aligned_basis(
            self.kin, q, self.line_dir, self.n_target,
            self.kin.q_mid, self.q_half, self.cfg.manip_damping)
        sigma = torch.einsum('bij,bi->bj', B_basis, g_phi)
        return sigma @ self._prior_verts.T

    def _refresh_n_target(self) -> None:
        """n(s) at the current arc progress (Rodrigues, batched); no-op for
        rate = 0 tasks. Consumers within one control period see n at the
        period's start, a <1 deg staleness at the sampled rates."""
        if not bool((self.n_rot_rate != 0).any()):
            return
        th = (self.n_rot_rate * self.arc_progress).unsqueeze(-1)
        k = self.n_rot_axis
        n0 = self.n0_target
        c, s = torch.cos(th), torch.sin(th)
        kxn = torch.linalg.cross(k, n0, dim=-1)
        kdn = (k * n0).sum(-1, keepdim=True)
        n = n0 * c + kxn * s + k * kdn * (1 - c)
        self.n_target = torch.where(
            (self.n_rot_rate != 0).unsqueeze(-1), n, self.n_target)

    def _path_frame(self, p: torch.Tensor):
        """(tangent, lateral_vec, lateral_dist) of the path at TCP position p.

        lateral_vec points from p to the closest point on the path and is
        orthogonal to tangent, so feeding it back adds no along-path motion.
        """
        return path_frame(p, self.p_start, self.path_d0, self.n_target,
                          self.path_kappa, self.path_amp, self.path_wavelen)

    def current_obs(self) -> torch.Tensor:
        """Public: obs at the current internal state (no step taken).

        Also refreshes ``self.line_dir`` to the instantaneous tangent, so an
        external controller reading ``env.line_dir`` between steps (the hybrid
        and classical eval loops do) sees the tangent at the current TCP and
        not a stale one. No-op on a straight ray.
        """
        p, R, _, _ = self.kin.tcp_fk_jac(self.q)
        tangent, _, _ = self._path_frame(p)
        self.line_dir = torch.where(
            torch.isfinite(tangent).all(-1, keepdim=True), tangent,
            self.line_dir)
        self._refresh_n_target()
        return self._compute_obs(p, R)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return
        spec = self.line_dist.sample(n_reset)
        self.q[mask] = spec["q0"]
        self.line_dir[mask] = spec["line_dir"]
        self.path_d0[mask] = spec["line_dir"]
        self.n_target[mask] = spec["n_target"]
        self.n0_target[mask] = spec["n_target"]
        if "n_rot_axis" in spec:
            self.n_rot_axis[mask] = spec["n_rot_axis"].to(
                device=self.device, dtype=self.kin.dtype)
            self.n_rot_rate[mask] = spec["n_rot_rate"].to(
                device=self.device, dtype=self.kin.dtype).reshape(-1)
        else:
            self.n_rot_axis[mask] = 0.0
            self.n_rot_rate[mask] = 0.0
        # Curvature is optional: distributions that predate the curved-path
        # extension describe straight rays only.
        if "kappa" in spec:
            self.path_kappa[mask] = spec["kappa"].to(
                device=self.device, dtype=self.kin.dtype).reshape(n_reset)
        else:
            self.path_kappa[mask] = 0.0
        if "amp" in spec:
            self.path_amp[mask] = spec["amp"].to(
                device=self.device, dtype=self.kin.dtype).reshape(n_reset)
            self.path_wavelen[mask] = spec["wavelen"].to(
                device=self.device, dtype=self.kin.dtype).reshape(n_reset)
        else:
            self.path_amp[mask] = 0.0
        self.arc_progress[mask] = 0.0
        self.t[mask] = 0
        self.a_prev[mask] = 0
        self.done_persistent[mask] = False
        self.episode_reward[mask] = 0.0
        self.episode_steps[mask] = 0
        # A task/seed-decoupled distribution may carry the task-defined ray
        # origin explicitly. This is intentionally not always FK(q0): an IK
        # projection has finite tolerance, while progress and lateral error
        # must remain anchored to the original task. Historical distributions
        # omit p0 and retain the exact old behavior.
        if 'p0' in spec:
            p0 = spec['p0'].to(device=self.device, dtype=self.kin.dtype)
            if p0.shape != (n_reset, 3):
                raise ValueError(
                    f'line distribution p0 must have shape ({n_reset}, 3), '
                    f'got {tuple(p0.shape)}')
            if not bool(torch.isfinite(p0).all().item()):
                raise ValueError('line distribution p0 must be finite')
            self.p_start[mask] = p0
        else:
            p_at_reset, _, _, _ = self.kin.tcp_fk_jac(self.q[mask])
            self.p_start[mask] = p_at_reset

        if self.cfg.w_margin != 0.0:
            q0 = self.q[mask]
            _, R0, _, _ = self.kin.tcp_fk_jac(q0)
            cos0 = (R0[:, :, 2] * self.n_target[mask]).sum(-1).clamp(-1., 1.)
            m_jl = ((self.q_half - (q0 - self.q_mid).abs())
                    / self.q_half).amin(dim=-1)
            m_cone = (cos0 - self.cos_cone) / (1.0 - self.cos_cone)
            tau = self.cfg.margin_tau
            self.phi_prev[mask] = -tau * torch.logsumexp(
                -torch.stack([m_jl, m_cone], dim=-1) / tau, dim=-1)

    # ---------------------------------------------------------------- API

    def reset(self) -> torch.Tensor:
        mask = torch.ones((self.n_envs,), device=self.device, dtype=torch.bool)
        self._reset_envs(mask)
        return self.current_obs()

    def step(self, actions: torch.Tensor, auto_reset: bool = True
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the batched env.

        auto_reset=True (default, training): terminated/truncated envs are
        re-sampled in-place; returned obs is the reset obs; info.terminal_obs
        holds the pre-reset terminal observation for PPO bootstrap.

        auto_reset=False (eval): done envs become frozen (state unchanged,
        reward 0, terminated/truncated False on subsequent steps). Caller
        polls `env.done_persistent` to know when all envs have finished.
        """
        actions = actions.clamp(-1.0, 1.0).to(device=self.device, dtype=self.kin.dtype)
        if self.cfg.speed_levels:
            speed_frac = actions[:, self.act_dim].clamp(
                min(self.cfg.speed_levels), 1.0)
            actions = actions[:, :self.act_dim]
        else:
            speed_frac = None
        a_scaled = actions * self.a_max
        active = ~self.done_persistent  # only matters when auto_reset=False

        p, R, J, _ = self.kin.tcp_fk_jac(self.q)
        J_p = J[:, :3, :]

        # Instantaneous path frame at the current TCP. `line_dir` is the
        # tangent from here on; it feeds the observation, the task-aligned
        # nullspace basis and the classical controller's directional
        # manipulability, none of which need to know the path is curved.
        tangent, lateral_vec, _ = self._path_frame(p)
        self.line_dir = tangent
        self._refresh_n_target()

        J_plus, sigma_min = damped_pinv(J_p, self.cfg.lambda_0, self.cfg.sigma_thr)
        B_basis, fb_mask = build_task_aligned_basis(
            self.kin, self.q, self.line_dir, self.n_target,
            self.kin.q_mid, self.q_half, self.cfg.manip_damping,
        )

        # Task-space command: feed-forward along the instantaneous tangent
        # plus proportional feedback on the distance to the path. On a
        # straight ray the damped pseudo-inverse alone keeps the realized TCP
        # velocity on-axis, and k_lateral = 0 recovers the pure feed-forward
        # command used for the submitted results. On a curved path the
        # feed-forward term alone cuts the corner by v·dt·|kappa|/2 per step,
        # an error that accumulates with arc length until the safety net
        # trips, so k_lateral > 0 is required there. lateral_vec is orthogonal
        # to the tangent, so this term never changes the along-path speed.
        v_eff = (self.v if speed_frac is None
                 else (self.v * speed_frac).unsqueeze(-1))
        x_dot = (v_eff * self.line_dir
                 + self.cfg.k_lateral * lateral_vec).unsqueeze(-1)
        qdot_task = (J_plus @ x_dot).squeeze(-1)
        qdot_null = (B_basis @ a_scaled.unsqueeze(-1)).squeeze(-1)
        qdot = qdot_task + qdot_null
        q_new = self.q + qdot * self.dt

        link_tfs = self.kin.link_transforms(q_new)
        p_new, R_new, _, _ = self.kin.tcp_fk_jac(q_new)
        z_new = R_new[:, :, 2]

        is_coll = self.collision.is_collided(link_tfs)
        jl_viol = ((q_new < self.lmt_lo) | (q_new > self.lmt_up)).any(dim=-1)
        cos_angle = (z_new * self.n_target).sum(-1).clamp(-1.0, 1.0)
        cone_viol = cos_angle < self.cos_cone

        # Distance from the TCP to the path — NOT to the initial tangent ray.
        # On a straight ray the two coincide; on an arc of radius R the ray
        # reference would flag a perfectly tracked path as violating after
        # sqrt(2 R * LATERAL_SAFETY_NET) of travel (0.2 m at R = 1 m),
        # which has nothing to do with kinematic capability. Safety-net
        # terminate only — the k_lateral feedback keeps this far below the cap.
        tangent_new, lateral_vec_new, lateral_err = self._path_frame(p_new)
        lateral_viol = lateral_err > LATERAL_SAFETY_NET

        new_t = self.t + 1
        truncated = new_t >= self.max_steps

        # Per-step EE travel along u_hat (meters this step). Damped pinv
        # guarantees this is ≤ v·dt; clip to non-negative to ignore numerical
        # backwards drift from finite-dt curvature.
        # On a wave the objective is how far the stroke advanced ALONG THE
        # AXIS, not the length of the zig-zag it traced: otherwise a larger
        # amplitude would inflate the score without covering more of the seam.
        # On a ray the two coincide, and on an arc the axis is meaningless, so
        # only the wave branch differs.
        is_wave = self.path_amp.abs() > 1e-6
        delta_progress = torch.where(
            is_wave,
            ((p_new - p) * self.path_d0).sum(-1),
            ((p_new - p) * self.line_dir).sum(-1))
        progress_norm = (delta_progress / (self.v * self.dt)).clamp(0.0, 1.0)
        arc_step = delta_progress.clamp_min(0.0)
        r_progress_per_env = self.cfg.w_progress * progress_norm

        reward = r_progress_per_env.clone()
        if self.cfg.w_margin != 0.0:
            m_jl = ((self.q_half - (q_new - self.q_mid).abs())
                    / self.q_half).amin(dim=-1)
            m_cone = (cos_angle - self.cos_cone) / (1.0 - self.cos_cone)
            tau = self.cfg.margin_tau
            phi_new = -tau * torch.logsumexp(
                -torch.stack([m_jl, m_cone], dim=-1) / tau, dim=-1)
            reward = reward + self.cfg.w_margin * (
                self.cfg.margin_gamma * phi_new - self.phi_prev)
            self.phi_prev = torch.where(
                self.done_persistent, self.phi_prev, phi_new)
        reward = torch.where(is_coll, reward + self.cfg.r_collision, reward)
        reward = torch.where(cone_viol & ~is_coll, reward + self.cfg.r_cone, reward)
        reward = torch.where(jl_viol & ~is_coll & ~cone_viol,
                             reward + self.cfg.r_jl, reward)

        # Framing B: lateral_viol is a terminating condition (hard constraint),
        # not bootstrapped. NOT included as bootstrap-truncation.
        terminated = is_coll | cone_viol | jl_viol | lateral_viol
        done = terminated | truncated

        term_reason = torch.full((self.n_envs,), TERM_ALIVE,
                                 device=self.device, dtype=torch.long)
        term_reason = torch.where(is_coll,
                                  torch.full_like(term_reason, TERM_COLLISION), term_reason)
        term_reason = torch.where(cone_viol & ~is_coll,
                                  torch.full_like(term_reason, TERM_CONE), term_reason)
        term_reason = torch.where(jl_viol & ~is_coll & ~cone_viol,
                                  torch.full_like(term_reason, TERM_JL), term_reason)
        term_reason = torch.where(lateral_viol & ~is_coll & ~cone_viol & ~jl_viol,
                                  torch.full_like(term_reason, TERM_LATERAL), term_reason)
        term_reason = torch.where(truncated & ~terminated,
                                  torch.full_like(term_reason, TERM_TRUNCATED), term_reason)

        # Chord projection on the tangent the step started from. Kept for the
        # historical training log only; on a curved path the meaningful
        # objective is self.arc_progress below.
        progress_now = ((p_new - self.p_start) * tangent).sum(-1)
        progress_now = torch.nan_to_num(progress_now, nan=0.0)

        # From here on line_dir is the post-step tangent, so terminal_obs and
        # the next controller call both refer to the new TCP position. Frozen
        # envs (auto_reset=False, already done) keep their last tangent.
        self.line_dir = torch.where(active.unsqueeze(-1), tangent_new,
                                    self.line_dir)

        # snapshot of obs at end-of-step (post-step q, this-step actions);
        # PPO bootstraps V(terminal_obs) for truncated episodes.
        terminal_obs = self._compute_obs(
            p_new, R_new, q=q_new, a_prev=actions)

        # Accumulate per-episode reward + step counter (before reset wipes them)
        ep_reward_finished = torch.zeros_like(reward)
        ep_steps_finished = torch.zeros_like(self.episode_steps)
        ep_progress_finished = torch.zeros_like(reward)

        if auto_reset:
            self.q = q_new
            self.t = new_t
            self.a_prev = actions
            self.episode_reward = self.episode_reward + reward
            self.episode_steps = self.episode_steps + 1
            self.arc_progress = self.arc_progress + arc_step
            new_done = done
            ep_reward_finished = torch.where(done, self.episode_reward,
                                             torch.zeros_like(self.episode_reward))
            ep_steps_finished = torch.where(done, self.episode_steps,
                                            torch.zeros_like(self.episode_steps))
            ep_progress_finished = torch.where(done, progress_now,
                                               torch.zeros_like(progress_now))
            ep_arc_finished = torch.where(done, self.arc_progress,
                                          torch.zeros_like(self.arc_progress))
            if done.any():
                self._reset_envs(done)
            obs = self.current_obs()
        else:
            new_done = done & active
            reward = torch.where(active, reward, torch.zeros_like(reward))
            terminated = terminated & active
            truncated = truncated & active
            term_reason = torch.where(active, term_reason,
                                      torch.full_like(term_reason, TERM_ALIVE))

            self.q = torch.where(active.unsqueeze(-1), q_new, self.q)
            self.t = torch.where(active, new_t, self.t)
            self.a_prev = torch.where(active.unsqueeze(-1), actions, self.a_prev)
            self.episode_reward = torch.where(active, self.episode_reward + reward,
                                              self.episode_reward)
            self.episode_steps = torch.where(active, self.episode_steps + 1,
                                             self.episode_steps)
            self.arc_progress = torch.where(active,
                                            self.arc_progress + arc_step,
                                            self.arc_progress)
            self.done_persistent = self.done_persistent | done
            ep_reward_finished = torch.where(new_done, self.episode_reward,
                                             torch.zeros_like(self.episode_reward))
            ep_steps_finished = torch.where(new_done, self.episode_steps,
                                            torch.zeros_like(self.episode_steps))
            ep_progress_finished = torch.where(new_done, progress_now,
                                               torch.zeros_like(progress_now))
            ep_arc_finished = torch.where(new_done, self.arc_progress,
                                          torch.zeros_like(self.arc_progress))
            obs = self.current_obs()

        # Episode-reward stats (only mean over envs that finished this step;
        # default 0/0 → reported as nan if no env finished). Used by PPO log_fn.
        n_finished = int(new_done.sum().item())
        if n_finished > 0:
            ep_reward_mean = float(ep_reward_finished[new_done].mean().item())
            ep_len_mean = float(ep_steps_finished[new_done].float().mean().item())
            ep_progress_mean = float(ep_progress_finished[new_done].mean().item())
            ep_arc_mean = float(ep_arc_finished[new_done].mean().item())
        else:
            ep_reward_mean = float("nan")
            ep_len_mean = float("nan")
            ep_progress_mean = float("nan")
            ep_arc_mean = float("nan")

        info = {
            "terminal_obs": terminal_obs,
            "term_reason": term_reason,
            "sigma_min": sigma_min,
            "episode_done": new_done,
            "r_progress_mean": float(r_progress_per_env.mean().item()),
            "lateral_err_mean": float(lateral_err.mean().item()),
            "lateral_err_max": float(lateral_err.max().item()),
            "ep_reward_mean": ep_reward_mean,
            "ep_len_mean": ep_len_mean,
            "ep_progress_mean": ep_progress_mean,
            "ep_arc_progress_mean": ep_arc_mean,
            "n_episodes_done": n_finished,
            # MGS fallback rates (batch-averaged) per anchor column.
            "fb_rate_e0": float(fb_mask[:, 0].float().mean().item()),
            "fb_rate_e1": float(fb_mask[:, 1].float().mean().item()),
            "fb_rate_e2": float(fb_mask[:, 2].float().mean().item()),
        }
        return obs, reward, terminated, truncated, info
