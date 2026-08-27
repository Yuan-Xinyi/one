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
    # Reset-time randomization of the wrist-roll joint (last joint). The TCP
    # position and tool axis are exactly invariant to it (rotation about the
    # tool axis), so this is a free symmetry augmentation that exposes the
    # critic to the full q7 range; 0 disables, x>0 draws uniformly within
    # the central x-fraction of the joint range.
    q7_reset_uniform: float = 0.0
    # Basis scaling: False = orthonormal columns (unit joint speed per
    # action unit, the published pipeline); True = keep each objective
    # column at its RAW projected-gradient magnitude times this gain, so
    # |a_k|=1 commands gain x the physically attainable component of that
    # objective (the free residual direction e3 stays unit). 0 disables
    # (orthonormal columns), 1 = raw magnitude, 10 = ten-fold.
    basis_raw_scale: float = 0.0
    # Physically normalized continuous action ("dir-frac"): the policy
    # outputs m+1 channels; the first m give a null-space DIRECTION (unit-
    # normalized), the last a FRACTION rho in [0,1] of the physically
    # available amplitude alpha_feas(q, dir), computed each substep from the
    # true joint-velocity limits qd_limit. No a_max is involved.
    # 0 = off; 1 = direction in B-coordinates (m+1 channels, v1);
    # 2 = direction in JOINT space, projected by the exact null projector
    #     (n+1 channels, basis-free final form).
    dir_frac_action: int = 0
    qd_limit: tuple = ()
    # a_prev stores the EXECUTED physical motion (qdot_null / qd_limit,
    # 2 rho - 1) instead of the raw network output. Raw u is
    # non-identifiable: u, 2u and u + r with r in row(J_p) all execute the
    # same physical direction after projection. dir_frac_action == 2 only.
    a_prev_executed: bool = False
    # Append per-joint actuator headroom AFTER the task term claims its
    # share: r+ = (qd_lim - qdot_task)/qd_lim and r- = (qd_lim +
    # qdot_task)/qd_lim (2n channels, one control period stale like
    # proj_scales; 1.0 at reset). Complements proj_scales: those report
    # geometric leverage, these report actuator budget.
    observe_headroom: bool = False
    # Number of a_prev frames in the observation (1 = the historical
    # single frame). K > 1 appends the K-1 older frames after a_prev,
    # newest first; frames are whatever a_prev records (raw or executed).
    a_prev_stack: int = 1
    # Observation components to omit, for leave-one-out ablations.
    # Supported keys: 'q_sq' (squared joint config), 'z_cross_n',
    # 'cos_angle', 'a_prev' (drops the executed-motion channel from the
    # observation only; the env still tracks it internally).
    obs_drop: tuple = ()
    # Metric used by the null projection that turns the actor's 7-dim
    # output into a direction. 0 = Euclidean joint metric (v2 mainline).
    # 1 = static velocity-limit metric D = diag(qd_limit): the actor picks
    #     the direction in "fraction of each joint's speed limit" space.
    # 2 = state-dependent headroom metric D = diag(qd_limit - |qdot_task|):
    #     direction picked in "fraction of remaining budget" space.
    # Either way the executed direction is D z / |D z| with z the projection
    # of u in the scaled coordinates, so ker J_p membership is exact.
    dv_metric: int = 0
    # Drop the separate rho channel: the fraction is carried by the norm
    # of the 7-dim output itself, rho = min(1, |P_N u|). Direction stays
    # the normalized projection. dir_frac_action == 2 only.
    rho_from_norm: bool = False
    # Append the four normalized CURRENT-state constraint margins
    # (jl, cone, corridor, collision) to the observation. Class-0
    # information: analytics of the current configuration, no forward
    # simulation. Refreshed in step() for the post-step state; 1.0 right
    # after reset (exact for corridor, approximate for the rest) unless
    # true_reset_obs is set.
    observe_margins: bool = False
    # Replace the placeholder reset values of the buffered observation
    # channels with the true quantities at q0: the four constraint margins
    # (placeholder 1.0) and the projected-gradient scales (placeholder 0).
    # Without this V(o(q0)) is blind to wall distances and null-space
    # mobility exactly at the state where candidate scoring queries it.
    # a_prev stays 0 at reset -- that IS the true value.
    true_reset_obs: bool = False
    # Fold the next-step joint-POSITION limits into alpha_feas:
    # |q + dt (qdot_task + alpha xi) - q_mid| <= q_half, still a per-joint
    # closed-form bound (no optimizer). alpha_safe = min(alpha_vel,
    # alpha_joint).
    alpha_joint: bool = False
    # Append the three projected-gradient magnitudes s_k = |P_N g_k| of the
    # basis objectives to the observation (one control period stale, like
    # a_prev).
    observe_proj_scales: bool = False
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
    # Binary advance gate: one trailing action channel, executed as
    # g = 1[a > 0]. g=1 runs the task-space command as usual; g=0 executes
    # NO task motion this step -- pure null-space reconfiguration, with the
    # whole velocity budget available to it (alpha_feas sees qdot_task = 0).
    # Progress reward is computed from realized motion, so a paused step
    # earns exactly 0: under the exit-time objective a pause pays only if
    # the reconfiguration buys more stroke than the time it costs.
    # dir_frac_action == 2 only; mutually exclusive with speed_levels.
    task_gate: bool = False
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
    raw_scale: float = 0.0,
    return_scales: bool = False,
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
    proj_norms = torch.stack([p1.norm(dim=-1), p2.norm(dim=-1),
                              p3.norm(dim=-1)], dim=-1)
    gnorm = torch.stack(
        [g1d.norm(dim=-1), g2d.norm(dim=-1), g3d.norm(dim=-1)],
        dim=-1).clamp_min(1e-20)

    used = torch.zeros((B_size, m), dtype=torch.bool, device=device)
    arange_m = torch.arange(m, device=device).view(1, -1)
    fb_flags: list[torch.Tensor] = []
    col_scales: list[torch.Tensor] = []

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
        col_scales.append(norm_v)
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

    if raw_scale:
        g = float(raw_scale)
        for k, sc in enumerate(col_scales):
            cols[k] = cols[k] * (g * sc).unsqueeze(-1)
    B_basis = torch.stack(cols, dim=-1).to(dtype)
    fb_mask = torch.stack(fb_flags, dim=-1)
    if return_scales:
        # (residual MGS norms, order-free projected norms)
        scales = torch.stack(col_scales, dim=-1).to(dtype)
        return B_basis, fb_mask, scales, proj_norms.to(dtype)
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
        # Policy-facing action dim: +1 fraction channel in dir-frac mode.
        _dfa = int(getattr(cfg, 'dir_frac_action', 0) or 0)
        _rfn = getattr(cfg, 'rho_from_norm', False)
        if _rfn:
            assert _dfa == 2, 'rho_from_norm requires dir_frac_action == 2'
        self.act_dim_policy = (self.act_dim + 1 if _dfa == 1 else
                               self.n_joints + (0 if _rfn else 1)
                               if _dfa == 2 else
                               self.act_dim)
        if getattr(cfg, 'task_gate', False):
            assert _dfa == 2 and not cfg.speed_levels, \
                'task_gate requires dir_frac_action == 2, no speed_levels'
        if _dfa == 2 and (cfg.speed_levels
                          or getattr(cfg, 'task_gate', False)):
            self.act_dim_policy += 1          # trailing speed/gate channel
        _od = set(getattr(cfg, 'obs_drop', ()) or ())
        assert _od <= {'q_sq', 'z_cross_n', 'cos_angle', 'a_prev'}, _od
        # 2n (q_norm, q_norm^2) + 13 task channels + a_prev; 31 for FR3.
        self.obs_dim = (2 * self.n_joints + 13 + self.act_dim_policy
                        - (self.n_joints if 'q_sq' in _od else 0)
                        - (3 if 'z_cross_n' in _od else 0)
                        - (1 if 'cos_angle' in _od else 0)
                        - (self.act_dim_policy if 'a_prev' in _od else 0)
                        + (3 if getattr(cfg, 'observe_proj_scales', False)
                           else 0)
                        + (2 * self.n_joints
                           if getattr(cfg, 'observe_headroom', False) else 0)
                        + (int(getattr(cfg, 'a_prev_stack', 1) or 1) - 1)
                        * self.act_dim_policy
                        + (4 if getattr(cfg, 'observe_margins', False)
                           else 0)
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
        self.a_prev = torch.zeros((B, self.act_dim_policy),
                                  device=self.device,
                                  dtype=d)
        self._proj_scales = torch.zeros((B, 3), device=self.device,
                                        dtype=torch.float32)
        if (getattr(cfg, 'dir_frac_action', False)
                or getattr(cfg, 'observe_headroom', False)):
            assert cfg.qd_limit, 'dir_frac_action requires qd_limit'
            self.qd_limit = torch.tensor(cfg.qd_limit, device=self.device,
                                         dtype=self.kin.dtype)
            assert self.qd_limit.shape[0] == self.n_joints
        if getattr(cfg, 'a_prev_executed', False):
            assert _dfa == 2, 'a_prev_executed requires dir_frac_action == 2'
        self._headroom = torch.ones((B, 2 * self.n_joints),
                                    device=self.device, dtype=d)
        self._ap_k = int(getattr(cfg, 'a_prev_stack', 1) or 1)
        self._a_hist = torch.zeros(
            (B, (self._ap_k - 1) * self.act_dim_policy),
            device=self.device, dtype=d)
        self._mg_obs = torch.ones((B, 4), device=self.device, dtype=d)
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
        _od = set(getattr(self.cfg, 'obs_drop', ()) or ())
        obs_parts = [
            q_norm,         # 7
            *([] if 'q_sq' in _od else [q_norm_sq]),      # 7
            self.line_dir,  # 3, u_hat
            z_tool,         # 3
            self.n_target,  # 3
            *([] if 'cos_angle' in _od else [cos_angle]),  # 1
            *([] if 'z_cross_n' in _od else [z_cross_n]),  # 3
            *([] if 'a_prev' in _od else [a_prev]),  # m (+1 dir-frac)
        ]
        if self._ap_k > 1:
            obs_parts.append(self._a_hist)        # (K-1) older a_prev frames
        if getattr(self.cfg, 'observe_margins', False):
            obs_parts.append(self._mg_obs)        # 4 current-state margins
        if getattr(self.cfg, 'observe_proj_scales', False):
            obs_parts.append(self._proj_scales)   # 3, one period stale
        if getattr(self.cfg, 'observe_headroom', False):
            obs_parts.append(self._headroom)      # 2n, one period stale
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
            self.kin.q_mid, self.q_half, self.cfg.manip_damping,
            raw_scale=self.cfg.basis_raw_scale)
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

    @torch.no_grad()
    def _twin_side(self, q_t: torch.Tensor):
        """Full admissibility check + geometry of one twin candidate."""
        lmt = ((q_t >= self.lmt_lo) & (q_t <= self.lmt_up)).all(dim=-1)
        tfs = self.kin.link_transforms(q_t)
        coll = self.collision.is_collided(tfs)
        p_t, R_t, _, _ = self.kin.tcp_fk_jac(q_t)
        tangent, _, lat = self._path_frame(p_t)
        cos = (R_t[:, :, 2] * self.n_target).sum(-1).clamp(-1.0, 1.0)
        feas = ((~coll) & lmt & (lat <= LATERAL_SAFETY_NET)
                & (cos >= self.cos_cone))
        mm = self.collision.min_margin(tfs) / 0.05
        return feas, p_t, R_t, tangent, lat, cos, mm

    @torch.no_grad()
    def compute_twin_obs(self, delta: float, null_space: bool = True,
                         margin_floor: float = 0.0):
        """Feasibility-certified null-space twin observation of the state.

        V* is constant along feasible self-motion directions: a pure
        reconfiguration changes neither the task-space progress nor the
        reachable remainder, so the critic queried at the current state
        and at a small certified step along ker J_p must agree
        (P_N grad V = 0). Samples one random null direction per env,
        tries +delta and -delta in random order, and emits the
        observation at the first feasible side with margins and tangent
        recomputed there; the convention-stale channels (a_prev, proj
        scales) are inherited from the anchor so both observations follow
        the training-time staleness convention. States at t=0 emit no
        pair: without true_reset_obs their anchor margins are the reset
        placeholder and the pair would compare different conventions.
        A pair never straddles an
        infeasible region -- the certification is the same constraint
        set that terminates an episode -- so the equality constraint
        cannot propagate across a component boundary.

        Returns (twin_obs, valid) of shapes (B, obs_dim) and (B,).
        """
        B = self.n_envs
        dt = self.kin.dtype
        if null_space:
            _, _, J, _ = self.kin.tcp_fk_jac(self.q)
            _, _, Vh = torch.linalg.svd(J[:, :3, :].double(),
                                        full_matrices=True)
            Nn = Vh.transpose(-1, -2)[..., 3:]
            r = torch.randn(B, Nn.shape[-1], 1, device=self.device,
                            dtype=torch.float64)
            xi = (Nn @ r).squeeze(-1)
        else:
            # Confound control: same certified-pair machinery, but the
            # direction is drawn in the FULL joint space. The implied
            # invariance is wrong (the pair may move along the path), so
            # matching gains here would mean the effect is generic
            # neighbor-consistency, not the null-space semantics.
            xi = torch.randn(B, self.n_joints, device=self.device,
                             dtype=torch.float64)
        nrm = xi.norm(dim=-1, keepdim=True)
        ok_dir = nrm.squeeze(-1) > 1e-8
        xi = (xi / nrm.clamp_min(1e-8)).to(dt)
        sgn = (torch.randint(0, 2, (B, 1), device=self.device) * 2 - 1).to(dt)
        qa = self.q + delta * sgn * xi
        qb = self.q - delta * sgn * xi
        fa, pa, Ra, ta, la, ca, ma = self._twin_side(qa)
        fb, pb, Rb, tb, lb, cb, mb = self._twin_side(qb)
        valid = (fa | fb) & ok_dir & (self.t > 0)
        w = fa.unsqueeze(-1)
        if margin_floor > 0.0:
            # Interior-only pairing: both sides must keep every normalized
            # margin above the floor (anchor from its buffered margins,
            # twin from the freshly computed ones below).
            valid &= self._mg_obs.amin(dim=-1) > margin_floor
        q_t = torch.where(w, qa, qb)
        p_t = torch.where(w, pa, pb)
        R_t = torch.where(w.unsqueeze(-1), Ra, Rb)
        tan = torch.where(w, ta, tb)
        lat = torch.where(fa, la, lb)
        cos = torch.where(fa, ca, cb)
        mm = torch.where(fa, ma, mb)
        sv = (self._mg_obs, self.line_dir)
        if getattr(self.cfg, 'observe_margins', False):
            m_jl = ((self.q_half - (q_t - self.q_mid).abs())
                    / self.q_half).amin(dim=-1)
            m_cone = (cos - self.cos_cone) / (1.0 - self.cos_cone)
            m_lat = (LATERAL_SAFETY_NET - lat) / LATERAL_SAFETY_NET
            _tw_mg = torch.stack([m_jl, m_cone, m_lat, mm], dim=-1)
            if margin_floor > 0.0:
                valid &= _tw_mg.amin(dim=-1) > margin_floor
            self._mg_obs = _tw_mg
        self.line_dir = tan
        obs_t = self._compute_obs(p_t, R_t, q=q_t, a_prev=self.a_prev)
        self._mg_obs, self.line_dir = sv
        return obs_t, valid

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
        if self.cfg.q7_reset_uniform > 0.0:
            lo, up = self.kin.lmt_lo[-1], self.kin.lmt_up[-1]
            mid, half = 0.5 * (lo + up), 0.5 * (up - lo)
            u = torch.rand(n_reset, device=self.device, dtype=self.q.dtype)
            q7 = mid + (2.0 * u - 1.0) * half * self.cfg.q7_reset_uniform
            qm = self.q[mask]
            qm[:, -1] = q7
            self.q[mask] = qm
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
        self._proj_scales[mask] = 0
        self._headroom[mask] = 1.0
        self._a_hist[mask] = 0
        self._mg_obs[mask] = 1.0
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

        if getattr(self.cfg, 'true_reset_obs', False):
            q0 = self.q[mask]
            if getattr(self.cfg, 'observe_margins', False):
                link_tfs0 = self.kin.link_transforms(q0)
                p0f, R0, _, _ = self.kin.tcp_fk_jac(q0)
                cos0 = (R0[:, :, 2] * self.n_target[mask]
                        ).sum(-1).clamp(-1., 1.)
                _m_jl = ((self.q_half - (q0 - self.q_mid).abs())
                         / self.q_half).amin(dim=-1)
                _m_cone = (cos0 - self.cos_cone) / (1.0 - self.cos_cone)
                _, _, lat0 = path_frame(p0f, self.p_start[mask],
                                        self.path_d0[mask],
                                        self.n_target[mask],
                                        self.path_kappa[mask],
                                        self.path_amp[mask],
                                        self.path_wavelen[mask])
                _m_lat = (LATERAL_SAFETY_NET - lat0) / LATERAL_SAFETY_NET
                _m_coll = self.collision.min_margin(link_tfs0) / 0.05
                self._mg_obs[mask] = torch.stack(
                    [_m_jl, _m_cone, _m_lat, _m_coll], dim=-1)
            if getattr(self.cfg, 'observe_proj_scales', False):
                # Same basis call as step(); at t=0 these are exactly the
                # scales the first control period will use, so the reset obs
                # is not even one period stale.
                _bo = build_task_aligned_basis(
                    self.kin, q0, self.line_dir[mask], self.n_target[mask],
                    self.kin.q_mid, self.q_half, self.cfg.manip_damping,
                    raw_scale=self.cfg.basis_raw_scale, return_scales=True)
                _df = int(getattr(self.cfg, 'dir_frac_action', 0) or 0)
                self._proj_scales[mask] = (_bo[3] if _df == 2
                                           else _bo[2]).float()

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
        dir_frac = int(getattr(self.cfg, 'dir_frac_action', 0) or 0)
        if dir_frac:
            raw_actions = actions                # (B, m+1) or (B, n+1)
            nd = self.act_dim if dir_frac == 1 else self.n_joints
            u_dir = actions[:, :nd]
            if getattr(self.cfg, 'rho_from_norm', False):
                rho = None                       # set after projection
                _sp_col = nd
            else:
                rho = 0.5 * (actions[:, nd] + 1.0)   # [0, 1]
                _sp_col = nd + 1
            # dir-frac + speed/gate: one extra trailing channel.
            if getattr(self.cfg, 'task_gate', False):
                speed_frac = (actions[:, _sp_col] > 0).to(actions.dtype)
            elif self.cfg.speed_levels:
                speed_frac = actions[:, _sp_col].clamp(
                    min(self.cfg.speed_levels), 1.0)
            else:
                speed_frac = None
            actions = u_dir
        elif self.cfg.speed_levels:
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
        want_scales = getattr(self.cfg, 'observe_proj_scales', False)
        _basis_out = build_task_aligned_basis(
            self.kin, self.q, self.line_dir, self.n_target,
            self.kin.q_mid, self.q_half, self.cfg.manip_damping,
            raw_scale=self.cfg.basis_raw_scale, return_scales=want_scales)
        if want_scales:
            B_basis, fb_mask, _sc_res, _sc_proj = _basis_out
            self._proj_scales = (_sc_proj.float()
                                 if int(getattr(self.cfg, 'dir_frac_action',
                                                0) or 0) == 2
                                 else _sc_res.float())
        else:
            B_basis, fb_mask = _basis_out

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
        if getattr(self.cfg, 'task_gate', False):
            # g=0 kills the ENTIRE task command (feed-forward and lateral
            # feedback): the paused step is pure self-motion.
            x_dot = x_dot * speed_frac.reshape(-1, 1, 1)
        qdot_task = (J_plus @ x_dot).squeeze(-1)
        if getattr(self.cfg, 'observe_headroom', False):
            _hr = torch.cat(
                [(self.qd_limit - qdot_task) / self.qd_limit,
                 (self.qd_limit + qdot_task) / self.qd_limit], dim=-1)
            self._headroom = torch.where(active.unsqueeze(-1), _hr,
                                         self._headroom)
        if dir_frac == 1:
            # v1: unit direction via the B chart
            un = actions / actions.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            xi_dir = (B_basis @ un.unsqueeze(-1)).squeeze(-1)   # unit, kerJ
        elif dir_frac == 2:
            # v2 (basis-free): joint-space output, exact null projection
            _, _, Vh_n = torch.linalg.svd(J_p.double(), full_matrices=True)
            Nn = Vh_n.transpose(-1, -2)[..., 3:]
            _dvm = int(getattr(self.cfg, 'dv_metric', 0) or 0)
            if _dvm:
                _Dv = (self.qd_limit.expand_as(qdot_task) if _dvm == 1 else
                       (self.qd_limit - qdot_task.abs()).clamp_min(
                           0.05 * self.qd_limit))
                _Js = (J_p * _Dv.unsqueeze(1)).double()   # column scaling
                _, _, Vh_n = torch.linalg.svd(_Js, full_matrices=True)
                Nn = Vh_n.transpose(-1, -2)[..., 3:]
                _z = (Nn @ (Nn.transpose(-1, -2)
                            @ actions.double().unsqueeze(-1))).squeeze(-1)
                xi_dir = _Dv.double() * _z
            else:
                xi_dir = (Nn @ (Nn.transpose(-1, -2)
                                @ actions.double().unsqueeze(-1))
                          ).squeeze(-1)
            _xi_nrm = xi_dir.norm(dim=-1, keepdim=True)
            xi_dir = (xi_dir / _xi_nrm.clamp_min(1e-6)).to(self.kin.dtype)
            if rho is None:                      # rho_from_norm
                rho = _xi_nrm.squeeze(-1).clamp(max=1.0).to(self.kin.dtype)
        if dir_frac:
            # max alpha >= 0 with |qdot_task + alpha*xi_dir| <= qd_limit
            denom_pos = xi_dir.clamp_min(1e-9)
            denom_neg = (-xi_dir).clamp_min(1e-9)
            head = (self.qd_limit - qdot_task) / denom_pos
            room = (self.qd_limit + qdot_task) / denom_neg
            bound = torch.where(xi_dir >= 0, head, room)
            if getattr(self.cfg, 'alpha_joint', False):
                # next-step joint-position limits, same per-joint linear
                # form: lmt_lo <= q + dt (qdot_task + alpha xi) <= lmt_up
                up_num = self.lmt_up - self.q - self.dt * qdot_task
                lo_num = self.q + self.dt * qdot_task - self.lmt_lo
                bj = torch.where(
                    xi_dir >= 0,
                    up_num / (self.dt * xi_dir.clamp_min(1e-9)),
                    lo_num / (self.dt * (-xi_dir).clamp_min(1e-9)))
                bound = torch.minimum(bound, bj)
            alpha_feas = bound.amin(dim=-1).clamp_min(0.0)
            qdot_null = (rho * alpha_feas).unsqueeze(-1) * xi_dir
            if getattr(self.cfg, 'a_prev_executed', False):
                _parts = [qdot_null / self.qd_limit]
                if not getattr(self.cfg, 'rho_from_norm', False):
                    _parts.append((2.0 * rho - 1.0).unsqueeze(-1))
                if speed_frac is not None:
                    _parts.append((2.0 * speed_frac - 1.0).unsqueeze(-1))
                raw_actions = torch.cat(_parts, dim=-1)
        else:
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

        if getattr(self.cfg, 'observe_margins', False):
            _m_jl = ((self.q_half - (q_new - self.q_mid).abs())
                     / self.q_half).amin(dim=-1)
            _m_cone = (cos_angle - self.cos_cone) / (1.0 - self.cos_cone)
            _m_lat = ((LATERAL_SAFETY_NET - lateral_err)
                      / LATERAL_SAFETY_NET)
            _m_coll = self.collision.min_margin(link_tfs) / 0.05
            _mg = torch.stack([_m_jl, _m_cone, _m_lat, _m_coll], dim=-1)
            self._mg_obs = torch.where(active.unsqueeze(-1), _mg,
                                       self._mg_obs)

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

        # Shift the a_prev history (newest first) before the post-step
        # snapshot so terminal_obs sees [a_t, a_{t-1}, ...] consistently.
        if self._ap_k > 1:
            _A = self.act_dim_policy
            _shifted = (torch.cat([self.a_prev, self._a_hist[:, :-_A]], -1)
                        if self._ap_k > 2 else self.a_prev)
            self._a_hist = torch.where(active.unsqueeze(-1), _shifted,
                                       self._a_hist)

        # snapshot of obs at end-of-step (post-step q, this-step actions);
        # PPO bootstraps V(terminal_obs) for truncated episodes.
        terminal_obs = self._compute_obs(
            p_new, R_new, q=q_new,
            a_prev=(raw_actions if dir_frac else actions))

        # Accumulate per-episode reward + step counter (before reset wipes them)
        ep_reward_finished = torch.zeros_like(reward)
        ep_steps_finished = torch.zeros_like(self.episode_steps)
        ep_progress_finished = torch.zeros_like(reward)

        if auto_reset:
            self.q = q_new
            self.t = new_t
            self.a_prev = raw_actions if dir_frac else actions
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
            self.a_prev = torch.where(
            active.unsqueeze(-1),
            (raw_actions if dir_frac else actions), self.a_prev)
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
            ep_progress_max = float(ep_progress_finished[new_done].max().item())
            ep_arc_mean = float(ep_arc_finished[new_done].mean().item())
        else:
            ep_reward_mean = float("nan")
            ep_len_mean = float("nan")
            ep_progress_mean = float("nan")
            ep_progress_max = float("nan")
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
            "ep_progress_max": ep_progress_max,
            "ep_arc_progress_mean": ep_arc_mean,
            "n_episodes_done": n_finished,
            # MGS fallback rates (batch-averaged) per anchor column.
            "fb_rate_e0": float(fb_mask[:, 0].float().mean().item()),
            "fb_rate_e1": float(fb_mask[:, 1].float().mean().item()),
            "fb_rate_e2": float(fb_mask[:, 2].float().mean().item()),
        }
        return obs, reward, terminated, truncated, info
