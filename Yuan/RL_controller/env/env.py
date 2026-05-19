"""Torch-batched Gymnasium-style env for FR3 position-only NSRL path-following.

Per rules.md, the "line" is an infinite ray (p_0, u_hat, n_target) — no length,
no success terminate. Agent's only objective is to maximize episode lifetime.

Per env state:
    q              (7,)  current joint config
    line_dir       (3,)  task direction u_hat (unit, world frame)
    n_target       (3,)  target normal for 30° cone (unit, world frame)
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

from Yuan.RL_controller.env.line_distribution import LineDistribution


OBS_DIM = 31
ACT_DIM = 4

# Framing B v4: lateral deviation is now controlled by a task-space PD term
# in the controller (see step()), not by reward or a tight termination cap.
# This bound is a safety net — only trips if PD feedback fails entirely.
LATERAL_SAFETY_NET = 0.02


@dataclass
class EnvConfig:
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
    # ---- Length-based shaping reward (sum-to-1 normalized weights) ----
    # Survival term now rewards EE TRAVEL not step count: each step earns
    # w_progress · clip(Δp · u_hat / (v·dt), 0, 1)  ∈ [0, w_progress].
    # When EE tracks the line at full speed → ~w_progress per step;
    # when stuck near singular → ~0 (no incentive to burn steps in place).
    #   r_t = w_progress · clip(Δp·u_hat/(v·dt), 0, 1)
    #       + w_jl   · scale · (||q_norm_prev||²_mean − ||q_norm_curr||²_mean)   # away from JL center
    #       + w_cone · scale · (cos_curr − cos_prev)                             # better alignment
    #       + w_dm   · scale · (w_u_curr − w_u_prev)                             # higher manipulability
    # Weights are normalized so w_progress + w_jl + w_cone + w_dm = 1 at runtime.
    w_progress: float = 0.25
    w_jl: float = 0.25
    w_cone: float = 0.25
    w_dm: float = 0.25
    delta_scale: float = 100.0        # K: per-step clipped-delta scale
    delta_lookback: int = 10          # N: metric_t − metric_{t−N}; defeats hi-freq oscillation hack
    w_terminal: float = 1.0           # weight on terminal telescoping bonus = K·Σw_i·(metric_final − metric_init)
    manip_damping: float = 1e-3


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

    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(J_p_d, full_matrices=True)
        N = Vh.transpose(-1, -2)[..., -4:]
        NNt = N @ N.transpose(-1, -2)

    p1 = (NNt @ g1d.unsqueeze(-1)).squeeze(-1)
    p2 = (NNt @ g2d.unsqueeze(-1)).squeeze(-1)
    p3 = (NNt @ g3d.unsqueeze(-1)).squeeze(-1)
    gnorm = torch.stack(
        [g1d.norm(dim=-1), g2d.norm(dim=-1), g3d.norm(dim=-1)],
        dim=-1).clamp_min(1e-20)

    used = torch.zeros((B_size, 4), dtype=torch.bool, device=device)
    arange4 = torch.arange(4, device=device).view(1, -1)
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
        used = used | ((~ok).unsqueeze(-1) & (arange4 == fb_idx.view(-1, 1)))
        fb_flags.append(~ok)
        return e

    e0 = _gs(p1, [], 0, g1d)
    e1 = _gs(p2, [e0], 1, g2d)
    e2 = _gs(p3, [e0, e1], 2, g3d)

    fb_idx = (~used).float().argmax(dim=-1)
    n_col = torch.gather(
        N, -1, fb_idx.view(-1, 1, 1).expand(-1, d, 1)).squeeze(-1)
    v3 = n_col
    for e in (e0, e1, e2):
        v3 = v3 - (v3 * e).sum(-1, keepdim=True) * e
    for e in (e0, e1, e2):
        v3 = v3 - (v3 * e).sum(-1, keepdim=True) * e
    e3 = v3 / v3.norm(dim=-1, keepdim=True).clamp_min(eps_abs)

    B_basis = torch.stack([e0, e1, e2, e3], dim=-1).to(dtype)
    fb_mask = torch.stack(fb_flags, dim=-1)
    return B_basis, fb_mask


# term_reason codes (success=1 removed: no success terminate in infinite-ray task)
TERM_ALIVE = 0
TERM_COLLISION = 2
TERM_CONE = 3
TERM_JL = 4
TERM_TRUNCATED = 5
TERM_LATERAL = 6


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
        # Normalize reward weights so they sum to 1
        total_w = cfg.w_progress + cfg.w_jl + cfg.w_cone + cfg.w_dm
        if abs(total_w - 1.0) > 1e-3:
            print(f"[env] normalizing reward weights: sum={total_w:.4f} → 1.0")
            self._w_progress = cfg.w_progress / total_w
            self._w_jl = cfg.w_jl / total_w
            self._w_cone = cfg.w_cone / total_w
            self._w_dm = cfg.w_dm / total_w
        else:
            self._w_progress = cfg.w_progress
            self._w_jl = cfg.w_jl
            self._w_cone = cfg.w_cone
            self._w_dm = cfg.w_dm
        self.kin = BatchedFR3Kinematics(device=self.device, tcp_offset=cfg.tcp_offset)
        self.collision = FR3SphereCollision(device=self.device)
        self.line_dist = line_dist

        self.lmt_lo = self.kin.lmt_lo
        self.lmt_up = self.kin.lmt_up
        self.q_mid = self.kin.q_mid
        self.q_half = 0.5 * (self.lmt_up - self.lmt_lo)

        B = self.n_envs
        d = self.kin.dtype
        self.q = torch.zeros((B, 7), device=self.device, dtype=d)
        self.line_dir = torch.zeros((B, 3), device=self.device, dtype=d)
        self.n_target = torch.zeros((B, 3), device=self.device, dtype=d)
        self.t = torch.zeros((B,), device=self.device, dtype=torch.long)
        self.a_prev = torch.zeros((B, ACT_DIM), device=self.device, dtype=d)
        self.done_persistent = torch.zeros((B,), device=self.device, dtype=torch.bool)
        # N-step lookback ring buffers for clipped delta reward.
        # Each metric has shape (B, N). NaN sentinel ⇒ delta = 0 for first N steps.
        N = int(cfg.delta_lookback)
        self.q_norm_sq_hist = torch.full((B, N), float("nan"), device=self.device, dtype=d)
        self.cos_angle_hist = torch.full((B, N), float("nan"), device=self.device, dtype=d)
        self.w_u_hist = torch.full((B, N), float("nan"), device=self.device, dtype=d)
        # Episode-start metric values for terminal telescoping bonus
        self.q_norm_sq_init = torch.full((B,), float("nan"), device=self.device, dtype=d)
        self.cos_angle_init = torch.full((B,), float("nan"), device=self.device, dtype=d)
        self.w_u_init = torch.full((B,), float("nan"), device=self.device, dtype=d)
        # Cumulative per-episode reward (logged on done)
        self.episode_reward = torch.zeros((B,), device=self.device, dtype=d)
        self.episode_steps = torch.zeros((B,), device=self.device, dtype=torch.long)
        # EE position at episode start; used to compute progress = (p_now - p_start)·u_hat
        self.p_start = torch.full((B, 3), float("nan"), device=self.device, dtype=d)

    # ---------------------------------------------------------------- helpers

    def _compute_obs(self, R_tcp: torch.Tensor) -> torch.Tensor:
        z_tool = R_tcp[:, :, 2]
        q_norm = (self.q - self.q_mid) / self.q_half
        # Explicit cone-relevant features so the actor doesn't have to
        # learn dot/cross from raw vectors (ReLU MLPs are bad at it):
        cos_angle = (z_tool * self.n_target).sum(-1, keepdim=True)  # 1
        z_cross_n = torch.linalg.cross(z_tool, self.n_target, dim=-1)  # 3
        # Quadratic JL proximity — makes |q_i_norm| close-to-1 directly visible
        q_norm_sq = q_norm * q_norm  # 7
        return torch.cat([
            q_norm,         # 7
            q_norm_sq,      # 7
            self.line_dir,  # 3, u_hat
            z_tool,         # 3
            self.n_target,  # 3
            cos_angle,      # 1
            z_cross_n,      # 3
            self.a_prev,    # 4
        ], dim=-1)          # = 31

    def current_obs(self) -> torch.Tensor:
        """Public: obs at the current internal state (no step taken)."""
        _, R, _, _ = self.kin.tcp_fk_jac(self.q)
        return self._compute_obs(R)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        n_reset = int(mask.sum().item())
        if n_reset == 0:
            return
        spec = self.line_dist.sample(n_reset)
        self.q[mask] = spec["q0"]
        self.line_dir[mask] = spec["line_dir"]
        self.n_target[mask] = spec["n_target"]
        self.t[mask] = 0
        self.a_prev[mask] = 0
        self.done_persistent[mask] = False
        self.q_norm_sq_hist[mask] = float("nan")
        self.cos_angle_hist[mask] = float("nan")
        self.w_u_hist[mask] = float("nan")
        self.q_norm_sq_init[mask] = float("nan")  # will be set on first step
        self.cos_angle_init[mask] = float("nan")
        self.w_u_init[mask] = float("nan")
        self.episode_reward[mask] = 0.0
        self.episode_steps[mask] = 0
        # Compute EE start position for the reset envs (FK on the just-sampled q0)
        p_at_reset, _, _, _ = self.kin.tcp_fk_jac(self.q[mask])
        self.p_start[mask] = p_at_reset

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
        a_scaled = actions * self.a_max
        active = ~self.done_persistent  # only matters when auto_reset=False

        p, R, J, _ = self.kin.tcp_fk_jac(self.q)
        J_p = J[:, :3, :]

        J_plus, sigma_min = damped_pinv(J_p, self.cfg.lambda_0, self.cfg.sigma_thr)
        B_basis, fb_mask = build_task_aligned_basis(
            self.kin, self.q, self.line_dir, self.n_target,
            self.kin.q_mid, self.q_half, self.cfg.manip_damping,
        )

        # Framing B v4: lateral correction is the controller's job, not the
        # policy's. p_ref tracks the ideal point on the line at the current
        # episode time; K_P pulls TCP back toward it in task space, leaving
        # the nullspace projection (and reward) free of lateral concerns.
        K_P = 5.0
        p_ref = self.p_start + (self.v * self.dt * self.t.to(p.dtype)).unsqueeze(-1) * self.line_dir
        x_dot = (self.v * self.line_dir + K_P * (p_ref - p)).unsqueeze(-1)
        qdot_task = (J_plus @ x_dot).squeeze(-1)
        qdot_null = (B_basis @ a_scaled.unsqueeze(-1)).squeeze(-1)
        qdot = qdot_task + qdot_null
        q_new = self.q + qdot * self.dt

        link_tfs = self.kin.link_transforms(q_new)
        p_new, R_new, J_new, _ = self.kin.tcp_fk_jac(q_new)
        z_new = R_new[:, :, 2]
        J_p_new = J_new[:, :3, :]

        is_coll = self.collision.is_collided(link_tfs)
        jl_viol = ((q_new < self.lmt_lo) | (q_new > self.lmt_up)).any(dim=-1)
        cos_angle = (z_new * self.n_target).sum(-1).clamp(-1.0, 1.0)
        cone_viol = cos_angle < self.cos_cone

        # Lateral deviation from the infinite ray (p_start, line_dir).
        # lateral_err = ‖(p − p_0) − ((p − p_0)·û) û‖. Safety-net terminate
        # only — PD feedback in the controller should keep this far below cap.
        delta_from_start = p_new - self.p_start
        along = (delta_from_start * self.line_dir).sum(-1, keepdim=True)
        lateral_err = (delta_from_start - along * self.line_dir).norm(dim=-1)
        lateral_viol = lateral_err > LATERAL_SAFETY_NET

        new_t = self.t + 1
        truncated = new_t >= self.max_steps

        # Per-step EE travel along u_hat (meters this step). Damped pinv
        # guarantees this is ≤ v·dt; clip to non-negative to ignore numerical
        # backwards drift from finite-dt curvature.
        delta_progress = ((p_new - p) * self.line_dir).sum(-1)
        progress_norm = (delta_progress / (self.v * self.dt)).clamp(0.0, 1.0)
        r_progress_per_env = self._w_progress * progress_norm

        # progress survival base + terminal penalty modifiers (penalties set to 0 by default)
        reward = r_progress_per_env.clone()
        reward = torch.where(is_coll, reward + self.cfg.r_collision, reward)
        reward = torch.where(cone_viol & ~is_coll, reward + self.cfg.r_cone, reward)
        reward = torch.where(jl_viol & ~is_coll & ~cone_viol,
                             reward + self.cfg.r_jl, reward)

        # ---- Incremental-delta shaping (positive = improvement) ----
        q_norm_new = (q_new - self.q_mid) / self.q_half
        q_norm_sq_now = (q_norm_new * q_norm_new).mean(-1)  # ∈ [0, 1]
        # directional manipulability w_u(q_new, u_hat)
        # Use torch.linalg.solve (Cholesky-friendly) + nan_to_num for numerical
        # robustness; torch.linalg.inv occasionally returns NaN on FR3 configs
        # with high condition number.
        JJt_dmp = (J_p_new @ J_p_new.transpose(-1, -2)
                   + (self.cfg.manip_damping ** 2) * torch.eye(
                       3, device=self.device, dtype=self.kin.dtype
                   ).expand(self.n_envs, 3, 3))
        u_col = self.line_dir.unsqueeze(-1)
        sol = torch.linalg.solve(JJt_dmp, u_col)  # JJt_dmp @ sol = u_col
        inv_quad = (u_col.transpose(-1, -2) @ sol).squeeze(-1).squeeze(-1)
        inv_quad = torch.nan_to_num(inv_quad, nan=1.0, posinf=1e12, neginf=1e-12).clamp_min(1e-12)
        w_u_now = inv_quad.pow(-0.5).clamp_min(1e-6)
        w_u_now = torch.nan_to_num(w_u_now, nan=1e-6)  # final safety net
        cos_now = cos_angle  # already clamped to [-1, 1]

        # Initialize episode-start metric values on first step (NaN sentinel)
        first_step_mask = torch.isnan(self.q_norm_sq_init)
        self.q_norm_sq_init = torch.where(first_step_mask, q_norm_sq_now, self.q_norm_sq_init)
        self.cos_angle_init = torch.where(first_step_mask, cos_now, self.cos_angle_init)
        self.w_u_init = torch.where(first_step_mask, w_u_now, self.w_u_init)

        # N-step lookback delta: compare metric_now vs metric_{t-N}.
        # Ring buffer slot for current step t is (t % N); reading from same
        # slot BEFORE overwriting gives metric_{t-N}.
        N = self.cfg.delta_lookback
        slot = (self.t % N).long()  # (B,) write/read slot for THIS step
        idx = slot.unsqueeze(-1)  # gather index along dim 1
        # Read metric_{t-N} from each env's slot
        old_qsq = self.q_norm_sq_hist.gather(1, idx).squeeze(-1)
        old_cos = self.cos_angle_hist.gather(1, idx).squeeze(-1)
        old_wu = self.w_u_hist.gather(1, idx).squeeze(-1)
        no_old = torch.isnan(old_qsq)  # first N steps: no history yet → delta = 0
        zero = torch.zeros_like(q_norm_sq_now)
        # Mod 4 rolled back: positive-only delta (only reward improvement,
        # never penalize per-step degradation). Telescoping bonus on the
        # final step still captures net signed init→final change.
        delta_jl = torch.where(no_old, zero,
                               (old_qsq - q_norm_sq_now).clamp(min=0.0))
        delta_cone = torch.where(no_old, zero,
                                 (cos_now - old_cos).clamp(min=0.0))
        delta_dm = torch.where(no_old, zero,
                               (w_u_now - old_wu).clamp(min=0.0))

        scale = self.cfg.delta_scale
        r_jl_term = self._w_jl * scale * delta_jl
        r_cone_term = self._w_cone * scale * delta_cone
        r_dm_term = self._w_dm * scale * delta_dm
        reward = reward + r_jl_term + r_cone_term + r_dm_term

        # Terminal telescoping bonus: at episode end, add signed
        # K · Σ w_i · (metric_final − metric_initial). Captures overall
        # trajectory improvement (positive) or degradation (negative).
        # Framing B: lateral_viol is a terminating condition (hard constraint),
        # not bootstrapped. NOT included as bootstrap-truncation.
        terminated = is_coll | cone_viol | jl_viol | lateral_viol
        done_now_for_bonus = terminated | truncated  # episodes ending this step
        bonus = (
            self._w_jl * scale * (self.q_norm_sq_init - q_norm_sq_now)
            + self._w_cone * scale * (cos_now - self.cos_angle_init)
            + self._w_dm * scale * (w_u_now - self.w_u_init)
        ) * self.cfg.w_terminal
        # Only apply bonus on the step where episode ends, and only for envs
        # whose init was set (i.e., we've had at least one step). Guard NaN init.
        bonus = torch.where(torch.isnan(self.q_norm_sq_init), zero, bonus)
        reward = reward + torch.where(done_now_for_bonus, bonus, zero)

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

        terminal_obs = self._compute_obs(R_new)  # snapshot of pre-reset obs

        # Accumulate per-episode reward + step counter (before reset wipes them)
        ep_reward_finished = torch.zeros_like(reward)  # cumulative ep reward for envs that just ended
        ep_steps_finished = torch.zeros_like(self.episode_steps)
        # EE progress along u_hat = (p_final - p_start) · u_hat (meters). Per episode end.
        progress_now = ((p_new - self.p_start) * self.line_dir).sum(-1)
        # Guard NaN (shouldn't happen since p_start is set in _reset_envs, but for first reset edge)
        progress_now = torch.nan_to_num(progress_now, nan=0.0)
        ep_progress_finished = torch.zeros_like(reward)

        if auto_reset:
            self.q = q_new
            self.t = new_t
            self.a_prev = actions
            # Write current metric values into ring buffer at slot for this step
            self.q_norm_sq_hist.scatter_(1, idx, q_norm_sq_now.unsqueeze(-1))
            self.cos_angle_hist.scatter_(1, idx, cos_now.unsqueeze(-1))
            self.w_u_hist.scatter_(1, idx, w_u_now.unsqueeze(-1))
            self.episode_reward = self.episode_reward + reward
            self.episode_steps = self.episode_steps + 1
            new_done = done
            # Snapshot finished episodes' cumulative reward + length before reset
            ep_reward_finished = torch.where(done, self.episode_reward,
                                             torch.zeros_like(self.episode_reward))
            ep_steps_finished = torch.where(done, self.episode_steps,
                                            torch.zeros_like(self.episode_steps))
            ep_progress_finished = torch.where(done, progress_now,
                                               torch.zeros_like(progress_now))
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
            # Conditionally write history slots (only for active envs)
            new_qsq = torch.where(active, q_norm_sq_now, old_qsq)
            new_cos = torch.where(active, cos_now, old_cos)
            new_wu = torch.where(active, w_u_now, old_wu)
            self.q_norm_sq_hist.scatter_(1, idx, new_qsq.unsqueeze(-1))
            self.cos_angle_hist.scatter_(1, idx, new_cos.unsqueeze(-1))
            self.w_u_hist.scatter_(1, idx, new_wu.unsqueeze(-1))
            self.episode_reward = torch.where(active, self.episode_reward + reward,
                                              self.episode_reward)
            self.episode_steps = torch.where(active, self.episode_steps + 1,
                                             self.episode_steps)
            self.done_persistent = self.done_persistent | done
            ep_reward_finished = torch.where(new_done, self.episode_reward,
                                             torch.zeros_like(self.episode_reward))
            ep_steps_finished = torch.where(new_done, self.episode_steps,
                                            torch.zeros_like(self.episode_steps))
            ep_progress_finished = torch.where(new_done, progress_now,
                                               torch.zeros_like(progress_now))
            obs = self.current_obs()

        # Episode-reward stats (only mean over envs that finished this step;
        # default 0/0 → reported as nan if no env finished). Used by PPO log_fn.
        n_finished = int(new_done.sum().item())
        if n_finished > 0:
            ep_reward_mean = float(ep_reward_finished[new_done].mean().item())
            ep_len_mean = float(ep_steps_finished[new_done].float().mean().item())
            ep_progress_mean = float(ep_progress_finished[new_done].mean().item())
        else:
            ep_reward_mean = float("nan")
            ep_len_mean = float("nan")
            ep_progress_mean = float("nan")

        # Terminal bonus magnitude (mean over envs that ended this step)
        if n_finished > 0:
            terminal_bonus_mean = float(bonus[new_done].mean().item())
        else:
            terminal_bonus_mean = float("nan")

        info = {
            "terminal_obs": terminal_obs,
            "term_reason": term_reason,
            "sigma_min": sigma_min,
            "episode_done": new_done,
            # Per-term reward means (scalars) for wandb decomposition panels
            "r_progress_mean": float(r_progress_per_env.mean().item()),
            "r_jl_mean": float(r_jl_term.mean().item()),
            "r_cone_mean": float(r_cone_term.mean().item()),
            "r_dm_mean": float(r_dm_term.mean().item()),
            "w_u_mean": float(w_u_now.mean().item()),
            "lateral_err_mean": float(lateral_err.mean().item()),
            "lateral_err_max": float(lateral_err.max().item()),
            "r_terminal_mean": terminal_bonus_mean,  # signed; only on done step
            "ep_reward_mean": ep_reward_mean,        # mean cum reward of envs that finished
            "ep_len_mean": ep_len_mean,              # mean length of envs that finished
            "ep_progress_mean": ep_progress_mean,    # mean (p_final − p_start)·u_hat in METERS
            "n_episodes_done": n_finished,
            # MGS fallback rates (batch-averaged) per anchor column.
            "fb_rate_e0": float(fb_mask[:, 0].float().mean().item()),
            "fb_rate_e1": float(fb_mask[:, 1].float().mean().item()),
            "fb_rate_e2": float(fb_mask[:, 2].float().mean().item()),
        }
        return obs, reward, terminated, truncated, info
