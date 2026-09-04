"""Nova2 tabletop drag environment with dirfrac-v2 null-space mechanics.

Task: the gripper has already rigidly grasped an object resting on the
table; drag it in SE(2) from its start pose to a goal xy position. The
goal yaw is free, the path and speed are free.

Fixed task components (3, equality, held by projection + feedback):
    - TCP height z (object stays on the table plane),
    - no tilt: world-frame angular velocity x, y = 0 (the grasp axis
      z_g keeps its initial direction).
Free components (3 = ker J_c): planar translation + yaw. These are the
policy's to use between start and goal.

Action (7 channels), dirfrac v2 (basis-free, velocity-limited):
    u in [-1,1]^6  joint-space direction proposal,
    rho = (a_7+1)/2 in [0,1]  fraction of the feasible amplitude.
u is projected by the exact null projector of J_c (SVD) and normalized
to a unit direction xi; alpha_feas is the per-joint closed-form bound
with  |qdot_hold + alpha xi| <= qd_limit;  the executed command is
    qdot = qdot_hold + rho * alpha_feas * xi.
No a_max, no Gram-Schmidt basis. qdot_hold is the 3-dim constraint
drift feedback (nominally zero).

Reward: ManiSkill-style staged dense reward,
    r = (1 - tanh(3 d_goal)) + is_placed * (1 - tanh(5 |qdot/qd_lim|)),
    r[success] = 3,  success = is_placed & is_static.
Success does NOT terminate the episode (hovering next to the goal would
otherwise strictly beat succeeding); constraint violations terminate
with no extra penalty -- the cost of dying is the foregone reward.

All heavy machinery is imported read-only from the IJRR dirfrac branch
(BatchedChainKinematics, damped_pinv); nothing outside Yuan/Qling is
modified.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import torch

from .ijrr_root import add_ijrr_path
add_ijrr_path()

from Yuan.IJRR.kinematics.batched_chain_kin import BatchedChainKinematics  # noqa: E402
from Yuan.IJRR.env.env import damped_pinv                                  # noqa: E402

from .nova2_spec import NOVA2, GRIPPER_TCP_OFFSET                          # noqa: E402
from .nova2_collision import Nova2DragCollision                            # noqa: E402


@dataclass
class DragEnvConfig:
    n_envs: int = 128
    dt: float = 0.05                     # 20 Hz control, same as v2
    device: str = 'cpu'
    seed: int = 0

    # dirfrac mechanics
    qd_limit: tuple = (math.radians(135.0),) * 6   # official Nova2 spec
    lambda_0: float = 0.05               # DLS damping baseline (v2 value)
    sigma_thr: float = 0.05
    k_z: float = 8.0                     # m/s per m of height drift
    k_tilt: float = 8.0                  # rad/s per rad of tilt drift

    # table / workspace (Qin Liang's sampling range; base frame = world,
    # table top at z = 0 per the regrasp env's base collision box)
    table_z: float = 0.0
    sample_x: tuple = (-0.7, 0.7)
    sample_y: tuple = (0.1, 1.0)
    workspace_margin: float = 0.02       # object may exceed range by this

    # start-state pool (task-from-configuration, v0: no grasp table yet)
    n_pool: int = 10000
    z_band: tuple = (0.03, 0.25)         # TCP height above table at start
    tilt_max_start_deg: float = 20.0     # z_g within this of straight-down
    pool_jl_margin: float = 0.95         # |q_bar| <= this at start
    min_goal_dist: float = 0.15
    max_goal_dist: float = 1.0
    # goals farther than this from the base are outside the Nova2 working
    # radius (0.625 m official) and can never be placed; keep a small
    # interior margin so the goal itself is not at the reach boundary
    goal_max_radius: float = 0.58

    # episode
    max_steps: int = 400                 # 20 s at 20 Hz
    goal_eps: float = 0.025              # is_placed radius (m)
    static_eps: float = 0.05             # |qdot/qd_lim| for is_static
    eps_z: float = 0.01                  # drift safety net (m)
    eps_tilt_deg: float = 3.0            # drift safety net (deg)

    # collision: generated sphere sets (drag/data/spheres/nova2), arm
    # self-collision incl. base + table halfspace over links 2..6.
    # Gripper spheres participate in self-collision only (its table
    # clearance is constant while z and tilt are locked).
    pool_chunk: int = 2048               # pairwise margins are memory-heavy

    # post-step Newton re-projection onto the constraint manifold
    # (z = z_ref, z_g = zg_ref). Kills integration drift by construction;
    # the velocity feedback k_z/k_tilt then only smooths within a step
    # and eps_z / eps_tilt become true never-firing safety nets.
    n_project_iters: int = 2

    # reward
    success_reward: float = 3.0

    # start-state curriculum, honored by subclasses that define one
    # ('mixed' = also start from mid-maneuver states; 'wp0' = task start
    # only). The base env ignores it.
    start_mode: str = 'mixed'

    # collision-margin speed governor: scale alpha_feas by
    # clamp(margin / margin_speed_scale, 0.12, 1). At the full velocity
    # budget one control step moves the TCP by centimetres, which is
    # instantly lethal inside mm-clearance passages; the governor makes
    # the feasible amplitude margin-aware the same way alpha_joint makes
    # it position-limit-aware. 0 disables (v0 behavior).
    margin_speed_scale: float = 0.0

    # free-regrasp iteration levers (rg45f3 autopsy: policies rotate
    # ~1 m afield, then strand -- the return leg has no gradient
    # off-plan). Only RegraspFreeEnv reads these.
    #   return_bonus: weight of a linear pull-home reward term once the
    #       yaw is solved (0 disables).
    #   ret_start_frac: fraction of resets started at far-field
    #       yaw-solved poses (return-leg curriculum; 0 disables).
    return_bonus: float = 0.0
    ret_start_frac: float = 0.0
    #   max_switches: per-episode regrasp budget. 6 pins at the cap in
    #       every rg45f* autopsy BEFORE the return leg -- a free
    #       (unplanned) extract+rotate burns 4-5 switches, leaving none
    #       for the insertion regrasp.
    max_switches: int = 6
    #   pose_curriculum: PLAN-FREE start curriculum -- mid states are
    #       sampled directly in container SE(2) pose space (goal-backoff
    #       corridor + free field, feasible-grasp filtered), never from
    #       a reference maneuver. Replaces the plan-state curriculum.
    pose_curriculum: bool = False
    #   rand_switch_budget: curriculum starts draw n_switches uniform
    #       in {0..max_switches} instead of always 0 -- true starts
    #       reach mid-maneuver states with the budget partly/fully
    #       spent, and a policy trained only on fresh-budget states is
    #       OOD there (rg45f6: mid-bucket 0.371 with fresh budget vs
    #       0.000 true-start, switches pinned at cap in every run).
    rand_switch_budget: bool = False
    #   switch_freeze: steps the arm is "in transit" after a committed
    #       switch (no motion, no release). 1 = legacy instant switch;
    #       realistic regrasping takes seconds, and at 1 step the
    #       policy exploits switches as free arm teleportation.
    switch_freeze: int = 1
    #   smooth_return: replace the yaw_ok-GATED pull-home term with a
    #       continuous product (1-|yaw_err|/pi)^2 * (1-d/1.2) -- the
    #       10-deg gate left the whole return leg without gradient
    #       outside a razor-thin yaw window (rg45f6/7/8: strand at
    #       0.7-1.1 m right after rotating).
    smooth_return: bool = False
    #   s1_yaw_gate_deg: yaw window that exempts the extraction term
    #       (re-approaching the slot re-raises d_extract; inside this
    #       window s1 stays full so returning is not punished).
    #       0 = legacy (use yaw_tol).
    s1_yaw_gate_deg: float = 0.0
    #   pose-curriculum bucket shares (true share = 1 - back - field).
    back_frac: float = 0.30
    field_frac: float = 0.30
    #   switch_explore_eps: with this probability a TRAINING-time
    #       committed switch picks a random feasible candidate instead
    #       of the critic argmax. Breaks the selector's
    #       self-confirmation trap (rg45f10a: from mouth-grasp states
    #       the critic re-picks a mouth grasp 85% of the time at 0.008
    #       success, while the non-mouth picks it never explores would
    #       succeed at 0.652). Eval (wp0) always uses pure argmax.
    switch_explore_eps: float = 0.0
    #   slope_theta_deg: ramp angle of the bottle plane-to-slope task
    #       (only BottleSlopeEnv reads it).
    slope_theta_deg: float = 15.0
    #   rand_goals: sample a random SE(2)-on-surface goal per episode
    #       (position across flat AND slope, heading +-45 deg) instead
    #       of the fixed straight-up-slope goal.
    rand_goals: bool = False
    #   rand_starts: sample the START pose from the same surface
    #       distribution as the goals (any-to-any surface policy).
    rand_starts: bool = False
    #   goal_yaw_range_deg: half-range of sampled headings.
    goal_yaw_range_deg: float = 45.0
    #   yaw_tol_deg: success yaw tolerance override (0 = env default).
    yaw_tol_deg: float = 0.0


class Nova2DragEnv:
    """Batched kinematic drag env. Interface mirrors NSRLBatchedEnv:
    reset() -> obs; step(a) -> (obs, reward, terminated, truncated, info);
    auto_reset semantics identical (terminal_obs in info)."""

    OBS_DIM = 27   # 6 q + 6 q^2 + 2 u_goal + 1 d + 2 yaw + 7 a_prev + 3 margins
    ACT_DIM = 7    # 6 direction + 1 rho

    def __init__(self, cfg: DragEnvConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.kin = BatchedChainKinematics(
            NOVA2, device=self.device, dtype=torch.float32,
            tcp_offset=GRIPPER_TCP_OFFSET)
        self.coll = Nova2DragCollision(device=self.device,
                                       dtype=torch.float32)
        self.n_joints = self.kin.n_joints
        B = cfg.n_envs
        self.gen = torch.Generator(device='cpu').manual_seed(cfg.seed)

        self.qd_limit = torch.tensor(cfg.qd_limit, device=self.device)
        self.lmt_lo, self.lmt_up = self.kin.lmt_lo, self.kin.lmt_up
        self.q_mid = self.kin.q_mid
        self.q_half = 0.5 * (self.lmt_up - self.lmt_lo)

        # per-env state
        self.q = torch.zeros((B, self.n_joints), device=self.device)
        self.z_ref = torch.zeros(B, device=self.device)
        self.zg_ref = torch.zeros((B, 3), device=self.device)
        self.start_xy = torch.zeros((B, 2), device=self.device)
        self.goal_xy = torch.zeros((B, 2), device=self.device)
        self.L0 = torch.ones(B, device=self.device)
        self.steps = torch.zeros(B, dtype=torch.long, device=self.device)
        self.a_prev = torch.zeros((B, self.ACT_DIM), device=self.device)
        self.qdot = torch.zeros((B, self.n_joints), device=self.device)
        self.done_persistent = torch.zeros(B, dtype=torch.bool,
                                           device=self.device)
        # collision margin cache: refreshed post-step and on reset so the
        # pairwise sphere pass runs once per transition, not twice
        self._coll_margin = torch.full((B,), 1.0, device=self.device)
        # episode accounting for PPO logging
        self.ep_reward = torch.zeros(B, device=self.device)
        self.ep_len = torch.zeros(B, device=self.device)

        # PPO interface (stage2_traj.ppo.train contract)
        self.n_envs = B
        self.obs_dim = self.OBS_DIM
        self.act_dim = self.ACT_DIM

        self._build_start_pool()
        self._move_tensors_to_device()

    def _move_tensors_to_device(self):
        """Blanket device normalization: constants built without an
        explicit device (grasp caches, box tables, ...) all land on
        self.device. Idempotent; subclasses call it again at the end of
        their own __init__ to cover attributes set after super()."""
        for k, v in list(vars(self).items()):
            if torch.is_tensor(v):
                setattr(self, k, v.to(self.device))
            elif (isinstance(v, list) and v
                  and all(torch.is_tensor(t) for t in v)):
                setattr(self, k, [t.to(self.device) for t in v])

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _frames(self, q):
        p, R, J, T_last = self.kin.tcp_fk_jac(q)
        z_g = R[:, :, 2]                       # gripper approach axis (world)
        return p, R, J, z_g

    def _collision_margin(self, q):
        """Min signed clearance (m): sphere self-collision (incl. base
        and gripper) and arm-vs-table halfspace."""
        tfs = Nova2DragCollision.augment(self.kin.link_transforms(q))
        return self.coll.combined_margin(tfs, self.cfg.table_z)

    def _object_xy(self, p, R):
        """Planar position of the manipulated object. Base env: the TCP
        point itself; subclasses with a grasp offset override this."""
        return p[:, :2]

    def _jl_margin(self, q):
        """Min normalized distance to the nearest joint limit, in [0,1]."""
        q_bar = (q - self.q_mid) / self.q_half
        return (1.0 - q_bar.abs()).amin(dim=1)

    def _workspace_margin(self, obj_xy):
        cx, cy = self.cfg.sample_x, self.cfg.sample_y
        m = torch.stack([
            obj_xy[:, 0] - cx[0], cx[1] - obj_xy[:, 0],
            obj_xy[:, 1] - cy[0], cy[1] - obj_xy[:, 1]], dim=1)
        return m.amin(dim=1) + self.cfg.workspace_margin

    # ------------------------------------------------------------------
    # start-state pool (v0: task-from-configuration; the real grasp-table
    # x IK-branch seed stage replaces this later)
    # ------------------------------------------------------------------
    def _pool_stamp(self):
        cfg = self.cfg
        import hashlib
        from .nova2_collision import SPHERE_DIR
        h = hashlib.md5()
        for fp in sorted(SPHERE_DIR.glob('*.json')):
            h.update(fp.read_bytes())
        return dict(seed=cfg.seed, n_pool=cfg.n_pool,
                    z_band=list(cfg.z_band),
                    tilt=cfg.tilt_max_start_deg,
                    jl=cfg.pool_jl_margin,
                    sx=list(cfg.sample_x), sy=list(cfg.sample_y),
                    qd=list(cfg.qd_limit),
                    spheres=h.hexdigest(),      # collision model fingerprint
                    version=3)

    def _build_start_pool(self):
        """Rejection + Newton-repair sampling of valid start states.

        Random configurations whose grasp axis is within
        tilt_max_start_deg of straight-down are Newton-projected onto an
        EXACTLY vertical z_g (and z clamped into the band), then filtered
        for workspace, joint-limit interior and collision. Every pooled
        start therefore has zg_ref = -z_hat exactly -- a clean top-down
        grasp. The pool is cached on disk with its filter parameters;
        changing any stamped parameter triggers a rebuild.
        """
        import json as _json
        import numpy as _np
        cfg = self.cfg
        cache = os.path.join(os.path.dirname(__file__), 'data',
                             f'q0_pool_s{cfg.seed}.npz')
        stamp = _json.dumps(self._pool_stamp(), sort_keys=True)
        if os.path.exists(cache):
            d = _np.load(cache, allow_pickle=False)
            if str(d['stamp']) == stamp:
                self.q0_pool = torch.as_tensor(
                    d['pool'], device=self.device, dtype=torch.float32)
                return
        cos_tilt = math.cos(math.radians(cfg.tilt_max_start_deg))
        z_down = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        chunks, total = [], 0
        # sample uniformly inside limits AND one turn (clamping after
        # scaling would pile ~half the wrist samples exactly at +-pi)
        lo = torch.maximum(self.q_mid - cfg.pool_jl_margin * self.q_half,
                           torch.full_like(self.q_mid, -math.pi))
        hi = torch.minimum(self.q_mid + cfg.pool_jl_margin * self.q_half,
                           torch.full_like(self.q_mid, math.pi))
        for it in range(4000):
            u = torch.rand((cfg.pool_chunk, self.n_joints),
                           generator=self.gen)
            q = lo + u.to(self.device) * (hi - lo)
            p, R, J, z_g = self._frames(q)
            rough = ((-z_g[:, 2] >= cos_tilt)
                     & (p[:, 2] >= cfg.table_z + cfg.z_band[0] - 0.05)
                     & (p[:, 2] <= cfg.table_z + cfg.z_band[1] + 0.05))
            if not rough.any():
                continue
            q = q[rough]
            p = p[rough]
            z_tgt = p[:, 2].clamp(cfg.table_z + cfg.z_band[0],
                                  cfg.table_z + cfg.z_band[1])
            q = self._project_q(q, z_tgt, z_down.expand(q.shape[0], 3),
                                iters=6)
            p, R, J, z_g = self._frames(q)
            ok = (p[:, 2] - z_tgt).abs() < 1e-4
            ok &= (-z_g[:, 2]) >= math.cos(math.radians(0.1))
            ok &= (p[:, 0] >= cfg.sample_x[0]) & (p[:, 0] <= cfg.sample_x[1])
            ok &= (p[:, 1] >= cfg.sample_y[0]) & (p[:, 1] <= cfg.sample_y[1])
            q_bar = (q - self.q_mid) / self.q_half
            ok &= q_bar.abs().amax(dim=1) <= cfg.pool_jl_margin
            ok &= self._collision_margin(q) > 0.0
            chunks.append(q[ok])
            total += int(ok.sum())
            if total >= cfg.n_pool:
                break
        pool = torch.cat(chunks, dim=0)
        if pool.shape[0] < 64:
            raise RuntimeError(
                f'start pool too small ({pool.shape[0]}); relax filters')
        self.q0_pool = pool[:cfg.n_pool]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        _np.savez(cache, pool=self.q0_pool.cpu().numpy(), stamp=stamp)

    def _sample_goals(self, start_xy):
        cfg = self.cfg
        B = start_xy.shape[0]
        goal = torch.empty_like(start_xy)
        todo = torch.ones(B, dtype=torch.bool, device=self.device)
        for _ in range(64):
            u = torch.rand((B, 2), generator=self.gen).to(self.device)
            cand = torch.stack([
                cfg.sample_x[0] + u[:, 0] * (cfg.sample_x[1] - cfg.sample_x[0]),
                cfg.sample_y[0] + u[:, 1] * (cfg.sample_y[1] - cfg.sample_y[0]),
            ], dim=1)
            d = (cand - start_xy).norm(dim=1)
            ok = (todo & (d >= cfg.min_goal_dist) & (d <= cfg.max_goal_dist)
                  & (cand.norm(dim=1) <= cfg.goal_max_radius))
            goal[ok] = cand[ok]
            todo &= ~ok
            if not todo.any():
                break
        if todo.any():   # fall back: step toward the workspace center
            center = torch.tensor([0.0, 0.35], device=self.device)
            v = center - start_xy[todo]
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-6)
            goal[todo] = start_xy[todo] + v * cfg.min_goal_dist
        return goal

    def _reset_envs(self, mask: torch.Tensor):
        n = int(mask.sum())
        if n == 0:
            return
        idx = torch.randint(0, self.q0_pool.shape[0], (n,),
                            generator=self.gen).to(self.device)
        q0 = self.q0_pool[idx]
        self.q[mask] = q0
        p, R, J, z_g = self._frames(q0)
        self.z_ref[mask] = p[:, 2]
        # pooled starts are Newton-repaired to an exactly vertical grasp
        # axis, so the held reference is exactly straight-down
        self.zg_ref[mask] = torch.tensor([0.0, 0.0, -1.0],
                                         device=self.device)
        self.start_xy[mask] = p[:, :2]
        goal = self._sample_goals(p[:, :2])
        self.goal_xy[mask] = goal
        self.L0[mask] = (goal - p[:, :2]).norm(dim=1).clamp_min(1e-6)
        self.steps[mask] = 0
        self.a_prev[mask] = 0.0
        self.qdot[mask] = 0.0
        self.done_persistent[mask] = False
        self._coll_margin[mask] = self._collision_margin(q0)
        self.ep_reward[mask] = 0.0
        self.ep_len[mask] = 0.0

    def reset(self):
        self._reset_envs(torch.ones(self.cfg.n_envs, dtype=torch.bool,
                                    device=self.device))
        return self._obs()

    # ------------------------------------------------------------------
    # observation / reward
    # ------------------------------------------------------------------
    def _obs(self):
        p, R, J, z_g = self._frames(self.q)
        q_bar = (self.q - self.q_mid) / self.q_half
        to_goal = self.goal_xy - p[:, :2]
        d = to_goal.norm(dim=1)
        u_goal = to_goal / d.clamp_min(1e-6).unsqueeze(1)
        d_tilde = (d / self.L0).clamp(max=2.0).unsqueeze(1)
        x_g = R[:, :, 0]
        yaw = torch.atan2(x_g[:, 1], x_g[:, 0])
        margins = torch.stack([
            self._jl_margin(self.q),
            (self._coll_margin / 0.10).clamp(0.0, 1.0),
            (self._workspace_margin(p[:, :2]) / 0.10).clamp(0.0, 1.0),
        ], dim=1)
        return torch.cat([
            q_bar, q_bar * q_bar, u_goal, d_tilde,
            torch.sin(yaw).unsqueeze(1), torch.cos(yaw).unsqueeze(1),
            self.a_prev, margins], dim=1)

    def _reward_info(self, obj_xy, qdot):
        cfg = self.cfg
        d = (obj_xy - self.goal_xy).norm(dim=1)
        place = 1.0 - torch.tanh(3.0 * d)
        qd_frac = (qdot / self.qd_limit).norm(dim=1)
        static = 1.0 - torch.tanh(5.0 * qd_frac)
        is_placed = d <= cfg.goal_eps
        is_static = qd_frac <= cfg.static_eps
        success = is_placed & is_static
        reward = place + static * is_placed.to(place.dtype)
        reward = torch.where(success,
                             torch.full_like(reward, cfg.success_reward),
                             reward)
        info = dict(d_goal=d, is_placed=is_placed, is_static=is_static,
                    success=success, raw_reward=reward)
        return reward / cfg.success_reward, info

    def _zg_target(self, z_g, zg_ref):
        """Position-level hold target for the grasp axis: the REFERENCE
        TILT at the CURRENT azimuth. A rigid grasp on an object that
        rotates about world z carries the tilted axis's horizontal
        component around with it; pinning the full reference vector
        (the old behavior) silently forbids object rotation for tilted
        grasps. Vertical grasps have a degenerate azimuth and keep the
        old fixed-vector behavior."""
        zz = zg_ref[:, 2:3]
        horiz_ref = (1.0 - zz * zz).clamp(min=0.0).sqrt()
        h = z_g[:, :2]
        hn = h.norm(dim=1, keepdim=True)
        tgt = torch.cat([h / hn.clamp_min(1e-6) * horiz_ref, zz], dim=1)
        degen = ((hn < 1e-3) | (horiz_ref < 1e-3)).expand_as(tgt)
        return torch.where(degen, zg_ref, tgt)

    def _constraint_rows(self, J, p, R):
        """Rows of the fixed task components (B, 3, 6). Default: pure
        [v_z, w_x, w_y] -- constant height + tilt hold on a horizontal
        plane. Surface-following subclasses (slope) return position-
        dependent COUPLED rows instead."""
        return J[:, [2, 3, 4], :]

    def _hold_targets(self, p, R):
        """Position-level constraint targets (z_tgt, zg_ref_eff) at the
        current state. Default: the constants set at reset."""
        return self.z_ref, self.zg_ref

    def _field_governor(self):
        """Speed scale for steep constraint-field regions (surface
        subclasses override; default none)."""
        return 1.0

    def _tilt_residual(self, R, z_g, zg_ref):
        """(B, 3) rotation-error vector of the orientation hold.
        Default: grasp-axis hold with free azimuth. Surface-following
        subclasses override with a BODY-frame hold (the grasp axis
        pins only 2 of the body's 3 rotational DOF; roll about it is
        invisible to this default)."""
        return torch.cross(z_g, self._zg_target(z_g, zg_ref), dim=1)

    def _residual_components(self, e_z, e_rot, p, R):
        """Project (e_z, e_rot) onto the constraint rows' directions.
        Default rows are [v_z, w_x, w_y]."""
        return torch.stack([e_z, e_rot[:, 0], e_rot[:, 1]], dim=1)

    def _project_q(self, q, z_tgt, zg_tgt, iters, tilt_only=False):
        """Newton-project q onto {z = z_tgt, orientation hold}."""
        cfg = self.cfg
        for _ in range(iters):
            p, R, J, z_g = self._frames(q)
            e_z = z_tgt - p[:, 2]
            if tilt_only:
                e_rot = self._tilt_residual(R, z_g, zg_tgt)
            else:
                e_rot = torch.cross(z_g, zg_tgt, dim=1)
            res = self._residual_components(e_z, e_rot, p, R)
            J_c = self._constraint_rows(J, p, R)
            J_c_pinv, _ = damped_pinv(J_c, cfg.lambda_0, cfg.sigma_thr)
            q = q + (J_c_pinv @ res.unsqueeze(-1)).squeeze(-1)
        return q

    def _project_to_manifold(self, active: torch.Tensor):
        """Newton-correct q back onto {z = z_ref, z_g = zg_ref}.

        Explicit Euler leaves O(|qdot|^2 dt^2) constraint drift per step;
        under fast wrist-heavy motion that reaches mm / degree scale and
        the drift safety net becomes the dominant (artificial) death
        cause. Two Newton iterations on the 3-dim residual keep the
        state on the manifold to numerical precision."""
        p, R, _, _ = self._frames(self.q)
        z_tgt, zg_ref_eff = self._hold_targets(p, R)
        q_proj = self._project_q(self.q, z_tgt, zg_ref_eff,
                                 self.cfg.n_project_iters, tilt_only=True)
        self.q = torch.where(active.unsqueeze(1), q_proj, self.q)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        cfg = self.cfg
        actions = actions.clamp(-1.0, 1.0).to(self.device,
                                              dtype=self.kin.dtype)
        u_dir = actions[:, :self.n_joints]
        rho = 0.5 * (actions[:, self.n_joints] + 1.0)
        active = ~self.done_persistent

        p, R, J, z_g = self._frames(self.q)

        # fixed task components (default [v_z, w_x, w_y]; subclasses may
        # return position-dependent coupled rows)
        J_c = self._constraint_rows(J, p, R)
        J_c_pinv, sigma_min = damped_pinv(J_c, cfg.lambda_0, cfg.sigma_thr)

        # constraint drift feedback (nominally zero)
        z_tgt0, zg_ref0 = self._hold_targets(p, R)
        e_z = z_tgt0 - p[:, 2]
        e_rot0 = self._tilt_residual(R, z_g, zg_ref0)
        x_hold = self._residual_components(cfg.k_z * e_z,
                                           cfg.k_tilt * e_rot0, p, R)
        qdot_hold = (J_c_pinv @ x_hold.unsqueeze(-1)).squeeze(-1)

        # dirfrac v2: exact null projection of the joint-space proposal
        _, _, Vh = torch.linalg.svd(J_c.double(), full_matrices=True)
        Nn = Vh.transpose(-1, -2)[..., 3:]                     # (B, 6, 3)
        xi = (Nn @ (Nn.transpose(-1, -2)
                    @ u_dir.double().unsqueeze(-1))).squeeze(-1)
        xi_nrm = xi.norm(dim=-1, keepdim=True)
        xi_dir = (xi / xi_nrm.clamp_min(1e-6)).to(self.kin.dtype)

        # velocity-limit amplitude: max alpha with
        # |qdot_hold + alpha xi_dir| <= qd_limit, per-joint closed form
        head = (self.qd_limit - qdot_hold) / xi_dir.clamp_min(1e-9)
        room = (self.qd_limit + qdot_hold) / (-xi_dir).clamp_min(1e-9)
        bound = torch.where(xi_dir >= 0, head, room)
        alpha_feas = bound.amin(dim=-1).clamp_min(0.0)
        if cfg.margin_speed_scale > 0.0:
            governor = (self._coll_margin
                        / cfg.margin_speed_scale).clamp(0.12, 1.0)
            alpha_feas = alpha_feas * governor
        alpha_feas = alpha_feas * self._field_governor()

        qdot_null = (rho * alpha_feas).unsqueeze(-1) * xi_dir
        qdot = qdot_hold + qdot_null
        qdot = torch.where(active.unsqueeze(1), qdot,
                           torch.zeros_like(qdot))
        self.qdot = qdot
        self.q = self.q + cfg.dt * qdot
        self._project_to_manifold(active)
        self.steps = self.steps + active.long()

        # executed-motion a_prev (v2 a_prev_executed semantics)
        self.a_prev = torch.cat(
            [qdot_null / self.qd_limit, (2.0 * rho - 1.0).unsqueeze(1)],
            dim=1) * active.unsqueeze(1)

        # post-step state
        p2, R2, _, z_g2 = self._frames(self.q)
        obj_xy = self._object_xy(p2, R2)

        reward, rinfo = self._reward_info(obj_xy, qdot)
        reward = reward * active.to(reward.dtype)

        # terminations (success does NOT terminate)
        z_tgt2, zg_ref2 = self._hold_targets(p2, R2)
        tilt = torch.asin(
            self._tilt_residual(R2, z_g2, zg_ref2)
            .norm(dim=1).clamp(max=1.0))
        died_jl = ((self.q <= self.lmt_lo)
                   | (self.q >= self.lmt_up)).any(dim=1)
        self._coll_margin = self._collision_margin(self.q)
        died_coll = self._coll_margin <= 0.0
        died_ws = self._workspace_margin(obj_xy) <= 0.0
        died_drift = (((p2[:, 2] - z_tgt2).abs() > cfg.eps_z)
                      | (tilt > math.radians(cfg.eps_tilt_deg)))
        violated = died_jl | died_coll | died_ws | died_drift
        terminated = violated & active
        truncated = (self.steps >= cfg.max_steps) & active & ~terminated

        info = dict(**rinfo, sigma_min=sigma_min,
                    alpha_feas=alpha_feas, tilt=tilt,
                    z_drift=(p2[:, 2] - z_tgt2),
                    died_jl=died_jl & active, died_coll=died_coll & active,
                    died_ws=died_ws & active,
                    died_drift=died_drift & active)

        # episode accounting + PPO logging contract
        self.ep_reward = self.ep_reward + reward
        self.ep_len = self.ep_len + active.float()
        info['r_progress_mean'] = float(reward[active].mean().item()
                                        if active.any() else 0.0)
        done = terminated | truncated
        obs_pre = self._obs()
        info['terminal_obs'] = obs_pre       # obs BEFORE any auto-reset
        n_done = int(done.sum())
        info['n_episodes_done'] = n_done
        info['episode_done'] = done
        if n_done > 0:
            net_prog = ((self.L0 - rinfo['d_goal']) / self.L0)[done]
            info['ep_reward_mean'] = float(self.ep_reward[done].mean())
            info['ep_len_mean'] = float(self.ep_len[done].mean())
            info['ep_progress_mean'] = float(net_prog.mean())
            info['ep_progress_max'] = float(net_prog.max())
            info['ep_success_mean'] = float(
                rinfo['success'][done].float().mean())
        if auto_reset:
            if n_done > 0:
                self._reset_envs(done)
                obs = self._obs()
            else:
                obs = obs_pre
        else:
            self.done_persistent = self.done_persistent | done
            obs = obs_pre
        return obs, reward, terminated, truncated, info
