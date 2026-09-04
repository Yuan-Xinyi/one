"""compare_exp bottle task: slide a lying bottle to a full SE(2) goal
pose using ONE of 20 given grasp candidates (no regrasp; the release
channel does not exist here).

Differences from the container line:
  - no slot/walls; obstacles = table, base, self, and the BOTTLE body
    (arm links 2..6 vs a lying-box approximation; the gripper touches
    the bottle by design and is excluded).
  - grasp axes are TILTED (45-deg approaches): the fixed task
    components still lock height and tilt, but the held reference axis
    zg_ref is each grasp's own initial axis, not the vertical.
  - initial grasp: at reset all 20 candidates are Newton-solved at the
    (slightly jittered) init pose and filtered for reach, limits and
    collision; training samples uniformly among the feasible ones,
    evaluation (start_mode='wp0') picks the CRITIC-argmax candidate.
  - object pose from EE pose via the full grasp transform:
        R_obj = R_ee R_g^T,  p_obj = p_ee - R_obj p_g
    heading theta = atan2(R_obj[1,2], R_obj[0,2]) (bottle long axis).
  - reward: position + orientation staged terms, settle term, success
    override 5 (success = 2.5 cm AND 10 deg AND static).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from .drag_env import DragEnvConfig, Nova2DragEnv
from .container_scenario import Study

EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'compare_exp')


def _read_task(path):
    tok = open(path).read().split()

    def grab(key, n):
        i = tok.index(key)
        return np.array([float(x) for x in tok[i + 1:i + 1 + n]])
    return dict(init_pos=grab('init_pos', 3),
                init_R=grab('init_rotmat', 9).reshape(3, 3),
                goal_pos=grab('goal_pos', 3),
                goal_R=grab('goal_rotmat', 9).reshape(3, 3))


class BottleCompareEnv(Nova2DragEnv):

    OBS_DIM = 35
    ACT_DIM = 7
    BOTTLE_R = 0.0375          # half width/height of the lying body
    BOTTLE_L = 0.168

    def __init__(self, cfg: DragEnvConfig, yaw_tol_deg: float = 10.0):
        task = _read_task(os.path.join(EXP_DIR, 'repair0_init_goal.txt'))
        g = np.load(os.path.join(EXP_DIR, 'grasps_G20.npz'))
        self.task = task
        self.yaw_tol = math.radians(yaw_tol_deg)
        self.g_R = torch.tensor(g['ac_rotmat'], dtype=torch.float32)
        self.g_p = torch.tensor(g['ac_pos'], dtype=torch.float32)
        self.n_grasps = self.g_R.shape[0]
        self.init_p = torch.tensor(task['init_pos'], dtype=torch.float32)
        self.init_R = torch.tensor(task['init_R'], dtype=torch.float32)
        self.goal_p = torch.tensor(task['goal_pos'], dtype=torch.float32)
        goal_R = task['goal_R']
        self.goal_heading = math.atan2(goal_R[1, 2], goal_R[0, 2])
        self.value_fn = None

        super().__init__(cfg)
        B = cfg.n_envs
        self.grasp_idx = torch.zeros(B, dtype=torch.long,
                                     device=self.device)
        self.goal_yaw = torch.full((B,), self.goal_heading,
                                   device=self.device)
        self._move_tensors_to_device()

    def set_value_fn(self, fn):
        self.value_fn = fn

    # -- object pose from EE --------------------------------------------
    def _obj_pose(self, p, R, gi):
        Rg = self.g_R[gi]                              # (B,3,3)
        R_obj = R @ Rg.transpose(-1, -2)
        p_obj = p - (R_obj @ self.g_p[gi].unsqueeze(-1)).squeeze(-1)
        heading = torch.atan2(R_obj[:, 1, 2], R_obj[:, 0, 2])
        return p_obj, heading

    def _object_xy(self, p, R):
        p_obj, heading = self._obj_pose(p, R, self.grasp_idx)
        self._obj_heading = heading
        return p_obj[:, :2]

    # -- collision: base checks + bottle body vs arm --------------------
    def _bottle_margin(self, q, obj_xy, heading):
        tfs = self.kin.link_transforms(q)
        aug = torch.cat([tfs, tfs[:, 6:7]], dim=1)
        pos = self.coll.sphere_positions(aug)
        keep = (self.coll.link_indices >= 2) & (self.coll.link_indices <= 6)
        pts, r = pos[:, keep], self.coll.radii[keep]
        # lying box: center half a length along the heading, z in
        # [0, 2R]; half extents (L/2, R) in the heading frame
        c, s = torch.cos(heading), torch.sin(heading)
        bx = obj_xy[:, 0] + c * self.BOTTLE_L / 2
        by = obj_xy[:, 1] + s * self.BOTTLE_L / 2
        loc_x = (c.unsqueeze(1) * (pts[..., 0] - bx.unsqueeze(1))
                 + s.unsqueeze(1) * (pts[..., 1] - by.unsqueeze(1)))
        loc_y = (-s.unsqueeze(1) * (pts[..., 0] - bx.unsqueeze(1))
                 + c.unsqueeze(1) * (pts[..., 1] - by.unsqueeze(1)))
        dx = loc_x.abs() - self.BOTTLE_L / 2
        dy = loc_y.abs() - self.BOTTLE_R
        dz = torch.maximum(0.0 - pts[..., 2],
                           pts[..., 2] - 2 * self.BOTTLE_R)
        d = torch.stack([dx, dy, dz], -1)
        outside = d.clamp(min=0).norm(dim=-1)
        inside = d.max(dim=-1).values.clamp(max=0)
        return (outside + inside - r).amin(dim=1)

    def _collision_margin(self, q):
        m = Nova2DragEnv._collision_margin(self, q)
        p, R, J, z_g = self._frames(q)
        p_obj, heading = self._obj_pose(p, R, self.grasp_idx)
        return torch.minimum(
            m, self._bottle_margin(q, p_obj[:, :2], heading))

    def _build_start_pool(self):
        self._collision_margin = \
            lambda q: Nova2DragEnv._collision_margin(self, q)
        try:
            Nova2DragEnv._build_start_pool(self)
        finally:
            del self._collision_margin

    # -- full-pose Newton solve -----------------------------------------
    def _solve_pose(self, p_des, R_des, seeds, iters=16):
        q = seeds.clone()
        for _ in range(iters):
            p, R, J, z_g = self._frames(q)
            e_p = p_des - p
            e_r = 0.5 * (torch.cross(R[:, :, 0], R_des[:, :, 0], dim=1)
                         + torch.cross(R[:, :, 1], R_des[:, :, 1], dim=1)
                         + torch.cross(R[:, :, 2], R_des[:, :, 2], dim=1))
            e = torch.cat([e_p, e_r], dim=1)
            JJt = (J @ J.transpose(-1, -2)
                   + 1e-5 * torch.eye(6, device=self.device))
            q = q + (J.transpose(-1, -2)
                     @ torch.linalg.solve(JJt, e.unsqueeze(-1))
                     ).squeeze(-1).clamp(-0.25, 0.25)
        p, R, J, z_g = self._frames(q)
        ok = (p_des - p).norm(dim=1) < 2e-3
        rerr = 0.5 * (torch.cross(R[:, :, 0], R_des[:, :, 0], dim=1)
                      + torch.cross(R[:, :, 1], R_des[:, :, 1], dim=1)
                      + torch.cross(R[:, :, 2], R_des[:, :, 2], dim=1))
        ok &= rerr.norm(dim=1) < 0.02
        ok &= ((q > self.lmt_lo) & (q < self.lmt_up)).all(dim=1)
        return q, ok

    # -- reset -----------------------------------------------------------
    def _reset_envs(self, mask: torch.Tensor):
        n = int(mask.sum())
        if n == 0:
            return
        # jittered init pose (goal stays exact -- it is THE task)
        jx = ((torch.rand(n, generator=self.gen).to(self.device) - 0.5)
              * 0.010)
        jy = ((torch.rand(n, generator=self.gen).to(self.device) - 0.5)
              * 0.010)
        jt = ((torch.rand(n, generator=self.gen).to(self.device) - 0.5)
              * math.radians(4))
        c, s = torch.cos(jt), torch.sin(jt)
        Rz = torch.zeros((n, 3, 3), device=self.device)
        Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
        Rz[:, 1, 0], Rz[:, 1, 1] = s, c
        Rz[:, 2, 2] = 1.0
        R_obj0 = Rz @ self.init_R.unsqueeze(0)
        p_obj0 = self.init_p.unsqueeze(0).repeat(n, 1)
        p_obj0[:, 0] += jx
        p_obj0[:, 1] += jy
        # all 20 candidates per env
        G = self.n_grasps
        rows = torch.arange(n, device=self.device).repeat_interleave(G)
        cand = torch.arange(G, device=self.device).repeat(n)
        R_des = R_obj0[rows] @ self.g_R[cand]
        p_des = p_obj0[rows] + (R_obj0[rows]
                                @ self.g_p[cand].unsqueeze(-1)).squeeze(-1)
        pool_p, _, _, _ = self._frames(self.q0_pool)
        near = torch.cdist(p_des, pool_p).argmin(dim=1)
        q_all, ok = self._solve_pose(p_des, R_des, self.q0_pool[near])
        if ok.any():
            oi = ok.nonzero().squeeze(1)
            p_o, head_o = self._obj_pose(*self._frames(q_all[oi])[:2],
                                         cand[oi])
            m = Nova2DragEnv._collision_margin(self, q_all[oi])
            m = torch.minimum(m, self._bottle_margin(
                q_all[oi], p_o[:, :2], head_o))
            ok[oi] = m > 0.0
        # pick per env: critic argmax at eval, random among feasible else
        q0 = torch.zeros((n, 6), device=self.device)
        gi0 = torch.zeros(n, dtype=torch.long, device=self.device)
        use_critic = (self.cfg.start_mode == 'wp0'
                      and self.value_fn is not None)
        if use_critic:
            vals = torch.full((n * G,), -1e9, device=self.device)
            rows_full = mask.nonzero().squeeze(1)
            save_q = self.q.clone()
            save_gi = self.grasp_idx.clone()
            save_ap = self.a_prev.clone()
            self.a_prev[rows_full] = 0.0     # true post-reset value
            # candidate-major chunks: at most ONE candidate per env row
            # per _obs pass (mixed chunks overwrite each other's states
            # and the values come out shuffled)
            for g_c in range(G):
                sel = (ok & (cand == g_c)).nonzero().squeeze(1)
                if not len(sel):
                    continue
                r_sel = rows_full[rows[sel]]
                self.q[r_sel] = q_all[sel]
                self.grasp_idx[r_sel] = cand[sel]
                obs = self._obs()
                vals[sel] = self.value_fn(obs)[r_sel]
                self.q = save_q.clone()
                self.grasp_idx = save_gi.clone()
            self.a_prev = save_ap
            score = vals
        else:
            score = torch.rand(n * G, generator=self.gen).to(self.device)
            score[~ok] = -1e9
        for r in range(n):
            sel = (rows == r).nonzero().squeeze(1)
            best = sel[score[sel].argmax()]
            assert bool(ok[best]), 'no feasible grasp at init pose'
            q0[r] = q_all[best]
            gi0[r] = cand[best]
        self.q[mask] = q0
        self.grasp_idx[mask] = gi0
        p, R, J, z_g = self._frames(q0)
        self.z_ref[mask] = p[:, 2]
        self.zg_ref[mask] = z_g            # each grasp's OWN tilted axis
        self.start_xy[mask] = self._obj_pose(p, R, gi0)[0][:, :2]
        self.goal_xy[mask] = self.goal_p[:2].unsqueeze(0).expand(n, 2)
        self.L0[mask] = 1.0
        self.steps[mask] = 0
        self.a_prev[mask] = 0.0
        self.qdot[mask] = 0.0
        self.done_persistent[mask] = False
        self.ep_reward[mask] = 0.0
        self.ep_len[mask] = 0.0
        self._coll_margin = self._collision_margin(self.q)

    # -- reward ----------------------------------------------------------
    def _reward_info(self, obj_xy, qdot):
        cfg = self.cfg
        d_pos = (obj_xy - self.goal_xy).norm(dim=1)
        yaw_err = torch.remainder(
            self._obj_heading - self.goal_yaw + math.pi,
            2 * math.pi) - math.pi
        s_pos = 1.0 - torch.tanh(3.0 * d_pos)
        # linear, not tanh: at the task's 161-deg initial error a
        # tanh(1.5 x) term is saturated flat (grad ~1e-3) and the
        # policy never discovers rotation; linear keeps a constant
        # gradient over the whole range
        s_yaw = 1.0 - yaw_err.abs() / math.pi
        qd_frac = (qdot / self.qd_limit).norm(dim=1)
        is_placed = (d_pos <= cfg.goal_eps) \
            & (yaw_err.abs() <= self.yaw_tol)
        s4 = is_placed.float() * (1.0 - torch.tanh(5.0 * qd_frac))
        is_static = qd_frac <= cfg.static_eps
        success = is_placed & is_static
        reward = s_pos + s_yaw + s4
        reward = torch.where(success, torch.full_like(reward, 5.0), reward)
        info = dict(d_goal=d_pos, yaw_err=yaw_err, is_placed=is_placed,
                    is_static=is_static, success=success,
                    raw_reward=reward)
        return reward / 5.0, info

    # -- obs -------------------------------------------------------------
    def _obs(self):
        p, R, J, z_g = self._frames(self.q)
        obj_p, heading = self._obj_pose(p, R, self.grasp_idx)
        q_bar = (self.q - self.q_mid) / self.q_half
        to_goal = self.goal_xy - obj_p[:, :2]
        d = to_goal.norm(dim=1)
        u_goal = to_goal / d.clamp_min(1e-6).unsqueeze(1)
        yaw_err = torch.remainder(heading - self.goal_yaw + math.pi,
                                  2 * math.pi) - math.pi
        gp = self.g_p[self.grasp_idx]
        gz = self.g_R[self.grasp_idx, :, 2]
        margins = torch.stack([
            self._jl_margin(self.q),
            (self._coll_margin / 0.10).clamp(0.0, 1.0),
            (self._workspace_margin(obj_p[:, :2]) / 0.10).clamp(0.0, 1.0),
        ], dim=1)
        return torch.cat([
            q_bar, q_bar * q_bar, u_goal, d.clamp(max=2.0).unsqueeze(1),
            torch.sin(yaw_err).unsqueeze(1), torch.cos(yaw_err).unsqueeze(1),
            torch.sin(heading).unsqueeze(1),
            torch.cos(heading).unsqueeze(1),
            gp / 0.1, gz,
            self.a_prev, margins], dim=1)[:, :self.OBS_DIM]
