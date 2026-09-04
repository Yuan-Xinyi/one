"""Bottle plane-to-slope drag task.

Scene: table plane z=0; a THETA ramp rises toward +y from the fold
line y = Y_FOLD (parallel to x). The lying bottle (long axis along +x,
compare_exp roll preserved) starts flat at y = Y_START and must be
pushed SLOPE_ADV up the ramp (translation task: goal heading = start
heading).

Constraint generalization (the only mechanical change vs the flat
bottle task): the fixed task components become position-dependent
COUPLED rows. Crossing the fold the rigid bottle BRIDGES -- rear
bottom edge on the table, front bottom edge on the ramp -- so height
and pitch are slaved to the advance:

    v_z(center) - c(y) v_y(center) = 0        c = dz_c/dy_c
    w_x         - k(y) v_y(center) = 0        k = dphi/dy_c
    w_y                            = 0

with c, k, z_c, phi tabulated from the closed 2-D bridging geometry
(axis-parallel crossing; yawed bridging is second-order for the small
yaws this task sees). On the flat both fields vanish and the rows
reduce EXACTLY to the parent's [v_z, w_x, w_y].

All of this enters through the two base-class hooks
(_constraint_rows, _hold_targets); projection, drift feedback and the
drift safety net then work unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .drag_env import DragEnvConfig, damped_pinv
from .bottle_env import BottleCompareEnv


def _rotx_batch(phi):
    c, s = torch.cos(phi), torch.sin(phi)
    R = torch.zeros((phi.shape[0], 3, 3), device=phi.device)
    R[:, 0, 0] = 1.0
    R[:, 1, 1], R[:, 1, 2] = c, -s
    R[:, 2, 1], R[:, 2, 2] = s, c
    return R


class BottleSlopeEnv(BottleCompareEnv):

    OBS_DIM = 39   # parent 35 + [phi/th, c/tan, k/kmax, goal_phi/th]
    THETA = math.radians(15.0)
    Y_FOLD = 0.48
    Y_START = 0.40
    SLOPE_ADV = 0.07

    def __init__(self, cfg: DragEnvConfig, yaw_tol_deg: float = 10.0):
        if cfg.yaw_tol_deg > 0:
            yaw_tol_deg = cfg.yaw_tol_deg
        super().__init__(cfg, yaw_tol_deg=yaw_tol_deg)
        self.THETA = math.radians(cfg.slope_theta_deg)   # instance value
        L, R = self.BOTTLE_L, self.BOTTLE_R
        # re-pose the compare_exp bottle: heading -> 0 (axis +x), same
        # roll, center at (0, Y_START) on the flat
        self.init_R = self.init_R.cpu()      # scene math on cpu; moved
        self.init_p = self.init_p.cpu()      # back at the end
        head0 = math.atan2(float(self.init_R[1, 2]),
                           float(self.init_R[0, 2]))
        c0, s0 = math.cos(-head0), math.sin(-head0)
        Rz = torch.tensor([[c0, -s0, 0.0], [s0, c0, 0.0], [0.0, 0.0, 1.0]])
        self.init_R = (Rz @ self.init_R).contiguous()
        z0 = float(self.init_p[2])
        self.init_p = torch.tensor([-L / 2, self.Y_START, z0])
        self._zc_flat = z0             # flat center height (= origin z,
        #                                axis horizontal)
        self.goal_phi = torch.full((cfg.n_envs,), 0.0)
        self._build_field_table()
        self._move_tensors_to_device()
        # anchor goal (the guaranteed fallback of the start/goal
        # compatibility chain) at a field-derived resting pose
        yc_g = self._anchor_yc()
        dz_g, phi_g, _, _ = self._fields(
            torch.tensor([yc_g], device=self.device))
        Rg1 = self._pose_R(torch.zeros(1, device=self.device),
                           phi_g)[0]
        self.goal_p = (torch.tensor([0.0, yc_g, z0 + float(dz_g[0])],
                                    device=self.device)
                       - (L / 2) * Rg1[:, 2])
        self.goal_heading = 0.0
        self.goal_yaw = torch.full((cfg.n_envs,), 0.0,
                                   device=self.device)
        self.goal_phi[:] = phi_g[0]
        # feasible-grasp set at the anchor
        R_des = Rg1.unsqueeze(0) @ self.g_R
        p_des = (self.goal_p.unsqueeze(0)
                 + (Rg1.unsqueeze(0)
                    @ self.g_p.unsqueeze(-1)).squeeze(-1))
        pool_p, _, _, _ = self._frames(self.q0_pool)
        near = torch.cdist(p_des, pool_p).argmin(dim=1)
        _, self._v1_goal_ok = self._solve_pose(p_des, R_des,
                                               self.q0_pool[near])
        assert bool(self._v1_goal_ok.any())

    def _anchor_yc(self):
        return self.Y_FOLD + self.SLOPE_ADV * math.cos(self.THETA)

    # -- bridging geometry --------------------------------------------
    # With the bottle yawed by psi, the support extremes along the
    # travel direction are the two DIAGONAL bottom corners; their
    # half-span is w_eff = R|cos psi| + (L/2)|sin psi| and the same 2-D
    # bridge geometry applies with that width (the bottom-face center
    # is the diagonal midpoint). Fields therefore live on a (psi, y)
    # grid.
    def _phi_of_yr(self, y_r, w_eff):
        w2 = 2 * w_eff
        if y_r + w2 <= self.Y_FOLD:
            return 0.0
        if y_r >= self.Y_FOLD:
            return self.THETA
        lo, hi = 0.0, self.THETA
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (w2 * math.sin(mid) < math.tan(self.THETA)
                    * (y_r + w2 * math.cos(mid) - self.Y_FOLD)):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _field_np(self, psi=0.0, y_c_query=None):
        """Closed-geometry center fields for one yaw; helper."""
        R = self.BOTTLE_R
        w_eff = (R * abs(math.cos(psi))
                 + (self.BOTTLE_L / 2) * abs(math.sin(psi)))
        span = max(12 * R, 3 * w_eff)
        yrs = np.linspace(self.Y_FOLD - span, self.Y_FOLD + span, 2001)
        ycs, zcs, phis = [], [], []
        for yr in yrs:
            phi = self._phi_of_yr(float(yr), w_eff)
            zr = (math.tan(self.THETA) * (yr - self.Y_FOLD)
                  if yr >= self.Y_FOLD else 0.0)
            yb = yr + w_eff * math.cos(phi)
            zb = zr + w_eff * math.sin(phi)
            ycs.append(yb - R * math.sin(phi))
            zcs.append(zb + R * math.cos(phi))
            phis.append(phi)
        ycs, zcs, phis = map(np.array, (ycs, zcs, phis))
        if y_c_query is not None:
            return (float(np.interp(y_c_query, ycs, zcs)) - R,)
        return ycs, zcs - R, phis      # z field relative to flat height

    PSI_MAX = math.pi

    def _build_field_table(self):
        ny, npsi = 701, 61
        yg = np.linspace(self.Y_FOLD - 0.35, self.Y_FOLD + 0.35, ny)
        pg = np.linspace(-self.PSI_MAX, self.PSI_MAX, npsi)
        dz = np.zeros((npsi, ny))
        ph = np.zeros((npsi, ny))
        for j, psi in enumerate(pg):
            ycs, z, phis = self._field_np(psi=float(psi))
            dz[j] = np.interp(yg, ycs, z)
            ph[j] = np.interp(yg, ycs, phis)
        c = np.gradient(dz, yg, axis=1)
        k = np.gradient(ph, yg, axis=1)
        zp = np.gradient(dz, pg, axis=0)     # dz/dpsi (spin coupling)
        pp = np.gradient(ph, pg, axis=0)     # dphi/dpsi
        self.tab_y = torch.tensor(yg, dtype=torch.float32)
        self.tab_psi = torch.tensor(pg, dtype=torch.float32)
        self.tab_dz = torch.tensor(dz, dtype=torch.float32)
        self.tab_phi = torch.tensor(ph, dtype=torch.float32)
        self.tab_c = torch.tensor(c, dtype=torch.float32)
        self.tab_k = torch.tensor(k, dtype=torch.float32)
        self.tab_zp = torch.tensor(zp, dtype=torch.float32)
        self.tab_pp = torch.tensor(pp, dtype=torch.float32)
        self.k_max = float(np.abs(k).max())

    def _fields(self, y_c, psi=None):
        """(dz, phi, c, k) at (center-y, yaw), bilinear, clamped."""
        if psi is None:
            psi = torch.zeros_like(y_c)
        y = y_c.clamp(float(self.tab_y[0]), float(self.tab_y[-1]))
        s = psi.clamp(float(self.tab_psi[0]), float(self.tab_psi[-1]))
        iy = torch.bucketize(y, self.tab_y).clamp(1, len(self.tab_y) - 1)
        ip = torch.bucketize(s, self.tab_psi).clamp(
            1, len(self.tab_psi) - 1)
        y0, y1 = self.tab_y[iy - 1], self.tab_y[iy]
        p0, p1 = self.tab_psi[ip - 1], self.tab_psi[ip]
        wy = (y - y0) / (y1 - y0).clamp_min(1e-9)
        wp = (s - p0) / (p1 - p0).clamp_min(1e-9)

        def pick(t):
            a = t[ip - 1, iy - 1] * (1 - wy) + t[ip - 1, iy] * wy
            b = t[ip, iy - 1] * (1 - wy) + t[ip, iy] * wy
            return a * (1 - wp) + b * wp
        self._pick = pick                    # for the psi-derivatives
        return (pick(self.tab_dz), pick(self.tab_phi),
                pick(self.tab_c), pick(self.tab_k))

    def _center(self, p, R):
        """Bottle CENTER position + the resting-family yaw parameter.

        The family is R = Rot_x(phi) Rz(psi) R_flat; the HORIZONTAL
        heading of the tilted axis differs from psi by up to ~6 deg at
        45-deg tilt, which at the bridge boundary maps to ~1 cm of
        center-height error. Recover psi exactly: phi from the body up
        axis, then un-tilt the long axis."""
        Rg = self.g_R[self.grasp_idx]
        R_obj = R @ Rg.transpose(-1, -2)
        p_obj = p - (R_obj @ self.g_p[self.grasp_idx].unsqueeze(-1)
                     ).squeeze(-1)
        u_b = R_obj[:, :, 1]
        phi_b = torch.atan2(-u_b[:, 1], u_b[:, 2])
        cpb, spb = torch.cos(phi_b), torch.sin(phi_b)
        axis = R_obj[:, :, 2]
        psi = torch.atan2(cpb * axis[:, 1] + spb * axis[:, 2],
                          axis[:, 0])
        center = p_obj + (self.BOTTLE_L / 2) * axis
        return center, psi

    # -- constraint hooks ---------------------------------------------
    def _constraint_rows(self, J, p, R):
        if not hasattr(self, 'grasp_idx'):      # start-pool build phase
            return super()._constraint_rows(J, p, R)
        center, hd = self._center(p, R)
        _, phi, c, k = self._fields(center[:, 1], hd)
        zp = self._pick(self.tab_zp)
        pp = self._pick(self.tab_pp)
        self._phi_cache = phi
        self._k_cache = k
        r = p - center                          # v_center = v_ee - w x r
        Jv, Jw = J[:, :3, :], J[:, 3:, :]
        v_cy = (Jv[:, 1, :] - r[:, 0:1] * Jw[:, 2, :]
                + r[:, 2:3] * Jw[:, 0, :])
        v_cz = (Jv[:, 2, :] - r[:, 1:2] * Jw[:, 0, :]
                + r[:, 0:1] * Jw[:, 1, :])
        cph = torch.cos(phi).unsqueeze(1)
        sph = torch.sin(phi).unsqueeze(1)
        # spin rate about the local normal (changes w_eff, hence the
        # fields: the psi-coupling terms keep fast in-place rotation on
        # the resting family)
        w_n = -sph * Jw[:, 1, :] + cph * Jw[:, 2, :]
        r1 = v_cz - c.unsqueeze(1) * v_cy - zp.unsqueeze(1) * w_n
        r2 = Jw[:, 0, :] - k.unsqueeze(1) * v_cy - pp.unsqueeze(1) * w_n
        # third row: w . t2 with t2 = (0, cos phi, sin phi) -- the free
        # rotation is about the LOCAL surface normal, not world z
        r3 = cph * Jw[:, 1, :] + sph * Jw[:, 2, :]
        return torch.stack([r1, r2, r3], dim=1)

    def _hold_targets(self, p, R):
        if not hasattr(self, 'grasp_idx'):
            return super()._hold_targets(p, R)
        center, hd = self._center(p, R)
        dz, phi, _, _ = self._fields(center[:, 1], hd)
        self._phi_cache = phi
        e_zc = (self._zc_flat + dz) - center[:, 2]
        # EE-equivalent target so the base e_z formula corrects the
        # OBJECT height error; the tilt reference stays in the FLAT
        # frame -- _zg_target (overridden) evaluates the azimuth-free
        # hold in the LOCAL surface frame using the cached phi
        z_tgt = p[:, 2] + e_zc
        return z_tgt, self.zg_ref

    def _tilt_residual(self, R, z_g, zg_ref):
        """BODY-frame orientation hold: the bottle's own up-axis must
        equal the local surface normal (yaw about the normal stays
        free). Holding only the GRASP axis (parent default) leaves the
        roll about that axis unconstrained -- yawing while tilted then
        winds the body off the resting family with every instrument
        reading green (the 44 mm penetration bug)."""
        if not hasattr(self, 'grasp_idx') \
                or not hasattr(self, '_phi_cache') \
                or self._phi_cache.shape[0] != R.shape[0]:
            return super()._tilt_residual(R, z_g, zg_ref)
        R_obj = R @ self.g_R[self.grasp_idx].transpose(-1, -2)
        u_b = R_obj[:, :, 1]                     # body up (world z when
        phi = self._phi_cache                    # flat, see init_R)
        n = torch.stack([torch.zeros_like(phi),
                         -torch.sin(phi), torch.cos(phi)], dim=1)
        return torch.cross(u_b, n, dim=1)

    K_KNEE = 14.0          # rad/m; single-fold bridging (~10.5) is
    #                        unaffected, steep convex edges crawl

    def _field_governor(self):
        if not hasattr(self, '_k_cache'):
            return 1.0
        return ((self.K_KNEE / self._k_cache.abs().clamp_min(1e-3)) ** 2
                ).clamp(0.02, 1.0)

    def _project_to_manifold(self, active):
        """Object-anchored Newton projection: the base version drives
        the EE z to a PRE-step target, but under a spin the EE-center
        lever arm rotates and the stale anchor injects (w x r)_z dt of
        height error per step (~5 mm at full spin rate). Re-measuring
        the residual ON THE OBJECT each iteration removes the anchor
        entirely."""
        if not hasattr(self, 'grasp_idx'):
            return super()._project_to_manifold(active)
        q = self.q
        for _ in range(self.cfg.n_project_iters):
            p, R, J, z_g = self._frames(q)
            center, psi = self._center(p, R)
            dz, phi, _, _ = self._fields(center[:, 1], psi)
            self._phi_cache = phi
            e_zc = (self._zc_flat + dz) - center[:, 2]
            e_rot = self._tilt_residual(R, z_g, self.zg_ref)
            res = self._residual_components(e_zc, e_rot, p, R)
            J_c = self._constraint_rows(J, p, R)
            J_c_pinv, _ = damped_pinv(J_c, self.cfg.lambda_0,
                                      self.cfg.sigma_thr)
            # per-iteration clamp: the projection corrects mm-scale
            # drift; a correction demanding a large arm move means the
            # command outran the field -- smear it over steps instead
            # of teleporting the arm
            dq = (J_c_pinv @ res.unsqueeze(-1)).squeeze(-1)
            q = q + dq.clamp(-0.03, 0.03)
        self.q = torch.where(active.unsqueeze(1), q, self.q)

    def _residual_components(self, e_z, e_rot, p, R):
        """Rows 2/3 live in the local surface plane: fold axis x and
        t2 = n x x = (0, cos phi, sin phi); the free rotation is about
        the local NORMAL, not world z."""
        if not hasattr(self, 'grasp_idx') \
                or not hasattr(self, '_phi_cache') \
                or self._phi_cache.shape[0] != e_rot.shape[0]:
            return super()._residual_components(e_z, e_rot, p, R)
        cph = torch.cos(self._phi_cache)
        sph = torch.sin(self._phi_cache)
        return torch.stack(
            [e_z, e_rot[:, 0],
             cph * e_rot[:, 1] + sph * e_rot[:, 2]], dim=1)

    # -- reset ----------------------------------------------------------
    def _reset_envs(self, mask):
        # candidate batches at reset carry their own grasp per row; the
        # init pose is flat, so the parent's flat-box margin is exact
        # there (see _bottle_margin)
        self._flat_margin_mode = True
        try:
            if self.cfg.rand_goals:
                self._reset_rand_goals(mask)
            else:
                super()._reset_envs(mask)
                self.goal_phi[mask] = self.THETA
        finally:
            self._flat_margin_mode = False

    def _goal_y_range(self):
        return 0.33, self.Y_FOLD + 0.10 * math.cos(self.THETA)

    def _goal_phi_ok(self, phi):
        """Accept only RESTING poses (flat or flush on the incline)."""
        return (phi < 0.02 * self.THETA) | (phi > 0.98 * self.THETA)

    def _sample_goals(self, n):
        """SE(2)-on-surface goals: y spans flat -> slope continuously;
        the bridge band is rejected so every goal is a RESTING pose
        (flat or flush on the ramp). Returns (center(n,3), psi(n),
        phi(n))."""
        dev = self.device
        y_lo, y_hi = self._goal_y_range()
        yc = torch.zeros(n, device=dev)
        ps = torch.zeros(n, device=dev)
        need = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(24):
            m = int(need.sum())
            if m == 0:
                break
            y_try = (y_lo + (y_hi - y_lo)
                     * torch.rand(m, generator=self.gen)).to(dev)
            p_try = ((torch.rand(m, generator=self.gen).to(dev) - 0.5)
                     * 2 * math.radians(self.cfg.goal_yaw_range_deg))
            _, phi_t, _, _ = self._fields(y_try, p_try)
            good = self._goal_phi_ok(phi_t)
            rows = need.nonzero().squeeze(1)[:m]
            acc = rows[good]
            yc[acc] = y_try[good]
            ps[acc] = p_try[good]
            need[acc] = False
        # leftovers (rare): flat fallback well clear of the fold
        if need.any():
            m = int(need.sum())
            yc[need] = 0.33 + 0.04 * torch.rand(m,
                                                generator=self.gen).to(dev)
            ps[need] = 0.0
        xg = ((torch.rand(n, generator=self.gen).to(dev) - 0.5) * 0.24)
        dz, phi, _, _ = self._fields(yc, ps)
        center = torch.stack([xg, yc, self._zc_flat + dz], dim=1)
        return center, ps, phi

    def _pose_R(self, psi, phi):
        """Surface resting orientation: yaw then field tilt."""
        n = psi.shape[0]
        dev = psi.device
        cps, sps = torch.cos(psi), torch.sin(psi)
        cph, sph = torch.cos(phi), torch.sin(phi)
        Rz = torch.zeros((n, 3, 3), device=dev)
        Rz[:, 0, 0], Rz[:, 0, 1] = cps, -sps
        Rz[:, 1, 0], Rz[:, 1, 1] = sps, cps
        Rz[:, 2, 2] = 1.0
        Rx = torch.zeros((n, 3, 3), device=dev)
        Rx[:, 0, 0] = 1.0
        Rx[:, 1, 1], Rx[:, 1, 2] = cph, -sph
        Rx[:, 2, 1], Rx[:, 2, 2] = sph, cph
        return Rx @ Rz @ self.init_R.unsqueeze(0)

    def _surface_clearance(self, pts, rr):
        """Per-sphere clearance vs the raised surface (beyond-fold ramp
        plane by default; terrain subclasses override)."""
        st, ct = math.sin(self.THETA), math.cos(self.THETA)
        d_ramp = -st * (pts[..., 1] - self.Y_FOLD) + ct * pts[..., 2]
        return torch.where(pts[..., 1] > self.Y_FOLD, d_ramp - rr,
                           torch.full_like(d_ramp, 1.0))

    def _bottle_margin_at(self, q, R_obj, p_obj):
        """Arm-vs-bottle box + surface margin at EXPLICIT object poses
        (candidate batches at reset carry their own pose per row)."""
        tfs = self.kin.link_transforms(q)
        aug = torch.cat([tfs, tfs[:, 6:7]], dim=1)
        pos = self.coll.sphere_positions(aug)
        keep = (self.coll.link_indices >= 2) & (self.coll.link_indices <= 6)
        pts, rr = pos[:, keep], self.coll.radii[keep]
        d_ramp = self._surface_clearance(pts, rr)
        loc = torch.einsum('bji,bsj->bsi', R_obj,
                           pts - p_obj.unsqueeze(1))
        BR, BL = self.BOTTLE_R, self.BOTTLE_L
        d = torch.stack([loc[..., 0].abs() - BR,
                         loc[..., 1].abs() - BR,
                         torch.maximum(-loc[..., 2], loc[..., 2] - BL)],
                        dim=-1)
        outside = d.clamp(min=0).norm(dim=-1)
        inside = d.max(dim=-1).values.clamp(max=0)
        m = (outside + inside - rr).amin(dim=1)
        return torch.minimum(m, d_ramp.amin(dim=1))

    def _reset_rand_goals(self, mask: torch.Tensor):
        n = int(mask.sum())
        if n == 0:
            return
        G = self.n_grasps
        dev = self.device
        if self.cfg.rand_starts:
            # start pose from the same surface distribution as goals
            ctr0, psi0, phi0 = self._sample_goals(n)
            R0 = self._pose_R(psi0, phi0)
            p0 = ctr0 - (self.BOTTLE_L / 2) * R0[:, :, 2]
        else:
            # jittered flat init pose (parent convention)
            jx = ((torch.rand(n, generator=self.gen).to(dev) - 0.5)
                  * 0.010)
            jy = ((torch.rand(n, generator=self.gen).to(dev) - 0.5)
                  * 0.010)
            jt = ((torch.rand(n, generator=self.gen).to(dev) - 0.5)
                  * math.radians(4))
            phi0 = torch.zeros(n, device=dev)
            c, s = torch.cos(jt), torch.sin(jt)
            Rz = torch.zeros((n, 3, 3), device=dev)
            Rz[:, 0, 0], Rz[:, 0, 1] = c, -s
            Rz[:, 1, 0], Rz[:, 1, 1] = s, c
            Rz[:, 2, 2] = 1.0
            R0 = Rz @ self.init_R.unsqueeze(0)
            p0 = self.init_p.unsqueeze(0).repeat(n, 1)
            p0[:, 0] += jx
            p0[:, 1] += jy
        rows = torch.arange(n, device=dev).repeat_interleave(G)
        cand = torch.arange(G, device=dev).repeat(n)
        pool_p, _, _, _ = self._frames(self.q0_pool)

        def solve_at(R_obj, p_obj):
            R_des = R_obj[rows] @ self.g_R[cand]
            p_des = p_obj[rows] + (R_obj[rows]
                                   @ self.g_p[cand].unsqueeze(-1)
                                   ).squeeze(-1)
            near = torch.cdist(p_des, pool_p).argmin(dim=1)
            return self._solve_pose(p_des, R_des, self.q0_pool[near])

        # start feasibility with retry: every env must end with a
        # start-feasible set that intersects the v1-goal set (the
        # fallback anchor); rows failing get resampled once, then the
        # flat default start
        from .drag_env import Nova2DragEnv
        for att in range(3):
            q_all, ok_start = solve_at(R0, p0)
            if ok_start.any():
                oi = ok_start.nonzero().squeeze(1)
                m = Nova2DragEnv._collision_margin(self, q_all[oi])
                m = torch.minimum(m, self._bottle_margin_at(
                    q_all[oi], R0[rows][oi], p0[rows][oi]))
                ok_start[oi] = m > 0.0
            bad = ~(ok_start.view(n, G)
                    & self._v1_goal_ok.unsqueeze(0)).any(dim=1)
            if not bad.any():
                break
            nb = int(bad.sum())
            if att == 0 and self.cfg.rand_starts:
                c2, s2, f2 = self._sample_goals(nb)
                R2 = self._pose_R(s2, f2)
                R0[bad] = R2
                p0[bad] = c2 - (self.BOTTLE_L / 2) * R2[:, :, 2]
                phi0[bad] = f2
            else:
                R0[bad] = self.init_R.unsqueeze(0)
                p0[bad] = self.init_p.unsqueeze(0)
                phi0[bad] = 0.0
        # goal sampling with start-goal grasp compatibility (2 rounds,
        # then the fixed v1 goal as a guaranteed-compatible fallback)
        gctr = torch.zeros((n, 3), device=dev)
        gpsi = torch.zeros(n, device=dev)
        gphi = torch.zeros(n, device=dev)
        compat = torch.zeros(n * G, dtype=torch.bool, device=dev)
        unresolved = torch.ones(n, dtype=torch.bool, device=dev)
        for rnd in range(3):
            if not unresolved.any():
                break
            if rnd < 2:
                ctr_t, psi_t, phi_t = self._sample_goals(n)
            else:                      # fallback: the anchor goal
                yc = torch.full((n,), self._anchor_yc(), device=dev)
                dzf, phif, _, _ = self._fields(yc)
                ctr_t = torch.stack(
                    [torch.zeros(n, device=dev), yc,
                     self._zc_flat + dzf], dim=1)
                psi_t = torch.zeros(n, device=dev)
                phi_t = phif
            R_g = self._pose_R(psi_t, phi_t)
            p_g = ctr_t - (self.BOTTLE_L / 2) * R_g[:, :, 2]
            _, ok_goal = solve_at(R_g, p_g)
            comp_t = ok_start & ok_goal
            take = unresolved & comp_t.view(n, G).any(dim=1)
            tr = take[rows]
            compat = torch.where(tr, comp_t, compat)
            gctr[take] = ctr_t[take]
            gpsi[take] = psi_t[take]
            gphi[take] = phi_t[take]
            unresolved &= ~take
        assert not bool(unresolved.any()), 'no compatible grasp-goal pair'
        # goal state (must precede the critic pass: goal enters obs)
        rows_full = mask.nonzero().squeeze(1)
        R_g = self._pose_R(gpsi, gphi)
        origin_g = gctr - (self.BOTTLE_L / 2) * R_g[:, :, 2]
        self.goal_xy[mask] = origin_g[:, :2]
        self.goal_yaw[mask] = torch.atan2(R_g[:, 1, 2], R_g[:, 0, 2])
        self.goal_phi[mask] = gphi
        # grasp selection among compatible candidates
        use_critic = (self.cfg.start_mode == 'wp0'
                      and self.value_fn is not None)
        if use_critic:
            vals = torch.full((n * G,), -1e9, device=dev)
            save_q = self.q.clone()
            save_gi = self.grasp_idx.clone()
            save_ap = self.a_prev.clone()
            self.a_prev[rows_full] = 0.0
            for g_c in range(G):
                sel = (compat & (cand == g_c)).nonzero().squeeze(1)
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
            score = torch.rand(n * G, generator=self.gen).to(dev)
            score[~compat] = -1e9
        gi0 = score.view(n, G).argmax(dim=1)
        ar = torch.arange(n, device=dev)
        assert bool(compat.view(n, G)[ar, gi0].all()), \
            'no compatible grasp at reset'
        q0 = q_all.view(n, G, 6)[ar, gi0]
        self.q[mask] = q0
        self.grasp_idx[mask] = gi0
        p, R, J, z_g = self._frames(q0)
        self.z_ref[mask] = p[:, 2]
        # tilt reference is stored in the FLAT frame: rotate the grasp
        # axis back by the START tilt (identity for flat starts)
        Rl0 = _rotx_batch(phi0)
        self.zg_ref[mask] = (Rl0.transpose(-1, -2)
                             @ z_g.unsqueeze(-1)).squeeze(-1)
        self.start_xy[mask] = self._obj_pose(p, R, gi0)[0][:, :2]
        self.L0[mask] = 1.0
        self.steps[mask] = 0
        self.a_prev[mask] = 0.0
        self.qdot[mask] = 0.0
        self.done_persistent[mask] = False
        self.ep_reward[mask] = 0.0
        self.ep_len[mask] = 0.0
        # full-batch margin with the EXACT tilted-box branch (starts
        # may be on the slope under rand_starts)
        self._flat_margin_mode = False
        self._coll_margin = self._collision_margin(self.q)

    def _bottle_margin(self, q, obj_xy, heading):
        tfs = self.kin.link_transforms(q)
        aug = torch.cat([tfs, tfs[:, 6:7]], dim=1)
        pos = self.coll.sphere_positions(aug)
        keep = (self.coll.link_indices >= 2) & (self.coll.link_indices <= 6)
        pts, rr = pos[:, keep], self.coll.radii[keep]
        m_ramp = self._surface_clearance(pts, rr).amin(dim=1)
        if getattr(self, '_flat_margin_mode', False):
            return torch.minimum(
                super()._bottle_margin(q, obj_xy, heading), m_ramp)
        p, R, _, _ = self._frames(q)
        Rg = self.g_R[self.grasp_idx]
        R_obj = R @ Rg.transpose(-1, -2)
        p_obj = p - (R_obj @ self.g_p[self.grasp_idx].unsqueeze(-1)
                     ).squeeze(-1)
        loc = torch.einsum('bji,bsj->bsi', R_obj,
                           pts - p_obj.unsqueeze(1))
        BR, BL = self.BOTTLE_R, self.BOTTLE_L
        d = torch.stack([loc[..., 0].abs() - BR,
                         loc[..., 1].abs() - BR,
                         torch.maximum(-loc[..., 2], loc[..., 2] - BL)],
                        dim=-1)
        outside = d.clamp(min=0).norm(dim=-1)
        inside = d.max(dim=-1).values.clamp(max=0)
        m = (outside + inside - rr).amin(dim=1)
        return torch.minimum(m, m_ramp)

    # -- obs ------------------------------------------------------------
    def _obs(self):
        base = super()._obs()
        p, R, _, _ = self._frames(self.q)
        center, hd = self._center(p, R)
        _, phi, c, k = self._fields(center[:, 1], hd)
        extra = torch.stack([phi / self.THETA,
                             c / math.tan(self.THETA),
                             k / max(self.k_max, 1e-6),
                             self.goal_phi / self.THETA], dim=1)
        return torch.cat([base, extra], dim=1)
