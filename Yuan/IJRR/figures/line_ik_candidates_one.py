#!/usr/bin/env python3
"""IK candidates at two points of one straight-line path, in one's viewer.

The task here is the plain one: the TCP follows a straight line at a fixed tool
orientation, so the 7-DoF FR3 keeps a 1-D self-motion manifold. One line is
drawn. Two points on it are picked — where the path starts and a point far along
it — and at each of them *every* collision-free IK solution of that pose is
drawn faint, with the one the successful trajectory actually passes through in
blue.

That is the whole argument in one picture: both points have a fistful of IK
solutions, so point-wise they are equally "reachable", but only one solution at
the first point continues into a solution at the second.

    conda activate one
    cd /home/lqin/one
    python Yuan/IJRR/figures/line_ik_candidates_one.py
    python Yuan/IJRR/figures/line_ik_candidates_one.py --law wln --way-frac 0.9
    python Yuan/IJRR/figures/line_ik_candidates_one.py --list-starts

Orbit with the mouse and take the screenshot yourself.
"""
from __future__ import annotations

import Yuan.IJRR.figures.fig_door_opening as F      # matplotlib before torch
from Yuan.IJRR.figures.door_scene_one import (_rotz, _tint,             # noqa: E402
                                              CAND_RGB, GOOD_RGB)

import argparse                                     # noqa: E402
import builtins                                     # noqa: E402
import math                                         # noqa: E402

import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402

import one.scene.scene_object_primitive as ossop    # noqa: E402
import one.viewer.world as ovw                      # noqa: E402
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (  # noqa: E402
    attach_pen_visual, make_fr3_with_pen,
)

LINE_RGB = (0.35, 0.36, 0.40)
MARK_RGB = (0.20, 0.21, 0.24)
PEN_RGB = (0.13, 0.13, 0.15)


def spawn_pen_arm(scene, base: F.BasePose, q: np.ndarray, rgb, alpha: float,
                  jaw_width: float):
    """One FR3 holding the pen, at a fixed pose and opacity."""
    arm, hand = make_fr3_with_pen(rotmat=_rotz(math.radians(base.yaw_deg)),
                                  pos=np.asarray(base.t, np.float32),
                                  jaw_width=jaw_width)
    arm.attach_to(scene)
    arm.fk(np.asarray(q, np.float32))
    pen = attach_pen_visual(arm, rgb=PEN_RGB, alpha=min(1.0, alpha + 0.25))
    _tint(arm, hand, rgb, alpha)
    for part in pen:                       # the pen keeps its own darker shade
        part.alpha = min(1.0, alpha + 0.25)
    return arm, hand, pen


def tool_frame(d: np.ndarray) -> np.ndarray:
    """Fixed tool orientation along the path: tool z down, tool x along the line."""
    z_ee = np.array([0.0, 0.0, -1.0])
    x_ee = d / np.linalg.norm(d)
    y_ee = np.cross(z_ee, x_ee)
    return np.stack([x_ee, y_ee, z_ee], axis=-1)


def rollout_line(arm: F.Arm, base: F.BasePose, p0: np.ndarray, d: np.ndarray,
                 R_t: np.ndarray, q_start: np.ndarray, law: F.Law,
                 length: float = 0.9, ds: float = 0.002, lam: float = 0.02,
                 iters: int = 3, pos_tol: float = 2e-3, rot_tol: float = 2e-2,
                 limit_margin: float = 0.01):
    """March the TCP along the line until the arm cannot follow any further.

    Same resolved-rate machinery and the same laws as the door rollout, with the
    door-specific collision checks dropped: only joint limits, tracking failure
    and self-collision stop it here.
    """
    s_grid = np.arange(0.0, length + 1e-9, ds)
    p_w = p0[None] + s_grid[:, None] * d[None]
    R_w = np.repeat(R_t[None], len(s_grid), axis=0)
    p_b, R_b = base.to_base(p_w, R_w)
    p_b = torch.as_tensor(p_b, dtype=F.DTYPE)
    R_b = torch.as_tensor(R_b, dtype=F.DTYPE)

    u_hat = torch.as_tensor(np.repeat((d / np.linalg.norm(d))[None], len(s_grid), 0)
                            @ base.R, dtype=F.DTYPE)
    q = torch.as_tensor(q_start, dtype=F.DTYPE).reshape(1, 7)
    lo, hi = arm.lo + limit_margin, arm.hi - limit_margin
    q_hist = np.zeros((len(s_grid), 7))
    n_ok, reason, boundary = 0, 0, False

    for t in range(len(s_grid)):
        if t > 0:
            if law.hybrid:
                qn = float(1.0 - arm.limit_margin(q))
                boundary = qn >= (law.tau_exit if boundary else law.tau_enter)
            use_classical = law.classical != 0.0 and not (law.hybrid and boundary)
            use_wln = law.wln or (law.hybrid and boundary)
            use_sat = law.sat or (law.hybrid and boundary)
            hit_limit = False
            for it in range(iters):
                p, R, J, _ = arm.fk(q)
                e = F.pose_error(p, R, p_b[t:t + 1], R_b[t:t + 1])
                g = torch.zeros_like(q)
                if it == 0 and law.w_center != 0.0:
                    g = g + law.w_center * (-(q - arm.mid) / (arm.half ** 2))
                if it == 0 and law.w_manip != 0.0:
                    g = g + law.w_manip * F.manip_grad(arm, q)
                g = g * ds
                if it == 0 and use_classical:
                    g = g + law.classical * ds / F.V_PATH_REF * F._clamp_null(
                        F.classical_null_velocity(arm, q, u_hat[t:t + 1]),
                        J, lam, F.A_MAX_REF)
                winv = None
                if use_wln:
                    dq_ref, _ = F.dls_solve(J, e, lam)
                    winv = 1.0 / F.wln_weight(arm, q, dq_ref)
                active = torch.ones_like(q)
                for _ in range(5):
                    w_act = active if winv is None else active * winv
                    dq, N = F.dls_solve(J * active.unsqueeze(1), e, lam, w_act)
                    dq = dq + (N @ g.unsqueeze(-1)).squeeze(-1)
                    dq = (dq * active).clamp(-0.35, 0.35)
                    out = ((q + dq) < lo) | ((q + dq) > hi)
                    hit_limit = hit_limit or bool(out.any())
                    if not use_sat or not bool((out & (active > 0)).any()):
                        break
                    active = active * (~out)
                q = (q + dq).clamp(lo, hi)

            p, R, _, _ = arm.fk(q)
            e = F.pose_error(p, R, p_b[t:t + 1], R_b[t:t + 1])
            if bool(e[0, :3].norm() > pos_tol or e[0, 3:].norm() > rot_tol):
                reason = 1 if hit_limit else 2
                break
            if float(arm.self_collision_margin(q)) <= 0.0:
                reason = 3
                break
        q_hist[t] = q.numpy()[0]
        n_ok = t + 1

    return s_grid, q_hist, n_ok, reason


def ik_candidates(arm: F.Arm, base: F.BasePose, p_w: np.ndarray, R_w: np.ndarray,
                  n_restart: int, want: int, dedup: float = 0.5, seed: int = 0):
    """Distinct collision-free IK solutions of one pose, spread over the branches."""
    p_b, R_b = base.to_base(p_w[None], R_w[None])
    g = torch.Generator().manual_seed(seed)
    q0 = arm.lo + torch.rand((n_restart, 7), generator=g, dtype=F.DTYPE) * (arm.hi - arm.lo)
    q, ok = F.solve_ik(arm, torch.as_tensor(p_b, dtype=F.DTYPE).expand(n_restart, 3),
                       torch.as_tensor(R_b, dtype=F.DTYPE).expand(n_restart, 3, 3), q0)
    ok = ok & (arm.self_collision_margin(q) > 0.0)
    q = q[ok]
    keep = []
    for i in range(q.shape[0]):
        if all((q[i] - k).abs().max() > dedup for k in keep):
            keep.append(q[i])
        if len(keep) >= want:
            break
    return [k.numpy() for k in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--law', default='hybrid', choices=list(F.LAW_BY_KEY))
    ap.add_argument('--start', type=int, default=None, metavar='K',
                    help='which IK solution at the line start to follow '
                         '(default: the one that gets furthest)')
    ap.add_argument('--list-starts', action='store_true')
    ap.add_argument('--length', type=float, default=1.2, help='line length [m]')
    ap.add_argument('--way-frac', type=float, default=1.0,
                    help='second point, as a fraction of the achieved length')
    ap.add_argument('--cand-alpha', type=float, default=0.07)
    ap.add_argument('--good-alpha', type=float, default=0.45)
    ap.add_argument('--want', type=int, default=12)
    ap.add_argument('--n-restart', type=int, default=1200)
    ap.add_argument('--dedup', type=float, default=0.30,
                    help='min joint distance between two candidates [rad]; the '
                         'self-motion manifold is continuous, so this sets how '
                         'densely it is sampled')
    ap.add_argument('--jaw', type=float, default=0.0)
    ap.add_argument('--base-z', type=float, default=0.70)
    args = ap.parse_args()

    arm = F.Arm(tcp_offset=F.PEN_TCP_OFFSET)   # the pen tip follows the line
    base = F.BasePose(x=0.0, y=0.0, z=args.base_z, yaw_deg=0.0)
    p0 = np.array([0.30, -0.55, base.z + 0.05])
    d = np.array([0.0, 1.0, 0.0])
    R_t = tool_frame(d)

    cands0 = ik_candidates(arm, base, p0, R_t, args.n_restart, args.want,
                           dedup=args.dedup)
    if not cands0:
        raise SystemExit('no IK solution at the start of the line')
    law = F.LAW_BY_KEY[args.law]
    rolls = [rollout_line(arm, base, p0, d, R_t, q, law, length=args.length)
             for q in cands0]
    reach = [s[n - 1] for s, _, n, _ in rolls]
    if args.list_starts:
        print('IK solutions at the start of the line:')
        for k, (q, r) in enumerate(zip(cands0, reach)):
            print(f'  K={k}  follows the line for {r * 100:5.1f} cm')
        return
    k = int(np.argmax(reach)) if args.start is None else args.start
    s_grid, q_hist, n_ok, reason = rolls[k]
    s_end = float(s_grid[n_ok - 1])
    ti = int(np.clip(round(args.way_frac * (n_ok - 1)), 0, n_ok - 1))
    print(f'law {law.name}; start K={k} of {len(cands0)} IK solutions')
    print(f'  followed the line for {s_end * 100:.1f} cm '
          f'({F.STOP_LABEL[reason]})')

    look = p0 + np.array([0.0, 0.5 * s_end, 0.0])
    eye = look + np.array([-2.6, -1.1, 1.5])
    world = ovw.World(cam_pos=tuple(eye), cam_lookat_pos=tuple(look))
    builtins.base = world

    # no pedestal: the arm base floats, nothing but the line and the postures
    a, b = p0, p0 + s_end * d
    ossop.cylinder(spos=tuple(a), epos=tuple(b), radius=0.006, segments=12,
                   rgb=LINE_RGB, alpha=1.0).attach_to(world.scene)

    for tag, t_idx in (('start', 0), ('way point', ti)):
        p_pt = p0 + s_grid[t_idx] * d
        ossop.sphere(pos=tuple(p_pt), radius=0.016, segments=16, rgb=MARK_RGB,
                     alpha=1.0).attach_to(world.scene)
        q_good = q_hist[t_idx]
        cands = ik_candidates(arm, base, p_pt, R_t, args.n_restart,
                              args.want, dedup=args.dedup)
        far = [q for q in cands if np.abs(q - q_good).max() > 0.3]
        for q in far:
            spawn_pen_arm(world.scene, base, q, CAND_RGB, args.cand_alpha,
                          args.jaw)
        spawn_pen_arm(world.scene, base, q_good, GOOD_RGB, args.good_alpha,
                      args.jaw)
        print(f'  {tag:10s} s = {s_grid[t_idx] * 100:5.1f} cm: {len(far)} other '
              f'IK solutions + the one on the trajectory')

    print('one viewer: drag to orbit, scroll to zoom, close the window to quit')
    world.run()


if __name__ == '__main__':
    main()
