"""Why a radial pass dies and a tangential pass does not.

Two traverses of the same workpiece plane, both heading +x, both required to
keep the tool axis inside a 30 deg cone about the plane normal:

  radial      start at (x0, 0)      -- the path runs straight at the base axis
  tangential  start at (x0, y_off)  -- the path sweeps past the base

The tool axis is drawn as an arrow at intervals along each traverse, coloured
by how far it has tipped from the plane normal. On the radial pass the arm has
to fold as its distance to the base axis collapses, and the redundancy is spent
undoing that fold, so the arrows lean over until the tool leaves the cone. On
the tangential pass the distance to the base axis barely changes; the motion is
mostly a rotation about joint 1, which a vertically held tool does not feel at
all, so the arrows stay upright.

    python -m Yuan.system_eval.viz_radial_vs_tangential --case radial \
        --save /tmp/radial.png
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _e = dict(os.environ)
    _e["LD_LIBRARY_PATH"] = _conda_lib + ":" + _e.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _e)

import argparse
import dataclasses
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent
from Yuan.system_eval.pen_collision import PenSphereCollision
from Yuan.system_eval.fig_slice_capacity import cone_ik_seeds

TABLE = "Yuan/unified_rl/runs/iksel_clean_v1/cvt_table_201600.npz"
RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
ENV_YAML = "Yuan/RL_controller/config.yaml"
CONE_DEG = 30.0

C_PLANE = np.array([0.80, 0.84, 0.90], np.float32)
C_PATH = np.array([0.35, 0.55, 0.95], np.float32)
C_TRACE = np.array([0.05, 0.55, 0.15], np.float32)
C_CONE = np.array([1.00, 0.65, 0.10], np.float32)


def tilt_colour(deg: float) -> np.ndarray:
    """Green at 0 deg, red at the 30 deg cone limit."""
    t = float(np.clip(deg / CONE_DEG, 0.0, 1.0))
    return np.array([0.15 + 0.80 * t, 0.75 - 0.65 * t, 0.15], np.float32)


@torch.no_grad()
def fly(env, agent, classical, q0, p0, d0, nt, tau_e=0.98, tau_x=0.94):
    """One episode; returns the joint history, TCP history and tool tilt."""
    dev, dt = env.device, env.kin.dtype
    n = q0.shape[0]
    env.line_dist = ScriptedLineDistribution({
        "q0": q0, "line_dir": d0, "n_target": nt, "p0": p0,
        "kappa": torch.zeros(n, device=dev, dtype=dt)})
    env.reset()
    qm, qh = env.q_mid, env.q_half
    mx = lambda q: ((q - qm).abs() / qh).max(-1).values
    using = mx(env.q) < tau_e
    hist, tcp, tilt = [], [], []
    term = np.full(n, -1)
    for _ in range(env.max_steps + 1):
        p, R, _, _ = env.kin.tcp_fk_jac(env.q)
        hist.append(env.q.clone().cpu().numpy())
        tcp.append(p.cpu().numpy())
        tilt.append(torch.rad2deg(torch.arccos(
            (R[:, :, 2] * nt).sum(-1).clamp(-1, 1))).cpu().numpy())
        cq = mx(env.q)
        using = torch.where(using, cq < tau_e, cq < tau_x)
        obs = env.current_obs()
        rl = agent.actor_mean(obs).clamp(-1.0, 1.0)
        B, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        qd = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cl = ((B.transpose(-1, -2) @ qd.unsqueeze(-1)).squeeze(-1)
              / env.a_max).clamp(-1.0, 1.0)
        _, _, _, _, info = env.step(torch.where(using.unsqueeze(-1), rl, cl),
                                    auto_reset=False)
        nd = info["episode_done"].cpu().numpy()
        if nd.any():
            term[nd] = info["term_reason"].cpu().numpy()[nd]
        if bool(env.done_persistent.all().item()):
            break
    return (np.stack(hist, 1), np.stack(tcp, 1), np.stack(tilt, 1),
            term, env.t.cpu().numpy(), env.arc_progress.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["radial", "tangential"], default="radial")
    ap.add_argument("--z0", type=float, default=0.20)
    ap.add_argument("--x0", type=float, default=-0.55)
    ap.add_argument("--y-off", type=float, default=0.60)
    ap.add_argument("--y0", type=float, default=None,
                    help="explicit start y; overrides --case")
    ap.add_argument("--reach-len", type=float, default=None,
                    help="pointwise-reachable length at this cell; drawn "
                         "thin beyond the part actually traversed")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--arrow-every", type=int, default=6)
    ap.add_argument("--preview", type=float, default=0.5)
    ap.add_argument("--cam-pos", type=float, nargs=3,
                    default=[1.55, -1.75, 1.45])
    ap.add_argument("--cam-look", type=float, nargs=3,
                    default=[-0.10, 0.15, 0.32])
    ap.add_argument("--plane-half", type=float, default=1.05)
    ap.add_argument("--win", type=int, nargs=2, default=[1280, 900])
    ap.add_argument("--save", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    y = (args.y0 if args.y0 is not None
         else (0.0 if args.case == "radial" else args.y_off))
    p0 = np.array([args.x0, y, args.z0], np.float32)
    d0 = np.array([1.0, 0.0, 0.0], np.float32)
    nv = np.array([0.0, 0.0, -1.0], np.float32)
    dev = torch.device(args.device)

    with open(REPO / ENV_YAML) as f:
        ycfg = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in ycfg["env"].items() if k in keys}
    env = NSRLBatchedEnv(EnvConfig(**{**kw, "n_envs": args.n_seeds,
                                      "k_lateral": 5.0}), None, dev)
    env.collision = PenSphereCollision(env.kin.tcp_offset, device=dev)

    T = np.load(REPO / TABLE)
    tree = cKDTree(np.concatenate([T["pos"] * 20.0, T["zax"]], 1).astype(np.float32))
    seeds, ok = cone_ik_seeds(env, tree, T, p0[None], nv, args.n_seeds,
                              np.random.default_rng(0))
    valid = np.nonzero(ok[0])[0]
    if not len(valid):
        raise SystemExit(f"no cone-IK solution at {p0}")

    with open(REPO / RL_CKPT / "config.yaml") as f:
        rc = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim, hidden_dim=rc["ppo"]["hidden_dim"],
                  init_log_std=rc["ppo"]["init_log_std"]).to(dev)
    agent.load_state_dict(torch.load(REPO / RL_CKPT / "agent.pt", map_location=dev))
    agent.eval()
    classical = ClassicalNullspaceController(env.kin)

    dt = env.kin.dtype
    S = args.n_seeds
    q_hist, tcp_hist, tilt_hist, term, steps, arc = fly(
        env, agent, classical,
        torch.as_tensor(np.nan_to_num(seeds[0]), device=dev, dtype=dt),
        torch.as_tensor(np.tile(p0, (S, 1)), device=dev, dtype=dt),
        torch.as_tensor(np.tile(d0, (S, 1)), device=dev, dtype=dt),
        torch.as_tensor(np.tile(nv, (S, 1)), device=dev, dtype=dt))
    arc_v = np.where(ok[0], arc, -1)
    b = int(np.argmax(arc_v))
    L = int(steps[b]) + 1
    qs, ps, ts = q_hist[b][:L], tcp_hist[b][:L], tilt_hist[b][:L]
    r0 = float(np.linalg.norm(p0[:2]))
    r1 = float(np.linalg.norm(ps[-1][:2]))
    print(f"[viz] {args.case}: start ({p0[0]:+.2f}, {p0[1]:+.2f}), "
          f"{len(valid)} seeds, best arc {arc[b]:.3f} m, {L-1} steps, "
          f"stop = {TERM_NAMES.get(int(term[b]), '?')}")
    print(f"[viz]   tool tilt {ts[0]:.1f} deg -> {ts[-1]:.1f} deg   "
          f"distance to base axis {r0:.3f} -> {r1:.3f} m")

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    import one.utils.math as oum
    from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3

    span = float(arc[b]) + args.preview
    ctr = p0 + np.array([span * 0.5, 0.0, 0.0], np.float32)

    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    import one.utils.math as oum
    from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3

    # One fixed camera for both cases, framing the whole robot and both paths,
    # so the two renders can be compared side by side.
    world = ovw.World(cam_pos=list(args.cam_pos),
                      cam_lookat_pos=list(args.cam_look),
                      win_size=tuple(args.win))
    ossop.frame().attach_to(world.scene)

    # workpiece plane
    half = args.plane_half
    ossop.box(half_extents=(half, half, 0.0004),
              pos=np.array([0.0, 0.0, args.z0 - 0.004], np.float32),
              rotmat=np.eye(3, dtype=np.float32), rgb=C_PLANE,
              alpha=0.30).attach_to(world.scene)

    # nominal path: solid over the traversed part, dashed beyond
    a0 = p0.astype(np.float32)
    a1 = (p0 + d0 * float(arc[b])).astype(np.float32)
    reach = float(args.reach_len) if args.reach_len else span
    a2 = (p0 + d0 * reach).astype(np.float32)
    # Thick: actually traversed. Thin: pointwise reachable but never reached —
    # every point of it admits a valid configuration on its own.
    ossop.cylinder(spos=a0, epos=a1, radius=0.0060, rgb=C_PATH,
                   alpha=1.0).attach_to(world.scene)
    ossop.cylinder(spos=a1, epos=a2, radius=0.0022, rgb=C_PATH,
                   alpha=0.55).attach_to(world.scene)
    ossop.sphere(pos=a2, radius=0.012,
                 rgb=np.array([0.25, 0.45, 0.95], np.float32),
                 alpha=0.55).attach_to(world.scene)
    ossop.sphere(pos=a0, radius=0.014, rgb=np.array([0.15, 0.45, 1.0], np.float32),
                 alpha=1.0).attach_to(world.scene)
    ossop.sphere(pos=ps[-1].astype(np.float32), radius=0.016,
                 rgb=np.array([0.9, 0.15, 0.15], np.float32),
                 alpha=1.0).attach_to(world.scene)

    # realised TCP trace
    for i in range(len(ps) - 1):
        ossop.cylinder(spos=ps[i].astype(np.float32), epos=ps[i + 1].astype(np.float32),
                       radius=0.003, rgb=C_TRACE, alpha=1.0).attach_to(world.scene)

    # the tool axis along the way, coloured by how far it has tipped
    fk = PenFrankaResearch3(name="pen", enable_cc=False)
    idxs = list(range(0, len(qs), max(1, args.arrow_every)))
    if idxs[-1] != len(qs) - 1:
        idxs.append(len(qs) - 1)
    for i in idxs:
        fk.goto_given_conf(qs[i].astype(np.float32))
        R = np.asarray(fk.manipulator.gl_tcp_rotmat, np.float32)
        tip = ps[i].astype(np.float32)
        ossop.arrow(spos=tip, epos=tip - R[:, 2] * 0.20,
                    rgb=tilt_colour(float(ts[i])), alpha=0.95).attach_to(world.scene)

    # the cone the tool must stay inside, at the start and at the stop
    for pos in (a0, ps[-1].astype(np.float32)):
        h = 0.20
        ossop.cone(spos=(pos - nv * h).astype(np.float32), epos=pos,
                   radius=float(h * math.tan(math.radians(CONE_DEG))), segments=32,
                   rgb=C_CONE, alpha=0.22).attach_to(world.scene)

    # arm at the start and at the stop, ghosts in between
    gi = list(range(0, len(qs), max(1, args.stride)))
    if gi[-1] != len(qs) - 1:
        gi.append(len(qs) - 1)
    for i in gi:
        edge = (i == 0 or i == gi[-1])
        r = PenFrankaResearch3(name="pen", enable_cc=False)
        r.goto_given_conf(qs[i].astype(np.float32))
        r.gen_meshmodel(alpha=1.0 if edge else 0.05).attach_to(world)

    world.set_caption(
        f"{args.case}: arc {arc[b]:.3f} m, tilt {ts[0]:.0f}->{ts[-1]:.0f} deg, "
        f"r {r0:.2f}->{r1:.2f} m, stop = {TERM_NAMES.get(int(term[b]), '?')}")
    if args.save:
        import pyglet
        outp = Path(args.save)
        outp.parent.mkdir(parents=True, exist_ok=True)

        def _grab(dtt):
            pyglet.image.get_buffer_manager().get_color_buffer().save(str(outp))
            print(f"[viz] saved -> {outp}")
            pyglet.app.exit()

        world.schedule_once(_grab, delay=1.2)
    world.run()


if __name__ == "__main__":
    main()
