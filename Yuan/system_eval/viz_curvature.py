"""Render one task of the curvature scan in the `one` viewer.

Shows, for a single task (p0, d, n) and a single seed, what changes when the
straight ray is replaced by a constant-curvature arc:

  * the workpiece plane      -- the plane through p0 whose normal is n_target;
                                the path lies in it and the tool must stay
                                inside a 30 deg cone about n_target
  * the tool-orientation cone at p0
  * the nominal path         -- solid over the arc length actually travelled,
                                dashed beyond it, because the path is unbounded
                                by construction and it is the robot that stops,
                                not the path
  * the realised TCP trace
  * the arm at the start pose and at the terminating pose, with faded
    intermediate poses

Every curvature is rolled out with exactly the environment the scan used, so
what is drawn is what was measured.

    # interactive window, one curvature
    python -m Yuan.system_eval.viz_curvature --kappa 4.0

    # write PNGs for a sequence of curvatures, then a contact sheet
    python -m Yuan.system_eval.viz_curvature --kappa 0 --save /tmp/k0.png
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as eval.hybrid).
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _new_env = dict(os.environ)
    _new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + _new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _new_env)

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, build_task_aligned_basis,
)
from Yuan.RL_controller.env.path_geometry import arc_point
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent

CAND = "Yuan/unified_rl/runs/iksel_final_n48/iksel_eval10k_candidates.npz"
EVALSET = "Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
ENV_YAML = "Yuan/RL_controller/config.yaml"

C_PLANE = np.array([0.78, 0.82, 0.88], np.float32)
C_PATH = np.array([0.35, 0.55, 0.95], np.float32)
C_TRACE = np.array([0.05, 0.55, 0.15], np.float32)
C_CONE = np.array([1.00, 0.65, 0.10], np.float32)
C_START = np.array([0.15, 0.45, 1.00], np.float32)
C_END = np.array([0.90, 0.15, 0.15], np.float32)
C_TAN = np.array([0.85, 0.20, 0.75], np.float32)


def pick_task(n_ik: int, task_seed: int, up_thresh: float):
    """A task whose plane is roughly horizontal, so the picture reads as a
    workpiece surface rather than an arbitrarily tilted slab."""
    cand = np.load(REPO / CAND)
    es = np.load(REPO / EVALSET, allow_pickle=True)
    tid = cand["task_indices"].astype(np.int64)
    ok = (cand["ik_ok"].sum(1) >= n_ik) & (es["cs_n_target"][tid][:, 2] > up_thresh)
    pool = tid[ok]
    if not len(pool):
        raise SystemExit(f"no task with n_target_z > {up_thresh}")
    t = int(np.random.default_rng(task_seed).choice(pool))
    r = int(np.nonzero(tid == t)[0][0])
    return t, dict(
        p0=es["cs_p0"][t].astype(np.float64),
        d0=es["cs_line_dir"][t].astype(np.float64),
        nt=es["cs_n_target"][t].astype(np.float64),
        seeds=cand["seeds"][r][np.nonzero(cand["ik_ok"][r])[0][:n_ik]],
        pilot=cand["q0_pilot"][r])


@torch.no_grad()
def rollout_traj(env, agent, classical, q0, p0, d0, nt, kappas,
                 tau_enter, tau_exit):
    """One episode per curvature, batched; returns the joint history."""
    n = len(kappas)
    dt = env.kin.dtype
    dev = env.device
    rep = lambda v: torch.as_tensor(np.repeat(v[None], n, 0), device=dev, dtype=dt)
    env.line_dist = ScriptedLineDistribution({
        "q0": torch.as_tensor(np.repeat(q0[None], n, 0), device=dev, dtype=dt),
        "line_dir": rep(d0), "n_target": rep(nt), "p0": rep(p0),
        "kappa": torch.as_tensor(kappas, device=dev, dtype=dt)})
    env.reset()

    qm, qh = env.q_mid, env.q_half
    mx = lambda q: ((q - qm).abs() / qh).max(-1).values
    using_rl = mx(env.q) < tau_enter
    hist = [env.q.clone().cpu().numpy()]
    term = np.full(n, -1)
    ep_len = np.zeros(n, int)
    done_before = env.done_persistent.clone()

    for _ in range(env.max_steps + 1):
        cq = mx(env.q)
        using_rl = torch.where(using_rl, cq < tau_enter, cq < tau_exit)
        obs = env.current_obs()
        rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)
        B, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping)
        qd = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cls_act = ((B.transpose(-1, -2) @ qd.unsqueeze(-1)).squeeze(-1)
                   / env.a_max).clamp(-1.0, 1.0)
        a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(a, auto_reset=False)
        nd = info["episode_done"]
        if bool(nd.any().item()):
            idx = nd.cpu().numpy()
            term[idx] = info["term_reason"][nd].cpu().numpy()
            ep_len[idx] = env.t[nd].cpu().numpy()
        # Frozen envs stop moving, so appending unconditionally just repeats
        # their last pose; trimmed per-env below.
        hist.append(env.q.clone().cpu().numpy())
        if bool(env.done_persistent.all().item()):
            break
    still = ~env.done_persistent.cpu().numpy()
    ep_len[still] = env.t.cpu().numpy()[still]
    term[still] = 5
    return np.stack(hist, 1), term, ep_len, env.arc_progress.cpu().numpy()


def nominal_path(task, kappa, arc_m, preview_extra, n=200):
    p0, d0, nt = task["p0"], task["d0"], task["nt"]
    tt = lambda v: torch.as_tensor(v, dtype=torch.float64).unsqueeze(0)
    s = torch.linspace(0.0, arc_m + preview_extra, n, dtype=torch.float64)
    pts = arc_point(tt(p0).expand(n, 3), tt(d0).expand(n, 3),
                    tt(nt).expand(n, 3),
                    torch.full((n,), kappa, dtype=torch.float64), s)
    return s.numpy(), pts.numpy().astype(np.float32)


def build_scene(world, task, kappa, q_hist, arc_m, s_grid, pts, stride,
                plane_pad, preview_extra, n_tangent):
    import one.scene.scene_object_primitive as ossop
    from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3
    import one.utils.math as oum

    p0, d0, nt = task["p0"], task["d0"], task["nt"]
    ctr = pts.mean(0)

    # ---- workpiece plane: normal = n_target, through p0 -------------------
    # Kept translucent; it is the reference the cone constraint is written
    # against, not an obstacle.
    span = float(np.abs(pts - ctr).max()) * 2.0 + plane_pad
    rot = oum.rotmat_between_vecs(np.array([0, 0, 1], np.float32),
                                  nt.astype(np.float32))
    ossop.box(half_extents=(span / 2, span / 2, 0.0004),
              pos=(ctr - 0.004 * nt).astype(np.float32), rotmat=rot,
              rgb=C_PLANE, alpha=0.34).attach_to(world.scene)

    # ---- tool-orientation cone at p0 (30 deg half-angle) -----------------
    h = 0.18
    ossop.cone(spos=(p0 + nt * h).astype(np.float32), epos=p0.astype(np.float32),
               radius=float(h * np.tan(np.deg2rad(30.0))), segments=32,
               rgb=C_CONE, alpha=0.30).attach_to(world.scene)

    # ---- nominal path: solid where travelled, dashed beyond --------------
    # Drawn thick and underneath; the realised trace is drawn thin on top, so
    # any visible blue means the TCP left the path.
    n_solid = max(2, int(round(len(s_grid) * arc_m / (arc_m + preview_extra))))
    for i in range(len(s_grid) - 1):
        if i < n_solid:
            ossop.cylinder(spos=pts[i], epos=pts[i + 1], radius=0.0060,
                           rgb=C_PATH, alpha=1.0).attach_to(world.scene)
        elif i % 2 == 0:
            ossop.cylinder(spos=pts[i], epos=pts[i + 1], radius=0.0042,
                           rgb=C_PATH, alpha=0.40).attach_to(world.scene)

    # ---- instantaneous tangent: the only path information the policy sees -
    # The 31-D observation carries this vector and n_target, never kappa and
    # never any lookahead. On a straight ray these arrows are all parallel;
    # on an arc they rotate, and that rotation is the entire difference.
    for j in range(n_tangent):
        s_j = arc_m * (j + 0.5) / n_tangent
        i = int(np.searchsorted(s_grid, s_j))
        i = min(max(i, 1), len(pts) - 2)
        tan = pts[i + 1] - pts[i - 1]
        tan /= max(float(np.linalg.norm(tan)), 1e-9)
        ossop.arrow(spos=pts[i], epos=pts[i] + tan * 0.13,
                    rgb=C_TAN, alpha=0.95).attach_to(world.scene)

    # ---- realised TCP trace ---------------------------------------------
    fk = PenFrankaResearch3(name="pen", enable_cc=False)
    tcp = []
    for q in q_hist:
        fk.goto_given_conf(q.astype(np.float32))
        tcp.append(np.asarray(fk.manipulator.gl_tcp_pos, np.float32))
    tcp = np.stack(tcp)
    for i in range(len(tcp) - 1):
        ossop.cylinder(spos=tcp[i], epos=tcp[i + 1], radius=0.0030,
                       rgb=C_TRACE, alpha=1.0).attach_to(world.scene)
    ossop.sphere(pos=tcp[0], radius=0.013, rgb=C_START,
                 alpha=1.0).attach_to(world.scene)
    ossop.sphere(pos=tcp[-1], radius=0.015, rgb=C_END,
                 alpha=1.0).attach_to(world.scene)

    # ---- arm: start and terminating pose opaque, the rest ghosted --------
    idxs = list(range(0, len(q_hist), max(1, stride)))
    if idxs[-1] != len(q_hist) - 1:
        idxs.append(len(q_hist) - 1)
    for i in idxs:
        edge = (i == 0 or i == idxs[-1])
        r = PenFrankaResearch3(name="pen", enable_cc=False)
        r.goto_given_conf(q_hist[i].astype(np.float32))
        r.gen_meshmodel(alpha=1.0 if edge else 0.05,
                        toggle_tcp_frame=False).attach_to(world)
    return ctr, tcp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="signed curvature [1/m]; 0 = the straight ray")
    ap.add_argument("--task-seed", type=int, default=7)
    ap.add_argument("--n-ik", type=int, default=12)
    ap.add_argument("--seed-idx", type=int, default=None,
                    help="which pool seed to fly; default = best at kappa=0")
    ap.add_argument("--up-thresh", type=float, default=0.85)
    ap.add_argument("--k-lateral", type=float, default=5.0)
    ap.add_argument("--tau-enter", type=float, default=0.98)
    ap.add_argument("--tau-exit", type=float, default=0.94)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--cam-dist", type=float, default=1.5)
    ap.add_argument("--fix-span", type=float, default=-1.0,
                    help="force the framing radius so panels at "
                         "different curvatures share one camera")
    ap.add_argument("--cam-tilt", type=float, default=52.0,
                    help="deg away from the surface normal; 0 = straight down")
    ap.add_argument("--n-tangent", type=int, default=5,
                    help="instantaneous-tangent arrows along the path")
    ap.add_argument("--plane-pad", type=float, default=0.22)
    ap.add_argument("--preview", type=float, default=0.45,
                    help="metres of unbounded path drawn past the stop point")
    ap.add_argument("--save", default=None, help="write a PNG and exit")
    ap.add_argument("--win", type=int, nargs=2, default=[1280, 900])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tid, task = pick_task(args.n_ik, args.task_seed, args.up_thresh)
    device = torch.device(args.device)

    with open(REPO / ENV_YAML) as f:
        env_yaml = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in env_yaml["env"].items() if k in keys}
    n_seed = task["seeds"].shape[0]
    env = NSRLBatchedEnv(
        EnvConfig(**{**env_kw, "n_envs": n_seed, "k_lateral": args.k_lateral}),
        line_dist=None, device=device)
    classical = ClassicalNullspaceController(env.kin)
    with open(REPO / RL_CKPT / "config.yaml") as f:
        rl_cfg = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=rl_cfg["ppo"]["hidden_dim"],
                  init_log_std=rl_cfg["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(REPO / RL_CKPT / "agent.pt",
                                     map_location=device))
    agent.eval()

    def fly(q0, kappa):
        e = NSRLBatchedEnv(
            EnvConfig(**{**env_kw, "n_envs": 1, "k_lateral": args.k_lateral}),
            line_dist=None, device=device)
        return rollout_traj(e, agent, ClassicalNullspaceController(e.kin), q0,
                            task["p0"], task["d0"], task["nt"], [kappa],
                            args.tau_enter, args.tau_exit)

    # Pick the seed the same way the scan's reference does: best on the line.
    if args.seed_idx is None:
        arcs = []
        for i in range(n_seed):
            *_, a = fly(task["seeds"][i], 0.0)
            arcs.append(float(a[0]))
        seed_idx = int(np.argmax(arcs))
        print(f"[viz] best-on-the-line seed = #{seed_idx} "
              f"(arc {arcs[seed_idx]:.3f} m at kappa=0)")
    else:
        seed_idx = args.seed_idx
    q0 = task["seeds"][seed_idx]

    hist, term, ep_len, arc = fly(q0, args.kappa)
    q_hist = hist[0][:ep_len[0] + 1]
    arc_m = float(arc[0])
    R = np.inf if args.kappa == 0 else 1.0 / args.kappa
    print(f"[viz] task {tid}  seed #{seed_idx}  kappa={args.kappa:+.2f} "
          f"(R={R:.3f} m)  arc={arc_m:.3f} m  steps={ep_len[0]}  "
          f"stop={TERM_NAMES.get(int(term[0]), '?')}")

    # Frame the camera on the scene before the window exists: the view matrix
    # is built in Camera.__init__ and setting look_at afterwards does not
    # rebuild it.
    s_grid, pts = nominal_path(task, args.kappa, arc_m, args.preview)
    pctr = pts.mean(0)
    ctr = 0.5 * (pctr + np.array([0.0, 0.0, 0.4], np.float32))
    # Look down the surface normal, offset away from the base, so the shape of
    # the path in the workpiece plane is what the picture is about. Edge-on
    # views make an arc indistinguishable from a line.
    nt = task["nt"].astype(np.float32)
    out = pctr - np.array([0.0, 0.0, float(pctr[2])], np.float32)
    out = out - float(out @ nt) * nt
    out = out / max(float(np.linalg.norm(out)), 1e-6)
    span = (args.fix_span if args.fix_span > 0
            else float(np.abs(pts - pctr).max()))
    dist = args.cam_dist * (0.75 + span)
    eye = pctr + nt * dist * np.cos(np.deg2rad(args.cam_tilt)) \
        + out * dist * np.sin(np.deg2rad(args.cam_tilt))

    import one.viewer.world as ovw
    world = ovw.World(cam_pos=eye.tolist(), cam_lookat_pos=ctr.tolist(),
                      win_size=tuple(args.win))
    import one.scene.scene_object_primitive as ossop
    ossop.frame().attach_to(world.scene)
    build_scene(world, task, args.kappa, q_hist, arc_m, s_grid, pts,
                args.stride, args.plane_pad, args.preview, args.n_tangent)
    world.set_caption(
        f"kappa={args.kappa:+.2f} 1/m  R={R:.3f} m  arc={arc_m:.3f} m  "
        f"stop={TERM_NAMES.get(int(term[0]), '?')}")

    if args.save:
        import pyglet
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)

        def _grab(dt):
            pyglet.image.get_buffer_manager().get_color_buffer().save(str(out))
            print(f"[viz] saved -> {out}")
            pyglet.app.exit()

        world.schedule_once(_grab, delay=1.0)
    world.run()


if __name__ == "__main__":
    main()
