"""Visualize policy (or baseline) rollout in `one` viewer.

Usage:
    # RL policy
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller rl --ckpt Yuan/RL_controller/runs11/agent.pt

    # Classical 4-term nullspace controller (hand-tuned strong baseline)
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller classical

    # GPM-JL only (weak baseline)
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller gpm

Hot keys (one's default):
    drag/scroll to orbit / zoom; ESC quits.
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (see train.py).
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import builtins
import math

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.RL_controller.env.baseline_controller import (
    GPMBaselineController, baseline_action_fn,
)
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.ppo import Agent


# CLI -------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--controller", choices=["rl", "classical", "gpm"], default=None,
                    help="rl=trained policy (needs --ckpt); classical=4-term hand-tuned NS; gpm=weak GPM-JL")
parser.add_argument("--ckpt", default=None,
                    help="agent state_dict path; required when --controller rl")
# Back-compat shim: --baseline was the old flag for GPM-only
parser.add_argument("--baseline", action="store_true",
                    help="(deprecated) alias for --controller gpm")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps-per-tick", type=int, default=1,
                    help="env steps per viewer tick; raise for fast-forward")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_yaml = yaml.safe_load(f)

# Resolve which controller to run
if args.baseline and args.controller is None:
    args.controller = "gpm"
if args.controller is None:
    args.controller = "rl" if args.ckpt is not None else None
if args.controller is None:
    parser.error("specify --controller {rl, classical, gpm} (or --ckpt for rl)")
if args.controller == "rl" and args.ckpt is None:
    parser.error("--controller rl requires --ckpt")

device = torch.device(args.device)


# Env -------------------------------------------------------------------------

env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": 1})
env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
# Viz only needs a handful of valid lines (one per episode); 500 is plenty
# and keeps the optional feasibility filter fast (~5s instead of ~130s).
env.line_dist = LineDistribution(
    kin=env.kin, collision=env.collision,
    n_pool=500,
    n_target_noise_deg=cfg_yaml["line_distribution"]["n_target_noise_deg"],
    seed=args.seed if args.seed is not None
         else cfg_yaml["line_distribution"]["train_seed"],
)
if cfg_yaml["line_distribution"].get("feasibility_filter", False):
    env.line_dist.filter_by_classical_controller(
        env_cfg, threshold_m=float(cfg_yaml["line_distribution"]["feasibility_threshold_m"]),
        verbose=False)


# Action source ----------------------------------------------------------------

if args.controller == "gpm":
    ctrl = GPMBaselineController(env.kin,
                                 k_jl=cfg_yaml["baseline"]["k_jl"],
                                 k_dm=float(cfg_yaml["baseline"].get("k_dm", 0.0)),
                                 manip_damping=float(cfg_yaml["baseline"].get("manip_damping", 1e-3)))
    action_fn = baseline_action_fn(ctrl)
    print(f"[viz] using GPM baseline (k_jl={ctrl.k_jl}, k_dm={ctrl.k_dm})")
elif args.controller == "classical":
    ctrl = ClassicalNullspaceController(env.kin)
    action_fn = cn_action_fn(ctrl)
    print("[viz] using classical 4-term nullspace controller "
          "(manip + JL center + cone gradient + q_ref attract)")
else:  # "rl"
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()

    @torch.no_grad()
    def action_fn(env_: NSRLBatchedEnv) -> torch.Tensor:
        mean = agent.actor_mean(env_.current_obs())
        return mean.clamp(-1.0, 1.0)
    print(f"[viz] loaded RL policy from {args.ckpt}")


# Viewer ----------------------------------------------------------------------

base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

# Hand + pen. With env `tcp_offset = 0.2034`, the env controls the pen tip;
# we set `use_pen_tcp=True` so scalar FR3's `_loc_tcp_tf` is also at the pen
# tip — `robot.gl_tcp_tf` then matches the env's EE position 1:1.
robot, hand = make_fr3_with_pen(use_pen_tcp=True)
robot.attach_to(base.scene)
attach_pen_visual(robot)
builtins.base = base
builtins.robot = robot
builtins.hand = hand

# Visualization handles — rebuilt on each new episode
_viz = {"u_hat_arrow": None, "n_target_arrow": None, "z_tool_arrow": None,
        "origin_frame": None, "line_ray": None}
ARROW_LEN_TASK = 0.30
ARROW_LEN_TOOL = 0.15
LINE_RAY_LEN = 1.5      # m, length of the dashed reference ray drawn from p_start
LINE_RAY_RADIUS = 0.003  # m
_cos_cone = math.cos(env_cfg.cone_deg * math.pi / 180.0)


def _detach(name: str):
    obj = _viz.get(name)
    if obj is not None:
        try:
            obj.detach_from(base.scene)
        except Exception:
            pass
        _viz[name] = None


def _tcp_pose():
    """Pen-tip position + flange rotation. With env `tcp_offset=0.2034`, the
    pen tip is the EE — the position task drives THIS point along `v·u_hat`.
    Rotation comes from the flange (z_tool = pen pointing direction)."""
    tcp_tf = robot.gl_tcp_tf
    return tcp_tf[:3, 3], robot.gl_flange_tf[:3, :3]


def _build_episode_viz():
    """Attach u_hat / n_target arrows + origin frame + reference ray, all
    anchored at the PEN TIP (= env control point)."""
    for name in ("u_hat_arrow", "n_target_arrow", "origin_frame", "line_ray"):
        _detach(name)
    p_tip, _ = _tcp_pose()
    u_hat = env.line_dir[0].cpu().numpy()
    n_target = env.n_target[0].cpu().numpy()
    _viz["u_hat_arrow"] = ossop.arrow(
        spos=p_tip, epos=p_tip + u_hat * ARROW_LEN_TASK,
        rgb=(0.2, 0.4, 1.0))
    _viz["u_hat_arrow"].attach_to(base.scene)
    _viz["n_target_arrow"] = ossop.arrow(
        spos=p_tip, epos=p_tip + n_target * ARROW_LEN_TASK,
        rgb=(0.2, 0.9, 0.2))
    _viz["n_target_arrow"].attach_to(base.scene)
    _viz["origin_frame"] = ossop.frame(pos=p_tip)
    _viz["origin_frame"].attach_to(base.scene)
    # Reference ray the PEN TIP should follow — from p_start along u_hat.
    _viz["line_ray"] = ossop.dashed_cylinder(
        spos=p_tip, epos=p_tip + u_hat * LINE_RAY_LEN,
        radius=LINE_RAY_RADIUS, rgb=(0.2, 0.4, 1.0), alpha=0.6)
    _viz["line_ray"].attach_to(base.scene)


def _update_tool_arrow():
    """z_tool arrow at pen tip: green if within cone, red if outside."""
    _detach("z_tool_arrow")
    p_tip, R = _tcp_pose()
    z_tool = R[:, 2]
    n_target = env.n_target[0].cpu().numpy()
    cos_ang = float(np.dot(z_tool, n_target))
    if cos_ang >= _cos_cone:
        rgb = (0.2, 0.9, 0.2)
    else:
        rgb = (0.95, 0.2, 0.2)
    _viz["z_tool_arrow"] = ossop.arrow(
        spos=p_tip, epos=p_tip + z_tool * ARROW_LEN_TOOL, rgb=rgb)
    _viz["z_tool_arrow"].attach_to(base.scene)


# Episode state ---------------------------------------------------------------

_state = {"episode": 0, "step": 0, "term_reason": None, "needs_init": True}


def _sync_robot():
    q = env.q[0].cpu().numpy().astype(np.float32)
    robot.fk(qs=q)


def _start_new_episode():
    env.reset()
    _sync_robot()
    _build_episode_viz()
    _update_tool_arrow()
    _state["episode"] += 1
    _state["step"] = 0
    _state["term_reason"] = None
    _state["needs_init"] = False
    print(f"[viz] episode {_state['episode']} started "
          f"(u_hat={env.line_dir[0].tolist()}, "
          f"n_target={env.n_target[0].tolist()})")


TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl", 5: "truncated"}


def tick(dt):
    if _state["needs_init"]:
        _start_new_episode()
        return

    for _ in range(args.steps_per_tick):
        a = action_fn(env)
        _, _, term, trunc, info = env.step(a, auto_reset=False)
        _state["step"] += 1
        if bool(env.done_persistent[0].item()):
            _state["term_reason"] = TERM_NAMES.get(
                int(info["term_reason"][0].item()), "?")
            break

    _sync_robot()
    _update_tool_arrow()

    if bool(env.done_persistent[0].item()):
        print(f"[viz] episode {_state['episode']} ended: "
              f"step={_state['step']}, reason={_state['term_reason']}")
        # Pause one tick, then start the next episode
        _state["needs_init"] = True


base.schedule_interval(tick, interval=env_cfg.dt)
print(f"[viz] tick = {env_cfg.dt*args.steps_per_tick:.3f}s "
      f"({args.steps_per_tick} env step(s)/tick). Ctrl-C or close window to exit.")
base.run()
