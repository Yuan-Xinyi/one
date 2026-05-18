"""Overlay viz: two transparent FR3 arms in the same scene, both following
the SAME (q0, u_hat, n_target) line spec but with different controllers.

Red transparent  = RL policy (--ckpt)
Blue transparent = Classical 4-term nullspace controller

When both episodes end (one or both terminate), a new shared spec is drawn
and both reset to it. You can see directly which controller survives longer
and how the joint configurations diverge over time.

Usage:
    python -m Yuan.RL_controller.visualize_overlay \\
        --config Yuan/RL_controller/config.yaml \\
        --ckpt   Yuan/RL_controller/runs11/agent.pt
"""
from __future__ import annotations

# Self-relaunch with LD_LIBRARY_PATH (see train.py)
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
from Yuan.RL_controller.env.line_distribution import LineDistribution, ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn
from Yuan.RL_controller.ppo import Agent


# CLI -------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True, help="RL agent ckpt")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=None,
                    help="line distribution seed (deterministic line sequence)")
parser.add_argument("--steps-per-tick", type=int, default=1,
                    help="env steps per viewer tick; raise for fast-forward")
parser.add_argument("--n-episodes", type=int, default=50,
                    help="number of shared line specs to cycle through")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_yaml = yaml.safe_load(f)
device = torch.device(args.device)


# Two envs (n_envs=1 each) -----------------------------------------------------
env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": 1})

def make_env():
    e = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    return e

env_rl = make_env()
env_cl = make_env()

# Pre-sample shared line specs (both envs see identical sequence)
# For viz, override n_pool to ~10× n_episodes — filter only this many (fast).
viz_pool = max(args.n_episodes * 10, 200)
sampler = LineDistribution(
    kin=env_rl.kin, collision=env_rl.collision,
    n_pool=viz_pool,
    n_target_noise_deg=cfg_yaml["line_distribution"]["n_target_noise_deg"],
    seed=args.seed if args.seed is not None
         else cfg_yaml["line_distribution"]["train_seed"],
)
if cfg_yaml["line_distribution"].get("feasibility_filter", False):
    sampler.filter_by_classical_controller(
        env_cfg, threshold_m=float(cfg_yaml["line_distribution"]["feasibility_threshold_m"]),
        verbose=False)
shared_specs = sampler.sample(args.n_episodes)
print(f"[overlay] pre-sampled {args.n_episodes} shared line specs (from pool of {viz_pool})")


# Action sources ---------------------------------------------------------------
rl_agent = Agent(env_rl.obs_dim, env_rl.act_dim,
                 hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                 init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
rl_agent.load_state_dict(torch.load(args.ckpt, map_location=device))
rl_agent.eval()
print(f"[overlay] loaded RL policy from {args.ckpt}")

@torch.no_grad()
def rl_action_fn(env_):
    return rl_agent.actor_mean(env_.current_obs()).clamp(-1.0, 1.0)

cl_ctrl = ClassicalNullspaceController(env_rl.kin)
cl_action_fn = cn_action_fn(cl_ctrl)
print(f"[overlay] classical nullspace controller ready")


# Viewer -----------------------------------------------------------------------
base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

# Two robots with different colors + transparency
RL_RGB = (0.95, 0.25, 0.20)     # red
CL_RGB = (0.20, 0.45, 0.95)     # blue
ALPHA = 0.45

robot_rl, hand_rl = make_fr3_with_pen(use_pen_tcp=True)
robot_rl.attach_to(base.scene)
attach_pen_visual(robot_rl, rgb=RL_RGB, alpha=ALPHA)
robot_rl.rgba = (*RL_RGB, ALPHA)

robot_cl, hand_cl = make_fr3_with_pen(use_pen_tcp=True)
robot_cl.attach_to(base.scene)
attach_pen_visual(robot_cl, rgb=CL_RGB, alpha=ALPHA)
robot_cl.rgba = (*CL_RGB, ALPHA)

builtins.base = base
builtins.robot_rl = robot_rl
builtins.robot_cl = robot_cl

# Shared-line viz handles
_viz = {"u_hat_arrow": None, "n_target_arrow": None, "origin_frame": None,
        "line_ray": None}
ARROW_LEN_TASK = 0.30
LINE_RAY_LEN = 1.5
LINE_RAY_RADIUS = 0.003


def _detach(name):
    obj = _viz.get(name)
    if obj is not None:
        try:
            obj.detach_from(base.scene)
        except Exception:
            pass
        _viz[name] = None


def _build_shared_viz(p_start, u_hat, n_target):
    for k in ("u_hat_arrow", "n_target_arrow", "origin_frame", "line_ray"):
        _detach(k)
    _viz["u_hat_arrow"] = ossop.arrow(
        spos=p_start, epos=p_start + u_hat * ARROW_LEN_TASK,
        rgb=(0.2, 0.4, 1.0))
    _viz["u_hat_arrow"].attach_to(base.scene)
    _viz["n_target_arrow"] = ossop.arrow(
        spos=p_start, epos=p_start + n_target * ARROW_LEN_TASK,
        rgb=(0.2, 0.9, 0.2))
    _viz["n_target_arrow"].attach_to(base.scene)
    _viz["origin_frame"] = ossop.frame(pos=p_start)
    _viz["origin_frame"].attach_to(base.scene)
    _viz["line_ray"] = ossop.dashed_cylinder(
        spos=p_start, epos=p_start + u_hat * LINE_RAY_LEN,
        radius=LINE_RAY_RADIUS, rgb=(0.2, 0.4, 1.0), alpha=0.6)
    _viz["line_ray"].attach_to(base.scene)


# Episode state ----------------------------------------------------------------
_state = {"ep_idx": -1, "needs_init": True, "step_count": 0,
          "rl_done": False, "cl_done": False,
          "rl_len": 0, "cl_len": 0,
          "rl_reason": None, "cl_reason": None}

TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl", 5: "truncated"}


def _set_env_to_spec(env, spec_idx):
    """Manually push spec[i] into env state without going through line_dist."""
    env.q[:] = shared_specs["q0"][spec_idx:spec_idx + 1]
    env.line_dir[:] = shared_specs["line_dir"][spec_idx:spec_idx + 1]
    env.n_target[:] = shared_specs["n_target"][spec_idx:spec_idx + 1]
    env.t.zero_()
    env.a_prev.zero_()
    env.B_prev_valid.zero_()
    env.done_persistent.zero_()


def _sync_robot(robot, env):
    q = env.q[0].cpu().numpy().astype(np.float32)
    robot.fk(qs=q)


def _start_new_episode():
    _state["ep_idx"] = (_state["ep_idx"] + 1) % args.n_episodes
    i = _state["ep_idx"]
    _set_env_to_spec(env_rl, i)
    _set_env_to_spec(env_cl, i)
    _sync_robot(robot_rl, env_rl)
    _sync_robot(robot_cl, env_cl)
    # Build shared viz anchored at the starting pen-tip position (use either robot)
    p_start = robot_rl.gl_tcp_tf[:3, 3]
    u_hat = env_rl.line_dir[0].cpu().numpy()
    n_target = env_rl.n_target[0].cpu().numpy()
    _build_shared_viz(p_start, u_hat, n_target)
    _state["needs_init"] = False
    _state["step_count"] = 0
    _state["rl_done"] = False
    _state["cl_done"] = False
    _state["rl_len"] = 0
    _state["cl_len"] = 0
    _state["rl_reason"] = None
    _state["cl_reason"] = None
    print(f"\n[overlay] episode {i + 1}/{args.n_episodes} started "
          f"u_hat={u_hat.round(2).tolist()} n_target={n_target.round(2).tolist()}")


def tick(dt):
    if _state["needs_init"]:
        _start_new_episode()
        return

    for _ in range(args.steps_per_tick):
        # Step RL env if alive
        if not _state["rl_done"]:
            a = rl_action_fn(env_rl)
            _, _, _, _, info = env_rl.step(a, auto_reset=False)
            if bool(env_rl.done_persistent[0].item()):
                _state["rl_done"] = True
                _state["rl_len"] = int(env_rl.t[0].item())
                _state["rl_reason"] = TERM_NAMES.get(int(info["term_reason"][0].item()), "?")
                print(f"  [RL ]   ended step={_state['rl_len']} reason={_state['rl_reason']}")
        # Step classical env if alive
        if not _state["cl_done"]:
            a = cl_action_fn(env_cl)
            _, _, _, _, info = env_cl.step(a, auto_reset=False)
            if bool(env_cl.done_persistent[0].item()):
                _state["cl_done"] = True
                _state["cl_len"] = int(env_cl.t[0].item())
                _state["cl_reason"] = TERM_NAMES.get(int(info["term_reason"][0].item()), "?")
                print(f"  [CL]   ended step={_state['cl_len']} reason={_state['cl_reason']}")
        _state["step_count"] += 1
        if _state["rl_done"] and _state["cl_done"]:
            break

    # Sync visuals
    _sync_robot(robot_rl, env_rl)
    _sync_robot(robot_cl, env_cl)

    # Both done → advance to next episode (with a tiny visual pause)
    if _state["rl_done"] and _state["cl_done"]:
        ratio = (_state["cl_len"] / max(_state["rl_len"], 1)) if _state["rl_len"] > 0 else float("inf")
        print(f"  [done]  RL={_state['rl_len']} CL={_state['cl_len']}  CL/RL ratio={ratio:.2f}")
        _state["needs_init"] = True


base.schedule_interval(tick, interval=env_cfg.dt)
print(f"[overlay] tick = {env_cfg.dt * args.steps_per_tick:.3f}s "
      f"({args.steps_per_tick} env step(s)/tick).")
print(f"          RED = RL policy   /   BLUE = classical nullspace controller")
base.run()
