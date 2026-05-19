"""Visualize policy (or baseline) rollout in `one` viewer.

Usage:
    # RL policy
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller rl --ckpt path/to/agent.pt

    # Classical 4-term nullspace controller (hand-tuned strong baseline)
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller classical

    # GPM-JL only (weak baseline)
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller gpm

    # Overlay: RL + baseline in same scene, both half-transparent
    # (black pen = RL, orange pen = baseline). Identical seeded episodes.
    python -m Yuan.RL_controller.visualize \\
        --config Yuan/RL_controller/config.yaml \\
        --controller rl --ckpt path/to/agent.pt --overlay classical

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
parser.add_argument("--overlay", choices=["classical", "gpm"], default="classical",
                    help="overlay a second baseline controller in the same scene; both robots run "
                         "identical episodes (same seeded line_dist) so trajectories can be compared "
                         "side-by-side. Both robots are rendered semi-transparent; pen colors distinguish them.")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--case", default=None,
                    help="comma-separated eval case indices (e.g. '0,5,17'); "
                         "replays specs from the eval holdout pool — same "
                         "holdout_seed/n_pool/feasibility filter as eval.py & "
                         "plot_joint_trajectories.py, so case i here == line i "
                         "in eval.csv. Looped: viewer cycles through the list "
                         "indefinitely. Omit to use the random viz pool.")
parser.add_argument("--steps-per-tick", type=int, default=1,
                    help="env steps per viewer tick; raise for fast-forward")
parser.add_argument("--slowdown", type=float, default=4.0,
                    help="playback slowdown factor; tick interval = env_dt * slowdown. "
                         "e.g. 4.0 = 0.25x speed (one env step every 4 * env_dt seconds)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg_yaml = yaml.safe_load(f)

# Resolve which controller to run
if args.controller is None:
    args.controller = "rl" if args.ckpt is not None else None
if args.controller is None:
    parser.error("specify --controller {rl, classical, gpm} (or --ckpt for rl)")
if args.controller == "rl" and args.ckpt is None:
    parser.error("--controller rl requires --ckpt")

device = torch.device(args.device)


# Env + runner construction --------------------------------------------------
# Each "runner" bundles one env + one action source + one rendered robot. With
# --overlay we build two runners that share an identical seeded LineDistribution
# (deterministic pool + lock-step sampling) so both controllers run the SAME
# episode and trajectories can be compared visually.

if args.overlay is not None and args.controller != "rl":
    parser.error("--overlay only makes sense together with --controller rl")

env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": 1})
seed_val = args.seed if args.seed is not None else cfg_yaml["line_distribution"]["train_seed"]


case_ids: list[int] | None = None
if args.case is not None:
    case_ids = [int(s) for s in args.case.split(",") if s.strip() != ""]
    if not case_ids:
        parser.error("--case parsed to an empty list")


class LoopingScriptedDist:
    """ScriptedLineDistribution that loops + tracks which eval case is next.

    Unlike eval's ScriptedLineDistribution (which exhausts after one pass),
    this wraps the cursor so the viewer can keep cycling through the chosen
    cases. Exposes `current_case_id()` so episode-start logging can print
    which eval case the viewer is showing.
    """
    def __init__(self, specs: dict[str, torch.Tensor], case_ids: list[int]):
        assert specs["q0"].shape[0] == len(case_ids)
        self._specs = specs
        self._case_ids = case_ids
        self._cursor = 0
        self._total = len(case_ids)

    def sample(self, n: int, generator: torch.Generator | None = None):
        idx = [(self._cursor + i) % self._total for i in range(n)]
        out = {k: v[idx].clone() for k, v in self._specs.items()}
        self._cursor = (self._cursor + n) % self._total
        return out

    def peek_case_id(self) -> int:
        return self._case_ids[self._cursor]


# Eval holdout (shared across all runners) — built once if --case is set.
_eval_specs_cache: dict | None = None


def _eval_holdout_specs() -> dict[str, torch.Tensor]:
    """Sample the same holdout as eval.py / plot_joint_trajectories.py."""
    global _eval_specs_cache
    if _eval_specs_cache is not None:
        return _eval_specs_cache
    line_cfg = cfg_yaml["line_distribution"]
    eval_cfg = cfg_yaml["eval"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    proxy = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    sampler = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_cfg["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    holdout = sampler.sample(eval_cfg["n_holdout"])
    assert case_ids is not None
    n_holdout = eval_cfg["n_holdout"]
    for cid in case_ids:
        if cid < 0 or cid >= n_holdout:
            raise SystemExit(f"--case {cid} out of range [0, {n_holdout})")
    _eval_specs_cache = {k: v[case_ids].clone() for k, v in holdout.items()}
    return _eval_specs_cache


def _build_env() -> NSRLBatchedEnv:
    e = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    if case_ids is not None:
        # Replay eval cases by index. Each runner gets an independent
        # LoopingScriptedDist over the SAME spec slice, so overlay runners
        # stay in lock-step (both advance the cursor identically per reset).
        e.line_dist = LoopingScriptedDist(_eval_holdout_specs(), case_ids)
    else:
        # Viz only needs a handful of valid lines (one per episode); 500 is
        # plenty and keeps the optional feasibility filter fast (~5s vs ~130s).
        e.line_dist = LineDistribution(
            kin=e.kin, collision=e.collision,
            n_pool=500,
            n_target_noise_deg=cfg_yaml["line_distribution"]["n_target_noise_deg"],
            seed=seed_val,
        )
        if cfg_yaml["line_distribution"].get("feasibility_filter", False):
            e.line_dist.filter_by_classical_controller(
                env_cfg, threshold_m=float(cfg_yaml["line_distribution"]["feasibility_threshold_m"]),
                verbose=False)
    return e


def _build_action_fn(kind: str, env_for_dims: NSRLBatchedEnv):
    if kind == "gpm":
        ctrl = GPMBaselineController(env_for_dims.kin,
                                     k_jl=cfg_yaml["baseline"]["k_jl"],
                                     k_dm=float(cfg_yaml["baseline"].get("k_dm", 0.0)),
                                     manip_damping=float(cfg_yaml["baseline"].get("manip_damping", 1e-3)))
        print(f"[viz] GPM baseline (k_jl={ctrl.k_jl}, k_dm={ctrl.k_dm})")
        return baseline_action_fn(ctrl)
    if kind == "classical":
        ctrl = ClassicalNullspaceController(env_for_dims.kin)
        print("[viz] classical 4-term nullspace controller "
              "(manip + JL center + cone gradient + q_ref attract)")
        return cn_action_fn(ctrl)
    # "rl"
    agent = Agent(env_for_dims.obs_dim, env_for_dims.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()

    @torch.no_grad()
    def action_fn(env_: NSRLBatchedEnv) -> torch.Tensor:
        return agent.actor_mean(env_.current_obs()).clamp(-1.0, 1.0)
    print(f"[viz] RL policy loaded from {args.ckpt}")
    return action_fn


# Viewer ----------------------------------------------------------------------

base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

# Build runners. Single runner unless --overlay is set, in which case we add a
# second baseline runner sharing the same seeded line_dist (deterministic ⇒
# identical episodes). Both robots are half-transparent; pen color distinguishes
# them: blue = RL (ours), red = overlay baseline.
runners: list[dict] = []


def _add_runner(kind: str, alpha: float, body_rgb, pen_rgb):
    env_i = _build_env()
    fn_i = _build_action_fn(kind, env_i)
    robot_i, hand_i = make_fr3_with_pen(use_pen_tcp=True)
    robot_i.attach_to(base.scene)
    attach_pen_visual(robot_i, rgb=pen_rgb, alpha=alpha)
    if body_rgb is not None:
        robot_i.rgb = body_rgb  # uniform tint across all links + mounted hand
    robot_i.alpha = alpha
    runners.append({"label": kind, "env": env_i, "fn": fn_i,
                    "robot": robot_i, "hand": hand_i,
                    "term_reason": None})


if args.overlay is None:
    _add_runner(args.controller, alpha=1.0,
                body_rgb=None, pen_rgb=(0.15, 0.15, 0.15))
else:
    # RL primary (blue body + blue pen) + baseline overlay (red body + red pen),
    # both at alpha=0.5. Body uses a lighter tint than the pen so geometry stays
    # readable while the color identity is unmistakable.
    _add_runner("rl", alpha=0.5,
                body_rgb=(0.45, 0.60, 0.95), pen_rgb=(0.10, 0.25, 0.95))
    _add_runner(args.overlay, alpha=0.5,
                body_rgb=(0.95, 0.55, 0.55), pen_rgb=(0.95, 0.10, 0.10))

# Aliases for legacy viz helpers — arrows / cone check key off the primary env.
primary = runners[0]
env: NSRLBatchedEnv = primary["env"]
robot = primary["robot"]
hand = primary["hand"]

builtins.base = base
builtins.robot = robot
builtins.hand = hand
builtins.runners = runners

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

_state = {"episode": 0, "step": 0, "needs_init": True}


def _sync_robots():
    for r in runners:
        q = r["env"].q[0].cpu().numpy().astype(np.float32)
        r["robot"].fk(qs=q)


def _all_done() -> bool:
    return all(bool(r["env"].done_persistent[0].item()) for r in runners)


def _start_new_episode():
    # Peek case id BEFORE reset advances the looping cursor (only meaningful
    # in --case mode; primary env's line_dist mirrors all runners').
    case_id_str = ""
    if case_ids is not None:
        case_id_str = f" [eval case {runners[0]['env'].line_dist.peek_case_id()}]"
    for r in runners:
        r["env"].reset()
        r["term_reason"] = None
    _sync_robots()
    _build_episode_viz()
    _update_tool_arrow()
    _state["episode"] += 1
    _state["step"] = 0
    _state["needs_init"] = False
    print(f"[viz] episode {_state['episode']}{case_id_str} started "
          f"(u_hat={env.line_dir[0].tolist()}, "
          f"n_target={env.n_target[0].tolist()})")


TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl", 5: "truncated"}


def tick(dt):
    if _state["needs_init"]:
        _start_new_episode()
        return

    for _ in range(args.steps_per_tick):
        for r in runners:
            env_r = r["env"]
            if bool(env_r.done_persistent[0].item()):
                continue  # frozen
            a = r["fn"](env_r)
            _, _, _, _, info = env_r.step(a, auto_reset=False)
            if bool(env_r.done_persistent[0].item()) and r["term_reason"] is None:
                r["term_reason"] = TERM_NAMES.get(
                    int(info["term_reason"][0].item()), "?")
                print(f"[viz] episode {_state['episode']} {r['label']} ended: "
                      f"step={_state['step']+1}, reason={r['term_reason']}")
        _state["step"] += 1
        if _all_done():
            break

    _sync_robots()
    _update_tool_arrow()

    if _all_done():
        # Pause one tick, then start the next episode
        _state["needs_init"] = True


tick_interval = env_cfg.dt * float(args.slowdown)
base.schedule_interval(tick, interval=tick_interval)
print(f"[viz] tick = {tick_interval:.3f}s "
      f"({args.steps_per_tick} env step(s)/tick, slowdown={args.slowdown}x → "
      f"playback {1.0/float(args.slowdown):.2f}x real-time per env step). "
      "Ctrl-C or close window to exit.")
base.run()
