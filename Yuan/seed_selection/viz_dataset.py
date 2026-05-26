"""Simple viewer for one seed_selection task.

Two modes:
  * NPZ mode  (--npz given): pick task `--task` from a built dataset.
  * RAW mode  (--npz omitted): sample a fresh task with --seed and run the
                                full label pipeline (LineDistribution +
                                build_labels_for_one_task) on the fly.

In both modes the scene shows:
    * the task plane patch (along line_dir, normal = n_target)
    * the n_target arrow (green) and line_dir arrow + dashed ray (blue)
    * one ghost FR3 per SMM label q0, overlaid transparently:
        - BLUE  transparent  → labels with L_clean >= --L-threshold
                               ("walks far" — what the dataset wants to keep)
        - GREY  transparent  → labels with L_clean <  --L-threshold
    * a solid cylinder per label of length L_clean × --target-distance-m
      along line_dir, drawn from that label's TCP.

Usage:
    # NPZ mode
    python -m Yuan.seed_selection.viz_dataset \\
        --npz Yuan/seed_selection/runs/pilot_day5/pilot_v2.npz --task 11

    # RAW mode (label pipeline runs live; ~10-60s/task on cuda)
    python -m Yuan.seed_selection.viz_dataset --seed 42
"""
from __future__ import annotations

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

import numpy as np

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)


parser = argparse.ArgumentParser()
parser.add_argument("--npz", default=None,
                    help="dataset NPZ (from dataset_builder). Omit → RAW mode.")
parser.add_argument("--task", type=int, default=0,
                    help="[NPZ mode] task index in the NPZ to visualize")
parser.add_argument("--seed", type=int, default=0,
                    help="[RAW mode] seed for fresh task sampling")
parser.add_argument("--config", default="Yuan/RL_controller/config.yaml",
                    help="[RAW mode] env config (for dt/v/a_max/tcp_offset/...)")
parser.add_argument("--raw-n-pool", type=int, default=64,
                    help="[RAW mode] LineDistribution pool size (just need 1 valid task)")
parser.add_argument("--raw-n-envs", type=int, default=32,
                    help="[RAW mode] n_envs for the batched rollout env")
parser.add_argument("--L-threshold", type=float, default=0.30,
                    help="L_clean >= this → blue (kept). Below → grey.")
parser.add_argument("--ray-len", type=float, default=1.5,
                    help="line ray length (m)")
parser.add_argument("--plane-size", type=float, default=0.6,
                    help="task plane patch width across line_dir (m)")
parser.add_argument("--alpha", type=float, default=0.35,
                    help="ghost robot transparency")
parser.add_argument("--target-distance-m", type=float, default=1.5,
                    help="meters used to denormalize L (rollout.DEFAULT_TARGET_DISTANCE_M)")
parser.add_argument("--other-robots", action="store_true",
                    help="[RAW mode] also draw robot ghosts for non-label candidates "
                         "(default: just the L cylinders, to keep the scene readable)")
args = parser.parse_args()


def _load_from_npz(npz_path: str, task_idx: int):
    z = np.load(npz_path, allow_pickle=False)
    N = int(z["L_seed"].shape[0])
    if not (0 <= task_idx < N):
        raise SystemExit(f"task index {task_idx} out of range [0, {N})")
    n_labels = int(z["n_labels"][task_idx])
    return {
        "p0":        z["cs_p0"][task_idx].astype(np.float32),
        "u_hat":     z["cs_line_dir"][task_idx].astype(np.float32),
        "n_target":  z["cs_n_target"][task_idx].astype(np.float32),
        "n_labels":  n_labels,
        "labels_q0": z["labels_q0"][task_idx, :n_labels].astype(np.float32),
        "labels_L":  z["labels_L_clean"][task_idx, :n_labels].astype(np.float32),
        "status":    str(z["status"][task_idx]),
        "L_seed":    float(z["L_seed"][task_idx]),
        # NPZ doesn't persist non-label candidates → empty.
        "other_q0":    np.zeros((0, 7), dtype=np.float32),
        "other_L":     np.zeros((0,), dtype=np.float32),
        "other_p_tip": np.zeros((0, 3), dtype=np.float32),
        "source":      f"NPZ {npz_path} task {task_idx}/{N}",
    }


def _sample_raw(seed: int, config_path: str, n_pool: int, n_envs: int):
    """Sample a fresh task and run the label pipeline live."""
    import torch
    import yaml
    from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
    from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
    from Yuan.RL_controller.env.line_distribution import LineDistribution
    from Yuan.seed_selection.label_builder import build_labels_for_one_task

    with open(config_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    line_cfg = cfg_yaml["line_distribution"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[viz_dataset:raw] device={device}  seed={seed}", flush=True)

    env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": n_envs})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)

    print(f"[viz_dataset:raw] building LineDistribution(n_pool={n_pool}, "
          f"noise={line_cfg['n_target_noise_deg']}°)", flush=True)
    line_dist = LineDistribution(
        kin=env.kin, collision=env.collision,
        n_pool=n_pool,
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=int(seed),
    )
    gen = torch.Generator(device=device).manual_seed(int(seed))
    spec = line_dist.sample(1, generator=gen)
    q0_seed = spec["q0"][0]
    line_dir = spec["line_dir"][0]
    n_target = spec["n_target"][0]
    p0, _, _, _ = env.kin.tcp_fk_jac(q0_seed.unsqueeze(0))
    p0 = p0[0]
    c = {"p0": p0, "line_dir": line_dir, "n_target": n_target}

    print(f"[viz_dataset:raw] running build_labels_for_one_task ...", flush=True)
    # Match Day-5 pilot defaults so live output is consistent with the dataset.
    out = build_labels_for_one_task(
        c, q0_seed,
        kin=env.kin, collision=env.collision,
        env=env, controller=controller,
        cone_angle_deg=5.0, n_orientations=10, n_ik_restarts=5,
        sample_per_branch=5, k=3, K_prime=6,
        tau_robust=0.0, n_perturb=4,
        perturb_d_deg=3.0, perturb_n_deg=3.0, perturb_p0_mm=8.0,
        L_min_abs=0.10, L_min_acceptable=0.20,
        target_distance_m=1.5,
        ik_dedup_rad=0.08, smm_dedup_rad=0.08,
        seed=int(seed),
        return_all_candidates=True,
    )
    nl = int(out["n_labels"])
    labels_q0_np = out["labels_q0"].detach().cpu().numpy().astype(np.float32)[:nl]
    all_q0_np = out["all_candidates_q0"].numpy().astype(np.float32)
    all_L_np = np.asarray(out["all_candidates_L"], dtype=np.float32)

    # Mark which entries in all_candidates_q0 are also in the top-k labels —
    # labels are picked verbatim from the scored set, so exact equality holds.
    is_label = np.zeros(all_q0_np.shape[0], dtype=bool)
    for lab in labels_q0_np:
        is_label |= np.all(all_q0_np == lab[None, :], axis=1)
    other_q0_np = all_q0_np[~is_label]
    other_L_np  = all_L_np[~is_label]

    # FK in a single batch to get TCP positions for non-label candidates;
    # needed for drawing their L-cylinders even when robot ghosts are off.
    other_p_tip_np = np.zeros((0, 3), dtype=np.float32)
    if other_q0_np.shape[0] > 0:
        q_t = torch.as_tensor(other_q0_np, device=env.device, dtype=env.kin.dtype)
        p_t, _, _, _ = env.kin.tcp_fk_jac(q_t)
        other_p_tip_np = p_t.detach().cpu().numpy().astype(np.float32)

    return {
        "p0":        p0.detach().cpu().numpy().astype(np.float32),
        "u_hat":     line_dir.detach().cpu().numpy().astype(np.float32),
        "n_target":  n_target.detach().cpu().numpy().astype(np.float32),
        "n_labels":  nl,
        "labels_q0": labels_q0_np,
        "labels_L":  np.asarray(out["labels_L_clean"], dtype=np.float32)[:nl],
        "status":    str(out["status"]),
        "L_seed":    float(out["L_seed"]),
        # Non-label scored candidates (for raw-mode overlay).
        "other_q0":    other_q0_np,
        "other_L":     other_L_np,
        "other_p_tip": other_p_tip_np,
        "source":      f"RAW seed={seed} (live label pipeline)",
    }


task = _load_from_npz(args.npz, args.task) if args.npz else \
       _sample_raw(args.seed, args.config, args.raw_n_pool, args.raw_n_envs)

p0       = task["p0"]
u_hat    = task["u_hat"]
n_target = task["n_target"]
n_labels = task["n_labels"]
labels_q0 = task["labels_q0"]
labels_L  = task["labels_L"]
status   = task["status"]
L_seed   = task["L_seed"]

other_q0    = task["other_q0"]
other_L     = task["other_L"]
other_p_tip = task["other_p_tip"]

print(f"[viz_dataset] {task['source']}")
print(f"  status={status}  L_seed={L_seed:.3f}")
print(f"  p0       = {p0.tolist()}")
print(f"  line_dir = {u_hat.tolist()}")
print(f"  n_target = {n_target.tolist()}")
print(f"  {n_labels} SMM labels (L_clean):")
for k in range(n_labels):
    tag = "BLUE " if labels_L[k] >= args.L_threshold else "grey "
    print(f"    [{k}] {tag} L_clean={labels_L[k]:.3f}  q0={labels_q0[k].tolist()}")
if other_q0.shape[0] > 0:
    print(f"  {other_q0.shape[0]} other scored candidates (TAN, transparent):")
    for k in range(other_q0.shape[0]):
        print(f"    ({k}) L_clean={other_L[k]:.3f}")


# Scene -----------------------------------------------------------------------

base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)


# Task plane patch (rectangular, long edge along line_dir, normal = n_target).
y_axis = np.cross(n_target, u_hat)
y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
plane_rotmat = np.stack([u_hat, y_axis, n_target], axis=1).astype(np.float32)
plane_center = p0 + u_hat * (args.ray_len * 0.5)
plane = ossop.box(
    pos=plane_center,
    half_extents=(args.ray_len * 0.5, args.plane_size * 0.5, 5e-4),
    rotmat=plane_rotmat,
    rgb=(0.85, 0.85, 0.55), alpha=0.20,
)
plane.attach_to(base.scene)

# line_dir dashed ray + arrow (blue), n_target arrow (green).
ray = ossop.dashed_cylinder(
    spos=p0, epos=p0 + u_hat * args.ray_len,
    radius=0.003, rgb=(0.2, 0.4, 1.0), alpha=0.9,
)
ray.attach_to(base.scene)
u_arrow = ossop.arrow(
    spos=p0, epos=p0 + u_hat * 0.30, rgb=(0.2, 0.4, 1.0))
u_arrow.attach_to(base.scene)
n_arrow = ossop.arrow(
    spos=p0, epos=p0 + n_target * 0.30, rgb=(0.2, 0.9, 0.2))
n_arrow.attach_to(base.scene)


# Colour palette.
#   - Labels above threshold → saturated BLUE
#   - Labels below threshold → mid GREY
#   - Other scored candidates → TAN, more transparent, thinner length-cylinder
BLUE_BODY = (0.10, 0.40, 1.00)   # saturated so it pops against TAN cluster
BLUE_PEN  = (0.00, 0.20, 1.00)
GREY_BODY = (0.55, 0.55, 0.55)
GREY_PEN  = (0.30, 0.30, 0.30)
TAN_BODY  = (0.85, 0.75, 0.45)
TAN_PEN   = (0.55, 0.45, 0.20)

other_alpha = max(args.alpha * 0.45, 0.10)
label_alpha = min(args.alpha + 0.20, 1.00)   # bump so labels read against TAN behind


# Non-label scored candidates FIRST (drawn underneath; raw mode only).
# Default: only the L-cylinder (thin TAN line). The cluster of robot ghosts
# at near-identical EE poses creates a thick haze that obscures the BLUE
# labels in the middle. Pass --other-robots to bring back the ghosts.
n_other_robots = 0
for k in range(other_q0.shape[0]):
    if args.other_robots:
        arm, _hand = make_fr3_with_pen(use_pen_tcp=True)
        arm.attach_to(base.scene)
        attach_pen_visual(arm, rgb=TAN_PEN, alpha=other_alpha)
        arm.rgb = TAN_BODY
        arm.alpha = other_alpha
        arm.fk(qs=other_q0[k])
        p_tip = arm.gl_tcp_tf[:3, 3].copy().astype(np.float32)
        n_other_robots += 1
    else:
        p_tip = other_p_tip[k].astype(np.float32)
    L = float(other_L[k])
    if np.isfinite(L) and L > 0.0:
        seg = ossop.cylinder(
            spos=p_tip,
            epos=p_tip + u_hat * (L * args.target_distance_m),
            radius=0.004, rgb=TAN_BODY, alpha=min(other_alpha + 0.3, 1.0),
        )
        seg.attach_to(base.scene)
if other_q0.shape[0] > 0:
    if args.other_robots:
        print(f"[viz_dataset] {n_other_robots} TAN ghost robots + L-cylinders "
              f"(other scored candidates, alpha={other_alpha:.2f})")
    else:
        print(f"[viz_dataset] {other_q0.shape[0]} TAN L-cylinders only "
              f"(other scored candidates; pass --other-robots to also draw ghosts)")


# Label ghosts LAST (drawn on top so they always read).
n_blue = 0
n_grey = 0
for k in range(n_labels):
    is_kept = labels_L[k] >= args.L_threshold
    body_rgb = BLUE_BODY if is_kept else GREY_BODY
    pen_rgb  = BLUE_PEN  if is_kept else GREY_PEN
    arm, _hand = make_fr3_with_pen(use_pen_tcp=True)
    arm.attach_to(base.scene)
    attach_pen_visual(arm, rgb=pen_rgb, alpha=label_alpha)
    arm.rgb = body_rgb
    arm.alpha = label_alpha
    arm.fk(qs=labels_q0[k])
    L = float(labels_L[k])
    if np.isfinite(L) and L > 0.0:
        p_tip = arm.gl_tcp_tf[:3, 3].copy().astype(np.float32)
        seg = ossop.cylinder(
            spos=p_tip,
            epos=p_tip + u_hat * (L * args.target_distance_m),
            radius=0.008, rgb=body_rgb, alpha=min(label_alpha + 0.2, 1.0),
        )
        seg.attach_to(base.scene)
    n_blue += int(is_kept)
    n_grey += int(not is_kept)

print(f"[viz_dataset] + {n_blue} blue (kept) + {n_grey} grey label ghosts "
      f"(threshold L_clean >= {args.L_threshold}, alpha={label_alpha:.2f})")

print("[viz_dataset] orbit / zoom; close window or Ctrl-C to exit.")

builtins.base = base
base.run()
