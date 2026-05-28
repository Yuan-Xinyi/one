"""Visualize trained-model q0 samples vs. ground-truth SMM labels for one task.

Operational pipeline matches eval_rollout.py:
  1. Sample N q0_dp from the diffusion model (with optional CFG).
  2. For each q0_dp, run `newton_project` to refine it onto the task's exact
     (p0, R_target_strict). Refined q0 has TCP on p0 with z = n_target.
  3. Rollout from the refined q0 to get the *actual* operational L_clean.

Scene contents:
  - Task scene (plane patch, line_dir arrow, n_target arrow).
  - GT labels: thick BLUE ghost robots + thick BLUE cylinders (length = L_clean
    from the NPZ).
  - Refined samples: ORANGE ghost robots + thin ORANGE cylinders (length =
    rollout L from the refined q0). RED if Newton IK failed to converge.
  - Optional (--show-raw): faint GREY shadows of the raw pre-IK samples.

Usage:
    python -m Yuan.seed_selection.viz_inference --ckpt path/to/step_N.pt \\
        --task 0 --n-samples 16 --cfg-w 1.5
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
from pathlib import Path

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

from Yuan.fr3_dit.training.task_cond_dit_q0 import denormalize_q
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import newton_project
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.seed_selection.batched_rollout import batched_rollout_many
from Yuan.seed_selection.label_builder import _build_R_target_strict
from Yuan.seed_selection.sample_q0 import ddim_sample_q0, load_ckpt


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=Path, required=True)
parser.add_argument("--data", type=Path,
                    default=Path("Yuan/seed_selection/runs/pilot_day5/pilot_20k.npz"))
parser.add_argument("--task", type=int, default=0,
                    help="task index in the dataset")
parser.add_argument("--n-samples", type=int, default=16)
parser.add_argument("--cfg-w", type=float, default=1.0,
                    help="classifier-free guidance weight (1.0 = no guidance)")
parser.add_argument("--ddim-steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--config", default="Yuan/RL_controller/config.yaml",
                    help="env config (for the rollout used to compute sample L)")
parser.add_argument("--no-rollout", action="store_true",
                    help="skip the rollout of samples (faster; no orange cylinders)")
parser.add_argument("--n-envs-rollout", type=int, default=32)
parser.add_argument("--show-raw", action="store_true",
                    help="also draw the RAW (un-refined) sample as a faint grey robot. "
                         "Default off: only the operationally-meaningful refined pose is drawn.")
parser.add_argument("--alpha", type=float, default=0.35,
                    help="ghost robot transparency")
parser.add_argument("--target-distance-m", type=float, default=1.5)
parser.add_argument("--ray-len", type=float, default=1.5)
parser.add_argument("--plane-size", type=float, default=0.6)
parser.add_argument("--plane-square-m", type=float, default=0.0,
                    help="if > 0: draw a SQUARE plane patch of this size centered on p0 "
                         "(overrides --plane-size and --ray-len for the patch). Makes the "
                         "infinite-plane collision constraint visible regardless of line_dir.")
parser.add_argument("--obstacle-slab-m", type=float, default=0.0,
                    help="if > 0: also draw a translucent reddish slab on the +n_target side "
                         "(the obstacle half-space). Slab thickness in m; centered along normal.")
parser.add_argument("--use-model", action="store_true",
                    help="use raw model weights instead of EMA (default: EMA)")
args = parser.parse_args()


# Reproducibility for the sample draw.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load dataset entry.
z = np.load(args.data, allow_pickle=False)
N = int(z["L_seed"].shape[0])
if not (0 <= args.task < N):
    raise SystemExit(f"--task {args.task} out of range [0, {N})")
p0       = z["cs_p0"][args.task].astype(np.float32)
line_dir = z["cs_line_dir"][args.task].astype(np.float32)
n_target = z["cs_n_target"][args.task].astype(np.float32)
n_labels = int(z["n_labels"][args.task])
labels_q0 = z["labels_q0"][args.task, :n_labels].astype(np.float32)
labels_L  = z["labels_L_clean"][args.task, :n_labels].astype(np.float32)
L_seed    = float(z["L_seed"][args.task])
status    = str(z["status"][args.task])

print(f"[viz] task={args.task}/{N}  status={status}  L_seed={L_seed:.3f}  "
      f"n_labels={n_labels}")
print(f"  p0       = {p0.tolist()}")
print(f"  line_dir = {line_dir.tolist()}")
print(f"  n_target = {n_target.tolist()}")
print(f"  labels_L = {labels_L.tolist()}")

# Load model and sample.
print(f"[viz] loading ckpt={args.ckpt}  cfg_w={args.cfg_w}  "
      f"weights={'model' if args.use_model else 'ema'}")
model, schedule, model_cfg, step = load_ckpt(args.ckpt, device, use_ema=not args.use_model)
print(f"[viz] ckpt step={step}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

c_vec = np.concatenate([p0, line_dir, n_target]).astype(np.float32)
c_rep = torch.from_numpy(c_vec).to(device).unsqueeze(0).expand(args.n_samples, -1).contiguous()
q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                        num_steps=args.ddim_steps, cfg_w=args.cfg_w)
raw_sample_q0 = denormalize_q(q_norm).cpu().numpy().astype(np.float32)  # (M, 7)

# Build env (needed for kin → IK; also reused for rollout below).
with open(args.config, "r") as f:
    cfg_yaml = yaml.safe_load(f)
env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": args.n_envs_rollout})
env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
controller = ClassicalNullspaceController(env.kin)
lo_np = env.kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
hi_np = env.kin.lmt_up.detach().cpu().numpy().astype(np.float32)

# Newton-refine each sample onto (p0, R_target_strict).
R_tgt = _build_R_target_strict(n_target, line_dir)
sample_q0 = np.zeros_like(raw_sample_q0)
ik_ok = np.zeros(args.n_samples, dtype=bool)
ik_err = np.zeros(args.n_samples, dtype=np.float32)
print(f"\n[viz] running Newton IK on {args.n_samples} samples to refine onto exact (p0, R_target)...")
for s in range(args.n_samples):
    q_ref, ok, err = newton_project(env.kin, raw_sample_q0[s], p0, R_tgt, lo_np, hi_np)
    sample_q0[s] = q_ref
    ik_ok[s] = bool(ok); ik_err[s] = float(err)
print(f"[viz] IK converged: {int(ik_ok.sum())}/{args.n_samples}")

# Joint-distance to labels (use REFINED — apples-to-apples with labels).
d_to_labels = np.linalg.norm(sample_q0[:, None, :] - labels_q0[None, :, :], axis=-1)  # (M, n)
nearest = d_to_labels.argmin(axis=1)
min_dist = d_to_labels.min(axis=1)
print(f"\n[viz] refined-sample → nearest-label distance (rad):")
for s in range(args.n_samples):
    ok_tag = "✓" if ik_ok[s] else "✗"
    print(f"  [{s:2d}] {ok_tag} d={min_dist[s]:.3f}  → label {nearest[s]} (L_label={labels_L[nearest[s]]:.3f})")
print(f"[viz] mean min_dist = {min_dist.mean():.3f}, fraction < 0.5 rad = "
      f"{100*(min_dist < 0.5).mean():.1f}%")
print(f"[viz] per-label coverage (any sample within 0.5 rad):")
for k in range(n_labels):
    hits = int((d_to_labels[:, k] < 0.5).sum())
    print(f"  label {k} (L={labels_L[k]:.3f}): {hits}/{args.n_samples} samples")

# Rollout from REFINED q0 (starts TCP exactly at p0). This is the operational L.
sample_L = np.full(args.n_samples, np.nan, dtype=np.float32)
if not args.no_rollout:
    print(f"\n[viz] rolling out {args.n_samples} refined samples to get operational L ...")
    cs_rep_list = [{"p0": torch.as_tensor(p0, device=device, dtype=env.kin.dtype),
                    "line_dir": torch.as_tensor(line_dir, device=device, dtype=env.kin.dtype),
                    "n_target": torch.as_tensor(n_target, device=device, dtype=env.kin.dtype)}
                   for _ in range(args.n_samples)]
    qs_t = torch.as_tensor(sample_q0, device=device, dtype=env.kin.dtype)
    res = batched_rollout_many(qs_t, cs_rep_list, env=env, controller=controller,
                                target_distance_m=args.target_distance_m)
    sample_L = res["L"].astype(np.float32)
    sample_L[~ik_ok] = np.nan  # mark IK-failed samples
    valid_L = sample_L[np.isfinite(sample_L)]
    if len(valid_L):
        print(f"  sample L_clean (IK-ok only): range [{valid_L.min():.3f}, {valid_L.max():.3f}], "
              f"mean {valid_L.mean():.3f}, best {valid_L.max():.3f}")
    print(f"  comparison: L_seed={L_seed:.3f}, max(labels_L)={labels_L.max():.3f}, "
          f"best_sample_L={(valid_L.max() if len(valid_L) else float('nan')):.3f}")


# Scene -----------------------------------------------------------------------
base = ovw.World(cam_pos=(1.5, 1.2, 1.2),
                 cam_lookat_pos=(0.0, 0.0, 0.4),
                 toggle_auto_cam_orbit=False)
ossop.frame().attach_to(base.scene)

# Task plane patch (normal = n_target, long edge along line_dir).
y_axis = np.cross(n_target, line_dir)
y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
plane_rotmat = np.stack([line_dir, y_axis, n_target], axis=1).astype(np.float32)
if args.plane_square_m > 0.0:
    # Square patch starting at p0 and extending only in +line_dir direction.
    # Half-extents: forward = plane_square_m/2 along line_dir (so its TOTAL length
    # equals plane_square_m), and ±plane_square_m/2 along the perpendicular y_axis.
    half = float(args.plane_square_m) * 0.5
    plane_center = p0 + line_dir * half        # patch center at p0 + half * line_dir
    plane_he = (half, half, 5e-4)              # extent (half, half, ~0) in (line_dir, y_axis, n_target)
else:
    plane_center = p0 + line_dir * (args.ray_len * 0.5)
    plane_he = (args.ray_len * 0.5, args.plane_size * 0.5, 5e-4)
plane = ossop.box(
    pos=plane_center,
    half_extents=plane_he,
    rotmat=plane_rotmat,
    rgb=(0.85, 0.85, 0.55), alpha=0.20,
)
plane.attach_to(base.scene)

# Optional: draw the +n_target obstacle half-space as a translucent reddish slab.
# A sphere whose center pierces into this slab is what the plane-collision filter flags.
if args.obstacle_slab_m > 0.0:
    thickness = float(args.obstacle_slab_m)
    # Slab centered at p0 + (thickness/2) * n_target  → spans p0 to p0 + thickness * n_target.
    obs_center = p0 + n_target * (thickness * 0.5)
    if args.plane_square_m > 0.0:
        half = float(args.plane_square_m) * 0.5
        obs_he = (half, half, thickness * 0.5)
    else:
        obs_he = (args.ray_len * 0.5, args.plane_size * 0.5, thickness * 0.5)
    obs = ossop.box(
        pos=obs_center,
        half_extents=obs_he,
        rotmat=plane_rotmat,
        rgb=(0.80, 0.20, 0.20), alpha=0.10,
    )
    obs.attach_to(base.scene)

# line_dir dashed ray (blue), n_target arrow (green).
ossop.dashed_cylinder(
    spos=p0, epos=p0 + line_dir * args.ray_len,
    radius=0.003, rgb=(0.2, 0.4, 1.0), alpha=0.9,
).attach_to(base.scene)
ossop.arrow(
    spos=p0, epos=p0 + line_dir * 0.30, rgb=(0.2, 0.4, 1.0),
).attach_to(base.scene)
ossop.arrow(
    spos=p0, epos=p0 + n_target * 0.30, rgb=(0.2, 0.9, 0.2),
).attach_to(base.scene)


# Palette
BLUE_BODY    = (0.10, 0.40, 1.00)     # GT labels
BLUE_PEN     = (0.00, 0.20, 1.00)
ORANGE_BODY  = (1.00, 0.55, 0.10)     # refined samples, IK-ok
ORANGE_PEN   = (0.80, 0.35, 0.00)
RED_BODY     = (0.85, 0.15, 0.15)     # IK-failed samples (warning)
RED_PEN      = (0.55, 0.05, 0.05)
GREY_BODY    = (0.50, 0.50, 0.50)     # raw (pre-IK) sample shadow
GREY_PEN     = (0.30, 0.30, 0.30)

label_alpha  = min(args.alpha + 0.20, 1.00)
sample_alpha = max(args.alpha * 0.70, 0.20)
raw_alpha    = max(args.alpha * 0.40, 0.12)


# Optionally draw RAW (pre-IK) samples as faint grey shadows.
n_raw_drawn = 0
if args.show_raw:
    for s in range(args.n_samples):
        arm, _ = make_fr3_with_pen(use_pen_tcp=True)
        arm.attach_to(base.scene)
        attach_pen_visual(arm, rgb=GREY_PEN, alpha=raw_alpha)
        arm.rgb = GREY_BODY; arm.alpha = raw_alpha
        arm.fk(qs=raw_sample_q0[s])
        n_raw_drawn += 1

# Draw REFINED samples (operational pose; TCP lands on p0).
n_sample_drawn = 0
n_failed_drawn = 0
for s in range(args.n_samples):
    arm, _ = make_fr3_with_pen(use_pen_tcp=True)
    arm.attach_to(base.scene)
    if ik_ok[s]:
        body_rgb, pen_rgb = ORANGE_BODY, ORANGE_PEN
    else:
        body_rgb, pen_rgb = RED_BODY, RED_PEN
        n_failed_drawn += 1
    attach_pen_visual(arm, rgb=pen_rgb, alpha=sample_alpha)
    arm.rgb = body_rgb; arm.alpha = sample_alpha
    arm.fk(qs=sample_q0[s])
    p_tip = arm.gl_tcp_tf[:3, 3].copy().astype(np.float32)
    L = float(sample_L[s])
    if np.isfinite(L) and L > 0.0:
        ossop.cylinder(
            spos=p_tip,
            epos=p_tip + line_dir * (L * args.target_distance_m),
            radius=0.004, rgb=body_rgb, alpha=min(sample_alpha + 0.3, 1.0),
        ).attach_to(base.scene)
    n_sample_drawn += 1


# Draw LABELS on top (always visible).
for k in range(n_labels):
    arm, _ = make_fr3_with_pen(use_pen_tcp=True)
    arm.attach_to(base.scene)
    attach_pen_visual(arm, rgb=BLUE_PEN, alpha=label_alpha)
    arm.rgb = BLUE_BODY
    arm.alpha = label_alpha
    arm.fk(qs=labels_q0[k])
    L = float(labels_L[k])
    if np.isfinite(L) and L > 0.0:
        p_tip = arm.gl_tcp_tf[:3, 3].copy().astype(np.float32)
        ossop.cylinder(
            spos=p_tip,
            epos=p_tip + line_dir * (L * args.target_distance_m),
            radius=0.009, rgb=BLUE_BODY, alpha=min(label_alpha + 0.2, 1.0),
        ).attach_to(base.scene)


print(f"\n[viz] scene: {n_labels} BLUE labels + "
      f"{n_sample_drawn-n_failed_drawn} ORANGE (IK-ok) + {n_failed_drawn} RED (IK-failed) samples"
      + (f" + {n_raw_drawn} GREY raw shadows" if args.show_raw else ""))
print(f"  step={step}  cfg_w={args.cfg_w}  task={args.task}  status={status}")
print(f"[viz] orbit/zoom; close window or Ctrl-C to exit.")

builtins.base = base
base.run()
