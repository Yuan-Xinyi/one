"""Static (PNG) visualization for one task's inference.

Same pipeline as viz_inference.py (sample → IK refine → rollout) but renders
a 4-panel matplotlib figure instead of an interactive 3D window. Use this
over SSH / on a headless box.

Panels:
  (a) 3D scene: p0, line_dir arrow, n_target arrow, label TCPs (blue dots)
      vs refined sample TCPs (orange dots; red = IK failed). Cylinders are
      drawn as line segments of length L (label_L or rollout_L).
  (b) Joint-space PCA of labels vs refined samples (q ∈ R^7 → 2D).
  (c) Per-sample rollout L distribution + the label L_clean reference lines.
  (d) Text panel: numerical summary.

Usage:
    python -m Yuan.seed_selection.viz_inference_static \\
        --ckpt path/to/step.pt --task 17267 --cfg-w 1.5 --n-samples 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa

from Yuan.fr3_dit.training.task_cond_dit_q0 import denormalize_q
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import newton_project
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.seed_selection.batched_rollout import batched_rollout_many
from Yuan.seed_selection.label_builder import _build_R_target_strict
from Yuan.seed_selection.sample_q0 import ddim_sample_q0, load_ckpt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--data', type=Path,
                   default=Path('Yuan/seed_selection/runs/pilot_day5/pilot_20k.npz'))
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--cfg-w', type=float, default=1.5)
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--config', default='Yuan/RL_controller/config.yaml')
    p.add_argument('--target-distance-m', type=float, default=1.5)
    p.add_argument('--n-envs-rollout', type=int, default=32)
    p.add_argument('--out', type=Path, default=None,
                   help='output PNG path (default: <ckpt-dir>/viz_task<i>_cfg<w>.png)')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load task.
    z = np.load(args.data, allow_pickle=False)
    p0 = z['cs_p0'][args.task].astype(np.float32)
    line_dir = z['cs_line_dir'][args.task].astype(np.float32)
    n_target = z['cs_n_target'][args.task].astype(np.float32)
    n_labels = int(z['n_labels'][args.task])
    labels_q0 = z['labels_q0'][args.task, :n_labels].astype(np.float32)
    labels_L = z['labels_L_clean'][args.task, :n_labels].astype(np.float32)
    L_seed = float(z['L_seed'][args.task])
    status = str(z['status'][args.task])
    q0_seed = z['q0_seeds'][args.task].astype(np.float32)

    # Load model + sample.
    model, schedule, _, step = load_ckpt(args.ckpt, device, use_ema=True)
    c_vec = np.concatenate([p0, line_dir, n_target]).astype(np.float32)
    c_rep = torch.from_numpy(c_vec).to(device).unsqueeze(0).expand(args.n_samples, -1).contiguous()
    q_norm = ddim_sample_q0(model, schedule, c_rep, device=device,
                            num_steps=args.ddim_steps, cfg_w=args.cfg_w)
    raw_q = denormalize_q(q_norm).cpu().numpy().astype(np.float32)

    # Env / kin.
    with open(args.config, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = EnvConfig(**{**cfg_yaml['env'], 'n_envs': args.n_envs_rollout})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    lo_np = env.kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = env.kin.lmt_up.detach().cpu().numpy().astype(np.float32)

    # IK refine.
    R_tgt = _build_R_target_strict(n_target, line_dir)
    refined_q = np.zeros_like(raw_q)
    ik_ok = np.zeros(args.n_samples, dtype=bool)
    for s in range(args.n_samples):
        qr, ok, _ = newton_project(env.kin, raw_q[s], p0, R_tgt, lo_np, hi_np)
        refined_q[s] = qr; ik_ok[s] = bool(ok)

    # Rollout.
    cs_rep_list = [{'p0': torch.as_tensor(p0, device=device, dtype=env.kin.dtype),
                    'line_dir': torch.as_tensor(line_dir, device=device, dtype=env.kin.dtype),
                    'n_target': torch.as_tensor(n_target, device=device, dtype=env.kin.dtype)}
                   for _ in range(args.n_samples)]
    qs_t = torch.as_tensor(refined_q, device=device, dtype=env.kin.dtype)
    res = batched_rollout_many(qs_t, cs_rep_list, env=env, controller=controller,
                                target_distance_m=args.target_distance_m)
    sample_L = res['L'].astype(np.float32)
    sample_L[~ik_ok] = np.nan

    # TCPs via FK.
    @torch.no_grad()
    def fk_tcp(q_batch_np):
        qt = torch.as_tensor(q_batch_np, device=device, dtype=env.kin.dtype)
        p_t, _, _, _ = env.kin.tcp_fk_jac(qt)
        return p_t.detach().cpu().numpy().astype(np.float32)
    label_tcp   = fk_tcp(labels_q0)
    refined_tcp = fk_tcp(refined_q)
    seed_tcp    = fk_tcp(q0_seed[None, :])[0]

    # Joint-distance to nearest label (refined)
    d_to_labels = np.linalg.norm(refined_q[:, None, :] - labels_q0[None, :, :], axis=-1)
    nearest = d_to_labels.argmin(axis=1)
    min_dist = d_to_labels.min(axis=1)
    covered = (d_to_labels < 0.5).any(axis=0)
    valid_L = sample_L[np.isfinite(sample_L)]
    best_L = float(valid_L.max()) if len(valid_L) else float('nan')
    ratio = best_L / float(np.nanmax(labels_L)) if np.isfinite(best_L) else float('nan')

    # --- Plot ---
    fig = plt.figure(figsize=(16, 10))

    # (a) 3D scene
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    # Base origin
    ax.scatter([0], [0], [0], c='k', s=40, marker='s', label='robot base')
    # p0
    ax.scatter([p0[0]], [p0[1]], [p0[2]], c='red', s=80, marker='*', label='p0 (start)')
    # line_dir and n_target arrows
    L_ray = 0.3
    ax.quiver(*p0, *(line_dir * L_ray), color='blue', alpha=0.85,
              arrow_length_ratio=0.1, linewidth=2, label='line_dir')
    ax.quiver(*p0, *(n_target * L_ray * 0.8), color='green', alpha=0.85,
              arrow_length_ratio=0.1, linewidth=2, label='n_target')
    # Seed TCP
    ax.scatter(*seed_tcp, c='grey', s=80, marker='X', label=f'q0_seed TCP (L={L_seed:.3f})')
    # Label TCPs (blue dots, size by L_label)
    for k in range(n_labels):
        ax.scatter(*label_tcp[k], c='blue', s=80 + 200 * labels_L[k],
                   marker='o', edgecolors='navy', linewidths=1,
                   label=f'label {k} (L={labels_L[k]:.3f})' if k < 3 else None)
        # Cylinder as a line
        seg_end = label_tcp[k] + line_dir * (labels_L[k] * args.target_distance_m)
        ax.plot([label_tcp[k, 0], seg_end[0]],
                [label_tcp[k, 1], seg_end[1]],
                [label_tcp[k, 2], seg_end[2]], 'b-', lw=2.5, alpha=0.6)
    # Refined sample TCPs (orange / red)
    for s in range(args.n_samples):
        color = 'orange' if ik_ok[s] else 'red'
        L = float(sample_L[s])
        size = 20 + 80 * (L if np.isfinite(L) and L > 0 else 0)
        ax.scatter(*refined_tcp[s], c=color, s=size, marker='.', alpha=0.6)
        if np.isfinite(L) and L > 0:
            seg_end = refined_tcp[s] + line_dir * (L * args.target_distance_m)
            ax.plot([refined_tcp[s, 0], seg_end[0]],
                    [refined_tcp[s, 1], seg_end[1]],
                    [refined_tcp[s, 2], seg_end[2]], color=color, lw=0.8, alpha=0.4)
    # FR3 nominal reach circle (xz plane at floor)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_title(f'3D scene  •  task={args.task}  status={status}  L_seed={L_seed:.3f}')
    ax.legend(fontsize=7, loc='upper left')
    # Equal aspect
    all_pts = np.vstack([label_tcp, refined_tcp, p0[None], seed_tcp[None]])
    rng_max = np.max(all_pts.max(axis=0) - all_pts.min(axis=0))
    ctr = all_pts.mean(axis=0)
    half = max(rng_max, 0.3) * 0.6
    ax.set_xlim(ctr[0]-half, ctr[0]+half); ax.set_ylim(ctr[1]-half, ctr[1]+half); ax.set_zlim(ctr[2]-half, ctr[2]+half)

    # (b) Joint-space PCA
    ax = fig.add_subplot(2, 2, 2)
    all_q = np.vstack([labels_q0, refined_q, q0_seed[None]])
    q_mean = all_q.mean(axis=0)
    q_c = all_q - q_mean
    u, s, vh = np.linalg.svd(q_c, full_matrices=False)
    proj = q_c @ vh[:2].T   # (N, 2)
    n_lab = labels_q0.shape[0]
    n_smp = refined_q.shape[0]
    lab_p = proj[:n_lab]
    smp_p = proj[n_lab:n_lab+n_smp]
    seed_p = proj[-1]
    ax.scatter(seed_p[0], seed_p[1], c='grey', s=140, marker='X', edgecolors='k', label='q0_seed', zorder=4)
    ax.scatter(lab_p[:, 0], lab_p[:, 1], c='blue', s=200, marker='o', edgecolors='navy',
               linewidths=1.5, label=f'labels (n={n_lab})', zorder=3)
    ok_smp = ik_ok
    if ok_smp.any():
        ax.scatter(smp_p[ok_smp, 0], smp_p[ok_smp, 1], c='orange', s=40, alpha=0.7,
                   label=f'samples IK ok ({int(ok_smp.sum())})', zorder=2)
    if (~ok_smp).any():
        ax.scatter(smp_p[~ok_smp, 0], smp_p[~ok_smp, 1], c='red', s=40, marker='x',
                   label=f'samples IK fail ({int((~ok_smp).sum())})', zorder=2)
    ax.set_xlabel(f'PC1 ({100*s[0]**2/np.sum(s**2):.0f}% var)')
    ax.set_ylabel(f'PC2 ({100*s[1]**2/np.sum(s**2):.0f}% var)')
    ax.set_title('Joint-space PCA: labels vs refined samples')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Per-sample rollout L vs labels
    ax = fig.add_subplot(2, 2, 3)
    ax.hist(valid_L, bins=20, color='orange', alpha=0.7, edgecolor='k',
            label=f'sample rollout L (n={len(valid_L)} valid)')
    for k, lL in enumerate(labels_L):
        ax.axvline(lL, color='blue', ls='--', lw=2, alpha=0.8,
                   label=f'label {k} L={lL:.3f}' if k < n_labels else None)
    ax.axvline(L_seed, color='grey', ls=':', lw=2, label=f'L_seed={L_seed:.3f}')
    if np.isfinite(best_L):
        ax.axvline(best_L, color='red', ls='-', lw=2, alpha=0.7,
                   label=f'best_of_N={best_L:.3f}')
    ax.set_xlabel('rollout L_clean')
    ax.set_ylabel('# samples')
    ax.set_title(f'Sample L distribution  •  best_L/max(label_L) = {ratio:.3f}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) Text summary
    ax = fig.add_subplot(2, 2, 4)
    ax.axis('off')
    r_3d = float(np.linalg.norm(p0))
    info = [
        f"Task #{args.task}   raw NPZ index: {args.task}",
        f"Status:        {status}",
        f"L_seed:        {L_seed:.3f}  (×1.5m = {L_seed*1.5:.3f}m)",
        f"max label L:   {float(np.nanmax(labels_L)):.3f}",
        f"best_of_{args.n_samples} L: {best_L:.3f}",
        f"recover_ratio: {ratio:.3f}",
        f"deployment_gain: {best_L/L_seed:.3f}× (best_L / L_seed)",
        '',
        f"c geometry:",
        f"  p0           = ({p0[0]:+.2f}, {p0[1]:+.2f}, {p0[2]:+.2f})   ||p0||={r_3d:.3f}m",
        f"  line_dir     = ({line_dir[0]:+.2f}, {line_dir[1]:+.2f}, {line_dir[2]:+.2f})",
        f"  n_target     = ({n_target[0]:+.2f}, {n_target[1]:+.2f}, {n_target[2]:+.2f})",
        '',
        f"Diagnostics:",
        f"  n_branches            : {int(z['n_branches_per_task'][args.task])}",
        f"  cone_ik n_succ/attempt: {int(z['cone_ik_n_successes'][args.task])}/{int(z['cone_ik_n_attempts'][args.task])}",
        f"  q0_seed term_reason   : {str(z['q0_seed_term_reason'][args.task])}",
        f"  q0_seed max_q_norm    : {float(z['q0_seed_max_q_norm'][args.task]):.3f}",
        f"  labels term_reason    : {z['labels_term_reason'][args.task].tolist()}",
        f"  labels max_q_norm     : {z['labels_max_q_norm'][args.task].tolist()}",
        '',
        f"Inference:",
        f"  ckpt step            : {step}",
        f"  cfg_w                : {args.cfg_w}",
        f"  n_samples            : {args.n_samples}",
        f"  IK convergence       : {int(ik_ok.sum())}/{args.n_samples}",
        f"  branches covered     : {int(covered.sum())}/{n_labels} (joint-dist on refined)",
        f"  sample L >= 0.20     : {int((valid_L >= 0.20).sum())}/{len(valid_L)}",
    ]
    ax.text(0.0, 1.0, '\n'.join(info), family='monospace', fontsize=9,
            verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout()
    out = args.out or (args.ckpt.parent /
                       f'viz_task{args.task}_cfg{args.cfg_w}_step{step}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
