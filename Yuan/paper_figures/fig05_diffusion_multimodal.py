"""Figure 5: multi-modal diffusion samples for one task, rendered as a
transparent overlay of N independent diffusion samples in the One viewer.

For the selected task we draw N samples from the trained diffusion model
(conditioned on c = (p0, d, n)), Newton-project each onto the strict
constraint manifold (p0 exactly, z aligned with n), and spawn one ghost
FR3 arm per valid sample with the chosen colour and transparency.

The point is to show that DP samples spread across distinct branches of
the self-motion manifold rather than collapsing to a single mode.

Usage:
    python -m Yuan.paper_figures.fig05_diffusion_multimodal
    python -m Yuan.paper_figures.fig05_diffusion_multimodal --task 1872 \\
        --n-samples 32 --alpha 0.25
"""
from __future__ import annotations

# Conda lib bootstrap (so the One viewer can find shared libraries).
import os, sys
_conda_lib = os.path.join(sys.prefix, 'lib')
if _conda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    new_env = dict(os.environ)
    new_env['LD_LIBRARY_PATH'] = _conda_lib + ':' + new_env.get('LD_LIBRARY_PATH', '')
    if __spec__ is not None and __spec__.name != '__main__':
        argv = [sys.executable, '-m', __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import rollout_seeds_batched
from Yuan.seed_selection.diffusion import (
    ddim_sample_q0, denormalize_q, load_ckpt,
)
from Yuan.seed_selection.smm import _build_R_target_strict, newton_project


COLOR_DP = (142/255, 207/255, 201/255)        # #8ECFC9  (paper's DP colour)

DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'

task_lists = [5721,2676,1777,7572,7012]
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=2676, help='eval-set task index')
    p.add_argument('--n-samples', type=int, default=32,
                   help='Number of diffusion samples to draw (pool size).')
    p.add_argument('--n-representatives', type=int, default=4,
                   help='Number of representative (maximally-spread) samples '
                        'to actually render. Selected from the IK-valid pool '
                        'via farthest-point sampling. Use 0 to render all.')
    p.add_argument('--alpha', type=float, default=0.30,
                   help='Per-ghost transparency in [0, 1].')
    p.add_argument('--color', type=str, default=None,
                   help='Optional hex colour for the ghosts (e.g. #8ECFC9). '
                        'Default is the paper DP teal.')
    p.add_argument('--ddim-steps', type=int, default=50)
    p.add_argument('--cfg-w', type=float, default=1.5)
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--target-distance-m', type=float, default=1.5)
    p.add_argument('--show-raw', action='store_true',
                   help='Also overlay the raw (pre-Newton) samples as faint '
                        'grey shadows.')
    p.add_argument('--stick-spacing', type=float, default=0.025,
                   help='Lateral spacing (m) between the parallel '
                        'length-cylinders, perpendicular to the motion '
                        'direction d.')
    return p.parse_args()


def _build_kin_env(env_yaml, device, n_envs: int = 1):
    """FR3 kinematics + rollout machinery. ``n_envs`` controls the batched
    rollout chunk size."""
    with open(env_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in cfg['env'].items() if k in valid}
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': int(n_envs)}),
                          line_dist=None, device=device)


def _hex_to_rgb(s):
    s = s.lstrip('#')
    return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def farthest_point_sample(qs: np.ndarray, k: int) -> np.ndarray:
    """Greedy farthest-point sampling in joint space.
    qs: (N, 7) candidate joint configurations.
    Returns: (min(k, N),) indices of selected samples, maximally spread.
    """
    N = len(qs)
    if k <= 0 or k >= N:
        return np.arange(N)
    selected = [0]
    dists = np.linalg.norm(qs - qs[0], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(dists))
        selected.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(qs - qs[nxt], axis=1))
    return np.array(selected, dtype=np.int64)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Task spec --------------------------------------------------------
    es = np.load(args.eval_set, allow_pickle=False)
    T = int(args.task)
    p0 = es['cs_p0'][T].astype(np.float32)
    d  = es['cs_line_dir'][T].astype(np.float32)
    nt = es['cs_n_target'][T].astype(np.float32)
    print(f'[fig05] task={T}  N={args.n_samples}  ddim={args.ddim_steps}  '
          f'cfg_w={args.cfg_w}  alpha={args.alpha}')

    # ---- Build kinematics + load diffusion model --------------------------
    env = _build_kin_env(cfg['env']['config_yaml'], dev,
                         n_envs=max(args.n_samples, 16))
    kin = env.kin
    classical = ClassicalNullspaceController(kin)
    lo_np = kin.lmt_lo.detach().cpu().numpy().astype(np.float32)
    hi_np = kin.lmt_up.detach().cpu().numpy().astype(np.float32)

    ckpt = cfg['diffusion']['ckpt']
    use_ema = bool(cfg['diffusion'].get('use_ema', True))
    model, schedule, _model_cfg, step = load_ckpt(Path(ckpt), dev,
                                                   use_ema=use_ema)
    print(f'[fig05] diffusion ckpt={ckpt} step={step}')

    # ---- Sample N q0 from the diffusion model -----------------------------
    torch.manual_seed(int(args.sample_seed))
    np.random.seed(int(args.sample_seed))
    c_np = np.concatenate([p0, d, nt]).astype(np.float32)             # (9,)
    c_rep = torch.from_numpy(c_np).to(dev).unsqueeze(0).repeat(args.n_samples, 1)
    q_norm = ddim_sample_q0(model, schedule, c_rep, device=dev,
                            num_steps=args.ddim_steps, cfg_w=args.cfg_w)
    raw_q = denormalize_q(q_norm).cpu().numpy().astype(np.float32)    # (N, 7)

    # ---- Newton-project each onto (p0, R_target_strict) -------------------
    R_tgt = _build_R_target_strict(nt, d)
    refined = np.zeros_like(raw_q)
    ok = np.zeros(args.n_samples, dtype=bool)
    for s in range(args.n_samples):
        q_ref, conv, _err = newton_project(kin, raw_q[s], p0, R_tgt, lo_np, hi_np)
        refined[s] = q_ref
        ok[s] = bool(conv)
    print(f'[fig05] Newton-projection ok: {ok.sum()}/{len(ok)} '
          f'({100*ok.mean():.1f}%)')

    # ---- Scene ------------------------------------------------------------
    base = ovw.World(cam_pos=(2.5, -1.0, 1.5),
                     cam_lookat_pos=(0.5, 0.0, 0.5))
    ossop.frame().attach_to(base.scene)

    # Reference line + target-normal arrow.
    line_len = float(args.target_distance_m)
    ossop.dashed_cylinder(
        spos=p0, epos=p0 + d * line_len,
        radius=0.003, rgb=(0.4, 0.4, 0.4), alpha=0.85,
    ).attach_to(base.scene)
    ossop.arrow(spos=p0, epos=p0 + nt * 0.12,
                radius=0.005, rgb=(0.2, 0.2, 0.2)).attach_to(base.scene)

    # Optional faint raw samples (pre-Newton) -- shadowy.
    if args.show_raw:
        for s in range(args.n_samples):
            arm, _ = make_fr3_with_pen(use_pen_tcp=True)
            arm.attach_to(base.scene)
            attach_pen_visual(arm, rgb=(0.55, 0.55, 0.55), alpha=0.10)
            arm.rgb = (0.55, 0.55, 0.55)
            arm.alpha = 0.10
            arm.fk(qs=raw_q[s])

    # Pick representative (most spread-out) subset via farthest-point sampling.
    valid_idx = np.where(ok)[0]
    valid_qs = refined[valid_idx]
    rep_local = farthest_point_sample(valid_qs, args.n_representatives)
    rep_idx = valid_idx[rep_local]
    print(f'[fig05] representative samples (FPS over {len(valid_idx)} valid): '
          f'rendering {len(rep_idx)}/{args.n_samples}  '
          f'pool indices = {rep_idx.tolist()}')

    # ---- Roll out each representative under the classical controller to
    #      get the path-following length L (m); used to draw a per-ghost
    #      cylinder visualising the L distribution.
    rep_qs = refined[rep_idx].astype(np.float32)
    B = len(rep_qs)
    p0_per = np.broadcast_to(p0[None, :], (B, 3)).copy().astype(np.float32)
    d_per  = np.broadcast_to(d[None,  :], (B, 3)).copy().astype(np.float32)
    n_per  = np.broadcast_to(nt[None, :], (B, 3)).copy().astype(np.float32)
    roll = rollout_seeds_batched(
        rep_qs, p0_per, d_per, n_per,
        env=env, controller='classical', classical=classical,
        agent=None, target_distance_m=line_len,
        progress_every_chunks=10**9, progress_prefix='',
    )
    L_per_rep = (roll['L'].astype(np.float32) * line_len)
    print(f'[fig05] L per representative (m): '
          + '  '.join(f'{v:.3f}' for v in L_per_rep))

    # Arm body keeps the FR3 default renderer colour unless --color is given;
    # the length-stick cylinders default to dark grey (overridable by --color).
    arm_rgb = _hex_to_rgb(args.color) if args.color else None
    stick_rgb = _hex_to_rgb(args.color) if args.color else (0.15, 0.15, 0.15)
    stick_alpha = min(1.0, args.alpha + 0.5)

    # Perpendicular direction in the rolling plane for laying out the
    # parallel length-cylinders side-by-side.
    perp = np.cross(d, nt).astype(np.float32)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        perp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        perp = perp / perp_norm
    K = len(rep_idx)
    centred = np.arange(K, dtype=np.float32) - (K - 1) / 2.0

    for j, s in enumerate(rep_idx):
        arm, _ = make_fr3_with_pen(use_pen_tcp=True)
        arm.attach_to(base.scene)
        if arm_rgb is not None:
            attach_pen_visual(arm, rgb=arm_rgb, alpha=args.alpha)
            arm.rgb = arm_rgb
        else:
            attach_pen_visual(arm, alpha=args.alpha)
        arm.alpha = args.alpha
        arm.fk(qs=refined[int(s)])
        # Parallel length stick: solid cylinder along d, length = L (m),
        # laterally offset so all K sticks lie side-by-side perpendicular
        # to the motion direction.
        L_m = float(L_per_rep[j])
        if L_m > 1e-3:
            spos = p0 + perp * (centred[j] * float(args.stick_spacing))
            ossop.cylinder(
                spos=spos, epos=spos + d * L_m,
                radius=0.004, rgb=stick_rgb, alpha=stick_alpha,
            ).attach_to(base.scene)
    print(f'[fig05] viewer ready. Close the window to exit.')
    base.run()


if __name__ == '__main__':
    main()
