"""Scan the eval set for tasks well-suited to a real-robot demo.

For each task we replay (with the SAME diffusion seed fig04 uses) the three
controllers — classical / rl / hybrid — in a batched rollout, recording
per-task progress and the per-joint q range actually visited. A task is a
good hardware candidate when:

  * all three trajectories stay (nearly) inside the joint limits, so the
    hardware clamp in `replay_franka_traj.py` barely fires, and
  * the hybrid controller's progress beats BOTH classical and pure-RL by a
    clear margin (the controller-ablation story we want to show on hardware).

Ranked candidates are printed; pass `--dump-top K` to write each one's
`fig04_traj_task<T>.npz` (and PNGs) via fig04_joint_trajectories.

Usage:
    python -m Yuan.paper_figures.scan_hw_tasks --dump-top 3
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.system_eval.rollout_controllers import (
    build_env, load_rl_agent,
)
from Yuan.RL_controller.env.env import build_task_aligned_basis
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution

DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
DEFAULT_DP_NPZ   = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--dp-seeds', default=DEFAULT_DP_NPZ)
    p.add_argument('--limit-margin', type=float, default=0.01,
                   help='clamp band used at replay time (rad)')
    p.add_argument('--max-clamp', type=float, default=0.02,
                   help='max allowed overshoot beyond a hard joint limit, any '
                        'mode/joint, for a task to count as hardware-feasible (rad)')
    p.add_argument('--min-hybrid-adv', type=float, default=0.05,
                   help='require hybrid_L - max(classical_L, rl_L) >= this')
    p.add_argument('--exclude', type=int, nargs='*', default=[7199],
                   help='task ids to skip (already-chosen demos)')
    p.add_argument('--dump-top', type=int, default=0,
                   help='render+save fig04 npz for the top-K candidates')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='only scan the first N tasks (debug)')
    return p.parse_args()


@torch.no_grad()
def _rollout_chunk(env, classical, agent, mode, qs, p0s, dirs, ntgts,
                   tau_e, tau_x):
    """One batched chunk under `mode`; returns progress + per-joint q range.

    q_min / q_max are tracked over every visited config (incl. the seed and
    the final step) so hardware joint-limit feasibility can be screened.
    """
    spec = {'q0': qs, 'line_dir': dirs, 'n_target': ntgts}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = p0s
    p_start = env.p_start.clone()
    line_dir = env.line_dir.clone()
    n = env.n_envs
    q_mid, q_half = env.q_mid, env.q_half
    cls_action = cn_action_fn(classical)

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    using_rl = _max_abs_qn(env.q) < tau_e          # hybrid initial branch
    q_min = env.q.clone()
    q_max = env.q.clone()
    progress = torch.zeros(n, dtype=env.kin.dtype, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)

    for _ in range(env.max_steps + 1):
        if mode == 'classical':
            a = cls_action(env)
        elif mode == 'rl':
            a = agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
        else:  # hybrid variant B
            cur = _max_abs_qn(env.q)
            using_rl = torch.where(using_rl, cur < tau_e, cur < tau_x)
            rl_act = agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping)
            q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
            cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
            cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)
            a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)

        _, _, _, _, info = env.step(a, auto_reset=False)
        # track q range only for envs still running (avoid post-done drift)
        live = (~finished).unsqueeze(-1)
        q_min = torch.where(live, torch.minimum(q_min, env.q), q_min)
        q_max = torch.where(live, torch.maximum(q_max, env.q), q_max)

        new_done = info['episode_done']
        if bool(new_done.any().item()):
            p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
            prog = ((p_now - p_start) * line_dir).sum(-1)
            progress[new_done] = prog[new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break

    if (~finished).any():
        nd = ~finished
        p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p_now - p_start) * line_dir).sum(-1)
        progress[nd] = prog[nd]
    return progress, q_min, q_max


@torch.no_grad()
def scan_mode(env, classical, agent, mode, qs, p0s, dirs, ntgts,
              tau_e, tau_x, target_dist):
    """Batched scan of all tasks under `mode`. Returns L, q_min, q_max arrays."""
    import math
    B = qs.shape[0]
    ne = env.n_envs
    dev, dt = env.device, env.kin.dtype
    qf = torch.as_tensor(qs, device=dev, dtype=dt)
    pf = torch.as_tensor(p0s, device=dev, dtype=dt)
    df = torch.as_tensor(dirs, device=dev, dtype=dt)
    nf = torch.as_tensor(ntgts, device=dev, dtype=dt)
    L = np.zeros(B, np.float32)
    qmin = np.zeros((B, 7), np.float32)
    qmax = np.zeros((B, 7), np.float32)
    nchunks = math.ceil(B / ne)
    for ci in range(nchunks):
        s, e = ci * ne, min((ci + 1) * ne, B)
        rn = e - s
        if rn == ne:
            qc, pc, dc, nc = qf[s:e], pf[s:e], df[s:e], nf[s:e]
        else:
            pad = ne - rn
            qc = torch.cat([qf[s:e], qf[e-1:e].expand(pad, 7)])
            pc = torch.cat([pf[s:e], pf[e-1:e].expand(pad, 3)])
            dc = torch.cat([df[s:e], df[e-1:e].expand(pad, 3)])
            nc = torch.cat([nf[s:e], nf[e-1:e].expand(pad, 3)])
        prog, qmn, qmx = _rollout_chunk(env, classical, agent, mode,
                                        qc, pc, dc, nc, tau_e, tau_x)
        L[s:e] = (prog[:rn] / target_dist).cpu().numpy()
        qmin[s:e] = qmn[:rn].cpu().numpy()
        qmax[s:e] = qmx[:rn].cpu().numpy()
        if ci % 10 == 0 or ci == nchunks - 1:
            print(f'  [{mode:9s}] {e}/{B}', flush=True)
    return L, qmin, qmax


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    es = np.load(args.eval_set)
    dp = np.load(args.dp_seeds)['seeds'].astype(np.float32)   # (B, 7)
    p0 = es['cs_p0'].astype(np.float32)
    d  = es['cs_line_dir'].astype(np.float32)
    nt = es['cs_n_target'].astype(np.float32)
    B = dp.shape[0]
    if args.max_tasks:
        B = min(B, args.max_tasks)
        dp, p0, d, nt = dp[:B], p0[:B], d[:B], nt[:B]
    print(f'scanning {B} tasks, seed source {Path(args.dp_seeds).name}')

    env = build_env(cfg['env']['config_yaml'], int(cfg['env']['n_envs']), dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    te = float(cfg['rl_controller']['tau_enter'])
    tx = float(cfg['rl_controller']['tau_exit'])
    target_dist = float(cfg['env']['target_distance_m'])
    lo = env.kin.lmt_lo.cpu().numpy().astype(np.float32)
    hi = env.kin.lmt_up.cpu().numpy().astype(np.float32)

    res = {}
    for mode in ('classical', 'rl', 'hybrid'):
        res[mode] = scan_mode(env, classical, agent, mode, dp, p0, d, nt,
                              te, tx, target_dist)

    L_cls, L_rl, L_hyb = res['classical'][0], res['rl'][0], res['hybrid'][0]
    # worst overshoot beyond a hard limit across all three modes (rad)
    overshoot = np.zeros(B, np.float32)
    for mode in ('classical', 'rl', 'hybrid'):
        _, qmn, qmx = res[mode]
        over = np.maximum.reduce([lo[None, :] - qmn, qmx - hi[None, :],
                                  np.zeros_like(qmn)]).max(axis=1)
        overshoot = np.maximum(overshoot, over)

    adv = L_hyb - np.maximum(L_cls, L_rl)
    feasible = overshoot <= args.max_clamp
    good = feasible & (adv >= args.min_hybrid_adv)
    for t in args.exclude:
        if 0 <= t < B:
            good[t] = False
    order = np.argsort(-adv)
    order = [int(t) for t in order if good[t]]

    print(f'\n{feasible.sum()} feasible (clamp<= {args.max_clamp}rad), '
          f'{good.sum()} also hybrid-dominant (adv>= {args.min_hybrid_adv})')
    print(f'\n{"rank":>4} {"task":>6} {"L_hyb":>6} {"L_cls":>6} {"L_rl":>6} '
          f'{"adv":>6} {"clamp":>6} {"bucket":>11}')
    for r, t in enumerate(order[:30]):
        print(f'{r:>4} {t:>6} {L_hyb[t]:>6.3f} {L_cls[t]:>6.3f} {L_rl[t]:>6.3f} '
              f'{adv[t]:>6.3f} {overshoot[t]:>6.4f} {str(es["bucket"][t]):>11}')

    if args.dump_top > 0:
        chosen = order[:args.dump_top]
        print(f'\ndumping fig04 npz for tasks: {chosen}')
        for t in chosen:
            subprocess.run([sys.executable, '-m',
                            'Yuan.paper_figures.fig04_joint_trajectories',
                            '--task', str(t)], check=True)


if __name__ == '__main__':
    main()
