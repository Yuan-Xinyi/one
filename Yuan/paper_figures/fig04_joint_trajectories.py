"""Figure 4: q6 trajectory for one task under each of the three controllers.

Outputs three separate PNGs (one per mode), each:
  - figsize 10x8
  - linewidth 3
  - y-axis = absolute FR3 joint 6 range (rad), with the lower/upper joint
    limits as the axis bounds.

Hybrid is coloured by the sub-controller active at each step (RL = red,
Classical = yellow); Classical mode is yellow throughout; RL mode is red
throughout.
"""
from __future__ import annotations
import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent


COLOR_CLASSICAL = '#FFBE7A'
COLOR_RL        = '#FA7F6F'

DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
DEFAULT_DP_NPZ   = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'
JOINT_IDX        = 5    # zero-based: q6 is index 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=7199)
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--dp-seeds', default=DEFAULT_DP_NPZ)
    p.add_argument('--out-dir', default=str(Path(__file__).parent))
    p.add_argument('--linewidth', type=float, default=5.0)
    p.add_argument('--threshold-linewidth', type=float, default=2.5)
    p.add_argument('--figsize', nargs=2, type=float, default=(7, 4))
    p.add_argument('--dpi', type=int, default=200)
    return p.parse_args()


# -----------------------------------------------------------------------
# Trajectory capture (n_envs = 1)
# -----------------------------------------------------------------------
def _build_env(env_yaml, device):
    with open(env_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in cfg['env'].items() if k in valid}
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), line_dist=None,
                          device=device)


def _load_agent(ckpt_dir, env, device):
    with open(Path(ckpt_dir) / 'config.yaml') as f:
        rl_cfg = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=rl_cfg['ppo']['hidden_dim'],
                  init_log_std=rl_cfg['ppo']['init_log_std']).to(device)
    state = torch.load(Path(ckpt_dir) / 'agent.pt', map_location=device,
                       weights_only=False)
    agent.load_state_dict(state['model'] if isinstance(state, dict)
                          and 'model' in state else state)
    agent.eval()
    return agent


def _classical_action(env, classical):
    q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping,
    )
    a = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    return (a / env.a_max).clamp(-1.0, 1.0)


def _rl_action(env, agent):
    with torch.no_grad():
        return agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)


def capture(env, classical, agent, q0, p0, d, n, mode, tau_e, tau_x, device):
    spec = {'q0': torch.as_tensor(q0[None], device=device, dtype=env.kin.dtype),
            'line_dir': torch.as_tensor(d[None], device=device, dtype=env.kin.dtype),
            'n_target': torch.as_tensor(n[None], device=device, dtype=env.kin.dtype)}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = torch.as_tensor(p0[None], device=device,
                                     dtype=env.kin.dtype)
    q_mid = env.q_mid; q_half = env.q_half

    def _max_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    using_rl = bool((_max_qn(env.q) < tau_e).item())
    q_traj = [env.q[0].detach().cpu().numpy().copy()]
    using_rl_per_step = []
    for _ in range(env.max_steps + 1):
        cur = float(_max_qn(env.q).item())
        if mode == 'hybrid':
            if using_rl and cur >= tau_e:
                using_rl = False
            elif (not using_rl) and cur < tau_x:
                using_rl = True
        elif mode == 'rl':
            using_rl = True
        else:
            using_rl = False
        using_rl_per_step.append(using_rl)
        a = _rl_action(env, agent) if using_rl else _classical_action(env, classical)
        _, _, _, _, info = env.step(a, auto_reset=False)
        q_traj.append(env.q[0].detach().cpu().numpy().copy())
        if bool(info['episode_done'][0].item()):
            break
    using_rl_per_step.append(using_rl_per_step[-1])
    return (np.stack(q_traj).astype(np.float32),
            np.array(using_rl_per_step, dtype=bool))


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------
Y_LO = 0.5
Y_HI = 2.0


def plot_single_q6(q_traj, using_rl_traj, mode, q_mid, q_half,
                   tau_enter, tau_exit,
                   x_max, y_lo, y_hi,
                   linewidth, threshold_lw,
                   figsize, dpi, out_path):
    q6 = q_traj[:, JOINT_IDX]
    t = np.arange(len(q6))

    fig, ax = plt.subplots(figsize=tuple(figsize))

    if mode == 'classical':
        ax.plot(t, q6, color=COLOR_CLASSICAL, linewidth=linewidth)
    elif mode == 'rl':
        ax.plot(t, q6, color=COLOR_RL, linewidth=linewidth)
    else:
        # Hybrid: colour each segment by the active sub-controller.
        pts = np.column_stack([t, q6])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        colors = [COLOR_RL if using_rl_traj[i] else COLOR_CLASSICAL
                  for i in range(len(segs))]
        lc = LineCollection(segs, colors=colors, linewidths=linewidth,
                            capstyle='round')
        ax.add_collection(lc)

        # Switching thresholds in q6 units (rho thresholds applied to the
        # q6 axis via mid +/- tau * half_range).
        for tau in (tau_enter, tau_exit):
            ax.axhline(q_mid + tau * q_half, color='black',
                       linestyle='--', linewidth=threshold_lw, alpha=0.7)
            ax.axhline(q_mid - tau * q_half, color='black',
                       linestyle='--', linewidth=threshold_lw, alpha=0.7)

    ax.set_xlabel('step', fontsize=14)
    ax.set_ylabel(r'$q_{6}$  (rad)', fontsize=14)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis='both', labelsize=12)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    es = np.load(args.eval_set)
    T = int(args.task)
    p0 = es['cs_p0'][T].astype(np.float32)
    d  = es['cs_line_dir'][T].astype(np.float32)
    nt = es['cs_n_target'][T].astype(np.float32)
    dp = np.load(args.dp_seeds)['seeds'][T].astype(np.float32)

    env = _build_env(cfg['env']['config_yaml'], dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = _load_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    te = float(cfg['rl_controller']['tau_enter'])
    tx = float(cfg['rl_controller']['tau_exit'])

    # FR3 joint 6 range -- used to convert the rho thresholds (which are
    # normalised against the full joint range) back into absolute radians
    # for the threshold lines on the q6 axis.
    q_lo = float(env.kin.lmt_lo[JOINT_IDX].cpu().item())
    q_hi = float(env.kin.lmt_up[JOINT_IDX].cpu().item())
    q_mid_6  = 0.5 * (q_lo + q_hi)
    q_half_6 = 0.5 * (q_hi - q_lo)
    print(f'q{JOINT_IDX+1} range: [{q_lo:.3f}, {q_hi:.3f}] rad  '
          f'(mid={q_mid_6:.3f}, half={q_half_6:.3f})')
    print(f'  tau_enter={te}  -> q6 thresholds {q_mid_6 - te*q_half_6:.3f} / '
          f'{q_mid_6 + te*q_half_6:.3f}')
    print(f'  tau_exit ={tx}  -> q6 thresholds {q_mid_6 - tx*q_half_6:.3f} / '
          f'{q_mid_6 + tx*q_half_6:.3f}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Capture all three trajectories first to derive shared axis bounds.
    captured = {}
    for mode in ('classical', 'rl', 'hybrid'):
        q_traj, urlt = capture(env, classical, agent, dp, p0, d, nt,
                               mode, te, tx, dev)
        captured[mode] = (q_traj, urlt)
        print(f'{mode:9s}: T={len(q_traj)-1:>3d} steps  '
              f'q6 range=[{q_traj[:, JOINT_IDX].min():.3f}, '
              f'{q_traj[:, JOINT_IDX].max():.3f}]')

    # Save the full 7-joint trajectories (one array per mode, possibly
    # different lengths) plus task metadata, for hardware replay (franky).
    npz_path = out_dir / f'fig04_traj_task{T}.npz'
    save_kw = {}
    for mode, (q_traj, urlt) in captured.items():
        save_kw[f'{mode}_q'] = q_traj.astype(np.float32)          # (T+1, 7)
        save_kw[f'{mode}_using_rl'] = urlt.astype(bool)           # (T+1,)
    np.savez(
        npz_path,
        task=np.int64(T),
        q0_seed=dp.astype(np.float32),      # start config (= traj[0] of all modes)
        cs_p0=p0, cs_line_dir=d, cs_n_target=nt,
        tau_enter=np.float32(te), tau_exit=np.float32(tx),
        joint_lo=env.kin.lmt_lo.detach().cpu().numpy().astype(np.float32),
        joint_hi=env.kin.lmt_up.detach().cpu().numpy().astype(np.float32),
        modes=np.array(list(captured.keys()), dtype='<U16'),
        **save_kw,
    )
    print(f'wrote {npz_path}  (modes: {", ".join(captured.keys())})')

    # Shared y-range fixed at [Y_LO, Y_HI] = [0.5, 1.0] (focused on the
    # lower part of q6 where the rho thresholds live).
    y_lo, y_hi = Y_LO, Y_HI
    # Shared x-range: the step at which Hybrid's q6 first leaves the visible
    # y window (i.e., crosses Y_HI). After that Hybrid is off-screen anyway,
    # so cutting at this step keeps the plot tight on the part of the
    # trajectory that fits the chosen y axis.
    hyb_q6 = captured['hybrid'][0][:, JOINT_IDX]
    above = np.where(hyb_q6 > y_hi)[0]
    x_max = int(above[0]) if above.size else len(hyb_q6) - 1
    print(f'shared axes: x in [0, {x_max}], y in [{y_lo:.3f}, {y_hi:.3f}]')

    # 2) Plot each mode with the shared bounds.
    for mode, (q_traj, urlt) in captured.items():
        out_path = out_dir / f'fig04_q{JOINT_IDX+1}_{mode}.png'
        plot_single_q6(q_traj, urlt, mode, q_mid_6, q_half_6, te, tx,
                       x_max, y_lo, y_hi,
                       args.linewidth, args.threshold_linewidth,
                       args.figsize, args.dpi, out_path)


if __name__ == '__main__':
    main()
