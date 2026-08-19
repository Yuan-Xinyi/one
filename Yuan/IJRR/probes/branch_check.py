"""Is PPO's trajectory a mirror / different self-motion branch of myopic's?

Checks, on task 215's two recorded trajectories:
  1. joint-space mirror test: q_PPO ?= M q_myopic with M = diag(-1,1,-1,1,-1,1,-1)
  2. TCP position agreement at equal t (both track the same line at v)
  3. arm angle psi(t): elbow position around the shoulder-wrist axis,
     referenced to the vertical plane -- the self-motion coordinate.
"""
import numpy as np
import torch
import yaml
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import sys
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
kin = env.kin

d = np.load(REPO / 'Yuan/IJRR/runs/single_task_ppo/traj_compare.npz')
qm = torch.tensor(d['myopic_q'], device=dev, dtype=kin.dtype)
qp = torch.tensor(d['PPO_q'], device=dev, dtype=kin.dtype)
T = min(len(qm), len(qp))

# 1. mirror test
M = torch.tensor([-1., 1., -1., 1., -1., 1., -1.], device=dev, dtype=kin.dtype)
raw = (qp[:T] - qm[:T]).abs().max().item()
mir = (qp[:T] - M * qm[:T]).abs().max().item()
print(f'max |q_PPO - q_myopic|          = {np.degrees(raw):7.1f} deg')
print(f'max |q_PPO - mirror(q_myopic)|  = {np.degrees(mir):7.1f} deg')

# 2. TCP agreement at equal t
pm, Rm, _, _ = kin.tcp_fk_jac(qm[:T])
pp, Rp, _, _ = kin.tcp_fk_jac(qp[:T])
dtcp = (pm - pp).norm(dim=-1)
zang = torch.rad2deg(torch.acos(
    (Rm[:, :, 2] * Rp[:, :, 2]).sum(-1).clamp(-1, 1)))
print(f'TCP distance at equal t: mean {dtcp.mean()*1000:.1f} mm, '
      f'max {dtcp.max()*1000:.1f} mm')
print(f'tool z-axis angle between arms: mean {zang.mean():.1f} deg, '
      f'max {zang.max():.1f} deg')


# 3. arm angle
def arm_angle(q):
    tfs = kin.link_transforms(q)                     # (B, 8, 4, 4)
    S = tfs[:, 2, :3, 3]                             # shoulder (joint-2 origin)
    E = tfs[:, 4, :3, 3]                             # elbow (joint-4 origin)
    W = tfs[:, 6, :3, 3]                             # wrist (joint-6 origin)
    u = (W - S) / (W - S).norm(dim=-1, keepdim=True)
    z = torch.zeros_like(u); z[:, 2] = 1.0
    r0 = z - (z * u).sum(-1, keepdim=True) * u
    r0 = r0 / r0.norm(dim=-1, keepdim=True)
    v = E - S
    vp = v - (v * u).sum(-1, keepdim=True) * u
    s = (u * torch.linalg.cross(r0, vp, dim=-1)).sum(-1)
    c = (r0 * vp).sum(-1)
    return torch.rad2deg(torch.atan2(s, c)), S, E, W


psi_m, _, Em, _ = arm_angle(qm)
psi_p, _, Ep, _ = arm_angle(qp)
print(f'arm angle psi: myopic [{psi_m.min():.0f}, {psi_m.max():.0f}] deg, '
      f'PPO [{psi_p.min():.0f}, {psi_p.max():.0f}] deg')
print(f'psi difference at equal t: mean '
      f'{(psi_p[:T]-psi_m[:T]).mean():.1f} deg, '
      f'start {psi_p[0]-psi_m[0]:.1f}, end {(psi_p[T-1]-psi_m[T-1]):.1f}')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dt = 0.05
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
ax = axes[0]
ax.plot(np.arange(len(psi_m)) * dt, psi_m.cpu(), color='tab:blue',
        label='myopic')
ax.plot(np.arange(len(psi_p)) * dt, psi_p.cpu(), color='tab:red',
        label='PPO')
ax.set_xlabel('time (s)'); ax.set_ylabel('arm angle $\\psi$ (deg)')
ax.set_title('self-motion coordinate (elbow around S-W axis)')
ax.legend(frameon=False)

ax = axes[1]
Emc, Epc = Em.cpu().numpy(), Ep.cpu().numpy()
ax.plot(Emc[:, 0], Emc[:, 1], color='tab:blue', label='myopic elbow')
ax.plot(Epc[:, 0], Epc[:, 1], color='tab:red', label='PPO elbow')
ax.plot(Emc[0, 0], Emc[0, 1], 'ko', ms=5)
pmc = pm.cpu().numpy()
ax.plot(pmc[:, 0], pmc[:, 1], color='0.5', lw=1, ls=':', label='TCP path')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.axis('equal')
ax.set_title('elbow path, top view (dot = start)')
ax.legend(frameon=False, fontsize=8)

ax = axes[2]
ax.plot(np.arange(T) * dt, zang.cpu(), color='tab:purple')
ax.set_xlabel('time (s)'); ax.set_ylabel('deg')
ax.set_title('tool-axis angle between the two arms\n(same TCP point, same cone)')
fig.tight_layout()
out = REPO / 'Yuan/IJRR/runs/single_task_ppo/branch_check.png'
fig.savefig(out, dpi=160)
print(f'wrote {out}')
