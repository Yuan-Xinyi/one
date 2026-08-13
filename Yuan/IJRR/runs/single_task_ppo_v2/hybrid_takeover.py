"""ISRR-style hybrid on task 27: PPO drives, classical nullspace takes over
at step k; sweep every takeover point k = 0..T_death and record how far the
classical continuation gets."""
import matplotlib                       # must precede torch on this box
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, sys, os
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import torch, yaml
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.eval.single_task_ppo import (_env_and_yaml, _load_task,
                                            SingleTaskDistribution,
                                            TERM_NAMES, TERM_TRUNCATED)
import Yuan.IJRR.eval.single_task_ppo as stp
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent

dev = torch.device('cuda')
stp.OUT = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
RUN = stp.OUT
y, env = _env_and_yaml(1, dev)
task, spec1 = _load_task(dev, env.kin.dtype)
env.line_dist = SingleTaskDistribution(spec1)

agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev))
agent.eval()
classical = ClassicalNullspaceController(env.kin)
fcl = cn_action_fn(classical)


@torch.no_grad()
def run_episode(switch_k):
    """PPO mean action for t < switch_k, classical afterwards."""
    env.reset()
    for t in range(env.max_steps):
        a = (agent.actor_mean(env.current_obs()) if t < switch_k
             else fcl(env))
        _, _, _, _, info = env.step(a, auto_reset=False)
        if bool(info['episode_done'][0]):
            term = int(info['term_reason'][0])
            break
    else:
        term = TERM_TRUNCATED
    pf, _, _, _ = env.kin.tcp_fk_jac(env.q[:1])
    prog = float(((pf[0] - env.p_start[0]) * env.line_dir[0]).sum())
    return prog, term, t + 1

prog_ppo, term_ppo, T = run_episode(10 ** 9)     # pure PPO
print(f"pure PPO      progress {prog_ppo:.4f} m  {T} steps  "
      f"term {TERM_NAMES[term_ppo]}")
prog_cl, term_cl, Tc = run_episode(0)            # pure classical
print(f"pure classical progress {prog_cl:.4f} m  {Tc} steps  "
      f"term {TERM_NAMES[term_cl]}")

rows = []
for k in range(0, T + 1):
    prog, term, steps = run_episode(k)
    rows.append((k, prog, term, steps - k))
    if k % 10 == 0 or prog > prog_ppo + 1e-4:
        print(f"switch @ {k:>3d} (s={k * 0.01:.2f})  final {prog:.4f} m  "
              f"term {TERM_NAMES[term]}  classical steps {steps - k}",
              flush=True)

ks = np.array([r[0] for r in rows])
fin = np.array([r[1] for r in rows])
extra = fin - np.minimum(ks * 0.01, prog_ppo)    # gain over handover point
best = int(fin.argmax())
print(f"\nbest hybrid: switch @ step {rows[best][0]} "
      f"(s={rows[best][0] * 0.01:.2f})  final {fin[best]:.4f} m  "
      f"term {TERM_NAMES[rows[best][2]]}")
print(f"pure PPO {prog_ppo:.4f}  ->  hybrid max {fin.max():.4f}  "
      f"(delta {fin.max() - prog_ppo:+.4f})")
np.savez(RUN / 'hybrid_takeover.npz', switch_k=ks, final=fin,
         term=np.array([r[2] for r in rows]),
         classical_steps=np.array([r[3] for r in rows]),
         ppo=prog_ppo, classical=prog_cl)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ks * 0.01, fin, 'o-', ms=3, lw=1.2, c='black',
        label='hybrid: PPO then classical takes over')
ax.plot(ks * 0.01, ks * 0.01, ls=':', c='gray', lw=1,
        label='progress already made at handover')
ax.axhline(prog_ppo, c='tab:blue', ls='--', lw=1.2,
           label=f'pure PPO ({prog_ppo:.2f} m)')
ax.axhline(1.0614, c='tab:red', ls='--', lw=1.2,
           label='best search (1.06 m)')
ax.set_xlabel('takeover point s (m along the line)')
ax.set_ylabel('final progress (m)')
ax.set_title('task 27: classical-nullspace takeover of the RL trajectory\n'
             '(every possible handover step)')
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = RUN / 'hybrid_takeover.png'
fig.savefig(out, dpi=150)
print('wrote', out)
