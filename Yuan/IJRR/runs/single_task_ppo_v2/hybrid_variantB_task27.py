"""The EXACT ISRR hybrid (variant B) on task 27: step-level hysteresis
switching on max|q_norm| with tau_enter/tau_exit, RL <-> classical, exactly
as in Yuan/system_eval/rollout_controllers.py."""
import matplotlib                       # must precede torch on this box
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, sys
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
q_mid, q_half = env.q_mid, env.q_half


def max_abs_qn(q):
    return ((q - q_mid).abs() / q_half).max(dim=-1).values


@torch.no_grad()
def run_variantB(tau_enter, tau_exit):
    env.reset()
    using_rl = bool(max_abs_qn(env.q[:1]) < tau_enter)
    switches, trace = 0, []
    for t in range(env.max_steps):
        qn = float(max_abs_qn(env.q[:1]))
        new_rl = qn < (tau_enter if using_rl else tau_exit)
        if new_rl != using_rl:
            switches += 1
            using_rl = new_rl
        trace.append((t, qn, using_rl))
        a = agent.actor_mean(env.current_obs()) if using_rl else fcl(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        if bool(info['episode_done'][0]):
            term = int(info['term_reason'][0])
            break
    else:
        term = TERM_TRUNCATED
    pf, _, _, _ = env.kin.tcp_fk_jac(env.q[:1])
    prog = float(((pf[0] - env.p_start[0]) * env.line_dir[0]).sum())
    return prog, term, t + 1, switches, trace

print("tau_e  tau_x   final(m)  steps  switches  term")
results = {}
for te, tx in [(0.98, 0.94), (0.98, 0.98), (0.95, 0.90), (0.90, 0.85),
               (0.85, 0.80), (0.80, 0.75)]:
    prog, term, steps, sw, trace = run_variantB(te, tx)
    results[(te, tx)] = (prog, term, steps, sw, trace)
    print(f"{te:.2f}  {tx:.2f}   {prog:.4f}   {steps:>4d}  {sw:>7d}   "
          f"{TERM_NAMES[term]}")

# trace figure for the paper setting 0.98/0.94
prog, term, steps, sw, trace = results[(0.98, 0.94)]
t_arr = np.array([r[0] for r in trace])
qn_arr = np.array([r[1] for r in trace])
rl_arr = np.array([r[2] for r in trace])
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_arr * 0.01, qn_arr, c='black', lw=1.4, label='max |q_norm|')
ax.axhline(0.98, c='tab:red', ls='--', lw=1, label='tau_enter 0.98')
ax.axhline(0.94, c='tab:orange', ls='--', lw=1, label='tau_exit 0.94')
ax.fill_between(t_arr * 0.01, 0, 1.05, where=~rl_arr, color='gray',
                alpha=0.25, label='classical in charge')
ax.set_xlabel('arc length s (m)')
ax.set_ylabel('max normalized joint excursion')
ax.set_ylim(0.5, 1.03)
ax.set_title(f'task 27, ISRR hybrid variant B (0.98/0.94): final '
             f'{prog:.3f} m, {sw} switches, term {TERM_NAMES[term]}')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(alpha=0.25)
fig.tight_layout()
out = RUN / 'hybrid_variantB.png'
fig.savefig(out, dpi=150)
print('wrote', out)
np.savez(RUN / 'hybrid_variantB.npz',
         taus=np.array([k for k in results]),
         final=np.array([v[0] for v in results.values()]),
         switches=np.array([v[3] for v in results.values()]))
