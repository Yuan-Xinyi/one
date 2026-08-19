"""Task 27: joint trajectories of all controllers in one figure.
7 stacked panels (normalized joints), x = arc progress; death points marked."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.eval.eval_curve import _agent
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent

dev = torch.device('cuda')
hl.SUB = 2
y = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
model.terms = [0, 1]

tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
i = 27
dt = env.kin.dtype
spec = {'q0': torch.tensor(tz['q0_seed'][i:i+1], dtype=dt, device=dev),
        'line_dir': torch.tensor(tz['cs_line_dir'][i:i+1], dtype=dt,
                                 device=dev),
        'n_target': torch.tensor(tz['cs_n_target'][i:i+1], dtype=dt,
                                 device=dev)}

classical = ClassicalNullspaceController(env.kin)
ag = _agent(REPO / hl.ROBOTS['fr3'][1], env.obs_dim, dev, act_dim=env.act_dim)
cont = ContAgent(env.obs_dim, env.act_dim).to(dev)
cont.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev))
cont.eval()

ARMS = {
    'Classical Gradient': (lambda e, dn, f=cn_action_fn(classical): f(e)),
    'Analytic Margin': hl.make_myopic(model),
    'Continuous PPO': (lambda e, dn: cont.actor_mean(e.current_obs())),
    'Vertex PPO': (lambda e, dn: ag.actor_mean(e.current_obs())),
    'Vertex PPO + Classical': hl.make_hybrid(env, ag, classical, 0.98, 0.94),
    'Critic-Guided Value': hl.make_vlook(model, env, ag),
}
COL = {'Classical Gradient': '#9aa0a6', 'Analytic Margin': '#5f8f5f',
       'Continuous PPO': '#a08cc0', 'Vertex PPO': '#e0a060',
       'Vertex PPO + Classical': '#1f77b4', 'Critic-Guided Value': '#d62728'}
STY = {'Classical Gradient': (1.8, ':'), 'Analytic Margin': (1.8, '--'),
       'Continuous PPO': (1.8, '-.'), 'Vertex PPO': (2.0, '--'),
       'Vertex PPO + Classical': (2.8, '-'),
       'Critic-Guided Value': (3.4, '-')}

trajs = {}
with torch.no_grad():
    for name, afn in ARMS.items():
        env.line_dist = ScriptedLineDistribution(
            {k: v.clone() for k, v in spec.items()})
        env.reset()
        done = torch.zeros(1, dtype=torch.bool, device=dev)
        Q, S = [env.q[0].clone().cpu().numpy()], [0.0]
        for _ in range(env.cfg.max_steps // 2):
            a = afn(env, done)
            for _ in range(2):
                env.step(a, auto_reset=False)
            Q.append(env.q[0].clone().cpu().numpy())
            S.append(float(env.arc_progress[0]))
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        trajs[name] = (np.array(S), np.array(Q))
        print(f'{name:24s} stroke {S[-1]:.3f} m')

# best searched trajectory (reach-tree, width 16384): known achievable bound
rt = np.load(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2/reachtree.npz')
qrt = torch.tensor(rt['q'], dtype=dt, device=dev)
with torch.no_grad():
    prt = env.kin.tcp_fk_jac(qrt)[0]
p0 = prt[0]
d0 = spec['line_dir'][0]
Srt = ((prt - p0) @ d0).float().cpu().numpy()
name_rt = f"Reach-tree search ({float(rt['progress']):.2f} m)"
trajs[name_rt] = (Srt, rt['q'])
COL[name_rt] = '#222222'
STY[name_rt] = (2.4, (0, (5, 2)))
print(f'search traj: {float(rt["progress"]):.3f} m, {rt["q"].shape[0]} states')

qmid = env.q_mid.cpu().numpy(); qhalf = env.q_half.cpu().numpy()
fig, axes2 = plt.subplots(2, 4, figsize=(32, 12), sharex=True)
axes = axes2.ravel()
for j in range(7):
    ax = axes[j]
    ax.axhspan(-1, -0.98, color='0.85', lw=0)
    ax.axhspan(0.98, 1.0, color='0.85', lw=0)
    ax.axhline(1.0, color='k', lw=0.7)
    ax.axhline(-1.0, color='k', lw=0.7)
    for name, (S, Q) in trajs.items():
        qn = (Q[:, j] - qmid[j]) / qhalf[j]
        lw, ls = STY[name]
        z = 5 if name == 'Critic-Guided Value' else (4 if name in ('Vertex PPO + Classical', name_rt) else 3)
        ax.plot(S, qn, color=COL[name], lw=lw, ls=ls, zorder=z,
                label=name if j == 0 else None)
        ax.plot(S[-1], qn[-1], 'o', color=COL[name], ms=9 if z == 5 else 7,
                zorder=z + 1, mec='white', mew=1.2)
    ax.set_ylabel(f'$q_{{{j+1}}}$ (norm.)', fontsize=18)
    ax.set_ylim(-1.12, 1.12)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=14)
    ax.set_title(f'joint {j+1}', fontsize=18)
# inset: q2 death zone
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
axi = inset_axes(axes[1], width='55%', height='58%', loc='lower right',
                 borderpad=2.0)
for name, (S, Q) in trajs.items():
    qn = (Q[:, 1] - qmid[1]) / qhalf[1]
    m = S <= 0.32
    lw, ls = STY[name]
    z = 5 if name == 'Critic-Guided Value' else (4 if name in ('Vertex PPO + Classical', name_rt) else 3)
    axi.plot(S[m], qn[m], color=COL[name], lw=lw, ls=ls, zorder=z)
    if S[-1] <= 0.32:
        axi.plot(S[-1], qn[-1], 'o', color=COL[name], ms=7, zorder=z + 1,
                 mec='white', mew=1.2)
axi.axhline(1.0, color='k', lw=0.8)
axi.set_xlim(0, 0.32); axi.set_ylim(0.72, 1.04)
axi.tick_params(labelsize=11)
axi.set_title('death zone (zoom)', fontsize=13)
mark_inset(axes[1], axi, loc1=2, loc2=4, fc='none', ec='0.6', lw=0.8)

h, l = axes[0].get_legend_handles_labels()
axes[7].axis('off')
axes[7].legend(h, l, loc='center', fontsize=20, framealpha=0.9)
for ax in axes[4:7]:
    ax.set_xlabel('arc progress along the task path [m]', fontsize=16)
fig.suptitle('Task 27: joint trajectories per controller '
             '(normalized joints; x = death point; '
             r'$\ell^{\mathrm{pw}} \geq 1.70$ m)', fontsize=22, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = ('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-'
       'vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/'
       'scratchpad/task27_trajs.png')
fig.savefig(out, dpi=150)
print('wrote', out)
