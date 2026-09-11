"""Fair replot: classical and RL from the SAME critic-picked start."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
OUT = MAIN/'runs/paper_fill/search_compare'
d = np.load(FU/'cone5_eval_v1.npz'); s5 = np.load(FU/'cone5_sel_10k.npz')
ct = np.load(FU/'cone5_cls_pick_traj.npz')
sub, lpw5, pick = d['sub'], d['lpw5'], s5['pick']
tz = np.load(A/'tasks_pool_fr3.npz')
TASKS = ct['tasks'].tolist()
rows = {int(s): i for i, s in enumerate(sub)}
idx = np.array([rows[t] for t in TASKS])
# RL trajectories from picked starts
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone5.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': len(idx)}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8k_cone5/agent.pt', map_location=dev))
ag.eval()
rdt = env.kin.dtype
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(pick[idx], dtype=rdt, device=dev),
     'line_dir': torch.tensor(tz['cs_line_dir'][sub[idx]], dtype=rdt, device=dev),
     'n_target': torch.tensor(tz['cs_n_target'][sub[idx]], dtype=rdt, device=dev)})
env.reset()
Qr, Sr = [env.q.cpu().numpy().copy()], [np.zeros(len(idx), np.float32)]
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
            Qr.append(env.q.cpu().numpy().copy())
            Sr.append(env.arc_progress.float().cpu().numpy().copy())
        if bool(env.done_persistent.all()):
            break
Qr, Sr = np.array(Qr), np.array(Sr)
Qc, Sc = ct['q'], ct['s']
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
for k, t in enumerate(TASKS):
    c = idx[k]
    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    for j in range(7):
        ax = axes[j//4, j%4]
        ax.plot(Sc[:, k], Qc[:, k, j], color='crimson', lw=2.2,
                label=f'Classical, same start ({Sc[-1, k]:.2f} m)')
        ax.plot(Sr[:, k], Qr[:, k, j], color='royalblue', lw=2.2,
                label=f'Cone5 RL ({Sr[-1, k]:.2f} m)')
        ax.axhline(LMT_LO[j], color='k', ls='--', lw=1.6)
        ax.axhline(LMT_UP[j], color='k', ls='--', lw=1.6)
        ax.set_ylim(min(LMT_LO[j], Qc[:, k, j].min(), Qr[:, k, j].min())-0.15,
                    max(LMT_UP[j], Qc[:, k, j].max(), Qr[:, k, j].max())+0.15)
        ax.set_title(f'Joint {j+1}', fontsize=16)
        ax.set_xlabel('arc length s (m)', fontsize=13)
        ax.set_ylabel('q (rad)', fontsize=13)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=13)
    ax = axes[1, 3]; ax.axis('off')
    ax.text(0.05, 0.7, f'Task {t} (5° cone)', fontsize=26, weight='bold')
    ax.text(0.05, 0.5, f'$\\ell^{{pw}}$(5°) = {lpw5[c]:.2f} m', fontsize=20)
    ax.text(0.05, 0.34, f'Classical: {Sc[-1, k]:.2f} m', fontsize=18, color='crimson')
    ax.text(0.05, 0.2, f'Cone5 RL: {Sr[-1, k]:.2f} m', fontsize=18, color='royalblue')
    ax.text(0.05, 0.06, 'SAME critic-picked start for both controllers',
            fontsize=14, weight='bold')
    fig.tight_layout()
    fig.savefig(OUT/f'cone5_t{t}_cmp.png', dpi=70)
    plt.close(fig)
    print(f't{t}: cls {Sc[-1, k]:.2f}  rl {Sr[-1, k]:.2f}', flush=True)
