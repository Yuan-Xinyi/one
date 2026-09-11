"""Joint-trajectory comparison at 5-deg cone: classical (from first
5-deg start) vs cone5-retrain + critic-picked start. Six showcase tasks,
2x4 panels, hardware limits dashed, full-range y."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
OUT = MAIN/'runs/paper_fill/search_compare'
d = np.load(FU/'cone5_eval_v1.npz')
s5 = np.load(FU/'cone5_sel_10k.npz')
sub, q0f, lpw5 = d['sub'], d['q0_first'], d['lpw5']
p_cls, p_sel, pick = d['p_cls'], s5['prog'], s5['pick']
tz = np.load(A/'tasks_pool_fr3.npz')
gap = p_sel - p_cls
cand = np.argsort(-gap)
CHOSEN = [int(c) for c in cand[:6]]
print('chosen rows:', [(int(sub[c]), round(float(p_cls[c]),2),
                        round(float(p_sel[c]),2)) for c in CHOSEN])

def roll_traj(cfgfile, ckpt, q0s, rows, classical=False):
    y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj'/cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    kw['cone_deg'] = 5.0
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': len(rows)}), None, dev)
    rdt = env.kin.dtype
    if classical:
        fn = cn_action_fn(ClassicalNullspaceController(env.kin))
    else:
        ag = Agent(env.obs_dim, env.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO/ckpt, map_location=dev))
        ag.eval()
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(q0s, dtype=rdt, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][sub[rows]], dtype=rdt, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][sub[rows]], dtype=rdt, device=dev)})
    env.reset()
    Q, S = [env.q.cpu().numpy().copy()], [np.zeros(len(rows), np.float32)]
    with torch.no_grad():
        for _ in range(env.cfg.max_steps//2):
            a = fn(env) if classical else ag.actor_mean(env.current_obs())
            for _ in range(2):
                env.step(a, auto_reset=False)
                Q.append(env.q.cpu().numpy().copy())
                S.append(env.arc_progress.float().cpu().numpy().copy())
            if bool(env.done_persistent.all()):
                break
    del env
    torch.cuda.empty_cache()
    return np.array(Q), np.array(S)

rows = np.array(CHOSEN)
Qc, Sc = roll_traj(str(Path(hl.ROBOTS['fr3'][0]).name), None,
                   q0f[rows], rows, classical=True)
Qr, Sr = roll_traj('config_line_cont_dirfrac_e8k_cone5.yaml',
                   'Yuan/IJRR/runs/rl_dirfrac_e8k_cone5/agent.pt',
                   pick[rows], rows)
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
for k, c in enumerate(CHOSEN):
    ti = int(sub[c])
    fig, axes = plt.subplots(2, 4, figsize=(32, 12))
    for j in range(7):
        ax = axes[j//4, j%4]
        ax.plot(Sc[:, k], Qc[:, k, j], color='crimson', lw=2.2,
                label=f'Classical ({Sc[-1, k]:.2f} m)')
        ax.plot(Sr[:, k], Qr[:, k, j], color='royalblue', lw=2.2,
                label=f'Cone5 RL + critic start ({Sr[-1, k]:.2f} m)')
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
    ax.text(0.05, 0.7, f'Task {ti} (5° cone)', fontsize=26, weight='bold')
    ax.text(0.05, 0.5, f'$\\ell^{{pw}}$(5°) = {lpw5[c]:.2f} m', fontsize=20)
    ax.text(0.05, 0.34, f'Classical: {Sc[-1, k]:.2f} m', fontsize=18, color='crimson')
    ax.text(0.05, 0.2, f'RL + critic start: {Sr[-1, k]:.2f} m', fontsize=18,
            color='royalblue')
    ax.text(0.05, 0.06, 'Starts differ: classical = first admissible;\n'
            'RL = critic-picked (selection is part of the method)', fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/f'cone5_t{ti}_cmp.png', dpi=70)
    plt.close(fig)
    print(f't{ti} plotted', flush=True)
