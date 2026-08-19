"""Two aggregate curves for the exit_multi generalization test:
(a) performance vs search width: unguided vs final-policy-guided ladder
(b) performance vs iteration: search and policy per round
All per-task values normalized by that task's UNGUIDED W=16384 result
(the vanilla full-width search reference)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RUN = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05'
           '/Yuan/IJRR/runs/single_task_ppo_v2')
d = np.load(RUN / 'exit_multi.npz', allow_pickle=True)
res = d['results'][0]
widths = list(d['widths'])            # 16384..64 (loop rounds)
ladder_ws = list(d['ladder_ws'])      # 256..1
tasks = list(d['tasks'])

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

# ---- (a) performance vs width ----
ax = axes[0]
uw = np.array([[res[t]['unguided'][W] for W in widths] for t in tasks])
lw = np.array([[res[t]['ladder'][W] for W in ladder_ws] for t in tasks])
ref = uw[:, 0]                        # unguided W=16384 per task
un = uw / ref[:, None]
ld = lw / ref[:, None]
for i, t in enumerate(tasks):
    ax.plot(widths, un[i], c='tab:gray', alpha=0.35, lw=0.9)
    ax.plot(ladder_ws, ld[i], c='tab:red', alpha=0.35, lw=0.9)
ax.plot(widths, un.mean(0), 'o-', c='black', lw=2.2,
        label='unguided search (mean of 10 tasks)')
ax.plot(ladder_ws, ld.mean(0), 's-', c='tab:red', lw=2.2,
        label='guided by final policy (mean)')
ax.axhline(1.0, c='gray', ls=':', lw=1)
ax.set_xscale('log')
ax.set_xlabel('search width W (log)')
ax.set_ylabel('progress / unguided-W16384 (per task)')
ax.set_title('(a) performance vs search width')
ax.legend(frameon=False, fontsize=9, loc='lower right')
ax.grid(alpha=0.25)

# ---- (b) performance vs iteration ----
ax = axes[1]
rs = np.array([[r[1] for r in res[t]['rounds']] for t in tasks]) / ref[:, None]
rp = np.array([[r[2] for r in res[t]['rounds']] for t in tasks]) / ref[:, None]
it = np.arange(len(widths))
for i, t in enumerate(tasks):
    ax.plot(it, rp[i], c='tab:red', alpha=0.3, lw=0.9)
ax.plot(it, rs.mean(0), 'o-', c='tab:blue', lw=2.2,
        label='search of that round (mean)')
ax.plot(it, rp.mean(0), 's-', c='tab:red', lw=2.2,
        label='policy after BC (mean)')
ax.axhline(1.0, c='gray', ls=':', lw=1)
for k, W in enumerate(widths):
    ax.annotate(f'W={W}', (it[k], rs.mean(0)[k]), textcoords='offset points',
                xytext=(0, -14), fontsize=8, ha='center', color='tab:blue')
ax.set_xticks(it)
ax.set_xlabel('iteration (search width shrinking)')
ax.set_ylabel('progress / unguided-W16384')
ax.set_title('(b) performance vs iteration')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(alpha=0.25)

fig.suptitle('iterative search-and-distill on 10 stratified tasks '
             '(thin lines = per task; dotted = vanilla full-width search '
             'reference)', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = RUN / 'exit_multi_curves.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)

# summary table
print(f"\n{'task':>5} {'myopic':>7} {'L_hi':>6} {'ung16384':>9} "
      f"{'ung64':>7} {'gui64':>7} {'gui1':>6} {'gui64/ung16384':>14}")
for i, t in enumerate(tasks):
    r = res[t]
    print(f"{t:>5} {r['myopic']:>7.3f} {r['L_hi']:>6.3f} "
          f"{uw[i,0]:>9.3f} {uw[i,-1]:>7.3f} {lw[i,1]:>7.3f} "
          f"{lw[i,-1]:>6.3f} {lw[i,1]/uw[i,0]:>14.2f}")
print(f"\nguided-W64 >= unguided-W16384 on "
      f"{(lw[:,1] >= uw[:,0] - 1e-3).sum()}/10 tasks; "
      f"mean ratio {np.mean(lw[:,1]/uw[:,0]):.2f}")
