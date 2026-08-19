"""Margin-triggered rescue on task 27: RL drives; when the minimum
normalized margin drops below tau_e, a myopic one-step lookahead targeting
the endangered term takes over; hands back above tau_x (hysteresis).

Rescuers: 'targeted' = myopic on the current argmin term only
          'deployed' = myopic on jl+cone (the deployed combination)
          'all4'     = myopic on all four terms"""
import matplotlib                       # must precede torch on this box
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
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

model_all = hl.StraightModel(env)               # margins: jl/cone/lat/coll
MARGIN_NAMES = ['jl', 'cone', 'lat', 'coll']
arms = {}
for i in range(4):                              # per-term myopic rescuers
    m = hl.StraightModel(env)
    m.terms = [i]
    arms[i] = hl.make_myopic(m)
m2 = hl.StraightModel(env)
m2.terms = [0, 1]
myo_deployed = hl.make_myopic(m2)
m4 = hl.StraightModel(env)
myo_all4 = hl.make_myopic(m4)


def margins_now():
    return model_all.margins(env.q[:1], env.p_start[:1], env.line_dir[:1],
                             env.n_target[:1])[0]


@torch.no_grad()
def run(tau_e, tau_x, rescuer):
    env.reset()
    rescued, n_rescues, rescue_steps, trace = False, 0, 0, []
    for t in range(env.max_steps):
        m = margins_now()
        mn, am = float(m.min()), int(m.argmin())
        if not rescued and mn < tau_e:
            rescued = True
            n_rescues += 1
        elif rescued and mn > tau_x:
            rescued = False
        trace.append((t, mn, am, rescued))
        if rescued:
            rescue_steps += 1
            if rescuer == 'targeted':
                a = arms[am](env, env.done_persistent)
            elif rescuer == 'deployed':
                a = myo_deployed(env, env.done_persistent)
            else:
                a = myo_all4(env, env.done_persistent)
        else:
            a = agent.actor_mean(env.current_obs())
        _, _, _, _, info = env.step(a, auto_reset=False)
        if bool(info['episode_done'][0]):
            term = int(info['term_reason'][0])
            break
    else:
        term = TERM_TRUNCATED
    pf, _, _, _ = env.kin.tcp_fk_jac(env.q[:1])
    prog = float(((pf[0] - env.p_start[0]) * env.line_dir[0]).sum())
    return prog, term, t + 1, n_rescues, rescue_steps, trace


# reference: pure PPO margin trace (tau_e < 0 never triggers)
prog0, term0, T0, _, _, trace0 = run(-1.0, -1.0, 'targeted')
mins0 = np.array([r[1] for r in trace0])
print(f"pure PPO: {prog0:.4f} m, {T0} steps, term {TERM_NAMES[term0]}; "
      f"min-margin percentiles p10 {np.percentile(mins0, 10):.3f} "
      f"p50 {np.percentile(mins0, 50):.3f} min {mins0.min():.3f}")

print("\nrescuer   tau_e  tau_x   final(m)  steps  rescues  rescue_steps  term")
best = (None, -1.0)
for rescuer in ('targeted', 'deployed', 'all4'):
    for te, tx in [(0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25),
                   (0.25, 0.35)]:
        prog, term, steps, nr, rs, trace = run(te, tx, rescuer)
        print(f"{rescuer:<9s} {te:.2f}  {tx:.2f}   {prog:.4f}   {steps:>4d}"
              f"  {nr:>6d}  {rs:>10d}   {TERM_NAMES[term]}", flush=True)
        if prog > best[1]:
            best = ((rescuer, te, tx, trace, term, nr), prog)

(rescuer, te, tx, trace, term, nr), prog = best
print(f"\nbest: {rescuer} tau {te}/{tx} -> {prog:.4f} m "
      f"(pure PPO {prog0:.4f}, best search 1.0614)")

t_arr = np.array([r[0] for r in trace]) * 0.01
mn_arr = np.array([r[1] for r in trace])
resc = np.array([r[3] for r in trace])
fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.plot(np.arange(len(mins0)) * 0.01, mins0, c='tab:blue', lw=1.2,
        label=f'pure PPO min margin ({prog0:.2f} m)')
ax.plot(t_arr, mn_arr, c='black', lw=1.4,
        label=f'best rescue run ({prog:.2f} m)')
ax.fill_between(t_arr, 0, mn_arr.max() * 1.05, where=resc, color='tab:red',
                alpha=0.15, label='myopic rescue engaged')
ax.axhline(te, c='tab:red', ls='--', lw=1, label=f'tau_e {te}')
ax.axhline(tx, c='tab:orange', ls='--', lw=1, label=f'tau_x {tx}')
ax.set_xlabel('arc length s (m)')
ax.set_ylabel('min normalized margin')
ax.set_title(f'task 27, margin-triggered myopic rescue: best = {rescuer} '
             f'tau {te}/{tx}, {nr} rescues, final {prog:.3f} m '
             f'({TERM_NAMES[term]})')
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
out = RUN / 'margin_rescue.png'
fig.savefig(out, dpi=150)
print('wrote', out)
