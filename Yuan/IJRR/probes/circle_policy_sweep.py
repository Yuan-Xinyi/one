"""Policy-side radius sweep: for each scenario, roll the flagship on
circles of every grid radius <= its r_pw (+1 step); largest closed r.
Uses circle_rpw_v1.npz scenarios; batched across (scenario, radius)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
d = np.load(FU/'circle_rpw_v1.npz')
sc, r_pw = d['scenario'], d['r_pw']
tz = np.load(A/'tasks_sel_arc.npz')
GRID = np.round(np.arange(0.05, 0.551, 0.025), 3)
jobs = []
for i, t in enumerate(sc):
    for r in GRID:
        if r <= r_pw[i] + 0.026:
            jobs.append((i, r))
jobs = np.array(jobs, dtype=np.float64)
print(f'{len(jobs)} rollouts for {len(sc)} scenarios', flush=True)
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 2000; kw['k_lateral'] = 5.0
B = 1024
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
dt_t = env.kin.dtype
prog = np.zeros(len(jobs), np.float32)
with torch.no_grad():
    for lo in range(0, len(jobs), B):
        hi = min(lo+B, len(jobs)); pad = B-(hi-lo)
        rows = sc[jobs[lo:hi, 0].astype(int)]
        rr = jobs[lo:hi, 1]
        sub = {'q0': torch.tensor(tz['q0_seed'][rows], dtype=dt_t),
               'line_dir': torch.tensor(tz['cs_line_dir'][rows], dtype=dt_t),
               'n_target': torch.tensor(tz['cs_n_target'][rows], dtype=dt_t),
               'kappa': torch.tensor(1.0/rr, dtype=dt_t)}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in sub.items()}
        sub = {k: v.to(dev) for k, v in sub.items()}
        env.line_dist = ScriptedLineDistribution(sub)
        env.reset()
        for _ in range(env.cfg.max_steps//2):
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
        print(f'{hi}/{len(jobs)}', flush=True)
closed = prog >= 2*np.pi*jobs[:, 1] - 1e-3
r_best = np.zeros(len(sc), np.float32)
for j, (i, r) in enumerate(jobs):
    if closed[j]:
        r_best[int(i)] = max(r_best[int(i)], r)
np.savez(FU/'circle_policy_sweep_v1.npz', scenario=sc, r_best=r_best,
         r_pw=r_pw, jobs=jobs, prog=prog)
ok = r_pw > 0
rt = r_best[ok]/np.maximum(r_pw[ok], 1e-9)
print(f'zero-shot: closed any {(r_best>0).mean()*100:.1f}%  '
      f'r_best/r_pw mean {rt.mean()*100:.1f} / p10 {np.percentile(rt,10)*100:.1f}', flush=True)
