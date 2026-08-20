"""E-prep: evaluate the margobs mainline on the TRAIN pool (valid rows)
at train granularity and emit per-pool-row oversampling weights:
w = 1 + 3*[progress < P25]  (worst quartile sampled 4x)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_margobs.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
B = 4096
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
lc = y['line_distribution']
dist = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=lc['n_pool'],
    n_target_noise_deg=lc['n_target_noise_deg'], seed=lc['train_seed'],
    env_cfg=env.cfg,
    feasibility_threshold_m=(float(lc['feasibility_threshold_m'])
                             if lc.get('feasibility_filter') else None),
    swing_max_deg=lc.get('swing_max_deg', 0.0))
ag = Agent(env.obs_dim, env.act_dim_policy).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_margobs_30M/agent.pt',
    map_location=dev))
ag.eval()

valid = torch.nonzero(dist.valid_mask, as_tuple=False).squeeze(-1)
NV = valid.shape[0]
prog = np.zeros(NV, np.float32)
dt = env.kin.dtype
with torch.no_grad():
    for lo in range(0, NV, B):
        hi = min(lo + B, NV)
        pad = B - (hi - lo)
        rows = valid[lo:hi]
        if pad:
            rows = torch.cat([rows, rows[:1].expand(pad)])
        env.line_dist = ScriptedLineDistribution(
            {'q0': dist.q_pool[rows].to(dev, dt),
             'line_dir': dist.line_dir_pool[rows].to(dev, dt),
             'n_target': dist.n_target_pool[rows].to(dev, dt)})
        env.reset()
        for _ in range(env.cfg.max_steps):
            a = ag.actor_mean(env.current_obs())
            env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'{hi}/{NV} mean {prog[:hi].mean():.4f}', flush=True)

p25 = np.percentile(prog, 25)
w_full = np.zeros(int(dist.valid_mask.shape[0]), np.float32)
w_full[valid.cpu().numpy()] = 1.0 + 3.0 * (prog < p25)
np.savez(FU + 'pool_weights.npz', w=w_full, prog=prog,
         valid=valid.cpu().numpy(), p25=p25)
print(f'weights saved: P25={p25:.4f}  oversampled {int((prog < p25).sum())}'
      f'/{NV}', flush=True)
