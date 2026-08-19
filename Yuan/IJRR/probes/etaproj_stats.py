"""Item 5 (stats only): projection retention eta = |P_N u| / |u| of the
dirfrac v2 actor's raw direction output, measured on eval-pool rollouts.
If eta << 1 on many states the actor wastes capacity on components the
projector discards."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_v2.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_dirfrac_v2_30M/agent.pt',
    map_location=dev))
ag.eval()
dt = env.kin.dtype
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(tz['q0_seed'][:B], dtype=dt, device=dev),
     'line_dir': torch.tensor(tz['cs_line_dir'][:B], dtype=dt, device=dev),
     'n_target': torch.tensor(tz['cs_n_target'][:B], dtype=dt, device=dev)})
env.reset()
etas, rhos = [], []
with torch.no_grad():
    for t in range(env.cfg.max_steps):
        a = ag.actor_mean(env.current_obs())
        u = a[:, :7].to(dt)
        _, _, J, _ = env.kin.tcp_fk_jac(env.q)
        _, _, Vh = torch.linalg.svd(J[:, :3, :].double(), full_matrices=True)
        Nn = Vh.transpose(-1, -2)[..., 3:]
        pu = (Nn @ (Nn.transpose(-1, -2) @ u.double().unsqueeze(-1))
              ).squeeze(-1)
        eta = (pu.norm(dim=-1) / u.double().norm(dim=-1).clamp_min(1e-9))
        alive = ~env.done_persistent
        etas.append(eta[alive].float().cpu().numpy())
        rhos.append((0.5 * (a[alive, 7] + 1)).float().cpu().numpy())
        env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
e = np.concatenate(etas); r = np.concatenate(rhos)
qs = [1, 5, 10, 25, 50, 75, 90, 99]
print(f'eta_proj over {len(e)} states: mean {e.mean():.3f}')
print('  pct ', {q: round(float(np.percentile(e, q)), 3) for q in qs})
print(f'  frac eta<0.5: {(e < 0.5).mean()*100:.1f}%   '
      f'eta<0.2: {(e < 0.2).mean()*100:.1f}%   eta<0.05: '
      f'{(e < 0.05).mean()*100:.2f}%')
print(f'rho: mean {r.mean():.3f}  pct ',
      {q: round(float(np.percentile(r, q)), 3) for q in qs})
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
         'etaproj_stats.npz', eta=e, rho=r)
