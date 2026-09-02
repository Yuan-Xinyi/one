"""Cobotta: roll a given config/ckpt on the 10k pool, save npz. argv: tag"""
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
tag = sys.argv[1]      # 'e8kXXL' or 'e8kXXL_v005'
dev = torch.device('cuda')
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
y = yaml.safe_load(open(REPO/f'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_cobotta_{tag}.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
B = 2500
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/f'Yuan/IJRR/runs/rl_dirfrac_cobotta_{tag}/agent.pt', map_location=dev))
ag.eval()
dt_t = env.kin.dtype
tz = np.load(A/'tasks_pool_cobotta.npz')
N = len(tz['q0_seed'])
out = np.zeros(N, np.float32)
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo+B, N); pad = B-(hi-lo)
        sub = {'q0': torch.tensor(tz['q0_seed'][lo:hi], dtype=dt_t),
               'line_dir': torch.tensor(tz['cs_line_dir'][lo:hi], dtype=dt_t),
               'n_target': torch.tensor(tz['cs_n_target'][lo:hi], dtype=dt_t)}
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
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
np.savez_compressed(FU/f'dirfrac_cobotta{tag}_10k.npz', prog=out)
print(f'{tag}: mean stroke {out.mean():.4f}', flush=True)
