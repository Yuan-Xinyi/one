import sys, dataclasses, time
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
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_xarm7_e8kXXL_v005.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
B = 2500
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
assert abs(env.v - 0.05) < 1e-9
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_v005/agent.pt', map_location=dev))
ag.eval()
dt_t = env.kin.dtype
tz = np.load(A/'tasks_pool_xarm7.npz')
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
np.savez_compressed(FU/'dirfrac_xarm7v005_10k.npz', prog=out)
base = np.load(FU/'pool_xarm7_straight.npz')
b = np.load(A/'bound_pool_xarm7_v2.npz'); w = np.load(A/'witness_pool_xarm7.npz')
ref = np.maximum(b['L_hi'], w['prog'])
for k in base.files:
    if k.endswith('_progress'):
        ref = np.maximum(ref, base[k])
v2 = np.load(FU/'dirfrac_xarm7e8k_straight.npz')['prog']
for tag, v in (('flagship v=0.20', v2), ('v=0.05        ', out)):
    r = np.maximum(ref, np.maximum(v2, out)); rt = v/np.maximum(r,1e-9)
    print(f'{tag}: stroke {v.mean():.4f}  ratio {rt.mean()*100:.2f} / {np.percentile(rt,10)*100:.2f}')
d = out - v2
print(f'paired: improved>1cm {(d>0.01).mean()*100:.1f}%  hurt>1cm {(d<-0.01).mean()*100:.1f}%  mean diff {d.mean()*100:+.1f} cm')
