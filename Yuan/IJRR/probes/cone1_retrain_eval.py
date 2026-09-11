"""Roll the cone5-retrained policy from the saved 5-deg starts."""
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
FU = MAIN/'runs/paper_fill/fam_unify'
d = np.load(FU/'cone1_eval_v1.npz')
q0, has, lpw5 = d['q0_first'], d['has'], d['lpw5']
sub = d['sub']
A = MAIN/'runs/paper_fill/ratio_assets'
tz = np.load(A/'tasks_pool_fr3.npz')
dd = tz['cs_line_dir'][sub].astype(np.float32)
nt = tz['cs_n_target'][sub].astype(np.float32)
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone1.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
B = 2000
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8k_cone1/agent.pt', map_location=dev))
ag.eval()
rdt = env.kin.dtype
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(q0, dtype=rdt, device=dev),
     'line_dir': torch.tensor(dd, dtype=rdt, device=dev),
     'n_target': torch.tensor(nt, dtype=rdt, device=dev)})
env.reset()
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
p = env.arc_progress.float().cpu().numpy()
np.savez(FU/'cone1_retrain_10k.npz', prog=p)
ref = np.maximum.reduce([lpw5, d['p_cls'], d['p_rl'], p])
for tag, v in (('classical', d['p_cls']), ('flagship-0shot', d['p_rl']),
               ('cone5-retrain', p)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:15s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
