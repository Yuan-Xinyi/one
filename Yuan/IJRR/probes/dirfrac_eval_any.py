"""Parametrized 10k actor-only eval for any dirfrac config/ckpt.
argv: <config.yaml rel to stage2_traj> <ckpt path> <out npz name> <tag>"""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

cfgfile, ckpt, outname, tag = sys.argv[1:5]
dev = torch.device('cuda')
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N, B = 10000, 2048
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy).to(dev)
ag.load_state_dict(torch.load(ckpt, map_location=dev))
ag.eval()

out = np.zeros(N, np.float32)
dt = env.kin.dtype
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        ids = np.arange(lo, hi)
        ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        env.line_dist = ScriptedLineDistribution(
            {'q0': torch.tensor(tz['q0_seed'][ip], dtype=dt, device=dev),
             'line_dir': torch.tensor(tz['cs_line_dir'][ip], dtype=dt,
                                      device=dev),
             'n_target': torch.tensor(tz['cs_n_target'][ip], dtype=dt,
                                      device=dev)})
        env.reset()
        for _ in range(env.cfg.max_steps // 2):
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'{tag} {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)

np.savez(FU + outname, prog=out)
A = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
b = np.load(A + 'bound_pool_fr3.npz')
w = np.load(A + 'witness_pool_fr3.npz')
base = np.load(FU + 'pool_fr3_straight.npz')
v2 = np.load(FU + 'dirfrac_v2_10k.npz')['prog']
ref = np.maximum(b['L_hi'], w['prog'])
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]:
    ref = np.maximum(ref, base[f'{a2}_progress'])
ref = np.maximum(np.maximum(ref, v2), out)
def stat(v, t):
    rt = v / np.maximum(ref, 1e-9)
    print(f'{t}: {v.mean():.4f}  {rt.mean()*100:.1f} / '
          f'{np.percentile(rt, 10)*100:.1f}   t27 {v[27]:.3f}', flush=True)
stat(out, tag)
stat(v2, 'dirfrac v2 baseline')
d = out - v2
print(f'{tag}-v2: improved {(d > 0.02).mean()*100:.1f}%  '
      f'hurt {(d < -0.02).mean()*100:.1f}%', flush=True)
