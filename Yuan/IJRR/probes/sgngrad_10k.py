"""Class-0 analytic law on the 10k pool: a = sgn(B^T grad Phi) at the
CURRENT state -- current-state kinematic analytics only, no one-step
forward simulation. Completes the privilege-class ladder."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution

dev = torch.device('cuda')
hl.SUB = 2
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N, B = 10000, 2048
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
y = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
model.terms = [0, 1]
fn = hl.make_sgngrad(model)

out = np.zeros(N, np.float32)
dt = env.kin.dtype
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
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(env.cfg.max_steps // 2):
        a = fn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    print(f'sgngrad {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
np.savez(FU + 'sgngrad_10k.npz', prog=out)
