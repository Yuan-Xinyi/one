"""Eval the q7-randomized agent: vlook on task 27 + the aligned 10k pool."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent

dev = torch.device('cuda')
hl.SUB = 2
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line_bscale.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
model.terms = [0, 1]
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_bscale_30M', env.obs_dim,
            dev, act_dim=env.act_dim)
vfn = hl.make_vlook(model, env, ag)
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N = tz['cs_p0'].shape[0]
dt = env.kin.dtype
out = np.zeros(N, np.float32)
for lo in range(0, N, B):
    hi = min(lo + B, N); pad = B - (hi - lo)
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
        a = vfn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    print(f'[bscale vlook] {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
np.savez('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
         'bscale_vlook_10k.npz', prog=out)
b = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
            'bound_pool_fr3.npz')
w = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
            'witness_pool_fr3.npz')
base = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
               'pool_fr3_straight.npz')
ref = np.maximum(np.maximum(b['L_hi'], w['prog']), out)
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]:
    ref = np.maximum(ref, base[f'{a2}_progress'])
rt = out / np.maximum(ref, 1e-9)
rt0 = base['vlook_progress'] / np.maximum(ref, 1e-9)
print(f'BASELINE vlook : stroke {base["vlook_progress"].mean():.4f} '
      f'ratio {rt0.mean()*100:.1f} / {np.percentile(rt0,10)*100:.1f}')
print(f'Q7RAND  vlook : stroke {out.mean():.4f} '
      f'ratio {rt.mean()*100:.1f} / {np.percentile(rt,10)*100:.1f}')
print(f'task 27: baseline {base["vlook_progress"][27]:.3f} '
      f'-> q7rand {out[27]:.3f}  (search 1.061)')
