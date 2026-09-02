"""DirFrac (rh2048XXL) zero-shot rollouts on the FR3 curved families,
row-aligned with ctrl_fr3_{fam}.npz (same tasks.pt specs, same env
protocol as fam_unify: k_lateral=5.0, SUB=2, arc_progress objective)."""
import sys, dataclasses, time, os
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
SMOKE = bool(int(os.environ.get('SMOKE', '0')))

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
kw['k_lateral'] = 5.0
B = 128 if SMOKE else 2500
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()

tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                   weights_only=False)


def fam_spec(fam):
    sp = tasks[f'test_{fam}']
    spec = {'q0': sp['q0'].clone(), 'p0': sp['p0'].clone(),
            'line_dir': sp['line_dir'].clone(),
            'n_target': sp['n_target'].clone()}
    for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
        if k in sp:
            spec[k] = sp[k].clone()
    return spec


@torch.no_grad()
def rollout(sub):
    env.line_dist = ScriptedLineDistribution(sub)
    env.reset()
    for _ in range(env.cfg.max_steps // 2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    return env.arc_progress.float().cpu().numpy().copy()


dt = env.kin.dtype
for fam in ('serpentine', 'nonplanar'):
    spec = fam_spec(fam)
    N = 128 if SMOKE else spec['q0'].shape[0]
    out = np.zeros(N, np.float32)
    t0 = time.time()
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        sub = {}
        for k, v in spec.items():
            t = v[lo:hi]
            if pad:
                t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
            sub[k] = t
        for k2 in ('q0', 'line_dir', 'n_target'):
            sub[k2] = sub[k2].to(device=dev, dtype=dt)
        out[lo:hi] = rollout(sub)[:hi - lo]
    base = np.load(FU / f'ctrl_fr3_{fam}.npz')
    ref = base['vlook_progress'][:N]
    print(f'[dirfrac-curved] {fam}: mean {out.mean():.4f} '
          f'(vlook row mean {ref.mean():.4f})  {time.time()-t0:.0f}s',
          flush=True)
    if not SMOKE:
        np.savez_compressed(FU / f'dirfrac_fr3e8k_{fam}_curved.npz', prog=out)
print('done', flush=True)
