"""Classical gradient controller on the 2500 arc tasks, full-circle cap
lifted: closure fractions to compare with the flagship zero-shot probe."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
y = yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 3000; kw['k_lateral'] = 5.0
B = 1250
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
fn = cn_action_fn(ClassicalNullspaceController(env.kin))
dt_t = env.kin.dtype
tz = np.load(A/'tasks_sel_arc.npz')
N = len(tz['q0_seed'])
prog = np.zeros(N, np.float32)
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo+B, N); pad = B-(hi-lo)
        sub = {'q0': torch.tensor(tz['q0_seed'][lo:hi], dtype=dt_t),
               'line_dir': torch.tensor(tz['cs_line_dir'][lo:hi], dtype=dt_t),
               'n_target': torch.tensor(tz['cs_n_target'][lo:hi], dtype=dt_t),
               'kappa': torch.tensor(tz['kappa'][lo:hi], dtype=dt_t)}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in sub.items()}
        sub = {k: v.to(dev) for k, v in sub.items()}
        env.line_dist = ScriptedLineDistribution(sub)
        env.reset()
        for _ in range(env.cfg.max_steps//2):
            a = fn(env)
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
circ = 2*np.pi/np.abs(tz['kappa'])
frac = prog/circ
z = np.load(FU/'circle_zeroshot_probe.npz')
lpw = z['lpw']; fr_rl = z['prog']/circ
np.savez(FU/'circle_classical_probe.npz', prog=prog)
m = lpw >= circ
print(f'classical: closure frac mean {frac.mean():.3f} med {np.median(frac):.3f} '
      f'closed {(frac>=1).mean()*100:.2f}%')
print(f'on pointwise-closable ({m.sum()} tasks): classical closed {(frac[m]>=1).mean()*100:.1f}% '
      f'(frac mean {frac[m].mean():.2f})  vs RL {(fr_rl[m]>=1).mean()*100:.1f}% '
      f'(frac mean {fr_rl[m].mean():.2f})')
