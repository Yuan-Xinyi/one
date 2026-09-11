"""Can ANY reactive law hold the 1-deg cone? Classical with cranked
orientation gain + tight margin, from certified starts."""
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
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
d = np.load(FU/'cone1_eval_v1.npz')
sub, q0, has, lpw = d['sub'], d['q0_first'], d['has'], d['lpw5']
tz = np.load(A/'tasks_pool_fr3.npz')
y = yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
kw['cone_deg'] = 1.0
N = 2000
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
rdt = env.kin.dtype
for kth, marg in ((2.0, 3.0), (8.0, 1.5), (20.0, 1.0)):
    ctrl = ClassicalNullspaceController(env.kin, angle_boundary_gain=kth,
                                        angle_margin_deg=marg,
                                        theta_max_deg=1.0)
    fn = cn_action_fn(ctrl)
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(q0, dtype=rdt, device=dev),
         'line_dir': torch.tensor(tz['cs_line_dir'][sub], dtype=rdt, device=dev),
         'n_target': torch.tensor(tz['cs_n_target'][sub], dtype=rdt, device=dev)})
    env.reset()
    with torch.no_grad():
        for _ in range(env.cfg.max_steps//2):
            a = fn(env)
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
    p = env.arc_progress.float().cpu().numpy()
    ref = np.maximum(lpw, p)
    rt = p[has]/np.maximum(ref[has], 1e-9)
    print(f'ktheta={kth} margin={marg}deg: stroke {p[has].mean():.3f}  '
          f'ratio {rt.mean()*100:.1f} / {np.percentile(rt,10)*100:.1f}', flush=True)
