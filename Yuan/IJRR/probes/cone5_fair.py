"""Fair rerun: classical rolled from the SAME critic-picked starts,
aggregate over the 2000 tasks + trajectories for the 6 showcase tasks."""
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
d = np.load(FU/'cone5_eval_v1.npz'); s5 = np.load(FU/'cone5_sel_10k.npz')
sub, has, lpw5, pick = d['sub'], d['has'], d['lpw5'], s5['pick']
tz = np.load(A/'tasks_pool_fr3.npz')
y = yaml.safe_load(open(REPO/hl.ROBOTS['fr3'][0]))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
kw['cone_deg'] = 5.0
N = 2000
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
rdt = env.kin.dtype
fn = cn_action_fn(ClassicalNullspaceController(env.kin))
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(pick, dtype=rdt, device=dev),
     'line_dir': torch.tensor(tz['cs_line_dir'][sub], dtype=rdt, device=dev),
     'n_target': torch.tensor(tz['cs_n_target'][sub], dtype=rdt, device=dev)})
env.reset()
QT, ST = [env.q.cpu().numpy().copy()], [np.zeros(N, np.float32)]
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = fn(env)
        for _ in range(2):
            env.step(a, auto_reset=False)
            QT.append(env.q.cpu().numpy().copy())
            ST.append(env.arc_progress.float().cpu().numpy().copy())
        if bool(env.done_persistent.all()):
            break
p = ST[-1]
np.savez(FU/'cone5_cls_pickstart_10k.npz', prog=p)
pr = np.load(FU/'cone5_retrain_10k.npz')['prog']
ps = s5['prog']
ref = np.maximum.reduce([lpw5, d['p_cls'], p, pr, ps])
for tag, v in (('classical @first', d['p_cls']), ('classical @critic-start', p),
               ('RL @first', pr), ('RL @critic-start', ps)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:24s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
# save showcase trajectories for replot
TASKS = [1046, 4677, 7006, 6001, 3916, 2552]
rows = {int(s): i for i, s in enumerate(sub)}
idx = [rows[t] for t in TASKS]
np.savez(FU/'cone5_cls_pick_traj.npz',
         q=np.array(QT)[:, idx], s=np.array(ST)[:, idx], tasks=np.array(TASKS))
print('traj saved', flush=True)
