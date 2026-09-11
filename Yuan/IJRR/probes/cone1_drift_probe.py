"""Zero-action drift probe at 1-deg cone: from certified starts, how fast
does pure position tracking rotate the tool axis out of the cone?"""
import sys, dataclasses, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
dev = torch.device('cuda')
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
d = np.load(FU/'cone1_eval_v1.npz')
sub, q0, has = d['sub'], d['q0_first'], d['has']
tz = np.load(A/'tasks_pool_fr3.npz')
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone1.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 400
N = 2000
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
rdt = env.kin.dtype
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(q0, dtype=rdt, device=dev),
     'line_dir': torch.tensor(tz['cs_line_dir'][sub], dtype=rdt, device=dev),
     'n_target': torch.tensor(tz['cs_n_target'][sub], dtype=rdt, device=dev)})
env.reset()
nt = env.n_target.clone()
ang0 = torch.rad2deg(torch.acos(
    (env.kin.tcp_fk_jac(env.q)[1][:, :, 2]*nt).sum(-1).clamp(-1, 1)))
a = torch.zeros(N, env.act_dim_policy, device=dev, dtype=rdt)
angs = [ang0.cpu().numpy()]
alive = []
with torch.no_grad():
    for t in range(6):
        env.step(a, auto_reset=False)
        ang = torch.rad2deg(torch.acos(
            (env.kin.tcp_fk_jac(env.q)[1][:, :, 2]*nt).sum(-1).clamp(-1, 1)))
        angs.append(ang.cpu().numpy())
        alive.append(float((~env.done_persistent).float().mean()))
angs = np.array(angs)[:, has]
print('tool-axis angle to cone axis (deg), median over tasks:')
for t in range(7):
    print(f'  after {t} substeps: {np.median(angs[t]):.2f}  p90 {np.percentile(angs[t],90):.2f}')
print('alive fraction per substep:', [f'{x:.2f}' for x in alive])
