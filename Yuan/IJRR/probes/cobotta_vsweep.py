"""Cobotta v-sweep counterfactual: at the SAME visited states, scale the
task speed and recompute saturation + alpha_feas (qdot_task is linear in v)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
MAIN = Path('/home/lqin/one/Yuan/IJRR')
dev = torch.device('cuda'); N = 1024
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_cobotta.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_cobotta_XXL/agent.pt', map_location=dev))
ag.eval()
dt_t = env.kin.dtype; nj = env.n_joints; lim = env.qd_limit
tz = np.load(MAIN/'runs/paper_fill/ratio_assets/tasks_pool_cobotta.npz')
env.line_dist = ScriptedLineDistribution({
    'q0': torch.tensor(tz['q0_seed'][:N], dtype=dt_t, device=dev),
    'line_dir': torch.tensor(tz['cs_line_dir'][:N], dtype=dt_t, device=dev),
    'n_target': torch.tensor(tz['cs_n_target'][:N], dtype=dt_t, device=dev)})
env.reset()
QT, XI = [], []
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            act = ~env.done_persistent
            if not bool(act.any()): break
            aa = a.clamp(-1,1).to(dtype=env.kin.dtype)
            _,_,J,_ = env.kin.tcp_fk_jac(env.q); J_p = J[:,:3,:]
            J_plus,_ = damped_pinv(J_p, env.cfg.lambda_0, env.cfg.sigma_thr)
            qt = (J_plus @ (env.v*env.line_dir).unsqueeze(-1)).squeeze(-1)
            _,_,Vh = torch.linalg.svd(J_p.double(), full_matrices=True)
            Nn = Vh.transpose(-1,-2)[...,3:]
            xi = (Nn @ (Nn.transpose(-1,-2) @ aa[:,:nj].double().unsqueeze(-1))).squeeze(-1)
            xi = (xi/xi.norm(dim=-1,keepdim=True).clamp_min(1e-6)).to(env.kin.dtype)
            QT.append(qt[act].cpu()); XI.append(xi[act].cpu())
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()): break
QT = torch.cat(QT); XI = torch.cat(XI); liml = lim.cpu()
print(f'{len(QT)} substeps  (baseline v = {env.v} m/s)')
print(f'{"v(m/s)":>8} {"sat mean":>9} {"sat p90":>8} {">100%":>7} {"alpha med":>10} {"alpha<0.05":>11}')
for s in (1.0, 0.5, 0.35, 0.25, 0.115, 0.0585):
    qt = QT*s
    sat = (qt.abs()/liml).amax(-1)
    head = (liml-qt)/XI.clamp_min(1e-9); room = (liml+qt)/(-XI).clamp_min(1e-9)
    al = torch.where(XI>=0, head, room).amin(-1).clamp_min(0)
    print(f'{0.2*s:8.3f} {sat.mean():9.3f} {np.percentile(sat.numpy(),90):8.3f} '
          f'{(sat>1).float().mean()*100:6.1f}% {al.median():10.3f} '
          f'{(al<0.05).float().mean()*100:10.1f}%')
