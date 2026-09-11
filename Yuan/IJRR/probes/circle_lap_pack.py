"""Render pack for the multi-lap circle rollout (task from tasks_sel_arc)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.path_geometry import arc_point
from Yuan.IJRR.stage2_traj.ppo import Agent
TI = int(sys.argv[1])
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'
OUT = MAIN/'runs/paper_fill/search_compare'
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 3000; kw['k_lateral'] = 5.0
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
dt_t = env.kin.dtype
tz = np.load(A/'tasks_sel_arc.npz')
sub = {'q0': torch.tensor(tz['q0_seed'][TI][None], dtype=dt_t, device=dev),
       'line_dir': torch.tensor(tz['cs_line_dir'][TI][None], dtype=dt_t, device=dev),
       'n_target': torch.tensor(tz['cs_n_target'][TI][None], dtype=dt_t, device=dev),
       'kappa': torch.tensor(tz['kappa'][TI][None], dtype=dt_t, device=dev)}
env.line_dist = ScriptedLineDistribution(sub)
env.reset()
QS, SS = [env.q[0].cpu().numpy().copy()], [0.0]
with torch.no_grad():
    for _ in range(env.cfg.max_steps//2):
        a = ag.actor_mean(env.current_obs())
        for _ in range(2):
            env.step(a, auto_reset=False)
            QS.append(env.q[0].cpu().numpy().copy())
            SS.append(float(env.arc_progress[0]))
        if bool(env.done_persistent.all()):
            break
QS, SS = np.array(QS), np.array(SS)
keep = np.concatenate([[True], np.diff(SS) > 1e-9])
QS, SS = QS[keep], SS[keep]
g = np.arange(0.0, SS[-1], 0.03)
QG = np.stack([np.interp(g, SS, QS[:, j]) for j in range(7)], 1).astype(np.float32)
q_t = torch.tensor(QG, dtype=dt_t, device=dev)
p, R, _, _ = env.kin.tcp_fk_jac(q_t)
# circle geometry for drawing
p0 = torch.tensor(tz['cs_p0'][TI][None], dtype=dt_t, device=dev)
d0 = torch.tensor(tz['cs_line_dir'][TI][None], dtype=dt_t, device=dev)
d0 = d0/d0.norm()
nt = torch.tensor(tz['cs_n_target'][TI][None], dtype=dt_t, device=dev)
nt = nt/nt.norm()
kap = torch.tensor(tz['kappa'][TI][None], dtype=dt_t, device=dev)
s_ring = torch.linspace(0, float(2*np.pi/abs(tz['kappa'][TI])), 145,
                        dtype=dt_t, device=dev)
ring = arc_point(p0.expand(145, 3), d0.expand(145, 3), nt.expand(145, 3),
                 kap.expand(145), s_ring).cpu().numpy()
np.savez_compressed(OUT/f'circle_t{TI}_pack.npz',
                    q=QG, s=g.astype(np.float32),
                    tip=p.cpu().numpy().astype(np.float32),
                    zax=R[:, :, 2].cpu().numpy().astype(np.float32),
                    ring=ring.astype(np.float32),
                    circ=np.float32(2*np.pi/abs(tz['kappa'][TI])))
print(f'pack: {len(QG)} frames, {SS[-1]:.2f} m, {SS[-1]*abs(tz["kappa"][TI])/(2*np.pi):.2f} laps')
