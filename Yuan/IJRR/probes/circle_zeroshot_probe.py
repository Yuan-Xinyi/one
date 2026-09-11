"""Zero-shot full-circle probe: roll the straight-trained flagship on the
2500 arc tasks with the episode cap lifted; report closure fraction
arc/(2*pi*r) and compare with the pointwise bound."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
A = MAIN/'runs/paper_fill/ratio_assets'; FU = MAIN/'runs/paper_fill/fam_unify'
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = 3000; kw['k_lateral'] = 5.0
B = 1250
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
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
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
circ = 2*np.pi/np.abs(tz['kappa'])
frac = prog/circ
b = np.load(A/'bound_sel_arc.npz'); w = np.load(A/'witness_sel_arc.npz')
lpw = np.maximum(np.maximum(b['L_hi'], w['prog']), prog)
np.savez(FU/'circle_zeroshot_probe.npz', prog=prog, circ=circ, lpw=lpw)
print(f'closure fraction: mean {frac.mean():.3f}  med {np.median(frac):.3f}  '
      f'p90 {np.percentile(frac,90):.3f}  max {frac.max():.3f}')
print(f'closed (frac>=1): {(frac>=1).mean()*100:.2f}%')
print(f'pointwise closable (lpw>=circ): {(lpw>=circ).mean()*100:.2f}%')
m = lpw >= circ
if m.any():
    print(f'on pointwise-closable tasks: policy closure {(frac[m]>=1).mean()*100:.1f}%, '
          f'frac mean {frac[m].mean():.3f}')
q = np.abs(tz['kappa'])
for lo_k, hi_k in ((0.5,1.0),(1.0,2.0),(2.0,3.0)):
    mm = (q>=lo_k)&(q<hi_k)
    print(f'kappa {lo_k}-{hi_k} (r {1/hi_k:.2f}-{1/lo_k:.2f}m): frac mean {frac[mm].mean():.3f}  '
          f'closed {(frac[mm]>=1).mean()*100:.1f}%  pw-closable {(lpw[mm]>=circ[mm]).mean()*100:.1f}%')
