"""Tight-cone selection leg: score the saved 5-deg candidate pools with
the cone5-retrained critic, roll the picked start per task."""
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
FU = MAIN/'runs/paper_fill/fam_unify'
A = MAIN/'runs/paper_fill/ratio_assets'
d = np.load(FU/'cone1_eval_v1.npz')
sub, has, lpw5 = d['sub'], d['has'], d['lpw5']
CQ, CT = d['cands_q'], d['cands_t']
tz = np.load(A/'tasks_pool_fr3.npz')
dd = tz['cs_line_dir'][sub].astype(np.float32)
nt = tz['cs_n_target'][sub].astype(np.float32)
N = len(sub)
y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8k_cone1.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO/'Yuan/IJRR/runs/rl_dirfrac_e8k_cone1/agent.pt', map_location=dev))
ag.eval()
rdt = env.kin.dtype
V = np.zeros(len(CQ), np.float32)
with torch.no_grad():
    for lo in range(0, len(CQ), B):
        hi = min(lo+B, len(CQ)); pad = B-(hi-lo)
        ids = CT[lo:hi]
        s2 = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt),
              'line_dir': torch.tensor(dd[ids], dtype=rdt),
              'n_target': torch.tensor(nt[ids], dtype=rdt)}
        if pad:
            s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
        s2 = {k: v.to(dev) for k, v in s2.items()}
        env.line_dist = ScriptedLineDistribution(s2)
        env.reset()
        V[lo:hi] = ag.get_value(env.current_obs()).float().cpu().numpy()[:hi-lo]
pick = d['q0_first'].copy()
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    pick[CT[lo]] = CQ[lo + int(np.argmax(V[lo:hi]))]
    lo = hi
env.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(pick[:B] if N <= B else pick, dtype=rdt, device=dev)[:N] if False else torch.tensor(pick, dtype=rdt, device=dev),
     'line_dir': torch.tensor(dd, dtype=rdt, device=dev),
     'n_target': torch.tensor(nt, dtype=rdt, device=dev)})
env2 = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
env2.line_dist = ScriptedLineDistribution(
    {'q0': torch.tensor(pick, dtype=rdt, device=dev),
     'line_dir': torch.tensor(dd, dtype=rdt, device=dev),
     'n_target': torch.tensor(nt, dtype=rdt, device=dev)})
env2.reset()
with torch.no_grad():
    for _ in range(env2.cfg.max_steps//2):
        a = ag.actor_mean(env2.current_obs())
        for _ in range(2):
            env2.step(a, auto_reset=False)
        if bool(env2.done_persistent.all()):
            break
p = env2.arc_progress.float().cpu().numpy()
np.savez(FU/'cone1_sel_10k.npz', prog=p, pick=pick)
pr = np.load(FU/'cone1_retrain_10k.npz')['prog']
ref = np.maximum.reduce([lpw5, d['p_cls'], d['p_rl'], pr, p])
for tag, v in (('classical', d['p_cls']), ('retrain-first', pr), ('retrain+critic', p)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:15s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
