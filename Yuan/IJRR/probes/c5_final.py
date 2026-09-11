"""Formal tight-cone finale: cone5 flagship @shared start, critic-picked
start (both controllers), 500-task pool oracle. Full 10k."""
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
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
FU = MAIN/'runs/paper_fill/fam_unify'; A = MAIN/'runs/paper_fill/ratio_assets'
z = np.load(FU/'cone5_full10k_assets.npz')
sub, q0f, has, lpw5 = z['sub'], z['q0_first'], z['has'], z['lpw5']
p_cls, p_z30 = z['p_cls'], z['p_rl']
CQ, CT = z['cands_q'], z['cands_t']
tz = np.load(A/'tasks_pool_fr3.npz')
N = len(sub)
CFG = 'config_line_cont_dirfrac_e8kXXL_cone5.yaml'
CKPT = 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_cone5/agent.pt'

def build(cfgfile, classical=False, B=2500):
    y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj'/cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    kw['cone_deg'] = 5.0
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin)), None
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO/CKPT, map_location=dev))
    ag.eval()
    return env, None, ag

def roll(env, fn, ag, q0s, rows):
    B = env.n_envs; rdt = env.kin.dtype
    out = np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for lo in range(0, len(rows), B):
            hi = min(lo+B, len(rows)); pad = B-(hi-lo)
            s2 = {'q0': torch.tensor(q0s[lo:hi], dtype=rdt),
                  'line_dir': torch.tensor(tz['cs_line_dir'][sub[rows[lo:hi]]], dtype=rdt),
                  'n_target': torch.tensor(tz['cs_n_target'][sub[rows[lo:hi]]], dtype=rdt)}
            if pad:
                s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
            s2 = {k: v.to(dev) for k, v in s2.items()}
            env.line_dist = ScriptedLineDistribution(s2)
            env.reset()
            for _ in range(env.cfg.max_steps//2):
                a = fn(env) if fn else ag.actor_mean(env.current_obs())
                for _ in range(2):
                    env.step(a, auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
            out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
    return out

tr = np.arange(N)
env, _, ag = build(CFG)
p_re = roll(env, None, ag, q0f, tr)
print('retrain @shared rolled', flush=True)
# critic scores
rdt = env.kin.dtype
V = np.zeros(len(CQ), np.float32)
B = env.n_envs
with torch.no_grad():
    for lo in range(0, len(CQ), B):
        hi = min(lo+B, len(CQ)); pad = B-(hi-lo)
        ids = CT[lo:hi]
        s2 = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt),
              'line_dir': torch.tensor(tz['cs_line_dir'][sub[ids]], dtype=rdt),
              'n_target': torch.tensor(tz['cs_n_target'][sub[ids]], dtype=rdt)}
        if pad:
            s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
        s2 = {k: v.to(dev) for k, v in s2.items()}
        env.line_dist = ScriptedLineDistribution(s2)
        env.reset()
        V[lo:hi] = ag.get_value(env.current_obs()).float().cpu().numpy()[:hi-lo]
pick = q0f.copy()
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    pick[CT[lo]] = CQ[lo + int(np.argmax(V[lo:hi]))]
    lo = hi
p_sel = roll(env, None, ag, pick, tr)
print('retrain @critic rolled', flush=True)
# oracle subsample with the retrain
osub = np.sort(np.random.default_rng(9).choice(np.nonzero(has)[0], 500, replace=False))
mask = np.isin(CT, osub)
Lo = roll(env, None, ag, CQ[mask], CT[mask])
orc = np.zeros(N, np.float32)
CTm = CT[mask]
for t in osub:
    m2 = CTm == t
    orc[t] = Lo[m2].max()
print('oracle rolled', flush=True)
del env, ag; torch.cuda.empty_cache()
envc, fnc, _ = build(str(Path(hl.ROBOTS['fr3'][0]).name), classical=True)
p_cp = roll(envc, fnc, None, pick, tr)
print('classical @critic rolled', flush=True)
np.savez(FU/'cone5_formal_10k.npz', sub=sub, lpw5=lpw5, p_cls=p_cls,
         p_z30=p_z30, p_re=p_re, p_sel=p_sel, p_cp=p_cp, orc=orc,
         osub=osub, pick=pick)
ref = np.maximum.reduce([lpw5, p_cls, p_z30, p_re, p_sel, p_cp, orc])
print('=== FORMAL 5-deg cone, 10k tasks ===', flush=True)
for tag, v in (('classical @shared', p_cls), ('classical @critic', p_cp),
               ('flagship30 0shot @shared', p_z30),
               ('cone5-flagship @shared', p_re),
               ('cone5-flagship @critic', p_sel)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:26s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.2f} / '
          f'{np.percentile(rt,10)*100:.2f}', flush=True)
m3 = np.zeros(N, bool); m3[osub] = True
rto = orc[m3]/np.maximum(ref[m3], 1e-9)
rts = p_sel[m3]/np.maximum(ref[m3], 1e-9)
print(f'pool-oracle(500) {rto.mean()*100:.1f}/{np.percentile(rto,10)*100:.1f}; '
      f'critic on same: {rts.mean()*100:.1f} -> capture '
      f'{rts.mean()/rto.mean()*100:.1f}%', flush=True)
