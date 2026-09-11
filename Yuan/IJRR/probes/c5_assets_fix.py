"""Re-roll the two baselines chunked over all 10k, fix the assets npz."""
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
CQ, CT = z['cands_q'], z['cands_t']
tz = np.load(A/'tasks_pool_fr3.npz')
N = len(sub)

def roll(cfgfile, ckpt, classical=False):
    y = yaml.safe_load(open(REPO/'Yuan/IJRR/stage2_traj'/cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps']*2)
    kw['cone_deg'] = 5.0
    B = 2500
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    rdt = env.kin.dtype
    if classical:
        fn = cn_action_fn(ClassicalNullspaceController(env.kin))
        ag = None
    else:
        ag = Agent(env.obs_dim, env.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO/ckpt, map_location=dev))
        ag.eval()
    out = np.zeros(N, np.float32)
    with torch.no_grad():
        for lo in range(0, N, B):
            hi = min(lo+B, N); pad = B-(hi-lo)
            s2 = {'q0': torch.tensor(q0f[lo:hi], dtype=rdt),
                  'line_dir': torch.tensor(tz['cs_line_dir'][sub[lo:hi]], dtype=rdt),
                  'n_target': torch.tensor(tz['cs_n_target'][sub[lo:hi]], dtype=rdt)}
            if pad:
                s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
            s2 = {k: v.to(dev) for k, v in s2.items()}
            env.line_dist = ScriptedLineDistribution(s2)
            env.reset()
            for _ in range(env.cfg.max_steps//2):
                a = fn(env) if classical else ag.actor_mean(env.current_obs())
                for _ in range(2):
                    env.step(a, auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
            out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi-lo]
    del env
    torch.cuda.empty_cache()
    return out

import Yuan.IJRR.eval.horizon_ladder as hl2
p_cls = roll(str(Path(hl2.ROBOTS['fr3'][0]).name), None, classical=True)
print('classical rolled', flush=True)
p_rl = roll('config_line_cont_dirfrac_e8kXXL_rm.yaml',
            'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt')
print('flagship-30 rolled', flush=True)
np.savez(FU/'cone5_full10k_assets.npz', sub=sub, q0_first=q0f, has=has,
         lpw5=lpw5, p_cls=p_cls, p_rl=p_rl, cands_q=CQ, cands_t=CT)
ref = np.maximum.reduce([lpw5, p_cls, p_rl])
for tag, v in (('classical', p_cls), ('flagship-30-0shot', p_rl)):
    rt = v[has]/np.maximum(ref[has], 1e-9)
    print(f'{tag:18s} stroke {v[has].mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt,10)*100:.1f}', flush=True)
