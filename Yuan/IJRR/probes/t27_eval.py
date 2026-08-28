"""Task-27 single-task eval: canonical-start stroke + per-segment survival.
argv: <config yaml basename> <ckpt path> [tag]"""
import matplotlib; matplotlib.use('Agg')
import sys, dataclasses, yaml, torch, numpy as np
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

cfgf, ckpt = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else ckpt
A = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
tz = np.load(A + 'tasks_pool_fr3.npz')
pool = np.load(FU + 't27_ray_starts.npz')
dev = torch.device('cuda')

y = yaml.safe_load(open('/home/lqin/one/Yuan/IJRR/.claude/worktrees/'
                        f'vigilant-hertz-799b05/Yuan/IJRR/stage2_traj/{cfgf}'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 512
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy,
           hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(ckpt, map_location=dev))
ag.eval()
rdt = env.kin.dtype
GATE_COL = 7 if env.act_dim_policy > 7 else None


def roll(q0, p0):
    n = len(q0)
    L = np.zeros(n, np.float32)
    paused = np.zeros(n, np.int32)
    for lo in range(0, n, B):
        hi = min(lo + B, n)
        d = {'q0': torch.tensor(q0[lo:hi], dtype=rdt),
             'p0': torch.tensor(p0[lo:hi], dtype=rdt),
             'line_dir': torch.tensor(
                 np.repeat(tz['cs_line_dir'][27][None], hi - lo, 0), dtype=rdt),
             'n_target': torch.tensor(
                 np.repeat(tz['cs_n_target'][27][None], hi - lo, 0), dtype=rdt)}
        if hi - lo < B:
            d = {k: torch.cat([v, v[-1:].expand(B - (hi - lo), *v.shape[1:])])
                 for k, v in d.items()}
        d = {k: v.to(dev) for k, v in d.items()}
        env.line_dist = ScriptedLineDistribution(d)
        env.reset()
        with torch.no_grad():
            for _ in range(env.cfg.max_steps // 2):
                a = ag.actor_mean(env.current_obs())
                if GATE_COL is not None:
                    pz = ((a[:, GATE_COL] <= 0)
                          & ~env.done_persistent).cpu().numpy()
                    paused[lo:hi] += pz[:hi - lo]
                for _ in range(2):
                    env.step(a, auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
        L[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return L, paused


L0, P0_ = roll(tz['q0_seed'][27][None].astype(np.float32),
               tz['cs_p0'][27][None].astype(np.float32))
print(f'[{tag}] canonical start: {L0[0]:.3f} m  paused {int(P0_[0])} steps'
      f'   (old 0.582 / gate 0.588 / vlook 0.831 / ceiling 1.70)', flush=True)

rng = np.random.default_rng(0)
S = pool['s']
rows, labels = [], []
for lo in np.arange(0.0, 1.7, 0.1):
    m = (S >= lo) & (S < lo + 0.1)
    if int(m.sum()) == 0:
        continue
    rows.append(rng.choice(np.nonzero(m)[0], min(48, int(m.sum())),
                           replace=False))
    labels.append(lo)
ids = np.concatenate(rows)
L, P = roll(pool['q'][ids], pool['p0'][ids])
off = 0
print('   s-bin    mean     best   best-reach  pause%', flush=True)
for lo, r in zip(labels, rows):
    k = len(r)
    seg, pseg, sseg = L[off:off + k], P[off:off + k], S[r]
    print(f'   {lo:.1f}   {seg.mean():7.3f}  {seg.max():.3f}   '
          f'{(sseg + seg).max():.3f}      {pseg.mean():.1f}', flush=True)
    off += k
np.savez(FU + f't27_eval_{tag}.npz', ids=ids, L=L, P=P, s=S[ids],
         canonical=L0[0])
