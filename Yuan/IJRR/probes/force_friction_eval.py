"""Tangential-friction experiment on the full 10k (critic-picked starts,
force-aware policy trained without friction):
  (a) at mu = 0, the normal-tangential cross-compliance ratio n^T C t / n^T C n
      along every trajectory -> how much a friction coefficient would shift
      the realised normal force for a commanded depth;
  (b) zero-shot rollouts at mu in {0.3, 0.6}: stroke ratio, max |force error|,
      force / stiffness terminations, versus mu = 0."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_FORCE, TERM_STIFF
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'; FU = MAIN / 'runs/paper_fill/fam_unify'
d = dict(np.load(FU / 'force_eval_10k.npz'))
sub, has, lpwf, pick_q = d['sub'], d['has'], d['lpwf'], d['pick_q']
tz = np.load(A / 'tasks_pool_fr3.npz')
dd = tz['cs_line_dir'][sub].astype(np.float32); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
N = len(sub); B = 2000
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_force.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
base = {k: v for k, v in y['env'].items() if k in keys}
base['dt'] /= 2; base['max_steps'] = int(y['env']['max_steps'] * 2)
FF = '--ff' in sys.argv[1:]
_ck = [a for a in sys.argv[1:] if a.startswith('--ckpt=')]
CKPT = _ck[0][7:] if _ck else 'force'


def roll(mu):
    kw = dict(base, force_kn_max=2000.0, force_set=5.0, force_tol=2.0, k_lateral=5.0,
              force_depth_ff=FF, force_mu=mu, n_envs=B)
    env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{CKPT}/agent.pt', map_location=dev))
    ag.eval(); rdt = env.kin.dtype
    prog = np.zeros(N, np.float32); fmax = np.zeros(N, np.float32); term = np.zeros(N, np.int64)
    ratios = []
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        def _t(a):
            t = torch.tensor(a[lo:hi], dtype=rdt, device=dev)
            return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])]) if pad else t
        env.line_dist = ScriptedLineDistribution({'q0': _t(pick_q), 'line_dir': _t(dd), 'n_target': _t(nt)})
        env.reset(); fm = torch.zeros(B, device=dev); tr = torch.zeros(B, dtype=torch.long, device=dev)
        with torch.no_grad():
            for _ in range(env.cfg.max_steps // 2):
                a = ag.actor_mean(env.current_obs())
                for _ in range(2):
                    _, _, te, tu, info = env.step(a, auto_reset=False)
                    alive = ~env.done_persistent
                    fm = torch.where(alive | te | tu, torch.maximum(fm, env._ferr.abs()), fm)
                    new = (te | tu) & (tr == 0)
                    tr = torch.where(new, info['term_reason'], tr)
                    if mu == 0.0 and alive.any():
                        ratios.append(env._fric_ratio[alive][:hi - lo].float().cpu().numpy())
                if bool(env.done_persistent.all()):
                    break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        fmax[lo:hi] = fm.float().cpu().numpy()[:hi - lo]; term[lo:hi] = tr.cpu().numpy()[:hi - lo]
    del env; torch.cuda.empty_cache()
    return prog, fmax, term, (np.concatenate(ratios) if ratios else None)


res = {}
for mu in (0.0, 0.3, 0.6):
    res[mu] = roll(mu)
    print(f'rolled mu={mu}', flush=True)
r = res[0.0][3]
print(f'cross ratio n^T C t / n^T C n along trajectories ({len(r)} steps): '
      f'median |r| {np.median(np.abs(r)):.3f}, p90 {np.percentile(np.abs(r), 90):.3f}, '
      f'p99 {np.percentile(np.abs(r), 99):.3f}; sign + {(r > 0).mean()*100:.0f}%', flush=True)
for mu in (0.3, 0.6):
    e = mu * np.abs(r) / (1 + mu * r)            # relative change of realised force
    print(f'  mu={mu}: relative force change |mu r/(1+mu r)| median {np.median(e)*100:.1f}%, '
          f'p90 {np.percentile(e, 90)*100:.1f}%, share > 20% (1 N at 5 N): {(e > 0.2).mean()*100:.1f}%', flush=True)
np.savez(FU / f'force_friction_10k{"_ff" if FF else ""}{"" if CKPT == "force" else "_" + CKPT}.npz',
         **{f'p_mu{mu}': v[0] for mu, v in res.items()}, **{f'fmax_mu{mu}': v[1] for mu, v in res.items()},
         **{f'term_mu{mu}': v[2] for mu, v in res.items()}, ratio_mu0=r)
ref = np.maximum.reduce([lpwf] + [v[0] for v in res.values()] + [d['p_sel']])
print(f'{"variant":12s} {"stroke":>7s} {"ratio":>12s} {"max|ferr| med/p90":>18s} {"force-term":>10s} {"stiff-term":>10s}')
for mu, (p, fm, tm, _) in res.items():
    rt = p[has] / np.maximum(ref[has], 1e-9)
    print(f'mu={mu:<8.1f} {p[has].mean():7.3f} {rt.mean()*100:6.1f} / {np.percentile(rt,10)*100:4.1f} '
          f'{np.median(fm[has]):8.2f} / {np.percentile(fm[has],90):4.2f} '
          f'{(tm[has]==TERM_FORCE).mean()*100:9.1f}% {(tm[has]==TERM_STIFF).mean()*100:9.1f}%', flush=True)
