"""Depth feed-forward check on the full 10k: the force-aware policy rolled
from the critic-picked starts with the feed-forward off / on, at force
tolerance +-2 N and +-1 N. Reports stroke ratio (denominator: the 10k
force bound and all realised strokes), per-task max |force error|, and
the share of force terminations."""
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


def roll(tol, ff):
    kw = dict(base, force_kn_max=2000.0, force_set=5.0, force_tol=tol, k_lateral=5.0,
              force_depth_ff=ff, n_envs=B)
    env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_force/agent.pt', map_location=dev))
    ag.eval(); rdt = env.kin.dtype
    prog = np.zeros(N, np.float32); fmax = np.zeros(N, np.float32); term = np.zeros(N, np.int64)
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
                if bool(env.done_persistent.all()):
                    break
        prog[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        fmax[lo:hi] = fm.float().cpu().numpy()[:hi - lo]; term[lo:hi] = tr.cpu().numpy()[:hi - lo]
    del env; torch.cuda.empty_cache()
    return prog, fmax, term


res = {}
for tol in (2.0, 1.0):
    for ff in (False, True):
        res[(tol, ff)] = roll(tol, ff)
        print(f'rolled tol={tol} ff={ff}', flush=True)
np.savez(FU / 'force_ff_10k.npz', **{f'p_t{t}_ff{int(f)}': v[0] for (t, f), v in res.items()},
         **{f'fmax_t{t}_ff{int(f)}': v[1] for (t, f), v in res.items()},
         **{f'term_t{t}_ff{int(f)}': v[2] for (t, f), v in res.items()})
ref = np.maximum.reduce([lpwf] + [v[0] for v in res.values()] + [d['p_sel']])
print(f'{"variant":22s} {"stroke":>7s} {"ratio":>12s} {"max|ferr| med/p90":>18s} {"force-term":>10s} {"stiff-term":>10s}')
for (tol, ff), (p, fm, tm) in res.items():
    rt = p[has] / np.maximum(ref[has], 1e-9)
    print(f'tol +-{tol:.0f} N ff={"on " if ff else "off"}      {p[has].mean():7.3f} '
          f'{rt.mean()*100:6.1f} / {np.percentile(rt,10)*100:4.1f} '
          f'{np.median(fm[has]):8.2f} / {np.percentile(fm[has],90):4.2f} '
          f'{(tm[has]==TERM_FORCE).mean()*100:9.1f}% {(tm[has]==TERM_STIFF).mean()*100:9.1f}%', flush=True)
