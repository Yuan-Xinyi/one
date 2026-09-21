"""Zero-step critic start selection under the implicit-force constraint.

Every admissible candidate saved by force_eval.py (cone 30, tube, limits,
collision, k_n <= cap) is scored by the retrained policy's critic at its
reset observation; the arg-max per task is rolled once, same substep
protocol as force_eval. Adds p_sel to force_eval_v1.npz and prints the
four-row comparison."""
import sys, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

CONE = 30.0
_kn = [a for a in sys.argv[1:] if a.startswith('--kn=')]
_ck = [a for a in sys.argv[1:] if a.startswith('--ckpt=')]
CKPT = _ck[0][7:] if _ck else 'force'
KN_MAX = float(_kn[0][5:]) if _kn else 2000.0
FORCE_KW = dict(force_kn_max=KN_MAX, force_set=5.0, force_tol=2.0, k_lateral=5.0)
dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'
FU = MAIN / 'runs/paper_fill/fam_unify'
_sfx = '' if KN_MAX == 2000.0 else f'_k{int(KN_MAX)}'
FULL = '--all' in sys.argv[1:]
OUTF = FU / (f'force_eval_10k{_sfx}.npz' if FULL else (f'force_eval{_sfx}.npz' if _sfx else 'force_eval_v1.npz'))
d = dict(np.load(OUTF))
sub, has, lpwf = d['sub'], d['has'], d['lpwf']
CQ, CT, CK = d['cands_q'], d['cands_t'], d['cands_kn']
N = len(sub)
tz = np.load(A / 'tasks_pool_fr3.npz')
dd = tz['cs_line_dir'][sub].astype(np.float32)
dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32)
nt /= np.linalg.norm(nt, axis=1, keepdims=True)

y = yaml.safe_load(open(REPO / f'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_{CKPT}.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
kw.update(FORCE_KW); kw['cone_deg'] = CONE
B = 4096
renv = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
ag = Agent(renv.obs_dim, renv.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO / f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{CKPT}/agent.pt',
                              map_location=dev))
ag.eval()
rdt = renv.kin.dtype


def specs(q, t):
    s = {'q0': torch.tensor(q, dtype=rdt), 'line_dir': torch.tensor(dd[t], dtype=rdt),
         'n_target': torch.tensor(nt[t], dtype=rdt)}
    pad = B - len(q)
    if pad:
        s = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s.items()}
    return {k: v.to(dev) for k, v in s.items()}


# ---- score every candidate at its reset observation ----
V = np.zeros(len(CQ), np.float32)
t0 = time.time()
with torch.no_grad():
    for lo in range(0, len(CQ), B):
        hi = min(lo + B, len(CQ))
        renv.line_dist = ScriptedLineDistribution(specs(CQ[lo:hi], CT[lo:hi]))
        renv.reset()
        V[lo:hi] = ag.get_value(renv.current_obs()).float().cpu().numpy()[:hi - lo]
print(f'scored {len(CQ)} candidates ({time.time() - t0:.0f}s)', flush=True)

pick_q = d['q0_first'].copy(); pick_kn = np.zeros(N, np.float32)
first_kn = np.zeros(N, np.float32)
lo = 0
while lo < len(CT):
    hi = lo
    while hi < len(CT) and CT[hi] == CT[lo]:
        hi += 1
    j = lo + int(np.argmax(V[lo:hi]))
    pick_q[CT[lo]] = CQ[j]; pick_kn[CT[lo]] = CK[j]; first_kn[CT[lo]] = CK[lo]
    lo = hi

# ---- roll the picks ----
p_sel = np.zeros(N, np.float32)
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        renv.line_dist = ScriptedLineDistribution(specs(pick_q[lo:hi], np.arange(lo, hi)))
        renv.reset()
        for _ in range(renv.cfg.max_steps // 2):
            a = ag.actor_mean(renv.current_obs())
            for _ in range(2):
                renv.step(a, auto_reset=False)
            if bool(renv.done_persistent.all()):
                break
        p_sel[lo:hi] = renv.arc_progress.float().cpu().numpy()[:hi - lo]
_k = '' if CKPT == 'force' else '_' + CKPT
d['p_sel' + _k] = p_sel; d['pick_q' + _k] = pick_q; d['cands_V' + _k] = V
np.savez(OUTF, **d)
print(f'picked-start k_n median {np.median(pick_kn[has]):.0f} N/m '
      f'(first-candidate {np.median(first_kn[has]):.0f})', flush=True)

rows = [('classical', d['p_cls']), ('flagship-0shot', d['p_rl']),
        ('force', d['p_force']), ('force+critic', d['p_sel'])] if CKPT != 'force' else \
       [('classical', d['p_cls']), ('flagship-0shot', d['p_rl']), ('force', d['p_force'])]
rows.append((CKPT if CKPT != 'force' else 'force+critic', p_sel) if CKPT == 'force' else (CKPT + '+critic', p_sel))
if CKPT != 'force' and f'p_{CKPT}' in d:
    rows.insert(-1, (CKPT, d[f'p_{CKPT}']))
ref = np.maximum.reduce([lpwf] + [v for _, v in rows])
for tag, v in rows:
    rt = v[has] / np.maximum(ref[has], 1e-9)
    print(f'{tag:15s} stroke {v[has].mean():.3f}  ratio {rt.mean() * 100:.1f} / '
          f'{np.percentile(rt, 10) * 100:.1f}', flush=True)
imp = (p_sel[has] > d['p_force'][has] + 0.01).mean(); wor = (p_sel[has] < d['p_force'][has] - 0.01).mean()
print(f'critic pick vs first candidate: better {imp * 100:.1f}% / worse {wor * 100:.1f}% of tasks', flush=True)
