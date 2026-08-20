"""Gate-vs-selection attribution for the ANALYTICAL MARGIN law.

Existing cells on the 10k pool: myopic (= margin-rank, NO gate) 0.5310,
vlook (gate + critic) 0.5625, random/zero baselines. Adds:
  random16    : uniform vertex          (no gate, no selection)
  gate_random : uniform among survivors (gate only)
  gate_margin : softmin-rank among survivors (gate + margin selection)
Gate = all four successor margins > 0 (same as vlook); ranking potential =
softmin over terms [0,1] (jl, cone), identical to the paper's myopic law."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution

dev = torch.device('cuda')
hl.SUB = 2
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N, B = 10000, 2048
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'

y = yaml.safe_load(open(REPO / hl.ROBOTS['fr3'][0]))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2
kw['max_steps'] = int(y['env']['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
model = hl.StraightModel(env)
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
model.terms = [0, 1]
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * env.act_dim, indexing='ij'),
             -1).reshape(-1, env.act_dim), dtype=torch.float32, device=dev)
K = verts.shape[0]
CH = 32768
gen = torch.Generator(device=dev).manual_seed(7)


@torch.no_grad()
def enum_all(e):
    Bn = e.n_envs
    qe = e.q.repeat_interleave(K, 0)
    ae = verts.unsqueeze(0).expand(Bn, -1, -1).reshape(Bn * K, -1)
    de = e.line_dir.repeat_interleave(K, 0)
    ne = e.n_target.repeat_interleave(K, 0)
    pe = e.p_start.repeat_interleave(K, 0)
    qn = torch.cat([model.step(qe[i:i + CH], de[i:i + CH], ne[i:i + CH],
                               ae[i:i + CH]) for i in range(0, Bn * K, CH)])
    mg = torch.cat([model.margins(qn[i:i + CH], pe[i:i + CH], de[i:i + CH],
                                  ne[i:i + CH])
                    for i in range(0, Bn * K, CH)])
    alive = (mg.amin(-1) > 0).reshape(Bn, K)
    tau = 0.1
    phi = (-tau * torch.logsumexp(-mg[:, model.terms] / tau, dim=-1)
           ).reshape(Bn, K)
    return alive, phi


def pick_gate_margin(e, done):
    alive, phi = enum_all(e)
    NEG = torch.full_like(phi, -1e9)
    sc = torch.where(alive, phi, NEG)
    pick = torch.where(alive.any(-1), sc.argmax(-1), phi.argmax(-1))
    return verts[pick]


def pick_gate_random(e, done):
    alive, phi = enum_all(e)
    r = torch.rand(alive.shape, device=dev, generator=gen)
    sc = torch.where(alive, r, torch.full_like(r, -1e9))
    pick = torch.where(alive.any(-1), sc.argmax(-1),
                       r.argmax(-1))
    return verts[pick]


def pick_random16(e, done):
    r = torch.randint(0, K, (e.n_envs,), device=dev, generator=gen)
    return verts[r]


@torch.no_grad()
def run(afn, tag):
    out = np.zeros(N, np.float32)
    dt = env.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        ids = np.arange(lo, hi)
        ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        env.line_dist = ScriptedLineDistribution(
            {'q0': torch.tensor(tz['q0_seed'][ip], dtype=dt, device=dev),
             'line_dir': torch.tensor(tz['cs_line_dir'][ip], dtype=dt,
                                      device=dev),
             'n_target': torch.tensor(tz['cs_n_target'][ip], dtype=dt,
                                      device=dev)})
        env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(env.cfg.max_steps // 2):
            a = afn(env, done)
            for _ in range(2):
                env.step(a, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'{tag} {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
    return out


res = {}
res['random16'] = run(pick_random16, 'random16')
res['gate_random'] = run(pick_gate_random, 'gate_random')
res['gate_margin'] = run(pick_gate_margin, 'gate_margin')
np.savez(FU + 'margin_attrib_10k.npz', **res)

A = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
b = np.load(A + 'bound_pool_fr3.npz')
w = np.load(A + 'witness_pool_fr3.npz')
base = np.load(FU + 'pool_fr3_straight.npz')
ref = np.maximum(b['L_hi'], w['prog'])
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]:
    ref = np.maximum(ref, base[f'{a2}_progress'])
for v in res.values():
    ref = np.maximum(ref, v)
def stat(v, t):
    rt = v / np.maximum(ref, 1e-9)
    print(f'{t}: {v.mean():.4f}  {rt.mean()*100:.1f} / '
          f'{np.percentile(rt, 10)*100:.1f}   t27 {v[27]:.3f}', flush=True)
print('--- margin-law attribution grid ---')
stat(res['random16'],  'D random16            ')
stat(res['gate_random'], 'C gate + random       ')
stat(base['myopic_progress'], 'B margin-rank, no gate')
stat(res['gate_margin'], 'A gate + margin-rank  ')
stat(base['vlook_progress'], '  (gate + critic ref) ')
