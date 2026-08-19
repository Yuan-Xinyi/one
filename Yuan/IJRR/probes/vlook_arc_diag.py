"""Diagnose the value-law-on-curves anomaly: is the vlook hard alive-gate
(min over ALL 4 margins incl. m_lat measured to the fictitious ray anchored
at p_start with the instantaneous tangent) the killer on arcs?

Arms, same 512 arc tasks, same q0 (candidate 0 of the v2_k32 pool):
  myopic      margin law, softmin over terms [jl, cone]      (reference)
  vlook       stock: gate = min(jl, cone, lat, coll) > 0     (suspected bug)
  vlook_fix   gate = min(jl, cone) > 0, same critic ranking  (like-for-like)
Also logs, for the stock arm, the fraction of envs whose gate killed ALL 32
successors at least once before death (smoking gun).
"""
import sys, time, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'selector_ood', MAIN / 'stage1_seed/selector_ood.py')
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent

N = 512
dev = torch.device('cuda')
env, model = so.build_base_env(N, dev)
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)

tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
spec = tasks['test_arc']
print('spec keys:', list(spec.keys()))
cd = cands['test_arc']['cands'].cpu().numpy()
print('cands shape', cd.shape)

dt = env.kin.dtype
sub = {'q0': torch.tensor(cd[:N, 0], dtype=dt),
       'p0': spec['p0'][:N]}
for key in ('line_dir', 'n_target', 'kappa'):
    if key in spec:
        sub[key] = spec[key][:N]
for k2 in ('q0', 'line_dir', 'n_target'):
    sub[k2] = sub[k2].to(device=dev, dtype=dt)
kap = np.abs(np.asarray(spec['kappa'][:N], dtype=np.float64))


def make_vlook_gated(model, env, agent, terms, gate_log=None, chunk=32768):
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]

    @torch.no_grad()
    def fn(e, done):
        B = e.n_envs
        qe = e.q.repeat_interleave(K, 0)
        ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(B * K, m)
        de = e.line_dir.repeat_interleave(K, 0)
        ne = e.n_target.repeat_interleave(K, 0)
        pe = e.p_start.repeat_interleave(K, 0)
        qn = torch.cat([model.step(qe[i:i + chunk], de[i:i + chunk],
                                   ne[i:i + chunk], ae[i:i + chunk])
                        for i in range(0, B * K, chunk)])
        mg = torch.cat([model.margins(qn[i:i + chunk], pe[i:i + chunk],
                                      de[i:i + chunk], ne[i:i + chunk])
                        for i in range(0, B * K, chunk)])
        alive = (mg[:, terms].amin(-1) > 0).reshape(B, K)
        if gate_log is not None:
            gate_log.logical_or_((~alive.any(-1)) & ~done)
        v = torch.cat([agent.critic(
            hl._obs_of(e, qn[i:i + chunk], de[i:i + chunk], ne[i:i + chunk],
                       ae[i:i + chunk])).squeeze(-1)
            for i in range(0, B * K, chunk)]).reshape(B, K)
        v = torch.where(alive, v, torch.full_like(v, -1e9))
        return verts[v.argmax(-1)]
    return fn


def run(afn):
    env.line_dist = ScriptedLineDistribution(sub)
    env.reset()
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
    for _ in range(env.max_steps // so.SUB):
        a = afn(env, done)
        for _ in range(so.SUB):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return env.arc_progress.float().cpu().numpy().copy()


t0 = time.time()
myo = run(hl.make_myopic(model))
print(f'myopic (margin law)      mean {myo.mean():.4f} m   [{time.time()-t0:.0f}s]',
      flush=True)

gate_hit = torch.zeros(N, dtype=torch.bool, device=dev)
t0 = time.time()
stock = run(make_vlook_gated(model, env, ag, [0, 1, 2, 3], gate_hit))
print(f'vlook stock (4-term gate) mean {stock.mean():.4f} m   '
      f'all-32-dead hit: {gate_hit.float().mean():.1%}  [{time.time()-t0:.0f}s]',
      flush=True)

t0 = time.time()
fix = run(make_vlook_gated(model, env, ag, [0, 1]))
print(f'vlook fixed (jl+cone gate) mean {fix.mean():.4f} m   [{time.time()-t0:.0f}s]',
      flush=True)

for lo, hi in [(0, 1), (1, 2), (2, 3)]:
    m = (kap >= lo) & (kap < hi)
    if m.sum():
        print(f'|kappa| in [{lo},{hi}):  n={m.sum():4d}  '
              f'myopic {myo[m].mean():.3f}  stock {stock[m].mean():.3f}  '
              f'fixed {fix[m].mean():.3f}')
np.savez('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-'
         'vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/'
         'scratchpad/vlook_arc_diag.npz',
         myo=myo, stock=stock, fix=fix, kappa=kap,
         gate_hit=gate_hit.cpu().numpy())
print('saved diag npz')
