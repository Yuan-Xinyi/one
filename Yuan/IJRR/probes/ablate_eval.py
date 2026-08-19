"""Ablation evals on the aligned FR3 10k pool: cont a_max 10/20 (actor +
cont-critic vlook under matching dynamics) and LSTM/TF vertex backbones
(history-window actor + history-window vlook under mainline dynamics)."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent as ContAgent
from Yuan.IJRR.stage2_traj.vertex_agent import (LSTMVertexAgent,
                                                TransformerVertexAgent)
from Yuan.IJRR.stage2_traj.history_env import HistoryStackEnv

dev = torch.device('cuda')
hl.SUB = 2
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N, B = 10000, 2048
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'


def mkenv(cfgfile, wrap_k=None, n=B):
    y = yaml.safe_load(open(REPO / ('Yuan/IJRR/stage2_traj/' + cfgfile)))
    kw = {k: v for k, v in y['env'].items()
          if k in {f.name for f in dataclasses.fields(EnvConfig)}}
    kw['dt'] /= 2
    kw['max_steps'] = int(y['env']['max_steps'] * 2)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': n}), None, dev)
    model = hl.StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    model.terms = [0, 1]
    if wrap_k:
        env = HistoryStackEnv(env, wrap_k)
    return env, model


@torch.no_grad()
def batched(env, afn):
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
    return out


def make_vlook_hist(model, env, agent, chunk=32768):
    """vlook for history agents: successor window = current window shifted
    left with the analytic successor observation appended."""
    m = model.act_dim
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * m, indexing='ij'),
                 -1).reshape(-1, m), dtype=torch.float32,
        device=model.q_mid.device)
    K = verts.shape[0]

    @torch.no_grad()
    def fn(e, done):
        Bn = e.n_envs
        base = e._env
        qe = base.q.repeat_interleave(K, 0)
        ae = verts.unsqueeze(0).expand(Bn, -1, -1).reshape(Bn * K, m)
        de = base.line_dir.repeat_interleave(K, 0)
        ne = base.n_target.repeat_interleave(K, 0)
        pe = base.p_start.repeat_interleave(K, 0)
        qn = torch.cat([model.step(qe[i:i + chunk], de[i:i + chunk],
                                   ne[i:i + chunk], ae[i:i + chunk])
                        for i in range(0, Bn * K, chunk)])
        mg = torch.cat([model.margins(qn[i:i + chunk], pe[i:i + chunk],
                                      de[i:i + chunk], ne[i:i + chunk])
                        for i in range(0, Bn * K, chunk)])
        alive = (mg.amin(-1) > 0).reshape(Bn, K)
        succ = hl._obs_of(base, qn, de, ne, ae)          # (B*K, D)
        win = e._h[:, 1:].repeat_interleave(K, 0)        # (B*K, k-1, D)
        stacked = torch.cat([win, succ.unsqueeze(1)],
                            dim=1).reshape(Bn * K, -1)
        v = torch.cat([agent.critic(stacked[i:i + chunk]).squeeze(-1)
                       for i in range(0, Bn * K, chunk)]).reshape(Bn, K)
        v = torch.where(alive, v, torch.full_like(v, -1e9))
        return verts[v.argmax(-1)]
    return fn


results = {}

# ---- continuous a_max 10 / 20 ----------------------------------------
for am in (10, 20):
    env, model = mkenv(f'config_line_cont_sqent_amax{am}.yaml')
    ag = ContAgent(env.obs_dim, env.act_dim).to(dev)
    ag.load_state_dict(torch.load(
        REPO / f'Yuan/IJRR/runs/rl_cont_sqent_amax{am}_30M/agent.pt',
        map_location=dev))
    ag.eval()
    results[f'cont{am}_actor'] = batched(
        env, lambda e, dn: ag.actor_mean(e.current_obs()))
    print(f'cont{am} actor done', results[f'cont{am}_actor'].mean(),
          flush=True)
    results[f'cont{am}_vlook'] = batched(env, hl.make_vlook(model, env, ag))
    print(f'cont{am} vlook done', results[f'cont{am}_vlook'].mean(),
          flush=True)
    del env, model, ag
    torch.cuda.empty_cache()

# ---- LSTM / TF backbones ---------------------------------------------
for kind, cls in (('lstm', LSTMVertexAgent), ('tf', TransformerVertexAgent)):
    env, model = mkenv(f'config_vertex_line_{kind}.yaml', wrap_k=8)
    ag = cls(obs_dim=env.obs_dim, act_dim=env.act_dim, history=8).to(dev)
    ag.load_state_dict(torch.load(
        REPO / f'Yuan/IJRR/runs/rl_vertex_line_{kind}_30M/agent.pt',
        map_location=dev))
    ag.eval()
    results[f'{kind}_actor'] = batched(
        env, lambda e, dn: ag.actor_mean(e.current_obs()))
    print(f'{kind} actor done', results[f'{kind}_actor'].mean(), flush=True)
    results[f'{kind}_vlook'] = batched(env,
                                       make_vlook_hist(model, env, ag))
    print(f'{kind} vlook done', results[f'{kind}_vlook'].mean(), flush=True)
    del env, model, ag
    torch.cuda.empty_cache()

np.savez(FU + 'ablate_eval_10k.npz', **results)

# ---- ratios against the (new) pooled reference ------------------------
A = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
b = np.load(A + 'bound_pool_fr3.npz')
w = np.load(A + 'witness_pool_fr3.npz')
base = np.load(FU + 'pool_fr3_straight.npz')
ref = np.maximum(b['L_hi'], w['prog'])
for a2 in [k[:-9] for k in base.files if k.endswith('_progress')]:
    ref = np.maximum(ref, base[f'{a2}_progress'])
for v in results.values():
    ref = np.maximum(ref, v)
def stat(v):
    rt = v / np.maximum(ref, 1e-9)
    return (f'{v.mean():.4f}  {rt.mean()*100:.1f} / '
            f'{np.percentile(rt, 10)*100:.1f}   t27 {v[27]:.3f}')
print('--- vs new bound refs ---')
print('MLP baseline actor :', stat(base['vertex_progress']))
print('MLP baseline vlook :', stat(base['vlook_progress']))
print('cont a_max=0.5 actor:', stat(base['cont_progress']))
for k in ('cont10_actor', 'cont10_vlook', 'cont20_actor', 'cont20_vlook',
          'lstm_actor', 'lstm_vlook', 'tf_actor', 'tf_vlook'):
    print(f'{k:19s}:', stat(results[k]))

