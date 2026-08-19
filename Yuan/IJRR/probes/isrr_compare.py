"""Score the ISRR arms (deployed multi-task RL, and the RL/classical variant-B
hybrid) on exactly the tasks used by the cross-task distillation run, under
this experiment's protocol (SUB=1), next to the classical law, the one-step
margin law and the distilled policy.

Ratios only; every arm is scored by the same rollout_first_episode.
"""
import sys, os, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.env.rollout import rollout_first_episode
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.eval_curve import _agent, _hybrid_fn
import Yuan.IJRR.eval.horizon_ladder as hl

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
RUN = REPO / 'Yuan/IJRR/runs/pool_v5'
SPLIT = os.environ.get('SPLIT', 'heldout')     # heldout | train
NB = 2048                                      # batched-eigvalsh safe size

env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': NB}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
perm = torch.randperm(valid.numel(),
                      generator=torch.Generator().manual_seed(7))
sel = valid[perm[:12000 + 300 + 24]]
ids = (np.arange(12000, 12300) if SPLIT == 'heldout'
       else np.arange(0, int(os.environ.get('NTRAIN', '3000'))))
print(f"split={SPLIT}  {len(ids)} tasks")

Q0 = pool.q_pool[sel].to(dev)
DIR = pool.line_dir_pool[sel].to(dev)
NTG = pool.n_target_pool[sel].to(dev)

classical = ClassicalNullspaceController(env.kin)
model = hl.StraightModel(env)
model.terms = [0, 1]
myo = hl.make_myopic(model)
rl = _agent(MAIN / 'runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
dist = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
dist.load_state_dict(torch.load(RUN / 'agent_pool.pt', map_location=dev))
dist.eval()

arms = {
    'classical': cn_action_fn(classical),
    'margin law (myopic)': lambda e: myo(e, e.done_persistent),
    'ISRR pure RL (30M)': lambda e: rl.actor_mean(e.current_obs()).clamp(-1, 1),
    'ISRR hybrid B .98/.94': _hybrid_fn(rl, classical, 0.98, 0.94),
    'ISRR hybrid B .98/.98': _hybrid_fn(rl, classical, 0.98, 0.98),
    'distilled policy': lambda e: dist.actor_mean(e.current_obs()),
}

res = {}
for name, fn in arms.items():
    out, t0 = [], time.time()
    for base in range(0, len(ids), NB):
        chunk = ids[base:base + NB]
        pad = NB - len(chunk)
        cp = np.concatenate([chunk, np.full(pad, chunk[0])]) if pad else chunk
        ii = torch.as_tensor(cp, device=dev)
        env.line_dist = ScriptedLineDistribution(
            {'q0': Q0[ii].clone(), 'line_dir': DIR[ii].clone(),
             'n_target': NTG[ii].clone()})
        st = rollout_first_episode(env, fn)
        out.append(st['episode_progress'].cpu().numpy()[:len(chunk)])
    res[name] = np.concatenate(out)
    print(f"  {name:<24s} {res[name].mean():.4f} m  ({time.time()-t0:.0f}s)")

cl = res['classical']
my = res['margin law (myopic)']
print(f"\n{'arm':<24s} {'mean m':>8s} {'x classical':>12s} "
      f"{'median x cl':>12s} {'x margin law':>13s} {'win vs cl':>10s}")
for name, p in res.items():
    r = p / np.maximum(cl, 1e-6)
    print(f"{name:<24s} {p.mean():>8.4f} {r.mean():>12.3f} "
          f"{np.median(r):>12.3f} "
          f"{np.mean(p / np.maximum(my, 1e-6)):>13.3f} "
          f"{(p > cl + 1e-6).mean():>10.3f}")
np.savez(RUN / f'isrr_compare_{SPLIT}.npz',
         ids=ids, **{k.replace(' ', '_'): v for k, v in res.items()})
print('wrote', RUN / f'isrr_compare_{SPLIT}.npz')

# ---- clean critic one-step lookahead: rank ALL 16 alive successors by V ----
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * env.act_dim,
                     indexing='ij'), -1).reshape(-1, env.act_dim),
                     dtype=torch.float32, device=dev)
mdl_all = hl.StraightModel(env)          # all four margins for the alive test


def make_vlook(agent_v):
    fenv_cache = {}

    @torch.no_grad()
    def fn(e):
        B = e.n_envs
        q = e.q
        qe = q.repeat_interleave(16, 0)
        ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(B * 16, -1)
        de = e.line_dir.repeat_interleave(16, 0)
        ne = e.n_target.repeat_interleave(16, 0)
        pe = e.p_start.repeat_interleave(16, 0)
        qn = mdl_all.step(qe, de, ne, ae)
        mg = mdl_all.margins(qn, pe, de, ne)
        alive = (mg.amin(-1) > 0).reshape(B, 16)
        key = B * 16
        if key not in fenv_cache:
            fenv_cache[key] = NSRLBatchedEnv(
                EnvConfig(**{**y['env'], 'n_envs': key}), None, dev)
        fenv = fenv_cache[key]
        fenv.line_dist = ScriptedLineDistribution(
            {'q0': qn, 'line_dir': de.clone(), 'n_target': ne.clone()})
        fenv.reset()
        ob = fenv.current_obs().clone()
        if os.environ.get('APREV', '1') == '1':
            ob[:, -4:] = ae          # the critic was trained with a_prev set
        v = agent_v.critic(ob).squeeze(-1).reshape(B, 16)
        v = torch.where(alive, v, torch.full_like(v, -1e9))
        return verts[v.argmax(-1)]
    return fn


print("\n--- clean critic one-step lookahead (all 16 successors ranked by V) ---")
for tag, ag in (('ISRR RL critic', rl), ('distilled critic', dist)):
    out = []
    t0 = time.time()
    for base in range(0, len(ids), 1024):
        chunk = ids[base:base + 1024]
        e2 = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                            'n_envs': len(chunk)}), None, dev)
        ii = torch.as_tensor(chunk, device=dev)
        e2.line_dist = ScriptedLineDistribution(
            {'q0': Q0[ii].clone(), 'line_dir': DIR[ii].clone(),
             'n_target': NTG[ii].clone()})
        st = rollout_first_episode(e2, make_vlook(ag))
        out.append(st['episode_progress'].cpu().numpy())
        del e2
    p = np.concatenate(out)
    r = p / np.maximum(cl, 1e-6)
    print(f"{tag:<20s} {p.mean():.4f} m  x{r.mean():.3f} classical  "
          f"x{np.mean(p / np.maximum(my, 1e-6)):.3f} margin law  "
          f"median x{np.median(r):.3f}  ({time.time()-t0:.0f}s)")
    np.savez(RUN / f'vlook_{tag.split()[0]}_{SPLIT}.npz', progress=p)

# ---- controls: how much of the gain is the ALIVE FILTER, not the critic? ----
def make_onestep(rank_mode):
    @torch.no_grad()
    def fn(e):
        B = e.n_envs
        qe = e.q.repeat_interleave(16, 0)
        ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(B * 16, -1)
        de = e.line_dir.repeat_interleave(16, 0)
        ne = e.n_target.repeat_interleave(16, 0)
        pe = e.p_start.repeat_interleave(16, 0)
        qn = mdl_all.step(qe, de, ne, ae)
        mg = mdl_all.margins(qn, pe, de, ne)
        alive = (mg.amin(-1) > 0).reshape(B, 16)
        if rank_mode == 'softmin4':
            sc = mdl_all.softmin_margin(qn, pe, de, ne).reshape(B, 16)
        elif rank_mode == 'random':
            sc = torch.rand(B, 16, device=dev)
        else:                      # progress of the successor along the line
            pf = e.kin.tcp_fk_jac(qn)[0]
            sc = ((pf - pe) * de).sum(-1).reshape(B, 16)
        sc = torch.where(alive, sc, torch.full_like(sc, -1e9))
        return verts[sc.argmax(-1)]
    return fn


print("\n--- one-step controls with the SAME alive filter ---")
for tag, mode in (('alive + softmin4', 'softmin4'),
                  ('alive + random', 'random'),
                  ('alive + progress', 'progress')):
    out = []
    for base in range(0, len(ids), 1024):
        chunk = ids[base:base + 1024]
        e2 = NSRLBatchedEnv(EnvConfig(**{**y['env'],
                            'n_envs': len(chunk)}), None, dev)
        ii = torch.as_tensor(chunk, device=dev)
        e2.line_dist = ScriptedLineDistribution(
            {'q0': Q0[ii].clone(), 'line_dir': DIR[ii].clone(),
             'n_target': NTG[ii].clone()})
        st = rollout_first_episode(e2, make_onestep(mode))
        out.append(st['episode_progress'].cpu().numpy())
        del e2
    p = np.concatenate(out)
    r = p / np.maximum(cl, 1e-6)
    print(f"{tag:<20s} {p.mean():.4f} m  x{r.mean():.3f} classical  "
          f"x{np.mean(p / np.maximum(my, 1e-6)):.3f} margin law  "
          f"median x{np.median(r):.3f}")
    np.savez(RUN / f'onestep_{mode}_{SPLIT}.npz', progress=p)
