"""Reviewer comment 3: approximate oracle for the true stroke-maximization
objective on the 1024 protocol tasks (FR3, paper protocol SUB=2).
Per-task maximum over many stochastic rollouts: eps-noised value law,
eps-noised one-step margin law, and stochastic vertex-PPO sampling.
First pass (eps=0 value law) must reproduce the ladder's vlook reference
(~0.564 m) before the sweep runs."""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
B = 1024
kw = dict(y['env']); kw['dt'] = kw['dt'] / 2; kw['max_steps'] = int(kw['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
ids = valid[:B]
model = hl.StraightModel(env)
import dataclasses
model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])   # plan at 50 ms
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * env.act_dim,
                     indexing='ij'), -1).reshape(-1, env.act_dim),
                     dtype=torch.float32, device=dev)
NV = verts.shape[0]
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill')
log = open(OUT / 'offline_oracle.log', 'a')


def say(m):
    print(m, flush=True)
    log.write(m + '\n'); log.flush()


@torch.no_grad()
def expand(scorer):
    """score all NV successors of the current batch state; -inf if dead"""
    qe = env.q.repeat_interleave(NV, 0)
    ae = verts.unsqueeze(0).expand(B, -1, -1).reshape(-1, env.act_dim)
    de = env.line_dir.repeat_interleave(NV, 0)
    ne = env.n_target.repeat_interleave(NV, 0)
    pe = env.p_start.repeat_interleave(NV, 0)
    CH = 32768
    qn = torch.cat([model.step(qe[i:i + CH], de[i:i + CH], ne[i:i + CH],
                               ae[i:i + CH]) for i in range(0, B * NV, CH)])
    mg = torch.cat([model.margins(qn[i:i + CH], pe[i:i + CH], de[i:i + CH],
                                  ne[i:i + CH]) for i in range(0, B * NV, CH)])
    aliveA = (mg.amin(-1) > 0).reshape(B, NV)
    sc = scorer(qn, de, ne, ae, mg).reshape(B, NV)
    return torch.where(aliveA, sc, torch.full_like(sc, -1e9)), aliveA


def sc_value(qn, de, ne, ae, mg):
    CH = 32768
    return torch.cat([ag.critic(hl._obs_of(env, qn[i:i + CH], de[i:i + CH],
                                           ne[i:i + CH], ae[i:i + CH])
                                ).squeeze(-1)
                      for i in range(0, qn.shape[0], CH)])


def sc_margin(qn, de, ne, ae, mg):
    return -0.1 * torch.logsumexp(-mg / 0.1, dim=-1)


@torch.no_grad()
def rollout(mode, eps, seed):
    env.line_dist = ScriptedLineDistribution(
        {'q0': pool.q_pool[ids].to(dev),
         'line_dir': pool.line_dir_pool[ids].to(dev),
         'n_target': pool.n_target_pool[ids].to(dev)})
    env.reset()
    g = torch.Generator(device='cpu').manual_seed(seed)
    prog = torch.zeros(B, device=dev)
    act = torch.zeros(B, dtype=torch.long, device=dev)
    for t in range(env.cfg.max_steps):
        live = ~env.done_persistent
        if not bool(live.any()):
            break
        if t % 2 == 0:                      # one 50 ms decision, held 2 steps
            if mode == 'ppo':
                logits = ag._logits_head(ag._actor_trunk(env.current_obs()))
                act = torch.distributions.Categorical(
                    logits=logits / max(eps, 1e-6)).sample()
            else:
                sc, aliveA = expand(sc_value if mode == 'value' else sc_margin)
                act = sc.argmax(-1)
                if eps > 0:
                    r = torch.rand(B, generator=g).to(dev)
                    noise = (r < eps) & aliveA.any(-1)
                    if bool(noise.any()):
                        w = aliveA.float() + 1e-9
                        rnd = torch.multinomial(w, 1).squeeze(-1)
                        act = torch.where(noise, rnd, act)
        env.step(verts[act], auto_reset=False)
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        cur = ((p - env.p_start) * env.line_dir).sum(-1)
        prog = torch.maximum(prog, torch.where(live, cur, prog))
    return prog


t0 = time.time()
ref = rollout('value', 0.0, 0)
say(f'[oracle] eps=0 value-law reference: {ref.mean():.4f} m '
    f'({time.time()-t0:.0f}s)  [ladder vlook ~0.564]')
best = ref.clone()
best_margin = rollout('margin', 0.0, 1)
say(f'[oracle] eps=0 margin-law: {best_margin.mean():.4f} m')
best = torch.maximum(best, best_margin)

PASSES = ([('value', e, 1000 + i) for i, e in enumerate(
             [0.05, 0.1, 0.2, 0.3] * 30)]              # 120 value passes
          + [('margin', e, 3000 + i) for i, e in enumerate(
             [0.1, 0.2, 0.3] * 20)]                    # 60 margin passes
          + [('ppo', tmp, 5000 + i) for i, tmp in enumerate(
             [1.0, 1.5, 2.0] * 20)])                   # 60 stochastic PPO
for k, (mode, eps, seed) in enumerate(PASSES):
    p = rollout(mode, eps, seed)
    best = torch.maximum(best, p)
    if k % 10 == 9:
        say(f'[oracle] pass {k+1}/{len(PASSES)} ({mode} {eps}): '
            f'pass mean {p.mean():.4f}, best-so-far {best.mean():.4f} m, '
            f'vlook/oracle {(ref/best.clamp(min=1e-9)).mean():.4f} '
            f'({time.time()-t0:.0f}s)')
np.savez(OUT / 'offline_oracle.npz', ref=ref.cpu().numpy(),
         best=best.cpu().numpy(), ids=ids.cpu().numpy())
say(f'[oracle] DONE: oracle mean {best.mean():.4f} m; vlook mean '
    f'{ref.mean():.4f}; per-task vlook/oracle mean '
    f'{(ref/best.clamp(min=1e-9)).mean():.4f} median '
    f'{(ref/best.clamp(min=1e-9)).median():.4f} '
    f'({time.time()-t0:.0f}s)')
