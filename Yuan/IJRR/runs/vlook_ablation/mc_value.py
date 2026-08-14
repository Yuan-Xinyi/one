"""Ablation: fit a value function OUTSIDE PPO and feed it to the same
one-step lookahead. Labels are the realized remaining arc length of a mixed
behaviour (margin law, classical law, and noisy variants of both), so the
target is the same quantity PPO's critic estimates, learned by plain
regression. If the lookahead gain survives, the gain belongs to 'a good
value function'; if it does not, it belongs to how PPO trains one.
"""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn as nn, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
import Yuan.IJRR.eval.horizon_ladder as hl

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
kw = dict(y['env'])
kw['n_envs'] = 1024
kw['dt'] = kw['dt'] / hl.SUB                 # paper protocol
kw['max_steps'] = int(y['env']['max_steps'] * hl.SUB)
env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
model = hl.StraightModel(env)
model.terms = [0, 1]
myo = hl.make_myopic(model)
cls = cn_action_fn(ClassicalNullspaceController(env.kin))
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * env.act_dim,
                     indexing='ij'), -1).reshape(-1, env.act_dim),
                     dtype=torch.float32, device=dev)

pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
env.line_dist = pool

OBS, LAB = [], []
t0 = time.time()
BEHAV = [('myopic', 0.0), ('myopic', 0.15), ('myopic', 0.4),
         ('classical', 0.0), ('classical', 0.3), ('random', 1.0)]
for bname, eps in BEHAV:
    env.reset()
    obs_l, alive_l, prog_l = [], [], []
    p0, u = env.p_start.clone(), env.line_dir.clone()
    for t in range(env.max_steps // hl.SUB):
        alive = ~env.done_persistent
        if not bool(alive.any()):
            break
        obs_l.append(env.current_obs().clone())
        alive_l.append(alive.clone())
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog_l.append(((p - p0) * u).sum(-1).clone())
        if bname == 'myopic':
            a = myo(env, env.done_persistent)
        elif bname == 'classical':
            a = cls(env)
        else:
            a = verts[torch.randint(0, verts.shape[0], (env.n_envs,),
                                    device=dev)]
        if eps > 0:
            r = torch.rand(env.n_envs, device=dev) < eps
            a = torch.where(r.unsqueeze(-1),
                            verts[torch.randint(0, verts.shape[0],
                                                (env.n_envs,), device=dev)],
                            a)
        for _ in range(hl.SUB):
            env.step(a, auto_reset=False)
    if not obs_l:
        continue
    O = torch.stack(obs_l); A = torch.stack(alive_l); P = torch.stack(prog_l)
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    final = ((p - p0) * u).sum(-1)
    # label = remaining arc length actually realized from that state, in the
    # same units PPO's critic sees (arc / (v*dt) per step, gamma-discounted)
    gamma = float(y['ppo'].get('gamma', 0.99))
    step_r = (P[1:] - P[:-1]).clamp_min(0) / (env.cfg.v * env.cfg.dt * hl.SUB)
    step_r = torch.cat([step_r, ((final - P[-1]).clamp_min(0)
                                 / (env.cfg.v * env.cfg.dt * hl.SUB))[None]])
    ret = torch.zeros_like(step_r)
    acc = torch.zeros(env.n_envs, device=dev)
    for t in range(step_r.shape[0] - 1, -1, -1):
        acc = step_r[t] + gamma * acc
        ret[t] = acc
    OBS.append(O[A]); LAB.append(ret[A])
    print(f"  {bname:<10s} eps {eps:.2f}: {int(A.sum())} states  "
          f"({time.time()-t0:.0f}s)", flush=True)

OBS = torch.cat(OBS).float(); LAB = torch.cat(LAB).float()
print(f"dataset {OBS.shape[0]} states, label mean {float(LAB.mean()):.2f} "
      f"max {float(LAB.max()):.1f}")

net = nn.Sequential(nn.Linear(env.obs_dim, 512), nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, 1)).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
n = OBS.shape[0]
hold = torch.arange(n, device=dev) % 10 == 0
for ep in range(40_000):
    idx = torch.randint(0, n, (4096,), device=dev)
    m = ~hold[idx]
    loss = nn.functional.smooth_l1_loss(net(OBS[idx][m]).squeeze(-1),
                                        LAB[idx][m])
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 5000 == 0:
        with torch.no_grad():
            pr = net(OBS[hold]).squeeze(-1)
            r2 = 1 - ((pr - LAB[hold]) ** 2).mean() / LAB[hold].var()
        print(f"  fit {ep:>6}  loss {float(loss):.3f}  holdout R2 "
              f"{float(r2):.3f}", flush=True)
out = REPO / 'Yuan/IJRR/runs/vlook_ablation/mc_value.pt'
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(net.state_dict(), out)
print('wrote', out)
