"""Plain epsilon-greedy exploration sweep on task 27: behavior = trained
policy, but each step with probability eps a uniformly random vertex is
executed instead. eps=1.0 is the fully random policy. 40960 episodes per
setting; record the frontier."""
import numpy as np
import torch
import yaml
from pathlib import Path
import sys, time

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
N = 4096
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': N}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
spec = {k: torch.tensor(task[k], device=dev, dtype=env.kin.dtype).unsqueeze(0)
        for k in ('q0', 'line_dir', 'n_target')}
env.line_dist = SingleTaskDistribution(spec)

agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev))
agent.eval()


@torch.no_grad()
def sweep(eps, rounds=10, seed0=777):
    best = -1.0
    hits = np.zeros(3)   # >0.72, >0.75, >0.80
    n_tot = 0
    for r in range(rounds):
        torch.manual_seed(seed0 + r)
        env.reset()
        p0 = env.p_start.clone()
        u = env.line_dir.clone()
        for t in range(env.max_steps):
            obs = env.current_obs()
            logits = agent._logits_head(agent._actor_trunk(obs))
            a_idx = torch.distributions.Categorical(logits=logits).sample()
            rnd = torch.randint(0, 16, (N,), device=dev)
            take = torch.rand(N, device=dev) < eps
            a_idx = torch.where(take, rnd, a_idx)
            env.step(agent.vertices[a_idx], auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p - p0) * u).sum(-1).cpu().numpy()
        best = max(best, float(prog.max()))
        hits += [(prog > 0.72).sum(), (prog > 0.75).sum(), (prog > 0.80).sum()]
        n_tot += N
    return best, hits, n_tot


for eps in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    t0 = time.time()
    best, hits, n = sweep(eps)
    print(f"eps={eps:<4}: {n} eps  max {best:.4f} m   "
          f">0.72: {int(hits[0]):>4}  >0.75: {int(hits[1]):>3}  "
          f">0.80: {int(hits[2]):>3}   ({time.time()-t0:.0f}s)", flush=True)
