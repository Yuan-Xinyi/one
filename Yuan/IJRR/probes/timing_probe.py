"""Per-decision wall time on task 27: PPO forward pass vs myopic vs the
reachtree expansion, all on the same GPU."""
import time
import numpy as np
import torch
import yaml
from pathlib import Path
import sys

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
import Yuan.IJRR.eval.horizon_ladder as hl
hl.SUB = 1

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 1}), None, dev)
model = hl.StraightModel(env)
model.terms = [0, 1]

RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
env.line_dist = ScriptedLineDistribution(
    {k: torch.tensor(task[k], device=dev, dtype=env.kin.dtype).unsqueeze(0)
     for k in ('q0', 'line_dir', 'n_target')})
env.reset()

agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev))
agent.eval()

myo = hl.make_myopic(model)
obs = env.current_obs()


def clock(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


with torch.no_grad():
    t_ppo = clock(lambda: agent.actor_mean(obs))
    t_myo = clock(lambda: myo(env, env.done_persistent))

    # one reachtree depth at the production width: 16384 states x 16 vertices
    W = 16384
    q = env.q.expand(W, 7).contiguous()
    verts = torch.tensor(
        np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'),
                 -1).reshape(-1, 4), dtype=torch.float32, device=dev)
    d = env.line_dir[0]
    n = env.n_target[0]
    p0 = env.p_start[0]

    def depth():
        qe = q.unsqueeze(1).expand(-1, 16, -1).reshape(-1, 7)
        ae = verts.unsqueeze(0).expand(W, -1, -1).reshape(-1, 4)
        for i in range(0, W * 16, 32768):
            j = min(i + 32768, W * 16)
            qn = model.step(qe[i:j], d.expand(j - i, 3), n.expand(j - i, 3),
                            ae[i:j])
            model.margins(qn, p0.expand(j - i, 3), d.expand(j - i, 3),
                          n.expand(j - i, 3))
    t_tree = clock(depth, n=5)

print(f"PPO actor forward (1 env):            {t_ppo:8.2f} ms / decision")
print(f"myopic (16 successors, 1 env):        {t_myo:8.2f} ms / decision")
print(f"reachtree depth (16384 x 16 states):  {t_tree:8.2f} ms / decision step")
