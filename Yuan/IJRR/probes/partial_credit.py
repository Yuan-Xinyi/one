"""No-partial-credit test: start ON the winning reachtree route at depth t0
(i.e., the first t0 'correct actions' are given for free), hand control to
(a) the trained PPO policy, (b) random actions. How far do they get vs the
0.70 baseline? If being 10-20 steps into the branch still dies at ~0.7,
partially-lucky episodes were invisible to the learning signal."""
import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
N = 2048
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': N}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz'); rt = np.load(RUN / 'reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]
agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent.pt', map_location=dev))
agent.eval()
verts = torch.tensor(np.stack(np.meshgrid(*[[-1.,1.]]*4, indexing='ij'), -1).reshape(-1,4),
                     dtype=torch.float32, device=dev)

@torch.no_grad()
def takeover(t0, mode, rounds=2):
    sp = {'q0': qr[t0][None], 'line_dir': d[None], 'n_target': n_t[None]}
    env.line_dist = SingleTaskDistribution(sp)
    best, mean = -1e9, 0.0
    for r in range(rounds):
        torch.manual_seed(9000 + r)
        env.reset()
        for t in range(400):
            if mode == 'ppo':
                logits = agent._logits_head(agent._actor_trunk(env.current_obs()))
                a_idx = torch.distributions.Categorical(logits=logits).sample()
            else:
                a_idx = torch.randint(0, 16, (N,), device=dev)
            env.step(verts[a_idx], auto_reset=False)
            if bool(env.done_persistent.all()): break
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p - p0) * d).sum(-1)
        best = max(best, float(prog.max())); mean += float(prog.mean()) / rounds
    return best, mean

print("start on the WINNING route at arc s0, then hand over control:")
print(f"{'s0(m)':>6} | ppo-takeover max / mean | random-takeover max / mean")
for t0 in (45, 55, 65, 75, 85, 95):
    bp, mp = takeover(t0, 'ppo')
    br, mr = takeover(t0, 'rand')
    print(f"{t0*0.01:>6.2f} |  {bp:.3f} / {mp:.3f}        |  {br:.3f} / {mr:.3f}", flush=True)
print("(reference: winning route itself reaches 1.061; PPO from q0 dies at 0.70)")
