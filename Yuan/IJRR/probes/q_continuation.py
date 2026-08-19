"""User-designed decisive experiment: at bottleneck states s on the winning
route, compare Q under different CONTINUATIONS of the same entry action.

  Q^{pi_PPO}(s, a*)     : search's entry action, then PPO
  Q^{pi_PPO}(s, a_safe) : PPO's own preferred action, then PPO
  Q^{pi_BC}(s, a*)      : search's entry action, then BC(search-distilled)
  Q^{pi_BC}(s, a_safe)  : PPO's action, then BC (bonus row)

Also report PPO's probability rank of a* among the 16 vertices at s.
Values = total arc from q0 reached by the rollout (deterministic policies).
"""
import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz')
rt = np.load(RUN / 'reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
astar_seq = rt['action_idx']
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]
verts = torch.tensor(np.stack(np.meshgrid(*[[-1.,1.]]*4, indexing='ij'),
                     -1).reshape(-1,4), dtype=torch.float32, device=dev)

def load(ck):
    ag = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                     hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(ck, map_location=dev)); ag.eval()
    return ag

ppo = load(RUN / 'agent.pt')                                   # 0.70 policy
bc = load(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2_ei/agent_bc.pt')  # 1.07

@torch.no_grad()
def Q(state, first_action, cont_agent):
    env.line_dist = SingleTaskDistribution(
        {'q0': state[None], 'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    env.step(verts[int(first_action)][None].expand(2, -1), auto_reset=False)
    for t in range(400):
        if bool(env.done_persistent.all()): break
        env.step(cont_agent.actor_mean(env.current_obs()), auto_reset=False)
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    return float(((p[0] - p0) * d).sum())

print("s = winning-route state at arc; a* = search entry action; "
      "a_safe = PPO's own argmax at s")
print(f"{'arc':>5} {'a*':>3} {'aS':>3} {'rankA*':>7} | "
      f"{'Q_PPO(s,a*)':>12} {'Q_PPO(s,aS)':>12} | "
      f"{'Q_BC(s,a*)':>11} {'Q_BC(s,aS)':>11}")
for t in (30, 40, 45, 50, 55, 60):
    s = qr[t]
    a_star = int(astar_seq[t])
    env.line_dist = SingleTaskDistribution(
        {'q0': s[None], 'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    with torch.no_grad():
        logits = ppo._logits_head(ppo._actor_trunk(env.current_obs()))[0]
    a_safe = int(logits.argmax())
    rank = int((logits > logits[a_star]).sum()) + 1
    row = (Q(s, a_star, ppo), Q(s, a_safe, ppo),
           Q(s, a_star, bc), Q(s, a_safe, bc))
    print(f"{t*0.01:5.2f} {a_star:>3} {a_safe:>3} {rank:>4}/16 | "
          f"{row[0]:>12.3f} {row[1]:>12.3f} | {row[2]:>11.3f} "
          f"{row[3]:>11.3f}", flush=True)
