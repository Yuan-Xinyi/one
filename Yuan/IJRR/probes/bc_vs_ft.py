"""BC-only vs anchored-PPO-fine-tuned: robustness under start-state noise
and action noise (the closed-loop value PPO is supposed to add)."""
import numpy as np, torch, yaml, sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
N = 512
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': N}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2_ei'
task = np.load(RUN / 'task.npz')
q0 = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)

class NoisyStart:
    def __init__(self, sigma): self.s = sigma
    def sample(self, n, generator=None):
        return {'q0': q0[None] + self.s * torch.randn(n, 7, device=dev,
                                                      dtype=q0.dtype),
                'line_dir': d[None].expand(n, 3).clone(),
                'n_target': n_t[None].expand(n, 3).clone()}

@torch.no_grad()
def run(agent, sigma_deg, act_eps):
    env.line_dist = NoisyStart(np.radians(sigma_deg))
    torch.manual_seed(11)
    env.reset()
    p0 = env.p_start.clone(); u = env.line_dir.clone()
    for t in range(300):
        a = agent.actor_mean(env.current_obs())
        if act_eps > 0:
            rnd = agent.vertices[torch.randint(0, 16, (N,), device=dev)]
            m = (torch.rand(N, device=dev) < act_eps).unsqueeze(-1)
            a = torch.where(m, rnd, a)
        env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()): break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = ((p - p0) * u).sum(-1)
    return float(prog.mean()), float(np.median(prog.cpu())), float(prog.min())

for tag, ck in (('BC-only ', RUN / 'agent_bc.pt'),
                ('anchored', RUN / 'agent.pt')):
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(ck, map_location=dev)); agent.eval()
    for sig, eps in ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (2.0, 0.0),
                     (0.0, 0.05), (0.0, 0.10)):
        mn, md, lo = run(agent, sig, eps)
        print(f"{tag} start-noise {sig:>3}deg act-noise {eps:>4}: "
              f"mean {mn:.3f}  median {md:.3f}  min {lo:.3f}", flush=True)
