import numpy as np, torch, yaml, sys, re
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

txt = open(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2_curr/train_stdout.log').read()
p = np.array([float(m.group(1)) for m in re.finditer(r'progress (\d+\.\d+) m', txt)])
print(f"curr eval: n={len(p)} peak={p.max():.4f} last10_mean={p[-10:].mean():.4f}")

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2}), None, dev)
RUN = REPO / 'Yuan/IJRR/runs/single_task_ppo_v2'
task = np.load(RUN / 'task.npz'); rt = np.load(RUN / 'reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)
q0_t = torch.tensor(task['q0'], device=dev, dtype=env.kin.dtype)
p0 = env.kin.tcp_fk_jac(q0_t[None])[0][0]

for tag, ck in (('archive ', REPO / 'Yuan/IJRR/runs/single_task_ppo_v2_archive/agent.pt'),
                ('currqlum', REPO / 'Yuan/IJRR/runs/single_task_ppo_v2_curr/agent.pt')):
    agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                        hidden_dim=y['ppo']['hidden_dim']).to(dev)
    agent.load_state_dict(torch.load(ck, map_location=dev)); agent.eval()
    row = []
    for t0 in (0, 35, 45, 55, 65, 75, 85, 95):
        qs = q0_t if t0 == 0 else qr[t0]
        env.line_dist = SingleTaskDistribution(
            {'q0': qs[None], 'line_dir': d[None], 'n_target': n_t[None]})
        env.reset()
        with torch.no_grad():
            for t in range(400):
                env.step(agent.actor_mean(env.current_obs()), auto_reset=False)
                if bool(env.done_persistent.all()): break
        p_, _, _, _ = env.kin.tcp_fk_jac(env.q)
        row.append(float(((p_ - p0) * d).sum(-1)[0]))
    print(f"{tag} from q0/0.35/0.45/0.55/0.65/0.75/0.85/0.95: "
          + "  ".join(f"{v:.3f}" for v in row), flush=True)
