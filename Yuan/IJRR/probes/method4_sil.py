"""Method 4: self-imitation learning (SIL) — PPO whose anchor is its OWN
best episode, re-harvested every round. No external data of any kind."""
import subprocess, sys
import numpy as np
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
PY = '/home/lqin/miniconda3/envs/one/bin/python'
RUN = 'single_task_ppo_v2_sil'
RUND = REPO / 'Yuan/IJRR/runs' / RUN
RUND.mkdir(exist_ok=True)
import shutil
shutil.copy(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2/task.npz',
            RUND / 'task.npz')

sys.path.insert(0, str(REPO))
import torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
envH = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2048}), None, dev)
env1 = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 2}), None, dev)
task = np.load(RUND / 'task.npz')
spec = {k: torch.tensor(task[k], device=dev, dtype=envH.kin.dtype).unsqueeze(0)
        for k in ('q0', 'line_dir', 'n_target')}
verts = torch.tensor(
    np.stack(np.meshgrid(*[[-1.0, 1.0]] * 4, indexing='ij'), -1).reshape(-1, 4),
    dtype=torch.float32, device=dev)
agent = VertexAgent(obs_dim=envH.obs_dim, act_dim=envH.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)


@torch.no_grad()
def harvest(ck):
    if ck:
        agent.load_state_dict(torch.load(ck, map_location=dev))
    agent.eval()
    envH.line_dist = SingleTaskDistribution(spec)
    envH.reset()
    A = np.zeros((2048, 200), dtype=np.int64)
    for t in range(200):
        logits = agent._logits_head(agent._actor_trunk(envH.current_obs()))
        a_idx = torch.distributions.Categorical(logits=logits).sample()
        A[:, t] = a_idx.cpu().numpy()
        envH.step(agent.vertices[a_idx], auto_reset=False)
        if bool(envH.done_persistent.all()):
            break
    p, _, _, _ = envH.kin.tcp_fk_jac(envH.q)
    prog = ((p - envH.p_start) * envH.line_dir).sum(-1)
    best = int(prog.argmax())
    steps = int(envH.episode_steps[best])
    # replay best in env1 collecting obs/act
    env1.line_dist = SingleTaskDistribution(spec)
    env1.reset()
    obs_l, act_l = [], []
    for t in range(steps):
        obs_l.append(env1.current_obs()[0].clone())
        act_l.append(int(A[best, t]))
        env1.step(verts[A[best, t]][None].expand(2, -1), auto_reset=False)
    return float(prog[best]), torch.stack(obs_l).cpu().numpy(), np.array(act_l)


best_prog = -1.0
ck = None
log = open(RUND / 'sil_driver.log', 'w')
for rnd in range(10):
    prog, obs, act = harvest(ck)
    if prog > best_prog:
        best_prog = prog
        np.savez(RUND / 'sil_anchor.npz', obs=obs, act=act,
                 ret=np.zeros(len(act)))
    msg = f"round {rnd}: harvested best {prog:.4f} (anchor {best_prog:.4f})"
    print(msg, flush=True); log.write(msg + '\n'); log.flush()
    cmd = [PY, '-m', 'Yuan.IJRR.eval.single_task_ppo', '--stage', 'train',
           '--run-dir', RUN, '--total-steps', '200000',
           '--eval-every', '200000',
           '--anchor-data', f'Yuan/IJRR/runs/{RUN}/sil_anchor.npz',
           '--anchor-coef', '0.5']
    if ck:
        cmd += ['--resume-from-ckpt', ck]
    subprocess.run(cmd, cwd=REPO, capture_output=True)
    ck = str(RUND / 'agent.pt')

# final deterministic eval from q0
agent.load_state_dict(torch.load(ck, map_location=dev)); agent.eval()
env1.line_dist = SingleTaskDistribution(spec)
env1.reset()
with torch.no_grad():
    for t in range(500):
        env1.step(agent.actor_mean(env1.current_obs()), auto_reset=False)
        if bool(env1.done_persistent.all()):
            break
p, _, _, _ = env1.kin.tcp_fk_jac(env1.q)
final = float(((p[0] - env1.p_start[0]) * env1.line_dir[0]).sum())
msg = f"SIL FINAL: q0 deterministic {final:.4f}, best harvested {best_prog:.4f}"
print(msg, flush=True); log.write(msg + '\n'); log.close()
