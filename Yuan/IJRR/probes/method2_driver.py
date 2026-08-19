"""Method 2: COMPETENCE-GATED reverse curriculum driver.

Windows over the reachtree bank, deepest first. After each 600k-step phase,
measure the policy's mean survival from the window-start states; advance the
window only when it clears the bar (else repeat, max 2 repeats). All from
scratch (no golden data, no BC): pure RL + resets + gating.
"""
import subprocess, sys, re
import numpy as np
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
PY = '/home/lqin/miniconda3/envs/one/bin/python'
RUN = 'single_task_ppo_v2_cgc'
RUND = REPO / 'Yuan/IJRR/runs' / RUN
RUND.mkdir(exist_ok=True)
import shutil
shutil.copy(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2/task.npz',
            RUND / 'task.npz')
BANK = 'Yuan/IJRR/runs/single_task_ppo_v2/reachtree_bank.npz'

sys.path.insert(0, str(REPO))
import torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
from Yuan.IJRR.eval.single_task_ppo import SingleTaskDistribution

dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': 8}), None, dev)
task = np.load(RUND / 'task.npz')
rt = np.load(REPO / 'Yuan/IJRR/runs/single_task_ppo_v2/reachtree.npz')
qr = torch.tensor(rt['q'], device=dev, dtype=env.kin.dtype)
d = torch.tensor(task['line_dir'], device=dev, dtype=env.kin.dtype)
n_t = torch.tensor(task['n_target'], device=dev, dtype=env.kin.dtype)


def survival_from(agent, t0s):
    outs = []
    for t0 in t0s:
        env.line_dist = SingleTaskDistribution(
            {'q0': qr[t0][None], 'line_dir': d[None], 'n_target': n_t[None]})
        env.reset()
        with torch.no_grad():
            for t in range(300):
                env.step(agent.actor_mean(env.current_obs()),
                         auto_reset=False)
                if bool(env.done_persistent.all()):
                    break
        outs.append(float(env.episode_steps.float().mean()))
    return float(np.mean(outs))


def eval_q0(agent):
    env.line_dist = SingleTaskDistribution(
        {'q0': torch.tensor(task['q0'], device=dev,
                            dtype=env.kin.dtype)[None],
         'line_dir': d[None], 'n_target': n_t[None]})
    env.reset()
    with torch.no_grad():
        for t in range(500):
            env.step(agent.actor_mean(env.current_obs()), auto_reset=False)
            if bool(env.done_persistent.all()):
                break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    return float(((p[0] - env.p_start[0]) * env.line_dir[0]).sum())


agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)

windows = [(95, 106), (85, 106), (75, 106), (65, 106), (55, 106),
           (45, 106), (30, 106), (15, 106), (0, 106)]
ckpt = None
log = open(RUND / 'driver.log', 'w')
for lo, hi in windows:
    for attempt in range(3):
        cmd = [PY, '-m', 'Yuan.IJRR.eval.single_task_ppo', '--stage', 'train',
               '--run-dir', RUN, '--restart-bank', BANK,
               '--restart-window', f'{lo},{hi}', '--restart-frac', '0.5',
               '--total-steps', '600000', '--eval-every', '600000']
        if ckpt:
            cmd += ['--resume-from-ckpt', ckpt]
        subprocess.run(cmd, cwd=REPO, capture_output=True)
        ckpt = str(RUND / 'agent.pt')
        agent.load_state_dict(torch.load(ckpt, map_location=dev))
        agent.eval()
        surv = survival_from(agent, list(range(lo, min(lo + 5, 106))))
        need = 0.6 * (106 - lo)
        q0p = eval_q0(agent)
        msg = (f"window[{lo},{hi}] attempt{attempt}: survival {surv:.1f} "
               f"(need {need:.1f})  q0-eval {q0p:.4f}")
        print(msg, flush=True)
        log.write(msg + '\n'); log.flush()
        if surv >= need:
            break
print("DRIVER-DONE", flush=True)
log.write("DRIVER-DONE\n")
log.close()
