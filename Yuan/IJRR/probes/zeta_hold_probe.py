"""FlashSAC-style noise-repetition probe on task 27.

Behavior policy = trained PPO policy, but each env intermittently enters a
HOLD: a uniformly random vertex executed for k ~ Zeta(s=2) consecutive steps
(capped). Compare against i.i.d. policy sampling: does temporally-held
exploration ever reach past the 0.70 plateau toward the 1.06 branch?
"""
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

K_MAX = 60
S = 2.0
pk = np.arange(1, K_MAX + 1, dtype=np.float64) ** (-S)
pk /= pk.sum()
pk_t = torch.tensor(pk, device=dev, dtype=torch.float32)

P_START = 0.05      # per-step probability of entering a hold when not holding


@torch.no_grad()
def run_batch(mode, rounds, seed0):
    """mode: 'iid' (pure policy sampling) or 'hold' (policy + zeta holds)."""
    best = -1.0
    all_prog = []
    for r in range(rounds):
        torch.manual_seed(seed0 + r)
        env.reset()
        p0 = env.p_start.clone()
        u = env.line_dir.clone()
        hold_left = torch.zeros(N, dtype=torch.long, device=dev)
        hold_idx = torch.zeros(N, dtype=torch.long, device=dev)
        for t in range(env.max_steps):
            obs = env.current_obs()
            logits = agent._logits_head(agent._actor_trunk(obs))
            a_idx = torch.distributions.Categorical(logits=logits).sample()
            if mode == 'hold':
                start = ((hold_left == 0)
                         & (torch.rand(N, device=dev) < P_START))
                if bool(start.any()):
                    n_s = int(start.sum())
                    hold_idx[start] = torch.randint(0, 16, (n_s,), device=dev)
                    hold_left[start] = torch.multinomial(
                        pk_t, n_s, replacement=True) + 1
                held = hold_left > 0
                a_idx = torch.where(held, hold_idx, a_idx)
                hold_left = (hold_left - held.long()).clamp_min(0)
            env.step(agent.vertices[a_idx], auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p - p0) * u).sum(-1)
        all_prog.append(prog.cpu().numpy())
        best = max(best, float(prog.max()))
    ap = np.concatenate(all_prog)
    return ap, best


for mode in ('iid', 'hold'):
    t0 = time.time()
    ap, best = run_batch(mode, rounds=10, seed0=1234)
    n = len(ap)
    print(f"{mode:>5s}: {n} episodes  max {best:.4f} m  "
          f"P>0.72 {(ap > 0.72).mean():.2e}  "
          f"P>0.80 {(ap > 0.80).mean():.2e}  "
          f"P>0.90 {(ap > 0.90).mean():.2e}  "
          f"P>1.00 {(ap > 1.00).mean():.2e}  ({time.time() - t0:.0f}s)",
          flush=True)
    np.save(RUN / f'probe_{mode}.npy', ap)
