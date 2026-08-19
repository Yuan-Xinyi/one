"""#9: evaluate the fixed-entropy continuous PPO (rl_cont_sqent_30M) on the
1024 ladder tasks, paper protocol SUB=2; also saturation statistics."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent, PPOConfig

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
B = 1024
kw = dict(y['env']); kw['dt'] /= 2; kw['max_steps'] = int(kw['max_steps'] * 2)
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
ids = valid[:B]

ag = Agent(env.obs_dim, env.act_dim).to(dev)
ag.load_state_dict(torch.load(
    REPO / 'Yuan/IJRR/runs/rl_cont_sqent_30M/agent.pt', map_location=dev))
ag.eval()

@torch.no_grad()
def rollout(fn):
    env.line_dist = ScriptedLineDistribution(
        {'q0': pool.q_pool[ids].to(dev),
         'line_dir': pool.line_dir_pool[ids].to(dev),
         'n_target': pool.n_target_pool[ids].to(dev)})
    env.reset()
    prog = torch.zeros(B, device=dev)
    acts = []
    a = torch.zeros(B, env.act_dim, device=dev)
    for t in range(env.cfg.max_steps):
        live = ~env.done_persistent
        if not bool(live.any()):
            break
        if t % 2 == 0:
            a = fn(env)
            acts.append(a[live].abs().cpu())
        env.step(a, auto_reset=False)
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        cur = ((p - env.p_start) * env.line_dir).sum(-1)
        prog = torch.maximum(prog, torch.where(live, cur, prog))
    return prog, torch.cat(acts)

cl = hl.ClassicalNullspaceController(env.kin)
fcl = hl.cn_action_fn(cl)
p_cl, _ = rollout(lambda e: fcl(e))
p_rl, acts = rollout(lambda e: ag.actor_mean(e.current_obs()))
ok = p_cl > 1e-6
r = (p_rl[ok] / p_cl[ok])
mx = acts.max(dim=-1).values
print(f'classical mean {p_cl.mean():.4f} m')
print(f'cont-sqent mean {p_rl.mean():.4f} m   ratio-to-classical '
      f'{r.mean():.4f} (median {r.median():.4f})')
print(f'saturation: median |a| {acts.median():.3f}; '
      f'component>0.9 frac {(acts > 0.9).float().mean():.3f}; '
      f'max-comp>0.9 frac {(mx > 0.9).float().mean():.3f}')
np.savez(REPO / 'Yuan/IJRR/runs/rl_cont_sqent_30M/ladder_eval.npz',
         p_cl=p_cl.cpu().numpy(), p_rl=p_rl.cpu().numpy())
