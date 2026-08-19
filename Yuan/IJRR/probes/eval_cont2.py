"""Continuous-PPO (fixed entropy) 10k evaluation on a robot's pool tasks."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

robot, ckpt = sys.argv[1], sys.argv[2]
hl.SUB = 2
dev = torch.device('cuda')
CFG, _ = hl.ROBOTS[robot]
y = yaml.safe_load(open(REPO / CFG))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2500
env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
ids_all = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:10000]
ag = Agent(env.obs_dim, env.act_dim).to(dev)
ag.load_state_dict(torch.load(REPO / ckpt / 'agent.pt', map_location=dev))
ag.eval()

prog_all = np.zeros(len(ids_all), np.float32)
with torch.no_grad():
    for base in range(0, len(ids_all), B):
        sel = ids_all[base:base + B]
        pad = B - len(sel)
        sp = torch.cat([sel, sel[:1].expand(pad)]) if pad else sel
        env.line_dist = ScriptedLineDistribution(
            {'q0': pool.q_pool[sp].to(dev),
             'line_dir': pool.line_dir_pool[sp].to(dev),
             'n_target': pool.n_target_pool[sp].to(dev)})
        env.reset()
        prog = torch.zeros(B, device=dev)
        a = torch.zeros(B, env.act_dim, device=dev)
        for t in range(env.cfg.max_steps):
            live = ~env.done_persistent
            if not bool(live.any()):
                break
            if t % 2 == 0:
                a = ag.actor_mean(env.current_obs())
            env.step(a, auto_reset=False)
            cur = env.arc_progress.float()
            prog = torch.maximum(prog, torch.where(live, cur, prog))
        prog_all[base:base + len(sel)] = prog[:len(sel)].cpu().numpy()
        print(f'[cont {robot}] {base+len(sel)}/{len(ids_all)} '
              f'mean {prog_all[:base+len(sel)].mean():.4f}', flush=True)
out = f'/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/cont_{robot}_10k.npz'
np.savez(out, prog=prog_all)
print('[cont] wrote', out, 'mean', prog_all.mean())
