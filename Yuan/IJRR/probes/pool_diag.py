"""Why does the distilled policy sit below the one-step margin law?
On held-out tasks: (a) top-1 action agreement with the law along the POLICY's
own states, (b) same along the LAW's states, (c) where each dies."""
import sys
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
import Yuan.IJRR.eval.horizon_ladder as hl

RUN = REPO / 'Yuan/IJRR/runs/pool_v4'
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))
N = 300
env = NSRLBatchedEnv(EnvConfig(**{**y['env'], 'n_envs': N}), None, dev)
model = hl.StraightModel(env)
model.terms = [0, 1]
myo = hl.make_myopic(model)
POW = torch.tensor([8.0, 4.0, 2.0, 1.0], device=dev)
verts = torch.tensor(np.stack(np.meshgrid(*[[-1., 1.]] * 4, indexing='ij'),
                     -1).reshape(-1, 4), dtype=torch.float32, device=dev)

pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
perm = torch.randperm(valid.numel(),
                      generator=torch.Generator().manual_seed(7))
sel = valid[perm[:3000 + 300 + 24]]
import os
OFF = int(os.environ.get('OFF','3000'))
ev = sel[OFF:OFF+300]
spec = {'q0': pool.q_pool[ev].to(dev), 'line_dir': pool.line_dir_pool[ev].to(dev),
        'n_target': pool.n_target_pool[ev].to(dev)}

agent = VertexAgent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                    hidden_dim=y['ppo']['hidden_dim']).to(dev)
agent.load_state_dict(torch.load(RUN / 'agent_pool.pt', map_location=dev))
agent.eval()


@torch.no_grad()
def run(driver):
    env.line_dist = ScriptedLineDistribution({k: v.clone()
                                              for k, v in spec.items()})
    env.reset()
    agree, n_st, steps = 0, 0, np.zeros(N, dtype=int)
    for t in range(env.max_steps):
        alive = ~env.done_persistent
        if not bool(alive.any()):
            break
        o = env.current_obs()
        lab = ((myo(env, env.done_persistent) > 0).float() * POW).sum(-1).long()
        pol = agent._logits_head(agent._actor_trunk(o)).argmax(-1)
        agree += int(((pol == lab) & alive).sum())
        n_st += int(alive.sum())
        steps[alive.cpu().numpy()] += 1
        act = verts[pol] if driver == 'policy' else verts[lab]
        env.step(act, auto_reset=False)
    pf = env.kin.tcp_fk_jac(env.q)[0]
    prog = ((pf - env.p_start) * env.line_dir).sum(-1).cpu().numpy()
    term = env.term_reason.cpu().numpy() if hasattr(env, 'term_reason') else None
    return agree / max(n_st, 1), prog, steps, term


a_pol, p_pol, s_pol, t_pol = run('policy')
a_myo, p_myo, s_myo, t_myo = run('myopic')
print(f"top-1 agreement with the margin law:")
print(f"  along the POLICY's own states : {a_pol:.3f}")
print(f"  along the LAW's states        : {a_myo:.3f}")
print(f"\nprogress: policy {p_pol.mean():.4f} m   law {p_myo.mean():.4f} m   "
      f"ratio {np.mean(p_pol / np.maximum(p_myo, 1e-6)):.3f}")
print(f"episode length: policy {s_pol.mean():.1f}  law {s_myo.mean():.1f}")
r = p_pol / np.maximum(p_myo, 1e-6)
print(f"per-task ratio: median {np.median(r):.3f}  "
      f"frac >= 1.0: {(r >= 1.0).mean():.3f}  "
      f"frac >= 0.9: {(r >= 0.9).mean():.3f}  frac < 0.5: {(r < 0.5).mean():.3f}")
np.savez(RUN / f'diag_{OFF}.npz', agree_policy=a_pol, agree_law=a_myo,
         prog_policy=p_pol, prog_law=p_myo, len_policy=s_pol, len_law=s_myo)
print('wrote', RUN / f'diag_{OFF}.npz')
