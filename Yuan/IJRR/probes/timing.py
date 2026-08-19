"""Per-decision latency and per-trajectory wall time of each controller,
at deployment-relevant batch sizes (1 task = the hardware case)."""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.eval.eval_curve import _agent

hl.SUB = 2
dev = torch.device('cuda')
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'))

for B in (1, 16, 128):
    kw = dict(y['env']); kw['n_envs'] = B
    kw['dt'] = kw['dt'] / hl.SUB
    kw['max_steps'] = int(y['env']['max_steps'] * hl.SUB)
    env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    env.line_dist = pool
    model = hl.StraightModel(env)
    model.terms = [0, 1]
    ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
                act_dim=env.act_dim)
    arms = {
        'classical': (lambda f: (lambda e: f(e)))(cn_action_fn(
            ClassicalNullspaceController(env.kin))),
        'RL policy': lambda e: ag.actor_mean(e.current_obs()),
        'margin law': (lambda f: (lambda e: f(e, e.done_persistent)))(
            hl.make_myopic(model)),
        'value lookahead': (lambda f: (lambda e: f(e, e.done_persistent)))(
            hl.make_vlook(model, env, ag)),
        'value beam 4x2': (lambda f: (lambda e: f(e, e.done_persistent)))(
            hl.make_vbeam(model, env, ag, 4, 2)),
    }
    print(f"\n=== batch {B} ===")
    for name, fn in arms.items():
        env.reset()
        ts, steps = [], 0
        with torch.no_grad():
            for t in range(150):
                torch.cuda.synchronize()
                t0 = time.time()
                a = fn(env)
                torch.cuda.synchronize()
                if t >= 10:                     # skip warmup
                    ts.append(time.time() - t0)
                for _ in range(hl.SUB):
                    env.step(a, auto_reset=False)
                steps += 1
                if bool(env.done_persistent.all()):
                    break
        ms = 1000 * np.mean(ts)
        print(f"  {name:<16s} {ms:7.1f} ms/decision  "
              f"({'fits' if ms < 50 else 'MISSES'} the 50 ms budget)  "
              f"~{ms * 60 / 1000:.1f} s per 60-step trajectory")
