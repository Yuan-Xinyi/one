"""Post-fix checks: straight smoke must reproduce 0.5678 exactly as before;
arc must jump from ~0.26 to ~0.7 with the repo make_vlook."""
import sys, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'selector_ood', MAIN / 'stage1_seed/selector_ood.py')
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

N = 1024
dev = torch.device('cuda')
env, model = so.build_base_env(N, dev)
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
vfn = hl.make_vlook(model, env, ag)


def run(sub):
    env.line_dist = ScriptedLineDistribution(sub)
    env.reset()
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
    for _ in range(env.max_steps // so.SUB):
        a = vfn(env, done)
        for _ in range(so.SUB):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return env.arc_progress.float().cpu().numpy().copy()


# straight smoke (same 1024 tasks as batch_k32 smoke stage; ref 0.5678)
pool = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=20000,
    n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
    feasibility_threshold_m=0.1, verbose=False)
ids = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
dt = env.kin.dtype
q0 = pool.q_pool[ids].to(device=dev, dtype=dt)
p0 = env.kin.tcp_fk_jac(q0)[0].float().cpu()
straight = run({'q0': q0, 'p0': p0,
                'line_dir': pool.line_dir_pool[ids].to(device=dev, dtype=dt),
                'n_target': pool.n_target_pool[ids].to(device=dev, dtype=dt)})
print(f'straight smoke: mean {straight.mean():.4f}  (pre-fix ref 0.5678)',
      flush=True)

# arc check (same 1024 tasks/q0 as vlook_arc_diag subset)
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)
cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt',
                   weights_only=False)
spec = tasks['test_arc']
cd = cands['test_arc']['cands'].cpu().numpy()
sub = {'q0': torch.tensor(cd[:N, 0], dtype=dt).to(dev),
       'p0': spec['p0'][:N],
       'line_dir': spec['line_dir'][:N].to(device=dev, dtype=dt),
       'n_target': spec['n_target'][:N].to(device=dev, dtype=dt),
       'kappa': spec['kappa'][:N]}
arc = run(sub)
print(f'arc with fixed gate: mean {arc.mean():.4f}  '
      f'(buggy 0.258, no-lat-gate 0.718, myopic 0.691 on the 512-subset)',
      flush=True)
