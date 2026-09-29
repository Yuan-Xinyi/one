"""Zero-shot transfer across bent tools for any straight-line model:
tip at length L along an axis tilted beta from the flange z (azimuth phi),
2000 (q0, d, n) tasks with the task frame rotated with the tool at q0.
argv: <tag> [config yaml] [ckpt]  -> fam_unify/<tag>_tool_sweep.npz"""
import sys, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn
from Yuan.IJRR.stage2_traj.ppo import Agent
from Yuan.IJRR.kinematics.batched_chain_kin import tool_rotmat
dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'flagship'
CFG = sys.argv[2] if len(sys.argv) > 2 else 'config_line_cont_dirfrac_e8kXXL_rm.yaml'
CKPT = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'
N = 2000
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / CFG))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw0 = {k: v for k, v in y['env'].items() if k in keys}
kw0['dt'] /= 2; kw0['max_steps'] = int(y['env']['max_steps'] * 2)
kw0.pop('tool_random', None); kw0.pop('table_paths', None)       # fixed tool per run, straight rays
tz = np.load(A / 'tasks_pool_fr3.npz')
sub = np.sort(np.random.default_rng(3).choice(len(tz['q0_seed']), N, replace=False))
Q0 = tz['q0_seed'][sub].astype(np.float64)
D0 = tz['cs_line_dir'][sub].astype(np.float64); NT0 = tz['cs_n_target'][sub].astype(np.float64)
env0 = NSRLBatchedEnv(EnvConfig(**{**kw0, 'n_envs': N, 'tool_tilt_deg': 0.0}), None, dev)
rdt = env0.kin.dtype
_, R0, _, _ = env0.kin.tcp_fk_jac(torch.as_tensor(Q0, dtype=rdt, device=dev))
R0 = R0.double().cpu().numpy(); del env0; torch.cuda.empty_cache()


def task_frame(beta, phi):
    Rt = tool_rotmat(beta, phi).astype(np.float64)
    S = np.einsum('bij,jk,blk->bil', R0, Rt, R0)
    return np.einsum('bij,bj->bi', S, D0), np.einsum('bij,bj->bi', S, NT0)


def build(L, beta, phi, classical):
    kw = dict(kw0); kw.update(tcp_offset=L, tool_tilt_deg=beta, tool_azimuth_deg=phi)
    if classical:
        kw.update(dir_frac_action=0, rho_from_norm=False, a_prev_executed=False)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    d, n = task_frame(beta, phi)
    env.line_dist = ScriptedLineDistribution({'q0': torch.tensor(Q0, dtype=rdt, device=dev),
                                             'line_dir': torch.tensor(d, dtype=rdt, device=dev),
                                             'n_target': torch.tensor(n, dtype=rdt, device=dev)})
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin))
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(CKPT, map_location=dev)); ag.eval()
    return env, (lambda e: ag.actor_mean(e.current_obs()))


@torch.no_grad()
def roll(env, fn):
    env.reset()
    for _ in range(env.cfg.max_steps // 2):
        a = fn(env)
        for _ in range(2):
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    return env.arc_progress.float().cpu().numpy()


grid = [(0.2034, 0.0, 0.0), (0.12, 0.0, 0.0), (0.30, 0.0, 0.0), (0.40, 0.0, 0.0)] + \
       [(L, b, 0.0) for L in (0.12, 0.2034, 0.30, 0.40) for b in (10.0, 20.0, 30.0, 45.0)] + \
       [(0.2034, 20.0, 90.0), (0.2034, 20.0, 180.0), (0.2034, 20.0, 270.0), (0.30, 60.0, 0.0), (0.50, 20.0, 0.0)]
res = {}; t0 = time.time()
for L, b, p in grid:
    for cls in (False, True):
        env, fn = build(L, b, p, cls); res[(L, b, p, cls)] = roll(env, fn); del env; torch.cuda.empty_cache()
    f, c = res[(L, b, p, False)], res[(L, b, p, True)]
    print(f'{TAG} L={L:.4f} beta={b:4.1f} phi={p:5.1f}  model {f.mean():.3f}  classical {c.mean():.3f}  gain x{f.mean()/c.mean():.2f}  '
          f'win {(f > c + 0.01).mean()*100:.1f}%  lose {(c > f + 0.01).mean()*100:.1f}%  p10 {np.percentile(f, 10):.3f}  [{time.time()-t0:.0f}s]', flush=True)
np.savez(FU / f'{TAG}_tool_sweep.npz', sub=sub, grid=np.array(grid),
         **{f'{"cls" if k[3] else "rl"}_{int(k[0]*10000)}_{int(k[1])}_{int(k[2])}': v for k, v in res.items()})
print('saved', flush=True)
