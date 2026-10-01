"""FastCycle vs NSRLBatchedEnv on one configuration at a time.

1. Random states along flagship rollouts: observation, joint velocity and the
   predictive stop of FastCycle against env.step on the same state.
2. Closed loop in pure simulation on 300 tasks: env-driven loop vs FastCycle
   loop (both re-plan every 50 ms from the integrated state).
3. Wall time per cycle of each.
"""
import sys, time, dataclasses
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
from Yuan.IJRR.deploy.fast_cycle import FastCycle

WT = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
ROBOT = sys.argv[1] if len(sys.argv) > 1 else 'xarm7'
CFG = {'xarm7': 'config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml',
       'fr3': 'config_line_cont_dirfrac_e8kXXL_rm.yaml'}[ROBOT]
CKPT = {'xarm7': 'runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt',
        'fr3': 'runs/rl_dirfrac_e8kXXL_rm/agent.pt'}[ROBOT]
TOOL = {'xarm7': (-0.0065, -0.0265, 0.206), 'fr3': None}[ROBOT]
dev_env = torch.device('cpu')
torch.set_num_threads(1)

y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / CFG))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw.update(dt=0.05, max_steps=10 ** 6, k_lateral=5.0, n_envs=1)
if TOOL is not None:
    kw.update(tcp_offset=TOOL[2], tool_xyz=TOOL)
env = NSRLBatchedEnv(EnvConfig(**kw), None, dev_env)
rdt = env.kin.dtype
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim'])
ag.load_state_dict(torch.load(WT / 'Yuan/IJRR' / CKPT, map_location='cpu')); ag.eval()
tz = np.load(MAIN / f'runs/paper_fill/ratio_assets/tasks_pool_{ROBOT}.npz')
fast = FastCycle(env, ag, policy_device='cpu')


def env_reset(i):
    env.line_dist = ScriptedLineDistribution({
        'q0': torch.tensor(tz['q0_seed'][i:i + 1], dtype=rdt),
        'line_dir': torch.tensor(tz['cs_line_dir'][i:i + 1], dtype=rdt),
        'n_target': torch.tensor(tz['cs_n_target'][i:i + 1], dtype=rdt)})
    return env.reset()[0].numpy().copy()


@torch.no_grad()
def env_cycle(q):
    """env-driven cycle from state q (what xarm7_realtime does without --fast)."""
    env.q[0] = torch.as_tensor(q, dtype=rdt)
    obs = env.current_obs()
    a = ag.actor_mean(obs)
    qb = env.q.clone()
    _, _, _, _, info = env.step(a, auto_reset=False)
    qdot = ((env.q - qb) / env.dt)[0].numpy().astype(np.float64)
    return obs[0].numpy().copy(), a[0].numpy().copy(), qdot, bool(env.done_persistent[0]), int(info['term_reason'][0])


# ---------------------------------------------------------------- 1. per-state agreement
rng = np.random.default_rng(0)
tasks = rng.choice(len(tz['q0_seed']), 60, replace=False)
d_obs, d_qdot, n_states, n_done_mismatch, n_done = [], [], 0, 0, 0
for i in tasks:
    obs0 = env_reset(i)
    q = env.q[0].numpy().astype(np.float64).copy()
    fast.reset(q, tz['cs_line_dir'][i], tz['cs_n_target'][i])
    for k in range(400):
        # both see the same state; the env also carries its own a_prev / scales
        o_e, a_e, qd_e, done_e, r_e = env_cycle(q)
        out = fast.step(q)
        d_obs.append(np.abs(o_e - out['obs']).max())
        d_qdot.append(np.abs(qd_e - out['qdot']).max())
        n_states += 1
        if done_e or out['done']:
            n_done += 1
            if done_e != out['done']:
                n_done_mismatch += 1
            break
        # advance along the ENV's integration so both stay on the env's trajectory
        q = (q + qd_e * 0.05)
d_obs, d_qdot = np.array(d_obs), np.array(d_qdot)
print(f'[1] {n_states} states: |obs diff| max {d_obs.max():.2e} (p99 {np.percentile(d_obs, 99):.2e}); '
      f'|qdot diff| max {d_qdot.max():.2e} rad/s (p99 {np.percentile(d_qdot, 99):.2e}); '
      f'stop verdict mismatches {n_done_mismatch}/{n_done}', flush=True)


# ---------------------------------------------------------------- 2. closed loop, 300 tasks
def loop_env(i, stroke=1.5):
    env_reset(i); q = env.q[0].numpy().astype(np.float64).copy(); p0 = env.p_start[0].numpy().copy()
    d = tz['cs_line_dir'][i] / np.linalg.norm(tz['cs_line_dir'][i])
    for k in range(2000):
        _, _, qd, done, _ = env_cycle(q)
        if done:
            break
        q = q + qd * 0.05
        if (env.kin.tcp_fk_jac(torch.as_tensor(q[None], dtype=rdt))[0][0].numpy() - p0) @ d >= stroke:
            break
    return float((env.kin.tcp_fk_jac(torch.as_tensor(q[None], dtype=rdt))[0][0].numpy() - p0) @ d)


def loop_fast(i, stroke=1.5):
    q = tz['q0_seed'][i].astype(np.float64)
    fast.reset(q, tz['cs_line_dir'][i], tz['cs_n_target'][i])
    for k in range(2000):
        out = fast.step(q)
        if out['done']:
            break
        q = out['q_new']
        if fast.state(q)['progress'] >= stroke:
            break
    return fast.state(q)['progress']


tasks2 = rng.choice(len(tz['q0_seed']), 300, replace=False)
se, sf = [], []
t_env = t_fast = 0.0
for i in tasks2:
    t0 = time.perf_counter(); se.append(loop_env(i)); t_env += time.perf_counter() - t0
    t0 = time.perf_counter(); sf.append(loop_fast(i)); t_fast += time.perf_counter() - t0
se, sf = np.array(se), np.array(sf)
dd = sf - se
print(f'[2] 300 tasks closed loop: env mean {se.mean():.4f} m, fast mean {sf.mean():.4f} m, '
      f'|diff| median {np.median(np.abs(dd)) * 1000:.2f} mm, max {np.abs(dd).max() * 100:.2f} cm, '
      f'>1 cm on {np.mean(np.abs(dd) > 0.01) * 100:.1f}% of tasks', flush=True)

# ---------------------------------------------------------------- 3. timing
i = int(tasks2[0]); q = tz['q0_seed'][i].astype(np.float64)
fast.reset(q, tz['cs_line_dir'][i], tz['cs_n_target'][i]); env_reset(i)
for _ in range(10): fast.step(q); env_cycle(q)
n = 200
t0 = time.perf_counter()
for _ in range(n): fast.step(q)
t_f = (time.perf_counter() - t0) / n * 1000
t0 = time.perf_counter()
for _ in range(n): env_cycle(q)
t_e = (time.perf_counter() - t0) / n * 1000
# policy alone and the rest
t0 = time.perf_counter()
for _ in range(n): fast.policy(fast.observation(q, fast.state(q)))
t_p = (time.perf_counter() - t0) / n * 1000
t0 = time.perf_counter()
for _ in range(n): fast.state(q)
t_s = (time.perf_counter() - t0) / n * 1000
print(f'[3] per cycle (CPU, 1 thread): FastCycle {t_f:.2f} ms  [state+obs+policy {t_p:.2f}, of which state {t_s:.2f}]  '
      f'vs env-driven {t_e:.2f} ms', flush=True)
if torch.cuda.is_available():
    fast_g = FastCycle(env, ag, policy_device='cuda')
    fast_g.reset(q, tz['cs_line_dir'][i], tz['cs_n_target'][i])
    for _ in range(10): fast_g.step(q)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fast_g.step(q)
    torch.cuda.synchronize()
    print(f'[3] FastCycle with the policy on the GPU: {(time.perf_counter() - t0) / n * 1000:.2f} ms per cycle', flush=True)
