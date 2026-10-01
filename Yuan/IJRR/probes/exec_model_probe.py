"""Position streaming vs joint-velocity command: what does the controller's
velocity ramp cost the policy?

The policy decides every 50 ms (task time) from the EXECUTED state, exactly
like the hardware loop.  Between decisions the commanded joint velocity is
executed through one of three models, in task-time units for a time scale s:
  exact      : q += qdot*h                           (position streaming: the
               arm tracks the integrated set-points, so the executed path IS the
               integrated path up to a small delay)
  velocity   : command delay d = 10 ms*s, first-order lag tau = 30 ms*s and
               acceleration limit a = 20 rad/s^2 / s^2 (the arm's own
               velocity-mode ramp), i.e. what joint-velocity mode does
Every sub-step the executed state is checked against the same constraints the
environment terminates on; the environment's own one-step prediction (from the
executed state, exact integration) is the predictive stop, as on hardware.
"""
import sys, dataclasses, math
sys.path.insert(0, '/home/lqin/one')
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, LATERAL_SAFETY_NET
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
dev = torch.device('cuda')
WT = '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/'
MAIN = '/home/lqin/one/Yuan/IJRR/'
B = 1000
H = 0.005            # sub-step [s, task time]
DT = 0.05            # decision period
NSUB = int(round(DT / H))
TOOL = (-0.0065, -0.0265, 0.206)     # XHand index fingertip
MAXT = 10.0          # s task time

y = yaml.safe_load(open('/home/lqin/one/Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw.update(dt=DT, max_steps=10 ** 6, tcp_offset=TOOL[2], tool_xyz=TOOL, k_lateral=5.0, n_envs=B)
env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
rdt = env.kin.dtype
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(WT + 'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt', map_location=dev))
ag.eval()
tz = np.load(MAIN + 'runs/paper_fill/ratio_assets/tasks_pool_xarm7.npz')
rng = np.random.default_rng(5)
sub = np.sort(rng.choice(len(tz['q0_seed']), B, replace=False))
spec = {'q0': torch.tensor(tz['q0_seed'][sub], dtype=rdt, device=dev),
        'line_dir': torch.tensor(tz['cs_line_dir'][sub], dtype=rdt, device=dev),
        'n_target': torch.tensor(tz['cs_n_target'][sub], dtype=rdt, device=dev)}
lo, up = env.kin.lmt_lo, env.kin.lmt_up


@torch.no_grad()
def measured_viol(q):
    p, R, _, _ = env.kin.tcp_fk_jac(q)
    _, _, lat = env._path_frame(p)
    cos = (R[:, :, 2] * env.n_target).sum(-1)
    coll = env.collision.is_collided(env.kin.link_transforms(q))
    jl = ((q < lo) | (q > up)).any(-1)
    cone = cos < env.cos_cone
    tube = lat > LATERAL_SAFETY_NET
    return coll | jl | cone | tube, p, dict(coll=coll, jl=jl, cone=cone, tube=tube)


@torch.no_grad()
def run(model, s=1.0, use_delay=True, use_lag=True, use_acc=True):
    env.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
    env.reset()
    q = env.q.clone()
    v = torch.zeros_like(q)
    active = torch.ones(B, dtype=torch.bool, device=dev)
    prog = torch.zeros(B, dtype=rdt, device=dev)
    reason = torch.zeros(B, dtype=torch.long, device=dev)      # 0 alive, 1 predicted, 2 measured
    tau = 0.03 * s
    a_lim = 20.0 / (s * s)
    delay = int(round(0.01 * s / H)) if use_delay else 0
    kinds = {k: torch.zeros(B, dtype=torch.bool, device=dev) for k in ('coll', 'jl', 'cone', 'tube')}
    cmd_queue = []
    p_prev = env.kin.tcp_fk_jac(q)[0]
    for k in range(int(MAXT / DT)):
        env.q[:] = q
        env.done_persistent[:] = ~active
        a = ag.actor_mean(env.current_obs())
        q_before = env.q.clone()
        env.step(a, auto_reset=False)
        qdot_cmd = (env.q - q_before) / DT
        pred = env.done_persistent & active
        reason = torch.where(pred & (reason == 0), torch.ones_like(reason), reason)
        active = active & ~pred
        cmd_queue.append(qdot_cmd)
        for j in range(NSUB):
            if model == 'exact':
                v = cmd_queue[-1]
            else:
                # command latency: the first `delay` sub-steps of a period still
                # execute the previous period's command
                tgt = cmd_queue[-1] if (j >= delay or len(cmd_queue) < 2) else cmd_queue[-2]
                v_lag = v + (tgt - v) * (1.0 - math.exp(-H / tau)) if use_lag else tgt
                dv = (v_lag - v).clamp(-a_lim * H, a_lim * H) if use_acc else (v_lag - v)
                v = v + dv
            q_new = q + v * H
            viol, p_new, kd = measured_viol(q_new)
            hit = viol & active
            for kk in kinds:
                kinds[kk] |= hit & kd[kk]
            reason = torch.where(hit & (reason == 0), 2 * torch.ones_like(reason), reason)
            # frozen envs keep their state
            q = torch.where(active.unsqueeze(-1), q_new, q)
            d_prog = ((p_new - p_prev) * env.line_dir).sum(-1).clamp_min(0.0)
            prog = prog + torch.where(active, d_prog, torch.zeros_like(d_prog))
            p_prev = torch.where(active.unsqueeze(-1), p_new, p_prev)
            active = active & ~hit
            v = torch.where(active.unsqueeze(-1), v, torch.zeros_like(v))
        if not bool(active.any()):
            break
    kinds = {k: float(v.float().mean()) * 100 for k, v in kinds.items()}
    return prog.float().cpu().numpy(), reason.cpu().numpy(), kinds


base, r0, _ = run('exact')
print(f'exact: mean {base.mean():.4f}', flush=True)
for s_ in (1.0, 0.5, 0.25):
    for name, kwargs in (('delay+lag+acc', {}), ('acc only', dict(use_delay=False, use_lag=False)),
                         ('delay only', dict(use_lag=False, use_acc=False))):
        st, r, kinds = run('velocity', s_, **kwargs)
        print(f's={s_:4.2f} {name:14s} mean {st.mean():.4f} ({(st - base).mean() * 100:+.1f} cm)  '
              f'measured stops {np.mean(r == 2) * 100:5.1f}%  of which coll {kinds["coll"]:.1f} jl {kinds["jl"]:.1f} '
              f'cone {kinds["cone"]:.1f} tube {kinds["tube"]:.1f}', flush=True)
