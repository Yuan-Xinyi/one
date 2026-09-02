"""Velocity-budget anatomy across arms: how much of the joint-velocity
budget does task tracking alone consume, and how much amplitude alpha_feas
is left for null-space adjustment?

Per robot: roll 1024 straight tasks with the mainline policy, and at every
substep mirror the env's own qdot_task / xi_dir / alpha_feas computation
(env.py step(), dir_frac==2 branch). The mirror is validated against the
executed (q_next - q)/dt, so every number is certified to be what the
controller actually experienced. Cobotta is additionally re-quantified
counterfactually with qd_limit x 2.7 at the SAME visited states.
"""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'
N = 1024
CFG = {'fr3': ('config_line_cont_dirfrac_e8kXXL_rm.yaml',
               'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'),
       'xarm7': ('config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml',
                 'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt'),
       'cobotta': ('config_line_cont_dirfrac_cobotta.yaml',
                   'Yuan/IJRR/runs/rl_dirfrac_cobotta_XXL/agent.pt')}

for robot, (cfg_f, ckpt) in CFG.items():
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfg_f))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2
    kw['max_steps'] = int(y['env']['max_steps'] * 2)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    assert int(getattr(env.cfg, 'dir_frac_action', 0)) == 2
    assert getattr(env.cfg, 'rho_from_norm', False)
    assert not getattr(env.cfg, 'alpha_joint', False)
    assert not int(getattr(env.cfg, 'dv_metric', 0) or 0)
    assert not getattr(env.cfg, 'task_gate', False) and not env.cfg.speed_levels
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / ckpt, map_location=dev))
    ag.eval()
    dt_t = env.kin.dtype
    nj = env.n_joints
    lim = env.qd_limit                                    # (nj,)

    tz = np.load(A / f'tasks_pool_{robot}.npz')
    sub = {'q0': torch.tensor(tz['q0_seed'][:N], dtype=dt_t, device=dev),
           'line_dir': torch.tensor(tz['cs_line_dir'][:N], dtype=dt_t,
                                    device=dev),
           'n_target': torch.tensor(tz['cs_n_target'][:N], dtype=dt_t,
                                    device=dev)}
    env.line_dist = ScriptedLineDistribution(sub)
    env.reset()

    S_task, ALPHA, RHO, EXEC, mirror_err = [], [], [], [], 0.0
    CF = [] if robot == 'cobotta' else None               # (s_task, alpha) x2.7
    with torch.no_grad():
        for _ in range(env.cfg.max_steps // 2):
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                act = ~env.done_persistent
                if not bool(act.any()):
                    break
                q_before = env.q.clone()
                # ---- mirror of env.step() dir_frac==2 -------------------
                aa = a.clamp(-1.0, 1.0).to(dtype=env.kin.dtype)
                u = aa[:, :nj]
                _, _, J, _ = env.kin.tcp_fk_jac(q_before)
                J_p = J[:, :3, :]
                J_plus, _ = damped_pinv(J_p, env.cfg.lambda_0,
                                        env.cfg.sigma_thr)
                x_dot = (env.v * env.line_dir).unsqueeze(-1)
                qdot_task = (J_plus @ x_dot).squeeze(-1)
                _, _, Vh = torch.linalg.svd(J_p.double(), full_matrices=True)
                Nn = Vh.transpose(-1, -2)[..., 3:]
                xi = (Nn @ (Nn.transpose(-1, -2)
                            @ u.double().unsqueeze(-1))).squeeze(-1)
                xin = xi.norm(dim=-1, keepdim=True)
                xi = (xi / xin.clamp_min(1e-6)).to(env.kin.dtype)
                rho = xin.squeeze(-1).clamp(max=1.0).to(env.kin.dtype)
                head = (lim - qdot_task) / xi.clamp_min(1e-9)
                room = (lim + qdot_task) / (-xi).clamp_min(1e-9)
                alpha = torch.where(xi >= 0, head, room
                                    ).amin(-1).clamp_min(0.0)
                qdot_null = (rho * alpha).unsqueeze(-1) * xi
                # ---- diagnostics on active envs -------------------------
                s_task = (qdot_task.abs() / lim).amax(-1)
                S_task.append(s_task[act].float().cpu())
                ALPHA.append(alpha[act].float().cpu())
                RHO.append(rho[act].float().cpu())
                EXEC.append((qdot_null.abs() / lim).amax(-1)[act]
                            .float().cpu())
                if CF is not None:
                    l2 = lim * 2.7
                    h2 = (l2 - qdot_task) / xi.clamp_min(1e-9)
                    r2 = (l2 + qdot_task) / (-xi).clamp_min(1e-9)
                    a2 = torch.where(xi >= 0, h2, r2).amin(-1).clamp_min(0.0)
                    CF.append(torch.stack(
                        [(qdot_task.abs() / l2).amax(-1)[act],
                         a2[act]]).float().cpu())
                # ---- step + mirror validation ---------------------------
                env.step(a, auto_reset=False)
                qd_real = (env.q - q_before) / env.dt
                qd_pred = qdot_task + qdot_null
                mirror_err = max(mirror_err, float(
                    (qd_real - qd_pred)[act].abs().max()))
            if bool(env.done_persistent.all()):
                break

    S = torch.cat(S_task).numpy(); AL = torch.cat(ALPHA).numpy()
    RH = torch.cat(RHO).numpy(); EX = torch.cat(EXEC).numpy()
    print(f'\n=== {robot} (nj={nj}, v={env.v} m/s, {len(S)} substeps, '
          f'mirror max err {mirror_err:.2e} rad/s) ===')
    print(f'  task saturation  max_i |qd_task|/lim : '
          f'mean {S.mean():.3f}  med {np.median(S):.3f}  '
          f'p90 {np.percentile(S, 90):.3f}   '
          f'>70% of budget: {(S > 0.7).mean() * 100:.1f}% of steps   '
          f'>100%: {(S > 1.0).mean() * 100:.1f}%')
    print(f'  alpha_feas (rad/s, chosen dir)       : '
          f'mean {AL.mean():.3f}  med {np.median(AL):.3f}  '
          f'p10 {np.percentile(AL, 10):.3f}   '
          f'<0.05: {(AL < 0.05).mean() * 100:.1f}% of steps')
    print(f'  policy rho (requested fraction)      : mean {RH.mean():.3f}')
    print(f'  executed null max_i |qd_null|/lim    : mean {EX.mean():.3f}  '
          f'med {np.median(EX):.3f}')
    if CF is not None:
        C = torch.cat(CF, dim=1).numpy()
        print(f'  [x2.7 limits, same states] task sat  : '
              f'mean {C[0].mean():.3f}  p90 {np.percentile(C[0], 90):.3f}  '
              f'>70%: {(C[0] > 0.7).mean() * 100:.1f}%')
        print(f'  [x2.7 limits, same states] alpha_feas: '
              f'mean {C[1].mean():.3f}  med {np.median(C[1]):.3f}  '
              f'<0.05: {(C[1] < 0.05).mean() * 100:.1f}%')
    del env, ag
    torch.cuda.empty_cache()
print('\nall done', flush=True)
