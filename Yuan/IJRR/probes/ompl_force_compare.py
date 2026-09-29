"""Per-task comparison pack + curves: OMPL (force-aware planner, how-far
protocol) vs the friction-aware policy on the same line, same friction.
argv: task index (into the mu=0.3 eval universe) ...
Writes fam_unify/ompl_force_cmp_t{i}.npz (arc-indexed q/tip/zax/k_eff for
both) and ompl_force_cmp_t{i}.png (2x4 joints with hardware limits, full
range, plus stiffness along the arc with the cap and the policy's force error)."""
import sys, dataclasses, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'Yuan/IJRR/probes'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent
from Yuan.IJRR.eval import line_bound as lb

MU, KN_MAX, F_SET, F_TOL = 0.3, 2000.0, 5.0, 2.0
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
WORK = REPO / 'Yuan/IJRR/probes/ompl_force_work'
dev = torch.device('cuda')
d = dict(np.load(FU / 'force_eval_10k_mu0.3.npz'))
sub, lpwf = d['sub'], d['lpwf']
pick = d['pick_q_force_mu03full_mu0.3']
tz = np.load(A / 'tasks_pool_fr3.npz')
p0 = tz['cs_p0'][sub].astype(np.float32)
dd = tz['cs_line_dir'][sub].astype(np.float32); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
env0 = lb.build_env(dev, 'stock', 64); dt0 = env0.kin.dtype
kq0 = torch.as_tensor(env0.kq_joint, device=dev, dtype=dt0)
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_force_mu03full.yaml'))
keys = {f.name for f in dataclasses.fields(EnvConfig)}
kw = {k: v for k, v in y['env'].items() if k in keys}
kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
kw.update(force_kn_max=KN_MAX, force_set=F_SET, force_tol=F_TOL, k_lateral=5.0, force_mu=MU, n_envs=1)
env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_force_mu03full/agent.pt', map_location=dev)); ag.eval()
LIM_LO = env.kin.lmt_lo.cpu().numpy(); LIM_UP = env.kin.lmt_up.cpu().numpy()


@torch.no_grad()
def fk(q, t):
    qt = torch.as_tensor(q, device=dev, dtype=dt0)
    p, R, _, _ = env0.kin.tcp_fk_jac(qt)
    tt = torch.as_tensor(np.tile(t, (len(q), 1)), device=dev, dtype=dt0)
    kn = lb.stiffness_along(env0, qt, R[:, :, 2], kq0, tt, MU)
    return p.float().cpu().numpy(), R[:, :, 2].float().cpu().numpy(), kn.float().cpu().numpy()


@torch.no_grad()
def roll_policy(i):
    rdt = env.kin.dtype
    env.line_dist = ScriptedLineDistribution({'q0': torch.tensor(pick[i:i+1], dtype=rdt, device=dev),
                                              'line_dir': torch.tensor(dd[i:i+1], dtype=rdt, device=dev),
                                              'n_target': torch.tensor(nt[i:i+1], dtype=rdt, device=dev)})
    env.reset(); Q, arc, fe, term = [env.q[0].float().cpu().numpy()], [0.0], [0.0], 'alive'
    for _ in range(env.cfg.max_steps):
        a = ag.actor_mean(env.current_obs())
        _, _, te, tr, info = env.step(a, auto_reset=False)
        Q.append(env.q[0].float().cpu().numpy()); arc.append(float(env.arc_progress[0])); fe.append(float(env._ferr[0]))
        if bool(te[0]) or bool(tr[0]):
            term = TERM_NAMES[int(info['term_reason'][0])]; break
    return np.array(Q, np.float32), np.array(arc, np.float32), np.array(fe, np.float32), term, float(env._depth0[0])


for i in [int(a) for a in sys.argv[1:]]:
    S = np.loadtxt(WORK / f'sol_{i}.txt', skiprows=1).reshape(-1, 8)
    oq, os_ = S[:, :7], S[:, 7]
    keep = np.concatenate([[True], np.diff(os_) > 1e-9]); oq, os_ = oq[keep], os_[keep]
    pq, parc, pfe, pterm, depth0 = roll_policy(i)
    # common arc grid (8 mm) for the overlay; each method frozen past its own end
    grid = np.arange(0.0, max(os_[-1], parc[-1]) + 1e-9, 0.008)
    def on_grid(q, s):
        return np.stack([np.interp(np.minimum(grid, s[-1]), s, q[:, j]) for j in range(7)], 1).astype(np.float32)
    OQ, PQ = on_grid(oq, os_), on_grid(pq, parc)
    otip, ozax, okn = fk(OQ, dd[i]); ptip, pzax, pkn = fk(PQ, dd[i])
    np.savez(FU / f'ompl_force_cmp_t{i}.npz', grid=grid, p0=p0[i], d=dd[i], n=nt[i], depth0=depth0,
             ompl_q=OQ, ompl_tip=otip, ompl_zax=ozax, ompl_kn=okn, ompl_end=os_[-1],
             pol_q=PQ, pol_tip=ptip, pol_zax=pzax, pol_kn=pkn, pol_end=parc[-1],
             pol_ferr=np.interp(np.minimum(grid, parc[-1]), parc, pfe), pol_term=pterm, bound=lpwf[i])
    # ---- curves: 2x4 joints (hardware limits, full range) + stiffness/force row
    fig = plt.figure(figsize=(16, 9.5)); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.05])
    om = grid <= os_[-1] + 1e-9; pm = grid <= parc[-1] + 1e-9
    for j in range(7):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        ax.plot(grid[om], OQ[om, j], color='#1b7f3b', lw=1.8, label='OMPL (force-aware)')
        ax.plot(grid[pm], PQ[pm, j], color='#3d6be0', lw=1.8, label='policy (force-aware)')
        ax.axhline(LIM_LO[j], color='k', ls='--', lw=1.0); ax.axhline(LIM_UP[j], color='k', ls='--', lw=1.0)
        pad = 0.08 * (LIM_UP[j] - LIM_LO[j]); ax.set_ylim(LIM_LO[j] - pad, LIM_UP[j] + pad)
        ax.set_title(f'J{j+1} [rad]', fontsize=10); ax.grid(alpha=0.25)
        if j == 0: ax.legend(fontsize=8, loc='upper right')
        if j >= 3: ax.set_xlabel('arc length [m]')
    ax = fig.add_subplot(gs[1, 3]); ax.axis('off')
    ax.text(0.0, 0.9, f'task {i} (pool idx {sub[i]})\nbound {lpwf[i]:.2f} m\nOMPL reached {os_[-1]:.2f} m\n'
            f'policy reached {parc[-1]:.2f} m ({pterm})\nmu = {MU}, cap {KN_MAX:.0f} N/m', fontsize=11, va='top')
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(grid[om], okn[om], color='#1b7f3b', lw=1.8, label='OMPL'); ax.plot(grid[pm], pkn[pm], color='#3d6be0', lw=1.8, label='policy')
    ax.axhline(KN_MAX, color='#b71c1c', ls='--', lw=1.2, label='cap'); ax.set_ylim(400, 2400)
    ax.set_ylabel('effective stiffness $k_{eff}$ [N/m]'); ax.set_xlabel('arc length [m]'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[2, 2:])
    ax.axhspan(-F_TOL, F_TOL, color='#cfe8d4', alpha=0.8, lw=0); ax.axhline(0, color='#2e7d32', ls='--', lw=1.0)
    ax.plot(grid[pm], np.interp(grid[pm], parc, pfe), color='#3d6be0', lw=1.8, label='policy force error')
    ax.set_ylim(-3, 3); ax.set_ylabel('contact-force error [N]'); ax.set_xlabel('arc length [m]'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.suptitle(f'OMPL vs force-aware policy, task {i}: same line, same start pool, mu = 0.3', fontsize=13)
    fig.tight_layout(); fig.savefig(FU / f'ompl_force_cmp_t{i}.png', dpi=110); plt.close(fig)
    print(f'task {i}: OMPL {os_[-1]:.2f} m, policy {parc[-1]:.2f} m ({pterm}), k_eff max ompl {okn[om].max():.0f} / policy {pkn[pm].max():.0f}', flush=True)
