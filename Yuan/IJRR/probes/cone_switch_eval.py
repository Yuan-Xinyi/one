"""Change the tolerance in the middle of a task.

2000 tasks of the 5-deg set with their shared (5-deg admissible) starts, so
every schedule starts from the same state. The cone is set per env from
its own arc progress s, every decision step:
    const30        30 deg throughout
    const5          5 deg throughout
    tighten_hard   30 deg for s < S0, 5 deg after (no warning)
    tighten_ramp   30 deg for s < S0, linear to 5 deg over [S0, S0+RAMP], 5 after
    relax           5 deg for s < S0, 30 deg after
The per-env cone drives the termination test, the cone-margin channel and
(mixed models only) the tolerance observation theta_max / 30.
Recorded per task: final stroke, tilt angle at the first decision with
s >= S0 and with s >= S0+RAMP, whether the task was alive there.
Models: mixed 480M / 240M (observe the cone), dedicated 30 deg and 5 deg
(margin feedback only), classical (theta_max fixed at 30 deg).
"""
import sys, dataclasses, json, time, math
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

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
N = 2000; S0 = 0.20; RAMP = 0.10
keys = {f.name for f in dataclasses.fields(EnvConfig)}
MODELS = {
    'mixed480': ('config_line_cont_dirfrac_e8kXXL_conemix480.yaml', 'runs/rl_dirfrac_e8kXXL_conemix480/agent.pt'),
    'mixed240': ('config_line_cont_dirfrac_e8kXXL_conemix.yaml', 'runs/rl_dirfrac_e8kXXL_conemix/agent.pt'),
    'dedicated30': ('config_line_cont_dirfrac_e8kXXL_rm.yaml', 'runs/rl_dirfrac_e8kXXL_rm/agent.pt'),
    'dedicated5': ('config_line_cont_dirfrac_e8kXXL_cone5.yaml', 'runs/rl_dirfrac_e8kXXL_cone5/agent.pt'),
    'classical': ('config_line_cont_dirfrac_e8kXXL_rm.yaml', None),
}


def schedule(name, s):
    hi, lo = 30.0, 5.0
    if name == 'const30':
        return torch.full_like(s, hi)
    if name == 'const5':
        return torch.full_like(s, lo)
    if name == 'tighten_hard':
        return torch.where(s < S0, hi, lo)
    if name == 'tighten_ramp':
        f = ((s - S0) / RAMP).clamp(0.0, 1.0)
        return hi + (lo - hi) * f
    if name == 'relax':
        return torch.where(s < S0, lo, hi)
    raise KeyError(name)


SCHEDULES = ['const30', 'const5', 'tighten_hard', 'tighten_ramp', 'relax']


def build(cfgfile, ckpt):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2); kw['cone_deg'] = 30.0
    if ckpt is None:
        kw.update(dir_frac_action=0, rho_from_norm=False, a_prev_executed=False)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    if ckpt is None:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin))
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim'],
               cond_log_std_index=(env.obs_dim - 1 if y['ppo'].get('cone_cond_log_std') else None)).to(dev)
    ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR' / ckpt, map_location=dev)); ag.eval()
    return env, (lambda e: ag.actor_mean(e.current_obs()))


def set_cone(env, cone_deg):
    env.cone_deg_b[:] = cone_deg.to(env.kin.dtype)
    env.cos_cone_b[:] = torch.cos(cone_deg.to(env.kin.dtype) * (math.pi / 180.0))
    env._cone_mixed = True


@torch.no_grad()
def tilt_deg(env):
    _, R, _, _ = env.kin.tcp_fk_jac(env.q)
    c = (R[:, :, 2] * env.n_target).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(c))


@torch.no_grad()
def roll(env, fn, sched, q0, ld, nt):
    rdt = env.kin.dtype
    env.line_dist = ScriptedLineDistribution({'q0': torch.tensor(q0, dtype=rdt).to(dev), 'line_dir': torch.tensor(ld, dtype=rdt).to(dev),
                                              'n_target': torch.tensor(nt, dtype=rdt).to(dev)})
    env.reset()
    set_cone(env, schedule(sched, env.arc_progress))
    nan = torch.full((N,), float('nan'), device=dev)
    tilt_a, tilt_b = nan.clone(), nan.clone()          # tilt at first decision with s >= S0 / s >= S0+RAMP (alive)
    tilt_max_after = torch.zeros((N,), device=dev)     # max tilt seen at decisions with s >= S0 (alive)
    for _ in range(env.cfg.max_steps // 2):
        s = env.arc_progress; alive = ~env.done_persistent
        set_cone(env, schedule(sched, s))
        t = tilt_deg(env).float()
        m = alive & (s >= S0) & torch.isnan(tilt_a); tilt_a[m] = t[m]
        m = alive & (s >= S0 + RAMP) & torch.isnan(tilt_b); tilt_b[m] = t[m]
        m = alive & (s >= S0); tilt_max_after[m] = torch.maximum(tilt_max_after[m], t[m])
        a = fn(env)
        for _ in range(2):
            env.step(a, auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    return (env.arc_progress.float().cpu().numpy(), tilt_a.cpu().numpy(), tilt_b.cpu().numpy(), tilt_max_after.cpu().numpy())


tz = np.load(A / 'tasks_pool_fr3.npz'); LD, NT = tz['cs_line_dir'], tz['cs_n_target']
z = np.load(FU / 'cone5_full10k_assets.npz')
sel = np.nonzero(z['has'])[0][:N]
sub = z['sub'][sel]; q0 = z['q0_first'][sel]; ld = LD[sub]; nt = NT[sub]
out = {}; rows = {}; t0 = time.time()
for mname, (cfg, ckpt) in MODELS.items():
    env, fn = build(cfg, ckpt)
    rows[mname] = {}
    for sched in SCHEDULES:
        p, ta, tb, tmax = roll(env, fn, sched, q0, ld, nt)
        out[f'{mname}/{sched}'] = p; out[f'{mname}/{sched}/tilt_at_S0'] = ta; out[f'{mname}/{sched}/tilt_at_S0R'] = tb; out[f'{mname}/{sched}/tilt_max_after'] = tmax
        reached = p >= S0
        died_now = reached & (p < S0 + 0.02)
        after = np.where(reached, p - S0, np.nan)
        r = dict(stroke=float(p.mean()), p10=float(np.percentile(p, 10)), reached_S0=float(reached.mean() * 100),
                 died_within_2cm=float(died_now.sum() / max(reached.sum(), 1) * 100),
                 stroke_after_S0=float(np.nanmean(after)), tilt_at_S0=float(np.nanmean(ta)), tilt_at_S0R=float(np.nanmean(tb)),
                 tilt_max_after=float(np.nanmean(np.where(reached, tmax, np.nan))))
        rows[mname][sched] = {k: round(v, 3) for k, v in r.items()}
        print(f'{mname:12s} {sched:13s} stroke {r["stroke"]:.3f} p10 {r["p10"]:.3f} | reached {S0:.2f}: {r["reached_S0"]:5.1f}%  died<2cm {r["died_within_2cm"]:5.1f}%  '
              f'after {r["stroke_after_S0"]:.3f} | tilt@S0 {r["tilt_at_S0"]:5.1f}  @S0+ramp {r["tilt_at_S0R"]:5.1f}  max-after {r["tilt_max_after"]:5.1f}   [{time.time()-t0:.0f}s]', flush=True)
    del env; torch.cuda.empty_cache()
np.savez(FU / 'cone_switch_eval.npz', sub=sub, q0=q0, **out)
json.dump(rows, open(FU / 'cone_switch_rows.json', 'w'), indent=1)
print('saved', FU / 'cone_switch_rows.json', flush=True)
