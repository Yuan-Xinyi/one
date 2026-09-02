"""xArm7 / Cobotta DirFrac evaluation: straight 10k pool + three curved
families, ratio recipes verified against the printed table rows (vlook)
before being applied to the DirFrac cells."""
import sys, dataclasses, time
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'
A = MAIN / 'runs/paper_fill/ratio_assets'
CFG = {'xarm7': 'config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml',
       'cobotta': 'config_line_cont_dirfrac_cobotta.yaml'}
CKPT = {'xarm7': 'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt',
        'cobotta': 'Yuan/IJRR/runs/rl_dirfrac_cobotta_XXL/agent.pt'}


def build(robot, batch, k_lateral=None):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / CFG[robot]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2
    kw['max_steps'] = int(y['env']['max_steps'] * 2)
    if k_lateral is not None:
        kw['k_lateral'] = k_lateral
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / CKPT[robot], map_location=dev))
    ag.eval()
    return env, ag


@torch.no_grad()
def run(env, ag, spec, N):
    out = np.zeros(N, np.float32)
    B = env.n_envs
    dt = env.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        sub = {}
        for k, v in spec.items():
            t = v[lo:hi]
            if pad:
                t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
            sub[k] = t
        for k2 in ('q0', 'line_dir', 'n_target'):
            sub[k2] = sub[k2].to(device=dev, dtype=dt)
        env.line_dist = ScriptedLineDistribution(sub)
        env.reset()
        for _ in range(env.cfg.max_steps // 2):
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


def stat(v, ref, tag):
    ref = np.maximum(ref, v)
    rt = v / np.maximum(ref, 1e-9)
    print(f'{tag}: stroke {v.mean():.3f}  ratio {rt.mean()*100:.1f} / '
          f'{np.percentile(rt, 10)*100:.1f}', flush=True)


for robot in ('xarm7',):
    # ---- straight 10k ---------------------------------------------------
    tz = np.load(A / f'tasks_pool_{robot}.npz')
    base = np.load(FU / f'pool_{robot}_straight.npz')
    for bname in ([f'bound_pool_{robot}_v2', f'bound_pool_{robot}']
                  if robot == 'xarm7' else [f'bound_pool_{robot}']):
        b = np.load(A / f'{bname}.npz')
        w = np.load(A / f'witness_pool_{robot}.npz')
        ref = np.maximum(b['L_hi'], w['prog'])
        for k in base.files:
            if k.endswith('_progress'):
                ref = np.maximum(ref, base[k])
        vl = base['vlook_progress']
        rt = vl / np.maximum(ref, 1e-9)
        print(f'[verify] {robot} {bname}: vlook {vl.mean():.3f} '
              f'{rt.mean()*100:.1f}/{np.percentile(rt,10)*100:.1f}',
              flush=True)
    env, ag = build(robot, 2500)
    spec = {'q0': torch.tensor(tz['q0_seed']),
            'line_dir': torch.tensor(tz['cs_line_dir']),
            'n_target': torch.tensor(tz['cs_n_target'])}
    t0 = time.time()
    v = run(env, ag, spec, len(tz['q0_seed']))
    np.savez_compressed(FU / f'dirfrac_{robot}e8k_straight.npz', prog=v)
    stat(v, ref, f'[dirfrac] {robot} straight ({time.time()-t0:.0f}s)')
    del env, ag
    torch.cuda.empty_cache()

    # ---- curved families -------------------------------------------------
    env, ag = build(robot, 2500, k_lateral=5.0)
    for fam in ('serpentine', 'nonplanar'):
        tz = np.load(A / f'tasks_selx_{fam}_{robot}.npz')
        spec = {'q0': torch.tensor(tz['q0_seed']),
                'line_dir': torch.tensor(tz['cs_line_dir']),
                'n_target': torch.tensor(tz['cs_n_target'])}
        for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
            if k in tz.files:
                spec[k] = torch.tensor(tz[k])
        b = np.load(A / f'bound_selx_{fam}_{robot}.npz')
        w = np.load(A / f'witness_selx_{fam}_{robot}.npz')
        c = np.load(FU / f'ctrl_{robot}_{fam}.npz')
        ref = np.maximum(b['L_hi'], w['prog'])
        for k in c.files:
            if k.endswith('_progress'):
                ref = np.maximum(ref, c[k])
        vl = c['vlook_progress']
        rt = vl / np.maximum(ref, 1e-9)
        print(f'[verify] {robot} {fam}: vlook {vl.mean():.3f} '
              f'{rt.mean()*100:.1f}/{np.percentile(rt,10)*100:.1f}',
              flush=True)
        v = run(env, ag, spec, len(tz['q0_seed']))
        np.savez_compressed(FU / f'dirfrac_{robot}e8k_{fam}.npz', prog=v)
        stat(v, ref, f'[dirfrac] {robot} {fam}')
    del env, ag
    torch.cuda.empty_cache()
print('all done', flush=True)
