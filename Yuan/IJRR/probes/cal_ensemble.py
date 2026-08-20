"""F: 3-seed margobs ensemble with calibrated start-state selection.

Class-0 legal: at t=0 each net's critic scores the CURRENT observation;
a per-net linear calibration (fit on train-pool rollouts) maps V_i(s0) to
predicted progress; the argmax net runs the whole episode. No forward
simulation, no gate, one extra forward per net at episode start."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot  # noqa
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = '/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/'
y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/'
                        'config_line_cont_dirfrac_margobs.yaml'))
kw = {k: v for k, v in y['env'].items()
      if k in {f.name for f in dataclasses.fields(EnvConfig)}}
CKPTS = ['rl_dirfrac_margobs_30M', 'rl_dirfrac_margobs_seedb_30M',
         'rl_dirfrac_margobs_seedc_30M']

# ---- calibration on 6144 train-pool tasks (eval protocol, SUB=2) --------
kw2 = dict(kw); kw2['dt'] = kw['dt'] / 2
kw2['max_steps'] = int(y['env']['max_steps'] * 2)
B = 2048
env = NSRLBatchedEnv(EnvConfig(**{**kw2, 'n_envs': B}), None, dev)
dt = env.kin.dtype
lc = y['line_distribution']
dist = LineDistribution.load_or_build(
    kin=env.kin, collision=env.collision, n_pool=lc['n_pool'],
    n_target_noise_deg=lc['n_target_noise_deg'], seed=lc['train_seed'],
    env_cfg=EnvConfig(**{**kw, 'n_envs': B}),
    feasibility_threshold_m=(float(lc['feasibility_threshold_m'])
                             if lc.get('feasibility_filter') else None),
    swing_max_deg=lc.get('swing_max_deg', 0.0))
valid = torch.nonzero(dist.valid_mask, as_tuple=False).squeeze(-1)
rng = np.random.default_rng(11)
cal_rows = valid[torch.tensor(
    rng.choice(valid.shape[0], 6144, replace=False))]

agents = []
for ck in CKPTS:
    a = Agent(env.obs_dim, env.act_dim_policy).to(dev)
    a.load_state_dict(torch.load(
        REPO / f'Yuan/IJRR/runs/{ck}/agent.pt', map_location=dev))
    a.eval()
    agents.append(a)


def scripted(rows):
    return ScriptedLineDistribution(
        {'q0': dist.q_pool[rows].to(dev, dt),
         'line_dir': dist.line_dir_pool[rows].to(dev, dt),
         'n_target': dist.n_target_pool[rows].to(dev, dt)})


@torch.no_grad()
def run_batches(agent_of_batch, rows_all, v0_out=None, which=None):
    N = rows_all.shape[0]
    out = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        rows = rows_all[lo:hi]
        if pad:
            rows = torch.cat([rows, rows[:1].expand(pad)])
        env.line_dist = scripted(rows)
        env.reset()
        obs0 = env.current_obs()
        if v0_out is not None:
            for k, a in enumerate(agents):
                v0_out[k, lo:hi] = a.get_value(obs0).float().cpu().numpy()[
                    :hi - lo]
        sel = (which[lo:hi] if which is not None
               else np.full(hi - lo, agent_of_batch, np.int64))
        selp = np.concatenate([sel, np.zeros(pad, np.int64)]) if pad else sel
        selt = torch.tensor(selp, device=dev)
        for _ in range(env.cfg.max_steps // 2):
            obs = env.current_obs()
            act = torch.zeros(env.n_envs, env.act_dim_policy, device=dev)
            for k, a in enumerate(agents):
                m = selt == k
                if bool(m.any()):
                    act[m] = a.actor_mean(obs[m])
            for _ in range(2):
                env.step(act, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'  {hi}/{N}', flush=True)
    return out


# per-net calibration rollouts
V0 = np.zeros((3, 6144), np.float32)
P = np.zeros((3, 6144), np.float32)
for k in range(3):
    print(f'[cal] rollout net {k}', flush=True)
    P[k] = run_batches(k, cal_rows, v0_out=V0 if k == 0 else None)
coef = []
for k in range(3):
    A_ = np.stack([V0[k], np.ones_like(V0[k])], 1)
    c, *_ = np.linalg.lstsq(A_, P[k], rcond=None)
    pred = A_ @ c
    r = np.corrcoef(pred, P[k])[0, 1]
    coef.append(c)
    print(f'[cal] net {k}: prog = {c[0]:.4f}*V + {c[1]:.4f}   corr {r:.3f}',
          flush=True)

# ---- ensemble on the 10k eval pool --------------------------------------
tz = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/ratio_assets/'
             'tasks_pool_fr3.npz')
N = 10000
V0e = np.zeros((3, N), np.float32)
# first pass: V(s0) for all nets on all tasks (reset-only, cheap)
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        ids = np.arange(lo, hi)
        ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        env.line_dist = ScriptedLineDistribution(
            {'q0': torch.tensor(tz['q0_seed'][ip], dtype=dt, device=dev),
             'line_dir': torch.tensor(tz['cs_line_dir'][ip], dtype=dt,
                                      device=dev),
             'n_target': torch.tensor(tz['cs_n_target'][ip], dtype=dt,
                                      device=dev)})
        env.reset()
        obs0 = env.current_obs()
        for k, a in enumerate(agents):
            V0e[k, lo:hi] = a.get_value(obs0).float().cpu().numpy()[:hi - lo]
pred = np.stack([coef[k][0] * V0e[k] + coef[k][1] for k in range(3)])
which = pred.argmax(0)
print('[ens] pick shares:', np.bincount(which, minlength=3) / N, flush=True)

rows_eval = torch.arange(N)
out = np.zeros(N, np.float32)
with torch.no_grad():
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        pad = B - (hi - lo)
        ids = np.arange(lo, hi)
        ip = np.concatenate([ids, np.full(pad, ids[0])]) if pad else ids
        env.line_dist = ScriptedLineDistribution(
            {'q0': torch.tensor(tz['q0_seed'][ip], dtype=dt, device=dev),
             'line_dir': torch.tensor(tz['cs_line_dir'][ip], dtype=dt,
                                      device=dev),
             'n_target': torch.tensor(tz['cs_n_target'][ip], dtype=dt,
                                      device=dev)})
        env.reset()
        sel = which[lo:hi]
        selp = (np.concatenate([sel, np.zeros(pad, np.int64)])
                if pad else sel)
        selt = torch.tensor(selp, device=dev)
        for _ in range(env.cfg.max_steps // 2):
            obs = env.current_obs()
            act = torch.zeros(env.n_envs, env.act_dim_policy, device=dev)
            for k, a in enumerate(agents):
                m = selt == k
                if bool(m.any()):
                    act[m] = a.actor_mean(obs[m])
            for _ in range(2):
                env.step(act, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'[ens] {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)

np.savez(FU + 'ensemble3_10k.npz', prog=out, which=which, V0=V0e,
         coef=np.array(coef))
mg = np.load(FU + 'dirfrac_margobs_10k.npz')['prog']
print(f'ensemble mean {out.mean():.4f}  vs margobs single {mg.mean():.4f}',
      flush=True)
