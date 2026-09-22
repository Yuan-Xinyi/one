"""Roll policies on the 10k friction universe from their critic-picked
starts and record the joint configuration at every 2 cm of arc, as
witnesses for the pointwise march (a point a method passed is certified
without search). argv: --mu=0.3 --ckpt=force_mu03full [--ckpt=...]
Output: fam_unify/force_witness_mu{mu}.npz with W (N, n_grid, 7), step,
p_start (executed ray origin per task), and per-ckpt progress."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

_mu = [a for a in sys.argv[1:] if a.startswith('--mu=')]; MU = float(_mu[0][5:]) if _mu else 0.0
CKPTS = [a[7:] for a in sys.argv[1:] if a.startswith('--ckpt=')] or ['force_mu03full']
STEPM, NG = 0.02, 91
dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
src = FU / (f'force_eval_10k_mu{MU}.npz' if MU > 0 else 'force_eval_10k.npz')
d = dict(np.load(src)); sub = d['sub']; N = len(sub)
tz = np.load(A / 'tasks_pool_fr3.npz')
dd = tz['cs_line_dir'][sub].astype(np.float32); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
nt = tz['cs_n_target'][sub].astype(np.float32); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
W = np.full((N, NG, 7), np.nan, np.float32); P0 = np.full((N, 3), np.nan, np.float32)
best = np.zeros(N, np.float32)
B = 2000
for ck in CKPTS:
    key = 'pick_q' + ('' if ck == 'force' else '_' + ck) + (f'_mu{MU}' if MU > 0 else '')
    pick = d[key]
    y = yaml.safe_load(open(REPO / f'Yuan/IJRR/stage2_traj/config_line_cont_dirfrac_e8kXXL_{ck}.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw.update(force_kn_max=2000.0, force_set=5.0, force_tol=2.0, k_lateral=5.0, force_mu=MU, n_envs=B)
    env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{ck}/agent.pt', map_location=dev)); ag.eval()
    rdt = env.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        def _t(a):
            t = torch.tensor(a[lo:hi], dtype=rdt, device=dev)
            return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])]) if pad else t
        env.line_dist = ScriptedLineDistribution({'q0': _t(pick), 'line_dir': _t(dd), 'n_target': _t(nt)})
        env.reset()
        Wb = np.full((B, NG, 7), np.nan, np.float32)
        p_start = env.p_start.float().cpu().numpy()
        with torch.no_grad():
            def snap():
                arc = env.arc_progress.float().cpu().numpy(); q = env.q.float().cpu().numpy()
                alive = (~env.done_persistent).cpu().numpy()
                r = np.rint(arc / STEPM).astype(int)
                on = (np.abs(arc - r * STEPM) <= 0.0026) & alive & (r < NG)
                for b in np.nonzero(on)[0]:
                    if np.isnan(Wb[b, r[b], 0]):
                        Wb[b, r[b]] = q[b]
            snap()
            for _ in range(env.cfg.max_steps // 2):
                a = ag.actor_mean(env.current_obs())
                for _ in range(2):
                    env.step(a, auto_reset=False); snap()
                if bool(env.done_persistent.all()):
                    break
        prog = env.arc_progress.float().cpu().numpy()[:hi - lo]
        for b in range(hi - lo):
            if prog[b] > best[lo + b]:
                best[lo + b] = prog[b]; W[lo + b] = Wb[b]; P0[lo + b] = p_start[b]
        print(f'{ck}: witnesses {hi}/{N}', flush=True)
    del env; torch.cuda.empty_cache()
np.savez(FU / f'force_witness_mu{MU}.npz', W=W, step=np.float32(STEPM), p_start=P0, best=best, ckpts=np.array(CKPTS))
print(f'saved: finite witness points {int(np.isfinite(W[:, :, 0]).sum())}, tasks with any {int(np.isfinite(W[:, 0, 0]).sum())}', flush=True)
