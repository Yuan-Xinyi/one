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
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn

_mu = [a for a in sys.argv[1:] if a.startswith('--mu=')]; MU = float(_mu[0][5:]) if _mu else 0.0
CKPTS = [a[7:] for a in sys.argv[1:] if a.startswith('--ckpt=')] or ['force_mu03full']
EXTRA = [a for a in ('--classical', '--flagship', '--first') if a in sys.argv[1:]]
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
def method_list():
    out = []
    for ck in CKPTS:
        key = 'pick_q' + ('' if ck == 'force' else '_' + ck) + (f'_mu{MU}' if MU > 0 else '')
        out.append((ck, f'config_line_cont_dirfrac_e8kXXL_{ck}.yaml', f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{ck}/agent.pt', d[key]))
        if '--first' in EXTRA:   # the same model from the first admissible candidate (the table's non-critic row)
            out.append((ck + '_first', f'config_line_cont_dirfrac_e8kXXL_{ck}.yaml', f'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_{ck}/agent.pt', d['q0_first']))
    if '--flagship' in EXTRA:
        out.append(('flagship', 'config_line_cont_dirfrac_e8kXXL_rm.yaml', 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', d['q0_first']))
    if '--classical' in EXTRA:
        out.append(('classical', str(Path(hl.ROBOTS['fr3'][0]).name), None, d['q0_first']))
    return out


for name, cfg, ckpt, starts in method_list():
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfg))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw.update(force_kn_max=2000.0, force_set=5.0, force_tol=2.0, k_lateral=5.0, force_mu=MU, n_envs=B, cone_deg=30.0)
    env = NSRLBatchedEnv(EnvConfig(**kw), None, dev)
    if ckpt is None:
        act = cn_action_fn(ClassicalNullspaceController(env.kin))
    else:
        ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(REPO / ckpt, map_location=dev)); ag.eval()
        act = lambda e: ag.actor_mean(e.current_obs())
    rdt = env.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        def _t(a):
            t = torch.tensor(a[lo:hi], dtype=rdt, device=dev)
            return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])]) if pad else t
        env.line_dist = ScriptedLineDistribution({'q0': _t(starts), 'line_dir': _t(dd), 'n_target': _t(nt)})
        env.reset()
        p_start = env.p_start.float().cpu().numpy()
        # full sample record (arc, q) per env, then linear interpolation onto the grid
        arcs, qs = [env.arc_progress.float().cpu().numpy()], [env.q.float().cpu().numpy()]
        alive_hist = [(~env.done_persistent).cpu().numpy()]
        with torch.no_grad():
            for _ in range(env.cfg.max_steps // 2):
                a = act(env)
                for _ in range(2):
                    env.step(a, auto_reset=False)
                    arcs.append(env.arc_progress.float().cpu().numpy()); qs.append(env.q.float().cpu().numpy())
                    alive_hist.append((~env.done_persistent).cpu().numpy())
                if bool(env.done_persistent.all()):
                    break
        arcs = np.stack(arcs); qs = np.stack(qs); alive_hist = np.stack(alive_hist)
        prog = env.arc_progress.float().cpu().numpy()
        for b in range(hi - lo):
            if prog[b] <= best[lo + b]:
                continue
            ok_steps = alive_hist[:, b] | np.concatenate([[True], alive_hist[:-1, b]])   # include the terminal step
            ar, qq = arcs[ok_steps, b], qs[ok_steps, b]
            keep_ = np.concatenate([[True], np.diff(ar) > 1e-6]); ar, qq = ar[keep_], qq[keep_]
            Wb = np.full((NG, 7), np.nan, np.float32)
            g = np.arange(NG) * STEPM; inside = g <= ar[-1] + 1e-9
            if inside.any() and len(ar) >= 2:
                Wb[inside] = np.stack([np.interp(g[inside], ar, qq[:, j]) for j in range(7)], 1)
            elif inside.any():
                Wb[inside] = qq[0]
            best[lo + b] = prog[b]; W[lo + b] = Wb; P0[lo + b] = p_start[b]
        print(f'{name}: witnesses {hi}/{N}', flush=True)
    del env; torch.cuda.empty_cache()
np.savez(FU / f'force_witness_mu{MU}.npz', W=W, step=np.float32(STEPM), p_start=P0, best=best, ckpts=np.array(CKPTS + EXTRA))
print(f'saved: finite witness points {int(np.isfinite(W[:, :, 0]).sum())}, tasks with any {int(np.isfinite(W[:, 0, 0]).sum())}', flush=True)
