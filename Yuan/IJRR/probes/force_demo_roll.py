"""Pick a demo task and roll both policies from the SAME critic-picked start
in the implicit-force env (eval protocol: dt/2, two substeps per action),
logging everything the video needs. Candidates: near-vertical pressing
tasks with a long force-aware stroke; among them the one where the
flagship fails earliest on the stiffness cap is chosen."""
import sys, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.stage2_traj.ppo import Agent

FORCE_KW = dict(force_kn_max=2000.0, force_set=5.0, force_tol=2.0, k_lateral=5.0)
dev = torch.device('cuda')
A = MAIN / 'runs/paper_fill/ratio_assets'; FU = MAIN / 'runs/paper_fill/fam_unify'
d = np.load(FU / 'force_eval_v1.npz')
sub, has, p_sel, pick_q = d['sub'], d['has'], d['p_sel'], d['pick_q']
tz = np.load(A / 'tasks_pool_fr3.npz')
nt = tz['cs_n_target'][sub].astype(np.float32); nt /= np.linalg.norm(nt, axis=1, keepdims=True)
dd = tz['cs_line_dir'][sub].astype(np.float32); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
down = nt @ np.float32([0, 0, -1])
cand = np.nonzero(has & (down > 0.95) & (p_sel > 1.0))[0]
print(f'{len(cand)} candidate tasks (near-vertical, force-aware stroke > 1 m)')


def roll(cfgfile, ckpt, idx):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw.update(FORCE_KW)
    B = len(idx)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / ckpt, map_location=dev)); ag.eval()
    rdt = env.kin.dtype
    env.line_dist = ScriptedLineDistribution(
        {'q0': torch.tensor(pick_q[idx], dtype=rdt, device=dev),
         'line_dir': torch.tensor(dd[idx], dtype=rdt, device=dev),
         'n_target': torch.tensor(nt[idx], dtype=rdt, device=dev)})
    env.reset()
    rec = {k: [] for k in ('q', 'tip', 'zax', 'kn', 'ferr', 'depth', 'arc', 'alive')}

    def snap():
        p, R, _, _ = env.kin.tcp_fk_jac(env.q)
        rec['q'].append(env.q.float().cpu().numpy()); rec['tip'].append(p.float().cpu().numpy())
        rec['zax'].append(R[:, :, 2].float().cpu().numpy())
        rec['kn'].append(env._kn.float().cpu().numpy()); rec['ferr'].append(env._ferr.float().cpu().numpy())
        rec['depth'].append(((env.cfg.force_set + env._ferr) / env._kn).float().cpu().numpy())
        rec['arc'].append(env.arc_progress.float().cpu().numpy())
        rec['alive'].append((~env.done_persistent).cpu().numpy())
    snap()
    term = np.full(B, -1, np.int64)
    with torch.no_grad():
        for _ in range(env.cfg.max_steps):
            a = ag.actor_mean(env.current_obs())
            _, _, te, tr, info = env.step(a, auto_reset=False)
            snap()
            newly = (te | tr).cpu().numpy() & (term < 0)
            term[newly] = info['term_reason'].cpu().numpy()[newly]
            if bool(env.done_persistent.all()):
                break
    out = {k: np.array(v, np.float32) for k, v in rec.items()}     # (T, B, ...)
    out['term'] = term
    out['p_start'] = env.p_start.float().cpu().numpy()
    out['depth0'] = env._depth0.float().cpu().numpy()
    return out


rl = roll('config_line_cont_dirfrac_e8kXXL_rm.yaml', 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt', cand)
fo = roll('config_line_cont_dirfrac_e8kXXL_force.yaml', 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_force/agent.pt', cand)
arc_rl, arc_fo = rl['arc'][-1], fo['arc'][-1]
names = np.array([TERM_NAMES[int(t)] if t >= 0 else 'alive' for t in rl['term']])
print('flagship term reasons:', {k: int((names == k).sum()) for k in set(names)})
for k_, (lo_, hi_) in enumerate(((0.15, 0.7), (0.05, 1.0), (0.0, 9.0))):
    good = (names == 'stiff') & (arc_fo > 1.0) & (arc_rl >= lo_) & (arc_rl <= hi_)
    if good.any():
        break
pool = np.nonzero(good)[0] if good.any() else np.arange(len(cand))
for jj in pool:
    print(f'   cand task {cand[jj]}: flagship {arc_rl[jj]:.2f} ({names[jj]}) / force {arc_fo[jj]:.2f}')
j = int(pool[np.argmax(arc_fo[pool] - arc_rl[pool])])
i = int(cand[j])
print(f'chosen task {i} (pool idx {sub[i]}): flagship {arc_rl[j]:.2f} m ({names[j]}), '
      f'force-aware {arc_fo[j]:.2f} m ({TERM_NAMES[int(fo["term"][j])] if fo["term"][j] >= 0 else "alive"})')


def one(out, j):
    alive = out['alive'][:, j]
    T = int(alive.sum()) + 1                      # steps until (and incl.) termination
    T = min(T, len(out['q']))
    return {'q': out['q'][:T, j], 'tip': out['tip'][:T, j], 'zax': out['zax'][:T, j],
            'kn': out['kn'][:T, j], 'ferr': out['ferr'][:T, j], 'depth': out['depth'][:T, j],
            'arc': out['arc'][:T, j],
            'term': TERM_NAMES[int(out['term'][j])] if out['term'][j] >= 0 else 'alive',
            'p_start': out['p_start'][j], 'depth0': float(out['depth0'][j])}


R, F = one(rl, j), one(fo, j)
for tag, o in (('flagship', R), ('force-aware', F)):
    print(f'  {tag}: {len(o["q"])} steps, arc {o["arc"][-1]:.3f} m, term {o["term"]}, '
          f'kn {o["kn"].min():.0f}..{o["kn"].max():.0f}, |ferr| max {np.abs(o["ferr"]).max():.2f} N')
np.savez(FU / 'force_demo_roll.npz', task=i, pool_idx=sub[i], n_target=nt[i], line_dir=dd[i],
         **{f'rl_{k}': v for k, v in R.items()}, **{f'fo_{k}': v for k, v in F.items()})
print('saved')
