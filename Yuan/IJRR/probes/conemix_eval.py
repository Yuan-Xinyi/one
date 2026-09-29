"""One checkpoint for every tolerance: evaluate the mixed-cone model
(config_line_cont_dirfrac_e8kXXL_conemix, theta_max ~ log-U[5, 30] deg with
theta_max / 30 in the observation) against the per-tolerance checkpoints.

  part 1  30 deg straight 10k, paper protocol (paired with the 30-deg flagship)
  part 2   5 deg straight 10k, c5_final protocol: shared first-candidate
          start, critic-picked start, paired with the 5-deg retrained model
  part 3  15 deg (no dedicated model): 2000 tasks, mixed model vs 30-deg
          flagship zero-shot vs classical

argv: [ckpt] [config yaml] [tag]  (default: the conemix run)
"""
import sys, dataclasses, time, json
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
CKPT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_conemix/agent.pt'
CFG_MIX = sys.argv[2] if len(sys.argv) > 2 else 'config_line_cont_dirfrac_e8kXXL_conemix.yaml'
TAG = sys.argv[3] if len(sys.argv) > 3 else 'conemix'
CFG_RM = 'config_line_cont_dirfrac_e8kXXL_rm.yaml'
CKPT_RM = REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'
B = 2500
keys = {f.name for f in dataclasses.fields(EnvConfig)}
tz = np.load(A / 'tasks_pool_fr3.npz')


def build(cfgfile, ckpt, cone, classical=False):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw['cone_deg'] = float(cone)
    if classical:
        kw.update(dir_frac_action=0, rho_from_norm=False, a_prev_executed=False)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin)), None
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(ckpt, map_location=dev))
    ag.eval()
    return env, (lambda e: ag.actor_mean(e.current_obs())), ag


@torch.no_grad()
def roll(env, fn, q0s, ld, nt):
    rdt = env.kin.dtype
    N = len(q0s); out = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        s2 = {'q0': torch.tensor(q0s[lo:hi], dtype=rdt), 'line_dir': torch.tensor(ld[lo:hi], dtype=rdt),
              'n_target': torch.tensor(nt[lo:hi], dtype=rdt)}
        if pad:
            s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
        env.line_dist = ScriptedLineDistribution({k: v.to(dev) for k, v in s2.items()})
        env.reset()
        for _ in range(env.cfg.max_steps // 2):
            a = fn(env)
            for _ in range(2):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


@torch.no_grad()
def critic_scores(env, ag, CQ, ld_rows, nt_rows):
    rdt = env.kin.dtype
    V = np.zeros(len(CQ), np.float32)
    for lo in range(0, len(CQ), B):
        hi = min(lo + B, len(CQ)); pad = B - (hi - lo)
        s2 = {'q0': torch.tensor(CQ[lo:hi], dtype=rdt), 'line_dir': torch.tensor(ld_rows[lo:hi], dtype=rdt),
              'n_target': torch.tensor(nt_rows[lo:hi], dtype=rdt)}
        if pad:
            s2 = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])]) for k, v in s2.items()}
        env.line_dist = ScriptedLineDistribution({k: v.to(dev) for k, v in s2.items()})
        env.reset()
        V[lo:hi] = ag.get_value(env.current_obs()).float().cpu().numpy().reshape(-1)[:hi - lo]
    return V


def pick_best(CQ, CT, V, q_default):
    pick = q_default.copy(); lo = 0
    while lo < len(CT):
        hi = lo
        while hi < len(CT) and CT[hi] == CT[lo]:
            hi += 1
        pick[CT[lo]] = CQ[lo + int(np.argmax(V[lo:hi]))]
        lo = hi
    return pick


def stat(v, ref, tag, mask=None):
    m = np.ones(len(v), bool) if mask is None else mask
    rt = v[m] / np.maximum(ref[m], 1e-9)
    print(f'  {tag:34s} stroke {v[m].mean():.3f}  ratio {rt.mean()*100:.1f} / {np.percentile(rt, 10)*100:.1f}', flush=True)
    return float(rt.mean() * 100), float(np.percentile(rt, 10) * 100)


res = {}
t0 = time.time()
# ---------------------------------------------------------------- part 1
print('=== part 1: 30 deg straight 10k (paper protocol) ===', flush=True)
Q, LD, NT = tz['q0_seed'], tz['cs_line_dir'], tz['cs_n_target']
env, fn, ag = build(CFG_MIX, CKPT, 30.0)
p30_mix = roll(env, fn, Q, LD, NT); del env, ag; torch.cuda.empty_cache()
env, fn, ag = build(CFG_RM, CKPT_RM, 30.0)
p30_rm = roll(env, fn, Q, LD, NT); del env, ag; torch.cuda.empty_cache()
b = np.load(A / 'bound_pool_fr3.npz'); w = np.load(A / 'witness_pool_fr3.npz')
base = np.load(FU / 'pool_fr3_straight.npz')
ref = np.maximum(b['L_hi'], w['prog'])
for k in base.files:
    if k.endswith('_progress'):
        ref = np.maximum(ref, base[k])
ref = np.maximum(np.maximum(ref, p30_rm), p30_mix)
res['p30_rm'] = stat(p30_rm, ref, '30-deg flagship'); res['p30_mix'] = stat(p30_mix, ref, 'mixed-cone model @30')
d = p30_mix - p30_rm
print(f'  paired: mixed better on {(d > 0.02).mean()*100:.1f}%, worse on {(d < -0.02).mean()*100:.1f}%  [{time.time()-t0:.0f}s]', flush=True)

# ---------------------------------------------------------------- part 2
print('=== part 2: 5 deg straight 10k (c5_final protocol) ===', flush=True)
z = np.load(FU / 'cone5_full10k_assets.npz'); f5 = np.load(FU / 'cone5_formal_10k.npz')
sub, q0f, has, lpw5 = z['sub'], z['q0_first'], z['has'], z['lpw5']
CQ, CT = z['cands_q'], z['cands_t']
ld5, nt5 = LD[sub], NT[sub]
env, fn, ag = build(CFG_MIX, CKPT, 5.0)
p5_mix_shared = roll(env, fn, q0f, ld5, nt5)
p5_mix_at_c5pick = roll(env, fn, f5['pick'], ld5, nt5)          # the 5-deg model's critic picks
V = critic_scores(env, ag, CQ, ld5[CT], nt5[CT])
pick_mix = pick_best(CQ, CT, V, q0f)
p5_mix_sel = roll(env, fn, pick_mix, ld5, nt5)
del env, ag; torch.cuda.empty_cache()
ref5 = np.maximum.reduce([lpw5, f5['p_cls'], f5['p_z30'], f5['p_re'], f5['p_sel'], f5['p_cp'],
                          p5_mix_shared, p5_mix_sel, p5_mix_at_c5pick])
res['p5_cls'] = stat(f5['p_cls'], ref5, 'classical @shared', has)
res['p5_z30'] = stat(f5['p_z30'], ref5, '30-deg flagship zero-shot @shared', has)
res['p5_re'] = stat(f5['p_re'], ref5, '5-deg retrained @shared', has)
res['p5_mix_shared'] = stat(p5_mix_shared, ref5, 'mixed-cone @shared', has)
res['p5_sel'] = stat(f5['p_sel'], ref5, '5-deg retrained @its critic', has)
res['p5_mix_at_c5pick'] = stat(p5_mix_at_c5pick, ref5, 'mixed-cone @5-deg critic pick', has)
res['p5_mix_sel'] = stat(p5_mix_sel, ref5, 'mixed-cone @its own critic', has)
print(f'  [{time.time()-t0:.0f}s]', flush=True)

# ---------------------------------------------------------------- part 3
json.dump(res, open(FU / f'{TAG}_cone_rows.json', 'w'), indent=1)
print('=== part 3: 15 deg, 2000 tasks (no dedicated model) ===', flush=True)
rng = np.random.default_rng(3); s15 = np.sort(rng.choice(len(Q), 2000, replace=False))
out15 = {}
for tag, (cfg, ck, cls) in {'classical': (CFG_RM, CKPT_RM, True),
                            'flagship30 zero-shot': (CFG_RM, CKPT_RM, False),
                            'mixed-cone': (CFG_MIX, CKPT, False)}.items():
    env, fn, ag = build(cfg, ck, 15.0, classical=cls)
    out15[tag] = roll(env, fn, Q[s15], LD[s15], NT[s15]); del env, ag; torch.cuda.empty_cache()
c = out15['classical']
for tag, v in out15.items():
    print(f'  {tag:22s} stroke {v.mean():.3f}  gain x{v.mean()/c.mean():.2f}  win vs classical {(v > c + 0.01).mean()*100:.1f}%  '
          f'p10 {np.percentile(v, 10):.3f}', flush=True)
dm = out15['mixed-cone'] - out15['flagship30 zero-shot']
print(f'  mixed vs flagship: better {(dm > 0.02).mean()*100:.1f}%  worse {(dm < -0.02).mean()*100:.1f}%', flush=True)
np.savez(FU / f'{TAG}_cone_eval.npz', p30_rm=p30_rm, p30_mix=p30_mix, ref30=ref, sub5=sub, has=has, ref5=ref5,
         p5_mix_shared=p5_mix_shared, p5_mix_sel=p5_mix_sel, p5_mix_at_c5pick=p5_mix_at_c5pick, pick_mix=pick_mix,
         s15=s15, **{f'p15_{k.replace(" ", "_")}': v for k, v in out15.items()})
print('saved', FU / f'{TAG}_cone_eval.npz', f'[{time.time()-t0:.0f}s]', flush=True)
