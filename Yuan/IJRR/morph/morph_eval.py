"""Evaluate a morphology-general token policy.

part 1  real arms, paper protocol (10k straight tasks each, Ratio mean/p10
        against the pointwise bound + witnesses + every recorded method):
        FR3, xArm7 (v = 0.2), Cobotta (v = 0.05, 6 DoF; a zero-shot
        morphology unless it was trained on), each paired with the arm's
        own flagship checkpoint and the classical law.
part 2  random variants: seen (training seeds) and unseen (fresh seed)
        length/speed variants of the FR3 and the xArm7, 2000 tasks each,
        token policy vs the classical law on the same tasks (no per-variant
        trained model exists).

argv: <tag> <ckpt> [config yaml (for env settings)]
"""
import sys, dataclasses, json, time, copy
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution, LineDistribution
from Yuan.IJRR.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn
from Yuan.IJRR.stage2_traj.ppo import Agent
from Yuan.IJRR.morph import chain_specs as cs
from Yuan.IJRR.morph.token_env import TokenEnv, env_config_for
from Yuan.IJRR.morph.token_agent import TokenAgent

FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
dev = torch.device('cuda'); B = 2500
TAG, CKPT = sys.argv[1], Path(sys.argv[2])
CFG = sys.argv[3] if len(sys.argv) > 3 else str(REPO / 'Yuan/IJRR/morph/config_morph_all.yaml')
y = yaml.safe_load(open(CFG))
base_cfg = dict(y['env'])
FLAG = {'fr3': ('config_line_cont_dirfrac_e8kXXL_rm.yaml', 'runs/rl_dirfrac_e8kXXL_rm/agent.pt'),
        'xarm7': ('config_line_cont_dirfrac_xarm7_e8kXXL_rm.yaml', 'runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt'),
        'cobotta': ('config_line_cont_dirfrac_cobotta_e8kXXL_v005.yaml', 'runs/rl_dirfrac_cobotta_e8kXXL_v005/agent.pt')}
keys = {f.name for f in dataclasses.fields(EnvConfig)}
agent = TokenAgent(hidden_dim=int(y['ppo']['hidden_dim']), d_model=int(y.get('agent', {}).get('d_model', 192)),
                   nhead=int(y.get('agent', {}).get('nhead', 4)), n_layers=int(y.get('agent', {}).get('n_layers', 3))).to(dev)
agent.load_state_dict(torch.load(CKPT, map_location=dev)); agent.eval()
res = {}; t00 = time.time()


def token_env(spec, n_envs):
    cfg = env_config_for(spec, n_envs, base_cfg, dt=0.025, max_steps=int(base_cfg['max_steps'] * 2))
    return TokenEnv(spec, cfg, None, dev)


def classical_env(spec, n_envs):
    cfg = env_config_for(spec, n_envs, base_cfg, dt=0.025, max_steps=int(base_cfg['max_steps'] * 2),
                         dir_frac_action=0, rho_from_norm=False, a_prev_executed=False, observe_proj_scales=False)
    kin, coll = cs.build_kin_collision(spec, dev)
    env = NSRLBatchedEnv(cfg, None, device=dev, kin=kin, collision=coll)
    return env, cn_action_fn(ClassicalNullspaceController(env.kin))


def flagship_env(robot):
    yy = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / FLAG[robot][0]))
    kw = {k: v for k, v in yy['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(yy['env']['max_steps'] * 2)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=yy['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(REPO / 'Yuan/IJRR' / FLAG[robot][1], map_location=dev)); ag.eval()
    return env, (lambda e: ag.actor_mean(e.current_obs()))


@torch.no_grad()
def roll(env, fn, spec_np, N, token=False):
    """spec_np: dict of numpy arrays (q0, line_dir, n_target[, p0])."""
    out = np.zeros(N, np.float32)
    base = env.env if token else env
    rdt = base.kin.dtype
    for lo in range(0, N, B):
        hi = min(lo + B, N); pad = B - (hi - lo)
        s2 = {}
        for k, v in spec_np.items():
            t = torch.tensor(v[lo:hi], dtype=rdt)
            if pad:
                t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
            s2[k] = t.to(dev)
        base.line_dist = ScriptedLineDistribution(s2)
        obs = env.reset()
        for _ in range(base.max_steps // 2):
            a = agent.actor_mean(obs) if token else fn(env)
            for _ in range(2):
                r = env.step(a, auto_reset=False)
                obs = r[0]
            if bool(base.done_persistent.all()):
                break
        out[lo:hi] = base.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


def stat(v, ref, tag):
    rt = v / np.maximum(ref, 1e-9)
    print(f'  {tag:34s} stroke {v.mean():.3f}  ratio {rt.mean()*100:.1f} / {np.percentile(rt, 10)*100:.1f}', flush=True)
    return [round(float(v.mean()), 4), round(float(rt.mean() * 100), 2), round(float(np.percentile(rt, 10) * 100), 2)]


# ================================================================ part 1
print(f'=== part 1: real arms, 10k straight tasks (token policy {TAG}) ===', flush=True)
res['real'] = {}
for robot in ('fr3', 'xarm7', 'cobotta'):
    spec = cs.with_spheres(copy.deepcopy(cs.BASE[robot]))
    tz = np.load(A / f'tasks_pool_{robot}.npz')
    sp = {'q0': tz['q0_seed'], 'line_dir': tz['cs_line_dir'], 'n_target': tz['cs_n_target']}
    N = len(sp['q0'])
    te = token_env(spec, B)
    p_tok = roll(te, None, sp, N, token=True); del te
    fe, ffn = flagship_env(robot)
    p_flag = roll(fe, ffn, sp, N); del fe
    ce, cfn = classical_env(spec, B)
    p_cls = roll(ce, cfn, sp, N); del ce
    torch.cuda.empty_cache()
    b = np.load(A / f'bound_pool_{robot}{"_v2" if robot == "xarm7" else ""}.npz'); w = np.load(A / f'witness_pool_{robot}.npz')
    ref = np.maximum(b['L_hi'], w['prog'])
    base = np.load(FU / f'pool_{robot}_straight.npz')
    for k in base.files:
        if k.endswith('_progress'):
            ref = np.maximum(ref, base[k])
    ref = np.maximum.reduce([ref, p_tok, p_flag, p_cls])
    res['real'][robot] = {'classical': stat(p_cls, ref, f'{robot}: classical'),
                          'flagship': stat(p_flag, ref, f'{robot}: own flagship (240M)'),
                          TAG: stat(p_tok, ref, f'{robot}: token policy {TAG}')}
    d = p_tok - p_flag
    print(f'  paired token vs flagship: better {(d > 0.02).mean()*100:.1f}% worse {(d < -0.02).mean()*100:.1f}%   [{time.time()-t00:.0f}s]', flush=True)
    np.savez(FU / f'morph_{TAG}_{robot}_10k.npz', tok=p_tok, flag=p_flag, cls=p_cls, ref=ref)

# ================================================================ part 2
print('=== part 2: random variants, 2000 tasks each (token policy vs classical) ===', flush=True)
res['variants'] = {}
groups = [('seen', 'fr3', 1, [0, 1]), ('seen', 'xarm7', 2, [0, 1]),
          ('unseen', 'fr3', 7, [0, 1, 2, 3]), ('unseen', 'xarm7', 8, [0, 1, 2, 3])]
for kind, robot, seed, ids in groups:
    rng = np.random.default_rng(seed)
    base = cs.with_spheres(copy.deepcopy(cs.BASE[robot]))
    for k in range(max(ids) + 1):
        v = cs.perturb_spec(base, rng, f'{robot}_v{seed}_{k}')
        if k not in ids:
            continue
        cs.register_specs([v])
        te = token_env(v, B)
        pool = LineDistribution.load_or_build(kin=te.env.kin, collision=te.env.collision, n_pool=2000, n_target_noise_deg=5.0, seed=42,
                                              env_cfg=env_config_for(v, B, base_cfg), feasibility_threshold_m=0.1, verbose=False)
        idx = torch.nonzero(pool.valid_mask).squeeze(-1)[:2000]
        sp = {'q0': pool.q_pool[idx].cpu().numpy(), 'line_dir': pool.line_dir_pool[idx].cpu().numpy(), 'n_target': pool.n_target_pool[idx].cpu().numpy()}
        N = len(idx)
        p_tok = roll(te, None, sp, N, token=True); del te
        ce, cfn = classical_env(v, B); p_cls = roll(ce, cfn, sp, N); del ce; torch.cuda.empty_cache()
        gain = float(p_tok.mean() / max(p_cls.mean(), 1e-9)); win = float((p_tok > p_cls + 0.01).mean() * 100); lose = float((p_cls > p_tok + 0.01).mean() * 100)
        res['variants'][v['name']] = dict(kind=kind, len_scale=[round(x, 2) for x in v['len_scale']], n=N,
                                          tok=round(float(p_tok.mean()), 3), cls=round(float(p_cls.mean()), 3), gain=round(gain, 2), win=round(win, 1), lose=round(lose, 1))
        print(f'  {kind:6s} {v["name"]:12s} scales {np.round(v["len_scale"], 2).tolist()}  token {p_tok.mean():.3f}  classical {p_cls.mean():.3f}  '
              f'gain x{gain:.2f}  win {win:.1f}%  lose {lose:.1f}%   [{time.time()-t00:.0f}s]', flush=True)
json.dump(res, open(FU / f'morph_{TAG}_rows.json', 'w'), indent=1)
print('saved', FU / f'morph_{TAG}_rows.json', flush=True)
