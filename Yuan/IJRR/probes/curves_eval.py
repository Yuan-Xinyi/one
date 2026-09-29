"""Evaluate a curve-trained model (generic-curve tables + preview observation)
against the straight-trained flagship.

  part 1  straight 10k, paper protocol (k_lateral 0) and with k_lateral 5
  part 2  serpentine / rotating-axis 2.5k families (analytic paths, ratio
          against the same references as the paper)
  part 3  start selection: first / random / critic / within-pool oracle on
          the K=32 candidate sets (straight: 2000-task subset of the 10k,
          paired with the flagship's saved labels; curved: full 2.5k)
  part 4  figure families as table paths from the straight pool tasks
          (circle, square, triangle, helix; 2000 each): flagship zero-shot
          vs curve model vs classical, stroke and completion

argv: <tag> [config yaml] [ckpt]; PARTS=1234 selects parts.
"""
import sys, os, dataclasses, time, json, math
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import matplotlib; matplotlib.use('Agg')
import numpy as np, torch, yaml
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn
from Yuan.IJRR.env.path_table import integrate_frames, M as TAB_M, DS as TAB_DS
from Yuan.IJRR.stage2_traj.ppo import Agent

dev = torch.device('cuda')
FU = MAIN / 'runs/paper_fill/fam_unify'; A = MAIN / 'runs/paper_fill/ratio_assets'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'curves'
CFG_C = sys.argv[2] if len(sys.argv) > 2 else 'config_line_cont_dirfrac_e8kXXL_curves.yaml'
CKPT_C = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_curves/agent.pt'
PARTS = os.environ.get('PARTS', '1234')
CFG_RM = 'config_line_cont_dirfrac_e8kXXL_rm.yaml'
CKPT_RM = REPO / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt'
B = 2500
keys = {f.name for f in dataclasses.fields(EnvConfig)}
res = {}
t00 = time.time()


def build(cfgfile, ckpt, classical=False, **over):
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj' / cfgfile))
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] /= 2; kw['max_steps'] = int(y['env']['max_steps'] * 2)
    kw.update(over)
    if classical:
        kw.update(dir_frac_action=0, rho_from_norm=False, a_prev_executed=False)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': B}), None, dev)
    if classical:
        return env, cn_action_fn(ClassicalNullspaceController(env.kin)), None
    ag = Agent(env.obs_dim, env.act_dim_policy, hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(ckpt, map_location=dev))
    ag.eval()
    return env, (lambda e: ag.actor_mean(e.current_obs())), ag


def _sub(spec, lo, hi):
    pad = B - (hi - lo); out = {}
    for k, v in spec.items():
        t = v[lo:hi]
        if pad:
            t = torch.cat([t, t[-1:].expand(pad, *t.shape[1:])])
        out[k] = t.to(dev)
    return out


@torch.no_grad()
def roll(env, fn, spec, N):
    out = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = ScriptedLineDistribution(_sub(spec, lo, hi))
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
def values(env, ag, spec, N):
    V = np.zeros(N, np.float32)
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = ScriptedLineDistribution(_sub(spec, lo, hi))
        V[lo:hi] = ag.get_value(env.reset()).float().cpu().numpy().reshape(-1)[:hi - lo]
    return V


def stat(v, ref, tag):
    rt = v / np.maximum(ref, 1e-9)
    print(f'  {tag:36s} stroke {v.mean():.3f}  ratio {rt.mean()*100:.1f} / {np.percentile(rt, 10)*100:.1f}', flush=True)
    return [round(float(v.mean()), 4), round(float(rt.mean() * 100), 2), round(float(np.percentile(rt, 10) * 100), 2)]


tz = np.load(A / 'tasks_pool_fr3.npz')
T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
# ================================================================ part 1
if '1' in PARTS:
    print('=== part 1: straight 10k ===', flush=True)
    spec = {'q0': T(tz['q0_seed']), 'line_dir': T(tz['cs_line_dir']), 'n_target': T(tz['cs_n_target'])}
    N = 10000
    b = np.load(A / 'bound_pool_fr3.npz'); w = np.load(A / 'witness_pool_fr3.npz'); base = np.load(FU / 'pool_fr3_straight.npz')
    ref = np.maximum(b['L_hi'], w['prog'])
    for k in base.files:
        if k.endswith('_progress'):
            ref = np.maximum(ref, base[k])
    p_rm = np.load(FU / 'e8kXXL_10k.npz')['prog']
    outs = {}
    for kl in (0.0, 5.0):
        env, fn, ag = build(CFG_C, CKPT_C, k_lateral=kl)
        outs[kl] = roll(env, fn, spec, N); del env, ag; torch.cuda.empty_cache()
    ref = np.maximum.reduce([ref, p_rm] + list(outs.values()))
    res['p1'] = {'flagship_kl0': stat(p_rm, ref, 'flagship (k_lateral 0, paper)')}
    for kl, v in outs.items():
        res['p1'][f'{TAG}_kl{int(kl)}'] = stat(v, ref, f'{TAG} (k_lateral {kl:.0f})')
    d = outs[0.0] - p_rm
    print(f'  paired vs flagship: better {(d > 0.02).mean()*100:.1f}% worse {(d < -0.02).mean()*100:.1f}%  [{time.time()-t00:.0f}s]', flush=True)
    np.savez(FU / f'{TAG}_straight10k.npz', kl0=outs[0.0], kl5=outs[5.0], ref=ref)

# ================================================================ part 2
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)


def fam_spec(key):
    sp = tasks[key]
    spec = {'q0': sp['q0'].clone().float(), 'p0': sp['p0'].clone().float(),
            'line_dir': sp['line_dir'].clone().float(), 'n_target': sp['n_target'].clone().float()}
    for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
        if k in sp:
            spec[k] = sp[k].clone().float()
    return spec


if '2' in PARTS:
    print('=== part 2: curved families 2.5k ===', flush=True)
    res['p2'] = {}
    env, fn, ag = build(CFG_C, CKPT_C, k_lateral=5.0)
    for fam, key in (('serpentine', 'test_serpentine'), ('rot', 'test_nonplanar')):
        spec = fam_spec(key); N = spec['q0'].shape[0]
        out = roll(env, fn, spec, N)
        fam2 = key.replace('test_', '')
        b = np.load(A / f'bound_sel_{fam2}.npz'); w = np.load(A / f'witness_sel_{fam2}.npz')
        ctrl = np.load(FU / f'ctrl_fr3_{fam2}.npz')
        p_rm = np.load(FU / f'dirfrac_fr3e8k_{fam2}_curved.npz')['prog']
        ref = np.maximum(b['L_hi'], w['prog'])[:N]
        for k in ctrl.files:
            ref = np.maximum(ref, ctrl[k][:N])
        ref = np.maximum(np.maximum(ref, p_rm), out)
        res['p2'][fam] = {'flagship': stat(p_rm, ref, f'{fam}: flagship'), TAG: stat(out, ref, f'{fam}: {TAG}')}
        d = out - p_rm
        print(f'  paired: better {(d > 0.02).mean()*100:.1f}% worse {(d < -0.02).mean()*100:.1f}%  [{time.time()-t00:.0f}s]', flush=True)
        np.savez(FU / f'{TAG}_{fam2}.npz', prog=out, ref=ref)
    del env, ag; torch.cuda.empty_cache()

# ================================================================ part 3
if '3' in PARTS:
    print('=== part 3: start selection (first / random / critic / oracle) ===', flush=True)
    cands = torch.load(MAIN / 'runs/selector_ood/v2_k32/cands.pt', weights_only=False)
    tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
    K = int(os.environ.get('SEL_K', 32))          # smoke: fewer candidates
    SEL_N = int(os.environ.get('SEL_N', 0))        # smoke: subset of every set
    res['p3'] = {}
    env, fn, ag = build(CFG_C, CKPT_C, k_lateral=5.0)
    for fam, key in (('straight', 'benchmark'), ('serpentine', 'test_serpentine'), ('rot', 'test_nonplanar')):
        nf = cands[key]['n_found'].numpy(); C = cands[key]['cands'].numpy()
        if key == 'benchmark':
            rows_sel = np.sort(np.random.default_rng(3).choice(C.shape[0], SEL_N or 2000, replace=False))
            spec_np = {'line_dir': tb['cs_line_dir'][rows_sel].astype(np.float32), 'n_target': tb['cs_n_target'][rows_sel].astype(np.float32)}
            b = np.load(MAIN / 'runs/paper_fill/bound_10000_final.npz'); w = np.load(MAIN / 'runs/paper_fill/witness_10k_v4.npz')
            saved = np.load(FU / 'fr3e8k_sel_straight.npz')
        else:
            rows_sel = np.arange(C.shape[0]) if not SEL_N else np.arange(SEL_N)
            sp = tasks[key]
            spec_np = {'p0': sp['p0'].numpy(), 'line_dir': sp['line_dir'].numpy(), 'n_target': sp['n_target'].numpy()}
            for kk in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
                if kk in sp:
                    spec_np[kk] = sp[kk].numpy()
            fam2 = key.replace('test_', '')
            b = np.load(A / f'bound_sel_{fam2}.npz'); w = np.load(A / f'witness_sel_{fam2}.npz')
            saved = np.load(FU / f'fr3e8k_sel_{fam}.npz')     # saved as 'rot', bounds as 'nonplanar'
        C, nf = C[rows_sel, :K], np.minimum(nf[rows_sel], K)
        N = C.shape[0]
        ref = np.maximum(b['L_hi'], w['prog'])[rows_sel]
        for kk in spec_np:
            spec_np[kk] = spec_np[kk] / np.linalg.norm(spec_np[kk], axis=1, keepdims=True) if kk in ('line_dir', 'n_target') else spec_np[kk]
        L = np.zeros((N, K), np.float32); V = np.full((N, K), np.nan, np.float32)
        t0 = time.time()
        for k in range(K):
            valid = nf > k
            spec = {'q0': T(np.nan_to_num(C[:, k], nan=0.0))}
            spec.update({kk: T(v) for kk, v in spec_np.items()})
            V[:, k] = values(env, ag, spec, N)
            L[:, k] = roll(env, fn, spec, N)
            L[~valid, k] = 0.0; V[~valid, k] = np.nan
            if (k + 1) % 8 == 0:
                print(f'    {fam}: cand {k+1}/{K} ({(time.time()-t0)/60:.1f} min)', flush=True)
        Mv = np.arange(K)[None, :] < nf[:, None]

        def rows(L, V):
            orc = np.where(Mv, L, -1e9).max(1); ok = orc > 1e-6
            first = L[np.arange(N), Mv.argmax(1)]
            rnd = np.where(Mv, L, 0).sum(1) / np.maximum(Mv.sum(1), 1)
            Vm = np.where(Mv & np.isfinite(V), V, -1e9); crit = L[np.arange(N), Vm.argmax(1)]
            out = {}
            for nme, v in (('first', first), ('random', rnd), ('critic', crit), ('oracle', orc)):
                rr = np.maximum(ref, v); rt = v / np.maximum(rr, 1e-9)
                out[nme] = [round(float(v[ok].mean()), 3), round(float(rt[ok].mean() * 100), 1),
                            round(float(np.percentile(rt[ok], 10) * 100), 1), round(float((v >= orc - 0.01)[ok].mean() * 100), 1)]
            return out
        r_new = rows(L, V)
        r_old = rows(saved['L'][rows_sel][:, :K], saved['V'][rows_sel][:, :K])
        res['p3'][fam] = {'flagship': r_old, TAG: r_new}
        for nme in ('first', 'random', 'critic', 'oracle'):
            print(f'  {fam:10s} {nme:7s} flagship {r_old[nme]}   {TAG} {r_new[nme]}   [stroke, ratio, p10, near-oracle%]', flush=True)
        np.savez(FU / f'{TAG}_sel_{fam}.npz', L=L, V=V, n_found=nf, rows=rows_sel)
    del env, ag; torch.cuda.empty_cache()

# ================================================================ part 4
if '4' in PARTS:
    print('=== part 4: figure families as table paths (2000 tasks each) ===', flush=True)
    rng = np.random.default_rng(3)
    s4 = np.sort(rng.choice(len(tz['q0_seed']), 2000, replace=False))
    N = 2000
    q0 = T(tz['q0_seed'][s4]); d0 = T(tz['cs_line_dir'][s4]); n0 = T(tz['cs_n_target'][s4])
    kin_env = build(CFG_RM, CKPT_RM, classical=True)[0]
    p0 = kin_env.kin.tcp_fk_jac(q0.to(dev))[0].cpu()
    del kin_env
    Mm1 = TAB_M - 1
    z = torch.zeros(N, Mm1)
    sgn = torch.where(torch.rand(N, generator=torch.Generator().manual_seed(1)) < 0.5, -1.0, 1.0)
    g = torch.Generator().manual_seed(2)
    fams = {}
    r = 0.15 + torch.rand(N, generator=g) * 0.35                          # circle radius
    fams['circle'] = (((sgn / r)[:, None]).expand(N, Mm1).clone(), z, z, z, 2 * math.pi * r)
    def polygon(nsides, a_lo, a_hi):
        a = a_lo + torch.rand(N, generator=g) * (a_hi - a_lo)
        corner = torch.zeros(N, Mm1); ang = sgn * (2 * math.pi / nsides)
        for j in range(1, 12):
            idx = (j * a / TAB_DS).round().long()
            ok = idx < Mm1
            corner[torch.arange(N)[ok], idx[ok]] = ang[ok]
        return (z, z, z, corner, nsides * a)
    fams['square'] = polygon(4, 0.3, 0.8)
    fams['triangle'] = polygon(3, 0.3, 0.9)
    R = 0.25 + torch.rand(N, generator=g) * 0.25; c = 0.05 + torch.rand(N, generator=g) * 0.15
    kap = R / (R * R + c * c); tor = c / (R * R + c * c)
    fams['helix'] = (z, (sgn * kap)[:, None].expand(N, Mm1).clone(), (sgn * tor)[:, None].expand(N, Mm1).clone(), z,
                     torch.full((N,), (TAB_M - 1) * TAB_DS))
    res['p4'] = {}
    envs = {'classical': build(CFG_RM, CKPT_RM, classical=True, table_paths=True, k_lateral=5.0),
            'flagship': build(CFG_RM, CKPT_RM, table_paths=True, k_lateral=5.0),
            TAG: build(CFG_C, CKPT_C, table_paths=True, k_lateral=5.0)}
    for fam, (kg, kn, tau, corner, length) in fams.items():
        pts, tan, nrm = integrate_frames(p0, d0, n0, kg, kn, tau, corner)
        spec = {'q0': q0, 'p0': pts[:, 0].clone(), 'line_dir': d0, 'n_target': n0,
                'path_pts': pts, 'path_tan': tan, 'path_nrm': nrm}
        length = torch.minimum(length, torch.full_like(length, (TAB_M - 1) * TAB_DS - 0.02)).numpy()
        outs = {k: roll(e, fn, spec, N) for k, (e, fn, _) in envs.items()}
        res['p4'][fam] = {}
        for k, v in outs.items():
            done = (v >= length - 0.01).mean() * 100
            res['p4'][fam][k] = [round(float(v.mean()), 3), round(float(done), 1)]
        c_ = outs['classical']; f_ = outs['flagship']; m_ = outs[TAG]
        print(f'  {fam:9s} length {length.mean():.2f} m | classical {c_.mean():.3f} ({(c_ >= length-0.01).mean()*100:.1f}% complete) | '
              f'flagship {f_.mean():.3f} ({(f_ >= length-0.01).mean()*100:.1f}%) | {TAG} {m_.mean():.3f} ({(m_ >= length-0.01).mean()*100:.1f}%) | '
              f'{TAG} vs flagship better {((m_ - f_) > 0.02).mean()*100:.1f}% worse {((m_ - f_) < -0.02).mean()*100:.1f}%', flush=True)
        np.savez(FU / f'{TAG}_fig_{fam}.npz', length=length, s4=s4, **outs)
    for e, _, ag in envs.values():
        del e, ag
    torch.cuda.empty_cache()

json.dump(res, open(FU / f'{TAG}_eval_rows.json', 'w'), indent=1)
print('saved', FU / f'{TAG}_eval_rows.json', f'[{time.time()-t00:.0f}s]', flush=True)
