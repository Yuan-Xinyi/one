"""Four-family table unification, post gate-fix (stage-dispatched, resumable).

Stages (argv[1]):
  relabel      v3_k32: relabel train_arc / test_arc / test_serpentine with the
               FIXED vlook gate; reuse straight / nonplanar / benchmark labels
  retrain      rankers on (train_straight + relabelled train_arc)
  selreport    per-task chosen strokes for every ranking rule x family
               (+ benchmark) -> sel_report.npz + report.json
  seltasks     refresh q0_seed (new oracle candidate) in tasks_sel_arc /
               tasks_sel_serpentine npz (witness starts)
  fr3_curved   controller rollouts on FR3 test_{arc,serpentine,nonplanar}:
               zero/classical/myopic/vertex/hybrid/vlook/cont/critxa
  xacb_tasks   build xarm7+cobotta curved task sets (2500/family) with
               path_pts/path_axes -> tasks_selx_{fam}_{robot}.npz
  xacb_roll    same rollouts on xarm7/cobotta curved sets (cont only if ckpt
               exists; xarm7 also critfr3)
  cont_curved  continuous-PPO rollouts on curved sets for robots whose cont
               ckpt appeared later (idempotent top-up of the rollout npz)
All rollouts: ladder protocol (dt/2, max_steps x2, model dt 50 ms),
k_lateral=5.0, stroke = env.arc_progress.
"""
import sys, os, time, json, dataclasses
from pathlib import Path
REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
MAIN = Path('/home/lqin/one/Yuan/IJRR')
sys.path.insert(0, str(REPO))
import numpy as np, torch, yaml
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'selector_ood', MAIN / 'stage1_seed/selector_ood.py')
so = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(so)
import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, damped_pinv
from Yuan.IJRR.env.line_distribution import (LineDistribution,
                                             ScriptedLineDistribution)
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.env.path_geometry import arc_point, serpentine_point
from Yuan.IJRR.eval.eval_curve import _agent

STAGE = sys.argv[1]
SMOKE = bool(int(os.environ.get('SMOKE', '0')))
dev = torch.device('cuda')
V2 = MAIN / 'runs/selector_ood/v2_k32'
V3 = MAIN / 'runs/selector_ood/v3_k32_fixedgate'
V3.mkdir(parents=True, exist_ok=True)
A = MAIN / 'runs/paper_fill/ratio_assets'
FU = MAIN / 'runs/paper_fill/fam_unify'
FU.mkdir(parents=True, exist_ok=True)
K = 32
BATCH = 512 if SMOKE else int(os.environ.get('FAM_BATCH', '4096'))
CHUNK = int(os.environ.get('FAM_CHUNK', '32768'))
STEP, MAXL = 0.01, 1.8
NG = int(round(MAXL / STEP)) + 1
CONT_CKPT = {'fr3': 'Yuan/IJRR/runs/rl_cont_sqent_30M',
             'xarm7': 'Yuan/IJRR/runs/rl_cont_sqent_xarm7_30M',
             'cobotta': 'Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M'}


def say(m):
    print(time.strftime('[%H:%M:%S] ') + m, flush=True)


def ladder_env(robot, batch, k_lateral=5.0):
    y = yaml.safe_load(open(REPO / hl.ROBOTS[robot][0]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] = kw['dt'] / 2
    kw['max_steps'] = int(y['env']['max_steps'] * 2)
    if k_lateral is not None:
        kw['k_lateral'] = k_lateral
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)
    model = hl.StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    model.terms = [0, 1]
    return env, model


def rollout(env, afn, sub_spec):
    env.line_dist = ScriptedLineDistribution(sub_spec)
    env.reset()
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
    for _ in range(env.cfg.max_steps // 2):
        a = afn(env, done)
        for _ in range(2):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return env.arc_progress.float().cpu().numpy().copy()


def batched_rollout(env, afn, spec, N):
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
        out[lo:hi] = rollout(env, afn, sub)[:hi - lo]
    return out


def make_cont_arm(env, robot):
    from Yuan.IJRR.stage2_traj.ppo import Agent
    ag = Agent(env.obs_dim, env.act_dim).to(dev)
    ag.load_state_dict(torch.load(REPO / CONT_CKPT[robot] / 'agent.pt',
                                  map_location=dev))
    ag.eval()

    @torch.no_grad()
    def fn(e, done):
        return ag.actor_mean(e.current_obs())
    return fn


def build_arms(env, model, robot, foreign=None, cont=True):
    classical = ClassicalNullspaceController(env.kin)
    ag = _agent(REPO / hl.ROBOTS[robot][1], env.obs_dim, dev,
                act_dim=env.act_dim)
    arms = {
        'zero': lambda e, dn: torch.zeros((e.n_envs, e.act_dim),
                                          device=e.device),
        'classical': (lambda e, dn, f=cn_action_fn(classical): f(e)),
        'myopic': hl.make_myopic(model),
        'vertex': (lambda e, dn, g_=ag: g_.actor_mean(e.current_obs())),
        'hybrid': hl.make_hybrid(env, ag, classical, 0.98, 0.94),
        'vlook': hl.make_vlook(model, env, ag),
    }
    if cont and (REPO / CONT_CKPT[robot] / 'agent.pt').exists():
        arms['cont'] = make_cont_arm(env, robot)
    if foreign is not None:
        fag = _agent(REPO / hl.ROBOTS[foreign][1], env.obs_dim, dev,
                     act_dim=env.act_dim)
        arms[f'crit_{foreign}'] = hl.make_vlook(model, env, fag)
    return arms


def fam_spec(tasks_dict, fam):
    sp = tasks_dict[f'test_{fam}']
    spec = {'q0': sp['q0'].clone(), 'p0': sp['p0'].clone(),
            'line_dir': sp['line_dir'].clone(),
            'n_target': sp['n_target'].clone()}
    for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
        if k in sp:
            spec[k] = sp[k].clone()
    return spec


# ======================== selector-side stages ==============================

def label_family_value(env, model, ag, spec, cands):
    vfn = hl.make_vlook(model, env, ag, chunk=CHUNK)
    N, Kc = cands.shape[:2]
    dt = env.kin.dtype
    rep = lambda t: t.repeat_interleave(Kc, 0)
    flat = {'q0': torch.tensor(cands.reshape(N * Kc, -1), dtype=dt),
            'p0': rep(spec['p0'])}
    for key in ('line_dir', 'n_target', 'kappa', 'amp', 'wavelen',
                'n_rot_axis', 'n_rot_rate'):
        if key in spec:
            flat[key] = rep(spec[key])
    L = np.zeros(N * Kc, np.float32)
    for lo in range(0, N * Kc, env.n_envs):
        hi = min(lo + env.n_envs, N * Kc)
        pad = env.n_envs - (hi - lo)
        sub = {k: v[lo:hi] for k, v in flat.items()}
        if pad:
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                   for k, v in sub.items()}
        for k2 in ('q0', 'line_dir', 'n_target'):
            sub[k2] = sub[k2].to(device=dev, dtype=dt)
        L[lo:hi] = rollout(env, vfn, sub)[:hi - lo]
        if (lo // env.n_envs) % 20 == 0:
            say(f'  [vlabel] {hi}/{N * Kc}')
    return L.reshape(N, Kc)


if STAGE == 'relabel':
    env, model = ladder_env('fr3', BATCH)
    ag = _agent(REPO / hl.ROBOTS['fr3'][1], env.obs_dim, dev,
                act_dim=env.act_dim)
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    cands = torch.load(V2 / 'cands.pt', weights_only=False)
    if (V3 / 'labels.pt').exists():
        labels = torch.load(V3 / 'labels.pt', weights_only=False)
    else:
        labels = dict(torch.load(V2 / 'labels.pt', weights_only=False))
        for k in ('train_arc', 'test_arc', 'test_serpentine'):
            labels.pop(k, None)
    for key in ('test_arc', 'test_serpentine', 'train_arc'):
        if key in labels:
            continue
        sp = tasks[key]
        spec = {'p0': sp['p0']}
        for k in ('line_dir', 'n_target', 'kappa', 'amp', 'wavelen',
                  'n_rot_axis', 'n_rot_rate'):
            if k in sp:
                spec[k] = sp[k]
        cd = cands[key]['cands'].numpy()
        if SMOKE:
            spec = {k: v[:64] for k, v in spec.items()}
            cd = cd[:64]
        say(f'[relabel] {key} ({cd.shape[0]}x{K}) with FIXED gate')
        L = label_family_value(env, model, ag, spec, cd)
        say(f'[relabel] {key}: mean best {L.max(1).mean():.3f} m '
            f'(v2 was {torch.load(V2 / "labels.pt", weights_only=False)[key].max(1).values.float().mean():.3f})')
        if not SMOKE:
            labels[key] = torch.tensor(L)
            torch.save(labels, V3 / 'labels.pt')
    if not SMOKE:
        (V3 / 'labels.done').write_text('ok')
    say('[relabel] done')

elif STAGE == 'retrain':
    env, model = ladder_env('fr3', BATCH)
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    cands = torch.load(V2 / 'cands.pt', weights_only=False)
    labels = torch.load(V3 / 'labels.pt', weights_only=False)
    Xc, Xr, Y, M = [], [], [], []
    for key in ('train_straight', 'train_arc'):
        spec = tasks[key]
        c = cands[key]['cands']
        Xc.append(so.cand_features(env, c.numpy(), spec))
        rt = so.road_table(spec)
        Xr.append(torch.tensor(rt.reshape(rt.shape[0], -1)))
        Y.append(labels[key].float())
        nf = cands[key]['n_found']
        M.append(torch.arange(K)[None, :] < nf[:, None])
    tr = (torch.cat(Xc), torch.cat(Xr), torch.cat(Y), torch.cat(M))
    nets = {}
    for cond in (True, False):
        say(f'[retrain] conditioned={cond}')
        nets[cond] = so.train_ranker(*tr, conditioned=cond, dev=dev)
    torch.save({'cond': nets[True].state_dict(),
                'nocond': nets[False].state_dict()}, V3 / 'rankers.pt')
    say('[retrain] done')

elif STAGE == 'selreport':
    env, model = ladder_env('fr3', BATCH)
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    cands = torch.load(V2 / 'cands.pt', weights_only=False)
    labels = torch.load(V3 / 'labels.pt', weights_only=False)
    rk = torch.load(V3 / 'rankers.pt', weights_only=False)
    tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
    bench = {'p0': torch.tensor(tb['cs_p0'], dtype=torch.float32),
             'line_dir': torch.tensor(tb['cs_line_dir'],
                                      dtype=torch.float32),
             'n_target': torch.tensor(tb['cs_n_target'],
                                      dtype=torch.float32),
             'q0': torch.tensor(tb['q0_seed'], dtype=torch.float32)}
    bench['line_dir'] /= bench['line_dir'].norm(dim=-1, keepdim=True)
    bench['n_target'] /= bench['n_target'].norm(dim=-1, keepdim=True)
    ALL = {**tasks, 'benchmark': bench}

    def selector_scores(key, which):
        spec = ALL[key]
        Xc = so.cand_features(env, cands[key]['cands'].numpy(), spec)
        rt = so.road_table(spec)
        Xr = torch.tensor(rt.reshape(rt.shape[0], -1))
        net = so.Ranker(Xc.shape[-1], Xr.shape[-1],
                        conditioned=(which == 'cond')).to(dev)
        net.load_state_dict(rk[which]); net.eval()
        with torch.no_grad():
            return net(Xc.to(dev).float(), Xr.to(dev).float()).cpu()

    def heuristic_scores(key):
        spec = ALL[key]
        C = cands[key]['cands'].to(dev)
        N = C.shape[0]
        nj = C.shape[-1]
        fq = C.reshape(-1, nj)
        fp = spec['p0'].to(dev).repeat_interleave(K, 0)
        fd = spec['line_dir'].to(dev).repeat_interleave(K, 0)
        fn = spec['n_target'].to(dev).repeat_interleave(K, 0)
        CH = 4096
        mjl, mc, ph, wd = [], [], [], []
        with torch.no_grad():
            for i in range(0, fq.shape[0], CH):
                m = model.margins(fq[i:i+CH], fp[i:i+CH], fd[i:i+CH],
                                  fn[i:i+CH])
                mjl.append(m[:, 0]); mc.append(m[:, 1])
                ph.append(-0.1 * torch.logsumexp(-m / 0.1, dim=-1))
                _, _, J, _ = env.kin.tcp_fk_jac(fq[i:i+CH])
                Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                                    env.cfg.sigma_thr)
                wd.append(1.0 / ((Jp @ fd[i:i+CH].unsqueeze(-1)).squeeze(-1)
                                 .norm(dim=-1) + 1e-9))
        r = lambda t: torch.cat(t).reshape(N, K).cpu()
        return {'manip': r(wd), 'orient': r(mc), 'jl': r(mjl), 'phim': r(ph)}

    out, rep_json = {}, {}
    for key in ('test_straight', 'test_arc', 'test_serpentine',
                'test_nonplanar', 'benchmark'):
        Y = labels[key].float()
        nf = cands[key]['n_found']
        M = (torch.arange(K)[None, :] < nf[:, None])
        Ym = torch.where(M, Y, torch.full_like(Y, -1e9))
        orc = Ym.max(1).values
        per = {'oracle': orc.numpy(),
               'first': Y.gather(1, M.float().argmax(1)[:, None])
                         .squeeze(1).numpy(),
               'random': ((Y * M).sum(1) / M.sum(1).clamp(min=1)).numpy()}
        scr = heuristic_scores(key)
        scr['selector'] = selector_scores(key, 'cond')
        for name, S in scr.items():
            pick = torch.where(M, S, torch.full_like(S, -1e9)).argmax(1)
            per[name] = Y.gather(1, pick[:, None]).squeeze(1).numpy()
        ok = (orc > 1e-6).numpy()
        per['valid'] = ok
        out.update({f'{key}__{n}': v for n, v in per.items()})
        rep_json[key] = {n: {'stroke': float(v[ok].mean()),
                             'frac': float((v[ok] / per['oracle'][ok]).mean())}
                         for n, v in per.items() if n != 'valid'}
        say(f'[selreport] {key}: ' + json.dumps(rep_json[key]))
    np.savez_compressed(FU / 'sel_report.npz', **out)
    json.dump(rep_json, open(FU / 'sel_report.json', 'w'), indent=1)
    say('[selreport] done')

elif STAGE == 'seltasks':
    cands = torch.load(V2 / 'cands.pt', weights_only=False)
    labels = torch.load(V3 / 'labels.pt', weights_only=False)
    for fam in ('arc', 'serpentine'):
        key = f'test_{fam}'
        f = A / f'tasks_sel_{fam}.npz'
        d = dict(np.load(f))
        L = labels[key]; nf = cands[key]['n_found']
        V = torch.arange(L.shape[1])[None, :] < nf[:, None]
        best = torch.where(V, L, torch.full_like(L, -1e9)).argmax(1)
        N = L.shape[0]
        d['q0_seed'] = cands[key]['cands'][torch.arange(N), best].numpy()
        np.savez_compressed(f, **d)
        say(f'[seltasks] refreshed q0_seed in {f.name}')

# ======================== controller-side stages ============================

elif STAGE == 'fr3_curved':
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    env, model = ladder_env('fr3', 2500 if not SMOKE else 128)
    arms = build_arms(env, model, 'fr3', foreign='xarm7')
    for fam in ('arc', 'serpentine', 'nonplanar'):
        f = FU / f'ctrl_fr3_{fam}.npz'
        if f.exists():
            continue
        spec = fam_spec(tasks, fam)
        N = spec['q0'].shape[0] if not SMOKE else 128
        spec = {k: v[:N] for k, v in spec.items()}
        res = {}
        for name, afn in arms.items():
            t0 = time.time()
            res[name] = batched_rollout(env, afn, spec, N)
            say(f'[fr3_curved] {fam}/{name}: mean {res[name].mean():.4f} '
                f'({time.time()-t0:.0f}s)')
        if not SMOKE:
            np.savez_compressed(f, **{f'{k}_progress': v
                                      for k, v in res.items()})
            say(f'[fr3_curved] wrote {f.name}')

elif STAGE == 'xacb_tasks':
    rng = np.random.default_rng(23)
    for robot in ('xarm7', 'cobotta'):
        env, _ = ladder_env(robot, 8)
        pool = LineDistribution.load_or_build(
            kin=env.kin, collision=env.collision, n_pool=20000,
            n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
            feasibility_threshold_m=0.1, verbose=False)
        valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
        idx = valid[:2500]
        for fam in ('arc', 'serpentine', 'nonplanar'):
            f = A / f'tasks_selx_{fam}_{robot}.npz'
            if f.exists():
                continue
            spec = so.make_specs(pool, idx, fam, rng)
            q0 = spec['q0'].to(dev, env.kin.dtype)
            p0 = env.kin.tcp_fk_jac(q0)[0].double().cpu()
            d = spec['line_dir'].double()
            n = spec['n_target'].double()
            d = d / d.norm(dim=-1, keepdim=True)
            n = n / n.norm(dim=-1, keepdim=True)
            N = idx.numel()
            s = torch.arange(NG, dtype=torch.float64) * STEP
            if fam == 'arc':
                kap = spec['kappa'].double()
                pts = torch.stack([arc_point(p0, d, n, kap,
                                   torch.full((N,), float(si),
                                              dtype=torch.float64))
                                   for si in s], dim=1)
                axes = n[:, None, :].expand(-1, NG, -1).clone()
            elif fam == 'serpentine':
                amp = spec['amp'].double(); wl = spec['wavelen'].double()
                xs = torch.arange(0, 2.6, 0.001, dtype=torch.float64)
                kk = 2.0 * torch.pi / wl.clamp_min(1e-3)
                dy = amp[:, None] * kk[:, None] * torch.cos(
                    kk[:, None] * xs[None, :])
                ds = torch.sqrt(1.0 + dy ** 2) * 0.001
                cum = torch.cumsum(ds, dim=1) - ds
                xi = torch.empty(N, NG, dtype=torch.float64)
                for i in range(N):
                    xi[i] = torch.from_numpy(
                        np.interp(s.numpy(), cum[i].numpy(), xs.numpy()))
                pts = serpentine_point(
                    p0[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                    d[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                    n[:, None, :].expand(-1, NG, -1).reshape(-1, 3),
                    amp[:, None].expand(-1, NG).reshape(-1),
                    wl[:, None].expand(-1, NG).reshape(-1),
                    xi.reshape(-1)).reshape(N, NG, 3)
                axes = n[:, None, :].expand(-1, NG, -1).clone()
            else:
                pts = p0[:, None, :] + s[None, :, None] * d[:, None, :]
                ax = spec['n_rot_axis'].double()
                ax = ax / ax.norm(dim=-1, keepdim=True)
                th = spec['n_rot_rate'].double()[:, None] * s[None, :]
                kxn = torch.linalg.cross(ax[:, None, :].expand(-1, NG, -1),
                                         n[:, None, :].expand(-1, NG, -1),
                                         dim=-1)
                kdn = (ax * n).sum(-1)[:, None, None]
                axes = (n[:, None, :] * torch.cos(th)[..., None]
                        + kxn * torch.sin(th)[..., None]
                        + ax[:, None, :] * kdn
                        * (1 - torch.cos(th))[..., None])
                axes = axes / axes.norm(dim=-1, keepdim=True)
            np.savez_compressed(
                f, cs_p0=p0.float().numpy(), cs_line_dir=d.float().numpy(),
                cs_n_target=n.float().numpy(),
                q0_seed=spec['q0'].float().numpy(),
                path_pts=pts.float().numpy(),
                path_axes=axes.float().numpy(),
                **{kk2: spec[kk2].float().numpy() for kk2 in
                   ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate')
                   if kk2 in spec})
            say(f'[xacb_tasks] wrote {f.name}')

elif STAGE in ('xacb_roll', 'cont_curved'):
    for robot in ('xarm7', 'cobotta'):
        env, model = ladder_env(robot, 2500 if not SMOKE else 128)
        foreign = 'fr3' if robot == 'xarm7' else None
        arms = build_arms(env, model, robot, foreign=foreign)
        if STAGE == 'cont_curved':
            arms = ({'cont': arms['cont']}
                    if 'cont' in arms else {})
        for fam in ('arc', 'serpentine', 'nonplanar'):
            f = FU / f'ctrl_{robot}_{fam}.npz'
            done_names = []
            old = {}
            if f.exists():
                old = dict(np.load(f))
                done_names = [k[:-9] for k in old if k.endswith('_progress')]
            todo = {n: a for n, a in arms.items() if n not in done_names}
            if not todo:
                continue
            tz = np.load(A / f'tasks_selx_{fam}_{robot}.npz')
            N = tz['cs_p0'].shape[0] if not SMOKE else 128
            spec = {'q0': torch.tensor(tz['q0_seed'][:N]),
                    'p0': torch.tensor(tz['cs_p0'][:N]),
                    'line_dir': torch.tensor(tz['cs_line_dir'][:N]),
                    'n_target': torch.tensor(tz['cs_n_target'][:N])}
            for k in ('kappa', 'amp', 'wavelen', 'n_rot_axis', 'n_rot_rate'):
                if k in tz.files:
                    spec[k] = torch.tensor(tz[k][:N])
            for name, afn in todo.items():
                t0 = time.time()
                r = batched_rollout(env, afn, spec, N)
                old[f'{name}_progress'] = r
                say(f'[{STAGE}] {robot}/{fam}/{name}: mean {r.mean():.4f} '
                    f'({time.time()-t0:.0f}s)')
                if not SMOKE:
                    np.savez_compressed(f, **old)
        del env, model, arms
        torch.cuda.empty_cache()

elif STAGE == 'pool_straight':
    # Aligned 6-arm (+cont/+foreign-critic) straight evals on the exact
    # tasks_pool_* slices the bounds/witnesses use. Supersedes the stale
    # ratio_*_10k npz (task-order mismatch) and the crosscritic/eval_cont2
    # CLI runs. Idempotent per arm (top-up).
    B2 = int(os.environ.get('POOL_BATCH', '2500'))
    for robot in ('fr3', 'xarm7', 'cobotta'):
        env, model = ladder_env(robot, B2, k_lateral=None)
        foreign = {'fr3': 'xarm7', 'xarm7': 'fr3'}.get(robot)
        arms = build_arms(env, model, robot, foreign=foreign)
        f = FU / f'pool_{robot}_straight.npz'
        old = dict(np.load(f)) if f.exists() else {}
        done_names = [k[:-9] for k in old if k.endswith('_progress')]
        todo = {n: a2 for n, a2 in arms.items() if n not in done_names}
        if todo:
            tz = np.load(A / f'tasks_pool_{robot}.npz')
            N = tz['cs_p0'].shape[0] if not SMOKE else 256
            spec = {'q0': torch.tensor(tz['q0_seed'][:N]),
                    'line_dir': torch.tensor(tz['cs_line_dir'][:N]),
                    'n_target': torch.tensor(tz['cs_n_target'][:N])}
            for name, afn in todo.items():
                t0 = time.time()
                r = batched_rollout(env, afn, spec, N)
                old[f'{name}_progress'] = r
                say(f'[pool_straight] {robot}/{name}: mean {r.mean():.4f} '
                    f'({time.time()-t0:.0f}s)')
                if not SMOKE:
                    np.savez_compressed(f, **old)
        del env, model, arms
        torch.cuda.empty_cache()

elif STAGE == 'fr3_cont_curved':
    # top-up: cont + any missing arm on the FR3 curved rollout npz
    tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt',
                       weights_only=False)
    env, model = ladder_env('fr3', 2500 if not SMOKE else 128)
    arms = build_arms(env, model, 'fr3', foreign='xarm7')
    for fam in ('arc', 'serpentine', 'nonplanar'):
        f = FU / f'ctrl_fr3_{fam}.npz'
        old = dict(np.load(f)) if f.exists() else {}
        done_names = [k[:-9] for k in old if k.endswith('_progress')]
        todo = {n: a for n, a in arms.items() if n not in done_names}
        if not todo:
            continue
        spec = fam_spec(tasks, fam)
        N = spec['q0'].shape[0] if not SMOKE else 128
        spec = {k: v[:N] for k, v in spec.items()}
        for name, afn in todo.items():
            r = batched_rollout(env, afn, spec, N)
            old[f'{name}_progress'] = r
            say(f'[fr3_cont_curved] {fam}/{name}: mean {r.mean():.4f}')
            if not SMOKE:
                np.savez_compressed(f, **old)

else:
    raise SystemExit(f'unknown stage {STAGE}')
say(f'STAGE {STAGE} COMPLETE')
