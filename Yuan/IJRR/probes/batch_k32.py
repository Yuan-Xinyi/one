"""Overnight batch: K=32 candidate pools + VALUE-LAW labels everywhere
(fixes the fact that all previous selector labels/fullsys strokes were
margin-law rollouts), retrained rankers, and all table slices.

Stages (resumable; each saves to OUT and skips if its file exists):
  smoke      labeler must reproduce the ladder vlook reference (~0.5678 m)
  cands      K=32/N_DIRS=40 pools for 20k train + 4x2500 test + 10k benchmark
  labels     value-law rollouts for every candidate of every pool
  train      cond + uncond rankers on the new train pools
  report     seed-table rows, ood-table rows, framework rows -> npz + txt
"""
import sys, time, dataclasses, json
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
from Yuan.IJRR.eval.eval_curve import _agent

# ---- pool upgrade ----------------------------------------------------------
so.K_CAND = 32
so.N_DIRS = 40
K = so.K_CAND
BATCH = 2048
OUT = MAIN / 'runs/selector_ood/v2_k32'
OUT.mkdir(parents=True, exist_ok=True)
log = open(OUT / 'batch.log', 'a')
def say(m):
    print(m, flush=True)
    log.write(time.strftime('[%H:%M:%S] ') + m + '\n'); log.flush()

dev = torch.device('cuda')
env, model = so.build_base_env(BATCH, dev)
ag = _agent(REPO / 'Yuan/IJRR/runs/rl_vertex_line_30M', env.obs_dim, dev,
            act_dim=env.act_dim)
vfn = hl.make_vlook(model, env, ag)

def label_family_value(spec, cands):
    """label_family with the VALUE law instead of the margin law."""
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
    blocks = env.max_steps // so.SUB
    for lo in range(0, N * Kc, BATCH):
        hi = min(lo + BATCH, N * Kc)
        n_b = hi - lo
        sub = {k: v[lo:hi] for k, v in flat.items()}
        if n_b < env.n_envs:
            pad = env.n_envs - n_b
            sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                   for k, v in sub.items()}
        for k2 in ('q0', 'line_dir', 'n_target'):
            sub[k2] = sub[k2].to(device=dev, dtype=dt)
        env.line_dist = ScriptedLineDistribution(sub)
        env.reset()
        done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
        for _ in range(blocks):
            a = vfn(env, done)
            for _ in range(so.SUB):
                env.step(a, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        L[lo:hi] = env.arc_progress[:n_b].float().cpu().numpy()
        if (lo // BATCH) % 10 == 0:
            say(f'  [vlabel] {hi}/{N * Kc}')
    return L.reshape(N, Kc)

# ---- stage: smoke ----------------------------------------------------------
if not (OUT / 'smoke.ok').exists():
    say('[smoke] labeler vs ladder vlook reference')
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=4242, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    valid = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)
    ids = valid[:1024]
    q0 = pool.q_pool[ids].cpu()
    p0 = env.kin.tcp_fk_jac(
        q0.to(device=dev, dtype=env.kin.dtype))[0].float().cpu()
    spec = {'p0': p0, 'line_dir': pool.line_dir_pool[ids].cpu(),
            'n_target': pool.n_target_pool[ids].cpu()}
    L = label_family_value(spec, q0.numpy()[:, None, :])
    say(f'[smoke] mean {L.mean():.4f} m (ladder vlook reference 0.5678)')
    assert abs(L.mean() - 0.5678) < 0.006, 'labeler does not reproduce vlook'
    (OUT / 'smoke.ok').write_text(f'{L.mean():.4f}')

# ---- stage: candidates -----------------------------------------------------
tasks = torch.load(MAIN / 'runs/selector_ood/v1/tasks.pt', weights_only=False)
# benchmark tasks for the framework table + l_pw fractions
tb = np.load(MAIN / 'runs/eval_10k_systematic/eval_set_10k.npz')
bench = {'p0': torch.tensor(tb['cs_p0'], dtype=torch.float32),
         'line_dir': torch.tensor(tb['cs_line_dir'], dtype=torch.float32),
         'n_target': torch.tensor(tb['cs_n_target'], dtype=torch.float32),
         'q0': torch.tensor(tb['q0_seed'], dtype=torch.float32)}
bench['line_dir'] /= bench['line_dir'].norm(dim=-1, keepdim=True)
bench['n_target'] /= bench['n_target'].norm(dim=-1, keepdim=True)
ALL = {**{k: v for k, v in tasks.items()}, 'benchmark': bench}

if not (OUT / 'cands.pt').exists():
    rng = np.random.default_rng(11)
    store = {}
    for key, spec in ALL.items():
        say(f'[cand] {key} (n={spec["p0"].shape[0]}, K={K}) ...')
        c, nf = so.gen_candidates(
            env, spec['p0'].numpy().astype(np.float32),
            spec['n_target'].numpy().astype(np.float32),
            spec['q0'].numpy().astype(np.float32), rng)
        store[key] = {'cands': torch.tensor(c), 'n_found': torch.tensor(nf)}
        say(f'[cand] {key}: mean found {nf.astype(float).mean():.2f} of {K}')
        torch.save(store, OUT / 'cands.pt')
    say('[cand] done')
cands = torch.load(OUT / 'cands.pt', weights_only=False)

# ---- stage: labels ---------------------------------------------------------
if not (OUT / 'labels.done').exists():
    if (OUT / 'labels.pt').exists():
        labels = torch.load(OUT / 'labels.pt', weights_only=False)
    else:
        labels = {}
    for key, spec in ALL.items():
        if key in labels:
            continue
        say(f'[label/value] {key} ...')
        L = label_family_value(spec, cands[key]['cands'].numpy())
        labels[key] = torch.tensor(L)
        say(f'[label/value] {key}: mean best {L.max(1).mean():.3f} m')
        torch.save(labels, OUT / 'labels.pt')
    (OUT / 'labels.done').write_text('ok')
labels = torch.load(OUT / 'labels.pt', weights_only=False)

# ---- stage: train rankers --------------------------------------------------
def pack(keys):
    Xc, Xr, Y, M = [], [], [], []
    for key in keys:
        spec = ALL[key]
        c = cands[key]['cands']
        Xc.append(so.cand_features(env, c.numpy(), spec))
        rt = so.road_table(spec)
        Xr.append(torch.tensor(rt.reshape(rt.shape[0], -1)))
        Y.append(labels[key].float())
        nf = cands[key]['n_found']
        M.append(torch.arange(K)[None, :] < nf[:, None])
    return (torch.cat(Xc), torch.cat(Xr), torch.cat(Y), torch.cat(M))

if not (OUT / 'rankers.pt').exists():
    say('[train] rankers on K=32 value-law pools')
    tr = pack(['train_straight', 'train_arc'])
    nets = {}
    for cond in (True, False):
        nets[cond] = so.train_ranker(*tr, conditioned=cond, dev=dev)
    torch.save({'cond': nets[True].state_dict(),
                'nocond': nets[False].state_dict()}, OUT / 'rankers.pt')
    say('[train] done')
rk = torch.load(OUT / 'rankers.pt', weights_only=False)

# ---- stage: report ---------------------------------------------------------
def selector_scores(key, which='cond'):
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
    fq = C.reshape(-1, 7)
    fp = spec['p0'].to(dev).repeat_interleave(K, 0)
    fd = spec['line_dir'].to(dev).repeat_interleave(K, 0)
    fn = spec['n_target'].to(dev).repeat_interleave(K, 0)
    CH = 4096
    mjl, mc, ph, wd = [], [], [], []
    with torch.no_grad():
        for i in range(0, fq.shape[0], CH):
            m = model.margins(fq[i:i+CH], fp[i:i+CH], fd[i:i+CH], fn[i:i+CH])
            mjl.append(m[:, 0]); mc.append(m[:, 1])
            ph.append(-0.1 * torch.logsumexp(-m / 0.1, dim=-1))
            _, _, J, _ = env.kin.tcp_fk_jac(fq[i:i+CH])
            Jp, _ = damped_pinv(J[:, :3, :], env.cfg.lambda_0,
                                env.cfg.sigma_thr)
            wd.append(1.0 / ((Jp @ fd[i:i+CH].unsqueeze(-1)).squeeze(-1)
                             .norm(dim=-1) + 1e-9))
    r = lambda t: torch.cat(t).reshape(N, K).cpu()
    return {'manip': r(wd), 'orient': r(mc), 'jl': r(mjl), 'phim': r(ph)}

def stats(key, extra_scores=None):
    Y = labels[key].float()
    nf = cands[key]['n_found']
    M = (torch.arange(K)[None, :] < nf[:, None])
    Ym = torch.where(M, Y, torch.full_like(Y, -1e9))
    orc = Ym.max(1).values
    ok = orc > 1e-6
    out = {}
    def add(name, v):
        out[name] = {'stroke': float(v[ok].mean()),
                     'frac': float((v[ok] / orc[ok]).mean())}
    add('first', Y.gather(1, M.float().argmax(1)[:, None]).squeeze(1))
    add('random', (Y * M).sum(1) / M.sum(1).clamp(min=1))
    scr = dict(heuristic_scores(key))
    scr['selector'] = selector_scores(key, 'cond')
    if extra_scores:
        scr.update(extra_scores)
    for name, S in scr.items():
        pick = torch.where(M, S, torch.full_like(S, -1e9)).argmax(1)
        add(name, Y.gather(1, pick[:, None]).squeeze(1))
    add('oracle', orc)
    return out

report = {}
for key in ['test_straight', 'test_arc', 'test_serpentine',
            'test_nonplanar', 'benchmark']:
    report[key] = stats(key)
    say(f'[report] {key}: ' + json.dumps(report[key]))
with open(OUT / 'report.json', 'w') as f:
    json.dump(report, f, indent=1)

# per-task arrays for the framework table / l_pw fractions
key = 'benchmark'
Y = labels[key].float(); nf = cands[key]['n_found']
M = (torch.arange(K)[None, :] < nf[:, None])
sel = selector_scores(key, 'cond')
pick = torch.where(M, sel, torch.full_like(sel, -1e9)).argmax(1)
np.savez(OUT / 'benchmark_arrays.npz',
         L=Y.numpy(), n_found=nf.numpy(), pick_cond=pick.numpy(),
         cands=cands[key]['cands'].numpy())
say('[report] wrote benchmark_arrays.npz — BATCH DONE')
