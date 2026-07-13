"""Consolidated CLEAN table stats. L clamped >=0 (extreme-motion length),
feasibility floor oracle_hyb>=0.10, buckets recomputed, per-task saved.
Produces tab:distill / tab:ranked / tab:ablate_exit / tab:progression, all on
the 5k test vs the fixed oracle_hyb reference, deployed pi_D single net."""
import os, sys
_l = os.path.join(sys.prefix, "lib")
if _l not in os.environ.get("LD_LIBRARY_PATH", ""):
    e = dict(os.environ); e["LD_LIBRARY_PATH"] = _l + ":" + e.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, e)
sys.path.insert(0, "/home/lqin/one"); os.chdir("/home/lqin/one")
sys.path.insert(0, "/home/lqin/one/Yuan/seed_selection/ranking")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from rank_v2 import obs_and_manip, dev
from Yuan.system_eval.rollout_controllers import build_env, rollout_seeds_batched, load_rl_agent
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
import gen_mix

P = Path('Yuan/seed_selection/runs/pipeline_v2')
PI0 = Path('Yuan/RL_controller/runs/rl_smmstart_30M')
CONV = Path('Yuan/RL_controller/runs/distill_v6_conv'); PID = CONV / 'round6'
DIFF = P / 'diffusion_q0_ckpts/step_300000.pt'; RK = P / 'ranker_piD_std'
TDM = 1.5; TE, TX = 0.985, 0.96; NC = 32; FLOOR = 0.10
env = build_env(PI0 / 'config.yaml', 16384, dev)
classical = ClassicalNullspaceController(env.kin)
ag0 = load_rl_agent(PI0, env, dev); agD = load_rl_agent(PID, env, dev)
te = np.load(P / 'test_tasks.npz')
p0 = te['cs_p0'].astype(np.float32); ld = te['cs_line_dir'].astype(np.float32); nt = te['cs_n_target'].astype(np.float32)
T = len(p0)
ev = np.load(P / 'eval_test_5k_perTask.npz'); ell = ev['ell_ref']
fin = np.isfinite(ell) & (ell >= FLOOR)
bucket = np.where(ell >= 0.80, 'Easy', np.where(ell >= 0.45, 'Medium', 'Difficult'))
print(f'[final] feasible (oracle>={FLOOR}): {fin.sum()}  E/M/D = '
      f"{((bucket=='Easy')&fin).sum()}/{((bucket=='Medium')&fin).sum()}/{((bucket=='Difficult')&fin).sum()}", flush=True)

sd, ok = gen_mix.gen_candidates(p0, ld, nt, DIFF, env.kin, dev, seed0=555)
first = np.array([np.nonzero(ok[i])[0][0] if ok[i].any() else 0 for i in range(T)])
seed_fv = sd[np.arange(T), first].astype(np.float32)


def roll1(seed, agent, ctrl, te_=TE, tx=TX):
    kw = dict(env=env, controller=ctrl, classical=classical, agent=agent, target_distance_m=TDM, progress_prefix='  ')
    if ctrl == 'hybrid_variantB':
        kw['tau_enter'] = te_; kw['tau_exit'] = tx
    return np.clip(rollout_seeds_batched(seed, p0, ld, nt, **kw)['L'].astype(np.float32) * TDM, 0.0, None)


def rollC(seeds3d, ok3d, agent, te_=1.0, tx=1.0):
    C = seeds3d.shape[1]; out = np.zeros((T, C), np.float32)
    flat = np.nonzero(ok3d.reshape(-1))[0]; tof = np.repeat(np.arange(T), C)
    qv = seeds3d.reshape(T * C, 7)[flat].astype(np.float32)
    r = rollout_seeds_batched(qv, p0[tof][flat], ld[tof][flat], nt[tof][flat], env=env,
                              controller='hybrid_variantB', classical=classical, agent=agent,
                              tau_enter=te_, tau_exit=tx, target_distance_m=TDM, progress_prefix='  ')
    out.reshape(-1)[flat] = np.clip(r['L'].astype(np.float32) * TDM, 0.0, None)
    return out


def S(a, m):
    v = a[m]; v = v[np.isfinite(v)]
    return [round(float(v.mean()), 3), round(float(v.std()), 3), round(float(v.min()), 3), round(float(v.max()), 3)]


def rows(l):
    pc = 100.0 * l / np.maximum(ell, 1e-9); o = {}
    for b in ('All', 'Easy', 'Medium', 'Difficult'):
        m = fin if b == 'All' else (bucket == b) & fin
        o[b] = {'l': S(l, m), 'pct': S(pc, m), 'n': int(m.sum())}
    return o


# tab:distill
print('[final] tab:distill 5 controllers...', flush=True)
distill = {'Classical': rows(roll1(seed_fv, ag0, 'classical')),
           'RL': rows(roll1(seed_fv, ag0, 'hybrid_variantB', 1.0, 1.0)),
           'piD': rows(roll1(seed_fv, agD, 'hybrid_variantB', 1.0, 1.0)),
           'Hybrid': rows(roll1(seed_fv, ag0, 'hybrid_variantB', TE, TX)),
           'Hybrid_piD': rows(roll1(seed_fv, agD, 'hybrid_variantB', TE, TX))}

# tab:ablate_exit
print('[final] tab:ablate_exit rounds...', flush=True)
exrows = {}
Lp0 = roll1(seed_fv, ag0, 'hybrid_variantB', 1.0, 1.0)
exrows['pi0'] = {'pct': round(float((100.0 * Lp0 / np.maximum(ell, 1e-9))[fin].mean()), 2),
                 'sw': round(100 * float((roll1(seed_fv, ag0, 'hybrid_variantB', TE, TX)[fin] > Lp0[fin] + 1e-4).mean()), 1)}
for k in range(7):
    ag = load_rl_agent(CONV / f'round{k}', env, dev)
    Lp = roll1(seed_fv, ag, 'hybrid_variantB', 1.0, 1.0); Lh = roll1(seed_fv, ag, 'hybrid_variantB', TE, TX)
    exrows[k] = {'pct': round(float((100.0 * Lp / np.maximum(ell, 1e-9))[fin].mean()), 2),
                 'sw': round(100 * float((Lh[fin] > Lp[fin] + 1e-4).mean()), 1)}

# tab:ranked + progression (pi_D standalone candidates)
print('[final] pi_D standalone 32-candidate roll...', flush=True)
dl = rollC(sd, ok, agD, 1.0, 1.0)
Xf = gen_mix.features(sd, ok, p0, ld, nt, env, obs_and_manip)
np.savez_compressed(P / 'final_piD_cand.npz', dl=dl, ok=ok, X=Xf)
n = np.load(RK / 'norm.npz'); Xn = ((Xf - n['mean']) / n['std']).astype(np.float32)


class Rank(nn.Module):
    def __init__(s, d=32):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


ens = [Rank().to(dev) for _ in range(10)]
for i, m in enumerate(ens): m.load_state_dict(torch.load(RK / f'net{i}.pt')); m.eval()
with torch.no_grad():
    fl = torch.from_numpy(Xn.reshape(-1, 32)).to(dev)
    sc = sum(torch.cat([m(fl[s:s+65536]) for s in range(0, len(fl), 65536)]) for m in ens).view(T, NC).cpu().numpy()
pick = np.where(ok, sc, -np.inf).argmax(1)
ranked = {'first_valid': rows(dl[np.arange(T), first]), 'ranked': rows(dl[np.arange(T), pick]),
          'best_of_batch': rows(np.where(ok, dl, -np.inf).max(1))}
def bok(K):
    b = np.clip(np.where(ok[:, :K], dl[:, :K], -np.inf).max(1), 0, None)
    return round(float((100.0 * b / np.maximum(ell, 1e-9))[fin].mean()), 2)
ranked['bestK'] = {k: bok(k) for k in (2, 4, 8)}

# tab:progression: train each objective on ranker_piD_std train_feat, score dl
print('[final] tab:progression objectives...', flush=True)
tf = np.load(RK / 'train_feat.npz'); Xtr, ytr, oktr = tf['X'], tf['y'], tf['ok']
mn, st, ysd = n['mean'], n['std'], float(n['ysd'])
Ntr = len(Xtr); rng = np.random.default_rng(0); tri = np.sort(rng.permutation(Ntr)[Ntr // 10:])
Xd = torch.from_numpy(((Xtr - mn) / st).astype(np.float32)).to(dev)
yd = torch.from_numpy((ytr / ysd).astype(np.float32)).to(dev); okd = torch.from_numpy(oktr).to(dev)
trd = torch.from_numpy(tri).to(dev)
def lf(kind, pred, yy, okk):
    dy = yy.unsqueeze(2) - yy.unsqueeze(1); dp = pred.unsqueeze(2) - pred.unsqueeze(1); po = okk.unsqueeze(2) & okk.unsqueeze(1)
    if kind == 'Pointwise': return ((pred - yy)[okk] ** 2).mean()
    if kind == 'RankNet':
        pv = po & (dy > 0.01); return Fn.softplus(-dp[pv]).mean() if pv.sum() > 0 else pred.sum() * 0
    if kind == 'ListNet':
        lp = torch.log_softmax(pred.masked_fill(~okk, -1e9), -1); tg = torch.softmax((yy / 0.05).masked_fill(~okk, -1e9), -1); return -(tg * lp).sum(-1).mean()
    if kind == 'LambdaRank':
        pv = po & (dy > 0.01); return (Fn.softplus(-dp) * dy.abs() * pv).sum() / pv.sum().clamp(min=1) if pv.sum() > 0 else pred.sum() * 0
    if kind == 'Contrastive':
        best = yy.masked_fill(~okk, -1e9).max(1, keepdim=True).values; pos = okk & (yy >= 0.97 * best)
        lg = pred.masked_fill(~okk, -1e9) / 0.1; lp = lg - torch.logsumexp(lg, -1, keepdim=True); return (-(lp * pos).sum(1) / pos.sum(1).clamp(min=1)).mean()
    if kind == 'ListMLE':
        big = pred.masked_fill(~okk, -1e9); order = torch.argsort(yy.masked_fill(~okk, -1e30), 1, descending=True)
        ps = torch.gather(big, 1, order); lse = torch.flip(torch.logcumsumexp(torch.flip(ps, [1]), 1), [1]); return ((lse - ps) * okk.gather(1, order)).sum(1).mean()
    pv = po & (dy > 0.01); pl = torch.relu(0.05 - dp[pv]).mean() if pv.sum() > 0 else pred.sum() * 0
    lp = torch.log_softmax(pred.masked_fill(~okk, -1e9), -1); tg = torch.softmax((yy / 0.05).masked_fill(~okk, -1e9), -1); return pl + 0.5 * (-(tg * lp).sum(-1).mean())
def fitk(kind, seed):
    torch.manual_seed(seed); net = Rank().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    for ep in range(60):
        o = trd[torch.randperm(len(trd), device=dev)]
        for s in range(0, len(o), 1024):
            ti = o[s:s + 1024]; pr = net(Xd[ti].reshape(-1, 32)).view(len(ti), NC)
            loss = lf(kind, pr, yd[ti], okd[ti]); opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net
prog = {}
Xtn = torch.from_numpy(Xn.reshape(-1, 32)).to(dev)
for kind in ['Pointwise', 'RankNet', 'ListNet', 'ListMLE', 'LambdaRank', 'Contrastive', 'pair+list']:
    nets = [fitk(kind, s) for s in range(3)]
    with torch.no_grad():
        s2 = sum(torch.cat([nn_(Xtn[s:s+65536]) for s in range(0, len(Xtn), 65536)]) for nn_ in nets).view(T, NC).cpu().numpy()
    pk = np.where(ok, s2, -np.inf).argmax(1); pc = 100.0 * dl[np.arange(T), pk] / np.maximum(ell, 1e-9)
    prog[kind] = round(float(pc[fin].mean()), 2)
    print(f'   {kind}: {prog[kind]}', flush=True)

json.dump({'distill': distill, 'exit': exrows, 'ranked': ranked, 'progression': prog,
           'meta': {'n_feasible': int(fin.sum()), 'floor': FLOOR}}, open(P / 'final_tables.json', 'w'), indent=2)
print('[final] wrote', P / 'final_tables.json', flush=True)
