"""Controlled test: for the SAME 1304 clean pilot tasks (disjoint from eval +
holdout), train the ranker on (a) SMM-enumerated candidates vs (b) diffusion
candidates. Isolates candidate SOURCE at fixed task count. Both rolled out
under the deployed controller (round-12 student), same features, same
RankNet loss, evaluated on the 10k eval set.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import obs_and_manip
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.system_eval.seed_sources import diffusion_seeds

CKPT = Path('Yuan/RL_controller/runs/exit_rounds7plus/final_avg')
TAU = (0.985, 0.96)
dev = torch.device('cuda')
OUT = Path('Yuan/seed_selection/runs/enum_vs_diff')
OUT.mkdir(parents=True, exist_ok=True)

p = np.load('Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz')
pc = np.load('Yuan/seed_selection/runs/pilot_20k/pilot_20k.plane_collision.npz')
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
used = set(z['src_idx'].tolist()) | set(
    np.load('Yuan/seed_selection/runs/fresh_holdout/fresh_set_2k.npz')['src_idx'].tolist())
kept = p['status'] == 'kept'
safe = ~pc['any_label_collides'].astype(bool)
vm = p['top_Kprime_valid_mask']
clean = kept & safe & (vm.sum(1) >= 2)
tid = np.array([i for i in np.where(clean)[0] if i not in used])
T = len(tid)
print(f'[data] {T} clean pilot training tasks', flush=True)

p0 = p['cs_p0'][tid].astype(np.float32)
ld = p['cs_line_dir'][tid].astype(np.float32)
nt = p['cs_n_target'][tid].astype(np.float32)

env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = __import__('Yuan.RL_controller.env.classical_nullspace',
                       fromlist=['ClassicalNullspaceController']
                       ).ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dc = cfg['diffusion']


def rollout_feats(seeds, ok):
    """seeds (T,C,7), ok (T,C) -> L (T,C), obs (T,C,31), mu (T,C) under deploy ctrl."""
    C = seeds.shape[1]
    L = np.zeros((T, C), np.float32)
    obs = np.zeros((T, C, 31), np.float32)
    mu = np.zeros((T, C), np.float32)
    flat = np.nonzero(ok.reshape(-1))[0]
    tof = np.repeat(np.arange(T), C)
    qv = seeds.reshape(T * C, 7)[flat].astype(np.float32)
    r = rollout_seeds_batched(qv, p0[tof][flat], ld[tof][flat], nt[tof][flat],
                              env=env, controller='hybrid_variantB',
                              classical=classical, agent=agent,
                              tau_enter=TAU[0], tau_exit=TAU[1], progress_prefix='  ')
    ob, m = obs_and_manip(env, qv, p0[tof][flat], ld[tof][flat], nt[tof][flat])
    L.reshape(-1)[flat] = r['L'] * 1.5
    obs.reshape(-1, 31)[flat] = ob
    mu.reshape(-1)[flat] = m
    return L, obs, mu


# (a) enumerated candidates
enum_q = p['top_Kprime_q'][tid].astype(np.float32)          # (T,6,7)
enum_ok = vm[tid]
if not (OUT / 'enum.npz').exists():
    print('[enum] rolling enumerated candidates', flush=True)
    Le, obe, mue = rollout_feats(enum_q, enum_ok)
    np.savez_compressed(OUT / 'enum.npz', L=Le, obs=obe, mu=mue, ok=enum_ok)
d = np.load(OUT / 'enum.npz')
enumL, enumObs, enumMu, enumOk = d['L'], d['obs'], d['mu'], d['ok']

# (b) diffusion candidates for the SAME tasks
if not (OUT / 'diff.npz').exists():
    print('[diff] sampling diffusion candidates', flush=True)
    fs = {'cs_p0': p0, 'cs_line_dir': ld, 'cs_n_target': nt}
    sd, sok = diffusion_seeds(fs, dc['ckpt'], n_samples=8, ddim_steps=50,
                              cfg_w=1.5, sample_seed=555, kin=env.kin,
                              device=dev, verbose=False)
    Ld, obd, mud = rollout_feats(sd, sok)
    np.savez_compressed(OUT / 'diff.npz', L=Ld, obs=obd, mu=mud, ok=sok)
d = np.load(OUT / 'diff.npz')
diffL, diffObs, diffMu, diffOk = d['L'], d['obs'], d['mu'], d['ok']

print(f'[data] enum: {enumOk.sum(1).mean():.1f} valid/task  '
      f'diff: {diffOk.sum(1).mean():.1f} valid/task', flush=True)


# ---- ranker (RankNet, single model) ----
class Rank(nn.Module):
    def __init__(s, d=32):
        super().__init__()
        s.n = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 512),
                            nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(s, x):
        return s.n(x).squeeze(-1)


# eval arrays (IK-fixed, no fallback, 24 diffusion cand) from prior cache
c = np.load('Yuan/seed_selection/runs/rank_phase0/ikfix_nofallback.npz')
ok24, L24 = c['ok24'], c['L24']
X10 = np.concatenate([c['obs24'], np.log(c['mu24'][..., None] + 1e-9)], -1)
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
fin = oh > 1e-6
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))


def train_eval(L, obs, mu, ok, name, mean=None, std=None):
    X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1).astype(np.float32)
    if mean is None:
        mean = X[ok].mean(0); std = X[ok].std(0) + 1e-6
    Xn = (X - mean) / std
    Xd = torch.from_numpy(Xn).to(dev)
    yd = torch.from_numpy(L).to(dev)
    okd = torch.from_numpy(ok).to(dev)
    Cc = L.shape[1]
    torch.manual_seed(0)
    net = Rank().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
    for ep in range(120):
        order = torch.randperm(T, device=dev)
        for s in range(0, T, 256):
            ti = order[s:s + 256]
            pred = net(Xd[ti].reshape(-1, 32)).view(len(ti), Cc)
            yy, okk = yd[ti], okd[ti]
            dy = yy.unsqueeze(2) - yy.unsqueeze(1)
            dp = pred.unsqueeze(2) - pred.unsqueeze(1)
            vp = okk.unsqueeze(2) & okk.unsqueeze(1) & (dy > 0)
            loss = (torch.nn.functional.softplus(-dp) * vp).sum() / vp.sum().clamp(min=1) \
                if vp.sum() > 0 else pred.sum() * 0
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    # eval on 10k
    X10n = torch.from_numpy((X10 - mean) / std).float()
    with torch.no_grad():
        flat = X10n.reshape(-1, 32)
        sc = torch.cat([net(flat[s:s + 65536].to(dev)).cpu()
                        for s in range(0, len(flat), 65536)]).view(len(L24), 24).numpy()
    pick = np.where(ok24, sc, -np.inf).argmax(1)
    Lp = np.where(ok24.any(1), L24[np.arange(len(L24)), pick], 0.0)
    pct = lambda m: 100 * (Lp[m] / oh[m]).mean()
    print(f'  {name:28s} 10k: All {pct(fin):.1f}  E {pct(fin&(bucket=="Easy")):.1f}'
          f'  M {pct(fin&(bucket=="Medium")):.1f}  D {pct(fin&(bucket=="Difficult")):.1f}',
          flush=True)


print(f'\n==== SAME {T} TASKS, candidate source isolated (RankNet, 10k eval) ====')
train_eval(diffL, diffObs, diffMu, diffOk, 'diffusion candidates (8/task)')
train_eval(enumL, enumObs, enumMu, enumOk, 'enumerated candidates (6/task)')
# combined
combL = np.concatenate([diffL, enumL], 1)
combObs = np.concatenate([diffObs, enumObs], 1)
combMu = np.concatenate([diffMu, enumMu], 1)
combOk = np.concatenate([diffOk, enumOk], 1)
train_eval(combL, combObs, combMu, combOk, 'combined (14/task)')
print('[done]', flush=True)
