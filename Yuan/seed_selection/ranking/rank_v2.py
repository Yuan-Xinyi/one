"""Seed-ranking v2: 9-way candidate set (8 DP + pilot seed), richer features,
fixed losses, ensembling. All ranker iteration is offline — eval-side L for
every candidate is already cached (rank_phase0 slots + r12m eval_10k L_hyb).

Stages: pilot (train-side pilot labels + obs/manip features for all slots),
fit (loss/feature grid on held-out capture), apply (one 10k evaluation).
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

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)

TRAIN = Path('Yuan/seed_selection/runs/rank_train')
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K = 8
dev = torch.device('cuda')


@torch.no_grad()
def obs_and_manip(env, qs, p0s, lds, nts):
    """t=0 obs (31) + manipulability sqrt(det(Jp Jp^T)) per row."""
    B = qs.shape[0]
    dtype = env.kin.dtype
    obs_l, mu_l = [], []
    for s in range(0, B, env.n_envs):
        e = min(s + env.n_envs, B)
        pad = env.n_envs - (e - s)
        def _t(x, w):
            t = torch.as_tensor(x[s:e], device=dev, dtype=dtype)
            return torch.cat([t, t[-1:].expand(pad, w)]) if pad else t
        env.line_dist = ScriptedLineDistribution(
            {'q0': _t(qs, 7), 'line_dir': _t(lds, 3), 'n_target': _t(nts, 3)})
        env.reset()
        env.p_start[:] = _t(p0s, 3)
        obs_l.append(env.current_obs()[:e - s].float().cpu())
        _, _, J, _ = env.kin.tcp_fk_jac(env.q)
        Jp = J[:, :3, :]
        mu = torch.sqrt(torch.det(Jp @ Jp.transpose(-1, -2)).clamp(min=1e-12))
        mu_l.append(mu[:e - s].float().cpu())
    return torch.cat(obs_l).numpy(), torch.cat(mu_l).numpy()


def stage_pilot():
    env = build_env(CKPT / 'config.yaml', 4096, dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(CKPT, env, dev)

    # ---- train side ----
    cd = np.load(TRAIN / 'candidates_K8.npz')
    q0p, p0 = cd['q0_pilot'], cd['p0']
    ld, nt = cd['line_dir'], cd['n_target']
    if not (TRAIN / 'L_pilot.npz').exists():
        r = rollout_seeds_batched(q0p, p0, ld, nt, env=env,
                                  controller='hybrid_variantB',
                                  classical=classical, agent=agent,
                                  tau_enter=TAU[0], tau_exit=TAU[1],
                                  progress_prefix='pilot-train ')
        np.savez_compressed(TRAIN / 'L_pilot.npz', L=r['L'])
        print('[v2] train pilot labeled', flush=True)
    if not (TRAIN / 'feat_v2.npz').exists():
        seeds = cd['seeds']
        N = seeds.shape[0]
        obs_p, mu_p = obs_and_manip(env, q0p, p0, ld, nt)
        obs_s = np.zeros((N, K, 31), np.float32)
        mu_s = np.zeros((N, K), np.float32)
        for si in range(K):
            obs_s[:, si], mu_s[:, si] = obs_and_manip(
                env, seeds[:, si], p0, ld, nt)
            print(f'[v2] train feat slot {si}', flush=True)
        np.savez_compressed(TRAIN / 'feat_v2.npz', obs_pilot=obs_p,
                            mu_pilot=mu_p, obs_slots=obs_s, mu_slots=mu_s)

    # ---- eval side (10k) ----
    z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    if not (P0DIR / 'feat_v2.npz').exists():
        pd = np.load(P0DIR / 'candidates_K8.npz')
        seeds = pd['seeds']
        n10 = seeds.shape[0]
        p0s = z['cs_p0'].astype(np.float32)
        lds = z['cs_line_dir'].astype(np.float32)
        nts = z['cs_n_target'].astype(np.float32)
        obs_p, mu_p = obs_and_manip(env, z['q0_seed'].astype(np.float32),
                                    p0s, lds, nts)
        obs_s = np.zeros((n10, K, 31), np.float32)
        mu_s = np.zeros((n10, K), np.float32)
        for si in range(K):
            obs_s[:, si], mu_s[:, si] = obs_and_manip(
                env, seeds[:, si], p0s, lds, nts)
            print(f'[v2] eval feat slot {si}', flush=True)
        np.savez_compressed(P0DIR / 'feat_v2.npz', obs_pilot=obs_p,
                            mu_pilot=mu_p, obs_slots=obs_s, mu_slots=mu_s)
    print('[v2] pilot stage done', flush=True)


# ---------------- fit ----------------
class Rank(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_matrix():
    """(N, 9, d) features, (N, 9) L in meters, (N, 9) valid, slot 8 = pilot."""
    cd = np.load(TRAIN / 'candidates_K8.npz')
    fv = np.load(TRAIN / 'feat_v2.npz')
    L = np.stack([np.load(TRAIN / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
    Lp = np.load(TRAIN / 'L_pilot.npz')['L']
    ik = cd['ik_ok']
    obs = np.concatenate([fv['obs_slots'],
                          fv['obs_pilot'][:, None, :]], axis=1)
    mu = np.concatenate([fv['mu_slots'], fv['mu_pilot'][:, None]], axis=1)
    X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], axis=-1)  # 32-d
    y = np.concatenate([L, Lp[:, None]], axis=1) * 1.5
    ok = np.concatenate([ik, np.ones((len(L), 1), bool)], axis=1)
    return X.astype(np.float32), y.astype(np.float32), ok


def capture_eval(scores, y, ok, y_first):
    s = np.where(ok, scores, -np.inf)
    pick = s.argmax(1)
    Lp = y[np.arange(len(y)), pick]
    y_best = np.where(ok, y, -np.inf).max(1)
    capt = (Lp - y_first).sum() / max((y_best - y_first).sum(), 1e-9)
    return Lp.mean(), 100 * capt


def fit_one(Xd, yd, taskd, okd, kind, seed, epochs=60):
    torch.manual_seed(seed)
    net = Rank(Xd.shape[-1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = Xd.shape[0]
    for ep in range(epochs):
        order = torch.randperm(N, device=dev)
        for s in range(0, N, 1024):          # 1024 tasks x 9 slots
            ti = order[s:s + 1024]
            X = Xd[ti].reshape(-1, Xd.shape[-1])
            pred = net(X).view(len(ti), 9)
            y = yd[ti]
            ok = okd[ti]
            if kind == 'point':
                loss = (((pred - y) ** 2) * ok).sum() / ok.sum()
            elif kind == 'list':
                logp = torch.log_softmax(pred.masked_fill(~ok, -1e9), -1)
                tgt = torch.softmax((y / 0.05).masked_fill(~ok, -1e9), -1)
                loss = -(tgt * logp).sum(-1).mean()
            else:  # pair: all valid within-task pairs with margin
                dy = y.unsqueeze(2) - y.unsqueeze(1)       # (B,9,9) y_i - y_j
                dp = pred.unsqueeze(2) - pred.unsqueeze(1)
                pv = ok.unsqueeze(2) & ok.unsqueeze(1) & (dy > 0.01)
                if pv.sum() == 0:
                    continue
                loss = torch.relu(0.05 - dp[pv]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net


def stage_fit_apply():
    X, y, ok = build_matrix()
    N = X.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N)
    n_val = N // 5
    va_t, tr_t = perm[:n_val], perm[n_val:]
    mean = X[tr_t][ok[tr_t]].mean(0)
    std = X[tr_t][ok[tr_t]].std(0) + 1e-6
    Xn = (X - mean) / std
    ysd = float(y[tr_t][ok[tr_t]].std())
    Xd = torch.from_numpy(Xn[tr_t]).to(dev)
    yd = torch.from_numpy(y[tr_t] / ysd).to(dev)
    okd = torch.from_numpy(ok[tr_t]).to(dev)
    taskd = None

    # val baselines: status quo = DP first-valid
    ik_dp = ok[va_t][:, :K]
    first_idx = np.argmax(ik_dp, 1)
    y_first = np.where(ik_dp.any(1),
                       y[va_t][np.arange(len(va_t)), first_idx],
                       y[va_t][:, K])          # fallback pilot
    y_best = np.where(ok[va_t], y[va_t], -np.inf).max(1)
    print(f'[v2-fit] val: first {y_first.mean():.4f}  pilot '
          f'{y[va_t][:, K].mean():.4f}  best-of-9 {y_best.mean():.4f}', flush=True)

    Xva = torch.from_numpy(Xn[va_t]).float().to(dev)

    @torch.no_grad()
    def score_val(net):
        return net(Xva.reshape(-1, X.shape[-1])).view(len(va_t), 9).cpu().numpy()

    nets, val_scores = {}, {}
    for kind in ('point', 'list', 'pair'):
        net = fit_one(Xd, yd, taskd, okd, kind, seed=0)
        sc = score_val(net)
        Lm, cap = capture_eval(sc, y[va_t], ok[va_t], y_first)
        print(f'[v2-fit] {kind:6s} val L {Lm:.4f}  capture {cap:.1f}%', flush=True)
        nets[kind], val_scores[kind] = net, (Lm, cap, sc)

    best_kind = max(val_scores, key=lambda k: val_scores[k][0])
    # ensemble 5 of the best kind
    ens = [nets[best_kind]]
    sc_sum = score_val(nets[best_kind]).copy()
    for sd_i in range(1, 5):
        n_i = fit_one(Xd, yd, taskd, okd, best_kind, seed=sd_i)
        ens.append(n_i)
        sc_sum += score_val(n_i)
    Lm, cap = capture_eval(sc_sum / 5, y[va_t], ok[va_t], y_first)
    print(f'[v2-fit] ens5-{best_kind} val L {Lm:.4f}  capture {cap:.1f}%', flush=True)

    torch.save({'nets': [n.state_dict() for n in ens], 'mean': mean,
                'std': std, 'ysd': ysd, 'kind': best_kind,
                'feat': 'obs31+logmu'}, TRAIN / 'ranker_v2.pt')

    # ---------------- apply to 10k ----------------
    fv = np.load(P0DIR / 'feat_v2.npz')
    L10 = np.stack([np.load(P0DIR / f'L_slot{si}.npz')['L']
                    for si in range(K)], 1) * 1.5
    Lp10 = np.load(CKPT / 'eval_10k.npz')['L_hyb'] * 1.5
    pd = np.load(P0DIR / 'candidates_K8.npz')
    ok10 = np.concatenate([pd['ik_ok'],
                           np.ones((len(L10), 1), bool)], 1)
    y10 = np.concatenate([L10, Lp10[:, None]], 1)
    obs10 = np.concatenate([fv['obs_slots'], fv['obs_pilot'][:, None, :]], 1)
    mu10 = np.concatenate([fv['mu_slots'], fv['mu_pilot'][:, None]], 1)
    X10 = np.concatenate([obs10, np.log(mu10[..., None] + 1e-9)], -1)
    X10n = torch.from_numpy((X10 - mean) / std).float()

    sc10 = np.zeros(y10.shape, np.float32)
    with torch.no_grad():
        for n_i in ens:
            flat = X10n.reshape(-1, X10.shape[-1]).to(dev)
            out = []
            for s in range(0, len(flat), 65536):
                out.append(n_i(flat[s:s + 65536]).cpu())
            sc10 += torch.cat(out).view(y10.shape).numpy()
    s = np.where(ok10, sc10, -np.inf)
    pick = s.argmax(1)
    L_pick = y10[np.arange(len(y10)), pick]

    res0 = np.load(P0DIR / 'phase0_results.npz')
    oh, L_first10 = res0['oh'], res0['L_first']
    fin = oh > 1e-9
    def pct(Lm_):
        return 100.0 * (Lm_[fin] / oh[fin]).mean()
    Lbest9 = np.where(ok10, y10, -np.inf).max(1)
    print(f'\n==== v2 10k RESULT (9-way, ens5-{best_kind}) ====')
    print(f'  first-valid      {pct(L_first10):.2f}%')
    print(f'  v1 ranked        93.24%')
    print(f'  v2 ranked        {pct(L_pick):.2f}%   (pilot picked on '
          f'{100*(pick==K).mean():.1f}% tasks)')
    print(f'  best-of-9        {pct(Lbest9):.2f}%')
    d10 = (L_pick[fin] - L_first10[fin]) / oh[fin] * 100
    se = d10.std(ddof=1) / np.sqrt(len(d10))
    print(f'  v2 - first: {d10.mean():+.2f}pp ± {se:.2f}')
    bucket = np.where(oh >= 0.80, 'Easy',
                      np.where(oh >= 0.45, 'Medium', 'Difficult'))
    for b in ('Easy', 'Medium', 'Difficult'):
        m = fin & (bucket == b)
        print(f'  {b:9s}: first {100*(L_first10[m]/oh[m]).mean():.1f}%  '
              f'v2 {100*(L_pick[m]/oh[m]).mean():.1f}%')
    tail = fin & (bucket == 'Easy') & (100*L_first10/np.maximum(oh, 1e-9) < 50)
    print(f'  Easy tail: first {100*(L_first10[tail]/oh[tail]).mean():.1f}%  '
          f'v2 {100*(L_pick[tail]/oh[tail]).mean():.1f}%  '
          f'still<50%: {(100*L_pick[tail]/oh[tail] < 50).sum()}/{tail.sum()}')
    np.savez_compressed(P0DIR / 'phase3_ranked_v2.npz', L_ranked=L_pick,
                        pick=pick, kind=np.str_(best_kind))
    print('[v2] done', flush=True)


if __name__ == '__main__':
    stage_pilot()
    stage_fit_apply()
