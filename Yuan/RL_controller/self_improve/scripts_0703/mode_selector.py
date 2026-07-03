"""A: task-level dual-mode selector — at t=0 choose soup3-pure vs r12m-hybrid.

Stage 'labels': sample fresh TRAIN-distribution tasks (never the 10k), roll
both modes, record (initial obs, L_pure, L_hyb).
Stage 'fit+apply': fit a margin regressor from initial obs; apply to the 10k
initial obs and score by slicing the existing per-task caches (deterministic
env => cached L is exactly what a deployed selector would get).
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)

RUNS = Path('Yuan/RL_controller/runs')
PURE_CKPT = RUNS / 'distill_soup3_s2_b975'
HYB_CKPT = RUNS / 'distill_r12m_b0.965_soup2'
TAU = (0.985, 0.96)
OUT = RUNS / 'mode_selector'
EVAL_SET = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
N_TRAIN_TASKS = 24576
CHUNK = 4096


@torch.no_grad()
def initial_obs(env, qs, p0s, lds, nts):
    """Exact t=0 obs under the eval convention (p_start overridden)."""
    outs = []
    B = qs.shape[0]
    dtype = env.kin.dtype
    for s in range(0, B, env.n_envs):
        e = min(s + env.n_envs, B)
        pad = env.n_envs - (e - s)
        def _t(x, w):
            t = torch.as_tensor(x[s:e], device=env.device, dtype=dtype)
            return torch.cat([t, t[-1:].expand(pad, w)]) if pad else t
        env.line_dist = ScriptedLineDistribution(
            {"q0": _t(qs, 7), "line_dir": _t(lds, 3), "n_target": _t(nts, 3)})
        env.reset()
        env.p_start[:] = _t(p0s, 3)
        outs.append(env.current_obs()[:e - s].float().cpu())
    return torch.cat(outs)


def stage_labels(device):
    OUT.mkdir(exist_ok=True)
    out = OUT / 'train_labels.npz'
    if out.exists():
        print('[sel] train_labels.npz exists, skip', flush=True)
        return
    _, cfg_yaml = load_agent(PURE_CKPT, device)
    env_kw = load_env_kw(cfg_yaml)
    line_cfg = cfg_yaml['line_distribution']
    threshold_m = (float(line_cfg['feasibility_threshold_m'])
                   if line_cfg.get('feasibility_filter', False) else None)
    proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, 'n_envs': 1}),
                           line_dist=None, device=device)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision, n_pool=line_cfg['n_pool'],
        n_target_noise_deg=line_cfg['n_target_noise_deg'],
        seed=line_cfg['train_seed'], env_cfg=EnvConfig(**{**env_kw, 'n_envs': 1}),
        feasibility_threshold_m=threshold_m)
    gen = torch.Generator(device=device).manual_seed(9100)
    tasks = pool.sample(N_TRAIN_TASKS, generator=gen)
    q0 = tasks['q0'].cpu().numpy().astype(np.float32)
    ld = tasks['line_dir'].cpu().numpy().astype(np.float32)
    nt = tasks['n_target'].cpu().numpy().astype(np.float32)
    p0, _, _, _ = proxy.kin.tcp_fk_jac(tasks['q0'].to(proxy.kin.dtype))
    p0 = p0.cpu().numpy().astype(np.float32)
    del proxy

    env = build_env(PURE_CKPT / 'config.yaml', CHUNK, device)
    classical = ClassicalNullspaceController(env.kin)
    obs0 = initial_obs(env, q0, p0, ld, nt).numpy()

    pure_agent = load_rl_agent(PURE_CKPT, env, device)
    r_pure = rollout_seeds_batched(q0, p0, ld, nt, env=env,
                                   controller='hybrid_variantB',
                                   classical=classical, agent=pure_agent,
                                   tau_enter=float('inf'), tau_exit=float('inf'),
                                   progress_prefix='sel-pure ')
    hyb_agent = load_rl_agent(HYB_CKPT, env, device)
    r_hyb = rollout_seeds_batched(q0, p0, ld, nt, env=env,
                                  controller='hybrid_variantB',
                                  classical=classical, agent=hyb_agent,
                                  tau_enter=TAU[0], tau_exit=TAU[1],
                                  progress_prefix='sel-hyb ')
    np.savez_compressed(out, obs0=obs0, L_pure=r_pure['L'], L_hyb=r_hyb['L'])
    dm = r_hyb['L'] - r_pure['L']
    print(f"[sel] labels: frac(pure>hyb) {100*(dm<0).mean():.1f}%  "
          f"mean|margin| {np.abs(dm).mean():.4f}", flush=True)


class MarginNet(nn.Module):
    def __init__(self, obs_dim=31, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def stage_fit_apply(device):
    d = np.load(OUT / 'train_labels.npz')
    obs = torch.from_numpy(d['obs0']).float()
    margin = torch.from_numpy(d['L_hyb'] - d['L_pure']).float()  # >0: hybrid
    n = obs.shape[0]
    sd = float(margin.std())
    tgt = margin / sd
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    n_val = n // 5
    va, tr = perm[:n_val], perm[n_val:]
    net = MarginNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    obs_tr, tgt_tr = obs[tr].to(device), tgt[tr].to(device)
    obs_va = obs[va].to(device)
    m_va = margin[va].numpy()
    for ep in range(60):
        order = torch.randperm(obs_tr.shape[0], device=device)
        for s in range(0, len(order), 4096):
            i = order[s:s + 4096]
            loss = ((net(obs_tr[i]) - tgt_tr[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred_va = net(obs_va).cpu().numpy() * sd

    # threshold sweep on val: choose pure when pred < theta
    Lp_va, Lh_va = d['L_pure'][va.numpy()], d['L_hyb'][va.numpy()]
    best = None
    for theta in np.arange(-0.02, 0.021, 0.002):
        pick_hyb = pred_va >= theta
        score = np.where(pick_hyb, Lh_va, Lp_va).mean()
        if best is None or score > best[1]:
            best = (float(theta), float(score))
    theta = best[0]
    print(f"[sel] val: theta={theta:+.3f}  selector L {best[1]:.4f}  "
          f"always-hyb {Lh_va.mean():.4f}  always-pure {Lp_va.mean():.4f}  "
          f"oracle {np.maximum(Lh_va, Lp_va).mean():.4f}", flush=True)
    torch.save({'state_dict': net.state_dict(), 'margin_std': sd,
                'theta': theta}, OUT / 'selector.pt')

    # ---- apply to 10k: initial obs -> choice -> slice caches ----
    z = np.load(EVAL_SET)
    qs, p0 = z['q0_seed'], z['cs_p0']
    ld, nt = z['cs_line_dir'], z['cs_n_target']
    Lo = z['max_label_L']
    env = build_env(PURE_CKPT / 'config.yaml', CHUNK, device)
    obs10k = initial_obs(env, qs, p0, ld, nt).to(device)
    with torch.no_grad():
        pred = (net(obs10k).cpu().numpy() * sd)
    pick_hyb = pred >= theta
    cp = np.load(PURE_CKPT / 'eval_10k.npz')
    ch = np.load(HYB_CKPT / 'eval_10k.npz')
    L_sel = np.where(pick_hyb, ch['L_hyb'], cp['L_pure'])
    v = Lo > 1e-6
    r_sel = L_sel[v] / Lo[v]
    r_hyb = ch['L_hyb'][v] / Lo[v]
    r_pure = cp['L_pure'][v] / Lo[v]
    r_max = np.maximum(r_hyb, r_pure)
    diff = r_sel - r_hyb
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"\n==== 10k SELECTOR RESULT ====")
    print(f"always-hyb(r12m)  {r_hyb.mean():.4f}")
    print(f"always-pure(soup3){r_pure.mean():.4f}")
    print(f"selector          {r_sel.mean():.4f}  (picked hyb on "
          f"{100*pick_hyb[v].mean():.1f}% tasks)")
    print(f"oracle ceiling    {r_max.mean():.4f}")
    print(f"selector - always-hyb: {diff.mean():+.4f} ± {se:.4f} "
          f"({'SIGNIFICANT' if abs(diff.mean()) > 2*se else 'tie'})")
    np.savez_compressed(OUT / 'eval_10k_selector.npz',
                        L_selector=L_sel, pick_hyb=pick_hyb, pred_margin=pred,
                        L_oracle=Lo, theta=np.float64(theta),
                        metric_ratio_selector_mean=np.float64(r_sel.mean()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['labels', 'fit', 'all'])
    args = ap.parse_args()
    dev = torch.device('cuda')
    if args.stage in ('labels', 'all'):
        stage_labels(dev)
    if args.stage in ('fit', 'all'):
        stage_fit_apply(dev)
