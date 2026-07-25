"""Kinematic-lookahead ("walk") features for static seed selection.

For every candidate q0, integrate a damped least-squares tracking step along
the task ray on the PURE KINEMATIC model: no environment, no physics stepping,
no learned policy. Records how far each branch can slide before violating the
tool cone, joint limits, collision, or tracking tolerance. This is feature
computation inside the single selector forward -- deployment stays
0 probes / 0 model rollouts / 1 real rollout.

Outputs per candidate (6 features):
  walk_len_m        ray distance survived (the headline feature)
  term_onehot(3)    cone / joint-limit / tracking-or-collision
  min_jl_margin     minimum joint-limit margin along the walk (rad)
  end_cone_cos      cone cosine at death (how it died)

Stages:
  walk    --set train|validation|external  -> ikpool_{set}_walk.npz
  analyze                                  -> signal report on E2 internal split
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
STEP_M = 0.01          # matches env v*dt
MAX_STEPS = 100        # 1.0 m > any observed progress
LAM = 0.05             # env lambda_0
POS_TOL = 0.015        # tracking-failure threshold (lenient vs 5mm validity)
CONE_DEG = 30.0
CHUNK = 32768


@torch.no_grad()
def kinematic_walk(kin, collision, q0, p0, line_dir, n_target, valid):
    """All inputs flat (B,...) on kin.device. Returns dict of (B,) tensors."""
    B = q0.shape[0]
    dev, dtype = kin.device, kin.dtype
    coslim = math.cos(math.radians(CONE_DEG))
    lo, hi = kin.lmt_lo, kin.lmt_up
    eye3 = torch.eye(3, device=dev, dtype=dtype)

    q = torch.where(valid.unsqueeze(-1), q0, kin.q_mid.expand_as(q0)).clone()
    alive = valid.clone()
    s = torch.zeros(B, device=dev, dtype=dtype)
    term = torch.zeros(B, dtype=torch.int8, device=dev)      # 0 alive/maxed, 1 cone, 2 jl, 3 track/coll
    min_margin = torch.minimum(q - lo, hi - q).min(-1).values
    end_cos = torch.ones(B, device=dev, dtype=dtype)

    for _ in range(MAX_STEPS):
        p, R, jac, _ = kin.tcp_fk_jac(q)
        cosang = (R[:, :, 2] * n_target).sum(-1)
        end_cos = torch.where(alive, cosang, end_cos)
        die_cone = alive & (cosang < coslim)
        err = (p0 + s.unsqueeze(-1) * line_dir) - p
        die_track = alive & (err.norm(dim=-1) > POS_TOL)
        die_coll = alive & collision.is_collided(kin.link_transforms(q))
        term = torch.where(die_cone, torch.tensor(1, dtype=torch.int8, device=dev), term)
        term = torch.where(die_track | die_coll,
                           torch.tensor(3, dtype=torch.int8, device=dev), term)
        alive = alive & ~(die_cone | die_track | die_coll)
        if not bool(alive.any()):
            break
        target = p0 + (s + STEP_M).unsqueeze(-1) * line_dir
        dp = target - p
        J = jac[:, :3, :]
        JJt = J @ J.transpose(-1, -2) + LAM * eye3
        dq = (J.transpose(-1, -2) @ torch.linalg.solve(JJt, dp.unsqueeze(-1))).squeeze(-1)
        q_new = q + torch.where(alive.unsqueeze(-1), dq, torch.zeros_like(dq))
        die_jl = alive & ((q_new < lo) | (q_new > hi)).any(-1)
        term = torch.where(die_jl, torch.tensor(2, dtype=torch.int8, device=dev), term)
        alive = alive & ~die_jl
        q = torch.where(alive.unsqueeze(-1), q_new, q)
        s = s + STEP_M * alive.to(dtype)
        margin = torch.minimum(q - lo, hi - q).min(-1).values
        min_margin = torch.where(alive, torch.minimum(min_margin, margin), min_margin)
    return {'walk_len_m': s, 'term': term, 'min_jl_margin': min_margin,
            'end_cone_cos': end_cos}


def stage_walk(args, device):
    which = args.set
    name = 'ikpool_candidates.npz' if which == 'train' else f'ikpool_{which}_candidates.npz'
    out = D / (f'ikpool_walk.npz' if which == 'train' else f'ikpool_{which}_walk.npz')
    if out.exists() and not args.force:
        print(f'[walk] {out.name} exists, skip', flush=True); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    ds = CachedSeedCandidateDataset.from_npz(D / name)
    ret = np.load(D / ('ikpool_returns.npz' if which == 'train'
                       else f'ikpool_{which}_returns.npz'))
    V = torch.as_tensor(ret['valid'])
    n, K = V.shape
    b = ds.batch
    q0 = b.q0.reshape(n * K, 7)
    p0 = b.p0[:, None, :].expand(-1, K, -1).reshape(-1, 3)
    ld = b.line_dir[:, None, :].expand(-1, K, -1).reshape(-1, 3)
    nt = b.n_target[:, None, :].expand(-1, K, -1).reshape(-1, 3)
    vf = V.reshape(-1)
    outs = {k: [] for k in ('walk_len_m', 'term', 'min_jl_margin', 'end_cone_cos')}
    for st in range(0, n * K, CHUNK):
        en = min(st + CHUNK, n * K)
        r = kinematic_walk(
            env.kin, env.collision,
            q0[st:en].to(device, env.kin.dtype),
            p0[st:en].to(device, env.kin.dtype),
            ld[st:en].to(device, env.kin.dtype),
            nt[st:en].to(device, env.kin.dtype),
            vf[st:en].to(device))
        for k in outs:
            outs[k].append(r[k].cpu())
        print(f'[walk {which}] {en}/{n*K} lanes', flush=True)
    merged = {k: torch.cat(v).reshape(n, K).numpy() for k, v in outs.items()}
    np.savez(out, **merged, task_indices=ret['task_indices'])
    print(f'[walk {which}] saved -> {out.name}', flush=True)


def stage_analyze(args, device):
    from scipy.stats import spearmanr
    ds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_candidates.npz')
    ret = np.load(D / 'ikpool_returns.npz')
    wk = np.load(D / 'ikpool_walk.npz')
    P, V, W = ret['progress_m'], ret['valid'], wk['walk_len_m']
    n = len(P)
    # E2 internal test split (identical construction)
    sig = np.concatenate([ds.batch.p0.numpy(), ds.batch.line_dir.numpy(),
                          ds.batch.n_target.numpy()], axis=1)
    _, gid = np.unique(np.round(sig, 6), axis=0, return_inverse=True)
    rng0 = np.random.default_rng(20260724)
    order = rng0.permutation(n)
    is_test = np.zeros(n, bool); seen = set(); cnt = 0
    for r in order:
        if cnt >= 3000:
            break
        g = gid[r]
        if g in seen:
            continue
        seen.add(g); is_test[gid == g] = True; cnt = is_test.sum()
    t = np.nonzero(is_test)[0]

    rho = []
    for i in t:
        v = V[i]
        if v.sum() >= 3:
            r_ = spearmanr(W[i][v], np.nan_to_num(P[i])[v]).statistic
            if np.isfinite(r_):
                rho.append(r_)
    Pt = np.nan_to_num(P[t]); Vt = V[t]; Wt = W[t]
    first = Vt.argmax(1); r_ = np.arange(len(t))
    ora = np.where(Vt, Pt, -np.inf).max(1)
    walk_sel = np.where(Vt, Wt, -np.inf).argmax(1)
    walk_cap = float((Pt[r_, walk_sel] - Pt[r_, first]).sum()
                     / (ora - Pt[r_, first]).sum() * 100)
    rep = {
        'n_test_tasks': int(len(t)),
        'within_task_spearman_walk': {
            'median': float(np.median(rho)), 'p25': float(np.percentile(rho, 25)),
            'p75': float(np.percentile(rho, 75)), 'n': len(rho)},
        'reference_best_static_feature_spearman': 0.28,
        'reference_trained_45d_mlp_spearman': 0.53,
        'walk_argmax_capture_pct': walk_cap,
        'reference_45d_selector_capture_pct': 62.7,
        'walk_mean_m': float(Wt[Vt].mean()),
        'progress_mean_m': float(Pt[Vt].mean()),
        'global_corr_walk_vs_progress': float(
            np.corrcoef(Wt[Vt], Pt[Vt])[0, 1]),
    }
    (D / 'ikpool_walk_signal.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('walk', 'analyze'))
    ap.add_argument('--set', default='train',
                    choices=('train', 'validation', 'external'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    globals()[f'stage_{args.stage}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
