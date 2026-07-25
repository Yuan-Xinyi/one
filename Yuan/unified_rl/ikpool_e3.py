"""E3: head-to-head on the canonical validation + external-dev sets.

New system  = IK pool (K=32) + selector trained on the full 18,432 train IK pool,
              deployed as one static pick + one C0 rollout.
Old system  = diffusion pool + production selector (cached policy_progress_m).
Both paired on identical held-out tasks; full robust statistics.

Stages:
  build --set validation|external   -> ikpool_{set}_{candidates,returns}.npz
  headtohead                        -> ikpool_e3_headtohead.json

Locked generation params imported from ikpool_build_full (same as E1).
Selector = listwise MLP (E2 track A), trained on the full train IK pool.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenRLController, rollout_selected_seeds
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate
from Yuan.unified_rl.ikpool_build_full import (
    C0_DIR, GEN_SEED, CONE_DEG, N_ORI, N_RESTART, JOINT_MARGIN, DEDUP_RAD,
    K_POOL, ROLL_CHUNK, _fps_select)

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
DIFFUSION_CAND = 'Yuan/seed_selection/runs/rank_train/candidates_K8.npz'
EXT_CAND = 'Yuan/unified_rl/runs/external_dev_v1/candidates_K8.npz'
OLD_VAL = 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz'
OLD_EXT = 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'
SEEDS, EPOCHS, TEMP, WD = [0, 1, 2], 250, 0.1, 1e-4


def _geom_source(which):
    """Return (p0, line_dir, n_target, fallback_q0, task_ids) for the set."""
    if which == 'validation':
        src = torch.load(f'{C0_DIR}/unified.pt', map_location='cpu', weights_only=False)
        vti = torch.as_tensor(src['validation_task_indices']).long()
        c = np.load(DIFFUSION_CAND, allow_pickle=True)
        return (c['p0'][vti], c['line_dir'][vti], c['n_target'][vti],
                c['q0_pilot'][vti], vti.numpy().astype(np.int64))
    c = np.load(EXT_CAND, allow_pickle=True)
    n = c['seeds'].shape[0]
    return (c['p0'], c['line_dir'], c['n_target'], c['q0_pilot'],
            np.arange(n, dtype=np.int64))


def stage_build(args, device):
    which = args.set
    p0a, lda, nta, fbq, tids = _geom_source(which)
    m = len(tids)
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    seeds = np.full((m, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, K_POOL), bool)
    nsol = np.zeros(m, np.int64)
    for i in range(m):
        rng = np.random.default_rng(GEN_SEED * 100000 + 700000000 +
                                    (0 if which == 'validation' else 10**9) + int(tids[i]))
        q = cone_constrained_ik_enumerate(
            p0=torch.as_tensor(p0a[i]), n_target=torch.as_tensor(nta[i]),
            line_dir=torch.as_tensor(lda[i]), kin=env1.kin, collision=env1.collision,
            cone_angle_deg=CONE_DEG, n_orientations=N_ORI, n_ik_restarts=N_RESTART,
            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, rng=rng)
        nsol[i] = q.shape[0]
        q = _fps_select(q, K_POOL)
        if q.shape[0]:
            seeds[i, :q.shape[0]] = q.cpu().numpy().astype(np.float32)
            ik_ok[i, :q.shape[0]] = True
        if (i + 1) % 400 == 0:
            print(f'[build {which}] {i+1}/{m} median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    cand_path = D / f'ikpool_{which}_candidates.npz'
    np.savez(cand_path, seeds=seeds, ik_ok=ik_ok, p0=p0a, line_dir=lda,
             n_target=nta, q0_pilot=fbq, task_indices=tids, n_solutions_raw=nsol)
    print(f'[build {which}] gen done median_sol={int(np.median(nsol))} empty={(nsol==0).sum()}', flush=True)

    # rollout all valid candidates under C0
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(cand_path)
    val = check_candidate_validity(
        env.kin, env.collision, ds.batch.to(env.kin.device, dtype=env.kin.dtype),
        cone_deg=env.cfg.cone_deg).valid.cpu()
    K = ds.batch.n_candidates
    prog = np.full((m, K), np.nan, np.float32)
    elen = np.full((m, K), -1, np.int64)
    pairs = torch.nonzero(val, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0]).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm, el = res.progress_m[:nr].cpu().numpy(), res.episode_len[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]; elen[int(p[j, 0]), int(p[j, 1])] = el[j]
        print(f'[roll {which}] {min(s+ROLL_CHUNK, pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(D / f'ikpool_{which}_returns.npz', progress_m=prog, episode_len=elen,
             valid=val.numpy(), task_indices=tids)
    print(f'[build {which}] rollout done -> ikpool_{which}_returns.npz', flush=True)


def _load_pool(which, device):
    ds = CachedSeedCandidateDataset.from_npz(D / f'ikpool_{which}_candidates.npz')
    ret = np.load(D / f'ikpool_{which}_returns.npz')
    X = _build_features(build_env_from_run(resolve_controller_dir(C0_DIR), 1, device).kin, ds, 4096)
    P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0)
    V = torch.as_tensor(ret['valid'])
    return X.to(device), P.to(device), V.to(device), ret['task_indices']


def _train_selector_on_full_train(device, seed):
    ds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_candidates.npz')
    ret = np.load(D / 'ikpool_returns.npz')
    X = _build_features(build_env_from_run(resolve_controller_dir(C0_DIR), 1, device).kin, ds, 4096).to(device)
    P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(device)
    V = torch.as_tensor(ret['valid']).to(device)
    mu, sd = X[V].mean(0), X[V].std(0).clamp_min(1e-6)
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    lo = torch.where(V, P, torch.tensor(1e9, device=device)).min(1, keepdim=True).values
    hi = torch.where(V, P, torch.tensor(-1e9, device=device)).max(1, keepdim=True).values
    T = ((P - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~V, 0.0)
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=WD)
    for _ in range(EPOCHS):
        opt.zero_grad()
        logit = mlp(Xz).squeeze(-1).masked_fill(~V, -1e9)
        tgt = torch.softmax((T / TEMP).masked_fill(~V, -1e9), 1)
        loss = -(tgt * torch.log_softmax(logit, 1).clamp_min(-30)).sum(1).mean()
        loss.backward(); opt.step()
    return mlp, mu, sd


def _boot_ci(delta, fps, n=20000, seed=7):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(fps, return_inverse=True)
    gsum = np.bincount(inv, weights=delta); gcnt = np.bincount(inv)
    gmean = gsum / gcnt
    means = gmean[rng.integers(0, len(gmean), size=(n, len(gmean)))].mean(1)
    return float(np.percentile(means, 2.5)) * 1e3, float(np.percentile(means, 97.5)) * 1e3


def _stats(new, old, first, ikora, difora, fps):
    d = new - old
    trim = np.sort(d); k = int(0.05 * len(d))
    tr = trim[k:-k] if k > 0 else trim
    lo, hi = _boot_ci(d, fps)
    return {
        'new_mean_m': float(new.mean()), 'old_mean_m': float(old.mean()),
        'paired_delta_mm': float(d.mean() * 1e3), 'paired_ci95_mm': [lo, hi],
        'trimmed5_delta_mm': float(tr.mean() * 1e3),
        'clipped50_delta_mm': float(np.clip(d, -0.05, 0.05).mean() * 1e3),
        'harm_gt1mm_pct': float((d < -1e-3).mean() * 100),
        'win_gt1mm_pct': float((d > 1e-3).mean() * 100),
        'new_capture_pct': float((new - first).sum() / (ikora - first).sum() * 100),
        'ik_first_m': float(first.mean()), 'ik_oracle_m': float(ikora.mean()),
        'diffusion_oracle_m': float(difora.mean()),
        'new_vs_diffusion_oracle_mm': float((new.mean() - difora.mean()) * 1e3),
    }


def stage_headtohead(args, device):
    report = {}
    # train 3 selectors on full train pool, ensemble their logits
    selectors = [_train_selector_on_full_train(device, s) for s in SEEDS]
    for which, oldpath in [('validation', OLD_VAL), ('external', OLD_EXT)]:
        X, P, V, tids = _load_pool(which, device)
        logits = []
        for mlp, mu, sd in selectors:
            Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
            with torch.no_grad():
                logits.append(mlp(Xz).squeeze(-1).masked_fill(~V, -1e9))
        sel = torch.stack(logits).mean(0).argmax(1)
        idx = torch.arange(len(P), device=device)
        new = P[idx, sel].cpu().numpy()
        first = P[idx, V.float().argmax(1)].cpu().numpy()
        ikora = torch.where(V, P, torch.tensor(-1e9, device=device)).max(1).values.cpu().numpy()
        old = np.load(oldpath, allow_pickle=True)
        oti = old['task_indices']
        order = {int(t): i for i, t in enumerate(oti)}
        perm = np.array([order[int(t)] for t in tids])
        old_pol = np.nan_to_num(old['policy_progress_m'])[perm]
        dif_ora = np.nan_to_num(old['best_progress_m'])[perm]
        # geometry fingerprints for bootstrap (each task unique geometry)
        fps = tids.astype(np.int64)
        report[which] = _stats(new, old_pol, first, ikora, dif_ora, fps)
        r = report[which]
        print(f'[{which}] new={r["new_mean_m"]:.4f} old={r["old_mean_m"]:.4f} '
              f'delta={r["paired_delta_mm"]:+.1f}mm CI{r["paired_ci95_mm"]} '
              f'trim={r["trimmed5_delta_mm"]:+.1f} cap={r["new_capture_pct"]:.1f}% '
              f'vs_dif_oracle={r["new_vs_diffusion_oracle_mm"]:+.1f}mm', flush=True)
    (D / 'ikpool_e3_headtohead.json').write_text(json.dumps(report, indent=1))
    print('\n' + json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('build', 'headtohead'))
    ap.add_argument('--set', choices=('validation', 'external'))
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    torch.manual_seed(GEN_SEED)
    globals()[f'stage_{args.stage}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
