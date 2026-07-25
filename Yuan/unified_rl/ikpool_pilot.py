"""IK-pool pilot: can a raw cone-IK candidate pool (K=32) beat the diffusion pool?

Answers two questions on ~500 held-out (ensemble-calibration) tasks under C0:
  (a) ceiling  : IK-pool complete-candidate oracle vs diffusion-pool oracle
  (b) signal   : is candidate progress rankable by the existing 45-D static
                 features / trained selector (transfer, no retraining)?

Stages (run sequentially, each caches its artifact):
  gen      -> ikpool_candidates.npz   (seeds/ik_ok/p0/line_dir/n_target/q0_pilot)
  roll     -> ikpool_returns.npz      (progress_m per task x candidate)
  analyze  -> prints report, saves ikpool_analysis.json
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController, rollout_selected_seeds)
from Yuan.unified_rl.validity import validate_cached_dataset
from Yuan.unified_rl.offline_seed_train import load_return_cache
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.materialize_seed_blend import _members_from_states
from Yuan.unified_rl.materialize_actor_q_selector import (
    _copy_ensemble_states, _ensemble_outputs)
from Yuan.unified_rl.provenance import (
    controller_fingerprint, file_fingerprint, state_dict_fingerprint)
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate

C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
ENSEMBLE = 'Yuan/unified_rl/runs/_multiseed/reprocheck_seed31000/unified.pt'
RETURNS = 'Yuan/unified_rl/runs/r2_full_returns_v1/train_returns.npz'
OUT_DIR = Path('Yuan/unified_rl/runs/_ikpool_pilot_v1')
N_TASKS = 500  # overridden by --n-tasks (smoke test)
K_POOL = 32
GEN_SEED = 20260723


def _load_shared(device):
    """Load C0 env(1) + source split + diffusion return cache rows."""
    controller_dir = resolve_controller_dir(C0_DIR)
    env = build_env_from_run(controller_dir, 1, device)
    ck = torch.load(ENSEMBLE, map_location='cpu', weights_only=False)
    prov = ck['offline_seed_ensemble_provenance']
    src_ref = prov['source_checkpoint']
    source_ck = torch.load(src_ref['path'], map_location='cpu', weights_only=False)
    cand_ref = prov['source_candidate_cache']
    dataset = CachedSeedCandidateDataset.from_npz(cand_ref['path'])
    dataset, _ = validate_cached_dataset(
        dataset, env.kin, env.collision, chunk_size=4096,
        cone_deg=env.cfg.cone_deg)
    train_dataset = dataset.select_source_tasks(
        torch.as_tensor(source_ck['train_task_indices']).cpu())
    controller_state_sha256 = state_dict_fingerprint(
        torch.load(controller_dir / 'agent.pt', map_location='cpu', weights_only=True))
    gamma = float(np.load(RETURNS, allow_pickle=True)['controller_gamma'])
    cached = load_return_cache(
        RETURNS, source=source_ck,
        source_artifact={'size': int(src_ref['size']), 'sha256': str(src_ref['sha256'])},
        candidate_artifact={'size': int(cand_ref['size']), 'sha256': str(cand_ref['sha256'])},
        controller_artifact=controller_fingerprint(controller_dir),
        controller_state_sha256=controller_state_sha256,
        objective='undiscounted', gamma=gamma, train_dataset=train_dataset)
    calib_idx = torch.as_tensor(
        ck['offline_ensemble_calibration_local_indices']).long()
    rows = calib_idx[:N_TASKS]
    return env, ck, train_dataset, cached, rows


def _fps_select(Q: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-point subset of joint configurations (max-min)."""
    n = Q.shape[0]
    if n <= k:
        return Q
    dist = torch.cdist(Q, Q)
    picked = [0]
    mind = dist[0].clone()
    for _ in range(k - 1):
        nxt = int(mind.argmax())
        picked.append(nxt)
        mind = torch.minimum(mind, dist[nxt])
    return Q[torch.as_tensor(picked, device=Q.device)]


def stage_gen(device):
    env, ck, train_dataset, cached, rows = _load_shared(device)
    batch = train_dataset.batch
    fb = train_dataset.fallback_index
    n_sol = np.zeros(len(rows), dtype=np.int64)
    seeds = np.full((len(rows), K_POOL, 7), np.nan, dtype=np.float32)
    ik_ok = np.zeros((len(rows), K_POOL), dtype=bool)
    p0 = np.zeros((len(rows), 3), dtype=np.float32)
    ldir = np.zeros((len(rows), 3), dtype=np.float32)
    ntgt = np.zeros((len(rows), 3), dtype=np.float32)
    pilot = np.zeros((len(rows), 7), dtype=np.float32)
    for i, r in enumerate(rows.tolist()):
        rng = np.random.default_rng(GEN_SEED * 1000 + i)
        q = cone_constrained_ik_enumerate(
            p0=batch.p0[r], n_target=batch.n_target[r], line_dir=batch.line_dir[r],
            kin=env.kin, collision=env.collision,
            cone_angle_deg=29.5, n_orientations=16, n_ik_restarts=8,
            joint_margin=0.02, dedup_rad=0.08, rng=rng)
        n_sol[i] = q.shape[0]
        q = _fps_select(q, K_POOL)
        m = q.shape[0]
        if m:
            seeds[i, :m] = q.cpu().numpy().astype(np.float32)
            ik_ok[i, :m] = True
        p0[i] = batch.p0[r].cpu().numpy()
        ldir[i] = batch.line_dir[r].cpu().numpy()
        ntgt[i] = batch.n_target[r].cpu().numpy()
        pilot[i] = batch.q0[r, fb].cpu().numpy()
        if (i + 1) % 50 == 0:
            print(f'[gen] {i+1}/{len(rows)} tasks; median solutions={int(np.median(n_sol[:i+1]))}',
                  flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / 'ikpool_candidates.npz',
             seeds=seeds, ik_ok=ik_ok, p0=p0, line_dir=ldir, n_target=ntgt,
             q0_pilot=pilot, task_indices=rows.numpy().astype(np.int64),
             n_solutions_raw=n_sol)
    print(f'[gen] done. raw solutions/task: median={int(np.median(n_sol))} '
          f'p10={int(np.percentile(n_sol,10))} p90={int(np.percentile(n_sol,90))} '
          f'min={int(n_sol.min())}', flush=True)


def stage_roll(device):
    chunk = 512
    controller_dir = resolve_controller_dir(C0_DIR)
    env = build_env_from_run(controller_dir, chunk, device)
    agent = load_controller_agent(controller_dir, env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(controller_dir)).gamma)
    ds = CachedSeedCandidateDataset.from_npz(OUT_DIR / 'ikpool_candidates.npz')
    ds, stats = validate_cached_dataset(
        ds, env.kin, env.collision, chunk_size=4096, cone_deg=env.cfg.cone_deg)
    print(f'[roll] physical validation: frac_valid={stats["frac_valid"]:.3f} '
          f'tasks retained={int(stats["n_tasks_retained"])}/{int(stats["n_tasks"])}',
          flush=True)
    n_tasks, K = len(ds), ds.batch.n_candidates
    progress = np.full((n_tasks, K), np.nan, dtype=np.float32)
    ep_len = np.full((n_tasks, K), -1, dtype=np.int64)
    pairs = torch.nonzero(ds.batch.valid, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for start in range(0, pairs.shape[0], chunk):
        p = pairs[start:start + chunk]
        n_real = p.shape[0]
        if n_real < chunk:
            p = torch.cat([p, p[-1:].expand(chunk - n_real, -1)])
        cands = ds.batch.index_select(p[:, 0]).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cands, p[:, 1].to(device), ctl, gamma=gamma)
        pm = res.progress_m[:n_real].cpu().numpy()
        el = res.episode_len[:n_real].cpu().numpy()
        for j in range(n_real):
            progress[int(p[j, 0]), int(p[j, 1])] = pm[j]
            ep_len[int(p[j, 0]), int(p[j, 1])] = el[j]
        print(f'[roll] {min(start+chunk, pairs.shape[0])}/{pairs.shape[0]} pairs', flush=True)
    np.savez(OUT_DIR / 'ikpool_returns.npz',
             progress_m=progress, episode_len=ep_len,
             valid=ds.batch.valid.numpy(),
             task_indices=ds.task_indices.numpy().astype(np.int64))
    print('[roll] done', flush=True)


def stage_analyze(device):
    env, ck, train_dataset, cached, rows = _load_shared(device)
    ik = np.load(OUT_DIR / 'ikpool_returns.npz')
    # task_indices already hold rows into the cached diffusion tables
    ik_rows = ik['task_indices']
    calib_rows = ik_rows
    ikP, ikV = ik['progress_m'], ik['valid']
    difP = cached.progress_m[torch.as_tensor(calib_rows)].numpy()
    difV = cached.valid[torch.as_tensor(calib_rows)].numpy()
    n = len(calib_rows)

    def _oracle(P, V):
        P = np.where(V, P, -np.inf)
        return P.max(axis=1)

    dif_first = difP[np.arange(n), difV.argmax(axis=1)]
    dif_oracle = _oracle(difP, difV)
    ik_first = ikP[np.arange(n), ikV.argmax(axis=1)]
    ik_oracle = _oracle(ikP, ikV)
    union_oracle = np.maximum(dif_oracle, ik_oracle)

    # spread among valid candidates
    def _spread(P, V):
        out = []
        for i in range(P.shape[0]):
            v = P[i][V[i]]
            if v.size >= 2:
                out.append(v.max() - v.min())
        return np.asarray(out)

    # ensemble transfer scoring on the IK pool
    ikds = CachedSeedCandidateDataset.from_npz(OUT_DIR / 'ikpool_candidates.npz')
    ikds, _ = validate_cached_dataset(
        ikds, env.kin, env.collision, chunk_size=4096, cone_deg=env.cfg.cone_deg)
    assert np.array_equal(ikds.task_indices.numpy(), ik_rows), 'row mismatch'
    feats = _build_features(env.kin, ikds, 4096)
    states, meta, arch = _copy_ensemble_states(ck, label='ensemble')
    members = _members_from_states(states, arch, device)
    validT = torch.as_tensor(ikV)
    actor, q, _ = _ensemble_outputs(
        members, feats, validT, torch.arange(len(ikds)), batch_size=1024, device=device)
    NEG = -np.inf
    actor_sel = np.where(ikV, actor, NEG).argmax(axis=1)
    aq_sel = np.where(ikV, actor + 0.2 * q / 0.01, NEG).argmax(axis=1)
    qonly_sel = np.where(ikV, q, NEG).argmax(axis=1)
    r = np.arange(n)
    picks = {
        'first_valid': ik_first,
        'ens_actor': ikP[r, actor_sel],
        'ens_actor_q_w0.2': ikP[r, aq_sel],
        'ens_feasibility_only': ikP[r, qonly_sel],
    }

    # within-task Spearman between feasibility head and true progress
    from scipy.stats import spearmanr
    rho_q, rho_a = [], []
    for i in range(n):
        v = ikV[i]
        if v.sum() >= 3:
            s1 = spearmanr(q[i][v], ikP[i][v]).statistic
            s2 = spearmanr(actor[i][v], ikP[i][v]).statistic
            if np.isfinite(s1): rho_q.append(s1)
            if np.isfinite(s2): rho_a.append(s2)

    # raw per-feature ranking power: does ANY static feature order the pool?
    featsN = feats.numpy()          # (n, K, 45) raw, un-normalized
    n_feat = featsN.shape[-1]
    rho_feat = np.zeros(n_feat)
    for f in range(n_feat):
        vals = []
        for i in range(n):
            v = ikV[i]
            if v.sum() >= 3 and np.isfinite(featsN[i, v, f]).all():
                s = spearmanr(featsN[i, v, f], ikP[i][v]).statistic
                if np.isfinite(s):
                    vals.append(s)
        rho_feat[f] = np.median(vals) if vals else 0.0
    top = np.argsort(-np.abs(rho_feat))[:8]
    # oracle-of-single-feature: pick argmax (or argmin if rho<0) of best feature
    fbest = int(top[0])
    sign = 1.0 if rho_feat[fbest] >= 0 else -1.0
    fsel = np.where(ikV, sign * featsN[..., fbest], NEG).argmax(axis=1)
    picks['best_single_feature'] = ikP[r, fsel]

    ikspread, difspread = _spread(ikP, ikV), _spread(difP, difV)
    rep = {
        'n_tasks': int(n),
        'valid_per_task_ik': float(ikV.sum(1).mean()),
        'valid_per_task_diffusion': float(difV.sum(1).mean()),
        'mean_progress_m': {
            'diffusion_first_valid': float(dif_first.mean()),
            'diffusion_oracle': float(dif_oracle.mean()),
            'ik_first_valid': float(ik_first.mean()),
            'ik_oracle': float(ik_oracle.mean()),
            'union_oracle': float(union_oracle.mean()),
            **{f'ik_{k}': float(v.mean()) for k, v in picks.items() if k != 'first_valid'},
        },
        'ceiling': {
            'ik_minus_diffusion_oracle_mm': float((ik_oracle - dif_oracle).mean() * 1e3),
            'union_minus_diffusion_oracle_mm': float((union_oracle - dif_oracle).mean() * 1e3),
            'ik_oracle_beats_diffusion_pct': float((ik_oracle > dif_oracle + 1e-3).mean() * 100),
            'diffusion_beats_ik_pct': float((dif_oracle > ik_oracle + 1e-3).mean() * 100),
        },
        'spread_max_minus_min_mm': {
            'ik_median': float(np.median(ikspread) * 1e3),
            'ik_p90': float(np.percentile(ikspread, 90) * 1e3),
            'diffusion_median': float(np.median(difspread) * 1e3),
            'diffusion_p90': float(np.percentile(difspread, 90) * 1e3),
        },
        'transfer_selection': {
            k: {
                'mean_m': float(v.mean()),
                'capture_pct': float(
                    (v - ik_first).sum() / max((ik_oracle - ik_first).sum(), 1e-9) * 100),
            } for k, v in picks.items()
        },
        'within_task_spearman': {
            'feasibility_head_median': float(np.median(rho_q)),
            'actor_head_median': float(np.median(rho_a)),
            'n_tasks_scored': len(rho_q),
        },
        'raw_feature_spearman_top8': {
            f'feat_{int(i)}': float(rho_feat[int(i)]) for i in top
        },
    }
    (OUT_DIR / 'ikpool_analysis.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('gen', 'roll', 'analyze', 'all'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--n-tasks', type=int, default=None)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    global N_TASKS, OUT_DIR
    if args.n_tasks is not None:
        N_TASKS = args.n_tasks
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
    device = torch.device(args.device)
    torch.manual_seed(GEN_SEED)
    stages = ('gen', 'roll', 'analyze') if args.stage == 'all' else (args.stage,)
    for s in stages:
        print(f'===== stage {s} =====', flush=True)
        globals()[f'stage_{s}'](device)


if __name__ == '__main__':
    main()
