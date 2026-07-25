"""E1: full-scale IK candidate pool + C0 complete-candidate return cache.

Over ALL 18,432 r2_grouped_best train rows. Per-task RNG is seeded by the
GLOBAL train-row index, so sharding is exact and order-independent. Every stage
is resumable: a shard whose output file already exists is skipped.

Stages:
  gen  --shard i/n   -> D/cand_shard{i}of{n}.npz     (per-task cone-IK, FPS->K)
  merge-cand          -> D/ikpool_candidates.npz      (concat shards, row order)
  roll --shard i/n   -> D/ret_shard{i}of{n}.npz       (C0 full-candidate rollout)
  merge-ret           -> D/ikpool_returns.npz          (concat, aligned to train rows)
  verify              -> prints E1 acceptance checklist

Row indices stored in task_indices are LOCAL train positions (0..18431),
i.e. positions into the r2_full_returns_v1 table, so E2 can join by them.

Alignment invariants (see IKPOOL_EXPERIMENT_MANUAL_ZH.md §0.4):
  - task_indices are train-row positions, NOT source cache indices.
  - roll uses check_candidate_validity (no drop/reindex) so per-shard tables
    stay aligned to their global row range even if a task has few valid slots.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenRLController, rollout_selected_seeds
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate

C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
SRC_CKPT = 'Yuan/unified_rl/runs/r2_grouped_best/unified.pt'
DIFFUSION_CAND = 'Yuan/seed_selection/runs/rank_train/candidates_K8.npz'

# locked generation parameters (manual E1.2) -- DO NOT change without a new run id
GEN_SEED = 20260724            # NEW value, distinct from pilot's 20260723
CONE_DEG = 29.5               # NOT the cone_ik default of 5.0; validation gate is 30
N_ORI, N_RESTART = 16, 8      # 128 DLS IK attempts/task
JOINT_MARGIN, DEDUP_RAD = 0.02, 0.08
K_POOL = 32                  # oracle-vs-K saturates at 16-24; 32 is the safe plateau
ROLL_CHUNK = 512             # locked; GPU-SVD batch dependence -> part of protocol


def _fps_select(Q: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy farthest-point subset in joint space (diversity-preserving)."""
    n = Q.shape[0]
    if n <= k:
        return Q
    dist = torch.cdist(Q, Q)
    picked, mind = [0], dist[0].clone()
    for _ in range(k - 1):
        nxt = int(mind.argmax())
        picked.append(nxt)
        mind = torch.minimum(mind, dist[nxt])
    return Q[torch.as_tensor(picked, device=Q.device)]


def _train_dataset(env):
    """Validated diffusion dataset restricted to the C0 train split.

    Only used for per-task geometry (p0/line_dir/n_target) and the classical
    fallback q0, indexed by local train position so IK task_indices align with
    the r2 return table.
    """
    src = torch.load(SRC_CKPT, map_location='cpu', weights_only=False)
    ds = CachedSeedCandidateDataset.from_npz(DIFFUSION_CAND)
    return ds.select_source_tasks(torch.as_tensor(src['train_task_indices']).cpu())


def _row_shard(n_rows, shard, limit):
    rows = np.arange(n_rows if limit is None else min(limit, n_rows))
    if shard is None:
        return rows
    i, n = shard
    return np.array_split(rows, n)[i]


def stage_gen(args, device):
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    td = _train_dataset(env)
    batch, fb = td.batch, td.fallback_index
    rows = _row_shard(len(td), args.shard, args.limit)
    tag = f'{args.shard[0]}of{args.shard[1]}' if args.shard else 'all'
    out = args.out_dir / f'cand_shard{tag}.npz'
    if out.exists() and not args.force:
        print(f'[gen] shard {tag} exists, skip', flush=True); return
    m = len(rows)
    seeds = np.full((m, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, K_POOL), bool)
    p0 = np.zeros((m, 3), np.float32); ldir = np.zeros((m, 3), np.float32)
    ntgt = np.zeros((m, 3), np.float32); pilot = np.zeros((m, 7), np.float32)
    nsol = np.zeros(m, np.int64)
    for i, r in enumerate(rows.tolist()):
        rng = np.random.default_rng(GEN_SEED * 100000 + int(r))   # GLOBAL row seed
        q = cone_constrained_ik_enumerate(
            p0=batch.p0[r], n_target=batch.n_target[r], line_dir=batch.line_dir[r],
            kin=env.kin, collision=env.collision, cone_angle_deg=CONE_DEG,
            n_orientations=N_ORI, n_ik_restarts=N_RESTART,
            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, rng=rng)
        nsol[i] = q.shape[0]
        q = _fps_select(q, K_POOL)
        if q.shape[0]:
            seeds[i, :q.shape[0]] = q.cpu().numpy().astype(np.float32)
            ik_ok[i, :q.shape[0]] = True
        p0[i] = batch.p0[r].cpu().numpy(); ldir[i] = batch.line_dir[r].cpu().numpy()
        ntgt[i] = batch.n_target[r].cpu().numpy(); pilot[i] = batch.q0[r, fb].cpu().numpy()
        if (i + 1) % 200 == 0:
            print(f'[gen {tag}] {i+1}/{m}  median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=p0, line_dir=ldir, n_target=ntgt,
             q0_pilot=pilot, task_indices=rows.astype(np.int64), n_solutions_raw=nsol)
    print(f'[gen {tag}] done -> {out.name}  n={m} median_sol={int(np.median(nsol))} '
          f'min_sol={int(nsol.min())} empty={(nsol==0).sum()}', flush=True)


def _concat_shards(out_dir, prefix, keys):
    files = sorted(out_dir.glob(f'{prefix}_shard*.npz'),
                   key=lambda p: int(p.stem.split('shard')[1].split('of')[0]))
    if not files:
        raise SystemExit(f'no {prefix} shards in {out_dir}')
    data = {k: [] for k in keys}
    for f in files:
        d = np.load(f)
        for k in keys:
            data[k].append(d[k])
    merged = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    ti = merged['task_indices']
    if not np.array_equal(ti, np.sort(ti)) or len(np.unique(ti)) != len(ti):
        raise SystemExit('merged task_indices are not strictly ascending/unique')
    return merged, files


def stage_merge_cand(args, device):
    keys = ('seeds', 'ik_ok', 'p0', 'line_dir', 'n_target', 'q0_pilot',
            'task_indices', 'n_solutions_raw')
    merged, files = _concat_shards(args.out_dir, 'cand', keys)
    np.savez(args.out_dir / 'ikpool_candidates.npz', **merged)
    print(f'[merge-cand] {len(files)} shards -> ikpool_candidates.npz '
          f'n_tasks={len(merged["task_indices"])}', flush=True)


def stage_roll(args, device):
    tag = f'{args.shard[0]}of{args.shard[1]}' if args.shard else 'all'
    out = args.out_dir / f'ret_shard{tag}.npz'
    if out.exists() and not args.force:
        print(f'[roll] shard {tag} exists, skip', flush=True); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(args.out_dir / 'ikpool_candidates.npz')
    all_rows = _row_shard(len(ds), args.shard, args.limit)
    lo, hi = int(all_rows[0]), int(all_rows[-1]) + 1
    sub = ds.batch.index_select(torch.arange(lo, hi))
    val = check_candidate_validity(
        env.kin, env.collision, sub.to(env.kin.device, dtype=env.kin.dtype),
        cone_deg=env.cfg.cone_deg).valid.cpu()
    R = hi - lo
    K = ds.batch.n_candidates
    progress = np.full((R, K), np.nan, np.float32)
    ep_len = np.full((R, K), -1, np.int64)
    pairs = torch.nonzero(val, as_tuple=False).long()   # local (0..R-1, 0..K-1)
    ctl = FrozenRLController(agent)
    for start in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[start:start + ROLL_CHUNK]
        n_real = p.shape[0]
        if n_real < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - n_real, -1)])
        cand = ds.batch.index_select(p[:, 0] + lo).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm, el = res.progress_m[:n_real].cpu().numpy(), res.episode_len[:n_real].cpu().numpy()
        for j in range(n_real):
            progress[int(p[j, 0]), int(p[j, 1])] = pm[j]
            ep_len[int(p[j, 0]), int(p[j, 1])] = el[j]
        print(f'[roll {tag}] {min(start+ROLL_CHUNK, pairs.shape[0])}/{pairs.shape[0]} pairs', flush=True)
    np.savez(out, progress_m=progress, episode_len=ep_len, valid=val.numpy(),
             task_indices=ds.task_indices.numpy()[lo:hi].astype(np.int64))
    print(f'[roll {tag}] done -> {out.name}  rows={R} valid_pairs={pairs.shape[0]}', flush=True)


def stage_merge_ret(args, device):
    merged, files = _concat_shards(
        args.out_dir, 'ret', ('progress_m', 'episode_len', 'valid', 'task_indices'))
    np.savez(args.out_dir / 'ikpool_returns.npz', **merged)
    nan_on_valid = np.isnan(merged['progress_m'])[merged['valid']].any()
    print(f'[merge-ret] {len(files)} shards -> ikpool_returns.npz '
          f'n_tasks={len(merged["task_indices"])} nan_on_valid={bool(nan_on_valid)}', flush=True)


def stage_verify(args, device):
    gen = np.load(args.out_dir / 'ikpool_candidates.npz')
    ret = np.load(args.out_dir / 'ikpool_returns.npz')
    P, V, nsol = ret['progress_m'], ret['valid'], gen['n_solutions_raw']
    assert np.array_equal(gen['task_indices'], ret['task_indices']), 'cand/ret task_indices mismatch'
    Kik = P.shape[1] - 1  # exclude classical fallback slot
    Pm = np.where(V[:, :Kik], P[:, :Kik], -np.inf)
    curve = {k: float(Pm[:, :k].max(1)[np.isfinite(Pm[:, :k].max(1))].mean())
             for k in (1, 2, 4, 8, 16, 24, 32)}
    rep = {
        'n_tasks': int(len(P)),
        'median_solutions': int(np.median(nsol)), 'min_solutions': int(nsol.min()),
        'empty_tasks': int((nsol == 0).sum()),
        'frac_valid': float(V.mean()), 'valid_per_task': float(V.sum(1).mean()),
        'nan_on_valid_slot': bool(np.isnan(P)[V].any()),
        'oracle_vs_K': curve,
        'K16_saturation_pct': float(curve[16] / curve[32] * 100),
    }
    checks = {
        'median_solutions in [40,60]': 40 <= rep['median_solutions'] <= 60,
        'frac_valid in [0.80,0.95]': 0.80 <= rep['frac_valid'] <= 0.95,
        'no empty tasks': rep['empty_tasks'] == 0,
        'no nan on valid slot': not rep['nan_on_valid_slot'],
        'K16 >= 95% of K32 oracle': rep['K16_saturation_pct'] >= 95.0,
    }
    rep['ACCEPTANCE'] = checks
    rep['ALL_PASS'] = all(checks.values())
    (args.out_dir / 'ikpool_e1_verify.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('gen', 'merge-cand', 'roll', 'merge-ret', 'verify'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--shard', default=None, help='i/n')
    ap.add_argument('--limit', type=int, default=None, help='cap total rows (smoke)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    args.out_dir = Path(args.out_dir)
    if args.shard is not None:
        i, n = args.shard.split('/')
        args.shard = (int(i), int(n))
        if not 0 <= args.shard[0] < args.shard[1]:
            raise SystemExit('bad --shard i/n')
    torch.manual_seed(GEN_SEED)
    device = torch.device(args.device)
    globals()[f'stage_{args.stage.replace("-", "_")}'](args, device)


if __name__ == '__main__':
    main()
