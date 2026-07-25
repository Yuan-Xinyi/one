"""E6: sealed 10k final evaluation (rules frozen in ikpool_sealed_v1_freeze.json).

Stages:
  audit      extra zero-overlap audit vs eval_set_10k (builder covers the rest)
  build-ik   cone-IK pool (K=32) for the sealed tasks
  roll       all valid IK candidates under C0, sharded + resumable
  final      published selector (from E4 JSON) -> one static pick per task,
             paired comparison vs the old-system sealed arm, full robust stats
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
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate
from Yuan.unified_rl.ikpool_build_full import (
    C0_DIR, CONE_DEG, N_ORI, N_RESTART, JOINT_MARGIN, DEDUP_RAD, K_POOL,
    ROLL_CHUNK, _fps_select)
from Yuan.unified_rl.ikpool_bidir import SetSel, _picks, _paired

S = Path('Yuan/unified_rl/runs/ikpool_sealed_v1')
IK_GEN_SEED = 2026072414
SYSTEMATIC_10K = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'


def _sealed_cands():
    return CachedSeedCandidateDataset.from_npz(S / 'candidates_K8.npz')


def stage_audit(args, device):
    c = np.load(S / 'candidates_K8.npz', allow_pickle=True)
    sig = np.concatenate([c['p0'], c['line_dir'], c['n_target']], 1).astype(np.float32)
    e = np.load(SYSTEMATIC_10K, allow_pickle=True)
    if 'p0' in e.files:
        esig = np.concatenate([e['p0'], e['line_dir'], e['n_target']], 1).astype(np.float32)
    else:  # systematic-10k schema uses the cs_ prefix
        esig = np.concatenate(
            [e['cs_p0'], e['cs_line_dir'], e['cs_n_target']], 1).astype(np.float32)
    a = {tuple(r) for r in sig.tolist()}
    b = {tuple(r) for r in esig.tolist()}
    inter = len(a & b)
    print(f'[audit] sealed n={len(sig)} unique={len(a)}; overlap vs systematic-10k = {inter}')
    if inter:
        raise SystemExit('OVERLAP DETECTED - sealed set invalid')
    (S / 'extra_audit.json').write_text(json.dumps(
        {'systematic_10k_overlap': 0, 'n_unique_geometries': len(a)}, indent=1))


def stage_build_ik(args, device):
    out = S / 'ik_candidates.npz'
    if out.exists():
        print('[build-ik] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    ds = _sealed_cands()
    b = ds.batch
    fb = ds.fallback_index
    m = len(ds)
    seeds = np.full((m, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, K_POOL), bool)
    nsol = np.zeros(m, np.int64)
    for i in range(m):
        rng = np.random.default_rng(IK_GEN_SEED * 100000 + i)
        q = cone_constrained_ik_enumerate(
            p0=b.p0[i], n_target=b.n_target[i], line_dir=b.line_dir[i],
            kin=env.kin, collision=env.collision, cone_angle_deg=CONE_DEG,
            n_orientations=N_ORI, n_ik_restarts=N_RESTART,
            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, rng=rng)
        nsol[i] = q.shape[0]
        q = _fps_select(q, K_POOL)
        if q.shape[0]:
            seeds[i, :q.shape[0]] = q.cpu().numpy().astype(np.float32)
            ik_ok[i, :q.shape[0]] = True
        if (i + 1) % 500 == 0:
            print(f'[build-ik] {i+1}/{m} median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=b.p0.numpy(), line_dir=b.line_dir.numpy(),
             n_target=b.n_target.numpy(), q0_pilot=b.q0[:, fb].numpy(),
             task_indices=np.arange(m, dtype=np.int64), n_solutions_raw=nsol)
    print(f'[build-ik] done median_sol={int(np.median(nsol))} empty={(nsol==0).sum()}', flush=True)


def stage_roll(args, device):
    i_sh, n_sh = args.shard
    out = S / f'ik_returns_shard{i_sh}of{n_sh}.npz'
    if out.exists():
        print(f'[roll] shard {i_sh} exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(S / 'ik_candidates.npz')
    rows = np.array_split(np.arange(len(ds)), n_sh)[i_sh]
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    sub = ds.batch.index_select(torch.arange(lo, hi))
    val = check_candidate_validity(env.kin, env.collision,
                                   sub.to(env.kin.device, dtype=env.kin.dtype),
                                   cone_deg=env.cfg.cone_deg).valid.cpu()
    R, K = hi - lo, ds.batch.n_candidates
    prog = np.full((R, K), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for st in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[st:st + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0] + lo).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
        if (st // ROLL_CHUNK) % 25 == 0:
            print(f'[roll {i_sh}/{n_sh}] {min(st+ROLL_CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(out, progress_m=prog, valid=val.numpy(),
             task_indices=np.arange(lo, hi, dtype=np.int64))
    print(f'[roll] shard {i_sh} done', flush=True)


def stage_final(args, device):
    files = sorted(S.glob('ik_returns_shard*.npz'),
                   key=lambda p: int(p.stem.split('shard')[1].split('of')[0]))
    P = np.concatenate([np.load(f)['progress_m'] for f in files], 0)
    V = np.concatenate([np.load(f)['valid'] for f in files], 0)
    np.savez(S / 'ik_returns.npz', progress_m=P, valid=V,
             task_indices=np.arange(len(P), dtype=np.int64))
    e4 = json.loads(Path('Yuan/unified_rl/runs/_multiseed_final/ikpool_e4.json').read_text())
    pub = e4['published_run_seed']
    sel_path = (Path('Yuan/unified_rl/runs/ikpool_full_v1/ikpool_selector_s0.pt')
                if pub == '0' else
                Path(f'Yuan/unified_rl/runs/_multiseed_final/sel_run{pub}.pt'))
    ck = torch.load(sel_path, map_location=device, weights_only=False)
    nets = []
    for st in ck['members']:
        n = SetSel().to(device); n.load_state_dict(st); n.eval(); nets.append(n)
    mu, sd = ck['mu'].to(device), ck['sd'].to(device)
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    ds = CachedSeedCandidateDataset.from_npz(S / 'ik_candidates.npz')
    X = _build_features(env.kin, ds, 4096).to(device)
    Pt = torch.nan_to_num(torch.as_tensor(P), nan=0.0).to(device)
    Vt = torch.as_tensor(V).to(device)
    pick = _picks(nets, mu, sd, X, Vt)
    idx = torch.arange(len(Pt), device=device)
    new = Pt[idx, pick].cpu().numpy()
    first = Pt[idx, Vt.float().argmax(1)].cpu().numpy()
    ora = torch.where(Vt, Pt, torch.tensor(-1e9, device=device)).max(1).values.cpu().numpy()
    old = np.load(S / 'eval_old_system.npz', allow_pickle=True)
    old_pol = np.nan_to_num(old['policy_progress_m'])
    old_first = np.nan_to_num(old['first_valid_progress_m'])
    old_ora = np.nan_to_num(old['best_progress_m'])
    rep = {
        'published_run_seed': pub, 'n_tasks': int(len(new)),
        'means_m': {'new': float(new.mean()), 'old': float(old_pol.mean()),
                    'ik_first': float(first.mean()), 'ik_oracle': float(ora.mean()),
                    'diffusion_first': float(old_first.mean()),
                    'diffusion_oracle': float(old_ora.mean())},
        'new_vs_old': _paired(new, old_pol),
        'new_vs_diffusion_oracle': _paired(new, old_ora),
        'capture_pct': float((new - first).sum() / (ora - first).sum() * 100),
        'old_capture_pct': float((old_pol - old_first).sum()
                                 / (old_ora - old_first).sum() * 100),
    }
    (S / 'ikpool_sealed_final.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('audit', 'build-ik', 'roll', 'final'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    i, n = args.shard.split('/')
    args.shard = (int(i), int(n))
    globals()[f'stage_{args.stage.replace("-", "_")}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
