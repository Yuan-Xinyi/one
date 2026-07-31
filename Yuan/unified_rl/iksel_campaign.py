"""Final campaign on the faithful-IKSel candidate layer (prereg: freeze v3).

Training pool = 20,000 tasks (18,432 historical + 1,568 fresh top-up).
Candidate layer = iksel_clean_pilot v2 (CVT-sampled 201,600-entry table,
32 cone directions, 6D rerank, reshuffle retries). Controller = HYBRID.

Stages:
  gen             --source train|topup|validation|external|sealed
  merge-train     concat train+topup candidate files -> 20k pool
  relabel         --source S --shard i/n   (hybrid, all valid candidates)
  merge-labels    --source S
  train-selector  --run-seed N   (on the 20k label table)
  train-enum-sel  enumeration-pool selector on its hybrid labels (fair arm)
  eval-dev        three-arm showdown on validation+external
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenHybridController, rollout_selected_seeds)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.ikpool_bidir import SetSel, _picks, _paired
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.seed_selection.smm.cone_ik import (
    _build_R_with_z, _dedup_q, _sample_in_cone)
from Yuan.unified_rl.iksel_clean_pilot import (
    _table_path, _minimal_rotvec, POS_SCALE, DEDUP_RAD, N_DIRS, CONE_DEG)

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
F = Path('Yuan/unified_rl/runs/ikpool_final')
G = Path(os.environ.get('IKSEL_DIR', 'Yuan/unified_rl/runs/iksel_final'))
CLEAN = Path('Yuan/unified_rl/runs/iksel_clean_v1')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
TAU_ENTER, TAU_EXIT = 0.985, 0.96
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '256'))
MEMBERS, EPOCHS, TEMP, WD = 5, 300, 0.1, 1e-4
GEN_SEED = 20260731
OLD = {'validation': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
       'external': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'}


def _geoms(source):
    paths = {
        'train': D / 'ikpool_candidates.npz',
        'topup': G / 'topup_tasks/candidates_K8.npz',
        'validation': D / 'ikpool_validation_candidates.npz',
        'external': D / 'ikpool_external_candidates.npz',
        'sealed': Path('Yuan/unified_rl/runs/iksel_sealed_v3/candidates_K8.npz'),
    }
    c = np.load(paths[source], allow_pickle=True)
    m = len(c['p0'])
    tids = c['task_indices'] if 'task_indices' in c.files else np.arange(m)
    return c['p0'], c['line_dir'], c['n_target'], c['q0_pilot'], np.asarray(tids, np.int64)


def _cand(source):
    return G / f'iksel_{source}_candidates.npz'


def _labels(source):
    return G / f'iksel_{source}_returns_hybrid.npz'


def stage_gen(args, device):
    source = args.source
    out = _cand(source)
    if out.exists():
        print(f'[gen {source}] exists, skip'); return
    if (source == 'validation' and N_DIRS == 32
            and (CLEAN / 'clean_validation_candidates_201600.npz').exists()):
        import shutil
        G.mkdir(parents=True, exist_ok=True)
        shutil.copy(CLEAN / 'clean_validation_candidates_201600.npz', out)
        print(f'[gen {source}] reused pilot artifact'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    T = np.load(_table_path(201600))
    feat = np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1).astype(np.float32)
    tree = cKDTree(feat)
    p0a, lda, nta, fbq, tids = _geoms(source)
    m = len(tids)
    MAX_TRY = 4
    seeds = np.full((m, N_DIRS, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, N_DIRS), bool)
    nsol = np.zeros(m, np.int64)
    solved = attempted = 0
    for i in range(m):
        p0, nt, ld = p0a[i], nta[i], lda[i]
        src_off = {'train': 0, 'topup': 1, 'validation': 2,
                   'external': 3, 'sealed': 4}[source] * 10_000_019
        rng = np.random.default_rng(GEN_SEED * 1000 + int(tids[i]) + src_off)
        dirs = _sample_in_cone(torch.as_tensor(nt, dtype=torch.float32),
                               CONE_DEG, N_DIRS - 1, rng)
        dirs = torch.cat([torch.as_tensor(nt, dtype=torch.float32).unsqueeze(0),
                          dirs]).numpy()
        qf = np.concatenate(
            [np.repeat(p0[None, :] * POS_SCALE, N_DIRS, 0), dirs], 1).astype(np.float32)
        _, idsK = tree.query(qf, k=200, workers=-1)
        dp = p0[None, None, :] - T['pos'][idsK]
        rv = _minimal_rotvec(T['zax'][idsK].reshape(-1, 3),
                             np.repeat(dirs, 200, 0)).reshape(N_DIRS, 200, 3)
        d6 = np.concatenate([dp, rv], -1)
        dq = np.einsum('dkje,dke->dkj', T['jinv6'][idsK], d6)
        order = (dq * dq).sum(-1).argsort(1)[:, :20]
        cand_q = T['q'][np.take_along_axis(idsK, order, 1)]
        pend = np.arange(N_DIRS)
        tried_sum = np.zeros((N_DIRS, 7), np.float32)
        n_tried = np.zeros(N_DIRS, np.int64)
        got = np.full((N_DIRS, 7), np.nan, np.float32)
        have = np.zeros(N_DIRS, bool)
        for _ in range(MAX_TRY):
            if not len(pend):
                break
            seed_q = cand_q[pend, 0]
            Rt = _build_R_with_z(
                torch.as_tensor(dirs[pend], device=device, dtype=env.kin.dtype),
                torch.as_tensor(ld, device=device, dtype=env.kin.dtype))
            q0 = torch.as_tensor(seed_q, device=device, dtype=env.kin.dtype)
            p0r = torch.as_tensor(p0, device=device, dtype=env.kin.dtype
                                  ).unsqueeze(0).expand(len(pend), 3)
            q_out, okc, _ = _batched_ik_project(env.kin, q0, p0r, Rt, branch_action=None)
            okn = okc.cpu().numpy()
            got[pend[okn]] = q_out[okc].cpu().numpy(); have[pend[okn]] = True
            fail = pend[~okn]
            tried_sum[fail] += seed_q[~okn]; n_tried[fail] += 1
            for fidx in fail:
                rest = cand_q[fidx, 1:]
                if len(rest):
                    dist = np.linalg.norm(
                        n_tried[fidx] * rest - tried_sum[fidx][None, :], axis=1)
                    cand_q[fidx, 1:] = rest[np.argsort(-dist)]
            cand_q = np.roll(cand_q, -1, axis=1)
            pend = fail
        attempted += N_DIRS; solved += int(have.sum())
        q_ok = torch.as_tensor(got[have], device=device, dtype=env.kin.dtype)
        if q_ok.shape[0]:
            kept = _dedup_q(q_ok, DEDUP_RAD)
            c = min(kept.shape[0], N_DIRS)
            seeds[i, :c] = kept[:c].cpu().numpy(); ik_ok[i, :c] = True
            nsol[i] = c
        if (i + 1) % 1000 == 0:
            print(f'[gen {source}] {i+1}/{m} solve={solved/attempted*100:.1f}% '
                  f'median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    G.mkdir(parents=True, exist_ok=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=p0a, line_dir=lda, n_target=nta,
             q0_pilot=fbq, task_indices=tids)
    print(f'[gen {source}] done solve={solved/attempted*100:.1f}% '
          f'median_sol={int(np.median(nsol))} empty={(nsol==0).sum()}', flush=True)


def stage_merge_train(args, device):
    out = _cand('train20k')
    if out.exists():
        print('[merge-train] exists, skip'); return
    a = np.load(_cand('train')); b = np.load(_cand('topup'))
    merged = {}
    for k in ('seeds', 'ik_ok', 'p0', 'line_dir', 'n_target', 'q0_pilot'):
        merged[k] = np.concatenate([a[k], b[k]], 0)
    n = len(merged['p0'])
    merged['task_indices'] = np.arange(n, dtype=np.int64)
    np.savez(out, **merged)
    print(f'[merge-train] {len(a["p0"])} + {len(b["p0"])} -> {n} tasks', flush=True)


def stage_relabel(args, device):
    source = args.source
    i_sh, n_sh = args.shard
    out = G / f'iksel_{source}_returns_hybrid_shard{i_sh}of{n_sh}.npz'
    if out.exists():
        print(f'[relabel] {out.name} exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    ctl = FrozenHybridController(agent, ClassicalNullspaceController(env.kin),
                                 TAU_ENTER, TAU_EXIT)
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(_cand(source))
    rows = np.array_split(np.arange(len(ds)), n_sh)[i_sh]
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    parts = []
    for s in range(lo, hi, 256):
        sub = ds.batch.index_select(torch.arange(s, min(s + 256, hi)))
        parts.append(check_candidate_validity(
            env.kin, env.collision, sub.to(env.kin.device, dtype=env.kin.dtype),
            cone_deg=env.cfg.cone_deg).valid.cpu())
    val = torch.cat(parts)
    Rn, K = hi - lo, ds.batch.n_candidates
    prog = np.full((Rn, K), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0] + lo).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
        if (s // ROLL_CHUNK) % 25 == 0:
            print(f'[relabel {source} {i_sh}/{n_sh}] '
                  f'{min(s+ROLL_CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(out, progress_m=prog, valid=val.numpy(),
             task_indices=ds.task_indices.numpy()[lo:hi])
    print(f'[relabel] done -> {out.name}', flush=True)


def stage_merge_labels(args, device):
    source = args.source
    files = sorted(G.glob(f'iksel_{source}_returns_hybrid_shard*.npz'),
                   key=lambda p: int(p.stem.split('shard')[1].split('of')[0]))
    data = {k: [] for k in ('progress_m', 'valid', 'task_indices')}
    for f in files:
        d = np.load(f)
        for k in data:
            data[k].append(d[k])
    merged = {k: np.concatenate(v, 0) for k, v in data.items()}
    np.savez(_labels(source), **merged)
    print(f'[merge-labels] {len(files)} shards -> {_labels(source).name} '
          f'n={len(merged["task_indices"])}', flush=True)


def _train_sel(X, P, V, run_seed, out, device):
    hub = nn.HuberLoss(delta=0.05, reduction='none')
    mu, sd = X[V].mean(0), X[V].std(0).clamp_min(1e-6)
    Xz_all = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    nets = []
    for mth in range(MEMBERS):
        seed = run_seed + 1000 * (mth + 1)
        g = torch.Generator().manual_seed(seed)
        boot = torch.randint(0, len(X), (len(X),), generator=g).to(device)
        Xz, Vb, Pb = Xz_all[boot], V[boot], P[boot]
        lo = torch.where(Vb, Pb, torch.tensor(1e9, device=device)).min(1, keepdim=True).values
        hi = torch.where(Vb, Pb, torch.tensor(-1e9, device=device)).max(1, keepdim=True).values
        Tt = ((Pb - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~Vb, 0.0)
        torch.manual_seed(seed)
        net = SetSel().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
        for _ in range(EPOCHS):
            opt.zero_grad()
            s, f = net(Xz, Vb)
            s = s.masked_fill(~Vb, -1e9)
            tgt = torch.softmax((Tt / TEMP).masked_fill(~Vb, -1e9), 1)
            rank = -(tgt * torch.log_softmax(s, 1).clamp_min(-30)).sum(1).mean()
            feas = (hub(f, Pb) * Vb.float()).sum() / Vb.float().sum()
            (rank + feas).backward(); opt.step()
        nets.append(net)
        print(f'[sel] member {mth+1}/{MEMBERS}', flush=True)
    torch.save({'members': [n.state_dict() for n in nets],
                'mu': mu.cpu(), 'sd': sd.cpu()}, out)
    print(f'[sel] saved {out.name}', flush=True)


def _load_pool_env(cand_path, labels_path, device):
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    ds = CachedSeedCandidateDataset.from_npz(cand_path)
    r = np.load(labels_path)
    X = _build_features(env.kin, ds, 4096).to(device)
    P = torch.nan_to_num(torch.as_tensor(r['progress_m']), nan=0.0).to(device)
    V = torch.as_tensor(r['valid']).to(device)
    return X, P, V


def stage_train_selector(args, device):
    out = G / f'sel_iksel_run{args.run_seed}.pt'
    if out.exists():
        print(f'[sel] {out.name} exists, skip'); return
    X, P, V = _load_pool_env(_cand('train20k'), _labels('train20k'), device)
    _train_sel(X, P, V, args.run_seed, out, device)


def stage_train_enum_sel(args, device):
    out = G / 'sel_enum_run0.pt'
    if out.exists():
        print('[sel-enum] exists, skip'); return
    X, P, V = _load_pool_env(D / 'ikpool_candidates.npz',
                             F / 'enumeration_train_returns_hybrid.npz', device)
    _train_sel(X, P, V, 0, out, device)


def _load_sel(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    nets = []
    for st in ck['members']:
        n = SetSel().to(device); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, ck['mu'].to(device), ck['sd'].to(device)


def stage_eval_dev(args, device):
    rep = {}
    sel_ik = _load_sel(G / 'sel_iksel_run0.pt', device)
    sel_en = _load_sel(G / 'sel_enum_run0.pt', device)
    for which in ('validation', 'external'):
        arms = {}
        X, P, V = _load_pool_env(_cand(which), _labels(which), device)
        pick = _picks(*sel_ik, X, V)
        i = torch.arange(len(P), device=device)
        arms['iksel'] = P[i, pick].cpu().numpy()
        first = P[i, V.float().argmax(1)].cpu().numpy()
        ora = torch.where(V, P, torch.tensor(-1e9, device=device)).max(1).values.cpu().numpy()
        Xe, Pe, Ve = _load_pool_env(
            D / f'ikpool_{which}_candidates.npz',
            F / f'enumeration_{which}_returns_hybrid.npz', device)
        picke = _picks(*sel_en, Xe, Ve)
        arms['enumeration_ablation'] = Pe[i, picke].cpu().numpy()
        rep[which] = {
            'means_m': {k: float(v.mean()) for k, v in arms.items()},
            'iksel_first_valid_m': float(first.mean()),
            'iksel_oracle_m': float(ora.mean()),
            'iksel_capture_pct': float((arms['iksel'] - first).sum()
                                       / (ora - first).sum() * 100),
            'iksel_vs_enumeration_ablation': _paired(
                arms['iksel'], arms['enumeration_ablation']),
        }
        r = rep[which]
        print(f"[dev {which}] iksel={r['means_m']['iksel']:.4f} "
              f"enum_abl={r['means_m']['enumeration_ablation']:.4f} "
              f"I-E {r['iksel_vs_enumeration_ablation']['delta_mm']:+.1f} "
              f"cap {r['iksel_capture_pct']:.1f}%", flush=True)
    (G / 'eval_dev.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=(
        'gen', 'merge-train', 'relabel', 'merge-labels',
        'train-selector', 'train-enum-sel', 'eval-dev'))
    ap.add_argument('--source', default='train',
                    choices=('train', 'topup', 'train20k', 'validation',
                             'external', 'sealed'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--run-seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    i, n = args.shard.split('/')
    args.shard = (int(i), int(n))
    globals()[f'stage_{args.stage.replace("-", "_")}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
