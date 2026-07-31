"""Final campaign: retrieval candidates + return-supervised selector + HYBRID.

Preregistration: runs/ikpool_final_freeze_v2.json. Win criterion: beat the
enumeration+SetSel arm under the identical hybrid controller on both dev sets
(fallback: union pool), and beat the diffusion baseline significantly.

Stages (all resumable; artifacts under runs/ikpool_final/):
  gen            --set train|validation|external|sealed   retrieval pools
                 (train queries exclude same-task donors)
  relabel        --set S --pool retrieval|enumeration --shard i/n
                 all-candidate rollouts under HYBRID
  merge          --set S --pool P
  train-selector --pool P --run-seed N
  eval-dev       three-arm paired comparison on validation+external
"""
import argparse, json, math, os
from pathlib import Path
import numpy as np
import torch
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
from Yuan.unified_rl.ikpool_retrieval_pilot import (
    _clamp_to_cone, POS_SCALE, Z_SCALE, DIR_SCALE, CONE_LIM,
    K_QUERY, K_RERANK, K_POOL, DEDUP_RAD)
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z, _dedup_q
from Yuan.unified_rl.ikpool_build_full import _fps_select

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
RP = Path('Yuan/unified_rl/runs/ikpool_retrieval_pilot')
F = Path('Yuan/unified_rl/runs/ikpool_final')
SEALED2 = Path('Yuan/unified_rl/runs/ikpool_sealed_v2')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
TAU_ENTER, TAU_EXIT = 0.985, 0.96
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '256'))
MEMBERS, EPOCHS, TEMP, WD = 5, 300, 0.1, 1e-4
OLD = {'validation': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
       'external': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'}


def _geom_source(which):
    if which == 'train':
        c = np.load(D / 'ikpool_candidates.npz')
    elif which == 'sealed':
        c = np.load(SEALED2 / 'candidates_K8.npz', allow_pickle=True)
        m = len(c['p0'])
        return c['p0'], c['line_dir'], c['n_target'], c['q0_pilot'], np.arange(m, dtype=np.int64)
    else:
        c = np.load(D / f'ikpool_{which}_candidates.npz')
    return c['p0'], c['line_dir'], c['n_target'], c['q0_pilot'], c['task_indices']


def _cand_path(which, pool):
    if pool == 'retrieval':
        return F / f'retr_{which}_candidates.npz'
    return (D / 'ikpool_candidates.npz' if which == 'train'
            else D / f'ikpool_{which}_candidates.npz' if which != 'sealed'
            else SEALED2 / 'ik_candidates.npz')


def _ret_path(which, pool):
    return F / f'{pool}_{which}_returns_hybrid.npz'


def stage_gen(args, device):
    which = args.set
    out = _cand_path(which, 'retrieval')
    if out.exists():
        print(f'[gen {which}] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    T = np.load(RP / 'table.npz')
    tree = cKDTree(T['feat'])
    p0a, lda, nta, fbq, tids = _geom_source(which)
    m = len(tids)
    self_ex = which == 'train'
    seeds = np.full((m, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, K_POOL), bool)
    nsol = np.zeros(m, np.int64)
    kq = K_QUERY + (64 if self_ex else 0)
    for i in range(m):
        p0, nt, ld = p0a[i], nta[i], lda[i]
        qf = np.concatenate([p0 * POS_SCALE, nt * Z_SCALE, ld * DIR_SCALE]).astype(np.float32)
        _, ids = tree.query(qf, k=kq)
        if self_ex:
            ids = ids[T['donor_task'][ids] != int(tids[i])][:K_QUERY]
        dp = p0[None, :] - T['pos'][ids]
        dq = np.einsum('ijk,ik->ij', T['jinv'][ids], dp)
        ang = np.arccos((T['zax'][ids] * nt[None, :]).sum(-1).clip(-1, 1))
        score = (dq * dq).sum(1) + np.maximum(ang - CONE_LIM, 0.0) ** 2 * 4.0
        keep = ids[np.argsort(score)[:K_RERANK]]
        z_t = _clamp_to_cone(T['zax'][keep], nt[None, :].repeat(len(keep), 0))
        Rt = _build_R_with_z(torch.as_tensor(z_t, device=device, dtype=env.kin.dtype),
                             torch.as_tensor(ld, device=device, dtype=env.kin.dtype))
        q0 = torch.as_tensor(T['q'][keep], device=device, dtype=env.kin.dtype)
        p0r = torch.as_tensor(p0, device=device, dtype=env.kin.dtype
                              ).unsqueeze(0).expand(len(keep), 3)
        q_out, ok, _ = _batched_ik_project(env.kin, q0, p0r, Rt, branch_action=None)
        q_ok = q_out[ok]
        if q_ok.shape[0]:
            kept = _fps_select(_dedup_q(q_ok, DEDUP_RAD), K_POOL)
            c = kept.shape[0]
            seeds[i, :c] = kept.cpu().numpy(); ik_ok[i, :c] = True
            nsol[i] = c
        if (i + 1) % 500 == 0:
            print(f'[gen {which}] {i+1}/{m} median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    F.mkdir(parents=True, exist_ok=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=p0a, line_dir=lda, n_target=nta,
             q0_pilot=fbq, task_indices=np.asarray(tids, dtype=np.int64))
    print(f'[gen {which}] done median_sol={int(np.median(nsol))} empty={(nsol==0).sum()}', flush=True)


def _hybrid_controller(env, device):
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    return FrozenHybridController(agent, ClassicalNullspaceController(env.kin),
                                  TAU_ENTER, TAU_EXIT)


def _chunked_valid(env, ds):
    parts = []
    for s in range(0, len(ds), 256):
        sub = ds.batch.index_select(torch.arange(s, min(s + 256, len(ds))))
        parts.append(check_candidate_validity(
            env.kin, env.collision, sub.to(env.kin.device, dtype=env.kin.dtype),
            cone_deg=env.cfg.cone_deg).valid.cpu())
    return torch.cat(parts)


def stage_relabel(args, device):
    which, pool = args.set, args.pool
    i_sh, n_sh = args.shard
    out = F / f'{pool}_{which}_returns_hybrid_shard{i_sh}of{n_sh}.npz'
    if out.exists():
        print(f'[relabel] {out.name} exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    ctl = _hybrid_controller(env, device)
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(_cand_path(which, pool))
    rows = np.array_split(np.arange(len(ds)), n_sh)[i_sh]
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    sub_ds = ds  # index arithmetic below is global
    val_parts = []
    for s in range(lo, hi, 256):
        sub = ds.batch.index_select(torch.arange(s, min(s + 256, hi)))
        val_parts.append(check_candidate_validity(
            env.kin, env.collision, sub.to(env.kin.device, dtype=env.kin.dtype),
            cone_deg=env.cfg.cone_deg).valid.cpu())
    val = torch.cat(val_parts)
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
            print(f'[relabel {pool} {which} {i_sh}/{n_sh}] '
                  f'{min(s+ROLL_CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(out, progress_m=prog, valid=val.numpy(),
             task_indices=ds.task_indices.numpy()[lo:hi])
    print(f'[relabel] done -> {out.name}', flush=True)


def stage_merge(args, device):
    which, pool = args.set, args.pool
    files = sorted(F.glob(f'{pool}_{which}_returns_hybrid_shard*.npz'),
                   key=lambda p: int(p.stem.split('shard')[1].split('of')[0]))
    data = {k: [] for k in ('progress_m', 'valid', 'task_indices')}
    for f in files:
        d = np.load(f)
        for k in data:
            data[k].append(d[k])
    merged = {k: np.concatenate(v, 0) for k, v in data.items()}
    np.savez(_ret_path(which, pool), **merged)
    print(f'[merge] {len(files)} shards -> {_ret_path(which, pool).name} '
          f'n={len(merged["task_indices"])}', flush=True)


def _load_train(pool, device):
    ds = CachedSeedCandidateDataset.from_npz(_cand_path('train', pool))
    ret = np.load(_ret_path('train', pool))
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    X = _build_features(env.kin, ds, 4096).to(device)
    P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(device)
    V = torch.as_tensor(ret['valid']).to(device)
    return X, P, V


def stage_train_selector(args, device):
    pool, rs = args.pool, args.run_seed
    out = F / f'sel_{pool}_run{rs}.pt'
    if out.exists():
        print(f'[train] {out.name} exists, skip'); return
    import torch.nn as nn
    X, P, V = _load_train(pool, device)
    hub = nn.HuberLoss(delta=0.05, reduction='none')
    mu, sd = X[V].mean(0), X[V].std(0).clamp_min(1e-6)
    Xz_all = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    nets = []
    for mth in range(MEMBERS):
        seed = rs + 1000 * (mth + 1)
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
        print(f'[train {pool} run{rs}] member {mth+1}/{MEMBERS}', flush=True)
    torch.save({'members': [n.state_dict() for n in nets],
                'mu': mu.cpu(), 'sd': sd.cpu()}, out)
    print(f'[train] saved {out.name}', flush=True)


def _load_sel(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    nets = []
    for st in ck['members']:
        n = SetSel().to(device); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, ck['mu'].to(device), ck['sd'].to(device)


def stage_eval_dev(args, device):
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    rep = {}
    for which in ('validation', 'external'):
        arms = {}
        pools = {}
        for pool in ('retrieval', 'enumeration'):
            ds = CachedSeedCandidateDataset.from_npz(_cand_path(which, pool))
            ret = np.load(_ret_path(which, pool))
            X = _build_features(env1.kin, ds, 4096).to(device)
            P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(device)
            V = torch.as_tensor(ret['valid']).to(device)
            sel = _load_sel(F / f'sel_{pool}_run0.pt', device)
            pick = _picks(*sel, X, V)
            idx = torch.arange(len(P), device=device)
            arms[pool] = P[idx, pick].cpu().numpy()
            first = P[idx, V.float().argmax(1)].cpu().numpy()
            ora = torch.where(V, P, torch.tensor(-1e9, device=device)).max(1).values.cpu().numpy()
            pools[pool] = {'first_m': float(first.mean()), 'oracle_m': float(ora.mean()),
                           'capture_pct': float((arms[pool] - first).sum()
                                                / (ora - first).sum() * 100)}
        # diffusion baseline arm: old ranker picks rolled under hybrid
        dif = np.load(F / f'diffusion_{which}_arm_hybrid.npz')
        arms['diffusion'] = np.nan_to_num(dif['policy_progress_m'])
        rep[which] = {
            'means_m': {k: float(v.mean()) for k, v in arms.items()},
            'pools': pools,
            'retrieval_vs_enumeration': _paired(arms['retrieval'], arms['enumeration']),
            'retrieval_vs_diffusion': _paired(arms['retrieval'], arms['diffusion']),
            'enumeration_vs_diffusion': _paired(arms['enumeration'], arms['diffusion']),
        }
        r = rep[which]
        print(f"[dev {which}] retr={r['means_m']['retrieval']:.4f} "
              f"enum={r['means_m']['enumeration']:.4f} dif={r['means_m']['diffusion']:.4f} "
              f"R-E {r['retrieval_vs_enumeration']['delta_mm']:+.1f}mm "
              f"CI{r['retrieval_vs_enumeration']['ci95_mm']}", flush=True)
    (F / 'eval_dev.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('gen', 'relabel', 'merge', 'train-selector', 'eval-dev'))
    ap.add_argument('--set', default='train',
                    choices=('train', 'validation', 'external', 'sealed'))
    ap.add_argument('--pool', default='retrieval', choices=('retrieval', 'enumeration'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--run-seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    i, n = args.shard.split('/')
    args.shard = (int(i), int(n))
    globals()[f'stage_{args.stage.replace("-", "_")}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
