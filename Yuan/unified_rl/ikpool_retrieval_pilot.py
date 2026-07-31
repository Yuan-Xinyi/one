"""Retrieval-based candidate generation pilot (IKSel-style, label-aware).

Table  : all validated E1 train-pool candidates (~570k configs) with their
         FK pose features and measured C0 progress (donor labels).
Query  : validation tasks (geometry-disjoint from the table by construction).
Method : KD-tree over (tcp position, tool z-axis, task line_dir) features ->
         Jacobian-metric rerank (IKSel) -> cone-clamped Newton refinement ->
         strict physical validation -> dedup + FPS to K=32.
Judged : refinement success, branch coverage, retrieval-pool oracle vs the
         enumeration-pool oracle (cached), S0 transfer, and the within-task
         rank signal of donor progress (the label-augmentation hypothesis).

Stages: table | gen | roll | analyze   (sequential driver runs all)
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenRLController, rollout_selected_seeds
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.ikpool_bidir import SetSel, _picks
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z, _dedup_q
from Yuan.unified_rl.ikpool_build_full import _fps_select

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
R = Path('Yuan/unified_rl/runs/ikpool_retrieval_pilot')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
# feature scaling: 5 cm position error ~ 1 rad of tool-axis mismatch
POS_SCALE, Z_SCALE, DIR_SCALE = 1.0 / 0.05, 1.0, 0.5
CONE_LIM = math.radians(29.5)
K_QUERY, K_RERANK, K_POOL = 256, 48, 32
import os
DEDUP_RAD = 0.08
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '512'))


def stage_table(args, device):
    out = R / 'table.npz'
    if out.exists():
        print('[table] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    cand = np.load(D / 'ikpool_candidates.npz')
    ret = np.load(D / 'ikpool_returns.npz')
    V = ret['valid'][:, :32]                       # IK slots only (skip fallback)
    q = cand['seeds'][V]                           # (N,7) valid configs
    prog = np.nan_to_num(ret['progress_m'][:, :32])[V]
    task_of = np.repeat(np.arange(V.shape[0]), V.sum(1))
    ldir = cand['line_dir'][task_of]
    n = len(q)
    print(f'[table] {n} valid configs from {V.shape[0]} train tasks', flush=True)
    pos = np.empty((n, 3), np.float32); zax = np.empty((n, 3), np.float32)
    jinv = np.empty((n, 7, 3), np.float32)
    for s in range(0, n, 65536):
        e = min(s + 65536, n)
        qt = torch.as_tensor(q[s:e], device=device, dtype=env.kin.dtype)
        p, rot, jac, _ = env.kin.tcp_fk_jac(qt)
        pos[s:e] = p.cpu().numpy(); zax[s:e] = rot[:, :, 2].cpu().numpy()
        jinv[s:e] = torch.linalg.pinv(jac[:, :3, :], rtol=1e-4).cpu().numpy()
        print(f'[table] fk {e}/{n}', flush=True)
    feat = np.concatenate([pos * POS_SCALE, zax * Z_SCALE, ldir * DIR_SCALE], 1)
    R.mkdir(parents=True, exist_ok=True)
    np.savez(out, q=q.astype(np.float32), pos=pos, zax=zax, jinv=jinv,
             feat=feat.astype(np.float32), donor_progress=prog.astype(np.float32),
             donor_task=task_of.astype(np.int64))
    print(f'[table] saved: {n} entries', flush=True)


def _clamp_to_cone(z, n_tgt, lim=CONE_LIM):
    """Slerp each z toward n_tgt so that angle(z', n_tgt) <= lim."""
    cos = (z * n_tgt).sum(-1, keepdims=True).clip(-1, 1)
    ang = np.arccos(cos)
    inside = ang[:, 0] <= lim
    t = np.zeros_like(ang)
    bad = ~inside
    t[bad] = (ang[bad] - lim) / np.maximum(ang[bad], 1e-9)
    s = np.sin(np.maximum(ang, 1e-9))
    z2 = (np.sin((1 - t) * ang) * z + np.sin(t * ang) * n_tgt) / s
    z2 = np.where(inside[:, None], z, z2)
    return z2 / np.linalg.norm(z2, axis=-1, keepdims=True)


def stage_gen(args, device):
    out = R / 'retrieval_candidates.npz'
    if out.exists():
        print('[gen] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    T = np.load(R / 'table.npz')
    tree = cKDTree(T['feat'])
    val = np.load(D / 'ikpool_validation_candidates.npz')
    m = len(val['p0'])
    seeds = np.full((m, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, K_POOL), bool)
    donor = np.full((m, K_POOL), np.nan, np.float32)
    stats = {'refined': 0, 'attempted': 0, 'per_task_solutions': []}
    for i in range(m):
        p0, nt, ld = val['p0'][i], val['n_target'][i], val['line_dir'][i]
        qfeat = np.concatenate([p0 * POS_SCALE, nt * Z_SCALE, ld * DIR_SCALE])
        _, ids = tree.query(qfeat.astype(np.float32), k=K_QUERY)
        # IKSel rerank: linearized joint-space step toward the target position
        dp = p0[None, :] - T['pos'][ids]
        dq = np.einsum('ijk,ik->ij', T['jinv'][ids], dp)
        ang = np.arccos((T['zax'][ids] * nt[None, :]).sum(-1).clip(-1, 1))
        score = (dq * dq).sum(1) + np.maximum(ang - CONE_LIM, 0.0) ** 2 * 4.0
        keep = ids[np.argsort(score)[:K_RERANK]]
        # cone-clamped Newton refinement from each retrieved seed
        z_t = _clamp_to_cone(T['zax'][keep], nt[None, :].repeat(len(keep), 0))
        Rt = _build_R_with_z(torch.as_tensor(z_t, device=device, dtype=env.kin.dtype),
                             torch.as_tensor(ld, device=device, dtype=env.kin.dtype))
        q0 = torch.as_tensor(T['q'][keep], device=device, dtype=env.kin.dtype)
        p0r = torch.as_tensor(p0, device=device, dtype=env.kin.dtype
                              ).unsqueeze(0).expand(len(keep), 3)
        q_out, ok, _ = _batched_ik_project(env.kin, q0, p0r, Rt, branch_action=None)
        okn = ok.cpu().numpy()
        stats['attempted'] += len(keep); stats['refined'] += int(okn.sum())
        q_ok = q_out[ok]
        d_ok = T['donor_progress'][keep][okn]
        if q_ok.shape[0]:
            # joint-space dedup, keeping donor alignment via index match
            kept = _dedup_q(q_ok, DEDUP_RAD)
            # map kept rows back to donor labels (first exact match)
            qn = q_ok.cpu().numpy(); kn = kept.cpu().numpy()
            idx = [int(np.argmin(np.linalg.norm(qn - k[None, :], axis=1)))
                   for k in kn]
            kept32 = _fps_select(kept, K_POOL)
            kn32 = kept32.cpu().numpy()
            idx32 = [idx[int(np.argmin(np.linalg.norm(kn - k[None, :], axis=1)))]
                     for k in kn32]
            c = kn32.shape[0]
            seeds[i, :c] = kn32; ik_ok[i, :c] = True
            donor[i, :c] = d_ok[idx32]
            stats['per_task_solutions'].append(c)
        else:
            stats['per_task_solutions'].append(0)
        if (i + 1) % 200 == 0:
            print(f'[gen] {i+1}/{m} refine_ok={stats["refined"]/max(stats["attempted"],1)*100:.1f}% '
                  f'median_sol={int(np.median(stats["per_task_solutions"]))}', flush=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=val['p0'], line_dir=val['line_dir'],
             n_target=val['n_target'], q0_pilot=val['q0_pilot'],
             task_indices=val['task_indices'], donor_progress=donor)
    sol = np.asarray(stats['per_task_solutions'])
    print(f'[gen] done refine_ok={stats["refined"]/stats["attempted"]*100:.1f}% '
          f'median_sol={int(np.median(sol))} empty={(sol==0).sum()}', flush=True)


def stage_roll(args, device):
    out = R / 'retrieval_returns.npz'
    if out.exists():
        print('[roll] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(R / 'retrieval_candidates.npz')
    val_parts = []
    for s in range(0, len(ds), 256):
        sub = ds.batch.index_select(torch.arange(s, min(s + 256, len(ds))))
        val_parts.append(check_candidate_validity(
            env.kin, env.collision, sub.to(env.kin.device, dtype=env.kin.dtype),
            cone_deg=env.cfg.cone_deg).valid.cpu())
    val = torch.cat(val_parts)
    n, K = val.shape
    prog = np.full((n, K), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0]).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
        if (s // ROLL_CHUNK) % 20 == 0:
            print(f'[roll] {min(s+ROLL_CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(out, progress_m=prog, valid=val.numpy(), task_indices=ds.task_indices.numpy())
    print('[roll] done', flush=True)


def stage_analyze(args, device):
    from scipy.stats import spearmanr
    gen = np.load(R / 'retrieval_candidates.npz')
    ret = np.load(R / 'retrieval_returns.npz')
    enum_ret = np.load(D / 'ikpool_validation_returns.npz')
    P, V = np.nan_to_num(ret['progress_m']), ret['valid']
    Pe, Ve = np.nan_to_num(enum_ret['progress_m']), enum_ret['valid']
    n = len(P); r = np.arange(n)
    o_r = np.where(V, P, -np.inf).max(1)
    o_e = np.where(Ve, Pe, -np.inf).max(1)
    ok = np.isfinite(o_r)
    # donor-label signal on retrieval pool (IK slots only)
    donor = gen['donor_progress']
    rho = []
    for i in range(n):
        v = V[i, :K_POOL] & np.isfinite(donor[i])
        if v.sum() >= 3:
            s = spearmanr(donor[i][v], P[i, :K_POOL][v]).statistic
            if np.isfinite(s):
                rho.append(s)
    # donor-argmax selector (zero learning, pure memory prior)
    d_sel = np.where(V[:, :K_POOL] & np.isfinite(donor), donor, -np.inf).argmax(1)
    d_prog = P[r, d_sel]
    first = P[r, V.argmax(1)]
    # S0 transfer on the retrieval pool
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    ck = torch.load(D / 'ikpool_selector_s0.pt', map_location=device, weights_only=False)
    nets = []
    for st in ck['members']:
        net = SetSel().to(device); net.load_state_dict(st); net.eval(); nets.append(net)
    ds = CachedSeedCandidateDataset.from_npz(R / 'retrieval_candidates.npz')
    X = _build_features(env1.kin, ds, 4096).to(device)
    pick = _picks(nets, ck['mu'].to(device), ck['sd'].to(device), X,
                  torch.as_tensor(V).to(device)).cpu().numpy()
    s0_prog = P[r, pick]
    rep = {
        'n_tasks': int(n), 'tasks_with_pool': int(ok.sum()),
        'valid_per_task': float(V[:, :K_POOL].sum(1).mean()),
        'oracle': {'retrieval_m': float(o_r[ok].mean()),
                   'enumeration_m': float(o_e[ok].mean()),
                   'delta_mm': float((o_r[ok] - o_e[ok]).mean() * 1e3),
                   'union_m': float(np.maximum(o_r, o_e)[ok].mean())},
        'first_valid_m': float(first.mean()),
        'donor_label_signal': {
            'within_task_spearman_median': float(np.median(rho)),
            'donor_argmax_progress_m': float(d_prog.mean()),
            'donor_argmax_capture_pct': float(
                (d_prog - first)[ok].sum() / (o_r - first)[ok].sum() * 100)},
        's0_transfer': {
            'progress_m': float(s0_prog.mean()),
            'capture_pct': float((s0_prog - first)[ok].sum()
                                 / (o_r - first)[ok].sum() * 100),
            'ref_s0_on_enumeration_m': 0.5629},
    }
    (R / 'retrieval_pilot.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('table', 'gen', 'roll', 'analyze', 'all'))
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    device = torch.device(args.device)
    stages = ('table', 'gen', 'roll', 'analyze') if args.stage == 'all' else (args.stage,)
    for s in stages:
        print(f'===== {s} =====', flush=True)
        globals()[f'stage_{s}'](args, device)


if __name__ == '__main__':
    main()
