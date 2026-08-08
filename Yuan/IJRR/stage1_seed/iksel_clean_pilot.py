"""Clean IKSel-CVT candidate layer, v1 (user spec, 2026-07-31).

Table   : IKSel's own CVT construction, fully self-contained -- 200k random
          joint samples -> k-means centroids -> store (q, tcp pos, tool z, J+),
          cKDTree over (pos/0.05, z). No experiment data anywhere.
Per task: sample 32 tool directions inside the 30-deg cone; each direction
          forms a full pose (p0, R with z = direction); query the table and
          take the TOP-1 entry (after IKSel's ||J+ dp|| rerank); 32 seeds
          total -> one DLS refinement each to its own pose -> strict physical
          validation -> 0.08 rad dedup. Downstream (selector / controller)
          unchanged.
Pilot   : validation tasks, pool rolled to termination under the HYBRID
          controller; ceiling compared against the enumeration pool's hybrid
          oracle. Two table sizes: 2048 (IKSel default) and 16384.

Stages: table | gen | roll | analyze   --n-cvt {2048,16384}
"""
import argparse, os
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree

from Yuan.IJRR.stage1_seed.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.stage1_seed.cone_ik import (
    _build_R_with_z, _dedup_q, _sample_in_cone)

T_GEOM = Path('Yuan/IJRR/runs/tasks')
C = Path('Yuan/IJRR/runs/iksel_clean_v1')
C0_DIR = 'Yuan/IJRR/runs/rl_smmstart_30M'
TAU_ENTER, TAU_EXIT = 0.985, 0.96
N_DIRS = int(os.environ.get('N_DIRS', '32'))          # cone directions per task
CONE_DEG = 29.5
POS_SCALE = 1.0 / 0.05
DEDUP_RAD = 0.08
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '256'))
GEN_SEED = 20260731


def _table_path(n_cvt):
    return C / f'cvt_table_{n_cvt}.npz'


def stage_table2(args, device):
    """v2: CVT-SAMPLED table (faithful to wrs SELIKSolver): Lloyd on a 2M pool
    with K=201,600 centroids -> every centroid is a table entry; store full
    6D jacobian pinv for the IKSel rerank."""
    n_cvt = 201600
    out = _table_path(n_cvt)
    if out.exists():
        print('[table2] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    kin = env.kin
    g = torch.Generator(device=device).manual_seed(GEN_SEED + 7)
    lo, hi = kin.lmt_lo, kin.lmt_up
    NPOOL = 2_000_000
    pool = lo + torch.rand(NPOOL, 7, generator=g, device=device,
                           dtype=kin.dtype) * (hi - lo)
    cent = pool[torch.randperm(NPOOL, generator=g, device=device)[:n_cvt]].clone()
    km_chunk = int(os.environ.get('KM_CHUNK', '512'))
    for it in range(8):
        newc = torch.zeros_like(cent)
        cnt = torch.zeros(n_cvt, device=device, dtype=kin.dtype)
        for s in range(0, NPOOL, km_chunk):
            e = min(s + km_chunk, NPOOL)
            a = torch.cdist(pool[s:e], cent).argmin(1)
            newc.index_add_(0, a, pool[s:e])
            cnt.index_add_(0, a, torch.ones(e - s, device=device, dtype=kin.dtype))
        keep = cnt > 0
        cent[keep] = newc[keep] / cnt[keep, None]
        print(f'[table2] lloyd iter {it} occupied={int(keep.sum())}', flush=True)
    q = cent.clamp(lo, hi)
    pos_l, zax_l, jinv_l = [], [], []
    for s in range(0, n_cvt, 16384):
        e = min(s + 16384, n_cvt)
        p, rot, jac, _ = kin.tcp_fk_jac(q[s:e])
        pos_l.append(p.cpu()); zax_l.append(rot[:, :, 2].cpu())
        jinv_l.append(torch.linalg.pinv(jac, rtol=1e-4).cpu())   # full 6D pinv (7x6)
    C.mkdir(parents=True, exist_ok=True)
    np.savez(out, q=q.cpu().numpy().astype(np.float32),
             pos=torch.cat(pos_l).numpy().astype(np.float32),
             zax=torch.cat(zax_l).numpy().astype(np.float32),
             jinv6=torch.cat(jinv_l).numpy().astype(np.float32))
    print('[table2] saved 201600 CVT-sampled entries', flush=True)


def _minimal_rotvec(z_from, z_to):
    """Axis-angle vector of the minimal rotation taking z_from -> z_to. (N,3)"""
    c = np.cross(z_from, z_to)
    s = np.linalg.norm(c, axis=-1, keepdims=True)
    d = (z_from * z_to).sum(-1, keepdims=True).clip(-1, 1)
    ang = np.arctan2(s, d)
    axis = np.where(s > 1e-9, c / np.maximum(s, 1e-9), np.zeros_like(c))
    return axis * ang


def stage_gen2(args, device):
    """v2 generation: k=200 query, full-6D IKSel rerank, failure-reshuffle
    retries (<=4 attempts per direction, faithful to wrs SELIKSolver.ik)."""
    n_cvt = 201600
    out = C / f'clean_validation_candidates_{n_cvt}.npz'
    if out.exists():
        print('[gen2] exists, skip'); return
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
    T = np.load(_table_path(n_cvt))
    feat = np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1).astype(np.float32)
    tree = cKDTree(feat)
    val = np.load(T_GEOM / 'validation_geoms.npz')
    m = len(val['p0'])
    MAX_TRY = 4
    seeds = np.full((m, N_DIRS, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, N_DIRS), bool)
    nsol = np.zeros(m, np.int64)
    solved = attempted_dirs = 0
    for i in range(m):
        p0, nt, ld = val['p0'][i], val['n_target'][i], val['line_dir'][i]
        rng = np.random.default_rng(GEN_SEED * 1000 + i)
        dirs = _sample_in_cone(torch.as_tensor(nt, dtype=torch.float32),
                               CONE_DEG, N_DIRS - 1, rng)
        dirs = torch.cat([torch.as_tensor(nt, dtype=torch.float32).unsqueeze(0),
                          dirs]).numpy()
        qf = np.concatenate(
            [np.repeat(p0[None, :] * POS_SCALE, N_DIRS, 0), dirs], 1).astype(np.float32)
        _, idsK = tree.query(qf, k=200, workers=-1)
        # full-6D IKSel rerank: dq = jinv6 @ (dpos, minimal-rotvec)
        dp = p0[None, None, :] - T['pos'][idsK]                       # (D,K,3)
        rv = _minimal_rotvec(T['zax'][idsK].reshape(-1, 3),
                             np.repeat(dirs, 200, 0)).reshape(N_DIRS, 200, 3)
        d6 = np.concatenate([dp, rv], -1)                             # (D,K,6)
        dq = np.einsum('dkje,dke->dkj', T['jinv6'][idsK], d6)
        order = (dq * dq).sum(-1).argsort(1)[:, :20]                  # top-20 kept
        cand_ids = np.take_along_axis(idsK, order, 1)                 # (D,20)
        # attempt rounds with failure-reshuffle (vectorized over directions)
        pend = np.arange(N_DIRS)
        cand_q = T['q'][cand_ids]                                     # (D,20,7)
        tried_sum = np.zeros((N_DIRS, 7), np.float32)
        n_tried = np.zeros(N_DIRS, np.int64)
        got = np.full((N_DIRS, 7), np.nan, np.float32)
        have = np.zeros(N_DIRS, bool)
        for attempt in range(MAX_TRY):
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
            # reshuffle remaining candidates of failed dirs: farthest from tried mean
            tried_sum[fail] += seed_q[~okn]; n_tried[fail] += 1
            for f in fail:
                rest = cand_q[f, 1:]
                if not len(rest):
                    continue
                dist = np.linalg.norm(
                    n_tried[f] * rest - tried_sum[f][None, :], axis=1)
                cand_q[f, 1:] = rest[np.argsort(-dist)]
            cand_q = np.roll(cand_q, -1, axis=1)  # advance to next candidate
            pend = fail
        attempted_dirs += N_DIRS; solved += int(have.sum())
        q_ok = torch.as_tensor(got[have], device=device, dtype=env.kin.dtype)
        if q_ok.shape[0]:
            kept = _dedup_q(q_ok, DEDUP_RAD)
            c = min(kept.shape[0], N_DIRS)
            seeds[i, :c] = kept[:c].cpu().numpy(); ik_ok[i, :c] = True
            nsol[i] = c
        if (i + 1) % 400 == 0:
            print(f'[gen2] {i+1}/{m} solve={solved/attempted_dirs*100:.1f}% '
                  f'median_sol={int(np.median(nsol[:i+1]))}', flush=True)
    np.savez(out, seeds=seeds, ik_ok=ik_ok, p0=val['p0'], line_dir=val['line_dir'],
             n_target=val['n_target'], q0_pilot=val['q0_pilot'],
             task_indices=val['task_indices'])
    print(f'[gen2] done solve={solved/attempted_dirs*100:.1f}% '
          f'median_sol={int(np.median(nsol))} empty={(nsol==0).sum()}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('table2', 'gen2'))
    ap.add_argument('--n-cvt', type=int, default=2048)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    globals()[f'stage_{args.stage}'](args, torch.device(args.device))


if __name__ == '__main__':
    main()
