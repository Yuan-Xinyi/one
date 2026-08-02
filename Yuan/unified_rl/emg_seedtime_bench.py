"""Micro-benchmark: per-task online seed-generation cost of the proposed
pipeline (cone sampling + KD-tree query + 6-D rerank + IK attempts with
failure reshuffle + selector features/forward) on 128 eval10k tasks.
Fills \\phSeedTime in the journal paper. Excludes the rollout itself,
which is common to every arm in Table 2.
"""
import os, time
os.environ.setdefault('N_DIRS', '48')
import numpy as np
import torch
from pathlib import Path
from scipy.spatial import cKDTree

from Yuan.unified_rl.iksel_clean_pilot import (
    _table_path, _minimal_rotvec, _sample_in_cone, _dedup_q,
    POS_SCALE, CONE_DEG, DEDUP_RAD, GEN_SEED, N_DIRS)
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z
from Yuan.flow_connectivity.batched_rollout import _batched_ik_project
from Yuan.unified_rl.ikpool_bidir import _picks
from Yuan.unified_rl.iksel_campaign import _load_pool_env, _load_sel, C0_DIR

device = torch.device('cuda:0')
G = Path('Yuan/unified_rl/runs/iksel_final_n48')
env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, device)
T = np.load(_table_path(201600))
feat = np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1).astype(np.float32)
tree = cKDTree(feat)
g = np.load('Yuan/unified_rl/runs/eval10k_geoms.npz')
M, MAX_TRY, WARM = 128, 4, 8

t_gen = []
for i in range(M + WARM):
    t0 = time.perf_counter()
    p0, nt, ld = g['p0'][i], g['n_target'][i], g['line_dir'][i]
    rng = np.random.default_rng(GEN_SEED * 1000 + i)
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
    cand_ids = np.take_along_axis(idsK, order, 1)
    pend = np.arange(N_DIRS)
    cand_q = T['q'][cand_ids]
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
        tried_sum[fail] += seed_q[~okn]; n_tried[fail] += 1
        for f in fail:
            rest = cand_q[f, 1:]
            if len(rest):
                dist = np.linalg.norm(
                    n_tried[f] * rest - tried_sum[f][None, :], axis=1)
                cand_q[f, 1:] = rest[np.argsort(-dist)]
        cand_q = np.roll(cand_q, -1, axis=1)
        pend = fail
    q_ok = torch.as_tensor(got[have], device=device, dtype=env.kin.dtype)
    if q_ok.shape[0]:
        _dedup_q(q_ok, DEDUP_RAD)
    torch.cuda.synchronize()
    if i >= WARM:
        t_gen.append(time.perf_counter() - t0)

# selector cost, amortized per task (features + 5-member forward)
X, P, V = _load_pool_env(G / 'iksel_eval10k_candidates.npz',
                         G / 'iksel_eval10k_returns_hybrid.npz', device)
sel = _load_sel(G / 'sel_mixed_run0.pt', device)
torch.cuda.synchronize(); t0 = time.perf_counter()
_picks(*sel, X, V)
torch.cuda.synchronize()
t_sel = (time.perf_counter() - t0) / len(X) * 1e3

t = np.array(t_gen) * 1e3
print(f'gen per-task: mean={t.mean():.1f}ms median={np.median(t):.1f}ms '
      f'p90={np.percentile(t,90):.1f}ms  selector={t_sel:.2f}ms/task '
      f'TOTAL mean={t.mean()+t_sel:.1f}ms')
