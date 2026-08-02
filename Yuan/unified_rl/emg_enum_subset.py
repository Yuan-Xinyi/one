"""Exhaustive-enumeration ablation on a 2,560-task subset of eval10k
(Table-2 ablation row): 128-restart cone-IK enumeration + hybrid relabel."""
import os, sys
from pathlib import Path
import numpy as np
import torch
from Yuan.unified_rl.checkpoint import (build_env_from_run, load_controller_agent,
    load_run_config, ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenHybridController, rollout_selected_seeds
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate
from Yuan.unified_rl.ikpool_build_full import (CONE_DEG, N_ORI, N_RESTART,
    JOINT_MARGIN, DEDUP_RAD, K_POOL, _fps_select)

OUT = Path('Yuan/unified_rl/runs/iksel_final_n48')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '128'))
N_SUB = 2560
dev = torch.device('cuda:0')

g = np.load('Yuan/unified_rl/runs/eval10k_geoms.npz')
rng = np.random.default_rng(20260802)
sub = np.sort(rng.choice(len(g['p0']), N_SUB, replace=False))
cand_p = OUT / 'enum_eval10k_sub_candidates.npz'
if not cand_p.exists():
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, dev)
    seeds = np.full((N_SUB, K_POOL, 7), np.nan, np.float32)
    ik_ok = np.zeros((N_SUB, K_POOL), bool)
    for i, r in enumerate(sub):
        rr = np.random.default_rng(20260802 * 1000 + int(r))
        q = cone_constrained_ik_enumerate(
            p0=torch.as_tensor(g['p0'][r]), n_target=torch.as_tensor(g['n_target'][r]),
            line_dir=torch.as_tensor(g['line_dir'][r]), kin=env.kin,
            collision=env.collision, cone_angle_deg=CONE_DEG, n_orientations=N_ORI,
            n_ik_restarts=N_RESTART, joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, rng=rr)
        q = _fps_select(q, K_POOL)
        if q.shape[0]:
            seeds[i, :q.shape[0]] = q.cpu().numpy(); ik_ok[i, :q.shape[0]] = True
        if (i+1) % 500 == 0: print(f'[enum-sub] gen {i+1}/{N_SUB}', flush=True)
    np.savez(cand_p, seeds=seeds, ik_ok=ik_ok, p0=g['p0'][sub], line_dir=g['line_dir'][sub],
             n_target=g['n_target'][sub], q0_pilot=g['q0_pilot'][sub],
             task_indices=sub.astype(np.int64))
    print('[enum-sub] candidates saved', flush=True)
ret_p = OUT / 'enum_eval10k_sub_returns.npz'
if not ret_p.exists():
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, dev)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, dev).eval()
    ctl = FrozenHybridController(agent, ClassicalNullspaceController(env.kin), 0.985, 0.96)
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(cand_p)
    parts = []
    for s in range(0, len(ds), 256):
        sb = ds.batch.index_select(torch.arange(s, min(s+256, len(ds))))
        parts.append(check_candidate_validity(env.kin, env.collision,
            sb.to(env.kin.device, dtype=env.kin.dtype), cone_deg=env.cfg.cone_deg).valid.cpu())
    val = torch.cat(parts)
    prog = np.full(tuple(val.shape), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s+ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK: p = torch.cat([p, p[-1:].expand(ROLL_CHUNK-nr, -1)])
        cb = ds.batch.index_select(p[:, 0]).to(device=dev, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cb, p[:, 1].to(dev), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr): prog[int(p[j,0]), int(p[j,1])] = pm[j]
        if (s // ROLL_CHUNK) % 40 == 0:
            print(f'[enum-sub] roll {min(s+ROLL_CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(ret_p, progress_m=prog, valid=val.numpy(), task_indices=sub.astype(np.int64))
    print('[enum-sub] returns saved', flush=True)
