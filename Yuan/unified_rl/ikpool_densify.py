"""E3 refinement (step 3): does DENSER IK enumeration close the ~6% pool gap?

For the val+external tasks where the K=32 IK oracle is still below the old
system's deployed pick, regenerate with a denser budget (24 orientations x 16
restarts = 384 IK attempts, FPS->K=48) and re-roll under C0. If the pool gap
closes, the ~6% residual is an enumeration-coverage artifact, not a manifold
limitation -> diffusion is genuinely unnecessary.
"""
import json
import numpy as np
import torch
from pathlib import Path

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenRLController, rollout_selected_seeds
from Yuan.unified_rl.validity import check_candidate_validity
from Yuan.seed_selection.smm.cone_ik import cone_constrained_ik_enumerate
from Yuan.unified_rl.ikpool_build_full import (C0_DIR, GEN_SEED, CONE_DEG, JOINT_MARGIN,
                                               DEDUP_RAD, ROLL_CHUNK, _fps_select)

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
OLD = {'validation': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
       'external': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'}
DENSE_ORI, DENSE_RESTART, DENSE_K = 24, 16, 48
dev = torch.device('cuda:0')


def gap_tasks(which):
    r = np.load(D / f'ikpool_{which}_returns.npz')
    P, V, tids = r['progress_m'], r['valid'], r['task_indices']
    Kik = P.shape[1] - 1
    ik_oracle = np.where(V[:, :Kik], P[:, :Kik], -np.inf).max(1)
    o = np.load(OLD[which], allow_pickle=True)
    order = {int(t): i for i, t in enumerate(o['task_indices'])}
    old_pol = np.nan_to_num(o['policy_progress_m'])[np.array([order[int(t)] for t in tids])]
    gap = np.nonzero(ik_oracle < old_pol - 1e-3)[0]
    cand = CachedSeedCandidateDataset.from_npz(D / f'ikpool_{which}_candidates.npz')
    return gap, cand, ik_oracle, old_pol


def run(which):
    gap, cand, ik_oracle32, old_pol = gap_tasks(which)
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, dev)
    m = len(gap)
    seeds = np.full((m, DENSE_K, 7), np.nan, np.float32)
    ik_ok = np.zeros((m, DENSE_K), bool)
    p0 = cand.batch.p0.numpy()[gap]; ld = cand.batch.line_dir.numpy()[gap]
    nt = cand.batch.n_target.numpy()[gap]
    fbq = cand.batch.q0.numpy()[gap, cand.fallback_index]
    for i, g in enumerate(gap.tolist()):
        rng = np.random.default_rng(GEN_SEED * 100000 + 900000000 + int(g))
        q = cone_constrained_ik_enumerate(
            p0=torch.as_tensor(p0[i]), n_target=torch.as_tensor(nt[i]),
            line_dir=torch.as_tensor(ld[i]), kin=env1.kin, collision=env1.collision,
            cone_angle_deg=CONE_DEG, n_orientations=DENSE_ORI, n_ik_restarts=DENSE_RESTART,
            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, rng=rng)
        q = _fps_select(q, DENSE_K)
        if q.shape[0]:
            seeds[i, :q.shape[0]] = q.cpu().numpy().astype(np.float32); ik_ok[i, :q.shape[0]] = True
    tmp = D / f'_dense_{which}.npz'
    np.savez(tmp, seeds=seeds, ik_ok=ik_ok, p0=p0, line_dir=ld, n_target=nt,
             q0_pilot=fbq, task_indices=gap.astype(np.int64))
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, dev)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, dev).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(tmp)
    val = check_candidate_validity(env.kin, env.collision,
                                   ds.batch.to(env.kin.device, dtype=env.kin.dtype),
                                   cone_deg=env.cfg.cone_deg).valid.cpu()
    K = ds.batch.n_candidates
    prog = np.full((m, K), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        c = ds.batch.index_select(p[:, 0]).to(device=dev, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, c, p[:, 1].to(dev), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
    dense_oracle = np.where(val.numpy(), np.nan_to_num(prog, nan=-1e9), -1e9).max(1)
    old_gap = old_pol[gap]
    closed = dense_oracle >= old_gap - 1e-3
    return {
        'which': which, 'n_gap_tasks': int(m),
        'gap_frac_of_set_pct': float(m / len(old_pol) * 100),
        'dense_closed': int(closed.sum()), 'dense_closed_pct': float(closed.mean() * 100),
        'K32_oracle_gap_mm': float((old_gap - ik_oracle32[gap]).mean() * 1e3),
        'K48dense_oracle_gap_mm': float((old_gap - dense_oracle).mean() * 1e3),
        'still_below_after_dense': int((~closed).sum()),
    }


rep = {w: run(w) for w in ('validation', 'external')}
print(json.dumps(rep, indent=1))
(D / 'ikpool_densify.json').write_text(json.dumps(rep, indent=1))
