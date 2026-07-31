"""Pilot: relabel the validation IK pool + diffusion pool under the HYBRID
controller (RL interior + classical near joint-limit boundary, hysteresis
tau 0.985/0.96) and compare per-candidate against the cached pure-C0 tables.

Answers, before committing to the full hybrid re-run:
  (a) uplift  : hybrid vs pure progress per pool (first / oracle / mean-valid)
  (b) ordering: within-task Spearman between hybrid and pure candidate returns
  (c) transfer: capture of the pure-trained S0 selector evaluated on hybrid
                returns (upper-bounds how much retraining matters)
"""
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenHybridController, rollout_selected_seeds)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.ikpool_bidir import SetSel, _picks

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
OUT = D / 'hybrid_pilot'
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
TAU_ENTER, TAU_EXIT = 0.985, 0.96
CHUNK = 512
dev = torch.device('cuda:0')


def roll_all(ds, valid, out_path):
    if out_path.exists():
        print(f'[hybrid] {out_path.name} exists, skip'); return np.load(out_path)
    env = build_env_from_run(resolve_controller_dir(C0_DIR), CHUNK, dev)
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, dev).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR))).gamma)
    ctl = FrozenHybridController(agent, ClassicalNullspaceController(env.kin),
                                 TAU_ENTER, TAU_EXIT)
    n, K = valid.shape
    prog = np.full((n, K), np.nan, np.float32)
    pairs = torch.nonzero(torch.as_tensor(valid), as_tuple=False).long()
    for s in range(0, pairs.shape[0], CHUNK):
        p = pairs[s:s + CHUNK]; nr = p.shape[0]
        if nr < CHUNK:
            p = torch.cat([p, p[-1:].expand(CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0]).to(device=dev, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(dev), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
        if (s // CHUNK) % 20 == 0:
            print(f'[hybrid {out_path.stem}] {min(s+CHUNK,pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    np.savez(out_path, progress_m=prog, valid=valid)
    return np.load(out_path)


def main():
    OUT.mkdir(exist_ok=True)
    rep = {}
    # ---- IK pool on validation tasks -----------------------------------
    ikds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_validation_candidates.npz')
    pure = np.load(D / 'ikpool_validation_returns.npz')
    Vik = pure['valid']
    hyb = roll_all(ikds, Vik, OUT / 'ik_validation_hybrid.npz')
    P0, P1 = np.nan_to_num(pure['progress_m']), np.nan_to_num(hyb['progress_m'])
    n = len(P0); r = np.arange(n)
    first = Vik.argmax(1)
    o0 = np.where(Vik, P0, -np.inf).max(1); o1 = np.where(Vik, P1, -np.inf).max(1)
    rho = [spearmanr(P0[i][Vik[i]], P1[i][Vik[i]]).statistic
           for i in range(n) if Vik[i].sum() >= 3]
    rho = [x for x in rho if np.isfinite(x)]
    # S0 transfer: pure-trained picks scored on hybrid returns
    env1 = build_env_from_run(resolve_controller_dir(C0_DIR), 1, dev)
    ck = torch.load(D / 'ikpool_selector_s0.pt', map_location=dev, weights_only=False)
    nets = []
    for st in ck['members']:
        net = SetSel().to(dev); net.load_state_dict(st); net.eval(); nets.append(net)
    X = _build_features(env1.kin, ikds, 4096).to(dev)
    pick = _picks(nets, ck['mu'].to(dev), ck['sd'].to(dev), X,
                  torch.as_tensor(Vik).to(dev)).cpu().numpy()
    sel1 = P1[r, pick]
    rep['ik_pool'] = {
        'pure':   {'first': float(P0[r, first].mean()), 'oracle': float(o0.mean())},
        'hybrid': {'first': float(P1[r, first].mean()), 'oracle': float(o1.mean())},
        'hybrid_minus_pure_oracle_mm': float((o1 - o0).mean() * 1e3),
        'within_task_spearman_median': float(np.median(rho)),
        's0_transfer_on_hybrid': {
            'mean_m': float(sel1.mean()),
            'capture_pct': float((sel1 - P1[r, first]).sum()
                                 / (o1 - P1[r, first]).sum() * 100)},
    }
    # ---- diffusion pool on the same tasks (old-system side) ------------
    src = torch.load(f'{C0_DIR}/unified.pt', map_location='cpu', weights_only=False)
    vti = torch.as_tensor(src['validation_task_indices']).long()
    difds = CachedSeedCandidateDataset.from_npz(
        'Yuan/seed_selection/runs/rank_train/candidates_K8.npz')
    difds = difds.index_select(vti)
    from Yuan.unified_rl.validity import validate_cached_dataset
    difds, _ = validate_cached_dataset(difds, env1.kin, env1.collision,
                                       chunk_size=4096, cone_deg=env1.cfg.cone_deg)
    Vd = difds.batch.valid.numpy()
    hybd = roll_all(difds, Vd, OUT / 'dif_validation_hybrid.npz')
    Pd1 = np.nan_to_num(hybd['progress_m'])
    m = len(Pd1); rm = np.arange(m)
    od1 = np.where(Vd, Pd1, -np.inf).max(1)
    old_pure = np.load(
        'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
        allow_pickle=True)
    rep['diffusion_pool_hybrid'] = {
        'first': float(Pd1[rm, Vd.argmax(1)].mean()), 'oracle': float(od1.mean()),
        'pure_oracle_ref': float(np.nanmean(old_pure['best_progress_m'])),
    }
    (OUT / 'hybrid_pilot.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
