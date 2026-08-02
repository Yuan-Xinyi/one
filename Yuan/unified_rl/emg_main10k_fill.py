"""Assemble the EMG journal paper's main-experiment numbers on eval_set_10k.

Inputs (all produced by _main10k_driver.sh + this script's roll stage):
  runs/iksel_final_n48/iksel_eval10k_{candidates,returns_hybrid}.npz
  runs/emg_analysis/heur_eval10k_{hybrid,classical,rl}_returns.npz
  runs/emg_analysis/enum_subset_returns.npz   (M4, 2,560-task ablation)
  runs/iksel_final_n48/main10k_picked_{classical,rl}.npz  (stage roll here)

Protocol = the conference paper's: per-task reference length l_ref = max
hybrid return over ALL candidates; report ratio-to-reference Mean/Std/
Min/Max overall and per difficulty bucket Easy(l_ref>=0.80) / Medium
(0.45..0.80) / Difficult(<0.45).

Stages:
  roll --controller classical|rl : roll the mixed-selector picks (1 seed/task)
  tables                         : print + save Table 1 / Table 2 JSON
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenRLController, rollout_selected_seeds)
from Yuan.unified_rl.iksel_campaign import _load_pool_env, _load_sel
from Yuan.unified_rl.ikpool_bidir import _picks
from Yuan.unified_rl.emg_problem_analysis import (
    FrozenClassicalController, DEFAULT_GAINS)

G = Path('Yuan/unified_rl/runs/iksel_final_n48')
A = Path('Yuan/unified_rl/runs/emg_analysis')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '128'))


def _lref_and_picks(dev):
    X, P, V = _load_pool_env(G / 'iksel_eval10k_candidates.npz',
                             G / 'iksel_eval10k_returns_hybrid.npz', dev)
    sel = _load_sel(G / 'sel_mixed_run0.pt', dev)
    pick = _picks(*sel, X, V)
    lref = torch.where(V, P, torch.tensor(-1e9, device=dev)).max(1).values
    ours_h = P[torch.arange(len(P), device=dev), pick]
    return pick.cpu().numpy(), lref.cpu().numpy(), ours_h.cpu().numpy()


def stage_roll(args, dev):
    out = G / f'main10k_picked_{args.controller}.npz'
    if out.exists():
        print(f'[roll] {out.name} exists, skip'); return
    pick, _, _ = _lref_and_picks(dev)
    c = np.load(G / 'iksel_eval10k_candidates.npz')
    m = len(c['p0'])
    tmp = G / '_picked_seed_tmp.npz'
    rows = np.arange(m)
    # column K (== n table candidates) is the q0_pilot fallback appended by
    # CachedSeedCandidateDataset; map those picks back to the pilot seed
    K = c['seeds'].shape[1]
    sel_seeds = np.where((pick < K)[:, None],
                         c['seeds'][rows, np.minimum(pick, K - 1)],
                         c['q0_pilot']).astype(np.float32)
    np.savez(tmp, seeds=sel_seeds[:, None],
             ik_ok=np.ones((m, 1), bool), p0=c['p0'],
             line_dir=c['line_dir'], n_target=c['n_target'],
             q0_pilot=c['q0_pilot'],
             task_indices=np.arange(m, dtype=np.int64))
    env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, dev)
    gamma = float(ppo_config_from_run(load_run_config(
        resolve_controller_dir(C0_DIR))).gamma)
    if args.controller == 'rl':
        agent = load_controller_agent(
            resolve_controller_dir(C0_DIR), env, dev).eval()
        ctl = FrozenRLController(agent)
    else:
        ctl = FrozenClassicalController(env.kin, DEFAULT_GAINS)
    ds = CachedSeedCandidateDataset.from_npz(tmp, include_fallback=False)
    prog = np.zeros(m, np.float32)
    for s in range(0, m, ROLL_CHUNK):
        r = torch.arange(s, min(s + ROLL_CHUNK, m))
        nr = len(r)
        if nr < ROLL_CHUNK:
            r = torch.cat([r, r[-1:].expand(ROLL_CHUNK - nr)])
        cb = ds.batch.index_select(r).to(device=dev, dtype=env.kin.dtype)
        res = rollout_selected_seeds(
            env, cb, torch.zeros(ROLL_CHUNK, dtype=torch.long, device=dev),
            ctl, gamma=gamma)
        prog[s:s + nr] = res.progress_m[:nr].cpu().numpy()
        if (s // ROLL_CHUNK) % 16 == 0:
            print(f'[roll {args.controller}] {s}/{m}', flush=True)
    np.savez(out, progress=prog)
    print(f'[roll] saved {out.name} mean={prog.mean():.4f}', flush=True)


def _stats(lens, lref, mask):
    l, r = lens[mask], lens[mask] / np.maximum(lref[mask], 1e-6) * 100
    return {'n': int(mask.sum()),
            'len': [round(float(x), 3) for x in (l.mean(), l.std(), l.min(), l.max())],
            'ratio': [round(float(x), 1) for x in (r.mean(), r.std(), r.min(), r.max())]}


def stage_tables(args, dev):
    _, lref, ours_h = _lref_and_picks(dev)
    buckets = {'All': np.ones(len(lref), bool), 'Easy': lref >= 0.80,
               'Medium': (lref >= 0.45) & (lref < 0.80),
               'Difficult': lref < 0.45}
    heur = {k: np.load(A / f'heur_eval10k_{k}_returns.npz')
            for k in ('hybrid', 'classical', 'rl')}
    arms = {'proposed+hybrid': ours_h,
            'proposed+classical': np.load(G / 'main10k_picked_classical.npz')['progress'],
            'proposed+rl': np.load(G / 'main10k_picked_rl.npz')['progress']}
    for k in ('hybrid', 'classical', 'rl'):
        arms[f'qmu+{k}'] = heur[k]['progress_mu']
        arms[f'qjl+{k}'] = heur[k]['progress_jl']
    t1 = {a: {b: _stats(v, lref, m) for b, m in buckets.items()}
          for a, v in arms.items()}
    # Table 2: seed-source comparison (hybrid controller), + gen time
    hgen = np.load(A / 'heur_eval10k.npz')
    t2 = {'q_mu': {'ratio': t1['qmu+hybrid']['All']['ratio'],
                   'len': t1['qmu+hybrid']['All']['len'],
                   't_ms': round(float(hgen['t_ms_mu']), 1)},
          'q_jl': {'ratio': t1['qjl+hybrid']['All']['ratio'],
                   'len': t1['qjl+hybrid']['All']['len'],
                   't_ms': round(float(hgen['t_ms_jl']), 1)},
          'proposed': {'ratio': t1['proposed+hybrid']['All']['ratio'],
                       'len': t1['proposed+hybrid']['All']['len']}}
    enum_f = G / 'enum_eval10k_sub_returns.npz'
    if enum_f.exists():
        sub = np.load(enum_f)['task_indices']
        Xe, Pe, Ve = _load_pool_env(G / 'enum_eval10k_sub_candidates.npz',
                                    enum_f, dev)
        sel = _load_sel(G / 'sel_mixed_run0.pt', dev)
        pe = _picks(*sel, Xe, Ve)
        el = Pe[torch.arange(len(Pe), device=dev), pe].cpu().numpy()
        eceil = torch.where(Ve, Pe, torch.tensor(-1e9, device=dev)) \
            .max(1).values.cpu().numpy()
        rs = np.maximum(lref[sub], 1e-6)
        t2['enum_subset'] = {
            'n': int(len(sub)),
            'ratio_enum_picked': [round(float(x), 1) for x in
                                  ((el / rs * 100).mean(), (el / rs * 100).std())],
            'ratio_enum_ceiling': round(float((eceil / rs * 100).mean()), 1),
            'ratio_proposed_same_subset': [round(float(x), 1) for x in
                ((ours_h[sub] / rs * 100).mean(), (ours_h[sub] / rs * 100).std())]}
    rep = {'lref_mean': round(float(lref.mean()), 4), 'table1': t1, 'table2': t2}
    (G / 'main10k_tables.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('roll', 'tables'))
    ap.add_argument('--controller', default='classical',
                    choices=('classical', 'rl'))
    args = ap.parse_args()
    globals()[f'stage_{args.stage}'](args, torch.device('cuda:0'))


if __name__ == '__main__':
    main()
