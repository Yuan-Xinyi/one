"""Head-to-head of the final IKSel system against the diffusion+ranker
system on the SAME validation tasks under the SAME hybrid controller,
both deployed (one seed, one rollout).

All inputs are cached; no rollouts are performed here.
  diffusion pool hybrid returns : runs/ikpool_full_v1/hybrid_pilot/dif_validation_hybrid.npz
  diffusion ranker picks        : runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz
  IKSel-48 pool hybrid returns  : runs/iksel_final_n48/iksel_validation_returns_hybrid.npz
Reference for the ratio is the union ceiling of both pools, so neither
side is measured against its own pool.
"""
import json
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.iksel_campaign import _load_pool_env, _load_sel, C0_DIR
from Yuan.unified_rl.ikpool_bidir import _picks

G = Path('Yuan/unified_rl/runs/iksel_final_n48')
H = Path('Yuan/unified_rl/runs/ikpool_full_v1/hybrid_pilot')
OLD = Path('Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/'
           'eval_validation_cmp1024.npz')
dev = torch.device('cuda:0')

# ---- diffusion side: rows follow the controller run's validation order --
src = torch.load(f'{C0_DIR}/unified.pt', map_location='cpu', weights_only=False)
vti = np.asarray(src['validation_task_indices']).astype(np.int64)
dh = np.load(H / 'dif_validation_hybrid.npz')
Pd, Vd = np.nan_to_num(dh['progress_m']), dh['valid']

old = np.load(OLD, allow_pickle=True)
oti = old['task_indices'].astype(np.int64)
perm = np.argsort(oti)[np.searchsorted(oti[np.argsort(oti)], vti)]
assert (oti[perm] == vti).all(), 'task alignment failed'
# consistency check: the validity mask must survive the permutation
assert (old['candidate_valid'][perm] == Vd).mean() > 0.99, 'column mismatch'

pick_d = old['policy_candidate_index'][perm].astype(np.int64)
r = np.arange(len(vti))
dif_deploy = Pd[r, pick_d]
dif_ceiling = np.where(Vd, Pd, -np.inf).max(1)
dif_first = Pd[r, Vd.argmax(1)]

# ---- proposed side: same tasks, mixed selector, one rollout ------------
X, P, V = _load_pool_env(G / 'iksel_validation_candidates.npz',
                         G / 'iksel_validation_returns_hybrid.npz', dev)
cand = np.load(G / 'iksel_validation_candidates.npz')
ours_ti = cand['task_indices'].astype(np.int64)
assert (ours_ti == vti).all(), 'proposed side task order differs'
sel = _load_sel(G / 'sel_mixed_run0.pt', dev)
pick_o = _picks(*sel, X, V)
Pn = P.cpu().numpy()
our_deploy = Pn[r, pick_o.cpu().numpy()]
our_ceiling = torch.where(V, P, torch.tensor(-1e9, device=dev)) \
    .max(1).values.cpu().numpy()
our_first = Pn[r, V.cpu().numpy().argmax(1)]

# ---- common reference: union ceiling of the two pools ------------------
ref = np.maximum(our_ceiling, dif_ceiling)
d = our_deploy - dif_deploy
boot = np.array([d[np.random.default_rng(s).integers(0, len(d), len(d))].mean()
                 for s in range(2000)])
rep = {
    'n_tasks': int(len(d)),
    'ratio_to_common_reference_pct': {
        'proposed': round(float((our_deploy / ref).mean() * 100), 2),
        'diffusion': round(float((dif_deploy / ref).mean() * 100), 2)},
    'deployed_mean_m': {'proposed': round(float(our_deploy.mean()), 4),
                        'diffusion': round(float(dif_deploy.mean()), 4)},
    'paired_delta_mm': round(float(d.mean() * 1e3), 2),
    'paired_ci95_mm': [round(float(np.percentile(boot, 2.5) * 1e3), 2),
                       round(float(np.percentile(boot, 97.5) * 1e3), 2)],
    'trimmed5_delta_mm': round(float(
        np.mean(np.sort(d)[int(.05 * len(d)):int(.95 * len(d))]) * 1e3), 2),
    'win_gt1mm_pct': round(float((d > 1e-3).mean() * 100), 1),
    'harm_gt1mm_pct': round(float((d < -1e-3).mean() * 100), 1),
    'pool_ceiling_m': {'proposed': round(float(our_ceiling.mean()), 4),
                       'diffusion': round(float(dif_ceiling.mean()), 4),
                       'ceiling_delta_mm': round(
                           float((our_ceiling - dif_ceiling).mean() * 1e3), 2)},
    'first_valid_m': {'proposed': round(float(our_first.mean()), 4),
                      'diffusion': round(float(dif_first.mean()), 4)},
    'capture_pct': {
        'proposed': round(float((our_deploy - our_first).sum()
                                / (our_ceiling - our_first).sum() * 100), 1),
        'diffusion': round(float((dif_deploy - dif_first).sum()
                                 / (dif_ceiling - dif_first).sum() * 100), 1)},
    'proposed_deploy_vs_diffusion_ceiling_mm': round(
        float((our_deploy - dif_ceiling).mean() * 1e3), 2),
}
(G / 'vs_diffusion.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
