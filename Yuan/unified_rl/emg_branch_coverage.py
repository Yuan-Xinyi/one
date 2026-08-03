"""Bridge between the branch analysis (Sec. IV) and the table-based
candidate layer (Sec. V): does a layer that never reasons about branches
actually place candidates on distinct branches?

For each of the 97 tasks whose SMM was traced in Part A, every candidate
of a given source is continued back to the exact nominal pose by the same
full-pose Newton projection used for the traced germs, then assigned to
the nearest traced branch (linkage radius 0.3 rad, the radius that
defined the branches). Candidates farther than that from every traced
point occupy a branch the enumeration germs never found.

Note on bias: the traced branch set was seeded from the *enumeration*
candidates, so this test is conservative for the proposed layer and
generous to the enumeration arm.

Output: runs/emg_analysis/branch_coverage.json
"""
import json
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.unified_rl.emg_problem_analysis import _project_full_pose
from Yuan.seed_selection.smm.cone_ik import _build_R_with_z

A = Path('Yuan/unified_rl/runs/emg_analysis')
D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
G = Path('Yuan/unified_rl/runs/iksel_final_n48')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
LINK = 0.3          # branch linkage radius used when the germs were deduped
POS_TOL = 5e-3

dev = torch.device('cuda:0')
env = build_env_from_run(resolve_controller_dir(C0_DIR), 1, dev)
kin = env.kin
lo, hi = kin.lmt_lo, kin.lmt_up

a1 = np.load(A / 'a1_smm.npz')
rows = json.loads((A / 'tasks.json').read_text())['rows']
cand_enum = np.load(D / 'ikpool_validation_candidates.npz')
V_enum = np.load(
    'Yuan/unified_rl/runs/ikpool_final/enumeration_validation_returns_hybrid.npz'
)['valid']
cand_iksel = np.load(G / 'iksel_validation_candidates.npz')
heur = np.load(A / 'heur_validation.npz')

SOURCES = {
    'proposed (table, 48 dirs)': lambda r: cand_iksel['seeds'][r][
        cand_iksel['ik_ok'][r]],
    'enumeration (128 restarts)': lambda r: cand_enum['seeds'][r][:32][
        V_enum[r][:32]],
    'q_mu': lambda r: heur['q_mu'][r][None, :],
    'q_jl': lambda r: heur['q_jl'][r][None, :],
}

res = {k: {'covered': [], 'available': [], 'hit_best': [], 'extra': [],
           'kept': []} for k in SOURCES}

for r in rows:
    m = a1['task'] == r
    if not m.any():
        continue
    br, qb, pr = a1['branch'][m], a1['q'][m], a1['progress'][m]
    branches = sorted(set(br.tolist()))
    best_branch = max(branches, key=lambda b: pr[br == b].max())
    qb_t = torch.as_tensor(qb, device=dev, dtype=kin.dtype)

    p_t = torch.as_tensor(cand_iksel['p0'][r], device=dev, dtype=kin.dtype)
    R_t = _build_R_with_z(
        torch.as_tensor(cand_iksel['n_target'][r], device=dev,
                        dtype=kin.dtype).unsqueeze(0),
        torch.as_tensor(cand_iksel['line_dir'][r], device=dev,
                        dtype=kin.dtype))[0]

    for name, get in SOURCES.items():
        q = get(r)
        q = q[np.isfinite(q).all(1)]
        if not len(q):
            res[name]['covered'].append(0); res[name]['available'].append(
                len(branches)); res[name]['hit_best'].append(0)
            res[name]['extra'].append(0); res[name]['kept'].append(0)
            continue
        qt = torch.as_tensor(q, device=dev, dtype=kin.dtype)
        qp = _project_full_pose(kin, qt, p_t.expand(len(qt), 3),
                                R_t.expand(len(qt), 3, 3), iters=5)
        ok = ((qp > lo + 0.01) & (qp < hi - 0.01)).all(-1)
        p_chk, _, _, _ = kin.tcp_fk_jac(qp)
        ok &= (p_chk - p_t).norm(dim=-1) < POS_TOL
        qp = qp[ok]
        if not len(qp):
            res[name]['covered'].append(0); res[name]['available'].append(
                len(branches)); res[name]['hit_best'].append(0)
            res[name]['extra'].append(0); res[name]['kept'].append(0)
            continue
        d = torch.cdist(qp, qb_t)                       # (n_cand, n_pts)
        dmin, imin = d.min(1)
        lab = np.where(dmin.cpu().numpy() < LINK, br[imin.cpu().numpy()], -1)
        cov = set(int(x) for x in lab if x >= 0)
        res[name]['covered'].append(len(cov))
        res[name]['available'].append(len(branches))
        res[name]['hit_best'].append(int(best_branch in cov))
        res[name]['extra'].append(int((lab < 0).sum()))
        res[name]['kept'].append(int(len(qp)))

rep = {}
for name, v in res.items():
    cov = np.array(v['covered'], float); av = np.array(v['available'], float)
    rep[name] = {
        'n_tasks': len(cov),
        'mean_candidates_at_nominal_pose': round(float(np.mean(v['kept'])), 1),
        'mean_branches_covered': round(float(cov.mean()), 2),
        'mean_branches_available': round(float(av.mean()), 2),
        'coverage_pct': round(float((cov / av).mean() * 100), 1),
        'best_branch_hit_pct': round(float(np.mean(v['hit_best']) * 100), 1),
        'mean_candidates_off_traced_set': round(float(np.mean(v['extra'])), 1),
    }
(A / 'branch_coverage.json').write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
