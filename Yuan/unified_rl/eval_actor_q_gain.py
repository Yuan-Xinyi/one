"""Per-seed matched-frozen actor-Q vs actor selector gain.

Reuses the certified materialize helpers so the arithmetic is identical to the
published actor-Q materialization, but reads the fit/model/calibration split
straight from the trained 45-D ensemble checkpoint (offline_seed_ensemble_complete)
instead of requiring a round_complete joint controller-source.

Reports, on the ensemble's own held-out model-select and calibration splits, how
much more net ray progress the actor-Q proposal captures than the actor-only
deployment (base) -- i.e. exactly the increment that ties the joint model with
the matched-frozen baseline in the sealed 2x2.
"""
import argparse, copy, json, math
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.unified_rl.validity import validate_cached_dataset
from Yuan.unified_rl.provenance import (
    file_fingerprint, controller_fingerprint, state_dict_fingerprint)
from Yuan.unified_rl.offline_seed_train import _calibrate_one_head, load_return_cache
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.materialize_seed_blend import _members_from_states
from Yuan.unified_rl.seed_deployment import deployment_config_from_checkpoint
from Yuan.unified_rl.materialize_actor_q_selector import (
    _copy_ensemble_states, _ensemble_outputs, _actor_q_proposal,
    _fixed_rule_report, _base_deployment_indices, _choose_model_candidate,
    _promotion_reasons, ACTOR_Q_WEIGHT_GRID, ACTOR_Q_SCALE_M, CONFIDENCE_Z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ensemble', required=True, help='trained 45-D ensemble unified.pt')
    ap.add_argument('--controller-ckpt', required=True, help='C0 controller dir (env + artifacts)')
    ap.add_argument('--return-cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--chunk', type=int, default=4096)
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ensemble, map_location='cpu', weights_only=False)
    if int(ck['seed_architecture']['feature_dim']) != 45:
        raise SystemExit('not a 45-D ensemble')

    controller_dir = resolve_controller_dir(args.controller_ckpt)
    controller_state_sha256 = state_dict_fingerprint(
        torch.load(controller_dir / 'agent.pt', map_location='cpu', weights_only=True))
    controller_artifact = controller_fingerprint(controller_dir)

    # candidate cache + source-checkpoint fingerprints as recorded by the ensemble
    prov = ck['offline_seed_ensemble_provenance']
    cand_ref = prov['source_candidate_cache']
    candidate_path = Path(cand_ref['path'])
    candidate_artifact = {'size': int(cand_ref['size']), 'sha256': str(cand_ref['sha256'])}
    src_ref = prov['source_checkpoint']
    source_artifact = {'size': int(src_ref['size']), 'sha256': str(src_ref['sha256'])}
    # the return cache was generated from the round_complete source checkpoint;
    # load_return_cache validates against that provenance, not the ensemble.
    source_ck = torch.load(src_ref['path'], map_location='cpu', weights_only=False)

    gamma = float(np.load(args.return_cache, allow_pickle=True)['controller_gamma'])
    env = build_env_from_run(controller_dir, 1, device)
    dataset = CachedSeedCandidateDataset.from_npz(candidate_path)
    dataset, _ = validate_cached_dataset(
        dataset, env.kin, env.collision, chunk_size=args.chunk, cone_deg=env.cfg.cone_deg)
    train_dataset = dataset.select_source_tasks(
        torch.as_tensor(source_ck['train_task_indices']).cpu())

    cached = load_return_cache(
        args.return_cache, source=source_ck, source_artifact=source_artifact,
        candidate_artifact=candidate_artifact, controller_artifact=controller_artifact,
        controller_state_sha256=controller_state_sha256,
        objective='undiscounted', gamma=gamma, train_dataset=train_dataset)

    model_idx = torch.as_tensor(ck['offline_ensemble_model_select_local_indices']).long()
    calib_idx = torch.as_tensor(ck['offline_ensemble_calibration_local_indices']).long()

    states, meta, arch = _copy_ensemble_states(ck, label='ensemble')
    members = _members_from_states(states, arch, device)

    features = _build_features(env.kin, train_dataset, args.chunk)
    sel = torch.cat([model_idx, calib_idx])
    actor, q, _ = _ensemble_outputs(members, features, cached.valid, sel,
                                    batch_size=1024, device=device)
    n_model = model_idx.numel()
    base_cfg = deployment_config_from_checkpoint(ck)

    def part(v, calib):
        return v[n_model:] if calib else v[:n_model]

    def split_arrays(idx):
        return (cached.valid[idx].numpy(), cached.progress_m[idx].numpy(),
                [cached.task_fingerprints[int(i)] for i in idx.tolist()])

    mv, mp, mfp = split_arrays(model_idx)
    cv, cp, cfp = split_arrays(calib_idx)
    base_model_sel = _base_deployment_indices(part(actor, False), part(q, False), mv, base_cfg)
    base_calib_sel = _base_deployment_indices(part(actor, True), part(q, True), cv, base_cfg)

    cands = []
    for w in ACTOR_Q_WEIGHT_GRID:
        prop, margin, first = _actor_q_proposal(part(actor, False), part(q, False), mv, w)
        thr = _calibrate_one_head('actor-q', prop, margin.astype(np.float64), first,
                                  mp, mv, mfp, CONFIDENCE_Z)
        rep = _fixed_rule_report(prop, margin, first, float(thr['threshold']),
                                 mp, mv, base_model_sel, mfp)
        reasons = _promotion_reasons(rep)
        cands.append({'weight': float(w), 'threshold_selection': thr,
                      'model_report': rep, 'eligible': not reasons})

    chosen = _choose_model_candidate(cands)
    out = {'ensemble': str(args.ensemble), 'seed': int(ck['offline_seed_ensemble_provenance']['settings']['seed']),
           'chosen_weight': None, 'model_gain_mm': None, 'calib_gain_mm': None,
           'calib_gain_lcb_mm': None, 'calib_harm_pct': None, 'calib_win_pct': None,
           'calib_trimmed_mm': None, 'promoted': False, 'n_model': int(n_model),
           'n_calib': int(calib_idx.numel())}
    if chosen is not None:
        out['chosen_weight'] = float(chosen['weight'])
        out['model_gain_mm'] = float(chosen['model_report']['paired_mean_delta_m']) * 1e3
        prop, margin, first = _actor_q_proposal(part(actor, True), part(q, True), cv, float(chosen['weight']))
        crep = _fixed_rule_report(prop, margin, first, float(chosen['threshold_selection']['threshold']),
                                  cp, cv, base_calib_sel, cfp)
        out['calib_gain_mm'] = float(crep['paired_mean_delta_m']) * 1e3
        out['calib_gain_lcb_mm'] = float(crep['paired_lower_bound_m']) * 1e3
        out['calib_harm_pct'] = float(crep['geometry_harm_rate_gt_1mm']) * 100
        out['calib_win_pct'] = float(crep['geometry_win_rate_gt_1mm']) * 100
        out['calib_trimmed_mm'] = float(crep['paired_trimmed_mean_delta_m']) * 1e3
        out['promoted'] = not _promotion_reasons(crep)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
