"""End-to-end system evaluation: seed x controller ablation.

Pipeline:
    build_eval_set.py    — stratified safe held-out task set (eval_set_*.npz)
    run_cell.py          — single-cell runner (cls_cls baseline, diff_cls
                           seed-only, cls_hyb controller-only, diff_hyb full
                           method, oracle_cls classical-label oracle)
    run_oracle_prime.py  — controller-aware oracle 'oracle_hyb' (max over
                           SMM top-K' under the hybrid deployment controller)
    aggregate.py         — CSV + markdown report + figures across cells

Cell naming = <seed_source>_<controller>:
    cls = pilot q0_seed,            classical = Yoshikawa nullspace
    diff = diffusion seed,          hyb       = hybrid (RL + Classical, variant B)
    oracle_cls = label-argmax seed (classical-label oracle, deployed under hybrid)
    oracle_hyb = max over SMM top-K' candidates evaluated under hybrid

Seed sources and controller wrappers live in `seed_sources.py` and
`rollout_controllers.py`; both import from `Yuan.seed_selection` and
`Yuan.RL_controller` only.
"""
