# Probe / evaluation scripts (session archive)

One-off diagnostic and evaluation scripts moved here verbatim from the
session scratchpad; they are the provenance for everything under
`runs/paper_fill/` (fam_unify, ratio_assets, selector_ood, ...).
Paths inside are absolute to the machine/worktree they ran on; treat them
as reproducibility records, not maintained tools.

Highlights (2026-08, algorithm-optimization phase):
- fam_unify.py           four-family relabel/retrain/report pipeline (paper tables)
- ablate_eval.py         a_max 10/20 cont + LSTM/TF vertex backbone ablations
- decoupled_probe.py     direction/magnitude decoupling, G-tensor, two-stage alpha readout
- snr_probe.py           probing-scale SNR: per-state Spearman vs pick quality
- gated_actor.py         gate-vs-critic decomposition (vertex actor + gate)
- alive_diag.py          death forensics: all-16-dead entrapment, cliff onset
- gate2_probe.py/gate2_10k.py  two-step viability gate (vlook2), 10k: 0.5804 vs 0.5625
- learned_gate.py        learned feasibility classifier (negative: false-permit 98%)
- dirfrac2_eval.py       dirfrac v2 actor-only 10k: 0.5079, ratio 79.9 / P10 28.4
- gated_dirfrac.py       diagnostic intent-preserving gate on dirfrac v2: 84.9 / 42.8
- l81_cont_10k.py        L81(+zero) candidate-set 10k eval
- witness_arms.py/witness_gen2.py  multi-arm witness strengthening
