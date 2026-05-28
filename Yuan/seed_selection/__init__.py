"""SMM-aware seed selection for diffusion-based start-q0 prediction.

Three subpackages:
    smm/        — SMM-aware label generation pipeline (perturb, cone-IK, walk,
                  rollout, robustness, label builder, dataset builder, parallel
                  build orchestration).
    diffusion/  — c → q0 diffusion model: definition, DataLoader, training,
                  DDIM sampling.
    eval/       — evaluation, post-build analysis, and visualization.
"""
