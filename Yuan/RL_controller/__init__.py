"""PPO-trained nullspace controller for FR3 path-following.

Five subpackages:
    env/       — torch-batched env, classical-nullspace baseline, line task
                 distribution, single-episode rollout helper.
    algo/      — PPO algorithm (Agent + train loop) and training entry point.
    eval/      — post-training evaluation: pure-controller diagnostics,
                 hybrid (RL + Classical) variant-A/B, kinematic-limit audit.
    analysis/  — deeper analyses of cached rollouts (escape geometry, policy
                 escape decisions, RL_worse / RL_better task subsets).
    viz/       — interactive policy-rollout viewer plus static scene / ghost
                 overlay renderers.
"""
