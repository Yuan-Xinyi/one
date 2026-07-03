"""Self-improvement loop: hybrid switching as a policy-improvement operator.

    pi_{k+1} = PPO+BC( verified rescue steps of hybrid(pi_k, classical) )

Round k:
    collect.py — hybrid(pi_k, classical) rollouts on training-distribution
                 tasks; record every classical-rescue step (obs, a_cls);
                 task-level win filter (hybrid must strictly outlive pure
                 pi_k on the same task) keeps only *verified* rescues.
    loop.py    — orchestrates rounds: collect -> joint PPO+BC fine-tune
                 (warm-start from pi_k) -> 10k-set eval. Convergence when
                 the operator runs dry: frac(hybrid > pure) -> 0 and
                 L_pure / L_hybrid -> 1 (rescue behavior fully internalized).
"""
