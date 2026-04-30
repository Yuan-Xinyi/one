"""SAC training for the farsighted-seed problem (v8).

Single-step contextual bandit specialisation:
    Q(c, q) ≈ L(c, q) / T  (deterministic, no Bellman bootstrap)
    π(q|c)  = MixtureGaussian(K=4) with reparameterised sampling

Per iteration
-------------
  1. Sample a batch of contexts c_i.
  2. Behaviour: q_i ~ π_θ(·|c_i).
  3. Run batched_rollout to get L_i (raw rollout length).
  4. Push (s_i, q_i, L_i, T_i) into FIFO replay buffer.
  5. K_Q steps of Q regression on minibatches drawn from buffer.
  6. K_π steps of policy update via reparam SAC objective.
  7. (Optional) auto-tune α to hit target entropy H_target = -action_dim.

No PPO clip, no KL early-stop, no on-policy advantage estimation: data is
deterministic, so each (s, q, L) tuple in the buffer is a permanent ground
truth label that we can revisit forever.
"""
from __future__ import annotations
import os, time, collections
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv, sample_raw_c
from Yuan.RL.policy import make_policy
from Yuan.RL.qnet import QNet, QEnsemble, ReplayBuffer
from Yuan.RL.batched_rollout import batched_rollout


def _maybe_init_wandb():
    if not cfg.WANDB_ENABLE:
        return None
    import wandb
    config = {n: v for n, v in vars(cfg).items()
              if n.isupper()
              and isinstance(v, (int, float, str, bool, tuple, list, type(None)))}
    return wandb.init(project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY,
                      name=cfg.WANDB_RUN_NAME, config=config)


# ----- batched-rollout-friendly task sampler -----
def _sample_training_batch(env: FarsightedSeedEnv, batch_size: int):
    """Sample `batch_size` training tasks via the env's randomized
    distribution. Returns (states_np, tasks)."""
    tasks = env._sample_tasks(batch_size)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0)
    return states.astype(np.float32), tasks


def _ucb_action_select(policy, qens: QEnsemble, states_t: torch.Tensor,
                       K: int, kappa: float = 1.0):
    """UCB action selection: for each state, sample K candidate actions
    from the (stochastic) policy; pick the one maximizing
        UCB(c, a) = Q_ensemble.mean(c, a) + κ · Q_ensemble.std(c, a).

    This biases the *behaviour distribution* toward (c, a) pairs the
    Q ensemble is uncertain about — i.e., adds densely-sample those
    regions in the replay buffer. Cheap: no extra env queries.
    """
    B = states_t.shape[0]
    cands = []
    for _ in range(K):
        with torch.no_grad():
            a, _ = policy.act(states_t, deterministic=False)
        cands.append(a)
    a_stack = torch.stack(cands, dim=0)              # (K, B, A)

    q_means, q_stds = [], []
    for k in range(K):
        q_means.append(qens.mean(states_t, a_stack[k]))
        q_stds.append(qens.std(states_t, a_stack[k]))
    q_means = torch.stack(q_means, dim=0)            # (K, B)
    q_stds  = torch.stack(q_stds,  dim=0)
    ucb = q_means + kappa * q_stds                   # (K, B)
    best_k = ucb.argmax(dim=0)                       # (B,)
    rows = torch.arange(B, device=states_t.device)
    return a_stack[best_k, rows]                     # (B, A)


def main():
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = _maybe_init_wandb()

    # SAC trains on rollouts collected from a randomized env. We hold the
    # MJCollider in serial-rollout-only paths (visualisation, evaluation),
    # not here — batched_rollout has its own sphere-based collision check.
    env = FarsightedSeedEnv(seed=cfg.SEED, randomize=True,
                            use_collision=False)
    state_dim = cfg.STATE_DIM
    action_dim = env.action_dim

    qmid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    qhalf = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)

    policy = make_policy(state_dim, action_dim, qmid, qhalf).to(device)
    qnet = QNet(state_dim, action_dim).to(device)
    qens = QEnsemble(state_dim, action_dim, m=cfg.Q_ENSEMBLE_M).to(device)
    buffer = ReplayBuffer(state_dim, action_dim, cfg.SAC_REPLAY_SIZE)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.SAC_LR_PI)
    opt_q  = torch.optim.Adam(qnet.parameters(), lr=cfg.SAC_LR_Q)
    opt_qe = torch.optim.Adam(qens.parameters(), lr=cfg.SAC_LR_Q)

    log_alpha = torch.tensor(np.log(cfg.SAC_ALPHA_INIT), device=device,
                             dtype=torch.float32, requires_grad=cfg.SAC_AUTO_ALPHA)
    opt_alpha = (torch.optim.Adam([log_alpha], lr=cfg.SAC_LR_ALPHA)
                 if cfg.SAC_AUTO_ALPHA else None)
    target_h = float(cfg.SAC_TARGET_H)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.CKPT_DIR, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("iter,buffer_size,mean_R,mean_len,mean_T,success_rate,"
                "q_loss,pi_loss,alpha,entropy,wall,reasons\n")

    t_start = time.time()

    # ---- warmup: fill buffer with stochastic-policy samples ----
    warm_target = max(int(cfg.SAC_WARMUP_ROLLOUTS), cfg.BATCH_SIZE)
    print(f"[warmup] collecting {warm_target} rollouts before learning starts")
    while len(buffer) < warm_target:
        states_np, tasks = _sample_training_batch(env, cfg.BATCH_SIZE)
        s_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            a_t, _ = policy.act(s_t, deterministic=False)
        a_np = a_t.cpu().numpy().astype(np.float32)
        c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
        v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
        e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
        T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)
        out = batched_rollout(a_np, c_np, v_np, e_np, T_np)
        L_np = np.asarray(out["lengths"], dtype=np.float32)
        buffer.add_batch(states_np, a_np, L_np, T_np.astype(np.float32))

    for it in range(1, cfg.N_ITERS + 1):
        # ----- sample tasks (no active task selection — too slow due to
        # reachability filter; we do UCB at ACTION level instead) -----
        states_np, tasks = _sample_training_batch(env, cfg.BATCH_SIZE)
        s_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)

        # UCB action selection once Q ensemble has signal
        if (cfg.ACTIVE_SAMPLING
                and it > cfg.SAC_WARMUP_ROLLOUTS // cfg.BATCH_SIZE):
            a_t = _ucb_action_select(policy, qens, s_t,
                                     K=cfg.ACTIVE_K, kappa=1.0)
        else:
            with torch.no_grad():
                a_t, _ = policy.act(s_t, deterministic=False)
        a_np = a_t.cpu().numpy().astype(np.float32)
        c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
        v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
        e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
        T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)
        out = batched_rollout(a_np, c_np, v_np, e_np, T_np)
        L_np = np.asarray(out["lengths"], dtype=np.float32)
        reasons_iter = list(out.get("reasons", []))
        buffer.add_batch(states_np, a_np, L_np, T_np.astype(np.float32))

        # ----- Q updates (main critic + ensemble) -----
        q_loss_val = 0.0
        qens_loss_val = 0.0
        for _ in range(cfg.SAC_K_Q):
            s_b, a_b, r_b = buffer.sample(cfg.SAC_BATCH, device)
            # main critic (used for policy gradient)
            q_pred = qnet(s_b, a_b)
            q_loss = ((q_pred - r_b) ** 2).mean()
            opt_q.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), cfg.GRAD_CLIP)
            opt_q.step()
            q_loss_val = float(q_loss.item())

            # ensemble: each member sees an independent bootstrap subsample
            qe_preds = qens(s_b, a_b)            # (M, B)
            mask = (torch.rand(qens.m, s_b.shape[0], device=device) < 0.8).float()
            qe_loss = ((qe_preds - r_b.unsqueeze(0)) ** 2 * mask).sum() \
                      / mask.sum().clamp_min(1.0)
            opt_qe.zero_grad()
            qe_loss.backward()
            torch.nn.utils.clip_grad_norm_(qens.parameters(), cfg.GRAD_CLIP)
            opt_qe.step()
            qens_loss_val = float(qe_loss.item())

        # ----- policy updates -----
        pi_loss_val = 0.0
        ent_val = 0.0
        for _ in range(cfg.SAC_K_PI):
            s_b, _, _ = buffer.sample(cfg.SAC_BATCH, device)
            a_repar, _, log_p = policy.rsample(s_b)
            q_val = qnet(s_b, a_repar)
            alpha = log_alpha.exp().detach()
            pi_loss = (alpha * log_p - q_val).mean()
            opt_pi.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.GRAD_CLIP)
            opt_pi.step()
            pi_loss_val = float(pi_loss.item())
            ent_val = -float(log_p.mean().item())

            if cfg.SAC_AUTO_ALPHA:
                # train alpha to push entropy toward target_h
                alpha_loss = -(log_alpha * (log_p.detach() + target_h)).mean()
                opt_alpha.zero_grad()
                alpha_loss.backward()
                opt_alpha.step()

        # ----- log -----
        if it % cfg.LOG_EVERY == 0 or it == 1:
            mean_R = float((L_np / np.maximum(T_np, 1)).mean())
            mean_len = float(L_np.mean())
            mean_T = float(T_np.mean())
            succ = float((L_np >= T_np).mean())
            wall = time.time() - t_start
            ctr = collections.Counter(reasons_iter)
            reason_str = "|".join(f"{k}:{v}" for k, v in ctr.most_common(4))
            alpha_now = float(log_alpha.exp().item())
            print(f"[{it:5d}/{cfg.N_ITERS}] "
                  f"buf={len(buffer):>5d} R={mean_R:.3f} "
                  f"len={mean_len:5.1f}/{mean_T:5.1f} succ={succ:.2f} "
                  f"q={q_loss_val:.4f} qens={qens_loss_val:.4f} "
                  f"pi={pi_loss_val:+.3f} a={alpha_now:.4f} H={ent_val:+.2f} "
                  f"({wall:.1f}s) [{reason_str}]")
            with open(log_path, "a") as f:
                f.write(f"{it},{len(buffer)},{mean_R},{mean_len},{mean_T},"
                        f"{succ},{q_loss_val},{pi_loss_val},{alpha_now},"
                        f"{ent_val},{wall},{reason_str}\n")
            if wandb_run is not None:
                wandb_run.log({
                    "train/mean_reward": mean_R,
                    "train/mean_length": mean_len,
                    "train/mean_T": mean_T,
                    "train/success_rate": succ,
                    "loss/q": q_loss_val,
                    "loss/pi": pi_loss_val,
                    "policy/alpha": alpha_now,
                    "policy/entropy": ent_val,
                    "buffer/size": len(buffer),
                    "time/wall_sec": wall,
                }, step=it)

        if it % cfg.CKPT_EVERY == 0:
            ckpt_path = os.path.join(cfg.CKPT_DIR, f"ckpt_{it:06d}.pt")
            torch.save({
                "iter": it,
                "policy_type": cfg.POLICY_TYPE,
                "action_mode": cfg.ACTION_MODE,
                "mixture_components": getattr(policy, "n_components", 1),
                "policy": policy.state_dict(),
                "qnet": qnet.state_dict(),
                "qens": qens.state_dict(),
                "log_alpha": float(log_alpha.detach().cpu().item()),
                "state_dim": cfg.STATE_DIM,
                "log_std": (policy.log_std.detach().cpu().numpy()
                            if hasattr(policy, "log_std") else None),
            }, ckpt_path)
            if wandb_run is not None:
                wandb_run.save(ckpt_path, policy="now")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
