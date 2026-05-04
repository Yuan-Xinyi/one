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


def _shape_rewards(lengths_np: np.ndarray,
                   T_np: np.ndarray,
                   reasons: list[str],
                   oracle_lengths_np: np.ndarray | None = None) -> np.ndarray:
    if oracle_lengths_np is None:
        denom = np.maximum(T_np.astype(np.float32), 1.0)
    else:
        denom = np.maximum(oracle_lengths_np.astype(np.float32),
                           float(cfg.REWARD_ORACLE_MIN_STEPS))
    reward = lengths_np.astype(np.float32) / denom
    penalty = np.zeros_like(reward, dtype=np.float32)
    reason_arr = np.asarray(reasons, dtype=object)
    penalty[reason_arr == "init_ik_fail"] += float(cfg.REWARD_FAIL_INIT_IK)
    penalty[reason_arr == "joint_limit"] += float(cfg.REWARD_FAIL_JOINT_LIMIT)
    penalty[reason_arr == "self_collision"] += float(cfg.REWARD_FAIL_SELF_COLLISION)
    penalty[reason_arr == "orient_err_exceeded"] += float(cfg.REWARD_FAIL_ORIENT)
    penalty[reason_arr == "pos_err_exceeded"] += float(cfg.REWARD_FAIL_POS)
    reward = reward - penalty
    return np.clip(reward, float(cfg.REWARD_CLIP_LO),
                   float(cfg.REWARD_CLIP_HI)).astype(np.float32)


def _sample_policy_action_batch(policy,
                                states_t: torch.Tensor,
                                n_samples: int) -> torch.Tensor:
    actions = []
    for _ in range(int(n_samples)):
        with torch.no_grad():
            a_t, _ = policy.act(states_t, deterministic=False)
        actions.append(a_t)
    return torch.stack(actions, dim=0)


def _collect_rollout_samples(policy,
                             states_np: np.ndarray,
                             tasks: list[dict],
                             device: torch.device):
    states_t = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    n_samples = max(1, int(cfg.SAC_ACTION_SAMPLES_PER_TASK))
    a_stack = _sample_policy_action_batch(policy, states_t, n_samples)
    batch_size = states_np.shape[0]
    a_np = a_stack.reshape(n_samples * batch_size, -1).cpu().numpy().astype(np.float32)

    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"] for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"] for t in tasks], dtype=np.int32)

    states_rep = np.tile(states_np, (n_samples, 1)).astype(np.float32)
    c_rep = np.tile(c_np, (n_samples, 1)).astype(np.float32)
    v_rep = np.tile(v_np, n_samples).astype(np.float32)
    e_rep = np.tile(e_np, n_samples).astype(np.float32)
    T_rep = np.tile(T_np, n_samples).astype(np.int32)

    out = batched_rollout(a_np, c_rep, v_rep, e_rep, T_rep)
    L_np = np.asarray(out["lengths"], dtype=np.float32)
    reasons = list(out.get("reasons", []))
    oracle_L_rep = None
    if bool(cfg.REWARD_USE_SAMPLED_ORACLE):
        L_grid = L_np.reshape(n_samples, batch_size)
        oracle_L = L_grid.max(axis=0).astype(np.float32)
        oracle_L_rep = np.tile(oracle_L, n_samples).astype(np.float32)
    r_np = _shape_rewards(L_np, T_rep, reasons, oracle_lengths_np=oracle_L_rep)
    return states_rep, a_np, L_np, T_rep, r_np, reasons, oracle_L_rep


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
    buffer = ReplayBuffer(state_dim, action_dim, cfg.SAC_REPLAY_SIZE)

    opt_pi = torch.optim.Adam(policy.parameters(), lr=cfg.SAC_LR_PI)
    opt_q  = torch.optim.Adam(qnet.parameters(), lr=cfg.SAC_LR_Q)

    # v13: train Q ensemble for deploy-time uncertainty-aware ranking
    # (mean - lambda*std). Decoupled from cfg.ACTIVE_SAMPLING — that flag
    # controlled v9-style task selection which ablated to a regression;
    # the ensemble itself is useful for q_ranked at deploy.
    qens = None
    opt_qe = None
    if cfg.ACTIVE_SAMPLING or getattr(cfg, "TRAIN_Q_ENSEMBLE", False):
        qens = QEnsemble(state_dim, action_dim, m=cfg.Q_ENSEMBLE_M).to(device)
        opt_qe = torch.optim.Adam(qens.parameters(), lr=cfg.SAC_LR_Q)

    log_alpha = torch.tensor(np.log(cfg.SAC_ALPHA_INIT), device=device,
                             dtype=torch.float32, requires_grad=cfg.SAC_AUTO_ALPHA)
    opt_alpha = (torch.optim.Adam([log_alpha], lr=cfg.SAC_LR_ALPHA)
                 if cfg.SAC_AUTO_ALPHA else None)
    target_h = float(cfg.SAC_TARGET_H)

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    log_path = os.path.join(cfg.CKPT_DIR, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("iter,buffer_size,mean_R,mean_raw_R,mean_oracle_raw_R,mean_len,mean_T,success_rate,"
                "q_loss,pi_loss,alpha,entropy,wall,reasons\n")

    t_start = time.time()

    # ---- warmup: fill buffer with stochastic-policy samples ----
    warm_target = max(int(cfg.SAC_WARMUP_ROLLOUTS), cfg.BATCH_SIZE)
    print(f"[warmup] collecting {warm_target} rollouts before learning starts")
    while len(buffer) < warm_target:
        states_np, tasks = _sample_training_batch(env, cfg.BATCH_SIZE)
        s_rep, a_np, L_np, T_np, r_np, _, _ = _collect_rollout_samples(
            policy, states_np, tasks, device)
        buffer.add_batch(s_rep, a_np, L_np, T_np.astype(np.float32), r_np)

    for it in range(1, cfg.N_ITERS + 1):
        # ----- sample tasks (no active task selection — too slow due to
        # reachability filter; we do UCB at ACTION level instead) -----
        states_np, tasks = _sample_training_batch(env, cfg.BATCH_SIZE)
        s_rep, a_np, L_np, T_np, r_np, reasons_iter, oracle_L_rep = _collect_rollout_samples(
            policy, states_np, tasks, device)
        buffer.add_batch(s_rep, a_np, L_np, T_np.astype(np.float32), r_np)

        # ----- Pairwise ranking-loss inputs from the FRESH K samples -----
        # _collect_rollout_samples returns rows tiled as
        #   row k*B + i = sample k of task i,   for k in [0,K), i in [0,B)
        # → reshape to (K, B, ...) then transpose to (B, K, ...) so that
        #   each row of the second axis is one task's K policy samples.
        rank_K = int(cfg.SAC_ACTION_SAMPLES_PER_TASK)
        rank_B = int(cfg.BATCH_SIZE)
        s_grid = torch.as_tensor(s_rep,  dtype=torch.float32, device=device
                                 ).view(rank_K, rank_B, -1).transpose(0, 1)
        a_grid = torch.as_tensor(a_np,   dtype=torch.float32, device=device
                                 ).view(rank_K, rank_B, -1).transpose(0, 1)
        r_grid = torch.as_tensor(r_np,   dtype=torch.float32, device=device
                                 ).view(rank_K, rank_B).transpose(0, 1)
        # shapes now: s_grid (B, K, S), a_grid (B, K, A), r_grid (B, K)

        # PER beta annealing: linearly ramp from PER_BETA -> PER_BETA_FINAL
        per_beta = float(cfg.PER_BETA)
        if cfg.PER_ENABLE and cfg.PER_BETA_ANNEAL_END > 0:
            t = min(1.0, it / float(cfg.PER_BETA_ANNEAL_END))
            per_beta = cfg.PER_BETA + t * (cfg.PER_BETA_FINAL - cfg.PER_BETA)

        # ----- Q updates -----
        q_loss_val = 0.0
        q_rank_val = 0.0
        qens_loss_val = 0.0
        rank_w = float(getattr(cfg, "Q_RANK_LOSS_WEIGHT", 0.0))
        rank_margin = float(getattr(cfg, "Q_RANK_MARGIN", 0.05))
        for _ in range(cfg.SAC_K_Q):
            s_b, a_b, r_b, idx_np, w_b = buffer.sample(
                cfg.SAC_BATCH, device, beta=per_beta)
            q_pred = qnet(s_b, a_b)
            td_err = q_pred - r_b
            # IS-weighted MSE; with PER off, w_b is all-ones so equiv. to mean
            q_loss_mse = (w_b * td_err ** 2).mean()

            # ----- Pairwise ranking loss on FRESH K samples -----
            # Forward Q on (B*K) flat, reshape to (B, K), then within each
            # task compare every (i, j) pair: sign(r_i - r_j) should match
            # sign(Q_i - Q_j). Hinge with margin.
            if rank_w > 0:
                q_rank = qnet(s_grid.reshape(-1, s_grid.shape[-1]),
                              a_grid.reshape(-1, a_grid.shape[-1])
                              ).view(rank_B, rank_K)
                q_diff = q_rank.unsqueeze(2) - q_rank.unsqueeze(1)   # (B,K,K)
                r_diff = r_grid.unsqueeze(2) - r_grid.unsqueeze(1)   # (B,K,K)
                sign_r = torch.sign(r_diff)
                # only train on pairs with non-trivial reward gap
                pair_mask = (r_diff.abs() > rank_margin * 0.5).float()
                hinge = torch.relu(rank_margin - sign_r * q_diff)
                rank_loss = (hinge * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)
                q_rank_val = float(rank_loss.item())
                q_loss = q_loss_mse + rank_w * rank_loss
            else:
                q_loss = q_loss_mse

            opt_q.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), cfg.GRAD_CLIP)
            opt_q.step()
            q_loss_val = float(q_loss_mse.item())

            # refresh priorities of the just-sampled rows with their |TD-err|
            buffer.update_priorities(
                idx_np, td_err.detach().abs().cpu().numpy())

            # ----- Q ensemble update (mirrors single Q's loss) -----
            if qens is not None:
                qe_preds = qens(s_b, a_b)
                mask = (torch.rand(qens.m, s_b.shape[0], device=device) < 0.8).float()
                qe_mse = ((qe_preds - r_b.unsqueeze(0)) ** 2 * mask).sum() \
                          / mask.sum().clamp_min(1.0)
                qe_loss = qe_mse
                if rank_w > 0:
                    # ensemble pairwise loss: average per-member ranking loss
                    qe_rank = qens(s_grid.reshape(-1, s_grid.shape[-1]),
                                   a_grid.reshape(-1, a_grid.shape[-1])
                                   ).view(qens.m, rank_B, rank_K)
                    qe_diff = qe_rank.unsqueeze(3) - qe_rank.unsqueeze(2)
                    # broadcast r_diff/sign_r across members
                    r_diff_b = r_diff.unsqueeze(0)
                    sign_r_b = sign_r.unsqueeze(0)
                    pair_mask_b = pair_mask.unsqueeze(0)
                    qe_hinge = torch.relu(rank_margin - sign_r_b * qe_diff)
                    qe_rank_loss = (qe_hinge * pair_mask_b).sum() / \
                                   (pair_mask_b.sum() * qens.m).clamp_min(1.0)
                    qe_loss = qe_mse + rank_w * qe_rank_loss
                opt_qe.zero_grad()
                qe_loss.backward()
                torch.nn.utils.clip_grad_norm_(qens.parameters(), cfg.GRAD_CLIP)
                opt_qe.step()
                qens_loss_val = float(qe_mse.item())

        # ----- policy updates -----
        pi_loss_val = 0.0
        ent_val = 0.0
        for _ in range(cfg.SAC_K_PI):
            s_b, _, _, _, _ = buffer.sample(cfg.SAC_BATCH, device,
                                            beta=per_beta)
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
            raw_R = L_np / np.maximum(T_np, 1)
            if oracle_L_rep is None:
                oracle_raw_R = np.ones_like(raw_R, dtype=np.float32)
            else:
                oracle_raw_R = oracle_L_rep / np.maximum(T_np, 1)
            mean_R = float(r_np.mean())
            mean_raw_R = float(raw_R.mean())
            mean_oracle_raw_R = float(oracle_raw_R.mean())
            mean_len = float(L_np.mean())
            mean_T = float(T_np.mean())
            succ = float((L_np >= T_np).mean())
            wall = time.time() - t_start
            ctr = collections.Counter(reasons_iter)
            reason_str = "|".join(f"{k}:{v}" for k, v in ctr.most_common(4))
            alpha_now = float(log_alpha.exp().item())
            print(f"[{it:5d}/{cfg.N_ITERS}] "
                  f"buf={len(buffer):>5d} R={mean_R:.3f} raw={mean_raw_R:.3f} "
                  f"orc={mean_oracle_raw_R:.3f} "
                  f"len={mean_len:5.1f}/{mean_T:5.1f} succ={succ:.2f} "
                  f"q={q_loss_val:.4f} qens={qens_loss_val:.4f} "
                  f"pi={pi_loss_val:+.3f} a={alpha_now:.4f} H={ent_val:+.2f} "
                  f"({wall:.1f}s) [{reason_str}]")
            with open(log_path, "a") as f:
                f.write(f"{it},{len(buffer)},{mean_R},{mean_raw_R},{mean_oracle_raw_R},"
                        f"{mean_len},{mean_T},"
                        f"{succ},{q_loss_val},{pi_loss_val},{alpha_now},"
                        f"{ent_val},{wall},{reason_str}\n")
            if wandb_run is not None:
                wandb_run.log({
                    "train/mean_reward": mean_R,
                    "train/mean_raw_reward": mean_raw_R,
                    "train/mean_oracle_raw_reward": mean_oracle_raw_R,
                    "train/policy_vs_sampled_oracle": (
                        mean_raw_R / max(mean_oracle_raw_R, 1e-6)),
                    "train/mean_length": mean_len,
                    "train/mean_T": mean_T,
                    "train/success_rate": succ,
                    "loss/q": q_loss_val,
                    "loss/q_rank": q_rank_val,
                    "loss/q_ensemble": qens_loss_val,
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
                "qens": qens.state_dict() if qens is not None else None,
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
