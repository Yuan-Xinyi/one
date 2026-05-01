"""Side-by-side visualization: deterministic policy vs null-space oracle.

Picks a task where the oracle (best of K uniform (phi, psi) samples) beats
the deterministic policy by the largest margin. Runs both rollouts and
displays:
  - Two FR3 arms (policy = orange tint via TCP marker, oracle = green tint)
  - Two TCP traces (blue for policy, magenta for oracle)
  - The shared task path: green dots up to whichever rollout reached further,
    red dots beyond
  - Both arms animate in sync over their respective q_traj.

Usage:
    python -m Yuan.RL.visualize_oracle_vs_policy
    python -m Yuan.RL.visualize_oracle_vs_policy --k 1000 --n-tasks 32 --seed 12345
    python -m Yuan.RL.visualize_oracle_vs_policy --task-idx 14
"""
from __future__ import annotations
import argparse, os, builtins
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.batched_rollout import batched_rollout
from Yuan.RL.rollout import build_target_rotmat, rollout
from Yuan.RL.controller import DLSController
from Yuan.RL.visualize_rollout_world import build_branch_rotmat


def _load_policy(ckpt_path, env, device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


def _rollout_chunked(actions_np, c_np, v_np, e_np, T_np, chunk=4096):
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out["lengths"], dtype=np.int32)
    return L


def find_demo_task(policy, env, n_tasks, K, seed, only_idx: int | None = None):
    """Sample n_tasks; for each compare det policy vs best-of-K nullspace.

    If ``only_idx`` is given, the heavy K-oracle sweep is computed ONLY for
    that single task. Other tasks get L_best=0 / a_orc=zeros and should not
    be selected by gap/good modes — used when the caller already knows
    which demo to visualize.
    """
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)
    env.rng = np.random.default_rng(seed)

    tasks = env._sample_tasks(n_tasks)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        a_det, _ = policy.act(states_t, deterministic=True)
    a_det_np = a_det.cpu().numpy().astype(np.float32)
    L_det = _rollout_chunked(a_det_np, c_np, v_np, e_np, T_np)

    if only_idx is not None:
        # K oracle sweep on a single task — saves N-1 task's worth of rollouts
        c_one = c_np[only_idx:only_idx+1]
        v_one = v_np[only_idx:only_idx+1]
        e_one = e_np[only_idx:only_idx+1]
        T_one = T_np[only_idx:only_idx+1]
        phi = rng.uniform(0.0, 2*np.pi, size=K).astype(np.float32)
        psi = rng.uniform(0.0, 2*np.pi, size=K).astype(np.float32)
        a_orc_K = np.stack([np.cos(phi), np.sin(phi),
                            np.cos(psi), np.sin(psi)], axis=-1).astype(np.float32)
        L_K = _rollout_chunked(a_orc_K,
                               np.tile(c_one, (K, 1)),
                               np.tile(v_one, K),
                               np.tile(e_one, K),
                               np.tile(T_one, K))
        L_best = np.zeros(n_tasks, dtype=np.int32)
        L_best[only_idx] = int(L_K.max())
        a_orc_best = np.zeros((n_tasks, 4), dtype=np.float32)
        a_orc_best[only_idx] = a_orc_K[int(L_K.argmax())]
    else:
        # full sweep across all N tasks
        phi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        psi = rng.uniform(0.0, 2*np.pi, size=(K, n_tasks)).astype(np.float32)
        a_orc = np.stack([np.cos(phi), np.sin(phi),
                          np.cos(psi), np.sin(psi)], axis=-1)
        a_orc_flat = a_orc.reshape(K*n_tasks, 4).astype(np.float32)
        rep_c = np.tile(c_np, (K, 1))
        rep_v = np.tile(v_np, K)
        rep_e = np.tile(e_np, K)
        rep_T = np.tile(T_np, K)
        L_flat = _rollout_chunked(a_orc_flat, rep_c, rep_v, rep_e, rep_T)
        L_orc = L_flat.reshape(K, n_tasks)
        L_best = L_orc.max(axis=0)
        a_orc_best = a_orc[L_orc.argmax(axis=0), np.arange(n_tasks)]

    return {
        "tasks":   tasks,
        "T":       T_np,
        "L_det":   L_det,
        "L_best":  L_best,
        "a_det":   a_det_np,
        "a_orc":   a_orc_best,
    }


def run_serial_rollout(env, action, task):
    c = task["c"]
    return rollout(env.arm, action.astype(np.float32),
                   c[:3], c[3:6], c[6:9],
                   mjc=None,
                   max_steps=task["T"],
                   v_path=task["v_path"],
                   eps_p=task["eps_p"])


def tcp_positions(arm_for_fk, q_traj):
    if not q_traj:
        return np.zeros((0, 3), dtype=np.float32)
    ctrl = DLSController(arm_for_fk)
    mask = arm_for_fk._chain.active_mask
    pts = np.empty((len(q_traj), 3), dtype=np.float32)
    for t, q_full in enumerate(q_traj):
        q_active = q_full[mask].astype(np.float32)
        p_tcp, _, _ = ctrl.fk_with_jac(q_active)
        pts[t] = p_tcp.astype(np.float32)
    return pts


def visualize(env, task, a_pol, info_pol, a_orc, info_orc, fps=30.0):
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    c = task["c"]
    p0, d, n = c[:3], c[3:6], c[6:9]
    T = task["T"]
    v_path = task["v_path"]
    L_pol = info_pol["length"]
    L_orc = info_orc["length"]
    path_len = float(T) * cfg.DT * v_path

    # Two arm instances (each = FR3 + Franka hand + pen) so we can fk them
    # independently each tick. The aux arm is for offline TCP queries.
    arm_pol, _ = make_fr3_with_pen()
    arm_orc, _ = make_fr3_with_pen()
    aux, _ = make_fr3_with_pen()

    pts_pol = tcp_positions(aux, info_pol["q_traj"])
    pts_orc = tcp_positions(aux, info_orc["q_traj"])

    base = ovw.World(cam_pos=(1.5, 1.2, 1.0),
                     cam_lookat_pos=(p0 + 0.18 * d).tolist(),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    arm_pol.attach_to(base.scene)
    arm_orc.attach_to(base.scene)
    # tint + make both arms semi-transparent so the overlap is readable
    arm_pol.rgb = (0.20, 0.50, 0.95)     # blue-ish (matches policy trace)
    arm_orc.rgb = (0.95, 0.30, 0.55)     # magenta-ish (matches oracle trace)
    arm_pol.alpha = 0.35
    arm_orc.alpha = 0.35
    # pen sticks: tint to match each arm so the pen is visually paired
    attach_pen_visual(arm_pol, rgb=(0.20, 0.50, 0.95), alpha=0.85)
    attach_pen_visual(arm_orc, rgb=(0.95, 0.30, 0.55), alpha=0.85)
    arm_pol.toggle_tcp(length_scale=0.12, radius_scale=0.5)
    arm_orc.toggle_tcp(length_scale=0.12, radius_scale=0.5)
    ossop.frame(length_scale=0.20, radius_scale=0.8).attach_to(base.scene)

    # task plane + normal + path direction (light alpha)
    ossop.plane(pos=tuple(p0), normal=tuple(n), size=(0.40, 0.40),
                rgb=(0.55, 0.55, 0.6), alpha=0.10).attach_to(base.scene)
    ossop.arrow(spos=tuple(p0), epos=tuple(p0 + 0.18 * n),
                shaft_radius=0.005, head_radius=0.012, head_length=0.025,
                rgb=(0.95, 0.20, 0.85), alpha=0.65).attach_to(base.scene)
    ossop.arrow(spos=tuple(p0), epos=tuple(p0 + path_len * d),
                shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                rgb=(0.10, 0.80, 0.85), alpha=0.65).attach_to(base.scene)

    # target frames at p0 (one per branch — they may differ in tool roll)
    R_tgt_pol = build_branch_rotmat(d, n, a_pol)
    R_tgt_orc = build_branch_rotmat(d, n, a_orc)
    ossop.frame(pos=p0, rotmat=R_tgt_pol, length_scale=0.14,
                radius_scale=0.55).attach_to(base.scene)

    # path dots: gray; overlay the policy and oracle reached prefixes
    for t in range(0, T + 1, max(1, T // 80)):
        p = p0 + t * cfg.DT * v_path * d
        ossop.sphere(pos=tuple(p), radius=0.0025,
                     rgb=(0.55, 0.55, 0.55), alpha=0.4).attach_to(base.scene)

    # Two TCP traces — line of small spheres (kept opaque so the trace stays
    # readable even though the arms are translucent)
    for t, p in enumerate(pts_pol):
        ossop.sphere(pos=tuple(p),
                     radius=0.0045 if t % 5 == 0 else 0.0028,
                     rgb=(0.10, 0.40, 0.95), alpha=1.0).attach_to(base.scene)
    for t, p in enumerate(pts_orc):
        ossop.sphere(pos=tuple(p),
                     radius=0.0045 if t % 5 == 0 else 0.0028,
                     rgb=(0.95, 0.20, 0.55), alpha=1.0).attach_to(base.scene)

    # End-of-trace markers (opaque)
    if len(pts_pol):
        ossop.sphere(pos=tuple(pts_pol[-1]), radius=0.012,
                     rgb=(0.05, 0.20, 0.85), alpha=1.0).attach_to(base.scene)
    if len(pts_orc):
        ossop.sphere(pos=tuple(pts_orc[-1]), radius=0.012,
                     rgb=(0.95, 0.10, 0.40), alpha=1.0).attach_to(base.scene)

    # Print status
    print()
    print(f"=== task: T={T} steps  ({path_len:.2f} m path) ===")
    print(f"  policy  : L={L_pol:>4d}  reason={info_pol['reason']}")
    print(f"            branch={np.array2string(a_pol, precision=3, suppress_small=True)}")
    print(f"  oracle  : L={L_orc:>4d}  reason={info_orc['reason']}")
    print(f"            branch={np.array2string(a_orc, precision=3, suppress_small=True)}")
    print(f"  ratio L_pol / L_orc = {L_pol / max(L_orc, 1):.3f}")
    print()
    print("legend: BLUE trace = policy TCP, MAGENTA trace = oracle TCP")
    print("        cyan arrow = path direction d, magenta arrow = surface normal n")

    # initial fk
    if info_pol["q_traj"]:
        arm_pol.fk(info_pol["q_traj"][0])
    else:
        arm_pol.fk(arm_pol.home_qs)
    if info_orc["q_traj"]:
        arm_orc.fk(info_orc["q_traj"][0])
    else:
        arm_orc.fk(arm_orc.home_qs)

    qpol = info_pol["q_traj"]
    qorc = info_orc["q_traj"]
    n_frames = max(len(qpol), len(qorc), 1)
    idx = [0]

    def tick(_dt):
        i = idx[0]
        if qpol:
            arm_pol.fk(qpol[min(i, len(qpol) - 1)])
        if qorc:
            arm_orc.fk(qorc[min(i, len(qorc) - 1)])
        idx[0] = (i + 1) % n_frames

    base.schedule_interval(tick, interval=1.0 / fps)
    base.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--n-tasks", type=int, default=32,
                    help="how many tasks to scan to pick the demo")
    ap.add_argument("--k", type=int, default=1000,
                    help="number of uniform (phi, psi) samples per task")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--task-idx", type=int, default=None,
                    help="override: pick this task index directly")
    ap.add_argument("--mode", choices=["gap", "good", "list"], default="gap",
                    help="gap: biggest oracle-policy length gap among feasible tasks; "
                         "good: highest ratio among feasible tasks; "
                         "list: print candidates and exit")
    ap.add_argument("--min-frac", type=float, default=0.10,
                    help="for --mode good: require L_best/T_max(K) >= this so the "
                         "demo isn't a tiny path; oracle-anchored, not absolute")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dry", action="store_true",
                    help="run rollouts only, skip the viewer")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    print(f"loading {args.ckpt}")
    policy = _load_policy(args.ckpt, env, device)

    # If --task-idx is given, skip the full N-task oracle sweep — saves
    # N-1 tasks' worth of rollouts (huge for N=200).
    only_idx = args.task_idx if (args.task_idx is not None
                                 and args.mode != "list") else None
    out = find_demo_task(policy, env, args.n_tasks, args.k, args.seed,
                         only_idx=only_idx)
    L_det, L_best, T_arr = out["L_det"], out["L_best"], out["T"]
    ratio = L_det.astype(np.float64) / np.maximum(L_best, 1)

    # Mark tasks where the oracle itself is degenerate so we can exclude them
    # from selection. ratio is meaningless when L_best == 0 (no feasible
    # nullspace solution found) or when L_best is trivially small.
    feasible = L_best > 0
    # how big the best oracle managed relative to its task — proxy for
    # "is this task interesting enough to demo at all"
    oracle_frac = L_best.astype(np.float64) / np.maximum(T_arr, 1)

    if args.mode == "list" or args.task_idx is None:
        rows = sorted(range(len(T_arr)),
                      key=lambda i: (-ratio[i] if feasible[i] else 1e9,
                                     -L_best[i]))
        print("\n  i   T   L_det L_best ratio  oracle/T  comment")
        for i in rows:
            tag = ""
            if not feasible[i]:
                tag = "infeasible (L_best=0)"
            elif ratio[i] >= 0.99:
                tag = "policy = oracle"
            elif ratio[i] <= 0.01:
                tag = "policy collapsed"
            print(f"  {i:>2d} {T_arr[i]:>4d} {L_det[i]:>5d} {L_best[i]:>5d} "
                  f"{ratio[i]:>5.2f}  {oracle_frac[i]:>7.2f}   {tag}")
        if args.mode == "list":
            return

    if args.task_idx is not None:
        idx = int(args.task_idx)
        if idx < 0 or idx >= len(T_arr):
            raise SystemExit(
                f"--task-idx {idx} is out of range for --n-tasks {args.n_tasks}. "
                f"Pass --n-tasks {idx + 1} or larger to include this task.")
    elif args.mode == "good":
        # require: feasible AND oracle covers at least min_frac of T
        # (so the demo isn't a tiny path even the oracle barely tackled).
        # Then pick highest ratio, tie-break by longest L_best.
        candidates = np.where(feasible & (oracle_frac >= args.min_frac))[0]
        if candidates.size == 0:
            print(f"no feasible task with oracle/T >= {args.min_frac}; "
                  f"falling back to all feasible")
            candidates = np.where(feasible)[0]
        idx = int(candidates[np.lexsort((-L_best[candidates],
                                         -ratio[candidates]))[0]])
    else:  # gap — biggest absolute gap is OK as a *visual* selector since
           # both legs are runs of the same task; this is not an evaluation
           # metric, just "where is the contrast most striking".
        gap = np.where(feasible, L_best - L_det, -1)
        idx = int(np.argmax(gap))

    print(f"\n--- selected task #{idx}: L_det={L_det[idx]}  L_oracle_best={L_best[idx]}  "
          f"ratio={ratio[idx]:.3f} ---")

    task = out["tasks"][idx]
    info_pol = run_serial_rollout(env, out["a_det"][idx], task)
    info_orc = run_serial_rollout(env, out["a_orc"][idx], task)
    print(f"  policy serial rollout: L={info_pol['length']}/{task['T']}  reason={info_pol['reason']}  q_traj_len={len(info_pol['q_traj'])}")
    print(f"  oracle serial rollout: L={info_orc['length']}/{task['T']}  reason={info_orc['reason']}  q_traj_len={len(info_orc['q_traj'])}")
    if args.dry:
        return
    visualize(env, task,
              out["a_det"][idx], info_pol,
              out["a_orc"][idx], info_orc,
              fps=args.fps)


if __name__ == "__main__":
    main()
