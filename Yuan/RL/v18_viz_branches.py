"""Show distinct IK branches at a fixed TCP pose, in rainbow.

Strategy:
  1. Pick a goal TCP pose (default: pen-down at (0.4, 0, 0.55))
  2. Build 8 seed q's via reflect-around-joint-mid on (J1, J3, J5)
     - this ensures broad coverage of sign space at the seed level
  3. Run IK from each seed + random seeds -> all converge to same TCP
  4. Bin converged q's by signature (sign(J1), sign(J4), sign(J6))
  5. Show one representative per unique branch in rainbow

All visualized arms reach the same TCP target - that's the point of
"branches": same end-effector pose, kinematically distinct configs.

Usage:
    python -m Yuan.RL.v18_viz_branches
    python -m Yuan.RL.v18_viz_branches --pos 0.45 0.1 0.6
    python -m Yuan.RL.v18_viz_branches --tilt-deg 30
    python -m Yuan.RL.v18_viz_branches --M 256 --alpha 0.45
"""
from __future__ import annotations
import argparse, builtins, itertools
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v18_data_prep import _dense_ik_at


# rainbow 8-color palette
RAINBOW8 = [
    (0.95, 0.20, 0.20),   # red
    (1.00, 0.55, 0.10),   # orange
    (1.00, 0.90, 0.10),   # yellow
    (0.20, 0.80, 0.30),   # green
    (0.10, 0.85, 0.85),   # cyan
    (0.20, 0.45, 0.95),   # blue
    (0.65, 0.30, 0.90),   # purple
    (0.95, 0.45, 0.80),   # pink
]


JOINT_LIMITS = np.array([
    [-2.97, +2.97],
    [-1.83, +1.83],
    [-2.97, +2.97],
    [-3.05, -0.05],
    [-2.97, +2.97],
    [-0.27, +4.53],
    [-2.97, +2.97],
], dtype=np.float32)
JOINT_MID = (JOINT_LIMITS[:, 0] + JOINT_LIMITS[:, 1]) / 2.0


# template positioned to give large reflection deltas on (J1, J3, J5)
Q_TEMPLATE = np.array([1.40, -0.60, 1.00, -2.00, 1.20, 1.50, 0.00], dtype=np.float32)


def branch_signature(q):
    return (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))


def reflect_seeds(template: np.ndarray, joints: list[int]) -> np.ndarray:
    """Build 2^len(joints) seeds by reflecting `joints` around their range mid."""
    seeds = []
    for signs in itertools.product([+1, -1], repeat=len(joints)):
        q = template.copy()
        for j, s in zip(joints, signs):
            q[j] = JOINT_MID[j] + s * (template[j] - JOINT_MID[j])
        q = np.clip(q, JOINT_LIMITS[:, 0] + 1e-3, JOINT_LIMITS[:, 1] - 1e-3)
        seeds.append(q)
    return np.stack(seeds, axis=0)


DEFAULT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.0], dtype=np.float32)


def pair_complementary(reps: list[np.ndarray],
                        weights: np.ndarray = DEFAULT_WEIGHTS,
                        ) -> list[list[np.ndarray]]:
    """Greedy nearest-neighbor pairing in weighted joint space.
    Two reps that differ in only one joint will have the smallest pairwise
    distance, so the closest pair = the "single-flip" pair. Returns list of
    pairs (each a list of 1 or 2 reps); odd-one-out becomes a singleton."""
    n = len(reps)
    if n == 0:
        return []
    R = np.stack(reps, axis=0)                                  # (n, 7)
    d = np.linalg.norm((R[:, None, :] - R[None, :, :]) * weights, axis=-1)
    np.fill_diagonal(d, np.inf)
    used = np.zeros(n, dtype=bool)
    pairs: list[list[np.ndarray]] = []
    while not used.all():
        # mask out already-used rows/cols
        d_masked = d.copy()
        d_masked[used, :] = np.inf
        d_masked[:, used] = np.inf
        if np.isinf(d_masked).all():
            for i in np.where(~used)[0]:
                pairs.append([reps[i]])
                used[i] = True
            break
        i, j = np.unravel_index(np.argmin(d_masked), d_masked.shape)
        pairs.append([reps[int(i)], reps[int(j)]])
        used[i] = used[j] = True
    return pairs


def cluster_by_jointspace(q_np: np.ndarray, threshold: float = 1.5,
                           weights: np.ndarray | None = None) -> list[np.ndarray]:
    """Greedy cluster q's by *weighted* Euclidean distance in joint space.

    Default weights focus on joints that determine arm body shape and ignore
    joints that are pure nullspace at typical poses:
      J1 J2 J3 J4 J5  J6   J7
       1  1  1  1  1  0.3   0     <- J7 is pen-axis roll (invisible),
                                     J6 is mostly fixed by pen orientation
    so we collapse nullspace variation into the same cluster.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    clusters: list[list[np.ndarray]] = []
    for q in q_np:
        if not clusters:
            clusters.append([q]); continue
        dists = [float(np.linalg.norm((q - c[0]) * weights)) for c in clusters]
        i = int(np.argmin(dists))
        if dists[i] < threshold:
            clusters[i].append(q)
        else:
            clusters.append([q])
    return [c[0] for c in clusters]


def pen_down_rotation(tilt_deg: float = 0.0) -> np.ndarray:
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    if abs(tilt_deg) > 1e-6:
        c, s = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        R = Ry @ R
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=float, nargs=3, default=[0.40, 0.0, 0.55])
    ap.add_argument("--tilt-deg", type=float, default=0.0,
                    help="pen-down orientation tilt around y-axis (deg)")
    ap.add_argument("--M", type=int, default=256,
                    help="random IK seeds in addition to the reflect seeds")
    ap.add_argument("--alpha", type=float, default=0.50,
                    help="arm transparency (each pair has 2 overlaid arms)")
    ap.add_argument("--cluster-thresh", type=float, default=1.5,
                    help="weighted joint-space distance below which two q's "
                         "are same branch (J7 weight 0, J6 weight 0.3)")
    ap.add_argument("--spacing", type=float, default=1.5,
                    help="spacing along y between adjacent branch arms (m)")
    ap.add_argument("--overlay", action="store_true",
                    help="overlay all arms at the same base (transparent stack) "
                         "instead of laying them out side by side")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    target_pos_np = np.array(args.pos, dtype=np.float32)
    R_target_np = pen_down_rotation(args.tilt_deg)
    target_pos = torch.as_tensor(target_pos_np, device=device, dtype=torch.float32)
    R_target = torch.as_tensor(R_target_np, device=device, dtype=torch.float32)

    # 16 reflect-around-mid seeds on (J1, J3, J5, J7) - broader sign coverage
    extra_seeds = reflect_seeds(Q_TEMPLATE, joints=[0, 2, 4, 6])

    print(f"target pos: {target_pos_np}")
    print(f"target R (pen down, tilt={args.tilt_deg}°):\n{R_target_np}\n")
    print(f"injected {extra_seeds.shape[0]} reflect-mid seeds (J1/J3/J5/J7 sign combos) "
          f"+ {args.M} random seeds")

    q_set, _ = _dense_ik_at(kin, target_pos, R_target, args.M, rng,
                             extra_seeds=extra_seeds, mix_boundary=True)
    print(f"IK: {q_set.shape[0]} solutions converged to target")
    if q_set.shape[0] == 0:
        print("no IK solutions - target unreachable; try different --pos")
        return

    q_np = q_set.cpu().numpy()
    # joint-space clustering: each cluster = one true kinematic branch
    reps = cluster_by_jointspace(q_np, threshold=args.cluster_thresh)
    print(f"distinct clusters (joint-space dist < {args.cluster_thresh} rad): "
          f"{len(reps)}\n")

    # sanity: each cluster rep reaches the target
    print(f"{'i':>2}  {'sig (J1,J4,J6)':>16}  {'TCP err':>8}  q (rad)")
    for i, q in enumerate(reps):
        q_t = torch.as_tensor(q, device=device, dtype=torch.float32).unsqueeze(0)
        p_pred, _ = kin.fk_batch(q_t)
        err = float((p_pred.squeeze(0) - target_pos).norm().item())
        sig = branch_signature(q)
        q_str = "[" + " ".join(f"{x:+.2f}" for x in q) + "]"
        print(f"  {i}  {str(sig):>16}  {err:>8.4f}  {q_str}")

    # build viewer
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop
    from Yuan.RL.fr3_with_pen import make_fr3_with_pen, attach_pen_visual

    # pair up "single-flip" complementary branches
    pairs = pair_complementary(reps)
    n_pairs = len(pairs)
    print(f"\npaired into {n_pairs} group(s) of complementary branches:")
    for pi, pair in enumerate(pairs):
        if len(pair) == 1:
            print(f"  pair {pi}: singleton")
        else:
            qa, qb = pair[0], pair[1]
            diffs = np.abs(qa - qb)
            top_j = np.argsort(-diffs)[:3]
            print(f"  pair {pi}: differ mostly at "
                  + ", ".join(f"J{j+1}(Δ={diffs[j]:+.2f})" for j in top_j))

    if args.overlay:
        offsets = [np.zeros(3, dtype=np.float32) for _ in range(n_pairs)]
        cam_lookat = tuple(target_pos_np)
        cam_pos = (1.6, 1.2, 1.0)
    else:
        ys = (np.arange(n_pairs) - (n_pairs - 1) / 2.0) * args.spacing
        offsets = [np.array([0.0, float(y), 0.0], dtype=np.float32) for y in ys]
        cam_lookat = (float(target_pos_np[0]), 0.0, float(target_pos_np[2]))
        cam_pos = (2.5, 0.0, 1.8 + 0.3 * n_pairs)

    base = ovw.World(cam_pos=cam_pos,
                     cam_lookat_pos=cam_lookat,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # within each pair: rainbow_a / rainbow_b colors, both transparent and overlaid
    color_idx = 0
    for pi, pair in enumerate(pairs):
        for q in pair:
            rgb = RAINBOW8[color_idx % 8]
            color_idx += 1
            arm, _ = make_fr3_with_pen()
            arm.set_rotmat_pos(rotmat=np.eye(3, dtype=np.float32),
                                pos=offsets[pi])
            arm.attach_to(base.scene)
            arm.rgb = rgb
            arm.alpha = float(args.alpha)
            attach_pen_visual(arm, rgb=rgb, alpha=0.85)
            arm.fk(q)
        # one target sphere + base frame per pair slot
        ossop.sphere(pos=tuple(target_pos_np + offsets[pi]), radius=0.020,
                     rgb=(0.05, 0.05, 0.05), alpha=0.95).attach_to(base.scene)
        ossop.frame(pos=tuple(offsets[pi]),
                    length_scale=0.15, radius_scale=0.6).attach_to(base.scene)

    print("\n[control]: spin viewer with mouse, close window to exit")
    base.run()


if __name__ == "__main__":
    main()
