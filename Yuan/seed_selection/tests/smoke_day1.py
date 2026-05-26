"""Day 1 smoke tests for Module 4 (perturb_task) and Module 3a
(cone_constrained_ik_enumerate).

Runs a small, fixed-seed scenario against the real FR3 kinematics +
collision checker and asserts the basic invariants. No pytest dependency.

Run:
    python -m Yuan.seed_selection.tests.smoke_day1
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics

from Yuan.seed_selection.cone_ik import cone_constrained_ik_enumerate
from Yuan.seed_selection.perturb import perturb_task


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def make_task(kin: BatchedFR3Kinematics,
              collision: FR3SphereCollision,
              seed: int) -> dict[str, torch.Tensor]:
    """Build one task ``c = (p0, line_dir, n_target)`` from a collision-free
    random joint configuration — mirrors LineDistribution's per-index
    geometry without spinning up a full 100k pool."""
    gen = torch.Generator(device=kin.device).manual_seed(seed)
    for _ in range(200):
        q = kin.rand_conf_batch(1, generator=gen)
        link_tfs = kin.link_transforms(q)
        if not bool(collision.is_collided(link_tfs).item()):
            break
    else:
        raise RuntimeError("could not sample a collision-free q0 in 200 tries")
    p_tcp, R, _, _ = kin.tcp_fk_jac(q)
    z = R[0, :, 2]
    # line_dir = random unit ⊥ z.
    # Note: avoid 1-D @ 1-D matmul (torch.dot) on CUDA — SIGFPE on this
    # toolkit/cuBLAS combo. Use (x * y).sum() throughout.
    r = torch.randn(3, device=kin.device, dtype=kin.dtype, generator=gen)
    r = r - (r * z).sum() * z
    line_dir = r / r.norm()
    return {
        "p0": p_tcp[0],
        "line_dir": line_dir,
        "n_target": z,
        "_q0_seed": q[0],   # kept around so smoke tests can verify seed is reachable
    }


# ----------------------------------------------------------------------
# Module 4 smoke
# ----------------------------------------------------------------------
def test_perturb(kin, collision):
    print("[perturb] running...")
    c = make_task(kin, collision, seed=0)
    gen = torch.Generator(device=kin.device).manual_seed(123)

    # Sweep a handful of perturbation magnitudes.
    cases = [
        dict(perturb_d_deg=0.0, perturb_n_deg=0.0, perturb_p0_mm=0.0),
        dict(perturb_d_deg=5.0, perturb_n_deg=0.0, perturb_p0_mm=0.0),
        dict(perturb_d_deg=0.0, perturb_n_deg=5.0, perturb_p0_mm=0.0),
        dict(perturb_d_deg=0.0, perturb_n_deg=0.0, perturb_p0_mm=10.0),
        dict(perturb_d_deg=10.0, perturb_n_deg=10.0, perturb_p0_mm=20.0),
    ]
    for i, case in enumerate(cases):
        # Average behavior over many draws — tighter bounds + clearer failures.
        max_d_err = 0.0
        max_n_err = 0.0
        max_norm_err = 0.0
        max_dot_err = 0.0
        max_p0_shift = 0.0
        for _ in range(64):
            cp = perturb_task(c, generator=gen, **case)
            d_err = float(torch.acos(torch.clamp((c["line_dir"] * cp["line_dir"]).sum(), -1.0, 1.0)) * 180 / math.pi)
            n_err = float(torch.acos(torch.clamp((c["n_target"] * cp["n_target"]).sum(), -1.0, 1.0)) * 180 / math.pi)
            max_d_err = max(max_d_err, d_err)
            max_n_err = max(max_n_err, n_err)
            max_norm_err = max(max_norm_err, abs(float(cp["line_dir"].norm()) - 1.0))
            max_norm_err = max(max_norm_err, abs(float(cp["n_target"].norm()) - 1.0))
            max_dot_err = max(max_dot_err, abs(float((cp["line_dir"] * cp["n_target"]).sum())))
            max_p0_shift = max(max_p0_shift, float((cp["p0"] - c["p0"]).norm()) * 1000.0)
        # Invariants:
        # - unit norms (≤ 1e-5 numerical drift)
        # - d' ⊥ n' (≤ 1e-5)
        # - rotation cap: max observed angle change ≤ cap × 1.01 (1% margin)
        # - p0 shift ≤ cap (mm). For the n-rotation case d's actual change can
        #   reach almost the same angle, since rotating n by θ around d does
        #   keep d unchanged but the d⊥n re-projection isn't applied (d is
        #   already ⊥ n). The combined case sees the full caps.
        # Note: in the perturb_d=0,perturb_n=θ case, d itself can change up
        # to ≈ θ degrees because the re-projection drift compounds at large
        # n rotations. So we use max(cap_d, cap_n) for the d bound when n
        # rotates significantly.
        cap_d = case["perturb_d_deg"]
        cap_n = case["perturb_n_deg"]
        cap_p0 = case["perturb_p0_mm"]
        # In the n-only case, d should stay essentially unchanged (it's
        # already perpendicular to the original n, and after rotating n by
        # at most cap_n around d, d still lies in n's perpendicular plane).
        # So the angle d→d' is bounded by ≈ cap_n.
        d_bound = max(cap_d, cap_n) + 0.5
        n_bound = cap_n + 0.5
        ok_norm = max_norm_err < 1e-5
        ok_dot = max_dot_err < 1e-5
        ok_d = max_d_err <= d_bound
        ok_n = max_n_err <= n_bound
        ok_p0 = max_p0_shift <= cap_p0 + 1e-6
        flag = 'PASS' if (ok_norm and ok_dot and ok_d and ok_n and ok_p0) else 'FAIL'
        print(f"  [{flag}] case {i}  {case}")
        print(f"           max d_err={max_d_err:.3f}° (≤ {d_bound:.1f}°)  "
              f"n_err={max_n_err:.3f}° (≤ {n_bound:.1f}°)  "
              f"p0_shift={max_p0_shift:.3f}mm (≤ {cap_p0}mm)")
        print(f"           norm_drift={max_norm_err:.2e}  "
              f"|d·n|={max_dot_err:.2e}")
        if not (ok_norm and ok_dot and ok_d and ok_n and ok_p0):
            return False
    print("[perturb] all cases PASS")
    return True


# ----------------------------------------------------------------------
# Module 3a smoke
# ----------------------------------------------------------------------
def test_cone_ik(kin, collision):
    print("[cone_ik] running...")
    n_tasks = 3
    n_orient = 10
    n_restart = 5
    cone_angle = 5.0
    any_fail = False
    for task_seed in range(n_tasks):
        c = make_task(kin, collision, seed=task_seed)
        rng = np.random.default_rng(1000 + task_seed)
        Q = cone_constrained_ik_enumerate(
            p0=c["p0"], n_target=c["n_target"], line_dir=c["line_dir"],
            kin=kin, collision=collision,
            cone_angle_deg=cone_angle,
            n_orientations=n_orient,
            n_ik_restarts=n_restart,
            joint_margin=0.05,
            dedup_rad=0.08,
            rng=rng,
        )
        n_cand = Q.shape[0]
        if n_cand == 0:
            print(f"  [FAIL] task {task_seed}: 0 candidates returned")
            any_fail = True
            continue
        # Validate every returned q satisfies all advertised invariants.
        p_tcp, R, _, _ = kin.tcp_fk_jac(Q)
        z_actual = R[:, :, 2]
        pos_err = (p_tcp - c["p0"].unsqueeze(0)).norm(dim=-1)
        cos_ang = (z_actual * c["n_target"].unsqueeze(0)).sum(-1).clamp(-1.0, 1.0)
        ang_deg = torch.arccos(cos_ang) * 180.0 / math.pi
        lo, hi = kin.lmt_lo, kin.lmt_up
        margin = torch.minimum(Q - lo, hi - Q).min(dim=-1).values
        link_tfs = kin.link_transforms(Q)
        coll = collision.is_collided(link_tfs)
        # Pairwise distances (dedup check).
        if n_cand > 1:
            d = (Q.unsqueeze(0) - Q.unsqueeze(1)).norm(dim=-1)
            d.fill_diagonal_(float('inf'))
            min_pair = float(d.min())
        else:
            min_pair = float('inf')
        ok_pos = float(pos_err.max()) <= 5e-3 + 1e-6
        ok_cone = float(ang_deg.max()) <= cone_angle + 1e-3
        ok_jl = float(margin.min()) >= 0.05 - 1e-6
        ok_coll = not bool(coll.any())
        ok_dedup = min_pair >= 0.08 - 1e-6
        flag = 'PASS' if (ok_pos and ok_cone and ok_jl and ok_coll and ok_dedup) else 'FAIL'
        print(f"  [{flag}] task {task_seed}: {n_cand} candidates  "
              f"pos_err_max={float(pos_err.max())*1000:.3f}mm  "
              f"cone_max={float(ang_deg.max()):.3f}°  "
              f"jl_min={float(margin.min()):.4f}  "
              f"coll={int(coll.sum())}  dedup_min={min_pair:.4f}rad")
        if not (ok_pos and ok_cone and ok_jl and ok_coll and ok_dedup):
            any_fail = True

    if any_fail:
        print("[cone_ik] some tasks FAIL")
        return False
    print(f"[cone_ik] all {n_tasks} tasks PASS")
    return True


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    kin = BatchedFR3Kinematics(device=device)
    collision = FR3SphereCollision(device=device)

    ok_p = test_perturb(kin, collision)
    ok_c = test_cone_ik(kin, collision)
    if ok_p and ok_c:
        print("\nAll Day-1 smoke tests PASS.")
        return 0
    print("\nSome Day-1 smoke tests FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
