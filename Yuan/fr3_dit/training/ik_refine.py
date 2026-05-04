"""IK-refinement helpers for the farsighted-IK eval protocol.

Predicted q0 is a seed — we refine it with classical IK targeting the exact
``path_start_xyz`` and pen-into-desk orientation, then run the tracker from the
refined q. This separates "did the model pick a good IK branch" (the actual
purpose of the DiT) from "did the q0 land on the right TCP" (which IK can fix).
"""
from __future__ import annotations

import numpy as np


def build_pen_target_rotmat(desk_normal: np.ndarray, first_seg_dir_world: np.ndarray) -> np.ndarray:
    """Construct the TCP-frame rotation matrix for the pen-down orientation.

    - z-axis = -desk_normal_unit  (pen points into the desk)
    - x-axis = projection of first_seg_dir onto the plane perpendicular to z, normalized
              (pen-frame x faces the upcoming stroke direction)
    - y-axis = z × x (right-handed)

    Returns a 3x3 rotation matrix whose columns are (x̂, ŷ, ẑ) in world coords.
    """
    z = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
    d = np.asarray(first_seg_dir_world, dtype=np.float64)
    x = d - z * float(np.dot(z, d))
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        # First seg parallel to desk normal (degenerate): fall back to a fixed perpendicular.
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = helper - z * float(np.dot(z, helper))
        x = x / max(float(np.linalg.norm(x)), 1e-12)
    else:
        x = x / nx
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float64)


def refine_q0_seed(
    pen_robot,
    q_seed: np.ndarray,
    target_pos: np.ndarray,
    target_rotmat: np.ndarray | None = None,
    desk_normal: np.ndarray | None = None,
    theta_max_deg: float = 30.0,
) -> tuple[np.ndarray, bool, dict]:
    """Run wrs IK from q_seed targeting (target_pos, ...).

    The IK target rotation defaults to **the seed's own TCP rotation** (preserving
    whatever orientation the model picked). This matches the project's data-gen
    convention: any TCP_z within ``theta_max_deg`` of ``-desk_normal`` is a valid
    pen-down orientation, so the model's orientation choice is part of the seed
    information that should be kept.

    Pass an explicit ``target_rotmat`` to override (e.g. for an oracle eval).
    If ``desk_normal`` is provided, the seed's TCP_z is checked against the cone
    and reported in info["seed_in_cone"] (informational only; IK still uses the
    seed's actual rotation as the target).

    Returns
    -------
    q_refined : np.ndarray (7,)   IK solution (or q_seed if IK fails).
    ok        : bool              True iff IK returned a non-None solution.
    info      : dict              tcp errors, ik status, z-axis diagnostics.
    """
    q_seed = np.asarray(q_seed, dtype=np.float32)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    # FK seed → record TCP, TCP_z; build target rotmat from seed if not provided.
    pen_robot.goto_given_conf(q_seed)
    tcp_seed = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    rot_seed = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)
    z_seed = rot_seed[:, 2]
    err_seed = float(np.linalg.norm(tcp_seed - target_pos))

    if target_rotmat is None:
        target_rotmat = rot_seed.copy()
    else:
        target_rotmat = np.asarray(target_rotmat, dtype=np.float64)

    seed_in_cone = None
    if desk_normal is not None:
        n = np.asarray(desk_normal, dtype=np.float64)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        cos_theta = float(np.dot(z_seed, -n))
        seed_in_cone = bool(cos_theta >= float(np.cos(np.deg2rad(theta_max_deg))))

    try:
        q_ref = pen_robot.ik(tgt_pos=target_pos, tgt_rotmat=target_rotmat,
                             seed_jnt_values=q_seed.astype(np.float64))
    except Exception as e:
        return q_seed.copy(), False, {
            "tcp_err_seed_m": err_seed, "tcp_err_refined_m": err_seed,
            "ik_failed": True, "ik_error": repr(e),
            "z_axis_seed": z_seed.tolist(), "seed_in_cone": seed_in_cone,
        }

    if q_ref is None:
        return q_seed.copy(), False, {
            "tcp_err_seed_m": err_seed, "tcp_err_refined_m": err_seed,
            "ik_failed": True, "ik_error": "ik returned None",
            "z_axis_seed": z_seed.tolist(), "seed_in_cone": seed_in_cone,
        }

    q_ref = np.asarray(q_ref, dtype=np.float32)
    pen_robot.goto_given_conf(q_ref)
    tcp_ref = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    err_ref = float(np.linalg.norm(tcp_ref - target_pos))
    z_ref = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)[:, 2]

    return q_ref, True, {
        "tcp_err_seed_m": err_seed, "tcp_err_refined_m": err_ref,
        "ik_failed": False,
        "z_axis_seed": z_seed.tolist(), "z_axis_refined": z_ref.tolist(),
        "seed_in_cone": seed_in_cone,
    }


def refine_batch(
    pen_robot,
    q_seeds: np.ndarray,                  # (N, 7)
    target_pos: np.ndarray,               # (3,)
    desk_normal: np.ndarray,              # (3,) — only for cone diagnostic in info dict
    theta_max_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Refine N candidates one-by-one, each preserving its own TCP orientation.

    The IK target rotation for candidate i is FK(q_seeds[i])'s TCP rotation, so
    refine = "slide TCP to target_pos while keeping the model's orientation
    choice". ``desk_normal`` is used only to flag whether each seed's TCP_z is
    inside the ``theta_max_deg`` cone (info["seed_in_cone"]).

    Returns (q_refined, ok_mask, per_cand_info).
    """
    q_out = q_seeds.astype(np.float32).copy()
    ok = np.zeros(q_seeds.shape[0], dtype=bool)
    info_list = []
    for i in range(q_seeds.shape[0]):
        q_ref, success, info = refine_q0_seed(
            pen_robot, q_seeds[i], target_pos,
            target_rotmat=None, desk_normal=desk_normal, theta_max_deg=theta_max_deg,
        )
        q_out[i] = q_ref
        ok[i] = success
        info_list.append(info)
    return q_out, ok, info_list
