"""Search for an optimal symmetric dual-XArm7 mounting config for
tabletop manipulation (position-only reach).

Setup: two XArm7 arms hanging from a rig (a ceiling plate). Task: reach
the POSITION of every point on a tabletop grid (TCP orientation free).

Optimization: grid search over 3 symmetric mount parameters
(shoulder half-separation dy, outward roll tilt, rig height z),
scored by (#both-reachable) * mean_manipulability on the table grid.
Right arm is always the mirror of left across the XZ plane.

Collision: both arms are posed at the same target simultaneously, but the
two arms' wrist/flange links are allowed to overlap (excluded cross-pairs);
only arm-body collisions are forbidden.
"""
import itertools
import time
import numpy as np

import one.collider.mj_collider as ocm
import one.utils.constant as ouc
from one import oum, ovw, ossop, xarm7


TABLE_Z = 0.0
TABLE_X = np.linspace(0.10, 0.35, 6, dtype=np.float32)
TABLE_Y = np.linspace(-0.20, 0.20, 9, dtype=np.float32)
# how many trailing links per arm (wrist + flange) may overlap the other arm
N_EE_LINKS = 2


def left_tf(dy_half, tilt_roll_deg, rig_z):
    """Mount transform for the LEFT arm (on +Y side, hanging down).
    Base rotation: flip about X (arm points down), then outward roll
    by +tilt_roll about X, then no yaw/pitch."""
    R_flip = oum.rotmat_from_axangle(
        ouc.StandardAxis.X, np.pi)
    R_tilt = oum.rotmat_from_axangle(
        ouc.StandardAxis.X, np.deg2rad(tilt_roll_deg))
    R = R_tilt @ R_flip
    pos = np.array([0.0, dy_half, rig_z], dtype=np.float32)
    return R.astype(np.float32), pos


def right_tf(dy_half, tilt_roll_deg, rig_z):
    """Mirror of left across XZ plane (negate Y + flip sign of X-rotation
    components that induce Y motion)."""
    M = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    R_left, p_left = left_tf(dy_half, tilt_roll_deg, rig_z)
    R = M @ R_left @ M
    pos = M @ p_left
    return R.astype(np.float32), pos.astype(np.float32)


def score_config(robot_l, robot_r, mjc, tcp_z_above=0.03):
    """Return (n_both_reachable, mean_w, per_cell_w) over the table grid.
    A cell counts if BOTH arms reach the position (orientation free) and the
    two arms posed together do not collide -- except the wrist/flange links,
    which are allowed to overlap (excluded in ``mjc``).
    per_cell_w shape (len(X), len(Y)) with nan where unreachable."""
    per_cell = np.full((len(TABLE_X), len(TABLE_Y)),
                       np.nan, dtype=np.float32)
    ws = []
    n_both = 0
    for ix, x in enumerate(TABLE_X):
        for iy, y in enumerate(TABLE_Y):
            tgt_pos = np.array([x, y, TABLE_Z + tcp_z_above],
                               dtype=np.float32)
            # position-only IK (orientation free)
            _sl = robot_l.ik_partial(tgt_pos, max_solutions=1)
            ql = _sl[0] if _sl else None
            _sr = robot_r.ik_partial(tgt_pos, max_solutions=1)
            qr = _sr[0] if _sr else None
            if ql is None or qr is None:
                continue
            # body collision (wrist/flange cross-pairs excluded in mjc)
            if mjc is not None:
                if mjc.is_collided(np.concatenate([ql, qr])):
                    continue
            # manipulability = sqrt(det(J J^T)) — averaged over both arms
            wl = _manip(robot_l, ql)
            wr = _manip(robot_r, qr)
            w = 0.5 * (wl + wr)
            per_cell[ix, iy] = w
            ws.append(w)
            n_both += 1
    mean_w = float(np.mean(ws)) if ws else 0.0
    return n_both, mean_w, per_cell


def _manip(robot, qs):
    chain = robot.chain('main')
    jnts = chain.jnts
    n = len(jnts)
    root_tf = np.eye(4, dtype=np.float32)
    root_tf[:3, :3] = robot.rotmat
    root_tf[:3, 3] = robot.pos
    wd_lnk = np.empty((n + 1, 4, 4), dtype=np.float32)
    wd_lnk[0] = root_tf
    wd_jnt = np.empty((n, 4, 4), dtype=np.float32)
    for k in range(n):
        wd_jnt[k] = wd_lnk[k] @ jnts[k].zero_tf
        wd_lnk[k + 1] = wd_jnt[k] @ jnts[k].motion_tf(qs[k])
    tcp = wd_lnk[-1, :3, 3]
    jac = np.zeros((6, n), dtype=np.float32)
    for k in range(n):
        ax = wd_jnt[k, :3, :3] @ jnts[k].ax
        p_k = wd_jnt[k, :3, 3]
        jac[3:6, k] = ax
        jac[0:3, k] = np.cross(ax, tcp - p_k)
    d = float(np.linalg.det(jac @ jac.T))
    return float(np.sqrt(max(d, 0.0)))


def build_pair(dy_half, tilt_roll_deg, rig_z):
    """Create two XArm7 instances plus a self/mutual collision checker.
    The two arms' last ``N_EE_LINKS`` links (wrist + flange) are allowed to
    overlap each other so both can target the same point; arm bodies still
    collide normally."""
    R_l, p_l = left_tf(dy_half, tilt_roll_deg, rig_z)
    R_r, p_r = right_tf(dy_half, tilt_roll_deg, rig_z)
    left = xarm7.XArm7(pos=p_l, rotmat=R_l)
    right = xarm7.XArm7(pos=p_r, rotmat=R_r)
    mjc = ocm.MJCollider()
    mjc.append(left)
    mjc.append(right)
    mjc.actors = [left, right]
    # exclude cross-arm collision among the trailing wrist/flange links
    n_l = len(left.runtime_lnks)
    n_r = len(right.runtime_lnks)
    ee_excludes = [((left, i), (right, j))
                   for i in range(n_l - N_EE_LINKS, n_l)
                   for j in range(n_r - N_EE_LINKS, n_r)]
    mjc.compile(margin=0.0, extra_excludes=ee_excludes)
    return left, right, mjc


def search():
    dy_halfs = [0.10, 0.15, 0.20, 0.25, 0.30]
    tilts = [0.0, 15.0, 30.0, 45.0, 60.0]
    rig_zs = [0.25, 0.35, 0.45, 0.55]
    best = None
    print(f"grid: {len(dy_halfs)*len(tilts)*len(rig_zs)} configs, "
          f"{len(TABLE_X)*len(TABLE_Y)} table cells each")
    t0 = time.time()
    for dy_half, tilt, rig_z in itertools.product(
            dy_halfs, tilts, rig_zs):
        left, right, mjc = build_pair(dy_half, tilt, rig_z)
        n_both, mean_w, grid = score_config(left, right, mjc)
        score = n_both * mean_w
        tag = (f"dy={dy_half:.2f} tilt={tilt:4.1f} "
               f"rig_z={rig_z:.2f}")
        print(f"  {tag}  n_both={n_both:3d} meanW={mean_w:.5f}  "
              f"score={score:.5f}")
        if best is None or score > best["score"]:
            best = {"score": score, "n_both": n_both,
                    "mean_w": mean_w, "grid": grid,
                    "dy_half": dy_half, "tilt": tilt, "rig_z": rig_z}
    print(f"[done in {time.time()-t0:.1f}s]  best: "
          f"dy_half={best['dy_half']:.2f}  tilt={best['tilt']:.1f}  "
          f"rig_z={best['rig_z']:.2f}  score={best['score']:.5f}  "
          f"n_both={best['n_both']}")
    return best


def visualize(best):
    base = ovw.World(cam_pos=(1.8, 1.8, 1.6),
                     cam_lookat_pos=(0.25, 0.0, 0.2))
    # rig plate
    plate = ossop.box(xyz_lengths=(0.3, 2 * best["dy_half"] + 0.1, 0.02),
                      pos=(0.0, 0.0, best["rig_z"] + 0.02),
                      rgb=ouc.BasicColor.GRAY)
    plate.attach_to(base.scene)
    # table
    table = ossop.box(xyz_lengths=(0.5, 0.6, 0.01),
                      pos=(0.3, 0.0, TABLE_Z - 0.005),
                      rgb=(0.85, 0.78, 0.6))
    table.attach_to(base.scene)

    R_l, p_l = left_tf(best["dy_half"], best["tilt"], best["rig_z"])
    R_r, p_r = right_tf(best["dy_half"], best["tilt"], best["rig_z"])
    left = xarm7.XArm7(pos=p_l, rotmat=R_l)
    right = xarm7.XArm7(pos=p_r, rotmat=R_r)
    left.attach_to(base.scene)
    right.attach_to(base.scene)

    # pose both arms at a center reach for visual (position-only)
    ctr = np.array([0.3, 0.0, TABLE_Z + 0.15], dtype=np.float32)
    _sl = left.ik_partial(ctr, max_solutions=1)
    ql = _sl[0] if _sl else None
    _sr = right.ik_partial(ctr, max_solutions=1)
    qr = _sr[0] if _sr else None
    if ql is not None:
        left.fk(qs=ql)
    if qr is not None:
        right.fk(qs=qr)

    # heat map: scatter points over table cells colored by manip
    pts, rgbs = [], []
    w = best["grid"]
    finite = w[np.isfinite(w)]
    if finite.size:
        lo, hi = float(finite.min()), float(finite.max())
        for ix, x in enumerate(TABLE_X):
            for iy, y in enumerate(TABLE_Y):
                if not np.isfinite(w[ix, iy]):
                    continue
                t = (w[ix, iy] - lo) / max(hi - lo, 1e-9)
                pts.append((x, y, TABLE_Z + 0.01))
                rgbs.append((t, 0.0, 1.0 - t))
    if pts:
        ossop.point_cloud(np.asarray(pts, dtype=np.float32),
                          np.asarray(rgbs, dtype=np.float32),
                          alpha=0.9).attach_to(base.scene)
    else:
        print("[visualize] no reachable table cells in best config — "
              "widen search ranges or check task/table placement.")

    ossop.frame().attach_to(base.scene)
    base.run()


if __name__ == "__main__":
    best = search()
    visualize(best)
