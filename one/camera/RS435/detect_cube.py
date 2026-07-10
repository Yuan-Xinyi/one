"""Detect a cube on the table from the calibrated D435, in the xArm7 BASE frame.

The eye-to-hand extrinsic ``T_base_cam`` (from ``register_camera_extrinsics.py``,
saved in ``camera_extrinsics.yaml``) maps the D435 colour-optical cloud into the
robot base frame: ``p_base = T_base_cam @ p_cam``. Because the calibration lives
in the base frame, everything here is base-frame too -- the same frame the arm's
IK / FK and ``l1cubepicking_xhand.py`` plan in. In that frame the table top is at
z ~= 0 and a 6 cm cube sits with its centre at z ~= +0.03.

Pipeline (all base-frame):
    1. capture one cloud, transform cam -> base with ``T_base_cam``;
    2. crop to the reachable workspace box in front of the robot;
    3. estimate the table height (dominant flat z band) -- or take it as given;
    4. keep points that stand ABOVE the table => the objects on it;
    5. Euclidean-cluster the object points (open3d DBSCAN);
    6. pick the cube-like cluster (footprint ~ cube size, height ~ cube size,
       optionally nearest a target colour), and report its centre + yaw.

``detect_cube_base(...)`` returns ``(center_xyz, yaw_rad, info)`` in the base frame
(``None`` if nothing cube-like is found). Yaw is the footprint's principal axis,
wrapped into [-45, 45) deg -- a cube face is 4-fold symmetric, and the antipodal
planner searches rolls anyway, so this is only a seed.

Run standalone to preview the detection (base-frame cloud + the fitted cube box):
    conda activate one
    python -m one.camera.RS435.detect_cube                 # live camera
    python -m one.camera.RS435.detect_cube --headless      # print pose only
"""
import os

import numpy as np
import numpy.typing as npt
import yaml

_THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(_THIS, 'camera_extrinsics.yaml')

CUBE_SIZE = 0.06                       # expected cube edge (m)

# Reachable table patch in front of the robot, BASE frame. The hardcoded cube in
# l1cubepicking_xhand.py sits at base ~= (0.35, 0.0, 0.03); this box brackets it
# generously while excluding the robot column (x<0.1) and the table far edge.
WORKSPACE = dict(x=(0.12, 0.62), y=(-0.35, 0.35), z=(-0.06, 0.20))

# Object-vs-table split and cluster tuning.
OBJ_Z_MARGIN = 0.012                   # a point this far above the table => object
CLUSTER_EPS = 0.015                    # DBSCAN neighbour radius (m)
CLUSTER_MIN_PTS = 25                   # DBSCAN min points per cluster


# ------------------------------------------------------------------ helpers ---
def load_extrinsics(path=DEFAULT_YAML):
    """Load ``T_base_cam`` (4x4) and the colour intrinsics dict from the yaml."""
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    T = np.asarray(d['T_base_cam'], dtype=np.float64)
    return T, d.get('color_intrinsics')


def apply_T(T, pts):
    """Apply a 4x4 homogeneous transform to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def capture_base_cloud(T_base_cam, z_near=0.15, z_far=2.0):
    """Grab one D435 cloud and return ``(pts_base(N,3), colors(N,3))`` in the base
    frame. Reuses ``register_camera_extrinsics.capture_realsense`` (same optical
    convention the extrinsic was calibrated with)."""
    from one.camera.RS435.register_camera_extrinsics import capture_realsense
    pts_cam, colors, _intr = capture_realsense(z_near, z_far)
    pts_base = apply_T(T_base_cam, pts_cam.astype(np.float64)).astype(np.float32)
    return pts_base, colors


def _crop(pts, colors, box):
    """Keep points inside an axis-aligned base-frame box ``dict(x,y,z=(lo,hi))``."""
    m = ((pts[:, 0] > box['x'][0]) & (pts[:, 0] < box['x'][1]) &
         (pts[:, 1] > box['y'][0]) & (pts[:, 1] < box['y'][1]) &
         (pts[:, 2] > box['z'][0]) & (pts[:, 2] < box['z'][1]))
    return pts[m], colors[m]


def estimate_table_z(pts, bins=80):
    """Table height = the densest z band of the cropped cloud (the table is the
    largest flat surface in the workspace). Robust to the small cube on top."""
    if len(pts) == 0:
        return 0.0
    z = pts[:, 2]
    hist, edges = np.histogram(z, bins=bins)
    k = int(np.argmax(hist))
    band = z[(z >= edges[k]) & (z <= edges[k + 1])]
    return float(np.median(band)) if len(band) else float(edges[k])


def _yaw_from_footprint(xy):
    """Principal-axis yaw of an (N,2) xy footprint, wrapped to [-pi/4, pi/4)."""
    c = xy - xy.mean(0)
    if len(c) < 3:
        return 0.0
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    ang = float(np.arctan2(vt[0, 1], vt[0, 0]))     # major axis direction
    return (ang + np.pi / 4) % (np.pi / 2) - np.pi / 4   # cube is 90-deg symmetric


# --------------------------------------------------------------- detection ---
def detect_cube_base(pts_base, colors, cube_size=CUBE_SIZE, workspace=WORKSPACE,
                     table_z=None, target_rgb=None):
    """Find the cube in a base-frame cloud.

    Returns ``(center_xyz(3,), yaw_rad, info)`` in the base frame, or ``None``.
    ``center`` z is pinned to ``table_z + cube_size/2`` (the cube rests on the
    table); x, y are the top-face centroid. ``target_rgb`` (optional 0-1 triple)
    biases cluster choice toward the cube's colour."""
    import open3d as o3d

    pts, cols = _crop(pts_base, colors, workspace)
    if len(pts) < CLUSTER_MIN_PTS:
        return None
    if table_z is None:
        table_z = estimate_table_z(pts)

    # points standing above the table surface = the objects on it
    obj = pts[:, 2] > table_z + OBJ_Z_MARGIN
    opts, ocols = pts[obj], cols[obj]
    if len(opts) < CLUSTER_MIN_PTS:
        return None

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(opts.astype(np.float64))
    labels = np.asarray(pc.cluster_dbscan(eps=CLUSTER_EPS,
                                          min_points=CLUSTER_MIN_PTS))
    if labels.max() < 0:
        return None

    best, best_score = None, None
    for lab in range(labels.max() + 1):
        m = labels == lab
        cpts, ccols = opts[m], ocols[m]
        lo, hi = cpts.min(0), cpts.max(0)
        fx, fy, h = float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])
        foot = max(fx, fy)
        # cube-likeness: footprint and height both near cube_size (a top-down view
        # sees mostly the top face, so height can read small -> only lightly weighted)
        score = abs(foot - cube_size) / cube_size + 0.3 * abs(h - cube_size) / cube_size
        # reject clearly non-cube blobs (a wall, a hand, the table edge)
        if foot > 2.5 * cube_size or foot < 0.3 * cube_size:
            continue
        if target_rgb is not None:
            score += float(np.linalg.norm(ccols.mean(0) - np.asarray(target_rgb)))
        if best_score is None or score < best_score:
            top = cpts[cpts[:, 2] > hi[2] - 0.015]          # top-face slab
            top = top if len(top) >= 3 else cpts
            center = np.array([top[:, 0].mean(), top[:, 1].mean(),
                               table_z + cube_size / 2], dtype=np.float32)
            best = dict(center=center, yaw=_yaw_from_footprint(top[:, :2]),
                        footprint=(fx, fy), height=h, n=int(m.sum()),
                        rgb=tuple(float(v) for v in ccols.mean(0)),
                        table_z=float(table_z))
            best_score = score
    if best is None:
        return None
    return best['center'], best['yaw'], best


# ------------------------------------------------------------------- preview ---
def _main():
    import argparse
    import one.utils.constant as ouc
    import one.utils.math as oum
    import one.scene.scene_object_primitive as ossop

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--yaml', default=DEFAULT_YAML, help='extrinsics yaml')
    ap.add_argument('--table-z', type=float, default=None,
                    help='force the table height (base-frame z, m); else estimated')
    ap.add_argument('--target-rgb', default=None,
                    help='cube colour bias, e.g. "0.85,0.2,0.2"')
    ap.add_argument('--headless', action='store_true', help='print pose, no window')
    args = ap.parse_args()

    T_base_cam, _ = load_extrinsics(args.yaml)
    target = (np.array([float(v) for v in args.target_rgb.split(',')])
              if args.target_rgb else None)

    print('[detect] capturing D435 cloud ...')
    pts, cols = capture_base_cloud(T_base_cam)
    print(f'[detect] {len(pts)} base-frame points')
    res = detect_cube_base(pts, cols, table_z=args.table_z, target_rgb=target)
    if res is None:
        print('[detect] no cube found -- check the workspace box / lighting')
        return
    center, yaw, info = res
    print(f'[detect] cube center (base) = [{center[0]:+.3f} {center[1]:+.3f} '
          f'{center[2]:+.3f}] m   yaw = {np.degrees(yaw):+.1f} deg')
    print(f'[detect] table_z = {info["table_z"]:+.3f} m   '
          f'footprint = {info["footprint"][0] * 1000:.0f} x '
          f'{info["footprint"][1] * 1000:.0f} mm   height = {info["height"] * 1000:.0f} mm   '
          f'pts = {info["n"]}   mean_rgb = '
          f'({info["rgb"][0]:.2f},{info["rgb"][1]:.2f},{info["rgb"][2]:.2f})')
    if args.headless:
        return

    import one.viewer.world as ovw
    base = ovw.World(cam_pos=(center[0] + 0.5, center[1] - 0.5, center[2] + 0.5),
                     cam_lookat_pos=tuple(float(v) for v in center))
    ossop.frame(length_scale=0.15).attach_to(base.scene)                 # base frame
    ossop.point_cloud(pts, cols).attach_to(base.scene)
    ossop.box(pos=tuple(float(v) for v in center),
              xyz_lengths=(CUBE_SIZE,) * 3,
              rotmat=oum.rotmat_from_axangle(ouc.StandardAxis.Z, yaw),
              rgb=(0.1, 0.9, 0.2), alpha=0.5).attach_to(base.scene)
    base.set_caption(f'cube @ base [{center[0]:.3f},{center[1]:.3f},{center[2]:.3f}] '
                     f'yaw {np.degrees(yaw):+.0f} deg')
    base.run()


if __name__ == '__main__':
    _main()
