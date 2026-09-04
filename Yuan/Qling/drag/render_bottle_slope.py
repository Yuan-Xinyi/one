"""Render a bottle plane-to-slope episode with the one viewer: table,
15-deg ramp, bottle driven by the grasp constraint, gripper, arm,
ghost goal bottle on the slope, breadcrumb trace on the surface.

Usage:
    /home/lqin/miniconda3/envs/one/bin/python \
        /home/lqin/one/Yuan/Qling/drag/render_bottle_slope.py \
        <traj.npz> <out.mp4> [cam_pos "x,y,z"] [lookat "x,y,z"]

npz: q (T,6), g_p (3), g_R (3,3), jaw (scalar) -- same contract as
render_bottle_traj plus nothing new; scene constants come from
BottleSlopeEnv.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, '/home/lqin/one')

import math                                          # noqa: E402
import numpy as np                                   # noqa: E402
import pyglet                                        # noqa: E402
import torch                                         # noqa: E402
from one import osso, ossop, ovw                     # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from drag.ijrr_root import add_ijrr_path             # noqa: E402
add_ijrr_path()
from Yuan.IJRR.kinematics.batched_chain_kin import (  # noqa: E402
    BatchedChainKinematics)
from drag.nova2_spec import NOVA2, GRIPPER_TCP_OFFSET  # noqa: E402
from drag.bottle_slope_env import BottleSlopeEnv     # noqa: E402

EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'compare_exp')
QLING_MESHES = os.path.join(os.path.dirname(__file__), '..', 'meshes')
WRS_MESHES = '/disk2/wrs_xinyi/wrs/robot_sim/manipulators/dobot_nova2/meshes'
ARM_LINKS = ['base_link0.stl', 'j1.stl', 'j2.stl', 'j3.stl',
             'j4.stl', 'j5.stl', 'j6.stl']
ACT_CENTER = 0.16
FPS_OUT = 15

THETA = BottleSlopeEnv.THETA          # overridden from the npz in main
Y_FOLD = BottleSlopeEnv.Y_FOLD


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


KNOTS = None


def surface_z(y):
    if KNOTS is not None:
        return float(np.interp(y, KNOTS[:, 0], KNOTS[:, 1]))
    return max(0.0, math.tan(THETA) * (y - Y_FOLD))


def main():
    traj_npz, out_mp4 = sys.argv[1], sys.argv[2]
    cam_pos = tuple(float(x) for x in sys.argv[3].split(',')) \
        if len(sys.argv) > 3 else (0.95, -0.10, 0.75)
    cam_look = tuple(float(x) for x in sys.argv[4].split(',')) \
        if len(sys.argv) > 4 else (-0.05, 0.50, 0.10)
    d = np.load(traj_npz)
    qs = torch.tensor(d['q'], dtype=torch.float32)
    g_p, g_R, jaw = d['g_p'], d['g_R'], float(d['jaw'])
    global THETA, KNOTS
    if 'theta_deg' in d:
        THETA = math.radians(float(d['theta_deg']))
    if 'knots' in d:
        KNOTS = d['knots']

    kin = BatchedChainKinematics(NOVA2, dtype=torch.float32,
                                 tcp_offset=GRIPPER_TCP_OFFSET)

    base = ovw.World(cam_pos=cam_pos,
                     cam_lookat_pos=cam_look,
                     win_size=(1280, 720))
    # table up to the fold; ramp beyond
    ossop.box(pos=(0, (Y_FOLD - 0.65) / 2 + 0.0, -0.015),
              half_extents=(0.85, (Y_FOLD + 0.65) / 2, 0.015),
              rgb=(0.82, 0.80, 0.76)).attach_to(base.scene)
    if KNOTS is not None:
        segs = []
        for k in range(len(KNOTS) - 1):
            (y0, z0k), (y1, z1k) = KNOTS[k], KNOTS[k + 1]
            if y1 < Y_FOLD - 0.2 or y0 > 1.0:
                continue
            segs.append((max(y0, Y_FOLD - 0.01), z0k
                         + (max(y0, Y_FOLD - 0.01) - y0)
                         * (z1k - z0k) / (y1 - y0),
                         min(y1, 0.95), z1k
                         + (min(y1, 0.95) - y1) * (z1k - z0k) / (y1 - y0)))
        for (y0, z0k, y1, z1k) in segs:
            if z0k == 0.0 and z1k == 0.0:
                continue          # table already drawn
            ln = math.hypot(y1 - y0, z1k - z0k)
            ang = math.atan2(z1k - z0k, y1 - y0)
            b = ossop.box(pos=(0, 0, 0),
                          half_extents=(0.6, ln / 2, 0.012),
                          rgb=(0.72, 0.74, 0.80))
            nrm = np.array([0.0, -math.sin(ang), math.cos(ang)])
            mid = np.array([0.0, (y0 + y1) / 2, (z0k + z1k) / 2])                 - 0.012 * nrm
            b.set_rotmat_pos(rotmat=rotx(ang), pos=mid)
            b.attach_to(base.scene)
    else:
        ramp_len = 0.45
        ct, st = math.cos(THETA), math.sin(THETA)
        ramp = ossop.box(pos=(0, 0, 0),
                         half_extents=(0.6, ramp_len / 2, 0.012),
                         rgb=(0.72, 0.74, 0.80))
        rc = np.array([0.0,
                       Y_FOLD + (ramp_len / 2) * ct + 0.012 * st,
                       (ramp_len / 2) * st - 0.012 * ct])
        ramp.set_rotmat_pos(rotmat=rotx(THETA), pos=rc)
        ramp.attach_to(base.scene)

    # ghost goal bottle: per-episode goal from the npz if present,
    # else the fixed env goal
    if 'goal_p' in d:
        goal_p, goal_R = d['goal_p'], d['goal_R']
    else:
        from drag.drag_env import DragEnvConfig
        env = BottleSlopeEnv(DragEnvConfig(
            n_envs=1, seed=0, device='cpu',
            slope_theta_deg=math.degrees(THETA)))
        goal_p = env.goal_p.numpy()
        goal_R = (rotx(THETA) @ env.init_R.numpy())
    goal_b = osso.SceneObject.from_file(
        os.path.join(EXP_DIR, 'bottle.stl'), rgb=(0.2, 0.55, 0.25),
        alpha=0.25)
    goal_b.set_rotmat_pos(rotmat=goal_R, pos=goal_p)
    goal_b.attach_to(base.scene)

    bottle = osso.SceneObject.from_file(
        os.path.join(EXP_DIR, 'bottle.stl'), rgb=(0.2, 0.55, 0.25))
    bottle.attach_to(base.scene)

    links = []
    for fname in ARM_LINKS:
        o = osso.SceneObject.from_file(os.path.join(WRS_MESHES, fname),
                                       rgb=(0.72, 0.72, 0.74))
        o.attach_to(base.scene)
        links.append(o)
    grip = osso.SceneObject.from_file(
        os.path.join(QLING_MESHES, 'base_v3.stl'), rgb=(0.35, 0.35, 0.38))
    grip.attach_to(base.scene)
    fingers, flocs = [], []
    for sgn, yawf in ((1.0, np.pi), (-1.0, 0.0)):
        lr = rotz(yawf)
        f = osso.SceneObject.from_file(
            os.path.join(QLING_MESHES, 'finger_v3.stl'),
            rgb=(0.3, 0.45, 0.75))
        f.attach_to(base.scene)
        fingers.append(f)
        flocs.append((lr, np.array([0.0, sgn * jaw / 2, 0.0])))

    tfs = kin.link_transforms(qs)
    p_all, R_all, _, _ = kin.tcp_fk_jac(qs)
    frames_dir = tempfile.mkdtemp(prefix='slope_frames_')
    hold = [0] * 8 + list(range(qs.shape[0])) + [qs.shape[0] - 1] * 12
    n_out = 0
    for t in hold:
        T = tfs[t].numpy()
        for i, o in enumerate(links):
            o.set_rotmat_pos(rotmat=T[i, :3, :3], pos=T[i, :3, 3])
        R_ee, p_ee = R_all[t].numpy(), p_all[t].numpy()
        base_pos = p_ee - R_ee @ np.array([0.0, 0.0, ACT_CENTER])
        grip.set_rotmat_pos(rotmat=R_ee, pos=base_pos)
        for f, (lr, lp) in zip(fingers, flocs):
            f.set_rotmat_pos(rotmat=R_ee @ lr, pos=base_pos + R_ee @ lp)
        R_obj = R_ee @ g_R.T
        p_obj = p_ee - R_obj @ g_p
        bottle.set_rotmat_pos(rotmat=R_obj, pos=p_obj)
        if t % 4 == 0:      # breadcrumb on the local surface
            zb = surface_z(float(p_obj[1])) + 0.005
            ossop.sphere(pos=(p_obj[0], p_obj[1], zb), radius=0.005,
                         rgb=(0.85, 0.25, 0.2)).attach_to(base.scene)
        base.switch_to()
        base.dispatch_events()
        base.on_draw()
        base.flip()
        pyglet.image.get_buffer_manager().get_color_buffer().save(
            os.path.join(frames_dir, f'f_{n_out:04d}.png'))
        n_out += 1
    base.close()
    subprocess.run(
        ['ffmpeg', '-y', '-framerate', str(FPS_OUT),
         '-i', os.path.join(frames_dir, 'f_%04d.png'),
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out_mp4],
        check=True, capture_output=True)
    print(f'wrote {out_mp4} ({n_out} frames), frames in {frames_dir}')


if __name__ == '__main__':
    main()
