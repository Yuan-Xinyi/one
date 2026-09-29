"""Pick a target in simulation, then run it on the xArm7.

1. Reads the current joint angles from the arm (or a simulated arm).
2. Generates K straight-line targets from the current pen tip: K directions
   evenly spaced in the plane orthogonal to the cone axis (default: the
   current tool axis).  Every target is rolled out in simulation with the
   flagship controller from the current configuration, which gives the
   predicted stroke, the stop reason and the predicted motion.
3. Opens the `one` viewer: solid arm = the real configuration, grey ray =
   the target, blue = the predicted pen path and the predicted end posture
   (translucent).  Candidates are sorted by predicted stroke, best first.

      <- / ->  or  Up / Down    previous / next target
      Enter                     execute the shown target on the arm
      R                         re-read the joints and regenerate targets
      C                         re-frame the camera (mouse: orbit / pan / zoom)
      Esc / Q                   quit (stops the arm first)

   During execution the solid arm follows the measured joints and the
   measured pen path is drawn in red; when the run ends the targets are
   regenerated from the new configuration.

    python -m Yuan.IJRR.deploy.xarm7_pick_and_run --ip 192.168.1.203 \
        --normal 0 0 -1 --stroke 0.8 --time-scale 0.5
    python -m Yuan.IJRR.deploy.xarm7_pick_and_run --sim --sim-task 17

The control loop is the one of xarm7_realtime.py (joint-velocity mode,
predictive and measured-state termination, velocity timeout).  It runs in
its own process, which owns the arm connection, so the viewer's rendering
cannot steal cycles from it; the viewer receives one message per cycle.
"""
from __future__ import annotations

import argparse
import builtins
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib  # noqa: F401  (import before torch)
matplotlib.use('Agg')
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PEN_LEN = 0.10
COL_PRED = (0.24, 0.42, 0.88)
COL_MEAS = (0.85, 0.15, 0.15)
COL_RAY = (0.55, 0.57, 0.62)
MESH_DIR = REPO / 'one/robots/manipulators/xarm/xarm7/meshes'


def unit(v):
    v = np.asarray(v, np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def rot_with_z(z):
    z = unit(z)
    h = np.array([1.0, 0.0, 0.0])
    if abs(h @ z) > 0.95:
        h = np.array([0.0, 1.0, 0.0])
    x = unit(h - (h @ z) * z)
    return np.stack([x, np.cross(z, x), z], 1).astype(np.float32)


def plane_basis(n):
    n = unit(n)
    h = np.array([1.0, 0.0, 0.0])
    if abs(h @ n) > 0.95:
        h = np.array([0.0, 1.0, 0.0])
    e1 = unit(h - (h @ n) * n)
    return e1, np.cross(n, e1)


# ------------------------------------------------------ robot process
def server_main(conn, args):
    """Owns the arm and a CPU controller; serves 'q', 'run', 'stop', 'quit'."""
    import torch
    torch.set_num_threads(1)
    from Yuan.IJRR.deploy.xarm7_realtime import (Controller, SimArm, XArm,
                                                 ASSETS, run_loop,
                                                 tool_xyz_from_args)
    ctrl = Controller(args.period, tool_xyz_from_args(args), args.cone,
                      args.k_lateral, 'cpu')
    try:
        if args.sim:
            if args.sim_q_deg is not None:
                q0 = np.radians(args.sim_q_deg)
            else:
                tz = np.load(ASSETS / 'tasks_pool_xarm7.npz')
                q0 = tz['q0_seed'][args.sim_task].astype(np.float64)
            arm = SimArm(q0, lag=args.sim_lag)
        else:
            arm = XArm(args.ip)
    except Exception as e:                  # noqa: BLE001
        conn.send(('error', f'cannot connect to the arm at {args.ip}: {e}'))
        return
    conn.send(('ready',))
    try:
        while True:
            msg = conn.recv()
            if msg[0] == 'q':
                conn.send(('q', arm.q()))
            elif msg[0] == 'run':
                _, d, n, stroke, ts, cap, log = msg

                def on_cycle(q, st, cmd):
                    conn.send(('cycle', q, st['p'], st['progress'], st['lateral'],
                               st['tilt_deg'], float(np.abs(cmd).max())))
                try:
                    prog, reason = run_loop(arm, ctrl, d, n, stroke, ts, cap, log,
                                            on_cycle=on_cycle)
                    conn.send(('done', float(prog), reason))
                except Exception as e:      # noqa: BLE001
                    try:
                        arm.stop()
                    except Exception:       # noqa: BLE001
                        pass
                    conn.send(('done', float('nan'), f'ERROR {e}'))
            elif msg[0] == 'stop':
                arm.stop()
                conn.send(('stopped',))
            elif msg[0] == 'quit':
                break
    finally:
        try:
            arm.stop()
        finally:
            arm.close()


class RemoteArm:
    def __init__(self, args):
        ctx = mp.get_context('spawn')
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=server_main, args=(child, args), daemon=True)
        self.proc.start()
        m = self.conn.recv()
        if m[0] != 'ready':
            raise SystemExit(f'[pick] {m[1]}')

    def q(self):
        self.conn.send(('q',))
        tag, q = self.conn.recv()
        assert tag == 'q'
        return q

    def run(self, d, n, stroke, ts, cap, log):
        self.conn.send(('run', np.asarray(d), np.asarray(n), float(stroke),
                        float(ts), float(cap), log))

    def poll(self):
        out = []
        while self.conn.poll():
            out.append(self.conn.recv())
        return out

    def stop(self):
        self.conn.send(('stop',))
        t0 = time.time()
        while time.time() - t0 < 5.0:
            m = self.conn.recv()
            if m[0] == 'stopped':
                return

    def close(self):
        try:
            self.conn.send(('quit',))
        except Exception:                   # noqa: BLE001
            pass
        self.proc.join(5.0)


# ------------------------------------------------------ candidate rollout
def simulate_candidates(ctrl, q0, dirs, n, stroke, device):
    """Roll every direction from q0 with the flagship (paper convention:
    25 ms sub-steps, action held for two).  Returns per-candidate stroke,
    stop reason, joint trajectory and tip path."""
    import dataclasses
    import torch
    from Yuan.IJRR.env.env import NSRLBatchedEnv, TERM_NAMES
    from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
    K = len(dirs)
    cfg = dataclasses.replace(ctrl.env.cfg, n_envs=K, dt=0.025,
                              max_steps=int(2 * stroke / (0.2 * 0.025)) + 40)
    env = NSRLBatchedEnv(cfg, None, torch.device(device))
    ag = ctrl.agent.to(torch.device(device))
    rdt = env.kin.dtype
    t = lambda x: torch.as_tensor(np.asarray(x, np.float64), dtype=rdt, device=env.device)
    env.line_dist = ScriptedLineDistribution({
        'q0': t(np.tile(q0, (K, 1))), 'line_dir': t(dirs),
        'n_target': t(np.tile(unit(n), (K, 1)))})
    with torch.no_grad():
        env.reset()
        Q = [env.q.cpu().numpy().copy()]
        P = [env.kin.tcp_fk_jac(env.q)[0].cpu().numpy().copy()]
        reason = np.full(K, -1, np.int64)
        for _ in range(cfg.max_steps // 2):
            a = ag.actor_mean(env.current_obs())
            for _ in range(2):
                _, _, _, _, info = env.step(a, auto_reset=False)
                tr = info['term_reason'].cpu().numpy()
                reason = np.where((reason < 0) & (tr > 0), tr, reason)
            Q.append(env.q.cpu().numpy().copy())
            P.append(env.kin.tcp_fk_jac(env.q)[0].cpu().numpy().copy())
            if bool(env.done_persistent.all()) or bool(
                    (env.arc_progress >= stroke).all()):
                break
    Q, P = np.stack(Q, 1), np.stack(P, 1)          # (K, T, 7), (K, T, 3)
    prog = env.arc_progress.float().cpu().numpy()
    ag.to(ctrl.dev)
    out = []
    for k in range(K):
        along = (P[k] - P[k, 0]) @ unit(dirs[k])
        lim = min(prog[k], stroke) - 1e-6
        end = int(np.argmax(along >= lim)) + 1 if along.max() >= lim else len(along)
        why = ('stroke target' if prog[k] >= stroke else
               TERM_NAMES.get(int(reason[k]), 'alive'))
        out.append(dict(d=unit(dirs[k]), stroke=float(min(prog[k], stroke)),
                        reason=why, q=Q[k, :end], tip=P[k, :end]))
    del env
    return out


# ---------------------------------------------------------------- viewer
def build_app(args):
    import pyglet
    import pyglet.gl as gl
    import torch
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from Yuan.IJRR.deploy.xarm7_realtime import Controller, tool_xyz_from_args

    class MeshArm:
        """xArm7 link meshes (URDF visuals, identity visual origins) placed
        with the same chain kinematics the controller uses."""

        def __init__(self, kin, scene, rgb, alpha):
            import trimesh
            self.kin = kin
            self.objs = []
            for nm in ['link_base'] + [f'link{i}' for i in range(1, 8)]:
                m = trimesh.load(str(MESH_DIR / f'{nm}.stl'), force='mesh')
                o = ossop.mesh(np.asarray(m.vertices, np.float32),
                               np.asarray(m.faces, np.uint32), rgb=rgb, alpha=alpha)
                o.attach_to(scene)
                self.objs.append(o)

        def fk(self, q):
            qt = torch.as_tensor(np.asarray(q, np.float64), dtype=self.kin.dtype,
                                 device=self.kin.device)[None]
            T = self.kin.link_transforms(qt)[0].cpu().numpy()
            for o, Ti in zip(self.objs, T):
                o.set_rotmat_pos(Ti[:3, :3].astype(np.float32),
                                 Ti[:3, 3].astype(np.float32))

    class PickWorld(ovw.World):
        def __init__(self, app, **kw):
            super().__init__(**kw)
            self.app = app
            self.label = pyglet.text.Label('', x=12, y=self.height - 12,
                                           anchor_x='left', anchor_y='top',
                                           width=self.width - 24, multiline=True,
                                           font_size=13, color=(20, 20, 20, 255))

        def on_draw(self):
            super().on_draw()
            gl.glDisable(gl.GL_DEPTH_TEST)
            self.label.y = self.height - 12
            self.label.width = self.width - 24
            self.label.draw()
            gl.glEnable(gl.GL_DEPTH_TEST)

        def on_key_press(self, symbol, modifiers):
            k = pyglet.window.key
            if symbol in (k.LEFT, k.UP):
                self.app.select(-1)
            elif symbol in (k.RIGHT, k.DOWN):
                self.app.select(+1)
            elif symbol in (k.ENTER, k.RETURN):
                self.app.execute()
            elif symbol == k.R:
                self.app.regenerate()
            elif symbol == k.C:
                self.app.frame_camera()
            elif symbol in (k.ESCAPE, k.Q):
                self.app.quit()
                return pyglet.event.EVENT_HANDLED

    class App:
        def __init__(self):
            self.args = args
            torch.set_num_threads(1)
            self.tool = tool_xyz_from_args(args)
            self.ctrl = Controller(args.period, self.tool, args.cone, args.k_lateral, 'cpu')
            self.arm = RemoteArm(args)
            self.running = False
            self.result = None
            self.status = ''
            self.meas_tips = []
            self.cands, self.sel = [], 0
            self.q_now = self.arm.q()
            self._report_pose('startup')
            self.n = self._cone_axis()
            self.world = PickWorld(self, cam_pos=(1.5, 1.5, 1.2),
                                   cam_lookat_pos=(0.3, 0.0, 0.3), win_size=(1400, 850))
            builtins.base = self.world
            self.scene = self.world.scene
            ossop.frame(axis_length=0.15).attach_to(self.scene)
            self.robot = MeshArm(self.ctrl.env.kin, self.scene, (0.86, 0.86, 0.89), 1.0)
            self.tool_len = float(self.tool[2])
            self.pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, self.tool_len), radius=0.007,
                                      rgb=(0.1, 0.1, 0.1), alpha=0.98)
            self.pen.attach_to(self.scene)
            self.ghost = MeshArm(self.ctrl.env.kin, self.scene, COL_PRED, 0.35)
            self.ghost_pen = ossop.cylinder(spos=(0, 0, 0), epos=(0, 0, self.tool_len),
                                            radius=0.007, rgb=COL_PRED, alpha=0.5)
            self.ghost_pen.attach_to(self.scene)
            self.dyn = []
            self.meas_ink = None
            self._set_robot(self.q_now)
            self.regenerate(first=True)
            self.frame_camera()
            self.world.schedule_interval(self.tick, interval=1 / 20.0)

        # ---- helpers
        def _report_pose(self, tag):
            tip, R = self.ctrl.fk(self.q_now)
            print(f'[pick] {tag}: joints read from the arm (deg) '
                  f'{np.degrees(self.q_now).round(1).tolist()}; tool tip {tip.round(3).tolist()} m, '
                  f'tool axis {R[:, 2].round(3).tolist()}', flush=True)

        def _cone_axis(self):
            return (unit(args.normal) if args.normal is not None
                    else self.ctrl.fk(self.q_now)[1][:, 2])

        def _set_robot(self, q):
            self.robot.fk(q)
            tip, R = self.ctrl.fk(np.asarray(q, np.float64))
            z = R[:, 2]
            self.pen.set_rotmat_pos(rot_with_z(z), (tip - self.tool_len * z).astype(np.float32))

        def _clear_dyn(self):
            for o in self.dyn:
                o.detach_from(self.scene)
            self.dyn = []

        def _show_candidate(self):
            self._clear_dyn()
            if not self.cands:
                self._update_label()
                return
            c = self.cands[self.sel]
            p0 = c['tip'][0]
            ray = np.stack([p0 + c['d'] * s for s in np.linspace(0, args.stroke, 40)])
            o = ossop.linsegs(np.stack([ray[:-1], ray[1:]], 1), radius=0.002,
                              srgbs=np.float32(COL_RAY), alpha=0.6)
            o.attach_to(self.scene); self.dyn.append(o)
            if len(c['tip']) > 1:
                o = ossop.linsegs(np.stack([c['tip'][:-1], c['tip'][1:]], 1), radius=0.004,
                                  srgbs=np.float32(COL_PRED), alpha=0.95)
                o.attach_to(self.scene); self.dyn.append(o)
            q_end = c['q'][-1]
            self.ghost.fk(q_end)
            tip, R = self.ctrl.fk(np.asarray(q_end, np.float64))
            z = R[:, 2]
            self.ghost_pen.set_rotmat_pos(rot_with_z(z), (tip - self.tool_len * z).astype(np.float32))
            self._update_label()

        def _update_label(self):
            if self.running:
                self.world.label.text = (f'EXECUTING target {self.sel + 1}/{len(self.cands)}   '
                                         f'{self.status}')
                return
            lines = []
            if self.result is not None:
                lines.append(f'last run: {self.result}')
            if self.cands:
                c = self.cands[self.sel]
                lines.append(f'target {self.sel + 1}/{len(self.cands)}   direction '
                             f'({c["d"][0]:+.2f}, {c["d"][1]:+.2f}, {c["d"][2]:+.2f})   '
                             f'predicted stroke {c["stroke"]:.3f} m   stop: {c["reason"]}')
            else:
                lines.append('no feasible target from this configuration')
            lines.append('Left/Right: previous/next    Enter: execute    R: regenerate    '
                         'C: re-frame    Esc: quit')
            self.world.label.text = '\n'.join(lines)

        def frame_camera(self):
            tip0, _ = self.ctrl.fk(self.q_now)
            look = 0.5 * (tip0 + np.array([0.0, 0.0, 0.30]))
            away = unit(np.array([tip0[0], tip0[1], 0.0]) + np.array([1e-3, 0, 0]))
            u = unit(0.9 * away + 0.55 * np.array([0, 0, 1.0]) - 0.35 * self.n)
            dist = 1.6 * (args.stroke + 0.6)
            self.world.camera.set_to((look + dist * u).astype(np.float32),
                                     look.astype(np.float32))

        # ---- actions
        def regenerate(self, first=False):
            if self.running:
                return
            if not first:
                self.q_now = self.arm.q()
                self._report_pose('regenerate')
                self._set_robot(self.q_now)
                self.n = self._cone_axis()
            e1, e2 = plane_basis(self.n)
            K = args.n_cands
            ang = np.arange(K) * 2 * math.pi / K
            dirs = np.stack([math.cos(a) * e1 + math.sin(a) * e2 for a in ang])
            t0 = time.perf_counter()
            cands = simulate_candidates(self.ctrl, self.q_now, dirs, self.n,
                                        args.stroke, args.sim_device)
            cands.sort(key=lambda c: -c['stroke'])
            self.cands, self.sel = cands, 0
            print(f'[pick] {K} targets simulated in {time.perf_counter() - t0:.1f}s: '
                  + ', '.join(f'{c["stroke"]:.2f}' for c in cands), flush=True)
            self._show_candidate()

        def select(self, step):
            if self.running or not self.cands:
                return
            self.sel = (self.sel + step) % len(self.cands)
            self._show_candidate()

        def execute(self):
            if self.running or not self.cands:
                return
            c = self.cands[self.sel]
            log = None
            if args.log_dir is not None:
                Path(args.log_dir).mkdir(parents=True, exist_ok=True)
                log = str(Path(args.log_dir) / f'run_{time.strftime("%Y%m%d_%H%M%S")}.npz')
            if self.meas_ink is not None:
                self.meas_ink.detach_from(self.scene); self.meas_ink = None
            self.meas_tips = []
            self.running, self.result, self.status = True, None, 'starting'
            self.exec_cand = c
            self.arm.run(c['d'], self.n, args.stroke, args.time_scale, args.qd_cap, log)
            self._update_label()

        def quit(self):
            try:
                self.arm.stop()
            finally:
                self.world.close()
                pyglet.app.exit()

        # ---- periodic view update (main thread only)
        def tick(self, _dt):
            finished = False
            for m in self.arm.poll():
                if m[0] == 'cycle':
                    _, q, p, prog, lat, tilt, qd = m
                    self._set_robot(q)
                    self.meas_tips.append(p)
                    self.status = (f'progress {prog:.3f} m   lateral {lat * 1000:.1f} mm   '
                                   f'tilt {tilt:.1f} deg   |qdot| {qd:.2f} rad/s')
                elif m[0] == 'done':
                    _, prog, reason = m
                    c = self.exec_cand
                    self.result = (f'{prog:.3f} m, {reason}   (predicted {c["stroke"]:.3f} m, '
                                   f'{c["reason"]})')
                    self.running = False
                    finished = True
            if self.running:
                T = np.asarray(self.meas_tips)
                if len(T) > 1:
                    if self.meas_ink is not None:
                        self.meas_ink.detach_from(self.scene)
                    self.meas_ink = ossop.linsegs(np.stack([T[:-1], T[1:]], 1), radius=0.004,
                                                  srgbs=np.float32(COL_MEAS), alpha=1.0)
                    self.meas_ink.attach_to(self.scene)
                self._update_label()
            if finished:
                print(f'[pick] run finished: {self.result}', flush=True)
                self.regenerate()
            if args.screenshot and not getattr(self, '_shot', False):
                self._shot = True
                self.world.schedule_once(self._screenshot, 1.0)

        def _screenshot(self, _dt):
            pyglet.image.get_buffer_manager().get_color_buffer().save(args.screenshot)
            print(f'[pick] screenshot -> {args.screenshot}', flush=True)
            if args.auto_test:
                self.execute()
                self.world.schedule_interval(self._auto_end, 0.5)
            else:
                self.quit()

        def _auto_end(self, _dt):
            if not self.running and self.result is not None:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    args.screenshot.replace('.png', '_end.png'))
                self.quit()

    return App()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ip', default=None, help='default: $ONE_ARM_IP or 192.168.1.205')
    ap.add_argument('--sim', action='store_true')
    ap.add_argument('--sim-task', type=int, default=0)
    ap.add_argument('--sim-q-deg', type=float, nargs=7, default=None,
                    help='--sim: start from these joint angles [deg] instead of a pool task')
    ap.add_argument('--sim-lag', type=float, default=0.03)
    ap.add_argument('--normal', type=float, nargs=3, default=None,
                    help='cone axis / plane normal (default: current tool axis)')
    ap.add_argument('--n-cands', type=int, default=12, help='directions to try')
    ap.add_argument('--stroke', type=float, default=0.8, help='target length [m]')
    ap.add_argument('--period', type=float, default=0.05)
    ap.add_argument('--time-scale', type=float, default=0.5)
    ap.add_argument('--qd-cap', type=float, default=1.0)
    ap.add_argument('--tcp', type=float, default=0.10, help='on-axis tool length [m]')
    ap.add_argument('--tool', choices=['pen', 'xhand_index'], default=None,
                    help='tool preset; xhand_index = XHand index fingertip, hand open')
    ap.add_argument('--tool-xyz', type=float, nargs=3, default=None,
                    help='tool tip (x y z) in the flange frame')
    ap.add_argument('--cone', type=float, default=30.0)
    ap.add_argument('--k-lateral', type=float, default=5.0)
    ap.add_argument('--log-dir', default=None)
    ap.add_argument('--sim-device', default=None,
                    help='device for the candidate rollouts (default: cuda if available)')
    ap.add_argument('--screenshot', default=None, help='debug: save a frame and quit')
    ap.add_argument('--auto-test', action='store_true',
                    help='debug: with --screenshot, also execute the best target')
    args = ap.parse_args()
    if args.sim_device is None:
        import torch
        args.sim_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.ip is None:
        from Yuan.IJRR.deploy.xarm7_realtime import DEFAULT_IP
        args.ip = DEFAULT_IP
    app = build_app(args)
    try:
        app.world.run()
    finally:
        try:
            app.arm.stop()
        finally:
            app.arm.close()


if __name__ == '__main__':
    main()
