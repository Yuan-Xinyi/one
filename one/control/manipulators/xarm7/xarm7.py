"""
Real-time control interface for the UFACTORY xArm7 (7-DOF).

This controller exposes two regimes:

  * Blocking moves (``move_j`` / ``move_p``) for setup, homing and point-to-point
    motion. These run in *position mode* (mode 0) and wait for completion.

  * Real-time servo streaming (``servo_j`` / ``servo_p`` / ``stream_jnt_path`` /
    ``realtime_loop``) for high-frequency (<=250 Hz) on-line control. These run in
    *servo motion mode* (mode 1): each call ships a single set-point that the
    controller executes immediately, so the caller streams a continuous command
    stream. Consecutive set-points must be close together (small joint / Cartesian
    deltas) or the controller faults out.

The xHand is mounted on the flange as the end-effector, so this class controls
the arm only -- it has no native gripper.

This driver depends ONLY on the official xArm-Python-SDK (``xarm``) and SciPy --
no WRS. Install the SDK with ``pip install xArm-Python-SDK``.

Reference: XArm Developer Manual
           XArm Python SDK (https://github.com/xArm-Developer/xArm-Python-SDK)
"""
import time
from typing import Optional, Callable
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation
from xarm.wrapper import XArmAPI

__VERSION__ = '0.2.0'


def _rotmat_to_rpy(rotmat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> xArm RPY (rad), extrinsic XYZ."""
    return _Rotation.from_matrix(np.asarray(rotmat)).as_euler('xyz')


def _rpy_to_rotmat(rpy) -> np.ndarray:
    """xArm RPY (rad), extrinsic XYZ -> 3x3 rotation matrix."""
    return _Rotation.from_euler('xyz', np.asarray(rpy, dtype=float)).as_matrix()


class XArm7X(object):
    def __init__(self, ip: str = "192.168.1.232", reset: bool = False):
        """
        :param ip: The ip address of the robot
        :param reset: reset (move home) the arm on connect, otherwise just enable it
        """
        assert isinstance(ip, str)
        self.ndof = 7
        # local mirror of the controller mode so the hot servo path never has to
        # round-trip the controller just to know which mode it is in.
        self._mode_cache = None

        self._arm_x = XArmAPI(port=ip)
        driver_v = self._arm_x.version_number
        # ensure the xarm driver is >= 1.9.0 (servo-streaming support)
        assert driver_v >= (1, 9, 0)
        # clear a latched emergency stop before anything else
        if self._arm_x.has_err_warn:
            err_code = self._arm_x.get_err_warn_code()[1][0]
            if err_code == 1 or err_code == 2:
                print("The Emergency Button is pushed in to stop!")
                input("Release the emergency button and press Enter to continue...")
        self._arm_x.clean_error()
        self._arm_x.motion_enable(enable=True)
        if reset:
            self._arm_x.reset(wait=True)
        self._arm_x.set_mode(0)
        self._arm_x.set_state(0)
        self._mode_cache = 0
        time.sleep(.5)

    # ------------------------------------------------------------------ units
    @staticmethod
    def mm_to_m(arr: np.ndarray) -> np.ndarray:
        """Convert a position (mm -> m)."""
        return np.asarray(arr) / 1000

    @staticmethod
    def m_to_mm(arr: np.ndarray) -> np.ndarray:
        """Convert a position (m -> mm)."""
        return np.asarray(arr) * 1000

    # ------------------------------------------------------------------ status
    @property
    def mode(self) -> int:
        """xArm mode (only valid when enable_report is True).
        0: position control, 1: servo motion, 2: joint teaching,
        4: joint velocity, 5: cartesian velocity.
        """
        return self._arm_x.mode

    @property
    def state(self):
        """Controller state. 1: in motion, 2: sleeping, 3: suspended, 4: stopping."""
        return self._arm_x.get_state()

    @property
    def cmd_num(self) -> int:
        """Number of buffered commands still in the controller cache."""
        code, cmd_num = self._arm_x.cmd_num
        self._ex_ret_code(code)
        return cmd_num

    @property
    def has_err_warn(self) -> bool:
        return self._arm_x.has_err_warn

    def _ex_ret_code(self, code):
        """Raise if an API return code is not 0 (success)."""
        if code != 0:
            raise Exception(f"The return code {code} is incorrect. Refer to the API for details")

    def clean_error(self):
        """Clear any error/warning and re-enable motion (e.g. after a servo fault)."""
        self._arm_x.clean_error()
        self._arm_x.motion_enable(enable=True)
        self._arm_x.set_mode(self._mode_cache if self._mode_cache is not None else 0)
        self._arm_x.set_state(0)
        time.sleep(.1)

    # ------------------------------------------------------------------ modes
    def _set_mode(self, mode: int):
        """Switch controller mode, skipping the round-trip if already there."""
        if self._mode_cache == mode:
            return
        self._arm_x.set_mode(mode)
        self._arm_x.set_state(0)
        self._mode_cache = mode
        time.sleep(.5)

    def enter_position_mode(self):
        """Position control mode (mode 0): blocking point-to-point moves."""
        self._set_mode(0)

    def enter_servo_mode(self):
        """Servo motion mode (mode 1): high-frequency real-time set-point streaming.
        Call this once before a burst of ``servo_j`` / ``servo_p`` commands."""
        self._set_mode(1)

    # backward-compatible aliases
    _position_mode = enter_position_mode
    _servo_mode = enter_servo_mode

    def reset(self):
        self._arm_x.reset()

    def homeconf(self):
        self.move_j(jnt_val=np.zeros(self.ndof))

    # ------------------------------------------------------------------ kinematics
    def ik(self, tgt_pos: np.ndarray, tgt_rot: Optional[np.ndarray]) -> np.ndarray:
        """Inverse kinematics via the controller.
        :param tgt_pos: position (m)
        :param tgt_rot: 3x3 rotation matrix or 1x3 RPY (rad); None keeps the current
        :return: 1x7 joint solution (rad)
        """
        tgt_pos = self.m_to_mm(tgt_pos)
        if tgt_rot is not None:
            tgt_rot = np.asarray(tgt_rot)
            tgt_rpy = _rotmat_to_rpy(tgt_rot) if tgt_rot.shape == (3, 3) else tgt_rot.flatten()[:3]
        else:
            tgt_rpy = np.zeros(3)
        tgt_pose = tgt_pos.tolist() + np.asarray(tgt_rpy).tolist()
        code, ik_s = self._arm_x.get_inverse_kinematics(pose=tgt_pose, input_is_radian=True,
                                                        return_is_radian=True)
        self._ex_ret_code(code)
        return np.array(ik_s[:self.ndof])

    def get_jnt_values(self) -> np.ndarray:
        """Current joint values (1x7 rad)."""
        code, jnt_val = self._arm_x.get_servo_angle(is_radian=True)
        self._ex_ret_code(code)
        return np.array(jnt_val[:self.ndof])

    def get_pose(self) -> (np.ndarray, np.ndarray):
        """Current flange pose: (position[m] (3,), rotation matrix (3,3))."""
        code, pose = self._arm_x.get_position(is_radian=True)
        self._ex_ret_code(code)
        return self.mm_to_m(np.array(pose[:3])), _rpy_to_rotmat(pose[3:])

    # ------------------------------------------------------------------ blocking moves
    def move_j(self,
               jnt_val: np.ndarray,
               speed: Optional[float] = None,
               is_rel_mov: bool = False,
               wait: bool = True) -> bool:
        """Blocking joint-space move (position mode).
        :param jnt_val: target joint values (1x7)
        :param speed: joint speed (rad/s)
        :param is_rel_mov: relative move or not
        :param wait: wait for completion
        """
        if isinstance(jnt_val, np.ndarray):
            jnt_val = jnt_val.tolist()
        assert isinstance(jnt_val, list) and len(jnt_val) == self.ndof
        self.enter_position_mode()
        suc = self._arm_x.set_servo_angle(angle=jnt_val, speed=speed, is_radian=True,
                                          relative=is_rel_mov, wait=wait)
        return suc == 0

    def move_p(self,
               pos: Optional[np.ndarray],
               rot: Optional[np.ndarray],
               speed: Optional[float] = None,
               path_rad: Optional[float] = None,
               is_rel_mov: bool = False,
               wait: bool = True) -> bool:
        """Blocking Cartesian move (position mode).
        :param pos: position (m) [x,y,z]; None keeps the current
        :param rot: 3x3 rotation matrix or RPY (rad); None keeps the current
        :param speed: linear/angular speed (mm/s, rad/s)
        :param path_rad: blend radius (>=0 -> arc-line, else line)
        :param is_rel_mov: relative move or not
        :param wait: wait for completion
        """
        assert pos is not None or rot is not None
        assert path_rad is None or path_rad >= 0
        self.enter_position_mode()
        if pos is not None:
            pos = self.m_to_mm(np.array(pos))
        else:
            pos = [None] * 3
        if rot is not None:
            rot = np.array(rot)
            rpy = _rotmat_to_rpy(rot) if rot.shape == (3, 3) else rot.flatten()[:3]
        else:
            rpy = [None] * 3
        suc = self._arm_x.set_position(x=pos[0], y=pos[1], z=pos[2],
                                       roll=rpy[0], pitch=rpy[1], yaw=rpy[2], speed=speed,
                                       is_radian=True, relative=is_rel_mov,
                                       radius=path_rad, wait=wait)
        return suc == 0

    # ------------------------------------------------------------------ real-time servo
    def servo_j(self, jnt_val: np.ndarray) -> int:
        """Stream ONE real-time joint set-point (servo mode).

        Non-blocking: returns as soon as the command is queued so the caller can
        push set-points at a fixed high rate (<=250 Hz). Consecutive set-points
        must be close together or the controller will fault. ``enter_servo_mode``
        is entered automatically on the first call.
        :param jnt_val: target joint values (1x7 rad)
        :return: API return code (0 = ok)
        """
        if isinstance(jnt_val, np.ndarray):
            jnt_val = jnt_val.tolist()
        assert len(jnt_val) == self.ndof
        self.enter_servo_mode()
        return self._arm_x.set_servo_angle_j(jnt_val, is_radian=True)

    def servo_p(self, pos: np.ndarray, rot: np.ndarray, is_tool_coord: bool = False) -> int:
        """Stream ONE real-time Cartesian set-point (servo mode).

        Same streaming contract as ``servo_j``. The pose must stay close to the
        previous one between calls.
        :param pos: position (m) [x,y,z]
        :param rot: 3x3 rotation matrix or RPY (rad)
        :param is_tool_coord: interpret the pose in the tool frame
        :return: API return code (0 = ok)
        """
        pos = self.m_to_mm(np.array(pos))
        rot = np.array(rot)
        rpy = _rotmat_to_rpy(rot) if rot.shape == (3, 3) else rot.flatten()[:3]
        mvpose = pos.tolist() + np.asarray(rpy).tolist()
        self.enter_servo_mode()
        return self._arm_x.set_servo_cartesian(mvpose, is_radian=True, is_tool_coord=is_tool_coord)

    @staticmethod
    def _resample(path, dt, max_jntvel=None):
        """Linearly densify a waypoint list to one point per ``dt``. With
        ``max_jntvel`` (rad/s, scalar or 1x7) each segment is spread over enough
        cycles that no joint exceeds it; without it each input segment becomes a
        single cycle (the caller must then supply already-dense waypoints)."""
        path = [np.asarray(p, dtype=float) for p in path]
        if len(path) < 2:
            return path
        out = [path[0]]
        for q0, q1 in zip(path[:-1], path[1:]):
            if max_jntvel is None:
                n = 1
            else:
                mv = np.asarray(max_jntvel, dtype=float)
                n = max(1, int(np.ceil(float(np.max(np.abs(q1 - q0) / mv)) / dt)))
            for i in range(1, n + 1):
                out.append(q0 + (q1 - q0) * (i / n))
        return out

    def stream_jnt_path(self,
                        path,
                        control_freq: float = 100.0,
                        max_jntvel=None,
                        start_frame_id: int = 0) -> None:
        """Real-time stream a joint-space path in servo mode at ``control_freq`` Hz.

        The waypoints are linearly resampled to one set-point per control cycle
        (velocity-capped when ``max_jntvel`` is given, see ``_resample``), then
        streamed, pacing each set-point with a wall-clock timer so the stream stays
        at the requested rate.

        :param path: [q0, q1, ...] joint waypoints (each 1x7)
        :param control_freq: streaming rate (Hz)
        :param max_jntvel: per-joint max velocity (rad/s); None spreads one input
                           segment over one cycle (waypoints must already be dense)
        :param start_frame_id: drop this many leading resampled frames
        """
        if not path:
            raise ValueError("The given path is empty!")
        dt = 1.0 / control_freq
        path = self._resample(path, dt, max_jntvel)[start_frame_id:]
        self.enter_servo_mode()
        next_t = time.perf_counter()
        for jnt_values in path:
            self._arm_x.set_servo_angle_j(np.asarray(jnt_values).tolist(), is_radian=True)
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()   # we fell behind; resync the clock

    def realtime_loop(self,
                      controller: Callable[[float, np.ndarray], Optional[np.ndarray]],
                      control_freq: float = 100.0,
                      duration: Optional[float] = None) -> None:
        """Closed-loop real-time control.

        Each cycle calls ``controller(t, q)`` -- ``t`` is the elapsed time (s) and
        ``q`` the latest measured joint values -- and servos to the returned target
        joint vector, paced at ``control_freq`` Hz. The loop stops when the
        controller returns ``None`` or ``duration`` (s) elapses.

        :param controller: callback (t, q) -> target_jnt (1x7) or None to stop
        :param control_freq: control rate (Hz)
        :param duration: optional time budget (s); None runs until controller stops
        """
        dt = 1.0 / control_freq
        self.enter_servo_mode()
        start = time.perf_counter()
        next_t = start
        while True:
            now = time.perf_counter()
            t = now - start
            if duration is not None and t >= duration:
                break
            q = self.get_jnt_values()
            tgt = controller(t, q)
            if tgt is None:
                break
            self._arm_x.set_servo_angle_j(np.asarray(tgt).tolist(), is_radian=True)
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()

    # backward-compatible name from the lite6 controller
    def move_jntspace_path(self, path, max_jntvel=None, max_jntacc=None,
                           start_frame_id=0, toggle_debug=False):
        """Deprecated alias of ``stream_jnt_path`` (``max_jntacc`` is ignored)."""
        self.stream_jnt_path(path, control_freq=100.0, max_jntvel=max_jntvel,
                             start_frame_id=start_frame_id)

    def __del__(self):
        try:
            self._arm_x.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    import numpy as np

    robot = XArm7X(ip="192.168.1.205")
    print("current joint values (rad):", np.round(robot.get_jnt_values(), 4))
