"""
Real-time control interface for the XHand (12-DOF dexterous hand) over RS-485.

The hand replies with a full state frame to every command, so two streaming
styles are provided:

  * ``move(..., read=False)`` -- fire-and-forget: ship a finger set-point and flush
    the reply without parsing it. This is the fast path for high-rate streaming;
    the RX buffer is drained each cycle so it never overflows.

  * ``move(..., read=True)`` -- ship a set-point and parse the returned 12 finger
    states, for closed-loop control / logging.

``stream_positions`` runs a wall-clock-paced control loop on top of these so the
caller just supplies a per-cycle target (and, optionally, reacts to feedback).
"""
import time
import struct
import atexit

import numpy as np
import serial

try:
    from . import data_type as xhand_bt
except ImportError:  # allow running as a standalone script
    import data_type as xhand_bt

# protocol framing (preserved from the original driver)
_FRAME_HEADER = 0xAA55
_SRC_ID = 0xFE          # PC
_DEST_ID = 0x80         # hand
_CMD_MOVE = 0x02
_CMD_VERSION = 0x13
# bytes per finger state -- derived from the wire format so it can't drift out of
# sync with it (the literal 22 here was wrong: the format packs to 24, which a live
# reply confirms -- 2208 = 12x24 fingers + 5x384 tactile blocks).
_FINGER_STATE_SIZE = struct.calcsize(xhand_bt.FINGER_STATE_FORMAT)  # 24
N_FINGERS = 12

# default position-mode gains / limits for a finger command
_DEFAULT_KP, _DEFAULT_KI, _DEFAULT_KD = 100, 0, 10
_DEFAULT_TOR_MAX, _DEFAULT_MODE = 300, 3


class XHandX:
    def __init__(self, port="COM3", baudrate=3000000, verbose=False):
        """Open the RS-485 serial connection to the hand.

        :param verbose: print every packet (debug only -- leave False for real-time
                        streaming, the prints alone cap the achievable rate).
        """
        self.verbose = verbose
        self._last_target = None   # last commanded 12-finger set-point (for slewed moves)
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5)
            print(f"Connected to {port} at {baudrate} baud")
            atexit.register(self.close)
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            self.ser = None  # mark connection as failed

    def close(self):
        """Close the serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed.")

    # ------------------------------------------------------------------ framing
    def calculate_crc(self, data):
        return xhand_bt.crc16(data)

    def _build_packet(self, command, data=b""):
        """Assemble a framed command packet (header + ids + cmd + len + data + crc)."""
        frame_header = struct.pack("<H", _FRAME_HEADER)
        src_id = struct.pack("<B", _SRC_ID)
        dest_id = struct.pack("<B", _DEST_ID)
        cmd = struct.pack("<B", command)
        data_length = struct.pack("<H", len(data))
        body = frame_header + src_id + dest_id + cmd + data_length + data
        return body + self.calculate_crc(body)

    def send_command(self, command, data=b"", read=True):
        """Send a framed command. When ``read`` (default) wait for and return the
        reply payload; otherwise flush the RX buffer and return immediately."""
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port is not open.")
            return None
        if not read:
            # fire-and-forget: drop any pending reply so the buffer can't overflow
            self.ser.reset_input_buffer()
        packet = self._build_packet(command, data)
        if self.verbose:
            print(f"Sending: {packet.hex()}")
        self.ser.write(packet)
        self.ser.flush()
        if not read:
            return None
        return self.read_response()

    def read_response(self):
        """Read and CRC-check one reply frame; return its data payload (or None)."""
        if not self.ser or not self.ser.is_open:
            print("Error: Serial port is not open.")
            return None
        # header: frame header(H) + src(B) + dest(B) + command(B) + length(H) = 7 bytes
        response_header = self.ser.read(7)
        if len(response_header) < 7:
            print("Error: Incomplete response header")
            return None
        _, _, _, command, data_length = struct.unpack("<HBBBH", response_header)
        data_length = int(data_length)
        response_data = self.ser.read(data_length + 2)   # payload + 2-byte CRC
        if len(response_data) < data_length + 2:
            print("Error: Incomplete response data")
            return None
        data = response_data[:-2]
        received_crc = response_data[-2:]
        computed_crc = self.calculate_crc(response_header + data)
        if received_crc != computed_crc:
            print("Error: CRC mismatch")
            return None
        if self.verbose:
            print(f"Received: {response_header.hex()} {response_data.hex()}")
        return data

    def get_version(self):
        """Get firmware version."""
        return self.send_command(_CMD_VERSION)

    # ------------------------------------------------------------------ commands
    @staticmethod
    def _finger_package(jnt_values, kp, ki, kd, tor_max, mode):
        """Pack a 12-finger position command into wire bytes."""
        if len(jnt_values) != N_FINGERS:
            raise ValueError(f"Expected exactly {N_FINGERS} joint values.")
        finger_package = xhand_bt.FingerCommandPackage()
        for i in range(N_FINGERS):
            cmd = xhand_bt.FingerCommand(
                id=i, kp=kp, ki=ki, kd=kd,
                position=float(jnt_values[i]),
                tor_max=tor_max, mode=mode, res0=0, res1=0, res2=0, res3=0)
            finger_package.set_command(i, cmd)
        return finger_package.to_bytes()

    @staticmethod
    def parse_finger_states(data):
        """Extract the 12 leading FingerState records from a reply payload."""
        if data is None or len(data) < _FINGER_STATE_SIZE * N_FINGERS:
            return None
        return [xhand_bt.FingerState.from_bytes(
                    data[i * _FINGER_STATE_SIZE:(i + 1) * _FINGER_STATE_SIZE])
                for i in range(N_FINGERS)]

    def move(self, jnt_values,
             kp=_DEFAULT_KP, ki=_DEFAULT_KI, kd=_DEFAULT_KD,
             tor_max=_DEFAULT_TOR_MAX, mode=_DEFAULT_MODE, read=False):
        """Stream ONE 12-finger position set-point (the real-time control primitive).

        :param jnt_values: 12 target finger positions
        :param kp, ki, kd: per-finger position gains
        :param tor_max: per-finger torque limit
        :param mode: finger control mode (3 = position)
        :param read: parse and return the 12 FingerState records (closed loop);
                     default False is the fast fire-and-forget streaming path
        :return: list[FingerState] if ``read`` else None
        """
        data = self._finger_package(jnt_values, kp, ki, kd, tor_max, mode)
        reply = self.send_command(_CMD_MOVE, data, read=read)
        self._last_target = np.asarray(jnt_values, dtype=float)
        return self.parse_finger_states(reply) if read else None

    def move_to(self, target, speed=1.0, freq=100.0, start=None,
                sync=False, read_feedback=False, **move_kwargs):
        """Drive the fingers to ``target`` at a CONSTANT speed (slew-rate limited).

        Unlike ``move`` / ``goto_given_conf`` (which snap to the goal in one packet),
        this streams set-points where every cycle each finger advances by a FIXED
        increment ``speed/freq`` toward its target. So every finger moves at exactly
        ``speed`` position-units per second regardless of how far it must travel --
        a uniform slew rate, not a fixed duration. A larger displacement simply takes
        proportionally longer (more cycles), never faster.

        Two modes:
          * ``sync=False`` (default): each finger keeps its own constant ``speed`` and
            stops when it reaches its target, so fingers with small displacement
            finish earlier. Speed is uniform across fingers.
          * ``sync=True``: all fingers are scaled to arrive together -- the farthest
            finger moves at ``speed``, the rest proportionally slower.

        Blocks until the goal is reached, paced by a wall-clock timer.

        :param target: 12 target finger positions
        :param speed: finger speed (position units per second)
        :param freq: streaming rate (Hz)
        :param start: start positions; default = last commanded target (if the hand
                      was never commanded, pass it explicitly or it won't move)
        :param sync: synchronize arrival (scale all fingers) instead of per-finger
                     constant speed
        :param read_feedback: parse and return the final 12 FingerState records
        :param move_kwargs: forwarded to ``move`` (kp/ki/kd/tor_max/mode)
        :return: final list[FingerState] if ``read_feedback`` else None
        """
        target = np.asarray(target, dtype=float)
        if target.shape[0] != N_FINGERS:
            raise ValueError(f"Expected exactly {N_FINGERS} joint values.")
        if start is None:
            start = self._last_target
        q = target.copy() if start is None else np.asarray(start, dtype=float).copy()

        step = max(speed / freq, 1e-9)          # fixed per-cycle increment
        dt = 1.0 / freq
        # at constant speed the move takes ceil(max_delta/step) cycles; +2 guards rounding
        max_iter = int(np.ceil(float(np.max(np.abs(target - q))) / step)) + 2
        states = None
        next_t = time.perf_counter()
        for _ in range(max_iter):
            delta = target - q
            if np.all(np.abs(delta) <= step):   # within one increment -> finish exactly
                q = target.copy()
            elif sync:                           # scale whole vector: farthest steps by `step`
                q = q + delta * (step / float(np.max(np.abs(delta))))
            else:                                # per-finger constant speed
                q = q + np.clip(delta, -step, step)
            states = self.move(q, read=read_feedback, **move_kwargs)
            if np.array_equal(q, target):
                break
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()   # fell behind; resync the clock
        return states

    def goto_given_conf(self, jnt_values):
        """Move all 12 fingers to the given configuration and return their states."""
        return self.move(jnt_values, read=True)

    # ------------------------------------------------------------------ real-time loop
    def stream_positions(self, controller, freq=100.0, duration=None,
                         read_feedback=False, **move_kwargs):
        """Run a wall-clock-paced real-time control loop.

        Each cycle calls ``controller(t, states)`` -- ``t`` is elapsed time (s) and
        ``states`` the latest list[FingerState] (or None when ``read_feedback`` is
        off) -- and streams the returned 12-vector target, paced at ``freq`` Hz. The
        loop ends when the controller returns None or ``duration`` (s) elapses.

        :param controller: callback (t, states) -> 12 targets, or None to stop
        :param freq: control rate (Hz)
        :param duration: optional time budget (s)
        :param read_feedback: read+parse the hand state each cycle and pass it in
        :param move_kwargs: forwarded to ``move`` (kp/ki/kd/tor_max/mode)
        """
        dt = 1.0 / freq
        start = time.perf_counter()
        next_t = start
        states = None
        while True:
            now = time.perf_counter()
            t = now - start
            if duration is not None and t >= duration:
                break
            tgt = controller(t, states)
            if tgt is None:
                break
            states = self.move(tgt, read=read_feedback, **move_kwargs)
            next_t += dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()   # fell behind; resync the clock


# **Example Usage**
if __name__ == "__main__":
    # hand = XHandX(port="COM3", baudrate=3000000)
    hand = XHandX(port="/dev/ttyUSB0", baudrate=3000000)
    hand.get_version()

    # one-shot move (snaps to the goal at full controller speed)
    # hand.goto_given_conf([0.2] * 12)

    # slow, smooth move: close at 0.3 units/s instead of snapping
    hand.move_to([0.6] * 12, speed=0.1)
    hand.move_to([0.0] * 12, speed=0.1)   # open again, gently

    # real-time streaming: sinusoidal open/close at 100 Hz for 5 s
    import math

    def sine_ctrl(t, states):
        a = 0.5 * (1 - math.cos(2 * math.pi * 0.5 * t))   # 0->1->0 every 2 s
        return [a] * 12

    hand.stream_positions(sine_ctrl, freq=100.0, duration=5.0)

    hand.close()
