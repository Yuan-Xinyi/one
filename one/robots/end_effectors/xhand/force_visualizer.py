"""
Interactive real-time fingertip-force visualizer for the XHand right (12-DOF
dexterous hand).

Streams the hand over RS-485 and, every cycle, draws the live contact force on
each of the five fingertips as an arrow at the (kinematically reconstructed)
fingertip: direction = the tactile pad's net (fx, fy, fz) mapped into the world,
length = |force| (scaled), colour = green (light) -> red (hard). The displayed
hand pose tracks the measured finger positions, so the model mirrors the real
hand as you squeeze something.

The force comes from the five fingertip tactile pads in every reply
(``XHandX.move_read_full`` -> the 5 ``SensorData`` blocks); the per-finger
joint torques are read in the same reply and printed alongside for reference.

----------------------------------------------------------------------- controls
  mouse:  right-drag orbit, middle-drag pan, scroll zoom
  SPACE   toggle squeeze (close all fingers <-> open)
  UP/DOWN increase / decrease the squeeze amount
  1..5    toggle an individual finger (thumb, index, middle, ring, pinky)
  0       enable all fingers      O   open all (amount -> 0)
  Z       re-zero (tare) the tactile pads -- do it with the fingers free
  T       toggle the per-finger joint-torque bars
  Q/ESC   quit

----------------------------------------------------------------------- run
  py -3.12 one/robots/end_effectors/xhand/force_visualizer.py
  REAL_HAND_PORT=/dev/ttyUSB0 py -3.12 .../force_visualizer.py   # override port

This is REAL-HARDWARE driven: it requires a connected XHand. If the serial port
can't be opened it prints why and exits.
"""
import os
import time
import builtins

import numpy as np
import pyglet.window.key as key

import one.utils.constant as ouc
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.end_effectors.xhand.xhand_right import XHandRight
from one.control.end_effector.xhand.xhand_x import XHandX

# ------------------------------------------------------------------- config
PORT = os.environ.get("REAL_HAND_PORT", "/dev/ttyUSB0")
BAUD = 3000000

# Finger order is the SAME for the hardware finger-ids, the 5 tactile pads, and
# the URDF, so a single ordered list ties pad k <-> fingertip link k <-> name k.
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_TIP_LINKS = ["thumb_rota_link2", "index_rota_link2", "mid_link2",
                    "ring_link2", "pinky_link2"]
# joint (= hardware finger-id) indices that belong to each finger, URDF order
# (thumb0-2, index0-2, middle0-1, ring0-1, pinky0-1).
FINGER_QIDX = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9], [10, 11]]

# Tactile pad -> fingertip-link frame rotation. The pad reports (fx, fy, fz) in
# its own frame; we don't have its mounting calibration, so default to identity
# (i.e. treat pad axes as the fingertip link axes) and expose it for tuning.
SENSOR_ROT = np.eye(3, dtype=np.float32)

FORCE_GAIN = 0.0016    # arrow length per unit force (m / count) -- tune on hw
FORCE_MAX = 80.0       # |force| mapped to full red
FORCE_DEADBAND = 2.0   # below this |force| draw nothing (sensor noise floor)

# Per-finger joint torque is read in the SAME reply (FingerState.torque) and is
# independent of the (uncalibrated) SENSOR_ROT, so it's a good cross-check on
# whether the force magnitudes look right. Drawn as a vertical bar gauge at each
# fingertip; the bar height = max |torque| over that finger's joints.
TORQUE_GAIN = 0.0004   # bar length per torque count (m / count) -- tune on hw
TORQUE_MAX = 200.0     # |torque| mapped to full red (tor_max default is 300)
TORQUE_DEADBAND = 5.0  # below this |torque| draw no bar
SHOW_TORQUE = True     # draw the torque bars (toggle live with T)

UPDATE_HZ = 30.0       # read + redraw rate
SLEW_STEP = 0.03       # max commanded-joint change per cycle (rad) -- gentle moves
SQUEEZE_STEP = 0.05    # squeeze-amount change per UP/DOWN press
PRINT_PERIOD = 0.3     # s between console force read-outs


def _fingertip_tf(hand, link_name):
    """World 4x4 transform of a (distal) link by its URDF name."""
    c = hand._compiled
    lnk = hand.runtime_lnks[c.lidx_map[hand.structure.lnk_map[link_name]]]
    return lnk.tf


def _grad_rgb(mag, mx):
    """Green (low) -> yellow -> red (>= mx)."""
    t = float(np.clip(mag / mx, 0.0, 1.0))
    return np.array([t, 1.0 - t, 0.0], dtype=np.float32)


class FingerForceViz:

    def __init__(self, hand_x, model, base):
        self.hand_x = hand_x
        self.model = model
        self.base = base

        # fully-curled 12-joint pose, derived from the model's power grasp -- the
        # squeeze target each finger is scaled toward.
        probe = XHandRight()
        probe.power(1.0)
        self.close_pose = probe.qs.copy().astype(float)
        self.open_pose = np.zeros(12, dtype=float)

        self.amount = 0.0                       # global squeeze in [0, 1]
        self.finger_on = np.ones(5, dtype=bool)  # which fingers participate
        self.q_cmd = self.open_pose.copy()      # slewed commanded pose
        self.arrows = [None] * 5                # live force-arrow scene objects
        self.torque_bars = [None] * 5           # live torque-gauge scene objects
        self.show_torque = SHOW_TORQUE
        self._print_acc = 0.0
        # tactile zero-offset per finger (fx, fy, fz): the resting bias each pad
        # reads with no contact. Subtracted from every sample; set by tare().
        self.bias = np.zeros((5, 3), dtype=np.float32)
        # resting joint-torque baseline (12,): position-hold torque carries a
        # non-zero preload, so the bars show |torque - baseline| (the change from
        # pressing), else they sit at a near-constant offset. Set by tare().
        self.torque_bias = np.zeros(12, dtype=np.float32)

    # ---------------------------------------------------------------- target
    def _target(self):
        tgt = self.open_pose.copy()
        for k in range(5):
            if self.finger_on[k]:
                for j in FINGER_QIDX[k]:
                    tgt[j] = self.amount * self.close_pose[j]
        return tgt

    # ---------------------------------------------------------------- tare
    def tare(self, settle_cycles=30, sample_cycles=30):
        """Capture the tactile zero-offset: hold the fingers free of contact,
        average the resting (fx, fy, fz) per pad, and store it as ``self.bias`` so
        subsequent forces read ~0 at rest. ``settle_cycles`` first drives the hand
        to the open pose and lets it settle (skip with 0 for an on-the-spot
        re-zero). Blocks for ~(settle+sample)/UPDATE_HZ seconds."""
        dt = 1.0 / UPDATE_HZ
        if settle_cycles:
            print("[tare] settling to rest -- keep fingertips free of contact...")
            self.q_cmd = self.open_pose.copy()
            for _ in range(settle_cycles):
                self.hand_x.move_read_full(self.q_cmd)
                time.sleep(dt)
        acc = np.zeros((5, 3), dtype=np.float64)
        tacc = np.zeros(12, dtype=np.float64)
        n = 0
        for _ in range(sample_cycles):
            fingers, sensors = self.hand_x.move_read_full(self.q_cmd)
            if sensors is not None:
                acc += [[s.fx, s.fy, s.fz] for s in sensors]
                n += 1
            if fingers is not None:
                tacc += [float(f.torque) for f in fingers]
            time.sleep(dt)
        if n:
            self.bias = (acc / n).astype(np.float32)
            self.torque_bias = (tacc / n).astype(np.float32)
        print("[tare] zeroed; resting bias |f| per finger = " + "  ".join(
            f"{FINGER_NAMES[k]}:{np.linalg.norm(self.bias[k]):.1f}"
            for k in range(5)))

    # ---------------------------------------------------------------- input
    def _handle_keys(self):
        im = self.base.input_manager
        if im.is_key_pressed_edge(key.SPACE):
            self.amount = 0.8 if self.amount < 0.05 else 0.0
        if im.is_key_pressed_edge(key.UP):
            self.amount = float(np.clip(self.amount + SQUEEZE_STEP, 0.0, 1.0))
        if im.is_key_pressed_edge(key.DOWN):
            self.amount = float(np.clip(self.amount - SQUEEZE_STEP, 0.0, 1.0))
        if im.is_key_pressed_edge(key.O):
            self.amount = 0.0
        if im.is_key_pressed_edge(key.Z):
            self.tare(settle_cycles=0)   # on-the-spot re-zero (fingers must be free)
        if im.is_key_pressed_edge(key.T):
            self.show_torque = not self.show_torque
        if im.is_key_pressed_edge(key._0):
            self.finger_on[:] = True
        for k, sym in enumerate((key._1, key._2, key._3, key._4, key._5)):
            if im.is_key_pressed_edge(sym):
                self.finger_on[k] = not self.finger_on[k]
        if im.is_key_pressed_edge(key.Q) or im.is_key_pressed_edge(key.ESCAPE):
            self.base.close()

    # ---------------------------------------------------------------- arrows
    def _redraw_force(self, sensors):
        for k in range(5):
            if self.arrows[k] is not None:
                self.arrows[k].detach_from(self.base.scene)
                self.arrows[k] = None
            if sensors is None:
                continue
            s = sensors[k]
            raw = np.array([s.fx, s.fy, s.fz], dtype=np.float32) - self.bias[k]
            f_link = SENSOR_ROT @ raw
            mag = float(np.linalg.norm(f_link))
            if mag < FORCE_DEADBAND:
                continue
            tf = _fingertip_tf(self.model, FINGER_TIP_LINKS[k])
            spos = tf[:3, 3]
            epos = spos + (tf[:3, :3] @ f_link) * FORCE_GAIN
            arr = ossop.arrow(spos=spos, epos=epos, rgb=_grad_rgb(mag, FORCE_MAX),
                              shaft_radius=0.0028, head_radius=0.007,
                              head_length=0.012)
            arr.attach_to(self.base.scene)
            self.arrows[k] = arr

    # ---------------------------------------------------------------- torque
    def _redraw_torque(self, fingers):
        """A vertical (world +Z) bar at each fingertip whose height = the finger's
        peak joint |torque|. Independent of SENSOR_ROT, so it's a direction-free
        sanity check on how hard each finger is actually pressing."""
        for k in range(5):
            if self.torque_bars[k] is not None:
                self.torque_bars[k].detach_from(self.base.scene)
                self.torque_bars[k] = None
            if fingers is None or not self.show_torque:
                continue
            tq = max(abs(float(fingers[j].torque) - self.torque_bias[j])
                     for j in FINGER_QIDX[k])
            if tq < TORQUE_DEADBAND:
                continue
            tf = _fingertip_tf(self.model, FINGER_TIP_LINKS[k])
            # offset a little along world +X so the bar sits beside the force arrow
            spos = tf[:3, 3] + np.array([0.012, 0.0, 0.0], dtype=np.float32)
            epos = spos + np.array([0.0, 0.0, tq * TORQUE_GAIN], dtype=np.float32)
            bar = ossop.cylinder(spos=spos, epos=epos, radius=0.004,
                                 rgb=_grad_rgb(tq, TORQUE_MAX))
            bar.attach_to(self.base.scene)
            self.torque_bars[k] = bar

    # ---------------------------------------------------------------- console
    def _print(self, dt, fingers, sensors):
        self._print_acc += dt
        if self._print_acc < PRINT_PERIOD:
            return
        self._print_acc = 0.0
        if sensors is None:
            print("[force] incomplete reply (no tactile data)")
            return
        cells = []
        for k in range(5):
            s = sensors[k]
            mag = float(np.linalg.norm(
                np.array([s.fx, s.fy, s.fz], dtype=np.float32) - self.bias[k]))
            mark = "" if self.finger_on[k] else "-"
            cells.append(f"{mark}{FINGER_NAMES[k]:>6} {mag:5.1f}")
        tq = ""
        if fingers is not None:
            # raw per-joint torque AND per-finger change-from-rest (what the bars
            # show) -- if 'raw' moves but 'dtau' stays flat, re-tare (press Z).
            raw = " ".join(f"{int(abs(fingers[j].torque)):>3}" for j in range(12))
            dtau = " ".join(
                f"{max(abs(float(fingers[j].torque) - self.torque_bias[j]) for j in FINGER_QIDX[k]):4.0f}"
                for k in range(5))
            tq = f"  | tau {raw}  | dtau {dtau}"
        print(f"[force] amt {self.amount:.2f} | "
              + "  ".join(cells) + tq)

    # ---------------------------------------------------------------- loop
    def update(self, dt):
        self._handle_keys()

        # slew the commanded pose toward the squeeze target (gentle, no snapping)
        target = self._target()
        delta = np.clip(target - self.q_cmd, -SLEW_STEP, SLEW_STEP)
        self.q_cmd = self.q_cmd + delta

        fingers, sensors = self.hand_x.move_read_full(self.q_cmd)

        # mirror the real hand: drive the model from the MEASURED finger positions
        # (identity map, hardware finger-id == URDF joint order). Fall back to the
        # commanded pose if the reply was incomplete.
        if fingers is not None:
            meas = np.array([f.position for f in fingers], dtype=np.float32)
            self.model.fk(qs=meas)
        else:
            self.model.fk(qs=self.q_cmd.astype(np.float32))

        self._redraw_force(sensors)
        self._redraw_torque(fingers)
        self._print(dt, fingers, sensors)


def main():
    hand_x = XHandX(port=PORT, baudrate=BAUD)
    if hand_x.ser is None or not hand_x.ser.is_open:
        raise SystemExit(
            f"No XHand on {PORT}. Set REAL_HAND_PORT to the right serial device "
            f"(this visualizer is real-hardware driven).")

    base = ovw.World(cam_pos=(0.35, -0.32, 0.22),
                     cam_lookat_pos=(0.0, 0.0, 0.10))
    builtins.base = base
    ossop.frame(length_scale=0.5, radius_scale=0.5).attach_to(base.scene)

    model = XHandRight()
    model.open_hand()
    model.attach_to(base.scene)

    viz = FingerForceViz(hand_x, model, base)
    builtins.viz = viz
    viz.tare()   # zero the tactile pads at rest before streaming (press Z to redo)
    base.schedule_interval(viz.update, interval=1.0 / UPDATE_HZ)

    print(__doc__)
    print(f"Connected; streaming fingertip force at {UPDATE_HZ:.0f} Hz. "
          f"SPACE=squeeze  UP/DOWN=amount  1-5=toggle finger  O=open  Q=quit")
    base.run()


if __name__ == "__main__":
    main()
