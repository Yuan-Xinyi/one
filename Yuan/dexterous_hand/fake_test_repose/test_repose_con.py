"""Replay an XHand action clip on the REAL hand, very slowly.

Parallel to ``test_repose_sim.py`` but drives the physical XHand over
serial via ``XHandX.goto_given_conf`` instead of the viewer.

Safety-first design (this moves real hardware):
  * Very slow by default (``--speed 0.1`` => 10x slower than the 20 Hz
    recording) with interpolated sub-steps so motion is smooth.
  * A gentle ``--ramp`` from a HOME pose into the clip's first frame,
    so the hand eases in instead of snapping.
  * Per-joint limit clamping (matches the sim/URDF limits).
  * ``--dry-run`` prints every target without opening the serial port.

The clip stores joints in *joint-level* order; the sim/hardware want a
specific per-id order. We remap by joint *name* (see HW_JOINT_ORDER).

Usage::

    # safe first: print targets only, no hardware
    python Yuan/dexterous_hand/fake_test_repose/test_repose_con.py --dry-run

    # very slow real replay (start with the hand open / near HOME)
    python Yuan/dexterous_hand/fake_test_repose/test_repose_con.py --port /dev/ttyUSB0 --speed 0.1

    # even slower, smoother
    python Yuan/dexterous_hand/fake_test_repose/test_repose_con.py --speed 0.05 --substeps 10

NOTE: open-loop finger playback only — it cannot reproduce the cube
reposing (the policy was closed-loop on the cube pose). Use it to check
hardware / finger trajectories.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]  # /home/lqin/one
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from Yuan.dexterous_hand.xhand_con.xhand_x import XHandX  # noqa: E402


CLIP_PATH = _HERE / "xhand_action_clip.npz"

# ---- Hardware finger-id order: id 0..11 -> joint name --------------------
# This is the canonical XHand joint order, fixed by xhand_right.urdf: the
# i-th revolute joint in the URDF is FingerCommand id i in goto_given_conf,
# and the same order is the sim's conf vector. So sim and hardware share
# this exact mapping — it is determined, not a guess.
HW_JOINT_ORDER = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)

# Per-joint limits (rad), from xhand_right.prepare_ms() / the URDF.
JOINT_LIMITS = {
    "thumb_joint0": (0.0, 1.83),
    "thumb_joint1": (-1.05, 1.57),
    "thumb_joint2": (-0.175, 1.83),
    "index_joint0": (-0.175, 0.175),
    "index_joint1": (0.0, 1.92),
    "index_joint2": (0.0, 1.92),
    "middle_joint0": (0.0, 1.92),
    "middle_joint1": (0.0, 1.92),
    "ring_joint0": (0.0, 1.92),
    "ring_joint1": (0.0, 1.92),
    "pinky_joint0": (0.0, 1.92),
    "pinky_joint1": (0.0, 1.92),
}

# HOME pose the ramp-in starts from (open hand = all zeros). The hand
# should be physically near this before starting.
HOME_CONF = np.zeros(len(HW_JOINT_ORDER), dtype=np.float32)


def load_clip(source: str):
    """Return (conf_traj [T, 12] in HW_JOINT_ORDER, control_dt)."""
    d = np.load(CLIP_PATH, allow_pickle=True)
    clip_names = [str(n) for n in d["joint_names"]]
    traj = np.asarray(d[source], dtype=np.float32)  # [T, 12] in clip order
    control_dt = float(d["control_dt"])

    name_to_col = {n: i for i, n in enumerate(clip_names)}
    missing = [n for n in HW_JOINT_ORDER if n not in name_to_col]
    if missing:
        raise KeyError(f"clip is missing joints {missing}; has {clip_names}")
    perm = [name_to_col[n] for n in HW_JOINT_ORDER]
    return traj[:, perm], control_dt


def clamp_conf(conf):
    """Clamp each joint to its limit; warn on any out-of-range value."""
    out = np.array(conf, dtype=np.float32)
    for i, name in enumerate(HW_JOINT_ORDER):
        lo, hi = JOINT_LIMITS[name]
        c = float(out[i])
        if c < lo or c > hi:
            print(f"  ! clamp {name}: {c:.3f} -> [{lo:.3f}, {hi:.3f}]")
        out[i] = min(max(c, lo), hi)
    return out


def interp_confs(a, b, n):
    """n intermediate confs from a (exclusive) to b (inclusive)."""
    ts = (np.arange(1, n + 1, dtype=np.float32) / n)[:, None]
    return (1.0 - ts) * a[None, :] + ts * b[None, :]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyUSB0",
                        help="serial port of the hand")
    parser.add_argument("--baudrate", type=int, default=3000000)
    parser.add_argument("--source", choices=["joint_pos", "joint_target"],
                        default="joint_pos",
                        help="which recorded trajectory to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed vs real-time (0.1 = 10x slower)")
    parser.add_argument("--substeps", type=int, default=5,
                        help="interpolated sends between recorded frames")
    parser.add_argument("--ramp", type=float, default=3.0,
                        help="seconds to ease from HOME into the first frame")
    parser.add_argument("--dry-run", action="store_true",
                        help="print targets only; do not open serial / move")
    args = parser.parse_args()

    conf_traj, control_dt = load_clip(args.source)
    conf_traj = np.stack([clamp_conf(c) for c in conf_traj])
    n_steps = conf_traj.shape[0]

    frame_dt = control_dt / max(args.speed, 1e-6)   # wall time per frame
    step_dt = frame_dt / max(args.substeps, 1)      # wall time per send
    total = n_steps * frame_dt
    print(f"{args.source}: {n_steps} frames, speed={args.speed} "
          f"=> frame_dt={frame_dt:.3f}s, {args.substeps} substeps "
          f"({step_dt:.3f}s/send), ~{total:.0f}s total"
          + (" [DRY RUN]" if args.dry_run else ""))

    hand = None
    if not args.dry_run:
        hand = XHandX(port=args.port, baudrate=args.baudrate)
        if hand.ser is None:
            raise SystemExit("failed to open serial port; aborting")
        hand.get_version()

    def send(conf, tag):
        if args.dry_run:
            print(f"  {tag}: " + " ".join(f"{v:+.3f}" for v in conf))
        else:
            hand.goto_given_conf([float(v) for v in conf])

    try:
        # --- ramp HOME -> first frame ---
        ramp_n = max(int(args.ramp / step_dt), 1) if args.ramp > 0 else 0
        if ramp_n:
            print(f"ramping into frame 0 over {args.ramp:.1f}s "
                  f"({ramp_n} steps)...")
            for k, c in enumerate(interp_confs(HOME_CONF, conf_traj[0], ramp_n)):
                send(c, f"ramp {k + 1}/{ramp_n}")
                time.sleep(step_dt)

        # --- play trajectory ---
        prev = conf_traj[0]
        send(prev, "frame 0")
        time.sleep(step_dt)
        for i in range(1, n_steps):
            for c in interp_confs(prev, conf_traj[i], args.substeps):
                send(c, f"frame {i}")
                time.sleep(step_dt)
            prev = conf_traj[i]
        print("done.")
    except KeyboardInterrupt:
        print("\ninterrupted by user.")
    finally:
        if hand is not None:
            hand.close()


if __name__ == "__main__":
    main()
