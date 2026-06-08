"""Replay an XHand action clip in the `one` viewer.

Reads ``xhand_action_clip.npz`` (200 steps @ 20 Hz) and drives the
12-dof XHandRight through the recorded trajectory.

The clip stores joints in *joint-level* order::

    index0, middle0, pinky0, ring0, thumb0,
    index1, middle1, pinky1, ring1, thumb1,
    index2, thumb2

but the sim's conf vector is in *per-finger* order
(thumb[:3], index[3:6], middle[6:8], ring[8:10], pinky[10:12]). We remap
by joint *name* so column order never has to be assumed.

Two trajectories are available:
  * ``joint_pos``     — measured angles in sim (default; smoother, physical)
  * ``joint_target``  — commanded targets (may jitter / overshoot)

Usage::

    python Yuan/dexterous_hand/fake_test_repose/test_repose_sim.py
    python Yuan/dexterous_hand/fake_test_repose/test_repose_sim.py --source joint_target
    python Yuan/dexterous_hand/fake_test_repose/test_repose_sim.py --speed 0.5 --loop

NOTE: this is open-loop playback of finger joints only — it does not (and
cannot) reproduce the cube reposing, since the policy was closed-loop on
the cube pose. It is for checking hardware / finger trajectories.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]  # /home/lqin/one
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse  # noqa: E402

import builtins  # noqa: E402
import numpy as np  # noqa: E402

import one.scene.scene_object_primitive as ossop  # noqa: E402
import one.viewer.world as ovw  # noqa: E402
from Yuan.dexterous_hand.xhand_sim.xhand_right import XHandRight  # noqa: E402


CLIP_PATH = _HERE / "xhand_action_clip.npz"

# Sim conf order — must match the joint add order in xhand_right.prepare_ms()
# (thumb[:3], index[3:6], middle[6:8], ring[8:10], pinky[10:12]).
SIM_JOINT_ORDER = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)


def load_clip(source: str):
    """Return (conf_traj [T, 12] in sim order, control_dt)."""
    d = np.load(CLIP_PATH, allow_pickle=True)
    clip_names = [str(n) for n in d["joint_names"]]
    traj = np.asarray(d[source], dtype=np.float32)  # [T, 12] in clip order
    control_dt = float(d["control_dt"])

    # Build clip-col -> sim-slot permutation by name.
    name_to_col = {n: i for i, n in enumerate(clip_names)}
    missing = [n for n in SIM_JOINT_ORDER if n not in name_to_col]
    if missing:
        raise KeyError(f"clip is missing joints {missing}; has {clip_names}")
    perm = [name_to_col[n] for n in SIM_JOINT_ORDER]
    return traj[:, perm], control_dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["joint_pos", "joint_target"],
                        default="joint_pos",
                        help="which recorded trajectory to replay")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback steps advanced per frame (real-time=1)")
    parser.add_argument("--loop", action="store_true",
                        help="loop the clip instead of holding the last frame")
    args = parser.parse_args()

    conf_traj, control_dt = load_clip(args.source)
    n_steps = conf_traj.shape[0]
    print(f"loaded {args.source}: {n_steps} steps @ "
          f"{1.0 / control_dt:.0f} Hz ({n_steps * control_dt:.1f} s)")

    base = ovw.World(cam_pos=(0.5, -0.5, 0.5), cam_lookat_pos=(0, 0, 0.05))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    xhand = XHandRight()
    xhand.goto_given_conf(conf_traj[0])
    xhand.attach_to(base.scene)

    state = {"i": 0.0}

    def animate(_dt, *_a, **_k):
        i = state["i"]
        if i >= n_steps - 1:
            if args.loop:
                i = 0.0
            else:
                i = n_steps - 1
                xhand.goto_given_conf(conf_traj[int(i)])
                return
        xhand.goto_given_conf(conf_traj[int(i)])
        state["i"] = i + args.speed

    base.schedule_interval(animate, control_dt)
    base.run()


if __name__ == "__main__":
    main()
