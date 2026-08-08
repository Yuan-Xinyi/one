# `Yuan/IJRR/figures` — figure code for the paper

One file per figure topic. Each file is standalone: it owns its scene
definition, its computation and its plotting, and it can be run directly.
Shared code lives where it already lives (`one.robots...` for kinematics and the
sphere collision model, `Yuan/IJRR/{env,kinematics,eval}` for the pipeline);
nothing in here is imported by the pipeline.

Conventions

* run from the repo root with the `one` conda environment
  (`conda activate one; cd /home/lqin/one`), or from anywhere — every file puts
  the repo root on `sys.path` itself;
* **import `matplotlib` before `torch`.** In this environment torch loads the
  system `libstdc++` and matplotlib's C extension then fails on
  `CXXABI_1.3.15`. The files here already do this, keep it that way when editing;
* `--save out.mp4` / `out.gif` writes the animation, `--save out.png` writes the
  same figure as a static frame at the end of the rollout, no `--save` opens an
  interactive window (needs a display; the backend is TkAgg here);
* the FR3 model, its joint limits and the sphere self-collision model are the
  ones the rest of the pipeline uses, so numbers here are comparable to the
  numbers in the paper.

| file | topic |
|---|---|
| `fig_door_opening.py` | opening a door: the achievable opening angle is a property of the whole trajectory, not of any point on it |
| `fr3_scene.py` | *helper, not a figure* — MuJoCo scene with the real FR3 meshes (arm + Franka hand), the wall, the hinged leaf and the handle, so figures can show the actual robot |
| `door_scene_one.py` | the same door rollouts inside **one's own real-time viewer** (`one.viewer.world`), as a transparent overlay you can orbit and screenshot |

## `fig_door_opening.py`

A vertical leaf on a vertical hinge, grasped at a handle bar. The bar is a
cylinder, so *how* the hand takes it (`grasp_roll_deg`, `grasp_flip`) and at
what height are one-time choices; after that the grasp is rigid, so the door
angle `theta` fixes the full 6-D end-effector pose and the 7-DoF FR3 keeps a 1-D
self-motion manifold. The measured quantity is `theta_max`: how far the door can
be pulled before a joint limit, a singularity, a self-collision or the wall /
the swinging leaf stops the arm.

The scene is tuned so that the door **can** be opened all the way (90°): one
start posture, one base placement and several resolution laws get there, and
everything else stalls. In the search behind this scene, 35 of 4986 combinations
of (base, grasp, start posture, resolution law) opened it fully.

The grasp frame is the **hand** frame (fingers closing horizontally across the
vertical bar); since the Franka Hand is mounted 45° about z on the flange and the
kinematics here is flange-based, `grasp_path` post-multiplies by that mount
rotation. Without it the fingers meet the bar at 45°, which is not a grasp.

```bash
# in one's own real-time viewer (transparent overlay, orbit and screenshot yourself)
python Yuan/IJRR/figures/door_scene_one.py --scenario init
python Yuan/IJRR/figures/door_scene_one.py --scenario base --ghosts 8
python Yuan/IJRR/figures/door_scene_one.py --scenario init --variant 0 --animate

# the matplotlib figures / animations
python Yuan/IJRR/figures/fig_door_opening.py --scenario init         # start joint angles
python Yuan/IJRR/figures/fig_door_opening.py --scenario redundancy   # null-space resolution
python Yuan/IJRR/figures/fig_door_opening.py --scenario base         # base placement
python Yuan/IJRR/figures/fig_door_opening.py --scenario height       # grasp height on the leaf
python Yuan/IJRR/figures/fig_door_opening.py --sweep                 # the landscape, static (~2 min)
```

Each scenario animates 3–4 rollouts side by side — the real FR3 meshes rendered
with MuJoCo, plus a schematic top view where the door angle is easiest to read —
together with the joint-limit margin along the way and a bar chart of
`theta_max` against what a point-wise reachability map would promise. Everything except the varied quantity is held fixed, and the start
posture is optimized away (best over all IK branches) in the `base` and `height`
scenarios so that only the varied quantity can explain the difference.

Numbers with the defaults in the file (FR3 on a 0.70 m platform, 0.80 m leaf,
handle at 0.70 m from the hinge and 0.95 m high, pulled toward the robot):

| scenario | `theta_max`, goal is 90° |
|---|---|
| start joint angles (same base, same controller, same TCP pose) | **90° (opens it)** / 62.5° / 10° |
| redundancy resolution (same start joint angles) | 60.5° minimum-norm, 30.5° classical (paper 0.8/0.4), 61.5° weighted least norm + clamping, **90°** hybrid (2 boundary switches) |
| base placement (15 cm apart) | **90°** / 68° / 50.5° |
| grasp height (0.70 / 0.95 / 1.20 m) | 75° / **90°** / **90°** |
| point-wise reachability map | 90° — the whole arc, for every one of the cases above |

The resolution laws are `--laws minnorm center manip wln clear classical hybrid`;
`classical` is `Yuan/IJRR/env/classical_nullspace.py` (directional manipulability
0.8 + joint centering 0.4, cone term dropped — a rigid grasp has no cone) and
`hybrid` is that law in the interior with the limit-aware law near a bound,
switched on `max|q-q_mid|/q_half` with the paper's 0.985/0.96 hysteresis. The
trained RL policy is *not* here: its observation and action spaces belong to the
position-plus-cone task, not to a 6-D rigid grasp.

`--overlay N` turns any scenario into a static figure instead of an animation:
N poses of each rollout and of the leaf, superimposed with rising opacity. The
meshes are then in FR3's own white — the variant colour stays on the title, the
TCP trace, the curves and the bars. `--render stick --overlay N` does the same
with link polylines (those keep the variant colour, it is all they have).

Rendering: the 3-D panels are MuJoCo renders of the real FR3 meshes
(`fr3_scene.py`), composited into the matplotlib figure, so the whole window
animates at roughly 10 fps (MuJoCo itself does 3 panels at ~75 fps; matplotlib
is the bottleneck). `--viewer` instead plays the rollouts in a native MuJoCo
window at full rate with a free camera and no plots. `--render stick` is the
polyline fallback for machines without a GL backend; if the default backend
fails, try `MUJOCO_GL=osmesa` (CPU, slower).

Useful flags: `--cam-azim / --cam-elev / --zoom` frame the mesh camera
(`--cam-azim 0` looks straight into the door from the robot's side),
`--dtheta` sets the door step (0.5° default), `--theta-end` the goal opening,
`--grid` the base-grid size of `--sweep`, `--no-reach` skips the reachability
map (about a second per variant), `--spheres` draws the collision spheres in
`--render stick`.
