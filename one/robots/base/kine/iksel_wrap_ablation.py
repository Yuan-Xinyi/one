"""Ablation: 2pi "limit rescue" (wrap_to_limits) on the SELIKSolver (iksel).

For each robot we draw N random in-limit joint configs, FK them to a target
pose (so a valid solution provably exists), and solve IK with the iksel solver
twice over the *same* targets:

    OFF : wrap_to_limits = False  (discard out-of-limit Newton solutions)
    ON  : wrap_to_limits = True   (try q +/- 2*pi*n to pull revolute joints
                                    back into range before discarding)

Because the only difference is post-convergence handling, the two runs are
perfectly paired. We report the success-rate delta, how many tasks were
recovered purely by the wrap, the pose accuracy of the returned solutions
(to confirm the 2pi offset leaves FK unchanged), and the mean Newton
iterations / wall time (to confirm the rescue adds no extra iterations).

Run:  python -m one.robots.base.kine.iksel_wrap_ablation
"""

import os
import time
import json
import numpy as np
from tqdm import tqdm

import one.utils.math as oum
import one.robots.base.kine.numik_sel as orbkis
from one.robots.manipulators.denso.cvr038.cvr038 import CVR038
from one.robots.manipulators.universal_robots.ur3.ur3 import UR3
from one.robots.manipulators.kawasaki.rs007l.rs007l import RS007L
from one.robots.manipulators.xarm.lite6.lite6 import Lite6
from one.robots.manipulators.franka.fr3.fr3 import FR3

ROBOTS = {
    'cbt': CVR038, 'ur3': UR3, 'rs007l': RS007L, 'lite6': Lite6, 'fr3': FR3,
}

N_TASKS = int(os.environ.get("ABLATION_N", "2000"))
SEED = int(os.environ.get("ABLATION_SEED", "0"))
ROBOT_LIST = os.environ.get(
    "ABLATION_ROBOTS", "ur3,rs007l,lite6,cbt,fr3").split(",")
OUT_JSON = "one/robots/base/kine/iksel_wrap_ablation.jsonl"


def rand_active_qs(chain, rng):
    lo = np.asarray(chain.lmt_lo, dtype=np.float32)
    up = np.asarray(chain.lmt_up, dtype=np.float32)
    return lo + rng.random(len(lo)).astype(np.float32) * (up - lo)


def solve_once(robot, tgt_rotmat, tgt_pos):
    """Return (success, pos_err_mm, rot_err_deg, dt) for the best solution."""
    t0 = time.time()
    sols = robot.ik_tcp(
        tgt_rotmat=tgt_rotmat, tgt_pos=tgt_pos, max_solutions=1)
    dt = time.time() - t0
    if not sols:
        return False, np.nan, np.nan, dt
    robot.fk(qs=sols[0])
    p = robot.gl_tcp_tf[:3, 3]
    r = robot.gl_tcp_tf[:3, :3]
    pos_err, rot_err, _ = oum.diff_between_poses(
        tgt_pos * 1000, tgt_rotmat, p * 1000, r)
    return True, pos_err, rot_err * 180 / np.pi, dt


def run_robot(name):
    robot = ROBOTS[name](pos=np.array([0.1, 0.3, 0.5], dtype=np.float32))
    chain = robot._chain
    data_dir = os.path.join(robot.structure.res_dir, "data")
    solver = orbkis.SELIKSolver(chain, data_dir=data_dir)
    robot._solver = solver

    rng = np.random.default_rng(SEED)
    # pre-draw the SAME targets used by both conditions
    targets = []
    for _ in range(N_TASKS):
        q = rand_active_qs(chain, rng)
        robot.fk(qs=q)
        targets.append((robot.gl_tcp_tf[:3, :3].copy(),
                        robot.gl_tcp_tf[:3, 3].copy()))

    res = {}
    for cond, flag in [("off", False), ("on", True)]:
        solver.wrap_to_limits = flag
        succ = 0
        pos, rot, ts = [], [], []
        ok_mask = np.zeros(N_TASKS, dtype=bool)
        for i, (rm, p) in enumerate(
                tqdm(targets, desc=f"{name}:wrap_{cond}", leave=False)):
            ok, pe, re, dt = solve_once(robot, rm, p)
            ts.append(dt)
            if ok:
                succ += 1
                ok_mask[i] = True
                pos.append(pe)
                rot.append(re)
        res[cond] = dict(
            succ=succ, ok_mask=ok_mask,
            pos=np.asarray(pos) if pos else np.array([np.nan]),
            rot=np.asarray(rot) if rot else np.array([np.nan]),
            t=np.asarray(ts))

    rescued = int(np.sum(res["on"]["ok_mask"] & ~res["off"]["ok_mask"]))
    lost = int(np.sum(~res["on"]["ok_mask"] & res["off"]["ok_mask"]))
    entry = {
        "robot": name, "n": N_TASKS,
        "succ_off": res["off"]["succ"],
        "succ_on": res["on"]["succ"],
        "rate_off": round(res["off"]["succ"] / N_TASKS * 100, 2),
        "rate_on": round(res["on"]["succ"] / N_TASKS * 100, 2),
        "delta_pp": round(
            (res["on"]["succ"] - res["off"]["succ"]) / N_TASKS * 100, 2),
        "rescued_tasks": rescued,
        "regressed_tasks": lost,
        "pos_err_mean_off_mm": round(float(np.mean(res["off"]["pos"])), 4),
        "pos_err_mean_on_mm": round(float(np.mean(res["on"]["pos"])), 4),
        "rot_err_mean_off_deg": round(float(np.mean(res["off"]["rot"])), 4),
        "rot_err_mean_on_deg": round(float(np.mean(res["on"]["rot"])), 4),
        "t_mean_off_ms": round(float(np.mean(res["off"]["t"])) * 1000, 3),
        "t_mean_on_ms": round(float(np.mean(res["on"]["t"])) * 1000, 3),
    }
    return entry


if __name__ == '__main__':
    open(OUT_JSON, "w").close()
    rows = []
    for name in ROBOT_LIST:
        e = run_robot(name)
        rows.append(e)
        with open(OUT_JSON, "a") as f:
            f.write(json.dumps(e) + "\n")
        print(f"\n[{e['robot']}] N={e['n']}  "
              f"rate {e['rate_off']:.2f}% -> {e['rate_on']:.2f}%  "
              f"(+{e['delta_pp']:.2f}pp, rescued={e['rescued_tasks']}, "
              f"regressed={e['regressed_tasks']})  "
              f"pos {e['pos_err_mean_off_mm']:.3f}->{e['pos_err_mean_on_mm']:.3f}mm  "
              f"t {e['t_mean_off_ms']:.2f}->{e['t_mean_on_ms']:.2f}ms")

    print("\n==== iksel wrap_to_limits ablation summary ====")
    hdr = (f"{'robot':7s} {'rate_off':>9s} {'rate_on':>9s} {'delta':>7s} "
           f"{'rescued':>8s} {'regress':>8s} {'pos_on(mm)':>11s}")
    print(hdr)
    for e in rows:
        print(f"{e['robot']:7s} {e['rate_off']:8.2f}% {e['rate_on']:8.2f}% "
              f"{e['delta_pp']:+6.2f}pp {e['rescued_tasks']:8d} "
              f"{e['regressed_tasks']:8d} {e['pos_err_mean_on_mm']:11.4f}")
