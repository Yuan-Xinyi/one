"""Small motion-planning helpers used by the cube-grasp pipeline.

Self-contained copies of the three helpers this pipeline needs from
``examples/l1picking/l1picking.py`` (kept here so ``Xinyi/grasp_objects`` runs
without depending on the examples tree). ``chain_planning_context`` carries the
``joint_limit_overrides`` extension used to keep the arm in its forward
hemisphere.
"""
import numpy as np

import one.motion.core.planning_context as omppc


TABLE_TOP_Z = 0.9   # nominal tabletop height in the sim world frame (m)


def chain_planning_context(robot, mjc, chain_name, joint_limit_overrides=None):
    """PlanningContext over the full qs with every joint NOT on ``chain_name``
    frozen at home -> the planner only explores that chain.

    ``joint_limit_overrides`` optionally narrows named joints to ``(low, high)``
    radians. Bounds are intersected with the URDF limits, never expanded.
    """
    c = robot._compiled
    chain = robot.chain(chain_name)
    lo = c.jlmt_low_by_idx.astype(np.float64).copy()
    hi = c.jlmt_high_by_idx.astype(np.float64).copy()
    home = robot.qs.astype(np.float64).copy()
    free = np.zeros(c.n_jnts, dtype=bool)
    free[chain.active_jnt_ids] = True
    lo[~free] = home[~free]
    hi[~free] = home[~free]
    for name, limits in (joint_limit_overrides or {}).items():
        if name not in robot.structure.jnt_map:
            raise ValueError(f'unknown joint limit override: {name}')
        jidx = c.jidx_map[robot.structure.jnt_map[name]]
        low, high = map(float, limits)
        lo[jidx] = max(lo[jidx], low)
        hi[jidx] = min(hi[jidx], high)
        if lo[jidx] > hi[jidx]:
            raise ValueError(f'empty joint limit override for {name}')
    return omppc.PlanningContext(collider=mjc, joint_limits=(lo, hi))


def plan_segment(planner, start, goal, max_iters=4000):
    """RRT-Connect a single joint-space segment ``start -> goal``.

    Thin wrapper over ``planner.solve`` returning a list of joint configs
    (densified, ``path[0] == start`` .. ``path[-1] == goal``) so callers can
    concatenate it with cartesian sub-paths (``traj += descend[1:]``).
    Raises if no collision-free path is found."""
    path = planner.solve(np.asarray(start, np.float64),
                         np.asarray(goal, np.float64), max_iters=max_iters)
    if path is None:
        raise RuntimeError("plan_segment: RRT-Connect found no path "
                           "start -> goal (unreachable or blocked)")
    return list(path)
