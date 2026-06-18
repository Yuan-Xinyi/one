"""XArm7 (7-DOF UFACTORY arm) as a `one`-native MechBase robot.

Loaded from ``xarm7.urdf`` (the same asset the wrs ``xarm7.py`` in this folder
uses), so it plugs into the `one` planning / collider / EE-mounting stack exactly
like Lite6 and the L1 arm -- positioned through cross-object ik, e.g.::

    arm.ik(p, R, chain='main', tcp=hand.eval_grasp_tcp(jw))

7 DOF is redundant, so there is no analytic solver: the chain uses the default
numerical SELIK solver, built lazily on first ``ik``.
"""
import os

import numpy as np

import one.utils.constant as ouc
import one.robots.base.mech_base as orbmb
import one.robots.base.urdf_loader as orul

_URDF_PATH = os.path.join(os.path.dirname(__file__), 'xarm7.urdf')


def prepare_mechstruct(collision_type=ouc.CollisionType.MESH):
    urdf_dir = os.path.dirname(os.path.abspath(_URDF_PATH))
    urdf = orul.load_robot_from_xacro(_URDF_PATH, base_dir=urdf_dir)
    return orul.urdf_to_mechstruct(
        urdf, urdf_dir, collision_type=collision_type, res_dir=urdf_dir)


class XArm7(orbmb.MechBase):
    """xArm7 as a MechBase. Registers the 'main' arm chain (link_base -> link7,
    numerical IK) and a 'flange' tcp on the last link, the same conventions Lite6
    uses, so existing arm code (chain='main', tcp='flange') works unchanged."""

    @classmethod
    def _build_structure(cls):
        return prepare_mechstruct()

    def __init__(self, rotmat=None, pos=None, home_qs=None, is_free=False):
        super().__init__(rotmat=rotmat, pos=pos,
                         home_qs=home_qs, is_free=is_free)
        c = self.structure.compiled
        self.add_chain('main', c.root_lnk, c.tip_lnks[0])   # numeric SELIK
        self.add_tcp('flange', self.runtime_lnks[-1])


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(1.6, 0.8, 1.2), cam_lookat_pos=(0.0, 0.0, 0.4))
    builtins.base = base
    ossop.frame().attach_to(base.scene)
    robot = XArm7()
    robot.attach_to(base.scene)
    base.run()
