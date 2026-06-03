import os
import numpy as np
import one.utils.math as oum
import one.utils.constant as ouc
import one.robots.base.mech_structure as orbms
import one.robots.end_effectors.ee_base as oreb


def prepare_ms():
    structure = orbms.MechStruct()
    mesh_dir = structure.default_mesh_dir
    palm_color = ouc.ExtendedColor.DIM_GRAY
    finger_color = ouc.BasicColor.GRAY

    def link(filename, rgb):
        return orbms.Link.from_file(
            os.path.join(mesh_dir, filename),
            loc_rotmat=None,
            loc_pos=None,
            collision_type=ouc.CollisionType.MESH,
            rgb=rgb)

    # ======= palm (root) ======== #
    palm_lnk = link("right_hand_link_.stl", palm_color)
    # palm flange offsets to each finger base (rotmat == identity)
    thumb_base = np.array([0.0228, -0.0095, 0.0305], dtype=np.float32)
    index_base = np.array([0.0265, -0.0065, 0.0899], dtype=np.float32)
    middle_base = np.array([0.004, -0.0065, 0.1082], dtype=np.float32)
    ring_base = np.array([-0.016, -0.0065, 0.1052], dtype=np.float32)
    pinky_base = np.array([-0.036, -0.0065, 0.1022], dtype=np.float32)

    structure.add_lnk(palm_lnk)

    # ======= thumb (3 dof) ======== #
    thumb_l0 = link("right_hand_thumb_bend_link_.stl", finger_color)
    thumb_l1 = link("right_hand_thumb_rota_link1_.stl", finger_color)
    thumb_l2 = link("right_hand_thumb_rota_link2_.stl", finger_color)
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=palm_lnk, child_lnk=thumb_l0,
        axis=-ouc.StandardAxis.Z, pos=thumb_base, lmt_lo=0.0, lmt_up=1.83))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=thumb_l0, child_lnk=thumb_l1,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.028599, 0.0083177, 0.00178], dtype=np.float32),
        rotmat=oum.rotmat_from_euler(0.2618, 0, 0.0407),
        lmt_lo=-1.05, lmt_up=1.57))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=thumb_l1, child_lnk=thumb_l2,
        axis=-ouc.StandardAxis.Y,
        pos=np.array([0.0553, 0.0, 0.0], dtype=np.float32),
        lmt_lo=-0.175, lmt_up=1.83))

    # ======= index (3 dof) ======== #
    index_l0 = link("right_hand_index_bend_link_.stl", finger_color)
    index_l1 = link("right_hand_index_rota_link1_.stl", finger_color)
    index_l2 = link("right_hand_index_rota_link2_.stl", finger_color)
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=palm_lnk, child_lnk=index_l0,
        axis=ouc.StandardAxis.Y, pos=index_base, lmt_lo=-0.175, lmt_up=0.175))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=index_l0, child_lnk=index_l1,
        axis=ouc.StandardAxis.X,
        pos=np.array([0.0, 0.0, 0.0178], dtype=np.float32),
        lmt_lo=0.0, lmt_up=1.92))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=index_l1, child_lnk=index_l2,
        axis=ouc.StandardAxis.X,
        pos=np.array([0.0, 0.0, 0.0558], dtype=np.float32),
        lmt_lo=0.0, lmt_up=1.92))

    # ======= middle (2 dof) ======== #
    middle_l0 = link("right_hand_mid_link1_.stl", finger_color)
    middle_l1 = link("right_hand_mid_link2_.stl", finger_color)
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=palm_lnk, child_lnk=middle_l0,
        axis=ouc.StandardAxis.X, pos=middle_base, lmt_lo=0.0, lmt_up=1.92))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=middle_l0, child_lnk=middle_l1,
        axis=ouc.StandardAxis.X,
        pos=np.array([0.0, 0.0, 0.0558], dtype=np.float32),
        lmt_lo=0.0, lmt_up=1.92))

    # ======= ring (2 dof) ======== #
    ring_l0 = link("right_hand_ring_link1_.stl", finger_color)
    ring_l1 = link("right_hand_ring_link2_.stl", finger_color)
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=palm_lnk, child_lnk=ring_l0,
        axis=ouc.StandardAxis.X, pos=ring_base, lmt_lo=0.0, lmt_up=1.92))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=ring_l0, child_lnk=ring_l1,
        axis=ouc.StandardAxis.X,
        pos=np.array([0.0, 0.0, 0.0558], dtype=np.float32),
        lmt_lo=0.0, lmt_up=1.92))

    # ======= pinky (2 dof) ======== #
    pinky_l0 = link("right_hand_pinky_link1_.stl", finger_color)
    pinky_l1 = link("right_hand_pinky_link2_.stl", finger_color)
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=palm_lnk, child_lnk=pinky_l0,
        axis=ouc.StandardAxis.X, pos=pinky_base, lmt_lo=0.0, lmt_up=1.92))
    structure.add_jnt(orbms.Joint(
        jnt_type=ouc.JntType.REVOLUTE, parent_lnk=pinky_l0, child_lnk=pinky_l1,
        axis=ouc.StandardAxis.X,
        pos=np.array([0.0, 0.0, 0.0558], dtype=np.float32),
        lmt_lo=0.0, lmt_up=1.92))

    structure.compile()
    return structure


class XHandRight(oreb.EndEffectorBase):
    """12-dof dexterous hand (thumb 3, index 3, middle/ring/pinky 2 each).

    Joint order matches the qs/conf vector: thumb[:3], index[3:6],
    middle[6:8], ring[8:10], pinky[10:12].
    """

    @classmethod
    def _build_structure(cls):
        return prepare_ms()

    def __init__(self, rotmat=None, pos=None):
        super().__init__(
            loc_tcp_tf=oum.tf_from_rotmat_pos(
                rotmat=oum.rotmat_from_euler(oum.pi / 2, oum.pi / 2, 0),
                pos=(0.0, -0.075, 0.075)))
        if rotmat is not None or pos is not None:
            self.set_rotmat_pos(rotmat=rotmat, pos=pos)

    def goto_given_conf(self, conf):
        self.fk(qs=conf)

    def rand_conf(self):
        lo = self._compiled.jlmt_low_by_idx
        hi = self._compiled.jlmt_high_by_idx
        return np.random.uniform(lo, hi).astype(np.float32)


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    base = ovw.World(cam_pos=(0.5, -0.5, 0.5), cam_lookat_pos=(0, 0, 0.05))
    builtins.base = base
    ossop.frame().attach_to(base.scene)
    xhand = XHandRight()
    # xhand.goto_given_conf(xhand.rand_conf())
    xhand.attach_to(base.scene)
    ossop.frame(pos=xhand.gl_tcp_tf[:3, 3],
                rotmat=xhand.gl_tcp_tf[:3, :3],
                color_mat=ouc.CoordColor.MYC).attach_to(base.scene)
    base.run()
