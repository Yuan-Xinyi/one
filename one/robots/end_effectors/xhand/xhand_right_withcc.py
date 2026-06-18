"""XHandRight with a transparent self-collision-sphere overlay.

Adds a sphere-based self-collision checker and its visualization to the plain
``XHandRight`` (everything grasp-related is inherited unchanged). Self-collision
follows a single rule: only the thumb fingertip is checked against the rest of
the hand. Contacts purely among the other fingers are ignored, the proximal
thumb links are excluded (they can't reach the hand), and directly connected
links are never checked.
"""

import os

import numpy as np

import one.utils.constant as ouc
from one.robots.end_effectors.xhand.xhand_right import XHandRight as _XHandRight, _URDF_PATH


class XHandRight(_XHandRight):
    # Thumb links and the subset (the fingertip) that is actually collision-checked.
    _THUMB_LINKS = ('thumb_bend_link', 'thumb_rota_link1', 'thumb_rota_link2')
    _THUMB_TIP_LINKS = ('thumb_rota_link2',)

    @property
    def collision_checker(self):
        """Lazily-built sphere self-collision checker for this hand (shares the
        same URDF / joint order as the qs vector). Built once and cached."""
        chk = getattr(self, '_collision_checker', None)
        if chk is None:
            from one.robots.end_effectors.xhand.sphere_collision_checker import (
                SphereCollisionChecker)
            chk = SphereCollisionChecker(
                os.path.abspath(_URDF_PATH),
                focus_links=self._THUMB_TIP_LINKS,
                focus_exclude_links=[l for l in self._THUMB_LINKS
                                     if l not in self._THUMB_TIP_LINKS])
            self._collision_checker = chk
        return chk

    def collision_sphere_world(self):
        """Current collision spheres in world coordinates for the hand's pose.

        Returns ``(centers, radii, in_collision)`` where ``centers`` is ``(n, 3)``
        in world frame, ``radii`` is ``(n,)`` and ``in_collision`` is a bool mask
        of spheres taking part in a self-collision at the current ``qs``."""
        chk = self.collision_checker
        local, hits = chk.check_collisions(np.asarray(self.qs, dtype=float))
        root = self.runtime_root_lnk.tf                 # world pose of the palm root
        centers = np.asarray(local) @ root[:3, :3].T + root[:3, 3]
        return centers, np.asarray(chk.sphere_radii), np.asarray(hits)

    def gen_collision_spheres(self, alpha=0.4, highlight=True,
                              free_rgb=ouc.BasicColor.CYAN,
                              hit_rgb=ouc.BasicColor.RED):
        """Build transparent (visual-only) SceneObject spheres for the current
        collision model. With ``highlight`` the spheres in self-collision are
        colored ``hit_rgb`` and the rest ``free_rgb``."""
        import one.scene.scene_object_primitive as ossop
        centers, radii, hits = self.collision_sphere_world()
        return [ossop.sphere(pos=tuple(map(float, c)), radius=float(r),
                             rgb=(hit_rgb if highlight and hit else free_rgb),
                             alpha=alpha, collision_type=None)
                for c, r, hit in zip(centers, radii, hits)]

    def show_collision_spheres(self, scene, **kwargs):
        """Build the collision spheres (``gen_collision_spheres`` kwargs) and
        attach them to ``scene``, replacing any previously shown set. Returns the
        list of attached SceneObjects."""
        self.hide_collision_spheres()
        self._sphere_scene = scene
        self._collision_sphere_objs = self.gen_collision_spheres(**kwargs)
        for o in self._collision_sphere_objs:
            o.attach_to(scene)
        return self._collision_sphere_objs

    def hide_collision_spheres(self):
        """Detach the spheres shown by ``show_collision_spheres`` (no-op if none)."""
        scene = getattr(self, '_sphere_scene', None)
        for o in getattr(self, '_collision_sphere_objs', []):
            o.detach_from(scene)
        self._collision_sphere_objs = []


if __name__ == '__main__':
    import builtins
    import one.viewer.world as ovw
    import one.scene.scene_object_primitive as ossop

    # one hand per grasp primitive, spread along y so they don't overlap.
    # (name, action, which center tcp to draw)
    PREVIEW = [
        ('open',   lambda h: h.open_hand(),      None),
        ('pinch',  lambda h: h.pinch(1.0),       'pinch_center'),
        ('tripod', lambda h: h.tripod(1.0),      'pinch_center'),
        ('power',  lambda h: h.power(1.0),       'power_center'),
    ]
    dy = 0.22   # y spacing between hands

    base = ovw.World(cam_pos=(0.6, -0.4, 0.5),
                     cam_lookat_pos=(0.0, -0.5 * dy * (len(PREVIEW) - 1), 0.08))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    hands = []
    for i, (name, action, tcp) in enumerate(PREVIEW):
        hand = XHandRight(pos=np.array([0.0, -i * dy, 0.0], dtype=np.float32))
        action(hand)
        hand.attach_to(base.scene)
        # small frame at the hand base to mark each one
        ossop.frame(pos=hand.runtime_root_lnk.tf[:3, 3],
                    rotmat=hand.runtime_root_lnk.tf[:3, :3]).attach_to(base.scene)
        if tcp is not None:
            hand.toggle_tcp(tcp, length_scale=0.15, radius_scale=0.25)
        # overlay the transparent self-collision spheres (red = in collision)
        spheres = hand.show_collision_spheres(base.scene, alpha=0.35)
        n_hit = int(hand.collision_sphere_world()[2].sum())
        hands.append(hand)
        print(f"hand {i}: {name}  (y = {-i * dy:.2f})  "
              f"spheres={len(spheres)} in_collision={n_hit}")
    builtins.hands = hands
    base.run()
