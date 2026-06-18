import os

import jax
import jax.numpy as jnp
import xmltodict
import numpy as np

from one.robots.end_effectors.xhand.sphere_collision_util import (
    axis_angle_to_matrix, xyzrpy_to_matrix, load_link_spheres)

# Per-link sphere decompositions live next to the URDF (one file per link).
_DEFAULT_SPHERE_DIR = os.path.join(os.path.dirname(__file__), 'collision_spheres')


class SphereCollisionChecker:
    def __init__(self, urdf_path, sphere_dir=_DEFAULT_SPHERE_DIR, solution_index=-1,
                 radius_scale=1.0, link_radius_scale=None,
                 ignore_adjacent=True, focus_links=None, focus_exclude_links=None,
                 disable_default_collisions=True,
                 default_conf=None, default_margin=-0.001):
        """Self-collision checker for the XHand built from sphere swept volumes.

        The XHand URDF carries mesh collision geometry, so the spheres come from
        the SPaSM ``collision_spheres/<mesh_basename>-spheres.json`` files: each
        link is matched to its sphere file via its collision (or visual) mesh
        filename. ``solution_index`` selects which candidate set to load per link
        (default ``-1`` = the finest set). ``radius_scale`` uniformly scales every
        sphere radius (e.g. ``0.95`` to shrink the whole model to 95%);
        ``link_radius_scale`` is an optional ``{link_name: factor}`` dict applied
        on top of it for specific links (e.g. ``{'palm': 0.9}`` to shrink only the
        palm spheres). Links with no matching sphere file fall back to any
        ``<sphere>`` collision geometry embedded in the URDF.

        With ``ignore_adjacent`` (the default) two directly connected links -- a
        link and its parent across a joint -- are never checked against each
        other, since a joint always lets their fitted spheres touch.

        ``focus_links`` optionally restricts checking to a single group of links:
        when given a collection of link names, only *cross-group* pairs are kept
        (one sphere on a focus link, the other on a non-focus link). For the hand
        this is used to check only thumb-vs-rest collisions -- pass the thumb link
        names and contacts purely among the other fingers are ignored.
        ``focus_exclude_links`` (only meaningful with ``focus_links``) drops the
        named links from checking entirely -- they are neither a focus link nor
        part of "the rest" -- so e.g. the non-tip thumb links don't get tested
        against the thumb tip, leaving only thumb-tip-vs-rest-of-hand pairs.

        On a compact hand the conservative sphere fits of the palm and of
        neighboring fingers also overlap at rest; with
        ``disable_default_collisions`` (the default)
        we build a MoveIt-style allowed-collision matrix: any sphere pair already
        closer than ``default_margin`` at ``default_conf`` (the open hand, all
        zeros) is permanently ignored, so a resting hand is collision-free and
        only genuinely new contacts are reported.
        """
        self.urdf_path = urdf_path
        self.sphere_dir = sphere_dir
        self.solution_index = solution_index
        self.radius_scale = radius_scale
        self.link_radius_scale = link_radius_scale or {}
        self.focus_links = set(focus_links) if focus_links is not None else None
        self.focus_exclude_links = set(focus_exclude_links or ())
        links_dict, joints_dict, self.link_order, self.base_link = self._parse_urdf_structure(urdf_path)
        self.sphere_offsets, self.sphere_radii, self.sphere_link_indices = self._prepare_sphere_data(links_dict)
        self.joint_statics, self.joint_axes, self.joint_types, self.q_indices, self.parent_indices = self._prepare_joint_kinematics(joints_dict)
        self.num_dof = int(self.q_indices.max()) + 1 if len(self.q_indices) else 0

        self.collision_mask = self._prepare_collision_masks(
            ignore_adjacent=ignore_adjacent,
            disable_default_collisions=disable_default_collisions,
            default_conf=default_conf, default_margin=default_margin)
        
        self._jit_update = jax.jit(self.compute_sphere_positions)
        self._jit_collision_dist = jax.jit(self.compute_self_collision_dist)
        self._jit_collision_cost = jax.jit(self.self_collision_cost)
        self._jit_check_collisions = jax.jit(self.check_collisions)

    def _parse_urdf_structure(self, urdf_path):
        with open(urdf_path, 'r') as f:
            robot_data = xmltodict.parse(f.read())['robot']
        links = {l['@name']: l for l in robot_data['link']}
        joints = {j['@name']: j for j in robot_data['joint']}
        parent_map = {j['child']['@link']: j for j in joints.values()}
        base_link = [l for l in links if l not in parent_map][0]
        child_map = {}
        for j in joints.values():
            child_map.setdefault(j['parent']['@link'], []).append(j['child']['@link'])
        link_order, queue = [], [base_link]
        while queue:
            curr = queue.pop(0)
            link_order.append(curr)
            queue.extend(child_map.get(curr, []))
        return links, joints, link_order, base_link

    def _mesh_basename(self, link):
        """Return the basename (sans extension) of a link's mesh geometry, taken
        from its collision element if present otherwise its visual, or None."""
        for tag in ('collision', 'visual'):
            if tag not in link:
                continue
            elems = link[tag] if isinstance(link[tag], list) else [link[tag]]
            for el in elems:
                mesh = el.get('geometry', {}).get('mesh')
                if mesh and '@filename' in mesh:
                    return os.path.splitext(os.path.basename(mesh['@filename']))[0]
        return None

    def _link_sphere_file(self, link):
        """Path to a link's ``*-spheres.json`` (derived from its mesh name), or
        None if the link has no mesh or no matching sphere file on disk."""
        base = self._mesh_basename(link)
        if base is None:
            return None
        path = os.path.join(self.sphere_dir, f'{base}-spheres.json')
        return path if os.path.exists(path) else None

    def _prepare_sphere_data(self, links_dict):
        offsets, radii, indices = [], [], []
        name_to_idx = {name: i for i, name in enumerate(self.link_order)}
        for name in self.link_order:
            link = links_dict[name]
            scale = self.radius_scale * self.link_radius_scale.get(name, 1.0)
            sphere_file = self._link_sphere_file(link)
            if sphere_file is not None:
                # XHand: spheres come from the per-link SPaSM json.
                origins, link_radii = load_link_spheres(sphere_file, self.solution_index)
                for origin, r in zip(origins, link_radii):
                    offsets.append(xyzrpy_to_matrix(origin, [0.0, 0.0, 0.0]))
                    radii.append(float(r) * scale)
                    indices.append(name_to_idx[name])
            elif 'collision' in link:
                # Fallback: spheres embedded directly in the URDF collision tags.
                colls = link['collision'] if isinstance(link['collision'], list) else [link['collision']]
                for col in colls:
                    if 'sphere' not in col.get('geometry', {}):
                        continue
                    xyz = [float(x) for x in col['origin']['@xyz'].split()]
                    rpy = [float(x) for x in col['origin']['@rpy'].split()]
                    offsets.append(xyzrpy_to_matrix(xyz, rpy))
                    radii.append(float(col['geometry']['sphere']['@radius']) * scale)
                    indices.append(name_to_idx[name])
        return jnp.array(offsets), jnp.array(radii), jnp.array(indices)

    def _prepare_joint_kinematics(self, joints_dict):
        movable = [j for j in joints_dict.values() if j.get('@type') in ['revolute', 'prismatic']]
        j_to_q = {j['@name']: i for i, j in enumerate(movable)}
        statics, axes, types, q_idxs, p_idxs = [], [], [], [], []
        name_to_idx = {name: i for i, name in enumerate(self.link_order)}
        for name in self.link_order:
            if name == self.base_link:
                statics.append(np.eye(4)); axes.append(np.zeros(3)); types.append(0); q_idxs.append(-1); p_indices = -1
            else:
                j = next(jt for jt in joints_dict.values() if jt['child']['@link'] == name)
                statics.append(xyzrpy_to_matrix([float(x) for x in j['origin']['@xyz'].split()], [float(x) for x in j['origin']['@rpy'].split()]))
                if '@type' in j and j['@type'] in ['revolute', 'prismatic']:
                    axes.append(np.array([float(x) for x in j['axis']['@xyz'].split()]))
                    types.append(1 if j['@type'] == 'revolute' else 2)
                    q_idxs.append(j_to_q[j['@name']])
                else:
                    axes.append(np.zeros(3)); types.append(0); q_idxs.append(-1)
                p_indices = name_to_idx[j['parent']['@link']]
            p_idxs.append(p_indices)
        return jnp.array(statics), jnp.array(axes), np.array(types), np.array(q_idxs), np.array(p_idxs)

    def _prepare_collision_masks(self, ignore_adjacent=True,
                                 disable_default_collisions=True,
                                 default_conf=None, default_margin=-0.001):
        num_spheres = len(self.sphere_link_indices)
        id_i, id_j = self.sphere_link_indices[:, None], self.sphere_link_indices[None, :]
        mask = (id_i != id_j)
        if ignore_adjacent:
            for name in self.link_order:
                if name == self.base_link: continue
                c_idx = self.link_order.index(name)
                p_idx = self.parent_indices[c_idx]
                if p_idx != -1:
                    mask &= ~((id_i == p_idx) & (id_j == c_idx))
                    mask &= ~((id_i == c_idx) & (id_j == p_idx))
        mask = mask & jnp.triu(jnp.ones((num_spheres, num_spheres), dtype=bool), k=1)
        if self.focus_links is not None:
            focus_idx = {i for i, n in enumerate(self.link_order) if n in self.focus_links}
            excl_idx = {i for i, n in enumerate(self.link_order) if n in self.focus_exclude_links}
            link_ids = np.asarray(self.sphere_link_indices)
            in_focus = jnp.array([int(li) in focus_idx for li in link_ids], dtype=bool)
            in_excl = jnp.array([int(li) in excl_idx for li in link_ids], dtype=bool)
            # keep only cross-group pairs (exactly one endpoint on a focus link)
            # and drop any pair touching an excluded link.
            cross = in_focus[:, None] != in_focus[None, :]
            keep = (~in_excl[:, None]) & (~in_excl[None, :])
            mask = mask & cross & keep
        if disable_default_collisions:
            conf = jnp.zeros(self.num_dof) if default_conf is None else jnp.asarray(default_conf)
            spheres = self.compute_sphere_positions(conf)
            diff = spheres[:, None, :] - spheres[None, :, :]
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)
            margin = dist - (self.sphere_radii[:, None] + self.sphere_radii[None, :])
            # ignore any pair already in contact at the resting configuration.
            mask = mask & (margin >= default_margin)
        return mask

    def _compute_collision_data(self, q):
        spheres = self.compute_sphere_positions(q)
        diff = spheres[:, None, :] - spheres[None, :, :]
        dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        margin_matrix = dist_matrix - (self.sphere_radii[:, None] + self.sphere_radii[None, :])
        return spheres, margin_matrix

    def compute_self_collision_dist(self, q):
        _, margin_matrix = self._compute_collision_data(q)
        return jnp.min(jnp.where(self.collision_mask, margin_matrix, 1e6))

    def self_collision_cost(self, q, scale=1, min_margin=-0.005):
        _, margin_matrix = self._compute_collision_data(q)
        return jnp.sum(jax.nn.relu(min_margin - margin_matrix) * self.collision_mask) * scale

    def check_collisions(self, q, margin=-0.005):
        spheres, margin_matrix = self._compute_collision_data(q)
        pairs = (margin_matrix < margin) & self.collision_mask
        return spheres, jnp.any(pairs, axis=1) | jnp.any(pairs, axis=0)

    def compute_sphere_positions(self, q):
        num_links = len(self.link_order)
        link_transforms = jnp.tile(jnp.identity(4), (num_links, 1, 1))
        for i in range(num_links):
            p_idx = int(self.parent_indices[i])
            if p_idx == -1: continue
            T_motion = jnp.identity(4)
            j_type, q_idx = int(self.joint_types[i]), int(self.q_indices[i])
            if j_type == 1: T_motion = axis_angle_to_matrix(self.joint_axes[i], q[q_idx])
            elif j_type == 2: T_motion = T_motion.at[0:3, 3].set(self.joint_axes[i] * q[q_idx])
            link_transforms = link_transforms.at[i].set(link_transforms[p_idx] @ self.joint_statics[i] @ T_motion)
        sphere_link_ts = link_transforms[self.sphere_link_indices]
        return jnp.einsum('nij,njk->nik', sphere_link_ts, self.sphere_offsets)[:, 0:3, 3]

    def update(self, q):
        return self._jit_update(q)