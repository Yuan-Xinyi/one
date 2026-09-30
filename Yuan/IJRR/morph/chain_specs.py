"""Serial-chain specifications for morphology-general training.

Every arm is a serial chain: joint i has a local transform (rotation, position)
in its parent frame, an axis in its own frame, position limits and a velocity
limit; the tool sits tcp_offset along the last frame's z past flange_pos.
FR3, xArm7 and Cobotta are the real arms (the FR3 spec reproduces
BatchedFR3Kinematics exactly, see verify_fr3_spec); perturb_spec draws
random variants by rescaling the link translations and the joint speed
limits, with the collision spheres of each link rescaled along.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from Yuan.IJRR.kinematics.batched_chain_kin import (BatchedChainKinematics, SPECS,
                                                    _rotx, _EYE3)

_HERE = Path(__file__).resolve().parents[1]                     # Yuan/IJRR
_FR3_SPHERES = _HERE.parents[1] / 'one/robots/manipulators/franka/fr3/collision_spheres'
_CHAIN_SPHERES = _HERE / 'kinematics/spheres'
FR3_LINK_NAMES = ['link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']

FR3 = dict(
    name='fr3',
    joints=[
        (_EYE3,             [0.0, 0.0, 0.333],    [0.0, 0.0, 1.0]),
        (_rotx(-math.pi/2), [0.0, 0.0, 0.0],      [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0, -0.316, 0.0],   [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0825, 0.0, 0.0],   [0.0, 0.0, 1.0]),
        (_rotx(-math.pi/2), [-0.0825, 0.384, 0.0], [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0, 0.0, 0.0],      [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.088, 0.0, 0.0],    [0.0, 0.0, 1.0]),
    ],
    lmt_lo=[-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    lmt_up=[2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    qdot_max=[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],   # the dir-frac limits
    flange_pos=[0.0, 0.0, 0.107],
    tcp_offset=0.2034,
    v=0.2,
)
XARM7 = dict(copy.deepcopy(SPECS['xarm7']), tcp_offset=0.10, v=0.2)
XARM7['qdot_max'] = [3.14] * 7
COBOTTA = dict(copy.deepcopy(SPECS['cobotta']), tcp_offset=0.10, v=0.05)
COBOTTA['qdot_max'] = [0.4032, 0.3922, 0.7257, 0.7391, 0.7391, 1.1093]   # factory limits
BASE = {'fr3': FR3, 'xarm7': XARM7, 'cobotta': COBOTTA}


def load_spheres(robot: str):
    """(centers (S,3), radii (S,), link_idx (S,)) numpy, link frames."""
    centers, radii, lidx = [], [], []
    if robot == 'fr3':
        files = [(_FR3_SPHERES / f'{n}-spheres.json', i) for i, n in enumerate(FR3_LINK_NAMES)]
    else:
        d = _CHAIN_SPHERES / robot
        n_links = len(BASE[robot]['joints']) + 1
        files = [(d / f'link{i}-spheres.json', i) for i in range(n_links)]
    for path, i in files:
        rec = json.loads(Path(path).read_text())
        spheres = rec[0]['spheres'] if isinstance(rec, list) else rec['spheres']
        for s in spheres:
            centers.append(s['origin']); radii.append(float(s['radius'])); lidx.append(i)
    return np.asarray(centers, np.float32), np.asarray(radii, np.float32), np.asarray(lidx, np.int64)


def with_spheres(spec: dict, robot: str | None = None) -> dict:
    """Attach the base arm's spheres to a spec (in place, returns spec)."""
    robot = robot or spec.get('base', spec['name'])
    c, r, l = load_spheres(robot)
    spec['spheres'] = dict(centers=c, radii=r, link_idx=l)
    spec.setdefault('base', robot)
    return spec


def perturb_spec(base: dict, rng: np.random.Generator, name: str,
                 len_scale=(0.75, 1.25), qd_scale=(0.7, 1.3)) -> dict:
    """Random variant: every joint's local translation scaled by one factor
    per joint, joint speed limits scaled per joint; the spheres of the link
    that a joint's translation spans are scaled by the same factor."""
    spec = copy.deepcopy(base)
    if 'spheres' not in spec:
        with_spheres(spec)
    spec['name'] = name
    n = len(spec['joints'])
    s = rng.uniform(len_scale[0], len_scale[1], size=n)
    joints = []
    for i, (rot, pos, axis) in enumerate(spec['joints']):
        joints.append((rot, [float(p) * float(s[i]) for p in pos], axis))
    spec['joints'] = joints
    spec['qdot_max'] = [float(q) * float(rng.uniform(qd_scale[0], qd_scale[1])) for q in spec['qdot_max']]
    # link i (frame after joint i, i>=1) carries the geometry up to joint
    # i+1, whose translation was scaled by s[i]; link 0 is the base.
    c = spec['spheres']['centers'].copy(); l = spec['spheres']['link_idx']
    for i in range(1, n):
        c[l == i] *= float(s[i])
    spec['spheres'] = dict(centers=c, radii=spec['spheres']['radii'].copy(), link_idx=l.copy())
    spec['len_scale'] = s.tolist()
    return spec


class ArraySphereCollision:
    """Sphere self-collision from arrays (same interface as the class-based
    checkers); pairs on the same, adjacent or two-apart links are ignored."""

    def __init__(self, centers, radii, link_idx, device=None, dtype=torch.float32, margin: float = 0.0):
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype; self.margin = float(margin)
        self.centers = torch.as_tensor(centers, device=self.device, dtype=dtype)
        self.radii = torch.as_tensor(radii, device=self.device, dtype=dtype)
        self.link_indices = torch.as_tensor(link_idx, device=self.device, dtype=torch.long)
        li, lj = self.link_indices[:, None], self.link_indices[None, :]
        upper = torch.triu(torch.ones(len(radii), len(radii), dtype=torch.bool, device=self.device), diagonal=1)
        self.mask = ((li - lj).abs() > 2) & upper

    def sphere_positions(self, link_tfs):
        tfs = link_tfs.to(device=self.device, dtype=self.dtype)[:, self.link_indices]
        ch = torch.cat([self.centers, torch.ones((self.centers.shape[0], 1), device=self.device, dtype=self.dtype)], -1)
        return (tfs @ ch.view(1, -1, 4, 1)).squeeze(-1)[..., :3]

    def margins(self, link_tfs):
        c = self.sphere_positions(link_tfs)
        dist = torch.linalg.norm(c[:, :, None, :] - c[:, None, :, :], dim=-1)
        return dist - (self.radii[:, None] + self.radii[None, :])

    def min_margin(self, link_tfs):
        m = self.margins(link_tfs)
        return torch.where(self.mask.unsqueeze(0), m, torch.full_like(m, 1e6)).amin(dim=(1, 2))

    def is_collided(self, link_tfs, margin=None):
        thr = self.margin if margin is None else float(margin)
        return self.min_margin(link_tfs) < thr


def build_kin_collision(spec: dict, device):
    """(BatchedChainKinematics, ArraySphereCollision) for a spec."""
    if 'spheres' not in spec:
        with_spheres(spec)
    kin = BatchedChainKinematics(spec, device=device, tcp_offset=float(spec.get('tcp_offset', 0.10)))
    sp = spec['spheres']
    coll = ArraySphereCollision(sp['centers'], sp['radii'], sp['link_idx'], device=device)
    return kin, coll


def register_specs(specs) -> None:
    """Make the arms rebuildable from their name alone (the pool
    feasibility filter constructs a temp env from the config's robot name)."""
    from Yuan.IJRR.env import env as _env
    for s in specs:
        _env.CUSTOM_ARMS[s['name']] = (lambda dev, s=s: build_kin_collision(s, dev))


def static_joint_features(spec: dict) -> np.ndarray:
    """Per-joint static token part (n, 16): local pos (3), local rotation
    as its first two columns (6), local axis (3), limits / pi (2), speed
    limit / 3.14 (1), position index from the base (1)."""
    n = len(spec['joints'])
    out = np.zeros((n, 16), np.float32)
    for i, (rot, pos, axis) in enumerate(spec['joints']):
        R = np.asarray(rot, np.float32)
        out[i, 0:3] = pos
        out[i, 3:9] = np.concatenate([R[:, 0], R[:, 1]])
        out[i, 9:12] = axis
        out[i, 12] = spec['lmt_lo'][i] / math.pi
        out[i, 13] = spec['lmt_up'][i] / math.pi
        out[i, 14] = spec['qdot_max'][i] / 3.14
        out[i, 15] = (i + 1) / n
    return out


def verify_fr3_spec(device='cpu'):
    """The FR3 chain spec must reproduce BatchedFR3Kinematics exactly."""
    from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
    ref = BatchedFR3Kinematics(device=device, tcp_offset=0.2034)
    kin = BatchedChainKinematics(FR3, device=device, tcp_offset=0.2034)
    q = ref.rand_conf_batch(256)
    p0, R0, J0, _ = ref.tcp_fk_jac(q); p1, R1, J1, _ = kin.tcp_fk_jac(q)
    e = max(float((p0 - p1).abs().max()), float((R0 - R1).abs().max()), float((J0 - J1).abs().max()))
    T0 = ref.link_transforms(q); T1 = kin.link_transforms(q)
    e2 = float((T0 - T1).abs().max())
    print(f'FR3 spec vs BatchedFR3Kinematics: max |diff| tip/R/J {e:.2e}, link transforms {e2:.2e}')
    return e < 1e-5 and e2 < 1e-5


if __name__ == '__main__':
    assert verify_fr3_spec()
    rng = np.random.default_rng(0)
    v = perturb_spec(with_spheres(copy.deepcopy(FR3)), rng, 'fr3_v0')
    kin, coll = build_kin_collision(v, 'cpu')
    q = kin.rand_conf_batch(64)
    print('variant', v['name'], 'len scales', np.round(v['len_scale'], 2), 'collided', int(coll.is_collided(kin.link_transforms(q)).sum()), '/ 64')
    print('static features', static_joint_features(v).shape)
