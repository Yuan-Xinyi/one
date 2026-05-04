#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_plane_trajectories.hdf5"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks.hdf5"

FR3_QDOT_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=np.float32)
FR3_QDDOT_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float32)
FR3_JERK_MAX = np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0], dtype=np.float32)

TOKEN_KIND_START = 0
TOKEN_KIND_SEGMENT = 1
TOKEN_KIND_CORNER = 2

TOKEN_LAYOUT = {
    "kind_onehot": 3,
    "dir_local": 3,
    "len_norm": 1,
    "delta_theta_sincos": 2,
    "axis_local": 3,
    "bisector_local": 3,
    "plane_normal_local": 3,
    "cum_len_norm": 1,
    "fourier_time": 8,
    "pad": 5,
}
TOKEN_DIM = sum(TOKEN_LAYOUT.values())  # 32

# Minimum sub-segment span (in steps) — avoid degenerate zero-length segments after slicing.
MIN_SUBSEG_STEPS = 20


@dataclass
class StitchConfig:
    # === Primary criterion: workspace intersection (TCP proximity) ===
    # Two anchors are candidates iff their TCP positions are within tcp_eps_m meters.
    tcp_eps_m: float = 0.02
    anchor_stride: int = 10

    # === Secondary criterion: C-space similarity ===
    # Among TCP-proximate candidates, keep only those whose weighted joint-space
    # distance ||W(q_A - q_B)|| < c_eps with W = diag(1 / span_i) is small enough
    # to imply they share a local IK branch. Dimensionless.
    c_eps: float = 0.1

    # === Optional further gates (default OFF) ===
    enforce_plane: bool = False
    plane_cos_threshold: float = 0.99
    enforce_physics: bool = False
    window_steps: int = 8
    vel_ratio: float = 0.5
    acc_ratio: float = 1.0
    jerk_ratio: float = 1.5

    # === Composite path enumeration ===
    max_hops: int = 3
    max_composites_per_seed: int = 16
    length_ref: float = 0.30
    fourier_bands: int = 4


@dataclass
class Trajectory:
    idx: int
    q: np.ndarray            # (T, 7)
    tcp: np.ndarray          # (T, 3)
    direction: np.ndarray    # (3,)
    plane_normal: np.ndarray # (3,)
    plane_point: np.ndarray  # (3,)
    plane_side: float
    length: float
    termination: int


@dataclass
class SubSegment:
    """A sliced portion of a raw trajectory: q[start:end], tcp[start:end]."""
    traj_id: int
    start: int   # inclusive
    end: int     # exclusive


def load_raw_trajectories(path: Path) -> tuple[list[Trajectory], dict]:
    """Load trajectories, auto-detecting one of two layouts:

    (A) `pen_fr3_plane_trajectories.hdf5`:
        /traj_XXXXXX  with q, tcp_pos, direction, plane_normal, plane_point as datasets
                      and plane_side, total_projected_length, num_points, termination_code as attrs.
    (B) `franka_research_3_gpu_trajectories_sub*.hdf5`:
        /trajectories/traj_XXXXXX  with q, tcp_pos as datasets;
                      direction, target_normal, start_pos, total_projected_length, num_points,
                      termination_code as attrs. No plane_side.
    """
    trajs: list[Trajectory] = []
    with h5py.File(path, "r") as f:
        root_attrs = {k: f.attrs[k] for k in f.attrs.keys()}
        if "trajectories" in f and isinstance(f["trajectories"], h5py.Group):
            container = f["trajectories"]
            layout = "gpu_nullspace"
        else:
            container = f
            layout = "plane_dataset"
        keys = sorted(k for k in container.keys() if k.startswith("traj_"))

        for idx, key in enumerate(keys):
            g = container[key]
            if int(g.attrs["termination_code"]) == 0:
                continue
            if int(g.attrs["num_points"]) < 3 * MIN_SUBSEG_STEPS:
                continue

            if layout == "plane_dataset":
                q_arr = g["q"][()]
                tcp_arr = g["tcp_pos"][()]
                direction = np.asarray(g["direction"][()], dtype=np.float32)
                plane_normal = np.asarray(g["plane_normal"][()], dtype=np.float32)
                plane_point = np.asarray(g["plane_point"][()], dtype=np.float32)
                plane_side = float(g.attrs["plane_side"])
            else:  # gpu_nullspace
                q_arr = g["q"][()]
                tcp_arr = g["tcp_pos"][()]
                direction = np.asarray(g.attrs["direction"], dtype=np.float32)
                plane_normal = np.asarray(g.attrs["target_normal"], dtype=np.float32)
                plane_point = np.asarray(g.attrs["start_pos"], dtype=np.float32)
                plane_side = -1.0  # schema convention from the source generator
            trajs.append(
                Trajectory(
                    idx=idx,
                    q=np.asarray(q_arr, dtype=np.float32),
                    tcp=np.asarray(tcp_arr, dtype=np.float32),
                    direction=direction,
                    plane_normal=plane_normal,
                    plane_point=plane_point,
                    plane_side=plane_side,
                    length=float(g.attrs["total_projected_length"]),
                    termination=int(g.attrs["termination_code"]),
                )
            )
    print(f"[load] read {len(trajs)} usable trajectories from {path} (layout={layout})")
    return trajs, root_attrs


def joint_span(trajs: list[Trajectory]) -> np.ndarray:
    """Per-joint range across the entire dataset (used as weighting denominator)."""
    all_q = np.concatenate([t.q for t in trajs], axis=0)
    span = all_q.max(0) - all_q.min(0)
    return np.maximum(span, 1e-3).astype(np.float32)


def c_space_embed(q: np.ndarray, span: np.ndarray) -> np.ndarray:
    """Weighted C-space embedding: W q where W = diag(1 / span).

    Distance in this space is dimensionless — 1.0 unit ≈ full joint range on one axis.
    The norm is the core topological adjacency metric:  ||W(q_A - q_B)|| < c_eps.
    """
    return q.astype(np.float32) / span


def build_anchors(
    trajs: list[Trajectory], stride: int, span: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-anchor arrays: list-position (N,), step_idx (N,), tcp (N,3), q_embed (N,7)."""
    pos_list, step_list, tcp_list, q_list = [], [], [], []
    for pos, t in enumerate(trajs):
        T = t.q.shape[0]
        steps = list(range(0, T, stride))
        if steps[-1] != T - 1:
            steps.append(T - 1)
        for s in steps:
            pos_list.append(pos)
            step_list.append(s)
            tcp_list.append(t.tcp[s])
            q_list.append(c_space_embed(t.q[s], span))
    return (
        np.asarray(pos_list, dtype=np.int32),
        np.asarray(step_list, dtype=np.int32),
        np.stack(tcp_list, axis=0).astype(np.float32),
        np.stack(q_list, axis=0).astype(np.float32),
    )


def poly_endpoint_kinematics(q_window: np.ndarray, dt: float, t_star: float) -> tuple[np.ndarray, np.ndarray]:
    w = q_window.shape[0]
    t = np.arange(w, dtype=np.float64) * dt
    coefs = np.polyfit(t, q_window.astype(np.float64), deg=3)
    c3, c2, c1, _ = coefs[0], coefs[1], coefs[2], coefs[3]
    v = 3.0 * c3 * t_star ** 2 + 2.0 * c2 * t_star + c1
    a = 6.0 * c3 * t_star + 2.0 * c2
    return v.astype(np.float32), a.astype(np.float32)


def seam_ok(tail_q: np.ndarray, head_q: np.ndarray, dt: float, cfg: StitchConfig) -> tuple[bool, float]:
    W = cfg.window_steps
    if tail_q.shape[0] < W or head_q.shape[0] < W:
        return False, np.inf
    v_tail, a_tail = poly_endpoint_kinematics(tail_q[-W:], dt, t_star=(W - 1) * dt)
    v_head, a_head = poly_endpoint_kinematics(head_q[:W], dt, t_star=0.0)
    dv = np.abs(v_head - v_tail)
    da = np.abs(a_head - a_tail)
    dj = np.abs(da) / max(dt * W, 1e-6)
    if np.any(dv > cfg.vel_ratio * FR3_QDOT_MAX):
        return False, np.inf
    if np.any(da > cfg.acc_ratio * FR3_QDDOT_MAX):
        return False, np.inf
    if np.any(dj > cfg.jerk_ratio * FR3_JERK_MAX):
        return False, np.inf
    return True, float(np.exp(-np.linalg.norm(dv / FR3_QDOT_MAX) - 0.5 * np.linalg.norm(da / FR3_QDDOT_MAX)))


def build_anchor_edges(
    trajs: list[Trajectory], dt: float, cfg: StitchConfig
) -> list[list[tuple[int, int, int, float]]]:
    """Workspace-first stitching.

    1) Primary — workspace intersection: anchors must be within ``tcp_eps_m`` meters
       in TCP space. This keeps composite trajectories visually connected.
    2) Secondary — C-space similarity: among TCP-proximate candidates, keep only
       those whose weighted joint-space distance ||W(q_A - q_B)|| < c_eps with
       W = diag(1/span). Prevents stitching across different IK branches that
       happen to share a TCP point.
    3) Optional — plane agreement, seam physics (off by default).

    Returns: edges_per_traj[a] = list of (exit_step_on_A, entry_traj_pos, entry_step_on_B, score).
    """
    span = joint_span(trajs)
    pos_id, step_idx, tcp_all, q_embed = build_anchors(trajs, cfg.anchor_stride, span)
    print(
        f"[anchors] total anchors={len(pos_id)} across {len(trajs)} trajectories "
        f"(stride={cfg.anchor_stride}); span={np.array2string(span, precision=3, suppress_small=True)}"
    )

    tree = cKDTree(tcp_all)
    pairs = tree.query_pairs(r=cfg.tcp_eps_m, output_type="ndarray")
    print(f"[anchors] raw TCP pairs under tcp_eps={cfg.tcp_eps_m*100:.1f}cm: {len(pairs)}")

    edges_per_traj: list[list[tuple[int, int, int, float]]] = [[] for _ in range(len(trajs))]
    if len(pairs) == 0:
        return edges_per_traj

    kept = rej_same = rej_len = rej_c = rej_plane = rej_phys = 0

    for u, v in pairs:
        for i_idx, j_idx in ((u, v), (v, u)):
            pos_a, step_a = int(pos_id[i_idx]), int(step_idx[i_idx])
            pos_b, step_b = int(pos_id[j_idx]), int(step_idx[j_idx])
            if pos_a == pos_b:
                rej_same += 1
                continue
            A, B = trajs[pos_a], trajs[pos_b]
            if step_a < MIN_SUBSEG_STEPS:
                rej_len += 1
                continue
            if step_b > B.q.shape[0] - MIN_SUBSEG_STEPS:
                rej_len += 1
                continue
            # Secondary: C-space similarity
            c_dist = float(np.linalg.norm(q_embed[i_idx] - q_embed[j_idx]))
            if c_dist > cfg.c_eps:
                rej_c += 1
                continue
            if cfg.enforce_plane:
                if float(np.dot(A.plane_normal, B.plane_normal)) < cfg.plane_cos_threshold:
                    rej_plane += 1
                    continue
                if A.plane_side * B.plane_side < 0:
                    rej_plane += 1
                    continue
            if cfg.enforce_physics:
                ok, _ = seam_ok(
                    A.q[max(0, step_a - cfg.window_steps) : step_a + 1],
                    B.q[step_b : step_b + cfg.window_steps + 1],
                    dt, cfg,
                )
                if not ok:
                    rej_phys += 1
                    continue
            tcp_dist = float(np.linalg.norm(tcp_all[i_idx] - tcp_all[j_idx]))
            score = float(
                np.exp(-tcp_dist / max(cfg.tcp_eps_m, 1e-9) - c_dist / max(cfg.c_eps, 1e-9))
            )
            edges_per_traj[pos_a].append((step_a, pos_b, step_b, score))
            kept += 1
    print(
        f"[edges] kept={kept} "
        f"(reject: same_traj={rej_same} short={rej_len} c_space={rej_c} "
        f"plane={rej_plane} phys={rej_phys})"
    )
    return edges_per_traj


def enumerate_paths(
    trajs: list[Trajectory],
    edges: list[list[tuple[int, int, int, float]]],
    cfg: StitchConfig,
) -> list[list[SubSegment]]:
    """Depth-first enumerate composite paths as lists of SubSegments.

    Each composite path starts at some traj a's head (step 0), walks along a until an exit anchor
    (step_a), jumps to traj b at entry step_b, etc. The terminal segment runs to its own tail.
    """
    paths: list[list[SubSegment]] = []
    # Cap paths emitted from a single seed to avoid combinatorial blow-up when out-degree is large.
    # 0 means "no cap" (legacy behavior); any positive value stops DFS for that seed once reached.
    per_seed_cap = int(cfg.max_composites_per_seed)
    for a_idx in range(len(trajs)):
        T_a = trajs[a_idx].q.shape[0]
        seed_count = [0]  # closure-mutable counter for paths rooted at this seed

        def dfs(prefix_segs: list[SubSegment], cur_idx: int, cur_entry: int):
            if per_seed_cap > 0 and seed_count[0] >= per_seed_cap:
                return
            # Option 1: terminate here
            T_cur = trajs[cur_idx].q.shape[0]
            terminal = SubSegment(traj_id=cur_idx, start=cur_entry, end=T_cur)
            if terminal.end - terminal.start >= MIN_SUBSEG_STEPS:
                paths.append(prefix_segs + [terminal])
                seed_count[0] += 1
                if per_seed_cap > 0 and seed_count[0] >= per_seed_cap:
                    return
            if len(prefix_segs) + 1 >= cfg.max_hops:
                return
            # Option 2: extend via any outgoing edge from cur_idx with exit > cur_entry + MIN_SUBSEG_STEPS
            for exit_step, next_idx, entry_step, _ in edges[cur_idx]:
                if per_seed_cap > 0 and seed_count[0] >= per_seed_cap:
                    return
                if exit_step - cur_entry < MIN_SUBSEG_STEPS:
                    continue
                if any(seg.traj_id == next_idx for seg in prefix_segs) or next_idx == cur_idx:
                    continue
                new_seg = SubSegment(traj_id=cur_idx, start=cur_entry, end=exit_step + 1)
                dfs(prefix_segs + [new_seg], next_idx, entry_step)

        dfs([], a_idx, 0)

    seed_counts: dict[int, int] = {}
    filtered: list[list[SubSegment]] = []
    for p in paths:
        seed = p[0].traj_id
        seed_counts.setdefault(seed, 0)
        if seed_counts[seed] >= cfg.max_composites_per_seed:
            continue
        seed_counts[seed] += 1
        filtered.append(p)
    seg_hist = np.bincount([len(p) for p in filtered]) if filtered else np.array([])
    print(
        f"[enum] total paths={len(paths)} kept={len(filtered)} "
        f"seg_count_hist={seg_hist.tolist()}"
    )
    return filtered


def build_local_frame(first_dir: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    x = first_dir / max(float(np.linalg.norm(first_dir)), 1e-12)
    z = plane_normal / max(float(np.linalg.norm(plane_normal)), 1e-12)
    z = z - np.dot(z, x) * x
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    y = np.cross(z, x)
    y = y / max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack([x, y, z]).astype(np.float32)


def fourier_features(s: float, bands: int) -> np.ndarray:
    ks = 2.0 ** np.arange(bands)
    out = np.empty(2 * bands, dtype=np.float32)
    out[0::2] = np.sin(ks * np.pi * s)
    out[1::2] = np.cos(ks * np.pi * s)
    return out


def token_fill(
    kind: int, dir_local: np.ndarray, length_m: float, delta_theta: float,
    axis_local: np.ndarray, bisector_local: np.ndarray, plane_normal_local: np.ndarray,
    cum_len_m: float, cfg: StitchConfig,
) -> np.ndarray:
    tok = np.zeros(TOKEN_DIM, dtype=np.float32)
    off = 0
    tok[off + kind] = 1.0
    off += TOKEN_LAYOUT["kind_onehot"]
    tok[off : off + 3] = dir_local; off += 3
    tok[off] = length_m / cfg.length_ref; off += 1
    tok[off] = np.sin(delta_theta); tok[off + 1] = np.cos(delta_theta); off += 2
    tok[off : off + 3] = axis_local; off += 3
    tok[off : off + 3] = bisector_local; off += 3
    tok[off : off + 3] = plane_normal_local; off += 3
    tok[off] = cum_len_m / cfg.length_ref; off += 1
    tok[off : off + 2 * cfg.fourier_bands] = fourier_features(cum_len_m / cfg.length_ref, cfg.fourier_bands)
    return tok


def tokenize_subseg_path(
    subsegs: list[SubSegment], trajs: list[Trajectory], cfg: StitchConfig
) -> dict:
    # Per-subseg TCP chunks and direction (from start->end of each subseg).
    tcp_chunks = [trajs[s.traj_id].tcp[s.start : s.end] for s in subsegs]
    q_chunks = [trajs[s.traj_id].q[s.start : s.end] for s in subsegs]
    plane_normal = trajs[subsegs[0].traj_id].plane_normal.astype(np.float32)
    seg_dirs = []
    seg_lens = []
    for tcp in tcp_chunks:
        if tcp.shape[0] < 2:
            seg_dirs.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            seg_lens.append(0.0)
            continue
        delta = tcp[-1] - tcp[0]
        ln = float(np.linalg.norm(delta))
        seg_dirs.append((delta / max(ln, 1e-12)).astype(np.float32))
        seg_lens.append(ln)

    R = build_local_frame(seg_dirs[0], plane_normal)
    R_T = R.T
    plane_normal_local = (R_T @ plane_normal).astype(np.float32)
    dirs_local = [R_T @ d for d in seg_dirs]

    tokens: list[np.ndarray] = []
    cum_len = 0.0
    tokens.append(token_fill(
        kind=TOKEN_KIND_START,
        dir_local=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        length_m=0.0, delta_theta=0.0,
        axis_local=np.zeros(3, dtype=np.float32),
        bisector_local=np.zeros(3, dtype=np.float32),
        plane_normal_local=plane_normal_local, cum_len_m=0.0, cfg=cfg,
    ))

    for k, seg in enumerate(subsegs):
        tokens.append(token_fill(
            kind=TOKEN_KIND_SEGMENT,
            dir_local=dirs_local[k], length_m=seg_lens[k], delta_theta=0.0,
            axis_local=np.zeros(3, dtype=np.float32),
            bisector_local=np.zeros(3, dtype=np.float32),
            plane_normal_local=plane_normal_local, cum_len_m=cum_len, cfg=cfg,
        ))
        cum_len += seg_lens[k]
        if k + 1 < len(subsegs):
            v1, v2 = dirs_local[k], dirs_local[k + 1]
            cross = np.cross(v1, v2)
            sin_t = float(np.dot(cross, plane_normal_local))
            cos_t = float(np.dot(v1, v2))
            delta_theta = float(np.arctan2(sin_t, cos_t))
            axis = cross / max(float(np.linalg.norm(cross)), 1e-12)
            bisect = v1 + v2
            bisect = bisect / max(float(np.linalg.norm(bisect)), 1e-12)
            tokens.append(token_fill(
                kind=TOKEN_KIND_CORNER,
                dir_local=np.zeros(3, dtype=np.float32),
                length_m=0.0, delta_theta=delta_theta,
                axis_local=axis.astype(np.float32),
                bisector_local=bisect.astype(np.float32),
                plane_normal_local=plane_normal_local, cum_len_m=cum_len, cfg=cfg,
            ))

    q_concat = [q_chunks[0]]
    tcp_concat = [tcp_chunks[0]]
    for k in range(1, len(subsegs)):
        prev, cur = subsegs[k - 1], subsegs[k]
        contiguous = (prev.traj_id == cur.traj_id) and (prev.end == cur.start)
        if contiguous:
            q_concat.append(q_chunks[k])
            tcp_concat.append(tcp_chunks[k])
        else:
            q_concat.append(q_chunks[k][1:])
            tcp_concat.append(tcp_chunks[k][1:])
    q_full = np.concatenate(q_concat, axis=0).astype(np.float32)
    tcp_full = np.concatenate(tcp_concat, axis=0).astype(np.float32)

    # Sub-segment metadata: per-segment (traj_id, start, end) triples.
    ss_meta = np.asarray([[s.traj_id, s.start, s.end] for s in subsegs], dtype=np.int32)
    seg_step_counts = np.asarray([s.end - s.start for s in subsegs], dtype=np.int32)

    return {
        "tokens": np.stack(tokens, axis=0).astype(np.float32),
        "q": q_full,
        "tcp": tcp_full,
        "start_q": q_chunks[0][0].astype(np.float32),
        "local_frame": R.astype(np.float32),
        "local_origin": tcp_chunks[0][0].astype(np.float32),
        "plane_normal": plane_normal,
        "subseg_meta": ss_meta,           # (K, 3)
        "seg_step_counts": seg_step_counts, # (K,)
        "seg_count": np.int32(len(subsegs)),
        "total_length": np.float32(sum(seg_lens)),
    }


def write_hdf5(
    out_path: Path, source_meta: dict, trajs: list[Trajectory],
    tasks: list[dict], cfg: StitchConfig,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_tokens_per_task = np.asarray([t["tokens"].shape[0] for t in tasks], dtype=np.int64)
    num_q_per_task = np.asarray([t["q"].shape[0] for t in tasks], dtype=np.int64)
    tok_offsets = np.concatenate([[0], np.cumsum(num_tokens_per_task)]).astype(np.int64)
    q_offsets = np.concatenate([[0], np.cumsum(num_q_per_task)]).astype(np.int64)
    token_flat = np.concatenate([t["tokens"] for t in tasks], axis=0).astype(np.float32)
    kind_flat = np.concatenate([np.argmax(t["tokens"][:, :3], axis=1) for t in tasks], axis=0).astype(np.uint8)
    q_flat = np.concatenate([t["q"] for t in tasks], axis=0).astype(np.float32)
    tcp_flat = np.concatenate([t["tcp"] for t in tasks], axis=0).astype(np.float32)
    local_frames = np.stack([t["local_frame"] for t in tasks], axis=0).astype(np.float32)
    local_origins = np.stack([t["local_origin"] for t in tasks], axis=0).astype(np.float32)
    plane_normals = np.stack([t["plane_normal"] for t in tasks], axis=0).astype(np.float32)
    start_qs = np.stack([t["start_q"] for t in tasks], axis=0).astype(np.float32)
    seg_counts = np.asarray([t["seg_count"] for t in tasks], dtype=np.uint16)
    total_lens = np.asarray([t["total_length"] for t in tasks], dtype=np.float32)

    subseg_meta_flat = np.concatenate([t["subseg_meta"] for t in tasks], axis=0).astype(np.int32)  # (S_total, 3)
    subseg_offset = np.concatenate([[0], np.cumsum(np.asarray([t["subseg_meta"].shape[0] for t in tasks]))]).astype(np.int64)
    seg_step_counts_flat = np.concatenate([t["seg_step_counts"] for t in tasks], axis=0).astype(np.int32)

    seeds = sorted({int(t["subseg_meta"][0, 0]) for t in tasks})

    with h5py.File(out_path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_raw_trajectories"] = len(trajs)
        meta.attrs["num_composites"] = len(tasks)
        meta.attrs["num_seeds"] = len(seeds)
        meta.attrs["token_dim"] = TOKEN_DIM
        meta.attrs["length_ref"] = cfg.length_ref
        meta.attrs["fourier_bands"] = cfg.fourier_bands
        meta.attrs["max_hops"] = cfg.max_hops
        meta.attrs["tcp_eps_m"] = cfg.tcp_eps_m
        meta.attrs["c_eps"] = cfg.c_eps
        meta.attrs["anchor_stride"] = cfg.anchor_stride
        meta.attrs["enforce_plane"] = cfg.enforce_plane
        meta.attrs["enforce_physics"] = cfg.enforce_physics
        meta.attrs["window_steps"] = cfg.window_steps
        for k, v in source_meta.items():
            try:
                meta.attrs[f"source_{k}"] = v
            except TypeError:
                continue
        layout_grp = meta.create_group("token_layout")
        for k, v in TOKEN_LAYOUT.items():
            layout_grp.attrs[k] = int(v)

        raw = f.create_group("raw_trajs")
        raw_q_offsets = np.concatenate([[0], np.cumsum(np.asarray([t.q.shape[0] for t in trajs]))]).astype(np.int64)
        raw_q_flat = np.concatenate([t.q for t in trajs], axis=0).astype(np.float32)
        raw_tcp_flat = np.concatenate([t.tcp for t in trajs], axis=0).astype(np.float32)
        raw.create_dataset("q_flat", data=raw_q_flat, compression="gzip")
        raw.create_dataset("tcp_flat", data=raw_tcp_flat, compression="gzip")
        raw.create_dataset("offset", data=raw_q_offsets)
        raw.create_dataset("direction", data=np.stack([t.direction for t in trajs]), compression="gzip")
        raw.create_dataset("plane_normal", data=np.stack([t.plane_normal for t in trajs]), compression="gzip")
        raw.create_dataset("plane_point", data=np.stack([t.plane_point for t in trajs]), compression="gzip")
        raw.create_dataset("plane_side", data=np.asarray([t.plane_side for t in trajs], dtype=np.float32))
        raw.create_dataset("length", data=np.asarray([t.length for t in trajs], dtype=np.float32))

        ts = f.create_group("tasks")
        ts.create_dataset("token_flat", data=token_flat, compression="gzip")
        ts.create_dataset("token_offset", data=tok_offsets)
        ts.create_dataset("token_kind", data=kind_flat)
        ts.create_dataset("qtraj_flat", data=q_flat, compression="gzip")
        ts.create_dataset("qtraj_offset", data=q_offsets)
        ts.create_dataset("tcp_flat", data=tcp_flat, compression="gzip")
        ts.create_dataset("start_q", data=start_qs, compression="gzip")
        ts.create_dataset("local_frame", data=local_frames, compression="gzip")
        ts.create_dataset("local_origin", data=local_origins, compression="gzip")
        ts.create_dataset("plane_normal", data=plane_normals, compression="gzip")
        ts.create_dataset("seg_count", data=seg_counts)
        ts.create_dataset("total_length", data=total_lens)
        ts.create_dataset("subseg_meta_flat", data=subseg_meta_flat)      # (S_total, 3): traj_id, start, end
        ts.create_dataset("subseg_offset", data=subseg_offset)
        ts.create_dataset("seg_step_counts_flat", data=seg_step_counts_flat)

    print(
        f"[write] {out_path} — raw={len(trajs)} seeds={len(seeds)} tasks={len(tasks)} "
        f"token_dim={TOKEN_DIM} tokens_total={int(num_tokens_per_task.sum())}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Workspace-first stitching: TCP intersection → C-space similarity → (optional) plane/physics."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # Primary criterion
    p.add_argument("--tcp-eps-m", type=float, default=0.02,
                   help="Primary: TCP-space seam radius in meters. Anchors closer than this are candidates.")
    p.add_argument("--c-eps", type=float, default=0.1,
                   help="Secondary: weighted C-space similarity bound (dimensionless). Rejects candidates "
                        "that share TCP but live on different IK branches. W = diag(1/span).")
    p.add_argument("--anchor-stride", type=int, default=10,
                   help="Subsample an anchor every N steps along each raw trajectory.")
    # Optional topological filters (off by default — maximize graph connectivity)
    p.add_argument("--enforce-plane", action="store_true",
                   help="Keep edges only between anchors whose plane normals agree (plane_cos) and same side.")
    p.add_argument("--plane-cos", type=float, default=0.99)
    p.add_argument("--enforce-physics", action="store_true",
                   help="Keep edges only when cubic-fit v/a/jerk at the seam stay within FR3 ratios.")
    p.add_argument("--window-steps", type=int, default=8)
    p.add_argument("--vel-ratio", type=float, default=0.5)
    p.add_argument("--acc-ratio", type=float, default=1.0)
    p.add_argument("--jerk-ratio", type=float, default=1.5)
    # Path enumeration
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--max-per-seed", type=int, default=16)
    p.add_argument("--length-ref", type=float, default=0.3)
    p.add_argument("--fourier-bands", type=int, default=4)
    p.add_argument("--include-singles", action="store_true")
    p.add_argument("--self-slice", type=int, default=0,
                   help="If >0, also emit tasks that split each long trajectory into N equal sub-segments "
                        "(geometrically continuous, Δθ≈0). Useful as a fallback when inter-trajectory "
                        "C-space neighbors are sparse.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StitchConfig(
        tcp_eps_m=args.tcp_eps_m,
        c_eps=args.c_eps,
        anchor_stride=args.anchor_stride,
        enforce_plane=args.enforce_plane,
        plane_cos_threshold=args.plane_cos,
        enforce_physics=args.enforce_physics,
        window_steps=args.window_steps,
        vel_ratio=args.vel_ratio,
        acc_ratio=args.acc_ratio,
        jerk_ratio=args.jerk_ratio,
        max_hops=args.max_hops,
        max_composites_per_seed=args.max_per_seed,
        length_ref=args.length_ref,
        fourier_bands=args.fourier_bands,
    )

    trajs, src_meta = load_raw_trajectories(args.input)
    if len(trajs) == 0:
        raise RuntimeError(f"No usable trajectories in {args.input}")
    dt = float(src_meta.get("dt", 0.01))

    edges = build_anchor_edges(trajs, dt, cfg)
    paths = enumerate_paths(trajs, edges, cfg)

    if args.include_singles:
        single_paths = [[SubSegment(traj_id=i, start=0, end=trajs[i].q.shape[0])] for i in range(len(trajs))]
        paths = single_paths + paths
        print(f"[enum] curriculum=on, appended {len(single_paths)} single-segment tasks")

    if args.self_slice >= 2:
        N = int(args.self_slice)
        self_slice_paths = []
        for i, t in enumerate(trajs):
            T = t.q.shape[0]
            if T < N * MIN_SUBSEG_STEPS:
                continue
            cuts = np.linspace(0, T, N + 1, dtype=np.int64)
            ss = [SubSegment(traj_id=i, start=int(cuts[k]), end=int(cuts[k + 1])) for k in range(N)]
            if all(s.end - s.start >= MIN_SUBSEG_STEPS for s in ss):
                self_slice_paths.append(ss)
        paths = paths + self_slice_paths
        print(f"[enum] self-slice={N}: appended {len(self_slice_paths)} geometrically-continuous {N}-segment tasks")

    if not paths:
        raise RuntimeError(
            "No composite paths discovered. Try --tcp-eps-m 0.02, smaller --anchor-stride, "
            "or --include-singles for a single-segment fallback."
        )

    tasks = [tokenize_subseg_path(p, trajs, cfg) for p in paths]
    token_counts = np.asarray([t["tokens"].shape[0] for t in tasks])
    seg_counts = np.asarray([int(t["seg_count"]) for t in tasks])
    print(
        f"[tok] tasks={len(tasks)} "
        f"tokens_per_task: min={token_counts.min()} max={token_counts.max()} mean={token_counts.mean():.2f} | "
        f"seg_count: min={seg_counts.min()} max={seg_counts.max()} mean={seg_counts.mean():.2f}"
    )

    write_hdf5(args.output, src_meta, trajs, tasks, cfg)


if __name__ == "__main__":
    main()
