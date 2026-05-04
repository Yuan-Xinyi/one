"""Retrieve a known-feasible training stroke whose shape best matches a target,
then apply a similarity transform (translate + in-plane rotate + uniform scale)
so it lines up with the character's geometry.

Why retrieval instead of pure DiT:
  - Every training task has a verified-feasible q-trajectory (data gen ran the tracker).
  - For our target stroke, the closest-by-shape training task is by definition
    in the densest part of DiT's training distribution → DiT gives strong q0
    candidates on its tokens.
  - The slight distortion introduced by the rigid alignment keeps the polyline in
    DiT's comfort zone while letting us COMPOSE it into a character (different
    starts, directions, total scale than what training had).

Pipeline:
    target polyline (canonical) ─┐
                                 ├→ retrieval index → matched training task
                                 │  (seg_count match, length/corner kNN)
                                 │
                                 ├→ similarity transform: translate · rotate · scale
                                 │  so retrieved.start = target.start, retrieved direction
                                 │  matches target's first-seg direction, total length matches.
                                 ↓
                          adjusted polyline → tokenize → DiT → IK refine → tracker
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"


@dataclass
class RetrievalMatch:
    task_idx: int
    distance: float                          # signature distance to query
    seg_lens: np.ndarray                     # (n_segs,) m, retrieved task's seg lengths
    corner_angles: np.ndarray                # (n_corners,) rad, retrieved task's corner angles
    polyline_world: np.ndarray               # (n_vertices, 3) — retrieved TCP vertices in world
    transform_R: np.ndarray                  # (3, 3) — applied rotation
    transform_scale: float                   # uniform scale factor applied
    transform_t: np.ndarray                  # (3,) — applied translation
    adjusted_polyline_world: np.ndarray      # (n_vertices, 3) — after similarity transform


class StrokeRetrieval:
    """Indexed view of the composite-tasks HDF5 for shape-based stroke retrieval."""

    def __init__(self, h5_path: Path = DEFAULT_DATA):
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            ts = f["tasks"]
            token_flat = np.asarray(ts["token_flat"][()], dtype=np.float32)
            token_kind = np.asarray(ts["token_kind"][()], dtype=np.uint8)
            self.token_offset = np.asarray(ts["token_offset"][()], dtype=np.int64)
            self.seg_count = np.asarray(ts["seg_count"][()], dtype=np.int32)
            self.local_origin = np.asarray(ts["local_origin"][()], dtype=np.float32)
            self.qtraj_offset = np.asarray(ts["qtraj_offset"][()], dtype=np.int64)
            self.length_ref = float(f["meta"].attrs["length_ref"])

        # Pre-extract per-task seg_lens and corner_angles as ragged arrays.
        # We use the start/segment/corner sequence in token_kind.
        self.task_seg_lens: list[np.ndarray] = []
        self.task_corner_angles: list[np.ndarray] = []
        # We also keep the world-frame seg directions so we can reconstruct polylines.
        self.task_dirs_local: list[np.ndarray] = []
        self.task_plane_normal_local: list[np.ndarray] = []
        for idx in range(self.seg_count.shape[0]):
            t_lo = int(self.token_offset[idx]); t_hi = int(self.token_offset[idx + 1])
            tokens = token_flat[t_lo:t_hi]; kinds = token_kind[t_lo:t_hi]
            seg_idx = np.where(kinds == 1)[0]
            corner_idx = np.where(kinds == 2)[0]
            seg_lens_norm = tokens[seg_idx, 6]
            seg_lens_m = seg_lens_norm * self.length_ref
            self.task_seg_lens.append(seg_lens_m.astype(np.float32))
            if corner_idx.size > 0:
                sin_t = tokens[corner_idx, 7]; cos_t = tokens[corner_idx, 8]
                angles = np.arctan2(sin_t, cos_t)
            else:
                angles = np.empty(0, dtype=np.float32)
            self.task_corner_angles.append(angles.astype(np.float32))
            self.task_dirs_local.append(tokens[seg_idx, 3:6].astype(np.float32))
            # plane_normal_local is at offset 15..18 (same for every token of one task; pick start token).
            start_tok_pos = np.where(kinds == 0)[0]
            if start_tok_pos.size > 0:
                self.task_plane_normal_local.append(tokens[int(start_tok_pos[0]), 15:18].astype(np.float32))
            else:
                self.task_plane_normal_local.append(np.array([0, 0, 1], dtype=np.float32))

        print(f"[retrieve] indexed {self.seg_count.shape[0]} tasks "
              f"({sum(s.shape[0] for s in self.task_seg_lens)} segments)")

    # ---- Polyline reconstruction ----

    def reconstruct_polyline_world(self, task_idx: int) -> np.ndarray:
        """Reconstruct the (n_vertices, 3) world-frame polyline for a stored task.

        Uses local_origin as start, then walks the per-segment local-frame directions
        through the recorded local_frame to get world-frame vertices.
        """
        with h5py.File(self.h5_path, "r") as f:
            local_frame = np.asarray(f["tasks/local_frame"][task_idx], dtype=np.float32)
        origin = self.local_origin[task_idx]
        dirs_local = self.task_dirs_local[task_idx]
        lens = self.task_seg_lens[task_idx]
        verts = [origin.copy()]
        cur = origin.copy()
        for d_l, L in zip(dirs_local, lens):
            d_w = local_frame @ d_l
            n = float(np.linalg.norm(d_w))
            if n < 1e-9:
                continue
            cur = cur + (d_w / n) * float(L)
            verts.append(cur.copy())
        return np.stack(verts, axis=0).astype(np.float32)

    # ---- Querying ----

    def query(
        self,
        target_polyline_world: np.ndarray,
        k: int = 5,
        position_weight: float = 2.0,     # mild: prefer nearby tasks, but shape comes first
        len_weight: float = 10.0,         # strong: total-length proportional match
        corner_weight: float = 3.0,       # strong: corner angles must look like target
        max_scale_dev: float = 0.30,      # hard filter: drop candidates >30% off length
        max_angle_dev_rad: float = 0.52,  # hard filter: drop candidates >30° corner mismatch
    ) -> list[RetrievalMatch]:
        """Find top-K training tasks that are CLOSE to the target (distance-prioritized).

        Score = position_weight · ||stored_start_xy − target_start_xy||      (translation magnitude)
              + len_weight        · |total_train / total_target − 1|         (scale deviation)
              + corner_weight     · sum |angle_train_i − angle_target_i|     (rad, looser)

        Only tasks with the same seg_count as the target are considered. Within that
        filter, retrieval prefers tasks whose start TCP is nearby — even if the corner
        angle is e.g. 87° vs target 90°, the user explicitly does NOT need exact shape.
        Similarity transform still aligns the retrieved polyline to the target geometry,
        but with weights this way the chosen training task should already sit close on
        the desk so the "transform magnitude" is small.
        """
        target_polyline = np.asarray(target_polyline_world, dtype=np.float32)
        # Compute target's intrinsic signature.
        seg_lens_t = []
        seg_dirs_t = []
        for i in range(target_polyline.shape[0] - 1):
            d = target_polyline[i + 1] - target_polyline[i]
            L = float(np.linalg.norm(d))
            if L < 1e-6:
                continue
            seg_lens_t.append(L)
            seg_dirs_t.append(d / L)
        if not seg_lens_t:
            raise ValueError("target polyline has no non-zero segments")
        seg_lens_t = np.asarray(seg_lens_t, dtype=np.float32)
        n_segs_t = len(seg_lens_t)
        total_len_t = float(seg_lens_t.sum())

        corner_angles_t = []
        for i in range(n_segs_t - 1):
            v1, v2 = seg_dirs_t[i], seg_dirs_t[i + 1]
            cos_t = float(np.dot(v1, v2))
            sin_t = float(np.linalg.norm(np.cross(v1, v2)))   # |sin| only — corner sign is ambiguous in 3D
            corner_angles_t.append(float(np.arctan2(sin_t, cos_t)))
        corner_angles_t = np.asarray(corner_angles_t, dtype=np.float32)

        # Filter by seg_count and score.
        candidate_idx = np.where(self.seg_count == n_segs_t)[0]
        if candidate_idx.size == 0:
            print(f"[retrieve] no training tasks with seg_count={n_segs_t}")
            return []

        target_start_xy = target_polyline[0, :2].astype(np.float64)
        scores = np.full(candidate_idx.shape[0], np.inf, dtype=np.float32)
        # Vectorized fast path for position term (position dominates → big speedup).
        cand_starts_xy = self.local_origin[candidate_idx, :2].astype(np.float64)
        pos_dist = np.linalg.norm(cand_starts_xy - target_start_xy[None, :], axis=1)  # (n_cand,) m

        for j, idx in enumerate(candidate_idx):
            seg_lens_db = self.task_seg_lens[idx]
            if seg_lens_db.shape[0] != n_segs_t:
                continue
            total_len_db = float(np.sum(seg_lens_db))
            # Length: relative scale-deviation from 1.0 (so big targets and tiny targets both
            # forgive their respective absolute differences proportionally).
            scale_dev = abs(total_len_t / max(total_len_db, 1e-9) - 1.0)
            if scale_dev > max_scale_dev:
                continue   # hard filter: too much scaling needed
            angle_diff = 0.0
            if n_segs_t >= 2:
                ang_db = np.abs(self.task_corner_angles[idx])
                ang_t = np.abs(corner_angles_t)
                angle_diff = float(np.sum(np.abs(ang_db - ang_t)))
                if angle_diff > max_angle_dev_rad:
                    continue   # hard filter: corner shape too different
            scores[j] = (
                position_weight * float(pos_dist[j])
                + len_weight * float(scale_dev)
                + corner_weight * angle_diff
            )

        order = np.argsort(scores)[: int(k)]
        matches: list[RetrievalMatch] = []
        for rank in range(int(k)):
            if rank >= order.shape[0]:
                break
            j = int(order[rank])
            idx = int(candidate_idx[j])
            poly_db = self.reconstruct_polyline_world(idx)
            R, scale, t, adj = self._compute_alignment(poly_db, target_polyline)
            matches.append(RetrievalMatch(
                task_idx=idx,
                distance=float(scores[j]),
                seg_lens=self.task_seg_lens[idx],
                corner_angles=self.task_corner_angles[idx],
                polyline_world=poly_db,
                transform_R=R, transform_scale=scale, transform_t=t,
                adjusted_polyline_world=adj,
            ))
        return matches

    # ---- Similarity transform: align retrieved polyline → target ----

    def _compute_alignment(
        self, src_poly: np.ndarray, tgt_poly: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Solve for (R, scale, t) such that scale·R·src + t best maps src to tgt.

        We anchor the START vertex (translation that puts src[0] at tgt[0]) and align
        the FIRST-segment direction to tgt's first-segment direction (rotation around
        any axis perpendicular to the segments). Scale = total_target_len / total_src_len.
        Subsequent segments inherit the same R+scale, so corner angles are preserved.
        """
        src = np.asarray(src_poly, dtype=np.float64); tgt = np.asarray(tgt_poly, dtype=np.float64)
        # Total length-based scale.
        src_total = float(np.sum(np.linalg.norm(np.diff(src, axis=0), axis=1)))
        tgt_total = float(np.sum(np.linalg.norm(np.diff(tgt, axis=0), axis=1)))
        scale = tgt_total / max(src_total, 1e-9)

        # First-segment direction unit vectors.
        src_d = src[1] - src[0]; src_d /= max(float(np.linalg.norm(src_d)), 1e-12)
        tgt_d = tgt[1] - tgt[0]; tgt_d /= max(float(np.linalg.norm(tgt_d)), 1e-12)

        # Rotation aligning src_d → tgt_d (Rodrigues).
        v = np.cross(src_d, tgt_d)
        sin_a = float(np.linalg.norm(v))
        cos_a = float(np.dot(src_d, tgt_d))
        if sin_a < 1e-9:
            R = np.eye(3) if cos_a > 0 else -np.eye(3)
        else:
            v_unit = v / sin_a
            K = np.array([
                [0,         -v_unit[2],  v_unit[1]],
                [v_unit[2],          0, -v_unit[0]],
                [-v_unit[1], v_unit[0],          0],
            ], dtype=np.float64)
            R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
        # Translate src origin to target origin AFTER scaling+rotating.
        t = tgt[0] - scale * (R @ src[0])
        adjusted = (scale * (R @ src.T)).T + t
        return R, float(scale), t, adjusted.astype(np.float32)


def _plot_match(target_poly: np.ndarray, m: RetrievalMatch, title: str, out_path: Path) -> None:
    """Top-down 2D plot: target vs retrieved (training) vs adjusted, on desk-plane XY."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    # Target (reference): gray dashed
    ax.plot(target_poly[:, 0], target_poly[:, 1], "--",
            color="0.4", lw=2.5, label="target (canonical char shape)")
    ax.scatter(target_poly[:, 0], target_poly[:, 1], color="0.4", s=40, zorder=3)
    # Retrieved (training, before similarity transform): blue
    src = m.polyline_world
    ax.plot(src[:, 0], src[:, 1], "-",
            color="#0044aa", lw=2.0, alpha=0.7, label=f"retrieved task #{m.task_idx} (raw)")
    ax.scatter(src[:, 0], src[:, 1], color="#0044aa", s=30, zorder=3, alpha=0.7)
    # Adjusted (after similarity transform): red
    adj = m.adjusted_polyline_world
    ax.plot(adj[:, 0], adj[:, 1], "-",
            color="#e24a33", lw=2.5, label=f"retrieved + similarity transform")
    ax.scatter(adj[:, 0], adj[:, 1], color="#e24a33", s=40, zorder=3)
    # Mark stroke starts
    ax.scatter([target_poly[0, 0]], [target_poly[0, 1]], marker="o", s=200, facecolor="none",
               edgecolor="0.4", lw=1.5, zorder=2, label="stroke start (target)")

    ax.set_xlabel("desk X (m)"); ax.set_ylabel("desk Y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_character(
    char: str,
    size_m: float,
    target_strokes: list[np.ndarray],
    matches_per_stroke: list[RetrievalMatch],
    out_path: Path,
) -> None:
    """One picture for the WHOLE character: all strokes overlaid, target / retrieved / adjusted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    # Per-stroke palette so the strokes are distinguishable.
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (tgt, m) in enumerate(zip(target_strokes, matches_per_stroke)):
        c = palette[i % len(palette)]
        # Target — gray dashed (background reference)
        ax.plot(tgt[:, 0], tgt[:, 1], "--", color="0.45", lw=2.0, alpha=0.85,
                label="canonical target" if i == 0 else None)
        ax.scatter(tgt[:, 0], tgt[:, 1], color="0.45", s=30, zorder=3, alpha=0.85)
        ax.annotate(f"s{i+1}", (float(tgt[0, 0]), float(tgt[0, 1])),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=10, color="0.3", alpha=0.85)
        # Retrieved raw — light dotted in stroke color (so you see how it sat in training)
        src = m.polyline_world
        ax.plot(src[:, 0], src[:, 1], ":", color=c, lw=1.4, alpha=0.45,
                label="retrieved raw (training shape)" if i == 0 else None)
        # Adjusted — solid in stroke color (this is what we'll feed to DiT)
        adj = m.adjusted_polyline_world
        ax.plot(adj[:, 0], adj[:, 1], "-", color=c, lw=2.6,
                label="adjusted (used for DiT input)" if i == 0 else None)
        ax.scatter(adj[:, 0], adj[:, 1], color=c, s=40, zorder=4)

    title = f"{char} @ size={size_m*100:.0f}cm — top-1 retrieval per stroke"
    info = "  ".join([f"s{i+1}:#{m.task_idx} d={m.distance:.3f} sc={m.transform_scale:.2f}"
                       for i, m in enumerate(matches_per_stroke)])
    ax.set_xlabel("desk X (m)"); ax.set_ylabel("desk Y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{title}\n{info}", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    # Demo: retrieve a match for 中's longest stroke (中央竖, 30cm @ size=0.15) and
    # report the alignment.
    import argparse
    from Yuan.fr3_dit.calligraphy.character_def import place_character

    p = argparse.ArgumentParser()
    p.add_argument("--char", type=str, default="中")
    p.add_argument("--size", type=float, default=0.15)
    p.add_argument("--stroke", type=int, default=None,
                   help="If set, query only this 1-based stroke index. Otherwise, retrieve "
                        "for ALL strokes and (with --save-plot) produce a whole-character figure.")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--save-plot", action="store_true", default=False,
                   help="Save a 2-D top-down plot. With --stroke = per-stroke comparison; "
                        "without --stroke = single whole-character figure (top-1 per stroke).")
    args = p.parse_args()

    desk_center = np.array([0.5, 0.0, -0.05], dtype=np.float32)
    desk_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    strokes = place_character(args.char, desk_center, desk_normal,
                               size_m=args.size, theta_rad=0.0)
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / "outputs"
    rt = StrokeRetrieval()

    if args.stroke is not None:
        # Single-stroke mode: print top-K matches, optionally save per-stroke detail plots.
        target_poly = strokes[args.stroke - 1]
        target_lens_cm = [round(np.linalg.norm(target_poly[i+1] - target_poly[i]) * 100, 1)
                          for i in range(target_poly.shape[0] - 1)]
        print(f"[target] {args.char} stroke {args.stroke}: {target_poly.shape[0]} verts, "
              f"seg_lens(cm)={target_lens_cm}")
        matches = rt.query(target_poly, k=args.k)
        print(f"\nTop-{args.k} matches:")
        for r, m in enumerate(matches):
            adj_lens_cm = [round(np.linalg.norm(m.adjusted_polyline_world[i+1] - m.adjusted_polyline_world[i]) * 100, 1)
                           for i in range(m.adjusted_polyline_world.shape[0] - 1)]
            print(f"  #{r+1}: task_idx={m.task_idx:>6}  dist={m.distance:.4f}  "
                  f"db_seg_lens(cm)={[round(L*100,1) for L in m.seg_lens.tolist()]}  "
                  f"adj_seg_lens(cm)={adj_lens_cm}  scale={m.transform_scale:.3f}")
        if args.save_plot and matches:
            for r, m in enumerate(matches[:min(3, len(matches))]):
                out_path = out_dir / f"retrieve_{args.char}_stroke{args.stroke}_rank{r+1}.svg"
                title = (f"{args.char} stroke {args.stroke} (size={args.size*100:.0f}cm) — "
                         f"rank {r+1}: task_idx={m.task_idx}, dist={m.distance:.4f}, "
                         f"scale={m.transform_scale:.3f}")
                _plot_match(target_poly, m, title, out_path)
                print(f"  saved {out_path}")
    else:
        # Whole-character mode: top-1 retrieval per stroke.
        print(f"[target] {args.char} @ size={args.size*100:.0f}cm — {len(strokes)} strokes")
        matches_per_stroke: list[RetrievalMatch] = []
        for i, target_poly in enumerate(strokes):
            ms = rt.query(target_poly, k=1)
            if not ms:
                raise RuntimeError(f"no retrieval candidate for stroke {i+1}")
            m = ms[0]
            adj_lens_cm = [round(np.linalg.norm(m.adjusted_polyline_world[j+1] - m.adjusted_polyline_world[j]) * 100, 1)
                           for j in range(m.adjusted_polyline_world.shape[0] - 1)]
            tgt_lens_cm = [round(np.linalg.norm(target_poly[j+1] - target_poly[j]) * 100, 1)
                           for j in range(target_poly.shape[0] - 1)]
            print(f"  stroke {i+1}/{len(strokes)}  "
                  f"target_lens(cm)={tgt_lens_cm}  "
                  f"matched task_idx={m.task_idx:>6}  dist={m.distance:.4f}  "
                  f"adj_lens(cm)={adj_lens_cm}  scale={m.transform_scale:.3f}")
            matches_per_stroke.append(m)
        if args.save_plot:
            out_path = out_dir / f"retrieve_{args.char}_size{int(args.size*100):02d}cm_whole.svg"
            _plot_character(args.char, args.size, list(strokes), matches_per_stroke, out_path)
            print(f"\nsaved {out_path}")
