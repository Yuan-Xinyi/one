"""DiT-as-prior + IK refine + tracker = feasibility oracle for individual strokes.

Loads everything once (DiT ckpt, FR3 GPU/CPU robots, plane tracker, IK solver) and
exposes a single ``evaluate_stroke`` method that returns per-candidate rollout results.

Usage:
    oracle = FeasibilityOracle(ckpt="dit_q0_v5_ckpts/final.pt")
    result = oracle.evaluate_stroke(tokenized, desk_center, desk_normal)
    if result.feasible: ...
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3, PenFrankaResearch3GPU
from Yuan.fr3_dit.data_generation.generate_fr3_plane_dataset import (
    PlaneConstrainedTracker,
    TrackerConfig,
    termination_label,
)
from Yuan.fr3_dit.training.ik_refine import refine_q0_seed
from Yuan.fr3_dit.training.task_cond_dit_q0 import (
    DDPMCosineSchedule,
    DiTq0Config,
    FR3_JOINT_LIMITS,
    TaskCondDiTq0,
    ddim_sample_q0,
    denormalize_q,
    normalize_q,
    q_sample,
    v_target_from,
)
from Yuan.fr3_dit.visualization.visualize_q0_rollout import rollout_from_q0

from Yuan.fr3_dit.calligraphy.polyline_to_tokens import TokenizedStroke


DEFAULT_CKPT = (Path(__file__).resolve().parents[1]
                / "experiments" / "outputs" / "dit_q0_v5_ckpts" / "final.pt")


@dataclass
class CandidateResult:
    rank: int
    dit_score: float                                    # higher = more in-distribution
    rmse_to_seed: float
    tcp_err_seed_cm: float
    tcp_err_refined_cm: float
    ik_ok: bool
    in_cone: Optional[bool]
    completed: bool
    completion_pct: float
    seg_completed: int
    n_segments: int
    distance_traveled_m: float
    target_total_m: float
    top_failure_label: str
    full_q_trajectory: np.ndarray = field(repr=False)   # (T_total, 7)
    q0_refined: np.ndarray = field(repr=False)          # (7,)


@dataclass
class StrokeResult:
    feasible: bool                          # ≥1 candidate completed all segments
    n_candidates: int
    n_success: int
    best_completion_pct: float
    best_idx: int                            # candidate index by completion (then RMSE)
    candidates: List[CandidateResult]
    notes: str = ""

    def best(self) -> CandidateResult:
        return self.candidates[self.best_idx]


class FeasibilityOracle:
    def __init__(
        self,
        ckpt: Path = DEFAULT_CKPT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_tokens: int = 11,
        cfg_w: float = 3.0,
        sampler_steps: int = 50,
        eta: float = 0.0,
        clip_x0: float = 1.2,
        n_candidates: int = 8,
        top_k_rollout: int | None = None,
        score_t_steps: tuple[int, ...] = (10, 50, 100, 200),
        score_n_repeat: int = 4,
        snap_q7: bool = True,
        # Tracker config (matches eval_tracker / visualize_q0_rollout defaults at the
        # time of writing — pass kwargs to override).
        theta_max_deg: float = 30.0,
        angle_null_gain: float = 1.0,
        angle_attract_gain: float = 2.0,
        max_steps_buffer: int = 30,
        verbose: bool = True,
    ):
        self.device = torch.device(device)
        self.cfg_w = float(cfg_w)
        self.sampler_steps = int(sampler_steps)
        self.eta = float(eta)
        self.clip_x0 = float(clip_x0)
        self.n_candidates = int(n_candidates)
        self.top_k_rollout = int(top_k_rollout) if top_k_rollout is not None else int(n_candidates)
        self.top_k_rollout = max(1, min(self.top_k_rollout, int(n_candidates)))
        self.score_t_steps = tuple(int(t) for t in score_t_steps)
        self.score_n_repeat = int(score_n_repeat)
        self.snap_q7 = bool(snap_q7)
        self.max_tokens = int(max_tokens)
        self.max_steps_buffer = int(max_steps_buffer)
        self.verbose = bool(verbose)

        # ---- DiT model ----
        if self.verbose: print(f"[oracle] loading DiT ckpt: {ckpt}")
        ckpt_dict = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.cfg = DiTq0Config(**ckpt_dict["cfg"])
        self.model = TaskCondDiTq0(self.cfg).to(self.device).eval()
        if "ema" in ckpt_dict and ckpt_dict["ema"] is not None:
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in ckpt_dict["ema"]:
                        p.copy_(ckpt_dict["ema"][n].to(self.device))
        else:
            self.model.load_state_dict(ckpt_dict["model"])
        self.schedule = DDPMCosineSchedule(T=int(ckpt_dict["T"])).to(self.device)
        if self.verbose:
            print(f"[oracle] DiT loaded: step={ckpt_dict.get('step', '?')} "
                  f"d_model={self.cfg.d_model} max_tokens={self.cfg.max_tokens}")

        # ---- IK robot (CPU, used for refine) ----
        # Use the canonical name "pen" so all PenFrankaResearch3 instances across the
        # project share a single SELIK CVT cache file (~60s build avoided per Python run).
        self.ik_robot = PenFrankaResearch3(name="pen", enable_cc=False)

        # ---- Tracker (GPU rollout) ----
        self.fr3_gpu = PenFrankaResearch3GPU(self.device)
        helpers = make_pen_collision_helpers(self.fr3_gpu.robot)
        self.tracker_config = TrackerConfig(
            theta_max_deg=float(theta_max_deg),
            angle_null_gain=float(angle_null_gain),
            angle_attract_gain=float(angle_attract_gain),
        )
        self.tracker = PlaneConstrainedTracker(
            robot=self.fr3_gpu.robot,
            self_collision_fn=helpers.self_collision_fn,
            sphere_positions_fn=helpers.sphere_positions_fn,
            sphere_radii=helpers.sphere_radii,
            sphere_link_indices=helpers.sphere_link_indices,
            config=self.tracker_config,
        )
        if self.verbose:
            print(f"[oracle] tracker: theta_max={theta_max_deg}° "
                  f"angle_null={angle_null_gain}  attract={angle_attract_gain}")

    # ---- DiT inference ----

    def _sample_q0_candidates(
        self,
        ts: TokenizedStroke,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (q0_raw_candidates (N, 7), gt_seed=None — no GT here)."""
        if ts.n_tokens > self.max_tokens:
            raise ValueError(f"stroke has {ts.n_tokens} tokens > max_tokens={self.max_tokens}")
        # Pad tokens to max_tokens.
        tokens_pad = np.zeros((self.max_tokens, ts.tokens.shape[1]), dtype=np.float32)
        tokens_pad[:ts.n_tokens] = ts.tokens
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_mask[:ts.n_tokens] = 1.0

        N = self.n_candidates
        tokens_t = torch.from_numpy(tokens_pad).unsqueeze(0).expand(N, -1, -1).contiguous().to(self.device)
        token_mask_t = torch.from_numpy(token_mask).unsqueeze(0).expand(N, -1).contiguous().to(self.device)
        q0_norm = ddim_sample_q0(
            self.model, self.schedule, tokens_t, token_mask_t,
            shape=(N, 7), device=self.device,
            num_steps=self.sampler_steps, eta=self.eta,
            cfg_w=self.cfg_w, clip_x0=self.clip_x0,
        )
        q0_raw = denormalize_q(q0_norm).cpu().numpy().astype(np.float32)
        if self.snap_q7:
            q0_raw[:, 6] = 0.0
        return q0_raw

    def _score_q0_candidates(
        self,
        q0_raw: np.ndarray,
        ts: TokenizedStroke,
    ) -> np.ndarray:
        """Score each candidate via DiT denoising loss at small t (option B).

        Lower v-pred error at small t = candidate sits closer to the model's training
        distribution = a more "confident" / in-distribution q0. Returned as positive
        scores (negated loss): higher = better.
        """
        # Pad tokens (same logic as _sample_q0_candidates).
        tokens_pad = np.zeros((self.max_tokens, ts.tokens.shape[1]), dtype=np.float32)
        tokens_pad[:ts.n_tokens] = ts.tokens
        token_mask = np.zeros((self.max_tokens,), dtype=np.float32)
        token_mask[:ts.n_tokens] = 1.0

        N = q0_raw.shape[0]
        tokens_t = torch.from_numpy(tokens_pad).unsqueeze(0).expand(N, -1, -1).contiguous().to(self.device)
        token_mask_t = torch.from_numpy(token_mask).unsqueeze(0).expand(N, -1).contiguous().to(self.device)

        # Normalize q0 to model's joint-limit space.
        q0_norm = torch.from_numpy(normalize_q(q0_raw).astype(np.float32)).to(self.device)

        losses = torch.zeros(N, device=self.device)
        with torch.no_grad():
            for t_step in self.score_t_steps:
                for _ in range(self.score_n_repeat):
                    t_idx = torch.full((N,), int(t_step), dtype=torch.long, device=self.device)
                    xt, eps = q_sample(q0_norm, t_idx, self.schedule)
                    v_target = v_target_from(q0_norm, eps, t_idx, self.schedule)
                    v_pred = self.model(xt, t_idx, tokens_t, token_mask_t)
                    losses += ((v_pred - v_target) ** 2).mean(dim=-1)
        denom = max(len(self.score_t_steps) * self.score_n_repeat, 1)
        losses = losses / denom
        return (-losses).cpu().numpy().astype(np.float32)   # higher score = better

    # ---- Per-stroke evaluation ----

    def evaluate_stroke(
        self,
        ts: TokenizedStroke,
        desk_center: np.ndarray,
        desk_normal: np.ndarray,
    ) -> StrokeResult:
        if ts.n_segments == 0:
            return StrokeResult(False, 0, 0, 0.0, -1, [], notes="empty stroke")
        q0_raw = self._sample_q0_candidates(ts)

        # DiT self-scoring (option B): rank candidates by denoising loss at small t.
        # Higher score = more in-distribution. We always score, but only roll out the
        # top-K (default K = n_candidates → no filtering, identical to before).
        scores = self._score_q0_candidates(q0_raw, ts)
        order = np.argsort(-scores)                                # descending by score
        rollout_indices = list(map(int, order[: self.top_k_rollout]))

        # Build segments arg expected by rollout_from_q0: [(dir_world, length_m), ...]
        segments = list(zip(ts.seg_dirs_world, ts.seg_lens))
        target_total = float(sum(ts.seg_lens))
        K = len(segments)

        candidates: List[CandidateResult] = []
        for rank_pos, i in enumerate(rollout_indices):
            q_seed = q0_raw[i]
            # IK refine: TCP target = stroke start (= ts.local_origin), orientation = seed's own.
            q_ref, ik_ok, info = refine_q0_seed(
                self.ik_robot, q_seed, ts.local_origin,
                target_rotmat=None, desk_normal=desk_normal,
                theta_max_deg=float(self.tracker_config.theta_max_deg),
            )
            # Roll out from refined q.
            full_q, seg_results = rollout_from_q0(
                self.tracker, self.tracker_config,
                q_ref, segments,
                np.asarray(desk_center, dtype=np.float32),
                np.asarray(desk_normal, dtype=np.float32),
                device=self.device, max_steps_buffer=self.max_steps_buffer,
            )
            n_seg_done = sum(1 for r in seg_results if r["completed"])
            traveled = sum(r.get("traveled_m", 0.0) for r in seg_results)
            completed_full = (n_seg_done == K)
            terms = [r["term_label"] for r in seg_results if not r["completed"]]
            top_fail = terms[0] if terms else ("all_segments_done" if completed_full else "unknown")
            candidates.append(CandidateResult(
                rank=int(i),
                dit_score=float(scores[i]),
                rmse_to_seed=0.0,  # not computed here; tracked elsewhere if needed
                tcp_err_seed_cm=info["tcp_err_seed_m"] * 100,
                tcp_err_refined_cm=info["tcp_err_refined_m"] * 100,
                ik_ok=bool(ik_ok),
                in_cone=info.get("seed_in_cone"),
                completed=completed_full,
                completion_pct=float(min(traveled / max(target_total, 1e-9), 1.0)),
                seg_completed=n_seg_done,
                n_segments=K,
                distance_traveled_m=traveled,
                target_total_m=target_total,
                top_failure_label=top_fail,
                full_q_trajectory=full_q,
                q0_refined=q_ref.astype(np.float32),
            ))

        # Pick best: prefer (IK-refined to exact target TCP) > (rolled out fully) > (more completion).
        # Without ik_ok in the key, a candidate with TCP 5cm off but slightly higher partial
        # completion would beat a candidate with TCP 0mm and 0% completion — and the visualizer
        # would render its q-trajectory starting from a wildly wrong TCP, looking like "drift".
        candidates_sorted_idx = sorted(
            range(len(candidates)),
            key=lambda j: (candidates[j].ik_ok, candidates[j].completed,
                           candidates[j].completion_pct),
            reverse=True,
        )
        best_idx = candidates_sorted_idx[0]
        n_success = sum(1 for c in candidates if c.completed)

        return StrokeResult(
            feasible=n_success > 0,
            n_candidates=len(candidates),
            n_success=n_success,
            best_completion_pct=float(candidates[best_idx].completion_pct),
            best_idx=best_idx,
            candidates=candidates,
        )


if __name__ == "__main__":
    # Tiny smoke test: tokenize 中's first stroke, query oracle, print result.
    from Yuan.fr3_dit.calligraphy.character_def import place_character
    from Yuan.fr3_dit.calligraphy.polyline_to_tokens import tokenize_stroke

    desk_center = np.array([0.5, 0.0, -0.05], dtype=np.float32)
    desk_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    strokes = place_character("中", desk_center, desk_normal, size_m=0.08)

    oracle = FeasibilityOracle()
    print(f"\nEvaluating 中 (4 strokes @ 8cm)...")
    for i, poly in enumerate(strokes):
        ts = tokenize_stroke(poly, desk_normal)
        r = oracle.evaluate_stroke(ts, desk_center, desk_normal)
        marker = "✓" if r.feasible else "✗"
        print(f"  stroke {i+1}/{len(strokes)} {marker}  "
              f"feasible={r.feasible}  "
              f"n_success={r.n_success}/{r.n_candidates}  "
              f"best_completion={r.best_completion_pct*100:.1f}%  "
              f"top_fail={r.best().top_failure_label}")
