"""LineDistribution — MC reachability sampler + feasibility filter.

A "line" is an infinite ray (p_0, u_hat, n_target). We pre-build a fixed pool
of line specs at init (q0, line_dir, n_target deterministic-per-index), then
`sample(n)` indexes into the pool. This lets us pre-filter the pool: lines
where even the classical nullspace controller cannot survive `threshold_m` of
EE travel are dropped, so the RL agent isn't asked to optimize on
intrinsically-infeasible tasks (which only adds noise to its gradient).

Also includes ScriptedLineDistribution for replaying a fixed spec list at eval.

Pools are cacheable to disk via `save()` / `load_or_build()` — building +
filtering a 100K pool takes ~2 min, but the result is deterministic given
(seed, n_pool, threshold_m, env v/dt/tcp_offset, n_target_noise_deg). The
training scripts auto-cache under `runs/_pool_cache/`.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path

import torch

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision


@contextlib.contextmanager
def _stock_tool(kin):
    """Run kinematics with the stock (single) tool even when an env has
    aliased per-env tool buffers into it."""
    sp = getattr(kin, '_stock_flange_p', None)
    if sp is None:
        yield
        return
    p, R = kin.flange_p, kin.flange_R
    kin.flange_p, kin.flange_R = sp, kin._stock_flange_R
    try:
        yield
    finally:
        kin.flange_p, kin.flange_R = p, R


class LineDistribution:
    def __init__(self,
                 kin: BatchedFR3Kinematics,
                 collision: FR3SphereCollision,
                 n_pool: int = 100_000,
                 n_target_noise_deg: float = 5.0,
                 seed: int | None = None,
                 batch_size: int = 8192,
                 swing_max_deg: float = 0.0,
                 wavelen_range: tuple[float, float] = (0.4, 1.2),
                 min_radius_m: float = 0.15,
                 cone_range: tuple[float, float] | None = None,
                 cone_log: bool = False,
                 curve_cfg: dict | None = None,
                 tool_cfg: dict | None = None):
        # Require explicit seed: cache_key() hashes the seed value, so a None
        # seed yields a fixed cache_path but a non-deterministic pool — the
        # first build wins and later "different" calls silently get the cached
        # contents. Force callers to pass an int.
        if seed is None:
            raise ValueError(
                "LineDistribution requires an explicit integer seed for "
                "reproducibility (cache key depends on it)")

        self.kin = kin
        self.device = kin.device
        self.dtype = kin.dtype
        self.n_target_noise = float(n_target_noise_deg) * math.pi / 180.0
        self.swing_max_deg = float(swing_max_deg)
        self.wavelen_range = tuple(wavelen_range)
        self.min_radius_m = float(min_radius_m)
        self.cone_range = (None if cone_range is None
                           else (float(cone_range[0]), float(cone_range[1])))
        self.cone_log = bool(cone_log)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        q_pool, z_pool = [], []
        n_remaining = n_pool
        while n_remaining > 0:
            b = min(batch_size, n_remaining)
            q = kin.rand_conf_batch(b, generator=gen)
            with _stock_tool(kin):
                _, R, _, _ = kin.tcp_fk_jac(q)
            z = R[:, :, 2]
            link_tfs = kin.link_transforms(q)
            ok = ~collision.is_collided(link_tfs)
            q_pool.append(q[ok])
            z_pool.append(z[ok])
            n_remaining -= int(ok.sum().item())
        self.q_pool = torch.cat(q_pool, dim=0)[:n_pool]
        self.z_pool = torch.cat(z_pool, dim=0)[:n_pool]
        n_pool = self.q_pool.shape[0]

        # Per-task cone half-angle [deg] for mixed-tolerance training. Drawn
        # before the n_target noise so the noise can be capped at half the
        # task's own cone: every start is admissible under its own tolerance.
        # Absent (the historical pools) nothing below changes.
        if self.cone_range is not None:
            lo, hi = self.cone_range
            u = torch.rand((n_pool,), device=self.device, dtype=self.dtype,
                           generator=gen)
            if self.cone_log:
                cone = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
            else:
                cone = lo + u * (hi - lo)
            self.cone_pool = cone
            _noise_amp = torch.minimum(
                torch.full_like(cone, self.n_target_noise),
                0.5 * cone * (math.pi / 180.0))
        else:
            self.cone_pool = None
            _noise_amp = self.n_target_noise

        # Per-task random tools: length L along an axis tilted by beta at
        # azimuth phi from the flange z. The tool axis at q0 replaces the
        # stock z_pool, so n_target is drawn around the actual tool axis.
        self.tool_cfg = dict(tool_cfg) if tool_cfg is not None else None
        if self.tool_cfg is not None:
            from Yuan.IJRR.kinematics.batched_chain_kin import tool_rotmats
            tc = dict(len_range=(0.05, 0.40), tilt_max_deg=45.0)
            tc.update(self.tool_cfg)
            u = lambda: torch.rand((n_pool,), device=self.device, dtype=self.dtype, generator=gen)
            L = tc['len_range'][0] + u() * (tc['len_range'][1] - tc['len_range'][0])
            beta = u() * math.radians(tc['tilt_max_deg'])
            phi = u() * 2.0 * math.pi
            R = tool_rotmats(beta, phi)
            lever = (R @ torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
                     .view(1, 3, 1)).squeeze(-1) * L.unsqueeze(-1)
            with _stock_tool(kin):
                fixed = kin.flange_p.reshape(-1, 3)[0] - (kin.flange_R.reshape(-1, 3, 3)[0]
                                                          @ torch.tensor([0.0, 0.0, kin.tcp_offset],
                                                                         device=self.device, dtype=self.dtype))
            self.tool_p_pool = fixed.unsqueeze(0) + lever
            self.tool_R_pool = R
            self.tool_lever_pool = lever
            self.tool_a0_pool = R[:, :, 2].clone()
            self.tool_len_pool, self.tool_tilt_pool = L, beta
            # tool axis at q0: R_last(q0) a0 (R_last = flange rotation)
            z_new = torch.empty_like(self.z_pool)
            with _stock_tool(kin):
                for lo in range(0, n_pool, 16384):
                    _, R_last, _, _ = kin.fk_jac(self.q_pool[lo:lo + 16384])
                    z_new[lo:lo + 16384] = (R_last @ self.tool_a0_pool[lo:lo + 16384].unsqueeze(-1)).squeeze(-1)
            self.z_pool = z_new
        else:
            self.tool_p_pool = None

        # Pre-generate full line spec for every pool entry (deterministic per index).
        # n_target = z_tool + small angular noise about a random axis ⊥ z.
        if self.n_target_noise > 0:
            axis = torch.randn((n_pool, 3), device=self.device, dtype=self.dtype, generator=gen)
            axis = axis - (axis * self.z_pool).sum(-1, keepdim=True) * self.z_pool
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            angle = (torch.rand((n_pool,), device=self.device, dtype=self.dtype, generator=gen)
                     * 2 - 1) * _noise_amp
            self.n_target_pool = (self.z_pool * torch.cos(angle).unsqueeze(-1)
                                  + axis * torch.sin(angle).unsqueeze(-1))
        else:
            self.n_target_pool = self.z_pool.clone()
        self.n_target_pool = self.n_target_pool / self.n_target_pool.norm(
            dim=-1, keepdim=True).clamp(min=1e-8)

        # line_dir = random unit vector ⊥ n_target
        r = torch.randn((n_pool, 3), device=self.device, dtype=self.dtype, generator=gen)
        r = r - (r * self.n_target_pool).sum(-1, keepdim=True) * self.n_target_pool
        self.line_dir_pool = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # ---- serpentine parameters -------------------------------------
        # What makes a wave hard for the arm is how far the direction of travel
        # swings, so the swing angle is sampled directly and the amplitude is
        # derived; sampling amplitude and wavelength independently would leave
        # the difficulty distribution implicit. swing = 0 is the straight ray,
        # so the ray is a member of this family rather than a separate case.
        if self.swing_max_deg > 0.0:
            swing = torch.rand((n_pool,), device=self.device, dtype=self.dtype,
                               generator=gen) * math.radians(self.swing_max_deg)
            lam = (self.wavelen_range[0]
                   + torch.rand((n_pool,), device=self.device, dtype=self.dtype,
                                generator=gen)
                   * (self.wavelen_range[1] - self.wavelen_range[0]))
            amp = lam * torch.tan(swing) / (2.0 * math.pi)
            # Reject waves tighter than the arm can track at all: they die in
            # the first bend and carry no learning signal. Redraw the swing
            # rather than the wavelength so the wavelength stays uniform.
            kappa_max = amp * (2.0 * math.pi / lam) ** 2
            too_tight = kappa_max > (1.0 / self.min_radius_m)
            for _ in range(16):
                if not bool(too_tight.any()):
                    break
                s2 = torch.rand((n_pool,), device=self.device, dtype=self.dtype,
                                generator=gen) * math.radians(self.swing_max_deg)
                swing = torch.where(too_tight, s2, swing)
                amp = lam * torch.tan(swing) / (2.0 * math.pi)
                kappa_max = amp * (2.0 * math.pi / lam) ** 2
                too_tight = kappa_max > (1.0 / self.min_radius_m)
            amp = torch.where(too_tight, torch.zeros_like(amp), amp)
        else:
            amp = torch.zeros((n_pool,), device=self.device, dtype=self.dtype)
            lam = torch.full((n_pool,), 0.8, device=self.device, dtype=self.dtype)
        self.amp_pool, self.wavelen_pool = amp, lam

        # ---- table paths (generic curves) -------------------------------
        # One random curve per task, starting at the task's own tip position
        # with its tangent d and normal n; kept on the CPU (100k x 201 x 9
        # floats) and moved to the device per sampled batch.
        self.curve_cfg = dict(curve_cfg) if curve_cfg is not None else None
        if self.curve_cfg is not None:
            from Yuan.IJRR.env.path_table import build_tables
            p0 = torch.empty((n_pool, 3), device=self.device, dtype=self.dtype)
            for lo in range(0, n_pool, 16384):
                if self.tool_p_pool is not None:
                    p0[lo:lo + 16384] = kin.fk_jac(self.q_pool[lo:lo + 16384],
                                                   self.tool_p_pool[lo:lo + 16384])[0]
                else:
                    with _stock_tool(kin):
                        p0[lo:lo + 16384] = kin.tcp_fk_jac(self.q_pool[lo:lo + 16384])[0]
            pts, tan, nrm = build_tables(p0, self.line_dir_pool, self.n_target_pool,
                                         gen, self.curve_cfg)
            self.tab_pts = pts.cpu(); self.tab_tan = tan.cpu(); self.tab_nrm = nrm.cpu()
        else:
            self.tab_pts = self.tab_tan = self.tab_nrm = None

        self.valid_mask = torch.ones(n_pool, dtype=torch.bool, device=self.device)
        self.n_pool = n_pool
        self._gen = gen

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum().item())

    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        gen = generator if generator is not None else self._gen
        valid_idx = torch.nonzero(self.valid_mask, as_tuple=False).squeeze(-1)
        n_valid = valid_idx.shape[0]
        if n_valid == 0:
            raise RuntimeError("LineDistribution has no valid lines (filter removed all)")
        pick = torch.randint(0, n_valid, (n,), device=self.device, generator=gen)
        idx = valid_idx[pick]
        out = {
            "q0": self.q_pool[idx],
            "line_dir": self.line_dir_pool[idx],
            "n_target": self.n_target_pool[idx],
            "amp": self.amp_pool[idx],
            "wavelen": self.wavelen_pool[idx],
        }
        if self.cone_pool is not None:
            out["cone_deg"] = self.cone_pool[idx]
        if self.tool_p_pool is not None:
            out["tool_p"] = self.tool_p_pool[idx]
            out["tool_R"] = self.tool_R_pool[idx]
            out["tool_lever"] = self.tool_lever_pool[idx]
            out["tool_a0"] = self.tool_a0_pool[idx]
        if self.tab_pts is not None:
            ic = idx.cpu()
            out["path_pts"] = self.tab_pts[ic].to(self.device)
            out["path_tan"] = self.tab_tan[ic].to(self.device)
            out["path_nrm"] = self.tab_nrm[ic].to(self.device)
            out["p0"] = out["path_pts"][:, 0].clone()
        return out

    # ---- disk cache ------------------------------------------------------

    def save(self, path) -> None:
        """Serialize pool to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_pool": self.q_pool.cpu(),
            "line_dir_pool": self.line_dir_pool.cpu(),
            "n_target_pool": self.n_target_pool.cpu(),
            "valid_mask": self.valid_mask.cpu(),
            "amp_pool": self.amp_pool.cpu(),
            "wavelen_pool": self.wavelen_pool.cpu(),
            "n_target_noise": self.n_target_noise,
            "cone_pool": (None if self.cone_pool is None
                          else self.cone_pool.cpu()),
            "curve_cfg": self.curve_cfg,
            "tab_pts": self.tab_pts, "tab_tan": self.tab_tan, "tab_nrm": self.tab_nrm,
            "tool_cfg": self.tool_cfg,
            "tool_p_pool": (None if self.tool_p_pool is None else self.tool_p_pool.cpu()),
            "tool_R_pool": (None if self.tool_p_pool is None else self.tool_R_pool.cpu()),
            "tool_lever_pool": (None if self.tool_p_pool is None else self.tool_lever_pool.cpu()),
            "tool_a0_pool": (None if self.tool_p_pool is None else self.tool_a0_pool.cpu()),
            "tool_len_pool": (None if self.tool_p_pool is None else self.tool_len_pool.cpu()),
            "tool_tilt_pool": (None if self.tool_p_pool is None else self.tool_tilt_pool.cpu()),
        }, path)

    @classmethod
    def load(cls, path, kin, collision) -> "LineDistribution":
        """Load pool from disk and rebuild an instance bound to (kin, collision)."""
        data = torch.load(path, map_location=kin.device, weights_only=False)
        obj = cls.__new__(cls)
        obj.kin = kin
        obj.collision = collision
        obj.device = kin.device
        obj.dtype = kin.dtype
        obj.n_target_noise = float(data["n_target_noise"])
        obj.q_pool = data["q_pool"].to(kin.device, dtype=kin.dtype)
        obj.line_dir_pool = data["line_dir_pool"].to(kin.device, dtype=kin.dtype)
        obj.n_target_pool = data["n_target_pool"].to(kin.device, dtype=kin.dtype)
        obj.valid_mask = data["valid_mask"].to(kin.device)
        n = obj.q_pool.shape[0]
        obj.amp_pool = (data["amp_pool"].to(kin.device) if "amp_pool" in data
                        else torch.zeros(n, device=kin.device, dtype=kin.dtype))
        obj.wavelen_pool = (data["wavelen_pool"].to(kin.device)
                            if "wavelen_pool" in data
                            else torch.full((n,), 0.8, device=kin.device,
                                            dtype=kin.dtype))
        obj.n_pool = obj.q_pool.shape[0]
        _cp = data.get("cone_pool", None)
        obj.cone_pool = (None if _cp is None
                         else _cp.to(kin.device, dtype=kin.dtype))
        obj.curve_cfg = data.get("curve_cfg", None)
        obj.tab_pts = data.get("tab_pts", None)
        obj.tab_tan = data.get("tab_tan", None)
        obj.tab_nrm = data.get("tab_nrm", None)
        obj.tool_cfg = data.get("tool_cfg", None)
        _tp = data.get("tool_p_pool", None)
        if _tp is None:
            obj.tool_p_pool = None
        else:
            _m = lambda k: data[k].to(kin.device, dtype=kin.dtype)
            obj.tool_p_pool = _m("tool_p_pool"); obj.tool_R_pool = _m("tool_R_pool")
            obj.tool_lever_pool = _m("tool_lever_pool"); obj.tool_a0_pool = _m("tool_a0_pool")
            obj.tool_len_pool = _m("tool_len_pool"); obj.tool_tilt_pool = _m("tool_tilt_pool")
        obj._gen = torch.Generator(device=kin.device)
        return obj

    @staticmethod
    def cache_key(seed, n_pool, n_target_noise_deg, env_cfg,
                  feasibility_threshold_m=None, cone_range=None,
                  cone_log=False, curve_cfg=None, tool_cfg=None) -> str:
        """Deterministic short key for cache filename. Include a_max since
        the feasibility filter's classical controller is clamped by it."""
        sig = (f"robot={getattr(env_cfg, 'robot', 'fr3')}|"
               f"seed={seed}|n={n_pool}|noise={n_target_noise_deg}|"
               f"v={env_cfg.v}|dt={env_cfg.dt}|tcp={env_cfg.tcp_offset}|"
               f"amax={env_cfg.a_max}|thr={feasibility_threshold_m}|"
               f"swing={getattr(env_cfg, '_swing_max_deg', 0.0)}|"
               f"lam={getattr(env_cfg, '_wavelen_range', (0.4, 1.2))}")
        if cone_range is not None:
            # appended only for mixed-cone pools so every historical key
            # (and cached pool) stays exactly what it was
            sig += (f"|cone={tuple(float(c) for c in cone_range)}"
                    f"|clog={bool(cone_log)}")
        if curve_cfg is not None:
            sig += "|curves=" + repr(sorted((str(k), v) for k, v in dict(curve_cfg).items()))
        if tool_cfg is not None:
            sig += "|tools=" + repr(sorted((str(k), v) for k, v in dict(tool_cfg).items()))
        return hashlib.md5(sig.encode()).hexdigest()[:10]

    @classmethod
    def load_or_build(cls, kin, collision, *,
                      n_pool, n_target_noise_deg, seed, env_cfg,
                      feasibility_threshold_m=None,
                      cache_dir="Yuan/IJRR/runs/_pool_cache",
                      swing_max_deg=0.0, wavelen_range=(0.4, 1.2),
                      min_radius_m=0.15, cone_range=None, cone_log=False,
                      curve_cfg=None, tool_cfg=None, verbose=True) -> "LineDistribution":
        """Try to load pool from cache; otherwise build (+ filter) and save.

        `feasibility_threshold_m=None` skips the filter (raw pool).
        """
        cache_dir = Path(cache_dir)
        env_cfg._swing_max_deg = swing_max_deg
        env_cfg._wavelen_range = tuple(wavelen_range)
        key = cls.cache_key(seed, n_pool, n_target_noise_deg,
                            env_cfg, feasibility_threshold_m,
                            cone_range=cone_range, cone_log=cone_log,
                            curve_cfg=curve_cfg, tool_cfg=tool_cfg)
        cache_path = cache_dir / f"pool_{key}.pt"
        if cache_path.exists():
            if verbose:
                print(f"[LineDist] loading cached pool from {cache_path}")
            return cls.load(cache_path, kin, collision)
        if verbose:
            print(f"[LineDist] no cache; building pool ({n_pool}) "
                  f"and saving to {cache_path}")
        obj = cls(kin=kin, collision=collision,
                  n_pool=n_pool,
                  n_target_noise_deg=n_target_noise_deg,
                  seed=seed,
                  swing_max_deg=swing_max_deg,
                  wavelen_range=wavelen_range,
                  min_radius_m=min_radius_m,
                  cone_range=cone_range, cone_log=cone_log,
                  curve_cfg=curve_cfg, tool_cfg=tool_cfg)
        if feasibility_threshold_m is not None:
            obj.filter_by_classical_controller(
                env_cfg, threshold_m=float(feasibility_threshold_m),
                verbose=verbose)
        obj.save(cache_path)
        return obj

    # ---- filter ----------------------------------------------------------

    def filter_by_classical_controller(self, env_cfg, threshold_m: float = 0.1,
                                       chunk_size: int = 1024,
                                       verbose: bool = True) -> dict:
        """Drop lines where classical_nullspace controller can't reach
        `threshold_m` of EE travel before terminating.

        Returns stats dict.
        """
        # Lazy imports to avoid circular dependency
        from dataclasses import replace
        from Yuan.IJRR.env.env import NSRLBatchedEnv
        from Yuan.IJRR.env.classical_nullspace import (
            ClassicalNullspaceController, cn_action_fn)
        from Yuan.IJRR.env.rollout import rollout_first_episode

        threshold_steps = int(math.ceil(threshold_m / (env_cfg.v * env_cfg.dt)))
        if verbose:
            print(f"[filter] testing {self.n_pool} lines against classical_nullspace "
                  f"controller (threshold = {threshold_m:.3f} m = {threshold_steps} steps)")
        keep = torch.zeros(self.n_pool, dtype=torch.bool, device=self.device)
        n0 = self.n_pool
        for start in range(0, n0, chunk_size):
            end = min(start + chunk_size, n0)
            chunk_n = end - start
            # The wave parameters must go in: filtering these tasks as if
            # they were straight rays would keep tasks the classical law cannot
            # actually start on, and drop none of the ones the bends kill.
            chunk_specs = {
                "q0": self.q_pool[start:end].clone(),
                "line_dir": self.line_dir_pool[start:end].clone(),
                "n_target": self.n_target_pool[start:end].clone(),
                "amp": self.amp_pool[start:end].clone(),
                "wavelen": self.wavelen_pool[start:end].clone(),
            }
            if self.tool_p_pool is not None:
                chunk_specs["tool_p"] = self.tool_p_pool[start:end].clone()
                chunk_specs["tool_R"] = self.tool_R_pool[start:end].clone()
                chunk_specs["tool_lever"] = self.tool_lever_pool[start:end].clone()
                chunk_specs["tool_a0"] = self.tool_a0_pool[start:end].clone()
            if self.tab_pts is not None:
                chunk_specs["path_pts"] = self.tab_pts[start:end].to(self.device)
                chunk_specs["path_tan"] = self.tab_tan[start:end].to(self.device)
                chunk_specs["path_nrm"] = self.tab_nrm[start:end].to(self.device)
                chunk_specs["p0"] = chunk_specs["path_pts"][:, 0].clone()
            # Build a temp env with chunk_n envs, scripted to these chunk specs.
            # The classical law emits basis-box actions, so the temp env is
            # always built with the basis-box interface whatever the training
            # config uses (a dir-frac config would otherwise crash here).
            # The cone stays cfg.cone_deg for every task: a mixed-cone pool
            # is screened at the relaxed tolerance, so the tight-cone tasks
            # are the same geometric tasks under a tighter cone, not a
            # differently filtered subset.
            chunk_cfg = replace(env_cfg, n_envs=chunk_n, dir_frac_action=0,
                                rho_from_norm=False, a_prev_executed=False,
                                task_gate=False, speed_levels=())
            env = NSRLBatchedEnv(chunk_cfg, line_dist=None, device=self.device)
            env.line_dist = ScriptedLineDistribution(chunk_specs)
            ctrl = ClassicalNullspaceController(env.kin)
            stats = rollout_first_episode(env, cn_action_fn(ctrl))
            ep_len = stats["episode_len"]
            keep[start:end] = ep_len >= threshold_steps
            if verbose and ((end // chunk_size) % 10 == 0 or end == n0):
                so_far = int(keep[:end].sum().item())
                print(f"[filter]   {end}/{n0}  kept {so_far} ({100*so_far/end:.1f}%)")
        self.valid_mask = keep
        n_valid = int(keep.sum().item())
        if verbose:
            print(f"[filter] done. {n_valid}/{n0} feasible ({100*n_valid/n0:.1f}%)")
        return {"n_initial": n0, "n_feasible": n_valid,
                "threshold_m": threshold_m, "threshold_steps": threshold_steps}


class ScriptedLineDistribution:
    """Replays a fixed list of line specs in order; used for eval + filter."""

    def __init__(self, specs: dict[str, torch.Tensor]):
        self._specs = specs
        self._cursor = 0
        self._total = specs["q0"].shape[0]

    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        if self._cursor + n > self._total:
            raise RuntimeError(f"ScriptedLineDistribution exhausted: need {n}, "
                               f"have {self._total - self._cursor}")
        out = {k: v[self._cursor:self._cursor + n] for k, v in self._specs.items()}
        self._cursor += n
        return out


class RayStartDistribution:
    """Single-task curriculum: one fixed (line_dir, n_target), start states
    drawn from a precomputed admissible pool sampled along the task's own
    ray. Each pool entry carries its ray anchor p0 = task_p0 + s * d, so an
    episode started mid-ray measures progress (and the lateral corridor)
    from that point onward -- the policy trains on every segment of the
    one line instead of only what it survives to reach."""

    def __init__(self, npz_path: str, device, dtype):
        import numpy as np
        d = np.load(npz_path)
        self.q = torch.tensor(d["q"], dtype=dtype, device=device)
        self.p0 = torch.tensor(d["p0"], dtype=dtype, device=device)
        self.s = torch.tensor(d["s"], dtype=dtype, device=device)
        self._dir = torch.tensor(d["line_dir"], dtype=dtype, device=device)
        self._nt = torch.tensor(d["n_target"], dtype=dtype, device=device)
        self._n = self.q.shape[0]
        # Optional per-start sampling weights (curriculum emphasis).
        if "weight" in d.files:
            w = torch.tensor(d["weight"], dtype=torch.float32, device=device)
            self._cum = torch.cumsum(w / w.sum(), 0)
        else:
            self._cum = None

    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        if self._cum is not None:
            idx = torch.searchsorted(
                self._cum, torch.rand(n, device=self.q.device))
            idx = idx.clamp_max(self._n - 1)
        else:
            idx = torch.randint(0, self._n, (n,), device=self.q.device)
        return {"q0": self.q[idx], "p0": self.p0[idx],
                "progress_offset": self.s[idx],
                "line_dir": self._dir.expand(n, 3).clone(),
                "n_target": self._nt.expand(n, 3).clone()}


class RayMixDistribution:
    """Generalized start curriculum over a full task pool.

    With probability 1 - p_ray an episode starts exactly as the base
    LineDistribution would (the task's own start state); with probability
    p_ray it starts from a precomputed, feasibility-certified mid-line
    state of the SAME task family: an admissible configuration at
    p0 + s0 * line_dir for s0 ~ U(0.02, 1.4). Every row carries an
    explicit p0 anchor and its progress offset, so progress and the
    lateral corridor are measured from the sampled start onward. The
    evaluation protocol is untouched -- this only changes what the policy
    gets to practice."""

    def __init__(self, base, reservoir_npz: str, p_ray: float = 0.7):
        import numpy as np
        self.base = base
        self.p_ray = float(p_ray)
        dev, dtype = base.q_pool.device, base.q_pool.dtype
        d = np.load(reservoir_npz)
        self.r_task = torch.tensor(d["task_idx"], device=dev)
        self.r_q = torch.tensor(d["q"], dtype=dtype, device=dev)
        self.r_s0 = torch.tensor(d["s0"], dtype=dtype, device=dev)
        self.r_p = torch.tensor(d["p_anchor"], dtype=dtype, device=dev)
        self.p0_task = torch.tensor(d["p0_task"], dtype=dtype, device=dev)
        self._n = self.r_q.shape[0]

    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        b = self.base
        dev = b.q_pool.device
        take_ray = torch.rand(n, device=dev) < self.p_ray
        # ray rows: reservoir entries; canonical rows: pool rows drawn the
        # same way base.sample would, but with the index kept so the row's
        # precomputed FK anchor can be attached.
        ridx = torch.randint(0, self._n, (n,), device=dev)
        valid_idx = torch.nonzero(b.valid_mask, as_tuple=False).squeeze(-1)
        cidx = valid_idx[torch.randint(0, len(valid_idx), (n,), device=dev)]
        task = torch.where(take_ray, self.r_task[ridx], cidx)
        q0 = torch.where(take_ray.unsqueeze(-1), self.r_q[ridx],
                         b.q_pool[cidx])
        p0 = torch.where(take_ray.unsqueeze(-1), self.r_p[ridx],
                         self.p0_task[cidx])
        off = torch.where(take_ray, self.r_s0[ridx],
                          torch.zeros_like(self.r_s0[ridx]))
        return {"q0": q0, "p0": p0, "progress_offset": off,
                "line_dir": b.line_dir_pool[task],
                "n_target": b.n_target_pool[task],
                "amp": b.amp_pool[task], "wavelen": b.wavelen_pool[task]}
