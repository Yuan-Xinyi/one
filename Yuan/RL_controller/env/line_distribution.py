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

import hashlib
import math
from pathlib import Path

import torch

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision


class LineDistribution:
    def __init__(self,
                 kin: BatchedFR3Kinematics,
                 collision: FR3SphereCollision,
                 n_pool: int = 100_000,
                 n_target_noise_deg: float = 5.0,
                 seed: int | None = None,
                 batch_size: int = 8192):
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

        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))

        q_pool, z_pool = [], []
        n_remaining = n_pool
        while n_remaining > 0:
            b = min(batch_size, n_remaining)
            q = kin.rand_conf_batch(b, generator=gen)
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

        # Pre-generate full line spec for every pool entry (deterministic per index).
        # n_target = z_tool + small angular noise about a random axis ⊥ z.
        if self.n_target_noise > 0:
            axis = torch.randn((n_pool, 3), device=self.device, dtype=self.dtype, generator=gen)
            axis = axis - (axis * self.z_pool).sum(-1, keepdim=True) * self.z_pool
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            angle = (torch.rand((n_pool,), device=self.device, dtype=self.dtype, generator=gen)
                     * 2 - 1) * self.n_target_noise
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
        return {
            "q0": self.q_pool[idx],
            "line_dir": self.line_dir_pool[idx],
            "n_target": self.n_target_pool[idx],
        }

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
            "n_target_noise": self.n_target_noise,
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
        obj.n_pool = obj.q_pool.shape[0]
        obj._gen = torch.Generator(device=kin.device)
        return obj

    @staticmethod
    def cache_key(seed, n_pool, n_target_noise_deg, env_cfg,
                  feasibility_threshold_m=None) -> str:
        """Deterministic short key for cache filename. Include a_max since
        the feasibility filter's classical controller is clamped by it."""
        sig = (f"seed={seed}|n={n_pool}|noise={n_target_noise_deg}|"
               f"v={env_cfg.v}|dt={env_cfg.dt}|tcp={env_cfg.tcp_offset}|"
               f"amax={env_cfg.a_max}|thr={feasibility_threshold_m}")
        return hashlib.md5(sig.encode()).hexdigest()[:10]

    @classmethod
    def load_or_build(cls, kin, collision, *,
                      n_pool, n_target_noise_deg, seed, env_cfg,
                      feasibility_threshold_m=None,
                      cache_dir="Yuan/RL_controller/runs/_pool_cache",
                      verbose=True) -> "LineDistribution":
        """Try to load pool from cache; otherwise build (+ filter) and save.

        `feasibility_threshold_m=None` skips the filter (raw pool).
        """
        cache_dir = Path(cache_dir)
        key = cls.cache_key(seed, n_pool, n_target_noise_deg,
                            env_cfg, feasibility_threshold_m)
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
                  seed=seed)
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
        from Yuan.RL_controller.env.env import NSRLBatchedEnv
        from Yuan.RL_controller.env.classical_nullspace import (
            ClassicalNullspaceController, cn_action_fn)
        from Yuan.RL_controller.env.rollout import rollout_first_episode

        threshold_steps = int(math.ceil(threshold_m / (env_cfg.v * env_cfg.dt)))
        if verbose:
            print(f"[filter] testing {self.n_pool} lines against classical_nullspace "
                  f"controller (threshold = {threshold_m:.3f} m = {threshold_steps} steps)")
        keep = torch.zeros(self.n_pool, dtype=torch.bool, device=self.device)
        n0 = self.n_pool
        for start in range(0, n0, chunk_size):
            end = min(start + chunk_size, n0)
            chunk_n = end - start
            chunk_specs = {
                "q0": self.q_pool[start:end].clone(),
                "line_dir": self.line_dir_pool[start:end].clone(),
                "n_target": self.n_target_pool[start:end].clone(),
            }
            # Build a temp env with chunk_n envs, scripted to these chunk specs
            chunk_cfg = replace(env_cfg, n_envs=chunk_n)
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
