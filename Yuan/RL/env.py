"""Contextual single-step environment for the farsighted-seed problem.

State (R^14):
    s = [ p0(3), d(3), n(3),                    # raw condition c
          v_path, eps_p, T/MAX_STEPS,            # task params  (DR or fixed)
          ||p0 - FK_pos(home)||,                 # FK-augmented features
          arccos(z_home . n) ]

Action: a = [cos(phi), sin(phi), cos(psi), sin(psi)] in branch mode,
        or q_seed in R^ndof in legacy joint-seed mode.
Reward: r = rollout_length / T  in [0, 1]      (per-task T, not a global T)
Done:   always True (one-step contextual bandit)
"""
from __future__ import annotations
import numpy as np

import Yuan.RL.config as cfg
from Yuan.RL.rollout import rollout
from Yuan.RL.controller import DLSController
from Yuan.RL.rollout import build_target_rotmat


# ----------------- helpers -----------------
def _sample_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _uniform_sphere_n(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit vector on S^2 via standard-normal + normalize.
    Density is *truly* uniform per unit area on the sphere (unlike the
    legacy tilt+azim sampling which biased toward the poles)."""
    while True:
        v = rng.standard_normal(3).astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-6:
            return v / nrm


def sample_raw_c(rng: np.random.Generator,
                 n_tilt_max: float | None = None,
                 n_tilt_range: tuple[float, float] | None = None,
                 p0_box: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Sample c = [p0, d, n] in R^9.

    n is sampled uniform on the full sphere by default. Constraints:
      - n_tilt_range = (lo, hi): keep only n whose tilt-from-+z is in band
      - n_tilt_max:              keep only n with tilt <= n_tilt_max
                                  (= polar cap of half-angle n_tilt_max)
    Both use rejection sampling on the uniform sphere.
    """
    if n_tilt_range is not None:
        lo, hi = float(n_tilt_range[0]), float(n_tilt_range[1])
        cos_lo, cos_hi = float(np.cos(hi)), float(np.cos(lo))  # note swap
        for _ in range(1000):
            n = _uniform_sphere_n(rng)
            if cos_lo <= float(n[2]) <= cos_hi:
                break
        else:
            n = _uniform_sphere_n(rng)
    elif n_tilt_max is not None and n_tilt_max < np.pi - 1e-6:
        cos_max = float(np.cos(n_tilt_max))
        for _ in range(1000):
            n = _uniform_sphere_n(rng)
            if float(n[2]) >= cos_max:
                break
        else:
            n = _uniform_sphere_n(rng)
    else:
        n = _uniform_sphere_n(rng)

    while True:
        v = _sample_unit_vec(rng)
        d = v - n * (v @ n)
        nrm = float(np.linalg.norm(d))
        if nrm > 1e-3:
            d = (d / nrm).astype(np.float32)
            break

    if p0_box is None:
        p0 = rng.uniform(cfg.P0_BOX_LO, cfg.P0_BOX_HI).astype(np.float32)
    else:
        lo, hi = p0_box
        p0 = rng.uniform(lo, hi).astype(np.float32)
    return np.concatenate([p0, d, n]).astype(np.float32)


def _build_mjcollider(arm):
    import one.collider.mj_collider as ocm
    mjc = ocm.MJCollider()
    mjc.append(arm)
    mjc.actors = [arm]
    mjc.compile(margin=0.0)
    return mjc


def _seed_manifold_penalty(seed_pos_err: np.ndarray,
                           seed_orient_err: np.ndarray) -> np.ndarray:
    pos_term = np.minimum(seed_pos_err / float(cfg.SEED_POS_ERR_SCALE), 1.0)
    orient_term = np.minimum(seed_orient_err / float(cfg.SEED_ORIENT_ERR_SCALE), 1.0)
    return float(cfg.SEED_MANIFOLD_COEF) * 0.5 * (pos_term + orient_term)


# ----------------- env -----------------
class FarsightedSeedEnv:
    """Single-step contextual env wrapping FR3 (or any redundant arm).

    Domain randomisation
    --------------------
    If `randomize=True` (training default), each reset() samples
        v_path  ~ U(DR_V_PATH)
        eps_p   ~ U(DR_EPS_POS)
        T       ~ U(DR_T)         (int)
        n_tilt  ~ U(DR_N_TILT)    (wider than eval)
    These task parameters are concatenated into the state vector so the
    policy can observe what setting it is acting in.

    For evaluation pass `randomize=False` and the env will use the fixed
    config defaults (V_PATH, EPS_POS, MAX_STEPS, N_TILT_MAX).
    """

    def __init__(self, arm=None, seed: int = 0,
                 use_collision: bool = cfg.USE_COLLISION_CHECK,
                 randomize: bool = True,
                 # OOD knobs (override sampling distributions)
                 n_tilt_range: tuple[float, float] | None = None,
                 p0_box: tuple[np.ndarray, np.ndarray] | None = None,
                 eval_T: int | None = None):
        if arm is None:
            from one.robots.manipulators.franka.fr3.fr3 import FR3
            arm = FR3()
        self.arm = arm
        self.ndof = int(arm.ndof)
        self.rng = np.random.default_rng(seed)
        self.lmt_lo = np.asarray(arm._chain.lmt_lo, dtype=np.float32)
        self.lmt_up = np.asarray(arm._chain.lmt_up, dtype=np.float32)
        self.q_mid  = 0.5 * (self.lmt_lo + self.lmt_up)
        self.q_half = 0.5 * (self.lmt_up - self.lmt_lo)
        if cfg.ACTION_MODE == "branch_descriptor":
            self.action_dim = int(cfg.BRANCH_ACTION_DIM)
            self.action_mid = np.zeros(self.action_dim, dtype=np.float32)
            self.action_half = np.ones(self.action_dim, dtype=np.float32)
        else:
            self.action_dim = self.ndof
            self.action_mid = self.q_mid.copy()
            self.action_half = self.q_half.copy()
        self.randomize = bool(randomize)
        self.n_tilt_range = n_tilt_range          # if None: use defaults
        self.p0_box = p0_box                      # if None: use defaults
        self.eval_T = eval_T                      # fixed T in non-randomize mode
        self._cur: dict | None = None
        self._reach_kin = None
        self._reach_actions = None
        self.mjc = _build_mjcollider(arm) if use_collision else None
        # cache home FK (TCP pose at home_qs)
        arm.fk(arm.home_qs)
        from Yuan.RL.controller import DLSController
        self._ctrl = DLSController(arm)              # cache for repeated FK
        p_home, R_home, _ = self._ctrl.fk_with_jac(
            arm.home_qs[arm._chain.active_mask].astype(np.float32))
        self.p_home = p_home.astype(np.float32)
        self.z_home = R_home[:, 2].astype(np.float32)

    def _training_p0_box(self):
        if self.p0_box is not None:
            return self.p0_box
        if self.randomize:
            return cfg.DR_P0_BOX_LO, cfg.DR_P0_BOX_HI
        return cfg.P0_BOX_LO, cfg.P0_BOX_HI

    def _sample_p0_in_shell(self, rng: np.random.Generator) -> np.ndarray:
        lo, hi = self._training_p0_box()
        r_min, r_max = cfg.DR_P0_RADIUS if self.randomize else (0.0, float('inf'))
        for _ in range(200):
            p0 = rng.uniform(lo, hi).astype(np.float32)
            radius = float(np.linalg.norm(p0))
            if r_min <= radius <= r_max:
                return p0
        return rng.uniform(lo, hi).astype(np.float32)

    def _branch_reach_actions(self):
        if self._reach_actions is not None:
            return self._reach_actions
        phi_vals = np.linspace(0.0, 2.0 * np.pi,
                               int(cfg.DR_REACHABILITY_PHI_SAMPLES),
                               endpoint=False)
        psi_vals = np.linspace(0.0, 2.0 * np.pi,
                               int(cfg.DR_REACHABILITY_PSI_SAMPLES),
                               endpoint=False)
        actions = []
        for phi in phi_vals:
            for psi in psi_vals:
                actions.append([np.cos(phi), np.sin(phi),
                                np.cos(psi), np.sin(psi)])
        self._reach_actions = np.asarray(actions, dtype=np.float32)
        return self._reach_actions

    def _reachability_device(self):
        import torch
        if cfg.BATCHED_ROLLOUT_DEVICE == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(cfg.BATCHED_ROLLOUT_DEVICE)

    def _reachable_mask(self, c_batch: np.ndarray) -> np.ndarray:
        if cfg.ACTION_MODE != "branch_descriptor":
            return np.ones(c_batch.shape[0], dtype=bool)
        import torch
        from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
        from Yuan.RL.batched_rollout import (
            build_branch_rotmat_batch,
            branch_project_multistart,
        )
        if self._reach_kin is None:
            self._reach_kin = BatchedFR3Kinematics(device=self._reachability_device())
        actions = self._branch_reach_actions()
        device = self._reach_kin.device
        c_t = torch.as_tensor(c_batch, device=device, dtype=torch.float32)
        action_t = torch.as_tensor(actions, device=device, dtype=torch.float32)
        num_tasks = c_t.shape[0]
        num_actions = action_t.shape[0]
        p0 = c_t[:, None, :3].expand(num_tasks, num_actions, 3).reshape(-1, 3)
        d = c_t[:, None, 3:6].expand(num_tasks, num_actions, 3).reshape(-1, 3)
        n = c_t[:, None, 6:9].expand(num_tasks, num_actions, 3).reshape(-1, 3)
        action_rep = action_t[None, :, :].expand(num_tasks, num_actions, 4).reshape(-1, 4)
        R_tgt = build_branch_rotmat_batch(d, n, action_rep)
        _, ok, _ = branch_project_multistart(self._reach_kin, p0, R_tgt, action_rep)
        return ok.view(num_tasks, num_actions).any(dim=1).detach().cpu().numpy()

    def _is_task_reachable(self, c: np.ndarray) -> bool:
        return bool(self._reachable_mask(c[None, :])[0])

    def _sample_p0_n_via_random_q(self, rng):
        """Sample p0, n by drawing a random joint config and forward-FK'ing.
        Guarantees (p0, n) is *physically reachable* by FR3 (one IK solution
        is the q we sampled). After the TCP_z=-n convention flip, n is the
        outward surface normal: n = -R_tcp[:, 2]."""
        chain = self.arm._chain
        lo = chain.lmt_lo + float(cfg.RANDOM_Q_MARGIN)
        hi = chain.lmt_up - float(cfg.RANDOM_Q_MARGIN)
        q = rng.uniform(lo, hi).astype(np.float32)
        p_tcp, R_tcp, _ = self._ctrl.fk_with_jac(q)
        p0 = p_tcp.astype(np.float32)
        n  = (-R_tcp[:, 2]).astype(np.float32)
        return p0, n, q

    def _sample_task_candidate(self) -> dict:
        rng = self.rng
        q_sample: np.ndarray | None = None
        if cfg.TASK_SAMPLE_MODE == "random_q":
            # guaranteed reachable: random q -> FK -> (p0, n)
            p0, n, q_sample = self._sample_p0_n_via_random_q(rng)
            # if OOD wants a tilt band, rejection-sample over q for it
            if self.n_tilt_range is not None:
                lo_tilt, hi_tilt = self.n_tilt_range
                cos_lo, cos_hi = float(np.cos(hi_tilt)), float(np.cos(lo_tilt))
                for _ in range(500):
                    if cos_lo <= float(n[2]) <= cos_hi:
                        break
                    p0, n, q_sample = self._sample_p0_n_via_random_q(rng)
            # d: random unit vector in the plane perp to n
            while True:
                v = _sample_unit_vec(rng)
                d = v - n * (v @ n)
                nrm = float(np.linalg.norm(d))
                if nrm > 1e-3:
                    d = (d / nrm).astype(np.float32)
                    break
            c = np.concatenate([p0, d, n]).astype(np.float32)
        else:
            # legacy "workspace" mode
            kwargs = dict(p0_box=self._training_p0_box())
            if self.n_tilt_range is not None:
                kwargs["n_tilt_range"] = self.n_tilt_range
            c = sample_raw_c(rng, **kwargs)
            c[:3] = self._sample_p0_in_shell(rng)

        if self.randomize:
            v_path = float(rng.uniform(*cfg.DR_V_PATH))
            eps_p  = float(rng.uniform(*cfg.DR_EPS_POS))
            T      = int(rng.integers(cfg.DR_T[0], cfg.DR_T[1] + 1))
        else:
            v_path = cfg.V_PATH
            eps_p  = cfg.EPS_POS
            T      = int(self.eval_T) if self.eval_T is not None else cfg.MAX_STEPS
        return {"c": c, "v_path": v_path, "eps_p": eps_p, "T": T,
                "q_sample": q_sample}

    def _sample_tasks(self, count: int) -> list[dict]:
        # random-q mode is guaranteed reachable by construction; skip filter
        if cfg.TASK_SAMPLE_MODE == "random_q":
            return [self._sample_task_candidate() for _ in range(count)]
        if (not self.randomize
                or not cfg.DR_SAMPLE_REACHABLE_ONLY
                or cfg.ACTION_MODE != "branch_descriptor"):
            return [self._sample_task_candidate() for _ in range(count)]

        tasks: list[dict] = []
        tries = 0
        while len(tasks) < count and tries < int(cfg.DR_REACHABILITY_TRIES):
            need = count - len(tasks)
            cand_count = max(need * 3, 16)
            candidates = [self._sample_task_candidate() for _ in range(cand_count)]
            c_batch = np.stack([task["c"] for task in candidates], axis=0)
            keep = self._reachable_mask(c_batch)
            tasks.extend([task for task, ok in zip(candidates, keep) if ok])
            tries += cand_count
        if len(tasks) < count:
            raise RuntimeError("Failed to sample enough reachable FR3 tasks.")
        return tasks[:count]

    # ----- task / state sampling -----
    def _sample_task(self) -> dict:
        return self._sample_tasks(1)[0]

    def _state_vec(self, task: dict) -> np.ndarray:
        c = task["c"]
        p0, n = c[:3], c[6:9]
        # FK-aug features
        dist_home_p0 = float(np.linalg.norm(p0 - self.p_home))
        cos_zhn = float(np.clip(self.z_home @ n, -1.0, 1.0))
        ang_zhn = float(np.arccos(cos_zhn))
        # geom-aug features (v9): how the task relates to FR3's reachable region
        shoulder = np.asarray(cfg.FR3_SHOULDER, dtype=np.float32)
        dist_p0_shoulder = float(np.linalg.norm(p0 - shoulder))
        reach_margin = float(max(0.0, cfg.FR3_REACH_RADIUS - np.linalg.norm(p0)))
        n_dot_grav = float(-n[2])    # +1 = normal pointing down, -1 = up
        # task params (T normalised to [0, 1])
        v_norm = task["v_path"]                          # already small magnitude
        eps_norm = task["eps_p"] * 1000.0                # -> mm scale
        T_norm = task["T"] / float(cfg.MAX_STEPS)
        feats = [
            c,                                                # 9
            np.array([v_norm, eps_norm, T_norm], dtype=np.float32),
            np.array([dist_home_p0, ang_zhn], dtype=np.float32),
        ]
        # v9 geom-aug only emitted when config asks for the wider state
        if cfg.STATE_DIM >= cfg.RAW_C_DIM + cfg.TASK_PARAM_DIM \
                + cfg.FK_AUG_DIM + cfg.GEOM_AUG_DIM:
            feats.append(np.array(
                [dist_p0_shoulder, reach_margin, n_dot_grav],
                dtype=np.float32))
        return np.concatenate(feats).astype(np.float32)

    # ----- gym-like API -----
    def reset(self) -> np.ndarray:
        self._cur = self._sample_task()
        return self._state_vec(self._cur)

    def step(self, q_seed: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self._cur is None:
            raise RuntimeError("Call reset() before step().")
        task = self._cur
        c = task["c"]
        p0, d, n = c[:3], c[3:6], c[6:9]
        q_seed = np.asarray(q_seed, np.float32)
        if cfg.ACTION_MODE == "joint_seed":
            q_seed = np.clip(q_seed, self.lmt_lo, self.lmt_up)
        info = rollout(self.arm, q_seed, p0, d, n,
                       mjc=self.mjc,
                       max_steps=task["T"],
                       v_path=task["v_path"],
                       eps_p=task["eps_p"])
        reward = info["length"] / float(task["T"])
        if cfg.SEED_MANIFOLD_REG and cfg.ACTION_MODE == "joint_seed":
            ctrl = DLSController(self.arm)
            R_tgt = build_target_rotmat(d, n)
            p_seed, R_seed, _ = ctrl.fk_with_jac(q_seed)
            seed_pos_err = float(np.linalg.norm(p0 - p_seed))
            seed_orient_err = float(np.arccos(np.clip(R_seed[:, 2] @ R_tgt[:, 2],
                                                      -1.0, 1.0)))
            penalty = float(_seed_manifold_penalty(
                np.asarray([seed_pos_err], dtype=np.float32),
                np.asarray([seed_orient_err], dtype=np.float32))[0])
            reward -= penalty
            info["seed_pos_err"] = seed_pos_err
            info["seed_orient_err"] = seed_orient_err
            info["seed_manifold_penalty"] = penalty
        info["reward"] = reward
        info["task"] = task
        s_out = self._state_vec(task)
        self._cur = None
        return s_out, reward, True, info

    # ----- batched, sequential collection -----
    def collect_batch(self, policy_sample_fn, batch_size: int):
        states = np.empty((batch_size, cfg.STATE_DIM), dtype=np.float32)
        # remember the full task dicts to replay during step()
        tasks: list[dict] = []
        tasks = self._sample_tasks(batch_size)
        for i, task in enumerate(tasks):
            states[i] = self._state_vec(task)
        actions, extra = policy_sample_fn(states)
        rewards = np.empty(batch_size, dtype=np.float32)
        lengths = np.empty(batch_size, dtype=np.int32)
        Ts      = np.empty(batch_size, dtype=np.int32)
        reasons: list[str] = []
        if cfg.BATCHED_ROLLOUT and self.mjc is None:
            from Yuan.RL.batched_rollout import batched_rollout
            c_batch = np.stack([task["c"] for task in tasks], axis=0)
            v_batch = np.asarray([task["v_path"] for task in tasks],
                                 dtype=np.float32)
            eps_batch = np.asarray([task["eps_p"] for task in tasks],
                                   dtype=np.float32)
            Ts[:] = np.asarray([task["T"] for task in tasks], dtype=np.int32)
            out = batched_rollout(actions, c_batch, v_batch, eps_batch, Ts)
            lengths[:] = out["lengths"]
            rewards[:] = lengths.astype(np.float32) / Ts.astype(np.float32)
            if cfg.SEED_MANIFOLD_REG and cfg.ACTION_MODE == "joint_seed":
                penalty = _seed_manifold_penalty(
                    out["seed_pos_err"], out["seed_orient_err"])
                rewards[:] = rewards - penalty.astype(np.float32)
            reasons = out["reasons"]
            return states, actions, rewards, lengths, Ts, extra, reasons

        for i in range(batch_size):
            self._cur = tasks[i]
            _, r, _, info = self.step(actions[i])
            rewards[i] = r
            lengths[i] = info["length"]
            Ts[i] = tasks[i]["T"]
            reasons.append(info["reason"])
        return states, actions, rewards, lengths, Ts, extra, reasons
