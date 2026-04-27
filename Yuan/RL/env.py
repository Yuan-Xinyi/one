"""Contextual single-step environment for the farsighted-seed problem.

State (R^14):
    s = [ p0(3), d(3), n(3),                    # raw condition c
          v_path, eps_p, T/MAX_STEPS,            # task params  (DR or fixed)
          ||p0 - FK_pos(home)||,                 # FK-augmented features
          arccos(z_home . n) ]

Action: a = q_seed in R^ndof
Reward: r = rollout_length / T  in [0, 1]      (per-task T, not a global T)
Done:   always True (one-step contextual bandit)
"""
from __future__ import annotations
import numpy as np

import Yuan.RL.config as cfg
from Yuan.RL.rollout import rollout


# ----------------- helpers -----------------
def _sample_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def sample_raw_c(rng: np.random.Generator,
                 n_tilt_max: float | None = None) -> np.ndarray:
    """Sample c = [p0, d, n] in R^9. n_tilt_max defaults to cfg.N_TILT_MAX
    (eval) but can be overridden for OOD splits."""
    if n_tilt_max is None:
        n_tilt_max = cfg.N_TILT_MAX
    tilt = rng.uniform(0.0, n_tilt_max)
    azim = rng.uniform(0.0, 2.0 * np.pi)
    n = np.array([np.sin(tilt) * np.cos(azim),
                  np.sin(tilt) * np.sin(azim),
                  np.cos(tilt)], dtype=np.float32)
    while True:
        v = _sample_unit_vec(rng)
        d = v - n * (v @ n)
        nrm = np.linalg.norm(d)
        if nrm > 1e-3:
            d = (d / nrm).astype(np.float32)
            break
    p0 = rng.uniform(cfg.P0_BOX_LO, cfg.P0_BOX_HI).astype(np.float32)
    return np.concatenate([p0, d, n]).astype(np.float32)


def _build_mjcollider(arm):
    import one.collider.mj_collider as ocm
    mjc = ocm.MJCollider()
    mjc.append(arm)
    mjc.actors = [arm]
    mjc.compile(margin=0.0)
    return mjc


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
                 p0_box: tuple[np.ndarray, np.ndarray] | None = None):
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
        self.randomize = bool(randomize)
        self.n_tilt_range = n_tilt_range          # if None: use defaults
        self.p0_box = p0_box                      # if None: use defaults
        self._cur: dict | None = None
        self.mjc = _build_mjcollider(arm) if use_collision else None
        # cache home FK (TCP pose at home_qs)
        arm.fk(arm.home_qs)
        from Yuan.RL.controller import DLSController
        ctrl = DLSController(arm)
        p_home, R_home, _ = ctrl.fk_with_jac(
            arm.home_qs[arm._chain.active_mask].astype(np.float32))
        self.p_home = p_home.astype(np.float32)
        self.z_home = R_home[:, 2].astype(np.float32)

    # ----- task / state sampling -----
    def _sample_task(self) -> dict:
        rng = self.rng
        # raw c
        if self.n_tilt_range is not None:
            tilt_max = float(rng.uniform(*self.n_tilt_range))
        elif self.randomize:
            tilt_max = float(rng.uniform(*cfg.DR_N_TILT))
        else:
            tilt_max = cfg.N_TILT_MAX
        # actually we sample one task at a time so just use tilt_max = sample
        # bound; we want tilt itself drawn within [0, tilt_max]
        c = sample_raw_c(rng, n_tilt_max=tilt_max)

        # override p0 box if requested (OOD)
        if self.p0_box is not None:
            lo, hi = self.p0_box
            c[:3] = rng.uniform(lo, hi).astype(np.float32)

        # task params
        if self.randomize:
            v_path = float(rng.uniform(*cfg.DR_V_PATH))
            eps_p  = float(rng.uniform(*cfg.DR_EPS_POS))
            T      = int(rng.integers(cfg.DR_T[0], cfg.DR_T[1] + 1))
        else:
            v_path = cfg.V_PATH
            eps_p  = cfg.EPS_POS
            T      = cfg.MAX_STEPS
        return {"c": c, "v_path": v_path, "eps_p": eps_p, "T": T,
                "n_tilt_max": tilt_max}

    def _state_vec(self, task: dict) -> np.ndarray:
        c = task["c"]
        p0, n = c[:3], c[6:9]
        # FK-aug features
        dist_home_p0 = float(np.linalg.norm(p0 - self.p_home))
        cos_zhn = float(np.clip(self.z_home @ n, -1.0, 1.0))
        ang_zhn = float(np.arccos(cos_zhn))
        # task params (T normalised to [0, 1])
        v_norm = task["v_path"]                          # already small magnitude
        eps_norm = task["eps_p"] * 1000.0                # -> mm scale
        T_norm = task["T"] / float(cfg.MAX_STEPS)
        return np.concatenate([
            c,                                                # 9
            np.array([v_norm, eps_norm, T_norm], dtype=np.float32),
            np.array([dist_home_p0, ang_zhn], dtype=np.float32),
        ]).astype(np.float32)

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
        q_seed = np.clip(np.asarray(q_seed, np.float32),
                         self.lmt_lo, self.lmt_up)
        info = rollout(self.arm, q_seed, p0, d, n,
                       mjc=self.mjc,
                       max_steps=task["T"],
                       v_path=task["v_path"],
                       eps_p=task["eps_p"])
        reward = info["length"] / float(task["T"])
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
        for i in range(batch_size):
            self._cur = self._sample_task()
            tasks.append(self._cur)
            states[i] = self._state_vec(self._cur)
            self._cur = None  # mark consumed; will be set again per-step below
        actions, extra = policy_sample_fn(states)
        rewards = np.empty(batch_size, dtype=np.float32)
        lengths = np.empty(batch_size, dtype=np.int32)
        Ts      = np.empty(batch_size, dtype=np.int32)
        reasons: list[str] = []
        for i in range(batch_size):
            self._cur = tasks[i]
            _, r, _, info = self.step(actions[i])
            rewards[i] = r
            lengths[i] = info["length"]
            Ts[i] = tasks[i]["T"]
            reasons.append(info["reason"])
        return states, actions, rewards, lengths, Ts, extra, reasons
