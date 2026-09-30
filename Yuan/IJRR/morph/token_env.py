"""Per-joint token observations and the multi-arm environment.

TokenEnv wraps one NSRLBatchedEnv (one arm) and rewrites its observation as
    [global (G), joint tokens (N_MAX x F), joint mask (N_MAX)]
so that one policy can serve arms with different joint counts. Every joint
token carries what the joint is (static chain features) and what it does
right now (its angle, the executed null-space motion, its limit margin, its
world axis and origin relative to the tool tip, and its two Jacobian
columns: how it moves and rotates the tool). The global token carries the
task and the arm-level margins exactly as in the paper's observation.

MultiArmEnv steps several TokenEnvs (different arms) as one batched env for
PPO: observations are concatenated, the 7-channel action is sliced to each
arm's joint count.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.morph.chain_specs import build_kin_collision, static_joint_features

N_MAX = 7
F_STATIC = 16
F_DYN = 16
F_TOK = F_STATIC + F_DYN          # 32
G_DIM = 21
OBS_DIM = G_DIM + N_MAX * F_TOK + N_MAX   # 252


def env_config_for(spec: dict, n_envs: int, base_cfg: dict, **over) -> EnvConfig:
    kw = dict(base_cfg)
    kw.update(robot=spec['name'], n_envs=n_envs, v=float(spec.get('v', 0.2)),
              tcp_offset=float(spec.get('tcp_offset', 0.10)),
              qd_limit=tuple(float(x) for x in spec['qdot_max']))
    kw.update(over)
    return EnvConfig(**kw)


class TokenEnv:
    """One arm; observation rewritten as tokens. Same step/reset contract as
    NSRLBatchedEnv (auto_reset, info['terminal_obs'], info['episode_done'])."""
    obs_dim = OBS_DIM
    act_dim_policy = N_MAX

    def __init__(self, spec: dict, cfg: EnvConfig, line_dist, device):
        kin, coll = build_kin_collision(spec, device)
        self.env = NSRLBatchedEnv(cfg, line_dist, device=device, kin=kin, collision=coll)
        self.spec = spec
        self.device = self.env.device
        self.n_envs = cfg.n_envs
        self.n = self.env.n_joints
        assert self.n <= N_MAX
        d = self.env.kin.dtype
        self.static = torch.as_tensor(static_joint_features(spec), device=self.device, dtype=d)   # (n, 16)
        self.axes = self.env.kin.axes                                                          # (n, 3) local
        self.mask = torch.zeros((N_MAX,), device=self.device, dtype=d); self.mask[:self.n] = 1.0
        self.q_half = self.env.q_half; self.q_mid = self.env.q_mid
        # offsets in the base observation (rm layout): q(n) q^2(n) d z n cos zxn a_prev(n) margins(4) proj(3)
        self._o_dir = 2 * self.n
        self._o_aprev = 2 * self.n + 13
        self._o_marg = self._o_aprev + self.n
        assert self.env.obs_dim == 3 * self.n + 20, self.env.obs_dim

    @property
    def done_persistent(self):
        return self.env.done_persistent

    @property
    def arc_progress(self):
        return self.env.arc_progress

    def _tokens(self, base_obs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B = q.shape[0]
        n = self.n
        p_tip, R_tip, J, _ = self.env.kin.tcp_fk_jac(q)
        T = self.env.kin.link_transforms(q)                       # (B, n+1, 4, 4)
        R_l = T[:, 1:, :3, :3]; o_l = T[:, 1:, :3, 3]             # joint frames after motion
        axis_w = torch.einsum('bnij,nj->bni', R_l, self.axes)     # world joint axes
        rel = o_l - p_tip.unsqueeze(1)                            # joint origin relative to the tip
        q_norm = (q - self.q_mid) / self.q_half
        jl = (self.q_half - (q - self.q_mid).abs()) / self.q_half
        a_prev = base_obs[:, self._o_aprev:self._o_aprev + n]
        Jp = J[:, :3, :].transpose(1, 2)                          # (B, n, 3)
        Jr = J[:, 3:, :].transpose(1, 2)
        dyn = torch.cat([q_norm.unsqueeze(-1), (q_norm * q_norm).unsqueeze(-1), a_prev.unsqueeze(-1),
                         jl.unsqueeze(-1), axis_w, rel, Jp, Jr], -1)       # (B, n, 16)
        tok = torch.cat([dyn, self.static.unsqueeze(0).expand(B, n, F_STATIC)], -1)
        if n < N_MAX:
            tok = torch.cat([tok, torch.zeros((B, N_MAX - n, F_TOK), device=self.device, dtype=tok.dtype)], 1)
        return tok

    def _global(self, base_obs: torch.Tensor) -> torch.Tensor:
        B = base_obs.shape[0]
        g = base_obs[:, self._o_dir:self._o_dir + 13]              # d, z, n, cos, zxn
        m = base_obs[:, self._o_marg:self._o_marg + 7]             # margins(4) + proj scales(3)
        nn = torch.full((B, 1), self.n / N_MAX, device=self.device, dtype=base_obs.dtype)
        return torch.cat([g, m, nn], -1)                           # 21

    def _obs(self, base_obs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        B = base_obs.shape[0]
        tok = self._tokens(base_obs, q).reshape(B, -1)
        return torch.cat([self._global(base_obs), tok, self.mask.unsqueeze(0).expand(B, N_MAX)], -1).float()

    def reset(self):
        base = self.env.reset()
        return self._obs(base, self.env.q)

    def current_obs(self):
        return self._obs(self.env.current_obs(), self.env.q)

    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        a = actions[:, :self.n]
        if auto_reset:
            # terminal observation must be built from the pre-reset state
            base, rew, term, trunc, info = self.env.step(a, auto_reset=False)
            done = term | trunc
            info['terminal_obs'] = self._obs(info['terminal_obs'], self.env.q)
            if bool(done.any()):
                self.env._reset_envs(done)
                self.env.done_persistent[:] = False
                base = self.env.current_obs()
            obs = self._obs(base, self.env.q)
            info['episode_done'] = done
            return obs, rew, term, trunc, info
        base, rew, term, trunc, info = self.env.step(a, auto_reset=False)
        info['terminal_obs'] = self._obs(info['terminal_obs'], self.env.q)
        return self._obs(base, self.env.q), rew, term, trunc, info


class MultiArmEnv:
    """Several TokenEnvs stepped together as one batched environment."""
    obs_dim = OBS_DIM
    act_dim = N_MAX
    act_dim_policy = N_MAX

    def __init__(self, envs: list[TokenEnv]):
        self.envs = envs
        self.device = envs[0].device
        self.sizes = [e.n_envs for e in envs]
        self.n_envs = sum(self.sizes)
        self.max_steps = max(e.env.max_steps for e in envs)
        self._cut = np.cumsum([0] + self.sizes)

    def _cat(self, xs):
        return torch.cat(list(xs), 0)

    def reset(self):
        return self._cat(e.reset() for e in self.envs)

    def current_obs(self):
        return self._cat(e.current_obs() for e in self.envs)

    @property
    def done_persistent(self):
        return self._cat(e.done_persistent for e in self.envs)

    @property
    def arc_progress(self):
        return self._cat(e.arc_progress for e in self.envs)

    def step(self, actions: torch.Tensor, auto_reset: bool = True):
        outs = [e.step(actions[lo:hi], auto_reset=auto_reset)
                for e, lo, hi in zip(self.envs, self._cut[:-1], self._cut[1:])]
        obs = self._cat(o[0] for o in outs); rew = self._cat(o[1] for o in outs)
        term = self._cat(o[2] for o in outs); trunc = self._cat(o[3] for o in outs)
        info = {'terminal_obs': self._cat(o[4]['terminal_obs'] for o in outs),
                'episode_done': self._cat(o[4]['episode_done'] for o in outs),
                'term_reason': self._cat(o[4]['term_reason'] for o in outs)}
        w = np.asarray(self.sizes, np.float64) / self.n_envs
        for k in ('r_progress_mean', 'lateral_err_mean', 'fb_rate_e0', 'fb_rate_e1', 'fb_rate_e2', 'sigma_min'):
            vals = [o[4][k] for o in outs]
            if torch.is_tensor(vals[0]):
                info[k] = self._cat(vals)
            else:
                info[k] = float(sum(float(v) * wi for v, wi in zip(vals, w)))
        info['lateral_err_max'] = max(float(o[4]['lateral_err_max']) for o in outs)
        info['force_err_max'] = 0.0
        n_done = [int(o[4]['n_episodes_done']) for o in outs]
        info['n_episodes_done'] = sum(n_done)
        for k in ('ep_reward_mean', 'ep_len_mean', 'ep_progress_mean', 'ep_arc_progress_mean'):
            num = sum(float(o[4][k]) * nd for o, nd in zip(outs, n_done) if nd > 0 and math.isfinite(float(o[4][k])))
            info[k] = num / sum(n_done) if sum(n_done) > 0 else float('nan')
        info['ep_progress_max'] = max(float(o[4]['ep_progress_max']) if o[4]['n_episodes_done'] > 0 else float('-inf') for o in outs)
        return obs, rew, term, trunc, info
