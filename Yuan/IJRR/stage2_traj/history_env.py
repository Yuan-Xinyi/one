"""Observation-history wrapper for sequence backbones under flat PPO.

ppo.train shuffles (T x B) samples across time, so temporal context cannot
live in the network state; it must live in the stored observation. This
wrapper maintains a rolling window of the last K base observations per env
and exposes their concatenation as the observation. On an auto-reset the
fresh episode's window is filled with its first observation, so no history
leaks across episode boundaries.
"""
from __future__ import annotations

import torch


_OWN = ('_env', 'k', 'base_obs_dim', 'obs_dim', '_h')


class HistoryStackEnv:
    def __init__(self, env, k: int):
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, 'k', int(k))
        object.__setattr__(self, 'base_obs_dim', env.obs_dim)
        object.__setattr__(self, 'obs_dim', env.obs_dim * int(k))
        object.__setattr__(self, '_h', None)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        if name in _OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)

    def _stack(self):
        return self._h.reshape(self._h.shape[0], -1)

    def reset(self):
        o = self._env.reset()
        object.__setattr__(self, '_h',
                           o.unsqueeze(1).repeat(1, self.k, 1).clone())
        return self._stack()

    def step(self, actions, auto_reset: bool = True):
        o, r, term, trunc, info = self._env.step(actions,
                                                 auto_reset=auto_reset)
        prev = self._h
        if isinstance(info, dict) and 'terminal_obs' in info:
            tobs = info['terminal_obs']
            tstack = torch.cat([prev[:, 1:], tobs.unsqueeze(1)], dim=1)
            info = dict(info)
            info['terminal_obs'] = tstack.reshape(tstack.shape[0], -1)
        h = torch.cat([prev[:, 1:], o.unsqueeze(1)], dim=1)
        if auto_reset:
            done = info['episode_done']
            if bool(done.any()):
                h[done] = o[done].unsqueeze(1)
        object.__setattr__(self, '_h', h)
        return self._stack(), r, term, trunc, info

    def current_obs(self):
        # refresh the newest slot only (line_dir refresh side effect kept)
        o = self._env.current_obs()
        h = self._h.clone()
        h[:, -1] = o
        object.__setattr__(self, '_h', h)
        return self._stack()
