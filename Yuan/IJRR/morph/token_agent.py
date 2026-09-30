"""Transformer actor-critic over joint tokens (morphology-general policy).

Input: the flat TokenEnv observation [global, N_MAX joint tokens, mask].
Each joint token is embedded, a learned joint-index embedding is added, the
global token is embedded with a type embedding, and a small encoder mixes
them (padded joints are masked out of attention). The actor reads one
scalar per joint token, the joint-space direction component u_i (the norm
of u carries the fraction, rho_from_norm); the critic reads the global
token. Padded joints receive zero mean and are excluded from the
log-probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from Yuan.IJRR.stage2_traj.ppo import Agent, _layer_init
from Yuan.IJRR.morph.token_env import N_MAX, F_TOK, G_DIM, OBS_DIM


class TokenAgent(Agent):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = N_MAX, hidden_dim: int = 512,
                 init_log_std: float = -0.5, squashed_entropy: bool = True,
                 d_model: int = 192, nhead: int = 4, n_layers: int = 3, **_ignored):
        nn.Module.__init__(self)
        assert obs_dim == OBS_DIM and act_dim == N_MAX, (obs_dim, act_dim)
        self.squashed_entropy = squashed_entropy
        self.d = d_model
        self.tok_embed = nn.Sequential(_layer_init(nn.Linear(F_TOK, d_model)), nn.ReLU(),
                                       _layer_init(nn.Linear(d_model, d_model)))
        self.glob_embed = nn.Sequential(_layer_init(nn.Linear(G_DIM, d_model)), nn.ReLU(),
                                        _layer_init(nn.Linear(d_model, d_model)))
        self.joint_pos = nn.Parameter(torch.zeros(1, N_MAX, d_model))
        self.type_emb = nn.Parameter(torch.zeros(1, 2, d_model))       # 0 = global, 1 = joint
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
                                           batch_first=True, dropout=0.0, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self._joint_head = nn.Sequential(_layer_init(nn.Linear(d_model, hidden_dim)), nn.ReLU(),
                                         _layer_init(nn.Linear(hidden_dim, 1), std=0.01))
        self._value_head = nn.Sequential(_layer_init(nn.Linear(d_model, hidden_dim)), nn.ReLU(),
                                         _layer_init(nn.Linear(hidden_dim, 1), std=1.0))
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    # ---- shared trunk
    def _split(self, x):
        B = x.shape[0]
        g = x[:, :G_DIM]
        tok = x[:, G_DIM:G_DIM + N_MAX * F_TOK].reshape(B, N_MAX, F_TOK)
        mask = x[:, G_DIM + N_MAX * F_TOK:] > 0.5                    # (B, N_MAX) True = real joint
        return g, tok, mask

    def _feat(self, x):
        g, tok, mask = self._split(x)
        hg = self.glob_embed(g).unsqueeze(1) + self.type_emb[:, 0:1]
        ht = self.tok_embed(tok) + self.joint_pos + self.type_emb[:, 1:2]
        seq = torch.cat([hg, ht], 1)                                  # (B, 1+N_MAX, d)
        pad = torch.cat([torch.zeros_like(mask[:, :1]), ~mask], 1)   # True = ignore
        h = self.norm(self.encoder(seq, src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:], mask

    def _mean(self, x):
        hg, ht, mask = self._feat(x)
        mu = self._joint_head(ht).squeeze(-1)                         # (B, N_MAX)
        return torch.where(mask, mu, torch.zeros_like(mu)), hg, mask

    def actor_mean(self, x):
        mu, _, _ = self._mean(x)
        return torch.tanh(mu)

    def get_value(self, x):
        hg, _, _ = self._feat(x)
        return self._value_head(hg).squeeze(-1)

    def _actor_dist(self, x):
        mu, hg, mask = self._mean(x)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mu)
        return Normal(mu, log_std.exp()), hg, mask

    def get_action_and_value(self, x, action=None):
        dist, hg, mask = self._actor_dist(x)
        z = dist.sample() if action is None else action
        m = mask.to(z.dtype)
        lp = dist.log_prob(z) - torch.log((1.0 - torch.tanh(z).pow(2)).clamp(min=1e-6))
        log_prob = (lp * m).sum(-1)
        if self.squashed_entropy:
            z_e = dist.rsample()
            lp_e = dist.log_prob(z_e) - torch.log((1.0 - torch.tanh(z_e).pow(2)).clamp(min=1e-6))
            entropy = -(lp_e * m).sum(-1)
        else:
            entropy = (dist.entropy() * m).sum(-1)
        value = self._value_head(hg).squeeze(-1)
        return z, log_prob, entropy, value, dist.scale.log()
