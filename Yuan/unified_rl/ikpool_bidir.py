"""Bidirectional closed loop on the IK candidate pool.

The framework's two directions, both with real load for the first time:
  backward: selector trained from complete controller returns (S0 on C0 cache)
  forward : controller adapted to the selector-induced reset distribution
            (C0 -> C1 via continued PPO under SeedPolicyLineDistribution)
  backward again: relabel all candidates under C1, retrain selector (S1)
  final: strict 2x2 (S0/S1 x C0/C1) + old-SOTA comparison on val/external.

Stages (each resumable; artifacts under runs/ikpool_full_v1/ and runs/ikpool_c1/):
  s0        train + save 5-member SetSel on the C0 return cache
  forward   continue PPO from C0 on S0-mixture resets (0.7/0.2/0.1), save C1
  gate      S0 picks: C0 (cache lookup) vs C1 (rollout) on validation/external
  relabel   re-roll all valid candidates under C1 (train + dev sets, sharded)
  s1        train + save SetSel on the C1 return cache
  eval2x2   four arms + old system, robust paired statistics
"""
import argparse, copy, json, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from Yuan.RL_controller.algorithms.ppo import train as ppo_train
from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import FrozenRLController, rollout_selected_seeds
from Yuan.unified_rl.validity import validate_cached_dataset, check_candidate_validity
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features
from Yuan.unified_rl.seed_distribution import SeedPolicyLineDistribution
from Yuan.unified_rl.seed_deployment import SeedDeploymentConfig

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
C0_DIR = 'Yuan/unified_rl/runs/r2_grouped_best'
C1_DIR = Path('Yuan/unified_rl/runs/ikpool_c1')
OLD = {'validation': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
       'external': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'}
MEMBERS, EPOCHS, TEMP, WD = 5, 300, 0.1, 1e-4
ROLL_CHUNK = 512
FORWARD_SEED = 20260725


class SetSel(nn.Module):
    """Per-candidate score + feasibility(metres) with a mean-pool set context."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.feas = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, X, V):
        e = self.enc(X)
        vf = V.unsqueeze(-1).float()
        ctx = (e * vf).sum(1) / vf.sum(1).clamp_min(1)
        h = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
        return self.score(h).squeeze(-1), self.feas(h).squeeze(-1)


class EnsembleSeedPolicy(nn.Module):
    """Adapter exposing distribution_and_values() for SeedPolicyLineDistribution."""

    def __init__(self, nets, mu, sd):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.register_buffer('mu', mu)
        self.register_buffer('sd', sd)

    @torch.no_grad()
    def distribution_and_values(self, features, valid):
        Xz = ((features - self.mu) / self.sd).masked_fill(~valid.unsqueeze(-1), 0.0)
        ss, ff = [], []
        for net in self.nets:
            s, f = net(Xz, valid)
            ss.append(s.masked_fill(~valid, -1e9))
            ff.append(f)
        score = torch.stack(ss).mean(0)
        feas = torch.stack(ff).mean(0)
        return torch.distributions.Categorical(logits=score), None, feas


def _kin(device):
    return build_env_from_run(resolve_controller_dir(C0_DIR), 1, device).kin


def _load_pool(which, device, returns_tag=''):
    name = 'ikpool' if which == 'train' else f'ikpool_{which}'
    cand = D / (f'{name}_candidates.npz' if which != 'train' else 'ikpool_candidates.npz')
    retp = D / (f'{name}_returns{returns_tag}.npz'
                if which != 'train' else f'ikpool_returns{returns_tag}.npz')
    ds = CachedSeedCandidateDataset.from_npz(cand)
    ret = np.load(retp)
    X = _build_features(_kin(device), ds, 4096).to(device)
    P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(device)
    V = torch.as_tensor(ret['valid']).to(device)
    return ds, X, P, V, ret['task_indices']


def _train_selector(X, P, V, device, out_path):
    mu, sd = X[V].mean(0), X[V].std(0).clamp_min(1e-6)
    Xz_all = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    hub = nn.HuberLoss(delta=0.05, reduction='none')
    nets = []
    for m in range(MEMBERS):
        g = torch.Generator().manual_seed(1000 * (m + 1))
        boot = torch.randint(0, len(X), (len(X),), generator=g).to(device)
        Xz, Vb, Pb = Xz_all[boot], V[boot], P[boot]
        lo = torch.where(Vb, Pb, torch.tensor(1e9, device=device)).min(1, keepdim=True).values
        hi = torch.where(Vb, Pb, torch.tensor(-1e9, device=device)).max(1, keepdim=True).values
        T = ((Pb - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~Vb, 0.0)
        torch.manual_seed(1000 * (m + 1))
        net = SetSel().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
        for _ in range(EPOCHS):
            opt.zero_grad()
            s, f = net(Xz, Vb)
            s = s.masked_fill(~Vb, -1e9)
            tgt = torch.softmax((T / TEMP).masked_fill(~Vb, -1e9), 1)
            rank = -(tgt * torch.log_softmax(s, 1).clamp_min(-30)).sum(1).mean()
            feas = (hub(f, Pb) * Vb.float()).sum() / Vb.float().sum()
            (rank + feas).backward(); opt.step()
        nets.append(net)
        print(f'[selector] member {m+1}/{MEMBERS} done', flush=True)
    torch.save({'members': [n.state_dict() for n in nets],
                'mu': mu.cpu(), 'sd': sd.cpu()}, out_path)
    print(f'[selector] saved -> {out_path}', flush=True)
    return nets, mu, sd


def _load_selector(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    nets = []
    for st in ck['members']:
        n = SetSel().to(device); n.load_state_dict(st); n.eval()
        nets.append(n)
    return nets, ck['mu'].to(device), ck['sd'].to(device)


@torch.no_grad()
def _picks(nets, mu, sd, X, V):
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    score = torch.stack([n(Xz, V)[0].masked_fill(~V, -1e9) for n in nets]).mean(0)
    return score.argmax(1)


def _roll_indices(ds, indices, controller_dir, device):
    """Roll one chosen candidate per task under the given controller."""
    env = build_env_from_run(resolve_controller_dir(controller_dir), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(controller_dir), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(controller_dir))).gamma)
    ctl = FrozenRLController(agent)
    n = len(ds)
    out = np.zeros(n, np.float32)
    for s in range(0, n, ROLL_CHUNK):
        rows = torch.arange(s, min(s + ROLL_CHUNK, n))
        nr = len(rows)
        if nr < ROLL_CHUNK:
            rows = torch.cat([rows, rows[-1:].expand(ROLL_CHUNK - nr)])
        cand = ds.batch.index_select(rows).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, indices[rows.to(device)].to(device), ctl, gamma=gamma)
        out[s:s + nr] = res.progress_m[:nr].cpu().numpy()
    return out


def _paired(new, old):
    d = new - old
    t = np.sort(d); k = int(0.05 * len(d)); tm = t[k:-k] if k else t
    rng = np.random.default_rng(7)
    boots = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(1)
    return {'delta_mm': float(d.mean() * 1e3),
            'ci95_mm': [float(np.percentile(boots, 2.5) * 1e3),
                        float(np.percentile(boots, 97.5) * 1e3)],
            'trimmed_mm': float(tm.mean() * 1e3),
            'harm_pct': float((d < -1e-3).mean() * 100),
            'win_pct': float((d > 1e-3).mean() * 100)}


# ---------------------------------------------------------------- stages
def stage_s0(args, device):
    _, X, P, V, _ = _load_pool('train', device)
    _train_selector(X, P, V, device, D / 'ikpool_selector_s0.pt')


def stage_forward(args, device):
    import dataclasses
    out_dir = Path(args.out_dir) if args.out_dir else C1_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    env = build_env_from_run(resolve_controller_dir(C0_DIR), 128, device)
    ds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_candidates.npz')
    ds, _ = validate_cached_dataset(ds, env.kin, env.collision,
                                    chunk_size=4096, cone_deg=env.cfg.cone_deg)
    nets, mu, sd = _load_selector(D / 'ikpool_selector_s0.pt', device)
    policy = EnsembleSeedPolicy(nets, mu, sd).to(device).eval()
    env.line_dist = SeedPolicyLineDistribution(
        ds, policy, env.kin,
        policy_prob=args.policy_prob,
        uniform_prob=(1.0 - args.policy_prob) * 0.75,
        fallback_prob=(1.0 - args.policy_prob) * 0.25,
        include_log_manip=True, include_ray_error=True,
        include_directional_dynamics=True,
        seed=FORWARD_SEED, seed_deployment=SeedDeploymentConfig())
    agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, device)
    cfg = ppo_config_from_run(load_run_config(resolve_controller_dir(C0_DIR)))
    cfg = dataclasses.replace(
        cfg, total_timesteps=args.steps, ent_coef=args.ent_coef,
        target_kl=args.target_kl, actor_warmup_updates=args.actor_warmup)

    def log_fn(stats):
        if 'update' in stats and (stats['update'] == 1 or stats['update'] % 20 == 0):
            print(f"[fwd] upd {stats['update']:>4} progress {stats.get('reward/progress', 0.0):.3f} "
                  f"kl {stats.get('train/approx_kl', 0.0):.4f}", flush=True)

    agent.train()
    torch.manual_seed(FORWARD_SEED + args.round)
    ppo_train(cfg, env, device, log_fn=log_fn, agent=agent)
    torch.save(agent.state_dict(), out_dir / 'agent.pt')
    shutil.copy(Path(C0_DIR) / 'config.yaml', out_dir / 'config.yaml')
    print(f'[forward] C saved -> {out_dir}', flush=True)


def stage_gate(args, device):
    c1 = args.c1_dir if args.c1_dir else str(C1_DIR)
    tag = Path(c1).name
    nets, mu, sd = _load_selector(D / 'ikpool_selector_s0.pt', device)
    rep = {'c1_dir': c1}
    for which in ('validation', 'external'):
        ds, X, P, V, tids = _load_pool(which, device)
        pick = _picks(nets, mu, sd, X, V)
        idx = torch.arange(len(P), device=device)
        s0c0 = P[idx, pick].cpu().numpy()          # cache lookup under C0
        s0c1 = _roll_indices(ds, pick, c1, device)  # rollout under the candidate C
        o = np.load(OLD[which], allow_pickle=True)
        order = {int(t): i for i, t in enumerate(o['task_indices'])}
        old_pol = np.nan_to_num(o['policy_progress_m'])[
            np.array([order[int(t)] for t in tids])]
        rep[which] = {
            'S0C0_mean_m': float(s0c0.mean()), 'S0C1_mean_m': float(s0c1.mean()),
            'forward_effect': _paired(s0c1, s0c0),
            'S0C1_vs_oldSOTA': _paired(s0c1, old_pol),
        }
        np.savez(D / f'gate_{which}_{tag}.npz', s0c0=s0c0, s0c1=s0c1,
                 pick=pick.cpu().numpy(), task_indices=tids)
        f = rep[which]['forward_effect']
        print(f"[gate {which}] S0C0={s0c0.mean():.4f} S0C1={s0c1.mean():.4f} "
              f"fwd {f['delta_mm']:+.1f}mm CI{f['ci95_mm']}", flush=True)
    (D / f'ikpool_gate_{tag}.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def stage_relabel(args, device):
    which, shard = args.set, args.shard
    name = 'ikpool_candidates.npz' if which == 'train' else f'ikpool_{which}_candidates.npz'
    out_name = (f'ikpool_returns_c1_shard{shard[0]}of{shard[1]}.npz'
                if which == 'train' else f'ikpool_{which}_returns_c1.npz')
    out = D / out_name
    if out.exists():
        print(f'[relabel] {out.name} exists, skip', flush=True); return
    c1 = args.c1_dir if args.c1_dir else str(C1_DIR)
    env = build_env_from_run(resolve_controller_dir(c1), ROLL_CHUNK, device)
    agent = load_controller_agent(resolve_controller_dir(c1), env, device).eval()
    gamma = float(ppo_config_from_run(load_run_config(resolve_controller_dir(c1))).gamma)
    ds = CachedSeedCandidateDataset.from_npz(D / name)
    n_all = len(ds)
    rows = np.array_split(np.arange(n_all), shard[1])[shard[0]] if which == 'train' \
        else np.arange(n_all)
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    sub = ds.batch.index_select(torch.arange(lo, hi))
    val = check_candidate_validity(env.kin, env.collision,
                                   sub.to(env.kin.device, dtype=env.kin.dtype),
                                   cone_deg=env.cfg.cone_deg).valid.cpu()
    R, K = hi - lo, ds.batch.n_candidates
    prog = np.full((R, K), np.nan, np.float32)
    pairs = torch.nonzero(val, as_tuple=False).long()
    ctl = FrozenRLController(agent)
    for s in range(0, pairs.shape[0], ROLL_CHUNK):
        p = pairs[s:s + ROLL_CHUNK]; nr = p.shape[0]
        if nr < ROLL_CHUNK:
            p = torch.cat([p, p[-1:].expand(ROLL_CHUNK - nr, -1)])
        cand = ds.batch.index_select(p[:, 0] + lo).to(device=device, dtype=env.kin.dtype)
        res = rollout_selected_seeds(env, cand, p[:, 1].to(device), ctl, gamma=gamma)
        pm = res.progress_m[:nr].cpu().numpy()
        for j in range(nr):
            prog[int(p[j, 0]), int(p[j, 1])] = pm[j]
        if (s // ROLL_CHUNK) % 20 == 0:
            print(f'[relabel {which} {shard[0]}/{shard[1]}] '
                  f'{min(s+ROLL_CHUNK, pairs.shape[0])}/{pairs.shape[0]}', flush=True)
    tids = ds.task_indices.numpy()[lo:hi].astype(np.int64)
    np.savez(out, progress_m=prog, valid=val.numpy(), task_indices=tids)
    print(f'[relabel] done -> {out.name}', flush=True)


def stage_merge_relabel(args, device):
    files = sorted(D.glob('ikpool_returns_c1_shard*.npz'),
                   key=lambda p: int(p.stem.split('shard')[1].split('of')[0]))
    data = {k: [] for k in ('progress_m', 'valid', 'task_indices')}
    for f in files:
        d = np.load(f)
        for k in data:
            data[k].append(d[k])
    merged = {k: np.concatenate(v, 0) for k, v in data.items()}
    np.savez(D / 'ikpool_returns_c1.npz', **merged)
    print(f'[merge] {len(files)} shards -> ikpool_returns_c1.npz '
          f'n={len(merged["task_indices"])}', flush=True)


def stage_s1(args, device):
    _, X, P, V, _ = _load_pool('train', device, returns_tag='_c1')
    _train_selector(X, P, V, device, D / 'ikpool_selector_s1.pt')


def stage_eval2x2(args, device):
    s0 = _load_selector(D / 'ikpool_selector_s0.pt', device)
    s1 = _load_selector(D / 'ikpool_selector_s1.pt', device)
    rep = {}
    for which in ('validation', 'external'):
        ds, X, P0, V, tids = _load_pool(which, device)          # C0 returns
        retc1 = np.load(D / f'ikpool_{which}_returns_c1.npz')
        P1 = torch.nan_to_num(torch.as_tensor(retc1['progress_m']), nan=0.0).to(device)
        V1 = torch.as_tensor(retc1['valid']).to(device)
        idx = torch.arange(len(P0), device=device)
        pick0 = _picks(*s0, X, V)
        pick1 = _picks(*s1, X, V)
        arms = {
            'S0C0': P0[idx, pick0].cpu().numpy(),
            'S0C1': P1[idx, pick0].cpu().numpy(),
            'S1C0': P0[idx, pick1].cpu().numpy(),
            'S1C1': P1[idx, pick1].cpu().numpy(),
        }
        o = np.load(OLD[which], allow_pickle=True)
        order = {int(t): i for i, t in enumerate(o['task_indices'])}
        perm = np.array([order[int(t)] for t in tids])
        old_pol = np.nan_to_num(o['policy_progress_m'])[perm]
        first1 = P1[idx, V1.float().argmax(1)].cpu().numpy()
        ora1 = torch.where(V1, P1, torch.tensor(-1e9, device=device)).max(1).values.cpu().numpy()
        rep[which] = {
            'means_m': {k: float(v.mean()) for k, v in arms.items()},
            'old_sota_m': float(old_pol.mean()),
            'effects': {
                'controller@S0 (S0C1-S0C0)': _paired(arms['S0C1'], arms['S0C0']),
                'selector@C1 (S1C1-S0C1)': _paired(arms['S1C1'], arms['S0C1']),
                'joint (S1C1-S0C0)': _paired(arms['S1C1'], arms['S0C0']),
                'S1C1_vs_oldSOTA': _paired(arms['S1C1'], old_pol),
            },
            'S1C1_capture_underC1_pct': float(
                (arms['S1C1'] - first1).sum() / (ora1 - first1).sum() * 100),
        }
        print(f'[2x2 {which}] ' + ' '.join(
            f'{k}={v.mean():.4f}' for k, v in arms.items()) +
            f' old={old_pol.mean():.4f}', flush=True)
    (D / 'ikpool_2x2.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('s0', 'forward', 'gate', 'relabel',
                                      'merge-relabel', 's1', 'eval2x2'))
    ap.add_argument('--set', default='train',
                    choices=('train', 'validation', 'external'))
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out-dir', default=None, help='forward: controller output dir')
    ap.add_argument('--steps', type=int, default=250000)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--target-kl', type=float, default=0.02)
    ap.add_argument('--actor-warmup', type=int, default=0)
    ap.add_argument('--policy-prob', type=float, default=0.7)
    ap.add_argument('--round', type=int, default=0)
    ap.add_argument('--c1-dir', default=None, help='gate/relabel: override C1 dir')
    args = ap.parse_args()
    i, n = args.shard.split('/')
    args.shard = (int(i), int(n))
    device = torch.device(args.device)
    globals()[f'stage_{args.stage.replace("-", "_")}'](args, device)


if __name__ == '__main__':
    main()
