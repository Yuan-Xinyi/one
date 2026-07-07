"""Final validation on a VIRGIN holdout: pilot tasks never selected into the
10k eval set (and never touched by any campaign stage — ranker trained on the
RL train pool, controller never saw pilot tasks).

Protocol mirrors the paper exactly:
  oracle'   : per-task max over SMM top-K' candidates rolled under the
              pi0 hybrid (config controller, tau 0.98/0.94) — frozen-oracle
              protocol replicated on fresh tasks.
  rows      : DP first-valid / ranked-25 (ranker_v4) / best-of-25 ceiling,
              all under the adopted controller (r12m @0.985/0.96);
              plus pi0-hybrid on the first-valid seed (paper Hybrid analog).
  robustness: q0 noise sigma in {0.02, 0.05} on the EXECUTED seed for
              first-valid vs ranked (is the picked seed more fragile?).
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch, yaml

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.system_eval.seed_sources import diffusion_seeds
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import Rank, obs_and_manip

OUT = Path('Yuan/seed_selection/runs/fresh_holdout_final')
OUT.mkdir(parents=True, exist_ok=True)
N_FRESH = 2048
CKPT_NEW = Path('Yuan/RL_controller/runs/exit_rounds7plus/final_avg')
TAU_NEW = (0.985, 0.96)
dev = torch.device('cuda')
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dc = cfg['diffusion']
CKPT_P0 = Path(cfg['rl_controller']['ckpt_dir'])
TAU_P0 = (float(cfg['rl_controller']['tau_enter']),
          float(cfg['rl_controller']['tau_exit']))

# ---- 1. build fresh subset ----
fs_npz = OUT / 'fresh_set_2k.npz'
if fs_npz.exists():
    fs = dict(np.load(fs_npz))
    print(f'[fresh] set cached: {len(fs["src_idx"])} tasks', flush=True)
else:
    pilot = np.load('Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz')
    pc = np.load('Yuan/seed_selection/runs/pilot_20k/pilot_20k.plane_collision.npz')
    used = set(np.load('Yuan/system_eval/runs/eval_10k_systematic/'
                       'eval_set_10k.npz')['src_idx'].tolist())
    status = pilot['status']
    kept = (status == 'kept') if status.dtype.kind == 'U' else status.astype(bool)
    safe = ~pc['any_label_collides'].astype(bool)
    L_seed = pilot['L_seed'].astype(np.float32)
    eligible = kept & safe & (L_seed >= 0.10)
    cand = np.array([i for i in np.where(eligible)[0] if i not in used])
    print(f'[fresh] eligible-and-unused: {len(cand)}', flush=True)
    rng = np.random.default_rng(777)
    pick = rng.choice(cand, size=min(N_FRESH, len(cand)), replace=False)
    pick.sort()
    fs = {
        'src_idx': pick,
        'cs_p0': pilot['cs_p0'][pick].astype(np.float32),
        'cs_line_dir': pilot['cs_line_dir'][pick].astype(np.float32),
        'cs_n_target': pilot['cs_n_target'][pick].astype(np.float32),
        'q0_seed': pilot['q0_seeds'][pick].astype(np.float32),
        'L_seed': L_seed[pick],
        'top_q': pilot['top_Kprime_q'][pick].astype(np.float32),
        'top_valid': pilot['top_Kprime_valid_mask'][pick].astype(bool),
    }
    np.savez_compressed(fs_npz, **fs)
    print(f'[fresh] set built: {len(pick)} tasks', flush=True)

N = len(fs['src_idx'])
p0s, lds, nts = fs['cs_p0'], fs['cs_line_dir'], fs['cs_n_target']

# ---- 2. oracle' under pi0 hybrid (frozen-oracle protocol) ----
orc_npz = OUT / 'oracle_prime.npz'
if orc_npz.exists():
    oh = np.load(orc_npz)['oh']
else:
    env = build_env(CKPT_P0 / 'config.yaml', 4096, dev)
    classical = ClassicalNullspaceController(env.kin)
    agent0 = load_rl_agent(CKPT_P0, env, dev)
    Kp = fs['top_q'].shape[1]
    # Roll ONLY valid (task, candidate) pairs — invalid slots are padding
    # garbage whose singular Jacobians crash eigvalsh (run_oracle_prime does
    # the same flattening).
    ti, si_ = np.where(fs['top_valid'])
    r = rollout_seeds_batched(
        fs['top_q'][ti, si_], p0s[ti], lds[ti], nts[ti], env=env,
        controller='hybrid_variantB', classical=classical, agent=agent0,
        tau_enter=TAU_P0[0], tau_exit=TAU_P0[1],
        progress_prefix='oracle-flat ')
    L_cand = np.full((N, Kp), -np.inf, np.float32)
    L_cand[ti, si_] = r['L']
    oh = L_cand.max(1) * 1.5
    oh = np.where(np.isfinite(oh), oh, 0.0)
    np.savez_compressed(orc_npz, oh=oh, L_cand=L_cand)
    del env, agent0
    torch.cuda.empty_cache()
    print(f'[fresh] oracle done (valid tasks: {(oh > 1e-6).sum()})', flush=True)

# ---- 3. DP candidates (fresh sampling seeds) ----
cand_npz = OUT / 'candidates25.npz'
env = build_env(CKPT_NEW / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT_NEW, env, dev)
if cand_npz.exists():
    d = np.load(cand_npz)
    seeds, ok = d['seeds'], d['ok']
else:
    fs_like = {'cs_p0': p0s, 'cs_line_dir': lds, 'cs_n_target': nts}
    s15, ok15 = diffusion_seeds(fs_like, dc['ckpt'], n_samples=16,
                                ddim_steps=50, cfg_w=1.5, sample_seed=13000,
                                kin=env.kin, device=dev, verbose=False)
    s10, ok10 = diffusion_seeds(fs_like, dc['ckpt'], n_samples=8,
                                ddim_steps=50, cfg_w=1.0, sample_seed=13001,
                                kin=env.kin, device=dev, verbose=False)
    seeds = np.concatenate([s15, s10, fs['q0_seed'][:, None, :]], 1)
    ok = np.concatenate([ok15, ok10, np.ones((N, 1), bool)], 1)
    np.savez_compressed(cand_npz, seeds=seeds, ok=ok)
    print(f'[fresh] candidates: IK ok {100*ok[:, :24].mean():.1f}%', flush=True)

# ---- 4. ranker pick ----
ck = torch.load('Yuan/seed_selection/runs/rank_train/ranker_final.pt',
                map_location=dev, weights_only=False)
nets = []
for sd in ck['nets']:
    n = Rank(32).to(dev); n.load_state_dict(sd); n.eval(); nets.append(n)
obs = np.zeros((N, 25, 31), np.float32)
mu = np.zeros((N, 25), np.float32)
for si in range(25):
    obs[:, si], mu[:, si] = obs_and_manip(env, seeds[:, si], p0s, lds, nts)
X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1)
Xn = torch.from_numpy((X - ck['mean']) / ck['std']).float()
sc = np.zeros((N, 25), np.float32)
with torch.no_grad():
    flat = Xn.reshape(-1, 32)
    for n_i in nets:
        out = [n_i(flat[s:s + 65536].to(dev)).cpu()
               for s in range(0, len(flat), 65536)]
        sc += torch.cat(out).view(N, 25).numpy()
pick = np.where(ok, sc, -np.inf).argmax(1)
q_ranked = seeds[np.arange(N), pick]
first_idx = np.argmax(ok[:, :16], 1)     # first IK-valid among w1.5 (deploy order)
has_w15 = ok[:, :16].any(1)
q_first = np.where(has_w15[:, None],
                   seeds[np.arange(N), first_idx], fs['q0_seed'])

# ---- 5. rollout rows ----
def roll(qs, agent_, tau, tag):
    f = OUT / f'L_{tag}.npz'
    if f.exists():
        return np.load(f)['L'] * 1.5
    r = rollout_seeds_batched(qs.astype(np.float32), p0s, lds, nts, env=env,
                              controller='hybrid_variantB', classical=classical,
                              agent=agent_, tau_enter=tau[0], tau_exit=tau[1],
                              progress_prefix=f'{tag} ')
    np.savez_compressed(f, L=r['L'], term=r['term_reason'])
    return r['L'] * 1.5

L_first = roll(q_first, agent, TAU_NEW, 'first_valid')
L_rank = roll(q_ranked, agent, TAU_NEW, 'ranked')
L_slots = np.stack([roll(seeds[:, si], agent, TAU_NEW, f'slot{si}')
                    for si in range(25)], 1)
Lbest = np.where(ok, L_slots, -np.inf).max(1)
agent_p0 = load_rl_agent(CKPT_P0, env, dev)
L_paper = roll(q_first, agent_p0, TAU_P0, 'p0hyb_firstvalid')

fin = oh > 1e-6
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()

print('\n==== FRESH HOLDOUT (paper pct of oracle_prime, virgin tasks) ====')
print(f'  pi0-hybrid + first-valid (paper analog)  {pct(L_paper):.2f}%   [10k: 90.6]')
print(f'  new ctrl  + first-valid                  {pct(L_first):.2f}%   [10k: 91.41]')
print(f'  new ctrl  + ranked-25                    {pct(L_rank):.2f}%   [10k: 98.38]')
print(f'  new ctrl  + best-of-25 (ceiling)         {pct(Lbest):.2f}%   [10k: 103.22]')
d = (L_rank[fin] - L_first[fin]) / oh[fin] * 100
se = d.std(ddof=1) / np.sqrt(len(d))
print(f'  ranked - first: {d.mean():+.2f}pp ± {se:.2f}')

# ---- 6. robustness: noise on the executed seed ----
rng = np.random.default_rng(4242)
print('\n==== ROBUSTNESS (q0 noise on executed seed) ====')
for sig in (0.02, 0.05):
    noise = rng.normal(0, sig, size=(N, 7)).astype(np.float32)
    Lf_n = roll(q_first + noise, agent, TAU_NEW, f'first_noise{sig}')
    Lr_n = roll(q_ranked + noise, agent, TAU_NEW, f'ranked_noise{sig}')
    print(f'  sigma={sig}: first {pct(Lf_n):.2f}% (drop {pct(L_first)-pct(Lf_n):.2f})'
          f'   ranked {pct(Lr_n):.2f}% (drop {pct(L_rank)-pct(Lr_n):.2f})')
np.savez_compressed(OUT / 'summary.npz', oh=oh, L_first=L_first,
                    L_rank=L_rank, L_best=Lbest, L_paper=L_paper, pick=pick)
print('[fresh] done', flush=True)
