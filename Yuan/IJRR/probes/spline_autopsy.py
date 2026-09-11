"""Where do spline deaths happen? Curvature at the death point vs the
spline's own distribution; regenerate splines with the probe's seed."""
import sys
sys.path.insert(0, '/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
import matplotlib; matplotlib.use('Agg')
import numpy as np
exec(open('/tmp/claude-1000/-home-lqin-one-Yuan-IJRR--claude-worktrees-vigilant-hertz-799b05/5877612c-7b98-459c-a55a-ae5c52eb5b25/scratchpad/spline_probe.py').read().split("SPL = [")[0].replace("N_SP = int(sys.argv[1])", "N_SP = 100").replace("TAG = sys.argv[2] if len(sys.argv) > 2 else 'smoke'", "TAG='x'"))
SPL = [sample_spline(rng) for _ in range(100)]
d = np.load('/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/spline_probe_v1.npz')
prog, lpw, L = d['prog'], d['lpw'], d['length']
assert np.allclose(L, [s[2] for s in SPL], atol=1e-4), 'seed mismatch'
k_death, k_med, k_p90, rmin = [], [], [], []
died = 0
for i, (pts, tan, Li) in enumerate(SPL):
    dt_ = np.gradient(tan, 0.005, axis=0)
    kap = np.linalg.norm(dt_, axis=1)
    kap = np.convolve(kap, np.ones(9)/9, 'same')
    k_med.append(np.median(kap)); k_p90.append(np.percentile(kap, 90))
    rmin.append(1/max(kap.max(), 1e-9))
    if prog[i] < min(lpw[i], Li) - 0.05:      # actually died mid-path
        j = min(int(prog[i]/0.005), len(kap)-1)
        k_death.append(kap[max(0, j-10):j+10].max())
        died += 1
k_death, k_med, k_p90 = map(np.array, (k_death, k_med, k_p90))
print(f'died mid-path: {died}/100;  min bend radius across splines: '
      f'med {np.median(rmin):.3f} m')
print(f'curvature at death: med {np.median(k_death):.2f} (r={1/np.median(k_death):.3f}m)')
print(f'spline curvature overall: med {np.median(k_med):.2f}  p90 {np.median(k_p90):.2f}')
hi = np.median(k_p90)
print(f'deaths at top-decile-curvature points: {(k_death > hi).mean()*100:.0f}%')
# survival vs tightest bend on the path ahead
tight = np.array([1/max(np.linalg.norm(np.gradient(s[1],0.005,axis=0),axis=1).max(),1e-9) for s in SPL])
rt = prog/np.maximum(lpw, 1e-9)
lo_r = tight < np.median(tight)
print(f'ratio | tighter-bend half: {rt[lo_r].mean()*100:.1f}  gentler half: {rt[~lo_r].mean()*100:.1f}')
