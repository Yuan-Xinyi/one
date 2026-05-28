# System eval — 5+1 cell ablation (N=8953)

All progress values in **meters** (raw EE displacement along the task line). Conservative variant: best-of-N over diffusion samples, all-IK-fail tasks counted as progress=0. Realistic variant: all-IK-fail tasks fall back to the q0_seed rollout (Cell A). Oracle used for recovery ratios = **E_prime** (controller-aware).

| cell | source |
|---|---|
| A | A: baseline (q0_seed + Classical) |
| B | B: seed ablation (Diffusion + Classical) |
| C | C: controller ablation (q0_seed + RL hybrid) |
| D | D: full method (Diffusion + RL hybrid) |
| E | E: seed-oracle (max-L_classical label + RL hybrid) |
| E_prime | E': controller-aware oracle (max over SMM top-K' + RL hybrid) |

## Overall (ALL tasks) — CONSERVATIVE (IK fail = 0)

| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 8953 | 0.3200 | 0.2500 | 0.4301 | 55.8 | 14.3 | 2.1 | 18.2 | 0.0000 |
| B | conservati | 8953 | 0.4597 | 0.3633 | 0.5901 | 95.3 | 40.2 | 10.5 | 43.9 | 0.0987 |
| C | conservati | 8953 | 0.4401 | 0.2987 | 0.6290 | 74.1 | 41.1 | 14.3 | 47.4 | 0.0600 |
| D | conservati | 8953 | 0.6199 | 0.4475 | 0.8284 | 96.5 | 67.5 | 32.5 | 95.0 | 0.2096 |
| E | conservati | 8953 | 0.5388 | 0.3798 | 0.7300 | 89.5 | 55.5 | 22.8 | 79.0 | 0.1449 |
| E_prime | conservati | 8953 | 0.6096 | 0.4286 | 0.8198 | 93.6 | 65.1 | 31.9 | 100.0 | 0.1992 |

## Overall (ALL tasks) — REALISTIC (IK fail → q0_seed fallback)

| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 8953 | 0.3200 | 0.2500 | 0.4301 | 55.8 | 14.3 | 2.1 | 18.2 | 0.0000 |
| B | realistic_ | 8953 | 0.4499 | 0.3516 | 0.5819 | 91.5 | 38.5 | 10.0 | 45.8 | 0.0898 |
| C | conservati | 8953 | 0.4401 | 0.2987 | 0.6290 | 74.1 | 41.1 | 14.3 | 47.4 | 0.0600 |
| D | realistic_ | 8953 | 0.6003 | 0.4201 | 0.8100 | 92.6 | 64.3 | 30.9 | 94.1 | 0.1901 |
| E | conservati | 8953 | 0.5388 | 0.3798 | 0.7300 | 89.5 | 55.5 | 22.8 | 79.0 | 0.1449 |
| E_prime | conservati | 8953 | 0.6096 | 0.4286 | 0.8198 | 93.6 | 65.1 | 31.9 | 100.0 | 0.1992 |

## Per-bucket break-down (conservative variant)

### Bucket: weak
| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 1453 | 0.1999 | 0.1799 | 0.2100 | 0.0 | 0.0 | 0.0 | 19.0 | 0.0000 |
| B | conservati | 1453 | 0.3596 | 0.3099 | 0.4802 | 81.1 | 22.6 | 6.5 | 42.4 | 0.1600 |
| C | conservati | 1453 | 0.2199 | 0.1800 | 0.2959 | 23.9 | 11.4 | 5.0 | 33.1 | 0.0195 |
| D | conservati | 1453 | 0.4999 | 0.3400 | 0.8275 | 85.4 | 49.7 | 29.4 | 91.4 | 0.3005 |
| E | conservati | 1453 | 0.3587 | 0.2299 | 0.6292 | 65.7 | 33.8 | 16.5 | 77.6 | 0.1597 |
| E_prime | conservati | 1453 | 0.4200 | 0.2798 | 0.7587 | 73.0 | 42.6 | 25.3 | 100.0 | 0.2294 |

### Bucket: medium-weak
| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 2500 | 0.2698 | 0.2498 | 0.2897 | 0.0 | 0.0 | 0.0 | 7.2 | 0.0000 |
| B | conservati | 2500 | 0.3801 | 0.3301 | 0.4708 | 94.2 | 21.3 | 6.5 | 36.3 | 0.1197 |
| C | conservati | 2500 | 0.3314 | 0.2700 | 0.4799 | 61.0 | 22.8 | 10.0 | 32.3 | 0.0603 |
| D | conservati | 2500 | 0.5585 | 0.3890 | 0.8628 | 96.2 | 56.8 | 31.8 | 93.0 | 0.2856 |
| E | conservati | 2500 | 0.4498 | 0.3402 | 0.6998 | 87.6 | 42.0 | 21.6 | 74.5 | 0.1797 |
| E_prime | conservati | 2500 | 0.5398 | 0.3796 | 0.8608 | 93.8 | 54.8 | 31.9 | 100.0 | 0.2701 |

### Bucket: medium
| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 3000 | 0.3601 | 0.3300 | 0.3999 | 100.0 | 0.0 | 0.0 | 17.7 | 0.0000 |
| B | conservati | 3000 | 0.4498 | 0.3898 | 0.5400 | 99.1 | 33.1 | 8.1 | 43.5 | 0.0705 |
| C | conservati | 3000 | 0.4600 | 0.3701 | 0.6089 | 93.6 | 41.4 | 13.3 | 50.8 | 0.0887 |
| D | conservati | 3000 | 0.5894 | 0.4483 | 0.7896 | 99.2 | 65.6 | 29.0 | 96.1 | 0.2105 |
| E | conservati | 3000 | 0.5297 | 0.4123 | 0.7086 | 96.7 | 55.4 | 20.7 | 80.0 | 0.1529 |
| E_prime | conservati | 3000 | 0.5899 | 0.4477 | 0.7902 | 99.4 | 65.2 | 29.1 | 100.0 | 0.2148 |

### Bucket: strong
| cell | variant | n | prog_med (m) | p25 | p75 | ≥0.3m | ≥0.5m | ≥0.75m | recov_vs_oracle | gain_med (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | conservati | 2000 | 0.5300 | 0.4801 | 0.6291 | 100.0 | 64.0 | 9.6 | 31.8 | 0.0000 |
| B | conservati | 2000 | 0.6107 | 0.5300 | 0.7299 | 99.8 | 85.0 | 21.4 | 54.9 | 0.0396 |
| C | conservati | 2000 | 0.6400 | 0.5403 | 0.7701 | 98.0 | 85.0 | 27.9 | 71.7 | 0.0698 |
| D | conservati | 2000 | 0.7086 | 0.6038 | 0.8393 | 99.8 | 94.4 | 40.8 | 97.9 | 0.1305 |
| E | conservati | 2000 | 0.6693 | 0.5607 | 0.7987 | 98.6 | 88.0 | 32.0 | 84.4 | 0.0931 |
| E_prime | conservati | 2000 | 0.7097 | 0.6080 | 0.8395 | 99.7 | 94.2 | 40.8 | 100.0 | 0.1311 |

## Diffusion-specific (IK convergence) — cells B, D

| cell | bucket | ik_rate | mean_n_ok | all_fail_rate |
|---|---|---|---|---|
| B | weak | 76.6 | 6.1287 | 16.8 |
| B | medium-weak | 88.8 | 7.1072 | 5.0 |
| B | medium | 92.3 | 7.3840 | 2.0 |
| B | strong | 92.6 | 7.4060 | 2.5 |
| B | ALL | 88.8 | 7.1079 | 5.4 |
| D | weak | 76.6 | 6.1287 | 16.8 |
| D | medium-weak | 88.8 | 7.1072 | 5.0 |
| D | medium | 92.3 | 7.3840 | 2.0 |
| D | strong | 92.6 | 7.4060 | 2.5 |
| D | ALL | 88.8 | 7.1079 | 5.4 |

## Ablation decomposition (median progress in meters, conservative)

| bucket | n | A (m) | Δ_B (m) | Δ_C (m) | Δ_D (m) | (Δ_B+Δ_C) | synergy |
|---|---|---|---|---|---|---|---|
| weak | 1453 | 0.200 | +0.160 | +0.020 | +0.300 | +0.180 | **+0.121** |
| medium-weak | 2500 | 0.270 | +0.120 | +0.060 | +0.286 | +0.180 | **+0.106** |
| medium | 3000 | 0.360 | +0.070 | +0.089 | +0.210 | +0.159 | **+0.051** |
| strong | 2000 | 0.530 | +0.040 | +0.070 | +0.130 | +0.109 | **+0.021** |

## D vs E vs E' (controller-aware oracle comparison)

**D vs E** (E = SMM-classical seed oracle, controller-mismatched):
- D > E on **83.9%** of tasks; median(D−E) = **+8.7 mm**

**D vs E'** (E' = controller-aware oracle over SMM top-K' under hybrid):
- D > E' on **50.9%** of tasks; median(D−E') = **+0.0 mm**

**E' vs E**:
- E' > E on **69.4%** of tasks; median(E'−E) = **+3.6 mm** (this measures how much the SMM seed-oracle leaves on the table by being controller-mismatched).
