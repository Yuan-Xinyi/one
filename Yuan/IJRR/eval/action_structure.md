# What the learned null-space command looks like

A closed-loop finding, stated only as far as the closed-loop measurements go.

## The three measurements that support it

All on 2048 straight-line tasks, identical start configurations, deterministic
evaluation.

| | median \|a\| | \|a\| > 0.9 | \|a\| > 0.99 |
|---|---|---|---|
| classical null-space law | 0.14 | 0.1% | 0.1% |
| learned policy | 1.00 | 89.2% | 80.2% |

1. The two resolutions occupy different regions of the command box: the
   classical law, being gradient ascent on a weighted objective, produces
   commands whose magnitude is that of the gradient and which essentially never
   touch the boundary; the learned policy is on the boundary most of the time.
2. Projecting the learned command onto the nearest vertex of `[-1,1]^4` costs
   nothing: +0.004 (policy alone) and +0.005 (hybrid) in the ratio to the
   classical law, both with bootstrap intervals spanning zero. The magnitude
   information the continuous parameterisation can express is not being used.
3. A policy whose action space *is* the 16-vertex set, trained from scratch
   with a categorical head, matches the continuous policy: 1.3235 against
   1.2977 on the serpentine family, interval spanning zero.

Together: **for this task the learned continuous policy behaves as a discrete
switching controller over the 16 extreme null-space directions.** What matters
is which extreme direction is selected as a function of state, not how far along
it the command is pushed.

## What is deliberately not claimed

That the optimal control is bang-bang. The behaviour above is qualitatively
consistent with the extremal-control structure that can arise in control-affine
problems with box-bounded controls, but establishing optimality for the
implemented discrete closed-loop system was not achieved and is out of scope.

Three claims were made during this investigation and withdrawn; they are
recorded so they are not made again:

* *"The saturation of the learned policy shows it matches the optimal bang-bang
  structure."* Not established. Saturation is a property of the learned policy.
* *"Free optimisation shows the optimum is interior."* The cross-entropy method
  is not a global optimiser, and the sequences it found at the default 50 ms
  integration step lose most of their advantage when the same commands are
  integrated at 25 ms — they exploited the discretisation rather than the
  dynamics. That whole line of evidence was discarded.
* *"J⁺(q) is nonlinear in q, so the Hamiltonian is not linear in the command"*,
  and *"σ crosses zero often, so singular arcs are common"*. Both are wrong: `f`
  and `G` may depend on `q` arbitrarily without affecting linearity in `a`, the
  costate already carries the future effect of the present command, and an
  isolated zero of the switching function is an ordinary switching instant, not
  a singular arc.

## Integration convergence

The control rate is 20 Hz throughout. Refining only the internal integration
step, with each command held across the substeps, changes the closed-loop
results by less than 2%:

| integration step | classical (m) | RL / classical | hybrid / classical |
|---|---|---|---|
| 50 ms | 0.3355 | 1.2900 | 1.3986 |
| 25 ms | 0.3338 | 1.2888 | 1.4114 |
| 12.5 ms | 0.3326 | 1.2880 | 1.4073 |

The ordering and the relative gaps are unchanged. Open-loop command sequences
are a different matter and are not converged at 50 ms, which is why no
open-loop planning result is reported.
