#!/bin/bash
# Test 3: single-channel ablation. 4× 1M-step PPO smoke runs, each isolating
# the contribution of one shaping channel on top of progress.
#
#   P0  wp=1.0  wjl=0.0  wcone=0.0  wdm=0.0   (progress only — baseline)
#   P1  wp=0.7  wjl=0.3  wcone=0.0  wdm=0.0   (+ jl)
#   P2  wp=0.7  wjl=0.0  wcone=0.3  wdm=0.0   (+ cone)
#   P3  wp=0.7  wjl=0.0  wcone=0.0  wdm=0.3   (+ dm)
#
# Runs sequentially. At the end, prints final eval/mean_progress_m for each.
#
# Usage:
#   bash Yuan/RL_controller/tests/run_channel_ablation.sh [OUT_BASE]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../../.." && pwd)"
cd "$REPO_ROOT"

OUT_BASE="${1:-Yuan/RL_controller/runs/ablation_$(date +%m%d_%H%M)}"
mkdir -p "$OUT_BASE"
echo "[ablation] writing to $OUT_BASE"

# Order: name wp wjl wcone wdm
SPECS=(
    "P0 1.0 0.0 0.0 0.0"
    "P1 0.7 0.3 0.0 0.0"
    "P2 0.7 0.0 0.3 0.0"
    "P3 0.7 0.0 0.0 0.3"
)

for spec in "${SPECS[@]}"; do
    read -r name wp wjl wcone wdm <<< "$spec"
    out="$OUT_BASE/$name"
    mkdir -p "$out"
    cfg="$out/config.yaml"
    # Generate config: base + weight overrides + 1M total_timesteps
    python -c "
import yaml
with open('Yuan/RL_controller/config.yaml') as f:
    y = yaml.safe_load(f)
y['env']['w_progress'] = float('$wp')
y['env']['w_jl']       = float('$wjl')
y['env']['w_cone']     = float('$wcone')
y['env']['w_dm']       = float('$wdm')
y['ppo']['total_timesteps'] = 1_000_000
# Faster eval cadence so we get more eval points within 1M steps.
y['train']['eval_every'] = 50_000
with open('$cfg','w') as f:
    yaml.safe_dump(y, f, sort_keys=False)
"
    echo ""
    echo "=== [$name] wp=$wp wjl=$wjl wcone=$wcone wdm=$wdm → $out ==="
    python -m Yuan.RL_controller.train \
        --config "$cfg" --out-dir "$out" \
        > "$out/stdout.log" 2>&1
    echo "  [$name] done. tail of stdout:"
    tail -3 "$out/stdout.log"
done

echo ""
echo "=== ablation summary (final eval/mean_progress_m per run) ==="
printf "%-4s  %-50s\n" "run" "final-eval line"
for name in P0 P1 P2 P3; do
    line=$(grep "eval @" "$OUT_BASE/$name/stdout.log" | tail -1 || echo "<no eval found>")
    printf "%-4s  %s\n" "$name" "$line"
done
echo ""
echo "[ablation] all done. logs under $OUT_BASE"
