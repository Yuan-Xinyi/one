#!/bin/bash
# Sync Yuan/IJRR/2026_Yuan_RAL <-> github.com:Yuan-Xinyi/emg-paper (Overleaf bridge).
# push: send local paper commits to GitHub (Overleaf then pulls via its UI).
#       If the remote has Overleaf-side commits, wrap the split head in a
#       merge commit so the push stays fast-forward (never force-push).
# pull: import the remote's current paper state into the monorepo
#       (archive checkout + commit; safe when local paper dir is clean).
set -e
REMOTE=git@github.com:Yuan-Xinyi/emg-paper.git
PREFIX=Yuan/IJRR/2026_Yuan_RAL
cd "$(git rev-parse --show-toplevel)"
case "${1:-push}" in
  push)
    git fetch "$REMOTE" main
    git subtree split --prefix=$PREFIX -b paper-only >/dev/null
    if git merge-base --is-ancestor FETCH_HEAD paper-only 2>/dev/null; then
      git push "$REMOTE" paper-only:main
    else
      M=$(git commit-tree 'paper-only^{tree}' -p paper-only -p FETCH_HEAD \
          -m "Merge Overleaf-side edits into the monorepo line")
      git push "$REMOTE" "$M":main
    fi
    git branch -D paper-only >/dev/null
    echo "pushed. In Overleaf: Menu -> GitHub -> Pull GitHub changes."
    ;;
  pull)
    git fetch "$REMOTE" main
    git archive FETCH_HEAD | tar -x -C $PREFIX
    git add $PREFIX
    git commit -m "Import Overleaf-side paper state" || echo "already up to date"
    ;;
  *) echo "usage: $0 [push|pull]"; exit 1 ;;
esac
