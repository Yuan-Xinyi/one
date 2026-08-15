#!/bin/bash
# Sync Yuan/IJRR/2026_Yuan_RAL <-> github.com:Yuan-Xinyi/emg-paper (Overleaf bridge).
# push: send local paper commits to GitHub (Overleaf then pulls via its UI)
# pull: fetch Overleaf-side edits (after pressing Push in Overleaf's GitHub menu)
set -e
REMOTE=git@github.com:Yuan-Xinyi/emg-paper.git
PREFIX=Yuan/IJRR/2026_Yuan_RAL
cd "$(git rev-parse --show-toplevel)"
case "${1:-push}" in
  push)
    git subtree split --prefix=$PREFIX -b paper-only >/dev/null
    git push $REMOTE paper-only:main
    git branch -D paper-only >/dev/null
    echo "pushed. In Overleaf: Menu -> GitHub -> Pull GitHub changes."
    ;;
  pull)
    git subtree pull --prefix=$PREFIX $REMOTE main -m "Merge Overleaf-side edits"
    echo "pulled Overleaf edits into $PREFIX."
    ;;
  *) echo "usage: $0 [push|pull]"; exit 1 ;;
esac
