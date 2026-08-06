"""Render one task at several curvatures and compose a labelled contact sheet.

Each panel is produced by a separate `viz_curvature` process (one pyglet window
per process), all with the same camera framing, so the panels differ only in
the curvature of the path.

    python -m Yuan.system_eval.viz_curvature_sheet \
        --kappas 0 1 4 --out Yuan/system_eval/runs/curvature_scan/sheet.png
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]


def crop_white(img: Image.Image, pad: int = 12) -> Image.Image:
    a = np.asarray(img.convert("RGB"))
    mask = (a < 246).any(axis=2)
    if not mask.any():
        return img
    ys, xs = np.nonzero(mask)
    return img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                     min(a.shape[1], xs.max() + pad),
                     min(a.shape[0], ys.max() + pad)))


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", nargs="+", type=float, default=[0.0, 1.0, 4.0])
    ap.add_argument("--fix-span", type=float, default=0.62)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--task-seed", type=int, default=7)
    ap.add_argument("--panel-h", type=int, default=760)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=None, help="dir to keep panel PNGs in")
    args, extra = ap.parse_known_args()

    tmp = Path(args.keep) if args.keep else Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    panels, captions = [], []
    for k in args.kappas:
        png = tmp / f"panel_k{k:+.2f}.png"
        cmd = [sys.executable, "-m", "Yuan.system_eval.viz_curvature",
               "--kappa", str(k), "--stride", str(args.stride),
               "--fix-span", str(args.fix_span),
               "--task-seed", str(args.task_seed),
               "--save", str(png)] + extra
        out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if "[viz] task" in l), "")
        if not png.exists():
            print(out.stdout[-2000:], out.stderr[-2000:])
            raise SystemExit(f"panel for kappa={k} was not written")
        m = re.search(r"R=([\d.]+|inf) m\)\s+arc=([\d.]+) m\s+steps=(\d+)"
                      r"\s+stop=(\w+)", line)
        if m is None:
            raise SystemExit(f"could not parse viz output for kappa={k}: {line!r}")
        R, arc, steps, stop = m.groups()
        R_txt = "straight" if k == 0 else f"R = {float(R):.2f} m"
        captions.append((f"kappa = {k:+.2f} 1/m   ({R_txt})",
                         f"arc length {arc} m   |   {steps} steps   |   stops on: {stop}"))
        panels.append(crop_white(Image.open(png)))
        print(line)

    H = args.panel_h
    panels = [p.resize((int(p.width * H / p.height), H), Image.LANCZOS)
              for p in panels]
    f_title, f_sub, f_leg = font(30), font(23), font(21)
    gap, top, bot = 26, 78, 116
    W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    sheet = Image.new("RGB", (W, H + top + bot), "white")
    dr = ImageDraw.Draw(sheet)

    x = gap
    for p, (t, s) in zip(panels, captions):
        sheet.paste(p, (x, top))
        dr.text((x + p.width // 2, 16), t, fill=(20, 20, 20), font=f_title,
                anchor="ma")
        dr.text((x + p.width // 2, 50), s, fill=(90, 90, 90), font=f_sub,
                anchor="ma")
        x += p.width + gap

    legend = [
        ((0.35, 0.55, 0.95), "nominal path (dashed = the unbounded continuation "
                             "the robot never got to)"),
        ((0.05, 0.55, 0.15), "realised TCP trace (inside the blue tube: on-path "
                             "to under 1 mm)"),
        ((0.85, 0.20, 0.75), "instantaneous tangent — the only path information "
                             "in the 31-D observation"),
        ((1.00, 0.65, 0.10), "30 deg tool-orientation cone about n_target; the "
                             "plane is the workpiece surface"),
    ]
    y = H + top + 8
    for i, (rgb, txt) in enumerate(legend):
        cx = gap + (i % 2) * (W // 2)
        cy = y + (i // 2) * 28
        col = tuple(int(255 * c) for c in rgb)
        dr.rectangle([cx, cy + 6, cx + 26, cy + 16], fill=col)
        dr.text((cx + 36, cy + 1), txt, fill=(55, 55, 55), font=f_leg)

    out = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[sheet] saved -> {out}")


if __name__ == "__main__":
    main()
