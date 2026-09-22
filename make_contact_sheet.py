"""Contact sheet of the device panels of every overlay in a results folder.

Usage:  python make_contact_sheet.py results_H5 [columns]
Writes  <results>/contact_sheet.png (and _partN.png if more than 24 devices).
"""
import glob
import os
import sys

import cv2
import numpy as np


def main(res_dir, cols=2, per_sheet=24):
    files = sorted(glob.glob(os.path.join(res_dir, "*_overlay.png")))
    if not files:
        print("no overlays in", res_dir)
        return
    tiles = []
    for f in files:
        im = cv2.imread(f)
        top = im[: int(0.46 * im.shape[0])]          # device + tracks panel only
        top = cv2.resize(top, (1100, int(1100 * top.shape[0] / top.shape[1])))
        tiles.append(top)
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    parts = [tiles[i:i + per_sheet] for i in range(0, len(tiles), per_sheet)]
    for n, part in enumerate(parts, 1):
        while len(part) % cols:
            part.append(np.full_like(part[0], 255))
        rows = [np.hstack(part[i:i + cols]) for i in range(0, len(part), cols)]
        out = os.path.join(res_dir, "contact_sheet.png" if len(parts) == 1 else f"contact_sheet_part{n}.png")
        cv2.imwrite(out, np.vstack(rows))
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results", int(sys.argv[2]) if len(sys.argv) > 2 else 2)
