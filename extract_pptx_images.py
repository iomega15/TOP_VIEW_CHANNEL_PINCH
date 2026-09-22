"""Extract every picture from a .pptx into a folder and write a manifest CSV.

Each picture is matched to the nearest caption text box below it on the same
slide, and the caption is parsed for H (layers) and W (DLP pixels). A small
override table fixes the few captions that are shared or mislabeled in the
deck, and records the scale-bar length printed by the microscope.

Usage:
    python extract_pptx_images.py "Week 1.pptx" images
"""
import csv
import os
import re
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

EMU_PER_IN = 914400.0

# Manual fixes for the Week 1 deck: (slide, shape name) -> (caption, H, W, view, scalebar_mm)
OVERRIDES = {
    (3, "Picture 6"): ("Fig.4 H=5, W=50 px", 5, 50, "top", 0.20),
    (3, "Content Placeholder 5"): ("Fig.4 H=5, W=50 px", 5, 50, "top", 0.20),
    (3, "Picture 13"): ("Fig.4 H=5, W=50 px, Layer 6 (side view, 87 deg tilt)", 5, 50, "side", 0.40),
    (3, "Picture 14"): ("Fig.5 H=5, W=150 px", 5, 150, "top", 0.20),
    (3, "Picture 15"): ("Fig.5 H=5, W=150 px", 5, 150, "top", 0.20),
    (3, "Picture 16"): ("Fig.5 H=5, W=150 px", 5, 150, "top", 0.20),
    (3, "Picture 17"): ("Fig.5 H=5, W=150 px", 5, 150, "top", 0.20),
    (4, "Picture 15"): ("Fig.8 H=10, W=130 px, L=4", 10, 130, "top", 2.00),
}
SCALEBAR_BY_SLIDE = {2: 0.20, 3: 0.20, 4: 2.00}


def iter_pictures(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_pictures(sh.shapes)
        elif hasattr(sh, "image"):
            yield sh


def iter_texts(shapes):
    for sh in shapes:
        if sh.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from iter_texts(sh.shapes)
        elif sh.has_text_frame and sh.text_frame.text.strip():
            yield sh


def nearest_caption(pic, texts):
    px0, px1 = pic.left, pic.left + pic.width
    pbot = pic.top + pic.height
    best, best_d = None, None
    for t in texts:
        if re.match(r"\s*Fig", t.text_frame.text) is None:
            continue
        tx = t.left + t.width / 2
        dy = t.top - pbot
        if dy < -0.6 * EMU_PER_IN:
            continue
        dx = 0 if px0 <= tx <= px1 else min(abs(tx - px0), abs(tx - px1))
        d = abs(dy) + 2 * dx
        if best is None or d < best_d:
            best, best_d = t, d
    return best.text_frame.text.strip() if best else ""


def parse_hw(caption):
    h = re.search(r"H\s*=\s*(\d+)", caption)
    w = re.search(r"W\s*=\s*(\d+)", caption)
    return (int(h.group(1)) if h else None, int(w.group(1)) if w else None)


def main(pptx_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prs = Presentation(pptx_path)
    rows = []
    for si, slide in enumerate(prs.slides, 1):
        texts = list(iter_texts(slide.shapes))
        for pi, pic in enumerate(iter_pictures(slide.shapes), 1):
            cap = nearest_caption(pic, texts)
            H, W = parse_hw(cap)
            view = "top"
            sb = SCALEBAR_BY_SLIDE.get(si, None)
            key = (si, pic.name)
            if key in OVERRIDES:
                cap, H, W, view, sb = OVERRIDES[key]
            ext = pic.image.ext
            tag = f"H{H}_W{W}" if H is not None and W is not None else "unlabeled"
            fname = f"s{si:02d}_p{pi:02d}_{tag}_{view}.{ext}"
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(pic.image.blob)
            rows.append(dict(file=fname, slide=si, shape=pic.name, caption=cap,
                             H_layers=H, W_px=W, view=view, scalebar_mm=sb,
                             left_in=round(pic.left / EMU_PER_IN, 2),
                             top_in=round(pic.top / EMU_PER_IN, 2)))
            print(f"{fname:40s} <- slide {si} {pic.name!r:24s} {cap}")
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n{len(rows)} images -> {out_dir}/manifest.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Week 1.pptx",
         sys.argv[2] if len(sys.argv) > 2 else "images")
