"""Top-view channel-formation classifier for 3D-printed printability devices (v2).

Input: top-view micrographs of a printability device (one design width W per
device in printer pixels, several parallel channels running from the top face
to the bottom face). The design width and layer count come from the file name
(`..._H5_W150_3.jpg`) or from `manifest.csv`; the microscope scale bar gives
the image scale; the printer pixel pitch (Asiga Ultra, 32 um) turns W into an
expected lumen width in image pixels, which anchors port detection, the cell
lattice and all width thresholds.

Per channel the output is a class and a set of measurements:
    FORMED      open from port to port, narrowest point >= neck_frac (75%) of the reference width
    NECKED      open from port to port but narrowed below 75% somewhere ("partial pinch");
                necked_len_frac = fraction of the length below 75%
    PINCHED     closed by cured resin between the side walls (narrow designs, or tapering tips);
                closed_len_frac = fraction of the length that is closed (point pinch -> ~0.02,
                half the channel -> ~0.5). blind = True when the lumen comes out of one
                port only and ends inside the device.
    TOUCHDOWN   the channel formed but the roof membrane rests on the floor (wide designs, or
                blunt lumen ends), often around a trapped bubble; roof_down_frac = fraction of
                the length where the roof is down (1.0 when no lumen is visible from above)
    NOT_FORMED  nothing beyond the port wedges (narrow designs)
    The regime (plug / touchdown / shape) comes from the design width, see P['plug_w_px'].
The reference width is the device's PRINTED width: the median over its tracked
channels of the 85th percentile of their open width.
plus touchdown_frac: fraction of the lumen area enclosed by an inner closed
outline (the roof membrane resting on the floor, 100% sag), min_width_frac and
its position, and a per-image debug overlay.

Pipeline:
  1. footer strip + scale bar (px); scale = bar_mm / bar_px; w_exp = W * 32 um / scale
  2. device detection on a local-texture map (robust to the illumination gradient)
  3. flat field; dark-line (vertical), dark-arc (horizontal), dark-blob maps
  4. ports = short dark components at the faces (crescents, wedges), filtered and
     split against w_exp; thin arcs as a paired fallback
  5. pairing across faces -> channels, pitch, lattice completion; walls = thin tall
     lines outside every cell
  6. two-terminal dynamic-programming tracking of the left and right lumen edge
  7. presence = continuity of the tracked line (hit fraction), smooth width, anchored
     to a port
  8. device-level reference width, debris-tolerant width smoothing, classification,
     touchdown detection

Usage:
    python topview_channel_pipeline.py --images images_student/H5 --out results_H5
    python topview_channel_pipeline.py --images images --out results_pptx --only s02_p03
    python topview_channel_pipeline.py --images images_student/H10 --out results_H10 --bar-mm 2.0 --pixel-um 32
"""
import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# parameters (lengths are fractions of the device height unless stated)
# ----------------------------------------------------------------------------
P = dict(
    # scale
    pixel_um=32.0,           # Asiga Ultra XY pixel pitch (PRINTER_PIXEL_SIZES[1] in the OpenSCAD)
    wall_mm=2.0,             # printability device: solid wall between channels; pitch = wall + W (measured on 7 devices, within 6%)
    pitch_tol=0.25,          # measured pitches are considered only within this fraction of the design pitch (wall 1.65-2.3 mm across devices)
    design_post_px=40,       # printability_device.scad: a post is 40 x 40 printer px; the solid between two channels is two posts side by side
    design_rows=4,           # printability_device.scad: a channel runs over 4 cells of two posts each -> design length = 4 x 2 x 40 px
    bar_mm=2.0,              # scale-bar length; the "0.20 mm" label on the H5 images is a 10x calibration slip
    um_per_px_default=13.5,  # used only when no scale bar is found (typical for the 1920 px frames)
    # Scale check against the device length (2026-11-19). The bar is read as bar_mm, but the frames
    # imaged in October 2026 carry 2.00 mm and 4.00 mm bars, and even read with the right label they
    # put the device at 1.3-1.4 x its design length (September frames: 0.89-1.12). q = device crop
    # height x bar scale / design length; outside [scale_q_lo, scale_q_hi] the bar does not describe
    # the frame and the scale is set so that q = scale_q_ref (the September median).
    scale_q_lo=0.85,
    scale_q_hi=1.15,
    scale_q_ref=0.985,
    scale_min_crop=0.6,      # ... but only when the device crop spans at least this fraction of the frame height (a short crop is a detection failure, not a ruler)
    # footer / device
    footer_dark=25,
    dev_tex_factor=3.0,
    # maps
    flat_sigma=0.12,
    line_se_frac=0.012, line_se_min=7, line_se_max=15,
    line_scale=0.30,         # a line 30% darker than the local mean -> response 1.0
    blob_dark=0.70,
    # ports
    port_band=0.16,
    face_skip=0.006,
    border_skip=0.03,
    crescent_dark=0.70,
    crescent_close=0.35,     # horizontal closing of the crescent mask, x design width (wide designs)
    port_max_h=0.35, port_min_h=0.012, port_min_area=10,
    port_smooth=0.004, port_rel_thr=0.05, port_abs_min_rows=2.5, port_min_mass_frac=0.15,
    port_w_min=0.15,         # port interval width, fraction of w_exp (W >= 12 px designs)
    port_w_max=2.2,
    port_split_above=1.6,    # intervals wider than this x w_exp are two merged ports
    frag_gap=0.25,           # same-face intervals closer than this x w_exp are fragments of one opening ...
    frag_merged_max=1.3,     # ... if the merged opening stays below this x w_exp
    narrow_design_px=12,     # below this w_exp the port is a wedge: width filters are absolute
    wedge_w_max_px=24,
    pair_tol=0.5, pair_tol_px=6, pair_tol_wexp=0.25,
    unpaired_mass_frac=0.5,
    arc_tau=0.5, arc_width_match=0.4,
    lattice_min_pairs=2, lattice_consistency=0.2, lattice_tol=0.3, lattice_margin=0.35,
    lattice_support_min=0.6, # with one port pair, complete the lattice only if the pitch has this port support
    pitch_min_over_wexp=1.1, # a pitch smaller than this x w_exp is not a channel pitch
    min_print_ratio=0.35,    # printed/design width below this means the tracked lines are not lumen edges
    # walls
    bin_tau=0.45, comp_min_h=0.03, wall_min_cover=0.6, wall_max_lines=1.5, wall_keep_out=0.4,
    # tracking
    roi_margin_px=3,
    roi_cap=1.1,             # tracking window half-width, x design width (wide designs)
    anchor_min_frac=0.8,     # port anchors are at least this x design width apart (crescents can be narrow)
    dp_lambda=0.06, dp_jump=2, dp_anchor=0.30,
    center_slack=0.3, center_wall=0.15,
    suppress_px=2, suppress_frac=0.10, suppress_max_px=8, suppress_gain=0.15,
    # presence
    hit_tau=0.35, hit_win=0.05, hit_frac=0.75, rough_max_px=2.5,
    anchor_depth=0.25,
    face_reach=0.10,         # a lumen reaches a face if present within this fraction of the length of it (tracks end in the crescent flare); a blind U ends farther in
    # classification
    w_smooth=0.02,           # median filter on the width (fraction of length)
    closed_min_len=0.015,    # closed runs shorter than this fraction of the length do not count
    debris_max_len=0.06,     # a closed run shorter than this where only ONE edge deflects is debris
    debris_asym=0.25,        # ... 'one edge': the other edge moved less than this fraction of the first
    taper_frac=0.4,          # a gap whose BOTH flanks get narrower than this x w_ref is a closure
    cap_tau=0.8,             # response that counts as 'dark' for a lumen tip (above speckle) ...
    cap_cover=0.8,           # ... whose union over the cap window must span this fraction of the cell width
    cap_reach=0.04,          # cap window reaches this fraction of the length beyond the end of a lumen run
    closed_min_px=5,         # two edge lines closer than this cannot be resolved: closed ...
    closed_line_px=4,        # ... or closer than the track repulsion radius plus this
    neck_frac=0.75,          # narrowest point below this x w_ref -> NECKED ("partially pinched")
    closed_frac=0.15,        # width below this x w_ref -> closed
    # a lumen run shorter than this fraction of the length, touching a face, is the port wedge itself
    # (the dark opening at the face), not a lumen. Review 2026-10 (Quynh): stubs of 5% of L and more
    # are channels that formed and then pinched; the old value (0.25) called them NOT_FORMED.
    wedge_max=0.04,
    not_formed_present=0.04, # open fraction below this -> NOT_FORMED (same reasoning)
    # rows where both walls are lost for longer than this fraction of the length are not open lumen:
    # a formed lumen draws a continuous outline, its absence over a long stretch means the lumen is
    # closed there (narrow design: resin plug; wide design: roof on the floor). Shorter gaps are
    # tracking dropouts and are bridged.
    dropout_max=0.12,
    fill_low_conf=0.3,       # a bridged gap longer than this fraction of the length -> channel confidence LOW
    # closure regime by design width W (printer px). Cross-sections (Roman, 2026-10-12): in narrow
    # channels the closure is uniform cured resin between the side walls (a plug, class PINCHED); in
    # wide channels it is the roof membrane resting on the floor, usually with a trapped bubble
    # (TOUCHDOWN). The review placed the boundary between W20 and W50; in between the shape of the
    # lumen ends decides: tips that taper to a point are a plug, blunt ends (the lumen keeps its width
    # up to a front across the channel) are the roof coming down.
    plug_w_px=25,            # W at or below this: closures are plugs
    touchdown_w_px=50,       # W at or above this: closures are touchdown (round 2: her touchdown calls start at W50)
    plug_taper=0.5,          # in between: mean inner/base width of the stubs below this -> plug
    touchdown_class_frac=0.10,  # roof on the floor over at least this fraction of the length -> class TOUCHDOWN
    touchdown_neck_frac=0.4, # touchdown regime: a lumen narrowed below this x w_ref is the roof front, roof down there
    edge_exclude=0.12,       # ignore the port flare when looking for the narrowest point
    touchdown_tau=0.5,       # dark-line threshold for the inner (touchdown) outline
    touchdown_min_frac=0.02, # flag touchdown (bubble, patch) when the enclosed area exceeds this fraction of the lumen
    touchdown_patch_frac=0.05,  # touchdown regime: an enclosed patch above this fraction of the lumen area -> class TOUCHDOWN
    front_tau=0.6,           # response that counts as a dark outline of a port pocket (wide_lattice)
    front_margin=0.10,       # wall_companions: skip this fraction of the length at each face (port rim, crescent)
    # port-to-port path through the lumen (lumen_path)
    path_tau=0.45,           # dark outlines above this response are barriers
    path_pad=0.12,           # the corridor is the wall span (or the design width) plus this x design width per side
    path_margin=0.12,        # seeds sit this fraction of the length inside each face
    path_speck=0.06,         # outline pieces shorter than this x design width are specks, not barriers
    path_seal=0.03,          # gaps in an outline up to this x design width are sealed
    # lattice of a wide design from the port pockets and the roof-step lines (wide_lattice)
    lat_band=0.25,           # face bands, fraction of the length
    lat_line_tau=0.3,        # line response that counts toward a step line
    lat_mid_weight=1.0,      # weight of one step line against one fully dark pocket window
    lat_pitch_tol=0.25,      # pitches searched around the design pitch
    lat_edge=0.15,           # a center stays at least this x design width inside the frame
    lat_gain=1.15,           # replace the port-based channels only if the fit scores this much better ...
    lat_shift=0.15,          # ... and moves some channel by at least this fraction of the pitch
    lat_min_score=0.25,      # ... and averages at least this score per channel
    # lumen entrances (the roof edge at each face) and the entrance-to-pinch metrics (2026-11)
    ent_band=0.12,           # the entrance is looked for within this fraction of the length from each face
    ent_min=0.12,            # dark fraction of the channel mouth, above the walls between channels, that counts as the edge
    ent_tol=0.03,            # a channel's own edge within this fraction of the length of the device median counts as detected
    ent_default=0.04,        # entrance position (fraction of the length from the face) when no channel shows the edge
    ent_attach=0.08,         # a lumen part belongs to a port if it starts within this fraction of the lumen length of the entrance
    ent_flare=0.07,          # per-half narrowest point: skip this fraction of the lumen length at the entrance (port flare)
    extent_taus=(0.45, 0.35, 0.25),  # outline thresholds tried, strict to loose, to find where each port pocket ends
    lat_min_w_px=0,          # designs at least this wide (printer px) get the lattice fit to pockets and step lines (0 = all designs; 50 = wide designs only, the rule before 2026-11-19)
    leak_jump=0.15,          # a pocket end that moves by more than this fraction of the lumen length between two outline thresholds is in dispute
    leak_gap=0.08,           # the longer pocket of a dispute is a leak when the tracked walls are missing over a run this long (fraction of the lumen length) inside the disputed stretch
    path_seal_fine=0.012,    # finer sealing of the outline gaps (x design width) for the second flood that follows a narrow tongue (pick_pockets)
    leak_fill=0.7,           # a flood row filled to less than this fraction of its own extent is seeping through surface grain, not inside a pocket
    pinch_min=0.02,          # the two port pockets count as separated only if they stay this fraction of the lumen length apart
    stub_hw=0.9,             # port stub: half-width of the search window (x design width, or x port width if larger)
    stub_open=0.12,          # port stub: opening kernel (x design width); cracks and edge lines thinner than this are not part of the stub
    stub_tol=0.02,           # port stub: it must begin within this fraction of the device length of the lumen entrance
    stub_blob=0.3,           # port stub: dark-blob response above this belongs to the stub body
    stub_max_len=0.25,       # port stub: search this fraction of the lumen length from the entrance; a piece that runs past it is not a stub
    sym_tol=0.05,            # pinch center within this fraction of the lumen length of mid-length -> symmetric
    bulge_min=0.10,          # mid-length width exceeding both half minima by this x reference width -> bulged
    comp_min=0.4,            # a companion line this long (fraction of the length) means the roof is down between the walls
    comp_band=0.15,          # wall_companions: search this x design width to either side of a wall
    open_wall_min=0.6,       # wide designs: both walls followed over at least this fraction of the core length -> open
)

CLASS_ORDER = ["FORMED", "NECKED", "PINCHED", "TOUCHDOWN", "NOT_FORMED"]
CLASS_COLOR = dict(FORMED="#2ca02c", NECKED="#ff7f0e", PINCHED="#d62728", TOUCHDOWN="#9467bd",
                   NOT_FORMED="#7f7f7f")


def runs_of(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def parse_name(fn):
    """H, W (printer px) and replicate from `..._H5_W150_3.jpg` or `s02_p03_H5_W30_top.jpg`."""
    m = re.search(r"H(\d+)_W(\d+)(?:_(\d+))?", os.path.basename(fn))
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)


# ----------------------------------------------------------------------------
# 1. footer + scale bar
# ----------------------------------------------------------------------------
def strip_footer(gray):
    dark_frac = (gray < P["footer_dark"]).mean(axis=1)
    h = gray.shape[0]
    y = h
    while y > int(0.6 * h) and dark_frac[y - 1] > 0.6:
        y -= 1
    footer = gray[y:, :]
    bar_px = 0
    if footer.size:
        for thr in (200, 160, 120):
            best = 0
            white = footer > thr
            for r in range(white.shape[0]):
                for a, b in runs_of(white[r]):
                    best = max(best, int(b - a))
            if best >= 0.03 * gray.shape[1]:      # a real bar is a few percent of the frame width
                bar_px = best
                break
            bar_px = max(bar_px, best)
    return gray[:y, :], bar_px


# ----------------------------------------------------------------------------
# 2. device detection
# ----------------------------------------------------------------------------
def texture_map(gray, k=21):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    m = cv2.blur(g, (k, k))
    m2 = cv2.blur(g * g, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0)) / np.maximum(m, 1.0)


def longest_run(flag, min_gap):
    f = ndi.binary_closing(flag, structure=np.ones(max(1, min_gap)))
    runs = runs_of(f)
    if not runs:
        return 0, len(flag)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return int(a), int(b)


def device_bbox(tex):
    h, w = tex.shape
    rowp = np.median(tex, axis=1)
    bg = np.percentile(rowp, 3)
    med = np.percentile(rowp, 60)
    tr = min(P["dev_tex_factor"] * bg, np.sqrt(bg * med)) if med > bg else 2 * bg
    y0, y1 = longest_run(rowp > tr, int(0.03 * h))
    colp = ndi.uniform_filter1d(np.median(tex[y0:y1], axis=0), max(5, int(0.02 * w)))
    x0, x1 = longest_run(colp > 0.6 * tr, int(0.03 * w))
    return (x0, y0, x1, y1), tr


def edge_angle(mask):
    h, w = mask.shape
    cols = np.arange(w)
    any_col = mask.any(axis=0)
    top = np.where(any_col, np.argmax(mask, axis=0), -1)
    bot = np.where(any_col, h - 1 - np.argmax(mask[::-1], axis=0), -1)
    angs = []
    for edge in (top, bot):
        ok = (edge > 2) & (edge < h - 3)
        if ok.sum() < 0.3 * w:
            continue
        x, y = cols[ok], edge[ok].astype(float)
        a, b = np.polyfit(x, y, 1)
        res = np.abs(y - (a * x + b))
        keep = res < max(2.0, 3 * np.median(res))
        if keep.sum() > 10:
            a, b = np.polyfit(x[keep], y[keep], 1)
        angs.append(np.degrees(np.arctan(a)))
    return float(np.mean(angs)) if angs else 0.0


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def detect_device(gray):
    tex = texture_map(gray)
    (x0, y0, x1, y1), thr = device_bbox(tex)
    py, px = int(0.06 * (y1 - y0)), int(0.03 * (x1 - x0))
    ya, yb = max(0, y0 - py), min(gray.shape[0], y1 + py)
    xa, xb = max(0, x0 - px), min(gray.shape[1], x1 + px)
    m = np.zeros(gray.shape, bool)
    sub = cv2.GaussianBlur(tex[ya:yb, xa:xb], (0, 0), 3) > thr
    sub = ndi.binary_opening(sub, iterations=2)
    m[ya:yb, xa:xb] = ndi.binary_fill_holes(largest_component(sub))
    ang = edge_angle(m)
    if abs(ang) > 10:
        ang = 0.0
    bg = int(np.median(gray[~m])) if (~m).any() else 255
    M = cv2.getRotationMatrix2D((0.5 * (x0 + x1), 0.5 * (y0 + y1)), ang, 1.0)
    rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderValue=bg)
    (x0, y0, x1, y1), _ = device_bbox(texture_map(rot))
    return rot, (x0, y0, x1, y1), ang


# ----------------------------------------------------------------------------
# 3. flat field and dark-structure maps
# ----------------------------------------------------------------------------
def dark_maps(dev):
    h = dev.shape[0]
    f = dev.astype(np.float32) + 1.0
    base = cv2.GaussianBlur(f, (0, 0), P["flat_sigma"] * h)
    N = np.clip(f / base, 0, 1.6)
    se_half = int(np.clip(round(P["line_se_frac"] * h), P["line_se_min"], P["line_se_max"]))
    se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * se_half + 1, 1))
    se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2 * se_half + 1))
    line = np.clip((cv2.morphologyEx(N, cv2.MORPH_CLOSE, se_h) - N) / P["line_scale"], 0, 1.5)
    arc = np.clip((cv2.morphologyEx(N, cv2.MORPH_CLOSE, se_v) - N) / P["line_scale"], 0, 1.5)
    blob = np.clip((P["blob_dark"] - N) / P["blob_dark"], 0, 1) * 1.5
    R = cv2.GaussianBlur(np.maximum(line, blob), (0, 0), sigmaX=0.7, sigmaY=1.0)
    return N, line, arc, blob, R


# ----------------------------------------------------------------------------
# 4-5. ports, channels, walls
# ----------------------------------------------------------------------------
def band_intervals(score, w, abs_min):
    s = ndi.gaussian_filter1d(score.astype(np.float32), max(1.5, P["port_smooth"] * w))
    edge = int(P["border_skip"] * w)
    s[:edge] = 0
    s[w - edge:] = 0
    if s.max() <= abs_min:
        return []
    thr = max(abs_min, P["port_rel_thr"] * s.max())
    flag = ndi.binary_closing(s > thr, structure=np.ones(max(3, int(0.012 * w))))
    out = []
    for a, b in runs_of(flag):
        seg = s[a:b]
        if b - a < 2:
            continue
        cx = float(a + np.sum(np.arange(b - a) * seg) / max(seg.sum(), 1e-6))
        out.append(dict(x=cx, width=float(b - a), left=float(a), right=float(b),
                        mass=float(seg.sum()), height=float(seg.max()), score=s))
    if out:
        ref = np.median([o["mass"] for o in out])
        out = [o for o in out if o["mass"] >= P["port_min_mass_frac"] * ref]
    return out


def size_filter_and_split(ports, w_exp):
    """Reject port intervals that cannot be a channel opening of this design
    width, and split intervals that are two merged openings."""
    out = []
    for p in ports:
        if w_exp >= P["narrow_design_px"]:
            if p["width"] > P["port_split_above"] * w_exp:
                k = int(round(p["width"] / (1.15 * w_exp)))
                if k >= 2:
                    a, b = int(p["left"]), int(p["right"])
                    s = p["score"][a:b]
                    cuts = [a + int(round(i * (b - a) / k)) for i in range(k + 1)]
                    # move each internal cut to the nearest score minimum
                    for i in range(1, k):
                        lo, hi = cuts[i] - int(0.2 * w_exp), cuts[i] + int(0.2 * w_exp)
                        lo, hi = max(a + 2, lo), min(b - 2, hi)
                        if hi > lo:
                            cuts[i] = lo + int(np.argmin(p["score"][lo:hi]))
                    for i in range(k):
                        aa, bb = cuts[i], cuts[i + 1]
                        seg = p["score"][aa:bb]
                        if bb - aa < 2 or seg.sum() <= 0:
                            continue
                        cx = float(aa + np.sum(np.arange(bb - aa) * seg) / seg.sum())
                        out.append(dict(x=cx, width=float(bb - aa), left=float(aa), right=float(bb),
                                        mass=float(seg.sum()), height=float(seg.max()),
                                        score=p["score"], source=p.get("source"), split=True))
                    continue
            if not (P["port_w_min"] * w_exp <= p["width"] <= P["port_w_max"] * w_exp):
                continue
        else:
            if p["width"] > P["wedge_w_max_px"]:
                continue
        out.append(p)
    return out


def merge_fragments(ports, w_exp):
    """On one face, intervals closer than 0.6 x the design width belong to the
    same opening (an arc broken into pieces, a crescent with a gap)."""
    if w_exp < P["narrow_design_px"] or len(ports) < 2:
        return ports
    out = [dict(ports[0])]
    for p in ports[1:]:
        q = out[-1]
        if p["left"] - q["right"] < P["frag_gap"] * w_exp and (p["right"] - q["left"]) <= P["frag_merged_max"] * w_exp:
            q["left"], q["right"] = min(q["left"], p["left"]), max(q["right"], p["right"])
            q["width"] = q["right"] - q["left"]
            q["mass"] += p["mass"]
            q["x"] = 0.5 * (q["left"] + q["right"])
            q["height"] = max(q["height"], p["height"])
            if p["source"] == "crescent":
                q["source"] = "crescent"
        else:
            out.append(dict(p))
    return out


def lattice_clean(ports):
    if len(ports) < 3:
        return ports
    pitch = float(np.median(np.diff(sorted(p["x"] for p in ports))))
    keep = []
    for p in sorted(ports, key=lambda q: -q["mass"]):
        if all(abs(p["x"] - k["x"]) > 0.5 * pitch for k in keep):
            keep.append(p)
    return sorted(keep, key=lambda q: q["x"])


def pair_ports(top, bot, w, w_exp):
    channels, used = [], set()
    for t in top:
        best = None
        for j, b in enumerate(bot):
            if j in used:
                continue
            d = abs(b["x"] - t["x"])
            tol = max(P["pair_tol_px"], P["pair_tol"] * 0.5 * (t["width"] + b["width"]), P["pair_tol_wexp"] * w_exp)
            if d <= tol and (best is None or d < best[1]):
                best = (j, d)
        c = dict(top=t, bot=None)
        if best is not None:
            used.add(best[0])
            c["bot"] = bot[best[0]]
        channels.append(c)
    for j, b in enumerate(bot):
        if j not in used:
            channels.append(dict(top=None, bot=b))
    for c in channels:
        c["x"] = np.mean([p["x"] for p in (c["top"], c["bot"]) if p])
    channels.sort(key=lambda c: c["x"])
    paired = [0.5 * (c["top"]["mass"] + c["bot"]["mass"]) for c in channels if c["top"] and c["bot"]]
    if not paired:
        if w_exp < P["narrow_design_px"]:
            return []                      # wedge-scale ports without a pair are specks
        ref = float(np.median([(c["top"] or c["bot"])["mass"] for c in channels])) if channels else 0.0
    else:
        ref = np.median(paired)
    # an unpaired crescent of a plausible size is a channel whose other port
    # is faint; only for wedge-scale designs (no size prior) demand mass
    channels = [c for c in channels if (c["top"] and c["bot"])
                or (c["top"] or c["bot"]).get("source") == "arc"
                or w_exp >= P["narrow_design_px"]
                or (c["top"] or c["bot"])["mass"] >= P["unpaired_mass_frac"] * ref]
    for c in channels:
        xt = c["top"]["x"] if c["top"] else None
        xb = c["bot"]["x"] if c["bot"] else None
        c["x_top"] = xt if xt is not None else xb
        c["x_bot"] = xb if xb is not None else xt
        c["x"] = 0.5 * (c["x_top"] + c["x_bot"])
        c["w_port"] = float(np.median([p["width"] for p in (c["top"], c["bot"]) if p]))
    return channels


def complete_lattice(channels, pitch, faces, w, w_exp, min_pairs=None):
    """The device is a regular array. With a pitch from the port pairs, snap
    every channel to a lattice index, keep the best channel per index (a pair
    beats a single port, then mass), drop off-lattice singles (specks), and
    fill empty indices from weaker port evidence or as predicted cells."""
    paired = [c for c in channels if c["top"] and c["bot"]]
    if min_pairs is None:
        min_pairs = P["lattice_min_pairs"]
    if pitch is None or len(paired) < min_pairs:
        return channels
    if not channels:
        return channels
    if not paired:
        # phase the lattice on the strongest single port
        best = max(channels, key=lambda c: sum(p["mass"] for p in (c["top"], c["bot"]) if p))
        paired = [best]
    xs = np.array([c["x"] for c in paired])
    k = np.round((xs - xs[0]) / pitch)
    x_ref = float(xs[0])
    if len(np.unique(k)) >= 2:
        A = np.column_stack([k, np.ones_like(k)])
        (pitch_fit, x_ref_fit), *_ = np.linalg.lstsq(A, xs, rcond=None)
        resid = np.abs(xs - (x_ref_fit + k * pitch_fit))
        if 0.85 * pitch < pitch_fit < 1.15 * pitch and resid.max() < 0.15 * pitch:
            pitch, x_ref = float(pitch_fit), float(x_ref_fit)

    def score(c):
        return (2 if (c["top"] and c["bot"]) else 1, sum(p["mass"] for p in (c["top"], c["bot"]) if p))

    slots = {}
    for c in channels:
        kk = int(round((c["x"] - x_ref) / pitch))
        xp = x_ref + kk * pitch
        if abs(c["x"] - xp) > P["lattice_tol"] * pitch:
            continue                      # off the lattice: not a cell
        if not (c["top"] and c["bot"]) and abs(c["x"] - xp) > 0.1 * pitch:
            c = dict(c, x=xp, x_top=xp, x_bot=xp, predicted="port")   # a single small port: trust the lattice
        if kk not in slots or score(c) > score(slots[kk]):
            slots[kk] = c
    w_med = float(np.median([c["w_port"] for c in channels]))
    margin = P["lattice_margin"] * pitch
    all_ports = faces[0] + faces[1]
    kmin = int(np.floor((margin - x_ref) / pitch))
    kmax = int(np.ceil((w - margin - x_ref) / pitch))
    for kk in range(kmin, kmax + 1):
        if kk in slots:
            continue
        xp = x_ref + kk * pitch
        if xp < margin or xp > w - margin:
            continue
        if xp - 0.5 * w_exp < P["border_skip"] * w or xp + 0.5 * w_exp > (1 - P["border_skip"]) * w:
            continue
        near = [p for p in all_ports if abs(p["x"] - xp) < P["lattice_tol"] * pitch]
        if near:
            p = max(near, key=lambda q: q["mass"])
            slots[kk] = dict(top=p if p in faces[0] else None, bot=p if p in faces[1] else None,
                             x=p["x"], x_top=p["x"], x_bot=p["x"], w_port=p["width"], predicted="port")
        else:
            slots[kk] = dict(top=None, bot=None, x=xp, x_top=xp, x_bot=xp,
                             w_port=w_exp if w_exp >= P["narrow_design_px"] else w_med, predicted="lattice")
    return [slots[kk] for kk in sorted(slots)]


def find_channels(N, line, arc, blob, h, w, w_exp, pitch_exp=None, n_expected=None):
    ey, ex = max(1, int(P["face_skip"] * h)), max(1, int(P["border_skip"] * w))

    def clear_edges(D):
        D[:ey, :] = 0
        D[h - ey:, :] = 0
        D[:, :ex] = 0
        D[:, w - ex:] = 0
        return D

    D = clear_edges((line > P["bin_tau"]).astype(np.uint8))
    lab, n = ndi.label(D, structure=np.ones((3, 3)))
    comps = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        ys, xs = sl
        hgt, wid = ys.stop - ys.start, xs.stop - xs.start
        if hgt < P["comp_min_h"] * h:
            continue
        area = int((lab[sl] == i).sum())
        comps.append(dict(x0=xs.start, x1=xs.stop, hgt=hgt, w=wid, thick=area / max(hgt, 1)))
    tall = [c["thick"] for c in comps if c["hgt"] > 0.5 * h]
    t_line = float(np.median(tall)) if tall else 2.0

    # crescents / wedges: short dark components at the faces, tall lines cut out first
    L = int(max(4, round(2.5 * t_line + 1)))
    C0 = clear_edges((N < P["crescent_dark"]).astype(np.uint8))
    tall_mask = np.zeros_like(D)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is not None and (sl[0].stop - sl[0].start) > P["port_max_h"] * h:
            tall_mask[sl][lab[sl] == i] = 1
    tall_mask = cv2.dilate(tall_mask, np.ones((3, 5), np.uint8))
    C0[tall_mask > 0] = 0
    if w_exp >= P["narrow_design_px"]:
        kx = int(max(3, round(P["crescent_close"] * w_exp))) | 1
        C0 = cv2.morphologyEx(C0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
        C0 = clear_edges(C0)
    labc, nc = ndi.label(C0, structure=np.ones((3, 3)))
    C = np.zeros_like(C0)
    for i, sl in enumerate(ndi.find_objects(labc), 1):
        if sl is None:
            continue
        ys, xs = sl
        hgt, wid = ys.stop - ys.start, xs.stop - xs.start
        if hgt > P["port_max_h"] * h or hgt < P["port_min_h"] * h or wid < 3:
            continue
        if ys.start > P["port_band"] * h and ys.stop < h - P["port_band"] * h:
            continue
        m = labc[sl] == i
        if m.sum() < P["port_min_area"]:
            continue
        C[sl][m] = 1
    A = clear_edges((cv2.blur(arc, (max(5, L), 1)) > P["arc_tau"]).astype(np.uint8))

    d = int(P["port_band"] * h)
    faces, sources = [], []
    for band in (slice(0, d), slice(h - d, h)):
        sc = C[band].sum(axis=0).astype(np.float32)
        crescents = band_intervals(sc - np.median(sc), w, P["port_abs_min_rows"])
        for p in crescents:
            p["source"] = "crescent"
        crescents = size_filter_and_split(crescents, w_exp)
        ports = list(crescents)
        if len(crescents) < 2:
            sa = A[band].sum(axis=0).astype(np.float32)
            arcs = band_intervals(sa - np.median(sa), w, P["port_abs_min_rows"])
            for a in arcs:
                a["source"] = "arc"
            arcs = size_filter_and_split(arcs, w_exp)
            for a in arcs:
                if not any(abs(a["x"] - c["x"]) < 0.5 * (a["width"] + c["width"]) for c in crescents):
                    ports.append(a)
        ports = merge_fragments(sorted(ports, key=lambda q: q["x"]), w_exp)
        ports = lattice_clean(ports)
        faces.append(ports)
        sources.append(f"{sum(p['source'] == 'crescent' for p in ports)}c+{sum(p['source'] == 'arc' for p in ports)}a")
    channels = pair_ports(faces[0], faces[1], w, w_exp)

    def arc_ok(c):
        if c["top"] and c["bot"]:
            if c["top"]["source"] == "arc" and c["bot"]["source"] == "arc":
                wt, wb = c["top"]["width"], c["bot"]["width"]
                return abs(wt - wb) <= P["arc_width_match"] * max(wt, wb)
            return True
        return (c["top"] or c["bot"])["source"] == "crescent"
    channels = [c for c in channels if arc_ok(c)]

    # channels closer than the design width are the same opening seen twice
    merged = []
    for c in channels:
        if merged and abs(c["x"] - merged[-1]["x"]) < 0.8 * w_exp:
            keep = merged[-1] if (merged[-1]["w_port"] >= c["w_port"]) else c
            merged[-1] = keep
        else:
            merged.append(c)
    channels = merged

    pitch = None
    paired = [c for c in channels if c["top"] and c["bot"]]
    if len(paired) >= 2:
        dif = np.diff([c["x"] for c in paired])
        base = float(np.min(dif))
        mult = np.round(dif / base)
        pitch = float(np.median(dif / np.maximum(mult, 1)))
    elif len(channels) >= 2:
        pitch = float(np.median(np.diff([c["x"] for c in channels])))
    if pitch is not None and pitch < P["pitch_min_over_wexp"] * w_exp:
        pitch = None
    # walls (thin, tall, regular) give an independent pitch estimate
    t_wall0 = float(np.percentile(tall, 25)) if tall else t_line
    wall_x = sorted(0.5 * (c["x0"] + c["x1"]) for c in comps
                    if c["thick"] <= P["wall_max_lines"] * t_wall0 + 1 and c["hgt"] >= P["wall_min_cover"] * h)
    wall_pitch = None
    if len(wall_x) >= 3:
        wd = np.diff(wall_x)
        wd = wd[wd > P["pitch_min_over_wexp"] * w_exp]
        if len(wd) >= 2:
            base = float(np.min(wd))
            mult = np.round(wd / base)
            cand = float(np.median(wd / np.maximum(mult, 1)))
            spread = np.abs(wd / np.maximum(mult, 1) - cand) / cand
            if np.mean(spread < 0.1) >= 0.6:
                wall_pitch = cand
    pitch_support = 0.0
    # choose among the candidate pitches the one whose lattice (phased on the
    # first pair) is best supported by the ports on both faces: most port mass
    # on lattice positions, fewest empty positions
    all_ports = faces[0] + faces[1]
    cands = [p for p in (pitch, wall_pitch) if p is not None]
    if len(all_ports) >= 3:
        pxs = np.array(sorted(p["x"] for p in all_ports))
        pd_ = np.diff(pxs)
        pd_ = pd_[pd_ > P["pitch_min_over_wexp"] * w_exp]
        if len(pd_):
            base = float(np.min(pd_))
            mult = np.round(pd_ / base)
            cands.append(float(np.median(pd_ / np.maximum(mult, 1))))
    # with the channel count known from the frame name, a face that shows nearly all its ports
    # gives the pitch directly as the span of the ports over their number; this survives a
    # misread scale bar, which the design pitch (wall + W in image px) does not
    span_cands = []
    if n_expected and n_expected >= 3:
        for ports in faces:
            # only a face that shows every port (at most one extra): fewer means a port is missing
            # somewhere in the row, more means fragments, and both corrupt the span
            if n_expected <= len(ports) <= n_expected + 1:
                xs_p = sorted(p["x"] for p in ports)
                cand = (xs_p[-1] - xs_p[0]) / (len(ports) - 1)
                if pitch_exp is None or abs(cand - pitch_exp) / pitch_exp <= 0.4:
                    span_cands.append(cand)
    if pitch_exp is not None:
        cands = [c for c in cands if abs(c - pitch_exp) / pitch_exp <= P["pitch_tol"]]
        cands.append(float(pitch_exp))
    cands += [c for c in span_cands if c > P["pitch_min_over_wexp"] * w_exp]
    if cands and (paired or all_ports):
        x0 = paired[0]["x"] if paired else max(all_ports, key=lambda p: p["mass"])["x"]
        total = sum(p["mass"] for p in all_ports) or 1.0

        def support(pc):
            kmin = int(np.floor((P["border_skip"] * w - x0) / pc))
            kmax = int(np.ceil(((1 - P["border_skip"]) * w - x0) / pc))
            pos = [x0 + k * pc for k in range(kmin, kmax + 1)
                   if P["border_skip"] * w + 0.5 * w_exp <= x0 + k * pc <= (1 - P["border_skip"]) * w - 0.5 * w_exp]
            if not pos:
                return -1.0
            matched = 0.0
            filled = 0
            for xp in pos:
                near = [p["mass"] for p in all_ports if abs(p["x"] - xp) < 0.2 * pc]
                if near:
                    filled += 1
                    matched += sum(near)
            return (matched / total) * (filled / len(pos))
        if os.environ.get('TV_DEBUG'):
            print('   pitch candidates', [(round(c, 1), round(support(c), 3)) for c in cands], 'pairs', len(paired), 'walls', [round(x) for x in wall_x])
        sup = {c: support(c) for c in cands}
        best = max(cands, key=lambda c: sup[c])
        if pitch_exp is not None and sup[float(pitch_exp)] >= 0.9 * sup[best]:
            best = float(pitch_exp)          # the design pitch wins near-ties
        # a measured pitch needs three real pairs and solid support to beat the design pitch,
        # unless it is the port-span pitch of a frame whose channel count is known
        if pitch_exp is not None and (len(paired) < 3 or sup[best] < 0.6) and best not in span_cands:
            best = float(pitch_exp)
        if sup[best] > 0 or best == pitch_exp:
            pitch = best
            pitch_support = max(sup[best], 0.6 if best == pitch_exp else 0.0)
    elif wall_pitch is not None and pitch is None:
        pitch = wall_pitch
    if pitch_exp is not None and (pitch is None or abs(pitch - pitch_exp) / pitch_exp > P["pitch_tol"]) \
            and pitch not in span_cands:
        pitch = float(pitch_exp)
        pitch_support = 0.6
    if pitch is not None and len(paired) >= 1:
        # keep unpaired channels only on the lattice defined by the pairs
        x0 = paired[0]["x"]
        channels = [c for c in channels if (c["top"] and c["bot"])
                    or abs(((c["x"] - x0) / pitch) - round((c["x"] - x0) / pitch)) < P["lattice_tol"]]
    channels = complete_lattice(channels, pitch, faces, w, w_exp,
                                min_pairs=0 if pitch_support >= P["lattice_support_min"] else P["lattice_min_pairs"])
    if len(channels) >= 2:
        pitch = float(np.median(np.diff([c["x"] for c in channels])))

    walls = []
    t_wall = float(np.percentile(tall, 25)) if tall else t_line
    for c in comps:
        if c["thick"] <= P["wall_max_lines"] * t_wall + 1 and c["hgt"] >= P["wall_min_cover"] * h:
            xm = 0.5 * (c["x0"] + c["x1"])
            keep_out = max(0.6 * w_exp + 4, (P["wall_keep_out"] * pitch) if pitch else 0)
            if not any(abs(xm - ch["x"]) < max(0.6 * ch["w_port"] + 2, keep_out) for ch in channels):
                walls.append(int(xm))
    walls = sorted(set(walls))
    n_pairs = sum(1 for c in channels if c["top"] and c["bot"] and not c.get("predicted"))
    confidence = "HIGH" if n_pairs >= 3 else ("MEDIUM" if n_pairs >= 2 else "LOW")
    diag = dict(t_line=round(t_line, 2), n_top=len(faces[0]), n_bot=len(faces[1]),
                port_source=f"{sources[0]}/{sources[1]}", n_pairs=n_pairs, confidence=confidence,
                pitch_support=round(pitch_support, 2))
    return channels, walls, pitch, diag, faces


# ----------------------------------------------------------------------------
# 6. two-terminal DP edge tracking
# ----------------------------------------------------------------------------
def track_edge(R, x_lo, x_hi, x_top, x_bot):
    sub = R[:, x_lo:x_hi]
    h, w = sub.shape
    J, lam, alpha = P["dp_jump"], P["dp_lambda"], P["dp_anchor"]
    xs = np.arange(w)
    C = np.empty((h, w), np.float32)
    back = np.zeros((h, w), np.int8)
    C[0] = -sub[0] + alpha * np.abs(xs - (x_top - x_lo))
    for y in range(1, h):
        prev = C[y - 1]
        best = np.full(w, np.inf, np.float32)
        arg = np.zeros(w, np.int8)
        for dd in range(-J, J + 1):
            cand = np.full(w, np.inf, np.float32)
            if dd >= 0:
                cand[dd:] = prev[:w - dd] + lam * abs(dd)
            else:
                cand[:w + dd] = prev[-dd:] + lam * abs(dd)
            better = cand < best
            best[better] = cand[better]
            arg[better] = dd
        C[y] = best - sub[y]
        back[y] = arg
    C[-1] += alpha * np.abs(xs - (x_bot - x_lo))
    path = np.empty(h, np.int32)
    path[-1] = int(np.argmin(C[-1]))
    for y in range(h - 1, 0, -1):
        path[y - 1] = path[y] - back[y, path[y]]
    return path + x_lo


# ----------------------------------------------------------------------------
# 7. tracking + presence (per channel, pass 1)
# ----------------------------------------------------------------------------
def track_channel(R, ch, x_lo, x_hi, h, w_max=None):
    wp = ch["w_port"]
    ys = np.arange(h)
    xc = ch["x_top"] + (ch["x_bot"] - ch["x_top"]) * ys / max(h - 1, 1)
    xx = np.arange(R.shape[1])[None, :]
    slack = max(3.0, P["center_slack"] * wp)
    wallL = np.clip(xx - (xc[:, None] + slack), 0, None) * P["center_wall"]
    wallR = np.clip((xc[:, None] - slack) - xx, 0, None) * P["center_wall"]
    sup = int(np.clip(round(P["suppress_frac"] * wp), P["suppress_px"], P["suppress_max_px"]))

    def suppress(path):
        m = np.ones_like(R)
        for dx in range(-sup, sup + 1):
            m[ys, np.clip(path + dx, 0, R.shape[1] - 1)] = P["suppress_gain"]
        return m

    aL = (ch["x_top"] - wp / 2, ch["x_bot"] - wp / 2)
    aR = (ch["x_top"] + wp / 2, ch["x_bot"] + wp / 2)

    def pair_score(xl, xr):
        sep = (xr - xl) >= 3
        if sep.mean() < 0.3:
            return -1.0
        return float(np.mean((R[ys, xl] + R[ys, xr]) * sep))

    # order 1: left edge first
    xL1 = track_edge(R - wallL, x_lo, x_hi, *aL)
    xR1 = track_edge(R * suppress(xL1) - wallR, x_lo, x_hi, *aR)
    xL1 = track_edge(R * suppress(xR1) - wallL, x_lo, x_hi, *aL)
    # order 2: right edge first (the first track may grab the wrong side's line
    # when the lumen is narrow or the center estimate is off)
    xR2 = track_edge(R - wallR, x_lo, x_hi, *aR)
    xL2 = track_edge(R * suppress(xR2) - wallL, x_lo, x_hi, *aL)
    xR2 = track_edge(R * suppress(xL2) - wallR, x_lo, x_hi, *aR)
    xL, xR = (xL1, xR1) if pair_score(xL1, xR1) >= pair_score(xL2, xR2) else (xL2, xR2)
    if np.median(xL) > np.median(xR):
        xL, xR = xR, xL

    win = max(7, int(P["hit_win"] * h))
    hitL = ndi.uniform_filter1d((R[ys, xL] > P["hit_tau"]).astype(np.float32), win)
    hitR = ndi.uniform_filter1d((R[ys, xR] > P["hit_tau"]).astype(np.float32), win)
    on_bound = (xL <= x_lo + 1) | (xR >= x_hi - 2)
    width = np.maximum(xR - xL, 0).astype(np.float32)
    rough = ndi.uniform_filter1d(np.abs(width - ndi.uniform_filter1d(width, win)), win)
    present = (hitL > P["hit_frac"]) & (hitR > P["hit_frac"]) & ~on_bound & (rough < P["rough_max_px"])
    if w_max is not None:
        present &= width <= w_max
    present = ndi.binary_closing(present, structure=np.ones(max(3, int(0.01 * h))))
    anchor = P["anchor_depth"] * h
    for a, b in runs_of(present):
        if a > anchor and b < h - anchor:
            present[a:b] = False
    width_raw = width.copy()
    width[~present] = np.nan
    return dict(xL=xL, xR=xR, present=present, width=width, width_raw=width_raw,
                x_lo=x_lo, x_hi=x_hi, w_port=wp, sup=sup, hitL=hitL, hitR=hitR)


# ----------------------------------------------------------------------------
# 8. classification (pass 2, with the device-level reference width)
# ----------------------------------------------------------------------------
def nanmedfilt(x, win):
    """Median filter that ignores NaN (returns NaN where the window is all NaN)."""
    n = len(x)
    out = np.full(n, np.nan, np.float32)
    r = win // 2
    for i in range(n):
        seg = x[max(0, i - r): i + r + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def measure_halves(w_cls, closed, h, reach, edge):
    """Width of the two lumen halves along the channel and the gap between their tips.

    Open rows are rows with a tracked width above the closed threshold. The top half is the open
    run that starts at the top face, the bottom half the run that ends at the bottom face (a
    continuous channel is split at mid-length). Per half, after skipping the port flare (`edge`
    rows): w_base = median width of the outer 20% (port end), w_inner = median of the innermost
    10% (tip or mid-length), taper = w_inner / w_base, narrowing = -d(width)/d(distance from the
    port) from a linear fit (px per px; positive = gets thinner toward the middle). tip_gap_frac =
    distance between the two inner tips as a fraction of the length (0 for a continuous channel;
    a missing half counts from the opposite face)."""
    ok = ~np.isnan(w_cls) & ~closed
    runs = runs_of(ok)
    start = max(reach, edge + 0.06 * h)     # a half may begin just past the port flare
    # the longest open run that starts near the face (slivers inside the flare are not the stub)
    tops = [(a, b) for a, b in runs if a <= start]
    bots = [(a, b) for a, b in runs if b >= h - start]
    top = max(tops, key=lambda r: r[1] - r[0]) if tops else None
    bot = max(bots, key=lambda r: r[1] - r[0]) if bots else None
    if top and bot and top == bot:                      # continuous channel: split at mid-length
        a, b = top
        top, bot = (a, (a + b) // 2), ((a + b) // 2, b)
    if os.environ.get("TV_DEBUG"):
        print(f"   halves: open runs {[(round(a / h, 2), round(b / h, 2)) for a, b in runs]} start<={start / h:.2f} top {top} bot {bot}")
    out = dict(tip_gap_frac=np.nan, open_top_len_frac=0.0, open_bot_len_frac=0.0)

    def half(run, from_top):
        a, b = run
        a2, b2 = (max(a, edge), b) if from_top else (a, min(b, h - edge))
        if b2 - a2 < 6:
            return dict(w_base=np.nan, w_inner=np.nan, taper=np.nan, narrowing=np.nan)
        w = w_cls[a2:b2]
        if not from_top:
            w = w[::-1]                                  # index 0 = port end
        n = len(w)
        good = ~np.isnan(w)
        base = float(np.nanmedian(w[: max(3, n // 5)]))
        inner = float(np.nanmedian(w[-max(3, n // 10):]))
        slope = float(np.polyfit(np.arange(n)[good], w[good], 1)[0]) if good.sum() >= 6 else np.nan
        return dict(w_base=base, w_inner=inner, taper=inner / base if base > 0 else np.nan, narrowing=-slope)

    if top:
        out.update({f"{k}_top": v for k, v in half(top, True).items()})
        out["open_top_len_frac"] = (top[1] - top[0]) / h
    if bot:
        out.update({f"{k}_bot": v for k, v in half(bot, False).items()})
        out["open_bot_len_frac"] = (bot[1] - bot[0]) / h
    if top or bot:
        out["tip_gap_frac"] = max(0, (bot[0] if bot else h) - (top[1] if top else 0)) / h
    for k in ("w_base_top", "w_inner_top", "taper_top", "narrowing_top",
              "w_base_bot", "w_inner_bot", "taper_bot", "narrowing_bot"):
        out.setdefault(k, np.nan)
    return out


def closure_regime(W_design, w_exp):
    """Which physical closure a mid-channel closure is (see P['plug_w_px'] and P['touchdown_w_px']):
    'plug' (narrow designs), 'touchdown' (wide designs) or 'shape' (decide from the lumen ends)."""
    if W_design is None:
        return "plug" if w_exp < 2 * P["narrow_design_px"] else "shape"
    if W_design <= P["plug_w_px"]:
        return "plug"
    if W_design >= P["touchdown_w_px"]:
        return "touchdown"
    return "shape"


def classify_channel(tr, h, w_ref, arc, w_exp, blob=None, R_full=None, W_design=None, has_port=True,
                     path=None, td_frac=0.0, companion=0.0, ent=None, ext=None):
    """Class and measurements of one tracked channel.

    Open rows are rows where both walls were followed (a lumen draws a continuous outline). Runs
    shorter than the port wedge do not count. Between open runs, a short gap is a tracking dropout
    and is bridged; a long gap, converging tips, or a dark cap mean the lumen is closed there. Rows
    between the last open run and a face are closed too (the lumen did not reach that face), except
    the port flare itself. A closure is a resin plug (PINCHED) in the narrow regime and the roof on
    the floor (TOUCHDOWN) in the wide regime; in between, tapering tips mean a plug, blunt ends mean
    the roof came down (closure_regime)."""
    present = tr["present"].copy()
    width = tr["width"].copy()
    win = max(5, int(P["w_smooth"] * h) | 1)
    reach, wedge = P["face_reach"] * h, P["wedge_max"] * h
    flare = int(P["edge_exclude"] * h)
    regime = closure_regime(W_design, w_exp)

    # lumen runs: present runs longer than the port wedge
    lumen_runs = [(a, b) for a, b in runs_of(present) if (b - a) > wedge]
    present[:] = False
    for a, b in lumen_runs:
        present[a:b] = True
    f_raw = float(present.mean())

    def capped(y0, y1):
        """Is the lumen closed off by a dark cap (rounded blind end, converging tips) in rows y0:y1?
        The union over these rows of strong dark structure must cover nearly the whole width between
        the two edges; a tracking dropout leaves the interior open."""
        y0, y1 = max(0, y0), min(h, y1)
        if y1 <= y0:
            return False
        xa = int(np.nanmin(tr["xL"][y0:y1])) + 1
        xb = int(np.nanmax(tr["xR"][y0:y1]))
        if xb - xa < 3:
            return False
        capmap = arc if blob is None else np.maximum(arc, blob)
        strong = (capmap[y0:y1, xa:xb] > P["cap_tau"]) | (R_full[y0:y1, xa:xb] > P["cap_tau"])
        cover = strong.any(axis=0).mean()
        return bool(cover > P["cap_cover"])

    longest_fill = 0.0
    if lumen_runs:
        # interior gaps
        for a, b in runs_of(~present):
            if a == 0 or b == h:
                continue
            fl = width[max(0, a - win):a]
            fr = width[b:b + win]
            fl, fr = fl[~np.isnan(fl)], fr[~np.isnan(fr)]
            ml = np.min(fl) if fl.size else np.nan
            mr = np.min(fr) if fr.size else np.nan
            tapers = (np.isfinite(ml) and ml < P["taper_frac"] * w_ref) and (np.isfinite(mr) and mr < P["taper_frac"] * w_ref)
            cw_in, cw_out = max(3, int(0.01 * h)), max(6, int(P["cap_reach"] * h))
            caps = capped(a - cw_in, a + cw_out) or capped(b - cw_out, b + cw_in)
            # a narrow lumen draws a crisp outline, so losing both walls for long means it is closed
            # there; a wide lumen's walls can fade while it stays open (review: "necked"), so a long
            # gap without taper or cap is bridged there and only lowers the confidence
            long_gap = regime == "plug" and (b - a) > P["dropout_max"] * h
            if tapers or caps or long_gap:
                width[a:b] = 0.0
            else:
                # dropout: bridge with the flank medians (single end rows are noisy), but keep the
                # raw tracked width where one edge was still followed (a faint neck survives)
                wl = np.median(fl) if fl.size else width[a - 1]
                wr = np.median(fr) if fr.size else width[min(b, h - 1)]
                bridge = np.interp(np.arange(a, b), [a - 1, b], [wl, wr])
                raw = nanmedfilt(tr["width_raw"][a:b].astype(np.float32), win)
                one_edge = (tr["hitL"][a:b] > P["hit_frac"]) | (tr["hitR"][a:b] > P["hit_frac"])
                width[a:b] = np.where(one_edge & (raw > 0), np.minimum(bridge, raw), bridge)
                longest_fill = max(longest_fill, (b - a) / h)
            present[a:b] = True
        # end gaps: the lumen stops before a face -> closed from there to the port flare
        a0, b0 = lumen_runs[0]
        a1, b1 = lumen_runs[-1]
        if a0 > reach:
            width[flare:a0] = 0.0
        if b1 < h - reach:
            width[b1:h - flare] = 0.0
    reaches_top = bool(lumen_runs) and lumen_runs[0][0] <= reach
    reaches_bot = bool(lumen_runs) and lumen_runs[-1][1] >= h - reach

    w_cls = nanmedfilt(width, win)
    ok = ~np.isnan(w_cls)
    f_present = float(ok.mean())
    core = ok.copy()
    core[:flare] = False
    core[h - flare:] = False
    if core.any():
        wc = np.where(core, w_cls, np.nan)
        i_min = int(np.nanargmin(wc))
        w_min, w_min_y = float(wc[i_min]), i_min / h
    else:
        w_min, w_min_y = np.nan, np.nan
    w_min_frac = w_min / w_ref if np.isfinite(w_min) else np.nan

    # two tracks that repel each other cannot get closer than the suppression radius plus a line
    # width: that is what a closed lumen looks like
    closed_px = max(P["closed_frac"] * w_ref, P["closed_min_px"], tr["sup"] + P["closed_line_px"])
    if regime == "touchdown":
        # the visible lumen narrows where the roof front comes down: roof down there too
        closed_px = max(closed_px, P["touchdown_neck_frac"] * w_ref)
    closed = core & (w_cls < closed_px)
    # drop short closed runs, and debris: a short closure where only one edge deflected (a real
    # pinch pulls both edges toward the center)
    for a, b in runs_of(closed):
        if (b - a) < P["closed_min_len"] * h:
            closed[a:b] = False
            continue
        if (b - a) < P["debris_max_len"] * h:
            fa, fb = max(0, a - win), min(h, b + win)
            xl_out = np.median(np.concatenate([tr["xL"][fa:a], tr["xL"][b:fb]]))
            xr_out = np.median(np.concatenate([tr["xR"][fa:a], tr["xR"][b:fb]]))
            dL = float(np.median(tr["xL"][a:b]) - xl_out)   # left edge moved right
            dR = float(xr_out - np.median(tr["xR"][a:b]))   # right edge moved left
            big, small = max(dL, dR), min(dL, dR)
            if big > 0 and small < P["debris_asym"] * big:
                closed[a:b] = False
                w_cls[a:b] = np.interp(np.arange(a, b), [a - 1, b], [w_cls[max(a - 1, 0)], w_cls[min(b, h - 1)]])
    fronts = np.zeros(h, bool)
    if regime != "plug" and R_full is not None:
        # the roof front crosses the lumen while the walls stay straight: roof down there
        fronts = roof_front_rows(tr, R_full, arc, blob, h, flare, tr["present"]) & core
        closed = closed | fronts
    necked = core & (w_cls < P["neck_frac"] * w_ref) & ~closed
    n_core = max(1, int(core.sum()))
    closed_len_frac = float(closed.sum()) / n_core
    necked_len_frac = float(necked.sum()) / n_core
    halves = measure_halves(w_cls, closed, h, reach, flare)

    def run_center(mask):
        rr = runs_of(mask)
        if not rr:
            return np.nan
        a, b = max(rr, key=lambda r: r[1] - r[0])
        return 0.5 * (a + b) / h

    blind = False
    roof_down_frac = 0.0
    chan_conf = "HIGH"
    if not lumen_runs or f_raw < P["not_formed_present"]:
        if regime == "touchdown":
            # a wide channel always forms; when nothing of it is visible from above the roof is on
            # the floor along the whole length (review 2026-10: "formed but the membrane touched down")
            cls, roof_down_frac = "TOUCHDOWN", 1.0
            chan_conf = "HIGH" if has_port else "LOW"
        else:
            cls = "NOT_FORMED"           # ports/wedges only
    elif closed.any():
        blind = not (reaches_top and reaches_bot)
        if regime == "plug":
            kind = "plug"
        elif regime == "touchdown" or fronts.any():
            kind = "touchdown"
        else:
            # a tip that tapers to a point on either side is a plug; the roof front leaves blunt ends
            tapers_ = [v for v in (halves.get("taper_top"), halves.get("taper_bot")) if np.isfinite(v)]
            t = float(np.min(tapers_)) if tapers_ else np.nan
            kind = "plug" if (np.isfinite(t) and t < P["plug_taper"]) else "touchdown"
            if np.isfinite(t) and abs(t - P["plug_taper"]) < 0.1:
                chan_conf = "LOW"
        if kind == "plug":
            cls = "PINCHED"
        else:
            roof_down_frac = closed_len_frac
            cls = "TOUCHDOWN" if closed_len_frac >= P["touchdown_class_frac"] else ("NECKED" if necked.any() else "FORMED")
    elif necked.any():
        cls = "NECKED"
    else:
        cls = "FORMED"
    path_block = None
    if regime == "touchdown" and path is not None:
        # Wide designs (review round 2, 2026-11). A wide channel always forms; what varies is whether
        # the roof stays up. When it comes down, the side walls vanish from the top view and only the
        # open pockets at the ports remain, each closed by a contact front, so the tracker's "walls"
        # there are fronts. The class therefore rests on positive evidence that the lumen is open:
        # a free path from port to port AND two walls followed over most of the length. Without
        # both, or with a roof patch inside the lumen, the roof is down (TOUCHDOWN). A neck, however
        # narrow, with a free path through it stays NECKED.
        p_ok, p_top, p_bot = path
        core_pres = float(tr["present"][flare:h - flare].mean()) if h > 2 * flare else 0.0
        if not p_ok:
            cls = "TOUCHDOWN"
            a_, b_ = max(flare, min(p_top, p_bot)), min(h - flare, max(p_top, p_bot))
            roof_down_frac = float(np.clip((b_ - a_) / n_core, P["touchdown_class_frac"], 1.0))
            path_block = 0.5 * (a_ + b_) / h
        elif core_pres < P["open_wall_min"]:
            cls, roof_down_frac = "TOUCHDOWN", float(np.clip(1.0 - core_pres, P["touchdown_class_frac"], 1.0))
            if core_pres >= 0.2:
                chan_conf = "LOW"            # walls partly followed, no front found: could be a faint open lumen
        elif td_frac >= P["touchdown_patch_frac"]:
            cls, roof_down_frac = "TOUCHDOWN", float(td_frac)
        elif companion >= P["comp_min"]:
            # each wall is doubled by a contact line: the roof lies on the floor between two side slivers
            cls, roof_down_frac = "TOUCHDOWN", float(min(1.0, companion))
        else:
            roof_down_frac = 0.0
            cls = "NECKED" if (necked.any() or closed_len_frac >= P["touchdown_class_frac"]) else "FORMED"
        blind = False
    # ---- lumen parts, entrances and where the pinch sits (2026-11) ----------------------------
    # The lumen is what is open: rows whose width is above the closed threshold (a narrow tongue
    # still counts). If it is in two parts, one at each port, the pinch is the stretch BETWEEN them
    # and cannot lie inside either; distances are measured from the lumen entrances (roof edges).
    e_top, e_bot = ent if ent is not None else (0.0, float(h - 1))
    L = max(1.0, float(e_bot - e_top))
    c_row = 0.5 * (e_top + e_bot)
    closed_px0 = max(P["closed_frac"] * w_ref, P["closed_min_px"], tr["sup"] + P["closed_line_px"])
    lum = np.nan_to_num(w_cls, nan=0.0) >= closed_px0
    lum = ndi.binary_closing(lum, structure=np.ones(max(3, int(P["closed_min_len"] * h)) | 1))
    parts = [(a, b) for a, b in runs_of(lum) if (b - a) > wedge]
    att = P["ent_attach"] * L
    # the stretches that are NOT lumen: before the first part (if it does not start at the entrance),
    # between parts, and after the last one. The pinch is the longest of them.
    gaps = []
    if parts:
        if parts[0][0] > e_top + att:
            gaps.append((float(e_top), float(parts[0][0])))
        for (a0, b0), (a1, b1) in zip(parts, parts[1:]):
            gaps.append((float(b0), float(a1)))
        if parts[-1][1] < e_bot - att:
            gaps.append((float(parts[-1][1]), float(e_bot)))
    else:
        gaps.append((float(e_top), float(e_bot)))
    single = not gaps                   # one blob from entrance to entrance
    tip_top, tip_bot = max(gaps, key=lambda g: g[1] - g[0]) if gaps else (float(e_top), float(e_top))
    is_closed = cls in ("PINCHED", "TOUCHDOWN")
    n_parts = len(parts)
    ext = pick_pockets(ext, lum, L) if (ext is not None and is_closed and regime != "plug") else None
    if ext is not None:
        rt, rb = float(ext[0]), float(ext[1])
        if path_block is not None:
            # Wide design with no free path: the two port pockets ARE the lumen parts, and the flood
            # decides both ends. The tracker cannot: it bridges a roof-down stretch (one blob), or it
            # follows the faint side lines of the cell past the end of a short pocket, which put the
            # upper end of the pinch inside the roof-down stretch (R3 H5 W110 ML7-10, channels 2-3).
            # The tracker still arbitrates which flood threshold to trust (pick_pockets).
            tip_top, tip_bot, single = rt, rb, False
            n_parts = 2
        elif single:
            pass
        elif (tip_top <= e_top and rt >= tip_bot) or (tip_bot >= e_bot and rb <= tip_top):
            # the stretch the tracker left out lies at an entrance and inside a port pocket: it is
            # the flare of the pocket, which the tracker did not follow, not a pinch
            tip_top, tip_bot = rt, rb
        else:
            if tip_top < rt < tip_bot:
                tip_top = rt
            if tip_top < rb < tip_bot:
                tip_bot = rb
    tip_top, tip_bot = float(np.clip(tip_top, e_top, e_bot)), float(np.clip(tip_bot, e_top, e_bot))
    pm = dict(n_lumen_parts=n_parts, tip_top_row=np.nan, tip_bot_row=np.nan,
              ent_to_pinch_top_frac=np.nan, ent_to_pinch_bot_frac=np.nan, pinch_len_frac=np.nan,
              pinch_center_offset_frac=np.nan, pinch_symmetric="",
              neck_depth_top=np.nan, neck_pos_top_frac=np.nan, neck_row_top=np.nan,
              neck_depth_bot=np.nan, neck_pos_bot_frac=np.nan, neck_row_bot=np.nan,
              w_center_frac=np.nan, bulge_excess_frac=np.nan, bulge=False,
              narrow_max_frac=np.nan, narrow_mean_frac=np.nan, lumen_quality=np.nan, w_med_px=np.nan,
              ent_top_row=float(e_top), ent_bot_row=float(e_bot), lumen_len_px=L,
              part_at_top=bool(parts) and parts[0][0] <= e_top + att,
              part_at_bot=bool(parts) and parts[-1][1] >= e_bot - att)
    has_stretch = is_closed and not single and tip_bot > tip_top
    if has_stretch:
        off = (0.5 * (tip_top + tip_bot) - c_row) / L
        pm.update(tip_top_row=tip_top, tip_bot_row=tip_bot,
                  ent_to_pinch_top_frac=(tip_top - e_top) / L, ent_to_pinch_bot_frac=(e_bot - tip_bot) / L,
                  pinch_len_frac=(tip_bot - tip_top) / L, pinch_center_offset_frac=off,
                  pinch_symmetric=bool(abs(off) <= P["sym_tol"]))
        if cls == "TOUCHDOWN" and path_block is not None:
            roof_down_frac = float(np.clip((tip_bot - tip_top) / n_core, P["touchdown_class_frac"], 1.0))
    if cls in ("FORMED", "NECKED"):
        f_e = P["ent_flare"] * L
        halves_rows = (("top", int(e_top + f_e), int(c_row)), ("bot", int(c_row), int(e_bot - f_e)))
        mins = {}
        w_open = np.where(np.nan_to_num(w_cls, nan=0.0) >= closed_px0, w_cls, np.nan)   # the channel is open: zeros are bookkeeping
        for nm, a, b in halves_rows:
            seg = w_open[max(0, a):max(0, b)]
            if seg.size and np.isfinite(seg).any():
                i = int(np.nanargmin(seg))
                mins[nm] = float(seg[i])
                row = max(0, a) + i
                pm["neck_depth_" + nm] = float(np.clip(1.0 - seg[i] / w_ref, 0.0, 1.0))
                pm["neck_row_" + nm] = float(row)
                pm["neck_pos_" + nm + "_frac"] = float((row - e_top) / L if nm == "top" else (e_bot - row) / L)
        # Narrowing of an open channel, two summaries of a width that varies along the length: the
        # deepest point (what limits the flow) and the mean over the length between the two port
        # flares (how much of the lumen is lost overall; rows wider than the reference count as 0,
        # so a bulge does not cancel a neck). Both as fractions of the reference width.
        depths = [pm["neck_depth_" + nm] for nm in ("top", "bot") if np.isfinite(pm["neck_depth_" + nm])]
        if depths:
            pm["narrow_max_frac"] = float(max(depths))
        seg = w_open[max(0, int(e_top + f_e)):max(0, int(e_bot - f_e))]
        if seg.size and np.isfinite(seg).any():
            pm["narrow_mean_frac"] = float(1.0 - np.nanmean(np.minimum(seg / w_ref, 1.0)))
            pm["w_med_px"] = float(np.nanmedian(seg))
            # Lumen quality of a channel that is open end to end: 1 minus the mean distance of its
            # width from the reference width along the length. A neck and a bulge both lower it; 1 is
            # a lumen at the reference width from port flare to port flare.
            pm["lumen_quality"] = float(1.0 - np.nanmean(np.minimum(np.abs(seg / w_ref - 1.0), 1.0)))
        ca, cb = int(c_row - 0.1 * L), int(c_row + 0.1 * L)
        seg = w_open[max(0, ca):max(0, cb)]
        if seg.size and np.isfinite(seg).any():
            wc_ = float(np.nanmedian(seg))
            pm["w_center_frac"] = wc_ / w_ref
            if len(mins) == 2:
                pm["bulge_excess_frac"] = (wc_ - max(mins.values())) / w_ref
                pm["bulge"] = bool(pm["bulge_excess_frac"] >= P["bulge_min"])
    if has_stretch:
        defect_y = 0.5 * (tip_top + tip_bot) / h       # between the two lumen parts, never inside one
    elif is_closed:
        defect_y = np.nan                               # one blob (roof patch, doubled walls): no single pinch row
    elif cls == "NECKED":
        deeper = max(("top", "bot"), key=lambda nm: (pm["neck_depth_" + nm] if np.isfinite(pm["neck_depth_" + nm]) else -1))
        defect_y = pm["neck_row_" + deeper] / h if np.isfinite(pm["neck_row_" + deeper]) else run_center(necked)
    else:
        defect_y = np.nan
    if not has_port and cls != "NOT_FORMED":
        chan_conf = "LOW"
    if longest_fill > P["fill_low_conf"]:
        chan_conf = "LOW"                # a long bridged stretch: the class rests on interpolation
    return dict(cls=cls, w_cls=w_cls, f_present=f_present, reaches_top=reaches_top, reaches_bot=reaches_bot,
                w_min=w_min, w_min_frac=w_min_frac, w_min_y=w_min_y, closed_len_frac=closed_len_frac,
                necked_len_frac=necked_len_frac, defect_y=defect_y, open_len_frac=f_raw,
                channel_confidence=chan_conf, longest_fill=longest_fill, blind=blind, regime=regime,
                roof_down_frac=roof_down_frac, **halves, **pm)


def wall_span(tr, present, h):
    """Straight-wall estimate of the lumen span: the median tracked wall positions over the rows
    where both walls were followed (the tracks wander where they were not)."""
    if present.sum() < max(5, 0.05 * h):
        return None
    xl = int(np.median(tr["xL"][present])) + 2
    xr = int(np.median(tr["xR"][present])) - 2
    return (xl, xr) if xr - xl >= 4 else None


def roof_front_rows(tr, R_full, arc, blob, h, flare, present):
    """Rows where the roof is on the floor, read from fronts that cross the lumen.

    Where the roof membrane meets the floor it draws a dark front across the channel (a U or a
    bubble outline), while the side walls stay where they are, so the tracker reports an open lumen
    of full width. A front is a row whose dark structure covers nearly the whole width between the
    walls (P['cap_cover']), outside the port flares. Two or more fronts enclose the roof-down
    stretch; a single front separates the open stub (the side with the tracked walls) from the
    roof-down rest."""
    mask = np.zeros(h, bool)
    span = wall_span(tr, present, h)
    if span is None:
        return mask
    a, b = span
    capmap = arc if blob is None else np.maximum(arc, blob)
    strong = np.maximum(capmap[:, a:b], R_full[:, a:b]) > P["cap_tau"]
    cover = strong.mean(axis=1)
    cover[:flare] = 0
    cover[h - flare:] = 0
    rows = np.where(cover > P["cap_cover"])[0]
    if rows.size == 0:
        return mask
    first, last = int(rows[0]), int(rows[-1])
    if last - first > 0.05 * h:
        mask[first:last] = True
    else:
        y0 = (first + last) // 2
        if present[:y0].sum() >= present[y0:].sum():
            mask[y0:h - flare] = True        # open stub from the top face, roof down beyond the front
        else:
            mask[flare:y0] = True
    return mask


def lumen_path(line, arc, tr, xc, h, w_exp):
    """Is there a way through the channel from one port to the other without crossing a dark outline?

    An open channel is a corridor between two walls. Where the roof rests on the floor the contact
    line closes the corridor: the open pocket at each port ends at a front (a funnel, a tongue, a
    bell), and no free path joins the two ports. The test floods the free space of the channel cell
    from a seed just inside each port and asks whether the two floods meet. Dark outlines above
    P['path_tau'] are barriers (specks removed, small gaps sealed); the cell sides are barriers too,
    so a path cannot leave the channel and come back.

    Returns (connected, reach_top, reach_bot): the last row reached from the top port and the first
    row reached from the bottom port (rows of the device); between them the roof is down."""
    m = int(P["path_margin"] * h)
    return _path_flood(line, arc, tr, xc, h, w_exp, P["path_tau"], m, h - m)[:3]


def pocket_extents(line, arc, tr, xc, h, w_exp, e_top, e_bot):
    """Candidates for where the open pocket at each port ends. The flood of lumen_path is repeated
    from seeds just inside the two entrances with the outline threshold lowered step by step
    (P['extent_taus']); a faint contact line that leaks at the strict threshold usually holds at a
    looser one. Returns one tuple per threshold at which the two floods stay apart, strict first:
    (last row of the top pocket, first row of the bottom pocket, last row of the top pocket's
    outline, first row of the bottom pocket's outline, fine), or None. `fine` is the same flood with
    a finer sealing of the outline gaps (P['path_seal_fine']): its two ends and the rows where each
    flood is not the clean inside of a pocket, or None if its floods met. pick_pockets() chooses."""
    L = max(1.0, float(e_bot - e_top))
    m_top, m_bot = int(e_top + 0.03 * L), int(e_bot - 0.03 * L)
    cands = []
    for tau in P["extent_taus"]:
        ok, rt, rb, ot, ob = _path_flood(line, arc, tr, xc, h, w_exp, tau, m_top, m_bot)
        if not ok:
            fine = _path_flood(line, arc, tr, xc, h, w_exp, tau, m_top, m_bot, seal=P["path_seal_fine"], sides=True)
            cands.append((rt, rb, ot, ob, None if fine[0] else (fine[1], fine[2], fine[5])))
    return cands or None


def pick_pockets(cands, lum, L):
    """Choose the pocket ends among the candidates of pocket_extents(): (end of the top pocket,
    start of the bottom pocket), or None.

    The strictest threshold whose floods do not leak wins. A flood has leaked when it crosses the
    other one, or when a looser threshold ends a pocket much earlier (P['leak_jump']) and the
    tracked walls are missing over a long run (P['leak_gap']) of the stretch in dispute: a pocket
    has walls, the roof-down stretch behind a leaking contact line has none. Where the tracked
    walls continue through the disputed stretch, the looser threshold has cut a real lumen at a
    speck, and the stricter one stands. `lum` marks the rows where the tracker sees a lumen."""
    if not cands:
        return None
    n = len(lum)

    def empty_run(a, b):
        a, b = int(np.clip(a, 0, n)), int(np.clip(b, 0, n))
        rr = runs_of(~lum[a:b]) if b > a else []
        return max((q[1] - q[0] for q in rr), default=0)

    def clean_gain(end, new, unclean, top):
        # the finer flood reaches farther than `end`; take it when the rows it adds are the clean
        # inside of a pocket (no long run that touches a corridor side or seeps through grain)
        a, b = (end, new) if top else (new, end)
        if unclean is None or b - a <= 0:
            return end
        a_, b_ = int(np.clip(a, 0, len(unclean))), int(np.clip(b, 0, len(unclean)))
        rr = runs_of(unclean[a_:b_]) if b_ > a_ else []
        return new if max((q[1] - q[0] for q in rr), default=0) <= P["leak_gap"] * L else end

    for i, (rt, rb, ot, ob, fine) in enumerate(cands):
        if rb - rt < P["pinch_min"] * L:
            continue                                    # the floods crossed or touch
        if i + 1 < len(cands):
            nrt, nrb = cands[i + 1][0], cands[i + 1][1]
            leak = False
            for a, b in ((nrt, rt), (rb, nrb)):         # rows this threshold adds to the top and to the bottom pocket
                if b - a > P["leak_jump"] * L and empty_run(a, b) > P["leak_gap"] * L:
                    leak = True
            if leak:
                continue
        # a pocket often ends in a spike too thin to flood: its outline reaches farther than its
        # interior. Use the outline ends when they lie between the two interiors and stay apart.
        t, b = (float(ot), float(ob)) if (ot is not None and ob is not None and rt <= ot < ob <= rb) else (float(rt), float(rb))
        # A tongue that hangs from a funnel through a narrow neck is cut off by the regular sealing
        # (R2 H5 W160 ML4-6, channel 3: the funnel ends at 0.25 of the length, its tongue at 0.63).
        # The finer flood follows it; its extra rows count when they are a clean inside.
        if fine is not None:
            t2 = clean_gain(t, float(fine[0]), fine[2][0], True) if fine[0] > t else t
            b2 = clean_gain(b, float(fine[1]), fine[2][1], False) if fine[1] < b else b
            if b2 - t2 >= P["pinch_min"] * L:
                t, b = t2, b2
        return t, b
    return None


def _path_flood(line, arc, tr, xc, h, w_exp, tau, m, m_bot, seal=None, sides=False):
    """Flood of the channel cell from a seed row m (top) and m_bot (bottom); see lumen_path.
    `seal` is the size of the closing that seals gaps in the outlines (x design width, default
    P['path_seal']). With `sides`, a sixth value is returned: for the top and for the bottom flood,
    the rows where it is not the clean inside of a pocket, because it touches a side of the corridor
    (not enclosed by an outline) or fills less than P['leak_fill'] of its own extent in that row
    (it seeps through surface grain)."""
    present = tr["present"]
    span = wall_span(tr, present, h)
    pad = max(6.0, P["path_pad"] * w_exp)
    if span is not None and (span[1] - span[0]) >= 0.5 * w_exp:
        xa, xb = span[0] - pad, span[1] + pad
    else:
        xa, xb = xc - 0.5 * w_exp - pad, xc + 0.5 * w_exp + pad
    xa, xb = int(max(tr["x_lo"], xa)), int(min(tr["x_hi"], xb))
    m, m_bot = int(max(0, m)), int(min(h, m_bot))
    # the seeds must sit inside the lumen: center them on the tracked walls when there are any (the
    # lattice position can be a fraction of the width off, and a seed outside the pocket floods the
    # bulk around it and reports a free path)
    seed_half = 0.2 * w_exp
    if span is not None and (span[1] - span[0]) >= 0.3 * w_exp:
        xc = 0.5 * (span[0] + span[1])
        seed_half = min(seed_half, 0.25 * (span[1] - span[0]))
    none = (True, h, 0, None, None) + ((None,) if sides else ())
    if xb - xa < 8 or m_bot - m < 10:
        return none
    B = (np.maximum(line, arc)[:, xa:xb] > tau)
    lab, n = ndi.label(B, structure=np.ones((3, 3)))
    if n:
        keep = np.zeros(n + 1, bool)
        min_len = max(8.0, P["path_speck"] * w_exp)
        for i, sl in enumerate(ndi.find_objects(lab), 1):
            if sl is not None and max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) >= min_len:
                keep[i] = True
        B = keep[lab]
    k = int(max(5 if seal is None else 3, round((P["path_seal"] if seal is None else seal) * w_exp))) | 1
    B = cv2.morphologyEx(B.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    free = ~B
    free[:m] = False
    free[m_bot:] = False
    labf, nf = ndi.label(free)                      # 4-connected: a diagonal outline still blocks
    if nf == 0:
        return (False, m, m_bot, None, None) + ((None,) if sides else ())
    c0, c1 = int(max(0, xc - xa - seed_half)), int(min(xb - xa, xc - xa + seed_half))
    if c1 - c0 < 2:
        c0, c1 = 0, xb - xa
    top_l = set(np.unique(labf[m:m + 3, c0:c1])) - {0}
    bot_l = set(np.unique(labf[m_bot - 3:m_bot, c0:c1])) - {0}
    if top_l & bot_l:
        return none
    rows_any = lambda labels: np.where(np.isin(labf, list(labels)).any(axis=1))[0] if labels else np.array([], int)
    rt, rb = rows_any(top_l), rows_any(bot_l)
    reach_top = int(rt.max()) if rt.size else m
    reach_bot = int(rb.min()) if rb.size else m_bot
    labB, _ = ndi.label(B, structure=np.ones((3, 3)))

    def outline_end(labels, top):
        """Last (top pocket) or first (bottom pocket) row of the outline pieces that bound a flood."""
        if not labels:
            return None
        near = cv2.dilate(np.isin(labf, list(labels)).astype(np.uint8), np.ones((k + 4, k + 4), np.uint8)) > 0
        ids = np.unique(labB[near & B])
        ids = ids[ids > 0]
        if not ids.size:
            return None
        rows = np.where(np.isin(labB, ids).any(axis=1))[0]
        rows = rows[(rows >= m) & (rows < m_bot)]
        if not rows.size:
            return None
        return int(rows.max()) if top else int(rows.min())

    out = (False, reach_top, reach_bot, outline_end(top_l, True), outline_end(bot_l, False))
    if sides:
        def unclean_rows(labels):
            mm = np.isin(labf, list(labels)) if labels else np.zeros(labf.shape, bool)
            cnt = mm.sum(axis=1)
            first = mm.argmax(axis=1)
            last = mm.shape[1] - 1 - mm[:, ::-1].argmax(axis=1)
            porous = (cnt > 0) & (cnt < P["leak_fill"] * (last - first + 1))
            return mm[:, 0] | mm[:, -1] | porous
        out += ((unclean_rows(top_l), unclean_rows(bot_l)),)
    return out


def find_entrances(N, arc, tracks, h, w, w_exp):
    """Rows of the two lumen entrances: the roof edge at the top face and at the bottom face.

    A device shows two horizontal edges near each face. The outer one is the edge of the base the
    walls stand on; it runs across the whole device. The inner one is the edge of the roof, where
    the lumen opens: a dark horizontal line (often two dark corners) across the mouth of each
    channel and absent over the solid walls between channels. The channel length that matters is
    the one between the two roof edges, so every along-channel distance is referred to them.

    For each channel mouth the dark fraction of every row in the face band is taken, minus the
    dark fraction of the same row over the walls between channels (this removes the base edge).
    The run of rows around the darkest one, above P['ent_min'], is that channel's edge. The roof edge is one
    straight line per face, so the device value is the median over the channels that show it, and a
    channel where the edge did not form inherits it.

    Returns dict(top=row, bot=row, det_top=[bool per track], det_bot=[...])."""
    band = int(P["ent_band"] * h)
    dark = ((N < P["crescent_dark"]) | (arc > P["front_tau"])).astype(np.float32)
    chan = np.zeros(w, bool)
    cols = []
    for t in tracks:
        half = 0.5 * max(w_exp, t["w_port"], 4.0) + 2
        a, b = int(max(0, t["x"] - half)), int(min(w, t["x"] + half + 1))
        cols.append((a, b))
        chan[a:b] = True
    wall = ~ndi.binary_dilation(chan, iterations=int(max(2, 0.15 * w_exp)))
    bx = int(P["border_skip"] * w)
    wall[:bx] = False
    wall[w - bx:] = False
    out = {}
    k = int(max(3, 0.006 * h)) | 1
    for name, rows in (("top", slice(0, band)), ("bot", slice(h - band, h))):
        d = dark[rows]
        ref = d[:, wall].mean(axis=1) if wall.sum() > 10 else np.zeros(d.shape[0], np.float32)
        per = []
        for a, b in cols:
            if b - a < 2:
                per.append(None)
                continue
            sc = ndi.uniform_filter1d(d[:, a:b].mean(axis=1) - ref, k)
            thr = max(P["ent_min"], 0.5 * float(sc.max()))
            rr = runs_of(sc >= thr)
            if not rr:
                per.append(None)
                continue
            i_max = int(np.argmax(sc))                          # the run that holds the darkest row
            ra, rb_ = next((q for q in rr if q[0] <= i_max < q[1]), rr[0])
            per.append(rows.start + 0.5 * (ra + rb_ - 1))
        found = [q for q in per if q is not None]
        d0 = int(P["ent_default"] * h)
        row = float(np.median(found)) if found else float(d0 if name == "top" else h - 1 - d0)
        out[name] = row
        out["det_" + name] = [q is not None and abs(q - row) <= P["ent_tol"] * h for q in per]
    return out


def port_stub(N, line, arc, blob, tr, h, w, w_exp, e_row, e_other, top):
    """Shape of the lumen remnant at one port of a channel whose lumen did not form there.

    A channel that did not form still shows a short stub at each port: a dark cup or crescent a
    fraction of the design width deep, shorter than the port wedge, so the wall tracker does not
    count it as lumen. Its shape is read from the image itself: dark body pixels and dark outlines
    in a window around the channel mouth are joined, holes are filled, and an opening removes what
    is thinner than P['stub_open'] x the design width (cracks, the roof edge, the base edge). The
    stub is the largest remaining piece that begins at the entrance; it is cut at the entrance row,
    so the base outside the roof edge is not part of it. In a frame with strong surface texture
    the joined pixels fill the window; such a piece is rejected and the stub is left unmeasured.

    Returns dict(len_px, w_px, area_px, contours) or None: length from the entrance to the tip,
    width (90th percentile of the row widths), area, and the outline for the overlay."""
    half = P["stub_hw"] * max(w_exp, tr["w_port"], 8.0)
    xa, xb = int(max(tr["x_lo"], tr["x"] - half, 0)), int(min(tr["x_hi"], tr["x"] + half, w))
    tol = int(max(3, P["stub_tol"] * h))
    far = e_row + (1 if top else -1) * P["stub_max_len"] * abs(e_other - e_row)
    ya, yb = (int(max(0, e_row - tol)), int(far)) if top else (int(far), int(min(h, e_row + tol + 1)))
    if xb - xa < 6 or yb - ya < 6:
        return None
    sl = (slice(ya, yb), slice(xa, xb))
    body = (N[sl] < P["crescent_dark"]) | (blob[sl] > P["stub_blob"]) | (np.maximum(line[sl], arc[sl]) > P["path_tau"])
    body = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    body = ndi.binary_fill_holes(body)
    k = int(max(3, round(P["stub_open"] * w_exp))) | 1
    body = cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    e_loc = int(round(e_row - ya))
    if top:
        body[:max(0, e_loc)] = False                    # outside the roof edge: the base, not the lumen
    else:
        body[min(body.shape[0], e_loc + 1):] = False
    lab, n = ndi.label(body)
    best, best_area = 0, 0
    for i, s in enumerate(ndi.find_objects(lab), 1):
        starts = (s[0].start <= e_loc + tol) if top else (s[0].stop >= e_loc - tol)
        area = int((lab[s] == i).sum())
        if starts and area > best_area:
            best, best_area = i, area
    if not best:
        return None
    m = lab == best
    # a stub is a compact shape inside the window. A piece that runs out of the window at its far
    # end, or spans it from side to side, is surface texture or a lumen, not a stub: not measured.
    runs_out = m[-1].any() if top else m[0].any()
    if runs_out or (m[:, 0].any() and m[:, -1].any()):
        return None
    rows = np.where(m.any(axis=1))[0]
    wid = m.sum(axis=1)
    tip = (rows.max() if top else rows.min()) + ya
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return dict(len_px=float(abs(tip - e_row)), w_px=float(np.percentile(wid[wid > 0], 90)), area_px=float(best_area),
                contours=[c[:, 0, :] + np.array([xa, ya]) for c in cs])


def wall_companions(line, tr, h, w_exp):
    """Length of the longest dark line that runs beside a tracked wall, as a fraction of the length.

    A thin roof that rests on the floor over the middle of a wide channel leaves a narrow open
    sliver along each side wall. From above each wall then shows as two parallel lines a few
    percent of the width apart (the wall and the contact line), with nothing between the two
    slivers. An open channel shows one line per wall. The search band reaches P['comp_band'] x the
    design width to either side of each tracked wall, without the wall line itself."""
    ys = np.arange(h)
    band = int(max(6, round(P["comp_band"] * w_exp)))
    skip = int(tr["sup"]) + 3
    B = line > P["path_tau"]
    best = 0.0
    m = int(P["front_margin"] * h)
    for path in (tr["xL"], tr["xR"]):
        strip = np.zeros((h, 2 * band + 1), bool)
        for d in range(-band, band + 1):
            if abs(d) <= skip:
                continue
            xs = np.clip(path + d, 0, line.shape[1] - 1)
            strip[:, d + band] = B[ys, xs]
        strip[:m] = False
        strip[h - m:] = False
        lab, n = ndi.label(strip, structure=np.ones((3, 3)))
        for sl in ndi.find_objects(lab):
            best = max(best, (sl[0].stop - sl[0].start) / float(h))
    return best


def touchdown(line, arc, tr, h):
    """Fraction of the lumen area enclosed by an inner closed dark outline: a bubble under the roof
    or a patch of roof on the floor. Both mean the lumen is not open there. The lumen span is taken
    from the straight walls over the whole stretch between the first and last tracked rows, so a
    bubble inside a bridged gap counts too."""
    present = tr["present"]
    if present.sum() < 0.1 * h:
        return 0.0, None
    span = wall_span(tr, present, h)
    if span is None:
        return 0.0, None
    ys = np.where(present)[0]
    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    inside = np.zeros(line.shape, bool)
    for y in range(y0, y1):
        if present[y]:
            a, b = int(tr["xL"][y]) + 3, int(tr["xR"][y]) - 2
        else:
            a, b = span
        if b > a:
            inside[y, a:b] = True
    if inside.sum() < 50:
        return 0.0, None
    M = (np.maximum(line, arc) > P["touchdown_tau"]) & inside
    M = ndi.binary_closing(M, structure=np.ones((3, 3)))
    free = inside & ~M
    lab, n = ndi.label(free)
    if n == 0:
        return 0.0, None
    # components of free space that touch the lumen boundary are the open lumen;
    # the rest are enclosed by an inner outline
    border = inside & ~ndi.binary_erosion(inside, structure=np.ones((3, 3)))
    touching = set(np.unique(lab[border & free]))
    enclosed = np.isin(lab, [i for i in range(1, n + 1) if i not in touching]) & free
    enclosed = enclosed | (M & ndi.binary_dilation(enclosed, iterations=2))
    frac = float(enclosed.sum()) / float(inside.sum())
    return frac, enclosed


def enforce_count(channels, k, pitch, w, w_exp, faces):
    """The frame name says how many channels it shows (its ML range). Refit the pitch on the
    channels that have port evidence, snap every channel to that lattice, and keep the window of
    k consecutive slots with the most port evidence, filling empty slots inside it with predicted
    cells. A missed channel then keeps its number and an extra cell at the frame edge is dropped.
    Real port pairs are never dropped; with more of them than k the channels come back unchanged."""
    if not k or pitch is None or not channels or len(channels) == k:
        return channels

    def score(c):
        if c["top"] and c["bot"] and not c.get("predicted"):
            return 3
        if (c["top"] or c["bot"]) and c.get("predicted") != "lattice":
            return 2
        return 0

    evid = sorted([c for c in channels if score(c) >= 2], key=lambda c: c["x"])
    x_ref = evid[0]["x"] if evid else channels[0]["x"]
    # the design pitch (wall + W) can be off by 10%, enough to drift a slot over ten cells: refit
    # it on the channels that show ports
    if len(evid) >= 3:
        xs = np.array([c["x"] for c in evid])
        dif = np.diff(xs)
        mult = np.maximum(np.round(dif / pitch), 1)
        p_fit = float(np.median(dif / mult))
        if 0.75 * pitch < p_fit < 1.33 * pitch:
            idx = np.concatenate([[0], np.cumsum(mult)])
            A = np.column_stack([idx, np.ones_like(idx)])
            (p_ls, x_ls), *_ = np.linalg.lstsq(A, xs, rcond=None)
            if 0.75 * pitch < p_ls < 1.33 * pitch:
                pitch, x_ref = float(p_ls), float(x_ls)
    slots = {}
    for c in channels:
        kk = int(round((c["x"] - x_ref) / pitch))
        if kk not in slots or score(c) > score(slots[kk]):
            slots[kk] = c
    # an edge cell may sit inside the detection border; let the lattice reach it
    half = 0.5 * max(w_exp, 4.0)
    lo_x, hi_x = max(half, 0.01 * w), w - max(half, 0.01 * w)
    kmin = int(np.ceil((lo_x - x_ref) / pitch))
    kmax = int(np.floor((hi_x - x_ref) / pitch))
    if kmax - kmin + 1 < k:          # the frame cannot hold k cells on this lattice
        return channels
    must = [kk for kk, c in slots.items() if score(c) == 3]
    best, best_key = None, None
    for start in range(kmin, kmax - k + 2):
        window = range(start, start + k)
        if any(m not in window for m in must):
            continue
        val = sum(score(slots[kk]) for kk in window if kk in slots)
        center = x_ref + (start + (k - 1) / 2) * pitch
        key = (val, -abs(center - w / 2))       # ties: the window centered in the frame
        if best is None or key > best_key:
            best, best_key = start, key
    if best is None:
        return channels
    all_ports = faces[0] + faces[1]
    w_med = float(np.median([c["w_port"] for c in channels]))
    out = []
    for kk in range(best, best + k):
        if kk in slots:
            out.append(slots[kk])
            continue
        xp = x_ref + kk * pitch
        near = [p for p in all_ports if abs(p["x"] - xp) < P["lattice_tol"] * pitch]
        if near:
            p = max(near, key=lambda q: q["mass"])
            out.append(dict(top=p if p in faces[0] else None, bot=p if p in faces[1] else None,
                            x=p["x"], x_top=p["x"], x_bot=p["x"], w_port=p["width"], predicted="port"))
        else:
            out.append(dict(top=None, bot=None, x=xp, x_top=xp, x_bot=xp,
                            w_port=w_exp if w_exp >= P["narrow_design_px"] else w_med, predicted="lattice"))
    if os.environ.get("TV_DEBUG"):
        print(f"   enforce_count: {len(channels)} -> {k}, pitch {pitch:.1f}, slots {sorted(slots)}, window {best}..{best + k - 1}")
    return out


def wide_lattice(N, line, arc, channels, k, pitch, pitch_exp, faces, h, w, w_exp):
    """Place the k channels of a wide design on the lattice that best fits the port pockets.

    When the roof of a wide channel is down, its ports are not crescents but open pockets (bells,
    funnels) whose outlines the port detector splits into fragments, and the lattice phased on such
    fragments can sit a fraction of a pitch off (review round 2: "wrong locate"). Two cues survive
    a collapsed roof. (1) Each pocket is a patch of dark outline at a face, as wide as the channel:
    the dark fraction of the two face bands, averaged over a channel-wide window, peaks at each
    channel center. (2) The roof thickness steps from one channel to the next along a straight line
    midway between them, which shows as a thin dark line over the full length. The lattice (first
    center, pitch) that maximizes pocket evidence at the k centers plus step-line evidence at the
    k-1 midpoints is compared with the channels found from the ports; it replaces them only when
    it scores clearly better and actually differs (P['lat_gain'], P['lat_shift']).

    The same two cues hold in narrow designs (2026-11-19, R1 H5 W7 ML1-10: the port-based lattice
    skipped the first lumen and put predicted channels on the step lines). A pinched lumen leaves a
    tapered stub at each port and nothing at mid-length, while a step line is straight and dark over
    the full length, so the fit now runs on every design at least P['lat_min_w_px'] wide (0 = all).

    Returns (channels, note)."""
    if not k or k < 2 or pitch_exp is None:
        return channels, ""
    band = int(P["lat_band"] * h)
    dark = ((N < P["crescent_dark"]) | (np.maximum(line, arc) > P["front_tau"])).astype(np.float32)
    ev = np.zeros(w, np.float32)
    cen = dark[int(0.35 * h):int(0.65 * h)].mean(axis=0)     # what is dark at mid-length too is a wall or a frame edge, not a pocket
    for sl in (slice(0, band), slice(h - band, h)):
        sc = dark[sl].mean(axis=0) - cen
        ev += np.clip(sc - np.median(sc), 0, None)
    Eb = ndi.uniform_filter1d(ev, int(max(5, round(w_exp))) | 1, mode="constant")
    Eb = Eb / max(float(Eb.max()), 1e-6)
    mid = (line[int(0.25 * h):int(0.75 * h)] > P["lat_line_tau"]).mean(axis=0).astype(np.float32)

    def score(xs, pc):
        xs = np.asarray(xs, float)
        if xs.min() < P["lat_edge"] * w_exp or xs.max() > w - 1 - P["lat_edge"] * w_exp:
            return -1.0
        s1 = float(Eb[np.clip(np.round(xs).astype(int), 0, w - 1)].sum())
        r = int(max(2, round(0.03 * pc)))
        s2 = 0.0
        for xm in 0.5 * (xs[1:] + xs[:-1]):
            a, b = int(max(0, xm - r)), int(min(w, xm + r + 1))
            s2 += float(mid[a:b].max()) if b > a else 0.0
        return s1 + P["lat_mid_weight"] * s2

    # the pitch read from the ports spreads 0.8-1.25 x the design pitch across frames (scale bar, wall width)
    cand_p = [pitch_exp * f for f in np.linspace(1 - P["lat_pitch_tol"], 1 + P["lat_pitch_tol"], 41)]
    best, best_s = None, -1.0
    for pc in cand_p:
        span = (k - 1) * pc
        for x0 in np.arange(P["lat_edge"] * w_exp, w - 1 - P["lat_edge"] * w_exp - span, 2.0):
            xs = x0 + pc * np.arange(k)
            sc = score(xs, pc)
            if sc > best_s:
                best, best_s = (xs, pc), sc
    if best is None:
        return channels, ""
    xs_new, p_new = best
    xs_old = np.array(sorted(c["x"] for c in channels), float)
    s_old = score(xs_old, pitch or pitch_exp) if len(xs_old) == k else -1.0
    shift = float(np.max(np.abs(xs_new - xs_old))) / p_new if len(xs_old) == k else 9.9
    note = f"lattice fit {best_s:.2f} vs ports {s_old:.2f}, shift {shift:.2f} pitch"
    if os.environ.get("TV_DEBUG"):
        print("   wide_lattice:", note, "new", [round(x) for x in xs_new], "old", [round(x) for x in xs_old])
    if best_s < P["lat_gain"] * max(s_old, 0.0) or shift < P["lat_shift"] or best_s < P["lat_min_score"] * k:
        return channels, note + " (kept)"
    out = []
    for x in xs_new:
        near = [[q for q in f_ if abs(q["x"] - x) < 0.5 * w_exp] for f_ in faces]
        top = max(near[0], key=lambda q: q["mass"]) if near[0] else None
        bot = max(near[1], key=lambda q: q["mass"]) if near[1] else None
        out.append(dict(top=top, bot=bot, x=float(x), x_top=float(x), x_bot=float(x), w_port=float(w_exp),
                        predicted="" if (top or bot) else "lattice"))
    return out, note + " (replaced)"


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def process_image(path, out_dir, meta=None, save_debug=True):
    name = os.path.splitext(os.path.basename(path))[0]
    Hl, Wd, rep = parse_name(path)
    if meta:
        Hl = Hl or (int(meta["H_layers"]) if meta.get("H_layers") else None)
        Wd = Wd or (int(meta["W_px"]) if meta.get("W_px") else None)
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    gray, bar_px = strip_footer(gray)
    scale_note = ""
    if bar_px >= 0.03 * gray.shape[1]:
        um_per_px = P["bar_mm"] * 1000.0 / bar_px
    else:
        um_per_px = P["um_per_px_default"]
        scale_note = f"scale bar not found ({bar_px} px): default {um_per_px} um/px used"
    rot, (x0, y0, x1, y1), ang = detect_device(gray)
    dev = rot[y0:y1, x0:x1]
    h, w = dev.shape
    # check the scale against the device length (see P['scale_q_lo'])
    um_per_px_bar, scale_source = um_per_px, "bar" if not scale_note else "default"
    len_design_um = P["design_rows"] * 2 * P["design_post_px"] * P["pixel_um"]
    q_scale = h * um_per_px / len_design_um
    if not (P["scale_q_lo"] <= q_scale <= P["scale_q_hi"]) and h >= P["scale_min_crop"] * gray.shape[0]:
        um_per_px = um_per_px * P["scale_q_ref"] / q_scale
        scale_source = "device length"
        scale_note = (scale_note + "; " if scale_note else "") + \
            f"scale bar disagrees with the device length (q {q_scale:.2f}): {um_per_px_bar:.2f} -> {um_per_px:.2f} um/px from the device length"
    w_exp = (Wd * P["pixel_um"] / um_per_px) if (Wd and np.isfinite(um_per_px)) else 30.0
    N, line, arc, blob, R = dark_maps(dev)
    pitch_exp = (P["wall_mm"] * 1000.0 / um_per_px + w_exp) if np.isfinite(um_per_px) else None
    _, ml_a, ml_b = parse_frame(path)
    n_expected = abs(ml_a - ml_b) + 1 if ml_a is not None else None
    channels, walls, pitch, diag, faces = find_channels(N, line, arc, blob, h, w, w_exp, pitch_exp, n_expected)
    channels = enforce_count(channels, n_expected, pitch, w, w_exp, faces)
    diag["n_expected"] = n_expected
    if Wd and Wd >= P["lat_min_w_px"] and n_expected:
        channels, diag["lattice"] = wide_lattice(N, line, arc, channels, n_expected, pitch, pitch_exp, faces, h, w, w_exp)
        if "replaced" in diag["lattice"] and len(channels) >= 2:
            pitch = float(np.median(np.diff([c["x"] for c in channels])))

    # pass 1: track every channel
    tracks = []
    xs = [c["x"] for c in channels]
    for i, c in enumerate(channels):
        half = 0.5 * pitch if pitch else max(0.7 * w_exp, 0.5 * w / max(1, len(channels)))
        lo = 0.5 * (xs[i - 1] + xs[i]) if i > 0 else xs[i] - half
        hi = 0.5 * (xs[i] + xs[i + 1]) if i < len(xs) - 1 else xs[i] + half
        for xw in walls:
            if lo < xw < c["x"]:
                lo = max(lo, xw)
            if c["x"] < xw < hi:
                hi = min(hi, xw)
        if w_exp >= P["narrow_design_px"]:
            lo = max(lo, c["x"] - P["roi_cap"] * w_exp)
            hi = min(hi, c["x"] + P["roi_cap"] * w_exp)
            c = dict(c, w_port=max(c["w_port"], P["anchor_min_frac"] * w_exp))
        # the tracking window stays off the frame border, except for an edge cell whose lumen
        # itself reaches into the border
        reach_x = 0.8 * max(w_exp, c["w_port"])
        bxl = int(min(P["border_skip"] * w + 2, max(2, c["x"] - reach_x)))
        bxr = int(min(P["border_skip"] * w + 2, max(2, w - c["x"] - reach_x)))
        x_lo = int(max(bxl, lo + P["roi_margin_px"]))
        x_hi = int(min(w - bxr, hi - P["roi_margin_px"]))
        if x_hi - x_lo < 6:
            continue
        w_max = None if w_exp >= P["narrow_design_px"] else max(4 * w_exp, 3) + 6
        tr = track_channel(R, c, x_lo, x_hi, h, w_max)
        tr.update(index=len(tracks) + 1, x=c["x"], x_frac=c["x"] / w, has_top=c["top"] is not None,
                  has_bot=c["bot"] is not None, predicted=c.get("predicted", ""))
        tracks.append(tr)

    # device-level reference width: the open width of the channels that did form
    e = int(P["edge_exclude"] * h)
    refs = [np.nanpercentile(t["width"][e:h - e], 85) for t in tracks
            if t["present"][e:h - e].mean() >= 0.3]
    refs = [r for r in refs if np.isfinite(r)]
    w_ref = float(np.median(refs)) if refs else float(w_exp)
    w_ref = max(w_ref, 2.0)
    if w_exp < P["narrow_design_px"]:
        w_ref = min(w_ref, max(4 * w_exp, 3) + 6)

    # pass 2: classify + touchdown
    results = []
    entr = find_entrances(N, arc, tracks, h, w, w_exp) if tracks else dict(top=0.0, bot=float(h - 1), det_top=[], det_bot=[])
    for i_tr, tr in enumerate(tracks):
        td_frac, td_mask = touchdown(line, arc, tr, h)
        lpath = lumen_path(line, arc, tr, tr["x"], h, w_exp)
        comp = wall_companions(line, tr, h, w_exp)
        ext = pocket_extents(line, arc, tr, tr["x"], h, w_exp, entr["top"], entr["bot"])
        cl = classify_channel(tr, h, w_ref, arc, w_exp, blob, R, W_design=Wd,
                              has_port=bool(tr["has_top"] or tr["has_bot"]), path=lpath, td_frac=td_frac,
                              companion=comp, ent=(entr["top"], entr["bot"]), ext=ext)
        tr.update(cl)
        span_ = wall_span(tr, tr["present"], h)
        tr.update(ent_det_top=bool(entr["det_top"][i_tr]), ent_det_bot=bool(entr["det_bot"][i_tr]),
                  x_lumen=float(0.5 * (span_[0] + span_[1])) if span_ is not None else float(tr["x"]))
        if cl["cls"] == "TOUCHDOWN" and not np.isfinite(cl["defect_y"]) and td_mask is not None \
                and td_frac >= P["touchdown_min_frac"]:
            tr["defect_y"] = float(np.nonzero(td_mask)[0].mean()) / h      # the roof patch itself
        tr.update(path_open=bool(lpath[0]), path_gap_frac=max(0, lpath[2] - lpath[1]) / h if not lpath[0] else 0.0,
                  wall_companion_frac=comp)
        if cl["regime"] == "touchdown" and cl["cls"] in ("FORMED", "NECKED") and td_frac >= P["touchdown_patch_frac"]:
            # a bubble or patch of roof on the floor inside a wide lumen: the roof came down there
            tr["cls"], tr["roof_down_frac"] = "TOUCHDOWN", max(tr["roof_down_frac"], td_frac)
            cl = dict(cl, cls="TOUCHDOWN")
        # port stubs: the remnant at a port where no tracked lumen part begins (both ports of a
        # NOT_FORMED channel, the blind side of a PINCHED one)
        for nm, top in (("top", True), ("bot", False)):
            st = None
            if cl["cls"] == "NOT_FORMED" or (cl["cls"] == "PINCHED" and not cl["part_at_" + nm]):
                st = port_stub(N, line, arc, blob, tr, h, w, w_exp, entr[nm], entr["bot" if top else "top"], top)
            tr["stub_" + nm] = st
            L_ = max(1.0, entr["bot"] - entr["top"])
            tr["stub_" + nm + "_len_frac"] = st["len_px"] / L_ if st else np.nan
            tr["stub_" + nm + "_len_px"] = st["len_px"] if st else np.nan
            tr["stub_" + nm + "_w_px"] = st["w_px"] if st else np.nan
            tr["stub_" + nm + "_area_px"] = st["area_px"] if st else np.nan
        # a stub is a lumen part: where the pinch of a PINCHED channel starts at an entrance because
        # the tracker found no part there, it starts at the tip of the stub instead
        if cl["cls"] == "PINCHED" and np.isfinite(tr["tip_top_row"]):
            t0, t1 = tr["tip_top_row"], tr["tip_bot_row"]
            if tr["stub_top"] and t0 <= entr["top"] + 1:
                t0 = min(entr["top"] + tr["stub_top"]["len_px"], t1)
            if tr["stub_bot"] and t1 >= entr["bot"] - 1:
                t1 = max(entr["bot"] - tr["stub_bot"]["len_px"], t0)
            if (t0, t1) != (tr["tip_top_row"], tr["tip_bot_row"]) and t1 > t0:
                off = (0.5 * (t0 + t1) - 0.5 * (entr["top"] + entr["bot"])) / L_
                tr.update(tip_top_row=float(t0), tip_bot_row=float(t1), ent_to_pinch_top_frac=(t0 - entr["top"]) / L_,
                          ent_to_pinch_bot_frac=(entr["bot"] - t1) / L_, pinch_len_frac=(t1 - t0) / L_,
                          pinch_center_offset_frac=off, pinch_symmetric=bool(abs(off) <= P["sym_tol"]),
                          defect_y=0.5 * (t0 + t1) / h)
        # length without lumen, one number for every class: 0 for a channel open end to end, the
        # pinch for a closed one, everything but the two port stubs for one that did not form
        if tr["cls"] in ("FORMED", "NECKED"):
            tr["gap_len_frac"] = 0.0
        elif tr["cls"] == "NOT_FORMED":
            stubs = sum(float(tr["stub_" + nm + "_len_frac"]) for nm in ("top", "bot") if np.isfinite(tr["stub_" + nm + "_len_frac"]))
            tr["gap_len_frac"] = float(np.clip(1.0 - stubs, 0.0, 1.0))
        else:
            tr["gap_len_frac"] = float(tr["pinch_len_frac"]) if np.isfinite(tr["pinch_len_frac"]) else np.nan
        # Printed against design. Width: the median width of an open channel between its port flares;
        # for a pinched channel the width of its lumen parts near the ports (after the flare). Length:
        # entrance to entrance against the design length of the roof. Pitch: center-to-center distance
        # of the channels against two posts plus the design width; the pitch does not depend on how
        # far the walls cured into the lumen, so it shows the scale of the print (and of the image).
        if tr["cls"] in ("FORMED", "NECKED"):
            w_print = float(tr["w_med_px"])
        elif tr["cls"] == "PINCHED":
            wb = [float(tr[k]) for k in ("w_base_top", "w_base_bot") if np.isfinite(tr[k])]
            w_print = float(np.mean(wb)) if wb else np.nan
        else:
            w_print = np.nan
        xs_ = sorted(t["x"] for t in tracks)
        pitch_meas = float(np.median(np.diff(xs_))) if len(xs_) >= 3 else pitch      # spacing of the final channel centers
        px_d = P["pixel_um"] / um_per_px if np.isfinite(um_per_px) else np.nan      # one printer pixel in image px
        len_design = P["design_rows"] * 2 * P["design_post_px"] * px_d
        pitch_design = (2 * P["design_post_px"] + Wd) * px_d if Wd else np.nan
        tr.update(w_print_px=w_print, w_print_over_design=w_print / w_exp if Wd else np.nan,
                  len_over_design=float(entr["bot"] - entr["top"]) / len_design if np.isfinite(len_design) else np.nan,
                  pitch_over_design=(pitch_meas / pitch_design) if (pitch_meas and np.isfinite(pitch_design)) else np.nan)
        tr.update(touchdown_frac=td_frac, td_mask=td_mask,
                  touchdown=(cl["cls"] == "TOUCHDOWN")
                  or (td_frac >= P["touchdown_min_frac"] and cl["cls"] in ("FORMED", "NECKED", "PINCHED")))
        results.append(tr)

    good = sum(1 for r in results if r["open_len_frac"] >= 0.6)
    n_pairs = diag["n_pairs"]
    if w_exp >= P["narrow_design_px"] and w_ref < P["min_print_ratio"] * w_exp:
        diag["confidence"] = "LOW"
        diag["note"] = "printed width implausibly small vs design: tracks are probably walls"
    elif n_pairs >= 3 or (n_pairs >= 1 and good >= 2):
        diag["confidence"] = "HIGH"
    elif n_pairs >= 1 or good >= 1:
        diag["confidence"] = "MEDIUM"
    else:
        diag["confidence"] = "LOW"
    if scale_note:
        diag["note"] = (diag.get("note", "") + "; " + scale_note).strip("; ")
    info = dict(file=os.path.basename(path), H=Hl, W=Wd, rep=rep, device_h_px=h, device_w_px=w,
                angle_deg=round(ang, 2), scalebar_px=bar_px, um_per_px=um_per_px, w_exp_px=w_exp,
                um_per_px_bar=um_per_px_bar, scale_source=scale_source, scale_q=q_scale,
                w_ref_px=w_ref, pitch_px=pitch, n_channels=len(results), n_walls=len(walls), diag=diag)
    if save_debug:
        save_overlay(dev, R, results, walls, faces, info, os.path.join(out_dir, name + "_overlay.png"))
    return info, results


def save_overlay(dev, R, results, walls, faces, info, out_png):
    h, w = dev.shape
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 2.2, 1.6], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax.imshow(dev, cmap="gray", vmin=0, vmax=255)
    for xw in walls:
        ax.axvline(xw, color="#1f77b4", lw=0.8, ls=":", alpha=0.9)
    for p in faces[0]:
        ax.plot([p["left"], p["right"]], [0.01 * h] * 2, color="cyan", lw=3, solid_capstyle="butt")
    for p in faces[1]:
        ax.plot([p["left"], p["right"]], [0.99 * h] * 2, color="cyan", lw=3, solid_capstyle="butt")
    ys = np.arange(h)
    for r in results:
        col = CLASS_COLOR[r["cls"]]
        ax.axvline(r["x_lo"], color="w", lw=0.4, ls="--", alpha=0.5)
        ax.axvline(r["x_hi"], color="w", lw=0.4, ls="--", alpha=0.5)
        for xp in (r["xL"], r["xR"]):
            ax.plot(np.where(r["present"], xp, np.nan), ys, color=col, lw=1.4)
            ax.plot(np.where(~r["present"], xp, np.nan), ys, color=col, lw=0.6, alpha=0.35)
        # the enclosed-patch mask is drawn where it is the evidence (a bubble or a roof patch inside a
        # lumen); in a TOUCHDOWN channel with a pinch stretch the pockets are the evidence, and over
        # surface grain the mask is only speckle (R2 H5 W160 ML4-6, channel 3)
        if r["td_mask"] is not None and r["touchdown"] and not (r["cls"] == "TOUCHDOWN" and np.isfinite(r["tip_top_row"])):
            yy, xx = np.nonzero(r["td_mask"])
            ax.scatter(xx[::3], yy[::3], s=1, color="yellow", alpha=0.35, linewidths=0)
        # lumen entrances (roof edges): solid where this channel shows the edge, dotted where inherited
        hw = 0.6 * max(info["w_exp_px"], r["w_port"], 6.0)
        for row, seen in ((r["ent_top_row"], r["ent_det_top"]), (r["ent_bot_row"], r["ent_det_bot"])):
            ax.plot([r["x_lumen"] - hw, r["x_lumen"] + hw], [row, row], color="#39ff14", lw=1.1, ls="-" if seen else ":")
        xm = r["x_lumen"]
        if np.isfinite(r["tip_top_row"]):
            # the pinch: from the end of the top lumen part to the start of the bottom one
            cap = 0.35 * hw
            ax.plot([xm, xm], [r["tip_top_row"], r["tip_bot_row"]], color="yellow", lw=1.2)
            for row in (r["tip_top_row"], r["tip_bot_row"]):
                ax.plot([xm - cap, xm + cap], [row, row], color="yellow", lw=1.6)
        if r["cls"] == "NECKED":
            for nm in ("top", "bot"):
                if np.isfinite(r["neck_row_" + nm]) and r["neck_depth_" + nm] >= 1 - P["neck_frac"]:
                    ax.plot([xm], [r["neck_row_" + nm]], marker="x", color="yellow", ms=8, mew=1.8)
        elif np.isfinite(r["defect_y"]):
            ax.plot([xm], [r["defect_y"] * h], marker="x", color="yellow", ms=9, mew=2)
        if r["bulge"]:
            ax.plot([xm], [0.5 * (r["ent_top_row"] + r["ent_bot_row"])], marker="D", mfc="none", mec="yellow", ms=8, mew=1.6)
        for nm in ("top", "bot"):                       # port stubs of a lumen that did not form there
            if r["stub_" + nm]:
                for c in r["stub_" + nm]["contours"]:
                    ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color="#ff9f1c", lw=0.9)
        extra = ""
        if r["cls"] == "PINCHED":
            extra = f"\nclosed {100 * r['closed_len_frac']:.0f}% of L"
        elif r["cls"] == "TOUCHDOWN":
            extra = f"\nroof down {100 * r['roof_down_frac']:.0f}% of L"
        elif r["cls"] == "NECKED":
            extra = f"\nmin {100 * r['w_min_frac']:.0f}%, {100 * r['necked_len_frac']:.0f}% of L"
        if r["blind"]:
            extra += "\n(blind: one port)"
        if np.isfinite(r["tip_gap_frac"]) and r["tip_gap_frac"] > 0 and r["cls"] == "PINCHED":
            extra += f"\ntip gap {100 * r['tip_gap_frac']:.0f}% of L"
        if r["touchdown"]:
            extra += f"\ntouchdown {100 * r['touchdown_frac']:.0f}%"
        if np.isfinite(r["ent_to_pinch_top_frac"]):
            extra += (f"\nent-pinch T{100 * r['ent_to_pinch_top_frac']:.0f} B{100 * r['ent_to_pinch_bot_frac']:.0f}"
                      f"\npinch {100 * r['pinch_len_frac']:.0f}, off {100 * r['pinch_center_offset_frac']:+.0f}"
                      + (" sym" if r["pinch_symmetric"] else ""))
        if r["cls"] in ("FORMED", "NECKED") and np.isfinite(r["neck_depth_top"]) and np.isfinite(r["neck_depth_bot"]):
            extra += (f"\nneck T{100 * r['neck_depth_top']:.0f}@{100 * r['neck_pos_top_frac']:.0f} "
                      f"B{100 * r['neck_depth_bot']:.0f}@{100 * r['neck_pos_bot_frac']:.0f}")
        if r["bulge"]:
            extra += f"\nbulge +{100 * r['bulge_excess_frac']:.0f}"
        if r["stub_top"] or r["stub_bot"]:
            um = info["um_per_px"]
            st = lambda nm: (f"{r['stub_' + nm + '_len_px'] * um:.0f}x{r['stub_' + nm + '_w_px'] * um:.0f}"
                             if r["stub_" + nm] else "-")
            extra += f"\nstub T{st('top')}\nstub B{st('bot')}"      # two lines: narrow designs leave no room for one
        if r["channel_confidence"] == "LOW":
            extra += "\n(low confidence)"
        tag = "*" if r["predicted"] else ""
        ax.text(r["x"], -0.02 * h, f"{r['index']}{tag}\n{r['cls']}{extra}", ha="center", va="bottom",
                fontsize=5.6, color=col, fontweight="bold")
    pitch_txt = "n/a" if info["pitch_px"] is None else f"{info['pitch_px']:.0f}"
    title = (f"{info['file']}   |   H{info['H']} W{info['W']}   |   {info['um_per_px']:.1f} um/px, "
             f"design lumen {info['w_exp_px']:.0f} px, printed reference {info['w_ref_px']:.0f} px, pitch {pitch_txt} px   |   "
             f"{info['n_channels']} channels, {info['n_walls']} walls, ports {info['diag']['port_source']}, "
             f"confidence {info['diag']['confidence']}")
    legend = ("green: lumen entrances (roof edges; dotted = edge not seen in this channel, device value used)   |   "
              "yellow bracket: pinch between the two lumen parts, x at its middle   |   "
              "ent-pinch T/B: entrance to pinch, top and bottom part (% of lumen length); pinch: its length; off: its middle minus mid-length\n"
              "neck T/B d@p: narrowest point of each half, d = % of the reference width lost, p = distance from that entrance (% of lumen length)"
              "   |   diamond: bulge, mid-length wider than both half minima (+ % of reference width)"
              "   |   orange outline, stub T/B: lumen remnant at a port where no lumen formed, length x width (um)")
    ax.set_title(title + "\n" + legend, fontsize=6.5, pad=58)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(np.clip(R, 0, 1.2), cmap="magma", aspect="auto")
    ax2.set_title("edge/blob response map used for tracking", fontsize=8)
    ax2.set_axis_off()

    ax3 = fig.add_subplot(gs[2])
    for r in results:
        col = CLASS_COLOR[r["cls"]]
        ax3.plot(ys / h, r["w_cls"] / info["w_ref_px"], color=col, lw=1.2, label=f"{r['index']} {r['cls']}")
    ax3.axhline(P["neck_frac"], color="k", lw=0.5, ls="--")
    ax3.axhline(P["closed_frac"], color="k", lw=0.5, ls=":")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.6)
    ax3.set_xlabel("position along channel (top face = 0, bottom face = 1)", fontsize=8)
    ax3.set_ylabel("width / reference width", fontsize=8)
    ax3.tick_params(labelsize=7)
    if results:
        ax3.legend(fontsize=6, ncol=min(6, len(results)), loc="upper center", bbox_to_anchor=(0.5, -0.35))
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


HERE = os.path.dirname(os.path.abspath(__file__))


def load_tags(path):
    """Frame tags: CSV with columns pattern (regex on the file name) and print_mode. The first
    matching row wins; unmatched frames are 'standard'."""
    tags = []
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("pattern"):
                    tags.append((re.compile(row["pattern"]), row["print_mode"].strip()))
    return tags


def print_mode_of(fn, tags):
    for rx, mode in tags:
        if rx.search(os.path.basename(fn)):
            return mode
    return "standard"


def read_manifest(images_dir):
    mf = os.path.join(images_dir, "manifest.csv")
    meta = {}
    if os.path.exists(mf):
        with open(mf, newline="") as f:
            for row in csv.DictReader(f):
                meta[row["file"]] = row
    return meta


def fmt(v, nd=3):
    if v is None:
        return ""
    try:
        if not np.isfinite(v):
            return ""
    except TypeError:
        return v
    return round(float(v), nd)


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def parse_frame(fn):
    """Replicate and membrane-layer range from `..._R1_top_H10_W70_ML7-10.jpg`.

    A printability device carries ten channels with roof thickness ML1..ML10; a wide design needs
    several overlapping frames, and `ML<a>-<b>` lists the channels of one frame from left to right
    (ML10-7 = ML10, 9, 8, 7). Returns (R, a, b), with None for anything the name does not carry.
    """
    base = os.path.basename(fn)
    r = re.search(r"_R(\d+)_", base)
    m = re.search(r"_ML(\d+)-(\d+)", base)
    return (int(r.group(1)) if r else None,
            int(m.group(1)) if m else None, int(m.group(2)) if m else None)


def assign_ml(rows):
    """Give every channel row its membrane-layer value and pick one observation per physical channel.

    Per frame, the n detected channels (left to right) are matched to the k = |a-b|+1 channels the
    file name lists:
      n == k  one to one;
      n >  k  the extra channels are partial ones at the frame edge: drop the channel nearest to an
              edge until n == k (ml_note 'edge channel outside ML range');
      n <  k  place the channels on the lattice with the measured pitch; if they then span exactly k
              slots the missing one is interior and the assignment is unambiguous, otherwise the
              frame is flagged ('fewer channels than ML range') and gets no ML values.
    Overlapping frames image the same channel twice: among rows with the same (H, W, R, ML) the one
    from a non-LOW device, with channel_confidence not LOW, farthest from the frame edge is kept
    (ml_use = True); the others are kept in the CSV with ml_use = False.
    """
    by_file = {}
    for r in rows:
        if r.get("channel") and r.get("cls") in CLASS_ORDER:
            by_file.setdefault(r["file"], []).append(r)
    for fn, chans in by_file.items():
        R, a, b = parse_frame(fn)
        for r in chans:
            r.update(R=R if R is not None else "", ML="", ml_note="", ml_use=False)
        if a is None:
            for r in chans:
                r["ml_note"] = "no ML range in file name"
            continue
        step = 1 if b >= a else -1
        mls = list(range(a, b + step, step))
        k = len(mls)
        chans = sorted(chans, key=lambda r: float(r["x_frac"]))
        edge = lambda r: min(float(r["x_frac"]), 1 - float(r["x_frac"]))
        keep = list(chans)
        while len(keep) > k:
            out = min((keep[0], keep[-1]), key=edge)
            out["ml_note"] = "edge channel outside ML range"
            keep.remove(out)
        if len(keep) == k:
            for r, ml in zip(keep, mls):
                r["ML"] = ml
            continue
        pitch = float(chans[0].get("pitch_px") or 0)
        wdev = float(chans[0].get("device_w_px") or 0)
        slots = [round((float(r["x_frac"]) - float(keep[0]["x_frac"])) * wdev / pitch) if pitch else None
                 for r in keep]
        if pitch and slots[-1] == k - 1 and len(set(slots)) == len(slots):
            for r, s in zip(keep, slots):
                r["ML"] = mls[s]
        else:
            for r in keep:
                r["ml_note"] = "fewer channels than ML range"

    best = {}
    for r in rows:
        if r.get("ML") in ("", None):
            continue
        key = (r["H_layers"], r["W_px"], r["R"], r["ML"], r.get("print_mode", "standard"))
        score = (r.get("device_confidence") != "LOW", r.get("channel_confidence") != "LOW",
                 min(float(r["x_frac"]), 1 - float(r["x_frac"])))
        if key not in best or score > best[key][0]:
            best[key] = (score, r)
    for score, r in best.values():
        r["ml_use"] = True
    return rows


CLASS_LETTER = dict(FORMED="F", NECKED="N", PINCHED="P", TOUCHDOWN="T", NOT_FORMED="X")


def plot_ml_map(cells, out_png, widths=None):
    """Categorical W x ML map per panel (H, and print mode when not standard): one cell per physical
    channel (replicates, if any, by majority). widths: {panel: [W, ...]} of every imaged design, so a
    design with no assignable channel still gets a gray column instead of disappearing."""
    Hs = sorted({c["panel"] for c in cells} | set(widths or {}))
    if not Hs:
        return
    fig, axes = plt.subplots(len(Hs), 1, figsize=(12, 3.6 * len(Hs)), squeeze=False)
    for ax, Hl in zip(axes[:, 0], Hs):
        sub = [c for c in cells if c["panel"] == Hl]
        Ws = sorted({c["W_px"] for c in sub} | set((widths or {}).get(Hl, [])))
        for x in range(len(Ws)):
            for y in range(10):
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#f2f2f2", edgecolor="white", lw=1.5))
        for c in sub:
            x, y = Ws.index(c["W_px"]), c["ML"] - 1
            color = "#d9d9d9" if c["excluded"] else CLASS_COLOR[c["cls"]]
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="white", lw=1.5))
            txt = "–" if c["excluded"] else CLASS_LETTER[c["cls"]] + ("*" if c["touchdown"] else "")
            ax.text(x, y, txt, ha="center", va="center", fontsize=7,
                    color="black" if c["excluded"] or c["cls"] == "NECKED" else "white")
        ax.set_xlim(-0.5, len(Ws) - 0.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xticks(range(len(Ws)))
        ax.set_xticklabels([str(w) for w in Ws], fontsize=7)
        ax.set_yticks(range(10))
        ax.set_yticklabels([str(i + 1) for i in range(10)], fontsize=7)
        ax.set_xlabel("design channel width W (printer px, 1 px = 32 um)")
        ax.set_ylabel("roof thickness ML (layers)")
        ax.set_title(f"H = {Hl}", loc="left", fontsize=9)
        ax.set_aspect("auto")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASS_ORDER]
    handles += [plt.Rectangle((0, 0), 1, 1, color="#d9d9d9"), plt.Rectangle((0, 0), 1, 1, color="#f2f2f2", ec="#bbbbbb")]
    labels = ([f"{CLASS_LETTER[c]} {c.replace('_', ' ').lower()}" for c in CLASS_ORDER]
              + ["– low-confidence device", "no channel assigned"])
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=7, frameon=False)
    fig.text(0.99, 0.005, "* touchdown (roof on the floor inside an open channel)", ha="right", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


SEQ_BLUE = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
FULL_REPLICATES = 3          # replicates per design in the sweep; a cell with fewer imaged ones is flagged on the maps
COUNT_RED = "#d03b3b"        # tag behind an open / imaged count in which half or fewer of the replicates are open (1/3, 1/2)
COUNT_YELLOW = "#f5c542"     # tag behind a count with most but not all replicates open (2/3), or all open but fewer than FULL_REPLICATES imaged
COUNT_GREEN = "#2e9e5b"      # tag behind a count in which all FULL_REPLICATES replicates were imaged and are open


def plot_value_map(cells, key, out_png, widths, title, closed_is_gray=True, partial=None, always_count=False,
                   scale=100.0, vmax=1.0, unit="%"):
    """W x ML map of one measured fraction (cells[key], shown in percent, mean over the replicates
    of a cell) on a one-hue light-to-dark scale; the color saturates at 100. With closed_is_gray, a
    cell with no value is gray and carries the letter of its class. With partial (default: same as
    closed_is_gray), the number is the mean of the open replicates only, and every cell with a number
    shows under it how many those are (open / imaged: 3/3, 2/2 or 1/1 when all are open). The count
    sits on a red tag when half or fewer of the imaged replicates are open (1/3, 1/2), on a yellow
    tag when more than half but not all are open (2/3) or all are open but fewer than
    FULL_REPLICATES were imaged (2/2, 1/1), and on a green tag when all FULL_REPLICATES were imaged
    and are open (3/3). always_count shows the count on a map of another kind, where the
    number is not a mean of the open replicates: there the red tag does not apply. scale, vmax and
    unit turn the percent map into a count map (scale 1, vmax = the largest count, unit = what is
    counted)."""
    partial = closed_is_gray if partial is None else partial
    tag = lambda bg: dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor="white", linewidth=0.4)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    panels = sorted({c["panel"] for c in cells} | set(widths or {}))
    if not panels:
        return
    # fixed margins in inches (title band on top, note band below), so nothing depends on a layout engine
    fig_h = 3.9 * len(panels) + 1.5
    fig, axes = plt.subplots(len(panels), 1, figsize=(12.6, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.925, top=1 - 0.75 / fig_h, bottom=1.15 / fig_h, hspace=0.36)
    fig.text(0.012, 1 - 0.12 / fig_h, title, ha="left", va="top", fontsize=9.5)
    for i_pn, (ax, pn) in enumerate(zip(axes[:, 0], panels)):
        sub = [c for c in cells if c["panel"] == pn]
        Ws = sorted({c["W_px"] for c in sub} | set((widths or {}).get(pn, [])))
        for x in range(len(Ws)):
            for y in range(10):
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#f2f2f2", edgecolor="white", lw=1.5))
        for c in sub:
            x, y = Ws.index(c["W_px"]), c["ML"] - 1
            v = c.get(key, np.nan)
            sub_txt, sub_bg = "", None
            if c["excluded"]:
                color, txt, ink = "#d9d9d9", "–", "black"
            elif np.isfinite(v):
                color = cmap(float(np.clip(v / vmax, 0, 1)))
                txt = f"{scale * v:.0f}"
                ink = "white" if v / vmax > 0.55 else "#1a1a1a"
                if always_count or partial:
                    k_open, n_img = c.get("n_open", 0), c.get("n", 0)
                    sub_txt = f"{k_open}/{n_img}"
                    if partial and k_open < n_img and 2 * k_open <= n_img:
                        sub_bg = COUNT_RED
                    elif (partial and k_open < n_img) or n_img < FULL_REPLICATES:
                        sub_bg = COUNT_YELLOW
                    elif k_open == n_img:
                        sub_bg = COUNT_GREEN
            elif closed_is_gray:
                color, txt, ink = "#8c8c8c", CLASS_LETTER[c["cls"]], "white"
            else:
                color, txt, ink = "#8c8c8c", "?", "white"
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=color, edgecolor="white", lw=1.5))
            if sub_txt:
                ax.text(x, y + 0.17, txt, ha="center", va="center", fontsize=6.5, color=ink)
                if sub_bg:
                    ax.text(x, y - 0.23, sub_txt, ha="center", va="center", fontsize=4.6,
                            color="#1a1a1a" if sub_bg == COUNT_YELLOW else "white", bbox=tag(sub_bg))
                else:
                    ax.text(x, y - 0.23, sub_txt, ha="center", va="center", fontsize=4.6, color=ink)
            else:
                ax.text(x, y, txt, ha="center", va="center", fontsize=6.5, color=ink)
        ax.set_xlim(-0.5, len(Ws) - 0.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xticks(range(len(Ws)))
        ax.set_xticklabels([str(w) for w in Ws], fontsize=7)
        ax.set_yticks(range(10))
        ax.set_yticklabels([str(i + 1) for i in range(10)], fontsize=7)
        ax.set_xlabel("design channel width W (printer px, 1 px = 32 um)")
        ax.set_ylabel("roof thickness ML (layers)")
        ax.set_title(f"H = {pn}", loc="left", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, scale * vmax))
    cb = fig.colorbar(sm, cax=fig.add_axes([0.94, 0.5 - 1.3 / fig_h, 0.012, 2.6 / fig_h]))
    cb.set_label(unit, fontsize=8)
    if unit != "%":
        cb.set_ticks(list(range(int(round(scale * vmax)) + 1)))
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    # the note takes up to three rows: one row runs past the right edge of the figure
    if closed_is_gray:
        notes = ["gray with a letter: no value for this class (P pinched, T touchdown, X not formed)   |   – low-confidence device"]
    else:
        notes = ["gray ?: no value could be measured   |   – low-confidence device"]
    if partial:
        notes.append("small k/n under a number: k of the n imaged replicates are open end to end; "
                     "the number is the mean of those k")
    elif unit == "%":
        notes.append("number = mean over the replicates of the cell")
    # row of count tags, each with its meaning: (sample, tag color, meaning, x of the sample)
    tags = []
    if partial:
        tags.append(("1/3", COUNT_RED, "half or fewer of the imaged replicates are open (1/3, 1/2)", 0.016))
    if partial or always_count:
        tags.append((f"{FULL_REPLICATES - 1}/{FULL_REPLICATES - 1}", COUNT_YELLOW,
                     "2/3 open, or all open but fewer than 3 replicates imaged (2/2, 1/1)" if partial
                     else f"fewer than {FULL_REPLICATES} replicates were imaged", 0.29 if partial else 0.016))
    if partial or always_count:
        tags.append((f"{FULL_REPLICATES}/{FULL_REPLICATES}", COUNT_GREEN,
                     f"all {FULL_REPLICATES} replicates were imaged and are open", 0.63 if partial else 0.30))
        notes.append("")
    for sample, bg, meaning, x0 in tags:
        fig.text(x0, 0.10 / fig_h, sample, ha="left", va="bottom", fontsize=6,
                 color="#1a1a1a" if bg == COUNT_YELLOW else "white", bbox=tag(bg))
        fig.text(x0 + 0.02, 0.10 / fig_h, meaning, ha="left", va="bottom", fontsize=7)
    for i, line in enumerate(reversed(notes)):
        if line:
            fig.text(0.012, (0.10 + 0.18 * i) / fig_h, line, ha="left", va="bottom", fontsize=7)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def summarize(rows, out_dir):
    """Per-frame and per-(H, W) summaries + a class-fraction plot; with ML-named files also a
    per-(H, W, ML) table and a W x ML map in which every physical channel is counted once."""
    chan = [r for r in rows if r.get("channel", 0) and r.get("cls") in CLASS_ORDER
            and r.get("device_confidence") != "LOW"]
    if not chan:
        return
    ml_mode = any(r.get("ML") not in ("", None) for r in rows)
    mode = lambda r: r.get("print_mode") or "standard"
    panel_of = lambda Hl, m: f"{Hl} layers" + ("" if m == "standard" else f", {m}")
    by_dev, by_hw = {}, {}
    for r in chan:
        by_dev.setdefault(r["file"], []).append(r)
        # pooled per-width statistics: with ML-named frames, only the kept observation of each channel;
        # frames printed in a different mode are pooled separately
        if not ml_mode or r.get("ml_use") in (True, "True"):
            by_hw.setdefault((r["H_layers"], r["W_px"], mode(r)), []).append(r)

    if ml_mode:
        groups = {}
        for r in rows:
            if r.get("ml_use") in (True, "True"):
                groups.setdefault((int(r["H_layers"]), mode(r), int(r["W_px"]), int(r["ML"])), []).append(r)
        cells, ml_rows = [], []
        for (Hl, m, Wd, ml), items in sorted(groups.items()):
            good = [r for r in items if r.get("device_confidence") != "LOW"]
            use = good or items
            cls = max(CLASS_ORDER, key=lambda c: (sum(r["cls"] == c for r in use), -CLASS_ORDER.index(c)))
            td = any(r["touchdown"] in (True, "True") for r in use)
            cells.append(dict(H_layers=Hl, panel=panel_of(Hl, m), W_px=Wd, ML=ml, cls=cls, touchdown=td,
                              excluded=not good))
            one = use[0]

            def mean_of(key, sel=None):
                vals = []
                for r in use:
                    if sel is None or r["cls"] in sel:
                        try:
                            v = float(r.get(key, ""))
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(v):
                            vals.append(v)
                return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)

            open_cls = ("FORMED", "NECKED")
            n_open = sum(r["cls"] in open_cls for r in use)
            gap, n_gap = mean_of("gap_len_frac")
            nmax, _ = mean_of("narrow_max_frac", open_cls)
            nmean, _ = mean_of("narrow_mean_frac", open_cls)
            # bulge: how much wider the middle is than both narrowings beside it; no bulge counts as 0
            bl = [max(0.0, float(r["bulge_excess_frac"])) for r in use
                  if r["cls"] in open_cls and r.get("bulge_excess_frac") not in ("", None)]
            bulge = float(np.mean(bl)) if bl else np.nan
            qual, _ = mean_of("lumen_quality", open_cls)
            w_ratio, _ = mean_of("w_print_over_design")
            len_ratio, _ = mean_of("lumen_len_over_design")
            pitch_ratio, _ = mean_of("pitch_over_design")
            cells[-1].update(gap=gap, narrow_max=nmax, narrow_mean=nmean, bulge=bulge, quality=qual,
                             w_ratio=w_ratio, len_ratio=len_ratio, pitch_ratio=pitch_ratio,
                             open_frac=n_open / len(use), n_open=n_open, n=len(use), n_rep=float(len(good)))
            ml_rows.append(dict(H_layers=Hl, print_mode=m, W_px=Wd, ML=ml, n_replicates=len(use), cls=cls,
                                gap_len_frac_mean=fmt(gap), n_gap=n_gap, n_open=n_open,
                                narrow_max_frac_mean=fmt(nmax), narrow_mean_frac_mean=fmt(nmean),
                                bulge_excess_frac_mean=fmt(bulge), lumen_quality_mean=fmt(qual),
                                open_fraction=fmt(n_open / len(use)), w_print_over_design_mean=fmt(w_ratio),
                                lumen_len_over_design_mean=fmt(len_ratio), pitch_over_design_mean=fmt(pitch_ratio),
                                excluded_low_confidence=not good, touchdown=td,
                                w_min_frac=one.get("w_min_frac", ""), closed_len_frac=one.get("closed_len_frac", ""),
                                necked_len_frac=one.get("necked_len_frac", ""), file=one["file"],
                                channel=one["channel"]))
        write_csv(os.path.join(out_dir, "summary_by_H_W_ML.csv"), ml_rows)
        widths = {}
        for r in rows:
            if r.get("H_layers") not in ("", None) and r.get("W_px") not in ("", None):
                widths.setdefault(panel_of(int(r["H_layers"]), mode(r)), set()).add(int(r["W_px"]))
        plot_ml_map(cells, os.path.join(out_dir, "channel_state_map_W_ML.png"), widths)
        plot_value_map(cells, "gap", os.path.join(out_dir, "gap_between_lumen_parts_map_W_ML.png"), widths,
                       "Length without lumen (% of the lumen length): the pinch between the two lumen parts; "
                       "0 = open end to end; a channel that did not form counts everything but its port stubs",
                       closed_is_gray=False)
        plot_value_map(cells, "narrow_max", os.path.join(out_dir, "narrowing_deepest_map_W_ML.png"), widths,
                       "Narrowing at the narrowest point of an open channel (% of the reference width lost)")
        plot_value_map(cells, "narrow_mean", os.path.join(out_dir, "narrowing_mean_map_W_ML.png"), widths,
                       "Mean narrowing of an open channel along its length (% of the reference width lost)")
        plot_value_map(cells, "bulge", os.path.join(out_dir, "bulge_mid_length_map_W_ML.png"), widths,
                       "Bulge of an open channel at mid-length: width of the middle minus the wider of the two "
                       "narrowest points beside it (% of the reference width; 0 = no bulge)")
        plot_value_map(cells, "quality", os.path.join(out_dir, "lumen_quality_map_W_ML.png"), widths,
                       "Lumen quality of a channel open end to end: 100 minus the mean distance of its width from the "
                       "reference width along the length (%; 100 = neither narrows nor bulges)")
        plot_value_map(cells, "open_frac", os.path.join(out_dir, "open_fraction_map_W_ML.png"), widths,
                       "Share of the printed channels that are open end to end (% of the replicates of a cell; "
                       "below each number: open / imaged)", closed_is_gray=False, always_count=True)
        plot_value_map(cells, "n_rep", os.path.join(out_dir, "replicates_map_W_ML.png"), widths,
                       "Replicates per design: number of imaged channels that count for a cell (one per physical "
                       "channel; channels of low-confidence devices are not counted)", closed_is_gray=False,
                       scale=1.0, vmax=float(FULL_REPLICATES), unit="replicates")
        plot_value_map(cells, "w_ratio", os.path.join(out_dir, "width_printed_vs_design_map_W_ML.png"), widths,
                       "Printed lumen width as % of the design width (open channels: median width along the length; "
                       "pinched channels: width of the lumen parts near the ports)", partial=False)
        plot_value_map(cells, "len_ratio", os.path.join(out_dir, "length_printed_vs_design_map_W_ML.png"), widths,
                       "Printed lumen length (roof edge to roof edge) as % of the design length", closed_is_gray=False)
        plot_value_map(cells, "pitch_ratio", os.path.join(out_dir, "pitch_printed_vs_design_map_W_ML.png"), widths,
                       "Channel pitch (center to center) as % of the design pitch: the scale of the print and of the image, "
                       "independent of how far the walls cured into the lumen", closed_is_gray=False)

    def num(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    def mean_of(items, key, nd=3):
        vals = [num(r.get(key)) for r in items]
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), nd) if vals else ""

    def agg(items):
        n = len(items)
        d = dict(n_channels=n)
        for c in CLASS_ORDER:
            d[f"frac_{c}"] = round(sum(1 for r in items if r["cls"] == c) / n, 3)
        d["frac_open_end_to_end"] = round(sum(1 for r in items if r["cls"] in ("FORMED", "NECKED")) / n, 3)
        mins = [r["w_min_frac"] for r in items if r["w_min_frac"] not in ("", None)]
        d["mean_min_width_frac"] = round(float(np.mean(mins)), 3) if mins else ""
        cl = [r["closed_len_frac"] for r in items if r["cls"] == "PINCHED"]
        d["mean_closed_len_frac_of_pinched"] = round(float(np.mean(cl)), 3) if cl else ""
        d["n_touchdown"] = sum(1 for r in items if r["touchdown"])
        # lumen geometry (see measure_halves): widths in um, taper = inner/base width of a half,
        # narrowing = width lost per unit length toward the middle, tip gap between the two halves
        d["mean_w_ref_um"] = mean_of(items, "w_ref_um", 1)
        d["mean_w_base_um"] = mean_of(items, "_w_base_um", 1)
        d["mean_taper"] = mean_of(items, "_taper")
        d["mean_narrowing"] = mean_of(items, "_narrowing", 4)
        d["mean_open_half_len_frac"] = mean_of(items, "_open_half_len_frac")
        d["mean_tip_gap_frac"] = mean_of(items, "_tip_gap_frac")
        d["mean_tip_gap_um"] = mean_of(items, "_tip_gap_um", 0)
        d["n_with_lumen"] = sum(1 for r in items if r["cls"] in ("FORMED", "NECKED", "PINCHED"))
        return d

    # per-channel means over the two halves, so a stub from either port counts once; geometry is
    # pooled only over channels with a lumen (FORMED, NECKED, PINCHED)
    pairs = {"_w_base_um": ("w_base_top_um", "w_base_bot_um"), "_taper": ("taper_top", "taper_bot"),
             "_narrowing": ("narrowing_top", "narrowing_bot"),
             "_open_half_len_frac": ("open_top_len_frac", "open_bot_len_frac")}
    for r in chan:
        has_lumen = r["cls"] in ("FORMED", "NECKED", "PINCHED")
        for k, (kt, kb) in pairs.items():
            vals = [v for v in (num(r.get(kt)), num(r.get(kb))) if v is not None]
            r[k] = float(np.mean(vals)) if vals and has_lumen else ""
        r["_tip_gap_frac"] = r.get("tip_gap_frac", "") if has_lumen else ""
        r["_tip_gap_um"] = r.get("tip_gap_um", "") if has_lumen else ""

    dev_rows = []
    for fn, items in by_dev.items():
        d = dict(file=fn, H_layers=items[0]["H_layers"], W_px=items[0]["W_px"],
                 device_confidence=items[0]["device_confidence"])
        d.update(agg(items))
        dev_rows.append(d)
    write_csv(os.path.join(out_dir, "device_summary.csv"), dev_rows)
    hw_rows = []
    for (Hl, Wd, m), items in sorted(by_hw.items(), key=lambda kv: (int(kv[0][0] or 0), kv[0][2], int(kv[0][1] or 0))):
        # a device is one replicate R when the names carry it; otherwise each frame counts as one
        d = dict(H_layers=Hl, print_mode=m, panel=panel_of(int(Hl), m), W_px=Wd,
                 n_devices=len({r.get("R") or r["file"] for r in items}), n_frames=len({r["file"] for r in items}))
        d.update(agg(items))
        hw_rows.append(d)
    write_csv(os.path.join(out_dir, "summary_by_H_W.csv"), hw_rows)
    # side-by-side comparison of print modes (same H and W, different mode)
    modes = sorted({r["print_mode"] for r in hw_rows})
    if len(modes) > 1:
        cmp_rows = []
        keys = ["n_channels", "n_with_lumen", "frac_open_end_to_end", "frac_PINCHED", "frac_NOT_FORMED", "mean_w_ref_um",
                "mean_w_base_um", "mean_taper", "mean_narrowing", "mean_open_half_len_frac", "mean_tip_gap_frac",
                "mean_tip_gap_um"]
        for (Hl, Wd) in sorted({(int(r["H_layers"]), int(r["W_px"])) for r in hw_rows}):
            row = dict(H_layers=Hl, W_px=Wd)
            for m in modes:
                hit = [r for r in hw_rows if int(r["H_layers"]) == Hl and int(r["W_px"]) == Wd and r["print_mode"] == m]
                for k in keys:
                    row[f"{k}[{m}]"] = hit[0][k] if hit else ""
            cmp_rows.append(row)
        write_csv(os.path.join(out_dir, "summary_by_print_mode.csv"), cmp_rows)

    # stacked class fractions vs W, one panel per H
    Hs = sorted({r["panel"] for r in hw_rows if r["H_layers"] not in ("", None)})
    if Hs:
        fig, axes = plt.subplots(len(Hs), 1, figsize=(12, 3.2 * len(Hs)), squeeze=False)
        for ax, Hl in zip(axes[:, 0], Hs):
            sub = sorted([r for r in hw_rows if r["panel"] == Hl], key=lambda r: int(r["W_px"]))
            xs = np.arange(len(sub))
            bottom = np.zeros(len(sub))
            for c in CLASS_ORDER:
                vals = np.array([r[f"frac_{c}"] for r in sub])
                ax.bar(xs, vals, bottom=bottom, color=CLASS_COLOR[c], label=c, width=0.8)
                bottom += vals
            ax.set_xticks(xs)
            ax.set_xticklabels([f"W{r['W_px']}\n(n={r['n_channels']})" for r in sub], fontsize=7)
            ax.set_ylabel("fraction of channels")
            ax.set_title(f"H = {Hl}: channel state vs design width (Asiga Ultra px)", fontsize=10)
            ax.set_ylim(0, 1)
        axes[0, 0].legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.35))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "class_fractions_vs_W.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


def _process_one(job):
    """Worker for --jobs: one frame, errors returned instead of raised."""
    fp, out_dir, m, save_debug = job
    try:
        return fp, process_image(fp, out_dir, m, save_debug=save_debug), None
    except Exception as e:
        return fp, None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None, help="substring filter on file names")
    ap.add_argument("--bar-mm", type=float, default=P["bar_mm"], help="physical length of the scale bar")
    ap.add_argument("--pixel-um", type=float, default=P["pixel_um"], help="printer XY pixel pitch")
    ap.add_argument("--no-debug", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="frames processed in parallel (default 1)")
    ap.add_argument("--tags", default=os.path.join(HERE, "frame_tags.csv"),
                    help="CSV of (pattern, print_mode) rows; frames are pooled per print mode")
    args = ap.parse_args()
    tags = load_tags(args.tags)
    P["bar_mm"], P["pixel_um"] = args.bar_mm, args.pixel_um
    os.makedirs(args.out, exist_ok=True)
    meta = read_manifest(args.images)
    files = sorted(glob.glob(os.path.join(args.images, "**", "*.jp*g"), recursive=True)
                   + glob.glob(os.path.join(args.images, "**", "*.png"), recursive=True))
    rows = []
    jobs = []
    for fp in files:
        fn = os.path.basename(fp)
        m = meta.get(fn, {})
        if m.get("view", "top") != "top":
            print(f"skip {fn} (view={m.get('view')})")
            continue
        if args.only and args.only not in fn:
            continue
        jobs.append((fp, args.out, m, not args.no_debug))
    if args.jobs > 1 and len(jobs) > 1:
        import multiprocessing
        pool = multiprocessing.Pool(min(args.jobs, len(jobs)))
        done = pool.imap(_process_one, jobs, chunksize=1)     # in file order
    else:
        pool, done = None, map(_process_one, jobs)
    for fp, res, err in done:
        fn = os.path.basename(fp)
        if res is None:
            print(f"FAILED {fn}: {err}")
            rows.append(dict(file=fn, error=err))
            continue
        info, results = res
        summary = ", ".join(f"{r['index']}:{r['cls'][:4]}" + (f"({100 * r['closed_len_frac']:.0f}%)" if r["cls"] == "PINCHED" else "")
                            + (f"({100 * r['roof_down_frac']:.0f}%)" if r["cls"] == "TOUCHDOWN" else "")
                            + ("+TD" if r["touchdown"] else "") for r in results)
        print(f"{fn[-26:]:26s} H{info['H']} W{str(info['W']):>3s} wexp {info['w_exp_px']:5.0f} wref {info['w_ref_px']:5.0f} "
              f"n {info['n_channels']:2d} conf {info['diag']['confidence']:6s} -> {summary}")
        base = dict(file=fn, H_layers=info["H"], W_px=info["W"], replicate=info["rep"],
                    scalebar_px=info["scalebar_px"], um_per_px=fmt(info["um_per_px"], 2),
                    um_per_px_bar=fmt(info["um_per_px_bar"], 2), scale_source=info["scale_source"],
                    scale_q=fmt(info["scale_q"], 3),
                    w_exp_px=fmt(info["w_exp_px"], 1), w_ref_px=fmt(info["w_ref_px"], 1),
                    device_h_px=info["device_h_px"], device_w_px=info["device_w_px"],
                    pitch_px=fmt(info["pitch_px"], 1), n_walls=info["n_walls"],
                    port_source=info["diag"]["port_source"], device_confidence=info["diag"]["confidence"],
                    print_mode=print_mode_of(fn, tags), w_ref_um=fmt(info["w_ref_px"] * info["um_per_px"], 1),
                    note=info["diag"].get("note", ""), lattice_note=info["diag"].get("lattice", ""))
        um = info["um_per_px"]
        for r in results:
            rows.append(dict(base, channel=r["index"], x_frac=fmt(r["x_frac"]), predicted=r["predicted"],
                             has_top_port=r["has_top"], has_bot_port=r["has_bot"], w_port_px=fmt(r["w_port"], 1),
                             cls=r["cls"], f_present=fmt(r["f_present"]), open_len_frac=fmt(r["open_len_frac"]),
                             reaches_top=r["reaches_top"], reaches_bot=r["reaches_bot"],
                             w_min_px=fmt(r["w_min"], 1), w_min_frac=fmt(r["w_min_frac"]), w_min_y_frac=fmt(r["w_min_y"]),
                             closed_len_frac=fmt(r["closed_len_frac"]), necked_len_frac=fmt(r["necked_len_frac"]),
                             defect_y_frac=fmt(r["defect_y"]), blind=r["blind"], channel_confidence=r["channel_confidence"],
                             touchdown=r["touchdown"], touchdown_frac=fmt(r["touchdown_frac"]),
                             regime=r["regime"], roof_down_frac=fmt(r["roof_down_frac"]),
                             path_open=r["path_open"], path_gap_frac=fmt(r["path_gap_frac"]),
                             wall_companion_frac=fmt(r["wall_companion_frac"]),
                             entrance_top_frac=fmt(r["ent_top_row"] / info["device_h_px"]),
                             entrance_bot_frac=fmt(r["ent_bot_row"] / info["device_h_px"]),
                             entrance_seen_top=r["ent_det_top"], entrance_seen_bot=r["ent_det_bot"],
                             lumen_len_um=fmt(r["lumen_len_px"] * um, 0), n_lumen_parts=r["n_lumen_parts"],
                             ent_to_pinch_top_frac=fmt(r["ent_to_pinch_top_frac"]),
                             ent_to_pinch_top_um=fmt(r["ent_to_pinch_top_frac"] * r["lumen_len_px"] * um, 0),
                             ent_to_pinch_bot_frac=fmt(r["ent_to_pinch_bot_frac"]),
                             ent_to_pinch_bot_um=fmt(r["ent_to_pinch_bot_frac"] * r["lumen_len_px"] * um, 0),
                             pinch_len_frac=fmt(r["pinch_len_frac"]),
                             pinch_len_um=fmt(r["pinch_len_frac"] * r["lumen_len_px"] * um, 0),
                             pinch_center_offset_frac=fmt(r["pinch_center_offset_frac"]),
                             pinch_symmetric=r["pinch_symmetric"],
                             neck_depth_top=fmt(r["neck_depth_top"]), neck_pos_top_frac=fmt(r["neck_pos_top_frac"]),
                             neck_depth_bot=fmt(r["neck_depth_bot"]), neck_pos_bot_frac=fmt(r["neck_pos_bot_frac"]),
                             w_center_frac=fmt(r["w_center_frac"]), bulge_excess_frac=fmt(r["bulge_excess_frac"]),
                             bulge=r["bulge"],
                             gap_len_frac=fmt(r["gap_len_frac"]), narrow_max_frac=fmt(r["narrow_max_frac"]),
                             narrow_mean_frac=fmt(r["narrow_mean_frac"]), lumen_quality=fmt(r["lumen_quality"]),
                             w_print_um=fmt(r["w_print_px"] * um, 0),
                             w_design_um=fmt(info["W"] * P["pixel_um"], 0) if info["W"] else "",
                             w_print_over_design=fmt(r["w_print_over_design"]),
                             lumen_len_design_um=fmt(P["design_rows"] * 2 * P["design_post_px"] * P["pixel_um"], 0),
                             lumen_len_over_design=fmt(r["len_over_design"]),
                             pitch_um=fmt(info["pitch_px"] * um, 0) if info["pitch_px"] else "",
                             pitch_over_design=fmt(r["pitch_over_design"]),
                             stub_top_len_frac=fmt(r["stub_top_len_frac"]), stub_top_len_um=fmt(r["stub_top_len_px"] * um, 0),
                             stub_top_w_um=fmt(r["stub_top_w_px"] * um, 0),
                             stub_top_area_um2=fmt(r["stub_top_area_px"] * um * um, 0),
                             stub_bot_len_frac=fmt(r["stub_bot_len_frac"]), stub_bot_len_um=fmt(r["stub_bot_len_px"] * um, 0),
                             stub_bot_w_um=fmt(r["stub_bot_w_px"] * um, 0),
                             stub_bot_area_um2=fmt(r["stub_bot_area_px"] * um * um, 0),
                             w_min_um=fmt(r["w_min"] * um, 1),
                             w_base_top_px=fmt(r["w_base_top"], 1), w_base_top_um=fmt(r["w_base_top"] * um, 1),
                             w_inner_top_px=fmt(r["w_inner_top"], 1), taper_top=fmt(r["taper_top"]),
                             narrowing_top=fmt(r["narrowing_top"], 4), open_top_len_frac=fmt(r["open_top_len_frac"]),
                             w_base_bot_px=fmt(r["w_base_bot"], 1), w_base_bot_um=fmt(r["w_base_bot"] * um, 1),
                             w_inner_bot_px=fmt(r["w_inner_bot"], 1), taper_bot=fmt(r["taper_bot"]),
                             narrowing_bot=fmt(r["narrowing_bot"], 4), open_bot_len_frac=fmt(r["open_bot_len_frac"]),
                             tip_gap_frac=fmt(r["tip_gap_frac"]), tip_gap_um=fmt(r["tip_gap_frac"] * info["device_h_px"] * um, 0),
                             error=""))
        if not results:
            rows.append(dict(base, channel=0, cls="NO_PORTS_FOUND",
                             error=f"{info['n_walls']} wall lines, no aligned port pair: treat all cells as NOT_FORMED"))
    if pool is not None:
        pool.close()
        pool.join()
    csv_path = os.path.join(args.out, "channel_results.csv")
    if rows and args.only and os.path.exists(csv_path):
        done = {r["file"] for r in rows}
        with open(csv_path, newline="") as f:
            old = [r for r in csv.DictReader(f) if r["file"] not in done]
        for r in old:   # CSV strings back to the types summarize() expects
            for k, v in list(r.items()):
                if isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", v):
                    r[k] = float(v)
            r["channel"] = int(r["channel"]) if r.get("channel") else 0
            r["touchdown"] = r.get("touchdown") in (True, "True")
        rows = old + rows
        rows.sort(key=lambda r: (r["file"], r.get("channel", 0)))
    if rows:
        assign_ml(rows)
        write_csv(csv_path, rows)
        summarize(rows, args.out)
        print(f"\nwrote {args.out}/channel_results.csv ({len(rows)} rows), device_summary.csv, summary_by_H_W.csv")


if __name__ == "__main__":
    main()
