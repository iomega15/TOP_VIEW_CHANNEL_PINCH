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
    PINCHED     closed somewhere (width below closed_frac, 15%, of the reference);
                closed_len_frac = fraction of the length that is closed (point pinch -> ~0.02,
                half the channel -> ~0.5). blind = True when the lumen comes out of one
                port only and ends inside the device.
    NOT_FORMED  nothing beyond the port wedges
    UNRESOLVED  both edges lost over more than half the length, no closure evidence
                (review by eye)
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
    bar_mm=2.0,              # scale-bar length; the "0.20 mm" label on the H5 images is a 10x calibration slip
    um_per_px_default=13.5,  # used only when no scale bar is found (typical for the 1920 px frames)
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
    gap_unresolved=0.5,      # a bridged gap longer than this fraction of the length -> UNRESOLVED (review by eye)
    cap_tau=0.8,             # response that counts as 'dark' for a lumen tip (above speckle) ...
    cap_cover=0.8,           # ... whose union over the cap window must span this fraction of the cell width
    cap_reach=0.04,          # cap window reaches this fraction of the length beyond the end of a lumen run
    closed_min_px=5,         # two edge lines closer than this cannot be resolved: closed ...
    closed_line_px=4,        # ... or closer than the track repulsion radius plus this
    fill_low_conf=0.3,       # a bridged tracking dropout longer than this fraction of the length -> channel confidence LOW
    neck_frac=0.75,          # narrowest point below this x w_ref -> NECKED ("partially pinched")
    closed_frac=0.15,        # width below this x w_ref -> closed
    not_formed_present=0.20,
    wedge_max=0.25,
    edge_exclude=0.12,       # ignore the port flare when looking for the narrowest point
    touchdown_tau=0.5,       # dark-line threshold for the inner (touchdown) outline
    touchdown_min_frac=0.02, # report touchdown when the enclosed area exceeds this fraction of the lumen
)

CLASS_ORDER = ["FORMED", "NECKED", "PINCHED", "NOT_FORMED", "UNRESOLVED"]
CLASS_COLOR = dict(FORMED="#2ca02c", NECKED="#ff7f0e", PINCHED="#d62728", NOT_FORMED="#7f7f7f",
                   UNRESOLVED="#17becf")


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


def find_channels(N, line, arc, blob, h, w, w_exp, pitch_exp=None):
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
    if pitch_exp is not None:
        cands = [c for c in cands if abs(c - pitch_exp) / pitch_exp <= P["pitch_tol"]]
        cands.append(float(pitch_exp))
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
        # a measured pitch needs three real pairs and solid support to beat the design pitch
        if pitch_exp is not None and (len(paired) < 3 or sup[best] < 0.6):
            best = float(pitch_exp)
        if sup[best] > 0 or best == pitch_exp:
            pitch = best
            pitch_support = max(sup[best], 0.6 if best == pitch_exp else 0.0)
    elif wall_pitch is not None and pitch is None:
        pitch = wall_pitch
    if pitch_exp is not None and (pitch is None or abs(pitch - pitch_exp) / pitch_exp > P["pitch_tol"]):
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
    top = next(((a, b) for a, b in runs if a <= reach), None)
    bot = next(((a, b) for a, b in reversed(runs) if b >= h - reach), None)
    if top and bot and top == bot:                      # continuous channel: split at mid-length
        a, b = top
        top, bot = (a, (a + b) // 2), ((a + b) // 2, b)
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


def classify_channel(tr, h, w_ref, arc, w_exp, blob=None, R_full=None):
    present = tr["present"].copy()
    width = tr["width"].copy()
    anchor = P["anchor_depth"] * h
    win = max(5, int(P["w_smooth"] * h) | 1)

    raw_runs = runs_of(present)
    f_raw = float(present.mean())
    only_wedges = (w_exp < 2 * P["narrow_design_px"]) and all(
        (b - a) <= P["wedge_max"] * h and (a <= anchor or b >= h - anchor) for a, b in raw_runs)

    def capped(y0, y1):
        """Is the lumen closed off by a dark cap (rounded blind end, converging
        tips) in rows y0:y1? The union over these rows of strong dark structure
        must cover nearly the whole width between the two edges; a tracking
        dropout leaves the interior open."""
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
    # interior gaps: a closure when both flanks taper (converging tips) or
    # when the gap is too long to be a tracking dropout; otherwise a dropout
    # that is bridged by interpolation
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
        if tapers or caps:
            width[a:b] = 0.0
        else:
            # dropout: bridge with the flank medians (single end rows are noisy),
            # but keep the raw tracked width where one edge was still followed
            # (a faint neck should not be erased by the bridge)
            wl = np.median(fl) if fl.size else width[a - 1]
            wr = np.median(fr) if fr.size else width[min(b, h - 1)]
            bridge = np.interp(np.arange(a, b), [a - 1, b], [wl, wr])
            raw = nanmedfilt(tr["width_raw"][a:b].astype(np.float32), win)
            one_edge = (tr["hitL"][a:b] > P["hit_frac"]) | (tr["hitR"][a:b] > P["hit_frac"])
            width[a:b] = np.where(one_edge & (raw > 0), np.minimum(bridge, raw), bridge)
            longest_fill = max(longest_fill, (b - a) / h)
        present[a:b] = True
    w_cls = nanmedfilt(width, win)
    ok = ~np.isnan(w_cls)
    f_present = float(ok.mean())

    reach, band = P["face_reach"] * h, P["port_band"] * h
    cap_win = max(4, int(0.02 * h))
    reaches_top = reaches_bot = False
    if raw_runs:
        a0, b0 = raw_runs[0]
        reaches_top = a0 <= reach
        a1, b1 = raw_runs[-1]
        reaches_bot = b1 >= h - reach
    # a detected crescent on a face proves the lumen opens there
    if tr.get("has_top") and not tr.get("predicted"):
        reaches_top = True
    if tr.get("has_bot") and not tr.get("predicted"):
        reaches_bot = True
    core = ok.copy()
    core[: int(P["edge_exclude"] * h)] = False
    core[h - int(P["edge_exclude"] * h):] = False
    if core.any():
        wc = np.where(core, w_cls, np.nan)
        i_min = int(np.nanargmin(wc))
        w_min, w_min_y = float(wc[i_min]), i_min / h
    else:
        w_min, w_min_y = np.nan, np.nan
    w_min_frac = w_min / w_ref if np.isfinite(w_min) else np.nan

    # two tracks that repel each other cannot get closer than the suppression
    # radius plus a line width: that is what a closed lumen looks like
    closed_px = max(P["closed_frac"] * w_ref, P["closed_min_px"], tr["sup"] + P["closed_line_px"])
    closed = core & (w_cls < closed_px)
    # drop short closed runs, and debris: a short closure where only one edge
    # deflected (a real pinch pulls both edges toward the center)
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
    necked = core & (w_cls < P["neck_frac"] * w_ref)
    n_core = max(1, int(core.sum()))
    closed_len_frac = float(closed.sum()) / n_core
    necked_len_frac = float(necked.sum()) / n_core
    halves = measure_halves(w_cls, closed, h, reach, int(P["edge_exclude"] * h))

    def run_center(mask):
        rr = runs_of(mask)
        if not rr:
            return np.nan
        a, b = max(rr, key=lambda r: r[1] - r[0])
        return 0.5 * (a + b) / h

    blind = False
    if only_wedges or f_raw < P["not_formed_present"]:
        cls = "NOT_FORMED"           # no lumen along the length (ports/wedges only, or roof down)
    elif reaches_top and reaches_bot:
        if closed.any():
            cls = "PINCHED"          # closed_len_frac says how much of the length
        elif necked.any():
            cls = "NECKED"
        else:
            cls = "FORMED"
    elif f_raw >= P["not_formed_present"]:
        # lumen from one port only: closed before it reaches the other face
        cls = "PINCHED"
        blind = True
        closed_len_frac = max(closed_len_frac, 1.0 - f_raw)
        ends = [b for a, b in raw_runs if a <= reach] + [a for a, b in raw_runs if b >= h - reach]
        defect_y = (ends[0] / h) if ends else np.nan
    else:
        cls = "NOT_FORMED"
    if not blind:
        defect_y = run_center(closed) if cls == "PINCHED" else (run_center(necked) if cls == "NECKED" else np.nan)
    chan_conf = "LOW" if longest_fill > P["fill_low_conf"] else "HIGH"
    if longest_fill > P["gap_unresolved"] and cls in ("FORMED", "NECKED"):
        cls = "UNRESOLVED"       # both edges lost over most of the length, no closure evidence
    return dict(cls=cls, w_cls=w_cls, f_present=f_present, reaches_top=reaches_top, reaches_bot=reaches_bot,
                w_min=w_min, w_min_frac=w_min_frac, w_min_y=w_min_y, closed_len_frac=closed_len_frac,
                necked_len_frac=necked_len_frac, defect_y=defect_y, open_len_frac=f_raw,
                channel_confidence=chan_conf, longest_fill=longest_fill, blind=blind, **halves)


def touchdown(line, arc, tr, h):
    """Fraction of the lumen area enclosed by an inner closed dark outline: the
    roof membrane resting on the floor. Bubbles/islands count too; both mean
    the lumen is not open there."""
    present = tr["present"]
    if present.sum() < 0.1 * h:
        return 0.0, None
    inside = np.zeros(line.shape, bool)
    ys = np.where(present)[0]
    for y in ys:
        a, b = int(tr["xL"][y]) + 3, int(tr["xR"][y]) - 2
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
    w_exp = (Wd * P["pixel_um"] / um_per_px) if (Wd and np.isfinite(um_per_px)) else 30.0
    rot, (x0, y0, x1, y1), ang = detect_device(gray)
    dev = rot[y0:y1, x0:x1]
    h, w = dev.shape
    N, line, arc, blob, R = dark_maps(dev)
    pitch_exp = (P["wall_mm"] * 1000.0 / um_per_px + w_exp) if np.isfinite(um_per_px) else None
    channels, walls, pitch, diag, faces = find_channels(N, line, arc, blob, h, w, w_exp, pitch_exp)

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
        bx = int(P["border_skip"] * w) + 2
        x_lo = int(max(bx, lo + P["roi_margin_px"]))
        x_hi = int(min(w - bx, hi - P["roi_margin_px"]))
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
    for tr in tracks:
        cl = classify_channel(tr, h, w_ref, arc, w_exp, blob, R)
        td_frac, td_mask = touchdown(line, arc, tr, h)
        tr.update(cl)
        tr.update(touchdown_frac=td_frac, td_mask=td_mask,
                  touchdown=(td_frac >= P["touchdown_min_frac"] and cl["cls"] in ("FORMED", "NECKED", "PINCHED")))
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
        if r["td_mask"] is not None and r["touchdown"]:
            yy, xx = np.nonzero(r["td_mask"])
            ax.scatter(xx[::3], yy[::3], s=1, color="yellow", alpha=0.35, linewidths=0)
        if np.isfinite(r["defect_y"]):
            ax.plot([r["x"]], [r["defect_y"] * h], marker="x", color="yellow", ms=9, mew=2)
        extra = ""
        if r["cls"] == "PINCHED":
            extra = f"\nclosed {100 * r['closed_len_frac']:.0f}% of L"
        elif r["cls"] == "NECKED":
            extra = f"\nmin {100 * r['w_min_frac']:.0f}%, {100 * r['necked_len_frac']:.0f}% of L"
        if r["blind"]:
            extra += "\n(blind: one port)"
        if np.isfinite(r["tip_gap_frac"]) and r["tip_gap_frac"] > 0 and r["cls"] == "PINCHED":
            extra += f"\ntip gap {100 * r['tip_gap_frac']:.0f}% of L"
        if r["touchdown"]:
            extra += f"\ntouchdown {100 * r['touchdown_frac']:.0f}%"
        if r["channel_confidence"] == "LOW":
            extra += "\n(low confidence)"
        tag = "*" if r["predicted"] else ""
        ax.text(r["x"], -0.02 * h, f"{r['index']}{tag}\n{r['cls']}{extra}", ha="center", va="bottom",
                fontsize=6.5, color=col, fontweight="bold")
    pitch_txt = "n/a" if info["pitch_px"] is None else f"{info['pitch_px']:.0f}"
    title = (f"{info['file']}   |   H{info['H']} W{info['W']}   |   {info['um_per_px']:.1f} um/px, "
             f"design lumen {info['w_exp_px']:.0f} px, printed reference {info['w_ref_px']:.0f} px, pitch {pitch_txt} px   |   "
             f"{info['n_channels']} channels, {info['n_walls']} walls, ports {info['diag']['port_source']}, "
             f"confidence {info['diag']['confidence']}")
    ax.set_title(title, fontsize=8.5, pad=40)
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


CLASS_LETTER = dict(FORMED="F", NECKED="N", PINCHED="P", NOT_FORMED="X", UNRESOLVED="U")


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
                    color="black" if c["excluded"] or c["cls"] in ("NECKED", "UNRESOLVED") else "white")
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
            ml_rows.append(dict(H_layers=Hl, print_mode=m, W_px=Wd, ML=ml, n_replicates=len(use), cls=cls,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="images")
    ap.add_argument("--out", default="results")
    ap.add_argument("--only", default=None, help="substring filter on file names")
    ap.add_argument("--bar-mm", type=float, default=P["bar_mm"], help="physical length of the scale bar")
    ap.add_argument("--pixel-um", type=float, default=P["pixel_um"], help="printer XY pixel pitch")
    ap.add_argument("--no-debug", action="store_true")
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
    for fp in files:
        fn = os.path.basename(fp)
        m = meta.get(fn, {})
        if m.get("view", "top") != "top":
            print(f"skip {fn} (view={m.get('view')})")
            continue
        if args.only and args.only not in fn:
            continue
        try:
            info, results = process_image(fp, args.out, m, save_debug=not args.no_debug)
        except Exception as e:
            print(f"FAILED {fn}: {e!r}")
            rows.append(dict(file=fn, error=repr(e)))
            continue
        summary = ", ".join(f"{r['index']}:{r['cls'][:4]}" + (f"({100 * r['closed_len_frac']:.0f}%)" if r["cls"] == "PINCHED" else "")
                            + ("+TD" if r["touchdown"] else "") for r in results)
        print(f"{fn[-26:]:26s} H{info['H']} W{str(info['W']):>3s} wexp {info['w_exp_px']:5.0f} wref {info['w_ref_px']:5.0f} "
              f"n {info['n_channels']:2d} conf {info['diag']['confidence']:6s} -> {summary}")
        base = dict(file=fn, H_layers=info["H"], W_px=info["W"], replicate=info["rep"],
                    scalebar_px=info["scalebar_px"], um_per_px=fmt(info["um_per_px"], 2),
                    w_exp_px=fmt(info["w_exp_px"], 1), w_ref_px=fmt(info["w_ref_px"], 1),
                    device_h_px=info["device_h_px"], device_w_px=info["device_w_px"],
                    pitch_px=fmt(info["pitch_px"], 1), n_walls=info["n_walls"],
                    port_source=info["diag"]["port_source"], device_confidence=info["diag"]["confidence"],
                    print_mode=print_mode_of(fn, tags), w_ref_um=fmt(info["w_ref_px"] * info["um_per_px"], 1),
                    note=info["diag"].get("note", ""))
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
