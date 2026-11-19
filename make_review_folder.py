"""Build a small, shareable folder for a reviewer to comment on the pipeline's calls, frame by frame.

The folder holds one JPEG overlay per frame (the picture the pipeline drew: tracked lumen walls in
class colors, port bars, class label per channel), a review spreadsheet with one row per frame that
lists what the pipeline decided and leaves comment columns to fill, and a README with instructions.
Nothing else is needed to review, so the folder stays small enough for a free Dropbox account.

Usage:
    python make_review_folder.py REVIEW_2026-09-30 results_H5_ML_2026-09-29 results_H10_ML_2026-09-29
"""
import csv
import glob
import os
import re
import sys

from PIL import Image
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

JPEG_QUALITY = 82
CLASSES = {
    "FORMED": "open from one port to the other, narrowest point at least 75% of the printed width",
    "NECKED": "open end to end but narrower than 75% somewhere (partial pinch)",
    "PINCHED": "closed by cured resin between the side walls; 'closed x% of L' = closed fraction of the length; 'blind' = open from one port only",
    "TOUCHDOWN": "formed, but the roof membrane rests on the floor (often around a trapped bubble); 'roof down x% of L' = fraction of the length",
    "NOT_FORMED": "no lumen beyond the port wedges",
}


def frame_meta(fn):
    m = re.search(r"_R(\d+)_top_H(\d+)_W(\d+)_ML(\d+)-(\d+)", fn)
    if not m:
        m2 = re.search(r"H(\d+)_W(\d+)", fn)
        return dict(R="", H=int(m2.group(1)) if m2 else "", W=int(m2.group(2)) if m2 else "", ML_range="", expected="")
    R, H, W, a, b = map(int, m.groups())
    return dict(R=R, H=H, W=W, ML_range=f"ML{a}-{b}", expected=abs(a - b) + 1)


def call_text(r):
    """One channel's decision in words."""
    txt = f"ch{r['channel']}"
    if r.get("ML") not in ("", None):
        txt += f" (ML{int(float(r['ML']))})"
    txt += f": {r['cls']}"
    if r["cls"] == "PINCHED":
        try:
            txt += f", closed {100 * float(r['closed_len_frac']):.0f}% of length"
        except ValueError:
            pass
        if r.get("blind") in ("True", True):
            txt += ", blind (one port only)"
    if r["cls"] == "TOUCHDOWN":
        try:
            txt += f", roof down {100 * float(r['roof_down_frac']):.0f}% of length"
        except (ValueError, KeyError):
            pass
    if r["cls"] == "NECKED" and r.get("w_min_frac") not in ("", None):
        txt += f", narrowest {100 * float(r['w_min_frac']):.0f}% of width"
    if r.get("touchdown") in ("True", True):
        txt += ", touchdown"
    if r.get("channel_confidence") == "LOW":
        txt += " [low confidence]"
    if r.get("predicted"):
        txt += " [cell placed by the lattice, no port seen]"
    return txt


def main(out_dir, results_dirs):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for rd in results_dirs:
        with open(os.path.join(rd, "channel_results.csv"), newline="") as f:
            chan = list(csv.DictReader(f))
        by_file = {}
        for r in chan:
            by_file.setdefault(r["file"], []).append(r)
        sub = os.path.join(out_dir, "overlays", os.path.basename(rd).split("_ML")[0].replace("results_", ""))
        os.makedirs(sub, exist_ok=True)
        for fn in sorted(by_file):
            base = os.path.splitext(fn)[0]
            src = os.path.join(rd, base + "_overlay.png")
            dst = os.path.join(sub, base + "_overlay.jpg")
            if os.path.exists(src) and not os.path.exists(dst):
                Image.open(src).convert("RGB").save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
            items = sorted(by_file[fn], key=lambda r: int(r["channel"] or 0))
            found = [r for r in items if r["cls"] in CLASSES]
            meta = frame_meta(fn)
            calls = "\n".join(call_text(r) for r in found) if found else "no channel found (no port pair)"
            rows.append(dict(overlay=os.path.relpath(dst, out_dir).replace(os.sep, "/"),
                             H=meta["H"], W=meta["W"], R=meta["R"], ML_range=meta["ML_range"],
                             expected_channels=meta["expected"], channels_found=len(found),
                             device_confidence=items[0].get("device_confidence", ""),
                             print_mode=items[0].get("print_mode", ""),
                             pipeline_calls=calls))

    # ---------------------------------------------------------------- spreadsheet
    wb = Workbook()
    ws = wb.active
    ws.title = "frames"
    head = ["overlay file", "H", "W", "R", "ML range", "channels expected", "channels found",
            "device confidence", "print mode", "what the code decided",
            "overall: good / partly / wrong", "what it got RIGHT", "what it got WRONG (channel, what it should be)",
            "entrance / pinch marks: what is off (channel, what you see)", "other comments"]
    ws.append(head)
    for c in range(1, len(head) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.fill = PatternFill("solid", fgColor="DDEBF7" if c <= 10 else "FFF2CC")
    for r in rows:
        ws.append([r["overlay"], r["H"], r["W"], r["R"], r["ML_range"], r["expected_channels"], r["channels_found"],
                   r["device_confidence"], r["print_mode"], r["pipeline_calls"], "", "", "", "", ""])
    widths = [58, 4, 5, 4, 9, 9, 9, 11, 14, 60, 16, 40, 48, 48, 40]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"good,partly,wrong"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"K2:K{len(rows) + 1}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(head))}{len(rows) + 1}"

    lg = wb.create_sheet("legend")
    lg.append(["class", "meaning"])
    for k, v in CLASSES.items():
        lg.append([k, v])
    lg.append([])
    lg.append(["colors in the overlay", "green FORMED, orange NECKED, red PINCHED, purple TOUCHDOWN, gray NOT_FORMED"])
    lg.append(["cyan bars at the faces", "ports (channel openings) the code found"])
    lg.append(["blue dotted lines", "device walls the code found"])
    lg.append(["yellow x", "where the code put the pinch or the narrowest point"])
    lg.append(["yellow fill", "touchdown: the roof resting on the floor inside an open channel"])
    lg.append(["numbers above the device", "channel numbers, left to right; use them in your comments"])
    lg.append(["* after a number", "cell placed by the lattice (no port seen there)"])
    lg.append(["low confidence", "the code itself is unsure about this channel or device"])
    lg.append(["lower panels", "the edge map the code tracked on, and the width along each channel (0 = top face, 1 = bottom face)"])
    lg.column_dimensions["A"].width = 26
    lg.column_dimensions["B"].width = 110
    for row in lg.iter_rows():
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    lg["A1"].font = lg["B1"].font = Font(bold=True)
    wb.save(os.path.join(out_dir, "review_sheet.xlsx"))

    # ---------------------------------------------------------------- README
    readme = f"""HOW TO REVIEW THE CHANNEL-DETECTION CODE ({len(rows)} frames)

What is in this folder
  overlays/H5/, overlays/H10/   one picture per frame you imaged. The top panel is your image with what the
                                code found drawn on it: the two walls of every channel it followed, colored by
                                its decision (green FORMED, orange NECKED, red PINCHED, purple TOUCHDOWN, gray
                                NOT_FORMED), the channel number and decision written above each channel, cyan
                                bars where it found the channel openings, blue dotted lines where it found
                                device walls, a yellow x where it put a pinch, yellow fill where it saw the
                                roof touching the floor. The two lower panels are for us; ignore them if you
                                like.
  review_sheet.xlsx             one row per frame. The blue columns are filled in (what the code decided, in
                                words). The yellow columns are yours.
  README.txt                    this file.

What we need from you
  For every frame, look at the overlay and tell us how the code did. We need the good and the bad:
  1. "overall": pick good, partly or wrong from the drop-down.
  2. "what it got RIGHT": say which channels it called correctly, in your own words.
     Example: "ch2-ch5 correct, they really are pinched near the middle; ch1 formed is right".
  3. "what it got WRONG": say which channel, what the code said, and what you see.
     Example: "ch3 is called NECKED but it is actually closed for about a quarter of its length";
     "ch6 was not found at all (no bar at the port)"; "the walls it drew on ch4 are the device wall,
     not the channel". If the number of channels found differs from the number expected, say which
     ones are missing or extra.
  4. "entrance / pinch marks": the overlay now also draws measurements. Tell us where one is off.
     - green lines across each channel near the two faces: the lumen entrances (the edge of the
       roof, where the lumen opens; not the edge of the base, which is farther out). Dotted
       means the code did not see the edge in that channel and used the one of its neighbors.
     - yellow bracket (a vertical line with two end bars) and an x at its middle: the pinch,
       drawn from where the upper lumen part ends to where the lower one begins. The x must lie
       between the two parts, never inside one.
     - "ent-pinch T.. B..": distance from the top entrance to the pinch and from the bottom
       entrance to the pinch, in percent of the lumen length. "pinch ..": its length. "off ..":
       how far its middle sits from mid-length (plus = toward the bottom face; "sym" = centered).
     - open channels: "neck T d@p B d@p" gives, for the top and bottom half, how much of the
       width is lost at the narrowest point (d, percent) and how far that point is from its
       entrance (p, percent of the lumen length). A yellow x marks a narrowing of 25% or more.
       A yellow diamond at mid-length marks a bulge: the middle is wider than both narrowings.
     - orange outline at a port: the short lumen stub that remains where the channel did not
       form (both ports of a NOT_FORMED channel, the closed side of a channel open from one
       port only). "stub T a x b" and "stub B a x b" give its length and width in micrometers
       at the top and bottom port. The outline should hug the dark cup at the port and nothing
       else.
     Example: "ch2: the lower end bar is inside the lumen, the lumen ends higher up";
     "bottom green line is on the base edge, not on the roof edge"; "ch1 has no bulge".
  5. "other comments": anything else, for example frames where the image itself is the problem
     (out of focus, debris, cut off), or where you are unsure what the right answer is.

  Plain language is fine. Short is fine. Use the channel numbers written above the device.
  If a frame is entirely correct, one word ("good") plus "all channels correct" is enough.

The vocabulary the code uses (also on the "legend" sheet)
  FORMED       open from one port to the other, narrowest point at least 75% of the printed width
  NECKED       open end to end but narrower than 75% somewhere (a partial pinch)
  PINCHED      closed somewhere; "closed x% of L" is the closed fraction of the channel length;
               "blind" means the lumen comes out of one port only
  NOT_FORMED   nothing beyond the port wedges, or the roof is down along the whole length
  TOUCHDOWN    the channel formed but the roof membrane rests on the floor, often around a trapped
               bubble; "roof down x% of L" is the fraction of the length where the roof is down
  touchdown    the roof membrane rests on the floor inside an otherwise open channel

When you are done, save review_sheet.xlsx in this same folder (keep the name) and tell Roman.
"""
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out_dir) for fn in fns)
    print(f"{len(rows)} frames -> {out_dir} ({total / 1e6:.0f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
