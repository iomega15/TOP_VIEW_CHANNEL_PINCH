"""Score a results folder against a reviewer's filled-in review_sheet.xlsx.

The reviewer wrote, per frame, which channels the code got right and what the wrong ones should
be, using the channel numbers of the overlays that were reviewed (the "what the code decided"
column of the sheet). This script turns those sentences into a per-channel truth table and scores
(a) the decisions the reviewer saw (baseline) and (b) a new results folder against it.

Usage:
    python score_against_review.py review_sheet.xlsx results_H5_... results_H10_... [--csv out.csv]

Truth vocabulary: FORMED, NECKED, PINCHED, NOT_FORMED, TOUCHDOWN ("formed but the membrane touched
down"). A channel the reviewer listed as right keeps the decision she saw. Sentences it cannot parse
are listed at the end, so the parsing can be checked.
"""
import argparse
import collections
import csv
import os
import re

import openpyxl

NUM = r"(?:ch(?:annel)?s?\s*)?(\d+(?:\s*(?:,|&|and|to|-|–)\s*\d+)*)"


def parse_list(s):
    """'1,2 and 4-10' / '1&3' / '2 to 6' -> [1, 2, 4, 5, ..., 10]."""
    out = []
    for part in re.split(r"\s*(?:,|&|and)\s*", s.strip()):
        m = re.fullmatch(r"(\d+)\s*(?:-|–|to)\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out += list(range(min(a, b), max(a, b) + 1))
        elif re.fullmatch(r"\d+", part):
            out.append(int(part))
    return out


def parse_decided(text):
    """'ch2 (ML3): FORMED, touchdown [low confidence]' lines -> {2: 'FORMED+td'}."""
    out = {}
    for m in re.finditer(r"ch(\d+)[^:]*:\s*([A-Z_]+)(,\s*touchdown)?", text or ""):
        out[int(m.group(1))] = m.group(2) + ("+td" if m.group(3) else "")
    return out


CLASS_WORDS = [
    (r"formed,?\s*but\s*(?:the\s*)?membr?ane?\s*touch", "TOUCHDOWN"),
    (r"formed\s*but\s*with\s*bubble", "TOUCHDOWN"),
    (r"membrane\s*touched?\s*down", "TOUCHDOWN"),
    (r"not\s*formed", "NOT_FORMED"),
    (r"pinch", "PINCHED"),
    (r"necked", "NECKED"),
    (r"formed", "FORMED"),
]


def class_of(phrase):
    p = phrase.lower()
    for rx, c in CLASS_WORDS:
        if re.search(rx, p):
            return c
    return None


def cell_text(v):
    """Excel turned '3-6' into a date (June 3): give the text back as 'day-month'."""
    if v is None:
        return ""
    if hasattr(v, "month") and hasattr(v, "day"):
        return f"{v.day}-{v.month}"
    return str(v)


def parse_truth(right, wrong, decided, n_found):
    """Per-channel truth from the reviewer's RIGHT and WRONG sentences."""
    truth, unparsed, flags = {}, [], set()
    right, wrong = cell_text(right), cell_text(wrong)
    r = (right or "").strip()
    if re.fullmatch(r"(?i)all\.?", r) or re.search(r"(?i)^all\b", r):
        truth.update(decided)
    else:
        for m in re.finditer(NUM, r):
            for k in parse_list(m.group(1)):
                if k in decided:
                    truth[k] = decided[k]
    w = (wrong or "").strip()
    if not w or re.fullmatch(r"(?i)nothing\.?", w):
        return truth, unparsed, flags
    for clause in re.split(r"\s*[;.]\s*(?=[A-Za-z]|\d)", w):
        c = clause.strip()
        if not c:
            continue
        lc = c.lower()
        if re.search(r"instead of \d+|fails? to detect|not detected|incorrect(ly)? (locat|number)|misidentif|no channel found|do not have channel", lc):
            flags.add("count/location")
        cls = class_of(c)
        if cls is None:
            unparsed.append(c)
            continue
        if re.search(r"\ball\b", lc) and not re.search(r"\d", lc.split(" are ")[0] if " are " in lc else lc):
            for k in range(1, max(n_found, len(decided)) + 1):
                truth.setdefault(k, cls)
            continue
        if "remaining" in lc:
            for k in range(1, max(n_found, len(decided)) + 1):
                truth.setdefault(k, cls)
            continue
        # several statements in one clause: "1 is formed, 2 is necked, 3 is formed but ..."
        segs = re.split(r",\s*(?=\d+\s*(?:is|are|&|,|and|-|to)\b)", c)
        if len(segs) > 1 and all(class_of(s) for s in segs):
            for s in segs:
                for m in re.finditer(NUM, s):
                    for k in parse_list(m.group(1)):
                        truth[k] = class_of(s)
            continue
        found = False
        for m in re.finditer(NUM, c):
            for k in parse_list(m.group(1)):
                truth[k] = cls
                found = True
        if not found:
            unparsed.append(c)
    return truth, unparsed, flags


def match(truth, got):
    """Class match; the touchdown flag only matters when the truth is a touchdown call."""
    if got is None:
        return False
    if truth in ("TOUCHDOWN", "FORMED+td"):
        return got in ("TOUCHDOWN", "TOUCHDOWN+td", "FORMED+td", "NECKED+td")
    return truth.split("+")[0] == got.split("+")[0]


def load_results(dirs):
    by_file = collections.defaultdict(dict)
    for d in dirs:
        with open(os.path.join(d, "channel_results.csv"), newline="") as f:
            for r in csv.DictReader(f):
                if r.get("channel") and r["channel"] != "0" and r.get("cls"):
                    lab = r["cls"] + ("+td" if r.get("touchdown") in ("True", True) and r["cls"] != "TOUCHDOWN" else "")
                    by_file[r["file"]][int(r["channel"])] = lab
    return by_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("results", nargs="*")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    ws = openpyxl.load_workbook(a.sheet, data_only=True)["frames"]
    rows = list(ws.iter_rows(values_only=True))
    hdr = rows[0]
    new = load_results(a.results) if a.results else None
    tot = collections.Counter()
    conf_old, conf_new = collections.Counter(), collections.Counter()
    out_rows, unparsed_all = [], []
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        if not d.get("overlay file"):
            continue
        fn = os.path.basename(str(d["overlay file"])).replace("_overlay.jpg", ".jpg")
        decided = parse_decided(d.get("what the code decided"))
        n_exp = int(d["channels expected"] or 0)
        n_found = int(d["channels found"] or 0)
        truth, unparsed, flags = parse_truth(d.get("what it got RIGHT"), d.get("what it got WRONG (channel, what it should be)"),
                                             decided, n_found)
        unparsed_all += [(fn, u) for u in unparsed]
        got_new = new.get(fn, {}) if new is not None else None
        same_count = got_new is not None and len(got_new) == n_found
        n_ok_old = n_ok_new = 0
        for k, t in sorted(truth.items()):
            o = decided.get(k)
            tot["channels"] += 1
            if o is not None and match(t, o):
                n_ok_old += 1
            conf_old[(t, o)] += 1
            if got_new is not None:
                if same_count:
                    g = got_new.get(k)
                else:   # count changed: compare as multisets, so numbering shifts do not count as errors
                    g = None
                    for kk, gg in got_new.items():
                        if match(t, gg) and ("used", kk) not in truth:
                            g = gg
                            truth[("used", kk)] = 1
                            break
                if g is not None and match(t, g):
                    n_ok_new += 1
                conf_new[(t, g)] += 1
        truth = {k: v for k, v in truth.items() if not isinstance(k, tuple)}
        tot["ok_old"] += n_ok_old
        tot["ok_new"] += n_ok_new
        tot["frames"] += 1
        tot["count_ok_old"] += int(n_found == n_exp)
        if got_new is not None:
            tot["count_ok_new"] += int(len(got_new) == n_exp)
        out_rows.append(dict(file=fn, H=d["H"], W=d["W"], R=d["R"], ML=d["ML range"], verdict=d["overall: good / partly / wrong"],
                             n_expected=n_exp, n_found_old=n_found, n_found_new=len(got_new) if got_new is not None else "",
                             n_truth=len(truth), ok_old=n_ok_old, ok_new=n_ok_new if got_new is not None else "",
                             flags=";".join(sorted(flags)), truth=" ".join(f"{k}:{v}" for k, v in sorted(truth.items())),
                             new=" ".join(f"{k}:{v}" for k, v in sorted((got_new or {}).items()))))
    print(f"{tot['frames']} frames, {tot['channels']} channels with a reviewer truth")
    print(f"baseline (what the reviewer saw): {tot['ok_old']}/{tot['channels']} channels right "
          f"({100 * tot['ok_old'] / max(1, tot['channels']):.0f}%), channel count right in {tot['count_ok_old']}/{tot['frames']} frames")
    if new is not None:
        print(f"new results:                      {tot['ok_new']}/{tot['channels']} channels right "
              f"({100 * tot['ok_new'] / max(1, tot['channels']):.0f}%), channel count right in {tot['count_ok_new']}/{tot['frames']} frames")
    for name, conf in (("baseline", conf_old), ("new", conf_new)):
        if name == "new" and new is None:
            continue
        print(f"\n{name}: truth -> decision (count)")
        by_t = collections.defaultdict(list)
        for (t, g), n in conf.items():
            by_t[t].append((n, g))
        for t in sorted(by_t):
            items = sorted(by_t[t], key=lambda x: (x[0], str(x[1])), reverse=True)
            n_t = sum(n for n, g in items)
            n_ok = sum(n for n, g in items if g is not None and match(t, g))
            print(f"  {t:11s} n={n_t:4d} right {100 * n_ok / n_t:3.0f}%  " + ", ".join(f"{g}:{n}" for n, g in items))
    if unparsed_all:
        print(f"\n{len(unparsed_all)} reviewer sentence(s) not parsed:")
        for fn, u in unparsed_all[:40]:
            print(f"  {fn[-34:]}: {u[:110]}")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            wr.writeheader()
            wr.writerows(out_rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
