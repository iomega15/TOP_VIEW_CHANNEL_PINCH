# Top-view channel-formation analysis of 3D-printed microfluidic channels

> **Status: under active development.** The preprint (22 September 2026)
> describes an earlier version with four classes; the `TOUCHDOWN` class, the
> closure regimes and the count enforcement were added after an expert review
> in October 2026. A second review round (November 2026) added the wide-design
> rule (a port-to-port path test, doubled walls), the lattice fit to lumen ends
> and step lines, and the entrance, pinch and stub measurements. Expect further
> changes.

Python program that looks at a top-view micrograph of a 3D-printed
*printability device* (a row of parallel enclosed channels running from one face
of the device to the other) and reports, for every channel, whether it formed
from end to end, where and how much it narrows, and over what fraction of its
length it is closed. It complements the side-view programs
[SAG_ANALYSIS_SAM_SEGMENTATION](https://github.com/iomega15/SAG_ANALYSIS_SAM_SEGMENTATION)
and [VALVE_CLOSURE_ANALYSIS](https://github.com/iomega15/VALVE_CLOSURE_ANALYSIS),
which see one cross-section per image. All three are described in the preprint
*Cross-Section and Top-View Image-Analysis Pipelines for Automated
Characterization of 3D-Printed Microfluidic Channels and Valves*
(Nguyen et al., 2026); the companion valve-design framework is
[engrXiv 10.31224/8053](https://doi.org/10.31224/8053).

## Classes (per channel)

| Class | Meaning |
|---|---|
| `FORMED` | open from port to port, narrowest point at least 75% of the device's printed reference width |
| `NECKED` | open from port to port, narrower than 75% somewhere (a partial pinch) |
| `PINCHED` | closed by cured resin between the side walls; `closed_len_frac` = closed fraction of the length; `blind` = open from one port only |
| `TOUCHDOWN` | the channel formed, but the roof membrane rests on the floor (often around a trapped bubble); `roof_down_frac` = fraction of the length where the roof is down, 1.0 when nothing of the lumen is visible from above |
| `NOT_FORMED` | nothing beyond the port wedges (a stub longer than 4% of the length counts as a lumen) |

Two things close a channel, and cross-sections tell them apart: in narrow
channels a uniform plug of cured resin joins the side walls; in wide channels
the roof sags onto the floor. From above both look like a lumen that stops. The
program decides by design width (`regime` column): at W <= 25 printer px a
closure is a plug (`PINCHED`), at W >= 50 px it is the roof (`TOUCHDOWN`), and in
between the lumen ends decide: tips that taper to a point (inner/base width
below 0.5) mean a plug, blunt ends mean the roof came down. These bounds are
the `plug_w_px`, `touchdown_w_px` and `plug_taper` parameters and came from two
expert reviews of 242 frames (October and November 2026).

A wide design (W >= `touchdown_w_px`) needs its own rule. When its roof comes
down, the side walls vanish from the top view and only an open pocket remains
at each port, closed by a curved contact line, so a wall tracker follows
contact lines instead of walls. The program therefore calls a wide channel open
only on positive evidence, and `TOUCHDOWN` otherwise:

1. a free path joins the two ports without crossing a dark outline
   (`path_open`; when it is blocked, `path_gap_frac` is the stretch between the
   two pockets),
2. both walls were followed over at least 60% of the length between the port
   flares (`open_wall_min`),
3. no closed outline inside the lumen covers 5% of its area or more
   (`touchdown_patch_frac`), and
4. no second dark line runs beside a wall for 40% of the length or more
   (`wall_companion_frac` against `comp_min`). Such a doubled wall is a thin
   roof lying on the floor between two open slivers along the walls.

A neck with a free path through it stays `NECKED`, however narrow it is.

Also reported: narrowest width and its position, necked fraction, defect
position, `touchdown_frac` (area enclosed by an inner closed outline inside an
open lumen, a bubble or a patch of roof on the floor), and channel/device
confidence (`LOW` for a cell placed without any visible port, classified
across a long bridged gap, or called `TOUCHDOWN` with walls partly followed and
no contact line found).

### Entrance-to-pinch measurements

A device shows two horizontal edges near each face. The outer edge belongs to
the base on which the walls stand and runs across the whole device. The inner
edge is the edge of the roof, where the lumen opens: a dark line across the
mouth of each channel that is absent over the solid walls between channels.
The program takes that inner edge as the *lumen entrance* and measures every
distance along the channel from it (`entrance_top_frac`, `entrance_bot_frac`,
`lumen_len_um`). The roof edge is one straight line per face, so the device
value is the median over the channels that show it; a channel whose edge did
not form inherits it (`entrance_seen_top`, `entrance_seen_bot` are then
`False`).

The lumen of a closed channel is in parts. The pinch is the longest stretch
that is not lumen (between two parts, or between a part and the far entrance),
so it can never lie inside a part. In a wide design with no free path the two
lumen parts are the port pockets, and both ends of the pinch come from the
flood of the path test, repeated at looser outline thresholds (`extent_taus`)
until the two pockets separate. The tracked walls do not set those ends,
because the tracker can follow the faint side lines of the channel cell past
the end of a short pocket. The tracked walls do decide which threshold to
trust. When a looser threshold ends a pocket more than 15% of the lumen length
earlier (`leak_jump`), the longer pocket stands only if the tracked walls
continue through the disputed stretch; a run without walls longer than 8% of
the lumen length (`leak_gap`) means that the flood leaked through a faint
contact line. The pockets also replace a stretch that the tracker left out at
an entrance when that stretch lies inside a pocket, because it is then the
flare of the pocket and not a pinch. Where a pocket ends in a spike, the end
is the tip of the spike, so the x at the middle of the bracket marks the middle
of the roof-down stretch and not the spike. A tongue that hangs from a funnel
through a narrow neck is cut off when the gaps in the outlines are sealed, so
the flood is repeated with a finer sealing (`path_seal_fine`). The rows that
the finer flood adds to a pocket count only when they are the clean inside of
a pocket: the flood must not touch a side of its search corridor there, and it
must fill at least 70% of its own extent in each row (`leak_fill`), which a
flood that seeps through surface grain does not. Reported for
`PINCHED` and `TOUCHDOWN` channels, as fractions of the lumen length and in
micrometers:

| Column | Meaning |
|---|---|
| `ent_to_pinch_top_frac`, `ent_to_pinch_bot_frac` | distance from the top and from the bottom entrance to the pinch |
| `pinch_len_frac` | length of the pinch |
| `pinch_center_offset_frac` | middle of the pinch minus mid-length (positive toward the bottom face) |
| `pinch_symmetric` | `True` when that offset is within 5% of the lumen length (`sym_tol`) |
| `n_lumen_parts` | number of separate lumen parts |

A closed channel whose lumen is one blob (a roof patch inside it, or doubled
walls) has no pinch stretch; the marker then sits on the patch.

For `FORMED` and `NECKED` channels the two halves are measured separately,
because the two narrowings rarely mirror each other: `neck_depth_top` and
`neck_depth_bot` give the fraction of the reference width lost at the
narrowest point of each half, and `neck_pos_top_frac` and `neck_pos_bot_frac`
its distance from that half's entrance. The first 7% of the lumen length at
each entrance is skipped (`ent_flare`, the port flare). `w_center_frac` is the
median width of the middle fifth of the lumen over the reference width. A
channel is `bulge = True` when that width exceeds both half minima by at least
10% of the reference width (`bulge_min`); `bulge_excess_frac` is the excess.

A channel that did not form still leaves a short *stub* at each port: a dark
cup or crescent shorter than the port wedge, which the wall tracker therefore
does not count as lumen. The program records the shape of that stub at both
ports of a `NOT_FORMED` channel and at the closed port of a `PINCHED` channel
that is open from one port only. It joins the dark pixels and the dark outlines
in a window around the channel mouth, fills the holes, removes everything
thinner than 12% of the design width (`stub_open`: cracks, the roof edge, the
base edge), and keeps the largest piece that begins at the entrance. The search
covers the first 25% of the lumen length (`stub_max_len`). A piece that runs
past that limit, or that spans the search window from side to side, is surface
texture or a lumen and not a stub, and the program leaves it unmeasured.
`stub_top_len_frac` and `stub_top_len_um` give the length from the entrance to
the tip, `stub_top_w_um` the width (90th percentile of the row widths, so the
flare at the mouth counts) and `stub_top_area_um2` the area; the `stub_bot_*`
columns hold the same for the bottom port. At a port where a tracked lumen part
begins, `ent_to_pinch_*` already gives the length of that part and the stub
columns stay empty. A stub is a lumen part: in a `PINCHED` channel the pinch
starts at the tip of the stub and not at the entrance.

The overlay draws all of this so that it can be checked by eye: green lines at
the two entrances (dotted where inherited), a yellow bracket over the pinch
with an x at its middle, a yellow x on each narrowing of 25% or more in an open
channel, a yellow diamond at mid-length for a bulge, and an orange outline
around each stub. The second title line
of every overlay explains the abbreviations in the channel labels.

### Lumen geometry by half

Lumen geometry per channel (columns `*_top` for the half that starts at the top
face of the frame, `*_bot` for the half that ends at the bottom face; a
continuous channel is split at mid-length): `w_base_*_um` = width near the port
(median of the outer 20% of the half, after the port flare), `w_inner_*_px` =
width at the tip or at mid-length (innermost 10%), `taper_*` = inner / base
(1 = constant width, 0.5 = tapers to half), `narrowing_*` = width lost per unit
length toward the middle from a linear fit (px per px, positive = thinning),
`open_*_len_frac` = length of the half as a fraction of the channel, and
`tip_gap_frac` / `tip_gap_um` = distance between the two inner tips (0 for a
continuous channel; a missing half counts from the opposite face). These tell
a wide stub that stops far from the middle from a narrow one whose tips nearly
meet. The *reference width*
is the printed width (median over the device's channels of the 85th percentile
of their open width), not the design width.

### Maps over design width and roof thickness

With ML-named frames the program draws one map per measured quantity, with the
design width W on the horizontal axis and the roof thickness ML on the vertical
axis, one panel per height and print mode. A cell holds the mean over its
replicates (one observation per physical channel; channels of low-confidence
devices do not count). The color runs from light to dark in one hue and
saturates at 100. `summary_by_H_W_ML.csv` holds the same cell means.

| Map (`*_map_W_ML.png`) | Quantity | Column in `channel_results.csv` |
|---|---|---|
| `channel_state` | class of the channel, by majority | `cls` |
| `replicates` | number of imaged channels that count for the cell | |
| `open_fraction` | share of the replicates that are open end to end, with the count open / imaged | |
| `gap_between_lumen_parts` | length without lumen, % of the lumen length | `gap_len_frac` |
| `narrowing_deepest` | width lost at the narrowest point, % of the reference width | `narrow_max_frac` |
| `narrowing_mean` | width lost on average along the length, % of the reference width | `narrow_mean_frac` |
| `bulge_mid_length` | width of the middle beyond both narrowings beside it, % of the reference width | `bulge_excess_frac` |
| `lumen_quality` | 100 minus the mean distance of the width from the reference width, % | `lumen_quality` |
| `width_printed_vs_design` | printed lumen width, % of the design width | `w_print_over_design` |
| `length_printed_vs_design` | printed lumen length, % of the design length | `lumen_len_over_design` |
| `pitch_printed_vs_design` | channel pitch, % of the design pitch | `pitch_over_design` |

The length without lumen is 0 for a channel that is open end to end, the pinch
for a closed channel, and everything except the two port stubs for a channel
that did not form. The two narrowing measures, the bulge and the quality apply
to open channels (`FORMED`, `NECKED`) and read the width between the two port
flares. The width of an open channel varies along its length, so two numbers
summarize the narrowing: the deepest point, which limits the flow, and the
mean, which says how much of the lumen is affected. In the mean, a row wider
than the reference counts as 0, so that a bulge does not cancel a neck. The
quality counts both: it is 1 minus the mean of |width / reference - 1| along
the length, so a lumen that neither narrows nor bulges scores 100% and both a
neck and a bulge lower the score. In these four maps a cell without an open
replicate is gray and carries the letter of its class. Every other cell shows
the mean of its open replicates with the count (open / imaged) under it: 1/3
means that one of three imaged replicates is open and the number is the value
of that one channel; 3/3, 2/2 and 1/1 mean that every imaged replicate is open.
The count sits on a red tag when half or fewer of the imaged replicates are
open (1/3, 1/2), on a yellow tag when two of three are open or when all are
open but fewer than three replicates were imaged (2/3, 2/2, 1/1), and on a
green tag (3/3) when all three were imaged and are open. On the
`open_fraction` map, where the number is the share itself, the red tag does
not apply.

The last three maps compare the print with the design. The printed width is
the median width of an open channel, or the width of the lumen parts near the
ports for a pinched channel. The printed length runs from one lumen entrance
to the other. The pitch is the center-to-center distance of the channels; it
does not depend on how far the walls cured into the lumen, so it shows the
scale of the print and of the image. The design dimensions come from the
defaults of the device generator (`printability_device.scad`): a post of
40 x 40 printer pixels (`design_post_px`), two posts between neighboring
channels, and a channel that runs over 4 cells of two posts each
(`design_rows`), which gives 2.56 mm of solid between channels and a channel
10.24 mm long at 32 um per pixel. Change the two parameters if a device was
printed with other values. On the 242 frames of the reviewed set the pitch is
87% to 97% of the design pitch, one value per imaging session, and the lumen
length follows it about two points lower. A ratio that changes with the
imaging session points to the scale of the image more than to the print, and
the images alone cannot tell the two apart; a caliper measurement of one device
can. The printed width is not a fixed fraction of the design width: at a
height of 5 layers the thinnest roof prints at about 80% to 95% of it and
thicker roofs at about 40% to 70%.

## How it works
1. Footer strip and scale bar (bar length in px; its physical length is `--bar-mm`,
   2.00 mm by default); design width in image px = W x printer pixel / scale.
2. Device detection on a local-texture map; tilt from the device edges; flat field.
3. Dark-line, dark-arc, and dark-blob response maps (1 = 30% darker than the local mean).
4. Ports (dark crescents/wedges at the faces), split or merged against the design width.
5. Port pairing, channel pitch (design pitch unless three or more pairs support a
   measured one; with the channel count known from the file name, the span of the
   ports on a face gives the pitch directly), lattice completion. When the name
   gives the count, the lattice is refit on the port-bearing channels and the
   window of that many consecutive slots with the most port evidence is kept, so
   an extra cell at the frame edge is dropped and a missed channel keeps its number.
   With a known count, a second lattice is then fitted to two cues that tell a
   lumen from the straight line between two channels. A lumen that pinched or
   whose roof came down leaves a dark shape at each port and little at
   mid-length (a tapered stub in a narrow design, a pocket in a wide one), so
   the dark fraction of the two face bands minus that of the mid-length band,
   averaged over a channel-wide window, peaks at the channel centers. The roof
   thickness steps along a straight line midway between channels, which stays
   dark over the full length. The second lattice replaces the port-based
   channels only when it scores at least 15% better and moves a channel by at
   least 15% of the pitch (`lat_gain`, `lat_shift`), or when the port-based
   count is wrong; `lattice_note` in `channel_results.csv` records the two
   scores and the decision. The fit ran on wide designs only until
   19 November 2026 and now runs on every design (`lat_min_w_px`).
6. Dynamic-programming tracking of the left and right wall of each channel, both orders.
7. Presence (continuity of the tracked line; runs shorter than the port wedge do
   not count), gap logic (tapering tips, a dark cap, or in narrow designs a gap
   longer than 12% of the length mean the lumen is closed there; shorter gaps are
   tracking dropouts and are bridged), reference width, classification by regime
   (wide designs by the four-part rule above), touchdown.

All tunables are in the `P` dictionary at the top of
`topview_channel_pipeline.py`, one comment each.

## Requirements
Python 3.10 with `opencv-python` (tested 5.0), `numpy` (2.2), `scipy` (1.15),
`matplotlib` (3.10). `python-pptx` only for `extract_pptx_images.py`.

    pip install -r requirements.txt

## Usage
File names must contain `_H<layers>_W<printer px>`, or the folder must hold a
`manifest.csv` (written by `extract_pptx_images.py`).

A printability device carries ten channels with roof (membrane) thickness
ML1 to ML10; wide designs need several overlapping frames. Name each frame
`..._R<replicate>_top_H<layers>_W<px>_ML<a>-<b>.jpg`, where `ML<a>-<b>` lists
the channels of that frame from left to right (`ML10-7` = ML10, 9, 8, 7). With
such names the program assigns an ML value to every channel, keeps one
observation per physical channel where frames overlap (the one farthest from the
frame edge), and writes `summary_by_H_W_ML.csv` and `channel_state_map_W_ML.png`
(a width x ML map per height). Frames whose detected channel count cannot be
matched to the ML range are flagged in `ml_note` and left out of the map.
`check_renamed_set.py` compares a renamed batch with the images already present.

    python topview_channel_pipeline.py --images examples --out results_examples
    python topview_channel_pipeline.py --images my_images --out results --only W100 --bar-mm 2.0 --pixel-um 32
    python make_contact_sheet.py results

Options: `--only <substring>` runs matching files, `--bar-mm` and `--pixel-um`
override the scale-bar length and the printer pixel pitch (32 um, Asiga Ultra),
`--no-debug` skips the overlays, `--jobs <n>` processes n frames in parallel,
`--tags <csv>` points at a frame-tag file
(default `frame_tags.csv` next to the script): rows of `pattern,print_mode`,
where `pattern` is a regular expression on the file name. Frames of different
print modes (for example the Asiga transparency mode on or off) are pooled and
mapped separately, and `summary_by_print_mode.csv` puts the modes side by side
per design (class fractions, widths, taper, tip gap). Setting the environment variable `TV_DEBUG=1`
prints per-channel tracking diagnostics.

Outputs: `channel_results.csv` (one row per channel), `device_summary.csv`
(one row per image), `summary_by_H_W.csv` (one row per design and print mode,
low-confidence devices excluded), `class_fractions_vs_W.png`, and one
`*_overlay.png` per image (tracked walls in class colors, response map, width
profile). ML-named frames add `summary_by_H_W_ML.csv` (one row per height,
print mode, width and roof thickness) and the eleven maps listed under "Maps
over design width and roof thickness".

`examples/` holds two top views: `example_H5_W50.jpg` (expert-reviewed; three
open-channel outcomes side by side) and `example_H10_W100_2.jpg`.

## Reviewing the calls
`make_review_folder.py <folder> <results dirs...>` builds a small folder for a
reviewer: one JPEG overlay per frame, a spreadsheet with one row per frame
(what the code decided, columns for what was right and wrong by channel
number, and a column for entrance and pinch marks that are off) and a README.
`score_against_review.py review_sheet.xlsx <results dirs...>` turns the filled-in sheet into a per-channel truth table and scores
both the decisions the reviewer saw and a new results folder against it, so a
code change can be measured before the next review round. It also counts the
channels the reviewer called mislocated or undetected, which it scores apart
from the class. `--round1` selects the sentence parser and the looser touchdown
match that the October 2026 scores used.

## Known limitations
The plug-versus-touchdown decision at W30 and W40 rests on the shape of the
lumen ends and is the least certain call. In a wide design, an open lumen whose
walls are too faint to follow is called `TOUCHDOWN` with low confidence, and a
trapped bubble that fills the lumen from wall to wall passes for an open
channel. At W5 to W20 a stub shorter than a few percent of the length is easily
missed, and in frames with strong surface texture the tracker can follow the
texture and report a stub that is not there. In a narrow design whose lumens
are too faint to show at the ports, the lattice fit has little to hold on to
and the channels can still sit off the lumens (such frames carry low
confidence). The entrance, pinch and stub
measurements have not yet been checked against a reviewer; a pocket that ends in a thin
spike is measured to where the spike becomes too thin to resolve, and a channel
placed off its lumen gives meaningless distances. Every frame is divided by a
blurred copy of itself before analysis, which removes a brightness gradient but
amplifies the grain of an under-exposed side; where the surface grain is as
dark as a faint contact line, the pocket ends are uncertain and a tongue too
thin to flood is measured short. A misread scale bar shifts every
micrometer value of that frame; the lattice does not depend on it when the file
name carries the channel count. The pitch map suggests that the scale of the
reviewed images is 3% to 13% off, which would shift every micrometer value and
every comparison with the design by that much. The narrowing, bulge and quality
values inherit every error of the wall tracker, so an isolated extreme value
needs a look at its overlay. Most cells of the reviewed set hold one or two
replicates (the `replicates` map), so a cell mean there is not a statistic. The thresholds were set against one reviewer's
calls on 242 frames of one resin, so the agreement figures are not an
independent accuracy estimate.

## License
MIT, see [`LICENSE`](LICENSE).
