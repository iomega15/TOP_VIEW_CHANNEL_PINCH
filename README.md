# Top-view channel-formation analysis of 3D-printed microfluidic channels

> **Status: under active development.** The classes and thresholds below are
> the version described in the preprint (22 September 2026); expect changes.

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
| `PINCHED` | closed somewhere (narrower than 15%); `closed_len_frac` = closed fraction of the length; `blind` = open from one port only |
| `NOT_FORMED` | nothing beyond the port wedges, or open over less than 20% of the length |
| `UNRESOLVED` | both walls lost over more than half the length with no closure evidence; review by eye |

Also reported: narrowest width and its position, necked fraction, defect
position, `touchdown_frac` (area enclosed by an inner closed outline, i.e. the
roof resting on the floor), and channel/device confidence. The *reference width*
is the printed width (median over the device's channels of the 85th percentile
of their open width), not the design width.

## How it works
1. Footer strip and scale bar (bar length in px; its physical length is `--bar-mm`,
   2.00 mm by default); design width in image px = W x printer pixel / scale.
2. Device detection on a local-texture map; tilt from the device edges; flat field.
3. Dark-line, dark-arc, and dark-blob response maps (1 = 30% darker than the local mean).
4. Ports (dark crescents/wedges at the faces), split or merged against the design width.
5. Port pairing, channel pitch (design pitch unless three or more pairs support a
   measured one), lattice completion.
6. Dynamic-programming tracking of the left and right wall of each channel, both orders.
7. Presence (continuity of the tracked line), gap logic (closure vs. tracking
   dropout vs. debris), reference width, classification, touchdown.

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
`--no-debug` skips the overlays. Setting the environment variable `TV_DEBUG=1`
prints per-channel tracking diagnostics.

Outputs: `channel_results.csv` (one row per channel), `device_summary.csv`
(one row per image), `summary_by_H_W.csv` (one row per design, low-confidence
devices excluded), `class_fractions_vs_W.png`, and one `*_overlay.png` per image
(tracked walls in class colors, response map, width profile).

`examples/` holds two top views: `example_H5_W50.jpg` (expert-reviewed; three
open-channel outcomes side by side) and `example_H10_W100_2.jpg`.

## Known limitations
Faint walls lead to `UNRESOLVED` or low-confidence calls; a channel ending in a
rounded blind tip without a dark rim can be bridged and reported as formed or
necked; touchdown needs a closed inner outline (air bubbles can count); the
thresholds were set on a small expert-reviewed set.

## License
MIT, see [`LICENSE`](LICENSE).
