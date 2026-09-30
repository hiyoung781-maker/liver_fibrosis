"""Poster figures for the v2 campaign: the funnel, and the trade-off plane.

Both are computed from the same functions the report used, not from numbers
retyped out of it. scripts/poster_figures.py is v1's and its numbers (19,485
sampled, 389 survivors, 20 leads) do not describe this run.

FIGURE 2 COLOURS BY THE RULE, NOT BY THE AXIS. The toxicity stage requires
every one of SR-MMP, NR-AhR and DILI to beat PLN-1474; it does not average
them. If a point's colour came from the composite on the y-axis, gen_02452
(0.1383) and gen_03050 (0.1442) would sit below the reference line while both
FAIL the stage, and the figure would contradict the filter it illustrates.
Position uses the composite because two axes are readable and four are not;
colour uses the rule. The claim the figure supports is therefore precise: no
molecule that clears every endpoint sits to the right of the noise band.

Axis labels are English. This machine has no Korean font, so matplotlib
cannot render Korean glyphs -- they come out as boxes, which is worse than
English on a poster.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funnel import PLN1474_CUTOFF
from lead_cards import SEED_SD, rank_noise_band
from lead_filter import lead_rows, stages, summarize
from selectivity import label_variants
from tradeoff import TOXICITY_ENDPOINTS, pareto_front
from tradeoff import rows as tradeoff_rows

__all__ = ["ARM_COLOURS", "STAGE_LABELS", "funnel_figure", "tradeoff_figure"]

ARM_COLOURS = {"TL-A": "#c0392b", "TL-C": "#2471a3"}
DISPLAY = {"TL-A-prime": "TL-A", "TL-A": "TL-A", "TL-C": "TL-C"}

STAGE_LABELS = {
    "start": "docked\nproperty window",
    "geometry": "1  geometry\nMIDAS + Asn224",
    "affinity": f"2  affinity\n< {PLN1474_CUTOFF:+.3f} kcal/mol",
    "interaction": "3  interaction\nPLIP metal complex",
    "selectivity": "4  selectivity\n> PLN-1474",
    "toxicity": "5  toxicity\nall 3 endpoints",
}

# How many of the three endpoints exceed the reference.
FAIL_STYLE = {
    0: ("#1e8449", 70, "all 3 endpoints pass"),
    1: ("#e67e22", 45, "1 endpoint over reference"),
    2: ("#95a5a6", 16, "2 over"),
    3: ("#d5d8dc", 12, "3 over"),
}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })
    return plt


def funnel_figure(counts: dict, out_path: Path, arm: str = "TL-C") -> dict:
    """A funnel: stacked trapezoids narrowing at each stage.

    WIDTH IS LOG-SCALED AND THE FIGURE SAYS SO. 5,212 molecules enter and 1
    leaves; width proportional to count makes every stage after the first a
    line one pixel wide, and width proportional to nothing at all is a
    decoration. log10 keeps the shape readable, and a reader who assumes
    width means count would misread it by three orders of magnitude, so the
    axis note is not optional.

    Each band carries the stage's THRESHOLD, not just its name. The funnel
    is the method; a band labelled only "affinity" tells no one what was
    required.
    """
    import numpy as np

    plt = _style()
    order = ["start"] + list(stages("off"))
    values = [counts[arm][s] for s in order]

    # log10(count), floored so a stage returning 1 still has a visible mouth
    # and a stage returning 0 closes it completely.
    widths = [np.log10(v) + 1 if v > 0 else 0.0 for v in values]
    widths = [w / widths[0] for w in widths]

    fig, axis = plt.subplots(figsize=(10.5, 7.4))
    band_height = 1.0
    palette = plt.cm.Blues(np.linspace(0.35, 0.92, len(order)))

    for i in range(len(order)):
        top, bottom = -i * band_height, -(i + 1) * band_height
        w_top = widths[i]
        w_bottom = widths[i + 1] if i + 1 < len(widths) else widths[i] * 0.9
        axis.fill(
            [-w_top / 2, w_top / 2, w_bottom / 2, -w_bottom / 2],
            [top, top, bottom, bottom],
            color=palette[i], edgecolor="white", linewidth=2, zorder=2)

        mid = (top + bottom) / 2
        label = STAGE_LABELS[order[i]].replace("\n", "  ·  ")
        axis.text(-widths[0] / 2 - 0.05, mid, label, ha="right", va="center",
                  fontsize=11.5, color="#1c2833")
        count = values[i]
        axis.text(widths[0] / 2 + 0.05, mid, f"{count:,}", ha="left",
                  va="center", fontsize=14, fontweight="bold",
                  color="#1a5276" if count else "#b03a2e")
        if i:
            previous = values[i - 1]
            share = f"{count / previous * 100:.1f}%" if previous else "—"
            axis.text(widths[0] / 2 + 0.05, mid - 0.24, f"({share} of above)",
                      ha="left", va="center", fontsize=9.5, color="#7f8c8d")

    axis.set_xlim(-1.55, 1.15)
    axis.set_ylim(-len(order) * band_height - 0.75, 0.85)
    axis.axis("off")
    axis.text(0, 0.55, f"{arm}  —  five-stage filter",
              ha="center", fontsize=15, fontweight="bold", color="#1c2833")
    axis.text(0, 0.22,
              "every threshold is relative to PLN-1474, a Phase 1 compound",
              ha="center", fontsize=10.5, color="#566573")
    axis.text(0, -len(order) * band_height - 0.42,
              "band width ∝ log10(molecules), not to the count itself — "
              f"{values[-1]:,} and {values[0]:,} share no linear axis",
              ha="center", fontsize=9, color="#7f8c8d", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "arm": arm,
            "counts": dict(zip(order, values))}


def tradeoff_figure(entries: list, band: float, sel_reference: float,
                    tox_reference: float, out_path: Path,
                    label_these: list | None = None) -> dict:
    plt = _style()

    fig, axis = plt.subplots(figsize=(8.6, 6.4))
    front = {e["label"] for e in pareto_front(entries)}

    # The region a molecule would have to reach to be both selective and
    # clean: right of the noise band AND below the reference toxicity.
    axis.axvspan(sel_reference - band, sel_reference + band,
                 color="#bdc3c7", alpha=0.35, lw=0, zorder=0)
    axis.axvline(sel_reference, color="#7f8c8d", ls="--", lw=1.2, zorder=1)
    axis.axhline(tox_reference, color="#7f8c8d", ls="--", lw=1.2, zorder=1)

    for n_over in (3, 2, 1, 0):
        colour, size, legend = FAIL_STYLE[n_over]
        group = [e for e in entries if len(e["tox_worse"]) == n_over
                 and e["selectivity"] is not None and e["toxicity"] is not None]
        if not group:
            continue
        axis.scatter([e["selectivity"] for e in group],
                     [e["toxicity"] for e in group],
                     s=size, c=colour, alpha=0.9 if n_over <= 1 else 0.6,
                     edgecolors="white" if n_over <= 1 else "none",
                     linewidths=0.8, zorder=4 - n_over + 2, label=legend)

    axis.scatter([sel_reference], [tox_reference], marker="*", s=320,
                 c="#8e44ad", edgecolors="white", linewidths=1.0, zorder=10,
                 label="PLN-1474 (reference)")

    for entry in entries:
        if label_these and entry["label"] not in label_these:
            continue
        if entry["selectivity"] is None or entry["toxicity"] is None:
            continue
        axis.annotate(entry["label"].replace("gen_", "gen "),
                      (entry["selectivity"], entry["toxicity"]),
                      textcoords="offset points", xytext=(9, 6),
                      fontsize=9.5, fontweight="bold",
                      color="#1e8449" if not entry["tox_worse"] else "#ba4a00")

    axis.set_xlabel("selectivity  (αvβ1 percentile − αvβ6 percentile)  →  more selective")
    axis.set_ylabel("toxicity, mean of SR-MMP / NR-AhR / DILI  →  worse")
    # The title states what the data shows, and is derived from it. A
    # hardcoded conclusion in a figure title is the one place a wrong claim
    # never gets checked -- it is not in the caption, so nobody re-reads it
    # when the numbers change.
    clean_selective = [
        e["label"] for e in entries
        if not e["tox_worse"] and e["selectivity"] is not None
        and e["selectivity"] > sel_reference + band]
    axis.set_title(
        "Selectivity and toxicity did not co-occur"
        if not clean_selective else
        f"{len(clean_selective)} molecule(s) clear every endpoint and the "
        "noise band", pad=12)
    axis.annotate(
        f"shaded = noise band (±{band:.3f}); a margin inside it is not a measurement",
        xy=(0.5, -0.145), xycoords="axes fraction", ha="center", fontsize=9,
        color="#555555")
    axis.legend(frameon=False, fontsize=9.5, loc="upper left")
    axis.grid(alpha=0.2)
    axis.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)

    return {"figure": str(out_path), "n": len(entries),
            "pareto": sorted(front),
            "clean_and_selective": clean_selective}


def _selectivity(path: Path, metric: str, isoform, reference) -> dict:
    data = json.loads(Path(path).read_text())
    key = {"delta": "deltas", "rank": "rank_selectivity"}[metric]
    tables = data.get(key) or {}
    name = isoform or (next(iter(tables)) if len(tables) == 1 else None)
    table = tables.get(name, {})
    ref = None
    for candidate in label_variants(reference):
        if candidate in table:
            ref = table[candidate]
            break
    verdicts = data.get({"delta": "calibration_delta",
                         "rank": "calibration_rank"}[metric]) or {}
    return {"values": table, "reference": ref,
            "calibration_passed": bool((verdicts.get(name) or {}).get("passed")),
            "calibration_ran": bool(data.get("calibration_ran", True))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", action="append", required=True,
                    metavar="NAME=FUNNEL:ADMET:SELECTIVITY",
                    help="repeat once per arm")
    p.add_argument("--main-arm", default="TL-C",
                    help="the arm figure 2 is drawn for")
    p.add_argument("--selectivity-metric", choices=("delta", "rank"),
                    default="rank")
    p.add_argument("--reference", default="PANEL_PLN-1474")
    p.add_argument("--selectivity-isoform", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = p.parse_args(argv)

    parsed = {}
    for item in args.arm:
        name, _, paths = item.partition("=")
        funnel_csv, admet_csv, selectivity_json = paths.split(":")
        parsed[DISPLAY.get(name, name)] = (funnel_csv, admet_csv,
                                           selectivity_json)

    # ---- figure 1 ----
    counts = {}
    for arm, (funnel_csv, admet_csv, selectivity_json) in parsed.items():
        selectivity = _selectivity(Path(selectivity_json),
                                   args.selectivity_metric,
                                   args.selectivity_isoform, args.reference)
        rows = lead_rows(funnel_csv, admet_csv, PLN1474_CUTOFF,
                         args.reference, selectivity, "off")
        summary = summarize(rows, "off")
        with open(funnel_csv, newline="") as handle:
            total = sum(1 for _ in csv.DictReader(handle))
        counts[arm] = {"start": total}
        for position, stage in enumerate(stages("off"), start=1):
            counts[arm][stage] = summary[f"stage_{position}_{stage}"]

    report = funnel_figure(counts, args.out_dir / "v2_fig1_funnel.png",
                           args.main_arm)
    print(f"wrote {report['figure']}  ({args.main_arm} only)")
    print("  " + "  ".join(f"{k}={v}" for k, v in report["counts"].items()))

    # ---- figure 2 ----
    funnel_csv, admet_csv, selectivity_json = parsed[args.main_arm]
    selectivity = _selectivity(Path(selectivity_json), args.selectivity_metric,
                               args.selectivity_isoform, args.reference)
    entries = tradeoff_rows(funnel_csv, admet_csv, selectivity, args.reference,
                            args.selectivity_metric)
    band = entries[0]["selectivity_band"] if entries else float("nan")

    with open(admet_csv, newline="") as handle:
        ref = {r["label"]: r for r in csv.DictReader(handle)}[args.reference]
    tox_reference = sum(float(ref[e]) for e in TOXICITY_ENDPOINTS) / 3

    label_these = [e["label"] for e in entries
                   if e["category"] in ("LEAD", "SELECTIVITY")]
    second = tradeoff_figure(entries, band, selectivity["reference"],
                             tox_reference,
                             args.out_dir / "v2_fig2_tradeoff.png", label_these)
    print(f"wrote {second['figure']}")
    print(f"  {second['n']} molecules cleared the structural stages")
    print(f"  labelled: {', '.join(label_these)}")
    print(f"  noise band {band:.3f}; toxicity reference {tox_reference:.4f}")
    print("  molecules clearing EVERY endpoint AND outside the noise band: "
          + (", ".join(second["clean_and_selective"])
             if second["clean_and_selective"] else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
