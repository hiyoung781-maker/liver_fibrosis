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
    "start": "sampled\n(property window)",
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


def funnel_figure(arms: dict, out_path: Path) -> dict:
    """Horizontal bars, one row per stage, both arms.

    Log x, because 5,212 and 1 do not share a linear axis. A count of zero
    has no place on a log axis and is written at the floor instead of drawn,
    so the reader is not left to infer it from a missing bar.
    """
    plt = _style()
    import numpy as np

    order = ["start"] + list(stages("off"))
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    height = 0.36
    positions = np.arange(len(order))[::-1]

    for offset, (arm, counts) in enumerate(arms.items()):
        colour = ARM_COLOURS[arm]
        values = [counts[s] for s in order]
        y = positions + (offset - 0.5) * height
        # A zero gets no bar. Clamping it to the axis floor draws a stub
        # that reads as "a few", and the one stage where an arm returns
        # nothing is the stage the figure exists to show.
        drawn = [v if v > 0 else 0 for v in values]
        axis.barh(y, drawn, height=height, color=colour,
                  alpha=0.85 if arm == "TL-C" else 0.55,
                  label=f"{arm}  ({'transfer learning' if arm == 'TL-A' else 'no transfer learning'})")
        for yi, value in zip(y, values):
            axis.text(max(value, 0.82) * 1.15, yi, f"{value:,}" if value else "0",
                      va="center", ha="left", fontsize=10,
                      fontweight="bold" if value <= 1 else "normal",
                      color=colour if value else "#b03a2e")

    axis.set_yticks(positions)
    axis.set_yticklabels([STAGE_LABELS[s] for s in order], fontsize=9.5)
    axis.set_xscale("log")
    axis.set_xlim(0.8, 12000)
    axis.set_xlabel("molecules (log scale)")
    axis.set_title("Five-stage filter, both arms — every threshold relative to PLN-1474")
    axis.legend(frameon=False, loc="lower right", fontsize=9.5)
    axis.grid(axis="x", alpha=0.25, which="both")
    axis.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "arms": arms}


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

    report = funnel_figure(counts, args.out_dir / "v2_fig1_funnel.png")
    print(f"wrote {report['figure']}")
    for arm, values in counts.items():
        print(f"  {arm:6s} " + "  ".join(f"{k}={v}" for k, v in values.items()))

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
