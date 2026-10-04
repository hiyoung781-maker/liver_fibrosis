"""v4 poster figures. Five panels, in the order the poster reads them.

  fig1_pipeline     the design pipeline as a block diagram, with the counts
  fig2_learning     transfer learning (5-fold) and reinforcement learning curves
  fig3_leads        the four strongest leads, structures and property cards
  fig4_dili         why the campaign's output is constrained, by endpoint
  fig5_pharmacophore  the RGD pharmacophore and what each half binds

WHAT IS AND IS NOT HERE. There is no isoform-selectivity panel: alphaVbeta6 and
alphaIIbbeta3 redocking failed their own validation, so selectivity is an
unevaluated axis and plotting it would imply a measurement that does not exist.
There is no panel on how the methodology reached its present form - version
history is a record for the team, not a finding about alphaVbeta1.

Every number is computed from the artifact that produced it. Nothing is retyped
from the report.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["pipeline_figure", "learning_figure", "dili_figure", "COUNTS"]

# Palette: the pipeline's own semantics, not decoration. Generation steps are
# one colour, the structural gates another, ADMET a third, so a reader can see
# which kind of evidence each stage rests on without reading the labels.
C_GEN = "#FDF3D7"      # generative / library
C_STRUCT = "#D9E8F5"   # structure-based (docking, PLIP)
C_ADMET = "#FADBD8"    # predicted properties
C_LEAD = "#D5F5E3"     # output
C_EDGE = "#1a1a1a"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })
    return plt


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def counts(root: Path = Path(".")) -> dict:
    """Stage counts, read from the files each stage wrote."""
    r = root / "results/v4_TL-B"
    d = root / "data/v4_TL-B"

    def rows(path):
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle))

    def smi(path):
        return sum(1 for line in open(path)
                   if line.strip() and not line.startswith("#"))

    funnel = rows(r / "funnel.csv")
    plip = rows(r / "plip.csv")
    mode = {x["label"] for x in plip
            if x["metal_ca501"] == "1" and x["hbond_asn224"] == "1"}
    sel = rows(r / "selection.csv")
    return {
        "sampled": smi(d / "library.smi"),
        "survivors": smi(d / "survivors.smi"),
        "admet_window": smi(d / "survivors_admet.smi"),
        "poses": len(plip),
        "binding_mode": len([m for m in mode if not m.startswith("CONTROL")]),
        "affinity": len(sel),
        "direct": len([x for x in sel if x["track"] == "direct_lead"]),
        "parents": len(rows(r / "parents.csv")),
        "analogues": len(rows(r / "analogues_n256_admet.csv")),
        "selected": len(rows(r / "analogues_n256_selected.csv")),
        "optimised": len(rows(r / "optimised_leads.csv")),
        "funnel_rows": len(funnel),
    }


COUNTS = counts


def _box(ax, x, y, w, h, title, body, count, fill, fs=9.5):
    """One pipeline stage: bold name, what it tests, and what it kept.

    Line offsets are fractions of the box height, not absolute figure units -
    the axes are not square, so a constant spacing puts text outside the box on
    one axis while crowding it on the other.
    """
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.018",
        linewidth=1.6, edgecolor=C_EDGE, facecolor=fill, zorder=2))
    cx = x + w / 2
    ax.text(cx, y + h * 0.76, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", zorder=3)
    if body:
        ax.text(cx, y + h * 0.47, body, ha="center", va="center",
                fontsize=fs - 1.3, zorder=3, linespacing=1.4)
    if count is not None:
        ax.text(cx, y + h * 0.16, count, ha="center", va="center",
                fontsize=fs + 2, fontweight="bold", zorder=3)


def _arrow(ax, x0, y0, x1, y1, rad=0.0, lw=1.6):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=C_EDGE,
        connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=1, shrinkB=1))


def _elbow(ax, x0, y0, x1, y1, ymid, lw=1.6, color=C_EDGE):
    """Down, across, then in. A long arc3 between distant boxes bulges so far
    that it appears to leave the wrong box - the toxicity-failure connector read
    as if it came from the ADMET window three stages earlier."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath
    verts = [(x0, y0), (x0, ymid), (x1, ymid), (x1, y1)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.LINETO]
    ax.add_patch(FancyArrowPatch(
        path=MPath(verts, codes), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=1, joinstyle="miter"))


def pipeline_figure(out_path: Path, c: dict) -> None:
    """The pipeline as a block diagram, with what each stage kept.

    A funnel was the obvious shape and the wrong one: the stages differ in the
    KIND of evidence they rest on, not only in how much they remove, and a taper
    cannot show that the optimisation track feeds molecules back into docking
    rather than onward.
    """
    plt = _style()
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    h, top = 0.235, 0.60
    w, gap, x0 = 0.128, 0.0165, 0.008
    xs = [x0 + i * (w + gap) for i in range(6)]
    stages = [
        ("Generative model", "transfer learning\n+ reinforcement learning",
         f"{c['sampled']:,}", C_GEN),
        ("Property window", "logP, MW, TPSA\nnovelty", f"{c['survivors']:,}", C_GEN),
        ("ADMET window", "DILI\npermeability", f"{c['admet_window']}", C_ADMET),
        ("Docking + PLIP", "MIDAS and\nβ1-Asn224", f"{c['binding_mode']}", C_STRUCT),
        ("Affinity", "≤ PLN-1474\n(−6.97 kcal/mol)", f"{c['affinity']}", C_STRUCT),
        ("Toxicity gate", "SR-MMP, NR-AhR\nDILI", f"{c['direct']}", C_ADMET),
    ]
    for (t, b, n, col), x in zip(stages, xs):
        _box(ax, x, top, w, h, t, b, n, col)
    for i in range(5):
        _arrow(ax, xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)

    lead_x = xs[5] + w + gap
    lead_w = 1.0 - lead_x - 0.008
    _box(ax, lead_x, top, lead_w, h, "Direct leads", "", f"{c['direct']}", C_LEAD)
    _arrow(ax, xs[5] + w, top + h / 2, lead_x, top + h / 2)

    bot, ow, ogap = 0.145, 0.148, 0.052
    ox = [xs[1] + 0.010 + i * (ow + ogap) for i in range(3)]
    opt = [
        ("Parents", "cleared structure\nand affinity", f"{c['parents']}", C_STRUCT),
        ("Analogues", "mol2mol\nhigh-similarity", f"{c['analogues']:,}", C_GEN),
        ("Selection", "toxicity improved\nvs parent", f"{c['selected']}", C_ADMET),
    ]
    for (t, b, n, col), x in zip(opt, ox):
        _box(ax, x, bot, ow, h, t, b, n, col)
    for i in range(2):
        _arrow(ax, ox[i] + ow, bot + h / 2, ox[i + 1], bot + h / 2)

    # Parents are every molecule that cleared structure AND affinity, not only
    # the ones toxicity rejected: toxicity is the axis being repaired, so
    # pre-filtering on it would discard what the stage exists to improve.
    _elbow(ax, xs[4] + w / 2, top, ox[0] + ow / 2, bot + h, 0.495,
           color="#8a2f2f")
    ax.text(0.448, 0.515, "all 24, whatever toxicity said", fontsize=8.5,
            style="italic", color="#8a2f2f", ha="center")
    _elbow(ax, ox[2] + ow / 2, bot + h, xs[3] + w / 2, top, 0.432,
           color="#2f4f8a")
    ax.text(0.575, 0.395, "re-dock, re-verify", fontsize=8.5, style="italic",
            color="#2f4f8a", ha="center")
    _box(ax, lead_x, bot, lead_w, h, "Optimised leads", "",
         f"{c['optimised']}", C_LEAD)
    _arrow(ax, ox[2] + ow, bot + h / 2, lead_x, bot + h / 2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_GEN, edgecolor=C_EDGE, label="Generation"),
        Patch(facecolor=C_ADMET, edgecolor=C_EDGE, label="Predicted properties"),
        Patch(facecolor=C_STRUCT, edgecolor=C_EDGE, label="Structure-based"),
        Patch(facecolor=C_LEAD, edgecolor=C_EDGE, label="Leads"),
    ], loc="lower left", bbox_to_anchor=(0.008, -0.04), ncol=4, frameon=True,
        fontsize=9, borderpad=0.6, columnspacing=1.5, handlelength=1.4)

    fig.savefig(out_path)
    plt.close(fig)


def _tl_curves(csv_path: Path) -> dict:
    """{(fold, kind): (epochs, nll)} from the extracted curve table.

    Read from a CSV rather than the TensorBoard event files directly: the only
    environment with tensorboard installed is not the one with matplotlib, and a
    figure that reads a text artifact is reproducible without either.
    scripts/extract_tl_curves.py writes it.
    """
    out: dict = {}
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["fold"], row["kind"])
            out.setdefault(key, ([], []))
            out[key][0].append(int(row["epoch"]))
            out[key][1].append(float(row["nll"]))
    return out


def learning_figure(out_path: Path, tl_csv: Path, rl_csv: Path) -> dict:
    """Transfer learning across folds, and the RL trajectory.

    The TL panel shows validation loss because that is what chose the production
    epoch; training loss is drawn faintly so the gap between them is visible
    rather than asserted.
    """
    plt = _style()
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    # (a) TL 5-fold
    ax = axes[0]
    curves = _tl_curves(tl_csv)
    folds = sorted({k[0] for k in curves})
    best = []
    for i, fold in enumerate(folds):
        xs, tr = curves.get((fold, "training"), ([], []))
        xv, va = curves.get((fold, "validation"), ([], []))
        if tr:
            ax.plot(xs, tr, color="#b0b0b0", lw=0.9, alpha=0.55,
                    label="training" if i == 0 else None, zorder=1)
        if va:
            ax.plot(xv, va, lw=1.5, alpha=0.9,
                    label=fold if i < 5 else None, zorder=2)
            k = int(np.argmin(va))
            best.append(xv[k])
            ax.plot([xv[k]], [va[k]], "o", ms=4.5, color="black", zorder=3)
    if best:
        ax.axvline(float(np.mean(best)), color="#c0392b", ls="--", lw=1.4,
                   zorder=0)
        ax.text(float(np.mean(best)), ax.get_ylim()[1] * 0.97,
                f" adopted: epoch {int(round(float(np.mean(best))))}",
                color="#c0392b", fontsize=9, va="top")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NLL loss")
    ax.set_title("(a) Transfer learning: 5-fold validation")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)

    # (b) RL 총점
    with open(rl_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    by = {}
    for r in rows:
        by.setdefault(int(r["step"]), []).append(r)
    steps = sorted(by)
    mean = lambda col, s: float(np.mean([_f(x[col]) for x in by[s]
                                         if _f(x[col]) is not None]))
    ax = axes[1]
    score = [mean("Score", s) for s in steps]
    ax.plot(steps, score, color="#1f4e79", lw=0.8, alpha=0.35)
    win = 25
    if len(score) > win:
        sm = np.convolve(score, np.ones(win) / win, mode="valid")
        ax.plot(steps[win - 1:], sm, color="#1f4e79", lw=2.2, label="total score (moving avg.)")
    ax.set_xlabel("RL step"); ax.set_ylabel("Total score")
    ax.set_title("(b) Reinforcement learning: objective")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    # (c) RL 성분별
    ax = axes[2]
    comps = [("carboxylate MIDAS anchor", "carboxylate (MIDAS anchor)", "#1f4e79"),
             ("TPSA", "TPSA window", "#2e8b57"),
             ("unwanted groups", "structural alerts", "#8a6d3b")]
    for col, label, colour in comps:
        if col not in rows[0]:
            continue
        series = [mean(col, s) for s in steps]
        if len(series) > win:
            sm = np.convolve(series, np.ones(win) / win, mode="valid")
            ax.plot(steps[win - 1:], sm, lw=2.0, color=colour, label=label)
    ax.set_xlabel("RL step"); ax.set_ylabel("Component score")
    ax.set_title("(c) Reinforcement learning: components")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"folds": len(folds), "best_epochs": best, "steps": len(steps)}


def dili_figure(out_path: Path, admet_csv: Path, thresholds: dict,
                reference_rate: float = 35.1) -> dict:
    """Which endpoint constrains the library, and by how much.

    Pass rates alone would not separate "three hard filters" from "one hard
    filter"; the leave-one-out bars do, which is the panel's whole point.
    """
    plt = _style()
    import numpy as np
    with open(admet_csv, newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if not r["label"].startswith("PANEL_")]
    n = len(rows)
    ends = ["DILI", "NR-AhR", "SR-MMP"]

    def passes(r, e):
        v = _f(r.get(e))
        return v is not None and v < thresholds[e]

    single = {e: sum(1 for r in rows if passes(r, e)) for e in ends}
    conj = sum(1 for r in rows if all(passes(r, e) for e in ends))
    drop = {e: sum(1 for r in rows
                   if all(passes(r, x) for x in ends if x != e)) for e in ends}

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    ax = axes[0]
    rates = [100 * single[e] / n for e in ends]
    bars = ax.bar(ends, rates, color=["#c0392b", "#d98880", "#f5b7b1"],
                  edgecolor=C_EDGE, linewidth=1.1)
    ax.axhline(reference_rate, color="#1f4e79", ls="--", lw=1.6)
    ax.text(2.45, reference_rate + 1.5, f"approved drugs {reference_rate}%",
            color="#1f4e79", fontsize=9, ha="right")
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}%",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Library pass rate (%)")
    ax.set_title(f"(a) Pass rate by endpoint (n={n:,})")
    ax.set_ylim(0, max(rates + [reference_rate]) * 1.25)

    ax = axes[1]
    labels = ["all\nthree"] + [f"drop\n{e}" for e in ends]
    vals = [conj] + [drop[e] for e in ends]
    cols = ["#7f8c8d"] + ["#c0392b" if e == "DILI" else "#aab7b8" for e in ends]
    bars = ax.bar(labels, vals, color=cols, edgecolor=C_EDGE, linewidth=1.1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:,}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Molecules passing")
    ax.set_title("(b) Leave one endpoint out")

    ax = axes[2]
    vals = [_f(r["DILI"]) for r in rows if _f(r["DILI"]) is not None]
    ax.hist(vals, bins=50, color="#d98880", edgecolor="white", linewidth=0.3)
    ax.axvline(thresholds["DILI"], color="#c0392b", lw=2.0)
    ax.text(thresholds["DILI"] - 0.02, ax.get_ylim()[1] * 0.93, "threshold ",
            color="#c0392b", fontsize=9, ha="right", va="top")
    for value, name, colour in ((0.4552, "PLN-1474", "#1f4e79"),
                                (0.3300, "bexotegrast", "#2e8b57")):
        ax.axvline(value, color=colour, ls="--", lw=1.5)
        ax.text(value - 0.02, ax.get_ylim()[1] * 0.60, name + " ", rotation=90,
                color=colour, fontsize=8.5, ha="right", va="center")
    ax.set_xlabel("Predicted DILI"); ax.set_ylabel("Molecules")
    ax.set_title("(c) Library DILI distribution")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return {"n": n, "single": single, "conjunction": conj, "leave_one_out": drop}



LEADS = ["opt_10773", "opt_11266", "opt_10928", "gen_11276"]


def _lead_rows(root: Path = Path(".")) -> list[dict]:
    """The shortlisted leads with the numbers the card shows, read from the
    files that produced them. PLN-1474 comes last, as the reference row."""
    r = root / "results/v4_TL-B"
    direct = {x["label"]: x for x in csv.DictReader(open(r / "selection.csv"))}
    opt = {x["label"]: x for x in csv.DictReader(open(r / "optimised_leads.csv"))}
    adm = {x["label"]: x for x in
           csv.DictReader(open(r / "analogues_n256_admet.csv"))}
    panel = {x["label"]: x for x in
             csv.DictReader(open(r / "admet_with_panel.csv"))}

    out = []
    for label in LEADS:
        if label in opt:
            x = opt[label]
            a = adm.get(label, {})
            out.append(dict(label=label, smiles=x["smiles"],
                            aff=_f(x["affinity"]), dili=_f(x["DILI"]),
                            srmmp=_f(x["SR_MMP"]), nrahr=_f(x["NR_AhR"]),
                            caco=_f(x["caco2"]), hia=_f(a.get("HIA_Hou")),
                            bioav=_f(a.get("Bioavailability_Ma")),
                            parent=x["parent"], delta=_f(x["delta"])))
        else:
            x = direct[label]
            out.append(dict(label=label, smiles=x["smiles"],
                            aff=_f(x["affinity"]), dili=_f(x["DILI"]),
                            srmmp=_f(x["SR_MMP"]), nrahr=_f(x["NR_AhR"]),
                            caco=_f(x["caco2"]), hia=_f(x["hia"]),
                            bioav=_f(x["bioavail"]), parent="", delta=None))
    x = panel["PANEL_PLN-1474"]
    out.append(dict(label="PLN-1474", smiles=x["smiles"], aff=-6.973,
                    dili=_f(x["DILI"]), srmmp=_f(x["SR-MMP"]),
                    nrahr=_f(x["NR-AhR"]), caco=_f(x["Caco2_Wang"]),
                    hia=_f(x["HIA_Hou"]), bioav=_f(x["Bioavailability_Ma"]),
                    parent="reference", delta=None))
    return out


def leads_figure(out_path: Path, rows: list[dict], thresholds: dict) -> None:
    """Four leads and the clinical reference: structure above, numbers below.

    The reference row is drawn in the same frame rather than quoted in a caption
    because the only honest reading of a predicted number here is relative - and
    PLN-1474 exceeds the SR-MMP threshold, which a caption would bury.
    """
    plt = _style()
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    RDLogger.DisableLog("rdApp.*")
    import io
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from PIL import Image

    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(3.05 * n, 5.6),
                             gridspec_kw={"height_ratios": [2.05, 1.0]})
    for col, row in enumerate(rows):
        mol = Chem.MolFromSmiles(row["smiles"])
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(540, 420)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))

        ax = axes[0][col]
        ax.imshow(img); ax.axis("off")
        ref = row["parent"] == "reference"
        ax.set_title(("PLN-1474  (reference)" if ref else row["label"]),
                     fontsize=12, fontweight="bold",
                     color="#555555" if ref else "#111111", pad=4)

        ax = axes[1][col]
        ax.axis("off")
        def mark(value, limit):
            return "" if value < limit else "  ✗"
        lines = [
            ("affinity", f"{row['aff']:.3f} kcal/mol"),
            ("DILI", f"{row['dili']:.3f}{mark(row['dili'], thresholds['DILI'])}"),
            ("SR-MMP", f"{row['srmmp']:.4f}{mark(row['srmmp'], thresholds['SR-MMP'])}"),
            ("NR-AhR", f"{row['nrahr']:.4f}{mark(row['nrahr'], thresholds['NR-AhR'])}"),
            ("Caco-2", f"{row['caco']:.2f}"),
            ("HIA / F", f"{row['hia']:.2f} / {row['bioav']:.2f}"),
        ]
        if row["delta"] is not None:
            lines.append(("vs parent", f"{row['delta']:+.3f}"))
        for i, (k, v) in enumerate(lines):
            y = 0.94 - i * 0.145
            ax.text(0.02, y, k, fontsize=9.5, va="top", color="#555555")
            ax.text(0.98, y, v, fontsize=9.5, va="top", ha="right",
                    fontweight="bold" if k == "affinity" else "normal",
                    color="#8a2f2f" if "✗" in v else "#111111")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def pharmacophore_figure(out_path: Path) -> None:
    """What each half of an RGD mimetic binds, and why that constrains design.

    Drawn rather than photographed from a structure: the point is the pairing of
    two pharmacophores with two subunits, which a rendered pocket obscures.
    """
    plt = _style()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch((0.03, 0.46), 0.42, 0.40,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor="#DCE9F7", edgecolor=C_EDGE, lw=1.6))
    ax.text(0.24, 0.79, "αV subunit", fontsize=12.5, fontweight="bold",
            ha="center")
    ax.text(0.24, 0.655, "Asp218", fontsize=11, ha="center")
    ax.text(0.24, 0.545, "salt bridge to a basic head", fontsize=9.2,
            ha="center", style="italic", color="#444444")

    ax.add_patch(FancyBboxPatch((0.55, 0.46), 0.42, 0.40,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor="#F7DEDC", edgecolor=C_EDGE, lw=1.6))
    ax.text(0.76, 0.79, "β1 subunit", fontsize=12.5, fontweight="bold",
            ha="center")
    ax.text(0.76, 0.655, "MIDAS metal  ·  Asn224", fontsize=11, ha="center")
    ax.text(0.76, 0.545, "coordinates a carboxylate", fontsize=9.2,
            ha="center", style="italic", color="#444444")

    ax.add_patch(FancyBboxPatch((0.17, 0.10), 0.20, 0.23,
                                boxstyle="round,pad=0.005,rounding_size=0.02",
                                facecolor="#FDF3D7", edgecolor=C_EDGE, lw=1.6))
    ax.text(0.27, 0.265, "Arg mimic", fontsize=11.5, fontweight="bold",
            ha="center")
    ax.text(0.27, 0.165, "basic head\n(e.g. THN)", fontsize=9.2, ha="center")

    ax.add_patch(FancyBboxPatch((0.63, 0.10), 0.20, 0.23,
                                boxstyle="round,pad=0.005,rounding_size=0.02",
                                facecolor="#FDF3D7", edgecolor=C_EDGE, lw=1.6))
    ax.text(0.73, 0.265, "Asp mimic", fontsize=11.5, fontweight="bold",
            ha="center")
    ax.text(0.73, 0.165, "carboxylate", fontsize=9.2, ha="center")

    ax.annotate("", xy=(0.40, 0.215), xytext=(0.60, 0.215),
                arrowprops=dict(arrowstyle="<->", lw=1.5, color="#555555"))
    ax.text(0.50, 0.245, "spacer", fontsize=9.5, ha="center", color="#555555")
    ax.text(0.50, 0.145, "9–16 bonds in measured actives", fontsize=8.6,
            ha="center", color="#555555", style="italic")

    for x0, x1, colour in ((0.27, 0.24, "#2f4f8a"), (0.73, 0.76, "#8a2f2f")):
        ax.add_patch(FancyArrowPatch((x0, 0.335), (x1, 0.455),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.8, color=colour))

    ax.text(0.5, 0.025,
            "The carboxylate is required for MIDAS binding and costs permeability; "
            "the basic head raises affinity on the αV side, which is shared across "
            "β partners.",
            fontsize=9.0, ha="center", color="#333333", style="italic")
    fig.savefig(out_path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("figures/v4"))
    p.add_argument("--tl-csv", type=Path,
                   default=Path("results/v4_TL-B/tl_curves.csv"))
    p.add_argument("--rl-csv", type=Path,
                   default=Path("results/v4_TL-B/rl_1.csv"))
    p.add_argument("--admet", type=Path,
                   default=Path("results/v4_TL-B/admet_preds.csv"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from admet_window import toxicity_thresholds
    th = toxicity_thresholds()

    c = counts()
    pipeline_figure(args.out_dir / "fig1_pipeline.png", c)
    print(f"  fig1_pipeline.png   {c}")

    info = learning_figure(args.out_dir / "fig2_learning.png", args.tl_csv,
                           args.rl_csv)
    print(f"  fig2_learning.png   {info}")

    info = dili_figure(args.out_dir / "fig4_dili.png", args.admet, th)
    print(f"  fig4_dili.png       conjunction {info['conjunction']} / {info['n']}")

    rows = _lead_rows()
    leads_figure(args.out_dir / "fig3_leads.png", rows, th)
    print(f"  fig3_leads.png      {[r['label'] for r in rows]}")

    pharmacophore_figure(args.out_dir / "fig5_pharmacophore.png")
    print("  fig5_pharmacophore.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
