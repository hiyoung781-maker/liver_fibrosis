"""Training curves for the two RL arms, and the TL sweep that preceded them.

Built from the run's own artefacts rather than from a logging service, so the
figures regenerate from the repository: REINVENT4's per-molecule summary CSV
(`results/v2_<arm>/rl_1.csv`, one row per sampled molecule per step) and the
TL sweep table (`results/v2_tl_sweep.csv`).

WHAT "LOSS" MEANS HERE. REINVENT4's DAP objective does not expose a single
loss; it reports three negative log-likelihoods per molecule, and their
relationship is the training signal:

  Prior NLL      the frozen prior's likelihood. Drift away from it is the
                 agent leaving the prior's chemistry.
  Agent NLL      the learning agent's own likelihood.
  Augmented NLL  prior NLL minus sigma * score, the target the agent is
                 regressed onto. Agent - Augmented IS the DAP loss term.

So the panel to read for convergence is Agent against Augmented: they meet as
the agent learns to reproduce the score-shifted prior.

EVERY PANEL IS A PER-STEP AGGREGATE over that step's 128 sampled molecules,
with the interquartile band drawn, because a mean alone hides whether the
batch agreed. A collapsing band with a rising mean is mode collapse and looks
identical to success in a mean-only plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["ARMS", "NLL_COLUMNS", "SCORE_COLUMNS", "per_step"]

ARMS = ("TL-A-prime", "TL-C")

# What the figure calls each arm. The directories and labels keep
# "TL-A-prime" -- v1 had its own TL-A, trained on a set that MIXED RGD and
# non-RGD, and the prime was what distinguished this arm from it. The figure
# says "TL-A" and carries a footnote so the two are not read as one.
DISPLAY = {"TL-A-prime": "TL-A", "TL-C": "TL-C"}

FOOTNOTE = ("TL-A = v2's non-RGD-only transfer-learning arm (directory "
            "TL-A-prime); TL-C = random prior, no transfer learning. "
            "v1's TL-A was a different arm, trained on a set mixing RGD and "
            "non-RGD.")

# Band edges are smoothed over this many steps. The median line is raw: the
# smoothing is for the shaded quartiles, whose per-step excursions to 0 make
# the panel unreadable without hiding anything the median does not already
# show.
BAND_WINDOW = 15

NLL_COLUMNS = {"Agent": "Agent NLL", "Prior": "Prior NLL",
               "Target": "Augmented NLL"}

# The section 5.1 scoring components, plotted as their RAW values rather than
# as REINVENT4's transformed scores. The transformed score of every component
# sits at 1.0 for nearly the whole run, so it shows only that the objective is
# satisfied, never by how much. Each panel draws its transform's own bounds,
# which is what makes the raw axis readable: the distance between the
# distribution and the bound IS the headroom the guard rail was set to catch.
#
# (column, title, y limits, [(y, label), ...] transform bounds)
RAW_PANELS = (
    ("carboxylate MIDAS anchor (raw)", "carboxylate count (MIDAS anchor)",
     (0, 5), [(1, "right_step high=1")]),
    ("TPSA (raw)", "TPSA, A^2  (double_sigmoid 40-115)",
     (0, 200), [(40, "low 40"), (115, "high 115")]),
    # SA score runs 1-10 and LOWER is easier to make, so the whole axis is
    # drawn: a panel clipped to the observed range would hide that the guard
    # rail never binds. reverse_sigmoid low=6 high=8 means full marks at or
    # below 6, zero at or above 8.
    ("SA score (raw)", "SA score  (1 easy - 10 hard; guard rail 6-8)",
     (1, 10), [(6, "full marks <= 6"), (8, "zero >= 8")]),
)

SCORE_COLUMNS = {c: title for c, title, _, _ in RAW_PANELS}

COLOURS = {"TL-A-prime": "#c0392b", "TL-C": "#2471a3"}


# REINVENT4 writes SMILES_state 1 for a valid molecule, 0 for one it could
# not parse and 2 for a duplicate. An invalid molecule has no descriptors, so
# its raw columns are written as 0 -- which is a real value on a count axis
# and an impossible one on an SA axis that starts at 1. Left in, those zeros
# drag every quartile band to the floor; they are a validity statistic, not a
# component value, so the panels exclude them and the count is reported.
VALID_STATE = 1


def per_step(csv_path: str, columns: list, drop_invalid: bool = False):
    """Per-step median and interquartile range for the named columns.

    Returns a DataFrame indexed by step with (column, stat) columns. Steps are
    REINVENT4's own `step`; each holds that step's sampled batch. With
    `drop_invalid`, molecules REINVENT4 could not parse are excluded.
    """
    import pandas as pd

    needed = ["step"] + columns + (["SMILES_state"] if drop_invalid else [])
    frame = pd.read_csv(csv_path, usecols=needed)
    if drop_invalid:
        frame = frame[frame["SMILES_state"] == VALID_STATE]
        # A raw 0 survives the state filter in a few hundred rows per arm
        # where the descriptor itself failed; SA score cannot be 0.
        for column in columns:
            if column.endswith("(raw)") and column.startswith("SA"):
                frame = frame[frame[column] > 0]
        frame = frame.drop(columns=["SMILES_state"])
    grouped = frame.groupby("step")[columns]
    out = grouped.median().add_suffix("|median")
    out = out.join(grouped.quantile(0.25).add_suffix("|q25"))
    out = out.join(grouped.quantile(0.75).add_suffix("|q75"))
    out = out.join(frame.groupby("step").size().rename("n"))
    return out.sort_index()


def _band(axis, frame, column, colour, label):
    axis.plot(frame.index, frame[f"{column}|median"], color=colour, lw=1.4,
              label=label)
    low = frame[f"{column}|q25"].rolling(BAND_WINDOW, center=True,
                                         min_periods=1).median()
    high = frame[f"{column}|q75"].rolling(BAND_WINDOW, center=True,
                                          min_periods=1).median()
    axis.fill_between(frame.index, low, high, color=colour, alpha=0.20, lw=0)


def _scaffolds_per_step(csv_path: str):
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=["step", "Scaffold"])
    return frame.groupby("step")["Scaffold"].nunique()


def build(results_dir: Path, out_path: Path, sweep_csv: Path | None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(NLL_COLUMNS) + ["Score"]
    raw_columns = [c for c, _, _, _ in RAW_PANELS]
    tables, raws, scaffolds, invalid = {}, {}, {}, {}
    for arm in ARMS:
        path = results_dir / f"v2_{arm}" / "rl_1.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the RL arm first")
        tables[arm] = per_step(str(path), columns)
        raws[arm] = per_step(str(path), raw_columns, drop_invalid=True)
        scaffolds[arm] = _scaffolds_per_step(str(path))
        invalid[arm] = int(tables[arm]["n"].sum() - raws[arm]["n"].sum())

    have_sweep = sweep_csv is not None and sweep_csv.exists()
    rows = 3 if have_sweep else 2
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.0 * rows))

    # Row 1: score, then the DAP loss pair per arm.
    axis = axes[0][0]
    for arm in ARMS:
        _band(axis, tables[arm], "Score", COLOURS[arm], DISPLAY[arm])
    axis.set_title("total score (4 components; CustomAlerts is a 0/1 factor)")
    axis.set_xlabel("step"); axis.set_ylabel("score"); axis.legend(frameon=False)

    for position, arm in enumerate(ARMS, start=1):
        axis = axes[0][position]
        frame = tables[arm]
        for column, style in (("Agent", "-"), ("Target", "--"), ("Prior", ":")):
            axis.plot(frame.index, frame[f"{column}|median"], style,
                      color=COLOURS[arm], lw=1.3, alpha=0.9,
                      label=NLL_COLUMNS[column])
        axis.set_title(f"{DISPLAY[arm]}: NLL  (Agent vs Augmented = the DAP loss)")
        axis.set_xlabel("step"); axis.set_ylabel("NLL")
        axis.legend(frameon=False, fontsize=8)

    # Row 2: the three scoring components as RAW values, both arms overlaid,
    # each with its transform's bounds drawn.
    for position, (column, title, ylim, bounds) in enumerate(RAW_PANELS):
        axis = axes[1][position]
        for arm in ARMS:
            _band(axis, raws[arm], column, COLOURS[arm], DISPLAY[arm])
        for y, text in bounds:
            axis.axhline(y, color="#7f8c8d", ls="--", lw=1.0)
            axis.annotate(text, xy=(0.99, y), xycoords=("axes fraction", "data"),
                          ha="right", va="bottom", fontsize=7.5,
                          color="#7f8c8d")
        axis.set_title(title)
        axis.set_xlabel("step"); axis.set_ylabel("raw value")
        axis.set_ylim(*ylim)
        if position == 0:
            axis.legend(frameon=False)

    if have_sweep:
        import pandas as pd
        sweep = pd.read_csv(sweep_csv)
        axis = axes[2][0]
        for arm in ARMS:
            smoothed = scaffolds[arm].rolling(BAND_WINDOW, center=True,
                                              min_periods=1).median()
            axis.plot(scaffolds[arm].index, scaffolds[arm].values,
                      color=COLOURS[arm], lw=0.5, alpha=0.25)
            axis.plot(smoothed.index, smoothed.values, color=COLOURS[arm],
                      lw=1.6, label=DISPLAY[arm])
        axis.set_title("unique Murcko scaffolds per step (of 128 sampled)")
        axis.set_xlabel("step"); axis.set_ylabel("unique scaffolds")
        axis.legend(frameon=False)

        axis = axes[2][1]
        axis.plot(sweep["epoch"], sweep["acid_pct"], "o-", color="#616a6b",
                  lw=1.4, ms=3)
        axis.set_title("TL sweep: carboxylic acid fraction")
        axis.set_xlabel("TL epoch"); axis.set_ylabel("% with COOH")

        axis = axes[2][2]
        axis.plot(sweep["epoch"], sweep["max_nn_tanimoto"], "o-",
                  color="#8e44ad", lw=1.4, ms=3, label="max NN-Tanimoto")
        axis.axhline(1.0, color="#c0392b", ls="--", lw=1.0)
        secondary = axis.twinx()
        secondary.plot(sweep["epoch"], sweep["unique_scaffolds"], "s-",
                       color="#16a085", lw=1.2, ms=3, label="unique scaffolds")
        secondary.set_ylabel("unique scaffolds / 1000")
        axis.set_title("TL sweep: memorisation (max NN-Tanimoto reaches 1.0)")
        axis.set_xlabel("TL epoch"); axis.set_ylabel("max NN-Tanimoto")
        axis.legend(frameon=False, loc="lower right", fontsize=8)
    else:
        axis = axes[1][2]

    fig.suptitle("v2 RL training curves", y=0.995, fontsize=13)
    fig.text(0.5, 0.004, FOOTNOTE, ha="center", va="bottom", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.022, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    return {
        "figure": str(out_path),
        "arms": {DISPLAY[arm]: {"steps": int(tables[arm].index.max()),
                       "batch": int(tables[arm]["n"].median()),
                       "score_first": float(tables[arm]["Score|median"].iloc[0]),
                       "score_last": float(tables[arm]["Score|median"].iloc[-1]),
                       "scaffolds_first": int(scaffolds[arm].iloc[0]),
                       "scaffolds_last": int(scaffolds[arm].iloc[-1])}
                 for arm in ARMS},
        "invalid": invalid,
        "sa_median_last": {DISPLAY[a]: float(raws[a]["SA score (raw)|median"].iloc[-1])
                           for a in ARMS},
        "sa_max": {DISPLAY[a]: float(raws[a]["SA score (raw)|q75"].max())
                   for a in ARMS},
        "sweep": bool(have_sweep),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--sweep-csv", type=Path,
                    default=Path("results/v2_tl_sweep.csv"))
    p.add_argument("--out", type=Path, default=Path("figures/v2_rl_curves.png"))
    args = p.parse_args(argv)

    try:
        report = build(args.results_dir, args.out, args.sweep_csv)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    print(f"wrote {report['figure']}")
    for arm, stats in report["arms"].items():
        print(f"  {arm:12s} steps {stats['steps']}  batch {stats['batch']}  "
              f"score {stats['score_first']:.3f} -> {stats['score_last']:.3f}  "
              f"scaffolds/step {stats['scaffolds_first']} -> "
              f"{stats['scaffolds_last']}")
    print(f"  SA score (raw, guard rail bites above 6): median at the last "
          f"step {report['sa_median_last']}")
    print(f"  molecules excluded as invalid: {report['invalid']}")
    if not report["sweep"]:
        print("  (TL sweep table absent; its panels were skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
