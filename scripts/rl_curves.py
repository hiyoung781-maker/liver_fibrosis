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

# The section 5.1 scoring components, as REINVENT4 names their columns.
SCORE_COLUMNS = {
    "carboxylate MIDAS anchor": "carboxylate (MIDAS anchor)",
    "TPSA": "TPSA (40-115 window)",
    "SA score": "SA score (guard rail)",
}

COLOURS = {"TL-A-prime": "#c0392b", "TL-C": "#2471a3"}


def per_step(csv_path: str, columns: list) -> "object":
    """Per-step median and interquartile range for the named columns.

    Returns a DataFrame indexed by step with (column, stat) columns. Steps are
    REINVENT4's own `step`; each holds that step's sampled batch.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=["step"] + columns)
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

    columns = list(NLL_COLUMNS) + ["Score"] + list(SCORE_COLUMNS)
    tables, scaffolds = {}, {}
    for arm in ARMS:
        path = results_dir / f"v2_{arm}" / "rl_1.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the RL arm first")
        tables[arm] = per_step(str(path), columns)
        scaffolds[arm] = _scaffolds_per_step(str(path))

    have_sweep = sweep_csv is not None and sweep_csv.exists()
    rows = 3 if have_sweep else 2
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4.0 * rows))

    # Row 1: score, then the DAP loss pair per arm.
    axis = axes[0][0]
    for arm in ARMS:
        _band(axis, tables[arm], "Score", COLOURS[arm], DISPLAY[arm])
    axis.set_title("total score (geometric mean of 3 components)")
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

    # Row 2: the three scoring components, both arms overlaid, then diversity.
    for position, (column, title) in enumerate(SCORE_COLUMNS.items()):
        axis = axes[1][position]
        for arm in ARMS:
            _band(axis, tables[arm], column, COLOURS[arm], DISPLAY[arm])
        axis.set_title(title)
        axis.set_xlabel("step"); axis.set_ylabel("component score")
        axis.set_ylim(-0.02, 1.02)
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
    if not report["sweep"]:
        print("  (TL sweep table absent; its panels were skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
