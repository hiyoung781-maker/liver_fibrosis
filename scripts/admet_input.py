"""Build the ADMET input set from the section 8.5b survivors, and summarise the output.

Section 8's item 6 applies ADMET prediction to SURVIVORS. The provisional run in
logs/admet_summary.txt used the section 8.2 set (7,767 molecules), because the docking
had not happened yet. Now that section 8.5b has run, the set that item 7 actually
selects from is the 1,587 ligands with at least one pose satisfying both geometry
criteria - a narrower and more meaningful population.

ADMET is a SCREEN, not a gate (section 7.5 removed it from section 9's criteria). The
reasons, measured: the best Caco-2 models reach MAE 0.26-0.28 log units while the same
compound's published Caco-2 values differ between laboratories by a median of 0.57 log
units, so model error and assay disagreement are the same size. And a Caco-2
prediction correlates with TPSA at r = -0.571 across our library - TPSA being an axis
we put into the RL objective ourselves, which is why a permeability comparison cannot
carry a non-circular claim.

Two steps, because the expensive one runs elsewhere:
  build   geometry.csv + library -> a CSV of SMILES for admet_predict
  report  admet_predict output   -> the endpoint table, survivors vs benchmark panel
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from rdkit import RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ["PANEL_PREFIX", "passing_labels", "write_admet_input"]

RDLogger.DisableLog("rdApp.*")

PANEL_PREFIX = "PANEL_"
# The control is a reference compound, not a candidate; it must not be scored as one.
EXCLUDE_LABELS = {"CONTROL_crystal"}


def passing_labels(geometry_csv: str) -> list[str]:
    """Labels with at least one pose that passed the section 8.5b geometry filter.

    Keyed on the label, not the row: a ligand qualifies if ANY of its poses passed,
    which is what section 8.5b means - the filter asks whether a binding mode exists,
    not whether every sampled mode is one.
    """
    passing: list[str] = []
    seen: set[str] = set()
    with open(geometry_csv, newline="") as handle:
        reader = csv.DictReader(handle)
        for column in ("label", "passes"):
            if column not in (reader.fieldnames or []):
                raise KeyError(f"{geometry_csv}: no '{column}' column "
                               f"(found {reader.fieldnames})")
        for row in reader:
            label = row["label"]
            if label in seen or label in EXCLUDE_LABELS:
                continue
            if row["passes"] == "1":
                seen.add(label)
                passing.append(label)
    return passing


def write_admet_input(out_csv: str, geometry_csv: str,
                      survivors_smi: str = "data/survivors.smi",
                      panel_smi: str = "data/benchmark_panel.smi") -> dict:
    """Write `smiles,label` for the geometry survivors plus the benchmark panel.

    The panel rides along in the same file so both are predicted by the same tool in
    the same run. Section 7.5 made the permeability comparison prediction-to-prediction
    rather than prediction-against-cpd-25's-measured-liabilities; that only holds if
    both sides go through one model.
    """
    from novelty import load_smi

    labels = set(passing_labels(geometry_csv))
    by_label = {label: smiles for smiles, label in load_smi(survivors_smi)}

    missing = sorted(labels - set(by_label))
    rows = [(by_label[label], label) for label in sorted(labels)
            if label in by_label]
    panel = [(smiles, f"{PANEL_PREFIX}{label}")
             for smiles, label in load_smi(panel_smi)]

    with open(out_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "label"])
        writer.writerows(rows + panel)

    return {
        "passing": len(labels),
        "written": len(rows),
        "panel": len(panel),
        "missing_from_survivors": missing,
    }


def summarise(preds_csv: str, out_txt: str | None = None) -> str:
    """The endpoint table: survivor percentiles beside each benchmark compound.

    Endpoint choice follows section 8 item 6 and the poster's use of it - a toxicity
    screen at the lead stage, reported briefly, with permeability alongside. Nothing
    here is a gate.
    """
    rows = list(csv.DictReader(open(preds_csv)))
    gen = [r for r in rows if not r["label"].startswith(PANEL_PREFIX)]
    panel = {r["label"][len(PANEL_PREFIX):]: r
             for r in rows if r["label"].startswith(PANEL_PREFIX)}

    TOX = [("hERG", "hERG block (lower better)"),
           ("AMES", "AMES mutagenicity"),
           ("DILI", "drug-induced liver injury"),
           ("ClinTox", "clinical trial toxicity"),
           ("Carcinogens_Lagunin", "carcinogenicity"),
           ("LD50_Zhu", "acute toxicity LD50 (higher safer)"),
           ("CYP3A4_Veith", "CYP3A4 inhibition"),
           ("CYP2C9_Veith", "CYP2C9 inhibition"),
           ("CYP2D6_Veith", "CYP2D6 inhibition")]
    PERM = [("Caco2_Wang", "Caco-2 (log cm/s, higher better)"),
            ("PAMPA_NCATS", "PAMPA high-permeability prob."),
            ("HIA_Hou", "intestinal absorption prob."),
            ("Bioavailability_Ma", "oral bioavailability prob."),
            ("Pgp_Broccatelli", "P-gp substrate prob."),
            ("Solubility_AqSolDB", "aqueous solubility (logS)")]
    REFERENCES = ("PLN-1474", "A1AFA", "bexotegrast", "CHEMBL4649232")

    out = [f"ADMET on the section 8.5b survivors: {len(gen)} molecules, "
           f"{len(panel)} panel compounds",
           "A SCREEN, NOT A GATE (section 7.5). Caco-2 model error (MAE 0.26-0.28 log)",
           "is the size of inter-laboratory disagreement on the same compound (median",
           "0.57 log), so a gap below ~0.5 log is not resolvable.", ""]

    def column(key, subset):
        return np.array([float(r[key]) for r in subset if r.get(key) not in (None, "")])

    for title, spec in (("TOXICITY", TOX), ("PERMEABILITY / ABSORPTION", PERM)):
        out.append("=" * 100)
        out.append(title)
        out.append("=" * 100)
        header = f"{'endpoint':36s} {'p25':>8s} {'median':>8s} {'p75':>8s} | "
        header += " ".join(f"{name[:10]:>10s}" for name in REFERENCES)
        out.append(header)
        for key, label in spec:
            values = column(key, gen)
            if values.size == 0:
                out.append(f"{label:36s} (absent from predictions)")
                continue
            line = (f"{label:36s} {np.percentile(values, 25):8.3f} "
                    f"{np.median(values):8.3f} {np.percentile(values, 75):8.3f} | ")
            line += " ".join(
                f"{float(panel[n][key]):10.3f}" if n in panel and panel[n].get(key)
                else f"{'-':>10s}" for n in REFERENCES)
            out.append(line)
        out.append("")

    caco = column("Caco2_Wang", gen)
    if caco.size and "PLN-1474" in panel:
        reference = float(panel["PLN-1474"]["Caco2_Wang"])
        tpsa = column("tpsa", gen)
        out += ["=" * 100,
                "Permeability against PLN-1474 (section 9.2's benchmark since 7.5)",
                "=" * 100,
                f"  PLN-1474 predicted Caco-2      {reference:.3f} log cm/s",
                f"  survivor median                {np.median(caco):.3f}"
                f"   (difference {np.median(caco) - reference:+.3f})",
                f"  above PLN-1474                 {(caco > reference).sum()} / "
                f"{caco.size} = {(caco > reference).mean():.1%}",
                f"  above by more than 0.5 log     {(caco > reference + 0.5).sum()} / "
                f"{caco.size} = {(caco > reference + 0.5).mean():.1%}",
                "       ^ the only figure that clears model error and assay spread"]
        if tpsa.size == caco.size:
            out.append(f"  Caco-2 vs TPSA correlation     "
                       f"r = {np.corrcoef(caco, tpsa)[0, 1]:+.3f}")
            out.append("       ^ how much of the signal is an axis we put in the RL")
            out.append("         objective ourselves - the circularity, made visible")

    text = "\n".join(out) + "\n"
    if out_txt:
        with open(out_txt, "w") as handle:
            handle.write(text)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="write the admet_predict input CSV")
    build.add_argument("--geometry", default="results/geometry.csv")
    build.add_argument("--survivors", default="data/survivors.smi")
    build.add_argument("--panel", default="data/benchmark_panel.smi")
    build.add_argument("--out", required=True)

    report = sub.add_parser("report", help="summarise admet_predict output")
    report.add_argument("--preds", required=True)
    report.add_argument("--out-txt", default=None)

    args = parser.parse_args(argv)

    if args.command == "build":
        counts = write_admet_input(args.out, args.geometry, args.survivors,
                                   args.panel)
        print(f"{args.geometry} -> {args.out}")
        for key in ("passing", "written", "panel"):
            print(f"  {key:24s} {counts[key]}")
        if counts["missing_from_survivors"]:
            print(f"  NOT FOUND in survivors: {len(counts['missing_from_survivors'])}"
                  f" -> {', '.join(counts['missing_from_survivors'][:10])}")
            print("  A passing label absent from data/survivors.smi means the two were")
            print("  produced from different libraries; do not proceed on that.")
            return 1
        print(f"\n  next: admet_predict --data_path {args.out} "
              f"--save_path <preds.csv> --smiles_column smiles")
        return 0

    print(summarise(args.preds, args.out_txt), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
