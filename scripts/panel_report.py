"""Score the section 3.4 positive benchmark panel under the section 5.1 objective.

Section 7's deliverable is the PER-COMPONENT reference distribution behind every
comparison the poster makes. The total is recorded but is not a basis for
comparison - it is the value of an objective we designed, so comparing on it is
circular (sections 3.4, 5.4).

Two arms, because section 6.5 measured that the objective's verdict can flip with
the tautomer written:
  as-curated  the panel exactly as data/benchmark_panel.smi holds it
  normalized  after section 8.0 canonicalization

Section 9.5 compares generated molecules (which are normalized - library.smi
column 1) against this panel, so the normalized arm is the one that makes that
comparison symmetric. Both are reported rather than one being chosen silently.

This module also cross-checks scripts/objective.py against REINVENT's own scoring
output. The two differ by exactly one filter - PAINS, which CustomAlerts cannot
express - and no panel molecule matches PAINS, so on this panel they must agree.
Section 8 runs objective.py over 19,485 molecules; if it disagrees with the real
scorer on the six compounds the poster cites, that has to surface here first.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from rdkit import Chem, RDLogger

import objective
from normalize import canonical_tautomer, flatten
from novelty import load_smi

__all__ = [
    "COMPONENTS",
    "PANEL",
    "crosscheck",
    "panel_rows",
    "read_reinvent_csv",
    "score_panel",
    "write_panel_smiles",
]

RDLogger.DisableLog("rdApp.*")

PANEL = "data/benchmark_panel.smi"

# objective.py key -> the column REINVENT writes, taken from the `name` fields in
# configs/_rl_scoring.frag. A rename there breaks read_reinvent_csv loudly.
COMPONENTS = {
    "total": "Score",
    "cooh": "carboxylate MIDAS anchor",
    "tpsa": "TPSA",
    "sascore": "SA score",
    "alerts": "unwanted groups",
}


def panel_rows(path: str = PANEL) -> list[tuple[str, str]]:
    """(smiles, label) for the six positive-panel compounds, in file order."""
    return load_smi(path)


def write_panel_smiles(out_path: str, normalize: bool, path: str = PANEL) -> list[str]:
    """Write plain SMILES for REINVENT and return the labels in the same order.

    REINVENT's smiles_file reader takes the whole line, so the label column and the
    comment header must be stripped. Its output CSV carries no labels, so the
    returned list is the only thing that maps CSV rows back to compounds - callers
    must zip against it rather than re-reading the panel.
    """
    labels, smiles = [], []
    for raw, label in panel_rows(path):
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            raise ValueError(f"{path}: {label} does not parse")
        mol = canonical_tautomer(mol) if normalize else flatten(mol)
        if mol is None:
            raise ValueError(f"{path}: {label} failed normalization")
        labels.append(label)
        smiles.append(Chem.MolToSmiles(mol))
    with open(out_path, "w") as handle:
        handle.write("\n".join(smiles) + "\n")
    return labels


def score_panel(normalize: bool, path: str = PANEL) -> dict[str, dict[str, float]]:
    """Per-component scores keyed by compound label, via the reference scorer."""
    labels, smiles = [], []
    for raw, label in panel_rows(path):
        mol = Chem.MolFromSmiles(raw)
        mol = canonical_tautomer(mol) if normalize else flatten(mol)
        labels.append(label)
        smiles.append(Chem.MolToSmiles(mol))
    scored = objective.score_smiles(smiles)
    if len(scored["smiles"]) != len(labels):
        raise ValueError("a panel molecule failed to parse; refusing to mislabel")
    return {
        label: {key: float(scored[key][i]) for key in COMPONENTS}
        for i, label in enumerate(labels)
    }


def read_reinvent_csv(path: str) -> list[dict[str, float]]:
    """REINVENT scoring output as objective.py-keyed rows, in file order.

    Raises KeyError on a missing component column: a silently absent component
    would let the cross-check pass by comparing nothing.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = [col for col in COMPONENTS.values() if col not in fields]
        if missing:
            raise KeyError(f"{path}: missing component columns {missing} "
                           f"(found {fields})")
        return [{key: float(row[col]) for key, col in COMPONENTS.items()}
                for row in reader]


def crosscheck(reference: dict[str, dict[str, float]], labels: list[str],
               reinvent: list[dict[str, float]]) -> dict[str, float]:
    """Max absolute per-component difference between the two scorers."""
    if len(labels) != len(reinvent):
        raise ValueError(
            f"{len(labels)} panel compounds but {len(reinvent)} scored rows; "
            "refusing to compare misaligned data"
        )
    return {
        key: float(max(abs(reference[label][key] - row[key])
                       for label, row in zip(labels, reinvent)))
        for key in COMPONENTS
    }


def _print_table(title: str, scores: dict[str, dict[str, float]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'compound':16s} {'COOH':>7s} {'TPSA':>7s} {'SAScore':>8s} "
          f"{'alerts':>7s} | {'total':>7s}")
    for label, row in scores.items():
        print(f"{label:16s} {row['cooh']:7.3f} {row['tpsa']:7.3f} "
              f"{row['sascore']:8.3f} {row['alerts']:7.0f} | {row['total']:7.3f}")
    print("  total is recorded, NOT a basis for comparison (§3.4, §5.4)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reinvent-csv", help="REINVENT scoring output, as-curated")
    parser.add_argument("--reinvent-csv-normalized",
                        help="REINVENT scoring output, normalized panel")
    args = parser.parse_args(argv)

    curated = score_panel(normalize=False)
    normalized = score_panel(normalize=True)
    _print_table("as-curated (data/benchmark_panel.smi)", curated)
    _print_table("normalized (§8.0) - the arm §9.5 compares against", normalized)

    print("\n=== normalization moves which compounds? ===")
    for label in curated:
        deltas = {k: normalized[label][k] - curated[label][k] for k in COMPONENTS}
        moved = {k: v for k, v in deltas.items() if abs(v) > 1e-9}
        print(f"  {label:16s} {moved if moved else 'unchanged'}")

    labels = [label for _, label in panel_rows()]
    for csv_path, reference, name in (
        (args.reinvent_csv, curated, "as-curated"),
        (args.reinvent_csv_normalized, normalized, "normalized"),
    ):
        if not csv_path:
            continue
        diffs = crosscheck(reference, labels, read_reinvent_csv(csv_path))
        print(f"\n=== cross-check vs REINVENT ({name}): max |difference| ===")
        for key, value in diffs.items():
            print(f"  {key:10s} {value:.3e}")
        worst = max(diffs.values())
        verdict = "AGREE" if worst < 1e-4 else "DISAGREE - investigate before §8"
        print(f"  worst component: {worst:.3e}  -> {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
