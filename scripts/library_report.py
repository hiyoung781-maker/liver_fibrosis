"""Descriptive distributions for a sampled library (blueprint section 6).

This module GATES NOTHING. Section 6's deliverable is a property and novelty
distribution; the cuts live in section 8's triage. Keeping the two apart matters
because section 9 pre-registers the triage thresholds - a report that quietly
filtered would make those thresholds unfalsifiable.

Reads the two-column library.smi that scripts/build_library.py writes and never
re-normalizes column 1. Column 1 is already tautomer-canonical, and re-running
the canonicalizer would repeat its ~0.14 s/molecule cost - 47 min per arm.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED

import objective
from normalize import FPGEN, canonical_tautomer
from novelty import load_smi, murcko

__all__ = [
    "normalization_discrepancy",
    "novelty_summary",
    "objective_summary",
    "percentiles",
    "property_summary",
    "read_library",
    "similarity_summary",
]

RDLogger.DisableLog("rdApp.*")

PROPERTIES = {
    "MW": Descriptors.MolWt,
    "cLogP": Crippen.MolLogP,
    "TPSA": Descriptors.TPSA,
    "HBD": Descriptors.NumHDonors,
    "HBA": Descriptors.NumHAcceptors,
    "RotB": Descriptors.NumRotatableBonds,
    "AromaticRings": Descriptors.NumAromaticRings,
    "HeavyAtoms": Descriptors.HeavyAtomCount,
    "QED": QED.qed,
}


def read_library(path: str) -> list[tuple[str, str]]:
    """(normalized, raw_canonical) pairs from a library.smi."""
    return [(smiles, label) for smiles, label in load_smi(path)]


def percentiles(values) -> dict:
    """n plus p10/p25/median/p75/p90. All-None when empty, so callers can print."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"n": 0, "p10": None, "p25": None, "median": None,
                "p75": None, "p90": None, "mean": None}
    return {
        "n": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "mean": float(array.mean()),
    }


def _mols(smiles_list: list[str]) -> list[Chem.Mol]:
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    return [m for m in mols if m is not None]


def property_summary(smiles_list: list[str]) -> dict:
    mols = _mols(smiles_list)
    return {name: percentiles([fn(m) for m in mols])
            for name, fn in PROPERTIES.items()}


def objective_summary(smiles_list: list[str]) -> dict:
    """Rates and score distribution under the section 5.1 RL objective.

    Reuses scripts/objective.py rather than restating the transforms, so the
    report can never drift from the objective the agent was trained against.
    SAScore comes from the same module for the same reason.
    """
    scored = objective.score_smiles(smiles_list)
    n = len(scored["smiles"])
    if n == 0:
        return {"n": 0, "carboxylate_rate": None, "alert_free_rate": None,
                "total": percentiles([]), "SAScore": percentiles([])}
    mols = _mols(scored["smiles"])
    return {
        "n": n,
        "carboxylate_rate": float(np.mean(scored["cooh"] > 0)),
        "alert_free_rate": float(np.mean(scored["alerts"] == 1.0)),
        "total": percentiles(scored["total"]),
        "SAScore": percentiles([objective.sascorer.calculateScore(m) for m in mols]),
    }


def novelty_summary(smiles_list: list[str], known: set[str]) -> dict:
    """Murcko novelty against `known`, computed WITHOUT re-normalizing.

    Deliberately does not call known_scaffolds.is_scaffold_novel: that helper
    normalizes its input, which is correct for raw generator output and wasteful
    here, where column 1 already went through section 8.0.

    An acyclic molecule's Murcko scaffold is "", which is absent from every
    reference set and so counts as novel. That is the right answer but a weak
    one, and the diversity filter buckets all acyclic molecules together, so the
    count is reported separately instead of being folded into the rate.
    """
    scaffolds = [murcko(m) for m in _mols(smiles_list)]
    if not scaffolds:
        return {"n": 0, "murcko_novel_rate": None, "n_unique_scaffolds": 0,
                "n_acyclic": 0}
    return {
        "n": len(scaffolds),
        "murcko_novel_rate": float(np.mean([s not in known for s in scaffolds])),
        "n_unique_scaffolds": len(set(scaffolds)),
        "n_acyclic": sum(1 for s in scaffolds if s == ""),
    }


def similarity_summary(smiles_list: list[str], reference_path: str) -> dict:
    """Nearest-neighbour Tanimoto to a reference set, per generated molecule.

    The reference side IS normalized here - reference .smi files hold published
    structures as curated, and section 8.0's rule is that both sides of a
    comparison pass through the same normalization. Column 1 is already
    normalized, so normalizing the reference is what makes the pair symmetric.
    """
    ref_mols = [canonical_tautomer(Chem.MolFromSmiles(s))
                for s, _ in load_smi(reference_path)]
    ref_fps = [FPGEN.GetCountFingerprint(m) for m in ref_mols if m is not None]
    nn = []
    for mol in _mols(smiles_list):
        sims = DataStructs.BulkTanimotoSimilarity(
            FPGEN.GetCountFingerprint(mol), ref_fps)
        if sims:
            nn.append(max(sims))
    return percentiles(nn)


def normalization_discrepancy(rows: list[tuple[str, str]]) -> dict:
    """How far apart are the two columns under the RL objective?

    The agent was rewarded on column 2 and judged on column 1. Tautomer choice
    shifts descriptors - PLN-1474 reads TPSA 100.0 as sampled and 100.6
    canonicalized - so a molecule can clear the RL TPSA window and miss the
    section 8.2 one. This reports the size of that effect rather than assuming
    it away. A nonzero carboxylate_changed would be the serious case: the MIDAS
    anchor gate deciding differently on the two structures.

    Reports n_shifted and max_shift alongside the percentiles because on the real
    library the percentiles are 0.00 all the way through p90 while dozens of
    molecules do flip - the effect lives entirely in a tail that a five-number
    summary erases.
    """
    normalized = [n for n, _ in rows]
    raw = [r for _, r in rows]
    a = objective.score_smiles(normalized)
    b = objective.score_smiles(raw)
    if len(a["smiles"]) != len(rows) or len(b["smiles"]) != len(rows):
        # Both columns came from RDKit round-trips, so this should be impossible;
        # say so rather than silently comparing misaligned arrays.
        raise ValueError("a library column failed to re-parse; refusing to "
                         f"compare {len(a['smiles'])} vs {len(b['smiles'])} "
                         f"of {len(rows)} rows")
    shift = np.abs(a["total"] - b["total"])
    return {
        "n": len(rows),
        "carboxylate_changed": int(np.sum((a["cooh"] > 0) != (b["cooh"] > 0))),
        "alerts_changed": int(np.sum(a["alerts"] != b["alerts"])),
        "identical_columns": int(np.sum(np.array(normalized) == np.array(raw))),
        "n_shifted": int(np.sum(shift > 1e-9)),
        "max_shift": float(shift.max()) if shift.size else 0.0,
        "total_score_shift": percentiles(shift),
    }


def _fmt(summary: dict) -> str:
    if summary["n"] == 0:
        return "n=0"
    return ("p10={p10:.2f} p25={p25:.2f} med={median:.2f} "
            "p75={p75:.2f} p90={p90:.2f}".format(**summary))


def report(path: str, known: set[str], reference_path: str) -> dict:
    rows = read_library(path)
    normalized = [n for n, _ in rows]
    return {
        "path": path,
        "n_molecules": len(rows),
        "properties": property_summary(normalized),
        "objective": objective_summary(normalized),
        "novelty": novelty_summary(normalized, known),
        "nn_tanimoto": similarity_summary(normalized, reference_path),
        "discrepancy": normalization_discrepancy(rows),
    }


def _print(result: dict) -> None:
    print(f"\n=== {result['path']}  ({result['n_molecules']} molecules) ===")
    print("-- properties --")
    for name, summary in result["properties"].items():
        print(f"  {name:14s} {_fmt(summary)}")
    obj = result["objective"]
    print("-- RL objective (section 5.1) --")
    print(f"  carboxylate     {obj['carboxylate_rate']:.3f}")
    print(f"  alert-free      {obj['alert_free_rate']:.3f}")
    print(f"  SAScore        {_fmt(obj['SAScore'])}")
    print(f"  total          {_fmt(obj['total'])}")
    nov = result["novelty"]
    print("-- novelty (section 8.3) --")
    print(f"  Murcko-novel    {nov['murcko_novel_rate']:.3f}"
          f"   unique scaffolds {nov['n_unique_scaffolds']}"
          f"   acyclic {nov['n_acyclic']}")
    print(f"  NN-Tanimoto    {_fmt(result['nn_tanimoto'])}")
    dis = result["discrepancy"]
    print("-- normalization discrepancy (column 1 vs column 2) --")
    print(f"  identical       {dis['identical_columns']}/{dis['n']}")
    print(f"  carboxylate differs  {dis['carboxylate_changed']}"
          f"   alerts differ  {dis['alerts_changed']}")
    print(f"  |score shift|  {_fmt(dis['total_score_shift'])}")
    print(f"  shifted at all  {dis['n_shifted']}/{dis['n']}"
          f"   max shift {dis['max_shift']:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("libraries", nargs="+", help="library .smi files")
    parser.add_argument("--known", default="data/known_scaffolds.smi")
    parser.add_argument("--reference", default="data/actives_extended.smi")
    args = parser.parse_args(argv)

    with open(args.known) as handle:
        known = {line.strip() for line in handle
                 if line.strip() and not line.startswith("#")}
    print(f"known scaffolds: {len(known)} from {args.known}")

    for path in args.libraries:
        _print(report(path, known, args.reference))
    return 0


if __name__ == "__main__":
    sys.exit(main())
