"""Turn a REINVENT sampling CSV into data/library.smi (blueprint section 6).

The pipeline section 6 prescribes: dedupe -> canonicalize -> tautomer normalize
(section 8.0). Dedupe runs TWICE, on purpose. REINVENT's `unique_molecules = true`
already collapses identical canonical SMILES, so the first pass is cheap
bookkeeping; the second pass, after tautomer canonicalization, is the one that
does work - two tautomers of the same compound are two distinct canonical SMILES
and would otherwise be counted as two molecules. The gap between
n_unique_canonical and n_unique_normalized measures how much of the sampled
library was tautomer-duplicated.

Each output row carries BOTH structures: the normalized SMILES in column 1, and
the pre-normalization canonical SMILES in column 2. Keeping the raw one is not
redundancy. The RL objective scored the raw generated SMILES, while every
downstream claim is made about the normalized structure, and tautomer choice
moves descriptors measurably - PLN-1474 reads QED 0.433 / TPSA 100.0 as sampled
and 0.462 / 100.6 canonicalized. A molecule can therefore pass the RL TPSA window
and fail the section 8.2 window. scripts/library_report.py reports that
discrepancy rate from these two columns instead of leaving it invisible.
"""

from __future__ import annotations

import argparse
import csv
import sys

from rdkit import Chem, RDLogger

import normalize
from normalize import canonical_tautomer, flatten

__all__ = [
    "build_library", "read_sampling_csv", "write_library",
    "output_path", "LIBRARY_COLUMNS", "input_path",
]

RDLogger.DisableLog("rdApp.*")

# v1 wrote to the arm-less results/leads.csv-style paths. Re-running v2 with the
# same paths would silently overwrite the previous arm's (or v1's) results,
# making the two RL arms incomparable. Every path below is scoped by arm.
LIBRARY_COLUMNS = ("smiles", "smiles_as_scored")


def input_path(arm: str) -> str:
    """The arm-scoped REINVENT sampling CSV this build reads from."""
    return f"results/v2_{arm}/sample.csv"


def output_path(arm: str) -> str:
    """The arm-scoped library.smi this build writes."""
    return f"data/v2_{arm}/library.smi"


def read_sampling_csv(path: str) -> list[str]:
    """SMILES column of a REINVENT sampling CSV, blank cells skipped.

    Raises KeyError if the column is absent rather than returning []: an empty
    list is indistinguishable from a run that sampled nothing, and would produce
    an empty library that looks like a successful build.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "SMILES" not in reader.fieldnames:
            raise KeyError(
                f"{path}: no SMILES column (found {reader.fieldnames})"
            )
        return [row["SMILES"] for row in reader if row.get("SMILES")]


def build_library(smiles_list: list[str]) -> dict:
    """Validate, dedupe, normalize. Returns rows plus every count worth reporting.

    rows: (normalized_smiles, raw_canonical_smiles), first occurrence wins, input
    order preserved so the library is reproducible rather than set-ordered.
    """
    failures_before = len(normalize.TAUTOMER_FAILURES)

    canonical: dict[str, Chem.Mol] = {}
    n_valid = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        n_valid += 1
        flat = flatten(mol)
        if flat is None:
            continue
        key = Chem.MolToSmiles(flat)
        canonical.setdefault(key, flat)

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw, flat in canonical.items():
        normalized_mol = canonical_tautomer(flat)
        if normalized_mol is None:
            continue
        normalized = Chem.MolToSmiles(normalized_mol)
        if normalized in seen:
            continue
        seen.add(normalized)
        rows.append((normalized, raw))

    return {
        "rows": rows,
        "n_input": len(smiles_list),
        "n_valid": n_valid,
        "n_unique_canonical": len(canonical),
        "n_unique_normalized": len(rows),
        "n_tautomer_failures": len(normalize.TAUTOMER_FAILURES) - failures_before,
    }


def write_library(path: str, result: dict, source: str) -> None:
    """Write library.smi: provenance header, then `normalized raw` per line."""
    lines = [
        "# Sampled library, blueprint section 6.",
        f"# source: {source}",
        "# column 1: tautomer-canonicalized SMILES (section 8.0) - the structure",
        "#           every downstream property and novelty claim is made about.",
        "# column 2: pre-normalization canonical SMILES - the structure the RL",
        "#           objective actually scored. Kept so the discrepancy between",
        "#           the two is measurable, not invisible.",
        f"# sampled: {result['n_input']}"
        f"  valid: {result['n_valid']}"
        f"  unique canonical: {result['n_unique_canonical']}"
        f"  unique normalized: {result['n_unique_normalized']}"
        f"  tautomer failures: {result['n_tautomer_failures']}",
    ]
    lines += [f"{normalized} {raw}" for normalized, raw in result["rows"]]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", help="e.g. TL-A-prime or TL-C; sets default --input/--output")
    parser.add_argument(
        "--input", help="REINVENT sampling CSV (default: derived from --arm)")
    parser.add_argument(
        "--output", help="library .smi to write (default: derived from --arm)")
    args = parser.parse_args(argv)

    if not args.arm and not (args.input and args.output):
        parser.error("either --arm or both --input and --output are required")

    input_csv = args.input or input_path(args.arm)
    output_smi = args.output or output_path(args.arm)

    import os
    os.makedirs(os.path.dirname(output_smi) or ".", exist_ok=True)

    sampled = read_sampling_csv(input_csv)
    result = build_library(sampled)
    write_library(output_smi, result, source=input_csv)

    print(f"{input_csv} -> {output_smi}")
    for key in ("n_input", "n_valid", "n_unique_canonical",
                "n_unique_normalized", "n_tautomer_failures"):
        print(f"  {key:22s} {result[key]}")
    collapsed = result["n_unique_canonical"] - result["n_unique_normalized"]
    print(f"  {'tautomer duplicates':22s} {collapsed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
