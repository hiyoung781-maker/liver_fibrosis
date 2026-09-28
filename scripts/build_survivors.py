"""Persist the section 8.2 survivor set as data/survivors.smi.

This is the docking input, so it has to be a committed artifact rather than a
number computed in memory: section 8.5's results are only meaningful as results
"about" a named set. Section 6.3 and the ADMET run both report 7,767 survivors of
data/library.smi, and tests/test_survivors.py pins that number to the committed
inputs so the set cannot drift under the results that cite it.

The cuts, cheapest first, exactly as section 8 orders them:
  8.1  carboxylate hard cut (the RL gate's floor is 6.31e-04, not 0)
  8.1  CustomAlerts + PAINS, via scripts/objective.py
  8.2  logP <= 5, MW 250-550, TPSA 40-115
  8.3  Murcko scaffold absent from data/known_scaffolds.smi

Column 1 of library.smi is used - the tautomer-canonical structure (section 8.0).
Section 6.5 measured why that matters: the alert verdict flips with the tautomer on
31 molecules, and the carboxylate verdict on 4.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

import objective
from known_scaffolds import DEFAULT_OUTPUT as KNOWN_SCAFFOLDS
from novelty import load_smi, murcko

__all__ = [
    "WINDOW", "passes_window", "write_survivors",
    "library_path", "survivors_path",
]

RDLogger.DisableLog("rdApp.*")

def library_path(arm: str) -> str:
    """The arm-scoped library.smi this arm's survivors are drawn from."""
    return f"data/v2_{arm}/library.smi"


def survivors_path(arm: str) -> str:
    """The arm-scoped survivors.smi this build writes."""
    return f"data/v2_{arm}/survivors.smi"

# Pre-registered in section 8.2. TPSA deliberately reuses the RL objective's window
# rather than restating it, so the two cannot diverge.
WINDOW = {
    "logp_max": 5.0,
    "mw_range": (250.0, 550.0),
    "tpsa_range": (objective.TPSA_WINDOW[0], objective.TPSA_WINDOW[1]),
}

DEFAULT_LIBRARY = "data/library.smi"
DEFAULT_OUTPUT = "data/survivors.smi"


def passes_window(mol: Chem.Mol | None) -> bool:
    """Section 8.2's property window. False for unparseable input."""
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    lo_mw, hi_mw = WINDOW["mw_range"]
    lo_tpsa, hi_tpsa = WINDOW["tpsa_range"]
    return (Crippen.MolLogP(mol) <= WINDOW["logp_max"]
            and lo_mw <= mw <= hi_mw
            and lo_tpsa <= tpsa <= hi_tpsa)


def write_survivors(out_path: str, library_path: str = DEFAULT_LIBRARY,
                    known: set[str] | None = None) -> dict[str, int]:
    """Apply section 8's cuts to a library and write the survivors with labels.

    Labels are `gen_NNNNN`, assigned in library order. They are the identity used
    by every downstream artifact - docking poses, geometry CSVs, ADMET rows - so
    they must stay stable: regenerating from the same library gives the same labels.
    """
    if known is None:
        with open(KNOWN_SCAFFOLDS) as handle:
            known = {line.strip() for line in handle
                     if line.strip() and not line.startswith("#")}

    smiles = [s for s, _ in load_smi(library_path)]
    scored = objective.score_smiles(smiles)
    mols = [Chem.MolFromSmiles(s) for s in scored["smiles"]]

    cooh = scored["cooh"] > 0
    alert_free = scored["alerts"] == 1.0
    window = np.array([passes_window(m) for m in mols])
    novel = np.array([murcko(m) not in known for m in mols])
    keep = cooh & alert_free & window & novel

    counts = {
        "input": len(smiles),
        "parsed": len(mols),
        "carboxylate": int(cooh.sum()),
        "alert_free": int((cooh & alert_free).sum()),
        "window": int((cooh & alert_free & window).sum()),
        "survivors": int(keep.sum()),
    }

    lines = [
        "# Section 8.2 survivors - the docking input for section 8.5.",
        f"# from: {library_path} (column 1, tautomer-canonical per section 8.0)",
        f"# known scaffolds: {len(known)} from {KNOWN_SCAFFOLDS}",
        f"# window: logP <= {WINDOW['logp_max']}, "
        f"MW {WINDOW['mw_range'][0]:.0f}-{WINDOW['mw_range'][1]:.0f}, "
        f"TPSA {WINDOW['tpsa_range'][0]:.0f}-{WINDOW['tpsa_range'][1]:.0f}",
        f"# cumulative: input {counts['input']}"
        f" -> carboxylate {counts['carboxylate']}"
        f" -> alert-free {counts['alert_free']}"
        f" -> logP/MW/TPSA window {counts['window']}"
        f" -> Murcko-novel {counts['survivors']}",
    ]
    index = 0
    for smi, k in zip(scored["smiles"], keep):
        if k:
            index += 1
            lines.append(f"{smi}\tgen_{index:05d}")
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", help="e.g. TL-A-prime or TL-C; sets default --library/--out")
    parser.add_argument("--library", help="default: derived from --arm")
    parser.add_argument("--out", help="default: derived from --arm")
    args = parser.parse_args(argv)

    if not args.arm and not (args.library and args.out):
        parser.error("either --arm or both --library and --out are required")

    library = args.library or library_path(args.arm)
    out = args.out or survivors_path(args.arm)

    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    counts = write_survivors(out, library)
    print(f"{library} -> {out}")
    for key, value in counts.items():
        print(f"  {key:14s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
