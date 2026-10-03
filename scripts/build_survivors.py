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
from make_rl_config import RL_TRANSFORM_OVERRIDES, campaign_for
from novelty import load_smi, murcko

__all__ = [
    "WINDOW", "V4_WINDOW", "window_for_arm", "passes_window", "write_survivors",
    "library_path", "survivors_path",
]

RDLogger.DisableLog("rdApp.*")

def library_path(arm: str) -> str:
    """The arm-scoped library.smi this arm's survivors are drawn from."""
    return f"data/{campaign_for(arm)}_{arm}/library.smi"


def survivors_path(arm: str) -> str:
    """The arm-scoped survivors.smi this build writes."""
    return f"data/{campaign_for(arm)}_{arm}/survivors.smi"

# Pre-registered in section 8.2. TPSA deliberately reuses the RL objective's window
# rather than restating it, so the two cannot diverge.
WINDOW = {
    "logp_max": 5.0,
    "mw_range": (250.0, 550.0),
    "tpsa_range": (objective.TPSA_WINDOW[0], objective.TPSA_WINDOW[1]),
}

# THE SAME RULE, APPLIED TO THE WINDOW v4 ACTUALLY GENERATED AGAINST. The comment
# above is the intent - the gate must not diverge from the RL objective - and for
# v4 that intent points somewhere else, because v4's RL window was re-derived to
# (60, 140) (results/v4_tpsa_window.md) while objective.TPSA_WINDOW stays frozen
# at the v2 blueprint's (40, 115). Reading objective.TPSA_WINDOW for a v4 arm
# would gate the library at a ceiling the library was never generated under:
# TL-B's sampled TPSA median is 121.8 (spec §3.4), above the old 115, so over
# half of what v4 makes would be discarded at the step that decides what gets
# docked - and §4.6's criterion (a) would read the shortfall as grounds to switch
# engines.
#
# WHY THIS IS A SECOND CONSTANT RATHER THAN AN EDIT TO WINDOW. §6.3 and the ADMET
# run report 7,767 survivors of data/library.smi, and tests/test_survivors.py
# RECOMPUTES that number rather than reading a committed file. Moving WINDOW
# would change the count under a result that already cites it. The v2 window
# stays the default; v4 passes its own, sourced from make_rl_config so the
# "cannot diverge" property holds on the v4 side too.
V4_WINDOW = {**WINDOW, "tpsa_range": (RL_TRANSFORM_OVERRIDES["TPSA"]["low"],
                                      RL_TRANSFORM_OVERRIDES["TPSA"]["high"])}


def window_for_arm(arm: str | None) -> dict:
    """V4_WINDOW for a v4 arm, the blueprint WINDOW otherwise.

    An undeclared arm gets the blueprint window: an arm has to be declared in
    make_rl_config.ARMS to be treated as v4, so forgetting to declare one fails
    toward the pre-registered value rather than silently adopting a wider gate.
    """
    return V4_WINDOW if campaign_for(arm) == "v4" else WINDOW

DEFAULT_LIBRARY = "data/library.smi"
DEFAULT_OUTPUT = "data/survivors.smi"


def passes_window(mol: Chem.Mol | None, window: dict | None = None) -> bool:
    """Section 8.2's property window. False for unparseable input.

    `window` defaults to the pre-registered WINDOW so every existing caller and
    the §6.3 survivor count are unaffected; a v4 arm passes V4_WINDOW.
    """
    if mol is None:
        return False
    window = WINDOW if window is None else window
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    lo_mw, hi_mw = window["mw_range"]
    lo_tpsa, hi_tpsa = window["tpsa_range"]
    return (Crippen.MolLogP(mol) <= window["logp_max"]
            and lo_mw <= mw <= hi_mw
            and lo_tpsa <= tpsa <= hi_tpsa)


def write_survivors(out_path: str, library_path: str = DEFAULT_LIBRARY,
                    known: set[str] | None = None,
                    window: dict | None = None) -> dict[str, int]:
    """Apply section 8's cuts to a library and write the survivors with labels.

    Labels are `gen_NNNNN`, assigned in library order. They are the identity used
    by every downstream artifact - docking poses, geometry CSVs, ADMET rows - so
    they must stay stable: regenerating from the same library gives the same labels.
    """
    window = WINDOW if window is None else window
    if known is None:
        with open(KNOWN_SCAFFOLDS) as handle:
            known = {line.strip() for line in handle
                     if line.strip() and not line.startswith("#")}

    smiles = [s for s, _ in load_smi(library_path)]
    scored = objective.score_smiles(smiles)
    mols = [Chem.MolFromSmiles(s) for s in scored["smiles"]]

    cooh = scored["cooh"] > 0
    alert_free = scored["alerts"] == 1.0
    in_window = np.array([passes_window(m, window) for m in mols])
    novel = np.array([murcko(m) not in known for m in mols])
    keep = cooh & alert_free & in_window & novel

    counts = {
        "input": len(smiles),
        "parsed": len(mols),
        "carboxylate": int(cooh.sum()),
        "alert_free": int((cooh & alert_free).sum()),
        "window": int((cooh & alert_free & in_window).sum()),
        "survivors": int(keep.sum()),
    }

    lines = [
        "# Section 8.2 survivors - the docking input for section 8.5.",
        f"# from: {library_path} (column 1, tautomer-canonical per section 8.0)",
        f"# known scaffolds: {len(known)} from {KNOWN_SCAFFOLDS}",
        f"# window: logP <= {window['logp_max']}, "
        f"MW {window['mw_range'][0]:.0f}-{window['mw_range'][1]:.0f}, "
        f"TPSA {window['tpsa_range'][0]:.0f}-{window['tpsa_range'][1]:.0f}",
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
        "--arm", help="e.g. TL-A-prime, TL-C, or TL-B; sets default "
                      "--library/--out and, for a v4 arm, the TPSA window")
    parser.add_argument("--library", help="default: derived from --arm")
    parser.add_argument("--out", help="default: derived from --arm")
    args = parser.parse_args(argv)

    if not args.arm and not (args.library and args.out):
        parser.error("either --arm or both --library and --out are required")

    library = args.library or library_path(args.arm)
    out = args.out or survivors_path(args.arm)
    window = window_for_arm(args.arm)

    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    counts = write_survivors(out, library, window=window)
    print(f"{library} -> {out}")
    print(f"  TPSA window    {window['tpsa_range'][0]:.0f}-"
          f"{window['tpsa_range'][1]:.0f}"
          f"{'  (v4, from make_rl_config)' if window is V4_WINDOW else '  (v2 blueprint)'}")
    for key, value in counts.items():
        print(f"  {key:14s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
