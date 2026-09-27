"""The novelty gate: is this Murcko scaffold already in the published record?

Blueprint section 8.3 and section 9.3. Binary, fingerprint-free, threshold-free.
The distance band it replaced passed 39 of 63 prior samples (62%) whose Murcko
scaffold was IDENTICAL to a published active's.

Both sides go through scripts.normalize.canonical_tautomer first: without it the
same compound yields two different Murcko SMILES (verified on PLN-1474).
"""

from __future__ import annotations

from typing import Sequence

from rdkit import Chem

from normalize import canonical_tautomer
from novelty import load_smi, murcko

__all__ = [
    "REFERENCE_FILES",
    "is_scaffold_novel",
    "known_scaffolds",
    "write_known_scaffolds",
]

# Every curated reference set, not just actives_extended. That file is defined as
# ChEMBL alphaVbeta1 actives <= 1 uM, which omits PLN-1474, bexotegrast, and A1AFA
# (pIC50 5.30, below the cut) - so on its own the gate would call a molecule
# rebuilding PLN-1474's scaffold novel. The union is 106 scaffolds.
#
# PLN-1474 IS in ChEMBL, as CHEMBL5933542 - an earlier version of this comment said
# its structure came from AdisInsight rather than a ChEMBL record, which was
# imprecise. ChEMBL holds it unnamed, with no synonyms and max_phase null, and its
# only activity records are three alphaVbeta6 entries (IC50 50 nM). It has NO
# alphaVbeta1 activity record, which is why the actives_extended query misses it
# and why this union exists. Blueprint section 7.5.
REFERENCE_FILES = (
    "data/actives_core.smi",
    "data/actives_core_B.smi",
    "data/actives_extended.smi",
    "data/benchmark_panel.smi",
    "data/similarity_refs.smi",
)
DEFAULT_OUTPUT = "data/known_scaffolds.smi"


def known_scaffolds(paths: Sequence[str] = REFERENCE_FILES) -> set[str]:
    """Murcko scaffolds of every published reference compound, tautomer-canonicalized."""
    scaffolds = set()
    for path in paths:
        for smiles, _ in load_smi(path):
            mol = canonical_tautomer(Chem.MolFromSmiles(smiles))
            if mol is None:
                continue
            scaffold = murcko(mol)
            if scaffold:  # "" means acyclic; no reference compound is acyclic
                scaffolds.add(scaffold)
    return scaffolds


def is_scaffold_novel(smiles: str, known: set[str]) -> bool | None:
    """True if this molecule's Murcko scaffold is absent from `known`.

    None for unparseable input, so the caller skips instead of crashing.
    An acyclic molecule has scaffold "" and is reported novel - correct, but
    the diversity filter buckets every acyclic molecule together, so the
    triage report states how many leads are acyclic.
    """
    mol = canonical_tautomer(Chem.MolFromSmiles(smiles)) if smiles else None
    if mol is None:
        return None
    return murcko(mol) not in known


def write_known_scaffolds(out_path: str = DEFAULT_OUTPUT) -> int:
    """Freeze the scaffold set to disk so triage does not recompute it."""
    scaffolds = sorted(known_scaffolds())
    lines = [
        "# Murcko scaffolds of every curated reference set - actives_core,",
        "# actives_core_B, actives_extended, benchmark_panel, similarity_refs -",
        "# tautomer-canonicalized per blueprint section 8.0. 106 scaffolds.",
        "# A generated molecule whose Murcko scaffold is absent here is",
        "# scaffold-novel (sections 8.3, 9.3). Regenerate with:",
        "#   python3 -c \"import sys; sys.path.insert(0,'scripts'); \\",
        "#     from known_scaffolds import write_known_scaffolds as w; w()\"",
    ]
    lines += scaffolds
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(scaffolds)
