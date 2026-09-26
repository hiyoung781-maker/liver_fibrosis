"""Shared structure normalization for every fingerprint and scaffold comparison.

Reference sets and measured sets MUST pass through the same normalization before
any Tanimoto or Murcko comparison. Blueprint section 8.0 explains why: the same
molecule written as a different tautomer scores 0.458 against itself and yields a
different Murcko SMILES, which lets a known active pass as scaffold-novel.

This module deliberately does NOT touch curate_actives.standardize(), whose output
is written to data/actives_core.smi - the TL-A training input. See the plan.
"""

from __future__ import annotations

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

__all__ = [
    "FPGEN",
    "TAUTOMER_FAILURES",
    "canonical_tautomer",
    "canonical_tautomer_smiles",
    "flatten",
]

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)

_TAUTOMER = rdMolStandardize.TautomerEnumerator()

# SMILES whose tautomer canonicalization raised. Callers report this count rather
# than letting a silent fallback desynchronize reference and measured sets.
TAUTOMER_FAILURES: list[str] = []


def flatten(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Drop stereochemistry - the de novo prior's vocabulary has no stereo tokens."""
    if mol is None:
        return None
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))


def canonical_tautomer(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Flatten, then pick RDKit's canonical tautomer. None if either step fails."""
    flat = flatten(mol)
    if flat is None:
        return None
    try:
        with rdBase.BlockLogs():
            return _TAUTOMER.Canonicalize(flat)
    except Exception:
        TAUTOMER_FAILURES.append(Chem.MolToSmiles(flat))
        return flat


def canonical_tautomer_smiles(smiles: str | None) -> str | None:
    """Canonical SMILES of the canonical tautomer, or None for unparseable input."""
    if not smiles:
        return None
    mol = canonical_tautomer(Chem.MolFromSmiles(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
