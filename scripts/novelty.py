"""Scaffold novelty: the cross-scaffold distance band and the Murcko gate.

Blueprint section 8.3. The band is REPORTING CONTEXT, not a gate - a distance
cut-off passes 62% of molecules that sit on a published Murcko scaffold. The gate
is Murcko scaffold membership, which lives in scripts/known_scaffolds.py.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from normalize import FPGEN, canonical_tautomer

__all__ = ["cross_scaffold_band", "load_smi", "murcko"]


def load_smi(path: str) -> list[tuple[str, str]]:
    """Read a .smi file as (smiles, label) pairs, skipping comments and blanks."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            rows.append((fields[0], " ".join(fields[1:])))
    return rows


def murcko(mol: Chem.Mol) -> str:
    """Murcko scaffold SMILES. Acyclic molecules give "" - see Review Focus 4."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def _prepare(smiles_list: list[str], normalize: bool) -> list[Chem.Mol]:
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if normalize:
            mol = canonical_tautomer(mol)
        else:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        if mol is not None:
            out.append(mol)
    return out


def cross_scaffold_band(smiles_list: list[str], normalize: bool) -> dict:
    """How far apart are two known actives that belong to different scaffolds?

    For each molecule, the nearest neighbour among molecules with a DIFFERENT Murcko
    scaffold. Returns n and the 25/50/75/90th percentiles of that distribution.
    """
    mols = _prepare(smiles_list, normalize)
    fps = [FPGEN.GetCountFingerprint(mol) for mol in mols]
    scaffolds = [murcko(mol) for mol in mols]

    cross = []
    for i in range(len(mols)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        other = [sims[j] for j in range(len(mols)) if scaffolds[j] != scaffolds[i]]
        if other:
            cross.append(max(other))

    band = {"n": len(cross)}
    for percentile in (25, 50, 75, 90):
        band[f"p{percentile}"] = round(float(np.percentile(cross, percentile)), 3)
    return band
