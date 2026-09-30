"""2D structure images for the presentation, drawn from SMILES.

WHERE THE SMILES COME FROM. Every structure is read from the file that
produced it -- the ADMET prediction table for the generated molecules, the
curated annotation table for compound 25, the benchmark panel for the
references. None is retyped here. A structure figure that disagrees with the
data behind it is the worst kind of figure, and the only reliable way to stop
that is to never have a second copy of the SMILES.

WHAT IS DRAWN ON EACH. The carboxylate that anchors the MIDAS is highlighted,
because it is the group section 8.5b's geometry gate measures and the group
that costs these molecules their permeability -- the tension the campaign set
out to test is visible in the structures themselves.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ["CARBOXYLATE", "FIGURES", "draw", "grid"]

CARBOXYLATE = "[CX3](=O)[OX2H1,OX1-]"

# label -> (source file, the column holding the SMILES, the key to match)
SOURCES = {
    "admet": ("results/v2_TL-C/admet_preds.csv", "smiles", "label"),
    "panel": ("data/benchmark_panel.smi", None, None),
    "actives": ("data/actives_annotated.csv", "smiles", "identifier"),
}

# What goes on the slides, in the order they are discussed.
FIGURES = [
    ("gen_01883", "admet", "LEAD"),
    ("gen_01431", "admet", "trade-off: selectivity vs DILI"),
    ("gen_02452", "admet", "trade-off: selectivity vs NR-AhR"),
    ("gen_03050", "admet", "trade-off: selectivity vs SR-MMP"),
    ("PANEL_PLN-1474", "admet", "reference (Phase 1)"),
    ("CHEMBL5532604", "actives", "compound 25 (IC50 0.32 nM)"),
]


def _read_smiles() -> dict:
    out = {}
    path, column, key = SOURCES["admet"]
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row[key], "admet")] = row[column]
    path, column, key = SOURCES["actives"]
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row[key], "actives")] = row[column]
    with open(SOURCES["panel"][0]) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    out[(parts[1], "panel")] = parts[0]
    return out


def draw(smiles: str, path: str, size=(500, 420), caption: str = "") -> None:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles}")
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.StraightenDepiction(mol)
    hits = [i for match in mol.GetSubstructMatches(Chem.MolFromSmarts(CARBOXYLATE))
            for i in match]
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.bondLineWidth = 2
    if caption:
        options.legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, legend=caption, highlightAtoms=hits,
        highlightAtomColors={i: (1.0, 0.85, 0.6) for i in hits})
    drawer.FinishDrawing()
    Path(path).write_bytes(drawer.GetDrawingText())


def grid(entries: list, path: str, per_row: int = 3,
         size=(420, 360)) -> None:
    """One sheet with every structure, for a single slide."""
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor

    mols, legends, highlights = [], [], []
    pattern = Chem.MolFromSmarts(CARBOXYLATE)
    for label, smiles, caption in entries:
        mol = Chem.MolFromSmiles(smiles)
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(f"{label}  —  {caption}")
        highlights.append([i for match in mol.GetSubstructMatches(pattern)
                           for i in match])
    image = Draw.MolsToGridImage(mols, molsPerRow=per_row, subImgSize=size,
                                 legends=legends,
                                 highlightAtomLists=highlights,
                                 useSVG=False, returnPNG=False)
    image.save(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("figures/structures"))
    p.add_argument("--grid", type=Path,
                    default=Path("figures/v2_structures.png"))
    args = p.parse_args(argv)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    smiles = _read_smiles()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries, missing = [], []
    for label, source, caption in FIGURES:
        found = smiles.get((label, source))
        if found is None:
            missing.append(f"{label} ({source})")
            continue
        display = label.replace("PANEL_", "")
        path = args.out_dir / f"{display}.png"
        draw(found, str(path), caption=f"{display} — {caption}")
        entries.append((display, found, caption))
        print(f"  {display:16s} {path}")

    if missing:
        sys.stderr.write(
            "SMILES not found for: " + ", ".join(missing)
            + "\n  Each label is looked up in the file that produced it; a "
              "miss means the label is spelled differently there, not that "
              "the molecule is absent.\n")
        return 1

    args.grid.parent.mkdir(parents=True, exist_ok=True)
    grid(entries, str(args.grid))
    print(f"  grid            {args.grid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
