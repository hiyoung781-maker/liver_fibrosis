"""2D structure images for the presentation, one PNG per molecule.

Same SMILES sources as structure_figures.py -- nothing is retyped here.
Each structure is drawn on a transparent background and saved directly
into figures/, named after the molecule (CHEMBL5532604 -> cpd25).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from structure_figures import CARBOXYLATE, FIGURES, _read_smiles  # noqa: E402

__all__ = ["draw_transparent", "RENAMES"]

# label -> output filename stem
RENAMES = {
    "CHEMBL5532604": "cpd25",
}


# Poster defaults. bondLineWidth is in device pixels, so it must be scaled with
# the canvas or the lines thin out as resolution rises: 8 at 1000x840 is the same
# visual weight as 4 at 500x420, at twice the print resolution. The old default
# (2 at 500x420) was drawn for screen and reads as hairlines at 1-2 m.
DEFAULT_SIZE = (1000, 840)
DEFAULT_BOND_WIDTH = 8


def draw_transparent(smiles: str, path: str, size=DEFAULT_SIZE,
                     bond_width: int = DEFAULT_BOND_WIDTH) -> None:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
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
    options.bondLineWidth = bond_width
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=hits,
        highlightAtomColors={i: (1.0, 0.85, 0.6) for i in hits})
    drawer.FinishDrawing()
    Path(path).write_bytes(drawer.GetDrawingText())


def smiles_from_csv(path: str, label_column: str = "label",
                    smiles_column: str = "smiles") -> dict:
    """{label: smiles} from any CSV carrying both columns.

    The default sources in structure_figures.SOURCES are the v2 run's files.
    A later run's survivors are in its own ADMET input, and retyping a SMILES
    to draw it is exactly how a figure ends up showing a different molecule
    from the one that was filtered.
    """
    import csv

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for column in (label_column, smiles_column):
            if column not in (reader.fieldnames or []):
                raise KeyError(f"{path}: no '{column}' column "
                               f"(found {reader.fieldnames})")
        return {row[label_column]: row[smiles_column] for row in reader
                if row[label_column] and row[smiles_column]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    p.add_argument("--smiles-csv", action="append", default=[],
                    help="A CSV with `label` and `smiles` columns, searched "
                         "before the built-in sources. Repeat; earlier wins.")
    p.add_argument("--labels",
                    help="Comma-separated labels to draw INSTEAD of the "
                         "built-in FIGURES list. Each is looked up in "
                         "--smiles-csv first, then in the built-in sources.")
    p.add_argument("--bond-width", type=int, default=DEFAULT_BOND_WIDTH,
                   help="bond line width in device pixels; scale it with "
                        "--size or the lines thin out")
    p.add_argument("--size", type=lambda v: tuple(int(x) for x in v.split("x")),
                   default=DEFAULT_SIZE, metavar="WxH",
                   help="canvas in pixels, e.g. 1000x840")
    args = p.parse_args(argv)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    smiles = _read_smiles()
    extra: dict = {}
    for path in reversed(args.smiles_csv):
        extra.update(smiles_from_csv(path))

    if args.labels:
        wanted = [(label.strip(), "csv", "")
                  for label in args.labels.split(",") if label.strip()]
    else:
        wanted = list(FIGURES)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    # Flat view of every built-in source, for labels given on the command line:
    # those arrive without the (label, source) pair the FIGURES list carries.
    flat = {label: value for (label, _source), value in smiles.items()}

    for label, source, _caption in wanted:
        found = None
        # A label may be spelled with or without the PANEL_ prefix depending on
        # which file produced it, so each lookup tries both spellings.
        for variant in (label, label.replace("PANEL_", ""), f"PANEL_{label}"):
            found = extra.get(variant) or smiles.get((variant, source)) \
                or flat.get(variant)
            if found is not None:
                break
        if found is None:
            missing.append(f"{label} ({source})")
            continue
        display = label.replace("PANEL_", "")
        display = RENAMES.get(display, display)
        path = args.out_dir / f"{display}.png"
        draw_transparent(found, str(path), size=args.size,
                         bond_width=args.bond_width)
        print(f"  {display:16s} {path}")

    if missing:
        sys.stderr.write(
            "SMILES not found for: " + ", ".join(missing)
            + "\n  Each label is looked up in the file that produced it; a "
              "miss means the label is spelled differently there, not that "
              "the molecule is absent.\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
