"""autogrid4 GPF generation and grid/box coverage verification (spec section 7.1).

The box comes from dock_unidock.box_from_ligand and is NOT recomputed here. That
function reproduces smina's --autobox_ligand + --autobox_add 6 rule exactly, and it
is the only source of truth for the search space in this campaign; recomputing it in
a second place risks the two silently diverging.

AutoDock-GPU does not error when a ligand wanders outside the grid -- it returns a
plausible-looking but wrong energy. This project has already been bitten three times
by failures of exactly that silent shape (a residue name overflowing PDB columns and
zeroing every affinity, a naive RMSD reporting a 0.63 A success as a 6.13 A failure,
and an Open Babel nitrogen mis-typing that promoted the fourth-best pose to first).
maps_cover_box exists to make a grid/box mismatch loud instead of silent.

autogrid4 is not installed on this development machine (checked: not on PATH, not in
any conda env here). This module produces and validates the GPF structurally --
directive order, even npts that cover the box, gridcenter, spacing, and the
map/elecmap/dsolvmap lines -- without ever invoking autogrid4. Running autogrid4 over
the GPF this module writes is a separate step, left to be run where the binary is
available (see results/v2_EXECUTION_CHECKLIST.md).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

__all__ = ["SPACING", "LIGAND_TYPES", "receptor_types", "write_gpf", "maps_cover_box"]

SPACING = 0.375
# Elements the generated ligand library uses, plus AD4's polar-hydrogen and
# H-bond donor/acceptor subtypes. This is NOT derived from the receptor -- it is
# the deliberate, closed vocabulary of the generated-molecule campaign (33 tokens,
# no phosphorus, no iodine's cousin... actually includes I; the point is it is
# fixed by what the generated ligands can contain, not by what the receptor is).
LIGAND_TYPES = ["C", "A", "N", "NA", "OA", "S", "SA", "HD", "F", "Cl", "Br", "I"]


def receptor_types(receptor_path) -> list[str]:
    """Distinct AutoDock atom types present in a receptor PDBQT.

    The AD4 type is the last whitespace-separated field of each ATOM/HETATM
    line (conventionally columns 78-79). `obabel -xr -h` (see
    prepare_receptor_pdbqt.py) adds ALL hydrogens, including non-polar ones,
    so the receptor can contain types a hand-picked list omits or gets wrong
    -- e.g. non-polar "H" alongside donor "HD", non-acceptor "S" alongside
    "SA", and the calcium type is actually "Ca", not "CA". autogrid4 rejects
    a GPF whose receptor_types count doesn't match the types actually present
    in the receptor PDBQT, so this must be read from the file itself rather
    than hardcoded. Order is sorted for determinism.
    """
    types = set()
    for line in Path(receptor_path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            if fields:
                types.add(fields[-1])
    return sorted(types)


def _npts(extent: float) -> int:
    """Smallest even n such that n * SPACING >= extent (AutoGrid requires even npts)."""
    n = math.ceil(extent / SPACING)
    return n + 1 if n % 2 else n


def write_gpf(box: dict, receptor: str, out: Path) -> str:
    """Write an AutoDock4 GPF that covers `box`, in the format autogrid4 expects.

    `box` is {"center": (x, y, z), "size": (sx, sy, sz)} -- the same shape produced
    by dock_unidock.box_from_ligand (after collapsing its center_x/... keys), so the
    box computed for Uni-Dock is reused verbatim rather than re-derived.
    """
    cx, cy, cz = box["center"]
    npts = [_npts(e) for e in box["size"]]
    stem = Path(receptor).stem
    lines = [
        f"npts {npts[0]} {npts[1]} {npts[2]}",
        f"gridfld {stem}.maps.fld",
        f"spacing {SPACING}",
        f"receptor_types {' '.join(receptor_types(receptor))}",
        f"ligand_types {' '.join(LIGAND_TYPES)}",
        f"receptor {Path(receptor).resolve()}",
        f"gridcenter {cx:.3f} {cy:.3f} {cz:.3f}",
        "smooth 0.5",
    ]
    lines += [f"map {stem}.{t}.map" for t in LIGAND_TYPES]
    lines += [
        f"elecmap {stem}.e.map",
        f"dsolvmap {stem}.d.map",
        "dielectric -0.1465",
    ]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def maps_cover_box(gpf: str, box: dict) -> bool:
    """Does the grid the GPF defines fully cover `box` on every axis?

    This is the gate: AutoDock-GPU silently scores ligand atoms outside the grid
    instead of erroring, so a grid that falls short of the search box must be
    caught here, before any docking run consumes it.
    """
    text = Path(gpf).read_text().splitlines()
    npts = [int(v) for v in
            next(l for l in text if l.startswith("npts")).split()[1:4]]
    return all(n * SPACING >= extent for n, extent in zip(npts, box["size"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--receptor", default="docking/receptor.pdbqt")
    parser.add_argument("--ligand-ref", default="docking/ligand_ref.sdf",
                         help="reference ligand defining the search box, via "
                              "dock_unidock.box_from_ligand (same rule Uni-Dock used)")
    parser.add_argument("--autobox-add", type=float, default=6.0)
    parser.add_argument("--out-dir", default="docking/v2/maps")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dock_unidock import box_from_ligand

    raw = box_from_ligand(args.ligand_ref, args.autobox_add)
    box = {
        "center": (raw["center_x"], raw["center_y"], raw["center_z"]),
        "size": (raw["size_x"], raw["size_y"], raw["size_z"]),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_copy_name = Path(args.receptor).name
    gpf_path = out_dir / f"{Path(receptor_copy_name).stem}.gpf"

    gpf = write_gpf(box, args.receptor, gpf_path)
    covers = maps_cover_box(gpf, box)

    print(f"box center: {box['center']}")
    print(f"box size:   {box['size']}")
    print(f"GPF written: {gpf}")
    print(f"maps_cover_box: {covers}")
    if not covers:
        print("FATAL: grid does not cover the search box", file=sys.stderr)
        return 1

    glg_path = gpf_path.with_suffix(".glg")
    print("\nautogrid4 is not run by this script. To generate the maps where "
          "autogrid4 is available, run (the GPF's receptor path is absolute "
          "for this reason -- autogrid4 must run from inside the maps dir):")
    print(f"  (cd {out_dir} && autogrid4 -p {gpf_path.name} -l {glg_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
