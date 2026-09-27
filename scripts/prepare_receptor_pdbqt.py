"""Convert the docking receptor to PDBQT and prove the MIDAS calcium survived.

Uni-Dock, like every Vina-family engine, wants a PDBQT receptor. That conversion is
the riskiest step in the whole section 8.5 pipeline, and the risk is plumbing rather
than science: section 8.5b's filter measures carboxylate-O to Ca501 of chain B, so a
converter that drops the metal or gives it a nonsensical atom type makes EVERY pose
fail a distance test - and the failure reads as chemistry, not as a bug.

Two errors of exactly this shape have already happened in this project: a 5-character
CCD residue name overflowing the PDB resName columns and zeroing every affinity, and
a naive index-wise RMSD reporting 6.13 A where the true value was 0.63 A. Hence this
module's real output is not the PDBQT but the verification table.

What is checked:
  1. Ca 501 of chain B is present at all.
  2. Its coordinates match the source PDB to within a tight tolerance.
  3. Its assigned AutoDock atom type is reported (not asserted - we record what the
     converter chose rather than pretending to know what it should be).
  4. Every receptor atom the filter's second anchor needs - Asn224 backbone O of
     chain B - is likewise present and unmoved.
  5. Nothing silently vanished: atom counts before and after, by record type.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import numpy as np

__all__ = [
    "ANCHORS",
    "compare_anchors",
    "convert_with_obabel",
    "read_pdb_atoms",
    "read_pdbqt_atoms",
]

# (chain, resSeq, resName, atomName) -> label. The two atoms section 8.5b measures.
ANCHORS = {
    ("B", "501", "CA", "CA"): "Ca501 (MIDAS)",
    ("B", "224", "ASN", "O"): "Asn224 backbone O",
}

TOLERANCE = 0.01  # A. A rigid conversion must not move anything at all.


def _parse(path: str, pdbqt: bool) -> list[dict]:
    """Columns 1-54 are identical in PDB and PDBQT; PDBQT adds charge and type."""
    atoms = []
    with open(path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom = {
                "record": line[:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip(),
                "resseq": line[22:26].strip(),
                "xyz": np.array([float(line[30:38]), float(line[38:46]),
                                 float(line[46:54])]),
            }
            if pdbqt:
                atom["adtype"] = line[77:79].strip() if len(line) > 77 else ""
                atom["charge"] = line[70:76].strip() if len(line) > 76 else ""
            atoms.append(atom)
    return atoms


def read_pdb_atoms(path: str) -> list[dict]:
    return _parse(path, pdbqt=False)


def read_pdbqt_atoms(path: str) -> list[dict]:
    return _parse(path, pdbqt=True)


def _key(atom: dict) -> tuple:
    return (atom["chain"], atom["resseq"], atom["resname"], atom["name"])


def compare_anchors(source: list[dict], converted: list[dict]) -> list[dict]:
    """One row per anchor: present before, present after, displacement, AD type.

    Chain is checked because chain A also holds four calciums, 35-50 A away. An
    earlier analysis in this project searched chain A for the MIDAS metal and got
    distances of 40-51 A, which is how easy this mistake is.
    """
    src = {_key(a): a for a in source}
    dst = {_key(a): a for a in converted}
    rows = []
    for key, label in ANCHORS.items():
        s, d = src.get(key), dst.get(key)
        rows.append({
            "label": label,
            "key": key,
            "in_source": s is not None,
            "in_converted": d is not None,
            "displacement": (float(np.linalg.norm(s["xyz"] - d["xyz"]))
                             if s is not None and d is not None else None),
            "adtype": d.get("adtype") if d else None,
            "charge": d.get("charge") if d else None,
        })
    return rows


def convert_with_obabel(pdb_path: str, out_path: str) -> None:
    """obabel -xr: rigid receptor, no torsions. Raises on a nonzero exit.

    `-xr` matters: without it Open Babel tries to find rotatable bonds in the
    protein and emits a flexible-residue PDBQT that Vina-family engines reject.
    """
    if shutil.which("obabel") is None:
        raise RuntimeError("obabel not on PATH - activate the docking/unidock env")
    result = subprocess.run(
        ["obabel", pdb_path, "-opdbqt", "-O", out_path, "-xr"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"obabel failed: {result.stderr.strip()[:400]}")


def _report(pdb_path: str, pdbqt_path: str) -> int:
    source = read_pdb_atoms(pdb_path)
    converted = read_pdbqt_atoms(pdbqt_path)

    print(f"\nsource    {pdb_path}: {len(source)} atoms")
    print(f"converted {pdbqt_path}: {len(converted)} atoms  "
          f"({len(converted) - len(source):+d})")

    for record in ("ATOM", "HETATM"):
        a = sum(1 for x in source if x["record"] == record)
        b = sum(1 for x in converted if x["record"] == record)
        print(f"  {record:7s} {a:6d} -> {b:6d}  ({b - a:+d})")

    src_ca = [a for a in source if a["resname"] == "CA"]
    dst_ca = [a for a in converted if a["resname"] == "CA" or a.get("adtype") == "CA"]
    print(f"\ncalcium ions: {len(src_ca)} in source, {len(dst_ca)} in PDBQT")
    for a in sorted(dst_ca, key=lambda x: (x["chain"], x["resseq"])):
        print(f"  {a['chain']}/{a['resseq']:>5s}  AD type {a.get('adtype','?'):>3s}"
              f"  charge {a.get('charge','?'):>7s}")

    print("\nanchors the section 8.5b filter measures:")
    ok = True
    for row in compare_anchors(source, converted):
        chain, seq, resname, name = row["key"]
        status = "OK"
        if not row["in_source"]:
            status, ok = "ABSENT IN SOURCE", False
        elif not row["in_converted"]:
            status, ok = "*** LOST IN CONVERSION ***", False
        elif row["displacement"] > TOLERANCE:
            status, ok = f"*** MOVED {row['displacement']:.3f} A ***", False
        print(f"  {row['label']:22s} {chain}/{seq:>4s} {resname:>4s} {name:<4s}"
              f"  AD type {str(row['adtype']):>3s}  {status}")

    print()
    if ok:
        print("VERDICT: both anchors survived unmoved. The geometry filter can run")
        print("         against this receptor.")
    else:
        print("VERDICT: FAILED. Do not dock against this receptor - every pose would")
        print("         fail the geometry filter for a reason that looks chemical.")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdb", default="docking/receptor.pdb")
    parser.add_argument("--out", default="docking/receptor.pdbqt")
    parser.add_argument("--verify-only", action="store_true",
                        help="skip conversion; just check an existing PDBQT")
    args = parser.parse_args(argv)

    if not args.verify_only:
        convert_with_obabel(args.pdb, args.out)
        print(f"obabel: {args.pdb} -> {args.out}")
    return _report(args.pdb, args.out)


if __name__ == "__main__":
    sys.exit(main())
