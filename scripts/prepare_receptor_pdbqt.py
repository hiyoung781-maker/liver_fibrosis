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
  6. The nitrogen donor/acceptor split, because checks 1-5 all PASSED on a receptor
     that was energetically wrong.

MEASURED FAILURE AND ITS FIX, kept here because checks 1-5 did not catch it.

`obabel -xr` on the hydrogen-free receptor.pdb typed 1,170 of its 1,212 nitrogens as
`NA`, AutoDock's H-bond ACCEPTOR. In a protein without hydrogens Open Babel cannot
tell a backbone amide NH (a donor) from an acceptor, so it defaulted nearly all of
them to acceptor, rewriting the H-bond landscape of the whole protein. Cost: the same
smina, ligand, box and settings scored -6.9 against receptor.pdb and -6.4 against that
receptor.pdbqt.

The fix is `-h`: add hydrogens before writing the PDBQT, so donor and acceptor become
decidable. Typing flips from 1,170 NA / 42 N to 101 NA / 1,111 N, with 1,592 HD polar
hydrogens, and smina scores -6.9 again - identical to the PDB.

What that mis-typing did to the section 8.5(a) control is worth stating exactly,
because it is subtle. On the corrected receptor smina's pose ranking is
  pose 1  -6.909  Ca 2.740  Asn224 2.895  PASSES
  pose 4  -6.670  Ca 2.357  Asn224 4.832  fails
and pose 4 is precisely what Uni-Dock had been returning as its best. The engine was
not finding a wrong pose; the mis-typed receptor was promoting the fourth-best pose to
first. smina and Uni-Dock agree to within 0.1 kcal/mol on the same receptor file.

Feeding Uni-Dock the raw PDB is NOT the fix - it scored -5.787 with the carboxylate
4.59 A off the calcium, worse than either PDBQT. The hydrogenated PDBQT is the input
to use.
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


def convert_with_obabel(pdb_path: str, out_path: str,
                        add_hydrogens: bool = True) -> None:
    """obabel -xr -h: rigid receptor, hydrogens added. Raises on a nonzero exit.

    `-xr` matters: without it Open Babel looks for rotatable bonds in the protein and
    emits a flexible-residue PDBQT that Vina-family engines reject.

    `-h` matters more, and is the whole subject of this module's docstring: without
    hydrogens the donor/acceptor typing is undecidable and Open Babel calls 97% of the
    nitrogens acceptors, which costs 0.5 kcal/mol and reorders the poses. It is a
    parameter rather than a constant only so a test can reproduce the broken receptor.
    """
    if shutil.which("obabel") is None:
        raise RuntimeError("obabel not on PATH - activate the docking/unidock env")
    cmd = ["obabel", pdb_path, "-opdbqt", "-O", out_path, "-xr"]
    if add_hydrogens:
        cmd.append("-h")
    result = subprocess.run(cmd, capture_output=True, text=True)
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

    src_n = [a for a in source if a["name"].startswith("N") or a["name"] == "N"]
    acceptors = [a for a in converted if a.get("adtype") == "NA"]
    donors = [a for a in converted if a.get("adtype") == "N"]
    total_n = len(acceptors) + len(donors)
    print(f"\nnitrogen typing: {total_n} nitrogens -> "
          f"{len(acceptors)} NA (acceptor), {len(donors)} N (non-acceptor)")
    if total_n and len(acceptors) / total_n > 0.5:
        print(f"  *** {len(acceptors)}/{total_n} = {len(acceptors)/total_n:.0%} typed as")
        print("      H-bond ACCEPTORS. In a protein, most nitrogens are backbone amide")
        print("      NH - DONORS. Open Babel cannot tell them apart without hydrogens,")
        print("      so a hydrogen-free input gets this wrong, and it is worth")
        print("      0.5 kcal/mol (smina scored -6.9 on the PDB, -6.4 on the PDBQT).")
        print("      Either add hydrogens before converting, or skip the conversion:")
        print("      Uni-Dock takes --receptor as PDB.")

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
    if total_n and len(acceptors) / total_n > 0.5:
        print("VERDICT: anchors are fine but the nitrogen typing above is not. This")
        print("         receptor scores 0.5 kcal/mol worse than the PDB it came from.")
        print("         Prefer --receptor <the PDB> over this file.")
    elif ok:
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
    parser.add_argument("--no-hydrogens", action="store_true",
                        help="omit obabel -h. Reproduces the broken receptor that "
                             "mistyped 97%% of nitrogens as acceptors; for tests only")
    args = parser.parse_args(argv)

    if not args.verify_only:
        convert_with_obabel(args.pdb, args.out,
                            add_hydrogens=not args.no_hydrogens)
        print(f"obabel: {args.pdb} -> {args.out}"
              f"{'' if args.no_hydrogens else ' (hydrogens added)'}")
    return _report(args.pdb, args.out)


if __name__ == "__main__":
    sys.exit(main())
