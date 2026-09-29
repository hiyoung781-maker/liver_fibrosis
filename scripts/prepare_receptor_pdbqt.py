"""Convert the docking receptor to PDBQT with Meeko, and prove the MIDAS calcium
survived with non-zero charge and correct typing.

WHY MEEKO REPLACED OPEN BABEL, in the order these failures actually happened on the
cluster:

1. `obabel -xr -h` produced a PDBQT with ALL partial charges zero. autogrid4 refused
   it: "No partial atomic charges were found in the receptor PDBQT file". This never
   mattered in the previous campaign, which used Uni-Dock/Vina - a scoring function
   with no electrostatic term. AutoDock4 has one, so charges are mandatory. The old
   script even printed "charge +0.000" for all six calciums and nobody noticed,
   because nothing downstream checked it.
2. `obabel ... --partialcharge gasteiger` then failed outright: "0 molecules
   converted".
3. Meeko's `mk_prepare_receptor.py` is the purpose-built AutoDock4/Vina receptor
   preparer and is installed on the cluster. It rejected the structure outright
   because 25 residues have truncated side chains - unremarkable for a 2.45 A X-ray
   structure where surface Lys/Arg/Glu/Tyr side chains are unresolved and simply
   absent from the deposit.

THE FIX: truncate to alanine, not delete.

Every one of the 25 residues retains exactly backbone + CB. Compare heavy-atom
counts: LYS has 9 heavy atoms, ARG 11, GLU 9, TYR 12; the observed "heavy_miss" from
Meeko for each matches losing everything past CB. In every case the 5 atoms that
remain are exactly N, CA, C, O, CB - and that set IS alanine's complete heavy-atom
content. So renaming the residue to ALA with those five atoms unmoved makes it match
Meeko's template with zero coordinate change and no atom deleted. Truncating an
unresolved side chain to alanine is standard crystallographic/modeling practice; the
point of this module is to make it loud and auditable rather than silent.

BINDING-SITE RESIDUES ARE NEVER TRUNCATED. The pose gate measures contacts to Ca
B/501, Asn B/224, Leu B/225, Tyr A/178 and Asp A/218. Silently alanine-ing one of
those would turn a measured contact into a missing one and the pipeline would report
a plausible-looking pass rate instead of an error. `truncate_incomplete_residues`
raises instead.

CALCIUM CHARGE. Gasteiger charges handle metals poorly and may leave calcium at
0.000, same failure mode as obabel, just quieter (autogrid4 does not refuse a file
with SOME nonzero charges). Ca2+ carries formal charge +2 and AutoDockTools assigns
the same; a zero charge would mean the MIDAS calcium contributes nothing to the
electrostatic map, which is the entire reason this campaign switched from Vina to
AutoDock4. `fix_calcium_charges` corrects and reports this.

NITROGEN TYPING is still checked for the same reason as before: the previous
Open-Babel-without-hydrogens conversion typed 1,170 of 1,212 nitrogens as acceptors
(should be mostly donors), promoting a fourth-best pose to first. The healthy split
measured with hydrogens present was 101 NA / 1,111 N. Meeko adds hydrogens itself, so
this should hold, but it is exactly the kind of check that "should hold" and then
silently doesn't.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = [
    "ANCHORS",
    "TOLERANCE",
    "STANDARD_HEAVY_ATOMS",
    "BINDING_SITE_RESIDUES",
    "BindingSiteTruncationError",
    "compare_anchors",
    "read_pdb_atoms",
    "read_pdbqt_atoms",
    "truncate_incomplete_residues",
    "build_meeko_command",
    "fix_calcium_charges",
    "parse_pdbqt_lines",
    "rewrite_charge_lines",
    "write_pdbqt",
    "persist_calcium_charge_fixes",
    "verify_receptor",
]

# (chain, resSeq, resName, atomName) -> label. The two atoms section 8.5b measures.
ANCHORS = {
    ("B", "501", "CA", "CA"): "Ca501 (MIDAS)",
    ("B", "224", "ASN", "O"): "Asn224 backbone O",
}

TOLERANCE = 0.01  # A. A rigid conversion must not move anything at all.

# Standard heavy-atom (non-hydrogen) counts for the 20 amino acids, including OXT
# is NOT counted here (OXT is a terminal-residue extra, not part of the standard
# residue template).
STANDARD_HEAVY_ATOMS = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6, "GLN": 9, "GLU": 9,
    "GLY": 4, "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11,
    "PRO": 7, "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}

# Atoms kept when a residue is truncated to ALA: full backbone, CB, and OXT if the
# residue happens to be a C-terminus (rare here but cheap to preserve).
ALANINE_ATOMS = {"N", "CA", "C", "O", "CB", "OXT"}

# The pose gate's binding-site residues. Hard-coded and never truncated: chain B
# 130/132/133/134/186/187/224/225/226/229/259 (contacts to Ca B/501, Asn224, Leu225)
# and chain A 121/178/218 (Tyr178, Asp218). If one of these is incomplete the run
# must stop, not quietly lose a measured contact.
BINDING_SITE_RESIDUES = {
    ("B", 130), ("B", 132), ("B", 133), ("B", 134), ("B", 186), ("B", 187),
    ("B", 224), ("B", 225), ("B", 226), ("B", 229), ("B", 259),
    ("A", 121), ("A", 178), ("A", 218),
}


class BindingSiteTruncationError(RuntimeError):
    """Raised when an incomplete binding-site residue is found. Never truncate it."""


def _parse_lines(lines: list[str], pdbqt: bool) -> list[dict]:
    """Columns 1-54 are identical in PDB and PDBQT; PDBQT adds charge and type.

    Pure function over already-read lines, so both parsing and (later)
    re-parsing a just-rewritten file are the same code path with no risk of
    the on-disk check silently reusing stale in-memory state.
    """
    atoms = []
    for line in lines:
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
            "line": line,
        }
        if pdbqt:
            atom["adtype"] = line[77:79].strip() if len(line) > 77 else ""
            atom["charge"] = line[70:76].strip() if len(line) > 76 else ""
        atoms.append(atom)
    return atoms


def _parse(path: str, pdbqt: bool) -> list[dict]:
    with open(path) as handle:
        return _parse_lines(handle.readlines(), pdbqt=pdbqt)


def read_pdb_atoms(path: str) -> list[dict]:
    return _parse(path, pdbqt=False)


def read_pdbqt_atoms(path: str) -> list[dict]:
    return _parse(path, pdbqt=True)


def parse_pdbqt_lines(lines: list[str]) -> list[dict]:
    """Parse already-in-memory PDBQT lines (e.g. from a just-opened file handle)
    without a second disk read. Used by the charge-fix/rewrite/reread cycle."""
    return _parse_lines(lines, pdbqt=True)


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


# ---------------------------------------------------------------------------
# Step 1-4: detect and truncate incomplete residues, refusing binding-site ones.
# ---------------------------------------------------------------------------

def _residue_groups(pdb_lines: list[str]) -> "dict[tuple, list[tuple[int, str]]]":
    """Group (index, line) pairs by (chain, resseq, resname), preserving order."""
    groups: dict[tuple, list[tuple[int, str]]] = {}
    for i, line in enumerate(pdb_lines):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip()
        resseq = line[22:26].strip()
        groups.setdefault((chain, resseq, resname), []).append((i, line))
    return groups


def truncate_incomplete_residues(
    pdb_lines: list[str],
    standard_heavy_atoms: dict[str, int] | None = None,
    binding_site: set[tuple[str, int]] | None = None,
) -> tuple[list[str], list[dict]]:
    """Rename incomplete standard-amino-acid residues to ALA, keeping only their
    backbone + CB (+ OXT) atoms, coordinates untouched.

    Returns (new_lines, truncations) where truncations is a list of
    {"chain", "resseq", "resname", "heavy_before"} for each residue that was
    renamed, in encounter order.

    Raises BindingSiteTruncationError if an incomplete residue is one of the
    hard-coded pose-gate binding-site residues - those must never be silently
    alanine'd, because the pose gate measures a contact to that exact side chain.
    """
    standard_heavy_atoms = standard_heavy_atoms or STANDARD_HEAVY_ATOMS
    binding_site = binding_site if binding_site is not None else BINDING_SITE_RESIDUES

    groups = _residue_groups(pdb_lines)
    drop_indices: set[int] = set()
    rename_indices: dict[int, str] = {}
    truncations: list[dict] = []

    for (chain, resseq, resname), entries in groups.items():
        if resname not in standard_heavy_atoms:
            continue
        heavy_names = {line[12:16].strip() for _, line in entries
                       if not line[12:16].strip().startswith("H")}
        n_heavy = len(heavy_names)
        if n_heavy >= standard_heavy_atoms[resname]:
            continue  # complete

        try:
            resseq_int = int(resseq)
        except ValueError:
            resseq_int = None

        if resseq_int is not None and (chain, resseq_int) in binding_site:
            raise BindingSiteTruncationError(
                f"{chain}:{resseq} {resname} is a binding-site residue "
                f"(heavy atoms {n_heavy}/{standard_heavy_atoms[resname]}) but is "
                "incomplete. Refusing to truncate it to ALA: the pose gate measures "
                "a contact to this side chain, and silently truncating it would turn "
                "a measured contact into a missing one without any error."
            )

        truncations.append({
            "chain": chain, "resseq": resseq, "resname": resname,
            "heavy_before": n_heavy,
        })
        for i, line in entries:
            atom_name = line[12:16].strip()
            if atom_name in ALANINE_ATOMS:
                rename_indices[i] = "ALA"
            else:
                drop_indices.add(i)

    new_lines = []
    for i, line in enumerate(pdb_lines):
        if i in drop_indices:
            continue
        if i in rename_indices:
            # resName occupies columns 18-20 (0-indexed 17:20).
            line = line[:17] + f"{rename_indices[i]:>3s}" + line[20:]
        new_lines.append(line)

    return new_lines, truncations


# ---------------------------------------------------------------------------
# Step 5: Meeko invocation (command construction is separated from execution so
# it is unit-testable without Meeko installed).
# ---------------------------------------------------------------------------

def build_meeko_command(
    truncated_pdb: str,
    output_basename: str,
    box: dict,
    box_radius: float = 8.0,
) -> list[str]:
    """Construct the mk_prepare_receptor.py argv. Does not execute it.

    `--box_center`/`--box_size` come from dock_unidock.box_from_ligand against
    docking/ligand_ref.sdf, imported rather than recomputed: scripts/prepare_maps.py
    builds the AutoGrid box from that same function, and a divergent box here would
    make every downstream geometric verdict silently answer a different question.
    """
    cx, cy, cz = box["center_x"], box["center_y"], box["center_z"]
    sx, sy, sz = box["size_x"], box["size_y"], box["size_z"]
    return [
        "mk_prepare_receptor.py",
        "--read_pdb", truncated_pdb,
        "--compute_charges", "--charge_model", "gasteiger",
        "-p",
        "--output_basename", output_basename,
        "--box_center", str(cx), str(cy), str(cz),
        "--box_size", str(sx), str(sy), str(sz),
        "--delete_bad_res_from_box_radius", str(box_radius),
    ]


def run_meeko(cmd: list[str]) -> None:
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(
            f"{cmd[0]} not on PATH - activate the meeko/docking env"
        )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"mk_prepare_receptor.py failed: "
                           f"{result.stderr.strip()[:2000]}")


# ---------------------------------------------------------------------------
# Step 6: post-Meeko verification and reporting.
# ---------------------------------------------------------------------------

def _charge_is_zero(atom: dict) -> bool:
    try:
        return float(atom.get("charge") or 0.0) == 0.0
    except ValueError:
        return True  # unparseable charge is treated as "not a valid nonzero charge"


def fix_calcium_charges(pdbqt_atoms: list[dict]) -> list[dict]:
    """Set any zero-charge calcium (adtype 'Ca') to +2.000 in the given in-memory
    atom dicts, returning the list of atoms that were fixed.

    Gasteiger handles metals poorly and may leave calcium at 0.000. Ca2+ carries
    formal charge +2 and AutoDockTools assigns the same; a zero charge here means
    the MIDAS calcium contributes nothing to the electrostatic map, removing the
    entire reason this campaign switched from Vina to AutoDock4.

    IMPORTANT: this only mutates the in-memory dicts. It does not touch any file
    on disk. Fixing the receptor autogrid4 actually reads requires calling
    `rewrite_charge_lines` + `write_pdbqt` on the fixed atoms' original lines and
    then re-reading the file - see `persist_calcium_charge_fixes` and `main()`.
    A prior version of this module fixed the charge only in memory, printed that
    it had, and returned exit 0 while the file on disk still read 0.000 - this
    split exists specifically so that mistake cannot happen silently again.
    """
    fixed = []
    for atom in pdbqt_atoms:
        if atom.get("adtype") != "Ca":
            continue
        if _charge_is_zero(atom):
            atom["charge"] = "+2.000"
            fixed.append(atom)
    return fixed


def rewrite_charge_lines(lines: list[str], fixes: dict[tuple, str]) -> list[str]:
    """Pure function: given the original PDBQT file lines and a map of
    (chain, resseq, resname, name) -> new charge string, return new lines with
    only the charge column (PDBQT columns 71-76, i.e. line[70:76]) replaced on
    matching ATOM/HETATM lines. Every other column, including coordinates, is
    passed through byte-identical.
    """
    new_lines = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            new_lines.append(line)
            continue
        key = (line[21].strip(), line[22:26].strip(), line[17:20].strip(),
               line[12:16].strip())
        if key in fixes:
            charge_field = f"{fixes[key]:>6s}"
            line = line[:70] + charge_field + line[76:]
        new_lines.append(line)
    return new_lines


def write_pdbqt(path: str, lines: list[str]) -> None:
    """The only place this module writes a PDBQT file. Kept as a one-line, fully
    testable function so 'the fix was written to disk' can be asserted directly
    rather than inferred from in-memory state."""
    with open(path, "w") as handle:
        handle.writelines(lines)


def persist_calcium_charge_fixes(path: str) -> tuple[list[dict], list[dict]]:
    """Read `path`, fix any zero-charge calcium, rewrite the file if any fix was
    needed, and RE-READ FROM DISK to confirm. Returns (fixed, disk_atoms) where
    disk_atoms reflects what is actually on disk after this call - never the
    in-memory-only state - so the final verdict is never based on a fix that
    silently failed to land in the file autogrid4 will read.
    """
    with open(path) as handle:
        lines = handle.readlines()
    atoms = parse_pdbqt_lines(lines)
    fixed = fix_calcium_charges(atoms)

    if fixed:
        fixes = {_key(a): a["charge"] for a in fixed}
        new_lines = rewrite_charge_lines(lines, fixes)
        write_pdbqt(path, new_lines)

    with open(path) as handle:
        disk_atoms = parse_pdbqt_lines(handle.readlines())
    return fixed, disk_atoms


def verify_receptor(source: list[dict], converted: list[dict]) -> dict:
    """Run every check step 6 requires and return a report dict. Does not print,
    raise, or fix anything - it only reports on the atoms it is given. Structured
    separately from I/O so it is unit-testable without Meeko or a real PDBQT
    file.

    Critically, this does NOT call fix_calcium_charges: a verdict that silently
    corrects a defect in memory and then reports success is exactly the bug this
    function must not reproduce. Callers that want fixes applied must persist
    them to disk first (persist_calcium_charge_fixes) and pass in atoms re-read
    from that disk state. Any calcium still at zero charge here fails verdict.
    """
    calciums = [a for a in converted if a.get("adtype") == "Ca"]
    wrong_case_ca = [a for a in converted
                     if a["resname"] == "CA" and a.get("adtype") not in ("Ca",)]
    zero_charge_ca = [a for a in calciums if _charge_is_zero(a)]

    acceptors = sum(1 for a in converted if a.get("adtype") == "NA")
    donors = sum(1 for a in converted if a.get("adtype") == "N")
    total_n = acceptors + donors
    acceptor_fraction = (acceptors / total_n) if total_n else 0.0

    anchor_rows = compare_anchors(source, converted)
    anchors_ok = all(
        row["in_source"] and row["in_converted"]
        and row["displacement"] is not None and row["displacement"] <= TOLERANCE
        for row in anchor_rows
    )

    ok = (
        len(calciums) == 6
        and not wrong_case_ca
        and not zero_charge_ca
        and anchors_ok
        and total_n > 0
        and acceptor_fraction <= 0.5
    )

    return {
        "calciums": calciums,
        "calcium_count": len(calciums),
        "zero_charge_calciums": zero_charge_ca,
        "wrong_case_calciums": wrong_case_ca,
        "acceptors": acceptors,
        "donors": donors,
        "total_n": total_n,
        "acceptor_fraction": acceptor_fraction,
        "anchor_rows": anchor_rows,
        "anchors_ok": anchors_ok,
        "ok": ok,
    }


def _print_report(report: dict) -> None:
    print(f"\ncalcium ions: {report['calcium_count']} found (need 6)")
    for a in sorted(report["calciums"], key=lambda x: (x["chain"], x["resseq"])):
        print(f"  {a['chain']}/{a['resseq']:>5s}  AD type {a.get('adtype','?'):>3s}"
              f"  charge {a.get('charge','?'):>7s}")
    if report["wrong_case_calciums"]:
        print("  *** calcium(s) not typed 'Ca' (case matters, autogrid4 is case "
              "sensitive) ***")
    if report["zero_charge_calciums"]:
        print(f"  *** {len(report['zero_charge_calciums'])} calcium charge(s) are "
              "STILL 0.000 ON DISK. Ca2+ is formal charge +2, AutoDockTools assigns "
              "the same, and a zero charge means the MIDAS calcium contributes "
              "nothing to the electrostatic map - this receptor must not be used "
              "for AutoDock4 as-is. ***")

    print(f"\nnitrogen typing: {report['total_n']} nitrogens -> "
          f"{report['acceptors']} NA (acceptor), {report['donors']} N (donor)")
    if report["total_n"] and report["acceptor_fraction"] > 0.5:
        print(f"  *** {report['acceptors']}/{report['total_n']} = "
              f"{report['acceptor_fraction']:.0%} typed as ACCEPTORS - "
              "the broken split. Healthy is ~101 NA / ~1111 N.")

    print("\nanchors the section 8.5b filter measures:")
    for row in report["anchor_rows"]:
        chain, seq, resname, name = row["key"]
        status = "OK"
        if not row["in_source"]:
            status = "ABSENT IN SOURCE"
        elif not row["in_converted"]:
            status = "*** LOST IN CONVERSION ***"
        elif row["displacement"] > TOLERANCE:
            status = f"*** MOVED {row['displacement']:.3f} A ***"
        print(f"  {row['label']:22s} {chain}/{seq:>4s} {resname:>4s} {name:<4s}"
              f"  AD type {str(row['adtype']):>3s}  {status}")

    print()
    if report["ok"]:
        print("VERDICT: receptor PDBQT passes all checks.")
    else:
        print("VERDICT: FAILED. Do not dock against this receptor.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pdb", default="docking/receptor.pdb")
    parser.add_argument("--out", default="docking/receptor.pdbqt")
    parser.add_argument("--truncated-pdb", default="docking/receptor_truncated.pdb",
                        help="intermediate PDB with unresolved side chains "
                             "renamed to ALA")
    parser.add_argument("--ligand-sdf", default="docking/ligand_ref.sdf")
    parser.add_argument("--box-radius", type=float, default=8.0,
                        help="--delete_bad_res_from_box_radius")
    parser.add_argument("--verify-only", action="store_true",
                        help="skip conversion; just check an existing PDBQT")
    args = parser.parse_args(argv)

    if not args.verify_only:
        with open(args.pdb) as handle:
            pdb_lines = handle.readlines()

        new_lines, truncations = truncate_incomplete_residues(pdb_lines)
        print(f"truncation: {len(truncations)} incomplete residue(s) renamed to ALA")
        for t in truncations:
            print(f"  {t['chain']}:{t['resseq']} {t['resname']} -> ALA")

        with open(args.truncated_pdb, "w") as handle:
            handle.writelines(new_lines)

        from dock_unidock import box_from_ligand
        box = box_from_ligand(args.ligand_sdf)

        out_basename = str(Path(args.out).with_suffix(""))
        cmd = build_meeko_command(args.truncated_pdb, out_basename, box,
                                  box_radius=args.box_radius)
        print("\n" + " ".join(cmd))
        run_meeko(cmd)
        print(f"meeko: {args.truncated_pdb} -> {args.out}")

    fixed, converted = persist_calcium_charge_fixes(args.out)
    if fixed:
        print(f"\ncalcium charge fix: {len(fixed)} zero-charge calcium(s) found, "
              f"corrected to +2.000, and rewritten to {args.out} "
              "(re-read from disk to confirm).")

    source = read_pdb_atoms(args.pdb)
    report = verify_receptor(source, converted)
    _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
