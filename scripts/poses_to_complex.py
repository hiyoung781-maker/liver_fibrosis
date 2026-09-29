"""Convert a docked pose PDBQT into a hydrogen-bearing protein-ligand complex PDB (spec Sec 8.0).

PLIP computes donor-H...acceptor angles, so the complex it reads must carry
explicit hydrogens on both the ligand and the receptor. PDBQT ligand files
merge nonpolar hydrogens into their heavy atoms, so they cannot be fed to
PLIP directly -- Open Babel is used to rebuild explicit hydrogens on the
ligand before it is stitched onto the (already-protonated) receptor PDB.

A single malformed pose must never abort a batch conversion: roughly 85,000
poses will pass through this converter across two docking arms, and one bad
file aborting a cluster run would waste hours. build_complex therefore
catches its failure modes and returns None instead of raising; build_many
collects the successes and records the failures with a reason.

THE LINK RECORD IS NOT OPTIONAL. PLIP merges a ligand and a neighbouring
metal into one composite binding site only when the input PDB carries a LINK
record joining them: it parses LINK lines in structure/preparation.py (line
129) and clusters ligand residues on exactly those links in identify_kmers.
A complex assembled from coordinate records alone therefore puts the calcium
in a binding site of its own -- and a ligand atom that belongs to a different
site is never a candidate target for that site's metal, so the MIDAS
coordination is reported NOWHERE, not in either site.

Measured on the deposited 8W30 ligand rebuilt through this converter:

  without LINK   LIG site: 0 metal_complexes
                 Ca B501 site: coordination 3 (Ser132 2.45, Ser134 2.50,
                 Glu229 2.39) -- the ligand contact simply absent
  with LINK      one site, longname LIG-CA: coordination 4, ligand contact
                 2.62 A, the three protein oxygens unchanged

2.62 A and coordination 4 are the deposited structure's own values, so the
LINK restores the crystal measurement rather than inventing one. Without this
record every converter-built pose fails the metal criterion of the section
8.5b gate for a reason that looks chemical and is purely a file-format
artefact.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Optional

__all__ = ["LIGAND_RESNAME", "MIDAS", "MIDAS_LINK_CUTOFF", "build_complex",
           "build_many", "midas_link_record"]

# Three characters is a hard PDB column limit. The deposited 8W30 ligand
# has a five-character CCD code (A1AFA); PLIP truncates that itself to a
# three-character hetid, and a residue name that overflows the PDB field
# has already zeroed every affinity once in this project. "LIG" survives.
LIGAND_RESNAME = "LIG"

# The MIDAS calcium: chain B, residue 501. Ca B502 is 6.44 A from the crystal
# ligand and does not participate; chain A's four calciums are 35-50 A away.
# Naming it explicitly keeps a receptor edit from silently linking another
# metal, which would fail every pose for a reason that looks chemical.
MIDAS = ("B", 501)

# Trigger window for emitting the LINK, NOT an interaction criterion. PLIP's
# own METAL_DIST_MAX is 3.0 A, so a window above that can never manufacture a
# coordination PLIP would not independently confirm -- the LINK only makes the
# ligand atom visible to PLIP as a candidate target. Below the window no LINK
# is written, because linking a pose that sits 124 A away (the pose_ok
# fixture) would fabricate a composite ligand out of two unrelated molecules.
MIDAS_LINK_CUTOFF = 4.0

# O, N and S are the metal-complex target elements PLIP recognises.
_METAL_TARGET_ELEMENTS = {"O", "N", "S"}


def _xyz(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def _element(line: str) -> str:
    """Element symbol, from columns 77-78 or, failing that, the atom name."""
    symbol = line[76:78].strip()
    if symbol:
        return symbol.capitalize()
    return line[12:16].strip()[:1].upper()


def midas_link_record(lig_lines: list, receptor_lines: list) -> Optional[str]:
    """A PDB LINK joining the ligand's nearest metal-target atom to the MIDAS.

    Returns None when the receptor has no Ca at MIDAS, when the ligand has no
    O/N/S atom, or when the closest such atom is farther than
    MIDAS_LINK_CUTOFF -- in every one of those cases there is no composite
    ligand to declare, and PLIP's verdict on the pose stands unchanged.

    The column layout below is the one PLIP's get_linkage reads:
    name1 [12:16], resname1 [17:20], chain1 [21], resseq1 [22:26];
    name2 [42:46], resname2 [47:50], chain2 [51], resseq2 [52:56].
    """
    metal = next(
        (l for l in receptor_lines
         if l.startswith(("ATOM", "HETATM"))
         and l[17:20].strip().upper() == "CA"
         and l[21] == MIDAS[0]
         and l[22:26].strip() == str(MIDAS[1])),
        None,
    )
    if metal is None:
        return None

    metal_xyz = _xyz(metal)
    candidates = [l for l in lig_lines if _element(l) in _METAL_TARGET_ELEMENTS]
    if not candidates:
        return None

    dist, nearest = min(
        (math.dist(_xyz(l), metal_xyz), l) for l in candidates
    )
    if dist > MIDAS_LINK_CUTOFF:
        return None

    return (
        "LINK        "
        f"{nearest[12:16].strip():>4} {LIGAND_RESNAME:<3} "
        f"{nearest[21]}{int(nearest[22:26]):>4}"
        "                "
        f"{metal[12:16].strip():>4} {metal[17:20].strip():<3} "
        f"{metal[21]}{int(metal[22:26]):>4}"
        f"  1555   1555 {dist:5.2f}"
    )


def _ligand_pdb(pose_pdbqt: str, out: Path) -> str:
    """Add explicit hydrogens to the docked ligand pose and convert to PDB.

    -h adds hydrogens appropriate for the existing bonding/charges in the
    PDBQT; it does not take a pH argument (unlike -p, which is avoided here
    and everywhere else in this project because it has been observed to
    silently drop metal ions from receptor structures).
    """
    subprocess.run(
        ["obabel", pose_pdbqt, "-opdb", "-O", str(out), "-h"],
        check=True, capture_output=True, timeout=120,
    )
    return str(out)


def build_complex(pose_pdbqt: str, receptor_pdb: str, out) -> Optional[str]:
    """Build a single receptor+ligand complex PDB from a docked pose.

    Returns the path to the written complex on success. Returns None on any
    failure (obabel error/timeout, missing/unreadable inputs, a ligand pose
    that converts to zero atoms) instead of raising, so a caller iterating
    over many poses can skip a bad one without losing the rest of the batch.
    """
    out = Path(out)
    try:
        ligand = _ligand_pdb(pose_pdbqt, out.with_suffix(".lig.pdb"))
        lig_lines = []
        for line in Path(ligand).read_text().splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            # Rewrite as HETATM/LIG so PLIP identifies this as the ligand
            # and distinguishes it from the receptor's own ATOM records.
            lig_lines.append("HETATM" + line[6:17] + LIGAND_RESNAME + line[20:])
        if not lig_lines:
            # obabel can exit 0 while converting zero atoms (e.g. an
            # unparsable PDBQT); that is a failure for our purposes.
            return None
        receptor = [
            l for l in Path(receptor_pdb).read_text().splitlines()
            if l.startswith(("ATOM", "HETATM"))
        ]
        if not receptor:
            return None
        # The LINK must precede the coordinate records: PLIP reads the whole
        # file either way, but a LINK after the ATOM block is not valid PDB
        # and other tools in this pipeline do read the complex.
        link = midas_link_record(lig_lines, receptor)
        header = [link] if link else []
        out.write_text("\n".join(header + receptor + lig_lines + ["END"]) + "\n")
        return str(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            OSError, ValueError, IndexError):
        return None


def build_many(poses: list, receptor_pdb: str, out_dir) -> dict:
    """Build complexes for many poses, isolating any single-pose failure.

    Returns {"built": [paths...], "failed": [(pose, reason), ...]}. Every
    pose in `poses` is attempted regardless of earlier failures.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = []
    failed = []
    for pose in poses:
        target = out_dir / (Path(pose).stem + ".complex.pdb")
        result = build_complex(pose, receptor_pdb, target)
        if result is None:
            failed.append((pose, "complex build failed"))
        else:
            built.append(result)
    return {"built": built, "failed": failed}
