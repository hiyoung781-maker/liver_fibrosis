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
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

__all__ = ["LIGAND_RESNAME", "build_complex", "build_many"]

# Three characters is a hard PDB column limit. The deposited 8W30 ligand
# has a five-character CCD code (A1AFA); PLIP truncates that itself to a
# three-character hetid, and a residue name that overflows the PDB field
# has already zeroed every affinity once in this project. "LIG" survives.
LIGAND_RESNAME = "LIG"


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
        out.write_text("\n".join(receptor + lig_lines + ["END"]) + "\n")
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
