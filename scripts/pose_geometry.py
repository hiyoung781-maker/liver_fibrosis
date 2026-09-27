"""Section 8.5b's geometric filter: does a docked pose make the two anchor contacts?

Docking score is NOT the criterion. Vina-family scoring functions have no metal
coordination term, so the single most important interaction - carboxylate to the
MIDAS Ca2+ - is invisible to the scorer (section 8.5). The design is therefore to
generate many poses and select on geometry, using the score only to break ties.

Two criteria, with thresholds derived in blueprint section 8.5:
  1. a carboxylate oxygen within CA_CUTOFF of Ca501 (chain B). 3.2 A sits inside
     the gap between this structure's first coordination shell (which ends at the
     ligand's own 2.62 A) and the next atom in (Asp259 OD1 at 3.30 A). Monodentate:
     the crystal ligand's second carboxylate oxygen is 4.54 A, so requiring both
     would reject the crystal structure itself.
  2. an H-bond donor within DONOR_CUTOFF of beta1-Asn224 backbone O. Measured
     2.63 A in the crystal; the partners of that same oxygen all sit in a 2.2-2.7 A
     band, and 3.5 A is the standard permissive N...O cutoff, 0.85 A above it.

RMSD, used only for the redocking control, goes through rdMolAlign.CalcRMS. A naive
index-wise RMSD reports 6.13 A for a pose whose true RMSD is 0.63 A, because
smina/obabel reorder atoms and the ligand has symmetric rings. That error turns a
passed validation into an apparent failure, so it is not available here by accident.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolAlign

__all__ = [
    "CA_CUTOFF",
    "DONOR_CUTOFF",
    "anchor_atoms",
    "crystal_rmsd",
    "measure_pose",
    "passes",
]

RDLogger.DisableLog("rdApp.*")

CA_CUTOFF = 3.2
DONOR_CUTOFF = 3.5

# Ca501 of chain B is the MIDAS calcium: 2.62 A from the crystal ligand. Ca502 is
# 6.44 A and does not participate; chain A's four calciums are 35-50 A away. Naming
# the residue explicitly keeps a future receptor edit from silently selecting
# another metal and failing every pose for a reason that looks chemical.
MIDAS = ("B", "501")
ASN224 = ("B", "224")

_CARBOXYLATE_O = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
# Donor heavy atoms: N or O carrying at least one hydrogen. The poses come from
# smina with hydrogens present, but total-H counting also works if they are implicit.
_DONOR = Chem.MolFromSmarts("[#7,#8;!H0]")


def anchor_atoms(receptor_path: str) -> dict[str, np.ndarray]:
    """Coordinates of Ca501 and Asn224 backbone O. Raises if either is absent."""
    found: dict[str, np.ndarray] = {}
    with open(receptor_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            res = line[17:20].strip()
            chain = line[21]
            seq = line[22:26].strip()
            name = line[12:16].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
            if res == "CA" and (chain, seq) == MIDAS:
                found["ca501"] = xyz
            elif res == "ASN" and (chain, seq) == ASN224 and name == "O":
                found["asn224_o"] = xyz
    missing = {"ca501", "asn224_o"} - set(found)
    if missing:
        raise ValueError(
            f"{receptor_path}: anchor atoms not found: {sorted(missing)}. "
            f"Expected Ca {MIDAS[0]}/{MIDAS[1]} and ASN {ASN224[0]}/{ASN224[1]} O - "
            "every pose would fail the filter for a non-chemical reason."
        )
    return found


def measure_pose(mol: Chem.Mol, anchors: dict[str, np.ndarray]) -> dict:
    """Both contact distances for one pose. inf when the pose has no such atom.

    ca_dist is the NEAREST carboxylate oxygen and ca_dist_far the other one, so a
    caller can see whether the contact is mono- or bidentate without re-measuring.
    """
    positions = mol.GetConformer().GetPositions()

    oxygens = sorted({idx for match in mol.GetSubstructMatches(_CARBOXYLATE_O)
                      for idx in match[1:]})
    ca_dists = sorted(float(np.linalg.norm(positions[i] - anchors["ca501"]))
                      for i in oxygens) or [float("inf")]

    donors = [idx for (idx,) in mol.GetSubstructMatches(_DONOR)]
    donor_dists = sorted(
        float(np.linalg.norm(positions[i] - anchors["asn224_o"])) for i in donors
    ) or [float("inf")]

    return {
        "ca_dist": ca_dists[0],
        "ca_dist_far": ca_dists[1] if len(ca_dists) > 1 else float("inf"),
        "donor_dist": donor_dists[0],
        "n_carboxylate_o": len(oxygens),
        "n_donors": len(donors),
    }


def passes(measurement: dict) -> bool:
    """Section 8.5b: both contacts, or the pose is rejected."""
    return (measurement["ca_dist"] <= CA_CUTOFF
            and measurement["donor_dist"] <= DONOR_CUTOFF)


def crystal_rmsd(pose: Chem.Mol, reference: Chem.Mol) -> float:
    """Symmetry- and mapping-aware RMSD, without superposition.

    CalcRMS, never a coordinate-order subtraction - see this module's docstring.
    """
    with rdBase.BlockLogs():
        return float(rdMolAlign.CalcRMS(Chem.RemoveHs(pose),
                                        Chem.RemoveHs(reference)))


def _score(mol: Chem.Mol) -> float:
    """Docking energy, kcal/mol. nan when the pose carries none.

    `affinity` is what scripts/poses_to_sdf.py writes from Uni-Dock's
    `REMARK VINA RESULT`; the smina tags come first because a pose SDF produced
    directly by smina/gnina uses those instead.
    """
    for tag in ("minimizedAffinity", "CNNaffinity", "affinity"):
        if mol.HasProp(tag):
            try:
                return float(mol.GetProp(tag))
            except ValueError:
                pass
    return float("nan")


def filter_poses(poses_path: str, receptor_path: str, out_csv: str,
                 out_sdf: str | None = None) -> dict[str, int]:
    """Measure every pose in an SDF, write a row each, and keep passing poses.

    The SDF holds all poses of all ligands in one file, as smina writes it; the
    ligand name comes from the molecule title, which prepare_ligands.py sets to the
    survivor label. Pose index restarts per ligand.
    """
    anchors = anchor_atoms(receptor_path)
    writer = Chem.SDWriter(out_sdf) if out_sdf else None
    counts = {"poses": 0, "passing": 0, "ligands": 0, "ligands_passing": 0}
    seen: set[str] = set()
    passing_ligands: set[str] = set()

    with open(out_csv, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["label", "pose", "affinity", "ca_dist", "ca_dist_far",
                      "donor_dist", "n_carboxylate_o", "n_donors", "passes"])
        pose_index: dict[str, int] = {}
        for mol in Chem.SDMolSupplier(poses_path, removeHs=False):
            if mol is None:
                continue
            counts["poses"] += 1
            label = mol.GetProp("_Name") or "unnamed"
            if label not in seen:
                seen.add(label)
                counts["ligands"] += 1
            pose_index[label] = pose_index.get(label, 0) + 1
            m = measure_pose(mol, anchors)
            ok = passes(m)
            if ok:
                counts["passing"] += 1
                passing_ligands.add(label)
                if writer is not None:
                    mol.SetProp("ca_dist", f"{m['ca_dist']:.3f}")
                    mol.SetProp("donor_dist", f"{m['donor_dist']:.3f}")
                    writer.write(mol)
            out.writerow([label, pose_index[label], f"{_score(mol):.3f}",
                          f"{m['ca_dist']:.3f}", f"{m['ca_dist_far']:.3f}",
                          f"{m['donor_dist']:.3f}", m["n_carboxylate_o"],
                          m["n_donors"], int(ok)])
    if writer is not None:
        writer.close()
    counts["ligands_passing"] = len(passing_ligands)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", required=True, help="smina output SDF")
    parser.add_argument("--receptor", default="docking/receptor.pdb")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-sdf", default=None,
                        help="passing poses only; omit to skip")
    args = parser.parse_args(argv)

    counts = filter_poses(args.poses, args.receptor, args.out_csv, args.out_sdf)
    for key, value in counts.items():
        print(f"  {key:16s} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
