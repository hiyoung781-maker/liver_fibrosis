"""Section 8.4's three structural observations, measured on section 8.5b's passing poses.

These are REPORTED, NOT GATED, and they are not evidence of selectivity. Section 8.4
demoted selectivity from a claim to an observation for reasons worth restating here,
because a table of numbers invites being read as proof:

  - alphaVbeta1 selectivity is set by BOTH subunits [114], and this project analyses
    only the beta side. The same review names alpha5's Trp157/Gln221/Ser224 as the
    alpha-side candidates; they are untouched here.
  - No selectivity-conferring contact has ever been observed crystallographically for
    this target. There was no alphaVbeta1 structure before 2020, and 8W30's own ligand
    is pIC50 5.30 and sits 7.00 A from Asp218 - it does not make the contact at all.
  - Measured cross-isoform IC50 is the only basis for a selectivity claim, and it is
    future work.

What is measured, per pose:

(i)  beta1-Leu225 SIDE-CHAIN contact - a ligand hydrophobic atom within 4.5 A of
     CB/CG/CD1/CD2. Aligning ITGB1 and ITGB3 betaI domains, Leu225 is the ONLY one of
     the five hydrophobic-pocket residues Sabat credits that differs in beta3, where it
     is Arg - a hydrophobic wall replaced by a charged side chain. Tyr133, Pro186,
     Cys187, Asn224 and Asp226 are all conserved. The side chain is the point:
     GLPG0187 and CWHM-12 reach Leu225's BACKBONE with the same 3-aminopropionyl
     scaffold and are still pan.

(ii) alphaV-Asp218 contact CHARACTER - within 4.0 A of OD1/OD2, separately: a neutral
     H-bond donor, or a basic Arg-mimic nitrogen. The distinction is the whole point.
     A basic head's salt bridge is associated with pan-alphaV activity - GSK3008348's
     1,8-naphthyridine salt-bridges Asp218 [114], and Sabat's THN introduction produced
     pan-alphaV [111] - while a neutral monodentate H-bond is not. The Arg-mimic
     definition is imported from scripts/curate_actives.py rather than restated, so this
     measurement and section 4's Arg-mimic percentage cannot drift apart.

(iii) alphaV-Tyr178 pi-stack - nearest ring-atom contact within 4.5 A, with an
     interplanar angle under 30 degrees (parallel) or over 60 (T-shaped). Centroid
     distance is reported but is NOT the criterion, which is a correction: section
     8.4 originally specified "centroid within 5.5 A", and the crystal structure
     fails it. 8W30 measures nearest atom 3.70 A, centroid 6.21 A, angle 74.5
     degrees - so the only crystallographically observed instance of this
     interaction would have been rejected by the test written to detect it.

     The cause is geometric. A T-shaped stack holds the rings perpendicular, which
     pushes their centroids apart while the closest atoms stay in contact; a
     centroid cutoff silently assumes parallel stacking. The crystal is T-shaped at
     74.5 degrees. So the distance criterion moves to the nearest ring atom, where
     the crystal's 3.70 A sits comfortably inside 4.5 A - the same cutoff criterion
     (i) uses for hydrophobic contact, and the usual permissive bound for an
     aromatic contact whose van der Waals range is 3.4-3.7 A.

     This follows how section 8.5's thresholds were derived: let the structure
     define the boundary rather than importing a convention.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger

__all__ = [
    "ASP218_CUTOFF",
    "FEATURES",
    "LEU225_CUTOFF",
    "TYR178_ATOM_CUTOFF",
    "TYR178_CENTROID_REFERENCE",
    "asp218_character",
    "leu225_contact",
    "observe",
    "receptor_features",
    "tyr178_stack",
]

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (chain, resSeq, atom names). Chain matters: Leu225 and Asn224 are beta1 (chain B)
# while Asp218 and Tyr178 are alphaV (chain A). Mixing them up is silent - an earlier
# analysis in this project searched chain A for the MIDAS calcium and got 40-51 A.
FEATURES = {
    # Side chain only. The backbone is deliberately excluded: pan-alphaV compounds
    # reach Leu225's backbone, so including N/CA/C/O would measure the wrong thing.
    "leu225": ("B", "225", ("CB", "CG", "CD1", "CD2")),
    "asp218": ("A", "218", ("OD1", "OD2")),
    "tyr178": ("A", "178", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")),
}

LEU225_CUTOFF = 4.5
ASP218_CUTOFF = 4.0
# The criterion. 8W30's nearest ring atom is 3.70 A; 4.5 A is the same bound criterion
# (i) uses and the usual permissive limit for an aromatic contact (vdW 3.4-3.7 A).
TYR178_ATOM_CUTOFF = 4.5
# Reported, not applied - see the module docstring. Section 8.4's original "centroid
# within 5.5 A" rejects the crystal structure, whose centroid distance is 6.21 A
# because the stack is T-shaped.
TYR178_CENTROID_REFERENCE = 6.21
PARALLEL_MAX = 30.0
T_SHAPED_MIN = 60.0

# Carbon not bonded to N or O, plus halogens - AutoDock's hydrophobic convention.
_HYDROPHOBIC = Chem.MolFromSmarts("[$([#6;!$([#6]~[#7,#8])]),$([F,Cl,Br,I])]")
# Any N or O carrying at least one hydrogen, counted implicitly so a crystal ligand
# without explicit H still measures. Excludes formally charged N, which criterion
# (ii) counts separately as the basic case.
_NEUTRAL_DONOR = Chem.MolFromSmarts("[$([#7,#8;!H0;!$([#7+])])]")


def _arg_mimic_patterns():
    """Imported from curate_actives so section 4 and section 8.4 share one definition."""
    from curate_actives import ARG_MIMIC_HEADS

    return {name: Chem.MolFromSmarts(s) for name, s in ARG_MIMIC_HEADS.items()}


def receptor_features(receptor_path: str) -> dict[str, np.ndarray]:
    """Coordinates of the three feature atom sets. Raises if any is incomplete."""
    wanted = {(c, s, n): key for key, (c, s, names) in FEATURES.items()
              for n in names}
    found: dict[str, list[np.ndarray]] = {key: [] for key in FEATURES}
    with open(receptor_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            key = wanted.get((line[21].strip(), line[22:26].strip(),
                              line[12:16].strip()))
            if key is not None:
                found[key].append(np.array([float(line[30:38]),
                                            float(line[38:46]),
                                            float(line[46:54])]))
    out = {}
    for key, (chain, seq, names) in FEATURES.items():
        if len(found[key]) != len(names):
            raise ValueError(
                f"{receptor_path}: {key} ({chain}/{seq}) gave "
                f"{len(found[key])} of {len(names)} atoms {names}. Every pose would "
                "report no contact, which reads as chemistry rather than a bad receptor."
            )
        out[key] = np.array(found[key])
    return out


def _min_distance(positions: np.ndarray, targets: np.ndarray) -> float:
    if len(positions) == 0:
        return float("inf")
    return float(np.linalg.norm(positions[:, None, :] - targets[None, :, :],
                                axis=2).min())


def leu225_contact(mol: Chem.Mol, features: dict) -> dict:
    """(i) Nearest ligand hydrophobic atom to Leu225's side-chain carbons."""
    xyz = mol.GetConformer().GetPositions()
    idx = [i for (i,) in mol.GetSubstructMatches(_HYDROPHOBIC)]
    distance = _min_distance(xyz[idx] if idx else np.empty((0, 3)),
                             features["leu225"])
    return {
        "leu225_min_dist": distance,
        "leu225_contact": int(distance <= LEU225_CUTOFF),
        "n_hydrophobic": len(idx),
    }


def asp218_character(mol: Chem.Mol, features: dict) -> dict:
    """(ii) Nearest neutral donor and nearest Arg-mimic nitrogen to Asp218's oxygens.

    Both are reported. A basic head within range is the finding literature ties to
    pan-alphaV activity; a neutral donor within range is not.
    """
    xyz = mol.GetConformer().GetPositions()
    donors = [i for (i,) in mol.GetSubstructMatches(_NEUTRAL_DONOR)]
    donor_dist = _min_distance(xyz[donors] if donors else np.empty((0, 3)),
                               features["asp218"])

    basic: set[int] = set()
    hits = []
    for name, pattern in _arg_mimic_patterns().items():
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            hits.append(name)
        for match in matches:
            basic.update(i for i in match
                         if mol.GetAtomWithIdx(i).GetSymbol() == "N")
    basic_dist = _min_distance(xyz[sorted(basic)] if basic else np.empty((0, 3)),
                               features["asp218"])
    # Any-atom distance too, because that is the quantity section 1's pocket table
    # records as "8W30 hit does not engage (7.0 A)" - it is an aromatic CH, not a
    # donor. Reporting only the donor distance made that documented number
    # irreproducible from this script (it measures 12.35 A for the crystal ligand).
    any_dist = _min_distance(xyz, features["asp218"])
    return {
        "asp218_any_atom_dist": any_dist,
        "asp218_donor_dist": donor_dist,
        "asp218_neutral_donor": int(donor_dist <= ASP218_CUTOFF),
        "asp218_basic_dist": basic_dist,
        "asp218_basic_contact": int(basic_dist <= ASP218_CUTOFF),
        "arg_mimic_heads": "|".join(hits),
    }


def _ring_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal of a ring, by SVD on the centred coordinates."""
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    return centroid, vh[2]


def tyr178_stack(mol: Chem.Mol, features: dict) -> dict:
    """(iii) Best ligand aromatic ring against Tyr178's ring: centroid distance, angle.

    "Best" is the smallest centroid distance, reported whether or not it satisfies the
    criterion, so the distribution can be read rather than only the pass rate.
    """
    xyz = mol.GetConformer().GetPositions()
    tyr_centroid, tyr_normal = _ring_plane(features["tyr178"])

    best = None
    rings = [r for r in mol.GetRingInfo().AtomRings()
             if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    for ring in rings:
        centroid, normal = _ring_plane(xyz[list(ring)])
        distance = float(np.linalg.norm(centroid - tyr_centroid))
        cosine = abs(float(np.dot(normal, tyr_normal)))
        angle = float(np.degrees(np.arccos(min(1.0, cosine))))
        nearest = _min_distance(xyz[list(ring)], features["tyr178"])
        # Ranked by the criterion's own measure. Ranking by centroid would pick a
        # different ring than the one the test judges.
        if best is None or nearest < best["tyr178_nearest_atom"]:
            best = {"tyr178_centroid_dist": distance, "tyr178_ring_angle": angle,
                    "tyr178_nearest_atom": nearest}
    if best is None:
        return {"tyr178_centroid_dist": float("inf"), "tyr178_ring_angle": float("nan"),
                "tyr178_nearest_atom": float("inf"), "tyr178_geometry": "no_aromatic_ring",
                "tyr178_stack": 0, "n_aromatic_rings": 0}

    angle = best["tyr178_ring_angle"]
    if angle <= PARALLEL_MAX:
        geometry = "parallel"
    elif angle >= T_SHAPED_MIN:
        geometry = "T_shaped"
    else:
        geometry = "oblique"
    best["tyr178_geometry"] = geometry
    best["tyr178_stack"] = int(
        best["tyr178_nearest_atom"] <= TYR178_ATOM_CUTOFF
        and geometry in ("parallel", "T_shaped"))
    best["n_aromatic_rings"] = len(rings)
    return best


def observe(mol: Chem.Mol, features: dict) -> dict:
    """All three observations for one pose."""
    row = {}
    row.update(leu225_contact(mol, features))
    row.update(asp218_character(mol, features))
    row.update(tyr178_stack(mol, features))
    return row


COLUMNS = [
    "label", "pose", "affinity",
    "leu225_min_dist", "leu225_contact", "n_hydrophobic",
    "asp218_any_atom_dist", "asp218_donor_dist", "asp218_neutral_donor",
    "asp218_basic_dist", "asp218_basic_contact", "arg_mimic_heads",
    "tyr178_centroid_dist", "tyr178_nearest_atom", "tyr178_ring_angle",
    "tyr178_geometry", "tyr178_stack", "n_aromatic_rings",
]


def observe_file(poses_path: str, receptor_path: str, out_csv: str) -> dict:
    """Measure every pose in an SDF and write one row each."""
    features = receptor_features(receptor_path)
    counts = {"poses": 0, "leu225": 0, "asp218_neutral": 0, "asp218_basic": 0,
              "tyr178": 0, "all_three": 0}
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for mol in Chem.SDMolSupplier(poses_path, removeHs=False):
            if mol is None:
                continue
            counts["poses"] += 1
            row = observe(mol, features)
            row["label"] = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            row["pose"] = mol.GetProp("pose") if mol.HasProp("pose") else ""
            row["affinity"] = (mol.GetProp("affinity")
                               if mol.HasProp("affinity") else "")
            for key in ("leu225_min_dist", "asp218_any_atom_dist",
                        "asp218_donor_dist", "asp218_basic_dist",
                        "tyr178_centroid_dist", "tyr178_nearest_atom",
                        "tyr178_ring_angle"):
                value = row[key]
                row[key] = "" if value != value else f"{value:.3f}"
            writer.writerow(row)
            counts["leu225"] += row["leu225_contact"]
            counts["asp218_neutral"] += row["asp218_neutral_donor"]
            counts["asp218_basic"] += row["asp218_basic_contact"]
            counts["tyr178"] += row["tyr178_stack"]
            counts["all_three"] += int(bool(row["leu225_contact"])
                                       and bool(row["asp218_neutral_donor"])
                                       and bool(row["tyr178_stack"]))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--poses", required=True,
                        help="passing-pose SDF from pose_geometry.py")
    parser.add_argument("--receptor", default="docking/receptor.pdb")
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args(argv)

    counts = observe_file(args.poses, args.receptor, args.out_csv)
    print(f"{args.poses} -> {args.out_csv}")
    total = counts["poses"] or 1
    for key in ("poses", "leu225", "asp218_neutral", "asp218_basic", "tyr178",
                "all_three"):
        pct = "" if key == "poses" else f"  ({counts[key] / total:5.1%})"
        print(f"  {key:16s} {counts[key]:7d}{pct}")
    print("\n  Observations, not selectivity. See section 8.4: only the beta side is")
    print("  analysed, no selectivity contact has been observed crystallographically")
    print("  for this target, and measured cross-isoform IC50 is future work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
