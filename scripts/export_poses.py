"""Export named docked poses as PDB, with a ChimeraX script that draws the §8.5b test.

The point is not to produce a picture but to make the FILTER visible. For every pose
exported, the .cxc draws distance monitors on exactly the two atom pairs
scripts/pose_geometry.py measured - the carboxylate oxygen nearest Ca501 of chain B,
and the hydrogen-bond donor nearest β1-Asn224's backbone O - so the figure shows the
numbers the selection actually used rather than a plausible-looking pose.

That matters most for counterexamples. A molecule can sit on the MIDAS calcium, score
better than every lead, and still be rejected because the Asn224 contact is absent;
`gen_02853` is such a case at -13.06 kcal/mol with Ca 2.98 Å and Asn224 8.37 Å. Drawing
both distances is what distinguishes "this pose is wrong" from "this pose looks fine".

RESIDUE NAMES ARE THREE CHARACTERS. Earlier in this project a five-character CCD name
(`A1AFA`) overflowed the PDB resName columns and shifted every field after it, which
zeroed all affinities in the affected file. Ligands are written as `LIG` and separated
by chain ID, with the source label kept in a REMARK.
"""

from __future__ import annotations

import argparse
import csv
import os
import string
import sys

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose_geometry import _score

__all__ = ["LIGAND_RESNAME", "export", "pose_atom_specs", "write_ligand_pdb"]

RDLogger.DisableLog("rdApp.*")

LIGAND_RESNAME = "LIG"          # three characters, deliberately
CHAIN_POOL = string.ascii_uppercase.replace("A", "").replace("B", "")  # receptor uses A,B


def _safe(label: str) -> str:
    """Filename-safe stem. A label like `ligand_crystal.pdb` would otherwise produce
    `ligand_crystal.pdb.pdb`, and a stray separator would escape the output directory."""
    keep = set(string.ascii_letters + string.digits + "_-")
    stem = "".join(c if c in keep else "_" for c in label)
    return stem.rstrip("_") or "pose"


def write_ligand_pdb(mol: Chem.Mol, path: str, chain: str, label: str) -> list[str]:
    """Write one pose as HETATM records. Returns the PDB atom names in atom order.

    Atom names are element + index, truncated to the four columns PDB allows, and the
    returned list lets the caller address a specific atom in ChimeraX by the same name
    it wrote.
    """
    conformer = mol.GetConformer()
    names, lines = [], [
        f"REMARK   1 label {label}",
        f"REMARK   1 chain {chain} resname {LIGAND_RESNAME}",
    ]
    counts: dict[str, int] = {}
    for index, atom in enumerate(mol.GetAtoms()):
        element = atom.GetSymbol()
        counts[element] = counts.get(element, 0) + 1
        name = f"{element}{counts[element]}"[:4]
        names.append(name)
        position = conformer.GetAtomPosition(index)
        # Fixed-column PDB, spelled out because getting it wrong is silent. Columns,
        # 1-indexed: 1-6 record, 7-11 serial, 13-16 atom name, 17 altLoc, 18-20 resName,
        # 22 chainID, 23-26 resSeq, 31-38/39-46/47-54 x/y/z, 55-60 occupancy,
        # 61-66 tempFactor, 77-78 element. The altLoc column is easy to omit, and
        # omitting it shifts resName and everything after it by one.
        lines.append(
            f"HETATM{index + 1:5d} {name:<4s} {LIGAND_RESNAME:>3s} {chain}"
            f"{1:4d}    {position.x:8.3f}{position.y:8.3f}{position.z:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}          {element:>2s}"
        )
    lines.append("END")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return names


def pose_atom_specs(mol: Chem.Mol, names: list[str], chain: str,
                    receptor: str = "docking/receptor.pdb") -> dict:
    """The two atom pairs the §8.5b filter measured, as ChimeraX specifiers.

    Recomputed here from the same functions the filter uses, so the drawn distance and
    the recorded distance cannot disagree.
    """
    from pose_geometry import _CARBOXYLATE_O, _DONOR, anchor_atoms

    anchors = anchor_atoms(receptor)
    positions = mol.GetConformer().GetPositions()

    def nearest(indices, target):
        if not indices:
            return None, float("inf")
        distances = [float(np.linalg.norm(positions[i] - target)) for i in indices]
        best = int(np.argmin(distances))
        return indices[best], distances[best]

    oxygens = sorted({i for match in mol.GetSubstructMatches(_CARBOXYLATE_O)
                      for i in match[1:]})
    donors = [i for (i,) in mol.GetSubstructMatches(_DONOR)]

    ca_index, ca_distance = nearest(oxygens, anchors["ca501"])
    donor_index, donor_distance = nearest(donors, anchors["asn224_o"])
    return {
        "ca_atom": f"/{chain}:1@{names[ca_index]}" if ca_index is not None else None,
        "ca_distance": ca_distance,
        "donor_atom": (f"/{chain}:1@{names[donor_index]}"
                       if donor_index is not None else None),
        "donor_distance": donor_distance,
    }


def _load_requested(poses_sdf: str, wanted: dict[str, int | None]) -> dict:
    """Pull the requested (label, pose) pairs out of the pose SDF in one pass.

    `wanted` maps a label to a 1-based pose number, or None for "the best affinity".
    One pass matters: the production pose SDF is 179 MB.
    """
    found: dict[str, dict] = {}
    for mol in Chem.SDMolSupplier(poses_sdf, removeHs=False):
        if mol is None:
            continue
        label = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        if label not in wanted:
            continue
        pose = int(mol.GetProp("pose")) if mol.HasProp("pose") else 0
        # _score, not GetProp("affinity"): it also reads smina's minimizedAffinity, so a
        # redock SDF works here as well as a Uni-Dock one.
        affinity = _score(mol)
        requested = wanted[label]
        if requested is not None:
            if pose == requested:
                found[label] = {"mol": mol, "pose": pose, "affinity": affinity}
        else:
            current = found.get(label)
            if current is None or affinity < current["affinity"]:
                found[label] = {"mol": mol, "pose": pose, "affinity": affinity}
    return found


def export(poses_sdf: str, wanted: dict[str, int | None], out_dir: str,
           receptor: str = "docking/receptor.pdb", name: str = "view") -> dict:
    """Write one PDB per requested pose plus a ChimeraX script that opens them all."""
    os.makedirs(out_dir, exist_ok=True)
    found = _load_requested(poses_sdf, wanted)
    missing = sorted(set(wanted) - set(found))

    entries = []
    for chain, label in zip(CHAIN_POOL, sorted(found)):
        record = found[label]
        path = os.path.join(out_dir, f"{_safe(label)}.pdb")
        names = write_ligand_pdb(record["mol"], path, chain, label)
        specs = pose_atom_specs(record["mol"], names, chain, receptor)
        entries.append({"label": label, "chain": chain, "path": path,
                        "pose": record["pose"], "affinity": record["affinity"],
                        **specs})

    cxc = os.path.join(out_dir, f"{name}.cxc")
    with open(cxc, "w") as handle:
        handle.write(_chimerax_script(entries, receptor))
    return {"exported": entries, "missing": missing, "cxc": cxc}


def _chimerax_script(entries: list[dict], receptor: str) -> str:
    """A ChimeraX script that opens the site and draws the two §8.5b distances."""
    lines = [
        "# Generated by scripts/export_poses.py",
        "# Section 8.5b's filter: a carboxylate oxygen within 3.2 A of Ca501 (chain B)",
        "# AND a hydrogen-bond donor within 3.5 A of beta1-Asn224's backbone O.",
        "# The distance monitors below are on the exact atom pairs the filter measured.",
        "",
        f"open {receptor}",
        "hide atoms",
        "hide cartoons",
        "show #1/A,B cartoons",
        "color #1 gray(70)",
        "transparency #1 60 cartoons",
        "",
        "# the two receptor anchors",
        "show #1/B:501 atoms",
        "style #1/B:501 sphere",
        "color #1/B:501 goldenrod",
        "show #1/B:224 atoms",
        "style #1/B:224 stick",
        "color #1/B:224 cornflowerblue",
        "# the section 8.4 observation residues, for context",
        "show #1/B:225 atoms",
        "show #1/A:178 atoms",
        "style #1/B:225,#1/A:178 stick",
        "color #1/B:225 darkseagreen",
        "color #1/A:178 plum",
        "label #1/B:501,#1/B:224,#1/B:225,#1/A:178 residues",
        "",
    ]
    palette = ["orange red", "medium purple", "dark cyan", "goldenrod",
               "forest green", "deep pink"]
    for index, entry in enumerate(entries):
        model = index + 2
        colour = palette[index % len(palette)]
        lines += [
            f"# {entry['label']}  pose {entry['pose']}  "
            f"affinity {entry['affinity']:.3f} kcal/mol",
            f"#   Ca501    {entry['ca_distance']:.2f} A "
            f"({'PASS' if entry['ca_distance'] <= 3.2 else 'FAIL'}, cutoff 3.2)",
            f"#   Asn224 O {entry['donor_distance']:.2f} A "
            f"({'PASS' if entry['donor_distance'] <= 3.5 else 'FAIL'}, cutoff 3.5)",
            f"open {entry['path']}",
            f"style #{model} stick",
            f"color #{model} {colour}",
        ]
        if entry["ca_atom"]:
            lines.append(f"distance #{model}{entry['ca_atom']} #1/B:501@CA "
                         "color yellow dashes 6")
        if entry["donor_atom"]:
            lines.append(f"distance #{model}{entry['donor_atom']} #1/B:224@O "
                         "color cyan dashes 6")
        lines.append("")

    lines += [
        "# frame the site on the MIDAS calcium",
        "view #1/B:501 @< 12",
        "lighting soft",
        "graphics silhouettes true",
        "set bgColor white",
        "",
        "# yellow dashes = carboxylate to Ca501 (cutoff 3.2 A)",
        "# cyan dashes   = donor to Asn224 backbone O (cutoff 3.5 A)",
    ]
    return "\n".join(lines) + "\n"


def _parse_requests(items: list[str]) -> dict[str, int | None]:
    """`gen_07401` means the best-affinity pose; `gen_02853:4` means pose 4."""
    wanted: dict[str, int | None] = {}
    for item in items:
        if ":" in item:
            label, pose = item.rsplit(":", 1)
            wanted[label] = int(pose)
        else:
            wanted[item] = None
    return wanted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", nargs="+",
                        help="labels, optionally LABEL:POSE; default is best affinity")
    parser.add_argument("--poses", default="docking/poses.sdf")
    parser.add_argument("--receptor", default="docking/receptor.pdb")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--name", default="view",
                        help="basename for the ChimeraX script")
    args = parser.parse_args(argv)

    result = export(args.poses, _parse_requests(args.labels), args.out_dir,
                    args.receptor, args.name)
    print(f"{args.poses} -> {args.out_dir}")
    for entry in result["exported"]:
        ca_ok = "PASS" if entry["ca_distance"] <= 3.2 else "FAIL"
        donor_ok = "PASS" if entry["donor_distance"] <= 3.5 else "FAIL"
        print(f"  {entry['label']:14s} chain {entry['chain']}  pose {entry['pose']:>2d}"
              f"  aff {entry['affinity']:7.3f}"
              f"  Ca {entry['ca_distance']:5.2f} {ca_ok}"
              f"  Asn224 {entry['donor_distance']:5.2f} {donor_ok}")
    if result["missing"]:
        print(f"  NOT FOUND in {args.poses}: {', '.join(result['missing'])}")
    print(f"\n  ChimeraX: open {result['cxc']}")
    return 1 if result["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
