"""Export v2 docked poses as PDB plus a ChimeraX script, for one receptor.

Same purpose as scripts/export_poses.py, which this does not replace: v1's
exporter reads a pose SDF and hard-codes alphaVbeta1's anchors (Ca501 of chain
B, beta1-Asn224 backbone O), because v1 only ever drew that one receptor. v2
draws a lead beside the reference compounds in TWO receptors, so the anchors
have to be arguments.

WHAT THE SCRIPT IS FOR. Not a picture -- a way to see the selection. The
alphaVbeta1 view draws the two distances section 8.5b measured, on the atom
pairs it measured them on, so a pose that looks reasonable and failed the
filter is distinguishable from one that passed. The isoform view draws the
same carboxylate-to-metal distance, which is what makes the two views
comparable: selectivity here IS the claim that this ligand reaches one MIDAS
and not the other.

THE METAL'S COLOUR COMES FROM ITS ELEMENT, never from a flag. alphaVbeta1's
MIDAS is Ca and alphaVbeta6's is Mg, the two views are read side by side, and
a colour chosen per run would let the same colour mean calcium in one figure
and magnesium in the next.

RESIDUE NAMES ARE THREE CHARACTERS, for the reason export_poses.py records:
a five-character CCD name (`A1AFA`) once overflowed the PDB resName columns
and shifted every field after it, zeroing the affinities in that file.

Paths in the .cxc are RELATIVE and the receptor is copied in beside the
ligands, so the output directory can be downloaded whole and opened on a
machine that has ChimeraX but not this repository.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import string
import sys

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_poses import LIGAND_RESNAME, _safe, write_ligand_pdb
from poses_to_sdf import poses_from_pdbqt
from selectivity import label_variants

__all__ = ["METAL_COLOURS", "LEAD_COLOURS", "REFERENCE_COLOURS",
           "metal_atom", "contact_residues", "export", "chimerax_script",
           "leads_from_csv"]

RDLogger.DisableLog("rdApp.*")

CHAIN_POOL = string.ascii_uppercase.replace("A", "").replace("B", "")

# Keyed by element, so a metal is the same colour in every figure this writes.
# Goldenrod for Ca is v1_leads.cxc's choice and is kept; Mg has to differ from
# it, since the two views are compared side by side.
METAL_COLOURS = {"CA": "goldenrod", "MG": "medium sea green",
                 "MN": "dark orange", "ZN": "slate gray"}
METAL_FALLBACK = "goldenrod"

# Warm for the molecules under test, cool for the references they are judged
# against, so which is which survives being read at a distance.
LEAD_COLOURS = ("orange red", "hot pink", "dark orange", "crimson", "magenta",
                "tomato")
REFERENCE_COLOURS = ("medium purple", "teal", "steel blue", "dark cyan",
                     "slate blue", "dark olive green")

# v1_leads.cxc's receptor styling, reproduced exactly.
# Everything the filter actually measured against, and nothing else. The
# pocket view is for reading one pose in context; this one is for overlaying
# several poses on the contacts that decided them, where the cartoon and
# forty context sticks are what stops you seeing which ligand reaches the
# metal.
KEY_ONLY_STYLE = """hide atoms
hide cartoons
color #1 gray(70)"""

RECEPTOR_STYLE = """hide atoms
hide cartoons
show #1/A,B cartoons
color #1 gray(70)
transparency #1 60 cartoons

# receptor chains, colored separately
color #1/A light steel blue cartoons
color #1/B rosy brown cartoons"""

CA_CUTOFF = 3.2                 # section 8.5b, carboxylate O to the MIDAS metal
DONOR_CUTOFF = 3.5              # section 8.5b, donor to the contact residue's O
CONTEXT_RADIUS = 4.5            # residues drawn as thin sticks around the site


def _parse_residue(text: str) -> tuple[str, str]:
    """`B/501` or `B:501` -> ("B", "501"). Kept as a string: PDB resSeq is a
    field, not a number, and insertion codes exist."""
    for separator in ("/", ":"):
        if separator in text:
            chain, _, resseq = text.partition(separator)
            return chain.strip(), resseq.strip()
    raise ValueError(f"{text!r}: expected CHAIN/RESSEQ, e.g. B/501")


def _atom_records(receptor: str):
    with open(receptor) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                yield line


def metal_atom(receptor: str, chain: str, resseq: str) -> dict:
    """The MIDAS metal: its element, its ChimeraX atom name and its position.

    Read from the structure rather than passed in, because the element decides
    the colour and a mislabelled flag would silently recolour the metal.
    """
    for line in _atom_records(receptor):
        if line[21] != chain or line[22:26].strip() != resseq:
            continue
        name = line[12:16].strip()
        element = (line[76:78].strip() or name).upper()
        return {"name": name, "element": element,
                "resname": line[17:20].strip(),
                "colour": METAL_COLOURS.get(element, METAL_FALLBACK),
                "xyz": np.array([float(line[30:38]), float(line[38:46]),
                                 float(line[46:54])])}
    raise ValueError(
        f"{receptor}: no atom at {chain}/{resseq}. The MIDAS metal is the "
        "anchor every distance in this view is drawn to; guessing one would "
        "draw distances to the wrong place.")


def backbone_o(receptor: str, chain: str, resseq: str) -> np.ndarray:
    for line in _atom_records(receptor):
        if (line[21] == chain and line[22:26].strip() == resseq
                and line[12:16].strip() == "O"):
            return np.array([float(line[30:38]), float(line[38:46]),
                             float(line[46:54])])
    raise ValueError(f"{receptor}: no backbone O at {chain}/{resseq}")


# Anything in here near the site is drawn as a sphere, not as a stick. The
# integrin beta-propeller/betaI interface carries three metal sites -- MIDAS,
# ADMIDAS and SyMBS -- and the neighbours matter: a pose that reaches ADMIDAS
# instead of MIDAS is a different binding mode, and a stick-styled ion in a
# residue list hides that there is a second metal there at all.
METALS = frozenset(METAL_COLOURS) | {"NA", "K", "FE", "CO", "NI", "CU", "CD"}


def contact_residues(receptor: str, points: np.ndarray,
                     radius: float = CONTEXT_RADIUS) -> list[tuple[str, str]]:
    """Receptor residues with any atom within `radius` of any ligand atom.

    Measured rather than listed, so the pocket drawn around a pose is the
    pocket that pose actually sits in. A hand-written residue list would have
    to be right for both receptors, and alphaVbeta6's numbering is not
    alphaVbeta1's.
    """
    if not len(points):
        return []
    residues: dict[tuple[str, str], None] = {}
    metals: dict[tuple[str, str], str] = {}
    for line in _atom_records(receptor):
        xyz = np.array([float(line[30:38]), float(line[38:46]),
                        float(line[46:54])])
        if np.min(np.linalg.norm(points - xyz, axis=1)) > radius:
            continue
        key = (line[21], line[22:26].strip())
        element = (line[76:78].strip() or line[12:16].strip()).upper()
        if line.startswith("HETATM") and element in METALS:
            metals[key] = element
        else:
            residues.setdefault(key, None)
    return sorted(residues), metals


def _carboxylate_and_donor(mol: Chem.Mol):
    from pose_geometry import _CARBOXYLATE_O, _DONOR

    oxygens = sorted({i for match in mol.GetSubstructMatches(_CARBOXYLATE_O)
                      for i in match[1:]})
    donors = [i for (i,) in mol.GetSubstructMatches(_DONOR)]
    return oxygens, donors


def _nearest(positions, indices, target):
    if not indices or target is None:
        return None, float("inf")
    distances = [float(np.linalg.norm(positions[i] - target)) for i in indices]
    best = int(np.argmin(distances))
    return indices[best], distances[best]


def _load(pose_dirs: list[str], label: str, pose: int | None):
    """The requested pose of one label, from whichever directory holds it."""
    for directory in pose_dirs:
        # The reference set is docked under either spelling; see
        # selectivity.label_variants for why this is not a guess.
        for spelling in label_variants(label):
            path = os.path.join(directory, f"{spelling}_out.pdbqt")
            if os.path.exists(path):
                break
        else:
            continue
        poses = poses_from_pdbqt(path, label=label)
        if not poses:
            continue
        if pose is None:
            return poses[0], 1
        if pose <= len(poses):
            return poses[pose - 1], pose
        raise ValueError(
            f"{path} holds {len(poses)} poses; {label}:{pose} was asked for. "
            "A file is not a pose -- Uni-Dock writes as many as the energy "
            "window kept, which differs per ligand.")
    return None, None


def export(pose_dirs: list[str], wanted: dict, out_dir: str, receptor: str,
           midas: tuple[str, str], contact: tuple[str, str] | None,
           leads: set, name: str = "view", title: str = "",
           residues: str = "pocket") -> dict:
    """Write the receptor copy, one PDB per pose and the ChimeraX script."""
    os.makedirs(out_dir, exist_ok=True)
    metal = metal_atom(receptor, *midas)
    contact_o = backbone_o(receptor, *contact) if contact else None

    shutil.copyfile(receptor, os.path.join(out_dir, "receptor.pdb"))

    entries, missing, all_points = [], [], []
    ordered = [l for l in wanted if l in leads] + \
              [l for l in wanted if l not in leads]
    lead_colours, reference_colours = iter(LEAD_COLOURS), iter(REFERENCE_COLOURS)

    for chain, label in zip(CHAIN_POOL, ordered):
        mol, pose = _load(pose_dirs, label, wanted[label])
        if mol is None:
            missing.append(label)
            continue
        path = os.path.join(out_dir, f"{_safe(label)}.pdb")
        names = write_ligand_pdb(mol, path, chain, label)
        positions = mol.GetConformer().GetPositions()
        all_points.append(positions)

        oxygens, donors = _carboxylate_and_donor(mol)
        metal_index, metal_distance = _nearest(positions, oxygens, metal["xyz"])
        # The filter measures the CARBOXYLATE oxygen, but a ligand can reach
        # the MIDAS with something else -- compound 25 does, at 7.4-9.8 A
        # carboxylate while PLIP still reports metal coordination. Drawing
        # only the carboxylate distance hides the contact that is actually
        # there, so the nearest coordinating heavy atom is drawn too, in a
        # different colour, whenever it is a different atom.
        coordinating = [i for i, atom in enumerate(mol.GetAtoms())
                        if atom.GetSymbol() in ("O", "N", "S")]
        near_index, near_distance = _nearest(positions, coordinating,
                                             metal["xyz"])
        donor_index, donor_distance = _nearest(positions, donors, contact_o)
        affinity = (float(mol.GetProp("affinity"))
                    if mol.HasProp("affinity") else None)
        is_lead = label in leads
        entries.append({
            "label": label, "chain": chain, "pose": pose,
            "path": os.path.basename(path), "affinity": affinity,
            "is_lead": is_lead,
            "colour": next(lead_colours if is_lead else reference_colours),
            "metal_atom": (f"@{names[metal_index]}"
                           if metal_index is not None else None),
            "metal_distance": metal_distance,
            "near_atom": (f"@{names[near_index]}"
                          if near_index is not None
                          and near_index != metal_index else None),
            "near_distance": near_distance,
            "near_element": (mol.GetAtomWithIdx(near_index).GetSymbol()
                             if near_index is not None else None),
            "donor_atom": (f"@{names[donor_index]}"
                           if donor_index is not None else None),
            "donor_distance": donor_distance,
        })

    points = np.vstack(all_points) if all_points else np.empty((0, 3))
    context, metals = contact_residues(receptor, points)
    context = [r for r in context
               if r != midas and (contact is None or r != contact)]
    # The MIDAS metal is drawn on its own terms above; these are its
    # neighbours, and which ones are present is part of what the view shows.
    neighbours = {k: v for k, v in metals.items() if k != midas}

    cxc_path = os.path.join(out_dir, f"{name}.cxc")
    with open(cxc_path, "w") as handle:
        handle.write(chimerax_script(entries, metal, midas, contact, context,
                                     title, neighbours, residues))
    return {"exported": entries, "missing": missing, "cxc": cxc_path,
            "metal": metal, "context": context, "neighbours": neighbours}


def chimerax_script(entries, metal, midas, contact, context, title="",
                    neighbours=None, residues: str = "pocket") -> str:
    midas_spec = f"#1/{midas[0]}:{midas[1]}"
    lines = [
        "# Generated by scripts/export_poses_v2.py",
        f"# {title}" if title else "# v2 docked poses",
        "# Section 8.5b's filter: a carboxylate oxygen within "
        f"{CA_CUTOFF} A of the MIDAS metal",
    ]
    if contact:
        lines.append(f"# AND a hydrogen-bond donor within {DONOR_CUTOFF} A of "
                     f"{contact[0]}/{contact[1]}'s backbone O.")
        lines.append("# The distance monitors below are on the exact atom "
                     "pairs the filter measured.")
    else:
        lines.append("# This receptor has no pre-registered donor contact, so "
                     "only the metal distance is drawn.")
        lines.append("# Inventing one would put a number on the figure that "
                     "no filter ever used.")

    lines += ["", "# ---- receptor ----", "open receptor.pdb",
              RECEPTOR_STYLE if residues == "pocket" else KEY_ONLY_STYLE,
              "", f"# the MIDAS metal: {metal['resname']} "
              f"{midas[0]}/{midas[1]}, element {metal['element']}",
              f"show {midas_spec} atoms",
              f"style {midas_spec} sphere",
              f"color {midas_spec} {metal['colour']}"]

    if contact:
        spec = f"#1/{contact[0]}:{contact[1]}"
        lines += ["", "# the pre-registered donor contact",
                  f"show {spec} atoms", f"style {spec} stick",
                  f"color {spec} cornflower blue atoms",
                  f"color {spec} byhetero atoms",
                  f"label {midas_spec}|{spec} residues"]
    else:
        lines += ["", f"label {midas_spec} residues"]

    for (chain, resseq), element in sorted((neighbours or {}).items()):
        spec = f"#1/{chain}:{resseq}"
        lines += ["", f"# a second metal site in reach of the poses: "
                  f"{element} {chain}/{resseq}",
                  f"show {spec} atoms", f"style {spec} sphere",
                  f"size {spec} atomRadius 0.8",
                  f"color {spec} {METAL_COLOURS.get(element, METAL_FALLBACK)}",
                  f"label {spec} residues"]

    if context and residues == "pocket":
        spec = "|".join(f"#1/{c}:{r}" for c, r in context)
        lines += ["", f"# pocket context: residues within {CONTEXT_RADIUS} A "
                  "of any exported pose, measured not listed",
                  f"show {spec} atoms", f"style {spec} stick",
                  f"size {spec} stickRadius 0.06",
                  f"color {spec} gray(55) atoms",
                  f"color {spec} byhetero atoms"]

    lines += ["", "# ---- ligand pose(s) ----"]
    for index, entry in enumerate(entries, start=2):
        role = "LEAD" if entry["is_lead"] else "reference"
        affinity = ("?" if entry["affinity"] is None
                    else f"{entry['affinity']:.3f}")
        lines.append(f"# {entry['label']}  ({role})  pose {entry['pose']}  "
                     f"affinity {affinity} kcal/mol")
        verdict = "PASS" if entry["metal_distance"] <= CA_CUTOFF else "FAIL"
        lines.append(f"#   {metal['element']:<8s} {entry['metal_distance']:.2f} A "
                     f"carboxylate ({verdict}, cutoff {CA_CUTOFF})")
        if entry.get("near_atom"):
            lines.append(f"#   {metal['element']:<8s} "
                         f"{entry['near_distance']:.2f} A nearest "
                         f"{entry['near_element']} -- NOT a carboxylate, so "
                         "the geometry gate does not count it")
        if contact:
            verdict = "PASS" if entry["donor_distance"] <= DONOR_CUTOFF else "FAIL"
            lines.append(f"#   donor    {entry['donor_distance']:.2f} A "
                         f"({verdict}, cutoff {DONOR_CUTOFF})")
        lines += [f"open {entry['path']}", f"style #{index} stick",
                  f"color #{index} {entry['colour']}",
                  f"color #{index} byhetero", ""]

    lines += ["# ---- distance monitors ----",
              "# chain is intentionally omitted from each ligand spec so the",
              "# atom name matches regardless of what chain ID the docking "
              "output used."]
    for index, entry in enumerate(entries, start=2):
        if entry["metal_atom"]:
            lines.append(f"distance #{index}{entry['metal_atom']} "
                         f"{midas_spec}@{metal['name']} color yellow dashes 6 "
                         "radius 0.06 decimalPlaces 2")
        if entry.get("near_atom"):
            lines.append(f"distance #{index}{entry['near_atom']} "
                         f"{midas_spec}@{metal['name']} color orange dashes 3 "
                         "radius 0.06 decimalPlaces 2")
        if contact and entry["donor_atom"]:
            lines.append(f"distance #{index}{entry['donor_atom']} "
                         f"#1/{contact[0]}:{contact[1]}@O color cyan dashes 6 "
                         "radius 0.06 decimalPlaces 2")

    if residues == "key":
        lines += ["", "# 다른 단백질 잔기는 모두 숨긴다 — 필터가 측정한 "
                  "접촉만 남긴다. 방향을 잡으려면 다음 줄을 살린다:",
                  "# show #1/A,B cartoons"]
    lines += ["", "# ---- view + rendering ----",
              f"view {midas_spec} @< 12", "lighting soft",
              "graphics silhouettes true", "set bgColor white", "",
              ("# receptor  : chain A light steel blue, chain B rosy brown "
               "(cartoons)" if residues == "pocket" else
               "# receptor  : hidden -- only the contacts the filter measured "
               "are drawn"),
              f"# {metal['element']:<9s} : {metal['colour']} sphere "
              f"({metal['resname']} {midas[0]}/{midas[1]}); the colour is "
              "keyed to the element,",
              "#             so it means the same metal in every view this "
              "writes"]
    for (chain, resseq), element in sorted((neighbours or {}).items()):
        lines.append(
            f"# {element:<9s} : {METAL_COLOURS.get(element, METAL_FALLBACK)} "
            f"sphere ({chain}/{resseq}) -- a NEIGHBOURING metal site, not the "
            "MIDAS one")
    for entry in entries:
        role = "LEAD     " if entry["is_lead"] else "reference"
        lines.append(f"# {role} : {entry['label']} carbons {entry['colour']}")
    lines += ["# heteroatoms by element on every ligand (N blue, O red, "
              "S yellow)",
              f"# yellow dashes = carboxylate to the MIDAS metal "
              f"(cutoff {CA_CUTOFF} A)",
              "# orange dashes = nearest O/N/S to the metal where that is NOT "
              "the carboxylate;",
              "#                 drawn because a ligand can coordinate the "
              "MIDAS with another atom,",
              "#                 which the geometry gate does not count"]
    if contact:
        lines.append(f"# cyan  dashes  = donor to {contact[0]}/{contact[1]} "
                     f"backbone O (cutoff {DONOR_CUTOFF} A)")
    return "\n".join(lines) + "\n"


def leads_from_csv(path: str) -> list[str]:
    """Labels that passed every stage, from lead_filter.py's --out-csv.

    Read rather than retyped. The labels are already in a file the filter
    wrote, and a hand-copied list is a second place for them to be wrong --
    silently, since a mistyped label exports nothing and the view simply
    lacks a molecule.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    if "all_stages" not in rows[0]:
        raise ValueError(
            f"{path} has no all_stages column; it is not a lead_filter.py "
            f"--out-csv (columns: {', '.join(sorted(rows[0]))})")
    return [r["label"] for r in rows if r["all_stages"] == "True"]


def _parse_requests(items: list[str]) -> dict:
    wanted: dict[str, int | None] = {}
    for item in items:
        label, _, pose = item.partition(":")
        wanted[label] = int(pose) if pose else None
    return wanted


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("labels", nargs="*", default=[],
                    help="LABEL or LABEL:POSE, added to whatever --leads-csv "
                         "and --controls supply")
    p.add_argument("--leads-csv",
                    help="lead_filter.py --out-csv; every row with "
                         "all_stages True is exported and drawn as a lead")
    p.add_argument("--controls", action="store_true",
                    help="also export the four §10.4 calibration controls, "
                         "drawn as references")
    p.add_argument("--poses", action="append", required=True,
                    help="directory of <label>_out.pdbqt; repeat to search "
                         "several")
    p.add_argument("--receptor", required=True)
    p.add_argument("--midas", required=True, metavar="CHAIN/RESSEQ",
                    help="the MIDAS metal, e.g. B/501 for 8W30")
    p.add_argument("--contact", metavar="CHAIN/RESSEQ",
                    help="the pre-registered donor contact, e.g. B/224 for "
                         "beta1-Asn224. Omit where none is registered")
    p.add_argument("--lead", action="append", default=[],
                    help="label to draw as a lead (warm colour); repeat")
    p.add_argument("--residues", choices=("pocket", "key"), default="pocket",
                    help="'pocket' draws residues within 4.5 A of a pose as "
                         "thin sticks, for reading one pose in context. "
                         "'key' draws only what the filter measured -- the "
                         "MIDAS metal, the registered donor contact and the "
                         "neighbouring metal sites -- and hides the cartoon, "
                         "for overlaying several poses on what decided them.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="view")
    p.add_argument("--title", default="")
    args = p.parse_args(argv)

    labels, leads = list(args.labels), set(args.lead)
    if args.leads_csv:
        try:
            found = leads_from_csv(args.leads_csv)
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        if not found:
            sys.stderr.write(
                f"{args.leads_csv}: no row passed every stage, so there is no "
                "lead to draw. That is a result, not an error -- but this "
                "view would be references only.\n")
        labels += found
        leads |= set(found)
    if args.controls:
        from selectivity import CALIBRATION
        labels += [c for group in CALIBRATION.values() for c in group]

    if not labels:
        sys.stderr.write("no labels: pass some, or --leads-csv, or "
                         "--controls.\n")
        return 2

    try:
        report = export(args.poses, _parse_requests(labels), args.out_dir,
                        args.receptor, _parse_residue(args.midas),
                        _parse_residue(args.contact) if args.contact else None,
                        leads, args.name, args.title, args.residues)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    print(f"wrote {report['cxc']}")
    print(f"  MIDAS {report['metal']['resname']} "
          f"({report['metal']['element']}) drawn {report['metal']['colour']}")
    for entry in report["exported"]:
        role = "LEAD" if entry["is_lead"] else "ref "
        print(f"  {role} {entry['label']:<20s} pose {entry['pose']}  "
              f"metal {entry['metal_distance']:.2f} A  "
              f"{entry['colour']}")
    print(f"  pocket context: {len(report['context'])} residues")
    for (chain, resseq), element in sorted(report["neighbours"].items()):
        print(f"  neighbouring metal {element} {chain}/{resseq} drawn "
              f"{METAL_COLOURS.get(element, METAL_FALLBACK)}")
    if report["missing"]:
        sys.stderr.write(
            f"no pose found for: {', '.join(report['missing'])}\n"
            "  A label absent from every --poses directory was not docked "
            "against this receptor.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
