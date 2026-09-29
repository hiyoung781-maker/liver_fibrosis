"""Split an integrin structure into receptor and MIDAS-bound ligand (spec §10).

Section 10.3 applies the SAME redocking gate to every isoform as to alphaVbeta1:
redock the structure's own crystal ligand and require PLIP's MIDAS metal
coordination plus a symmetry-corrected RMSD under 2.0 A. That needs the
ligand and the receptor separated, and the MIDAS metal identified, for a
structure this code has never seen.

NOTHING IS HARDCODED PER STRUCTURE. The MIDAS-bound ligand is found by
measurement: among the non-solvent HETATM residues large enough to be a
ligand, and the metal ions, the pair with the smallest interatomic distance
IS the MIDAS complex. Hardcoding a ligand code per isoform would not survive
the next structure, and hardcoding the wrong one fails every pose for a
reason that looks chemical -- this project has already spent a day on
exactly that shape of bug.

THE METAL IS NOT ALWAYS CALCIUM. 8W30's MIDAS is Ca, 6MK0's is Mn, 9CZD's is
Mg. A calcium-specific rule finds nothing on two of the three.

ONLY CHAINS THAT TOUCH THE LIGAND ARE KEPT. 9CZD carries a 17E6 Fab as a
crystallisation chaperone, and its heavy chain's CDR reaches to 6.08 A of the
ligand (Ser55) -- outside contact but inside the docking box, which is the
ligand's extent plus 6 A. Keeping it would put a non-biological wall at the
box edge, and would build this receptor differently from 8W30's and 6MK0's,
which have only their two integrin chains. The whole point of the Delta axis
is that the two receptors differ in the protein and in nothing else, so the
rule is contact with the ligand, not proximity and not chain name. That the
deposited site was determined with the Fab 6 A away is a limitation to
state, not to model.

Section 10.2's own structure list did not survive review: none of 1L5G,
3VI4, 4UM9 or 6UJA has a small-molecule ligand, so none can pass 10.3 as
written. alphaVbeta3 moved to 6MK0 and alphaVbeta6 to 9CZD; alpha5beta1 and
alphaVbeta8 have no small-molecule structure in the PDB at all and are
excluded. See results/v2_isoform_structures.md.

THE LIGAND'S BOND ORDERS COME FROM A TEMPLATE, NEVER FROM GEOMETRY. The
first version of this script wrote the ligand SDF with Open Babel, which
infers bond orders from coordinates and got BOTH isoform ligands wrong in
the Arg-mimic head: 6MK0's aromatic 1,8-naphthyridine came back reduced to a
tetrahydronaphthyridine, and 9CZD's tetrahydronaphthyridine came back as an
imine. Each error inverts the hydrogen-bonding character of the group that
binds the alphaV subunit's Asp218, so the section 10.3 gate asked whether
the crystal pose of a molecule that was not the crystal ligand could be
reproduced. --ligand-ccd is therefore required, and scripts/ligand_template
refuses any template that does not match the extracted ligand. See
results/v2_isoform_structures.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

__all__ = ["DroppedNearLigandError", "METAL_NAMES",
           "SOLVENT_AND_ADDITIVES", "binding_site_residues",
           "chains_near", "find_midas_ligand", "split_structure"]


class DroppedNearLigandError(RuntimeError):
    """A glycan or additive sits in the binding site.

    Meeko has no residue template for these and stops on them, so they
    have to come out of the receptor -- but one inside the pocket is
    part of the pocket. Removing it would change the site while every
    score still looked normal, which is how this project lost six
    calciums to `obabel -p 7.4` once already.
    """

# Divalent cations that occupy MIDAS/ADMIDAS/SyMBS across the integrin family.
METAL_NAMES = {"CA", "MG", "MN", "ZN", "CO", "NI", "CD"}

# Waters, ions, glycans and crystallisation additives: never the ligand.
SOLVENT_AND_ADDITIVES = {
    "HOH", "WAT", "DOD",
    "NAG", "BMA", "MAN", "GLC", "FUC", "GAL", "NDG", "A2G", "BGC", "SIA",
    "XYP", "FUL", "GLA", "RAM",
    "SO4", "PO4", "GOL", "EDO", "ACT", "PEG", "PG4", "MPD", "TRS", "IMD",
    "CIT", "MES", "EPE", "DMS", "IOD", "FMT", "TAR", "OXL", "NH4", "UNX",
    "NA", "CL", "K", "BR", "F",
} | METAL_NAMES

MIN_LIGAND_ATOMS = 8
MIDAS_MAX_DISTANCE = 3.5


def _atoms(lines: list) -> list:
    out = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        out.append({
            "het": line.startswith("HETATM"),
            "name": line[17:20].strip().upper(),
            "chain": line[21],
            "seq": line[22:26].strip(),
            "xyz": xyz,
            "line": line,
        })
    return out


def _key(atom: dict) -> tuple:
    try:
        return (atom["name"], atom["chain"], int(atom["seq"]))
    except ValueError:
        return (atom["name"], atom["chain"], atom["seq"])


def find_midas_ligand(lines: list) -> dict:
    """The (ligand, metal) pair whose atoms come closest. Measurement, not a table.

    Returns {"ligand": (resname, chain, resseq), "metal": (...),
             "distance": float}. Raises when no ligand-metal pair is within
    MIDAS_MAX_DISTANCE, because a structure with no MIDAS-bound small molecule
    cannot serve section 10.3 and guessing one would be worse than stopping.
    """
    atoms = _atoms(lines)
    groups = defaultdict(list)
    metals = []
    for atom in atoms:
        if not atom["het"]:
            continue
        if atom["name"] in METAL_NAMES:
            metals.append(atom)
        elif atom["name"] not in SOLVENT_AND_ADDITIVES:
            groups[_key(atom)].append(atom)

    candidates = {k: v for k, v in groups.items() if len(v) >= MIN_LIGAND_ATOMS}
    if not candidates:
        raise ValueError(
            f"no HETATM residue with at least {MIN_LIGAND_ATOMS} atoms that is "
            "not solvent, an ion, a glycan or an additive. This structure has "
            "no small-molecule ligand, so section 10.3's gate cannot be run on "
            "it -- see results/v2_isoform_structures.md.")
    if not metals:
        raise ValueError("no divalent cation found; there is no MIDAS to bind.")

    best = None
    for key, group in candidates.items():
        for metal in metals:
            distance = min(math.dist(a["xyz"], metal["xyz"]) for a in group)
            if best is None or distance < best[0]:
                best = (distance, key, _key(metal))
    distance, ligand, metal = best
    if distance > MIDAS_MAX_DISTANCE:
        raise ValueError(
            f"closest ligand-metal contact is {distance:.2f} A, beyond "
            f"{MIDAS_MAX_DISTANCE} A: no ligand is MIDAS-bound here.")
    return {"ligand": ligand, "metal": metal, "distance": distance}


def chains_near(lines: list, ligand_key: tuple, cutoff: float = 4.5) -> set:
    """Polymer chains with at least one atom within `cutoff` of the ligand.

    The default is a CONTACT distance, not a proximity one. Measured on the
    three structures this campaign uses:

        8W30   A 2.62   B 2.36
        6MK0   A 2.60   B 2.29
        9CZD   A 2.64   B 2.36   C 20.14   D 6.08

    Chains A and B are the integrin in all three. 9CZD's C and D are a 17E6
    Fab, and D reaches 6.08 A -- close enough that a 12 A rule keeps it, far
    enough that it touches nothing. Keeping it would make this receptor
    differently constructed from the other two, which is exactly what the
    Delta axis cannot tolerate.
    """
    atoms = _atoms(lines)
    ligand = [a["xyz"] for a in atoms if a["het"] and _key(a) == ligand_key]
    if not ligand:
        raise ValueError(f"ligand {ligand_key} not found")
    near = set()
    for atom in atoms:
        if atom["het"] or atom["chain"] in near:
            continue
        if any(math.dist(atom["xyz"], l) <= cutoff for l in ligand):
            near.add(atom["chain"])
    return near


def binding_site_residues(lines: list, ligand_key: tuple,
                          cutoff: float = 5.0) -> set:
    """(chain, resseq) of polymer residues within `cutoff` of the ligand.

    Derived from the structure rather than listed per isoform, so the guard
    that refuses to truncate a binding-site residue works on a structure this
    code has never seen.
    """
    atoms = _atoms(lines)
    ligand = [a["xyz"] for a in atoms if a["het"] and _key(a) == ligand_key]
    site = set()
    for atom in atoms:
        if atom["het"]:
            continue
        try:
            seq = int(atom["seq"])
        except ValueError:
            continue
        if any(math.dist(atom["xyz"], l) <= cutoff for l in ligand):
            site.add((atom["chain"], seq))
    return site


def split_structure(lines: list, ligand_key: tuple,
                    keep_chains: set | None = None,
                    near_cutoff: float = 4.5) -> tuple:
    """(receptor lines, ligand lines).

    The receptor keeps its polymer and its METALS -- removing the MIDAS ion
    would delete the interaction this whole campaign is about -- and drops
    the ligand itself, waters, glycans, cryoprotectants and other additives,
    plus any chain not in `keep_chains`.

    Glycans and additives have to go because Meeko has no residue template
    for them and stops: 6MK0 carries 8 NAG, 9CZD carries NAG, four GOL and an
    ACT. 8W30's prepared receptor had none, so the alphaVbeta1 path never met
    this.

    They never go silently. One within `near_cutoff` of the ligand is part of
    the binding site, and raises DroppedNearLigandError rather than being
    removed -- taking it out would change the pocket while every score still
    looked normal.
    """
    ligand = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[17:20].strip().upper()
        try:
            key = (name, line[21], int(line[22:26].strip()))
        except ValueError:
            key = (name, line[21], line[22:26].strip())
        if key == ligand_key:
            ligand.append(line)
    if not ligand:
        raise ValueError(f"ligand {ligand_key} not found")
    ligand_xyz = [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                  for l in ligand]

    receptor, intruders = [], {}
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[17:20].strip().upper()
        chain = line[21]
        seq = line[22:26].strip()
        try:
            key = (name, chain, int(seq))
        except ValueError:
            key = (name, chain, seq)
        if key == ligand_key:
            continue
        if keep_chains is not None and chain not in keep_chains:
            continue
        # Metals stay; every other non-polymer HETATM goes.
        if line.startswith("HETATM") and name not in METAL_NAMES \
                and name in SOLVENT_AND_ADDITIVES:
            if name not in ("HOH", "WAT", "DOD"):
                try:
                    xyz = (float(line[30:38]), float(line[38:46]),
                           float(line[46:54]))
                except ValueError:
                    continue
                if min(math.dist(xyz, l) for l in ligand_xyz) <= near_cutoff:
                    intruders[key] = round(
                        min(math.dist(xyz, l) for l in ligand_xyz), 2)
            continue
        receptor.append(line)

    if intruders:
        raise DroppedNearLigandError(
            f"these non-polymer residues sit within {near_cutoff} A of the "
            f"ligand and would be removed: {intruders}. One of them is part "
            "of the binding site; decide explicitly rather than dropping it.")
    return receptor, ligand


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--structure", required=True, help="Deposited PDB file.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chain-cutoff", type=float, default=4.5,
                    help="Keep polymer chains that CONTACT the ligand within "
                         "this distance; the rest are crystallisation "
                         "chaperones. See chains_near's docstring for the "
                         "measured per-chain distances.")
    p.add_argument("--site-cutoff", type=float, default=5.0)
    p.add_argument("--ligand-ccd",
                    help="The ligand's PDB chemical component id, used to fetch "
                         "its deposited bond orders. Give the FULL id: a "
                         "five-character code does not fit the PDB residue "
                         "field, so 9CZD's A1A6H appears in the file as A1A.")
    p.add_argument("--ligand-smiles",
                    help="The template SMILES directly, instead of fetching it.")
    args = p.parse_args(argv)

    if not args.ligand_ccd and not args.ligand_smiles:
        sys.stderr.write(
            "one of --ligand-ccd or --ligand-smiles is required. The ligand's "
            "bond orders are taken from its deposited chemical component, "
            "never inferred from coordinates: Open Babel's perception got both "
            "isoform ligands wrong in the group that binds alphaV Asp218.\n")
        return 2

    lines = Path(args.structure).read_text().splitlines()
    found = find_midas_ligand(lines)
    keep = chains_near(lines, found["ligand"], args.chain_cutoff)
    site = binding_site_residues(lines, found["ligand"], args.site_cutoff)
    receptor, ligand = split_structure(lines, found["ligand"], keep_chains=keep)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "receptor.pdb").write_text("\n".join(receptor + ["END"]) + "\n")
    (out / "ligand.pdb").write_text("\n".join(ligand + ["END"]) + "\n")

    from rdkit import Chem

    from ligand_template import (TemplateMismatchError, assign_from_template,
                                 ccd_smiles)

    template = args.ligand_smiles
    if template is None:
        try:
            template = ccd_smiles(args.ligand_ccd)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"could not fetch the chemical component {args.ligand_ccd!r}: "
                f"{exc}\nPass --ligand-smiles instead. Bond orders are never "
                "inferred from coordinates here -- that is the defect this "
                "argument exists to prevent.\n")
            return 1

    try:
        ligand_mol = assign_from_template("\n".join(ligand + ["END"]), template)
    except TemplateMismatchError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sdf = out / "ligand.sdf"
    writer = Chem.SDWriter(str(sdf))
    ligand_mol.SetProp("_Name", f"XTAL_{out.name}")
    writer.write(ligand_mol)
    writer.close()

    all_chains = sorted({l[21] for l in lines
                         if l.startswith("ATOM")})
    manifest = {
        "structure": args.structure,
        "ligand": list(found["ligand"]),
        "midas_metal": list(found["metal"]),
        "midas_distance": round(found["distance"], 3),
        "ligand_template": template,
        "ligand_ccd": args.ligand_ccd,
        "chains_all": all_chains,
        "chains_kept": sorted(keep),
        "chains_dropped": sorted(set(all_chains) - keep),
        "binding_site_residues": sorted([list(r) for r in site]),
        "ligand_atoms": len(ligand),
        "receptor_atoms": len(receptor),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"{args.structure}")
    print(f"  ligand         {found['ligand']}  {len(ligand)} atoms")
    print(f"  template       {args.ligand_ccd or '(explicit SMILES)'}")
    print(f"                 {Chem.MolToSmiles(ligand_mol)}")
    print(f"  MIDAS metal    {found['metal']}  at {found['distance']:.2f} A")
    print(f"  chains kept    {sorted(keep)}")
    print(f"  chains dropped {sorted(set(all_chains) - keep) or '(none)'}")
    print(f"  binding site   {len(site)} residues within {args.site_cutoff} A")
    print(f"  -> {out}/receptor.pdb, ligand.sdf, manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
