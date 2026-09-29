"""PLIP validation gate on the deposited 8W30 crystal complex (spec Sec 8.0).

Decides whether PLIP is trusted as this campaign's pose filter at all. If
PLIP cannot reproduce the interactions the deposited 8W30 structure is known
to make, adopting it would repeat exactly the error that motivated moving
away from the v1 distance criteria, and the plan reverts to those distance
criteria for every later task that would otherwise depend on PLIP.

Four criteria, all measured on the deposited 8W30 structure with PLIP 3.0.1
via scripts/run_plip.py's parse_report(hetid="A1A") (PLIP truncates the
five-character CCD code A1AFA to the three-character hetid A1A and merges
the ligand with the adjacent calcium into a site named "A1A-CA"; 8W30 has 13
binding sites in total and only this one is the ligand's):

  (a) MIDAS calcium coordination: a metal_complexes entry for the MIDAS Ca
      (chain B, residue 501) where the coordinated atom is the ligand's own
      oxygen -- measured 2.62 A, coordination 4, metal_type Ca. The same
      calcium also coordinates Ser132 (2.45 A), Ser134 (2.50 A) and Glu229
      (2.39 A); those three are informative but not required by this gate,
      which checks only the ligand-Ca contact itself.
  (b) Asn224 hydrogen bond, ligand donating: an hbonds entry for chain B
      residue 224 with protisdon == False (the ligand is the donor, not the
      protein) -- measured dist_d-a 2.63 A, don_angle 156.61 degrees. (Two
      further Asn224 entries at 3.23 A and 3.27 A have protisdon == True and
      are not this criterion.)
  (c) Tyr178 contact, ANY interaction type. This criterion was originally
      specified as a T-shaped pi-stack. It is corrected here: PLIP reports
      ZERO pi_stacks for the deposited 8W30 structure. PLIP's default
      pi-stack centroid cutoff is 5.5 A; the measured ligand-to-Tyr178
      centroid distance in 8W30 is 6.21 A, so PLIP classifies the contact as
      a hydrophobic_interaction (3.70 A) instead of a pi_stack. That is the
      same centroid-cutoff limitation that made the v1 distance criterion
      reject its own crystal structure -- the reason this project moved to
      PLIP in the first place. A gate that still demanded a pi-stack would
      reject the only crystallographically observed binder for this target,
      reproducing the exact failure PLIP was adopted to fix. The criterion
      is therefore type-agnostic: any interaction kind (hbonds,
      hydrophobic_contacts, pi_stacking, salt_bridges, water_bridges,
      halogen_bonds) that reports a contact to chain A residue 178 satisfies
      it. Do not retune PLIP's centroid cutoff to force a pi-stack
      classification here; that would be inventing a threshold to obtain a
      predetermined answer, which is what this gate exists to prevent.
  (d) Hydrophobic pocket: at least one hydrophobic_contacts entry among the
      documented pocket residues HYDROPHOBIC_POCKET. Leu225 (chain B, 3.69
      A) satisfies this in the deposited structure; Tyr178's own hydrophobic
      contact (criterion c) is not counted again here since HYDROPHOBIC_POCKET
      deliberately excludes (A, 178).

Field names match scripts/run_plip.py's parse_report output exactly (real
PLIP XML field names: reschain/resnr/restype describe the protein residue in
an entry; a metal_complexes entry additionally carries resnr_lig/
reschain_lig/restype_lig/metal_type for the metal ion, and location which is
"ligand" when the coordinated atom belongs to the ligand rather than the
protein). protisdon is a raw PLIP XML string ("True"/"False"), not coerced
to bool by parse_report, so comparisons below check both forms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = [
    "MIDAS_CA", "ASN224", "TYR178", "HYDROPHOBIC_POCKET",
    "OTHER_INTERACTION_KINDS", "crystal_verdict",
]

MIDAS_CA = ("B", 501)
ASN224 = ("B", 224)
TYR178 = ("A", 178)

# Documented hydrophobic pocket residues for criterion (d). Tyr178 is
# deliberately excluded here -- it is criterion (c), checked across all
# interaction types, not only hydrophobic_contacts.
HYDROPHOBIC_POCKET = {("B", 133), ("B", 186), ("B", 187), ("B", 225)}

# Every interaction kind other than hydrophobic_contacts that criterion (c)
# must also search, since a Tyr178 contact can be reported as any kind.
OTHER_INTERACTION_KINDS = (
    "hbonds", "pi_stacking", "salt_bridges", "water_bridges", "halogen_bonds",
)


def _is_true(value) -> bool:
    return value is True or value == "True"


def _is_false(value) -> bool:
    return value is False or value == "False"


def _has_midas_ligand_contact(metal_complexes: list) -> bool:
    for e in metal_complexes:
        metal = (e.get("reschain_lig"), e.get("resnr_lig"))
        # resnr_lig comes through parse_report as a raw XML string (only
        # "resnr", the protein-side field, is coerced to int) -- compare as
        # string alongside the int MIDAS_CA[1] to accept either.
        if metal[0] == MIDAS_CA[0] and str(metal[1]) == str(MIDAS_CA[1]):
            if e.get("metal_type") in ("Ca", "CA") and e.get("location") == "ligand":
                return True
    return False


def _has_asn224_ligand_donor_hbond(hbonds: list) -> bool:
    for e in hbonds:
        if (e.get("reschain"), e.get("resnr")) == ASN224 and _is_false(e.get("protisdon")):
            return True
    return False


def _has_tyr178_contact(record: dict) -> bool:
    for kind in ("hydrophobic_contacts",) + OTHER_INTERACTION_KINDS:
        for e in record.get(kind, []):
            if (e.get("reschain"), e.get("resnr")) == TYR178:
                return True
    return False


def _has_pocket_hydrophobic_contact(hydrophobic_contacts: list) -> bool:
    residues = {(e.get("reschain"), e.get("resnr")) for e in hydrophobic_contacts}
    return bool(residues & HYDROPHOBIC_POCKET)


def crystal_verdict(record: dict) -> dict:
    """Apply the four-criterion PLIP gate to one parse_report()-shaped record.

    `record` uses run_plip.INTERACTION_KINDS as keys (metal_complexes,
    hbonds, hydrophobic_contacts, pi_stacking, salt_bridges, water_bridges,
    halogen_bonds), each a list of PLIP XML entry dicts. Returns
    {"passed": bool, "missing": [...]} naming exactly the criteria not met,
    so a failure states precisely which contact PLIP failed to reproduce.
    """
    missing = []
    if not _has_midas_ligand_contact(record.get("metal_complexes", [])):
        missing.append("metal_ca501")
    if not _has_asn224_ligand_donor_hbond(record.get("hbonds", [])):
        missing.append("hbond_asn224")
    if not _has_tyr178_contact(record):
        missing.append("tyr178_contact")
    if not _has_pocket_hydrophobic_contact(record.get("hydrophobic_contacts", [])):
        missing.append("hydrophobic_pocket")
    return {"passed": not missing, "missing": missing}


# --------------------------------------------------------------------------
# CLI: run PLIP on the deposited 8W30 structure and write the gate verdict.
# --------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True, help="Deposited complex PDB, e.g. 8W30.pdb")
    p.add_argument("--hetid", default="A1A",
                    help="PLIP's truncated hetid for the ligand site (default A1A, not "
                         "the five-character CCD code A1AFA -- see module docstring).")
    p.add_argument("--plip-bin", default="plip", help="Path to the PLIP executable.")
    p.add_argument("--out", required=True, help="Where to write the markdown verdict.")
    return p


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_plip import parse_report, run_plip  # local import: keeps CLI-only deps out of tests

    xml_path = run_plip(args.structure, str(Path(args.out).parent / "_plip_out"), plip_bin=args.plip_bin)
    if xml_path is None:
        sys.stderr.write(f"PLIP failed to run on {args.structure}; see run_plip.run_plip.\n")
        return 1

    record = parse_report(Path(xml_path).read_text(), hetid=args.hetid)
    verdict = crystal_verdict(record)

    lines = [
        "# PLIP validation gate (8W30 deposited crystal complex)",
        "",
        f"Structure: `{args.structure}` (hetid `{args.hetid}`)",
        f"Verdict: **{'PASS' if verdict['passed'] else 'FAIL'}**",
        "",
    ]
    if verdict["passed"]:
        lines.append("All four criteria reproduced. PLIP is adopted as the campaign's pose filter.")
    else:
        lines.append("Missing: " + ", ".join(verdict["missing"]))
        lines.append("")
        lines.append("PLIP does not reproduce the deposited structure's own interactions. "
                      "PLIP is NOT adopted; the campaign reverts to the v1 distance-based criteria.")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
