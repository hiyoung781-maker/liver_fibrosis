"""Meeko PDBQT for an isoform receptor, from prepare_isoform's manifest (spec §10.3).

The alphaVbeta1 receptor was prepared by scripts/prepare_receptor_pdbqt.py,
whose binding-site list and anchor atoms are 8W30's own. Section 10 docks the
same ligands against alphaVbeta3 (6MK0) and alphaVbeta6 (9CZD), whose sites
are different residues and whose MIDAS ions are Mn and Mg. This driver reuses
that module's reusable parts -- the truncation of incomplete residues and the
Meeko command -- and takes everything structure-specific from the manifest
prepare_isoform.py wrote.

WHAT IS AND IS NOT CHECKED. The MIDAS ion must survive the conversion: losing
it deletes the interaction the whole campaign is about, and `obabel -p 7.4`
once removed all six calciums from the alphaVbeta1 receptor while the scores
still looked normal. Partial charges are NOT checked here. They mattered when
the engine was AutoDock-GPU, whose AD4 scoring has an electrostatic term; the
redocking gate rejected that engine and Uni-Dock's Vina scoring has no charge
term at all, so a zero-charge metal changes nothing for this campaign.

The box comes from the isoform's OWN crystal ligand, through the same
dock_unidock.box_from_ligand the production run uses. Section 10.1 compares
the SAME ligand's score across receptors and takes the difference, so a box
built by a different rule on one receptor would put a systematic offset into
exactly the quantity that is supposed to cancel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["binding_site_from_manifest", "metal_present", "prepare"]


def binding_site_from_manifest(manifest: dict) -> set:
    """(chain, resseq) pairs the truncation step must never alanine.

    prepare_isoform derives these from the structure -- every polymer residue
    within 5 A of the crystal ligand -- rather than listing them per isoform.
    An incomplete side chain among them stops the run, because the pose gate
    measures contacts to those exact side chains.
    """
    return {(chain, int(seq))
            for chain, seq in manifest.get("binding_site_residues", [])}


def metal_present(pdbqt_path: str, metal: tuple) -> bool:
    """Is the MIDAS ion still in the converted receptor?

    `metal` is (resname, chain, resseq) as the manifest records it.
    """
    resname, chain, resseq = metal[0].upper(), metal[1], str(metal[2])
    for line in Path(pdbqt_path).read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if (line[17:20].strip().upper() == resname and line[21] == chain
                and line[22:26].strip() == resseq):
            return True
    return False


def prepare(isoform_dir: str, box_radius: float = 8.0) -> dict:
    """Truncate, convert with Meeko, and verify. Returns a report dict."""
    from dock_unidock import box_from_ligand
    from prepare_receptor_pdbqt import (build_meeko_command, run_meeko,
                                        truncate_incomplete_residues)

    base = Path(isoform_dir)
    manifest = json.loads((base / "manifest.json").read_text())
    receptor_pdb = base / "receptor.pdb"
    ligand_sdf = base / "ligand.sdf"

    lines = receptor_pdb.read_text().splitlines()
    truncated, truncations = truncate_incomplete_residues(
        lines, binding_site=binding_site_from_manifest(manifest))
    truncated_pdb = base / "receptor_truncated.pdb"
    truncated_pdb.write_text("\n".join(truncated + ["END"]) + "\n")

    box = box_from_ligand(str(ligand_sdf))
    basename = str(base / "receptor")
    run_meeko(build_meeko_command(str(truncated_pdb), basename, box,
                                  box_radius=box_radius))

    pdbqt = f"{basename}.pdbqt"
    if not Path(pdbqt).exists():
        raise FileNotFoundError(f"Meeko produced no {pdbqt}")

    metal = manifest["midas_metal"]
    return {
        "isoform_dir": str(base),
        "pdbqt": pdbqt,
        "box": box,
        "truncated_residues": len(truncations),
        "binding_site_residues": len(binding_site_from_manifest(manifest)),
        "midas_metal": metal,
        "midas_present": metal_present(pdbqt, metal),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--isoform-dir", required=True,
                    help="Directory prepare_isoform.py wrote (receptor.pdb, "
                         "ligand.sdf, manifest.json).")
    p.add_argument("--box-radius", type=float, default=8.0,
                    help="Meeko --delete_bad_res_from_box_radius.")
    args = p.parse_args(argv)

    report = prepare(args.isoform_dir, args.box_radius)
    print(f"{report['isoform_dir']}")
    print(f"  box center  ({report['box']['center_x']:.3f}, "
          f"{report['box']['center_y']:.3f}, {report['box']['center_z']:.3f})")
    print(f"  box size    ({report['box']['size_x']:.3f}, "
          f"{report['box']['size_y']:.3f}, {report['box']['size_z']:.3f})")
    print(f"  truncated   {report['truncated_residues']} incomplete residues "
          f"(none of the {report['binding_site_residues']} binding-site ones)")
    print(f"  MIDAS {report['midas_metal']}  present in PDBQT: "
          f"{report['midas_present']}")
    print(f"  -> {report['pdbqt']}")

    if not report["midas_present"]:
        sys.stderr.write(
            "the MIDAS ion did not survive the conversion. Docking against "
            "this receptor would search a site with no metal, and every pose "
            "would fail the coordination criterion for a reason that is not "
            "chemical.\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
