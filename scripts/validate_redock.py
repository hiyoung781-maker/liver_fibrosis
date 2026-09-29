"""Redocking validation gate (spec section 7.4).

Decides whether AutoDock-GPU is trusted at all: redock the crystal ligand and
check that the redocked pose reproduces the two observed contacts from two
independent measurements each - distance (a/b, via pose_geometry) and PLIP
(d) - plus a symmetry-aware RMSD to the crystal pose. Distance and PLIP check
the same two contacts by different methods and can disagree: distance alone
does not see angles, so a pose can sit inside a distance cutoff and still not
be the interaction PLIP recognizes (e.g. an H-bond donor angle that fails
PLIP's geometric test). When they disagree, this module records the
disagreement in the verdict and follows PLIP, since PLIP is the method that
actually checks the geometry distance cannot.

If this gate does not pass, nothing below it can be trusted: AutoDock-GPU
would not be filtering compounds on real structural evidence, and the plan
falls back to the Uni-Dock regime for this campaign.

CRYSTAL_CONTACTS below are measured on the deposited 8W30 structure with
PLIP 3.0.1: ligand-to-MIDAS-calcium 2.62 A (coordination 4, metal type Ca;
same shell also carries Ser132 2.45 A, Ser134 2.50 A, Glu229 2.39 A - those
three are not part of this gate, which only checks the ligand-Ca distance
itself), and the Asn224 hydrogen bond at a 2.63 A donor-acceptor distance
with a 156.61 degree donor angle, ligand donating (the protein is NOT the
donor). PLIP truncates the five-character CCD code A1AFA to the three-
character hetid A1A and composes the site as A1A-CA (ligand and calcium
merged) - do not assume the hetid is A1AFA when parsing PLIP XML elsewhere.
PLIP reports ZERO pi-stacks for 8W30 (its default centroid cutoff is 5.5 A;
8W30's Tyr178 centroid distance is 6.21 A, so Tyr178 comes back only as a
hydrophobic contact at 3.70 A, same shell as Leu225 at 3.69 A) - this gate
therefore checks only the metal complex and the Asn224 H-bond, not a
pi-stack that PLIP does not report for this structure.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from pose_geometry import CA_CUTOFF, DONOR_CUTOFF

__all__ = [
    "CRYSTAL_CONTACTS",
    "RMSD_CUTOFF",
    "best_pose",
    "measure_redocked_pose",
    "verdict",
    "reproducibility",
]

# 8W30, measured with PLIP 3.0.1 on the deposited structure (see module
# docstring). This is the bar a redocked pose must reach; the gate itself
# (verdict, below) must be no more permissive than these documented values.
CRYSTAL_CONTACTS = {"ca_dist": 2.62, "donor_dist": 2.63}

# rdMolAlign.CalcRMS (symmetry- and mapping-aware), never index-wise coordinate
# subtraction - see pose_geometry.py's docstring for why a naive RMSD turned a
# real 0.63 A success into an apparent 6.13 A failure on this exact ligand.
RMSD_CUTOFF = 2.0


def verdict(measurement: dict) -> dict:
    """Apply the four-criterion redock gate to one pose's measurements.

    measurement keys: ca_dist, donor_dist, rmsd (all floats/distances in
    Angstrom), plip_metal_ca501, plip_hbond_asn224 (both bool, from PLIP's
    independent interaction report on the same redocked pose).

    Returns {"passed": bool, "failed": [...], "disagreements": [...]}.
    "disagreements" holds a contact where the distance criterion passed but
    PLIP did not confirm it - distance criteria don't see angles, PLIP does,
    so this is the observable signature of an angle failure. Any entry in
    "disagreements" also appears in "failed", since PLIP's judgment governs.
    """
    failed: list[str] = []
    disagreements: list[str] = []

    if measurement["ca_dist"] > CA_CUTOFF:
        failed.append("ca_dist")
    if measurement["donor_dist"] > DONOR_CUTOFF:
        failed.append("donor_dist")
    if measurement["rmsd"] >= RMSD_CUTOFF:
        failed.append("rmsd")
    if not measurement["plip_metal_ca501"]:
        failed.append("plip_metal_ca501")
    if not measurement["plip_hbond_asn224"]:
        failed.append("plip_hbond_asn224")

    # Distance passed but PLIP disagrees: the real-world case is a pose whose
    # Asn224 donor-acceptor distance is inside DONOR_CUTOFF but whose donor
    # angle fails PLIP's H-bond geometry test (PLIP checks angles; the raw
    # distance criterion above does not). Record it, and PLIP's "failed"
    # entry above already makes the gate follow PLIP's judgment.
    if measurement["ca_dist"] <= CA_CUTOFF and not measurement["plip_metal_ca501"]:
        disagreements.append("ca_dist")
    if measurement["donor_dist"] <= DONOR_CUTOFF and not measurement["plip_hbond_asn224"]:
        disagreements.append("donor_dist")

    return {"passed": not failed, "failed": failed, "disagreements": disagreements}


def best_pose(poses: list[dict]) -> dict:
    """The TOP pose of a .dlg, by affinity - never simply poses[0].

    spec section 7.4 applies the gate to the top pose. AutoDock-GPU writes
    its DOCKED blocks in RUN order, not energy order: the control ligand's
    own .dlg opens -5.89, -6.23, -6.30, -6.30, -6.02, so poses[0] is the
    worst of the first five. Sorting is therefore explicit here rather than
    left to the file's order.

    Poses whose energy line did not parse carry affinity None. parse_dlg
    keeps them deliberately (a v1 regression put score extraction inside the
    structure-parsing try and dropped the pose count to zero), so they are
    kept here too - sorted last, never selected over a scored pose, never
    discarded. Ties break on the lower rank so the choice is reproducible
    regardless of sort stability.
    """
    if not poses:
        raise ValueError(
            "no DOCKED poses in this .dlg. That is a DOCKING failure, not a "
            "gate failure - returning a verdict here would report it as the "
            "latter. Check the shard's autodock_gpu log."
        )
    return min(
        poses,
        key=lambda p: (p.get("affinity") is None,
                       p.get("affinity") if p.get("affinity") is not None else 0.0,
                       p.get("rank", 0)),
    )


def reproducibility(affinities: list[float]) -> dict:
    """Standard deviation and range of best-passing-pose affinity across seeds.

    Affinity is now a hard filter rather than a tie-break (spec section 7.7):
    in the v1 campaign, 5 of 20 leads sat within 0.05 kcal/mol of the cutoff,
    smaller than the 0.16 kcal/mol difference measured between two engines.
    This function reduces a list of same-ligand, seed-varied affinities (one
    "best passing pose" value per seed) to the two numbers the lead report
    needs to say how many molecules sit within engine noise of the cutoff.
    """
    if len(affinities) < 2:
        raise ValueError("need at least 2 seeds to measure reproducibility")
    return {
        "sd": statistics.stdev(affinities),
        "range": max(affinities) - min(affinities),
        "n": len(affinities),
    }


# --------------------------------------------------------------------------
# CLI: apply the pre-registered gate to a pose AutoDock-GPU has already
# produced. The control ligand is docked as part of the ordinary batch (it
# appears in each arm's ligand index as CONTROL_crystal), so this CLI reads
# that .dlg rather than invoking the docking engine itself: re-docking here
# would gate on a different run than the one the campaign actually used.
#
# ENVIRONMENT. This module needs RDKit (pose_geometry) in the same process
# and Open Babel + PLIP as subprocesses. On this machine those live in
# different conda envs, so run it as:
#
#   PATH=$HOME/miniconda3/envs/docking/bin:$PATH \
#   ~/miniconda3/envs/reinvent/bin/python scripts/validate_redock.py ...
#
# No docking output or affinity number is invented anywhere below; every
# number in the report is read from the .dlg, measured from its coordinates,
# or reported by PLIP.
# --------------------------------------------------------------------------

_PLIP_HETID = "LIG"  # poses_to_complex.LIGAND_RESNAME, as PLIP reports it


def _pose_to_sdf(pdbqt_path: Path, sdf_path: Path) -> None:
    subprocess.run(["obabel", str(pdbqt_path), "-osdf", "-O", str(sdf_path)],
                   check=True, capture_output=True, timeout=120)


def _load_reference(reference_sdf: str):
    from rdkit import Chem
    ref = next(iter(Chem.SDMolSupplier(reference_sdf, removeHs=False)), None)
    if ref is None:
        raise ValueError(f"{reference_sdf}: RDKit could not read a reference molecule")
    return ref


def _rmsd_to_reference(pose_mol, reference_mol):
    """Symmetry-aware RMSD, or (None, reason) when the graphs cannot be matched.

    CalcRMS needs the two molecules to be the same graph. The pose arrives
    through PDBQT -> Open Babel, whose perceived bond orders need not match
    the reference SDF's, and CalcRMS then raises rather than returning a
    number. Assigning the reference's bond orders onto the pose fixes the
    common case; when even that fails, this returns no number and names the
    reason, because a fabricated RMSD would silently decide criterion (c).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from pose_geometry import crystal_rmsd

    try:
        return crystal_rmsd(pose_mol, reference_mol), None
    except (ValueError, RuntimeError) as exc:
        first = str(exc)
    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(reference_mol, pose_mol)
        return crystal_rmsd(fixed, reference_mol), None
    except (ValueError, RuntimeError) as exc:
        return None, (f"CalcRMS could not match the pose to the reference "
                      f"({first}); after AssignBondOrdersFromTemplate: {exc}")


def measure_redocked_pose(pose: dict, receptor_pdb: str, reference_sdf: str,
                          workdir: Path, plip_bin: str) -> dict:
    """Measure one parsed .dlg pose against all four criteria of spec 7.4.

    Returns the dict verdict() consumes (ca_dist, donor_dist, rmsd,
    plip_metal_ca501, plip_hbond_asn224) plus the diagnostics the report
    needs: affinity, rank, the PLIP record's own missing-criteria list, and
    rmsd_note when the RMSD could not be computed.
    """
    from rdkit import Chem

    from pose_geometry import anchor_atoms, measure_pose
    from poses_to_complex import build_complex
    from run_plip import parse_report, run_plip
    from validate_plip import crystal_verdict

    workdir.mkdir(parents=True, exist_ok=True)
    stem = f"pose_{pose['rank']:03d}"
    pose_pdbqt = workdir / f"{stem}.pdbqt"
    pose_pdbqt.write_text(pose["pdbqt_block"] + "\n")

    pose_sdf = workdir / f"{stem}.sdf"
    _pose_to_sdf(pose_pdbqt, pose_sdf)
    mol = next(iter(Chem.SDMolSupplier(str(pose_sdf), removeHs=False)), None)
    if mol is None:
        raise ValueError(f"{pose_sdf}: RDKit could not read the converted pose")

    geometry = measure_pose(mol, anchor_atoms(receptor_pdb))
    rmsd, rmsd_note = _rmsd_to_reference(mol, _load_reference(reference_sdf))

    complex_pdb = build_complex(str(pose_pdbqt), receptor_pdb,
                                workdir / f"{stem}.complex.pdb")
    if complex_pdb is None:
        raise ValueError(f"{pose_pdbqt}: complex build failed; PLIP cannot be run")
    xml_path = run_plip(complex_pdb, str(workdir / f"{stem}_plip"),
                        plip_bin=plip_bin)
    if xml_path is None:
        raise ValueError(f"{complex_pdb}: PLIP failed to run; see run_plip.run_plip")
    record = parse_report(Path(xml_path).read_text(), hetid=_PLIP_HETID)
    plip = crystal_verdict(record)

    return {
        "rank": pose["rank"],
        "affinity": pose["affinity"],
        "ca_dist": geometry["ca_dist"],
        "ca_dist_far": geometry["ca_dist_far"],
        "donor_dist": geometry["donor_dist"],
        # A missing RMSD must not pass criterion (c) by default: inf fails it
        # the same way a bad pose would, and rmsd_note says why.
        "rmsd": rmsd if rmsd is not None else float("inf"),
        "rmsd_note": rmsd_note,
        "plip_metal_ca501": "metal_ca501" not in plip["missing"],
        "plip_hbond_asn224": "hbond_asn224" not in plip["missing"],
        "plip_missing": plip["missing"],
        "complex_pdb": complex_pdb,
        "plip_xml": xml_path,
    }


def _gate_report(label: str, dlg: str, measurement: dict, decision: dict,
                 n_poses: int) -> str:
    ok = decision["passed"]
    lines = [
        "# Redocking validation gate (spec section 7.4)",
        "",
        f"Control ligand `.dlg`: `{dlg}`",
        f"Poses parsed: {n_poses}. Gate applied to the TOP pose by affinity "
        f"(rank {measurement['rank']}, "
        f"{measurement['affinity']} kcal/mol).",
    ]
    if label:
        lines.append(f"Protonation state docked: **{label}**")
    lines += [
        "",
        f"Verdict: **{'PASS' if ok else 'FAIL'}**",
        "",
        "| criterion | measured | threshold | result |",
        "|---|---|---|---|",
        f"| (a) carboxylate O to Ca501 | {measurement['ca_dist']:.2f} A | "
        f"<= {CA_CUTOFF} A | {'pass' if 'ca_dist' not in decision['failed'] else 'FAIL'} |",
        f"| (b) donor to Asn224 backbone O | {measurement['donor_dist']:.2f} A | "
        f"<= {DONOR_CUTOFF} A | {'pass' if 'donor_dist' not in decision['failed'] else 'FAIL'} |",
        f"| (c) symmetry-aware RMSD (CalcRMS) | {measurement['rmsd']:.2f} A | "
        f"< {RMSD_CUTOFF} A | {'pass' if 'rmsd' not in decision['failed'] else 'FAIL'} |",
        f"| (d) PLIP Ca501 metal complex | "
        f"{'reported' if measurement['plip_metal_ca501'] else 'not reported'} | "
        f"required | {'pass' if 'plip_metal_ca501' not in decision['failed'] else 'FAIL'} |",
        f"| (d) PLIP Asn224 H-bond | "
        f"{'reported' if measurement['plip_hbond_asn224'] else 'not reported'} | "
        f"required | {'pass' if 'plip_hbond_asn224' not in decision['failed'] else 'FAIL'} |",
        "",
        f"Crystal reference: Ca {CRYSTAL_CONTACTS['ca_dist']} A, "
        f"Asn224 donor-acceptor {CRYSTAL_CONTACTS['donor_dist']} A "
        "(PLIP 3.0.1 on the deposited 8W30 structure).",
    ]
    if measurement.get("rmsd_note"):
        lines += ["", f"RMSD not computed: {measurement['rmsd_note']} "
                       "Criterion (c) is recorded as failed rather than assumed."]
    if decision["disagreements"]:
        lines += [
            "",
            "## Distance/PLIP disagreement",
            "",
            "These contacts are inside their distance cutoff but PLIP does not "
            "confirm them. Distance criteria do not check angles and PLIP does, "
            "so this is the observable form of the angle problem section 8.0 "
            "raised. Per spec 7.4 the gate follows PLIP: "
            + ", ".join(decision["disagreements"]) + ".",
        ]
    lines += ["", "---", ""]
    if ok:
        lines.append(
            "AutoDock-GPU reproduces the deposited pose. It is adopted as this "
            "campaign's docking engine.")
    else:
        lines.append(
            "Failed: " + ", ".join(decision["failed"]) + ". Per spec 7.4, "
            "**AutoDock-GPU ranks and filters nothing**; the campaign reverts "
            "to the Uni-Dock regime and this file is the record of why.")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dlg", nargs="+", required=True,
                    help="AutoDock-GPU .dlg for the redocked control ligand. "
                         "With --reproducibility, one .dlg per seed.")
    p.add_argument("--receptor", default="docking/v2/receptor_h.pdb",
                    help="Protonated receptor PDB: supplies the Ca501/Asn224 "
                         "anchors AND is the receptor half of the PLIP complex.")
    p.add_argument("--reference", default="docking/ligand_ref.sdf",
                    help="Crystal ligand pose, for the symmetry-aware RMSD.")
    p.add_argument("--plip-bin", default="plip", help="Path to the PLIP executable.")
    p.add_argument("--workdir", type=Path, default=None,
                    help="Where per-pose intermediates are written "
                         "(default: a temporary directory that is discarded).")
    p.add_argument("--protonation", choices=["neutral", "anion"], default=None,
                    help="Label recording which protonation state of the control "
                         "was docked. The state is decided in ligand preparation "
                         "upstream; this only records it in the report.")
    p.add_argument("--reproducibility", action="store_true",
                    help="Seed-variation mode (spec 7.7): reduce the top-pose "
                         "affinity of each --dlg to sd and range.")
    p.add_argument("--out", type=Path, required=True,
                    help="Markdown report path (JSON is written alongside it).")
    return p


def _run_gate(args, workdir: Path) -> int:
    from dock_autodock_gpu import parse_dlg

    dlg = args.dlg[0]
    poses = parse_dlg(Path(dlg).read_text())
    top = best_pose(poses)
    measurement = measure_redocked_pose(top, args.receptor, args.reference,
                                        workdir, args.plip_bin)
    decision = verdict(measurement)
    report = _gate_report(args.protonation, dlg, measurement, decision, len(poses))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"dlg": dlg, "n_poses": len(poses), "protonation": args.protonation,
         "measurement": measurement, "verdict": decision}, indent=2) + "\n")
    print(report)
    return 0 if decision["passed"] else 1


def _run_reproducibility(args, workdir: Path) -> int:
    from dock_autodock_gpu import parse_dlg

    per_seed = []
    for dlg in args.dlg:
        poses = parse_dlg(Path(dlg).read_text())
        top = best_pose(poses)
        if top["affinity"] is None:
            raise ValueError(f"{dlg}: top pose carries no parsed affinity; "
                             "reproducibility cannot be measured from it.")
        per_seed.append({"dlg": dlg, "affinity": top["affinity"],
                         "rank": top["rank"], "n_poses": len(poses)})

    stats = reproducibility([s["affinity"] for s in per_seed])
    lines = [
        "# AutoDock-GPU affinity reproducibility across seeds (spec section 7.7)",
        "",
        f"n seeds: {stats['n']}   sd: {stats['sd']:.3f} kcal/mol   "
        f"range: {stats['range']:.3f} kcal/mol",
        "",
        "| .dlg | top pose | affinity (kcal/mol) |",
        "|---|---|---|",
    ]
    lines += [f"| `{s['dlg']}` | rank {s['rank']} of {s['n_poses']} | "
              f"{s['affinity']:.2f} |" for s in per_seed]
    lines += [
        "",
        "Affinity is a hard FILTER in this campaign, not a tie-break. These two "
        "numbers say how wide a band around the PLN-1474 cutoff sits inside "
        "engine noise; molecules inside that band are reported as such rather "
        "than counted as passes or failures.",
    ]
    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"per_seed": per_seed, "stats": stats}, indent=2) + "\n")
    print(report)
    return 0


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    missing = [f for f in ([args.receptor] if not args.reproducibility else [])
               + ([args.reference] if not args.reproducibility else [])
               + list(args.dlg) if not Path(f).exists()]
    if missing:
        sys.stderr.write("missing input file(s): " + ", ".join(missing) + "\n")
        return 2

    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        return (_run_reproducibility if args.reproducibility else _run_gate)(
            args, args.workdir)
    with tempfile.TemporaryDirectory() as tmp:
        return (_run_reproducibility if args.reproducibility else _run_gate)(
            args, Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
