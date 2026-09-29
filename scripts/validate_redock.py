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
import shutil
import statistics
import sys
from pathlib import Path

from pose_geometry import CA_CUTOFF, DONOR_CUTOFF

__all__ = [
    "CRYSTAL_CONTACTS",
    "RMSD_CUTOFF",
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
# CLI: real redocking, the protonation decision, and the engine-reproducibility
# measurement (spec section 7.4/7.7) all require autodock_gpu, which is not
# installed anywhere on this machine (checked: not on PATH, not in any conda
# env). This CLI is the intended entry point for those three runs once the
# binary is available (see results/v2_EXECUTION_CHECKLIST.md for the exact
# commands and the pre-registered protonation rule); it refuses to fabricate
# output and exits with a pointer to the checklist instead of pretending to
# dock. No docking output or affinity number is invented anywhere below.
# --------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protonation", choices=["neutral", "anion"],
                    help="Redock the control ligand in this protonation state "
                         "and apply the gate (Step 5, protonation decision).")
    p.add_argument("--reproducibility", action="store_true",
                    help="Measure engine affinity reproducibility across seeds "
                         "(Step 6, spec section 7.7).")
    p.add_argument("--ligands", nargs="+", default=[],
                    help="Ligand ids to redock (e.g. A1AFA PLN-1474).")
    p.add_argument("--seeds", default=None,
                    help="Seed range for --reproducibility, e.g. '1..10'.")
    p.add_argument("--out", type=Path, help="Where to write the JSON result.")
    return p


def main(argv=None) -> int:
    _args = _build_arg_parser().parse_args(argv)

    if shutil.which("autodock_gpu") is None:
        sys.stderr.write(
            "autodock_gpu not found on PATH. Real redocking, the protonation "
            "decision, and the engine reproducibility measurement all require "
            "it; none can run on this machine. See the Task 10 entries in "
            "results/v2_EXECUTION_CHECKLIST.md for the exact commands to run "
            "once the binary is available on K-BDS. Refusing to fabricate "
            "docking output or affinity numbers.\n"
        )
        return 1

    # Real execution path (unreachable here, but left intact for K-BDS):
    # dock via scripts/dock_autodock_gpu.py, parse with parse_dlg, measure
    # with pose_geometry.measure_pose / crystal_rmsd, run PLIP on the
    # resulting complex, and feed the results into verdict()/reproducibility()
    # above. Deliberately not implemented further here - see the checklist.
    raise NotImplementedError(
        "autodock_gpu is present but the redocking pipeline wiring itself "
        "was not exercised on this machine; see results/v2_EXECUTION_CHECKLIST.md"
    )


if __name__ == "__main__":
    sys.exit(main())
