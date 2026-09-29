"""Independent RMSD check: index-wise correspondence, no graph matching (§7.4, §10.3).

RMSD measurement has been the failure point three times in this campaign, and
twice the number was wrong in the direction that decided a gate:

  - a naive index-wise RMSD reported 6.13 A for a pose whose true value was
    0.63 A, because Open Babel reorders atoms and the ligand has symmetric
    rings (v1);
  - CalcRMS returned 2.151 A for an alphaVbeta1 pose while AutoDock's own
    table said 1.46 A, because Open Babel re-protonated the carboxylate on
    the way out of the .dlg and the two molecules were not the same graph.

This module measures the one correspondence that needs no graph matching at
all. A docked pose carries its input ligand's ATOM ORDER, so lining the two
files up index by index IS the correct mapping, and the check is simply
whether the atom names agree in order.

WHAT THE COMPARISON PROVES. CalcRMS minimises over every symmetry-equivalent
mapping, so its result can never EXCEED the index-wise value. If it does, it
did not find the correct correspondence and the gate was decided on a number
that measures nothing. If the two agree, the gate's value stands.

CENTROID SHIFT IS REPORTED BESIDE IT, because a large RMSD has two very
different causes. A pose that flipped in place keeps its centroid and can
still make the anchor contacts; a pose that moved to another pocket does not.
The verdict is the same either way, but what the failure MEANS is not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["heavy_atoms", "index_wise_rmsd", "compare"]

# PDBQT merges nonpolar hydrogens into their heavy atoms; HD and HS are the
# polar ones Meeko keeps. Both gates measure heavy-atom RMSD.
_HYDROGEN_TYPES = {"H", "HD", "HS"}


def heavy_atoms(lines) -> list:
    """(atom name, xyz) for every non-hydrogen ATOM/HETATM record, in file order."""
    out = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = (line[77:79].strip() or line[76:78].strip()
                   or line[12:16].strip()[:1]).upper()
        if element in _HYDROGEN_TYPES:
            continue
        out.append((line[12:16].strip(),
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    return out


def index_wise_rmsd(pose: list, reference: list) -> dict:
    """RMSD over the index correspondence, plus whether that correspondence holds.

    `names_match` is the honest part: when it is False the index mapping is
    an assumption, not a fact, and the number below it means nothing.
    """
    names_match = [a[0] for a in pose] == [a[0] for a in reference]
    if len(pose) != len(reference):
        return {"names_match": names_match, "n_pose": len(pose),
                "n_reference": len(reference), "rmsd": None,
                "centroid_shift": None,
                "note": "different heavy-atom counts; no correspondence exists"}
    p = np.array([a[1] for a in pose])
    r = np.array([a[1] for a in reference])
    return {
        "names_match": names_match,
        "n_pose": len(pose),
        "n_reference": len(reference),
        "rmsd": float(np.sqrt(((p - r) ** 2).sum(1).mean())),
        "centroid_shift": float(np.linalg.norm(p.mean(0) - r.mean(0))),
        "max_atom_displacement": float(np.linalg.norm(p - r, axis=1).max()),
        "note": None,
    }


def compare(poses_pdbqt: str, reference_pdbqt: str, top_n: int = 5) -> dict:
    """Measure every pose against the reference, best affinity first."""
    from dock_unidock import parse_unidock_pdbqt

    reference = heavy_atoms(Path(reference_pdbqt).read_text().splitlines())
    poses = parse_unidock_pdbqt(Path(poses_pdbqt).read_text())
    if not poses:
        raise ValueError(f"{poses_pdbqt}: no MODEL blocks")

    rows = []
    for pose in poses:
        measured = index_wise_rmsd(
            heavy_atoms(pose["pdbqt_block"].splitlines()), reference)
        measured.update(rank=pose["rank"], affinity=pose["affinity"])
        rows.append(measured)
    rows.sort(key=lambda r: (r["affinity"] is None, r["affinity"] or 0.0))
    return {"reference": reference_pdbqt, "poses": poses_pdbqt,
            "n_poses": len(rows), "rows": rows, "top_n": top_n}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--poses", required=True, help="Uni-Dock output PDBQT.")
    p.add_argument("--reference", required=True,
                    help="The prepared crystal ligand PDBQT that was docked.")
    p.add_argument("--gate-rmsd", type=float, default=None,
                    help="The RMSD the gate reported, to compare against. "
                         "CalcRMS minimises over symmetry, so a gate value "
                         "ABOVE the index-wise one means it found no valid "
                         "correspondence.")
    p.add_argument("--top-n", type=int, default=5,
                    help="How many poses to print, best affinity first.")
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    result = compare(args.poses, args.reference)
    rows = result["rows"]
    top = rows[0]

    print(f"\n=== {args.label or args.poses} ===")
    print(f"  heavy atoms: pose {top['n_pose']}  reference {top['n_reference']}")
    print(f"  atom names identical in order: {top['names_match']}")
    if not top["names_match"]:
        print("    the index correspondence is an ASSUMPTION here, not a fact; "
              "the numbers below do not measure what they claim")
    print(f"\n  {'':>2} {'pose':>5} {'affinity':>9} {'index RMSD':>11} "
          f"{'centroid':>9} {'max atom':>9}")
    for row in rows[:args.top_n]:
        mark = "<-" if row is top else ""
        rmsd = "n/a" if row["rmsd"] is None else f"{row['rmsd']:.3f}"
        cent = "n/a" if row["centroid_shift"] is None else f"{row['centroid_shift']:.3f}"
        mx = "n/a" if row.get("max_atom_displacement") is None else \
            f"{row['max_atom_displacement']:.3f}"
        print(f"  {mark:>2} {row['rank']:>5} {row['affinity']:>9.3f} "
              f"{rmsd:>11} {cent:>9} {mx:>9}")

    best = min((r for r in rows if r["rmsd"] is not None),
               key=lambda r: r["rmsd"], default=None)
    if best is not None and best is not top:
        print(f"\n  closest pose to the crystal: rank {best['rank']} at "
              f"{best['rmsd']:.3f} A, affinity {best['affinity']:.3f} "
              f"({best['affinity'] - top['affinity']:+.3f} vs the top pose)")

    if args.gate_rmsd is not None and top["rmsd"] is not None:
        print(f"\n  gate reported {args.gate_rmsd:.3f} A; index-wise "
              f"{top['rmsd']:.3f} A")
        if args.gate_rmsd > top["rmsd"] + 0.05:
            print("    the gate value is ABOVE the index-wise correspondence. "
                  "CalcRMS minimises over symmetry-equivalent mappings, so it "
                  "cannot legitimately exceed it: the gate found no valid "
                  "correspondence and its verdict rests on a number that "
                  "measures nothing. RE-JUDGE.")
        else:
            print("    consistent: CalcRMS is at or below the index-wise value, "
                  "which is what symmetry correction does. The gate value stands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
