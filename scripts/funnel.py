"""Per-ligand funnel: geometry gate, then the PLN-1474 affinity filter (§8.5b, §8.7, §9.1).

Docking produced one row per POSE in two files -- pose_geometry's distances
and plip_batch's interaction record. Selection happens per LIGAND, so this
joins them and reduces.

TWO RULES THAT ARE EASY TO GET BACKWARDS, BOTH MEASURED ON THE PANEL RUN:

1. A ligand passes section 8.5b if ANY of its poses makes both anchor
   contacts. Selection is on binding mode, not on score.

2. The section 8.7 affinity filter then reads the best PASSING pose, never
   the best pose overall. The panel makes the difference concrete:
   CHEMBL4649232 holds the panel's best affinity at -8.012 kcal/mol with
   ZERO passing poses, while PLN-1474's best passing pose is -7.290. Scoring
   on the best pose overall would promote a compound whose binding mode the
   filter had just rejected.

PLN1474_CUTOFF is that -7.290: measured in the adopted engine (Uni-Dock /
Vina), the adopted box, and the adopted protonation state (anion). It is a
RELATIVE reference, not an absolute affinity claim (spec §12), and no value
from the discarded AutoDock-GPU run can serve in its place.

POSE COUNT IS A CONFOUND, NOT A DETAIL. TL-A-prime got a median of 3 poses
per ligand and TL-C 6; 13.5% of TL-A-prime ligands got exactly one pose
against 5.2% of TL-C's. A one-pose ligand gets one chance at the geometry
gate and a fifteen-pose ligand gets fifteen, so the arms' raw pass rates
compare pose counts rather than chemistry, to TL-A-prime's systematic
disadvantage. The pass rate is therefore reported per-ligand AND stratified
by pose count. Nothing is corrected for; the stratified table is the honest
comparison and the raw rate remains the pre-registered one.

A pose that PLIP could not process costs that pose's PLIP columns, never the
ligand. Dropping the ligand would shrink the denominator of every rate here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

__all__ = ["PLN1474_CUTOFF", "POSE_BUCKETS", "ligand_rows", "stratify",
           "summarize"]

# PLN-1474's best PASSING pose, measured in the adopted engine, box and
# protonation state (results/v2_panel_geometry.csv). Section 8.7 keeps
# candidates that score BELOW this -- a tie has not surpassed the reference.
PLN1474_CUTOFF = -7.290

POSE_BUCKETS = (("1", 1, 1), ("2-4", 2, 4), ("5-7", 5, 7), ("8-15", 8, 15),
                ("16+", 16, 10**6))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read(path: str) -> list:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def ligand_rows(geometry_csv: str, plip_csv: str,
                cutoff: float = PLN1474_CUTOFF) -> list:
    """One row per ligand, joining the per-pose geometry and PLIP records."""
    plip_by_pose = {}
    for row in _read(plip_csv):
        pose = _int(row.get("pose"))
        if pose is not None:
            plip_by_pose[(row["label"], pose)] = row

    poses = defaultdict(list)
    for row in _read(geometry_csv):
        pose = _int(row.get("pose"))
        if pose is None:
            continue
        poses[row["label"]].append({
            "pose": pose,
            "affinity": _float(row.get("affinity")),
            "ca_dist": _float(row.get("ca_dist")),
            "donor_dist": _float(row.get("donor_dist")),
            "passes": _int(row.get("passes")) == 1,
        })

    rows = []
    for label, entries in poses.items():
        passing = [e for e in entries if e["passes"]]
        affinities = [e["affinity"] for e in entries if e["affinity"] is not None]
        passing_affinities = [e["affinity"] for e in passing
                              if e["affinity"] is not None]
        best_passing = min(passing_affinities) if passing_affinities else None

        metal = hbond = False
        for entry in passing:
            record = plip_by_pose.get((label, entry["pose"]))
            if record is None or record.get("status") != "ok":
                continue
            metal = metal or record.get("metal_ca501") == "1"
            hbond = hbond or record.get("hbond_asn224") == "1"

        rows.append({
            "label": label,
            "n_poses": len(entries),
            "n_passing_poses": len(passing),
            "geometry_pass": bool(passing),
            "best_affinity": min(affinities) if affinities else None,
            "best_passing_affinity": best_passing,
            "affinity_pass": best_passing is not None and best_passing < cutoff,
            "plip_metal_any_passing": metal,
            "plip_hbond_any_passing": hbond,
            "min_ca_dist": min((e["ca_dist"] for e in entries
                                if e["ca_dist"] is not None), default=None),
            "min_donor_dist": min((e["donor_dist"] for e in entries
                                   if e["donor_dist"] is not None), default=None),
        })
    rows.sort(key=lambda r: r["label"])
    return rows


def summarize(rows: list, cutoff: float = PLN1474_CUTOFF) -> dict:
    """The funnel, narrowing at each stage."""
    geometry = [r for r in rows if r["geometry_pass"]]
    affinity = [r for r in geometry if r["affinity_pass"]]
    return {
        "ligands": len(rows),
        "geometry_pass": len(geometry),
        "affinity_pass": len(affinity),
        "affinity_cutoff": cutoff,
        "plip_metal_and_hbond": sum(
            1 for r in affinity
            if r["plip_metal_any_passing"] and r["plip_hbond_any_passing"]),
        "plip_metal_only": sum(
            1 for r in affinity
            if r["plip_metal_any_passing"] and not r["plip_hbond_any_passing"]),
        "plip_neither": sum(
            1 for r in affinity
            if not r["plip_metal_any_passing"] and not r["plip_hbond_any_passing"]),
        "survivors": sorted(r["label"] for r in affinity),
    }


def stratify(rows: list) -> dict:
    """Geometry pass rate per pose-count bucket.

    The arms differ systematically in poses per ligand, and the geometry gate
    passes a ligand if any pose does, so the raw rates are not comparable
    between arms. These buckets are.
    """
    out = {}
    for name, low, high in POSE_BUCKETS:
        bucket = [r for r in rows if low <= r["n_poses"] <= high]
        if not bucket:
            continue
        passed = sum(1 for r in bucket if r["geometry_pass"])
        out[name] = {
            "ligands": len(bucket),
            "geometry_pass": passed,
            "rate": round(100.0 * passed / len(bucket), 2),
        }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--geometry", required=True, help="pose_geometry.py --out-csv")
    p.add_argument("--plip", required=True, help="plip_batch.py --out-csv")
    p.add_argument("--cutoff", type=float, default=PLN1474_CUTOFF,
                    help="Section 8.7's affinity filter: keep the best PASSING "
                         "pose strictly below this. Default is PLN-1474's own "
                         "best passing pose in the adopted engine and box.")
    p.add_argument("--label", default="", help="Arm name, for the printed header.")
    p.add_argument("--out-csv", type=Path, help="Per-ligand rows.")
    p.add_argument("--out-json", type=Path, help="Funnel summary + strata.")
    args = p.parse_args(argv)

    for path in (args.geometry, args.plip):
        if not Path(path).exists():
            sys.stderr.write(f"missing input: {path}\n")
            return 2

    rows = ligand_rows(args.geometry, args.plip, args.cutoff)
    summary = summarize(rows, args.cutoff)
    strata = stratify(rows)

    name = args.label or args.geometry
    print(f"\n=== {name} ===")
    print(f"  ligands                       {summary['ligands']}")
    print(f"  geometry gate (8.5b)          {summary['geometry_pass']}"
          f"  ({100.0 * summary['geometry_pass'] / max(summary['ligands'], 1):.1f}%)")
    print(f"  + affinity < {args.cutoff:.3f} (8.7)   {summary['affinity_pass']}")
    print(f"      PLIP metal AND H-bond     {summary['plip_metal_and_hbond']}")
    print(f"      PLIP metal only           {summary['plip_metal_only']}")
    print(f"      PLIP neither              {summary['plip_neither']}")
    print("\n  geometry pass rate by pose count "
          "(the arms differ in poses/ligand; raw rates are not comparable)")
    print(f"  {'poses':>8} {'ligands':>8} {'pass':>6} {'rate':>7}")
    for bucket, data in strata.items():
        print(f"  {bucket:>8} {data['ligands']:>8} {data['geometry_pass']:>6} "
              f"{data['rate']:>6.1f}%")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows
                                    else ["label"])
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"summary": summary, "strata": strata}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
