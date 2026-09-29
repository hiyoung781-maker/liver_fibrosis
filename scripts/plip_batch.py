"""PLIP over every docked pose, one row each (spec §8.0).

PLIP is the campaign's interaction evidence: it checks donor-H...acceptor
ANGLES, which the distance criteria in pose_geometry cannot, and it reports
the MIDAS metal coordination that the Vina scoring function has no term for.
This module runs it across a whole docking arm and writes one row per pose.

PER POSE, NOT PER LIGAND. A ligand passes section 8.5b if ANY of its poses
makes the two anchor contacts, so the pose is the unit that has to be
recorded. Roughly 53,000 of them across the two arms.

WHY RAW COUNTS TRAVEL WITH THE VERDICT. The four section 8.0 criteria come
from validate_plip.crystal_verdict, so this batch and the crystal gate cannot
drift apart. But the compound 25 gate decision has not been made yet, and a
criterion that was not recorded cannot be applied afterwards -- v1 discarded
42,419 poses and could then apply no criterion retrospectively at all. The
per-kind counts are therefore written for every pose regardless of verdict.

CPU, NOT GPU. PLIP has no GPU implementation; run this on a cpu64-only node.
The 8gpu partition bills 8 node-hours per wall hour whether or not a GPU is
touched, and the guide's CPU queues are cpu32-only and cpu64-only.

RESUMABLE. 53,000 poses is long enough that a run gets interrupted -- one
already was, every worker's plip killed with SIGINT. --resume reads the
partial CSV back and skips the poses already in it, so an interruption costs
minutes rather than hours. A truncated final row is not counted as done.

A pose that PLIP cannot process is recorded with status="failed" and kept in
the output. Dropping it would silently shrink the denominator of every pass
rate computed downstream.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_plip import INTERACTION_KINDS

__all__ = ["FIELDS", "already_done", "iter_poses", "row_from_record",
           "run_pose", "thread_limits"]

# poses_to_complex.LIGAND_RESNAME, as PLIP reports it in <identifiers><hetid>.
PLIP_HETID = "LIG"

FIELDS = (["label", "pose", "affinity", "status",
           "metal_ca501", "hbond_asn224", "tyr178_contact", "hydrophobic_pocket"]
          + [f"n_{kind}" for kind in INTERACTION_KINDS])


def thread_limits() -> dict:
    """Thread-count variables to pin to 1 in each worker.

    64 worker processes each importing numpy will each size a thread pool to
    the whole node, so the node runs 64x64 threads over 64 cores and spends
    its time in the scheduler. The user's own node template sets
    OMP_NUM_THREADS=1 for this reason; the workers set it for themselves here
    so a plain command line does not have to remember.
    """
    return {name: "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def already_done(out_csv) -> set:
    """(label, pose) pairs a previous, interrupted run already wrote.

    The interruption lands mid-write, so the final line can be a fragment.
    A short row is dropped rather than counted, because counting it would
    leave that pose permanently unmeasured with nothing to show for it.
    """
    path = Path(out_csv)
    if not path.exists():
        return set()
    done = set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") and row.get("pose") and row.get("status"):
                try:
                    done.add((row["label"], int(row["pose"])))
                except ValueError:
                    continue
    return done


def iter_poses(pose_dir: str, pattern: str = "*_out.pdbqt"):
    """Every pose of every ligand in a Uni-Dock output directory.

    Yields {label, pose, affinity, pdbqt_block}. The label is the filename
    with `_out` removed, which is how Uni-Dock names outputs after the input
    ligand; the pose number is its rank within that file. A file with no
    MODEL block yields nothing rather than raising -- pose_census reports
    those separately, and one of them must not stop a 53,000-pose batch.
    """
    from dock_unidock import parse_unidock_pdbqt

    for path in sorted(Path(pose_dir).glob(pattern)):
        label = path.stem.removesuffix("_out")
        for pose in parse_unidock_pdbqt(path.read_text()):
            yield {"label": label, "pose": pose["rank"],
                   "affinity": pose["affinity"],
                   "pdbqt_block": pose["pdbqt_block"]}


def row_from_record(entry: dict, record: dict, status: str = "ok") -> dict:
    """One CSV row from a parse_report() record, plus the four 8.0 criteria.

    The criteria are computed by validate_plip.crystal_verdict, not
    reimplemented here, so a change to the gate reaches the batch.
    """
    from validate_plip import crystal_verdict

    missing = crystal_verdict(record)["missing"] if status == "ok" else None
    row = {
        "label": entry["label"],
        "pose": entry["pose"],
        "affinity": entry.get("affinity"),
        "status": status,
        "metal_ca501": "" if missing is None else int("metal_ca501" not in missing),
        "hbond_asn224": "" if missing is None else int("hbond_asn224" not in missing),
        "tyr178_contact": "" if missing is None else int("tyr178_contact" not in missing),
        "hydrophobic_pocket": "" if missing is None else int(
            "hydrophobic_pocket" not in missing),
    }
    for kind in INTERACTION_KINDS:
        row[f"n_{kind}"] = len(record.get(kind, [])) if status == "ok" else ""
    return row


def run_pose(entry: dict, receptor_pdb: str, plip_bin: str) -> dict:
    """Build the complex for one pose, run PLIP, and return its row.

    Never raises: a pose PLIP cannot process comes back with status="failed"
    and stays in the output, so no pass rate silently loses its denominator.
    """
    from poses_to_complex import build_complex
    from run_plip import parse_report, run_plip

    with tempfile.TemporaryDirectory() as tmp:
        try:
            pose_pdbqt = Path(tmp) / "pose.pdbqt"
            pose_pdbqt.write_text(entry["pdbqt_block"] + "\n")
            complex_pdb = build_complex(str(pose_pdbqt), receptor_pdb,
                                        Path(tmp) / "complex.pdb")
            if complex_pdb is None:
                return row_from_record(entry, {}, status="complex_failed")
            xml_path = run_plip(complex_pdb, str(Path(tmp) / "plip"),
                                plip_bin=plip_bin)
            if xml_path is None:
                return row_from_record(entry, {}, status="plip_failed")
            record = parse_report(Path(xml_path).read_text(), hetid=PLIP_HETID)
            return row_from_record(entry, record)
        except Exception:  # noqa: BLE001 - one pose must not stop the batch
            return row_from_record(entry, {}, status="failed")


_CONTEXT: dict = {}


def _init(receptor_pdb: str, plip_bin: str) -> None:
    os.environ.update(thread_limits())
    _CONTEXT["receptor"] = receptor_pdb
    _CONTEXT["plip_bin"] = plip_bin


def _work(entry: dict) -> dict:
    return run_pose(entry, _CONTEXT["receptor"], _CONTEXT["plip_bin"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pose-dir", required=True,
                    help="Uni-Dock --dir output containing *_out.pdbqt files.")
    p.add_argument("--receptor", default="docking/v2/receptor_h.pdb",
                    help="Protonated receptor PDB. PLIP needs explicit "
                         "hydrogens on both sides to judge donor angles.")
    p.add_argument("--plip-bin", default="plip")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="Worker processes. PLIP is CPU-only; run on cpu64.")
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--resume", action="store_true",
                    help="Skip poses already present in --out-csv and append. "
                         "A 53,000-pose run that is interrupted then costs "
                         "minutes to finish rather than starting over.")
    args = p.parse_args(argv)

    if not Path(args.receptor).exists():
        sys.stderr.write(f"receptor not found: {args.receptor}\n")
        return 2

    entries = list(iter_poses(args.pose_dir))
    if not entries:
        sys.stderr.write(f"{args.pose_dir}: no poses found\n")
        return 2

    total = len(entries)
    done = already_done(args.out_csv) if args.resume else set()
    if done:
        entries = [e for e in entries if (e["label"], e["pose"]) not in done]
        print(f"resuming: {len(done)} of {total} poses already recorded")
    print(f"{len(entries)} poses from {args.pose_dir} on {args.jobs} workers",
          flush=True)
    if not entries:
        print("nothing left to do")
        return 0

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0}
    append = bool(done)
    with open(args.out_csv, "a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not append:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init,
                                 initargs=(args.receptor, args.plip_bin)) as pool:
            for n, row in enumerate(
                    pool.map(_work, entries, chunksize=args.chunk), start=1):
                writer.writerow(row)
                handle.flush()
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                if n % 2000 == 0:
                    print(f"  {n}/{len(entries)}", flush=True)

    print(f"wrote {args.out_csv}")
    for status, n in sorted(counts.items()):
        print(f"  {status:16s} {n}")
    failed = sum(n for s, n in counts.items() if s != "ok")
    if failed:
        sys.stderr.write(
            f"{failed} pose(s) did not produce a PLIP record. They are kept in "
            "the CSV with their status so no pass rate loses its denominator, "
            "but they must not be counted as interaction failures.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
