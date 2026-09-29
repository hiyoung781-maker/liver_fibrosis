"""AutoDock-GPU 실행과 .dlg 파싱 (spec §7).

박스는 dock_unidock.box_from_ligand를 임포트해 쓴다. 여기서 다시 계산하면
검색 공간이 오류 없이 달라지고, 게이트는 다른 질문에 답하게 된다.

엔진이 Vina에서 AD4로 바뀌므로 affinity 값은 v1(Uni-Dock/Vina)과 비교할 수
없다. PLN-1474의 기준값도 이 엔진에서 새로 산출한다.

FORMAT CAVEAT: autodock_gpu is not installed anywhere on this machine (checked:
not on PATH, not in any conda env), so the .dlg parser below could not be run
against real output. The DOCKED:-prefixed .dlg format with an "Estimated Free
Energy of Binding" line is the classic AutoDock4-lineage output that
AutoDock-GPU inherits (it implements the AD4 scoring function over precomputed
grid maps), and AutoDock-GPU also offers an alternate --xmloutput mode this
module does NOT handle. Treat parse_dlg as UNVERIFIED against a real binary
until confirmed on K-BDS -- see results/v2_EXECUTION_CHECKLIST.md.

BATCH CLI. Task 16 drives this module as `python scripts/dock_autodock_gpu.py
--ligands <index> --maps <dir> --out <dir>`, so a batch entry point is provided
here rather than left for Task 16 to invent. AutoDock-GPU uses one GPU per
process, same as dock_unidock.run_multi_gpu, and for the same measured reason:
the ligand library sorts into size classes (v1: small 4,627 / medium 3,128 /
large 8, at torsion ceilings 8/16/20), so a CONTIGUOUS shard can collect every
large ligand and straggle long after the rest finish. The split below is
round-robin, reusing dock_unidock.split_index rather than re-implementing it.

--ligands is a plain-text file, one already-prepared ligand PDBQT path per
line -- the same convention scripts/ligands_to_pdbqt.py's index output and
dock_unidock's --ligand_index use. (Task 16's brief calls this file
"<file.smi>"; per the standing note that this plan has named a nonexistent
CLI flag or file five times already, that name is treated as intent, not a
requirement to parse SMILES here -- AutoDock-GPU takes PDBQT ligands, not
SMILES, so the index-of-paths convention is what the rest of this codebase
already uses and is what this CLI implements.)

AMENDMENT (real --help now known, v1.6-20-gbe06a13 built on the cluster).
Two changes from the original one-process-per-ligand design:

1. --filelist batching. AutoDock-GPU issues one invocation per SHARD now, not
   per ligand, using --filelist instead of --lfile: a bare per-ligand process
   reloads the grid maps every time, and with ~9,726 ligands that reload
   overhead can exceed the docking itself. FILELIST FORMAT CAVEAT: the layout
   written here (first line the .fld path, then one line of ligand PDBQT path
   followed by one line of its --resnam output name, per ligand) is what
   AutoDock-GPU's batch mode is documented to expect, but it could NOT be
   verified against the real binary -- it is not installed on this
   development machine. This is UNVERIFIED in the same sense as parse_dlg's
   FORMAT CAVEAT above, and must be confirmed with a two-ligand test run
   before the full 9,726-ligand batch -- see results/v2_EXECUTION_CHECKLIST.md.

2. --devnum, not CUDA_VISIBLE_DEVICES. --devnum is AutoDock-GPU's own device
   selector and avoids a layer of indirection, but it COUNTS FROM 1, not 0.
   Shard index 0 (this module's internal, zero-based numbering, matching
   dock_unidock.split_index) maps to --devnum 1, shard index 1 to --devnum 2,
   and so on. An off-by-one here silently sends work to the wrong GPU, or
   fails outright on a machine with fewer devices than the highest shard
   implies -- it is made explicit here (`device + 1`) rather than left to an
   implicit CUDA_VISIBLE_DEVICES-style zero base.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dock_unidock import box_from_ligand, split_index  # noqa: F401  (재수출)

__all__ = ["NRUN", "build_command", "build_filelist_command", "write_filelist",
           "parse_dlg", "box_from_ligand", "find_fld", "run_batch"]

NRUN = 20

_ENERGY = re.compile(r"Estimated Free Energy of Binding\s*=\s*(-?\d+\.\d+)")


def build_command(fld: str, ligand: str, out_prefix: str, seed: int) -> list[str]:
    """Single-ligand invocation. Kept for callers that dock one ligand at a
    time (e.g. the two-ligand filelist-format verification run); the batch
    path (run_batch) no longer uses this -- see build_filelist_command."""
    return ["autodock_gpu", "--ffile", fld, "--lfile", ligand,
            "--resnam", out_prefix, "--nrun", str(NRUN), "--seed", str(seed)]


def write_filelist(fld: str, ligands: list[str], out_dir: str,
                   filelist_path: str) -> str:
    """Write one AutoDock-GPU --filelist batch file for a shard.

    CONFIRMED FORMAT (verified against the real v1.6-20-gbe06a13 binary on the
    cluster): first line is the .fld map descriptor path, then for each
    ligand a line with its PDBQT path followed by a line with its --resnam
    output name (the ligand's stem). The layout itself was never wrong --
    see results/v2_EXECUTION_CHECKLIST.md.

    ALL THREE PATH KINDS MUST BE ABSOLUTE. AutoDock-GPU resolves any relative
    path in a filelist against the filelist's OWN directory, not the process's
    working directory -- observed directly on the cluster: a filelist at
    docking/ligand_shards/filelist_0.txt containing the relative fld path
    docking/v2/maps/meeko_receptor.maps.fld made AutoDock-GPU look for
    docking/ligand_shards/docking/v2/maps/meeko_receptor.maps.fld and fail
    with "Can't open fld file". This is the same class of bug already fixed
    once in scripts/prepare_maps.py, where the GPF's `receptor` directive had
    to become absolute because autogrid4 runs from inside the maps directory
    (see prepare_maps.py's `Path(receptor).resolve()`) -- a program that
    changes/implies its own working directory for a file it reads turns any
    relative path in that file into a trap. Resolve everything here for the
    same reason.
    """
    lines = [str(Path(fld).resolve())]
    for ligand in ligands:
        resnam = str((Path(out_dir) / Path(ligand).stem).resolve())
        lines.append(str(Path(ligand).resolve()))
        lines.append(resnam)
    path = Path(filelist_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def build_filelist_command(filelist: str, devnum: int, seed: int) -> list[str]:
    """One autodock_gpu invocation per shard, batched via --filelist.

    devnum is AutoDock-GPU's own --devnum, which COUNTS FROM 1 (not 0, unlike
    CUDA_VISIBLE_DEVICES). Callers pass `device + 1` for a zero-based shard
    index -- see the module docstring's AMENDMENT note.
    """
    return ["autodock_gpu", "--filelist", filelist, "--devnum", str(devnum),
            "--nrun", str(NRUN), "--seed", str(seed)]


def parse_dlg(text: str) -> list[dict]:
    """.dlg의 DOCKED 블록에서 포즈를 뽑는다.

    에너지가 없어도 포즈는 버리지 않는다 - v1에서 점수 추출 실패가 구조 파싱과
    같은 try 안에 있어 포즈를 0개로 만든 회귀가 있었다.
    """
    poses: list[dict] = []
    block: list[str] = []
    affinity = None
    rank = 0
    for line in text.splitlines():
        if not line.startswith("DOCKED:"):
            continue
        payload = line[len("DOCKED:"):].lstrip()
        if payload.startswith("MODEL"):
            block, affinity = [payload], None
            rank += 1
            continue
        match = _ENERGY.search(payload)
        if match:
            affinity = float(match.group(1))
            continue
        if payload.startswith("USER"):
            continue
        block.append(payload)
        if payload.startswith("ENDMDL"):
            poses.append({"rank": rank, "affinity": affinity,
                          "pdbqt_block": "\n".join(block)})
    return poses


def find_fld(maps_dir: str) -> str:
    """Locate the single `*.maps.fld` grid field file inside a maps directory.

    scripts/prepare_maps.py names it `<receptor stem>.maps.fld`; the stem isn't
    known to this caller, so it is discovered rather than guessed.
    """
    candidates = sorted(Path(maps_dir).glob("*.maps.fld"))
    if not candidates:
        raise FileNotFoundError(f"{maps_dir}: no *.maps.fld file found")
    if len(candidates) > 1:
        raise ValueError(f"{maps_dir}: multiple *.maps.fld files found: "
                          f"{[str(c) for c in candidates]}")
    return str(candidates[0])


def run_batch(ligands_index: str, maps_dir: str, out_dir: str, gpus: int = 1,
             shard_dir: str = "docking/ligand_shards", seed: int = 42,
             dry_run: bool = False) -> dict:
    """One `autodock_gpu` process per SHARD, batched via --filelist.

    Ligands are split round-robin over `gpus` shards (dock_unidock.split_index,
    so the round-robin logic and its size-class reasoning live in exactly one
    place). Each shard gets one --filelist batch file (write_filelist -- see
    its UNVERIFIED format note) and one autodock_gpu invocation pinned to its
    device via --devnum, which counts from 1: shard index `device` (zero-based)
    is pinned with `--devnum device + 1`.
    """
    fld = find_fld(maps_dir)
    shards = split_index(ligands_index, gpus, shard_dir)
    os.makedirs(out_dir, exist_ok=True)

    plan: dict = {"fld": fld, "shards": {}}
    processes = []
    for device, shard in enumerate(shards):
        ligands = [line.strip() for line in Path(shard).read_text().splitlines()
                   if line.strip()]
        devnum = device + 1
        filelist_path = str(Path(shard_dir) / f"filelist_{device}.txt")
        filelist = write_filelist(fld, ligands, out_dir, filelist_path)
        command = build_filelist_command(filelist, devnum, seed)
        plan["shards"][device] = {"shard": shard, "n_ligands": len(ligands),
                                  "filelist": filelist, "devnum": devnum,
                                  "command": command}
        if dry_run:
            continue
        log = os.path.join(out_dir, f"autodock_gpu{device}.log")
        handle = open(log, "w")
        handle.write("  ".join(command) + "\n")
        processes.append((device, command, log, handle,
                          subprocess.Popen(command, stdout=handle,
                                           stderr=subprocess.STDOUT)))

    if dry_run:
        return plan

    results = {}
    failures = []
    for device, cmd, log, handle, process in processes:
        status = process.wait()
        handle.close()
        n_ligands = plan["shards"][device]["n_ligands"]
        # count .dlg files this shard's resnam prefixes should have produced
        shard_out_prefixes = [
            Path(line).stem for i, line in
            enumerate(Path(plan["shards"][device]["filelist"]).read_text()
                     .splitlines())
            if i >= 2 and i % 2 == 0
        ]
        produced = sum(1 for prefix in shard_out_prefixes
                       if list(Path(out_dir).glob(f"{prefix}*.dlg")))
        results.setdefault(device, []).append({"cmd": cmd, "status": status,
                                               "dlg_produced": produced,
                                               "n_ligands": n_ligands})
        # A failure is a nonzero exit status OR an exit that produced no .dlg
        # output at all -- a shard that "succeeds" with zero poses would hand
        # every one of its ligands downstream as silently, plausibly docked
        # with zero poses, the same failure shape (a plausible-looking number
        # masking a real break) this project has hit three times before: the
        # 5-char CCD residue name zeroing every affinity, the naive RMSD
        # reporting 0.63 A as 6.13, and the Open Babel nitrogen mis-typing
        # that promoted the fourth-best pose to first. A printed warning would
        # not have stopped any of those from propagating -- this must raise.
        if status != 0 or (n_ligands > 0 and produced == 0):
            tail = ""
            try:
                tail = "\n".join(Path(log).read_text().splitlines()[-40:])
            except OSError:
                pass
            failures.append(
                f"shard {device} (devnum {plan['shards'][device]['devnum']}) "
                f"failed: exit status {status}, {produced}/{n_ligands} "
                f".dlg outputs found. Command: {' '.join(cmd)}\n"
                f"--- tail of {log} ---\n{tail}"
            )
    plan["results"] = results
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(processes)} autodock_gpu shard(s) "
            "failed:\n\n" + "\n\n".join(failures)
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ligands", required=True,
                        help="file listing one prepared ligand PDBQT path per line "
                             "(same convention as dock_unidock's --ligand_index)")
    parser.add_argument("--maps", required=True,
                        help="directory containing the *.maps.fld grid produced by "
                             "scripts/prepare_maps.py + autogrid4")
    parser.add_argument("--out", required=True, help="output directory for .dlg files")
    parser.add_argument("--gpus", type=int, default=1,
                        help="run this many autodock_gpu processes, one per GPU")
    parser.add_argument("--shard-dir", default="docking/ligand_shards")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned commands and the fld path, run nothing")
    args = parser.parse_args(argv)

    import shutil
    if not args.dry_run and shutil.which("autodock_gpu") is None:
        print("autodock_gpu not on PATH - see results/v2_EXECUTION_CHECKLIST.md",
              file=sys.stderr)
        return 1

    plan = run_batch(args.ligands, args.maps, args.out, gpus=args.gpus,
                     shard_dir=args.shard_dir, seed=args.seed,
                     dry_run=args.dry_run)
    print(f"fld: {plan['fld']}")
    for device, info in plan["shards"].items():
        print(f"  GPU {device}: {info['n_ligands']} ligands -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
