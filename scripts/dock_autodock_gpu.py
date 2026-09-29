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
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from dock_unidock import box_from_ligand, split_index  # noqa: F401  (재수출)

__all__ = ["NRUN", "build_command", "parse_dlg", "box_from_ligand",
           "find_fld", "run_batch"]

NRUN = 20

_ENERGY = re.compile(r"Estimated Free Energy of Binding\s*=\s*(-?\d+\.\d+)")


def build_command(fld: str, ligand: str, out_prefix: str, seed: int) -> list[str]:
    return ["autodock_gpu", "--ffile", fld, "--lfile", ligand,
            "--resnam", out_prefix, "--nrun", str(NRUN), "--seed", str(seed)]


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
    """One `autodock_gpu` process per ligand, ligand list split round-robin over GPUs.

    Mirrors dock_unidock.run_multi_gpu: one GPU per process (AutoDock-GPU does not
    itself batch across ligands), CUDA_VISIBLE_DEVICES pins each shard's device, and
    the split reuses dock_unidock.split_index so the round-robin logic - and its
    size-class reasoning - lives in exactly one place.
    """
    fld = find_fld(maps_dir)
    shards = split_index(ligands_index, gpus, shard_dir)
    os.makedirs(out_dir, exist_ok=True)

    plan: dict = {"fld": fld, "shards": {}}
    processes = []
    for device, shard in enumerate(shards):
        ligands = [line.strip() for line in Path(shard).read_text().splitlines()
                   if line.strip()]
        commands = [build_command(fld, lig, str(Path(out_dir) / Path(lig).stem),
                                  seed)
                    for lig in ligands]
        plan["shards"][device] = {"shard": shard, "n_ligands": len(ligands),
                                  "commands": commands}
        if dry_run:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
        log = os.path.join(out_dir, f"autodock_gpu{device}.log")
        handle = open(log, "w")
        for cmd in commands:
            handle.write("  ".join(cmd) + "\n")
            processes.append((device, cmd, log, handle,
                              subprocess.Popen(cmd, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT)))

    if dry_run:
        return plan

    results = {}
    for device, cmd, log, handle, process in processes:
        status = process.wait()
        results.setdefault(device, []).append({"cmd": cmd, "status": status})
    for _, _, log, handle, _ in processes:
        handle.close()
    plan["results"] = results
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
