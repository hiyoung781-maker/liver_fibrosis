"""Section 10.3's redocking gate, per isoform.

> 각 구조의 자체 결정 리간드를 같은 프로토콜로 재도킹해 **PLIP이 MIDAS 금속
> 배위를 보고 + 대칭보정 RMSD < 2.0 Å**를 만족해야 한다. **재현하지 못하는
> 아이소폼은 분석에서 제외**하고 기록한다. αvβ1과 동일한 기준이다.

Two criteria, not alphaVbeta1's four. Sections 8.5b's distance criteria are
Ca501 and beta1-Asn224 by residue number, which do not exist in another
receptor; what transfers is the MIDAS coordination and the RMSD.

THE MIDAS ION IS NOT CALCIUM HERE. 8W30's is Ca B501, 6MK0's is Mn B708,
9CZD's is Mg B2102. validate_plip's contact test is written against the
calcium and returns False for both isoforms, which would exclude them from
the selectivity analysis for a reason that is not chemical -- the same shape
of artefact that made the alphaVbeta1 gate fail twice before it was believed.
The metal is read from the manifest prepare_isoform wrote.

AN UNCOMPUTABLE RMSD IS A FAILURE, never a pass. rmsd_to_reference returns no
number when the pose and the reference are not the same molecular graph, and
a gate that read that as success would adopt a receptor on a measurement that
was never made.

What this gate decides: whether an isoform enters the selectivity analysis at
all. A failure excludes that receptor and is recorded; it does not stop the
campaign, because section 10.4's calibration gate can still run on whichever
isoforms survive.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["RMSD_CUTOFF", "midas_ligand_contact", "pose_table", "verdict"]

# Section 10.3: "alphaVbeta1과 동일한 기준이다." The same 2.0 A that the
# alphaVbeta1 gate used, and that AutoDock-GPU failed at 2.15 A.
RMSD_CUTOFF = 2.0

# poses_to_complex.LIGAND_RESNAME, as PLIP reports it.
PLIP_HETID = "LIG"


def midas_ligand_contact(metal_complexes: list, midas: tuple) -> bool:
    """Does PLIP report the LIGAND coordinating this isoform's MIDAS ion?

    `midas` is (resname, chain, resseq) as the manifest records it. The
    entry must name that ion AND have location == "ligand": the MIDAS is
    always coordinated by the protein, which says nothing about whether the
    ligand reaches it.
    """
    resname, chain, resseq = midas[0].upper(), midas[1], str(midas[2])
    for entry in metal_complexes:
        if entry.get("location") != "ligand":
            continue
        if (entry.get("metal_type") or "").upper() != resname:
            continue
        if entry.get("reschain_lig") != chain:
            continue
        if str(entry.get("resnr_lig")) != resseq:
            continue
        return True
    return False


def verdict(rmsd, plip_midas: bool) -> dict:
    """Section 10.3's two criteria. `rmsd` may be None -- that is a failure."""
    failed = []
    if rmsd is None or rmsd >= RMSD_CUTOFF:
        failed.append("rmsd")
    if not plip_midas:
        failed.append("plip_midas")
    return {"passed": not failed, "failed": failed}


def pose_table(records: list, gated_rank: int) -> list:
    """Rows for the per-pose record, best affinity first.

    A failed gate has to say WHICH failure it is. If no pose comes near the
    crystal, the search did not find the binding mode; if a pose does and the
    scoring function ranks another first, that is the ranking failure
    AutoDock-GPU showed on alphaVbeta1. Both are a FAIL -- the marked row is
    the verdict -- but they are different findings and the analysis that
    follows differs.
    """
    ordered = sorted(
        records,
        key=lambda r: (r.get("affinity") is None,
                       r.get("affinity") if r.get("affinity") is not None else 0.0,
                       r.get("rank", 0)))
    return [{"rank": r["rank"], "affinity": r["affinity"], "rmsd": r["rmsd"],
             "plip_midas": r["plip_midas"], "gated": r["rank"] == gated_rank}
            for r in ordered]


def _format_pose_table(rows: list) -> list:
    lines = [
        "## 전체 포즈 기록",
        "",
        "게이트는 표시된 행이다(§10.3은 상위 포즈로 판정한다). 나머지는 "
        "**어느 종류의 실패인지**를 말한다 — 결정 포즈 근처에 아무 포즈도 없으면 "
        "탐색이 결합 양식을 못 찾은 것이고, 가까운 포즈가 있는데 점수가 다른 것을 "
        "1위로 올렸다면 그것은 순위 실패다. 대안 판정이 아니다.",
        "",
        "| | 포즈 | affinity | RMSD | PLIP MIDAS |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        aff = f"{r['affinity']:.2f}" if r["affinity"] is not None else "-"
        rmsd = f"{r['rmsd']:.2f}" if r["rmsd"] is not None else "계산 불가"
        lines.append(f"| {'<-' if r['gated'] else ''} | {r['rank']} | {aff} | "
                     f"{rmsd} | {'예' if r['plip_midas'] else '아니오'} |")
    lines.append("")
    return lines


def _report(name: str, manifest: dict, measurement: dict, decision: dict,
            n_poses: int, rows: list | None = None) -> str:
    ok = decision["passed"]
    rmsd = measurement["rmsd"]
    lines = [
        f"# 아이소폼 재도킹 검증 게이트 — {name} (spec §10.3)",
        "",
        f"구조: `{manifest['structure']}`",
        f"결정 리간드: `{manifest['ligand'][0]} {manifest['ligand'][1]}"
        f"{manifest['ligand'][2]}`, MIDAS {manifest['midas_metal'][0]} "
        f"{manifest['midas_metal'][1]}{manifest['midas_metal'][2]} "
        f"(결정 좌표에서 {manifest['midas_distance']} Å)",
        f"포즈 {n_poses}개, affinity 최저 포즈(rank {measurement['rank']}, "
        f"{measurement['affinity']} kcal/mol)에 적용",
        "",
        f"판정: **{'PASS' if ok else 'FAIL'}**",
        "",
        "| 기준 | 측정 | 문턱 | 결과 |",
        "|---|---|---|---|",
        f"| 대칭보정 RMSD (CalcRMS) | "
        f"{'계산 불가' if rmsd is None else f'{rmsd:.2f} Å'} | < {RMSD_CUTOFF} Å | "
        f"{'pass' if 'rmsd' not in decision['failed'] else 'FAIL'} |",
        f"| PLIP MIDAS 금속 배위 (리간드 쪽) | "
        f"{'보고됨' if measurement['plip_midas'] else '없음'} | 필수 | "
        f"{'pass' if 'plip_midas' not in decision['failed'] else 'FAIL'} |",
        "",
    ]
    if measurement.get("rmsd_note"):
        lines += [f"RMSD 미산출 사유: {measurement['rmsd_note']}", ""]
    if rows:
        lines += _format_pose_table(rows)
    lines += ["---", ""]
    if ok:
        lines.append(
            f"{name}는 자체 결정 포즈를 재현한다. §10의 선택성 분석에 포함한다.")
    else:
        lines.append(
            "실패: " + ", ".join(decision["failed"]) + ". spec §10.3대로 "
            f"**{name}를 선택성 분석에서 제외**하고 이 파일을 그 기록으로 둔다. "
            "αvβ1에 적용한 것과 같은 기준이다.")
    return "\n".join(lines) + "\n"


def _measure(pose, isoform_dir: Path, manifest: dict, workdir: Path,
             plip_bin: str) -> dict:
    from poses_to_complex import build_complex
    from run_plip import parse_report, run_plip
    from validate_redock import _load_reference, _pose_to_sdf, rmsd_to_reference

    from rdkit import Chem

    workdir.mkdir(parents=True, exist_ok=True)
    stem = f"pose_{pose['rank']:03d}"
    pose_pdbqt = workdir / f"{stem}.pdbqt"
    pose_pdbqt.write_text(pose["pdbqt_block"] + "\n")

    pose_sdf = workdir / f"{stem}.sdf"
    _pose_to_sdf(pose_pdbqt, pose_sdf)
    mol = next(iter(Chem.SDMolSupplier(str(pose_sdf), removeHs=False)), None)
    if mol is None:
        raise ValueError(f"{pose_sdf}: RDKit could not read the converted pose")

    reference = _load_reference(str(isoform_dir / f"XTAL_{isoform_dir.name}.pdbqt"),
                                workdir)
    rmsd, rmsd_note = rmsd_to_reference(mol, reference)

    metal = manifest["midas_metal"]
    midas = (metal[0], metal[1], metal[2])
    complex_pdb = build_complex(str(pose_pdbqt), str(isoform_dir / "receptor.pdb"),
                                workdir / f"{stem}.complex.pdb",
                                midas=(midas[1], midas[2], midas[0]))
    if complex_pdb is None:
        raise ValueError(f"{pose_pdbqt}: complex build failed")
    xml_path = run_plip(complex_pdb, str(workdir / f"{stem}_plip"),
                        plip_bin=plip_bin)
    if xml_path is None:
        raise ValueError(f"{complex_pdb}: PLIP failed to run")
    record = parse_report(Path(xml_path).read_text(), hetid=PLIP_HETID)

    return {
        "rank": pose["rank"],
        "affinity": pose["affinity"],
        "rmsd": rmsd,
        "rmsd_note": rmsd_note,
        "plip_midas": midas_ligand_contact(record["metal_complexes"], midas),
        "n_metal_complexes": len(record["metal_complexes"]),
        "complex_pdb": complex_pdb,
        "plip_xml": xml_path,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--isoform-dir", required=True,
                    help="Directory prepare_isoform.py wrote.")
    p.add_argument("--poses", required=True,
                    help="Uni-Dock output PDBQT from redocking the crystal ligand.")
    p.add_argument("--plip-bin", default="plip")
    p.add_argument("--all-poses", action="store_true",
                    help="Measure every pose, not only the one the gate judges, "
                         "and record them. The verdict is unchanged: it always "
                         "comes from the top pose by affinity.")
    p.add_argument("--workdir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    from dock_unidock import parse_unidock_pdbqt
    from validate_redock import best_pose

    isoform_dir = Path(args.isoform_dir)
    manifest = json.loads((isoform_dir / "manifest.json").read_text())
    poses = parse_unidock_pdbqt(Path(args.poses).read_text())
    top = best_pose(poses)

    def run(workdir: Path) -> int:
        to_measure = poses if args.all_poses else [top]
        records = [_measure(p, isoform_dir, manifest, workdir, args.plip_bin)
                   for p in to_measure]
        measurement = next(r for r in records if r["rank"] == top["rank"])
        decision = verdict(measurement["rmsd"], measurement["plip_midas"])
        rows = pose_table(records, top["rank"]) if args.all_poses else None
        report = _report(isoform_dir.name, manifest, measurement, decision,
                         len(poses), rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        args.out.with_suffix(".json").write_text(json.dumps(
            {"isoform": isoform_dir.name, "poses_file": args.poses,
             "n_poses": len(poses), "manifest": manifest,
             "measurement": measurement, "verdict": decision,
             "all_poses": rows}, indent=2) + "\n")
        print(report)
        return 0 if decision["passed"] else 1

    if args.workdir is not None:
        args.workdir.mkdir(parents=True, exist_ok=True)
        return run(args.workdir)
    with tempfile.TemporaryDirectory() as tmp:
        return run(Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
