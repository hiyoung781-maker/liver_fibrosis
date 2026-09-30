"""Selectivity: the Delta axis, §10.4's calibration gate, and the rank-sum (spec §10).

§10.1 defines the axis as a score DIFFERENCE for the same ligand:

    Delta_s = affinity(subtype s) - affinity(alphaVbeta1)

ITS JUSTIFICATION IS ONLY HALF TRUE, AND THE HALF THAT FAILS IS THE ONE THAT
MATTERS. The spec argues that because Vina-family scores scale with ligand
size and atom count, the same ligand's difference cancels that systematic
error. It cancels the LIGAND-side error. It does nothing about the
RECEPTOR-side offset: a larger, more buried or more hydrophobic pocket scores
EVERY ligand better, so Delta conflates "this ligand prefers alphaVbeta1"
with "alphaVbeta6 scores everything generously" -- and selectivity is exactly
the difference between those two. Score differences across targets are used
in the literature but are known to be unreliable, and the usual mitigation is
per-target normalisation.

THE PRIMARY MEASURE IS THEREFORE A WITHIN-RECEPTOR RANK. Each receptor ranks
the whole target set by affinity, inside that receptor, and

    selectivity(L) = percentile_target(L) - percentile_subtype(L)

A large value means L is near the top against alphaVbeta1 and near the bottom
against the off-target. Because each ranking is normalised inside its own
receptor, a uniform receptor offset cancels exactly -- a test pins that,
alongside one showing Delta reports such an offset as uniform selectivity.

A RANK NEEDS A POPULATION. Ranking the six reference compounds among
themselves is too thin; the full geometry-passing set is docked against each
validated isoform so the percentiles mean something.

Delta is still computed and reported, because it is the pre-registered
number. The rank measure leads.

§10.4 is a PRE-REGISTERED gate on the protocol itself, not on any molecule:
CWHM-12 and GLPG0187 must come out NON-selective and PLN-1474 and
CHEMBL4649232 SELECTIVE. The answer key is measured IC50, which makes this a
stronger control than a decoy set. If the protocol cannot reproduce it, no
selectivity statement is made about any generated molecule and criterion 6 is
recorded unevaluable.

WHAT ONE ISOFORM COSTS THIS GATE. Three of the four registered isoforms are
out: alpha5beta1 and alphaVbeta8 have no small-molecule structure in the PDB,
and alphaVbeta3 failed §10.3 (see results/v2_isoform_structures.md). Only
alphaVbeta6 remains, and by measured IC50 both non-selective controls are
nearly equipotent on alphaVbeta1 and alphaVbeta6 -- CWHM-12 1.8 against
1.5 nM, GLPG0187 1.3 against 1.4 nM. The gate can be run but it has little to
discriminate with, and that is reported with the result rather than left for
a reader to work out.

§10.1's rank-sum: per validated isoform, rank the target set by Delta
DESCENDING (1 = most selective); a molecule's rank-sum is the reverse of the
sum of its ranks, so a LARGER value is more selective. Per-isoform score
scales are absorbed by the ranking step.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

__all__ = ["CALIBRATION", "LigandMismatchError", "MEASURED_IC50",
           "best_affinities", "calibration_verdict", "control_values", "deltas",
           "ligand_identity", "merge_affinities", "percentile_ranks",
           "rank_selectivity", "rank_sum", "shared_ligands"]


class LigandMismatchError(RuntimeError):
    """One label, two different molecules.

    Both the Delta and the rank comparison assume the SAME ligand file was
    docked against both receptors. Uni-Dock copies the input PDBQT's
    `REMARK SMILES` into its output, so that line identifies what was actually
    docked -- and it is how the protomer mix-up would have been caught, where
    a label meant one molecule in one run and another in the next.
    """

# spec §10.4's controls, and the measured IC50 (nM) that is their answer key.
CALIBRATION = {
    "non_selective": ("CWHM-12", "GLPG0187"),
    "selective": ("PLN-1474", "CHEMBL4649232"),
}

MEASURED_IC50 = {
    "CWHM-12": {"avb1": 1.8, "avb3": 0.8, "avb6": 1.5, "avb8": 0.2},
    "GLPG0187": {"avb1": 1.3, "avb3": 3.7, "avb6": 1.4, "avb8": 1.2},
}


def best_affinities(pose_dir: str, pattern: str = "*_out.pdbqt") -> dict:
    """label -> lowest affinity over that ligand's poses.

    §10.6 keeps geometry as a REPORTED observation for isoforms rather than a
    gate, because the MIDAS carboxylate contact is class-invariant (measured
    2.08-2.65 A across six structures) and would penalise everything equally.
    The Delta axis therefore reads affinity alone here.
    """
    from dock_unidock import parse_unidock_pdbqt

    out = {}
    for path in sorted(Path(pose_dir).glob(pattern)):
        label = path.stem.removesuffix("_out")
        scores = [p["affinity"] for p in parse_unidock_pdbqt(path.read_text())
                  if p["affinity"] is not None]
        if scores:
            out[label] = min(scores)
    return out


def deltas(target: dict, subtype: dict) -> dict:
    """Delta = subtype - target, for ligands present on BOTH sides.

    A ligand missing from either side is dropped, never defaulted to zero: a
    zero would read as "equally potent on both", which is a claim.
    """
    return {label: subtype[label] - target[label]
            for label in target if label in subtype}


def ligand_identity(pose_text: str):
    """The `REMARK SMILES` of a Uni-Dock output, or None if it carries none."""
    for line in pose_text.splitlines():
        if line.startswith("REMARK SMILES ") and "IDX" not in line:
            return line[len("REMARK SMILES "):].strip()
    return None


def _identities(pose_dir: str, pattern: str = "*_out.pdbqt") -> dict:
    out = {}
    for path in sorted(Path(pose_dir).glob(pattern)):
        out[path.stem.removesuffix("_out")] = ligand_identity(path.read_text())
    return out


def shared_ligands(target_dir: str, subtype_dir: str) -> set:
    """Labels present in both directories, verified to be the same molecule.

    A label whose REMARK SMILES differs between the two raises: comparing
    those scores would compare two molecules, not two receptors.
    """
    target, subtype = _identities(target_dir), _identities(subtype_dir)
    shared = set(target) & set(subtype)
    mismatched = {label: (target[label], subtype[label]) for label in shared
                  if target[label] is not None and subtype[label] is not None
                  and target[label] != subtype[label]}
    if mismatched:
        listed = "; ".join(f"{k}: {a} vs {b}"
                           for k, (a, b) in sorted(mismatched.items())[:5])
        raise LigandMismatchError(
            f"{len(mismatched)} label(s) name different molecules in "
            f"{target_dir} and {subtype_dir}: {listed}. Dock the SAME prepared "
            "PDBQT against both receptors -- otherwise the comparison is "
            "between molecules, not receptors.")
    return shared


def merge_affinities(pose_dirs: list, pattern: str = "*_out.pdbqt") -> dict:
    """Best affinity per label across several directories, identity-checked.

    alphaVbeta1's scores live in more than one directory -- the two arms and
    the reference set were docked separately -- so the target side is
    assembled from all of them. A label appearing twice as two different
    molecules raises; as the same molecule, the better score wins.
    """
    merged: dict = {}
    seen: dict = {}
    for directory in pose_dirs:
        identities = _identities(directory, pattern)
        for label, score in best_affinities(directory, pattern).items():
            identity = identities.get(label)
            if label in seen and identity is not None \
                    and seen[label] is not None and seen[label] != identity:
                raise LigandMismatchError(
                    f"{label} is {seen[label]} in one directory and "
                    f"{identity} in {directory}. These are different "
                    "molecules under one label.")
            seen.setdefault(label, identity)
            if label not in merged or score < merged[label]:
                merged[label] = score
    return merged


def percentile_ranks(scores: dict) -> dict:
    """label -> percentile within this receptor, 1.0 for the strongest binder.

    Normalised inside the receptor, which is the whole point: it removes the
    receptor-side offset that a score difference leaves in. Ties share a
    percentile. Returns {} for fewer than two molecules, because a percentile
    needs a population and one molecule is not one.
    """
    if len(scores) < 2:
        return {}
    # ascending affinity = strongest first; percentile 1.0 is the strongest.
    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    n = len(ordered)
    out: dict = {}
    position = 0
    while position < n:
        end = position
        while end + 1 < n and ordered[end + 1][1] == ordered[position][1]:
            end += 1
        # mean rank over the tie group, mapped so the strongest gets 1.0
        mean_index = (position + end) / 2.0
        value = 1.0 - mean_index / (n - 1)
        for index in range(position, end + 1):
            out[ordered[index][0]] = value
        position = end + 1
    return out


def rank_selectivity(target: dict, subtype: dict) -> dict:
    """percentile(target) - percentile(subtype), for ligands on both sides.

    Positive means the ligand ranks higher against alphaVbeta1 than against
    the off-target: selective. A uniform offset between the two receptors
    cancels exactly, which a score difference cannot do.
    """
    shared = [label for label in target if label in subtype]
    if len(shared) < 2:
        return {}
    target_pct = percentile_ranks({k: target[k] for k in shared})
    subtype_pct = percentile_ranks({k: subtype[k] for k in shared})
    return {k: target_pct[k] - subtype_pct[k] for k in shared}


def _inventory(target_dirs: list, subtype_dirs: list,
               pattern: str = "*_out.pdbqt") -> str:
    """Per-directory pose counts, for when the two sides share no ligand.

    A count of zero says the docking has not finished or wrote elsewhere; two
    non-zero counts that still share nothing says the labels differ between
    the runs. The two need different fixes and neither is visible from the
    shared-ligand count alone.
    """
    lines = []
    for role, group in (("αvβ1 (target)", target_dirs),
                        ("isoform", subtype_dirs)):
        for directory in group:
            path = Path(directory)
            if not path.is_dir():
                lines.append(f"  {role:16s} {directory}  -- NOT A DIRECTORY")
                continue
            poses = sorted(path.glob(pattern))
            sample = ", ".join(f.name[:-len("_out.pdbqt")] for f in poses[:3])
            lines.append(f"  {role:16s} {directory}  {len(poses)} poses"
                         + (f"  e.g. {sample}" if sample else ""))
    return "\n".join(lines) + "\n"


def _control_key(control: str, table: dict):
    """The key `table` uses for a §10.4 control, or None.

    The reference set is docked twice under two naming conventions -- bare
    (`PLN-1474`, from data/selectivity_refs.smi) on the selectivity side and
    prefixed (`PANEL_PLN-1474`) in the geometry and ADMET tables -- so a
    lookup that knows only one of them reports a docked control as missing.
    """
    for key in (control, f"PANEL_{control}", control.replace("PANEL_", "")):
        if key in table:
            return key
    return None


def control_values(table: dict) -> dict:
    """{control: value or None} for the four §10.4 controls.

    THE SINGLE SOURCE for "what did the controls do". The verdict and the
    report it prints must not look this up separately: they did, under two
    different naming conventions, and produced a report whose FAIL quoted
    real numbers above a table saying every control was absent. One function,
    one answer.
    """
    out = {}
    for group in CALIBRATION.values():
        for name in group:
            key = _control_key(name, table)
            out[name] = None if key is None else table[key]
    return out


def calibration_verdict(delta: dict) -> dict:
    """§10.4: do the four controls come out the way their IC50s say?

    Passing requires every selective control's Delta to exceed every
    non-selective control's, and all four to be present. A missing control
    fails rather than being skipped -- the gate's whole value is that its
    answer is known in advance.
    """
    failed = []
    # The reference set is docked under whichever name its input file used --
    # bare from data/selectivity_refs.smi, prefixed from docking/panel_pdbqt.
    # A lookup that knows only one of them reports a control that IS docked as
    # missing, which is indistinguishable from one that was never docked.
    values = control_values(delta)
    missing = sorted(n for n, v in values.items() if v is None)
    if missing:
        failed.append(f"controls absent from the Delta table: {missing}")
        return {"passed": False, "failed": failed, "missing": missing}

    selective = {n: values[n] for n in CALIBRATION["selective"]}
    non_selective = {n: values[n] for n in CALIBRATION["non_selective"]}
    for sel_name, sel in selective.items():
        for non_name, non in non_selective.items():
            if sel <= non:
                failed.append(
                    f"{non_name} (Delta {non:+.3f}) is at least as selective "
                    f"as {sel_name} ({sel:+.3f}); the measured IC50s say the "
                    "opposite")
    return {"passed": not failed, "failed": failed, "missing": [],
            "selective": selective, "non_selective": non_selective}


def rank_sum(per_isoform: dict) -> dict:
    """§10.1's selectivity rank-sum. Larger = more selective.

    `per_isoform` is {isoform: {label: Delta}}. Each isoform ranks its own
    molecules by Delta descending, and the reverse of the summed ranks is the
    score, normalised by how many isoforms actually ranked that molecule so
    one missing from an isoform is not penalised for it.
    """
    totals: dict = {}
    counts: dict = {}
    for values in per_isoform.values():
        ordered = sorted(values, key=lambda k: -values[k])
        n = len(ordered)
        for position, label in enumerate(ordered, start=1):
            # reverse score: 1st of n scores n, last scores 1
            totals[label] = totals.get(label, 0.0) + (n - position + 1)
            counts[label] = counts.get(label, 0) + 1
    return {label: totals[label] / counts[label] for label in totals}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-poses", action="append", required=True,
                    metavar="DIR",
                    help="A Uni-Dock output directory for alphaVbeta1. Repeat "
                         "it: the target's scores are spread over the two arms "
                         "and the reference set, which were docked separately.")
    p.add_argument("--isoform", action="append", default=[],
                    metavar="NAME=DIR[,DIR...]",
                    help="A validated isoform and its pose directories, comma "
                         "separated. Repeat the flag per isoform. The "
                         "directory list is comma separated for the same "
                         "reason --target-poses repeats: the four §10.4 "
                         "controls are docked separately from the generated "
                         "molecules, and an isoform side missing them cannot "
                         "run the gate at all. Only isoforms that passed "
                         "§10.3 belong here.")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    if not args.isoform:
        sys.stderr.write(
            "no isoform given. Section 10.4's gate and the selectivity axis "
            "both need at least one isoform that passed 10.3; with none, "
            "criterion 6 is unevaluable and that is the result to report.\n")
        return 2

    try:
        target = merge_affinities(args.target_poses)
    except LigandMismatchError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    delta_by_isoform, rank_by_isoform, shared_counts = {}, {}, {}
    for item in args.isoform:
        name, _, directory = item.partition("=")
        dirs = [d for d in directory.split(",") if d]
        # Every label used here is verified to name the same molecule on both
        # sides. Without that the comparison is between molecules.
        try:
            shared = set()
            for target_dir in args.target_poses:
                for subtype_dir in dirs:
                    shared |= shared_ligands(target_dir, subtype_dir)
            subtype_all = merge_affinities(dirs)
        except LigandMismatchError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        if not shared:
            # Nothing to compare is not a result, it is a broken input, and
            # reporting it as "0 shared" produced a report that looked like a
            # failed calibration gate. Name every directory and what is in it.
            sys.stderr.write(
                f"{name}: no ligand is present in both receptors, so there is "
                "nothing to compare.\n" + _inventory(args.target_poses, dirs))
            return 2
        subtype = {k: v for k, v in subtype_all.items() if k in shared}
        restricted = {k: v for k, v in target.items() if k in shared}
        shared_counts[name] = len(shared)
        delta_by_isoform[name] = deltas(restricted, subtype)
        rank_by_isoform[name] = rank_selectivity(restricted, subtype)

    # A control absent from BOTH sides was never docked against the isoform;
    # a control the target has but the isoform does not is a docking that did
    # not produce a pose. The gate reports the same FAIL for each, and they
    # need different fixes, so name which one happened.
    controls = list(CALIBRATION["selective"]) + list(CALIBRATION["non_selective"])
    control_status = {}
    for item in args.isoform:
        name, _, directory = item.partition("=")
        values = control_values(rank_by_isoform.get(name, {}))
        missing = [c for c in controls if values[c] is None]
        control_status[name] = {
            "missing": missing,
            "absent_from_target": [c for c in missing
                                   if _control_key(c, target) is None],
            "absent_from_isoform": [c for c in missing
                                    if _control_key(c, target) is not None],
        }

    rank_verdicts = {n: calibration_verdict(t) for n, t in rank_by_isoform.items()}
    delta_verdicts = {n: calibration_verdict(t) for n, t in delta_by_isoform.items()}
    scores = rank_sum(rank_by_isoform)

    isoforms = list(rank_by_isoform)
    lines = [
        "# 선택성 축과 §10.4 보정 게이트",
        "",
        "대상 수용체 포즈: " + ", ".join(f"`{d}`" for d in args.target_poses)
        + f"   분자 {len(target)}개",
        "아이소폼별 공통 리간드 수: "
        + ", ".join(f"{n} {c}개" for n, c in shared_counts.items()),
        "",
        "공통 리간드는 두 수용체에서 `REMARK SMILES`가 일치하는 것만이다 — "
        "같은 라벨이 서로 다른 분자를 가리키면 실행을 멈춘다. protomer를 고치기 "
        "전과 후의 실행이 같은 라벨을 썼고, 그 검증이 없으면 수용체가 아니라 "
        "분자를 비교하게 된다.",
        f"아이소폼: {', '.join(isoforms)}",
        "",
        "## 주 지표는 수용체 내 순위다",
        "",
        "§10.1은 축을 점수 차이 `Δ = affinity(아이소폼) − affinity(αvβ1)`로 "
        "정의하고, Vina 점수가 리간드 크기에 비례하므로 같은 리간드의 차이가 그 "
        "계통 오차를 약분한다고 논증한다. **리간드 쪽 오차는 약분되지만 수용체 쪽 "
        "오프셋은 남는다** — 더 크거나 더 묻힌 포켓은 모든 리간드에 더 좋은 점수를 "
        "주므로, Δ는 \"이 리간드가 αvβ1을 선호한다\"와 \"이 아이소폼이 점수를 후하게 "
        "준다\"를 구분하지 못한다. 선택성은 정확히 그 구분이다.",
        "",
        "따라서 각 수용체에서 대상 집합 전체를 affinity로 순위화하고",
        "",
        "```",
        "selectivity(L) = 백분위_αvβ1(L) − 백분위_아이소폼(L)",
        "```",
        "",
        "를 쓴다. 각 순위가 그 수용체 안에서 정규화되므로 균일한 수용체 오프셋이 "
        "정확히 소거된다. Δ도 사전등록 수치로 함께 보고한다.",
    ]
    if len(isoforms) == 1:
        lines += ["", "**아이소폼 한 종으로 실행했다.** 사전등록된 네 종 중 α5β1·αvβ8은 "
                  "소분자 결정 구조가 없고 αvβ3은 §10.3을 통과하지 못했다. 남은 "
                  "αvβ6에서 두 비선택성 대조군은 실험 IC50이 αvβ1과 거의 같으므로"
                  "(CWHM-12 1.8 대 1.5 nM, GLPG0187 1.3 대 1.4 nM) 이 게이트의 "
                  "변별력은 약하다."]

    for name in isoforms:
        for label, verdicts, table in (("순위 기반 (주)", rank_verdicts,
                                        rank_by_isoform[name]),
                                       ("Δ 기반 (사전등록)", delta_verdicts,
                                        delta_by_isoform[name])):
            verdict = verdicts[name]
            lines += ["", f"## {name} — {label} 보정 게이트: "
                      f"**{'PASS' if verdict['passed'] else 'FAIL'}**", "",
                      "| 화합물 | 값 | 기대 |", "|---|---|---|"]
            for group, expectation in (("selective", "선택적"),
                                       ("non_selective", "비선택적")):
                for control in CALIBRATION[group]:
                    value = control_values(table)[control]
                    shown = "없음" if value is None else f"{value:+.3f}"
                    lines.append(f"| {control} | {shown} | {expectation} |")
            if not verdict["passed"]:
                lines += [""] + [f"- {reason}" for reason in verdict["failed"]]
                status = control_status.get(name, {})
                if status.get("absent_from_isoform"):
                    lines += [
                        "",
                        "**이 FAIL은 게이트가 틀렸다는 뜻이 아니라 돌지 못했다는 "
                        "뜻이다.** 다음 대조군이 αvβ1에는 있고 " + name +
                        "에는 없다: " +
                        ", ".join(status["absent_from_isoform"]) + ". 즉 이 "
                        "아이소폼에 도킹된 적이 없다. 참조 화합물을 같은 수용체에 "
                        "도킹해 `--isoform " + name + "=<생성분자>,<참조>` 로 "
                        "다시 실행해야 판정할 수 있다."]
                if status.get("absent_from_target"):
                    lines += [
                        "",
                        "다음 대조군은 αvβ1 쪽에도 없다: " +
                        ", ".join(status["absent_from_target"]) + "."]

    never_docked = any(control_status.get(n, {}).get("absent_from_isoform")
                       for n in isoforms)
    if never_docked:
        lines += ["", "**게이트 미실행.** 대조군이 아이소폼 쪽에 도킹되지 않아 "
                  "판정 자체가 불가능했다. 이 상태의 선택성 값으로 필터를 돌리면 "
                  "검증되지 않은 축으로 후보를 자르게 된다."]
    elif not all(v["passed"] for v in rank_verdicts.values()):
        lines += ["", "**주 지표의 보정 게이트가 미통과했다. spec §10.4대로 생성 "
                  "분자에 대한 어떤 선택성 진술도 하지 않으며 판정 기준 6번을 "
                  "평가 불가로 기록한다.**"]

    lines += ["", "## 순위 기반 선택성 (값이 클수록 선택적)", "",
              "| 분자 | rank-sum | "
              + " | ".join(f"sel {n} | Δ {n}" for n in isoforms) + " |",
              "|---|---|" + "---|---|" * len(isoforms)]
    for label in sorted(scores, key=lambda k: -scores[k])[:args.top]:
        cells = " | ".join(
            (f"{rank_by_isoform[n][label]:+.3f}"
             if label in rank_by_isoform[n] else "—") + " | " +
            (f"{delta_by_isoform[n][label]:+.3f}"
             if label in delta_by_isoform[n] else "—")
            for n in isoforms)
        lines.append(f"| {label} | {scores[label]:.2f} | {cells} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"n_target": len(target), "rank_selectivity": rank_by_isoform,
         "deltas": delta_by_isoform, "calibration_rank": rank_verdicts,
         "calibration_delta": delta_verdicts, "rank_sum": scores,
         # A gate that could not run is not a gate that failed, and the two
         # need different fixes. `calibration_ran` false means a control was
         # never docked against the isoform.
         "calibration_ran": not never_docked,
         "control_status": control_status},
        indent=2) + "\n")
    print("\n".join(lines))
    return 0 if all(v["passed"] for v in rank_verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
