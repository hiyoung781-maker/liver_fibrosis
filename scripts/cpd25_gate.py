"""§8.2's interaction gate, decided from compound 25 exactly as pre-registered.

The gate is `Ca501 metal complex` AND a secondary polar anchor. Only the
anchor's DEFINITION was left open, and §8.2 fixed the rule for choosing it
before any docking was run:

  judge on compound 25's BEST-AFFINITY pose AMONG THOSE where PLIP reports a
  Ca501 metal complex, then

  1. PLIP reports a beta1-Asn224 BACKBONE O hydrogen bond
     -> gate unchanged:  metal AND Asn224 backbone H-bond
  2. no Asn224, but a beta1-Leu225 BACKBONE O hydrogen bond
     -> gate relaxed:    metal AND any beta1 backbone O in 223-226
  3. neither
     -> gate reduced:    metal complex alone; the secondary contact is
                         demoted to an observation
  4. compound 25 yields no metal-complex pose at all
     -> gate unchanged, and the fact is recorded

WHY THE RULE PICKS THAT POSE. Not the best pose overall: a top-scoring pose
that misses the MIDAS is not a model of how this chemotype binds, and reading
the anchor off it would define the gate from a pose the gate itself rejects.

WHAT THIS SCRIPT IS NOT. It does not choose between the four cases on any
grounds other than what PLIP reports. The reason §8.2 was written before
docking is that the leads are known by the time it runs, and one of them
(gen_03673) has no PLIP hydrogen bond at all -- so cases 1 and 2 remove it
while case 3 keeps it. Running the registered rule mechanically is what keeps
that from being a choice.

COMPOUND 25 IS IN THE TRAINING SET (data/folds/actives_core_fold3_train.smi).
That is not a problem for this use -- the gate is a statement about binding
geometry, not about generalisation -- but it is recorded here so the reader
does not have to discover it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = ["CASES", "GATE_RULES", "decide", "gate_predicate"]

CPD25 = "CHEMBL5532604"

CASES = {
    1: "Asn224 backbone H-bond present",
    2: "no Asn224, Leu225 backbone H-bond present",
    3: "neither secondary anchor present",
    4: "no pose with a Ca501 metal complex",
}

# What each case makes the gate. The key is the column combination applied to
# a pose row; `columns` names the plip.csv columns the filter must AND.
GATE_RULES = {
    1: {"name": "metal + Asn224 backbone H-bond",
        "columns": ["metal_ca501", "hbond_asn224"]},
    2: {"name": "metal + any beta1 223-226 backbone H-bond",
        "columns": ["metal_ca501", "hbond_anchor_window"]},
    3: {"name": "metal complex alone",
        "columns": ["metal_ca501"]},
    4: {"name": "metal + Asn224 backbone H-bond (unchanged; cpd 25 gave no "
                "metal-complex pose)",
        "columns": ["metal_ca501", "hbond_asn224"]},
}


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def decide(rows: list) -> dict:
    """Apply §8.2 to compound 25's PLIP rows. `rows` are plip.csv dicts."""
    usable = [r for r in rows if r.get("status") == "ok"]
    metal = [r for r in usable if _int(r.get("metal_ca501")) == 1]
    if not metal:
        return {"case": 4, "reason": CASES[4], "pose": None,
                "n_poses": len(usable), "n_metal_poses": 0,
                "gate": GATE_RULES[4]}

    # "the best-affinity pose among those reporting a metal complex"
    judged = min(metal, key=lambda r: (_float(r.get("affinity")) is None,
                                       _float(r.get("affinity"))))
    if _int(judged.get("hbond_asn224")) == 1:
        case = 1
    elif _int(judged.get("hbond_anchor_window")) == 1:
        case = 2
    else:
        case = 3
    return {"case": case, "reason": CASES[case],
            "pose": {"pose": judged.get("pose"),
                     "affinity": _float(judged.get("affinity")),
                     "hbond_asn224": _int(judged.get("hbond_asn224")),
                     "hbond_asn224_sidechain": _int(
                         judged.get("hbond_asn224_sidechain")),
                     "hbond_anchor_window": _int(
                         judged.get("hbond_anchor_window")),
                     "hbond_bb_residues": judged.get("hbond_bb_residues"),
                     "tyr178_contact": _int(judged.get("tyr178_contact")),
                     "hydrophobic_pocket": _int(
                         judged.get("hydrophobic_pocket"))},
            "n_poses": len(usable), "n_metal_poses": len(metal),
            "gate": GATE_RULES[case]}


def gate_predicate(columns: list):
    """A function testing one plip.csv row against the decided gate."""
    def passes(row: dict) -> bool:
        return all(_int(row.get(c)) == 1 for c in columns)
    return passes


def render(verdict: dict, label: str) -> str:
    gate = verdict["gate"]
    lines = [
        "# §8.2 상호작용 게이트 — compound 25로 판정",
        "",
        f"판정 대상: `{label}` (Sabat 2024 compound 25). PLIP이 Ca501 metal "
        "complex를 보고하는 포즈 중 affinity 최상위 포즈로 판정한다 — "
        "MIDAS를 놓친 최상위 포즈는 이 화학형이 어떻게 결합하는지의 모델이 "
        "아니므로, 거기서 앵커를 읽으면 게이트가 스스로 기각할 포즈로 "
        "게이트를 정하게 된다.",
        "",
        f"- 사용 가능한 포즈 **{verdict['n_poses']}**개 중 metal complex "
        f"보고 **{verdict['n_metal_poses']}**개",
        f"- **판정: 경우 {verdict['case']} — {verdict['reason']}**",
        f"- **게이트: {gate['name']}**",
        f"- 적용 컬럼: `{'` AND `'.join(gate['columns'])}`",
        "",
    ]
    pose = verdict.get("pose")
    if pose:
        lines += ["판정 포즈의 PLIP 결과:", "",
                  "| 항목 | 값 |", "|---|---|",
                  f"| 포즈 | {pose['pose']} |",
                  f"| affinity | {pose['affinity']:+.3f} kcal/mol |",
                  f"| Asn224 backbone 수소결합 | {pose['hbond_asn224']} |",
                  f"| Asn224 곁사슬 수소결합 (기록용) | "
                  f"{pose['hbond_asn224_sidechain']} |",
                  f"| β1 223–226 backbone 수소결합 | "
                  f"{pose['hbond_anchor_window']} |",
                  f"| backbone O 수소결합 잔기 | "
                  f"{pose['hbond_bb_residues'] or '없음'} |",
                  f"| Tyr178 접촉 (게이트 아님) | {pose['tyr178_contact']} |",
                  f"| 소수성 포켓 (게이트 아님) | {pose['hydrophobic_pocket']} |",
                  ""]
    lines += [
        "§8.2는 **Tyr178 π-stacking과 Leu225 측쇄 소수성 접촉을 어떤 경우에도 "
        "게이트로 삼지 않는다**고 사전등록했다 — 전자는 통과 집단에서 방향족 "
        "고리 농축이 1.03배(=없음)로 측정되었고 compound 1의 벤질기에 특이적인 "
        "접촉이며, 후자는 v1에서 64.3%로 변별력이 약했다. 둘 다 위 표에 "
        "관측치로 남는다.",
        "",
        "**기록:** compound 25는 `data/folds/actives_core_fold3_train.smi`에 "
        "포함된 학습 셋 분자다. 게이트는 결합 기하에 대한 진술이지 일반화에 "
        "대한 진술이 아니므로 이 용도에는 문제가 없으나, 독자가 스스로 "
        "발견하게 두지 않는다.",
        "",
        "**기록:** §8.2의 판정 규칙은 도킹 전에 확정되었으나, 그 적용은 리드가 "
        "산출된 뒤에 이루어졌다. 규칙이 먼저 고정되어 있었다는 점이 이것을 "
        "사후 조정과 구분하며, 적용 시점은 여기에 남긴다.",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plip", required=True,
                    help="plip_batch.py --out-csv holding compound 25's poses")
    p.add_argument("--label", default=CPD25,
                    help=f"compound 25's label in that CSV (default {CPD25})")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--out-json", type=Path)
    args = p.parse_args(argv)

    # dock_panel.py writes PANEL_-prefixed labels; selectivity_refs.smi and
    # the ChEMBL id are bare. Fifth place this has bitten the campaign, so it
    # goes through the one function that knows the answer.
    from selectivity import label_variants
    wanted = set(label_variants(args.label))
    with open(args.plip, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r.get("label") in wanted]
    if not rows:
        sys.stderr.write(
            f"{args.plip}: no rows labelled any of {sorted(wanted)}. §8.2 "
            "decides the gate from compound 25 and cannot be applied without "
            "it; dock and run PLIP on it first.\n")
        return 2
    if "hbond_anchor_window" not in rows[0]:
        sys.stderr.write(
            f"{args.plip} predates the anchor-window columns, so case 2 "
            "cannot be evaluated and a verdict here would silently be "
            "restricted to cases 1, 3 and 4. Re-run plip_batch.py.\n")
        return 2

    verdict = decide(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(verdict, args.label))
    if args.out_json:
        args.out_json.write_text(json.dumps(verdict, indent=2) + "\n")
    print(render(verdict, args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
