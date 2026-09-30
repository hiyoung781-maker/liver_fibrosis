"""How the lead set moves with the affinity cutoff, across §7.7's own range.

THIS IS NOT A RELAXED FILTER. The pre-registered §8.7 cutoff is −7.290,
PLN-1474's best passing pose at seed 42, and the headline result is whatever
that value gives. What this adds is the cutoff's OWN uncertainty, which §7.7
measured before any of this: re-docking PLN-1474 under five seeds put its top
pose anywhere from −7.319 to −6.802. The reference value is a draw from that
distribution, so a lead set computed at one draw is a lead set computed at one
draw, and reporting its sensitivity is part of reporting it.

`results/v2_reproducibility_and_isoforms.md` already tabulates survivor counts
across exactly these cutoffs. This carries the same sweep through to the end
of the filter.

WHAT MAKES THIS DIFFERENT FROM MOVING THE GOALPOST. The cutoffs are not
chosen here; they are the five seeds' measured values plus the median, fixed
before the sweep runs and reported together. A reader sees the whole curve,
including the pre-registered point, rather than the point that gave the
answer someone wanted. If a molecule appears only at the loosest cutoff, the
table says so on the same line.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funnel import PLN1474_CUTOFF
from lead_filter import lead_rows, stages, summarize
from selectivity import label_variants

__all__ = ["CUTOFFS", "sweep"]

# §7.7, results/v2_reproducibility_and_isoforms.md. PLN-1474 re-docked under
# five seeds: top pose −7.319 to −6.802, median −7.126. The pre-registered
# value is the seed-42 draw.
CUTOFFS = (
    (-7.319, "최고 시드"),
    (PLN1474_CUTOFF, "사전등록 (시드 42)"),
    (-7.126, "중앙값 시드"),
    (-6.802, "최악 시드"),
)


def sweep(funnel_csv, admet_csv, selectivity, caco2_mode, reference) -> list:
    out = []
    for cutoff, note in CUTOFFS:
        rows = lead_rows(funnel_csv, admet_csv, cutoff, reference,
                         selectivity, caco2_mode)
        summary = summarize(rows, caco2_mode)
        out.append({"cutoff": cutoff, "note": note, "summary": summary,
                    "leads": summary["leads"],
                    "rows": {r["label"]: r for r in rows}})
    return out


def render(results, caco2_mode, metric, arm) -> str:
    stage_names = stages(caco2_mode)
    lines = [
        f"# 컷오프 민감도 — {arm}",
        "",
        "**이것은 완화된 필터가 아니다.** 사전등록 §8.7 컷오프는 "
        f"{PLN1474_CUTOFF:+.3f}(PLN-1474 최고 통과 포즈, 시드 42)이며 "
        "대표 결과는 그 값이 주는 것이다. 여기 더해진 것은 **컷오프 자신의 "
        "불확실성**이고, 그것은 §7.7이 이 모든 것에 앞서 측정했다 — 같은 "
        "화합물을 시드 5개로 재도킹하면 1위 포즈가 −7.319에서 −6.802까지 "
        "움직인다. 기준값은 그 분포에서 뽑은 한 점이다.",
        "",
        f"선택성 지표: **{metric}**   Caco-2: **{caco2_mode}**",
        "",
        "| 컷오프 | | " + " | ".join(stage_names) + " | 리드 |",
        "|---|---|" + "---|" * (len(stage_names) + 1),
    ]
    for result in results:
        counts = " | ".join(
            str(result["summary"][f"stage_{i}_{s}"])
            for i, s in enumerate(stage_names, start=1))
        mark = " **←**" if abs(result["cutoff"] - PLN1474_CUTOFF) < 1e-9 else ""
        lines.append(f"| {result['cutoff']:+.3f}{mark} | {result['note']} | "
                     f"{counts} | **{len(result['leads'])}** |")

    first_seen = {}
    for result in results:
        for label in result["leads"]:
            first_seen.setdefault(label, result)

    lines += ["", "## 리드로 처음 나타나는 컷오프", "",
              "| 분자 | 처음 나타나는 컷오프 | affinity | 선택성 | 독성 평균 | "
              "Caco-2 |", "|---|---|---|---|---|---|"]
    for label, result in sorted(
            first_seen.items(),
            key=lambda kv: -abs(kv[1]["cutoff"])):
        row = result["rows"][label]
        sel = ("—" if row.get("selectivity") is None
               else f"{row['selectivity']:+.3f}")
        tox = "—" if row.get("toxicity") is None else f"{row['toxicity']:.4f}"
        caco2 = "—" if row.get("caco2") is None else f"{row['caco2']:.3f}"
        at = (f"{result['cutoff']:+.3f}"
              + (" (사전등록)" if abs(result["cutoff"] - PLN1474_CUTOFF) < 1e-9
                 else f" ({result['note']})"))
        lines.append(f"| {label} | {at} | "
                     f"{row['best_passing_affinity']:+.3f} | {sel} | {tox} | "
                     f"{caco2} |")

    preregistered = next(r for r in results
                         if abs(r["cutoff"] - PLN1474_CUTOFF) < 1e-9)
    later = [l for l in first_seen if l not in preregistered["leads"]]
    lines += ["", f"사전등록 컷오프에서의 리드: "
              f"**{len(preregistered['leads'])}**개"
              + (f" — {', '.join(preregistered['leads'])}"
                 if preregistered["leads"] else ""),
              "",
              (f"더 느슨한 컷오프에서만 나타나는 분자: **{len(later)}**개"
               f" — {', '.join(later)}. 이들은 사전등록 기준을 통과하지 "
               "못했으며, 기준값이 다른 시드에서 뽑혔다면 통과했을 분자다. "
               "**리드로 제시할 때 반드시 이 사실을 병기한다.**")
              if later else "더 느슨한 컷오프에서 추가로 나타나는 분자는 없다."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--funnel", required=True)
    p.add_argument("--admet", required=True)
    p.add_argument("--selectivity", type=Path, required=True)
    p.add_argument("--selectivity-isoform", default=None)
    p.add_argument("--selectivity-metric", choices=("delta", "rank"),
                    default="rank")
    p.add_argument("--caco2", choices=("off", "better", "margin"),
                    default="better")
    p.add_argument("--reference", default="PANEL_PLN-1474")
    p.add_argument("--arm", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    data = json.loads(args.selectivity.read_text())
    key = {"delta": "deltas", "rank": "rank_selectivity"}[args.selectivity_metric]
    tables = data.get(key) or {}
    name = args.selectivity_isoform or (
        next(iter(tables)) if len(tables) == 1 else None)
    table = tables.get(name, {})
    sel_reference = None
    for candidate in label_variants(args.reference):
        if candidate in table:
            sel_reference = table[candidate]
            break
    if sel_reference is None:
        sys.stderr.write(
            f"{args.selectivity}: {args.reference} has no {args.selectivity_metric} "
            "value, so there is no reference to filter against.\n")
        return 2
    verdicts = data.get({"delta": "calibration_delta",
                         "rank": "calibration_rank"}[args.selectivity_metric]) or {}
    selectivity = {"values": table, "reference": sel_reference,
                   "calibration_passed": bool((verdicts.get(name) or {}).get("passed")),
                   "calibration_ran": bool(data.get("calibration_ran", True))}

    results = sweep(args.funnel, args.admet, selectivity, args.caco2,
                    args.reference)
    text = render(results, args.caco2, args.selectivity_metric,
                  args.arm or args.funnel)
    if not selectivity["calibration_passed"]:
        text += ("\n**경고:** 이 지표의 §10.4 보정 게이트를 통과하지 않았다. "
                 "선택성 값의 타당성이 미검증이며 모든 행에 병기해야 한다.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
