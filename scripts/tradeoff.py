"""Trade-off cases among the molecules whose BINDING the filter supports.

THIS DOES NOT PRODUCE LEADS. The pre-registered five-stage filter produces
leads, and for TL-C it produces one. This reports the molecules that cleared
every STRUCTURAL stage -- geometry, affinity against PLN-1474, and section
8.2's interaction gate -- and then shows where each sits on the two axes that
actually trade off against each other, so a reader can see the shape of the
compromise instead of a single survivor.

Every molecule here failed the pre-registered filter unless it is marked
LEAD. Presenting one as a lead because it looks good on one axis is the thing
this file exists to make harder, not easier: each row carries what it failed
and by how much.

THE MARGINS ARE COMPARED AGAINST NOISE, NOT AGAINST ZERO. Section 7.7
re-docked PLN-1474 under five seeds and measured SD 0.156 kcal/mol, range
0.517. A selectivity Delta is a difference of two independently docked
scores, so its band is about sqrt(2) wider. A molecule that beats the
reference by less than that has not been shown to beat it, and saying so is
the difference between a case worth discussing and a number.

THE CATEGORIES, and what each one may be claimed to be:

  LEAD          passes every pre-registered stage.
  SELECTIVITY   selectivity margin clears the noise band, one toxicity
                endpoint above the reference. A genuine trade-off case.
  SAFETY        every toxicity endpoint below the reference, selectivity
                margin inside the noise. Affinity and tolerability, NO
                selectivity claim.
  BORDERLINE    affinity within the reference's own seed range of the
                cutoff. Neither pass nor fail is supportable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from funnel import PLN1474_CUTOFF
from lead_cards import DELTA_SD, SEED_RANGE, SEED_SD, margin_verdict

__all__ = ["TOXICITY_ENDPOINTS", "categorise", "pareto_front", "rows"]

TOXICITY_ENDPOINTS = ("SR-MMP", "NR-AhR", "DILI")
REFERENCE = "PANEL_PLN-1474"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pareto_front(entries: list) -> list:
    """Non-dominated on (selectivity up, toxicity composite down).

    Two axes, not four: a front over three toxicity endpoints plus
    selectivity is almost entirely front, which tells a reader nothing. The
    composite is the pre-registered §9.2 mean, and the per-endpoint values
    travel on every row so the mean never hides a single bad one.
    """
    front = []
    for candidate in entries:
        if candidate["selectivity"] is None or candidate["toxicity"] is None:
            continue
        dominated = any(
            other is not candidate
            and other["selectivity"] is not None and other["toxicity"] is not None
            and other["selectivity"] >= candidate["selectivity"]
            and other["toxicity"] <= candidate["toxicity"]
            and (other["selectivity"] > candidate["selectivity"]
                 or other["toxicity"] < candidate["toxicity"])
            for other in entries)
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda e: -e["selectivity"])


def categorise(entry: dict) -> str:
    # BORDERLINE is tested FIRST, before LEAD. A molecule that clears the
    # affinity cutoff by less than the reference's own seed range has not
    # been shown to clear it, and calling that a lead states more than the
    # measurement supports -- whatever it does on the other axes.
    if entry["borderline_affinity"]:
        return "BORDERLINE"
    if entry["all_stages"]:
        return "LEAD"
    margin = entry["selectivity_margin"]
    if margin is not None and margin >= DELTA_SD and len(entry["tox_worse"]) == 1:
        return "SELECTIVITY"
    if not entry["tox_worse"] and entry["admet_status"] == "ok":
        return "SAFETY"
    return "other"


def rows(funnel_csv, admet_csv, selectivity, reference=REFERENCE) -> list:
    with open(funnel_csv, newline="") as handle:
        funnel = list(csv.DictReader(handle))
    with open(admet_csv, newline="") as handle:
        admet = {r["label"]: r for r in csv.DictReader(handle)}
    ref = admet.get(reference)
    if ref is None:
        raise ValueError(f"{admet_csv}: no row labelled {reference!r}")
    ref_endpoints = {e: _float(ref.get(e)) for e in TOXICITY_ENDPOINTS}

    values = selectivity.get("values") or {}
    sel_reference = selectivity.get("reference")

    out = []
    for row in funnel:
        if row.get("geometry_pass") != "True":
            continue
        affinity = _float(row.get("best_passing_affinity"))
        if affinity is None:
            continue
        # The structural stages only. Everything past this point is what
        # trades off; everything before it is whether the pose is a pose.
        if affinity >= PLN1474_CUTOFF:
            continue
        if row.get("plip_gate_pass") == "False":
            continue

        label = row["label"]
        prediction = admet.get(label)
        selectivity_value = values.get(label)
        worse = []
        if prediction:
            for endpoint in TOXICITY_ENDPOINTS:
                value = _float(prediction.get(endpoint))
                if value is None or value >= ref_endpoints[endpoint]:
                    worse.append(endpoint)
        else:
            worse = list(TOXICITY_ENDPOINTS)

        composite = None
        if prediction:
            got = [_float(prediction.get(e)) for e in TOXICITY_ENDPOINTS]
            if all(v is not None for v in got):
                composite = sum(got) / len(got)

        entry = {
            "label": label,
            "affinity": affinity,
            "affinity_margin": PLN1474_CUTOFF - affinity,
            "selectivity": selectivity_value,
            "selectivity_margin": (selectivity_value - sel_reference
                                   if selectivity_value is not None
                                   and sel_reference is not None else None),
            "toxicity": composite,
            "tox_worse": worse,
            "endpoints": {e: _float(prediction.get(e)) if prediction else None
                          for e in TOXICITY_ENDPOINTS},
            "admet_status": "ok" if prediction else "missing",
            "plip_hbond": row.get("plip_hbond_any_passing") == "True",
            "caco2": _float(prediction.get("Caco2_Wang")) if prediction else None,
            # Inside the reference's own seed range of the cutoff.
            "borderline_affinity": abs(affinity - PLN1474_CUTOFF) < SEED_RANGE,
        }
        entry["selectivity_pass"] = (entry["selectivity_margin"] is not None
                                     and entry["selectivity_margin"] > 0)
        # Passing the stage and having shown something are different. The
        # stage asks only that the margin be positive; this says whether the
        # margin is larger than the noise on it. gen_01883 passes by 0.005
        # against a band of 0.221.
        entry["selectivity_within_noise"] = (
            entry["selectivity_pass"]
            and entry["selectivity_margin"] < DELTA_SD)
        entry["all_stages"] = (entry["selectivity_pass"] and not worse
                               and entry["admet_status"] == "ok")
        entry["category"] = categorise(entry)
        out.append(entry)
    return out


def render(entries, ref_endpoints, sel_reference, arm) -> str:
    front = pareto_front(entries)
    front_labels = {e["label"] for e in front}
    lines = [
        f"# 논의 후보 — {arm}",
        "",
        "**이 문서는 리드를 만들지 않는다.** 리드는 사전등록 5단 필터가 "
        "정하며 TL-C에서 1개다. 여기 있는 것은 **구조 단계를 모두 통과한** "
        "분자들 — geometry(§8.5b), affinity < "
        f"{PLN1474_CUTOFF:+.3f}(§8.7), 상호작용 게이트(§8.2) — 이고, "
        "그 다음 실제로 서로 상충하는 두 축 위에 어디 있는지를 보인다. "
        "`LEAD` 표시가 없는 행은 **전부 사전등록 필터를 통과하지 못했다.**",
        "",
        f"마진은 0이 아니라 잡음과 비교한다. §7.7이 PLN-1474를 시드 5개로 "
        f"재도킹해 SD **{SEED_SD:.3f}** kcal/mol, 폭 **{SEED_RANGE:.3f}**을 "
        f"측정했다. Δ는 독립적으로 도킹된 두 점수의 차이이므로 잡음 대역이 "
        f"약 √2배인 **{DELTA_SD:.3f}**다.",
        "",
        f"구조 단계 통과: **{len(entries)}**개. Pareto front(선택성↑ · 독성↓) "
        f"위: **{len(front)}**개.",
        "",
        "| 분류 | 분자 | affinity | 마진 | 선택성 Δ | 마진 | 독성 평균 | "
        "기준 초과 엔드포인트 | Pareto |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = {"LEAD": 0, "SELECTIVITY": 1, "SAFETY": 2, "BORDERLINE": 3,
             "other": 4}
    shown = [e for e in entries if e["category"] != "other"
             or e["label"] in front_labels]
    shown.sort(key=lambda e: (order[e["category"]],
                              -(e["selectivity_margin"] or -99)))
    for e in shown:
        sel = ("—" if e["selectivity"] is None else f"{e['selectivity']:+.3f}")
        selm = ("—" if e["selectivity_margin"] is None
                else f"{e['selectivity_margin']:+.3f}")
        tox = "—" if e["toxicity"] is None else f"{e['toxicity']:.4f}"
        worse = ", ".join(e["tox_worse"]) if e["tox_worse"] else "없음"
        if e.get("selectivity_within_noise"):
            selm += " ⚠"
        lines.append(
            f"| {e['category']} | {e['label']} | {e['affinity']:+.3f} | "
            f"{e['affinity_margin']:+.3f} | {sel} | {selm} | {tox} | {worse} | "
            f"{'●' if e['label'] in front_labels else ''} |")

    lines += ["", "## 분류가 주장할 수 있는 것", "",
              "- **LEAD** — 사전등록 단계를 전부 통과. 유일하게 리드라고 "
              "부를 수 있다.",
              "- **SELECTIVITY** — 선택성 마진이 잡음 대역을 넘고, 독성 "
              "엔드포인트 하나가 기준을 초과. 실제 trade-off 사례다. "
              "**독성 기준 미달이므로 리드가 아니다.**",
              "- **SAFETY** — 독성 세 엔드포인트가 모두 기준 이하이나 "
              "선택성 마진이 잡음 이내. 친화도와 내약성은 말할 수 있고 "
              "**선택성은 말할 수 없다.**",
              "- **BORDERLINE** — affinity가 컷오프에서 기준값 자신의 시드 "
              f"폭({SEED_RANGE:.3f}) 이내. 통과도 탈락도 근거가 없다.",
              "",
              f"**⚠** — 선택성 단계는 통과했으나 마진이 Δ 잡음 대역"
              f"({DELTA_SD:.3f}) 이내다. 단계 통과와 무언가를 보인 것은 "
              "다르다. 이 표시가 붙은 분자에 대해 **선택성을 주장하지 "
              "않는다.**",
              "", "Pareto front는 선택성(↑)과 §9.2 독성 평균(↓) 두 축에서 "
              "지배당하지 않는 분자다. 세 엔드포인트를 각각 축으로 놓으면 "
              "거의 전부가 front에 올라 아무것도 말해주지 않으므로 평균을 "
              "쓰되, 엔드포인트별 값을 같은 행에 실어 평균이 나쁜 하나를 "
              "가리지 못하게 한다."]

    for e in shown[:8]:
        sel = "—" if e["selectivity"] is None else f"{e['selectivity']:+.3f}"
        verdict = margin_verdict(e["selectivity_margin"] or 0, DELTA_SD)
        lines += ["", f"### {e['label']}  ({e['category']})", "",
                  f"- affinity **{e['affinity']:+.3f}** "
                  f"({margin_verdict(e['affinity_margin'], SEED_SD)})",
                  f"- 선택성 Δ **{sel}** ({verdict})"]
        for endpoint, value in e["endpoints"].items():
            if value is None:
                continue
            ratio = (f"기준의 {value / ref_endpoints[endpoint] * 100:.0f}%"
                     if ref_endpoints[endpoint] else "—")
            mark = " ← 기준 초과" if endpoint in e["tox_worse"] else ""
            lines.append(f"    - {endpoint} {value:.4f} 대 "
                         f"{ref_endpoints[endpoint]:.4f} ({ratio}){mark}")
        if e["caco2"] is not None:
            lines.append(f"- Caco-2 {e['caco2']:.3f} (게이트 아님)")
        lines.append(f"- PLIP 수소결합: {'있음' if e['plip_hbond'] else '없음'}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--funnel", required=True)
    p.add_argument("--admet", required=True)
    p.add_argument("--selectivity", type=Path, required=True)
    p.add_argument("--selectivity-isoform", default=None)
    p.add_argument("--selectivity-metric", choices=("delta", "rank"),
                    default="delta")
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument("--arm", default="")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    data = json.loads(args.selectivity.read_text())
    key = {"delta": "deltas", "rank": "rank_selectivity"}[args.selectivity_metric]
    tables = data.get(key) or {}
    name = args.selectivity_isoform or (
        next(iter(tables)) if len(tables) == 1 else None)
    table = tables.get(name, {})
    from selectivity import label_variants
    sel_reference = None
    for candidate in label_variants(args.reference):
        if candidate in table:
            sel_reference = table[candidate]
            break
    selectivity = {"values": table, "reference": sel_reference}

    try:
        entries = rows(args.funnel, args.admet, selectivity, args.reference)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    with open(args.admet, newline="") as handle:
        ref = {r["label"]: r for r in csv.DictReader(handle)}[args.reference]
    ref_endpoints = {e: _float(ref.get(e)) for e in TOXICITY_ENDPOINTS}

    text = render(entries, ref_endpoints, sel_reference, args.arm or args.funnel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
