"""Lead cards: every margin the selection claims, beside the noise on it.

Each filter stage asks whether a molecule beats PLN-1474. None of them asks by
HOW MUCH, and that is the question a lead card has to answer, because section
7.7 measured how large a Uni-Dock margin has to be before it means anything:
re-docking PLN-1474 under five seeds gave SD 0.156 kcal/mol and a range of
0.517 (results/v2_reproducibility_and_isoforms.md). A molecule that beats the
reference by 0.005 kcal/mol has not beaten it.

THE SELECTIVITY AXIS IS NOISIER THAN THE AFFINITY AXIS. Delta is a difference
of two independently docked scores, so if each carries the reference's own SD
the difference carries about sqrt(2) times it. A selectivity margin is
therefore compared against a wider band than an affinity margin, and this file
computes both rather than reusing one threshold for both.

WHAT THIS DOES NOT DO. It does not re-rank and it does not drop anything. The
filter's output is the filter's output; the card says which of its margins
survive the engine's own variability, so a reader can see that two of the four
leads pass the selectivity stage inside the noise on that stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

__all__ = ["SEED_SD", "SEED_RANGE", "DELTA_SD", "rank_noise_band",
           "margin_verdict", "cards"]

# §7.7, PLN-1474 re-docked under five seeds. The reference is the noisiest
# ligand measured (11 rotatable bonds); A1AFA, with 5, gave SD 0.004.
SEED_SD = 0.156
SEED_RANGE = 0.517

# A difference of two independently docked scores. Independence is an
# assumption and is stated as one: the two receptors share a ligand
# conformer search, so the true figure is between SEED_SD and this.
DELTA_SD = SEED_SD * math.sqrt(2)

TOXICITY_ENDPOINTS = ("SR-MMP", "NR-AhR", "DILI")
CONTEXT = ("hERG", "AMES", "Solubility_AqSolDB", "CYP3A4_Veith")


def rank_noise_band(scores, sd: float = SEED_SD) -> float:
    """The noise band for a RANK selectivity margin, in percentile units.

    DELTA_SD is in kcal/mol and a rank selectivity is a difference of
    percentiles, so comparing one to the other is a units error -- 0.221 on a
    scale that runs from -1 to 1 would flag almost everything as noise, and
    did, until this existed.

    The conversion is empirical and needs no assumption about the shape of
    the distribution: perturbing a score by one seed SD moves that molecule
    past however many others lie within `sd` of it, so the rank displacement
    is that count over the population. The median over all molecules is the
    typical displacement, and the band is sqrt(2) times it because the
    selectivity is a difference of two independently ranked receptors.

    The estimate uses the target receptor's distribution for both sides. The
    isoform's own scores are not in the selectivity JSON, and the two
    populations are the same molecules docked twice, so their densities are
    close but not identical -- stated rather than hidden.
    """
    values = sorted(v for v in scores if v is not None)
    n = len(values)
    if n < 3:
        return float("nan")
    import bisect
    displacements = [
        (bisect.bisect_right(values, v + sd) - bisect.bisect_left(values, v))
        / (n - 1)
        for v in values]
    displacements.sort()
    median = displacements[len(displacements) // 2]
    return median * math.sqrt(2)


def margin_verdict(margin: float, sd: float) -> str:
    """How a margin stands against the engine's own seed variability."""
    if margin <= 0:
        return "없음"
    if margin < sd:
        return f"잡음 이내 (< {sd:.3f})"
    if margin < 2 * sd:
        return f"잡음의 1-2배 ({margin / sd:.1f}x)"
    return f"잡음의 {margin / sd:.1f}배"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cards(leads_csv: str, admet_csv: str, affinity_cutoff: float,
          selectivity_reference: float | None, reference: str) -> list:
    with open(leads_csv, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["all_stages"] == "True"]
    with open(admet_csv, newline="") as handle:
        admet = {r["label"]: r for r in csv.DictReader(handle)}
    ref = admet.get(reference)
    if ref is None:
        raise ValueError(
            f"{admet_csv}: no row labelled {reference!r}. Every margin on the "
            "card is measured against it.")

    out = []
    for row in rows:
        affinity = _float(row["best_passing_affinity"])
        selectivity = _float(row["selectivity"])
        card = {
            "label": row["label"],
            "affinity": affinity,
            "affinity_margin": (affinity_cutoff - affinity
                                if affinity is not None else None),
            "selectivity": selectivity,
            "selectivity_margin": (
                selectivity - selectivity_reference
                if selectivity is not None and selectivity_reference is not None
                else None),
            "toxicity": _float(row.get("toxicity")),
            "caco2": _float(row.get("caco2")),
            "caco2_better": row.get("caco2_better_than_reference") == "True",
            "n_poses": row.get("n_poses"),
            "plip_metal": row.get("plip_metal_any_passing") == "True",
            "plip_hbond": row.get("plip_hbond_any_passing") == "True",
            "endpoints": {e: (_float(row.get(e)), _float(ref.get(e)))
                          for e in TOXICITY_ENDPOINTS},
            "context": {c: _float(row.get(c)) for c in CONTEXT if c in row},
        }
        card["affinity_verdict"] = margin_verdict(card["affinity_margin"] or 0,
                                                  SEED_SD)
        card["selectivity_verdict"] = margin_verdict(
            card["selectivity_margin"] or 0, DELTA_SD)
        out.append(card)
    return out


def render(cards_, affinity_cutoff, selectivity_reference, reference) -> str:
    lines = [
        "# 리드 카드 — 주장하는 마진과 그 마진 위의 잡음",
        "",
        f"기준은 {reference}다. affinity 기준값 {affinity_cutoff:+.3f} kcal/mol, "
        + (f"선택성 기준값 {selectivity_reference:+.3f}."
           if selectivity_reference is not None else "선택성 기준값 없음."),
        "",
        f"§7.7은 PLN-1474를 시드 5개로 재도킹해 SD **{SEED_SD:.3f}** kcal/mol, "
        f"폭 **{SEED_RANGE:.3f}**을 측정했다. 필터의 각 단계는 기준을 "
        "이겼는지만 묻고 **얼마나** 이겼는지는 묻지 않는다. 0.005 kcal/mol로 "
        "이긴 것은 이긴 것이 아니다.",
        "",
        f"Δ는 독립적으로 도킹된 두 점수의 차이이므로 잡음이 약 √2배인 "
        f"**{DELTA_SD:.3f}**로 커진다. 두 수용체가 리간드 형태 탐색을 "
        "공유하므로 실제 값은 두 수치 사이에 있다.",
        "",
        "| 분자 | affinity | 마진 | 판정 | 선택성 Δ | 마진 | 판정 |",
        "|---|---|---|---|---|---|---|",
    ]
    for card in cards_:
        lines.append(
            f"| {card['label']} | {card['affinity']:+.3f} | "
            f"{card['affinity_margin']:+.3f} | {card['affinity_verdict']} | "
            + (f"{card['selectivity']:+.3f} | {card['selectivity_margin']:+.3f} | "
               f"{card['selectivity_verdict']} |"
               if card["selectivity_margin"] is not None else "— | — | — |"))

    surviving = [c for c in cards_
                 if (c["selectivity_margin"] or 0) >= DELTA_SD]
    lines += ["", f"**선택성 마진이 Δ 잡음을 넘는 리드: {len(surviving)}/"
              f"{len(cards_)}** — "
              + (", ".join(c["label"] for c in surviving) if surviving
                 else "없음") + ".",
              "", "나머지는 선택성 단계를 통과했으나 그 통과가 엔진의 시드 "
              "변동 안에서 일어났다. 필터의 결과를 바꾸지 않되, 선택성을 "
              "근거로 그 분자를 내세우지는 않는다."]

    for card in cards_:
        lines += ["", f"## {card['label']}", "",
                  f"- affinity **{card['affinity']:+.3f}** kcal/mol "
                  f"({card['affinity_verdict']}), 포즈 {card['n_poses']}개",
                  f"- 선택성 Δ **{card['selectivity']:+.3f}** "
                  f"({card['selectivity_verdict']})",
                  f"- 독성 평균 **{card['toxicity']:.4f}**"]
        for endpoint, (value, reference_value) in card["endpoints"].items():
            ratio = (f"기준의 {value / reference_value * 100:.0f}%"
                     if value is not None and reference_value else "—")
            lines.append(f"    - {endpoint} {value:.4f} 대 "
                         f"{reference_value:.4f} ({ratio})")
        lines.append(
            f"- Caco-2 {card['caco2']:.3f} "
            + ("(기준보다 나음, 게이트 아님)" if card["caco2_better"]
               else "(기준보다 못함, 게이트 아님)"))
        plip = []
        plip.append("금속 O" if card["plip_metal"] else "금속 X")
        plip.append("수소결합 O" if card["plip_hbond"] else "수소결합 X")
        lines.append(f"- PLIP: {', '.join(plip)}"
                     + ("" if card["plip_hbond"] else
                        "  ← §8.0 상호작용 게이트를 적용했다면 탈락한다"))
        if card["context"]:
            lines.append("- 필터 밖 지표: " + ", ".join(
                f"{k} {v:.3f}" for k, v in card["context"].items()
                if v is not None))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--leads", required=True, help="lead_filter.py --out-csv")
    p.add_argument("--admet", required=True)
    p.add_argument("--affinity-cutoff", type=float, required=True)
    p.add_argument("--selectivity", type=Path,
                    help="selectivity.py JSON, for the reference's own Delta")
    p.add_argument("--selectivity-isoform", default=None)
    p.add_argument("--selectivity-reference", type=float,
                    help="the reference's own Delta, when the selectivity "
                         "JSON is not to hand. Overrides the JSON.")
    p.add_argument("--reference", default="PANEL_PLN-1474")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    selectivity_reference = args.selectivity_reference
    if selectivity_reference is None and args.selectivity and args.selectivity.exists():
        data = json.loads(args.selectivity.read_text())
        tables = data.get("deltas") or {}
        name = args.selectivity_isoform or (
            next(iter(tables)) if len(tables) == 1 else None)
        table = tables.get(name, {})
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from selectivity import label_variants
        for key in label_variants(args.reference):
            if key in table:
                selectivity_reference = table[key]
                break

    try:
        built = cards(args.leads, args.admet, args.affinity_cutoff,
                      selectivity_reference, args.reference)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    text = render(built, args.affinity_cutoff, selectivity_reference,
                  args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
