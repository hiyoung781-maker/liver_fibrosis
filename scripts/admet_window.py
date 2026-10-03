"""§5.5: apply the pre-registered Caco-2 and toxicity cuts BEFORE docking.

WHY THIS RUNS BEFORE DOCKING, not after. build_survivors.py's own docstring
declares the rule - "The cuts, cheapest first, exactly as section 8 orders
them" - and the funnel violated it. v3 docked its whole survivor set, then
discarded 99.3% of what passed the geometry and affinity gates on an ADMET
prediction that costs minutes: 295 -> affinity 238 -> toxicity 2. At v4's scale
that is 15 GPU-hours spent to produce molecules a cheap predictor would have
rejected first.

THE REORDERING IS NOT THE WHOLE POINT, THOUGH. Measured on v3's own data, the
set that passes affinity AND reaches BCS high permeability AND clears §9.2's
toxicity thresholds is EMPTY - 37 molecules reach high permeability with
affinities of -9.107 to -7.85, far stronger than v3's chosen leads, and every
one of them fails toxicity (tox mean 0.48-0.79 against 0.07-0.19 for the clean
ones). corr(Caco2_Wang, SR-MMP) = +0.489: permeability and SR-MMP oppose each
other. An empty intersection cannot be found by reordering filters, and it
cannot be found by Pareto selection either - a front over an empty region
returns the trade-off curve, not a usable lead. What running the cuts early
buys is learning that in minutes instead of after the docking.

ABSOLUTE THRESHOLDS, NOT PLN-1474's. The campaign used `caco2 > PLN-1474` as
its permeability bar, but PLN-1474 predicts 2.59e-06 cm/s - clinging to the
BOTTOM of the BCS moderate band. A bar set there says only "not catastrophically
bad". It is the same defect section 2.1 found in the old TPSA ceiling, which was
anchored on two alphaV compounds and penalised one of them: a reference-anchored
threshold carries no information when the reference itself sits on the boundary.

Pre-registered in the design note BEFORE these numbers were computed:

  Tier 1 (primary)   Caco2_Wang > -5.0   BCS/FDA high,     Fa 80-100%
  Tier 2 (fallback)  Caco2_Wang > -5.7   BCS moderate low, Fa 50-80%
  toxicity           section 9.2 unchanged - all three endpoints strictly below
                     the 60th percentile of 559 approved acidic drugs

Tier 2 is declared in advance because v3's Tier 1 intersection was empty, so
v4's may be too; loosening the threshold after seeing a zero would be exactly
the post-hoc move section 4.6 exists to prevent. Both counts are always
reported. No threshold is invented here - toxicity comes from
approved_drug_gate.json, permeability from the BCS bands.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lead_filter import CACO2_ENDPOINT, TOXICITY_ENDPOINTS  # noqa: E402

__all__ = ["CACO2_TIERS", "toxicity_thresholds", "read_admet", "admet_verdict",
           "apply_window", "amended_window", "DILI_ENDPOINT", "DILI_MAX_DEFAULT",
           "CONVERSION_BANDS", "DEFAULT_GATE_JSON"]

# log10(Papp in cm/s). BCS/FDA: high > -5.0 (Papp > 10e-6, Fa 80-100%);
# moderate -5.7..-5.0 (2-10e-6, Fa 50-80%); low < -5.7 (Fa < 50%).
CACO2_TIERS = (("tier1_bcs_high", -5.0), ("tier2_bcs_moderate", -5.7))

DEFAULT_GATE_JSON = "results/v3_metal_asn224/approved_drug_gate.json"

DILI_ENDPOINT = "DILI"

# THE AMENDED RULE (section 9.3.1 change, recorded in results/v4_threshold_ledger.md).
#
# The pre-registered rule is the section 9.2 conjunction, and on v4 it returns 2
# molecules - too few to fill an optimisation track or to support the comparison
# against v3's three leads. The amendment is NOT "the conjunction gave too few".
# Three things justify it, and the counts under the pre-registered rule are
# reported beside the amended ones so the change cannot hide its own cost.
#
# (1) THE CONJUNCTION CONTRADICTS THIS STAGE'S OWN DESIGN. optimize_leads.py:
# "Parents are molecules that cleared the structure and affinity stages, not the
# handful that also cleared toxicity ... toxicity is the axis being repaired, so
# pre-filtering on it would discard exactly the molecules the stage exists to
# improve." Gating all three endpoints before docking does that pre-filtering.
#
# (2) ONE ENDPOINT CARRIES THE CONSTRAINT. Measured on v4's 13,767: DILI passes
# 1.7% with a library median of 0.9649 against a 0.669635 threshold, while
# SR-MMP passes 20.7% and NR-AhR 7.2%. Dropping DILI alone takes the conjunction
# from 73 to 807; dropping either other endpoint leaves 105 or 101. So DILI is
# the axis, and SR-MMP/NR-AhR belong to the repair stage, where
# select_analogues already re-applies them as "no worse than the parent".
#
# (3) 0.75 IS DERIVED, NOT CHOSEN TO HIT A COUNT. v3's 1,990 analogues give the
# rate at which an unguided high-similarity analogue crosses the DILI threshold,
# as a function of where its parent sat (CONVERSION_BANDS below). The rate falls
# off a cliff: 54.1% just above the threshold, 8.5%, 2.3%, then ZERO in 825
# pairs above 0.92. A parent above the cliff cannot be repaired at this sample
# budget, so the window's job is to place parents where the search succeeds.
# 0.75 is the top of the 54% band.
DILI_MAX_DEFAULT = 0.75

# Measured on results/v3_opt/analogues_admet.csv (1,990 analogues, 32 per parent,
# run_type = "sampling" - unguided, no scoring function). (parent DILI band,
# pairs, converted, rate). This is the UNDIRECTED baseline: v3 put no toxicity
# term in generation, which is why the distribution did not move (49.0% improved,
# 51.0% worsened, median delta +0.0004). It bounds what this sample budget
# yields, NOT what chemistry allows - a larger budget buys conversion in the
# weaker bands (~44 samples/parent for 0.85-0.92, >800 above 0.92), and ADMET
# throughput, not generation, is the ceiling there.
CONVERSION_BANDS = (
    ((0.669635, 0.750), 74, 40, 0.541),
    ((0.750, 0.850), 272, 23, 0.085),
    ((0.850, 0.920), 434, 10, 0.023),
    ((0.920, 1.010), 825, 0, 0.000),
)


def toxicity_thresholds(path: str = DEFAULT_GATE_JSON,
                        percentile: float = 60.0) -> dict[str, float]:
    """The §9.2 thresholds, read from approved_drug_gate.py's own output.

    Read rather than restated. The campaign has already been bitten twice by a
    constant copied into a second place and then frozen there - the TPSA window
    (§2.2) and PROPERTY_WINDOW (§5.4) - and this is the same shape of value.
    """
    data = json.loads(open(path).read())
    if data.get("percentile") != percentile:
        raise ValueError(f"{path}: adopted percentile is {data.get('percentile')}, "
                         f"not {percentile}; §9.2 pins {percentile}")
    for entry in data["sweep"]:
        if entry["percentile"] == percentile:
            thresholds = {e: float(entry["thresholds"][e])
                          for e in TOXICITY_ENDPOINTS}
            return thresholds
    raise ValueError(f"{path}: no sweep entry at percentile {percentile}")


def read_admet(path: str) -> dict[str, dict]:
    """{smiles: prediction} from an admet_predict output CSV.

    Keyed on the SMILES string because that is what admet_predict echoes back;
    the caller feeds it the survivors' column-1 (tautomer-canonical) SMILES, so
    the join is exact rather than a re-canonicalisation that could drift.
    """
    out: dict[str, dict] = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in (CACO2_ENDPOINT, *TOXICITY_ENDPOINTS)
                   if c not in (reader.fieldnames or [])]
        if missing:
            raise KeyError(f"{path}: missing column(s) {missing}")
        key = "smiles" if "smiles" in (reader.fieldnames or []) else reader.fieldnames[0]
        for row in reader:
            out[row[key]] = row
    return out


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def admet_verdict(prediction: dict | None, caco2_floor: float,
                  thresholds: dict[str, float]) -> dict:
    """Per-molecule verdict against one Caco-2 tier and the §9.2 toxicity rule.

    A MISSING prediction fails, it does not pass by default. This gate decides
    what gets docked; treating an absent value as acceptable would let exactly
    the molecules the predictor could not handle through to the expensive stage.
    """
    if not prediction:
        return {"caco2": None, "caco2_pass": False,
                "toxicity_pass": False, "pass": False}
    caco2 = _float(prediction.get(CACO2_ENDPOINT))
    caco2_pass = caco2 is not None and caco2 > caco2_floor
    # §9.2: EVERY endpoint strictly below its threshold, not their mean - the
    # same comparison lead_filter.py makes, so the two stages cannot disagree
    # about what "clears toxicity" means.
    tox_pass = True
    for endpoint in TOXICITY_ENDPOINTS:
        value = _float(prediction.get(endpoint))
        if value is None or value >= thresholds[endpoint]:
            tox_pass = False
            break
    return {"caco2": caco2, "caco2_pass": caco2_pass,
            "toxicity_pass": tox_pass, "pass": caco2_pass and tox_pass}


def apply_window(entries: list[tuple[str, str]], admet: dict[str, dict],
                 thresholds: dict[str, float]) -> dict:
    """Counts and surviving entries for EVERY pre-registered tier.

    Both tiers are always computed, never just the one that happens to be
    non-empty: the fallback only stays honest if its count is reported next to
    the primary's rather than quietly replacing it.
    """
    result = {"input": len(entries), "tiers": {}, "toxicity_pass": 0,
              "admet_missing": 0}
    for smiles, _label in entries:
        if smiles not in admet:
            result["admet_missing"] += 1
    # Toxicity is tier-independent, so it is counted once.
    for smiles, _label in entries:
        v = admet_verdict(admet.get(smiles), float("-inf"), thresholds)
        if v["toxicity_pass"]:
            result["toxicity_pass"] += 1
    for name, floor in CACO2_TIERS:
        kept, caco2_only = [], 0
        for smiles, label in entries:
            v = admet_verdict(admet.get(smiles), floor, thresholds)
            if v["caco2_pass"]:
                caco2_only += 1
            if v["pass"]:
                kept.append((smiles, label))
        result["tiers"][name] = {"floor": floor, "caco2_pass": caco2_only,
                                 "survivors": len(kept), "entries": kept}
    return result


def amended_window(entries: list[tuple[str, str]], admet: dict[str, dict],
                   thresholds: dict[str, float], dili_max: float = DILI_MAX_DEFAULT,
                   caco2_floor: float = CACO2_TIERS[1][1]) -> dict:
    """The section 9.3.1 amendment: one Caco-2 floor and a DILI ceiling.

    SR-MMP and NR-AhR are recorded, not gated - they are the repair stage's
    target, and select_analogues re-applies them against the parent. The
    stratification by failure count is reported for the same reason
    optimize_leads.py reports it: a high-similarity analogue is far likelier to
    repair one endpoint than three, so the mix of parents predicts the yield.
    """
    kept, strata, bands = [], {0: 0, 1: 0, 2: 0, 3: 0}, {"already_passing": 0,
                                                         "conversion_band": 0}
    caco2_pass = dili_pass = 0
    for smiles, label in entries:
        prediction = admet.get(smiles)
        if not prediction:
            continue
        caco2 = _float(prediction.get(CACO2_ENDPOINT))
        dili = _float(prediction.get(DILI_ENDPOINT))
        if caco2 is None or dili is None:
            continue
        if caco2 > caco2_floor:
            caco2_pass += 1
        if dili < dili_max:
            dili_pass += 1
        if not (caco2 > caco2_floor and dili < dili_max):
            continue
        kept.append((smiles, label))
        failures = sum(
            1 for endpoint in TOXICITY_ENDPOINTS
            if (value := _float(prediction.get(endpoint))) is None
            or value >= thresholds[endpoint])
        strata[failures] = strata.get(failures, 0) + 1
        if dili < thresholds[DILI_ENDPOINT]:
            bands["already_passing"] += 1
        else:
            bands["conversion_band"] += 1
    return {"input": len(entries), "caco2_floor": caco2_floor,
            "dili_max": dili_max, "caco2_pass": caco2_pass,
            "dili_pass": dili_pass, "survivors": len(kept),
            "strata": strata, "dili_bands": bands, "entries": kept}


def read_survivors(path: str) -> list[tuple[str, str]]:
    """(smiles, label) rows of a survivors.smi, comments skipped."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", required=True,
                        help="§8.2 survivors.smi (the descriptor cuts' output)")
    parser.add_argument("--admet", required=True,
                        help="admet_predict output CSV for those SMILES")
    parser.add_argument("--out", required=True,
                        help="docking input to write (the adopted tier)")
    parser.add_argument("--gate-json", default=DEFAULT_GATE_JSON)
    parser.add_argument("--dili-max", type=float, default=None,
                        metavar="X",
                        help="apply the §9.3.1 AMENDED rule instead of the "
                             "pre-registered §9.2 conjunction: Caco-2 above "
                             "--caco2-floor and DILI below X, with SR-MMP and "
                             "NR-AhR recorded rather than gated. The "
                             "pre-registered counts are printed either way. "
                             f"The derived value is {DILI_MAX_DEFAULT} - see "
                             "CONVERSION_BANDS and v4_threshold_ledger.md")
    parser.add_argument("--caco2-floor", type=float, default=CACO2_TIERS[1][1],
                        help="Caco-2 floor for the amended rule "
                             f"(default {CACO2_TIERS[1][1]}, BCS moderate)")
    args = parser.parse_args(argv)

    thresholds = toxicity_thresholds(args.gate_json)
    entries = read_survivors(args.survivors)
    admet = read_admet(args.admet)
    result = apply_window(entries, admet, thresholds)

    print(f"{args.survivors} + {args.admet}")
    print(f"  input                {result['input']}")
    print(f"  ADMET prediction 없음  {result['admet_missing']}  (실패 처리)")
    print(f"  §9.2 독성 통과         {result['toxicity_pass']}")
    for endpoint, value in thresholds.items():
        print(f"      {endpoint:8} < {value:.6f}")
    print()
    for name, floor in CACO2_TIERS:
        t = result["tiers"][name]
        print(f"  {name}  (Caco2 > {floor})")
        print(f"      Caco2 통과          {t['caco2_pass']}")
        print(f"      ∩ 독성 통과         {t['survivors']}")

    if args.dili_max is not None:
        amended = amended_window(entries, admet, thresholds, args.dili_max,
                                 args.caco2_floor)
        print()
        print(f"  §9.3.1 AMENDED  (Caco2 > {amended['caco2_floor']}, "
              f"DILI < {amended['dili_max']})")
        print(f"      Caco2 통과          {amended['caco2_pass']}")
        print(f"      DILI 통과           {amended['dili_pass']}")
        print(f"      ∩ 채택             {amended['survivors']}")
        print(f"      그중 DILI 이미 통과 (< {thresholds[DILI_ENDPOINT]:.6f}): "
              f"{amended['dili_bands']['already_passing']}  -> 직행 리드 후보")
        print(f"      그중 전환 구간      "
              f"{amended['dili_bands']['conversion_band']}  -> 최적화 부모 "
              f"(v3 실측 전환율 {CONVERSION_BANDS[0][3]:.0%})")
        print("      §9.2 실패 엔드포인트 수별 (최적화 전망):")
        for count in sorted(amended["strata"]):
            print(f"          {count}개 실패  {amended['strata'][count]}")
        if not amended["entries"]:
            print("\n수정안으로도 0이다. 출력을 쓰지 않는다.", file=sys.stderr)
            return 1
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        lines = [
            "# §5.5 ADMET window survivors - the docking input.",
            f"# from: {args.survivors} + {args.admet}",
            f"# RULE: §9.3.1 amended - Caco2_Wang > {amended['caco2_floor']} "
            f"and DILI < {amended['dili_max']}",
            "#   SR-MMP and NR-AhR are recorded, not gated: they are the repair",
            "#   stage's target and select_analogues re-applies them against the",
            "#   parent (optimize_leads.py). See results/v4_threshold_ledger.md.",
            f"# PRE-REGISTERED, reported so the amendment does not hide its cost:",
            f"#   tier1 (Caco2 > {CACO2_TIERS[0][1]}) ∩ §9.2 conjunction = "
            f"{result['tiers'][CACO2_TIERS[0][0]]['survivors']}",
            f"#   tier2 (Caco2 > {CACO2_TIERS[1][1]}) ∩ §9.2 conjunction = "
            f"{result['tiers'][CACO2_TIERS[1][0]]['survivors']}",
            f"# cumulative: input {amended['input']} -> Caco2 "
            f"{amended['caco2_pass']} -> ∩ DILI {amended['survivors']}",
            f"#   of which DILI already below {thresholds[DILI_ENDPOINT]:.6f}: "
            f"{amended['dili_bands']['already_passing']}; in the conversion band: "
            f"{amended['dili_bands']['conversion_band']}",
        ]
        lines += [f"{smiles}\t{label}" for smiles, label in amended["entries"]]
        with open(args.out, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"\n  adopted amended: {amended['survivors']} -> {args.out}")
        return 0

    # The adopted tier is Tier 1 unless it is empty - the rule was fixed in the
    # design note before these counts existed.
    adopted = None
    for name, _floor in CACO2_TIERS:
        if result["tiers"][name]["survivors"] > 0:
            adopted = name
            break
    if adopted is None:
        print("\n모든 tier가 0이다. 이것이 결과다 — 이 chemotype과 이 목적함수로는"
              "\n경구 적합 리드가 없고, 결론은 보상함수가 이 항들을 들어야 한다는 것이다"
              "\n(§5.5). 도킹을 돌릴 집합이 없으므로 출력을 쓰지 않는다.", file=sys.stderr)
        return 1
    if adopted != CACO2_TIERS[0][0]:
        print(f"\n경고: Tier 1(BCS 고투과)이 0이므로 사전등록된 대안 {adopted}로"
              f"\n내려갔다. 최종 보고에 한계로 명시할 것 (§5.5).", file=sys.stderr)

    t = result["tiers"][adopted]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines = [
        "# §5.5 ADMET window survivors - the docking input.",
        f"# from: {args.survivors} + {args.admet}",
        f"# adopted tier: {adopted} (Caco2_Wang > {t['floor']})",
        f"# toxicity: §9.2, all of {', '.join(TOXICITY_ENDPOINTS)} below "
        f"{args.gate_json} p60",
        f"# cumulative: input {result['input']} -> Caco2 {t['caco2_pass']}"
        f" -> ∩ toxicity {t['survivors']}",
    ]
    lines += [f"{smiles}\t{label}" for smiles, label in t["entries"]]
    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n  adopted {adopted}: {t['survivors']} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
