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
           "apply_window", "DEFAULT_GATE_JSON"]

# log10(Papp in cm/s). BCS/FDA: high > -5.0 (Papp > 10e-6, Fa 80-100%);
# moderate -5.7..-5.0 (2-10e-6, Fa 50-80%); low < -5.7 (Fa < 50%).
CACO2_TIERS = (("tier1_bcs_high", -5.0), ("tier2_bcs_moderate", -5.7))

DEFAULT_GATE_JSON = "results/v3_metal_asn224/approved_drug_gate.json"


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
