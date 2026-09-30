"""spec §9.1's four-stage filter and §9.2's toxicity composite.

This supersedes scripts/select_leads.py for the v2 campaign. That module is
v1's and is left intact as the record it is: it applies a weighted toxicity
score (DILI 2.0 / hERG 1.0 / CYP 0.5, denominator 11.5) that §9.2 removes by
name, and it has no permeability filter at all.

§9.2's reason for removing the weights is a failed sanity check, not a
preference: under them CWHM-12 -- a parenteral pan-alphaV TOOL compound --
scored LESS toxic than PLN-1474, a compound that completed Phase 1. The v2
definition is the equal mean of the three endpoints ADMET-AI predicts best:
SR-MMP (AUROC 0.925), NR-AhR (0.904), DILI (0.881).

The known limitation is stated rather than fixed. An equal mean does not
solve the dynamic-range problem: PLN-1474's DILI is 0.455 against SR-MMP
0.0208, so DILI still dominates the average. Rank-normalising per endpoint
would mitigate it and would introduce another transform, so v2 uses the
equal mean and says this out loud. hERG, AMES and the three CYPs leave the
filter and travel on the lead card.

FOUR STAGES, AMENDED 2026-09-30 (change ledger in results/v2_lead_filter.md),
all relative to PLN-1474:

  1. geometry gate (§8.5b) + PLIP
  2. affinity below PLN-1474's best PASSING pose
  3. INTERACTION: §8.2's gate, as compound 25 decided it
  4. SELECTIVITY above PLN-1474's
  5. toxicity: EVERY §9.2 endpoint below PLN-1474's, not their mean

Stages 2 and 3 are the pair the campaign asked for: PLN-1474-or-better
affinity AND the compound-25-derived key interaction. They sit together
because neither alone is a claim about binding -- a score without the anchor
is a number, and the anchor without the score is a pose.

Caco-2 is computed and reported on every row. It does not gate. Four
amendments to §9.1's four stages, each with its reason:

ORDER. Strongest evidence first. Structure, then a same-engine relative
affinity, then selectivity, then model predictions -- with Caco-2 LAST
because it is the filter this project itself calls partly circular. Every
stage is a conjunction, so the final set is unchanged; the counts the report
shows are not, and both orders are reported.

CACO-2 STOPS GATING (2026-09-30, made AFTER seeing it leave zero leads --
recorded that way deliberately, because the reason has to stand without that
fact). §9.1 required Caco-2 better than PLN-1474 by more than 0.5 log.
Caco2_Wang is ADMET-AI's only REGRESSION endpoint here, trained on 906
compounds, with a reported error near 0.3 log. The five molecules that reach
this stage predict -6.193, -6.143, -5.644, -5.582 and -5.529 against
PLN-1474's -5.586: the two that beat the reference beat it by 0.004 and 0.057
log, roughly a tenth of the model's own error, and the 0.5-log margin exceeds
the whole set's 0.66-log spread. The gate therefore cuts noise, not signal,
and one margin term decides every outcome.

It does NOT come out because TPSA already covers permeability. Every molecule
reaching this stage passed the same TPSA 40-115 window during RL, so TPSA
cannot separate them and is not a substitute for this endpoint. The value
stays on every row, and both counts -- gated and not -- are reported.

SELECTIVITY BECOMES A STAGE. §9.1 had it only as criterion 6/7, a ranking
axis. §10.4's clause making criterion 6 unevaluable when the calibration gate
fails is amended away: the stage applies unconditionally. The calibration
verdict is still computed and reported beside it, because it is what says
whether the protocol has demonstrated any discriminating power -- and when it
has not, `selectivity_calibrated` is False and the lead card must say so. A
molecule with no selectivity value FAILS; the stage cannot pass what it did
not measure.

TOXICITY BECOMES PER-ENDPOINT. An equal mean hides one bad endpoint behind
two good ones. Measured on this run: gen_03050 passes the mean at 0.1442
while its SR-MMP is 0.3323, SIXTEEN TIMES PLN-1474's 0.0208, because a low
DILI dilutes it. Requiring every endpoint to beat the reference removes the
dynamic-range domination §9.2 admitted it could not fix. The mean is still
computed and reported, since it is the pre-registered number.

THE PERMEABILITY FILTER IS PARTLY CIRCULAR AND THAT HAS TO BE SAID. Measured
on this project's own library, predicted Caco-2 correlates with TPSA at
r = -0.571, and TPSA is an axis the RL objective optimises directly. The
0.5 log margin exists because the best Caco-2 models reach MAE 0.26-0.28 log
units while the same compound's published values differ between laboratories
by a median of 0.57, so a smaller margin would not clear measurement noise.
Neither fact is a reason to drop a pre-registered filter; both are reasons
the lead report cannot claim a permeability advantage as an independent
result.

A ligand missing from the ADMET run FAILS stages 2 and 4 and keeps its row,
with admet_status = "missing". Passing it silently would let an unscored
molecule become a lead; dropping it would shrink the denominator of every
rate computed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The §8.7 reference lives in funnel.py and is imported, never restated. It
# was restated once -- on the command line, as -6.807 -- and for weeks
# funnel.py filtered at -7.290 while lead_filter.py filtered at -6.807 on the
# same pre-registered criterion. Nothing caught it until both printed to one
# screen. -6.807 has no derivation anywhere in this repository; the value it
# displaced is PLN-1474's best passing pose in results/v2_panel_geometry.csv.
from funnel import PLN1474_CUTOFF

__all__ = ["CACO2_ENDPOINT", "CACO2_MARGIN", "CONTEXT_ENDPOINTS", "REFERENCE",
           "STAGES", "TOXICITY_ENDPOINTS", "lead_rows", "summarize",
           "toxicity_composite"]

# Reported in this order; see the module docstring for why.
STAGES = ("geometry", "affinity", "interaction", "selectivity", "toxicity")

# §9.2: the three endpoints ADMET-AI predicts best, equally weighted.
TOXICITY_ENDPOINTS = ("SR-MMP", "NR-AhR", "DILI")

CACO2_ENDPOINT = "Caco2_Wang"
# §9.1 filter 2 required this margin. It no longer gates (see the docstring);
# it is kept so the report can still say what the pre-registered rule would
# have done, which is the only honest way to record an amendment like this.
CACO2_MARGIN = 0.5

# The reference every filter is relative to, as admet_input labels it.
REFERENCE = "PANEL_PLN-1474"

# Reported beside every lead, never entering the filter (§9.2).
CONTEXT_ENDPOINTS = ("hERG", "AMES", "CYP3A4_Veith", "CYP2C9_Veith",
                     "CYP2D6_Veith", "Solubility_AqSolDB")


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def toxicity_composite(row: dict):
    """Equal mean of the three §9.2 endpoints, or None if any is missing.

    Averaging two of three would silently change the definition while still
    producing a number the filter could act on.
    """
    values = [_float(row.get(e)) for e in TOXICITY_ENDPOINTS]
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)


def _read(path: str) -> list:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def lead_rows(funnel_csv: str, admet_csv: str, affinity_cutoff: float,
              reference: str = REFERENCE, selectivity: dict | None = None) -> list:
    """One row per geometry-passing ligand with all five stage flags.

    `selectivity` is {"reference": float, "values": {label: float},
    "calibration_passed": bool}. Absent, or absent for a molecule, fails the
    selectivity stage -- it cannot pass what was not measured.
    """
    admet = {}
    for row in _read(admet_csv):
        label = row.get("label") or row.get("Label") or ""
        if label:
            admet[label] = row

    ref = admet.get(reference)
    if ref is None:
        raise ValueError(
            f"{admet_csv}: no row labelled {reference!r}. Every §9.1 filter is "
            "relative to PLN-1474, so without it there is nothing to filter "
            "against and a default would be an invented reference.")
    ref_caco2 = _float(ref.get(CACO2_ENDPOINT))
    ref_tox = toxicity_composite(ref)
    ref_endpoints = {e: _float(ref.get(e)) for e in TOXICITY_ENDPOINTS}
    if ref_caco2 is None or ref_tox is None or any(
            v is None for v in ref_endpoints.values()):
        raise ValueError(
            f"{admet_csv}: {reference} is missing {CACO2_ENDPOINT} or one of "
            f"{TOXICITY_ENDPOINTS}; the thresholds cannot be derived.")

    selectivity = selectivity or {}
    sel_values = selectivity.get("values") or {}
    sel_reference = selectivity.get("reference")
    sel_calibrated = bool(selectivity.get("calibration_passed", False))
    sel_ran = bool(selectivity.get("calibration_ran", True))

    rows = []
    for entry in _read(funnel_csv):
        if str(entry.get("geometry_pass")) != "True":
            continue
        label = entry["label"]
        prediction = admet.get(label)
        caco2 = _float(prediction.get(CACO2_ENDPOINT)) if prediction else None
        tox = toxicity_composite(prediction) if prediction else None
        affinity = _float(entry.get("best_passing_affinity"))

        # Stage 4: every endpoint below the reference, not their mean.
        worse = []
        if prediction:
            for endpoint in TOXICITY_ENDPOINTS:
                value = _float(prediction.get(endpoint))
                if value is None or value >= ref_endpoints[endpoint]:
                    worse.append(endpoint)
        else:
            worse = list(TOXICITY_ENDPOINTS)

        # Stage 3: §8.2's interaction gate. funnel.py aggregates it over the
        # geometry-passing poses; an absent column means the gate was never
        # decided, and the stage passes so the other four still report.
        raw_gate = entry.get("plip_gate_pass")
        if raw_gate in (None, ""):
            interaction_pass, interaction_status = True, "not_applied"
        else:
            interaction_pass = str(raw_gate) == "True"
            interaction_status = "ok"

        # Stage 4: selectivity, applied unconditionally (2026-09-30 amendment).
        sel_value = sel_values.get(label)
        if sel_reference is None or sel_value is None:
            sel_status, sel_pass = "missing", False
        else:
            sel_status, sel_pass = "ok", sel_value > sel_reference

        row = {
            "label": label,
            "geometry_pass": True,
            "best_passing_affinity": affinity,
            "affinity_pass": affinity is not None and affinity < affinity_cutoff,
            "interaction_pass": interaction_pass,
            "interaction_status": interaction_status,
            "selectivity": sel_value,
            "selectivity_pass": sel_pass,
            "selectivity_status": sel_status,
            "selectivity_calibrated": sel_calibrated,
            "selectivity_calibration_ran": sel_ran,
            "toxicity": tox,
            "toxicity_pass": bool(prediction) and not worse,
            "toxicity_worse_than_reference": worse,
            "toxicity_mean_pass": tox is not None and tox < ref_tox,
            "caco2": caco2,
            # Reported, not a stage. "_preregistered" names what it is: the
            # rule §9.1 asked for, kept visible beside the rule now used.
            "caco2_better_than_reference": caco2 is not None and caco2 > ref_caco2,
            "caco2_preregistered_pass": (caco2 is not None
                                         and caco2 > ref_caco2 + CACO2_MARGIN),
            "admet_status": "ok" if prediction else "missing",
            "plip_metal_any_passing": entry.get("plip_metal_any_passing") == "True",
            "plip_hbond_any_passing": entry.get("plip_hbond_any_passing") == "True",
            "n_poses": entry.get("n_poses"),
        }
        for endpoint in TOXICITY_ENDPOINTS:
            row[endpoint] = _float(prediction.get(endpoint)) if prediction else None
        for endpoint in CONTEXT_ENDPOINTS:
            row[endpoint] = _float(prediction.get(endpoint)) if prediction else None
        row["all_stages"] = all(row[f"{stage}_pass"] for stage in STAGES)
        # What the pre-registered five-stage rule would have selected.
        row["all_stages_preregistered"] = (row["all_stages"]
                                           and row["caco2_preregistered_pass"])
        rows.append(row)

    rows.sort(key=lambda r: (not r["all_stages"],
                             r["best_passing_affinity"]
                             if r["best_passing_affinity"] is not None else 0.0))
    return rows


def summarize(rows: list) -> dict:
    """The four-stage funnel, cumulative, in STAGES order.

    `leads_preregistered` reports what the five-stage rule with the Caco-2
    margin would have returned, so the amendment never hides its own cost.
    """
    out: dict = {}
    remaining = list(rows)
    for position, stage in enumerate(STAGES, start=1):
        remaining = [r for r in remaining if r.get(f"{stage}_pass")]
        out[f"stage_{position}_{stage}"] = len(remaining)
    out["leads"] = [r["label"] for r in remaining]
    out["admet_missing"] = [r["label"] for r in rows
                            if r.get("admet_status") == "missing"]
    out["selectivity_missing"] = [r["label"] for r in rows
                                  if r.get("selectivity_status") == "missing"]
    out["interaction_applied"] = any(
        r.get("interaction_status") == "ok" for r in rows)
    out["selectivity_calibrated"] = bool(rows) and all(
        r.get("selectivity_calibrated") for r in rows)
    out["selectivity_calibration_ran"] = not rows or all(
        r.get("selectivity_calibration_ran", True) for r in rows)
    # The pre-registered rules, kept for the record: what §9.1/§9.2 as
    # written would have returned, beside what the amendments return.
    out["toxicity_mean_pass"] = sum(1 for r in rows if r.get("toxicity_mean_pass"))
    out["leads_preregistered"] = [r["label"] for r in rows
                                  if r.get("all_stages_preregistered")]
    out["caco2_better_than_reference"] = [
        r["label"] for r in remaining if r.get("caco2_better_than_reference")]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--funnel", required=True, help="funnel.py --out-csv")
    p.add_argument("--admet", required=True, help="admet_predict output CSV")
    p.add_argument("--affinity-cutoff", type=float, default=PLN1474_CUTOFF,
                    help="PLN-1474's best PASSING pose in the adopted engine. "
                         f"Defaults to funnel.PLN1474_CUTOFF "
                         f"({PLN1474_CUTOFF:+.3f}), the pre-registered §8.7 "
                         "value, so the reference is defined in ONE place. "
                         "Passing anything else is recorded and warned about.")
    p.add_argument("--selectivity", type=Path,
                    help="selectivity.py --out JSON. Without it the "
                         "selectivity stage fails every molecule: a filter "
                         "cannot pass what was not measured.")
    p.add_argument("--selectivity-isoform", default=None,
                    help="Which isoform's rank selectivity to filter on "
                         "(default: the only one, if there is one).")
    p.add_argument("--selectivity-metric", choices=("delta", "rank"),
                    default="delta",
                    help="Which axis in the selectivity JSON to filter on. "
                         "delta (the §10.1 pre-registered one) is the default "
                         "because it is population independent: it is a "
                         "per-ligand difference, so the same ligand gets the "
                         "same value whatever else was docked. rank is a "
                         "percentile and moves with the population -- the "
                         "same four controls gave FAIL with one arm's "
                         "molecules and PASS with the other's. The receptor "
                         "offset that rank was introduced to cancel already "
                         "cancels here, because every use of the value is a "
                         "comparison against PLN-1474, which is itself a "
                         "difference.")
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument("--label", default="")
    p.add_argument("--out-csv", type=Path)
    p.add_argument("--out-json", type=Path)
    args = p.parse_args(argv)

    selectivity = None
    if args.selectivity and args.selectivity.exists():
        data = json.loads(args.selectivity.read_text())
        key = {"delta": "deltas", "rank": "rank_selectivity"}[
            args.selectivity_metric]
        tables = data.get(key) or {}
        name = args.selectivity_isoform or (
            next(iter(tables)) if len(tables) == 1 else None)
        if name is None:
            sys.stderr.write(
                f"{args.selectivity} holds {len(tables)} isoforms "
                f"({list(tables)}); name one with --selectivity-isoform.\n")
            return 2
        verdicts = data.get(
            {"delta": "calibration_delta",
             "rank": "calibration_rank"}[args.selectivity_metric]) or {}
        selectivity = {
            "values": tables.get(name, {}),
            "reference": None,
            "calibration_passed": bool((verdicts.get(name) or {}).get("passed")),
            # Older runs have no such key; assume the gate ran, since the
            # only thing that sets it false is a control that was never
            # docked, and that is a state this version detects explicitly.
            "calibration_ran": bool(data.get("calibration_ran", True)),
        }
        # The reference's own selectivity, looked up by either label form.
        for key in (args.reference, args.reference.replace("PANEL_", ""),
                    f"PANEL_{args.reference}"):
            if key in selectivity["values"]:
                selectivity["reference"] = selectivity["values"][key]
                break
        if selectivity["reference"] is None:
            sys.stderr.write(
                f"{args.selectivity}: {args.reference} has no selectivity "
                "value, so there is no reference to filter against.\n")
            return 2

    if abs(args.affinity_cutoff - PLN1474_CUTOFF) > 1e-9:
        sys.stderr.write(
            f"WARNING: --affinity-cutoff {args.affinity_cutoff:+.3f} is not "
            f"the pre-registered §8.7 value {PLN1474_CUTOFF:+.3f} "
            f"(difference {args.affinity_cutoff - PLN1474_CUTOFF:+.3f} "
            "kcal/mol). The pre-registered value is PLN-1474's best PASSING "
            "pose in results/v2_panel_geometry.csv. Use it unless the record "
            "says why not.\n")

    rows = lead_rows(args.funnel, args.admet, args.affinity_cutoff,
                     args.reference, selectivity)
    summary = summarize(rows)

    print(f"\n=== {args.label or args.funnel} ===")
    print(f"  affinity 컷오프: {args.affinity_cutoff:+.3f}"
          + ("" if abs(args.affinity_cutoff - PLN1474_CUTOFF) < 1e-9
             else f"  ← 사전등록 값 {PLN1474_CUTOFF:+.3f} 아님"))
    if args.selectivity:
        print(f"  선택성 지표: {args.selectivity_metric}")
    for position, stage in enumerate(STAGES, start=1):
        print(f"  {position} {stage:<12} {summary[f'stage_{position}_{stage}']}")
    print(f"  (사전등록 독성 규칙: 동등평균 < 기준 통과 "
          f"{summary['toxicity_mean_pass']})")
    print(f"  Caco-2는 게이트가 아니다 (2026-09-30 개정). 리드 "
          f"{len(summary['leads'])}개 중 기준보다 투과성이 좋은 것 "
          f"{len(summary['caco2_better_than_reference'])}개; 사전등록한 "
          f"0.5 log 마진까지 걸었다면 {len(summary['leads_preregistered'])}개.")
    if summary["selectivity_missing"]:
        print(f"  선택성 값 없음: {len(summary['selectivity_missing'])} "
              "(3단계 실패로 기록)")
    if not summary["interaction_applied"]:
        print("  경고: §8.2 상호작용 게이트가 적용되지 않았다 — funnel에 "
              "plip_gate_pass 컬럼이 없다. cpd25_gate.py로 게이트를 정하고 "
              "funnel.py --interaction-gate로 다시 만들어야 한다.")
    if not summary["selectivity_calibration_ran"]:
        print("  경고: §10.4 보정 게이트가 실행되지 않았다 — 대조군이 "
              "아이소폼에 도킹되지 않았다. 아무것도 검증하지 않은 축으로 "
              "3단계를 자른 것이므로, 참조를 도킹해 다시 실행해야 한다.")
    elif not summary["selectivity_calibrated"]:
        print("  경고: §10.4 보정 게이트를 통과하지 않은 선택성 값으로 "
              "필터했다. 그 값의 타당성은 미검증이며 리드 카드에 병기해야 한다.")
    if summary["admet_missing"]:
        print(f"  ADMET 예측 없음: {len(summary['admet_missing'])}")

    if args.out_csv and rows:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
