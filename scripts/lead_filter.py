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

FOUR STAGES, IN THE PRE-REGISTERED ORDER (§9.1), all relative to PLN-1474:

  1. geometry gate (§8.5b)
  2. Caco-2 better than PLN-1474 by MORE THAN 0.5 log
  3. affinity below PLN-1474's best PASSING pose
  4. toxicity composite below PLN-1474's

Every stage is a conjunction, so the order does not change the final set --
but it changes the counts the report shows, and the pre-registered order is
the one to report.

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
import sys
from pathlib import Path

__all__ = ["CACO2_ENDPOINT", "CACO2_MARGIN", "CONTEXT_ENDPOINTS", "REFERENCE",
           "TOXICITY_ENDPOINTS", "lead_rows", "summarize", "toxicity_composite"]

# §9.2: the three endpoints ADMET-AI predicts best, equally weighted.
TOXICITY_ENDPOINTS = ("SR-MMP", "NR-AhR", "DILI")

CACO2_ENDPOINT = "Caco2_Wang"
# §9.1 filter 2: "PLN-1474보다 > 0.5 log 우위". Higher Caco-2 = more permeable.
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
              reference: str = REFERENCE) -> list:
    """One row per geometry-passing ligand with all four §9.1 flags."""
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
    if ref_caco2 is None or ref_tox is None:
        raise ValueError(
            f"{admet_csv}: {reference} is missing {CACO2_ENDPOINT} or one of "
            f"{TOXICITY_ENDPOINTS}; the thresholds cannot be derived.")

    rows = []
    for entry in _read(funnel_csv):
        if str(entry.get("geometry_pass")) != "True":
            continue
        label = entry["label"]
        prediction = admet.get(label)
        caco2 = _float(prediction.get(CACO2_ENDPOINT)) if prediction else None
        tox = toxicity_composite(prediction) if prediction else None
        affinity = _float(entry.get("best_passing_affinity"))

        row = {
            "label": label,
            "geometry_pass": True,
            "caco2": caco2,
            "caco2_pass": caco2 is not None and caco2 > ref_caco2 + CACO2_MARGIN,
            "best_passing_affinity": affinity,
            "affinity_pass": affinity is not None and affinity < affinity_cutoff,
            "toxicity": tox,
            "toxicity_pass": tox is not None and tox < ref_tox,
            "admet_status": "ok" if prediction else "missing",
            "plip_metal_any_passing": entry.get("plip_metal_any_passing") == "True",
            "plip_hbond_any_passing": entry.get("plip_hbond_any_passing") == "True",
            "n_poses": entry.get("n_poses"),
        }
        for endpoint in CONTEXT_ENDPOINTS:
            row[endpoint] = _float(prediction.get(endpoint)) if prediction else None
        row["all_four"] = (row["caco2_pass"] and row["affinity_pass"]
                           and row["toxicity_pass"])
        rows.append(row)

    rows.sort(key=lambda r: (not r["all_four"],
                             r["best_passing_affinity"]
                             if r["best_passing_affinity"] is not None else 0.0))
    return rows


def summarize(rows: list) -> dict:
    """The four-stage funnel, cumulative, in §9.1's order."""
    stage1 = [r for r in rows if r["geometry_pass"]]
    stage2 = [r for r in stage1 if r["caco2_pass"]]
    stage3 = [r for r in stage2 if r["affinity_pass"]]
    stage4 = [r for r in stage3 if r["toxicity_pass"]]
    return {
        "stage_1_geometry": len(stage1),
        "stage_2_caco2": len(stage2),
        "stage_3_affinity": len(stage3),
        "stage_4_toxicity": len(stage4),
        "leads": [r["label"] for r in stage4],
        "admet_missing": [r["label"] for r in stage1
                          if r.get("admet_status") == "missing"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--funnel", required=True, help="funnel.py --out-csv")
    p.add_argument("--admet", required=True, help="admet_predict output CSV")
    p.add_argument("--affinity-cutoff", type=float, required=True,
                    help="PLN-1474's best PASSING pose in the adopted engine.")
    p.add_argument("--reference", default=REFERENCE)
    p.add_argument("--label", default="")
    p.add_argument("--out-csv", type=Path)
    p.add_argument("--out-json", type=Path)
    args = p.parse_args(argv)

    rows = lead_rows(args.funnel, args.admet, args.affinity_cutoff,
                     args.reference)
    summary = summarize(rows)

    print(f"\n=== {args.label or args.funnel} ===")
    print(f"  1 기하 게이트          {summary['stage_1_geometry']}")
    print(f"  2 + Caco-2 > 참조+{CACO2_MARGIN}   {summary['stage_2_caco2']}")
    print(f"  3 + affinity < {args.affinity_cutoff:.3f}  {summary['stage_3_affinity']}")
    print(f"  4 + 독성 < 참조         {summary['stage_4_toxicity']}")
    if summary["admet_missing"]:
        print(f"  ADMET 예측 없음: {len(summary['admet_missing'])} "
              f"(2·4단계 실패로 기록, 분모에서 빼지 않음)")

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
