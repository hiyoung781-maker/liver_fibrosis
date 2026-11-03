"""Section 8.7's final selection: affinity hard filter, then a weighted toxicity rank.

Two stages, in this order, because they answer different questions.

STAGE 1 - HARD FILTER on binding affinity, against PLN-1474 docked with the identical
protocol (scripts/dock_panel.py). PLN-1474 is the reference because it is the only
alphaVbeta1 inhibitor to have entered the clinic, and section 7.5 already made it the
benchmark in place of cpd 25, which was never proposed as a lead. A hard filter is
defensible here in a way it is not for toxicity: affinity is what the docking measures
directly, and both sides come from the same engine, receptor and box.

STAGE 2 - HARD FILTER on predicted permeability, against the same compound. Section
9.2 (as revised by 7.5) pre-registers "beats PLN-1474 on predicted permeability" as a
success criterion, and section 9 was frozen before sampling - so a lead that misses it
is not a lead. The margin required is 0.5 log units, the figure section 7.5 established
as the resolvable floor: the best Caco-2 models reach MAE 0.26-0.28 while the same
compound's published values differ between laboratories by a median of 0.57 log, so a
smaller gap is not a difference.

WHY THE FILTER STOPS THERE. Extending it to the toxicity endpoints - "better than
PLN-1474 on every one" - was measured and rejected. It leaves ZERO molecules, and the
reason is instructive rather than disappointing: PLN-1474 already sits at 0.0093 on
NR-AhR, 0.0208 on SR-MMP, 0.0393 on SR-p53 and 0.0437 on CYP2C9. Individually only
2-7% of the permeability survivors beat each of those, and none beats all of them. More
to the point, deciding that 0.0089 is "better than" 0.0093 is a resolution an AUROC-0.90
classifier does not have. That is the same principle the 0.5 log floor enforces for
permeability, and an all-endpoint conjunction would violate it nine times over.

On the composite, PLN-1474 scores 0.144 against a candidate median of 0.286 - it sits
in the top 5% of our own molecules. Only 2 of the 216 permeability survivors beat it
outright, and those two are flagged in the output: they are the molecules that support
the strongest claim available (better affinity, better permeability past the noise
floor, and a better overall toxicity profile than the only alphaVbeta1 inhibitor to
reach the clinic).

STAGE 3 - RANK on toxicity. NOT a gate, and the reason is measured: this chemotype runs
high on DILI across the board, including the published compounds. CHEMBL4649232 - the
most potent non-RGD active in the curated set, pIC50 9.78 - scores 0.980, and A1AFA, the
8W30 crystal ligand, scores 0.825. Any absolute DILI cutoff rejects the published
actives before it rejects our molecules. Ranking is also the more robust use of a
miscalibrated model: comparing molecules to each other cancels systematic bias, while a
threshold inherits it whole.

ENDPOINT CHOICE follows ADMET-AI's own reported per-endpoint performance (bundled in the
package as resources/data/admet.csv), not dataset size - a proxy that misleads here,
since DILI has only 475 training compounds and still reaches AUROC 0.881.

Weights put the liver first, because the indication is liver fibrosis: the drug is for a
liver that is already damaged, so hepatotoxicity is not one liability among many.

  weight 2.0, hepatic    DILI      AUROC 0.881 / AUPRC 0.878
                         SR-MMP    AUROC 0.925 - mitochondrial membrane potential
                                   disruption, a principal mechanism of DILI, and the
                                   single best-performing endpoint in the panel
                         SR-p53    AUROC 0.880 - DNA-damage response
                         NR-AhR    AUROC 0.904 - xenobiotic response, hepatic
  weight 1.0, general    hERG      AUROC 0.839 / AUPRC 0.906
                         AMES      AUROC 0.882 / AUPRC 0.896
  weight 0.5, DDI        CYP3A4    AUROC 0.912   these are interaction risk rather than
                         CYP2C9    AUROC 0.908   direct injury, so they inform the rank
                         CYP2D6    AUROC 0.886   without dominating it

EXCLUDED, with the number that excludes them:
  ClinTox              AUPRC 0.607 - AUROC 0.928 is class-imbalance inflation; on a
                       rare-positive endpoint AUPRC is the honest metric
  Carcinogens_Lagunin  AUPRC 0.608
  LD50_Zhu             R^2 0.596
  Bioavailability_Ma   AUROC 0.716 - and not a toxicity endpoint
  Skin_Reaction        AUROC 0.718 - and an irrelevant route

Every endpoint is a probability of the unwanted outcome, so they share an orientation
and LOWER IS BETTER throughout. The score is their weighted mean.

The benchmark panel is scored on the same composite and printed beside the leads, so
"least toxic" has something to be least toxic THAN. And the final set is forced onto
distinct Murcko scaffolds - a lead list that is one series would not support the
novelty claim the project is built on.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__all__ = [
    "EXCLUDED_ENDPOINTS",
    "TOXICITY_WEIGHTS",
    "best_pose_per_ligand",
    "select_leads",
    "toxicity_score",
]

RDLogger.DisableLog("rdApp.*")

PANEL_PREFIX = "PANEL_"
REFERENCE = "PLN-1474"

# log units of Caco-2. Section 7.5's resolvable floor: model MAE 0.26-0.28 against a
# median inter-laboratory spread of 0.57 log on the same compound.
PERMEABILITY_FLOOR = 0.5
PERMEABILITY_ENDPOINT = "Caco2_Wang"

# Vina-family scores for drug-like ligands fall in roughly -4 to -13 kcal/mol. 218 of
# the 7,763 docked molecules (1.07% of poses) carry values outside that - down to
# -226.61 and up to +50.08 - from poses that clashed or sat 7-14 A off the site. None of
# them has a geometry-passing pose, so nothing downstream was affected; this guard makes
# that a property of the code rather than a fact about one run. An affinity-ranked
# selection would have put -226.61 first.
AFFINITY_RANGE = (-15.0, 0.0)

TOXICITY_WEIGHTS = {
    # hepatic - the indication is liver fibrosis
    "DILI": 2.0,
    "SR-MMP": 2.0,
    "SR-p53": 2.0,
    "NR-AhR": 2.0,
    # general liabilities
    "hERG": 1.0,
    "AMES": 1.0,
    # drug-drug interaction risk, not direct injury
    "CYP3A4_Veith": 0.5,
    "CYP2C9_Veith": 0.5,
    "CYP2D6_Veith": 0.5,
}

# Kept as data so the reason travels with the decision.
EXCLUDED_ENDPOINTS = {
    "ClinTox": "AUPRC 0.607 (AUROC 0.928 is class-imbalance inflation)",
    "Carcinogens_Lagunin": "AUPRC 0.608",
    "LD50_Zhu": "R^2 0.596",
    "Bioavailability_Ma": "AUROC 0.716, and not a toxicity endpoint",
    "Skin_Reaction": "AUROC 0.718, and an irrelevant route",
}

# Reported beside every lead even though they do not enter the score.
CONTEXT_ENDPOINTS = ["Caco2_Wang", "ClinTox", "Carcinogens_Lagunin", "LD50_Zhu",
                     "Solubility_AqSolDB"]


def toxicity_score(row: dict, weights: dict[str, float] = TOXICITY_WEIGHTS) -> float:
    """Weighted mean of the endpoint probabilities. Lower is better. nan if none parse."""
    total = 0.0
    weight_sum = 0.0
    for endpoint, weight in weights.items():
        value = row.get(endpoint)
        if value in (None, ""):
            continue
        try:
            total += weight * float(value)
        except ValueError:
            continue
        weight_sum += weight
    return total / weight_sum if weight_sum else float("nan")


def best_pose_per_ligand(geometry_csv: str) -> dict[str, dict]:
    """Best PASSING pose per ligand, by affinity. Poses that failed are ignored.

    Section 8.5b asks whether a binding mode exists; the affinity that represents a
    ligand is therefore the best among the poses that satisfy the geometry, not the
    best overall - a better-scoring pose that misses the anchors is not a binding mode
    this project accepts.
    """
    best: dict[str, dict] = {}
    with open(geometry_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("passes") != "1":
                continue
            try:
                affinity = float(row["affinity"])
            except (KeyError, TypeError, ValueError):
                continue
            low, high = AFFINITY_RANGE
            if not low <= affinity <= high:
                continue
            label = row["label"]
            if label not in best or affinity < float(best[label]["affinity"]):
                best[label] = row
    return best


def _panel_reference(panel_geometry_csv: str) -> float:
    best = best_pose_per_ligand(panel_geometry_csv)
    label = f"{PANEL_PREFIX}{REFERENCE}"
    if label not in best:
        raise ValueError(
            f"{panel_geometry_csv}: no passing, scored pose for {label}. The hard "
            "filter has no reference - run scripts/dock_panel.py first."
        )
    return float(best[label]["affinity"])


def _selectivity(selectivity_csv: str) -> dict[str, dict]:
    """Per-ligand OR over its poses for each section 8.4 observation."""
    out: dict[str, dict] = {}
    with open(selectivity_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            entry = out.setdefault(row["label"],
                                   {"leu225": 0, "tyr178": 0, "asp218_basic": 0})
            entry["leu225"] |= int(row.get("leu225_contact") or 0)
            entry["tyr178"] |= int(row.get("tyr178_stack") or 0)
            entry["asp218_basic"] |= int(row.get("asp218_basic_contact") or 0)
    return out


def select_leads(geometry_csv: str, panel_geometry_csv: str, admet_csv: str,
                 survivors_smi: str, selectivity_csv: str | None = None,
                 n_leads: int = 20,
                 permeability_floor: float = PERMEABILITY_FLOOR) -> dict:
    """Affinity filter, permeability filter, toxicity rank, distinct scaffolds."""
    from known_scaffolds import murcko
    from novelty import load_smi

    reference = _panel_reference(panel_geometry_csv)
    poses = best_pose_per_ligand(geometry_csv)
    with open(admet_csv, newline="") as handle:
        admet = {r["label"]: r for r in csv.DictReader(handle)}
    smiles = {label: s for s, label in load_smi(survivors_smi)}
    selectivity = _selectivity(selectivity_csv) if selectivity_csv else {}

    panel_admet = admet.get(f"{PANEL_PREFIX}{REFERENCE}")
    if panel_admet is None:
        raise ValueError(f"{admet_csv}: no {PANEL_PREFIX}{REFERENCE} row - the "
                         "permeability and composite references are undefined")
    reference_caco = float(panel_admet[PERMEABILITY_ENDPOINT])
    reference_tox = toxicity_score(panel_admet)

    counts = {"affinity": 0, "permeability": 0}
    candidates = []
    for label, pose in poses.items():
        if label.startswith(PANEL_PREFIX) or label not in smiles:
            continue
        if label not in admet:
            continue
        affinity = float(pose["affinity"])
        if affinity >= reference:            # stage 1: strictly better than PLN-1474
            continue
        counts["affinity"] += 1
        try:
            caco = float(admet[label][PERMEABILITY_ENDPOINT])
        except (KeyError, TypeError, ValueError):
            continue
        # stage 2: better by more than the resolvable floor, not merely better
        if caco <= reference_caco + permeability_floor:
            continue
        counts["permeability"] += 1
        score = toxicity_score(admet[label])
        if score != score:                   # nan
            continue
        observations = selectivity.get(label, {})
        candidates.append({
            "label": label, "smiles": smiles[label], "affinity": affinity,
            "toxicity_score": score,
            "scaffold": murcko(Chem.MolFromSmiles(smiles[label])),
            "ca_dist": pose.get("ca_dist"), "donor_dist": pose.get("donor_dist"),
            "caco2": caco,
            "beats_reference_toxicity": int(score < reference_tox),
            "leu225": observations.get("leu225", 0),
            "tyr178": observations.get("tyr178", 0),
            "asp218_basic": observations.get("asp218_basic", 0),
            "admet": admet[label],
        })

    candidates.sort(key=lambda c: c["toxicity_score"])

    leads, used = [], set()
    for candidate in candidates:
        if candidate["scaffold"] in used:
            continue
        used.add(candidate["scaffold"])
        leads.append(candidate)
        if len(leads) >= n_leads:
            break

    panel = []
    for label, row in admet.items():
        if not label.startswith(PANEL_PREFIX):
            continue
        panel.append({"label": label[len(PANEL_PREFIX):],
                      "toxicity_score": toxicity_score(row), "admet": row})
    panel.sort(key=lambda p: p["toxicity_score"])

    return {
        "reference_affinity": reference,
        "reference_caco2": reference_caco,
        "reference_toxicity": reference_tox,
        "permeability_floor": permeability_floor,
        "passing": len(poses),
        "after_affinity": counts["affinity"],
        "after_permeability": counts["permeability"],
        "after_hard_filter": len(candidates),
        "distinct_scaffolds": len({c["scaffold"] for c in candidates}),
        "beating_reference_toxicity": sum(c["beats_reference_toxicity"]
                                          for c in candidates),
        "leads": leads,
        "panel": panel,
    }


def _report(result: dict, n_leads: int) -> str:
    out = [
        "Section 8.7 final selection",
        "=" * 104,
        "  All three references are PLN-1474, the only alphaVbeta1 inhibitor to reach",
        "  the clinic (section 7.5). Affinity comes from scripts/dock_panel.py with the",
        "  identical protocol; permeability and toxicity from the same ADMET run.",
        "",
        f"  reference affinity   {result['reference_affinity']:.3f} kcal/mol"
        "   (best PASSING pose - a pose that misses the anchors is not a binding mode)",
        f"  reference Caco-2     {result['reference_caco2']:.3f} log cm/s",
        f"  reference toxicity   {result['reference_toxicity']:.3f}  (composite)",
        "",
        f"  geometry survivors   {result['passing']}",
        f"  after affinity       {result['after_affinity']}"
        f"   (strictly better than PLN-1474)",
        f"  after permeability   {result['after_permeability']}"
        f"   (better by more than {result['permeability_floor']:.1f} log - section 9.2)",
        f"  distinct scaffolds   {result['distinct_scaffolds']}",
        f"  beating PLN-1474 on the toxicity composite too: "
        f"{result['beating_reference_toxicity']}",
        f"  leads selected       {len(result['leads'])} of {n_leads} requested,"
        " one per Murcko scaffold",
        "",
        "Affinity and permeability are HARD FILTERS: the first is what docking measures",
        "directly, both sides from one engine and box; the second is pre-registered in",
        "section 9.2, with a 0.5 log margin because model error (MAE 0.26-0.28) is the",
        "size of inter-laboratory spread on the same compound (median 0.57 log).",
        "",
        "Extending the filter to every toxicity endpoint was measured and rejected: it",
        "leaves ZERO molecules. PLN-1474 already scores 0.0093 on NR-AhR and 0.0208 on",
        "SR-MMP, and calling 0.0089 'better' than 0.0093 is a resolution an AUROC-0.90",
        "classifier does not have - the same objection the 0.5 log floor encodes.",
        "",
        "Toxicity is therefore a RANK. Weights: hepatic endpoints (DILI, SR-MMP,",
        "SR-p53, NR-AhR) x2 because the indication is liver fibrosis; hERG and AMES x1;",
        "CYP inhibition x0.5 as interaction risk. Lower is better. ClinTox,",
        "carcinogenicity, LD50, bioavailability and skin reaction are excluded on their",
        "own reported performance - see the module docstring.",
        "",
        "=" * 104,
        "BENCHMARK PANEL on the same composite - what 'least toxic' is measured against",
        "=" * 104,
        f"  {'compound':26s} {'tox score':>10s}  {'DILI':>7s} {'SR-MMP':>7s} "
        f"{'hERG':>7s} {'AMES':>7s}",
    ]
    for entry in result["panel"]:
        row = entry["admet"]
        out.append(f"  {entry['label']:26s} {entry['toxicity_score']:10.3f}  "
                   + "  ".join(f"{float(row[k]):5.3f}" if row.get(k) else "    -"
                               for k in ("DILI", "SR-MMP", "hERG", "AMES")))

    out += ["", "=" * 104,
            "LEADS   * = also beats PLN-1474 on the toxicity composite",
            "        Per-endpoint values are printed so the reader can see WHICH",
            "        endpoints are better and which are worse, rather than trusting a",
            "        single composite number.",
            "=" * 104,
            f"  {'#':>2s} {'label':12s} {'tox':>7s} {'aff':>7s} {'Caco2':>7s} "
            f"{'Ca':>5s} {'don':>5s} {'DILI':>6s} {'MMP':>6s} {'p53':>6s} "
            f"{'AhR':>6s} {'hERG':>6s} {'AMES':>6s} {'L225':>4s} {'Y178':>4s}"]
    reference_row = {"label": f"{REFERENCE} (ref)",
                     "toxicity_score": result["reference_toxicity"],
                     "affinity": result["reference_affinity"],
                     "caco2": result["reference_caco2"]}
    panel_admet = next((p["admet"] for p in result["panel"]
                        if p["label"] == REFERENCE), {})
    get_ref = lambda k: (float(panel_admet[k]) if panel_admet.get(k)
                         else float("nan"))
    out.append(
        f"  {'--':>2s} {reference_row['label']:12s} "
        f"{reference_row['toxicity_score']:7.3f} {reference_row['affinity']:7.2f} "
        f"{reference_row['caco2']:7.3f} {'':>5s} {'':>5s} "
        f"{get_ref('DILI'):6.3f} {get_ref('SR-MMP'):6.3f} {get_ref('SR-p53'):6.3f} "
        f"{get_ref('NR-AhR'):6.3f} {get_ref('hERG'):6.3f} {get_ref('AMES'):6.3f} "
        f"{'':>4s} {'':>4s}")
    out.append("  " + "-" * 102)
    for index, lead in enumerate(result["leads"], start=1):
        row = lead["admet"]
        get = lambda k: float(row[k]) if row.get(k) else float("nan")
        mark = "*" if lead["beats_reference_toxicity"] else " "
        out.append(
            f"  {index:2d}{mark}{lead['label']:11s} {lead['toxicity_score']:7.3f} "
            f"{lead['affinity']:7.2f} {lead['caco2']:7.3f} "
            f"{float(lead['ca_dist']):5.2f} {float(lead['donor_dist']):5.2f} "
            f"{get('DILI'):6.3f} {get('SR-MMP'):6.3f} {get('SR-p53'):6.3f} "
            f"{get('NR-AhR'):6.3f} {get('hERG'):6.3f} {get('AMES'):6.3f} "
            f"{lead['leu225']:4d} {lead['tyr178']:4d}")

    flagged = [l["label"] for l in result["leads"] if l["asp218_basic"]]
    if flagged:
        out += ["", f"  pan-alphaV flag (Asp218 basic contact, section 8.4 ii): "
                    f"{', '.join(flagged)}"]
    out += ["", "  Every number here is a prediction. The ADMET models reach AUROC",
            "  0.84-0.93 on held-out benchmark splits whose leaderboard positions an",
            "  independent audit found largely unreproducible, and no selectivity",
            "  observation is evidence of selectivity (section 8.4). Wet-lab work is",
            "  what would settle any of it."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--geometry", default="results/geometry.csv")
    parser.add_argument("--panel-geometry", default="results/panel_geometry.csv")
    parser.add_argument("--admet", default="results/admet_survivors.csv")
    parser.add_argument("--survivors", default="data/survivors.smi")
    parser.add_argument("--selectivity", default="results/selectivity.csv")
    parser.add_argument("--n-leads", type=int, default=20)
    parser.add_argument("--out-csv", default="results/leads.csv")
    parser.add_argument("--out-txt", default=None)
    args = parser.parse_args(argv)

    result = select_leads(args.geometry, args.panel_geometry, args.admet,
                          args.survivors, args.selectivity, args.n_leads)
    text = _report(result, args.n_leads)
    print(text, end="")
    if args.out_txt:
        with open(args.out_txt, "w") as handle:
            handle.write(text)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = ["rank", "label", "smiles", "scaffold", "toxicity_score", "affinity",
              "ca_dist", "donor_dist", "caco2", "beats_reference_toxicity",
              "leu225", "tyr178", "asp218_basic"]
    fields += list(TOXICITY_WEIGHTS) + CONTEXT_ENDPOINTS
    with open(args.out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for index, lead in enumerate(result["leads"], start=1):
            row = {"rank": index, **{k: lead[k] for k in
                                     ("label", "smiles", "scaffold",
                                      "toxicity_score", "affinity", "ca_dist",
                                      "donor_dist", "caco2",
                                      "beats_reference_toxicity", "leu225",
                                      "tyr178", "asp218_basic")}}
            row.update({k: lead["admet"].get(k, "")
                        for k in list(TOXICITY_WEIGHTS) + CONTEXT_ENDPOINTS})
            writer.writerow(row)
    print(f"\n  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
