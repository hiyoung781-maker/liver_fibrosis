"""A toxicity gate calibrated on approved drugs, not on one Phase 1 compound.

WHY THE EXISTING RULE CANNOT BE A GATE. Section 9.2 keeps a molecule when
SR-MMP, NR-AhR and DILI are each below PLN-1474's predicted value. Measured
against the 559 approved carboxylic-acid drugs in DrugBank that are not in this
repository's own benchmark, PLN-1474's three thresholds sit at the 65.3rd,
57.4th and 47.4th percentile and let 28.3% of them through. So the rule is
neither strict nor lenient in any stated sense -- it is wherever one Phase 1
compound happens to fall, on three axes, and nobody chose that position.

WHY results/v3_metal_asn224/admet_oracle_gate.md CANNOT BE A GATE EITHER. That
benchmark is 7 withdrawn and 9 marketed drugs, 63 comparison pairs, assembled
from recall; and its best-separating structural axis was chosen AFTER looking at
which drugs were in the withdrawn group. A bootstrap CI of [0.786, 1.000] and a
permutation p of 0.0004 say the separation is not an accident of the labels --
they say nothing at all about generalisation, and resampling the same sixteen
molecules cannot undo feature selection performed on them. Promoting that to a
gate would be fitting the filter to sixteen compounds.

WHAT THIS DOES INSTEAD. The threshold on each endpoint is a PERCENTILE of the
approved carboxylic-acid drug population, so "pass" has a statable meaning:
*this molecule scores at least as clean as the median approved acid drug on
every one of the three endpoints.* One knob (`--percentile`), one
interpretation, and a specificity that is known by construction rather than
discovered.

WHY ACIDS ONLY. Every molecule this campaign designs carries a carboxylate --
it is the MIDAS anchor, required by the reward function. Calibrating against
all 2,845 approved drugs would set the bar using chemotypes the campaign cannot
produce. 573 of them are acids with all three endpoints predicted; holding the
benchmark out removes 14 of those, leaving the 559 the thresholds are computed
from. (`n_excluded` in the JSON counts NAMES in the exclusion list, 17, not acid
drugs removed -- the two numbers do not sum to 573 and were never meant to.)

TWO POPULATIONS, TWO JOBS, AND THEY ARE NOT INTERCHANGEABLE. The 559 fix WHERE
each threshold sits: the 60th percentile of their predicted values is 0.6696 /
0.0111 / 0.0107. The 16 held-out drugs (7 withdrawn, 9 marketed) fix WHICH
percentile is used: the sweep in results/v3_metal_asn224/approved_drug_gate.md
keeps all 7 withdrawn drugs out up to p70 and first admits one at p80, while
marketed retention saturates at 5/9 by p60. So the threshold VALUES rest on 559
compounds and the CHOICE of 60 rests on 16 - a quantity worth stating separately,
because the second number is small enough that p60 should be read as the
strictest point inside a verified range rather than as a located optimum.

THE REFERENCE POPULATION IS CONTAMINATED AND THE OUTPUT SAYS SO. DrugBank's
"approved" flag covers drugs that were approved and later withdrawn -- six of
this repository's seven withdrawn benchmark drugs carry it. Those sixteen are
removed so the benchmark stays held out, but other withdrawn drugs certainly
remain among the 559 and have not been identified. The gate is therefore
calibrated against "approved, mostly not withdrawn", and a threshold set at the
median of that population is marginally more permissive than one set against a
clean population would be. The direction is known; the size is not.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lead_filter import TOXICITY_ENDPOINTS  # noqa: E402

__all__ = ["CARBOXYLATE", "build_reference", "percentile_thresholds",
           "joint_pass_rate", "apply_gate"]

CARBOXYLATE = "[CX3](=O)[OX2H1,OX1-]"


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_reference(drugbank_csv: str, exclude_names: set,
                    acids_only: bool = True) -> list:
    """Approved drugs with all three endpoints, carboxylic acids by default.

    `exclude_names` is matched case-insensitively on the drug name and exists
    for one purpose: keeping the validation benchmark out of the population the
    thresholds are derived from. Without it the gate would be calibrated on the
    molecules it is then tested against.
    """
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    pattern = Chem.MolFromSmarts(CARBOXYLATE)

    excluded = {n.strip().lower() for n in exclude_names}
    out = []
    for row in csv.DictReader(open(drugbank_csv)):
        name = (row.get("name") or "").strip()
        if name.lower() in excluded:
            continue
        values = {e: _float(row.get(e)) for e in TOXICITY_ENDPOINTS}
        if any(v is None for v in values.values()):
            continue
        mol = Chem.MolFromSmiles(row.get("smiles", ""))
        if mol is None:
            continue
        n_cooh = len(mol.GetSubstructMatches(pattern))
        if acids_only and n_cooh < 1:
            continue
        out.append({"name": name, "n_cooh": n_cooh, **values})
    return out


def percentile_thresholds(reference: list, percentile: float) -> dict:
    """The value below which `percentile`% of the reference population sits.

    Computed per endpoint with linear interpolation, the same convention
    numpy.percentile uses, written out so the gate has no numpy dependency in
    its definition and the convention cannot drift.
    """
    if not 0 < percentile < 100:
        raise ValueError(f"percentile must be in (0, 100), not {percentile}")
    if not reference:
        raise ValueError("empty reference population: nothing to calibrate on")

    thresholds = {}
    for endpoint in TOXICITY_ENDPOINTS:
        values = sorted(r[endpoint] for r in reference)
        index = (len(values) - 1) * percentile / 100.0
        low = int(index)
        high = min(low + 1, len(values) - 1)
        frac = index - low
        thresholds[endpoint] = values[low] * (1 - frac) + values[high] * frac
    return thresholds


def joint_pass_rate(population: list, thresholds: dict) -> float:
    """Fraction passing EVERY endpoint.

    The marginal percentile is not the gate's specificity: the three endpoints
    correlate, so thresholds at the 50th percentile each let 24.3% through
    jointly, not 12.5% and not 50%. The joint rate is the number to report.
    """
    if not population:
        return 0.0
    passing = sum(
        1 for r in population
        if all(r[e] < thresholds[e] for e in TOXICITY_ENDPOINTS))
    return passing / len(population)


def apply_gate(rows: list, thresholds: dict) -> list:
    """Tag each row with its per-endpoint verdict and the joint one."""
    out = []
    for row in rows:
        over = [e for e in TOXICITY_ENDPOINTS
                if row.get(e) is None or row[e] >= thresholds[e]]
        out.append({**row, "over": over, "kept": not over})
    return out


def _load_labelled(benchmark_csv: str, preds_csvs: list) -> list:
    """Benchmark drugs joined to their predictions.

    The benchmark carries the answer key and the structures; the endpoint
    values live in an admet_predict output. Reading only the benchmark gives
    every row a missing endpoint, every row is then skipped, and the
    "exclusions" silently become none -- which is how the reference population
    came back as 573 instead of 559 the first time this ran.
    """
    preds = {}
    for path in preds_csvs:
        for row in csv.DictReader(open(path)):
            values = {e: _float(row.get(e)) for e in TOXICITY_ENDPOINTS}
            if all(v is not None for v in values.values()):
                preds.setdefault(row["label"], values)

    rows, unmatched = [], []
    for row in csv.DictReader(open(benchmark_csv)):
        label = row.get("label", "")
        values = preds.get(label)
        if values is None:
            unmatched.append(label)
            continue
        rows.append({"label": label, "outcome": row.get("outcome", ""),
                     **values})
    if unmatched:
        sys.stderr.write(
            f"{len(unmatched)} benchmark drug(s) have no predictions and are "
            f"neither excluded nor validated: {', '.join(unmatched)}\n")
    return rows


def render(result: dict) -> str:
    t = result["thresholds"]
    out = [
        "# 승인 약물로 보정한 독성 게이트",
        "",
        "임계값을 단일 화합물이 아니라 **승인 카복실산 약물 집단의 백분위**로",
        "둔다. 따라서 통과에 진술 가능한 의미가 생긴다 — *이 분자는 세 지표",
        "모두에서 승인된 산성 약물의 중앙값만큼 깨끗하다.*",
        "",
        f"- 보정 집단: 승인 카복실산 약물 **{result['n_reference']}개** "
        f"(DrugBank, 검증 벤치마크 {result['n_excluded']}개 제외)",
        f"- 백분위: **{result['percentile']:g}**",
        f"- 보정 집단 결합 통과율: **{100 * result['reference_pass_rate']:.1f}%** "
        "(구성상 알려진 특이도)",
        "",
        "| 엔드포인트 | 게이트 임계값 | PLN-1474 값 | PLN-1474의 집단 내 백분위 |",
        "|---|---|---|---|",
    ]
    for endpoint in TOXICITY_ENDPOINTS:
        pln = result["pln_values"].get(endpoint)
        pct = result["pln_percentiles"].get(endpoint)
        out.append(
            f"| {endpoint} | {t[endpoint]:.4f} | "
            + (f"{pln:.4f}" if pln is not None else "—") + " | "
            + (f"{pct:.1f}" if pct is not None else "—") + " |")

    out += [
        "",
        f"기존 §9.2 규칙(PLN-1474 미만)의 결합 통과율은 이 집단에서 "
        f"**{100 * result['pln_pass_rate']:.1f}%**다. 즉 현행 규칙은 엄격하지도",
        "느슨하지도 않고, **한 Phase 1 화합물이 우연히 놓인 자리**다.",
        "",
        "## Held-out 검증 — 임상 결말이 알려진 16개",
        "",
        "이 16개는 보정 집단에서 제외했으므로 임계값을 보는 데 쓰이지 않았다.",
        "",
        "| 군 | n | 게이트 통과 | 통과율 |",
        "|---|---|---|---|",
    ]
    for group in ("marketed", "withdrawn"):
        info = result["holdout"].get(group)
        if not info:
            continue
        out.append(f"| {group} | {info['n']} | {info['kept']} | "
                   f"{100 * info['rate']:.1f}% |")
    out += [
        "",
        f"철회군 통과율 {100 * result['holdout']['withdrawn']['rate']:.1f}% 대 "
        f"현존군 {100 * result['holdout']['marketed']['rate']:.1f}%.",
        "",
        "**이 검증의 한계를 먼저 적는다.** n이 16이므로 통과율 차이의 신뢰구간이",
        "넓고, 이 결과는 게이트가 **알려진 정답을 뒤집지 않는다**까지만 말한다.",
        "게이트의 정당성은 이 16개가 아니라 보정 집단 "
        f"{result['n_reference']}개에서 나오며, 그쪽은 특이도가 구성상 고정돼",
        "있다.",
        "",
    ]
    if result.get("sweep"):
        out += [
            "## 게이트 강도 스윕",
            "",
            "리드 수는 게이트를 어디에 두느냐에 달려 있다. §2.4가 결합력",
            "컷오프에 한 것과 같은 이유로, 그 의존성을 숨기지 않고 적는다.",
            "",
            "| p | SR-MMP | NR-AhR | DILI | 참조 통과 | 철회 | 현존 | 리드 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        n_w = result["holdout"]["withdrawn"]["n"]
        n_m = result["holdout"]["marketed"]["n"]
        for entry in result["sweep"]:
            th = entry["thresholds"]
            mark = " ←" if entry["percentile"] == result["percentile"] else ""
            out.append(
                f"| {entry['percentile']:g}{mark} | {th['SR-MMP']:.4f} | "
                f"{th['NR-AhR']:.4f} | {th['DILI']:.4f} | "
                f"{100 * entry['reference_pass_rate']:.1f}% | "
                f"{entry['withdrawn_kept']}/{n_w} | "
                f"{entry['marketed_kept']}/{n_m} | "
                f"**{entry['n_leads']}** |")
        out += [
            "",
            "철회군이 0으로 유지되는 구간이 이 게이트의 **검증된 운용 범위**다.",
            "그 범위를 벗어나면 게이트가 알려진 정답을 뒤집기 시작한다.",
            "",
        ]

    if result.get("candidates"):
        out += [
            "## 후보 분자",
            "",
            "| 분자 | SR-MMP | NR-AhR | DILI | 게이트 | 초과 |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["candidates"]:
            out.append(
                f"| {row['label']} | "
                + " | ".join(f"{row[e]:.4f}" for e in TOXICITY_ENDPOINTS)
                + f" | {'**통과**' if row['kept'] else '탈락'} | "
                f"{', '.join(row['over']) if row['over'] else '—'} |")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--drugbank", default=None,
                   help="DrugBank approved CSV with ADMET-AI predictions. "
                        "Defaults to the copy bundled with the installed "
                        "admet_ai package.")
    p.add_argument("--benchmark", default="data/acid_drug_benchmark.csv",
                   help="Held out of the reference population AND used to "
                        "validate the thresholds it did not inform.")
    p.add_argument("--percentile", type=float, default=50.0)
    p.add_argument("--candidates", action="append", default=[],
                   help="admet_predict output to score against the gate")
    p.add_argument("--labels", default="",
                   help="comma-separated labels from --candidates to report")
    p.add_argument("--all-drugs", action="store_true",
                   help="Calibrate on every approved drug, not only the acids. "
                        "Off by default: this campaign's reward function makes "
                        "a carboxylate mandatory, so the non-acids are "
                        "chemotypes it cannot produce.")
    p.add_argument("--sweep", default="",
                   help="Comma-separated percentiles to tabulate alongside the "
                        "chosen one. The lead count depends on where the gate "
                        "is set, so the dependence is reported rather than "
                        "hidden behind one number -- the same reason section "
                        "2.4 sweeps the affinity cutoff.")
    p.add_argument("--sweep-candidates",
                   help="lead_filter --out-csv. Its rows that already cleared "
                        "selectivity are counted at each swept percentile.")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    drugbank = args.drugbank
    if drugbank is None:
        import admet_ai
        drugbank = str(Path(admet_ai.__file__).parent
                       / "resources" / "data" / "drugbank_approved.csv")
    if not Path(drugbank).exists():
        sys.stderr.write(f"missing DrugBank reference: {drugbank}\n")
        return 2

    bench = _load_labelled(args.benchmark, args.candidates)
    if not bench:
        sys.stderr.write(
            "no benchmark drug matched a predictions file. The reference "
            "population would then include every drug the gate is validated "
            "against, which is the circularity this gate exists to avoid.\n")
        return 2
    bench_names = {r["label"].replace("DRUG_", "").replace("_", " ")
                   for r in bench}

    reference = build_reference(drugbank, bench_names,
                                acids_only=not args.all_drugs)
    thresholds = percentile_thresholds(reference, args.percentile)

    gated_bench = apply_gate(bench, thresholds)
    holdout = {}
    for group in ("marketed", "withdrawn"):
        subset = [r for r in gated_bench if r["outcome"] == group]
        kept = [r for r in subset if r["kept"]]
        holdout[group] = {"n": len(subset), "kept": len(kept),
                          "rate": len(kept) / len(subset) if subset else 0.0,
                          "kept_labels": [r["label"] for r in kept]}

    # PLN-1474's own position, so the report can say what the old rule was.
    pln_values, pln_percentiles = {}, {}
    candidates = []
    wanted = [x.strip() for x in args.labels.split(",") if x.strip()]
    pool = {}
    for path in args.candidates:
        for row in csv.DictReader(open(path)):
            values = {e: _float(row.get(e)) for e in TOXICITY_ENDPOINTS}
            if any(v is None for v in values.values()):
                continue
            pool.setdefault(row["label"], values)
    for endpoint in TOXICITY_ENDPOINTS:
        pln = (pool.get("PANEL_PLN-1474") or {}).get(endpoint)
        if pln is None:
            continue
        pln_values[endpoint] = pln
        pln_percentiles[endpoint] = 100.0 * sum(
            1 for r in reference if r[endpoint] < pln) / len(reference)
    pln_pass_rate = (
        joint_pass_rate(reference, pln_values) if len(pln_values) == 3 else 0.0)

    for label in wanted:
        values = pool.get(label)
        if values is None:
            sys.stderr.write(f"no predictions for {label}\n")
            continue
        candidates.append({"label": label, **values})
    candidates = apply_gate(candidates, thresholds)

    sweep = []
    if args.sweep:
        pool_rows = []
        if args.sweep_candidates:
            for row in csv.DictReader(open(args.sweep_candidates)):
                if str(row.get("selectivity_pass")) != "True":
                    continue
                if str(row.get("affinity_pass")) != "True":
                    continue
                values = {e: _float(row.get(e)) for e in TOXICITY_ENDPOINTS}
                if all(v is not None for v in values.values()):
                    pool_rows.append({"label": row["label"], **values})
        for text_p in args.sweep.split(","):
            try:
                value = float(text_p)
            except ValueError:
                continue
            th = percentile_thresholds(reference, value)
            kept = [r["label"] for r in apply_gate(pool_rows, th) if r["kept"]]
            sweep.append({
                "percentile": value,
                "thresholds": th,
                "reference_pass_rate": joint_pass_rate(reference, th),
                "withdrawn_kept": sum(
                    1 for r in apply_gate(
                        [x for x in gated_bench if x["outcome"] == "withdrawn"],
                        th) if r["kept"]),
                "marketed_kept": sum(
                    1 for r in apply_gate(
                        [x for x in gated_bench if x["outcome"] == "marketed"],
                        th) if r["kept"]),
                "n_leads": len(kept),
                "leads": kept[:8],
            })

    result = {
        "percentile": args.percentile,
        "sweep": sweep,
        "acids_only": not args.all_drugs,
        "drugbank": drugbank,
        "n_reference": len(reference),
        "n_excluded": len(bench),
        "thresholds": thresholds,
        "reference_pass_rate": joint_pass_rate(reference, thresholds),
        "pln_values": pln_values,
        "pln_percentiles": pln_percentiles,
        "pln_pass_rate": pln_pass_rate,
        "holdout": holdout,
        "candidates": candidates,
    }

    text = render(result)
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        args.out.with_suffix(".json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
