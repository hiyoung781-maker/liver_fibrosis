#!/usr/bin/env python3
"""Blueprint section 4 D3 acceptance checks.

Samples from each focused prior and from the untouched prior (the TL-C baseline),
then reports the checks that decide which arm carries forward. The chemotype-drift
row is the decisive one: if TL-B raises the Arg-mimic fraction or TPSA, it has primed
the region section 0 is trying to escape, and it is out regardless of its other numbers.

    python3 scripts/d3_report.py --priors A=priors/focused_A.prior B=priors/focused_B.prior \
        --baseline REINVENT4/priors/reinvent.prior --n 1000 [--wandb]
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent

# The same SMARTS section 3.1 uses, so "chemotype drift" means the same thing here
# as it does in the curation log.
ARG_MIMIC = [
    "[NX3][CX3](=[NX2])[NX3]",
    "[$(C1CCc2cccnc2N1),$(C1CCc2ccc[nX2]c2N1)]",
    "[nX2]1ccccc1[NX3;H1,H2]",
    "[nX2]1c([NX3])[nX3]c2ccccc12",
    "[NX3;H1,H2;!$(N=*)][c]([nX2,nX3])[nX2,nX3]",
    "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(Na);!$(N[SX4]);!$([N+](=O))]([CX4])[CX4,#1]",
]
_HEADS = [Chem.MolFromSmarts(s) for s in ARG_MIMIC]
_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)


def sample(model: Path, n: int, work: Path, tag: str) -> list[str]:
    cfg = work / f"sample_{tag}.toml"
    out = work / f"samples_{tag}.csv"
    cfg.write_text(
        'run_type = "sampling"\ndevice = "cuda:0"\n\n[parameters]\n'
        f'model_file = "{model}"\noutput_file = "{out}"\n'
        f"num_smiles = {n}\nunique_molecules = true\n"
    )
    subprocess.run(
        ["reinvent", "-l", str(work / f"sample_{tag}.log"), str(cfg)],
        check=True, capture_output=True,
    )
    rows = list(csv.DictReader(out.open()))
    if not rows:
        return []
    col = next(c for c in rows[0] if "smiles" in c.lower())
    return [r[col] for r in rows if r[col]]


def describe(smiles: list[str], refs: list) -> dict:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [m for m in mols if m is not None]
    if not valid:
        return {"sampled": len(smiles), "validity": 0.0}
    acid = sum(1 for m in valid if m.HasSubstructMatch(_ACID))
    arg = sum(1 for m in valid if any(m.HasSubstructMatch(p) for p in _HEADS))
    scaffolds = set()
    for m in valid:
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    nn = []
    for m in valid:
        fp = _FPGEN.GetCountFingerprint(m)
        nn.append(max(DataStructs.TanimotoSimilarity(fp, r) for r in refs))
    tpsa = [Descriptors.TPSA(m) for m in valid]
    return {
        "sampled": len(smiles),
        "validity": 100 * len(valid) / len(smiles),
        "carboxylic_acid_pct": 100 * acid / len(valid),
        "arg_mimic_pct": 100 * arg / len(valid),
        "unique_scaffolds": len(scaffolds),
        "scaffolds_per_100": 100 * len(scaffolds) / len(valid),
        "nn_tanimoto_mean": float(np.mean(nn)),
        "nn_tanimoto_p90": float(np.percentile(nn, 90)),
        "tpsa_median": float(np.median(tpsa)),
        "mw_median": float(np.median([Descriptors.MolWt(m) for m in valid])),
        "qed_median": float(np.median([Descriptors.qed(m) for m in valid])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priors", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--baseline", default="REINVENT4/priors/reinvent.prior")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--refs", default="data/actives_core.smi")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    ref_smiles = [
        l.split("\t")[0] for l in (ROOT / args.refs).open()
        if not l.startswith("#") and l.strip()
    ]
    refs = [_FPGEN.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in ref_smiles]
    print(f"NN-Tanimoto reference set: {args.refs} ({len(refs)} molecules)\n")

    targets = [("TL-C baseline", Path(args.baseline))]
    for spec in args.priors:
        name, _, path = spec.partition("=")
        targets.append((name, Path(path)))

    results = {}
    with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
        work = Path(tmp)
        for name, model in targets:
            if not model.exists():
                print(f"  skipping {name}: {model} not found")
                continue
            print(f"sampling {args.n} from {name} ...", flush=True)
            results[name] = describe(sample(model, args.n, work, name.replace(" ", "_")), refs)

    if not results:
        sys.exit("no priors could be sampled")

    base = results.get("TL-C baseline", {})
    rows = ["| check | " + " | ".join(results) + " |",
            "|---" * (len(results) + 1) + "|"]
    # Per-metric precision: NN-Tanimoto lives in a narrow band, so one decimal
    # collapses every prior onto the same number and says nothing.
    keys = {"validity": 1, "carboxylic_acid_pct": 1, "arg_mimic_pct": 1,
            "unique_scaffolds": 0, "nn_tanimoto_mean": 3, "nn_tanimoto_p90": 3,
            "tpsa_median": 1, "mw_median": 1, "qed_median": 3}
    for k, prec in keys.items():
        cells = [f"{results[n].get(k, float('nan')):.{prec}f}" for n in results]
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    keys = list(keys)
    print("\n" + "\n".join(rows))

    print("\nacceptance (section 4):")
    for name, r in results.items():
        if name == "TL-C baseline":
            continue
        checks = [
            ("validity >= 95%", r["validity"] >= 95),
            ("carboxylic acid above baseline",
             r["carboxylic_acid_pct"] > base.get("carboxylic_acid_pct", 0) + 5),
            ("NN-Tanimoto above baseline",
             r["nn_tanimoto_mean"] > base.get("nn_tanimoto_mean", 0)),
            ("not collapsed (scaffolds > 200 per 1000)", r["unique_scaffolds"] > 200),
        ]
        drift = (r["arg_mimic_pct"] - base.get("arg_mimic_pct", 0),
                 r["tpsa_median"] - base.get("tpsa_median", 0))
        print(f"  {name}:")
        for label, ok in checks:
            print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"    chemotype drift vs baseline: Arg-mimic {drift[0]:+.1f} pp, "
              f"TPSA {drift[1]:+.1f}  <- the arm-selection criterion")

    if args.wandb:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "avb1-denovo"),
                         group="D3-report", job_type="acceptance",
                         name="d3-acceptance", reinit=True)
        table = wandb.Table(columns=["prior"] + keys)
        for name, r in results.items():
            table.add_data(name, *[r.get(k) for k in keys])
        run.log({"d3_acceptance": table})
        for name, r in results.items():
            for k, v in r.items():
                run.summary[f"{name}/{k}"] = v
        run.finish()
        print(f"\n  wandb [{os.environ.get('WANDB_MODE', 'online')}]: acceptance table logged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
