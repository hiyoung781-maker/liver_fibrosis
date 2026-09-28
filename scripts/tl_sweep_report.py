#!/usr/bin/env python3
"""TL-A' 체크포인트 스윕의 집계와 epoch 선택 (spec §4.2).

선택 규칙은 도킹 전에 확정된 것이며, 결과를 보고 바꾸지 않는다.
암기 지표는 여기서 선택에 쓰이지 않고 보고 컬럼으로만 남는다 - v1에서 암기
꼬리가 하류의 Murcko 신규성 게이트(known_scaffolds.smi)에서 제거되는 것이
확인되었기 때문이다. 다만 신규성을 주장할 때는 이 꼬리가 몇 개 제거되었는지
반드시 함께 보고해야 한다.

CLI usage (spec §4.2 step 5 - requires the K-BDS GPU sweep to have produced
checkpoints first, see scripts/run_d3.sh --mode production-sweep):

    python scripts/tl_sweep_report.py --checkpoints priors/sweep_A_prime \
        --n-samples 1000 --out results/v2_tl_sweep.csv

Checkpoints are discovered as REINVENT4 names them: `<prefix>.<epoch>.chkpt`
for every save epoch except the last, and bare `<prefix>` for the final
epoch (num_epochs). Epoch 0 is the untouched de novo prior (--baseline),
included so the sweep curve has a starting point.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = ["MIN_EPOCH", "SCAFFOLD_FLOOR", "select_epoch"]

MIN_EPOCH = 100
SCAFFOLD_FLOOR = 700

ROOT = Path(__file__).resolve().parent.parent


def select_epoch(rows: list[dict]) -> int | None:
    """규칙을 만족하는 가장 높은 epoch. 없으면 None (= TL-A′ 탈락)."""
    qualifying = [r["epoch"] for r in rows
                  if r["epoch"] >= MIN_EPOCH
                  and r["unique_scaffolds"] >= SCAFFOLD_FLOOR]
    return max(qualifying) if qualifying else None


# --------------------------------------------------------------------- CLI

def discover_checkpoints(prefix: Path, final_epoch: int) -> dict[int, Path]:
    """Map epoch -> checkpoint path for a REINVENT4 TL run.

    Intermediate epochs are saved as `<prefix>.<epoch>.chkpt`; the final
    epoch is saved as bare `<prefix>` (no .chkpt suffix).
    """
    found: dict[int, Path] = {}
    pattern = re.compile(re.escape(prefix.name) + r"\.(\d+)\.chkpt$")
    parent = prefix.parent
    if parent.is_dir():
        for candidate in parent.iterdir():
            m = pattern.match(candidate.name)
            if m:
                found[int(m.group(1))] = candidate
    if prefix.exists():
        found[final_epoch] = prefix
    return found


def sample(model: Path, n: int, work: Path, tag: str) -> list[str]:
    """Sample `n` SMILES from a REINVENT4 model file. Mirrors d3_report.sample."""
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
    """Per-checkpoint metrics: acid_pct, unique_scaffolds, max_nn_tanimoto,
    tpsa_median, qed_median. Requires rdkit, imported lazily so `select_epoch`
    (and its tests) never need rdkit installed."""
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid = [m for m in mols if m is not None]
    if not valid:
        return {"acid_pct": 0.0, "unique_scaffolds": 0, "max_nn_tanimoto": 0.0,
                "tpsa_median": 0.0, "qed_median": 0.0}
    acid = sum(1 for m in valid if m.HasSubstructMatch(_ACID))
    scaffolds = set()
    for m in valid:
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
        except Exception:
            pass
    nn = [0.0] * len(valid)
    if refs:
        from normalize import FPGEN
        nn = []
        for m in valid:
            fp = FPGEN.GetCountFingerprint(m)
            nn.append(max(DataStructs.TanimotoSimilarity(fp, r) for r in refs))
    return {
        "acid_pct": 100 * acid / len(valid),
        "unique_scaffolds": len(scaffolds),
        "max_nn_tanimoto": float(max(nn)) if nn else 0.0,
        "tpsa_median": float(np.median([Descriptors.TPSA(m) for m in valid])),
        "qed_median": float(np.median([Descriptors.qed(m) for m in valid])),
    }


def run_sweep(checkpoints: dict[int, Path], n_samples: int, refs_path: str,
              work: Path) -> list[dict]:
    """Sample every checkpoint and describe it. Real sampling requires `reinvent`
    (REINVENT4) on PATH and a GPU; see scripts/run_d3.sh --mode production-sweep."""
    from rdkit import Chem
    from normalize import FPGEN

    refs = []
    refs_file = ROOT / refs_path
    if refs_file.exists():
        for line in refs_file.open():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                refs.append(FPGEN.GetCountFingerprint(mol))

    rows = []
    for epoch in sorted(checkpoints):
        model = checkpoints[epoch]
        smiles = sample(model, n_samples, work, f"epoch{epoch}")
        metrics = describe(smiles, refs)
        rows.append({"epoch": epoch, **metrics})
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["epoch", "acid_pct", "unique_scaffolds", "max_nn_tanimoto",
                  "tpsa_median", "qed_median"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["epoch"]):
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                     help="prefix, e.g. priors/sweep_A_prime (no .prior suffix)")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-epoch", type=int, default=200)
    ap.add_argument("--refs", default="data/actives_extended.smi",
                     help="memorization reference set for max_nn_tanimoto")
    ap.add_argument("--baseline", default="REINVENT4/priors/reinvent.prior",
                     help="untouched prior sampled as epoch 0")
    args = ap.parse_args()

    prefix = Path(args.checkpoints)
    checkpoints = discover_checkpoints(prefix, args.final_epoch)
    baseline = ROOT / args.baseline
    if baseline.exists():
        checkpoints.setdefault(0, baseline)
    if not checkpoints:
        sys.exit(f"no checkpoints found at prefix {prefix}")

    with tempfile.TemporaryDirectory(dir=ROOT / "logs") as tmp:
        rows = run_sweep(checkpoints, args.n_samples, args.refs, Path(tmp))

    write_csv(rows, Path(args.out))
    print(f"wrote {len(rows)} rows to {args.out}")

    chosen = select_epoch(rows)
    if chosen is None:
        print("NO EPOCH QUALIFIES (scaffolds >= 700 for epoch >= 100) - "
              "TL-A' is dropped as an arm; see results/v2_tl_decision.md")
    else:
        print(f"selected epoch: {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
