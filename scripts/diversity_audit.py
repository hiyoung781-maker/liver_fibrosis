"""Does the RL agent memorize its training set, or collapse onto few scaffolds?

Transfer learning ran 179 epochs on 191 compounds (data/actives_core_B.smi). At
that ratio the agent could in principle be reproducing its training set rather
than generalizing, and RL on a narrow objective could additionally collapse onto
a handful of scaffolds. Either failure would make every downstream count a count
of near-duplicates. Neither is visible in the leads alone - 13 molecules say
nothing about the 13,767 they were drawn from - so the measurement has to be made
on the library.

Three quantities, each answering a different failure:
  memorization    nearest-neighbour Tanimoto from each generated molecule to the
                  TRAINING set. High values mean the agent is reciting.
  internal        1 - mean pairwise Tanimoto WITHIN the sample. Low values mean
  diversity       mode collapse, regardless of what the training set looked like.
  scaffold reuse  unique Murcko scaffolds, and how many coincide with training
                  scaffolds. Separates "new molecules" from "new frameworks".

Sampling rather than the full library: pairwise Tanimoto is O(n^2), and 13,767
molecules is 95M pairs. --sample fixes the draw with --seed so the numbers are
reproducible.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

GEN = AllChem.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles: list[str]) -> tuple[list, list[Chem.Mol]]:
    """Morgan fingerprints and mols, unparseable entries dropped from both."""
    fps, mols = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(GEN.GetFingerprint(mol))
        mols.append(mol)
    return fps, mols


def nearest_neighbour(query_fps: list, reference_fps: list) -> list[float]:
    """For each query, the max Tanimoto against the reference set."""
    return [max(DataStructs.BulkTanimotoSimilarity(fp, reference_fps))
            for fp in query_fps]


def internal_diversity(fps: list, pairs: int, rng: random.Random) -> float:
    """1 - mean Tanimoto over `pairs` random distinct pairs.

    Random pairs rather than all pairs: the mean of a random sample of pairs is an
    unbiased estimate of the mean over all of them, at O(pairs) instead of O(n^2).
    """
    n = len(fps)
    if n < 2:
        raise ValueError("internal diversity needs at least 2 molecules")
    sims = []
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1          # uniform over j != i, without a rejection loop
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return 1.0 - statistics.mean(sims)


def scaffolds(mols: list[Chem.Mol]) -> list[str]:
    out = []
    for mol in mols:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return out


def read_smiles(path: str) -> list[str]:
    """Column 1 of a .smi/.smi-with-labels file, comments skipped."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split()[0])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", default="data/v4_TL-B/library.smi")
    parser.add_argument("--training", default="data/actives_core_B.smi")
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--pairs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    library = read_smiles(args.library)
    if args.sample and args.sample < len(library):
        library = rng.sample(library, args.sample)
    training = read_smiles(args.training)

    gen_fps, gen_mols = fingerprints(library)
    train_fps, train_mols = fingerprints(training)
    print(f"library  {args.library}: {len(gen_mols)} parsed"
          f" (sample of {args.sample})" if args.sample else "")
    print(f"training {args.training}: {len(train_mols)} parsed")

    nn = nearest_neighbour(gen_fps, train_fps)
    nn_sorted = sorted(nn)
    def pct(p): return nn_sorted[min(int(p * len(nn_sorted)), len(nn_sorted) - 1)]
    print("\n== memorization: nearest-neighbour Tanimoto to the training set ==")
    print(f"  mean {statistics.mean(nn):.3f}  median {statistics.median(nn):.3f}"
          f"  p90 {pct(0.90):.3f}  p99 {pct(0.99):.3f}  max {max(nn):.3f}")
    for cut in (0.4, 0.6, 0.85, 0.95):
        n = sum(1 for v in nn if v >= cut)
        print(f"    >= {cut:<5} {n:6d} ({100 * n / len(nn):5.2f}%)")

    div = internal_diversity(gen_fps, args.pairs, rng)
    print(f"\n== mode collapse: internal diversity ({args.pairs} random pairs) ==")
    print(f"  mean pairwise Tanimoto {1 - div:.3f}  ->  internal diversity {div:.3f}")

    gen_scaf, train_scaf = scaffolds(gen_mols), set(scaffolds(train_mols))
    uniq = set(gen_scaf)
    print("\n== scaffold reuse: unique Murcko scaffolds ==")
    print(f"  {len(uniq)} unique in {len(gen_scaf)} molecules"
          f" ({100 * len(uniq) / len(gen_scaf):.1f}%)")
    print(f"  training scaffolds: {len(train_scaf)}"
          f"   also generated: {len(uniq & train_scaf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
