"""Turn data/survivors.smi into 3D SDF shards for the section 8.5 docking run.

Sharding exists for load balance, not parallelism itself: the run fans out with
`xargs -P`, and many small shards let a fast core pick up more work instead of
waiting on a slow one. Docking time scales with rotatable bonds (survivor median 7,
p90 12), so shard durations vary by several-fold.

Each molecule's SDF title is its survivor label (`gen_NNNNN`), because that is the
identity every downstream artifact keys on - pose CSVs, passing-pose SDFs, ADMET
rows. smina preserves the title, so the label survives into its output.

Shard 0 is reserved for the redocking control: the crystal ligand, docked with the
settings the production shards use. Section 8.5(a) requires that validation, and
running it inside the same job on the same machine is what makes it a validation OF
THIS RUN rather than of the local workstation where it was first measured.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import shutil

__all__ = ["CONTROL_SHARD", "embed", "write_shards"]

RDLogger.DisableLog("rdApp.*")

CONTROL_SHARD = "shard_0000_control.sdf"
DEFAULT_SHARD_SIZE = 10


def embed(smiles: str, seed: int = 42) -> Chem.Mol | None:
    """One ETKDG conformer, MMFF-relaxed, hydrogens explicit.

    One conformer is enough because docking searches ligand torsions itself; the
    embedding only has to be a valid starting geometry. Returns None if embedding
    or optimization fails, so the caller can count and report the loss instead of
    writing a molecule with no coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None     # keep going; a failed relax is a dropped ligand, reported
    return mol


def write_shards(survivors_path: str, out_dir: str,
                 shard_size: int = DEFAULT_SHARD_SIZE,
                 control_sdf: str | None = None) -> dict:
    """Write shard SDFs and return counts plus the shard file list."""
    from novelty import load_smi

    rows = load_smi(survivors_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shards: list[str] = []
    if control_sdf:
        shutil.copyfile(control_sdf, out / CONTROL_SHARD)
        shards.append(CONTROL_SHARD)

    failures: list[str] = []
    index, writer, written = 0, None, 0
    for position, (smiles, label) in enumerate(rows):
        if position % shard_size == 0:
            if writer is not None:
                writer.close()
            index += 1
            name = f"shard_{index:04d}.sdf"
            shards.append(name)
            writer = Chem.SDWriter(str(out / name))
        mol = embed(smiles)
        if mol is None:
            failures.append(label)
            continue
        mol.SetProp("_Name", label)
        writer.write(mol)
        written += 1
    if writer is not None:
        writer.close()

    (out / "shards.txt").write_text("\n".join(shards) + "\n")
    return {
        "ligands": len(rows),
        "embedded": written,
        "embed_failures": len(failures),
        "failure_labels": failures,
        "shards": len(shards),
        "shard_size": shard_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--survivors", default="data/survivors.smi")
    parser.add_argument("--out-dir", default="docking/ligands")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--control", default="docking/ligand_ref.sdf",
                        help="crystal ligand for the redocking control shard")
    args = parser.parse_args(argv)

    counts = write_shards(args.survivors, args.out_dir, args.shard_size,
                          args.control)
    print(f"{args.survivors} -> {args.out_dir}")
    for key in ("ligands", "embedded", "embed_failures", "shards", "shard_size"):
        print(f"  {key:16s} {counts[key]}")
    if counts["failure_labels"]:
        print(f"  failed: {', '.join(counts['failure_labels'][:20])}"
              + (" ..." if len(counts["failure_labels"]) > 20 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
