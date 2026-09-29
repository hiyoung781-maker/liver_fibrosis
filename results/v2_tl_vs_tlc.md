# TL-C vs TL-A-prime: head-to-head measurement

n = 2000 sampled per prior. See notes below for reference sets and settings.

| metric | TL-C (untrained, reinvent.prior) | TL-A-prime (epoch 180, focused_A_prime.prior) |
|---|---|---|
| n sampled | 1957 | 1848 |
| validity % | 100.0% | 100.0% |
| n valid | 1957 | 1848 |
| n unique valid molecules | 1957 | 1848 |
| carboxylic acid % (95% CI) | 8.9% (7.7-10.2) | 54.1% (51.8-56.4) |
| Arg-mimetic % (95% CI) | 22.7% (20.9-24.6) | 14.8% (13.2-16.5) |
| Arg-mimetic % difference (TL-A' minus TL-C), 95% CI | — | -7.9 pp (-10.4, -5.5) |
| unique Murcko scaffolds per 1000 sampled | 929.5 | 697.5 |
| scaffolds-per-molecule ratio | 0.929 | 0.698 |
| **NOVELTY GATE: % with Murcko scaffold in known_scaffolds.smi (106 scaffolds)** | **3.4%** (66/1957) | **17.6%** (325/1848) |
| NN-Tanimoto vs actives_extended.smi: mean | 0.291 | 0.420 |
| NN-Tanimoto: median | 0.290 | 0.374 |
| NN-Tanimoto: p90 | 0.356 | 0.669 |
| NN-Tanimoto: **MAX** (memorization signal) | **0.591** | **1.000** |
| % at NN-Tanimoto >= 0.680 (p25 of novelty band) | 0.0% (0/1957) | 8.9% (165/1848) |
| TPSA median | 72.8 | 102.2 |
| MW median | 377.4 | 402.2 |
| QED median | 0.604 | 0.531 |
| SAScore median | 2.761 | 2.845 |
| % passing property window (logP<=5.0, MW 250-550, TPSA 40-115) | 57.3% (1121/1957) | 46.5% (859/1848) |
| **joint: carboxylic acid AND scaffold-novel (context, most decision-relevant single number)** | **8.3%** (163/1957) | **40.6%** (750/1848) |

## Notes

- Sample size: n = 2000 requested per prior via REINVENT4 `sampling` run_type, `unique_molecules = true`, `device = "cuda:0"`, identical TOML template for both priors (the only difference is `model_file`). REINVENT4 v4.5.11. reinvent conda env (/home/seyoung/miniconda3/envs/reinvent/bin/python).
- NN-Tanimoto reference set: `data/actives_extended.smi` (190 lines read, 190 successfully tautomer-normalized) — this is the SAME reference set for both priors, unlike the two earlier (non-comparable) measurements in scripts/d3_report.py (actives_core.smi, 25 molecules) and scripts/tl_sweep_report.py (actives_extended.smi, 190).
- Fingerprint: Morgan radius 3, feature invariants (`GetMorganFeatureAtomInvGen`), count fingerprints (`GetCountFingerprint`), via `scripts/normalize.FPGEN` — same generator object used for scaffold-gate reference construction elsewhere in this repo.
- Tautomer normalization: `scripts.normalize.canonical_tautomer` (flatten stereochemistry -> LargestFragmentChooser -> Uncharger -> RDKit TautomerEnumerator.Canonicalize) applied to BOTH sampled molecules and reference molecules before Murcko scaffold extraction and before fingerprinting for NN-Tanimoto, exactly as `scripts/known_scaffolds.py` does for the real gate.
- Known-scaffold set: `scripts.known_scaffolds.known_scaffolds()` computed live from the same 5 reference files known_scaffolds.smi was built from (106 scaffolds; matches the file's header count).
- Property window: imported from `scripts.build_survivors.WINDOW` / `passes_window` rather than retyped; logP computed with `rdkit.Chem.Crippen.MolLogP` inside that function.
- SAScore: RDKit's Ertl/Schuffenhauer implementation from `REINVENT4/reinvent_plugins/components/SAScore/sascorer.py` (same module REINVENT4 itself uses for its SAScore component).
- Tautomer canonicalization failures (fell back to the flattened, non-canonicalized molecule): TL-C=0, TL-A-prime=0.
- TL-C carboxylic-acid % measured here (8.9%) falls inside the 6.3-9.3% range spanned by the two earlier (differently-configured) measurements; the SMARTS is identical (`curate_actives.CARBOXYLIC_ACID`) and the difference between those two earlier runs is attributable to sampling variance at their n (this run uses n=2000, giving a tighter Wilson interval of (7.7, 10.2)).
- n_valid used as the denominator for all percentages (not n_sampled), matching d3_report.py's and tl_sweep_report.py's convention.
