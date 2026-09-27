# 10-Day Blueprint: De Novo Design of Integrin αvβ1 Inhibitors with REINVENT4

**Project:** evidence-based target prioritization → in silico inhibitor design for liver fibrosis
**Target:** integrin αvβ1 (hepatic stellate cell–intrinsic, direct antifibrotic target)
**Reference structure:** PDB 8W30 — αvβ1 headpiece in complex with TR01225179 [112]
**Companion files:** `avb1_target_decision_brief.md` (argument architecture + attack surface), `reinvent4_avb1_scoring_config_sketch.toml` (annotated, syntax-verified configs), `avb1_pocket_3d.png`, `avb1_pocket_contacts.png/.svg`, `avb1_8w30_contacts.csv`
**Curated data (§3.1, executed):** `scripts/curate_actives.py` → `data/actives_core.smi`, `data/actives_core_B.smi` (+ `_train`/`_valid` splits), `data/actives_extended.smi`, `data/similarity_refs.smi`, `data/benchmark_panel.smi`, `data/actives_annotated.csv`, `data/CURATION_LOG.md`
**Compute environment (D1, executed):** `TRANSFER.md`, `scripts/kbds_probe.sh`, `scripts/setup_kbds.sh`, `env/`, `slurm/` — REINVENT4 pinned at **v4.5.11**, torch **2.5.1+cu118**, conda env **`reinvent`**

---

## 0. Design hypothesis (one sentence)

The αvβ1 **binding** pharmacophore — a MIDAS-metal-coordinating carboxylate, an H-bond donor to β1-Asn224, and an aromatic element against αv-Tyr178 — can be carried by **novel, more drug-like scaffolds** than the published azabenzimidazolone series, whose 1.3% rat oral bioavailability and series-wide MDCK permeability < 0.1×10⁻⁶ cm/s are a **series-specific ceiling, not a target-intrinsic one** [111].

Everything below is engineered to test that sentence — and to be able to say, honestly, if it fails.

---

## 1. What the pocket requires — structure translated into scoring-function terms

Derived from the 8W30 coordinate analysis (see `avb1_8w30_contacts.csv`):

| # | Pocket feature | Geometry in 8W30 | Consequence for generation/scoring |
|---|---|---|---|
| 1 | **Carboxylate → MIDAS Ca²⁺ (β1 Ca501)** | 2.62 Å, monodentate (second carboxylate O at 4.54 Å) | **Non-negotiable anchor.** All known αvβ1/αv-class inhibitors are carboxylates (RGD mimetics). → `MatchingSubstructure` carboxylic acid in RL; post-hoc docking filter: carboxylate O → Ca501 ≤ 3.2 Å |
| 2 | **Ligand amide NH → β1-Asn224 backbone O** | 2.63 Å H-bond. **Asn224 is conserved in β3** — this confers binding, not selectivity (§8.4) | Strong secondary anchor → post-hoc H-bond filter (donor within 3.5 Å of Asn224 O); in RL captured indirectly via similarity to actives, all of which make this contact |
| 3 | **Aromatic contact with αv-Tyr178** | 3.70 Å closest atoms, T-shaped/offset (centroid 6.21 Å, ring angle 74.6°) | ≥1 aromatic ring expected; not enforced directly in the objective — confirmed by the §8.5 docking geometric filter and by qualitative review |
| 4 | **β1 hydrophobic subpocket** (Pro186 3.16 Å, Tyr133 3.17 Å, Cys187 3.88 Å, Lys182 3.82 Å, Leu225 2.95 Å). **Only Leu225 differs in β3** (which carries Arg) — if selectivity comes from anywhere, it is here (§8.4) | dichlorophenyl of the hit sits here | Hydrophobic bulk is tolerated — SlogP up to about 3 is **clearly permitted**; potency is decided here. **Note:** this row was the source of §5's `SlogP` penalty threshold of "about 3", but it is a qualitative **permitted floor** derived from contact distances ("at least up to 3 is fine"), not "penalize above 3" — a misreading that turned a permitted lower bound into a penalty upper bound. §5.4 corrects this misreading and removes `SlogP` from the objective |
| 5 | **αv-Asp218 — NOT engaged by the 8W30 hit** (7.0 Å) | engaged by cpd 25's benzimidazolone NH in the published series [111]. **How it is engaged is what matters** — a salt bridge from a basic Arg mimetic is associated with pan-αv activity (the 1,8-naphthyridine of GSK3008348 [114]), a neutral monodentate H-bond is not. Being an αv residue, it cannot account for selectivity among αvβx (§8.4) | Growth vector. Deliberately **not** encoded as a substructure filter (would over-constrain novelty); used in manual inspection of top docked poses |
| 6 | **Interface water HOH2107** | 4.68 Å from ligand; bridges αv-Glu121 (2.6 Å) / β1-Ser177 (3.0 Å) | Possible water-mediated or water-displacement gains; post-hoc inspection only |
| — | MIDAS-loop residues Ser132/Ser134/Glu229 | 3.0–3.3 Å from carboxylate | Context for the metal site; no direct constraint |

**The permeability problem this creates:** the carboxylate is fixed by the pocket, so permeability must be bought back on the *rest* of the molecule — TPSA budget, lipophilic efficiency, rotatable bonds, MW. This is exactly what Step 2 RL optimizes (Section 5 — a single stage represents this with the TPSA axis, while the remaining axes are viewed broadly through the §8.2 triage window), and it is why the tension between the MIDAS carboxylate and permeability is pre-registered as an honest falsification risk (Section 9).

---

## 2. Mode selection — which REINVENT4 run types, and why

REINVENT4 [39, 40] provides run types `sampling`, `scoring`, `transfer_learning`, `staged_learning`, `enumeration`, and generators Reinvent (de novo), Mol2Mol, LibInvent, LinkInvent, Pepinvent.

| Mode / generator | Role | Verdict |
|---|---|---|
| Reinvent de novo + `transfer_learning` | Focus the prior on curated αvβ1 actives | **PRIMARY — Step 1** |
| Reinvent de novo + `staged_learning` | RL against the multi-parametric objective | **PRIMARY — Step 2** |
| `sampling` | Draw the final library from the best agent | **PRIMARY — Step 3** |
| `scoring` | Push the benchmark panel through the identical funnel | **PRIMARY — Step 4** |
| Mol2Mol | Close-analog arm around cpd 25 | **SECONDARY / optional** — similarity-constrained by construction, so it cannot deliver scaffold novelty; useful only as a "fix cpd 25's properties locally" fallback |
| LibInvent | Decorate a **fixed** scaffold | **REJECTED** — the hypothesis is scaffold novelty; fixing the azabenzimidazolone core presupposes the answer the project is trying to escape |
| LinkInvent | Link two fragments | **REJECTED** — no fragment pair to link; single contiguous pocket |
| Pepinvent | Peptides | **REJECTED** — small-molecule project |
| `enumeration` | Reaction-based enumeration | **REJECTED** — not generative; produces no novelty |

**Why de novo Reinvent is the only defensible primary:** the gap documented in prior art is the absence of a *public, selective, oral-like* αvβ1 chemotype — CWHM-12 is unselective [103], cpd 25 is selective but not drug-like (oral F 1.3%) [111], PLN-1474 is drug-like but proprietary and discontinued [88], bexotegrast is a dual αvβ6/αvβ1 developed for lung disease [98]. Scaffold novelty **is** the deliverable, and only the de-novo generator plus a scaffold diversity filter can produce it.

---

## 3. Data preparation (D1–D2)

### 3.1 Actives sets — EXECUTED, see `data/CURATION_LOG.md`

Built by `scripts/curate_actives.py` against ChEMBL_37. All source IDs are verified, not assumed.

**Sources.** ChEMBL document `CHEMBL5500400` = Sabat 2024 [111] (doi 10.1021/acs.jmedchem.4c00743), 26 molecules — the paper's numbering runs to 25 plus the external comparator C8, so ChEMBL's coverage of the series is **complete and no PDF transcription is needed**. ChEMBL target `CHEMBL2111407` = Integrin αvβ1, 413 activities over 253 molecules. RCSB CCD `A1AFA` = the 8W30 ligand TR01225179, *N*-(2,4-dichlorobenzoyl)-L-phenylalanine — taken from the CCD because ChEMBL's copy (`CHEMBL5556476`) has lost the stereocentre. PubChem for the named tool compounds.

**The chemotype split that drives everything below.** Molecules are tagged by whether they carry an Arg-mimetic basic head (guanidine/cyclic amidine, tetrahydronaphthyridine, 2-aminopyridine, benzimidazol-2-amine, 2-aminoazole, aliphatic amine). Those that do bind in the zwitterionic RGD mode; the 8W30 ligand and the Sabat series carry none.

| | n | Murcko scaffolds | RGD-zwitterion | MW | TPSA | QED |
|---|---|---|---|---|---|---|
| all ChEMBL αvβ1 actives ≤ 1 µM | 197 | 103 | 88% | 498 | 147 | 0.38 |
| **non-RGD only** | **24** | — | 0% | — | ~146 | — |
| 8W30 ligand (A1AFA) | 1 | — | 0% | 338 | 66 | 0.88 |

**The binding constraint: the chemotype cannot be scaled up.** Loosening the potency cut does not help — non-RGD molecules number 35 at ≤ 1 µM, 36 at ≤ 10 µM and 43 with no cut at all, of which 11 were later reclassified as RGD and 4 proved to be a curation error (below). Requiring *both* the non-RGD chemotype *and* a permeability-friendly envelope (TPSA ≤ 140, MW ≤ 550, QED ≥ 0.3) leaves **n = 11** in the whole public record. That near-disjointness is the strongest available evidence that the gap in §0 is real, and it is why the TL set is small by construction rather than by choice.

**Two ChEMBL data-quality findings, both acted on.** Document `CHEMBL1629507` contributes four molecules whose assay reads *"Inhibition of human integr**ase** alphaVbeta1 … in african green monkey Vero E6 cells"* — the paper is *Small molecule inhibitors of hantavirus infection* and no record carries a quantitative value. They are not αvβ1 actives. Separately, three Sabat molecules sit at IC50 = 10000 nM exactly (a censored "> 10 µM") and are SAR negatives. All are excluded and listed in the curation log.

**Two molecule counts, both correct.** Curation dedupes by isomeric InChIKey; the `.smi` files dedupe again on the *flattened* SMILES, because the prior cannot tell stereoisomers apart. Seven core_B pairs collapse that way, so core_B is **200 curated molecules → 193 the generator can distinguish**, and extended is **197 → 190**. The `.smi` counts are the ones quoted below; `actives_annotated.csv` keeps all 200 isomeric records.

**Stereochemistry: the `.smi` files are flattened, and this is not a detail.** The de novo prior's vocabulary, read straight out of `REINVENT4/priors/reinvent.prior`, is 33 tokens and contains **no stereochemistry at all** — no `@`, `@@`, `/` or `\`:

```
# %10 ( ) - 0 1 2 3 4 5 6 7 8 9 = Br C Cl F N O S [N+] [N-] [O-] [S+] [n+] [nH] c n o s
```

REINVENT4 filters out-of-vocabulary SMILES silently, so feeding isomeric input to transfer learning would have shrunk the sets without any error:

| set | n | isomeric survives | flattened survives |
|---|---|---|---|
| core (TL-A) | 29 | **2** | **29** |
| core_B (TL-B) | 193 | 30 | 193 |
| extended | 190 | 30 | 190 |
| similarity_refs | 6 | 1 | 6 |
| benchmark panel | 6 | **0** | 6 |

Every `.smi` file therefore carries flattened SMILES, and `curate_actives.py` gates on the prior's actual vocabulary so this cannot regress. The isomeric form — including the 8W30 ligand's L configuration — is preserved in `actives_annotated.csv` (`smiles` column) and is what the docking stage (§8.5) consumes.

**Sets written** (standardisation: RDKit cleanup → largest fragment → neutralise → canonical isomeric SMILES → InChIKey dedupe → flatten for the `.smi` files):

| file | n | role |
|---|---|---|
| `data/actives_core.smi` | **29** | **TL-A input + inception.** 25 non-RGD + 4 RGD tool compounds |
| `data/actives_core_B.smi` | **193** | **TL-B input.** TL-A plus every confirmed RGD-zwitterion active |
| `data/actives_extended.smi` | 190 | **novelty NN-Tanimoto baseline only** — not TL, not QSAR |
| `data/similarity_refs.smi` | 6 | one molecule per scoring endpoint (see §5) |
| `data/benchmark_panel.smi` | 6 | §3.4 positives |
| `data/folds/actives_core_fold{1..5}_{train,valid}.smi` | 29 per fold | TL-A **diagnostic** folds, scaffold-disjoint |
| `data/folds/actives_core_B_fold{1..5}_{train,valid}.smi` | 193 per fold | TL-B **diagnostic** folds, scaffold-disjoint |
| `data/actives_annotated.csv` | 254 rows | every candidate: isomeric `smiles` **and** `smiles_flat`, membership flags, potency, chemotype, physchem |

The four RGD tool compounds are in the core deliberately, for potency signal; the `chemotype` column keeps their contribution separable at D3.

**QC gates (all three pass).** Every emitted SMILES is representable by the prior's vocabulary. Every core molecule carries a carboxylic acid. No core molecule carries a protonatable nitrogen that the Arg-mimic patterns missed — this second gate exists because an early SMARTS omission mislabelled 11 2-aminoimidazole RGD mimetics as non-RGD.

**C8 is deliberately absent.** It is not a Sabat compound but an external inhibitor the paper cites, and the name resolves to nothing in PubChem. The closest ChEMBL candidate (`CHEMBL3957812`, pIC50 7.9 on the N-arylsulfonyl-L-proline scaffold) matches on a single coincident number, so it is recorded in the log rather than guessed into the panel.

### 3.2 Counter-screen actives (selectivity deselection)
- Pull ChEMBL actives (IC50/Ki ≤ 1 µM) for integrin **αvβ3** and **α5β1**.
- **ChEMBL target IDs verified against ChEMBL_37** (IC50 record counts in brackets): αvβ1 `CHEMBL2111407` [314], αvβ3 `CHEMBL1907598` [2158], α5β1 `CHEMBL2095226` [766], αvβ6 `CHEMBL2111416` [1361], αvβ8 `CHEMBL3430892` [276]. The Sabat document alone carries cross-integrin IC50 for αvβ3/α5β1/αvβ6/αvβ8/αvβ5/α4β1, so the selectivity picture for the primary series comes free with §3.1.
- Note from the §3.1 audit: **αvβ1 selectivity does not require abandoning the RGD zwitterion.** Document `CHEMBL3862028` (*N-Arylsulfonyl-L-proline … Potent and Selective αvβ1 Inhibitors*, ACS Med Chem Lett 2016) contributes 36 molecules that are selective and sub-nM yet all carry an Arg mimic. Selectivity and chemotype are independent axes — the non-RGD core is chosen for **permeability**, not selectivity. Expect this on the poster.
- ~~Use post-hoc: flag any generated molecule with ECFP4 Tanimoto ≥ 0.5 to a counter-screen active.~~ → **This clause is withdrawn (2026-09-27)**, for two reasons. First, **the set was never produced** — there is no αvβ3/α5β1 actives file under `data/`, and although §10's D2 is marked "all data frozen", this item was missed. Second, as the caveat above says of itself, similarity is not activity, and selectivity and chemotype are independent axes. §8.4 replaces it with three structural observations and §9.4 removes selectivity from the criteria.

### 3.3 Potency model (`ChemProp2`) — **NO-GO, decided at curation time**

The ≥ 100-molecule condition cannot be met in any defensible form:

- **Assay heterogeneity.** The 413 αvβ1 activities span **35 distinct assays** — cell adhesion, solid-phase receptor assay, ELISA-TMB, fluorescence-polarisation displacement, HEK293 binding — run under different cation conditions. Integrin potency depends strongly on Mn²⁺/Mg²⁺ activation, so pooled pIC50 variance is substantially assay, not chemistry.
- **No coherent subset is large enough.** The biggest single assay holds **n = 36** (`CHEMBL3863781`, "unknown origin" cell adhesion, 6.7 log span); the next are 30, 27, 26, 25. Nothing approaches 100.
- **No negative class for a classifier either.** Only 32 molecules measure > 1 µM and just 10 carry a censored (`>`) relation.

**This verdict does not depend on the compute platform.** The K-BDS glibc constraint (§11) separately removed the `chemprop` *package* from the install, but that is a later, unrelated decision. The reason ChemProp2 is out is the assay record itself, which is the same on any machine — do not let the two be conflated when the proposal is challenged.

**Consequence:** the potency proxy is `TanimotoSimilarity`, which makes §5's endpoint handling load-bearing rather than optional. The real gain from the ChEMBL pull is not a QSAR model but a **novelty baseline computed over 197 actives instead of ~30**, which materially strengthens the §8.3 scaffold-novelty claim.

### 3.4 Benchmark panel (frozen on D2)
- **Positives:** `data/benchmark_panel.smi` — the 8W30 ligand A1AFA, the most potent non-RGD Sabat compound (cpd 25 class), PLN-1474, bexotegrast, CWHM-12, GLPG0187. C8 is excluded; see §3.1.
- **The negative panel is discarded (2026-09-26).** The objective gates on the carboxylate (§5.1), so a molecule without a carboxylic acid has a total-score ceiling of 6.31e-04. Random drug-like ChEMBL molecules mostly lack a carboxylic acid, so a negative panel would pile up at that floor, and "the generated set beats the negative panel" would just be re-measuring "the generated molecules carry a carboxylic acid" — a fact the objective already enforces and §5.5 already reports as a pass rate. It carries no information, so it is not built.
- The only comparison worth making is against the positive panel, and only on **non-circular axes**: §8.5 docking geometry, §9.2 ADMET-AI permeability, Murcko novelty (§8.3), and alert/counter-screen flags. **Total score is not used for comparison** — it is the value of an objective function we designed, so comparing on it is circular (§5.4).
- This panel passes through the **identical** scoring + triage funnel as generated molecules. This is what makes every poster comparison honest.

---

## 4. Step 1 — Transfer learning: focus the prior (D3)

**Rationale:** the random prior covers all of drug-like space; αvβ1 inhibitors are a thin sliver (mandatory carboxylate + specific geometry). TL shifts sampling density toward the target region so RL does not waste hundreds of steps rediscovering the carboxylate.

**Three arms, decided at D3 on data rather than by argument.** A TL run is minutes on GPU, so the "is 29 molecules enough?" question is answered empirically instead of by convention. REINVENT4's own reference config (`configs/transfer_learning.toml`) uses 100 molecules and calls > 200 "large".

| arm | input | n | RGD-zwitterion | rationale |
|---|---|---|---|---|
| **TL-A** | `data/actives_core.smi` | 29 | 14% | chemotype-pure; the hypothesis direction |
| **TL-B** | `data/actives_core_B.smi` | 193 | 88% | conventional TL size; strongest binding signal, but primes the agent toward the space §0 is trying to escape |
| **TL-C** | none — RL straight from `reinvent.prior` | — | — | the no-TL baseline; also §10's global fallback |

**Recipe** (full block in `reinvent4_avb1_scoring_config_sketch.toml`):
- `run_type = "transfer_learning"`, device CUDA
- Priors **ship with REINVENT4 v4.5.11** at `REINVENT4/priors/` (ten files with `sha512.sum`, all verified) — no Zenodo download. Input = `REINVENT4/priors/reinvent.prior`, output = `focused_A.prior` / `focused_B.prior`
- `num_epochs = 10`, `batch_size = 32`, `num_refs = 0`
- **No `isomeric_smiles` key.** v4.5.11's TL config has no such field and its `GlobalConfig` sets pydantic `extra="forbid"`, so including it is a hard startup error. It would be wrong regardless: the prior cannot represent stereochemistry (§3.1), which is why the input `.smi` files are flattened.
- **`randomize_all_smiles = true`** — this is not optional at n = 29. The defaults (`randomize_smiles = true`, `randomize_all_smiles = false`) randomise each SMILES **once at file read** and then train on that fixed string every epoch, so there is no augmentation at all. Setting `randomize_all_smiles = true` routes through `Dataset._getitem_with_randomization`, which draws a fresh random SMILES on every access — the standard remedy for a small transfer set (the source comments "if true much shallower minimum"). Verified in `reinvent/runmodes/TL/run_transfer_learning.py` and `reinvent/models/reinvent/models/dataset.py`; the field and its default are identical in v4.5.11 and `main`.
- `num_epochs` is **not guessed** — it comes from the diagnostic pass below.

**Two kinds of TL run, and they must not be confused.**

| | input | `validation_smiles_file` | purpose |
|---|---|---|---|
| **Diagnostic** (5 runs per arm) | `data/folds/actives_<set>_fold<i>_train.smi` | the matching `_valid.smi` | find the epoch at which held-out NLL turns up |
| **Production** (1 run per arm) | the **full** `data/actives_core.smi` (29) or `actives_core_B.smi` (193) | **none** | the prior that actually generates, trained for the epoch count the folds identified |

An earlier draft of this section fed the full set to training *and* named a subset of it as validation — the validation molecules were inside the training set, so held-out NLL would have measured memorised data and detected nothing. Keep the two runs separate.

Withholding molecules permanently is not an option at this n: 6 of 29 is 21% of the entire public non-RGD chemotype. The folds exist to choose a hyperparameter, not to define the model.

**Why folds rather than one split.** The folds are scaffold-disjoint because a random split leaked — 50% of TL-A's validation molecules and 65% of TL-B's shared a Murcko scaffold with a training molecule, since this data is a few congeneric series, and held-out NLL then measures recall of a near-twin rather than generalisation. But one scaffold split is not enough either: 29 molecules over 19 scaffolds puts ~6 in validation, where a single molecule moves the estimate by 17% and the result is dominated by which scaffolds happened to land there. **Report mean and range across the five folds; never a single number.**

TL-A's folds are uneven by construction — fold 1 is the 9-molecule azabenzimidazolone series, which is one scaffold and cannot be split. That fold is the hardest test in the set and is expected to score worst; that is information, not a defect.

**Limits, stated rather than hidden.**
- **Chemotype cannot be stratified.** The core holds only 4 RGD-zwitterions in total, so they cannot be spread evenly over 5 scaffold-disjoint folds (they fall 0/1/1/0/2). Potency is spread: every fold spans roughly pIC50 6–9.
- **Never compare TL-A's held-out NLL with TL-B's.** Different set sizes and scaffold counts make the comparison confounded. Arms are chosen on the D3 sampling diagnostics below, with chemotype drift decisive.
- **Held-out NLL is a secondary diagnostic, never a gate.** There is a fair counter-argument that scaffold generalisation is not TL's job at all — TL concentrates probability mass near known actives, and the RL diversity filter does the scaffold hopping. Under that reading, held-out-scaffold NLL under-rates a perfectly good TL run. The gate is the sampling checks; the NLL number chooses `num_epochs` and answers the "did you validate on held-out actives?" question when it is asked.

### D3 result — EXECUTED, TL-A selected

Run on a local TITAN Xp (sm_61), not K-BDS; the environments are pinned identically (§11).

**Arm selection.** Sampling 1,000 from each prior, against the untouched prior as the TL-C baseline:

| | TL-C baseline | **TL-A (200 ep)** | TL-B (48 ep) |
|---|---|---|---|
| validity | 100.0 | 100.0 | 100.0 |
| carboxylic acid % | 7.6 | **51.1** | 62.2 |
| **Arg-mimic %** | 21.8 | **20.2 (−1.6 pp)** | **48.7 (+26.0 pp)** |
| unique scaffolds / 1000 | 946 | 747 | 854 |
| TPSA median | 71.7 | 103.7 | 107.0 |
| QED median | 0.588 | 0.508 | 0.438 |

**TL-B is out on the pre-registered criterion**, confirmed across three independent samplings: its Arg-mimic drift grew with training (+17.5 → +26.0 pp), which is what a set that is 88% RGD-zwitterion must do. **TL-A carries forward**: it raises the carboxylate density 6.7-fold while leaving the chemotype untouched, which is the whole point of restricting the core set to non-RGD actives in §3.1.

**Epoch count came from a sweep, not the fold mean.** The fold diagnostic hit its ceiling twice (5 of 10 folds at budget 20, 2 of 10 at budget 50), so the "best epoch" it reported was the budget, not a minimum. A sweep of the production prior — one run to 200 epochs, checkpointing every 20, sampling 1,000 at each — showed the metrics trade smoothly rather than turning over:

| epoch | 0 | 40 | 80 | 120 | 160 | 200 |
|---|---|---|---|---|---|---|
| carboxylic acid % | 8.0 | 28.0 | 43.8 | 47.0 | 49.6 | 55.4 |
| Arg-mimic % | 21.8 | 21.6 | 18.5 | 21.3 | 22.7 | 18.5 |
| unique scaffolds | 923 | 874 | 827 | 806 | 779 | 736 |
| QED median | 0.599 | 0.518 | 0.519 | 0.491 | 0.513 | 0.482 |

No collapse anywhere — 736 unique scaffolds per 1,000 is far above the §4 bar of "≫ the 29 actives". A separate 200-epoch probe on one fold did find an interior validation-NLL minimum at epoch 60, but that is one fold, and the five-fold spread at budget 50 was 17–50; the number is too dispersed to set a hyperparameter alone. **Production uses 200 epochs**, chosen on the sampling metrics.

**The honest cost of 200 epochs: the prior memorises part of its training set.** Against `actives_extended` (190 molecules), TL-A's samples are 95.1% below the §8.3 novelty threshold, but 4.9% fall inside the known-series similarity band, 1.2% above its p90, and the maximum NN-Tanimoto is **1.000** — some samples reproduce a training active verbatim. The untouched prior's maximum is 0.506, so that entire tail is TL-induced. It is filtered downstream by §8.3 and penalised by the RL diversity filter, but any novelty claim must state that this tail was removed first.

**TL raises TPSA and lowers QED** (+32.0 and −0.080 against baseline). That is expected — the training actives sit at TPSA ~145 — and it is Step 2 RL's job to pull them back, not TL's. Do not describe TL as improving drug-likeness; TL plants the carboxylate pharmacophore, RL shapes the properties.

**D3 acceptance checks** (sample 1,000 from each focused prior):
- validity ≥ 95%
- fraction containing a carboxylic acid markedly higher than the random-prior baseline
- mean NN-Tanimoto to actives shifted up vs random prior
- **not collapsed:** unique Murcko scaffolds in the 1,000 samples ≫ number of actives (if samples ≈ the actives themselves, reduce `num_epochs` to ~5)
- **chemotype drift (the arm-selection criterion):** fraction of samples carrying an Arg-mimetic basic head, scored with the same SMARTS as §3.1, plus the TPSA distribution. TL-B is expected to raise both; if it does, it has primed the wrong region and TL-A or TL-C carries forward regardless of its other numbers.

---

## 5. Step 2 — Single-stage learning (RL): a single-stage design (D4–D5)

> **2026-09-26 redesign.** The former two-stage curriculum (Stage 1 focus, 200 steps → Stage 2 optimize, 400 steps) has been discarded and replaced with a single stage of 600 steps. Full rationale and measurements are in `docs/superpowers/specs/2026-09-26-step2-single-stage-rl-design.md`.

**Shared settings:** `learning_strategy` type `"dap"`, `sigma = 128`, `rate = 0.0001`; `batch_size = 128` (GPU); `diversity_filter` = `IdenticalMurckoScaffold`, `bucket_size = 25`, `minscore = 0.4` (**applied throughout**); `inception` seeded with actives (a memory of "what a good molecule looks like"); aggregation is `geometric_mean`; `max_steps = 1000`.
* DAP (Differentiable Augmented Posterior): recomputes the log-likelihood in differentiable form at every step and backpropagates through it. It is REINVENT4's default recommendation and converges faster and more stably than alternative strategies such as MAULI/MASCOF/SDAP. Keep `dap` unless there is a specific reason not to.
* sigma: the weight that determines how hard the score pushes against the prior probability. Since the score is normalized to 0–1, sigma = 128 means a molecule with score 1.0 is preferred over the prior by 128 nats of log-likelihood.
  - Low (60–80): stays close to the prior → good synthesizability/chemical plausibility but slow score improvement
  - High (150–256): score rises fast, but exploitation gets so aggressive that the agent collapses onto one or two scaffolds (mode collapse) or reward-hacks (exploits loopholes in the scoring function for unrealistic structures)
  - 128 is REINVENT4's standard default and a reasonable starting point when mixing multiple objectives
* bucket_size = 25: once 25 molecules of the same scaffold have accumulated, any further molecule of that scaffold has its score forced to zero. From the agent's perspective this reads as "this scaffold no longer pays" → it is pushed to explore a new one.
* minscore = 0.4: only molecules scoring ≥ 0.4 are counted toward the bucket. This stops junk molecules from occupying a scaffold slot and blocking a promising scaffold early.
* inception (seeded with actives): an experience-replay buffer. It keeps the highest-scoring molecules seen so far in a small memory (typically `memory_size` ~100) and mixes a sample of them (`sample_size`, typically 10) into every batch's loss.
  - **What seeding (feeding known actives into `smiles_file`) does:** early in RL the agent rarely produces high-scoring molecules, so the gradient signal is sparse. Pre-loading known actives into the buffer gives it positive examples of "roughly what the target looks like" from step 1, which substantially speeds convergence.
  - Caveat: if the seed is too strong, the agent orbits the seed molecules and novelty suffers. Always check the Tanimoto similarity between the resulting molecules and the seeds.
  - Verify first that the seed molecules actually score well under the scoring function — a low-scoring seed sends the wrong signal instead.

The diversity filter caps how many molecules per identical Murcko scaffold can keep contributing reward — this is the mechanism that forces scaffold hopping and is what makes the novelty claim possible. Prior work [113] applied the diversity filter only in the final stage, to drive *convergence* onto an enriched chemotype; our goal runs the opposite direction, so it is kept in force throughout.

### 5.1 The objective — three scored endpoints and one filter

| Component | Kind | Role | Transform |
|---|---|---|---|
| `GroupCount` carboxylate | **scored** (w=1.0) — **gate** | MIDAS anchor, non-negotiable | `right_step(high=1)` → present 1.0 / absent 0.0 |
| `TPSA` | **scored** (w=1.0) | permeability axis for an acidic ligand | `double_sigmoid(low=40, high=115, coef_div=120, coef_si=coef_se=20)` |
| `SAScore` | **scored** (w=0.5) | **guard rail — not an optimization axis** (§5.4) | `reverse_sigmoid(low=6.0, high=8.0, k=0.5)` |
| `CustomAlerts` | **filter** (0/1 mask) | structural liabilities | below |

```toml
[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name = "carboxylate MIDAS anchor"
weight = 1.0
params.smarts = "[CX3](=O)[OX2H1,OX1-]"
transform.type = "right_step"
transform.high = 1
```

**Aggregation — confirmed in the REINVENT4 source and verified against real output.** `reinvent/scoring/scorer.py:159-178` puts only scored components into `aggregate`; penalty components are multiplied onto the result afterward. This configuration uses no penalty components, so

```
total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SAScore, 0.5)]) × alert_filter
      = prod(max(scoreᵢ, 1e-8) ** (wᵢ / Σw)) × alert_filter          # aggregators/means.py
```

This re-implementation was checked against real REINVENT output in `logs/cmp.vigHoB/new3.csv` — errors of TPSA 2.2e-07, sigmoid 8.7e-08, **total 1.9e-08** (the 19 molecules caught by the filter are excluded, since REINVENT overwrites every component to 0 for them).

**Why a gate and not `MatchingSubstructure` — the old configuration taught the agent to drop the carboxylate.** `comp_matching_substructure.py:63` **hardcodes** the score as `0.5 * (1.0 + match)` (0.5 when absent, not tunable). Scoring prior samples under that configuration gives:

| prior sample | n | total-score median | p90 | max |
|---|---|---|---|---|
| carboxylic acid **present** | 254 | **0.060** | 1.000 | 1.000 |
| carboxylic acid **absent** | 189 | **0.500** | 0.500 | 0.500 |

Without a carboxylic acid the score is pinned at 0.500; with one, the median is only 0.060 — because the carboxylic acid raises TPSA by roughly 37 and pushes the molecule out of the window. **The agent's expected-reward-optimal strategy was to drop the MIDAS carboxylate** — exactly the structure this project has declared non-negotiable.

`CustomAlerts` does not fix this — it scores "0 if matched," so it cannot express **absence**, and SMARTS has no molecule-level negation. Hence `GroupCount` + `right_step` is used as a scored endpoint. Because `geometric_mean` clamps 0 to 1e-8, the total-score ceiling when the carboxylic acid is absent is **6.31e-04** (all 219 of the sampled priors sit below it) — about 128 nats short of a perfect score under DAP σ=128. Not literally zero, but a gate in every practical sense. Raising `w_COOH` pushes the floor lower still (w=3 → 4.6e-06), but it dilutes the TPSA exponent and flattens the gradient, so **w=1** is kept.

`CustomAlerts` patterns:

```
[NX3;H2][c]            # primary anilines only — anilides/sulfonanilides/THN excluded (§5.4)
[*;r8] [*;r9] [*;r10]
N=[N+]=[N-]
C(=O)Cl
[SH]
[Nr0][Nr0]
```

PAINS is **not expressible in this TOML** — `reinvent_plugins` has no PAINS-capable component, and the RL run therefore applies only these eight SMARTS. PAINS is applied downstream instead: in `scripts/objective.py`'s offline reference scorer and in the §8 triage filter, both of which run the RDKit PAINS catalogue directly.

**Why the objective is limited to three endpoints:** prior work [113] gave docking score, QED and SAscore **equal weights of 0.3 each** (REINVENT4 normalizes internally, so the effective weight is 1/3 each). This is not a matter of taste but a condition for steerability. Under `geometric_mean`, each endpoint is a multiplicative veto, so adding endpoints collapses the achievable ceiling, and the result is "a molecule optimized purely for the objective function" rather than "a molecule that fits our purpose." Fine-grained filtering belongs to §8's post-hoc triage and qualitative review, not the objective.

### 5.2 What was removed from the objective

| removed | reason | moved to |
|---|---|---|
| **all of Stage 1** (200 steps) | §5.3 | dropped |
| `TanimotoSimilarity`, 6 references | structurally unachievable (§5.3) | §5.5 diagnostic |
| `TanimotoSimilarity`, 2 anti-targets | already present in §8.4 — redundant | kept in §8.4 |
| `SlogP` | threshold of "3" has no basis, and its direction is backwards (§5.4) | §8.2 triage window (logP ≤ 5) |
| `QED` | cannot distinguish a clinical compound from a random prior sample (§5.4) | §8.7 reported metric |
| `MolecularWeight`, `NumRotBond`, `HBD` | redundant with QED/TPSA, five-way overlap (§5.1) | §8.2 triage window |

§3.3 rules out `ChemProp2` and similarity is removed here too, so **the objective has no potency term.** This is a deliberate division of labor — potency signal comes from TL-A (carboxylic acid 7.6% → 51.1%, §4) and inception, and potency judgment is made post-hoc by the §8.5 docking geometric filter.

> **`TanimotoSimilarity` takes only one reference per endpoint — verified in the source and reproduced.** `comp_similarity.py` emits one score array per reference SMILES (`scores.extend([... for fingerprint in fingerprints])`), but `compute_scores.py` zips those arrays against the **endpoint** list (`:107` in v4.5.11, `:175` in `main` — the same defect in both versions). With N references under one endpoint, `zip` truncates: only the first reference is scored and the rest are **silently discarded**. REINVENT4 also has no similarity component that aggregates over a reference set (`reinvent/chemistry/similarity.py`'s max-aggregating `calculate_tanimoto` is used only by the Mol2Mol sampler). **This is the decisive reason similarity is dropped from the objective** — declaring each of the 6 references as its own independent endpoint is the only way around it, and that runs straight into the impossibility condition of §5.3.

### 5.3 Why a single stage and not a curriculum

**(1) The "stay near the actives" signal was triplicated.** TL-A already concentrates the prior on the active region (measured at D3), inception replays actives into the RL memory, and Stage 1's similarity term pulled toward the actives again for 200 steps. TL and inception are passive and free. Stage 1 was the third copy of the same signal, and the only one that spent 200 RL steps to do it.

**(2) A "pull then push" sequence contradicts the novelty hypothesis.** The old §5 monitoring paragraph said as much on its own — "similarity **falls** in Stage 2, and that fall is the sign of scaffold novelty appearing." Stage 1 spent 200 steps raising a value that Stage 2 then spent 400 steps undoing. If novelty is the goal, the honest move is never to raise it in the first place.

**(3) The premise for a curriculum only half holds.** The argument for staging is that the initial objective is ill-conditioned — under the default transforms, prior samples score near zero, leaving no gradient to learn from. Prior work [113] is exactly that condition (equal weights + REINVENT4's default transforms + 3 stages × 30 epochs). We have, over the last three iterations, calibrated every window to the measured prior distribution instead. That was reconfirmed (`logs/cmp.vigHoB/new3.csv` — 488 of 493 molecules in `molecules.smi` match `samples_prior.csv`, so this is effectively the TL-A prior sample distribution):

| component | p10 | median | p90 | score < 0.05 |
|---|---|---|---|---|
| TPSA | 0.000 | **0.894** | 1.000 | 38% |
| SlogP | 0.008 | **0.943** | 1.000 | 16% |
| QED | 0.170 | **0.468** | 0.788 | 4% |
| carboxylic acid | 0.500 | 1.000 | 1.000 | 4% |
| alerts | 1.000 | 1.000 | 1.000 | 4% |
| sim A1AFA | 0.000 | **0.004** | 0.982 | 65% |
| sim CHEMBL4649232 | 0.000 | **0.002** | 0.885 | 70% |
| sim CHEMBL5532604 | 0.000 | **0.001** | 0.984 | 71% |
| **total** (geometric_mean) | 0.000 | **0.011** | 0.332 | **70%** |

**The property axes are well-conditioned. The similarity axis is not, at all.** And the similarity axis's ill-conditioning is not a calibration failure — it is a **logical impossibility**: the 6 molecules in `data/similarity_refs.smi` were deliberately chosen to be **mutually dissimilar** (max pairwise Tanimoto 0.45); declaring each as its own endpoint and combining them under `geometric_mean` demands "resemble all 6 mutually-dissimilar molecules at once." No weighting fixes that.

So the prescription is not a curriculum but **removing similarity from the objective**. Doing that lets `geometric_mean` be kept as-is — the AND problem arose because one endpoint was unachievable, and removing it restores the multiplicative aggregation to its intended purpose (preventing an escape route where similarity *or* properties alone is satisfied). TL already did the focusing work, so the premise for a curriculum is gone.

### 5.4 Component-by-component rationale

**The `TPSA` window = `(40, 115)`, anchored on two oral clinical compounds.** The cpd 25 series cannot be used to justify the window — its high TPSA (median 133.3, max 170.0) reflects that **it was never proposed as an oral structure**, so citing it would just restate §0's premise that "cpd 25 is outside the window" (circular). The external anchors are the only two αv carboxylates that actually advanced orally:

| compound | TPSA | recorded basis |
|---|---|---|
| **PLN-1474** | **100.6** | oral, αvβ1-selective, Phase 1 completed cleanly (84 subjects, no DLT) [85] |
| **bexotegrast** | **112.5** | oral dual αvβ6/αvβ1 inhibitor, Phase 2a INTEGRIS [98, 99] |
| A1AFA | 66.4 | the 8W30 crystal ligand — structural floor |
| cpd 25 series (25 Sabat compounds) | median 133.3 / max 170.0 | oral F 1.3%, MDCK < 0.1×10⁻⁶ [111] — never proposed as oral |
| GLPG0187 / CWHM-12 | 158.7 / 172.4 | unselective / non-oral |

The floor of 40 sits below the crystal ligand (66.4), so it penalizes nothing. The ceiling of 115 is set **with a margin over the higher of the two oral clinical compounds (bexotegrast, 112.5)**. Window comparison (carboxylate gate applied; the "acid subset" is the 254 carboxylic-acid-bearing prior samples):

| window | bexotegrast | PLN-1474 | A1AFA | acid median | acid > 0.5 | Sabat median |
|---|---|---|---|---|---|---|
| **(40, 115) — adopted** | 0.878 | 0.998 | 1.000 | **0.178** | 43% | **0.060** |
| (40, 120) | 0.978 | 1.000 | 1.000 | 0.372 | 48% | 0.130 |
| (45, 125) | 0.997 | 1.000 | 1.000 | 0.661 | 52% | 0.276 |
| (50, 130) | 1.000 | 1.000 | 0.999 | 0.910 | 59% | 0.546 |
| (60, 140) — rejected | 1.000 | 1.000 | 0.968 | 0.976 | 63% | **0.968** |

The acid median of 0.178 at `(40, 115)` is **the intended direction** — the acids TL leaves behind sit at high TPSA (p25 102 / median 128 / p75 158 among carboxylic-acid-bearing samples), and RL's job is to pull that down. At the same time, **33% already score above 0.95 and 43% above 0.5**, so there is enough positive signal to imitate at step 0 — neither vanishing nor saturated. Widening the window raises the acid median but rewards the very Sabat series the design is trying to escape: already 0.546 at `(50, 130)`, 0.968 at `(60, 140)`. Treating bexotegrast as "allowed but borderline" (0.878) is honest, since a lung-indication dual inhibitor is not the optimum for a liver αvβ1 compound. The Sabat series falling outside the window is a **consequence**, not the justification.

> **Calculation correction (2026-09-27).** An earlier version of this table applied a subset filter of `total score > 1e-3`, which excluded low scorers and **over-reported** the acid median at (40,115) as 0.647. The figures above are the corrected ones. The window choice does not change — it rests on the two oral anchors and the exclusion of the Sabat series, not on the median.

**This constrains the poster wording.** "The generated molecules have lower TPSA than cpd 25" is guaranteed by construction and cannot be presented as a finding. The permeability claim must be carried by §9.2 (ADMET-AI, an independent model) and §8.5 (docking geometry).

**For the same reason, ADMET-AI is not put in the RL objective.** Doing so would make §9.2 circular too. Keeping TPSA in RL and ADMET-AI in post-hoc evaluation is what keeps the permeability claim non-circular — the same kind of deliberate constraint as §12's "no docking inside RL."

**`SlogP` removed.** Three files disagreed — this document said "reverse_sigmoid, penalize above ~3", `reinvent4_avb1_scoring_config_sketch.toml` said `double_sigmoid(low=1, high=4)`, and the actually-run `logs/cmp.vigHoB/new3.toml` used `reverse_sigmoid(low=2, high=5, k=0.4)`. The 0.5-crossing of `reverse_sigmoid(2, 5, 0.4)` is actually **3.5** (logP 3.0 already scores 0.823, 3.25 scores 0.683), so "about 3" pointed at where the penalty becomes noticeable, not its crossing point. The source is §1's pocket table row 4 ("hydrophobic bulk tolerated → SlogP up to about 3"), but that row is a qualitative **permitted floor** derived from contact distances ("at least up to 3 is fine"), not "penalize above 3" — a misreading that turned a permitted lower bound into a penalty upper bound. The low/high values themselves are not derived from either document. More importantly, the direction is wrong: this project builds an **anion** with a mandatory carboxylate, and §9.2 requires "beat cpd 25 on predicted permeability." For an acidic compound the permeability bottleneck is the ionized carboxylate, not excess lipophilicity, so trimming logP above 3 would degrade the very axis this project claims to improve. Prior work [113]'s own reference is not a soft penalty either, but a hard filter (exclude MW > 500 or logP > 5). → moved to the §8.2 triage window.

**`QED` removed.** PLN-1474 (the only αvβ1 inhibitor to reach the clinic) has a QED of **0.4619**, and the TL-A prior sample median QED is **0.468**. The one clinical-stage compound is indistinguishable from an ordinary prior sample, so it cannot serve as an optimization axis. (0.433 is the value computed on a non-aromatic tautomer and is an artifact of §8.0.) → kept only as a §8.7 reported metric.

**`SAScore` is a guard rail, and the window is `(6, 8, k=0.5)`.** First, the raw SA distribution:

| set | SA median | SA p90 | SA max |
|---|---|---|---|
| `actives_core` | 3.08 | 3.30 | 3.75 |
| `actives_extended` | 3.48 | 4.27 | **5.10** |
| `benchmark_panel` | 3.29 | 3.48 | 3.75 |
| TL-A prior (488) | 2.91 | 3.72 | **5.02** |

The sketch's `(3, 6, k=0.5)` is **unfit as a guard rail.** It looks harmless at the median (≥ 0.98 across every set) but bites in the tail — `actives_extended`'s worst molecule (SA 5.10) scores **0.091**, the prior's worst scores **0.121**. That penalizes known ChEMBL αvβ1 actives, the same class of error as the aniline filter below. Curves by window candidate:

| window | SA=3.0 | SA=4.0 | SA=5.1 | SA=6.0 | SA=7.0 | SA=8.0 | min over all reference sets |
|---|---|---|---|---|---|---|---|
| `(3, 6, k=0.5)` — sketch | 0.997 | 0.872 | **0.091** | 0.003 | 0.000 | 0.000 | **0.093** |
| `(5, 8, k=0.4)` | 1.000 | 1.000 | 0.987 | 0.823 | 0.177 | 0.010 | 0.987 |
| `(6, 9, k=0.4)` | 1.000 | 1.000 | 0.999 | 0.990 | 0.823 | 0.177 | 0.999 |
| **`(6, 8, k=0.5)` — adopted** | 1.000 | 1.000 | **1.000** | 0.997 | 0.500 | 0.003 | **0.99998** |

`(6, 8, k=0.5)` is adopted. The **worst molecule across the reference sets and prior samples scores 0.99998** — `reverse_sigmoid` is a logistic curve, so it approaches 1.0 asymptotically without ever landing exactly on 1.000. `actives_extended`'s worst molecule (SA 5.095) scores 0.999983, and the prior's worst (SA 5.017) scores 0.999989; after `geometric_mean`'s 0.2 exponent, that lowers a total score by **2.2e-06** — effectively no contribution at all. And it does not start biting until SA > 6, the conventional synthesizability boundary, becoming 0.5 at SA 7 and a near-veto at SA 8. `(6, 9, k=0.4)` is too permissive — still 0.823 at SA 7 — while `(5, 8, k=0.4)` starts biting at SA 6, earlier than the conventional boundary.

**This document does not claim SAScore optimizes synthesizability** — at step 0 it scores 0.99998 or better, contributing nothing measurable (a total is depressed by 2.2e-06), a form of insurance that only engages if 600 steps of RL drift into territory the prior never sampled.

**Honest summary: at step 0, the only optimization axes actually alive are the carboxylate anchor and TPSA.** Given that TL has already done the focusing, that is the consistent conclusion.

**Aniline pattern replaced — `[NH2,NH][c]` → `[NX3;H2][c]`.** The old pattern was measured with RDKit across the whole dataset (a match means `CustomAlerts` — a filter — zeroes the total score):

| set | `[NH2,NH][c]` (old) | `[NX3;H2][c]` (new) |
|---|---|---|
| `actives_core` (29) | **9/29 (31%)** | 0/29 |
| `actives_extended` (190) | **164/190 (86%)** | 0/190 |
| `similarity_refs` (6) | **3/6** | 0/6 |
| `benchmark_panel` (6) | **5/6** | 0/6 |
| TL-A prior samples (488) | **118/488 (24%)** | 12/488 (2.5%) |

The 5 zeroed in `benchmark_panel` are **PLN-1474, bexotegrast, CWHM-12, GLPG0187, CHEMBL4649232** — only A1AFA survives. Two consequences follow: **(a) it caught tetrahydro-1,8-naphthyridine (THN)** — the aryl-attached NH with a single H matches `[NH2,NH]`, and THN is the standard Arg-mimic head for this integrin class, so the filter was banning the class's core pharmacophore instead of reactive metabolites. **(b) §7 was in a state that would have invalidated it** — with 5 of the 6 positive-panel molecules zeroed on **every component** (a filter blankets all of them), the reference distribution itself collapses, and the then-current §9.5 wording ("the generated set beats the negative panel on total score and matches the positive panel") would pass no matter how bad the generated molecules were. `results/` was empty, so no published figure was contaminated. (§9.5 was separately revised — for a different reason, the circularity of total-score comparison — to non-circular axis comparisons.)

The reactive-metabolite risk around anilines is real (this series has that history [111]), but what is needed is a pattern limited to **primary aromatic amines**. `[NX3;H2;!$(N[!#6]);!$(NC=O);!$(NS(=O)=O)][c]` and `[NX3;H2][c]` give **identical** results on every test — `H2` already excludes N-acyl, N-sulfonyl and N-heteroatom substitution (those leave ≤ 1 H). The three exclusions are redundant, so the simpler pattern is adopted. 2-aminopyridine/2-aminopyrimidine both match `[NX3;H2][c]`; excluding heterocycles would need `[NX3;H2][c;!$(c~n);!$(c~o);!$(c~s)]`, but the difference is 12 vs 9 prior samples (0.6 pp) and both reference sets score 0 either way, so the stricter, simpler pattern is used.

The rest of the alert set is unremarkable — union match rates are `actives_core` 0/29, `actives_extended` 1/190, `similarity_refs` 0/6, `benchmark_panel` 0/6, prior 19/488 (3.9%). One gap this pass closes: this document had described "PAINS/reactive/aniline" as if PAINS were one of the RL config's patterns, but PAINS cannot be expressed in the TOML at all (no PAINS-capable component exists in `reinvent_plugins`), so the RL run never applied it — only `scripts/objective.py` and the §8 triage do. Measured divergence on the prior sample: the eight SMARTS alone catch 31/488, the full `objective.py` filter (eight SMARTS ∪ PAINS) catches 45/488, and PAINS alone catches 15/488 — free on `actives_core` (0/29), `similarity_refs` (0/6) and `benchmark_panel` (0/6), but `actives_extended` (190) loses 3: **CHEMBL244434, CHEMBL244013** (both via PAINS `mannich_A(296)`) and **CHEMBL4756602** (via `[Nr0][Nr0]`). These three are the real cost that spec success criterion 2 and blueprint §10's D4 checkpoint ("confirm zero reference compounds score zero") need to report honestly.

### 5.5 Monitoring — diagnostics, not reward (every 100 steps)

With similarity removed from the objective, the old monitoring expectation ("similarity falls in Stage 2 = novelty signal") is discarded and replaced with:

- NN-Tanimoto distribution against `actives_extended` — **after tautomer normalization** (§8.0)
- carboxylic acid pass rate
- number of unique Murcko scaffolds
- alert match rate
- SAScore distribution (whether the guard rail has actually begun to engage)

None of these feed the reward — this keeps the novelty claim measurable without paying for it in the objective. **Intervention criterion:** intervene only if NN-Tanimoto collapses below ~0.2 **and, at the same time,** the carboxylic acid pass rate drops. Neither alone triggers intervention, and do not "fix" it by raising a similarity weight — that component no longer exists.

---

## 6. Step 3 — Sampling (D6)

- `run_type = "sampling"` from **the final RL agent** (and, for comparison, from the D3-selected focused prior)
- 20,000–50,000 SMILES, unique molecules only
- Output: `library.smi` → dedupe → canonicalize → **tautomer normalization (§8.0)** → property/novelty distributions

## 7. Step 4 — Scoring the benchmark panel (D6, after sampling)

- `run_type = "scoring"` with the **identical scoring function from §5.1** applied to the frozen benchmark panel
- **Precondition:** `CustomAlerts` must be running the replaced aniline pattern from §5.4 (`[NX3;H2][c]`). With the old `[NH2,NH][c]`, 5 of the 6 positive-panel molecules score 0, and §9.5 would false-pass — do not run §7 without this check
- Output: **per-component** scores for cpd 25, PLN-1474, bexotegrast, CWHM-12, GLPG0187 (total score is also recorded but not used as a basis for comparison — §3.4) — the reference distribution behind every comparison shown on the poster

---

## 8. Post-hoc triage funnel (D7–D8)

Order matters — cheap to expensive:

0. **Tautomer normalization — a precondition for every fingerprint comparison.** Apply `rdMolStandardize.TautomerEnumerator().Canonicalize()` to every SMILES. It must be applied **identically to the reference set and to what is being measured**.

   `scripts/curate_actives.py:222-239`'s `standardize()` only ran `Cleanup` + `LargestFragmentChooser` + `Uncharger` and never normalized tautomers. That opens a false-positive path in §8.3 and §9.3. The exposure was measured (fraction whose representation changes on normalization, and that molecule's Tanimoto to *its own* other tautomer):

   | set | representation changed | self-Tanimoto median | **below 0.680** |
   |---|---|---|---|
   | `actives_core` (29) | 0/29 | — | 0 |
   | `benchmark_panel` (6) | 0/6 | — | 0 |
   | `actives_extended` (190) | 14/190 (7.4%) | 0.736 | 1 |
   | **TL-A prior samples (488)** | **43/488 (8.8%)** | 0.621 (min 0.379) | **29** |

   That is, **29 of 488 generated molecules (5.9%) pass §8.3's "scaffold-novel" test against their own other tautomer.** PLN-1474 is a case in point — written as the non-aromatic amidine tautomer of its tetrahydro-1,8-naphthyridine instead of the aromatic form, it keeps the same molecular formula (`C24H37N3O4`) but QED 0.4332 vs 0.4619, aromatic ring count 0 vs 1, and **an NN-Tanimoto to its own other tautomer of 0.458**. A known clinical compound can reproduce itself exactly and still pass as a novel molecule.

   The asymmetry is the core of the problem: ChEMBL actives arrive in a consistent aromatic form (core 0%, bench 0%), but generated molecules come out however the model emits them — the reference side is normalized and only the measured side wobbles. The aniline alert is tautomer-dependent for the same reason (the non-aromatic form does not match `[NH2,NH][c]`), which had opened a SMILES-level reward-hacking path where RL could dodge the filter by changing representation. §5.4's `[NX3;H2][c]` closes that path since neither form matches it, but putting normalization upstream is the root fix.

   **Applied to:** generated/prior samples, `data/actives_extended.smi`, `data/actives_core.smi`, the anti-target counter-screen references (§8.4), and a recomputed `data/novelty_band.json` — **done, table in §8.3 below** (extended p25 0.710 → 0.680).
1. **Validity / dedupe / carboxylate re-check / CustomAlerts re-check** (with §5.4's replaced pattern). The carboxylate is a gate in RL, but its floor is not literally zero (6.31e-04), so it is re-applied here as a **hard cut**.
2. **Property envelope:** use the same TPSA window as RL, and view the axes removed from the objective through a **wide window** here — logP ≤ 5 (the hard-filter precedent of prior work [113]), MW 250–550, RotB. Narrowing the objective and widening triage is the point of this redesign (§5.1).
3. **Novelty — calibrated against the actives themselves, not a borrowed constant.** Nearest-neighbour Tanimoto against **`data/actives_extended.smi` (190 molecules, 103 Murcko scaffolds — every ChEMBL αvβ1 active ≤ 1 µM)**, using Morgan radius 3, feature invariants, **count** fingerprints (after §8.0's tautomer normalization).

   The earlier criterion, NN-Tanimoto < 0.4, is **withdrawn**. It is a convention for binary ECFP4, and this project does not use binary ECFP4 — the same pair scores far higher on a feature-based count fingerprint. Measured against its own reference data the rule does not hold: **83% of the core actives and 99% of the extended ones sit at ≥ 0.4 from an active of a different Murcko scaffold**, so this rule would call almost the entire public αvβ1 series "not novel" relative to itself.

   The replacement is the distance the data itself defines — how far apart two known actives are when they belong to different scaffolds (`data/novelty_band.json`, recomputed by `curate_actives.py`):

   | reference set | tautomer | p25 | median | p75 | p90 |
   |---|---|---|---|---|---|
   | `actives_core` (29, 19 scaffolds) | as-is | 0.552 | 0.676 | 0.732 | 0.745 |
   | `actives_core` (29, 19 scaffolds) | **normalized** | 0.552 | 0.676 | 0.732 | 0.745 |
   | `actives_extended` (190, 103 scaffolds) | as-is | 0.710 | 0.775 | 0.831 | 0.922 |
   | `actives_extended` (190, 103 scaffolds) | **normalized — adopted** | **0.680** | 0.765 | 0.831 | 0.922 |

   Recomputed with §8.0's tautomer normalization applied (reproducing `curate_actives.py:627-641`'s logic exactly — running it without normalization exactly reproduces the current `data/novelty_band.json` figures of 0.552 / 0.710, which validates the reproduction). `actives_core` is unchanged, since none of its tautomers change; `actives_extended` has 14 that change, and its p25 drops from **0.710 to 0.680** — a slightly **stricter** band.

   **But using the distance threshold as a gate is also withdrawn (2026-09-26).** Measurement shows it is broken: applying `NN-Tanimoto < 0.680` to the 63 prior samples whose Murcko scaffold is **identical** to one of the 103 public actives lets **39 (62%) pass as "scaffold-novel."** That calls a molecule sitting on a published scaffold a novel chemotype. Distance is continuous and scaffold identity is discrete; a continuous cutoff cannot answer the discrete question ("has this scaffold been proposed before").

   **Adopted criterion — Murcko scaffold novelty (binary).** A generated molecule is scaffold-novel if its Murcko scaffold is **not** in the 106-member set in `data/known_scaffolds.smi`. No fingerprint dependence, no threshold, one sentence. And since the objective's diversity filter is already `IdenticalMurckoScaffold`, the reward mechanism and the success criterion now use **the same unit**.

   **The reference set is the union of five curated reference files, not `actives_extended` alone** (`actives_core`, `actives_core_B`, `actives_extended`, `benchmark_panel`, `similarity_refs` → 106, produced by `scripts/known_scaffolds.py`). `actives_extended` alone (103) **is missing the scaffolds of PLN-1474, bexotegrast and A1AFA** — because that file is defined as "ChEMBL αvβ1 actives ≤ 1 µM": PLN-1474's structure comes from AdisInsight, not a ChEMBL activity record [88], and A1AFA sits at pIC50 5.30, below the potency cut. Using the standalone set means **a generated molecule that reproduces PLN-1474's scaffold would pass as "scaffold-novel"** — despite being the one clinical αvβ1-selective compound, public since August 2023. The scaffolds of the public compounds the poster compares against must be in the reference set. `tests/test_known_scaffolds.py` asserts, for each of the three compounds, "absent from the standalone set, present in the union."

   - 420 of 488 (86.1%) TL-A prior samples pass — a loose filter. Bite comes from §8.5 docking geometry and §9.2 permeability.
   - **Exact-structure mismatch alone is not enough.** TL-A at 200 epochs memorizes (§4: max NN-Tanimoto = 1.000, some samples reproduce a training active verbatim). Adding one methyl group makes a "structure not seen before," so this criterion alone cannot answer "isn't this just cpd 25 with a methyl added?" Murcko does answer it.
   - **NN-Tanimoto is kept as a reported number, not a gate.** Of the 420 that pass on Murcko, **5 sit at NN ≥ 0.680 (max 0.738)**, so analog objections are possible. Reporting each lead's NN value alongside **the name of its nearest known active** lets a reader judge without a threshold. The band table above is retained only as context for reading that value — `data/novelty_band.json` is not a gate.
   - **Tautomer normalization bites directly here (§8.0).** Verified: PLN-1474's two tautomers give **different Murcko SMILES** without normalization (`...C1=CC=C2CCCN=C2N1` vs `...c1ccc2c(n1)NCCC2`), and identical Murcko SMILES with it. Without normalization, the novelty claim is breakable by a representation change.
4. **Selectivity — reported as an observation, not claimed (revised 2026-09-27).** The former clause, flagging Tanimoto ≥ 0.5 to αvβ3/α5β1 actives, is **withdrawn**: the counter-screen set §3.2 specifies was never produced, so the clause is unexecutable, and §3.2 itself notes that similarity ≠ activity. Instead, **measure and report** these three on the poses that clear §8.5.

   (i) **β1-Leu225 side-chain contact** — is a ligand hydrophobic atom within 4.5 Å of the Leu225 side chain (CB/CG/CD1/CD2)? Aligning the βI domains of ITGB1 (P05556, 20-residue signal peptide removed) and ITGB3 (P05106) shows that of the five residues Sabat names as the source of selectivity, **only Leu225 differs in β3**, which carries **Arg** there — a hydrophobic wall replaced by a charged side chain. Tyr133, Pro186, Cys187, Asn224 and Asp226 are all conserved. This explains why GLPG0187 and CWHM-12 engage the Leu225 **backbone** through the same 3-aminopropionyl framework and are nonetheless pan: the pocket needs the Leu **side chain**.

   (ii) **The character of the αv-Asp218 contact** — within 4.0 Å of the Asp218 carboxylate oxygens, is there (a) a neutral hydrogen-bond donor, or (b) a basic nitrogen (an Arg-mimetic head)? (b) is associated with pan-αv activity in the literature: the 1,8-naphthyridine of the αvβ6 clinical candidate GSK3008348 forms a **salt bridge** with Asp218 [114], and Sabat's THN incorporation produced pan-αv inhibition [111].

   (iii) **αv-Tyr178 π-stack** — is a ligand aromatic ring centroid within 5.5 Å of the Tyr178 ring centroid, with an inter-plane angle either under 30° (parallel) or 60–90° (T-shaped)? Measured in 8W30: closest atom 3.70 Å, centroid 6.21 Å, ring angle 74.6°.

   **These are features reported as correlated with selectivity; they are not evidence of selectivity.** αvβ1 selectivity is determined by **both** the β subunit (against αvβ3/αvβ5/αvβ6/αvβ8) and the α subunit (against α5β1/α4β1/α8β1) [114], and this work addresses none of the latter — the same review names α5's Trp157, Gln221 and Ser224 as candidates and we do not analyse them. Nor has the contact said to confer selectivity ever been observed crystallographically: no αvβ1 structure existed as of 2020 [114], and the ligand in the first one, 8W30, has pIC50 5.30 and sits 7.00 Å from Asp218. **Measured cross-integrin IC50s are the only basis on which selectivity can be judged, and that is future work.**
5. **Docking — a geometric filter, not an affinity ranking:**
   - Receptor: 8W30 chains A+B; strip ligand TR01225179 and waters (record the water decision; HOH2107 is the candidate to retain in a sensitivity run)
   - Tool: smina / AutoDock Vina; ~20 Å box on the ligand centroid
   - **Known limitation:** Vina-class scoring functions do not model metal coordination — the single most important interaction (carboxylate → MIDAS Ca²⁺) is invisible to the scorer. Therefore:
     - **(a) Validation:** redock TR01225179. Require carboxylate-O → Ca501 ≤ 3.2 Å in the top pose (ideally also heavy-atom RMSD < 2 Å against the crystal pose). **If validation fails, do not rank anything by docking** — fall back to similarity/QSAR-based triage and state that on the poster.
     - **(b) Post-hoc filter:** accept only poses with carboxylate O → Ca501 ≤ 3.2 Å **and** an H-bond donor within 3.5 Å of β1-Asn224 backbone O. Docking score is used only to break ties.

   **Validation result — EXECUTED, PASSED (2026-09-27).** In a conda `docking` environment (smina 2020.12.10, openbabel), TR01225179 was redocked into 8W30 chain A+B with the ligand, waters and glycans removed and **all six Ca²⁺ retained**: `--autobox_ligand` plus `--autobox_add 6 --exhaustiveness 16 --num_modes 20 --seed 42`.

   | pose | affinity | O→Ca501 | N→Asn224 O | RMSD |
   |---|---|---|---|---|
   | **1 (top)** | −6.91 | **2.72 Å** | **2.89 Å** | **0.63 Å** |
   | 2 | −6.83 | 2.31 | 3.27 | 1.75 |
   | 3 | −6.74 | 2.47 | 3.10 | 1.83 |
   | crystal | — | 2.62 | 2.63 | — |

   The top three poses all fall within 2 Å RMSD and the affinity ranking tracks the RMSD ranking. **Pose generation succeeded despite the missing metal term** — retaining Ca²⁺ in the receptor makes MIDAS a narrow polar concavity in which steric exclusion alone fixes where a carboxylate can sit. What the missing term still costs is **affinity ranking**, which is precisely why score is used only to break ties.

   > **A caution about the RMSD.** Computed index-wise, assuming the atom order is preserved, the RMSD is 6.13 Å — smina/obabel reorder atoms relative to the input and the molecule carries symmetry (phenyl, dichlorophenyl). Measured with `rdMolAlign.CalcRMS`, which accounts for atom mapping and symmetry without superposing coordinates, it is 0.63 Å. **A naive RMSD turns a passed validation into an apparent failure.**

   **Why the thresholds are these values.**

   *carboxylate-O → Ca501 ≤ 3.2 Å* — First, **the first coordination shell of this structure ends at that boundary.** Ranking the O/N atoms coordinating Ca501 by distance: β1-Glu229 OE2 at 2.39 Å, β1-Ser132 OG at 2.45 Å, β1-Ser134 OG at 2.50 Å, **the ligand OXT at 2.62 Å** — then a **gap** — β1-Asp259 OD1 at 3.30 Å and β1-Asp130 OD2 at 3.85 Å. The void between 2.62 and 3.30 separates the first shell from the second, and 3.2 Å sits inside it. The cut-off is therefore not a convention but **a boundary this structure defines**. Second, Ca²⁺–O coordination normally runs 2.3–2.6 Å, so 3.2 Å is a generous ceiling that absorbs the coordinate error implied by a 2.45 Å structure and a rigid-receptor assumption. Third, it leaves 0.58 Å of headroom over the crystallographic 2.62 Å, and the redocked top pose came in at 2.72 Å (+0.10) — neither too tight nor too loose.

   **The contact is judged monodentate.** Of the ligand's two carboxylate oxygens only `OXT` is close at 2.62 Å; the other (`O`) sits at **4.54 Å**. The condition is therefore **"either oxygen within 3.2 Å"** — requiring both would reject the crystal structure itself. **Ca501 is named** because Ca502 is 6.44 Å from the ligand and takes no part.

   *H-bond donor → β1-Asn224 backbone O ≤ 3.5 Å* — The crystallographic value is **2.63 Å**. Compared against the same oxygen's other partners: Leu225 N at 2.25 Å (the backbone chain), the ligand N at 2.63 Å, Asn224 N at 2.65 Å — all in the 2.2–2.7 Å band. 3.5 Å is the standard generous cut-off for an N···O hydrogen bond and leaves 0.85 Å above that band. Asn224 exists only in chain B, as does the ligand, so this is an intra-chain contact.

   **This contact confers binding, not selectivity.** Sabat 2024 states that the MIDAS and Asn224 contacts "substantiated the observed αvβ1 affinity but not necessarily isoform preference", and a β1/β3 sequence alignment confirms the structural reason: **Asn224 is conserved in β3** (§8.4).

   *Pose generation parameters* — `--exhaustiveness 16 --num_modes 20`, `--seed 42`. With no metal term there is no reason to expect the correct geometry to rank first, so **generating many poses and filtering them geometrically** is the design of (b). The redocking put the top three within 2 Å, so 20 modes suffice.

   *Water treatment* — the default receptor strips all waters. A second receptor retaining **HOH A2107** alone (4.68 Å from the ligand, at the subunit interface) is built for sensitivity analysis, because Sabat 2024 reports obtaining selectivity by engaging that **interface crystallographic water** instead of αv-Asp218. Report the difference between the two receptors.
6. **ADMET:** apply ADMET-AI (or equivalent) to survivors — permeability proxy, solubility, microsomal stability, hERG, CYPs. The decision baseline is cpd 25's *measured* liabilities (MDCK < 0.1×10⁻⁶ cm/s, oral F 1.3% [111]). Predictions must clear that bar directionally.
7. **Final selection:** roughly 10–20 leads satisfying: geometric filter pass, predicted permeability better than cpd 25, **Murcko scaffold absent from §8.3's 106-member set (`data/known_scaffolds.smi`)**, clean alerts, carboxylate present. QED and NN-Tanimoto (with the name of the nearest active) are reported alongside as **metrics, not gates**.

---

## 9. Pre-registered success criteria (freeze before D6 sampling)

Through the **identical** funnel, the proposal succeeds if at least one generated molecule satisfies:
1. passes the geometric docking filter (MIDAS + Asn224 contacts preserved),
2. beats cpd 25 on the predicted permeability axes,
3. **Murcko scaffold absent from the 106-member set in `data/known_scaffolds.smi`** (§8.3 — the union of five curated reference files; `actives_extended` alone, 103, is missing PLN-1474, bexotegrast and A1AFA). Both sides are judged after §8.0's tautomer normalization — without it, the same molecule can yield a different Murcko SMILES. NN-Tanimoto and the nearest known active's name are reported alongside as **metrics, not a gate**,
4. carries no CustomAlerts flags (§5.4's replaced aniline pattern) — the **counter-screen clause is withdrawn** (§8.4): the set was never produced and the mechanism is weak. **Selectivity is not part of these criteria**,
5. and, scored with the identical funnel and tools as the positive panel (PLN-1474, bexotegrast, CWHM-12, GLPG0187, cpd 25), is comparable on **non-circular axes** — passes the docking geometric filter on equal footing, beats cpd 25 on ADMET-AI permeability, and has a novel Murcko scaffold. **Total score is not used for comparison** (it is the value of an objective function we designed, so it is circular — §3.4, §5.4). The negative-panel clause is dropped (§3.4).

**Falsification clause:** if nothing satisfies 1–4, report exactly that. It is evidence the carboxylate–permeability tension may be **target-intrinsic** — which is itself a finding, and the honest answer to the strongest attack on this project.

---

## 10. Day-by-day schedule

| Day | Work | Checkpoint / fallback |
|---|---|---|
| D1 | **Done.** Environment fixed against a live K-BDS probe (§11); `setup_kbds.sh` + `slurm/00_smoke.sbatch` ready; priors are the v4.5.11 in-repo bundle, checksum-verified. **§3.1 curation also done** — `scripts/curate_actives.py`, outputs in `data/`, provenance in `data/CURATION_LOG.md` | sets frozen: core 29 / core_B 200 / extended 197 / refs 6 / panel 6 |
| D2 | Counter-screen sets (**target IDs verified, §3.2**) | all data frozen. **Negative panel dropped (§3.4)** |
| D3 | TL-A and TL-B runs + diagnostics on all three arms | one arm selected on the D3 checks, **chemotype drift decisive**; if TL-B raises the Arg-mimic fraction or TPSA, it is out regardless of its other numbers |
| D4 | **Preceding work:** add tautomer normalization to the comparison layer (`scripts/normalize.py`) → recompute `novelty_band.json` (§8.0); verify the `CustomAlerts` aniline pattern replacement (§5.4). **`curate_actives.standardize()` is deliberately left untouched** — its output is written to `data/actives_core.smi`, the TL-A training input and RL inception seed, and adding tautomer canonicalization there would invalidate the completed D3 run; normalization lives only in the comparison layer (`scripts/normalize.py:8-9`, `scripts/curate_actives.py:243-246`). Then RL single stage, steps 0–300 | start only after confirming no reference-set molecule scores 0 **beyond the one recorded in §5.4** — the eight SMARTS the RL run applies zero `CHEMBL4756602` in `actives_extended` via `[Nr0][Nr0]` (acyclic N–N, the hydrazine class). That pattern is a legitimate reactive alert and 1 of 190 is not the aniline catastrophe of 164/190, so it is accepted rather than hidden. PAINS is absent from the RL config and applies only in `objective.py` / the §8 triage, where the count is 3 (`CHEMBL244434` and `CHEMBL244013` via `mannich_A(296)`). Log §5.5 diagnostics every 100 steps |
| D5 | RL single stage, steps 300–600 | final agent; watch for collapse onto known series (NN-Tanimoto rising above 0.6 → raise DF `minscore` 0.4 → 0.5, restart from an earlier checkpoint if needed). **No similarity-weight adjustment is possible** — that component is not in the objective (§5.2) |
| D6 | Sampling 20–50k; benchmark panel scoring | `library.smi` + reference score distributions |
| D7 | Docking setup + TR01225179 redocking validation | validated protocol **or** documented fallback to similarity/QSAR-only triage |
| D8 | Dock + geometric filter + ADMET + counter-screens | triage table |
| D9 | Benchmark comparison, lead selection, results table + figures | top 10–20 leads |
| D10 | Buffer: poster assembly (pocket figures already done), 7-min talk rehearsal, attack-surface prep | — |

**Global fallbacks:**
- **No GPU on the day:** `batch_size` 32–64, halve `max_steps`, sample 10k — everything still completes.
- **TL underperforms the prior:** that is arm TL-C, already in the D3 comparison — carry the prior forward with a larger inception memory and say so. (The original "< 30 actives → skip TL" rule is superseded: the core is 29 because the public non-RGD chemotype is that small, and whether that is too small is now measured at D3 rather than assumed.)
- **Docking validation fails:** similarity/QSAR-only triage; limitation stated explicitly.

---

## 11. Compute notes — K-BDS (KISTI), executed

Runs on the KISTI K-BDS cluster. The stack was chosen from a live probe of the login node (`scripts/kbds_probe.sh`), not from assumption; the full reasoning is in `TRANSFER.md`.

| | |
|---|---|
| Login node | `kbds.kisti.re.kr` (OTP) — **the only SSH/scp path**; `kbds-dm.kisti.re.kr` is FTP-only (port 21, port 22 closed) |
| OS, scheduler | CentOS 7.9.2009 (**glibc 2.17**), Slurm 21.08.5, exclusive nodes, 120 h wall clock |
| GPU | A100 40 GB, **driver 470.57.02** |
| Queues | `1gpu`, `2gpu` (A100 40 G); `4gpu`, `8gpu` (A100 80 G); `debug-1gpu` (8 h cap) |
| Environment | conda env **`reinvent`**, Python 3.11 |

**Three measurements fixed the whole stack:**

1. **glibc 2.17** — REINVENT4 `main` pins `torch==2.12.0`, which ships `manylinux_2_28` wheels only and cannot run here. **REINVENT4 is pinned at `v4.5.11`**, the newest release whose torch pin (2.5.1) still has a glibc 2.17 wheel — and which ships the priors in-repo. The same constraint rules out PyPI `rdkit` (also `manylinux_2_28`), so **rdkit comes from conda-forge** (`__glibc >=2.17`).
2. **Driver 470.57.02** — CUDA 12 needs driver ≥ 525, so the cluster's `compilers/cuda/12.4` and `12.8` modules are unusable for this. CUDA 11.8 needs only ≥ 450. The single deviation from v4.5.11's pin is **`torch==2.5.1+cu124` → `2.5.1+cu118`** — same torch, different CUDA build, exactly the substitution its `pyproject.toml` comment anticipates. **No CUDA module is loaded**: the pip wheel carries its own runtime and a loaded toolkit would shadow it.
3. **Outbound internet is open** (pypi, conda-forge, zenodo, github, rcsb all 200, no proxy) — so packages install directly on K-BDS and no `conda-pack` bundle is needed. Apptainer and Singularity are both broken on the login node (`libsubid.so.3` missing), so the container route was unavailable.

**Three declared dependencies are dropped**, each backing one scoring component that REINVENT4 simply skips when it cannot import it: `openEye-toolkits` (licensed, ROCS only), `chemprop` (§3.3 rules ChemProp2 a no-go, and it pulls in scipy/scikit-learn whose current releases are `manylinux_2_28` only) and `descriptastorus` (a hidden chemprop dependency no REINVENT4 module imports). Hence `pip install --no-deps` for REINVENT4 itself; `setup_kbds.sh` then asserts that all nine components the RL objective names did register.

**v4.5.11 under-declares `torchvision` and `scipy`** — both are imported at module level on the path from `reinvent.Reinvent` (via the sampling reports) yet neither is in its `pyproject.toml`. With `--no-deps` that becomes a crash on first run, so `scripts/check_imports.py` walks the module-level import graph and `setup_kbds.sh` runs it before installing. torchvision is pinned to `0.20.1+cu118` (0.20.1 declares `torch==2.5.1` exactly, and it needs the same CUDA index); scipy is capped below 1.17, which went `manylinux_2_28`.

Pip installs binary-only (`--only-binary=:all:`). Without it, a package whose newest release ships only `manylinux_2_28` wheels makes pip fall back to the sdist, and the build dies several dependencies deep blaming the wrong package — the first attempt failed on *"NumPy requires GCC >= 9.3"* when the real cause was pandas 2.3.3 dropping its `manylinux_2_17` wheel. Every bound in `env/requirements-pip.txt` was checked against PyPI's actual cp311 wheel tags.

**Commands**

- Transfer: see `TRANSFER.md` (2.3 GB of the 45 GB tree; the rest is finished upstream work)
- Build once on the login node: `bash scripts/setup_kbds.sh`
- Smoke test on a GPU: `mkdir -p logs && sbatch slurm/00_smoke.sbatch` — runs one TL epoch on the real TL-A split plus 200-molecule sampling, and fails if validity drops below 80%
- Run: `reinvent -l run.log config.toml`
- Data curation: `python3 scripts/curate_actives.py` (RDKit only; ChEMBL/PubChem/RCSB responses cached under `data/.cache`, so the run is reproducible offline and the exact ChEMBL release is recorded in the log).
- GPU: RL `batch_size = 128`; submit to `1gpu` (one A100 is enough, and it is the cheapest queue at 1 node-hour per hour). Every GPU partition was fully allocated when probed — expect to queue.

---

## 12. What this protocol deliberately does NOT do

- **No in-RL docking** (DockStream/Maize): too slow for a 10-day project and metal-coordination-blind anyway; docking is a post-hoc geometric filter.
- **No LibInvent around the azabenzimidazolone core:** that would presuppose the chemotype the project is trying to escape.
- **No predicted-affinity claims:** Vina scores are not affinities, especially at a metal site. Claims are framed as geometry + properties + novelty.
- **No wet-lab promise:** the deliverable is a prioritized, falsifiable in silico lead set with pre-registered criteria.

---

## References

- [39] MolecularAI — REINVENT4 GitHub repository
- [40] REINVENT4, J. Cheminformatics (2024)
- [88] AdisInsight — PLN-1474 development status (structure public Aug 2023; no development as of May 2025)
- [98] Lancaster et al., INTEGRIS-IPF, Am. J. Respir. Crit. Care Med. (2024) — bexotegrast
- [103] Henderson et al., Nat. Med. (2013) — HSC αv deletion; CWHM-12
- [111] Sabat et al., J. Med. Chem. (2024) — Design and Discovery of a Potent and Selective Inhibitor of Integrin αvβ1
- [112] PDB 8W30 — αvβ1 headpiece + TR01225179
- [113] Qie, Wang, Li et al., *Molecular Diversity* (2026), doi 10.1007/s11030-026-11625-z — REINVENT4 stage-wise RL for EGFR inhibitor design + experimental validation. The **prior RL protocol** this project references
- [114] Zheng Y, Leftheris K, *J. Med. Chem.* **2020**, 63, 5675–5696, doi 10.1021/acs.jmedchem.9b01869 — *Insights into Protein–Ligand Interactions in Integrin Complexes: Advances in Structure Determinations*. Source for: selectivity among αvβx being determined by the β subunit because αv is shared; the absence of any αvβ1 structure as of 2020; α5's Trp157/Gln221/Ser224; the Asp218 salt bridge of GSK3008348; and the 2,6-dichlorophenyl motif being a general SDL-proximal feature across integrin inhibitors.
