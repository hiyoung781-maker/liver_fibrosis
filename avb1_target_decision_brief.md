# Target Decision Brief — Integrin αvβ1 for Liver Fibrosis
### Evidence-based target prioritization and argument architecture for a REINVENT4-based lead-proposal project

**Project:** In silico design of novel αvβ1 integrin inhibitors as direct antifibrotic leads for liver fibrosis (MASH-centric).
**Status of this document:** decision record + argument scaffold for a 7-minute presentation and A0 poster. Every factual claim carries a numbered reference (list at the end).

---

## 1. The design hypothesis (the project's single sentence)

> **The αvβ1 selectivity pharmacophore mapped by prior work — a MIDAS-metal carboxylate anchor, a β1-Asn224 H-bond, an αv-Tyr178 aromatic contact, and a β1 hydrophobic subpocket — can be carried by novel scaffolds with materially better passive permeability and oral-likeness than the published tool compound; i.e., the tool-compound ceiling is a property of the explored series, not of the target.**

The generative campaign is the *test* of this hypothesis. AI (REINVENT4) is the instrument, not the subject.

---

## 2. Problem statement

- **Fibrosis stage — not inflammation or steatosis — is the dominant predictor of liver-related death** in NAFLD/MASH: F4 carries HR ≈ 10.9 for death or transplant [52]; a Swedish biopsy cohort found fibrosis stage, but not NASH, predicts mortality and severe liver disease (HR up to ≈ 105 for F4) [53]; the prospective NASH CRN cohort confirmed F3–F4 drive liver complications and death [55].
- **MASH is common** — ~6% of US adults (~14.9 M people) [78].
- **The two approved drugs are metabolic and indirect.** Resmetirom (THR-β agonist, approved March 2024 for noncirrhotic F2–F3 MASH) improved fibrosis by ≥1 stage in only 24–26% vs 14% on placebo [33, 36, 38]. Semaglutide (GLP-1 RA, approved August 2025, same population) improved fibrosis in 36.8% vs 22.4% [78, 79, 80]. Neither is approved for cirrhosis; neither targets the fibrotic machinery directly.
- **Direct antifibrotic development is a graveyard.** Selonsertib (ASK1) failed two Phase 3 trials [1, 6]; cenicriviroc (CCR2/5) failed Phase 3 AURORA [2]; emricasan (pan-caspase) failed [5]; simtuzumab (anti-LOXL2) and others joined the list [8]. Roughly 70% of NASH programs fail the Phase 2→3 transition and ~42% fail Phase 3→approval [7].
- **Lesson the field learned:** targets with human genetic support are >2× more likely to win approval [110, 107] — which is why genetics anchored our prioritization matrix, and why its absence for αvβ1 is disclosed as a weakness (Section 7), not hidden.

**Gap:** no approved therapy acts directly on the fibrogenic cell (the hepatic stellate cell, HSC) or its matrix output. A direct antifibrotic is the rational add-on to the metabolic standard of care.

---

## 3. Prioritization framework and matrix

Criteria (each scored 0–2; evidence required for a nonzero score):

| # | Criterion | Why it matters for this project |
|---|-----------|--------------------------------|
| C1 | Directness (HSC-intrinsic or matrix-directed) | User-imposed hard filter; defines "direct antifibrotic" |
| C2 | Human validation (genetics or clinical PoC) | Genetics doubles success odds [110, 107] |
| C3 | Small-molecule tractability + known chemotypes | REINVENT4 needs priors and a scoring anchor |
| C4 | Structural data | Docking-based triage needs a pocket |
| C5 | Competitive whitespace | A proposal needs an unfilled gap |
| C6 | Story coherence in 7 min | Festival constraint |

| Target | C1 | C2 | C3 | C4 | C5 | C6 | Verdict |
|--------|----|----|----|----|----|----|---------|
| **Integrin αvβ1** | ✓✓ HSC-intrinsic [10, 103] | ✓ in-vivo genetics + clinical class PoC [98, 103]; no germline GWAS | ✓✓ public SAR series + clinical compounds [85, 111] | ✓✓ PDB 8W30 [112] | ✓✓ liver program orphaned [86, 88] | ✓✓ | **RECOMMENDED** |
| HSD17B13 | ✗ hepatocyte enzyme, indirect [30, 31] | ✓✓ LOF genetics [25, 27] | ✓✓ inhibitors + structures [21, 45] | ✓✓ | ✓ RNAi-only clinic [57, 61] | ✓ | Rejected: fails directness; phenocopy risk (catalytic inhibition ≠ protein loss) [16, 24]; possible harm signal in cirrhosis [28] |
| PNPLA3 | ✗ metabolic | ✓✓ strongest genetics [66] | ✗ gain-of-function neomorph; ASO/siRNA-only [67, 70, 71] | ✗ | ✓ | ✓ | Rejected: no small-molecule strategy |
| LOXL2/LOX | ✓✓ matrix-directed | ✗ simtuzumab failed in liver fibrosis [8] | ✓ | ✓ | ✗ | ✗ | Rejected: target failed in this exact indication |
| ALK5 (TGFBR1) | ✓✓ | ✗ systemic TGF-β blockade toxicity | ✓✓ | ✓✓ | ✗ | ✓ | Rejected: the cautionary tale αvβ1 solves |
| MARC1 | ✗ metabolic | ✓ genetics [75] | ✗ no chemotypes | ✗ | ✓✓ | ✗ | Rejected: human/mouse divergence [74] |
| α8β1/α11β1 | ✓✓ HSC-specific [9] | ✗ | ✗ no inhibitors | ✗ | ✓✓ | ✓ | Rejected: no generative priors |
| NOX1/4, TEAD/YAP | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | Rejected: thinner validation, no liver PoC |
| THR-β / GLP-1 | ✗ | ✓✓ | ✓✓ | ✓✓ | ✗ approved [33, 78] | ✓ | Rejected: no whitespace |

The matrix is poster-ready and demonstrates that step 2 (evidence-based prioritization) was a real filter — including rejecting our own first candidate (HSD17B13) when the directness criterion was applied strictly.

---

## 4. Why αvβ1: the evidence chain

1. **Causal, cell-intrinsic biology.** Deleting αv integrins in PDGFRβ+ pericyte-lineage cells (HSCs) protects mice from CCl₄-induced liver fibrosis; the small-molecule αvβ1 inhibitor CWHM-12 reduces fibrosis; the pathway is core across solid organs (Henderson, *Nat Med* 2013) [103, 14].
2. **Human HSC mechanism.** αvβ1 is the **most abundant αv integrin on human HSCs**, and its blockade suppresses procollagen-1 production through a **noncanonical pathway independent of SMAD phosphorylation or TGF-β activation** [10, 111] — i.e., αvβ1 controls fibrogenesis by more than latent-TGF-β activation (relevant to the redundancy counterargument, Section 7). RGD-integrin blockade also induces HSC senescence (Kitsugi 2023, cited in [111]). Reviews: [12, 9].
3. **Small-molecule proof in a liver-fibrosis model.** Sabat et al. (*J Med Chem* 2024) developed a selective non-RGD αvβ1 inhibitor (azabenzimidazolone **25**: cellular αvβ1 pIC₅₀ 6.37; >50-fold selective over α4β1/α5β1/α8β1/αvβ3 and αvβ6/αvβ8) and showed **dose-dependent antifibrotic efficacy in the rat CDHFD model** [111].
4. **Structural basis.** Co-crystal structure of the human αvβ1 headpiece with inhibitor TR01225179 (PDB 8W30) [112] — analyzed in Section 6.
5. **Clinical validation of the class.** PLN-1474 (oral, αvβ1-selective) completed a clean Phase 1 (84 volunteers, no dose-limiting toxicity) with preclinical liver-fibrosis efficacy [85]. Bexotegrast (oral dual αvβ6/αvβ1) showed antifibrotic signals (FVC, quantitative lung fibrosis imaging, biomarkers) and good tolerability in IPF Phase 2a INTEGRIS [98, 99, 101].
6. **The whitespace.** Novartis terminated PLN-1474 in 2023 while divesting its *entire* NASH portfolio (tropifexor and the FGF21 program were cut in the same period) — a strategic, not scientific, decision [86, 87, 90]. No liver αvβ1 program has been reported active since [88, 89].

---

## 5. Argument architecture (how to structure the case around prior research)

**Principle: prior research is the problem definition, not the competition.** The Sabat paper supplies the foundation, the gap, the generative priors, the structure, and the benchmark.

**Step 1 — Foundation (what prior research proved; cite, don't re-argue).** Target causality [103], human HSC mechanism [10], in-vivo small-molecule efficacy [111], human-safe oral compound [85], class PoC in fibrosis [98].

**Step 2 — Gap (what prior research explicitly did NOT deliver, in its own numbers).**

| Compound | Selective for αvβ1? | Oral-like / drug-like? | Public SAR? | Status |
|----------|--------------------|-----------------------|-------------|--------|
| CWHM-12 (tool) | ✗ pan-αv [103, 111] | ✗ | ✓ | literature tool |
| Sabat cpd 25 | ✓✓ (>50-fold) [111] | ✗ **oral F = 1.3 %** in rat (s.c. 59 %); series-wide MDCK permeability at assay floor (<0.1 × 10⁻⁶ cm/s); efficacy needed **50 mg/kg BID s.c.** [111] | ✓ | self-declared **tool compound** [111] |
| PLN-1474 | ✓ [85] | ✓ (oral, Ph1-clean) [85] | ✗ proprietary | **discontinued 2023 (strategic)** [86, 87] |
| Bexotegrast | partial (dual αvβ6/αvβ1) [98] | ✓ | partial | in clinic — for **lung**, not liver [98] |

**The empty cell: a public, αvβ1-selective, oral-like chemotype. That cell is the project.**

**Step 3 — Design hypothesis** (Section 1), stated before any generation. Falsifiable by construction.

**Step 4 — Test.** Transfer learning on public actives → reinforcement learning with a multi-component score encoding the pocket pharmacophore + permeability axes + selectivity counter-screens → docking triage into 8W30 → ADMET → novelty analysis (full protocol in the companion blueprint).

**Step 5 — Pre-registered success criteria.** A generated lead must: (i) dock into 8W30 comparably to TR01225179 and preserve the MIDAS + Asn224 contacts; (ii) beat cpd 25 on predicted permeability/lipophilicity axes; (iii) stay clean on αvβ3/α5β1/αvβ6 counter-screens; (iv) be scaffold-novel vs the public series (nearest-neighbor Tanimoto below a pre-set ceiling); (v) outscore two baselines run through the identical funnel — known actives (positive control) and random drug-like ChEMBL molecules (negative control). **If nothing beats the tool compound on the target axes while retaining contacts, the hypothesis is falsified — and the poster reports that.** Pre-registered falsifiability is the rigor signal that separates a proposal from a tutorial.

**Step 6 — Limitations, stated before the judges raise them.** In silico ≠ validated; the deliverable is a reasoned, benchmarked proposal plus a defined experimental confirmation path (competitive ELISA binding as in [111]; procollagen-1 assay in human HSCs as in [10]).

---

## 6. Binding-pocket characterization (PDB 8W30, analyzed in this work)

Structure: human αvβ1 headpiece co-crystallized with **TR01225179** (= N-(2,4-dichlorobenzoyl)-L-phenylalanine, the paper's screening hit "acid 1") [111, 112]. Distances below were measured directly from the deposited coordinates (this work; table in `avb1_8w30_contacts.csv`; figures `avb1_pocket_3d.png`, `avb1_pocket_contacts.png`).

**Four anchors (what a ligand must keep):**

| Anchor | Geometry (measured) | Note |
|--------|--------------------|------|
| **MIDAS metal coordination** | carboxylate OXT → Ca501 (β1): **2.62 Å**, monodentate | the carboxylate is non-negotiable for orthosteric RGD-integrin binding; flanked by β1-Ser132/Ser134/Glu229 (3.0–3.3 Å) |
| **β1-Asn224 H-bond** | backbone O ← ligand amide NH: **2.63 Å** | central amide is the H-bond donor |
| **αv-Tyr178 aromatic contact** | closest ring atoms **3.70 Å**; centroid–centroid 6.2 Å, ring-plane angle ~75° (T-shaped/offset) | engages the ligand's phenylalanine ring |
| **β1 hydrophobic subpocket** | dichlorophenyl ring amid Pro186 (3.16 Å), Tyr133 (3.17), Cys187 (3.88), Lys182 (3.82) | shape-driven selectivity element |

**Two growth vectors (what prior art reached for — and generative design should too):**

| Vector | Geometry | Note |
|--------|----------|------|
| **αv-Asp218** | **7.0 Å** from the hit — *not engaged* in 8W30 | the benzimidazolone NH of cpd 25 was designed to donate to Asp218; key potency/selectivity vector [111] |
| **Interface water HOH2107** | 4.7 Å from the hit; bridges αv-Glu121 and β1-Ser177 | later analogs were designed toward this water [111] |

**Design implications (the bridge to the REINVENT4 protocol):**
- The pocket is **metalloenzyme-like**: scoring must preserve the carboxylate–MIDAS anchor (substructure constraint + post-docking metal-distance filter), because standard docking scores do not model metal coordination.
- **Selectivity is a β1-subpocket problem** (Asn224/Leu225 region + hydrophobic pocket shape), so counter-screens against αvβ3/α5β1/αvβ6 belong *inside* the generative objective, not as an afterthought.
- **The permeability ceiling is structural**: the MIDAS carboxylate plus the H-bonding amide raise TPSA and lower logD — exactly the axes the generative score must optimize against, and exactly why cpd 25 ended as a tool compound (oral F 1.3 %) [111]. An optional exploratory arm may test acid bioisosteres (tetrazole, acylsulfonamide), accepting the potency risk — that trade-off *is* the hypothesis test.

---

## 7. Attack surface (anticipated counterarguments and defenses)

1. **"Prior research already made a selective αvβ1 inhibitor with in-vivo efficacy — what are you adding?"** The paper's own data answer this: cpd 25 is a self-declared *tool compound* — oral F 1.3 %, efficacy only at 50 mg/kg s.c. BID, permeability at the assay floor [111]. The project targets the empty cell (public + selective + oral-like), with cpd 25 as the benchmark to beat, not the result to reproduce.
2. **"No human germline genetics — your own matrix says genetics doubles success."** Conceded as the weakest pillar [110, 107]. Defense: ITGB1 is ubiquitously essential, so purifying selection removes common LOF variants — GWAS-silence is expected even for causal targets. Validation instead comes from cell-type-specific in-vivo genetics [103], human HSC pharmacology [10], and clinical class PoC [98].
3. **"Novartis walked away from PLN-1474 — maybe they saw something."** The public record shows a portfolio-wide NASH divestment (multiple unrelated assets cut simultaneously) [86, 87, 90]; no data-driven signal was disclosed. Honest answer: undisclosed data cannot be excluded — present as an open question, not a resolved one.
4. **"Selectivity across 24 integrins is a minefield."** Correct, and that is why counter-screens (αvβ3, α5β1, αvβ6, αvβ8) are built into the generative scoring function. >50-fold selectivity is demonstrably achievable in this pocket [111]. Note the αvβ6-selective antibody BG00011 had safety issues in IPF [102] — a point in favor of β1-selective (not pan-αv) inhibition.
5. **"TGF-β can be activated by other routes (αvβ6/β8, thrombospondin-1, MMPs, ROS) — blockade will be partial."** Partly true for the TGF-β arm; but αvβ1's procollagen control in human HSCs is partly TGF-β-activation-*independent* (noncanonical) [10, 111]. Frame as HSC-axis control, not total pathway blockade.
6. **"The MIDAS carboxylate may intrinsically cap permeability — your hypothesis may be false."** Correct, and stated as the pre-registered falsification condition (Section 5, step 5). Mitigations: acidic-bioisostere arm; prodrug precedent in the integrin literature; and the fallback claim that the campaign maps *how far* permeability can be pushed while keeping the pharmacophore — a quantitative answer either way.
7. **"Chronic TGF-β modulation could be unsafe long-term (tissue repair, tumor surveillance)."** Integrin-restricted, activation-level blockade is precisely the proposed safety mechanism versus systemic TGF-β blockade — but it is a hypothesis, and long-term safety is unknowable from this project. Say so.
8. **"Generative output is not a drug."** Agreed. The deliverable is a benchmarked, falsifiable lead *proposal* with an explicit experimental confirmation path — which is the level of claim the festival format asks for.

---

## 8. The 7-minute story arc

1. **(0:00–1:00) Fibrosis is what kills** — prognosis data [52, 53, 55]; MASH scale [78].
2. **(1:00–2:00) Two approvals, zero direct antifibrotics** — resmetirom/semaglutide are metabolic, F2–F3 only, modest fibrosis effect [33, 38, 78, 80]; the graveyard [1, 2, 5, 7, 8].
3. **(2:00–3:00) The TGF-β dilemma** — master regulator, but systemically undruggable; nature activates latent TGF-β *locally* via integrins; on human HSCs the switch is αvβ1 [10, 103].
4. **(3:00–4:00) Validated, then orphaned** — in-vivo genetics [103], selective inhibitor efficacy [111], human-safe Ph1 compound [85], class PoC in fibrosis [98] — and the strategic discontinuation that left the target empty [86].
5. **(4:00–5:00) The real gap, in the prior art's own numbers** — the three-compounds table; cpd 25's 1.3 % oral bioavailability [111]; the design hypothesis.
6. **(5:00–6:15) The test** — REINVENT4 funnel (TL → RL with pharmacophore + selectivity + permeability scoring → 8W30 docking → ADMET → novelty), benchmark panel, pre-registered success criteria [40].
7. **(6:15–7:00) Lead proposal + limitations** — 3–5 proposed leads vs references; falsification condition; experimental confirmation path.

---

## 9. References

1. Harrison SA et al. Selonsertib … STELLAR trials. *J Hepatol* 2020. doi:10.1016/j.jhep.2020.02.027
2. Anstee QM et al. Cenicriviroc … AURORA Phase III. *Clin Gastroenterol Hepatol* 2024. doi:10.1016/j.cgh.2023.04.003
5. Harrison SA et al. Emricasan in NASH F1–F3. *J Hepatol* 2020. doi:10.1016/j.jhep.2019.11.024
6. Rinella ME, Noureddin M. STELLAR 3 and 4: Lessons from the fall of Icarus. *J Hepatol* 2020. doi:10.1016/j.jhep.2020.04.034
7. Ratziu V, Friedman SL. Why Do So Many NASH Trials Fail? *Gastroenterology* 2023. doi:10.1053/j.gastro.2020.05.046
8. Drenth JPH, Schattenberg JM. The NASH drug development graveyard. *Expert Opin Investig Drugs* 2020. doi:10.1080/13543784.2020.1839888
9. Yokosaki Y, Nishimichi N. α8β1 and α11β1 on activated stellate cells. *Int J Mol Sci* 2021. doi:10.3390/ijms222312794
10. Han Z et al. Integrin αVβ1 regulates procollagen I … in human hepatic stellate cells. *Biochem J* 2021. doi:10.1042/bcj20200749
12. Rahman S et al. Integrins as a drug target in liver fibrosis. *Liver Int* 2022. doi:10.1111/liv.15157
14. Hinz B. It has to be the αv: myofibroblast integrins activate latent TGF-β1. *Nat Med* 2013. doi:10.1038/nm.3421
16. Ma Y et al. Antisense oligonucleotide targeting Hsd17b13 in a fibrosis mouse model. *J Lipid Res* 2024. doi:10.1016/j.jlr.2024.100514
21. Chen L et al. Highly potent, selective, liver-targeting HSD17B13 inhibitor. *J Med Chem* 2025. doi:10.1021/acs.jmedchem.5c00119
24. Ma Y et al. HSD17B13 deficiency does not protect mice from obesogenic diet injury. *Hepatology* 2021. doi:10.1002/hep.31517
25. Abul-Husn NS et al. A protein-truncating HSD17B13 variant and protection from chronic liver disease. *NEJM* 2018. doi:10.1056/nejmoa1712191
27. Stender S, Romeo S. HSD17B13 as a therapeutic target. *Liver Int* 2020. doi:10.1111/liv.14411
28. Gil-Gómez A et al. HSD17B13 rs72613567 and hepatic decompensation in cirrhosis. *Int J Mol Sci* 2022. doi:10.3390/ijms231911840
30. Vilar-Gomez E et al. HSD17B13 rs72613567 fibrosis protection mediation. *Clin Gastroenterol Hepatol* 2023. doi:10.1016/j.cgh.2022.11.002
31. Luukkonen PK et al. HSD17B13 variant, phospholipids and fibrosis in NAFLD. *JCI Insight* 2020. doi:10.1172/jci.insight.132158
33. FDA press release: Rezdiffra approval, March 14, 2024. fda.gov
36. Madrigal Pharmaceuticals press release, March 14, 2024.
38. Fierce Pharma: "FDA approves first MASH drug…", March 14, 2024.
40. Loeffler HH, He J et al. Reinvent 4: Modern AI-driven generative molecule design. *J Cheminform* 2024. doi:10.1186/s13321-024-00812-5
45. RCSB PDB 8G89 (HSD17B13–inhibitor complex).
52. Angulo P et al. Liver fibrosis, but no other histologic features, predicts long-term outcomes in NAFLD. *Gastroenterology* 2015. doi:10.1053/j.gastro.2015.04.043
53. Hagström H et al. Fibrosis stage but not NASH predicts mortality in NAFLD. *J Hepatol* 2017. doi:10.1016/j.jhep.2017.07.027
55. Sanyal AJ et al. Prospective study of outcomes in adults with NAFLD (NASH CRN). *N Engl J Med* 2021. (PMC8881985)
57. Sanyal AJ et al. Phase I RNAi (rapirosiran) targeting HSD17B13 in MASH. *J Hepatol* 2025. doi:10.1016/j.jhep.2025.05.031
61. Mak LY et al. Phase I/II ARO-HSD in NASH. *J Hepatol* 2022. doi:10.1016/j.jhep.2022.11.022
66. Lindén D, Tesz G, Loomba R. Targeting PNPLA3 to treat MASH. *Liver Int* 2024. doi:10.1111/liv.16186
67. Armisen J et al. AZD2693 (PNPLA3 ASO) Phase I. *J Hepatol* 2025. doi:10.1016/j.jhep.2024.12.046
70. Sherman DJ et al. PNPLA3-I148M is a neomorph. *Cell Rep* 2025. doi:10.1016/j.celrep.2025.116371
71. Wang Y et al. PNPLA3(148M) gain-of-function via ATGL inhibition. *J Hepatol* 2025. doi:10.1016/j.jhep.2024.10.048
73. Coyne ES et al. Loss of mARC1 reduces fibrosis in mouse CLD models. *Hepatol Commun* 2025. doi:10.1097/hc9.0000000000000637
74. Smagris E et al. Divergent role of MARC1 in human and mouse. *PLOS Genet* 2024. doi:10.1371/journal.pgen.1011179
75. Emdin CA et al. A missense variant in MARC1 and protection against liver disease. *PLOS Genet* 2020. doi:10.1371/journal.pgen.1008629
78. FDA: "FDA Approves Treatment for Serious Liver Disease Known as MASH" (Wegovy), August 15, 2025. fda.gov
79. Novo Nordisk press release: Wegovy approved for MASH, August 15, 2025.
80. Novo Nordisk company announcement (ESSENCE data), August 15, 2025.
85. Pliant Therapeutics press release: PLN-1474 Phase 1 completion and transfer to Novartis, March 16, 2021.
86. Pliant Therapeutics 8-K: Novartis termination of the collaboration (PLN-1474), February 17, 2023.
87. Fierce Biotech: "Novartis punts Pliant-partnered NASH prospect…", February 24, 2023.
88. AdisInsight drug profile: PLN-1474 (structure added Aug 2023; no development reported as of May 2025).
89. PatSnap Synapse drug profile: PLN-1474 (discontinued).
90. Endpoints News: "Novartis retreat from NASH takes down $80M alliance with Pliant", February 24, 2023.
98. Lancaster L et al. Bexotegrast in IPF: INTEGRIS-IPF trial. *Am J Respir Crit Care Med* 2024. doi:10.1164/rccm.202403-0636oc
99. Wuyts W et al. Long-term safety/antifibrotic activity of bexotegrast (ERS abstract). *Eur Respir J* 2023. doi:10.1183/13993003.congress-2023.oa1423
101. INTEGRIS-IPF full text, PMC11351797.
102. Wang F, Zhu M, Luo F. INTEGRIS-IPF: A New Hope for Tomorrow (editorial; BG00011 and GSK3008348 context). *Am J Respir Crit Care Med* 2024. doi:10.1164/rccm.202407-1295ed
103. Henderson NC et al. Targeting of αv integrin identifies a core molecular pathway that regulates fibrosis in several organs. *Nat Med* 2013. doi:10.1038/nm.3282
107. King EA, Davis JW, Degner JF. Are drug targets with genetic support twice as likely to be approved? *PLOS Genet* 2019. doi:10.1101/513945
110. Nelson MR et al. The support of human genetic evidence for approved drug indications. *Nat Genet* 2015. doi:10.1038/ng.3314
111. Sabat M et al. Design and Discovery of a Potent and Selective Inhibitor of Integrin αvβ1. *J Med Chem* 2024;67:10306–10320. doi:10.1021/acs.jmedchem.4c00743
112. RCSB PDB 8W30: crystal structure of the αvβ1 integrin headpiece with TR01225179. doi:10.2210/pdb8W30/pdb

*Companion files: `avb1_10day_blueprint.md` (execution plan + REINVENT4 protocol), `reinvent4_avb1_scoring_config_sketch.toml` (annotated config), `avb1_pocket_3d.png`, `avb1_pocket_contacts.png/.svg`, `avb1_8w30_contacts.csv`.*
