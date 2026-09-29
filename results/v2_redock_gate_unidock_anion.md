# Redocking validation gate (spec section 7.4)

Engine: **Uni-Dock**
Control ligand poses: `docking/v2/unidock/control_anion/CONTROL_crystal_out.pdbqt`
Poses parsed: 4. Gate applied to the TOP pose by affinity (rank 1, -6.913 kcal/mol).
Protonation state docked: **anion**

Verdict: **PASS**

| criterion | measured | threshold | result |
|---|---|---|---|
| (a) carboxylate O to Ca501 | 2.71 A | <= 3.2 A | pass |
| (b) donor to Asn224 backbone O | 2.89 A | <= 3.5 A | pass |
| (c) symmetry-aware RMSD (CalcRMS) | 0.63 A | < 2.0 A | pass |
| (d) PLIP Ca501 metal complex | reported | required | pass |
| (d) PLIP Asn224 H-bond | reported | required | pass |

Crystal reference: Ca 2.62 A, Asn224 donor-acceptor 2.63 A (PLIP 3.0.1 on the deposited 8W30 structure).

## Every pose, for the record

The gate is the marked row (spec 7.4 gates on the top pose by affinity). The others say whether the engine missed by a little everywhere or only here; they are not alternative verdicts.

| | pose | affinity | Ca dist | Asn224 dist | RMSD | PLIP metal | PLIP H-bond |
|---|---|---|---|---|---|---|---|
| <- | 1 | -6.91 | 2.71 | 2.89 | 0.63 | yes | yes |
|  | 2 | -6.91 | 2.72 | 2.90 | 0.63 | yes | yes |
|  | 3 | -6.90 | 2.72 | 2.88 | 0.63 | yes | yes |
|  | 4 | -6.89 | 2.74 | 2.91 | 0.66 | yes | yes |

---

AutoDock-GPU reproduces the deposited pose. It is adopted as this campaign's docking engine.
