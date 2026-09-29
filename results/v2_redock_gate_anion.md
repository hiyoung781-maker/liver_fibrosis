# Redocking validation gate (spec section 7.4)

Control ligand `.dlg`: `docking/v2/poses/control_anion/CONTROL_crystal.dlg`
Poses parsed: 20. Gate applied to the TOP pose by affinity (rank 1, -7.81 kcal/mol).
Protonation state docked: **anion**

Verdict: **FAIL**

| criterion | measured | threshold | result |
|---|---|---|---|
| (a) carboxylate O to Ca501 | 2.51 A | <= 3.2 A | pass |
| (b) donor to Asn224 backbone O | 2.81 A | <= 3.5 A | pass |
| (c) symmetry-aware RMSD (CalcRMS) | 2.15 A | < 2.0 A | FAIL |
| (d) PLIP Ca501 metal complex | reported | required | pass |
| (d) PLIP Asn224 H-bond | reported | required | pass |

Crystal reference: Ca 2.62 A, Asn224 donor-acceptor 2.63 A (PLIP 3.0.1 on the deposited 8W30 structure).

## Every pose, for the record

The gate is the marked row (spec 7.4 gates on the top pose by affinity). The others say whether the engine missed by a little everywhere or only here; they are not alternative verdicts.

| | pose | affinity | Ca dist | Asn224 dist | RMSD | PLIP metal | PLIP H-bond |
|---|---|---|---|---|---|---|---|
| <- | 1 | -7.81 | 2.51 | 2.81 | 2.15 | yes | yes |
|  | 12 | -7.81 | 2.50 | 2.81 | 2.16 | yes | yes |
|  | 19 | -7.77 | 2.51 | 2.79 | 2.14 | yes | yes |
|  | 7 | -7.76 | 2.51 | 2.80 | 2.13 | yes | yes |
|  | 8 | -7.76 | 2.49 | 2.80 | 2.13 | yes | yes |
|  | 10 | -7.76 | 2.50 | 2.79 | 2.19 | yes | yes |
|  | 20 | -7.75 | 2.52 | 2.80 | 2.10 | yes | yes |
|  | 11 | -7.72 | 2.49 | 2.79 | 2.10 | yes | yes |
|  | 4 | -7.71 | 2.50 | 2.77 | 2.01 | yes | yes |
|  | 5 | -7.53 | 2.47 | 2.70 | 0.55 | yes | yes |
|  | 2 | -7.52 | 2.50 | 2.68 | 0.54 | yes | yes |
|  | 17 | -7.51 | 2.50 | 2.70 | 0.54 | yes | yes |
|  | 3 | -7.50 | 2.53 | 2.68 | 0.54 | yes | yes |
|  | 15 | -7.50 | 2.48 | 2.71 | 0.54 | yes | yes |
|  | 13 | -7.49 | 2.48 | 2.71 | 0.53 | yes | yes |
|  | 6 | -7.47 | 2.49 | 2.71 | 0.54 | yes | yes |
|  | 14 | -7.47 | 2.48 | 2.69 | 0.55 | yes | yes |
|  | 16 | -7.45 | 2.51 | 2.69 | 0.57 | yes | yes |
|  | 18 | -7.45 | 2.50 | 2.69 | 0.56 | yes | yes |
|  | 9 | -6.44 | 12.99 | 5.05 | 6.70 | no | no |

---

Failed: rmsd. Per spec 7.4, **AutoDock-GPU ranks and filters nothing**; the campaign reverts to the Uni-Dock regime and this file is the record of why.
