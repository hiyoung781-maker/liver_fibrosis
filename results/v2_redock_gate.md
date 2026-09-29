# Redocking validation gate (spec section 7.4)

Control ligand `.dlg`: `docking/v2/poses/TL-A-prime/CONTROL_crystal.dlg`
Poses parsed: 20. Gate applied to the TOP pose by affinity (rank 5, -6.77 kcal/mol).
Protonation state docked: **neutral**

Verdict: **FAIL**

| criterion | measured | threshold | result |
|---|---|---|---|
| (a) carboxylate O to Ca501 | 2.57 A | <= 3.2 A | pass |
| (b) donor to Asn224 backbone O | 2.81 A | <= 3.5 A | pass |
| (c) symmetry-aware RMSD (CalcRMS) | 2.15 A | < 2.0 A | FAIL |
| (d) PLIP Ca501 metal complex | reported | required | pass |
| (d) PLIP Asn224 H-bond | reported | required | pass |

Crystal reference: Ca 2.62 A, Asn224 donor-acceptor 2.63 A (PLIP 3.0.1 on the deposited 8W30 structure).

## Every pose, for the record

The gate is the marked row (spec 7.4 gates on the top pose by affinity). The others say whether the engine missed by a little everywhere or only here; they are not alternative verdicts.

| | pose | affinity | Ca dist | Asn224 dist | RMSD | PLIP metal | PLIP H-bond |
|---|---|---|---|---|---|---|---|
| <- | 5 | -6.77 | 2.57 | 2.81 | 2.15 | yes | yes |
|  | 18 | -6.76 | 2.56 | 2.80 | 2.12 | yes | yes |
|  | 12 | -6.50 | 2.55 | 2.68 | 0.53 | yes | yes |
|  | 17 | -6.50 | 2.56 | 2.67 | 0.53 | yes | yes |
|  | 6 | -6.49 | 2.55 | 2.69 | 0.52 | yes | yes |
|  | 8 | -6.49 | 2.56 | 2.67 | 0.53 | yes | yes |
|  | 11 | -6.49 | 2.56 | 2.67 | 0.53 | yes | yes |
|  | 13 | -6.49 | 2.53 | 2.70 | 0.53 | yes | yes |
|  | 2 | -6.48 | 2.54 | 2.70 | 0.52 | yes | yes |
|  | 4 | -6.48 | 2.59 | 2.70 | 0.51 | yes | yes |
|  | 7 | -6.48 | 2.54 | 2.69 | 0.52 | yes | yes |
|  | 15 | -6.48 | 2.53 | 2.70 | 0.52 | yes | yes |
|  | 16 | -6.48 | 2.55 | 2.69 | 0.54 | yes | yes |
|  | 19 | -6.48 | 2.52 | 2.71 | 0.52 | yes | yes |
|  | 20 | -6.48 | 2.49 | 2.70 | 0.53 | yes | yes |
|  | 9 | -6.47 | 2.54 | 2.71 | 0.53 | yes | yes |
|  | 10 | -6.47 | 2.53 | 2.70 | 0.53 | yes | yes |
|  | 3 | -6.46 | 2.56 | 2.69 | 0.52 | yes | yes |
|  | 14 | -6.38 | 11.95 | 2.65 | 7.73 | no | yes |
|  | 1 | -5.90 | 3.64 | 2.83 | 2.73 | no | yes |

---

Failed: rmsd. Per spec 7.4, **AutoDock-GPU ranks and filters nothing**; the campaign reverts to the Uni-Dock regime and this file is the record of why.
