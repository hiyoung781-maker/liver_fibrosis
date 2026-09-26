# Step 2 single-stage RL scoring block (blueprint section 5.1).
#
#   total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SA, 0.5)]) * alert_filter
#
# The carboxylate is a GATE, not a MatchingSubstructure penalty: that component
# hardcodes 0.5 * (1.0 + match), and under it a molecule WITHOUT the carboxylate
# scored a flat 0.500 while molecules WITH it had a median of 0.203, because the
# acid adds ~37 TPSA and falls out of the window. The agent's best move was to drop
# the MIDAS anchor. GroupCount + right_step makes absence cost 6.31e-04 instead.
#
# TanimotoSimilarity is deliberately absent. The six references were chosen to be
# mutually DISSIMILAR (max pairwise Tanimoto 0.45), one endpoint each, under a
# geometric mean - which demands a molecule resemble six mutually unlike molecules
# at once. Prior-sample medians were 0.002-0.004 with 70% under 0.05. Potency comes
# from TL-A and inception; potency judgement comes from the section 8.5 docking
# geometry filter.

[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name = "carboxylate MIDAS anchor"
weight = 1.0
params.smarts = ["[CX3](=O)[OX2H1,OX1-]"]
transform.type = "right_step"
transform.high = 1

# Window anchored on the two orally advanced alphaV carboxylates: PLN-1474
# (TPSA 100.6, Phase 1 completed) and bexotegrast (112.5, Phase 2a). The cpd 25
# series cannot justify the window - its high TPSA (median 133.3) just restates
# that it was never proposed as an oral structure.
[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name = "TPSA"
weight = 1.0
transform.type = "double_sigmoid"
transform.low = 40.0
transform.high = 115.0
transform.coef_div = 120.0
transform.coef_si = 20.0
transform.coef_se = 20.0

# GUARD RAIL, not an optimization axis. Every reference molecule and every prior
# sample scores exactly 1.000 here, so this term contributes nothing at step 0; it
# only bites if 600 steps of RL drift past SA 6, the conventional hard-to-make line.
# The earlier (3, 6) window gave actives_extended's worst molecule (SA 5.10) a 0.093
# - punishing a published active.
[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name = "SA score"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.low = 6.0
transform.high = 8.0
transform.k = 0.5

# The aniline pattern is [NX3;H2][c], primary aromatic amines only. The earlier
# [NH2,NH][c] zeroed 164 of 190 actives_extended (86%) and 5 of 6 benchmark
# positives, because an aryl-NH with one hydrogen matches - which catches
# tetrahydro-1,8-naphthyridine, the standard integrin Arg-mimic head.
[[stage.scoring.component]]
[stage.scoring.component.CustomAlerts]
[[stage.scoring.component.CustomAlerts.endpoint]]
name = "unwanted groups"
params.smarts = ["[NX3;H2][c]", "[*;r8]", "[*;r9]", "[*;r10]", "N=[N+]=[N-]", "C(=O)Cl", "[SH]", "[Nr0][Nr0]"]
