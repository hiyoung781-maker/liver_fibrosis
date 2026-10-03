"""Generate the Task-6 RL arm configs from configs/_rl_scoring.frag.

Every arm runs the IDENTICAL objective with inception removed (spec section
4.1). Arms differ ONLY in `agent_file` and the output paths that name the arm.
Everything else - scoring components and their transforms, learning_strategy,
batch_size, max_steps, diversity_filter, seed, batch ordering - is generated
from the same template so the files cannot drift apart by hand-editing one.

Unlike scripts/make_scoring_config.py (section 7, `run_type = "scoring"`, which
strips the `stage.` prefix), configs/_rl_scoring.frag already carries the
`[[stage.scoring.component]]` headers a `staged_learning` RL config needs.

WHY THE FRAGMENT IS NO LONGER INLINED VERBATIM. It used to be, and the comment
saying so was the defect: v4 re-derived the TPSA window (results/v4_tpsa_window.md)
and the new bounds were hand-edited into configs/rl_C.toml and
configs/rl_A_prime.toml ONLY, leaving the fragment at the old (40, 115). Any
regeneration - including the first `--arm B` run - would have silently restored
the window v4 exists to replace, and the "do not edit by hand" header made that
look impossible.

The fragment cannot simply be updated, because it is NOT RL-only. It is also the
source of configs/score_panel.toml (make_scoring_config.py), the column names
scripts/panel_report.py reads, and the tail that
reinvent4_avb1_scoring_config_sketch.toml is asserted to end with
(tests/test_config.py). That chain is the v2 blueprint record and v2/v3 reported
panel numbers rest on it.

So the fragment stays frozen as the blueprint objective, and v4's divergence from
it lives HERE, as the two declared deviations below. One source of truth for the
RL objective, and the divergence is a tested declaration rather than a silent
rewrite of a pre-registration document.
"""

from __future__ import annotations

import argparse
import re
import sys

FRAG = "configs/_rl_scoring.frag"

COMPONENT_HEADER = "[[stage.scoring.component]]"

ARMS = {
    "A-prime": {
        "tag": "v2_TL-A-prime",
        "agent_file": "priors/focused_A_prime.prior",
    },
    "B": {
        "tag": "v4_TL-B",
        "agent_file": "priors/focused_B.prior",
    },
    "C": {
        "tag": "v2_TL-C",
        "agent_file": "REINVENT4/priors/reinvent.prior",
    },
}

# A tag is "<campaign>_TL-<arm>", and that campaign prefix is what every later
# stage scopes its paths by: results/v4_TL-B/sample.csv, data/v4_TL-B/library.smi,
# and so on. Deriving the mapping from ARMS keeps ONE place that knows TL-B is a
# v4 arm - build_library.py and build_survivors.py each spelled "v2_" into an
# f-string, which is why neither could express a v4 arm at all.
ARM_CAMPAIGNS = {tag.split("_", 1)[1]: tag.split("_", 1)[0]
                 for tag in (arm["tag"] for arm in ARMS.values())}


def campaign_for(arm_label: str | None, default: str = "v2") -> str:
    """'v4' for 'TL-B', 'v2' for the v2 arms, `default` for anything undeclared.

    Defaulting to v2 means a new arm has to be declared in ARMS before the rest
    of the pipeline treats it as v4 - an undeclared arm fails toward the
    pre-registered behaviour instead of silently inheriting v4's.
    """
    return ARM_CAMPAIGNS.get(arm_label, default)

# ---------------------------------------------------------------------------
# v4 DEVIATION 1: SAScore is dropped from the RL objective.
#
# The fragment documents it as a GUARD RAIL, not an optimization axis: every
# reference molecule and every prior sample scores 0.99998 or better, so at step
# 0 it moves a total by 2.2e-06. Its only function was to brake synthetic-
# accessibility drift over 1000 steps. Dropping it (user decision, 2026-10-03)
# removes that brake, and core_B has more room to drift than core did - MW
# median 465.6 vs 398.4, QED median 0.411 vs 0.547 (spec section 3.4).
#
# WHAT ELSE MOVES, because REINVENT normalizes geometric-mean weights by their
# sum (reinvent/scoring/aggregators/means.py: `scores ** (weights/sum_weights)`).
# Removing weight 0.5 takes the divisor from 2.5 to 2.0, so every remaining
# component's exponent rises from 0.4 to 0.5:
#
#   carboxylate present: total TPSA^0.4 -> TPSA^0.5          (totals fall)
#   carboxylate absent:  6.31e-04 -> 1.0e-04                 (MIDAS gate sharpens ~6x)
#   diversity_filter minscore 0.4 needs TPSA >= 0.101 -> 0.160 (stricter)
#
# The 6.31e-04 in the fragment's own comment is the measured value and matches
# 1e-8 ** 0.4 exactly, which is what confirms this exponent model rather than
# assuming it.
#
# SA REMAINS OBSERVABLE. SAScore is still in configs/score_panel.toml, so the
# sampled library's SA distribution is measured after the fact and reported on
# the lead cards - an observation, not a gate, the same standing the alphaIIbbeta3
# span has (spec section 6).
RL_DROP_COMPONENTS = ("SAScore",)

# v4 DEVIATION 2: the TPSA window, re-derived in results/v4_tpsa_window.md.
# 115 came from two alphaV compounds and penalised one of them (bexotegrast
# 112.5 scored 0.72); 140 is Veber's published oral-bioavailability threshold.
# The floor moves 40 -> 60, non-binding for every reference and every current
# lead (minimum TPSA 77.8) while keeping the only polarity floor this objective
# has. coef_div/si/se are deliberately unchanged so the only moving parts are
# the two bounds.
RL_TRANSFORM_OVERRIDES = {"TPSA": {"low": 60.0, "high": 140.0}}

# The fragment's preamble states the aggregation formula. Dropping a component
# makes that line false, and a generated config that misdescribes its own
# objective is the defect this module's docstring is about.
_FORMULA_OLD = "#   total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SA, 0.5)]) * alert_filter"
_FORMULA_NEW = "#   total = geometric_mean([(COOH, 1.0), (TPSA, 1.0)]) * alert_filter"

# Same reason, one paragraph down: 6.31e-04 is 1e-8 ** 0.4, the cost of a missing
# carboxylate while SAScore held the divisor at 2.5. Without it the exponent is 0.5
# and the cost is 1e-04. Leaving the measured-but-now-wrong number in a generated
# file would contradict the deviation note appended below it.
_PENALTY_OLD = "# the MIDAS anchor. GroupCount + right_step makes absence cost 6.31e-04 instead."
_PENALTY_NEW = ("# the MIDAS anchor. GroupCount + right_step makes absence cost 1.0e-04 instead\n"
                "# (6.31e-04 before SAScore was dropped - see the v4 deviations below).")


def split_components(frag_text: str) -> tuple[str, list[tuple[str, str]]]:
    """The fragment's preamble, then one (component_name, text) per component.

    A component's block starts at the comment lines immediately ABOVE its
    `[[stage.scoring.component]]` header, not at the header itself. Every
    component in the fragment is preceded by the paragraph explaining why its
    transform is shaped the way it is, and a split made at the header would
    strand each rationale above the wrong component - or, when a component is
    dropped, leave its justification behind explaining a block that is gone.
    """
    lines = frag_text.splitlines(keepends=True)

    starts: list[int] = []
    for index, line in enumerate(lines):
        if line.strip() != COMPONENT_HEADER:
            continue
        start = index
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        if start > 0 and not lines[start - 1].strip():
            start -= 1
        starts.append(start)

    if not starts:
        raise ValueError(f"no {COMPONENT_HEADER} blocks found")

    preamble = "".join(lines[: starts[0]])
    bounds = starts + [len(lines)]
    blocks = []
    for begin, end in zip(bounds, bounds[1:]):
        text = "".join(lines[begin:end])
        match = re.search(r"^\[stage\.scoring\.component\.(\w+)\]", text, re.M)
        if match is None:
            raise ValueError(f"component block has no type header:\n{text}")
        blocks.append((match.group(1), text))
    return preamble, blocks


def override_transform(block: str, overrides: dict[str, float]) -> str:
    """Rewrite `transform.<key>` values inside ONE component's block.

    Scoped to a single block on purpose: SAScore carries transform.low/high too,
    so a document-wide substitution would move the guard rail's window while
    aiming at TPSA's.
    """
    for key, value in overrides.items():
        pattern = re.compile(rf"^transform\.{key}\s*=.*$", re.M)
        if not pattern.search(block):
            raise ValueError(f"transform.{key} not present in block")
        block = pattern.sub(f"transform.{key} = {value}", block, count=1)
    return block


def rl_scoring(frag_text: str) -> str:
    """The fragment with v4's two declared deviations applied."""
    preamble, blocks = split_components(frag_text)

    names = [name for name, _ in blocks]
    for dropped in RL_DROP_COMPONENTS:
        if dropped not in names:
            raise ValueError(f"cannot drop {dropped!r}: not in {names}")
    for component in RL_TRANSFORM_OVERRIDES:
        if component not in names:
            raise ValueError(f"cannot override {component!r}: not in {names}")

    if _FORMULA_OLD not in preamble:
        raise ValueError("fragment preamble no longer states the aggregation "
                         "formula this generator rewrites; re-check it by hand")
    preamble = preamble.replace(_FORMULA_OLD, _FORMULA_NEW)
    if _PENALTY_OLD not in preamble:
        raise ValueError("fragment preamble no longer states the carboxylate-absence "
                         "penalty this generator corrects; re-check it by hand")
    preamble = preamble.replace(_PENALTY_OLD, _PENALTY_NEW)
    preamble = preamble.rstrip("\n") + "\n" + "\n".join([
        "#",
        "# v4 DEVIATIONS FROM THE FRAGMENT, applied by scripts/make_rl_config.py",
        "# (see its module docstring for why they live there and not in the",
        "# fragment, which is also the source of configs/score_panel.toml):",
        f"#   dropped components: {', '.join(RL_DROP_COMPONENTS)}",
        "#     SAScore was a guard rail scoring 0.99998+ on every reference, so",
        "#     it moved a total by 2.2e-06 at step 0 and only braked SA drift over",
        "#     1000 steps. Removing weight 0.5 renormalizes the geometric mean:",
        "#     each remaining exponent 0.4 -> 0.5, so totals fall, carboxylate",
        "#     absence costs 1.0e-04 instead of 6.31e-04, and minscore 0.4 now",
        "#     needs a TPSA component of 0.160 rather than 0.101. SA stays",
        "#     measurable through configs/score_panel.toml as an observation.",
        "#   transform overrides: TPSA low 40 -> 60, high 115 -> 140",
        "#     results/v4_tpsa_window.md. The old ceiling penalised bexotegrast",
        "#     (TPSA 112.5 -> 0.72) and zeroed 58% of core_B; 140 is Veber's oral",
        "#     threshold and 60 is non-binding for every reference and lead.",
        "",
    ])

    kept = []
    for name, block in blocks:
        if name in RL_DROP_COMPONENTS:
            continue
        if name in RL_TRANSFORM_OVERRIDES:
            block = override_transform(block, RL_TRANSFORM_OVERRIDES[name])
        kept.append(block)
    return preamble + "".join(kept)


def build_config(arm: str, frag_path: str = FRAG) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {sorted(ARMS)}")
    tag = ARMS[arm]["tag"]
    agent_file = ARMS[arm]["agent_file"]

    with open(frag_path) as handle:
        scoring = rl_scoring(handle.read())

    preamble = "\n".join([
        f"# GENERATED by scripts/make_rl_config.py --arm {arm} - do not edit by hand.",
        f"# Scoring components come from {frag_path} (spec section 5.1) with the two",
        "# v4 deviations that generator declares - NOT verbatim; see below and its",
        "# module docstring. Hand-editing a window into this file instead is exactly",
        "# how v4's re-derived TPSA bounds came to exist only in the generated output.",
        "# Inception is deliberately absent - spec section 4.1: inception.py truncates",
        "# by score and never re-admits an evicted seed, and its loss uses the PRIOR",
        "# log-likelihood, which would push the arms in opposite directions and",
        "# contaminate the one comparison this task exists to make.",
        "#",
        f"# Arm {arm}: differs from every other arm ONLY in `agent_file` below and the",
        "# output paths that name this arm. Scoring, learning_strategy, batch_size,",
        "# max_steps, diversity_filter, and batch ordering are identical across arms.",
        'run_type = "staged_learning"',
        'device = "cuda:0"',
        f'tb_logdir = "logs/{tag}/tb"',
        "",
        "[parameters]",
        f'agent_file = "{agent_file}"',
        'prior_file = "REINVENT4/priors/reinvent.prior"',
        "batch_size = 128",
        "unique_sequences = true",
        "randomize_smiles = true",
        f'summary_csv_prefix = "results/{tag}/rl"',
        "",
        "[learning_strategy]",
        'type = "dap"',
        "sigma = 128",
        "rate = 0.0001",
        "",
        "[[stage]]",
        "max_steps = 1000",
        'termination = "simple"',
        "max_score = 1.0",
        f'chkpt_file = "results/{tag}/rl_final.chkpt"',
        "",
        "[stage.diversity_filter]",
        'type = "IdenticalMurckoScaffold"',
        "bucket_size = 25",
        "minscore = 0.4",
        "",
        "[stage.scoring]",
        'type = "geometric_mean"',
        "",
    ])
    return preamble + scoring


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    text = build_config(args.arm)
    with open(args.out, "w") as handle:
        handle.write(text)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
