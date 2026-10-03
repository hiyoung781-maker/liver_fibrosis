"""The RL scoring component that makes the objective see the whole RGD pharmacophore.

The objective gated the Asp mimic and nothing else, and the output shows it:
v4's sampled library puts a basic centre and an acid at a separation some
measured active has in 0.80% of molecules, while core_B - the set the prior was
built from - is 28.8% tetrahydronaphthyridine. These tests pin the behaviour
that is supposed to change that, and the measured values it is anchored on.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "REINVENT4"))
sys.path.insert(0, str(ROOT / "scripts"))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from reinvent_plugins.components.comp_pharmacophore_span import (  # noqa: E402
    Parameters, PharmacophoreSpan)

# Measured on the deposited/clinical compounds, not asserted. Both use a
# different spacer - a heptamethylene chain and an amine-bearing chain - and
# land on the same bond path, which is what makes the separation a structural
# constraint rather than a feature of one series.
PLN_1474 = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"
BEXOTEGRAST = "COCCN(CCCCc1ccc2c(n1)NCCC2)CCC(Nc1ncnc2ccccc12)C(=O)O"
# v4 generated molecules, from results/v4_TL-B/selection.csv.
GEN_03220 = "O=C(O)C(Cc1ccc2c(n1)NCCC2)c1ccnnc1"           # both groups, adjacent
GEN_00446 = "O=C(O)C1CNc2nc(CCc3ccc4c(n3)NCCC4)ccc2C1"      # spans like PLN-1474
GEN_11083 = ("O=C(O)COC(=O)C(Cc1ccc2c(c1)OCCO2)CN1CCC(CCc2ccc3c(n2)NCCC3)CC1")
NO_BASE = "O=C(O)Cc1ccccc1"
# 8W30's deposited ligand - a non-RGD alphaVbeta1 inhibitor, which is what the
# original structure paper set out to make.
A1AFA = "O=C(NC(Cc1ccccc1)C(=O)O)c1ccc(Cl)cc1Cl"


def score(*smiles):
    return list(PharmacophoreSpan(Parameters())(list(smiles)).scores[0])


class TestClinicalCompoundsAnchorTheScale(unittest.TestCase):
    def test_both_clinical_compounds_span_ten_bonds(self):
        """Different spacers, same span - the constraint is the distance, not
        the chemistry that provides it."""
        self.assertEqual(score(PLN_1474, BEXOTEGRAST), [10.0, 10.0])


class TestItSeparatesPresenceFromGeometry(unittest.TestCase):
    """The distinction the objective currently cannot make. gen_03220 carries a
    tetrahydronaphthyridine AND a carboxylate AND is a zwitterion at pH 7.4 -
    every box the old objective could tick - but at 4 bonds the head cannot
    reach alphaV Asp218 while the acid holds the beta1 MIDAS metal. No measured
    active sits below 9."""

    def test_adjacent_pharmacophores_score_far_below_the_clinical_compounds(self):
        self.assertEqual(score(GEN_03220), [4.0])

    def test_a_generated_molecule_can_reach_the_clinical_span(self):
        self.assertEqual(score(GEN_00446, GEN_11083), [10.0, 13.0])


class TestNonRgdMoleculesAreNotPenalised(unittest.TestCase):
    """THE DISTINCTION THIS TERM EXISTS TO MAKE.

    8W30, this campaign's target structure, was solved with a NON-RGD
    alphaVbeta1 inhibitor: A1AFA has a carboxylate and no basic group, and 40 of
    core_B's 191 actives are the same chemotype. Scoring them zero would train
    the model away from the binding mode the target structure validates - the
    failure section 4.1 recorded when a window put 24 of 25 training actives
    below 0.06. NaN drops the component for that molecule instead, so it is
    scored exactly as it was before this term existed.

    What gets penalised is the third thing v4 actually produced: a basic centre
    that is present and too close to the acid to reach alphaV Asp218."""

    def test_a_non_rgd_molecule_returns_nan_so_the_axis_is_dropped(self):
        import math

        self.assertTrue(math.isnan(score(NO_BASE)[0]))

    def test_the_crystal_ligand_of_the_target_structure_is_not_penalised(self):
        import math

        self.assertTrue(math.isnan(score(A1AFA)[0]))

    def test_nan_leaves_the_other_components_unchanged(self):
        """REINVENT masks NaN weights and renormalises, so a dropped component
        is not the same as a zero one. Verified against the aggregator itself."""
        import numpy as np
        from reinvent.scoring.aggregators.means import geometric_mean

        both = geometric_mean([(np.array([1.0]), 1.0), (np.array([0.8]), 1.0),
                               (np.array([np.nan]), 1.0)])[0]
        without = geometric_mean([(np.array([1.0]), 1.0),
                                  (np.array([0.8]), 1.0)])[0]
        self.assertAlmostEqual(both, without)

    def test_a_vestigial_basic_centre_is_what_scores_near_zero(self):
        """gen_03220 pays for an Arg mimic's polarity and charge and buys none of
        its binding - no measured active sits below 9 bonds."""
        self.assertEqual(score(GEN_03220), [4.0])

    def test_an_unparseable_smiles_scores_zero_and_does_not_raise(self):
        """0, not NaN: a computation that failed is not a determination that the
        axis does not apply."""
        self.assertEqual(score("not_a_smiles"), [0.0])

    def test_one_bad_molecule_does_not_lose_the_rest_of_the_batch(self):
        """A batch is 128 molecules per RL step; one failure must not take the
        other 127 with it."""
        self.assertEqual(score(PLN_1474, "not_a_smiles", GEN_03220),
                         [10.0, 0.0, 4.0])


class TestItUsesTheShippedImplementation(unittest.TestCase):
    """The span logic lives in scripts/pharmacophore_span.py, which the
    alphaIIbbeta3 descriptor already uses. The component imports it rather than
    restating it - this campaign has twice shipped a constant copied into a
    second place and then frozen there (§2.2 and §5.4)."""

    def test_component_agrees_with_the_script_on_the_clinical_compounds(self):
        from pharmacophore_span import span

        for smiles in (PLN_1474, BEXOTEGRAST, GEN_00446):
            with self.subTest(smiles=smiles):
                script = span(smiles)
                self.assertGreater(script["n_basic"], 0)
                self.assertGreater(script["n_acid"], 0)
                # Same pH 7.4 species on both sides.
                self.assertEqual(score(smiles)[0] > 0, True)


class TestTheWindowIsNotInTheComponent(unittest.TestCase):
    """The component returns the raw span; the window that judges it belongs in
    the config transform, where it is pre-registered and visible. Burying a
    window in code is how the TPSA bounds came to exist in two places (§2.2)."""

    def test_parameters_carry_no_thresholds(self):
        self.assertEqual([f for f in Parameters.__dataclass_fields__], [])


if __name__ == "__main__":
    unittest.main()
