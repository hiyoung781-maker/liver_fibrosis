import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from selectivity_observations import (
    ASP218_CUTOFF,
    FEATURES,
    LEU225_CUTOFF,
    PARALLEL_MAX,
    T_SHAPED_MIN,
    TYR178_ATOM_CUTOFF,
    TYR178_CENTROID_REFERENCE,
    observe,
    observe_file,
    receptor_features,
)

RECEPTOR = "docking/receptor.pdb"
CRYSTAL = "docking/ligand_ref.sdf"
REDOCK = "docking/redock_poses.sdf"


class TestFeatureDefinition(unittest.TestCase):
    def test_subunit_assignment(self):
        """Leu225 is beta1 (chain B); Asp218 and Tyr178 are alphaV (chain A). Getting
        this wrong is silent - an earlier analysis here searched chain A for the MIDAS
        calcium and measured 40-51 A."""
        self.assertEqual(FEATURES["leu225"][0], "B")
        self.assertEqual(FEATURES["asp218"][0], "A")
        self.assertEqual(FEATURES["tyr178"][0], "A")

    def test_leu225_uses_side_chain_only(self):
        """The side chain is the whole point: GLPG0187 and CWHM-12 reach Leu225's
        BACKBONE with the same scaffold and are still pan-alphaV."""
        atoms = FEATURES["leu225"][2]
        self.assertEqual(set(atoms), {"CB", "CG", "CD1", "CD2"})
        for backbone in ("N", "CA", "C", "O"):
            self.assertNotIn(backbone, atoms)

    def test_tyr178_uses_the_phenol_ring(self):
        self.assertEqual(set(FEATURES["tyr178"][2]),
                         {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"})

    def test_cutoffs(self):
        self.assertEqual(LEU225_CUTOFF, 4.5)
        self.assertEqual(ASP218_CUTOFF, 4.0)
        self.assertEqual(TYR178_ATOM_CUTOFF, 4.5)
        self.assertEqual((PARALLEL_MAX, T_SHAPED_MIN), (30.0, 60.0))


class TestReceptorFeatures(unittest.TestCase):
    def test_finds_all_three_complete(self):
        feats = receptor_features(RECEPTOR)
        for key, (_, _, names) in FEATURES.items():
            self.assertEqual(len(feats[key]), len(names), key)

    def test_incomplete_receptor_raises(self):
        """Silence here would make every pose report no contact, which reads as
        chemistry rather than a bad receptor."""
        with self.assertRaises(ValueError):
            receptor_features(CRYSTAL)


class TestCrystalStructureReproducesTheRecordedValues(unittest.TestCase):
    """The blueprint records 8W30's measurements; this script must reproduce them or it
    is measuring something else."""

    @classmethod
    def setUpClass(cls):
        cls.row = observe(Chem.MolFromMolFile(CRYSTAL, removeHs=False),
                          receptor_features(RECEPTOR))

    def test_tyr178_nearest_atom_is_370(self):
        self.assertAlmostEqual(self.row["tyr178_nearest_atom"], 3.70, places=2)

    def test_tyr178_centroid_is_621(self):
        self.assertAlmostEqual(self.row["tyr178_centroid_dist"], 6.21, places=2)

    def test_tyr178_ring_angle_is_746(self):
        self.assertAlmostEqual(self.row["tyr178_ring_angle"], 74.6, places=0)

    def test_tyr178_geometry_is_t_shaped(self):
        self.assertEqual(self.row["tyr178_geometry"], "T_shaped")

    def test_asp218_any_atom_is_700(self):
        """Section 1's pocket table records 'the 8W30 hit does not engage (7.0 A)'.
        That is the any-atom distance, an aromatic CH - not a donor. Reporting only
        the donor distance left that documented number irreproducible here."""
        self.assertAlmostEqual(self.row["asp218_any_atom_dist"], 7.00, places=2)

    def test_the_nearest_donor_is_much_further_than_the_nearest_atom(self):
        self.assertGreater(self.row["asp218_donor_dist"],
                           self.row["asp218_any_atom_dist"])


class TestCriterionMustAdmitTheCrystalStructure(unittest.TestCase):
    """Section 8.4's original criterion (iii) - 'centroid within 5.5 A' - REJECTS the
    crystal structure, whose centroid distance is 6.21 A.

    That is the only crystallographically observed instance of this interaction, so a
    test that rejects it is measuring the wrong thing. The cause is geometric: a
    T-shaped stack holds the rings perpendicular, pushing the centroids apart while
    the closest atoms stay in contact, so a centroid cutoff assumes parallel stacking.
    8W30 is T-shaped at 74.5 degrees. The criterion therefore uses the nearest ring
    atom, where the crystal's 3.70 A sits inside 4.5 A.
    """

    @classmethod
    def setUpClass(cls):
        cls.row = observe(Chem.MolFromMolFile(CRYSTAL, removeHs=False),
                          receptor_features(RECEPTOR))

    def test_crystal_passes_criterion_iii(self):
        self.assertEqual(self.row["tyr178_stack"], 1)

    def test_crystal_passes_criterion_i(self):
        self.assertEqual(self.row["leu225_contact"], 1)
        self.assertLess(self.row["leu225_min_dist"], LEU225_CUTOFF)

    def test_crystal_makes_no_asp218_contact_as_documented(self):
        """Section 8.4(ii) exists to distinguish HOW a ligand reaches Asp218. The
        crystal ligand does not reach it at all, which is why the blueprint says no
        selectivity contact has been observed crystallographically."""
        self.assertEqual(self.row["asp218_neutral_donor"], 0)
        self.assertEqual(self.row["asp218_basic_contact"], 0)
        self.assertEqual(self.row["arg_mimic_heads"], "")

    def test_the_old_centroid_criterion_would_have_rejected_it(self):
        """Documents the defect so the criterion is not "simplified" back."""
        self.assertGreater(self.row["tyr178_centroid_dist"], 5.5)
        self.assertAlmostEqual(TYR178_CENTROID_REFERENCE, 6.21, places=2)


class TestArgMimicDefinitionIsShared(unittest.TestCase):
    def test_imported_from_curate_actives_not_restated(self):
        """Section 4 measures an Arg-mimic percentage with these patterns. A second
        copy here would let the two drift apart."""
        import selectivity_observations as so
        from curate_actives import ARG_MIMIC_HEADS

        self.assertEqual(set(so._arg_mimic_patterns()), set(ARG_MIMIC_HEADS))

    def test_a_tetrahydronaphthyridine_is_detected(self):
        """THN is the motif Sabat's paper ties to pan-alphaV activity."""
        import selectivity_observations as so

        thn = Chem.MolFromSmiles(
            "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1")  # PLN-1474
        hit = any(thn.HasSubstructMatch(p)
                  for p in so._arg_mimic_patterns().values())
        self.assertTrue(hit)


class TestObserveFile(unittest.TestCase):
    def test_writes_a_row_per_pose_with_counts(self):
        with TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "sel.csv")
            counts = observe_file(REDOCK, RECEPTOR, out)
            self.assertEqual(counts["poses"], 20)
            lines = Path(out).read_text().strip().splitlines()
            self.assertEqual(len(lines), 21)   # header + 20

    def test_counts_never_exceed_the_pose_count(self):
        with TemporaryDirectory() as tmp:
            counts = observe_file(REDOCK, RECEPTOR, str(Path(tmp) / "s.csv"))
            for key in ("leu225", "tyr178", "leu225_and_tyr178", "asp218_reached",
                        "asp218_neutral", "asp218_basic"):
                self.assertLessEqual(counts[key], counts["poses"], key)

    def test_there_is_no_all_three_conjunction(self):
        """(ii) asks about the CHARACTER of an Asp218 contact, and the basic case is the
        one tied to pan-alphaV activity - so multiplying it into a merit score points
        the wrong way. A pose that never reaches Asp218 has failed nothing; the crystal
        ligand sits 7.00 A away. Measured on the real passing set, only 5 of 4,065
        poses reach it at all, so any conjunction including (ii) is ~0 by arithmetic
        and says nothing about the molecules."""
        with TemporaryDirectory() as tmp:
            counts = observe_file(REDOCK, RECEPTOR, str(Path(tmp) / "s.csv"))
            self.assertNotIn("all_three", counts)
            self.assertIn("leu225_and_tyr178", counts)

    def test_asp218_reached_is_the_any_atom_test_not_the_donor_test(self):
        """Whether a ligand reaches Asp218 at all is an any-atom question; whether it
        does so with a donor or a basic head is the character question."""
        row = observe(Chem.MolFromMolFile(CRYSTAL, removeHs=False),
                      receptor_features(RECEPTOR))
        self.assertEqual(row["asp218_reached"], 0)
        self.assertGreater(row["asp218_any_atom_dist"], 4.0)

    def test_the_filter_discriminates_among_redocked_poses(self):
        """All 20 poses passing, or none, would mean the measurement is not measuring."""
        with TemporaryDirectory() as tmp:
            counts = observe_file(REDOCK, RECEPTOR, str(Path(tmp) / "s.csv"))
            self.assertGreater(counts["leu225"], 0)
            self.assertLess(counts["leu225"], counts["poses"])


if __name__ == "__main__":
    unittest.main()
