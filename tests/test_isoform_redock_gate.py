import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from isoform_redock_gate import (RMSD_CUTOFF, midas_ligand_contact, verdict)


def metal_entry(chain="B", resnr=708, metal="Mn", location="ligand", dist=2.13):
    return {"reschain_lig": chain, "resnr_lig": str(resnr), "metal_type": metal,
            "location": location, "dist": dist, "restype": "LIG"}


class TestMidasLigandContact(unittest.TestCase):
    """Section 10.3 asks the same question of every isoform as section 8.0 asks
    of alphaVbeta1 -- does PLIP report the ligand coordinating the MIDAS ion --
    but the ion differs: Ca B501 in 8W30, Mn B708 in 6MK0, Mg B2102 in 9CZD.
    validate_plip's version is written against the calcium and would return
    False for both isoforms, excluding them for a reason that is not chemical."""

    MIDAS_AVB3 = ("MN", "B", 708)
    MIDAS_AVB6 = ("MG", "B", 2102)

    def test_the_isoform_metal_is_recognised(self):
        self.assertTrue(midas_ligand_contact([metal_entry()], self.MIDAS_AVB3))

    def test_another_metal_at_the_same_site_does_not_count(self):
        entry = metal_entry(metal="Ca")
        self.assertFalse(midas_ligand_contact([entry], self.MIDAS_AVB3))

    def test_another_site_does_not_count(self):
        # An ADMIDAS or SyMBS ion coordinating the ligand is not the MIDAS.
        entry = metal_entry(resnr=709)
        self.assertFalse(midas_ligand_contact([entry], self.MIDAS_AVB3))

    def test_a_protein_side_coordination_does_not_count(self):
        # The MIDAS is always coordinated by the protein; that says nothing
        # about whether the LIGAND reaches it.
        entry = metal_entry(location="protein.sidechain")
        self.assertFalse(midas_ligand_contact([entry], self.MIDAS_AVB3))

    def test_metal_names_compare_case_insensitively(self):
        # PLIP writes "Mn"; the manifest records "MN" from the PDB.
        self.assertTrue(midas_ligand_contact([metal_entry(metal="mn")],
                                             self.MIDAS_AVB3))

    def test_magnesium_isoform(self):
        entry = metal_entry(chain="B", resnr=2102, metal="Mg")
        self.assertTrue(midas_ligand_contact([entry], self.MIDAS_AVB6))

    def test_no_entries_is_a_clean_false(self):
        self.assertFalse(midas_ligand_contact([], self.MIDAS_AVB3))


class TestVerdict(unittest.TestCase):
    """Two criteria, both required (spec 10.3). An isoform that fails is
    EXCLUDED from the selectivity analysis and recorded -- the same rule that
    rejected AutoDock-GPU for alphaVbeta1, applied per receptor."""

    def test_passes_when_both_hold(self):
        self.assertTrue(verdict(rmsd=0.8, plip_midas=True)["passed"])

    def test_rmsd_at_the_cutoff_is_a_failure(self):
        v = verdict(rmsd=RMSD_CUTOFF, plip_midas=True)
        self.assertFalse(v["passed"])
        self.assertIn("rmsd", v["failed"])

    def test_missing_metal_coordination_fails(self):
        v = verdict(rmsd=0.5, plip_midas=False)
        self.assertFalse(v["passed"])
        self.assertIn("plip_midas", v["failed"])

    def test_an_uncomputable_rmsd_fails_rather_than_passing(self):
        # rmsd_to_reference returns None when the graphs do not match. A gate
        # that treated that as a pass would adopt a receptor on a measurement
        # that was never made.
        v = verdict(rmsd=None, plip_midas=True)
        self.assertFalse(v["passed"])
        self.assertIn("rmsd", v["failed"])

    def test_the_cutoff_matches_alphaVbeta1s(self):
        # Section 10.3: "alphaVbeta1과 동일한 기준이다."
        self.assertEqual(RMSD_CUTOFF, 2.0)


if __name__ == "__main__":
    unittest.main()
