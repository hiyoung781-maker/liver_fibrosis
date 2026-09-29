import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_redock import CRYSTAL_CONTACTS, verdict


def m(ca=2.68, donor=2.92, rmsd=0.63, plip_metal=True, plip_hbond=True):
    return {"ca_dist": ca, "donor_dist": donor, "rmsd": rmsd,
            "plip_metal_ca501": plip_metal, "plip_hbond_asn224": plip_hbond}


class TestRedockGate(unittest.TestCase):
    """spec §7.4. If this gate does not pass, AutoDock-GPU filters nothing."""

    def test_passes_when_all_four_criteria_are_met(self):
        self.assertTrue(verdict(m())["passed"])

    def test_fails_on_rmsd(self):
        """2.5 A >= RMSD_CUTOFF (2.0): rdMolAlign.CalcRMS must be symmetry-aware,
        or a real 0.63 A success gets reported as a 6+ A failure (see
        pose_geometry's docstring for the v1 post-mortem)."""
        v = verdict(m(rmsd=2.5))
        self.assertFalse(v["passed"])
        self.assertIn("rmsd", v["failed"])

    def test_fails_when_plip_does_not_see_the_metal_complex(self):
        v = verdict(m(plip_metal=False))
        self.assertFalse(v["passed"])
        self.assertIn("plip_metal_ca501", v["failed"])

    def test_distance_plip_disagreement_is_recorded(self):
        """A pose can sit 2.7 A from Asn224 backbone O (inside DONOR_CUTOFF=3.5)
        and still not be a hydrogen bond, because distance criteria don't check
        angles and PLIP does. When they disagree, record it and follow PLIP."""
        v = verdict(m(donor=2.7, plip_hbond=False))
        self.assertFalse(v["passed"])
        self.assertIn("donor_dist", v["disagreements"])

    def test_documented_crystal_values(self):
        """8W30, measured with PLIP 3.0.1 on the deposited structure: Ca 2.62 A,
        Asn224 donor-acceptor 2.63 A (ligand donates, not the protein)."""
        self.assertAlmostEqual(CRYSTAL_CONTACTS["ca_dist"], 2.62)
        self.assertAlmostEqual(CRYSTAL_CONTACTS["donor_dist"], 2.63)


if __name__ == "__main__":
    unittest.main()
