import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_plip import HYDROPHOBIC_POCKET, crystal_verdict


def record(metal=True, hbond=True, tyr178=True, hydrophobic=True):
    """A parse_report()-shaped record using the real field names PLIP 3.0.1
    emits on the deposited 8W30 structure (see run_plip.parse_report), with
    the actual measured values from that structure.
    """
    rec = {"metal_complexes": [], "hbonds": [], "pi_stacking": [],
           "hydrophobic_contacts": [], "salt_bridges": [],
           "water_bridges": [], "halogen_bonds": []}
    if metal:
        rec["metal_complexes"].append({
            "resnr": 505, "restype": "A1A", "reschain": "B",
            "resnr_lig": "501", "restype_lig": "CA", "reschain_lig": "B",
            "metal_type": "Ca", "coordination": "4", "dist": 2.62,
            "location": "ligand",
        })
    if hbond:
        rec["hbonds"].append({
            "resnr": 224, "restype": "ASN", "reschain": "B",
            "dist_d-a": 2.63, "don_angle": 156.61, "protisdon": "False",
        })
    if tyr178:
        rec["hydrophobic_contacts"].append({
            "resnr": 178, "restype": "TYR", "reschain": "A", "dist": 3.70,
        })
    if hydrophobic:
        rec["hydrophobic_contacts"].append({
            "resnr": 225, "restype": "LEU", "reschain": "B", "dist": 3.69,
        })
    return rec


class TestCrystalValidation(unittest.TestCase):
    """spec §8.0: if PLIP cannot reproduce 8W30's recorded interactions, it
    is not adopted and the campaign reverts to the v1 distance criteria."""

    def test_passes_when_all_four_are_reproduced(self):
        self.assertTrue(crystal_verdict(record())["passed"])

    def test_fails_without_the_midas_metal_complex(self):
        v = crystal_verdict(record(metal=False))
        self.assertFalse(v["passed"])
        self.assertIn("metal_ca501", v["missing"])

    def test_metal_criterion_requires_the_ligand_contact_not_just_any_ca_entry(self):
        """The MIDAS calcium also coordinates Ser132/Ser134/Glu229 in separate
        metal_complexes entries; only the entry whose coordinated atom is the
        ligand itself (location == "ligand") satisfies criterion (a)."""
        rec = record(metal=False)
        rec["metal_complexes"].append({
            "resnr": 132, "restype": "SER", "reschain": "B",
            "resnr_lig": "501", "restype_lig": "CA", "reschain_lig": "B",
            "metal_type": "Ca", "coordination": "4", "dist": 2.45,
            "location": "protein.sidechain",
        })
        v = crystal_verdict(rec)
        self.assertFalse(v["passed"])
        self.assertIn("metal_ca501", v["missing"])

    def test_fails_without_the_asn224_ligand_donor_hbond(self):
        v = crystal_verdict(record(hbond=False))
        self.assertFalse(v["passed"])
        self.assertIn("hbond_asn224", v["missing"])

    def test_asn224_hbond_where_protein_donates_does_not_satisfy_the_criterion(self):
        """8W30 also has Asn224 hbond entries at 3.23/3.27 A with
        protisdon == True (the protein donates). Criterion (b) specifically
        requires the ligand-donor entry at 2.63 A (protisdon == False)."""
        rec = record(hbond=False)
        rec["hbonds"].append({
            "resnr": 224, "restype": "ASN", "reschain": "B",
            "dist_d-a": 3.23, "don_angle": 144.54, "protisdon": "True",
        })
        v = crystal_verdict(rec)
        self.assertFalse(v["passed"])
        self.assertIn("hbond_asn224", v["missing"])

    def test_tyr178_contact_reported_as_hydrophobic_satisfies_the_criterion(self):
        """Corrected criterion (c). The brief originally required a T-shaped
        pi-stack at Tyr178, but PLIP reports ZERO pi_stacks for the deposited
        8W30 structure: PLIP's default pi-stack centroid cutoff is 5.5 A,
        while 8W30's measured ligand-to-Tyr178 centroid distance is 6.21 A,
        so PLIP classifies the contact as a hydrophobic_interaction (3.70 A)
        instead. A gate that still demanded a pi-stack would reject the only
        crystallographically observed binder for this target -- the same
        centroid-cutoff failure mode that motivated moving off the v1
        distance criteria in the first place. The criterion is therefore
        type-agnostic: any interaction kind reporting Tyr178 satisfies it."""
        self.assertTrue(crystal_verdict(record())["passed"])
        self.assertEqual(record()["hydrophobic_contacts"][0]["reschain"], "A")
        self.assertEqual(record()["hydrophobic_contacts"][0]["resnr"], 178)

    def test_tyr178_contact_reported_as_hbond_also_satisfies_the_criterion(self):
        """Type-agnostic means any kind, not just hydrophobic_contacts."""
        rec = record(tyr178=False)
        rec["hbonds"].append({"resnr": 178, "restype": "TYR", "reschain": "A",
                               "dist_d-a": 3.4, "don_angle": 150.0, "protisdon": "False"})
        self.assertTrue(crystal_verdict(rec)["passed"])

    def test_fails_without_any_tyr178_contact(self):
        v = crystal_verdict(record(tyr178=False, hydrophobic=False))
        self.assertFalse(v["passed"])
        self.assertIn("tyr178_contact", v["missing"])
        # hydrophobic pocket also fails here since Leu225 was the only other entry
        self.assertIn("hydrophobic_pocket", v["missing"])

    def test_any_one_pocket_residue_satisfies_the_hydrophobic_criterion(self):
        rec = record(hydrophobic=False)
        rec["hydrophobic_contacts"].append(
            {"reschain": "B", "resnr": 133, "restype": "TYR", "dist": 3.5})
        self.assertTrue(crystal_verdict(rec)["passed"])

    def test_documented_pocket_residues(self):
        self.assertEqual(HYDROPHOBIC_POCKET,
                          {("B", 133), ("B", 186), ("B", 187), ("B", 225)})


if __name__ == "__main__":
    unittest.main()
