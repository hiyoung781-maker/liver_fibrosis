import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_isoform import (METAL_NAMES, binding_site_residues, chains_near,
                             find_midas_ligand, split_structure)

STRUCTURES = {"8W30": "8W30.pdb", "6MK0": "structures/6MK0.pdb",
              "9CZD": "structures/9CZD.pdb"}


def lines(pdb):
    return Path(STRUCTURES[pdb]).read_text().splitlines()


def have(pdb):
    return Path(STRUCTURES[pdb]).exists()


@unittest.skipUnless(have("8W30"), "8W30.pdb not present")
class TestFindMidasLigandOn8W30(unittest.TestCase):
    """The detection has to reproduce, on the structure whose answer is
    already known, what was measured by hand: A1A B505 sits 2.62 A from the
    MIDAS calcium Ca B501. Hardcoding a ligand code per isoform would not
    survive the next structure; hardcoding the wrong one would fail every
    pose for a reason that looks chemical."""

    def test_finds_the_deposited_ligand_and_the_midas_metal(self):
        found = find_midas_ligand(lines("8W30"))
        self.assertEqual(found["ligand"], ("A1A", "B", 505))
        self.assertEqual(found["metal"], ("CA", "B", 501))
        self.assertAlmostEqual(found["distance"], 2.62, places=1)

    def test_it_is_not_the_other_calcium(self):
        # Ca B502 is 6.44 A away and does not participate; chain A's four are
        # 35-50 A off. Selecting one of those would move the whole box.
        self.assertNotEqual(find_midas_ligand(lines("8W30"))["metal"][2], 502)


@unittest.skipUnless(have("6MK0") and have("9CZD"), "isoform structures absent")
class TestFindMidasLigandOnIsoforms(unittest.TestCase):
    """The two isoforms that survived the section 10.2 review use different
    metals from 8W30 -- Mn in 6MK0, Mg in 9CZD -- so a calcium-specific rule
    would find nothing."""

    def test_6mk0_avb3(self):
        found = find_midas_ligand(lines("6MK0"))
        self.assertEqual(found["ligand"][0], "JUY")
        self.assertEqual(found["metal"][0], "MN")
        self.assertLess(found["distance"], 3.5)

    def test_9czd_avb6(self):
        found = find_midas_ligand(lines("9CZD"))
        self.assertEqual(found["ligand"][0], "A1A")
        self.assertEqual(found["metal"][0], "MG")
        self.assertLess(found["distance"], 3.5)

    def test_every_metal_this_project_meets_is_recognised(self):
        for name in ("CA", "MG", "MN"):
            self.assertIn(name, METAL_NAMES)


@unittest.skipUnless(have("9CZD"), "9CZD absent")
class TestChainSelection(unittest.TestCase):
    """9CZD carries a 17E6 Fab as a crystallisation chaperone. Its heavy
    chain reaches 6.08 A of the ligand -- close enough that a proximity rule
    keeps it, far enough that it touches nothing. Keeping it would build this
    receptor differently from 8W30's and 6MK0's, which have only their two
    integrin chains, and the Delta axis depends on the two receptors
    differing in the protein and nothing else."""

    def test_only_the_integrin_chains_are_kept(self):
        found = find_midas_ligand(lines("9CZD"))
        near = chains_near(lines("9CZD"), found["ligand"])
        self.assertEqual(near, {"A", "B"})

    def test_a_proximity_rule_would_have_kept_the_fab(self):
        # Pins the reason the default is a contact distance: at 12 A the Fab
        # heavy chain comes back, and nothing in the output would say so.
        found = find_midas_ligand(lines("9CZD"))
        self.assertIn("D", chains_near(lines("9CZD"), found["ligand"], 12.0))


@unittest.skipUnless(have("8W30"), "8W30.pdb absent")
class TestSplitAndSite(unittest.TestCase):
    def test_split_removes_the_ligand_and_the_waters_from_the_receptor(self):
        found = find_midas_ligand(lines("8W30"))
        receptor, ligand = split_structure(lines("8W30"), found["ligand"],
                                           keep_chains={"A", "B"})
        self.assertTrue(ligand)
        self.assertFalse([l for l in receptor if l[17:20].strip() == "A1A"])
        self.assertFalse([l for l in receptor if l[17:20].strip() == "HOH"])

    def test_the_midas_metal_stays_in_the_receptor(self):
        # Removing it would delete the interaction the whole campaign is about.
        found = find_midas_ligand(lines("8W30"))
        receptor, _lig = split_structure(lines("8W30"), found["ligand"],
                                         keep_chains={"A", "B"})
        self.assertTrue([l for l in receptor
                         if l[17:20].strip() == "CA" and l[21] == "B"
                         and l[22:26].strip() == "501"])

    def test_binding_site_residues_include_the_known_anchors(self):
        # beta1-Asn224 and the MIDAS serines are the documented contacts.
        found = find_midas_ligand(lines("8W30"))
        site = binding_site_residues(lines("8W30"), found["ligand"], cutoff=5.0)
        self.assertIn(("B", 224), site)
        self.assertIn(("B", 132), site)


if __name__ == "__main__":
    unittest.main()
