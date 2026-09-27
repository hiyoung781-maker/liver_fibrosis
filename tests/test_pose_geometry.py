import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pose_geometry import (
    CA_CUTOFF,
    DONOR_CUTOFF,
    anchor_atoms,
    crystal_rmsd,
    measure_pose,
    passes,
)

RECEPTOR = "docking/receptor.pdb"
CRYSTAL = "docking/ligand_ref.sdf"
REDOCK = "docking/redock_poses.sdf"


class TestCutoffs(unittest.TestCase):
    def test_cutoffs_are_the_documented_ones(self):
        """§8.5's thresholds, with their derivation recorded in the blueprint:
        3.2 Å sits in the gap between Ca501's first shell (ends 2.62 Å) and the
        next atom (3.30 Å); 3.5 Å sits 0.85 Å above Asn224 O's 2.2-2.7 Å band."""
        self.assertEqual(CA_CUTOFF, 3.2)
        self.assertEqual(DONOR_CUTOFF, 3.5)


class TestAnchorAtoms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.anchors = anchor_atoms(RECEPTOR)

    def test_finds_the_midas_calcium_in_chain_b(self):
        """MIDAS is on the β subunit. Ca501 is 2.62 Å from the crystal ligand;
        Ca502 is 6.44 Å and does not participate, and the chain-A calciums are
        35-50 Å away. Picking the wrong one silently fails every pose."""
        self.assertIn("ca501", self.anchors)
        self.assertEqual(self.anchors["ca501"].shape, (3,))

    def test_finds_asn224_backbone_oxygen_in_chain_b(self):
        self.assertIn("asn224_o", self.anchors)

    def test_missing_anchor_raises(self):
        """A receptor edited to drop the calcium would otherwise make every pose
        fail the filter for a reason that looks chemical."""
        with self.assertRaises(ValueError):
            anchor_atoms(CRYSTAL)


class TestCrystalLigandPassesItsOwnFilter(unittest.TestCase):
    """The filter must accept the structure it was derived from. If the crystal
    pose fails, the thresholds are wrong, not the pose."""

    def test_crystal_pose_satisfies_both_criteria(self):
        mol = Chem.MolFromMolFile(CRYSTAL, removeHs=False)
        result = measure_pose(mol, anchor_atoms(RECEPTOR))
        self.assertLess(result["ca_dist"], CA_CUTOFF)
        self.assertLess(result["donor_dist"], DONOR_CUTOFF)
        self.assertTrue(passes(result))

    def test_measured_distances_match_the_recorded_crystal_values(self):
        """Blueprint §8.5 records 2.62 Å and 2.63 Å for the crystal contact."""
        mol = Chem.MolFromMolFile(CRYSTAL, removeHs=False)
        result = measure_pose(mol, anchor_atoms(RECEPTOR))
        self.assertAlmostEqual(result["ca_dist"], 2.62, places=1)
        self.assertAlmostEqual(result["donor_dist"], 2.63, places=1)

    def test_monodentate_judging_the_far_oxygen_is_not_required(self):
        """§8.5: only OXT is at 2.62 Å; the other carboxylate O is 4.54 Å. The
        criterion is 'either oxygen within 3.2 Å'. Requiring both would fail the
        crystal structure itself."""
        mol = Chem.MolFromMolFile(CRYSTAL, removeHs=False)
        result = measure_pose(mol, anchor_atoms(RECEPTOR))
        self.assertGreater(result["ca_dist_far"], 4.0)


class TestRedockedPoses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.poses = [m for m in Chem.SDMolSupplier(REDOCK, removeHs=False)
                     if m is not None]
        cls.anchors = anchor_atoms(RECEPTOR)

    def test_reads_all_twenty_poses(self):
        self.assertEqual(len(self.poses), 20)

    def test_top_pose_passes_and_reproduces_the_recorded_values(self):
        """§8.5's validation table: pose 1 gives 2.72 Å / 2.89 Å / RMSD 0.63 Å."""
        result = measure_pose(self.poses[0], self.anchors)
        self.assertAlmostEqual(result["ca_dist"], 2.72, places=1)
        self.assertAlmostEqual(result["donor_dist"], 2.89, places=1)
        self.assertTrue(passes(result))

    def test_some_poses_fail_so_the_filter_discriminates(self):
        """A filter that passes all 20 poses would not be filtering. smina
        generated poses across the whole box; most should not coordinate Ca501."""
        verdicts = [passes(measure_pose(p, self.anchors)) for p in self.poses]
        self.assertTrue(any(verdicts))
        self.assertFalse(all(verdicts))


class TestCrystalRmsd(unittest.TestCase):
    def test_symmetry_aware_rmsd_reproduces_063(self):
        """The recorded lesson: a naive index-wise RMSD gives 6.13 Å for this same
        pose because smina/obabel reorder atoms and the molecule has symmetric
        phenyl and dichlorophenyl rings. CalcRMS gives 0.63 Å. Using the naive
        number turns a passed validation into an apparent failure."""
        ref = Chem.MolFromMolFile(CRYSTAL, removeHs=False)
        pose = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))
        self.assertAlmostEqual(crystal_rmsd(pose, ref), 0.63, places=1)

    def test_naive_index_rmsd_is_much_worse_proving_the_hazard(self):
        """Two separate hazards, both real, demonstrated in order.

        First, the files do not even agree on atom count - the crystal reference
        carries 22 atoms and the smina pose 24 - so a coordinate subtraction
        raises before it can mislead. That one is loud.

        The dangerous one is quiet: strip hydrogens so both sides have 22 heavy
        atoms, and an index-wise RMSD still reports several angstroms, because
        smina/obabel reorder atoms and the ligand has symmetric phenyl and
        dichlorophenyl rings. CalcRMS resolves the mapping and reports 0.63 A.
        """
        ref = Chem.MolFromMolFile(CRYSTAL, removeHs=False)
        pose = next(iter(Chem.SDMolSupplier(REDOCK, removeHs=False)))

        with self.assertRaises(ValueError):
            a = ref.GetConformer().GetPositions()
            b = pose.GetConformer().GetPositions()
            (a - b) ** 2

        ref_heavy, pose_heavy = Chem.RemoveHs(ref), Chem.RemoveHs(pose)
        self.assertEqual(ref_heavy.GetNumAtoms(), pose_heavy.GetNumAtoms())
        a = ref_heavy.GetConformer().GetPositions()
        b = pose_heavy.GetConformer().GetPositions()
        naive = float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))
        self.assertGreater(naive, 3.0)
        self.assertLess(crystal_rmsd(pose, ref), 1.0)


if __name__ == "__main__":
    unittest.main()
