import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validate_redock import (CRYSTAL_CONTACTS, best_pose, pose_table,
                             rmsd_to_reference, verdict)


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


class TestBestPose(unittest.TestCase):
    """spec 7.4 applies the gate to the TOP pose. AutoDock-GPU writes its
    DOCKED blocks in RUN order, not energy order: the control ligand's own
    .dlg opens -5.89, -6.23, -6.30, -6.30, -6.02, so taking poses[0] would
    gate on the worst of the first five. The top pose is selected here by
    affinity, explicitly."""

    def test_selects_the_lowest_affinity_not_the_first_block(self):
        poses = [{"rank": 1, "affinity": -5.89}, {"rank": 2, "affinity": -6.23},
                 {"rank": 3, "affinity": -6.30}, {"rank": 4, "affinity": -6.30},
                 {"rank": 5, "affinity": -6.02}]
        self.assertEqual(best_pose(poses)["rank"], 3)

    def test_ties_break_on_rank_so_selection_is_deterministic(self):
        # -6.30 appears twice above; a reproducible gate must not depend on
        # sort stability across Python versions.
        poses = [{"rank": 4, "affinity": -6.30}, {"rank": 3, "affinity": -6.30}]
        self.assertEqual(best_pose(poses)["rank"], 3)

    def test_poses_without_an_affinity_sort_last_but_are_not_dropped(self):
        # parse_dlg deliberately keeps a pose whose energy line failed to
        # parse (a v1 regression made that drop the pose count to zero); such
        # a pose must never win the selection, and must never be discarded.
        poses = [{"rank": 1, "affinity": None}, {"rank": 2, "affinity": -6.1}]
        self.assertEqual(best_pose(poses)["rank"], 2)

    def test_all_affinities_missing_still_returns_a_pose(self):
        poses = [{"rank": 2, "affinity": None}, {"rank": 1, "affinity": None}]
        self.assertEqual(best_pose(poses)["rank"], 1)

    def test_empty_pose_list_raises_rather_than_returning_none(self):
        # A .dlg with no DOCKED block is a docking failure, not a gate
        # failure; silently returning None would report it as the latter.
        with self.assertRaises(ValueError):
            best_pose([])


def _rec(rank, affinity, rmsd=1.0):
    return {"rank": rank, "affinity": affinity, "ca_dist": 2.6,
            "donor_dist": 2.8, "rmsd": rmsd, "plip_metal_ca501": True,
            "plip_hbond_asn224": True}


class TestPoseTable(unittest.TestCase):
    """--all-poses records every pose so a failed gate can say whether the
    engine missed by a little on every pose or only on the one that scored
    best. It is a RECORD, not a second gate: exactly one row is marked as
    the gated one, and it is the top pose by affinity."""

    def test_rows_are_ordered_best_affinity_first(self):
        rows = pose_table([_rec(1, -5.89), _rec(5, -6.77), _rec(3, -6.30)], 5)
        self.assertEqual([r["rank"] for r in rows], [5, 3, 1])

    def test_exactly_one_row_is_marked_as_the_gated_pose(self):
        rows = pose_table([_rec(1, -5.89), _rec(5, -6.77)], 5)
        self.assertEqual([r["rank"] for r in rows if r["gated"]], [5])

    def test_a_better_rmsd_on_another_pose_does_not_move_the_mark(self):
        # The whole point: pose 1 redocks better but scores worse. Marking it
        # would be choosing the pose after seeing the answer.
        rows = pose_table([_rec(1, -5.89, rmsd=0.6), _rec(5, -6.77, rmsd=2.15)], 5)
        self.assertTrue(next(r for r in rows if r["rank"] == 5)["gated"])
        self.assertFalse(next(r for r in rows if r["rank"] == 1)["gated"])


class TestRmsdRefusesMismatchedGraphs(unittest.TestCase):
    """CalcRMS minimises over every symmetry-equivalent atom mapping, so its
    result can never EXCEED the correct correspondence's RMSD. On the cluster
    it returned 2.151 A for a pose AutoDock itself measures at 1.46 A against
    the same reference -- because Open Babel re-protonated the carboxylate on
    the way out of the .dlg, so the pose read back as ...C(=O)O while the
    reference SDF was ...C(=O)[O-]. Zero substructure matches, and a number
    returned anyway. A number that decides a pre-registered criterion must not
    come from a correspondence that does not exist."""

    @staticmethod
    def _mol(smiles):
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        return mol

    def test_no_number_when_the_graphs_differ(self):
        rmsd, note = rmsd_to_reference(self._mol("CC(=O)O"),
                                       self._mol("CC(=O)[O-]"))
        self.assertIsNone(rmsd)
        self.assertIn("not the same molecular graph", note)

    def test_the_note_names_both_graphs(self):
        # The cluster failure was invisible precisely because nothing printed
        # what the two molecules actually were.
        _rmsd, note = rmsd_to_reference(self._mol("CC(=O)O"),
                                        self._mol("CC(=O)[O-]"))
        self.assertIn("C(=O)O", note)
        self.assertIn("[O-]", note)

    def test_matching_graphs_still_produce_a_number(self):
        mol = self._mol("CC(=O)O")
        rmsd, note = rmsd_to_reference(mol, mol)
        self.assertIsNone(note)
        self.assertAlmostEqual(rmsd, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
