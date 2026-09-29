import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from poses_to_complex import (LIGAND_CHAIN, MIDAS, MIDAS_LINK_CUTOFF,
                              build_complex, build_many, midas_link_record)

RECEPTOR = "docking/v2/receptor_h.pdb"
GOOD_POSE = "tests/fixtures/pose_ok.pdbqt"
BROKEN_POSE = "tests/fixtures/pose_broken.pdbqt"
# The deposited 8W30 ligand, in the receptor's own frame: its OXT sits
# 2.62 A from Ca B501, the distance the crystal gate is written against.
MIDAS_POSE = "tests/fixtures/pose_midas.pdbqt"
# The same pose with the chain column blanked and resSeq 1 -- how a pose
# actually arrives from an AutoDock-GPU .dlg. The chain-bearing fixture above
# hid a real defect for exactly this reason, so both are exercised.
MIDAS_POSE_NOCHAIN = "tests/fixtures/pose_midas_nochain.pdbqt"


class TestComplexBuilding(unittest.TestCase):
    """PLIP computes donor-H...acceptor angles, so it needs explicit
    hydrogens. PDBQT merges nonpolar hydrogens into heavy atoms, so it
    cannot be handed to PLIP as-is."""

    def test_complex_contains_both_receptor_and_ligand(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(GOOD_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            text = Path(out).read_text()
            # LIG marks the ligand residue name; ATOM marks receptor atoms.
            # Both must be present for PLIP to see a binding site at all.
            self.assertIn("LIG", text)
            self.assertIn("ATOM", text)

    def test_ligand_has_explicit_hydrogens(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(GOOD_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            lig_lines = [l for l in Path(out).read_text().splitlines()
                         if "LIG" in l and l.startswith("HETATM")]
            # Without explicit H, PLIP cannot compute donor angles and its
            # hydrogen-bond verdicts silently change (documented failure:
            # 1,170/1,212 protein N atoms were mistyped as acceptors).
            self.assertTrue(any(l[76:78].strip() == "H" for l in lig_lines))


class TestFailureIsolation(unittest.TestCase):
    """One pose failing to convert must not stop a batch of ~85,000 poses
    from finishing; failures are recorded and the rest proceed."""

    def test_broken_pose_returns_none_instead_of_raising(self):
        with TemporaryDirectory() as tmp:
            self.assertIsNone(
                build_complex(BROKEN_POSE, RECEPTOR, Path(tmp) / "c.pdb"))

    def test_build_many_reports_failures_and_keeps_going(self):
        with TemporaryDirectory() as tmp:
            result = build_many([GOOD_POSE, BROKEN_POSE, GOOD_POSE],
                                RECEPTOR, Path(tmp))
            self.assertEqual(len(result["built"]), 2)
            self.assertEqual(len(result["failed"]), 1)


class TestMidasLinkRecord(unittest.TestCase):
    """PLIP merges a ligand and a metal into one composite binding site ONLY
    when the input PDB carries a LINK record joining them (preparation.py
    parses LINK at line 129 and clusters on it in identify_kmers). Without
    that line the calcium becomes its own binding site, and a ligand atom in
    a different site is never a candidate target for that site's metal - so
    the MIDAS coordination is reported NOWHERE. Measured on the deposited
    ligand: no LINK gives the LIG site 0 metal_complexes and the Ca B501 site
    coordination 3 (Ser132 2.45, Ser134 2.50, Glu229 2.39, ligand absent);
    with LINK the site becomes LIG-CA with coordination 4 and the ligand
    contact at 2.62 A, reproducing the crystal reference exactly."""

    def test_pose_at_midas_gets_a_link_record(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            links = [l for l in Path(out).read_text().splitlines()
                     if l.startswith("LINK")]
            self.assertEqual(len(links), 1)

    def test_link_record_columns_are_what_plip_parses(self):
        # PLIP's get_linkage reads these exact slices; a shifted column
        # silently yields a link between the wrong residues, or a crash.
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            link = next(l for l in Path(out).read_text().splitlines()
                        if l.startswith("LINK"))
            self.assertEqual(link[17:20].strip(), "LIG")
            self.assertEqual(link[21], LIGAND_CHAIN)
            # The residue number is assigned by build_complex, not inherited
            # from the pose file; TestLigandChainIsExplicit pins that it
            # matches the ligand's own atoms and collides with nothing.
            self.assertTrue(link[22:26].strip().isdigit())
            self.assertEqual(link[47:50].strip(), "CA")
            self.assertEqual(link[51], MIDAS[0])
            self.assertEqual(int(link[52:56]), MIDAS[1])

    def test_link_precedes_the_coordinate_records(self):
        # PLIP reads the whole file, but a LINK after the ATOM records is
        # not a valid PDB and other tools in this pipeline do care.
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            lines = Path(out).read_text().splitlines()
            first_coord = next(i for i, l in enumerate(lines)
                               if l.startswith(("ATOM", "HETATM")))
            link_idx = next(i for i, l in enumerate(lines)
                            if l.startswith("LINK"))
            self.assertLess(link_idx, first_coord)

    def test_pose_far_from_the_midas_gets_no_link(self):
        # pose_ok sits 124 A from Ca B501. Emitting a LINK there would
        # fabricate a composite ligand out of two unrelated molecules.
        with TemporaryDirectory() as tmp:
            out = build_complex(GOOD_POSE, RECEPTOR, Path(tmp) / "c.pdb")
            self.assertFalse([l for l in Path(out).read_text().splitlines()
                              if l.startswith("LINK")])

    def test_cutoff_is_permissive_relative_to_plips_own_metal_distance(self):
        # PLIP's METAL_DIST_MAX is 3.0 A. The LINK only makes the ligand
        # atom VISIBLE as a candidate; PLIP still applies its own cutoff and
        # geometry, so a trigger window above 3.0 can never manufacture an
        # interaction PLIP would not independently confirm.
        self.assertGreater(MIDAS_LINK_CUTOFF, 3.0)

    def test_midas_link_record_returns_none_when_receptor_lacks_the_calcium(self):
        lig = ["HETATM    1  O   LIG B 505       6.057 118.151  44.471"
               "  1.00  0.00           O  "]
        self.assertIsNone(midas_link_record(lig, []))


class TestLigandChainIsExplicit(unittest.TestCase):
    """A docked pose arrives with no chain id. Open Babel then relabels the
    chain-less HETATM residue as chain Z, while a LINK written from the
    original columns says chain "" -- and PLIP's identify_kmers looks the
    residue up as (name, chain, num), does not find ('LIG', '', 1) against
    ('LIG', 'Z', 1), and DROPS the link without a word. Observed on the
    cluster: the LINK was present in the complex, the sites still came back
    as separate LIG (chain Z) and CA (chain B), and criterion (a) of the
    section 7.4 gate failed at a 2.57 A contact -- inside PLIP's own 3.0 A
    METAL_DIST_MAX. build_complex therefore assigns the chain and residue
    number itself instead of inheriting whatever the pose file carried."""

    def test_ligand_lines_carry_an_explicit_chain(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE_NOCHAIN, RECEPTOR, Path(tmp) / "c.pdb")
            lig = [l for l in Path(out).read_text().splitlines()
                   if l.startswith("HETATM") and l[17:20] == "LIG"]
            self.assertTrue(lig)
            self.assertEqual({l[21] for l in lig}, {LIGAND_CHAIN})

    def test_ligand_residue_number_does_not_collide_with_the_receptor(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE_NOCHAIN, RECEPTOR, Path(tmp) / "c.pdb")
            lines = Path(out).read_text().splitlines()
            lig_res = {(l[21], l[22:26].strip()) for l in lines
                       if l.startswith("HETATM") and l[17:20] == "LIG"}
            self.assertEqual(len(lig_res), 1)
            receptor_res = {(l[21], l[22:26].strip()) for l in lines
                            if l.startswith(("ATOM", "HETATM"))
                            and l[17:20] != "LIG"}
            self.assertFalse(lig_res & receptor_res)

    def test_link_names_the_same_chain_and_residue_as_the_ligand_atoms(self):
        # This is the equality PLIP's identify_kmers tests. If it does not
        # hold, the link is discarded silently and the metal contact vanishes.
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE_NOCHAIN, RECEPTOR, Path(tmp) / "c.pdb")
            lines = Path(out).read_text().splitlines()
            link = next(l for l in lines if l.startswith("LINK"))
            lig = next(l for l in lines
                       if l.startswith("HETATM") and l[17:20] == "LIG")
            self.assertEqual(link[17:20], lig[17:20])
            self.assertEqual(link[21], lig[21])
            self.assertEqual(link[22:26], lig[22:26])

    def test_chainless_pose_still_gets_its_link(self):
        with TemporaryDirectory() as tmp:
            out = build_complex(MIDAS_POSE_NOCHAIN, RECEPTOR, Path(tmp) / "c.pdb")
            self.assertEqual(
                len([l for l in Path(out).read_text().splitlines()
                     if l.startswith("LINK")]), 1)


if __name__ == "__main__":
    unittest.main()
