import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_receptor_pdbqt import (
    ANCHORS,
    BINDING_SITE_RESIDUES,
    TOLERANCE,
    BindingSiteTruncationError,
    build_meeko_command,
    compare_anchors,
    fix_calcium_charges,
    read_pdb_atoms,
    truncate_incomplete_residues,
    verify_receptor,
)

PDB = "docking/receptor.pdb"

BOX = {
    "center_x": 1.0, "center_y": 2.0, "center_z": 3.0,
    "size_x": 20.0, "size_y": 20.0, "size_z": 20.0,
}


def _pdb_line(record, serial, name, resname, chain, resseq, x, y, z):
    # Column layout matches _parse(): name at [12:16], resname at [17:20],
    # chain at [21], resseq at [22:26], xyz at [30:38]/[38:46]/[46:54].
    line = [" "] * 80
    line[0:6] = list(f"{record:<6s}")
    line[6:11] = list(f"{serial:5d}")
    line[12:16] = list(f"{name:<4s}")
    line[17:20] = list(f"{resname:>3s}")
    line[21] = chain
    line[22:26] = list(f"{resseq:4d}")
    line[30:38] = list(f"{x:8.3f}")
    line[38:46] = list(f"{y:8.3f}")
    line[46:54] = list(f"{z:8.3f}")
    line[54:60] = list(f"{1.00:6.2f}")
    line[60:66] = list(f"{0.00:6.2f}")
    line[76:78] = list(f"{name[0]:>2s}")
    return "".join(line) + "\n"


def _incomplete_lys(chain="A", resseq=999):
    """A synthetic LYS with only N, CA, C, O, CB - the truncated pattern the
    cluster's 25 residues all share."""
    atoms = [("N", 0.0, 0.0, 0.0), ("CA", 1.0, 0.0, 0.0), ("C", 2.0, 0.0, 0.0),
              ("O", 3.0, 0.0, 0.0), ("CB", 1.0, 1.0, 0.0)]
    return [_pdb_line("ATOM", i + 1, name, "LYS", chain, resseq, x, y, z)
            for i, (name, x, y, z) in enumerate(atoms)]


def _complete_lys(chain="A", resseq=998):
    atoms = [("N", 0.0, 0.0, 0.0), ("CA", 1.0, 0.0, 0.0), ("C", 2.0, 0.0, 0.0),
              ("O", 3.0, 0.0, 0.0), ("CB", 1.0, 1.0, 0.0), ("CG", 1.0, 2.0, 0.0),
              ("CD", 1.0, 3.0, 0.0), ("CE", 1.0, 4.0, 0.0), ("NZ", 1.0, 5.0, 0.0)]
    return [_pdb_line("ATOM", i + 1, name, "LYS", chain, resseq, x, y, z)
            for i, (name, x, y, z) in enumerate(atoms)]


class TestAnchorDefinition(unittest.TestCase):
    def test_anchors_name_chain_b_explicitly(self):
        """Chain A holds four more calciums, 35-50 A from the ligand. An earlier
        analysis in this project searched chain A and got exactly those numbers, so
        the chain is part of the key, not an afterthought."""
        for chain, _, _, _ in ANCHORS:
            self.assertEqual(chain, "B")

    def test_tolerance_is_tight(self):
        """A rigid-receptor conversion must not move an atom at all; anything above
        a hundredth of an angstrom means the converter did something unexpected."""
        self.assertLessEqual(TOLERANCE, 0.01)


class TestParsers(unittest.TestCase):
    def test_reads_both_atom_and_hetatm(self):
        """The six calcium HETATM records must not be lost by code that only looks
        at HETATM or only at ATOM."""
        atoms = read_pdb_atoms(PDB)
        records = {a["record"] for a in atoms}
        self.assertIn("ATOM", records)
        self.assertIn("HETATM", records)
        calciums = [a for a in atoms if a["resname"] == "CA"]
        self.assertEqual(len(calciums), 6)


class TestTruncateIncompleteResidues(unittest.TestCase):
    """Step 1-4 of the Meeko route: rename unresolved side chains to ALA rather
    than deleting the residue, and never do this to a binding-site residue."""

    def test_incomplete_lysine_is_renamed_to_ala_with_coords_unchanged(self):
        """The cluster's 25 truncated residues (LYS/ARG/GLU/TYR) all retain exactly
        N, CA, C, O, CB - which is alanine's full heavy-atom set. Renaming to ALA
        must not move a single coordinate."""
        lines = _incomplete_lys(chain="A", resseq=999)
        new_lines, truncations = truncate_incomplete_residues(lines)

        self.assertEqual(len(truncations), 1)
        self.assertEqual(truncations[0]["resname"], "LYS")
        self.assertEqual(truncations[0]["chain"], "A")
        self.assertEqual(truncations[0]["resseq"], "999")

        self.assertEqual(len(new_lines), 5)
        kept_names = [line[12:16].strip() for line in new_lines]
        self.assertEqual(set(kept_names), {"N", "CA", "C", "O", "CB"})
        for line in new_lines:
            self.assertEqual(line[17:20], "ALA")

        # Byte-identical coordinates: compare the original atom lines against the
        # renamed ones by (name -> xyz string), since only columns 18-20 change.
        orig_by_name = {line[12:16].strip(): line[30:54] for line in lines}
        new_by_name = {line[12:16].strip(): line[30:54] for line in new_lines}
        self.assertEqual(orig_by_name, new_by_name)

    def test_complete_residue_is_untouched(self):
        lines = _complete_lys(chain="A", resseq=998)
        new_lines, truncations = truncate_incomplete_residues(lines)
        self.assertEqual(truncations, [])
        self.assertEqual(new_lines, lines)

    def test_incomplete_binding_site_residue_raises_instead_of_truncating(self):
        """The pose gate measures a contact to these exact side chains. Silently
        alanine-ing one would turn a measured contact into a missing one and the
        pipeline would report a plausible pass rate instead of an error."""
        chain, resseq = next(iter(BINDING_SITE_RESIDUES))
        lines = _incomplete_lys(chain=chain, resseq=resseq)
        with self.assertRaises(BindingSiteTruncationError):
            truncate_incomplete_residues(lines)

    def test_binding_site_set_matches_the_pose_gate(self):
        """Hard-coded per the campaign's pose-gate anchors: Ca B/501, Asn B/224,
        Leu B/225, Tyr A/178, Asp A/218, plus their surrounding MIDAS-loop
        residues."""
        self.assertIn(("B", 224), BINDING_SITE_RESIDUES)
        self.assertIn(("B", 225), BINDING_SITE_RESIDUES)
        self.assertIn(("A", 178), BINDING_SITE_RESIDUES)
        self.assertIn(("A", 218), BINDING_SITE_RESIDUES)


class TestBuildMeekoCommand(unittest.TestCase):
    """Command construction is separated from subprocess execution specifically so
    it is testable without Meeko installed."""

    def test_command_has_required_flags(self):
        cmd = build_meeko_command("truncated.pdb", "out/receptor", BOX,
                                  box_radius=8.0)
        self.assertEqual(cmd[0], "mk_prepare_receptor.py")
        self.assertIn("--read_pdb", cmd)
        self.assertIn("truncated.pdb", cmd)
        self.assertIn("--compute_charges", cmd)
        self.assertIn("--charge_model", cmd)
        self.assertIn("gasteiger", cmd)
        self.assertIn("-p", cmd)
        self.assertIn("--output_basename", cmd)
        self.assertIn("out/receptor", cmd)
        self.assertIn("--delete_bad_res_from_box_radius", cmd)
        self.assertIn("8.0", cmd)

    def test_box_values_come_through_unmodified(self):
        """box_from_ligand's values must reach the command exactly - a divergent
        box here vs. prepare_maps.py would make every downstream geometric verdict
        silently answer a different question."""
        cmd = build_meeko_command("t.pdb", "out", BOX, box_radius=8.0)
        self.assertIn("--box_center", cmd)
        self.assertIn("1.0", cmd)
        self.assertIn("2.0", cmd)
        self.assertIn("3.0", cmd)
        self.assertIn("--box_size", cmd)
        self.assertIn("20.0", cmd)

    def test_box_radius_is_configurable(self):
        cmd = build_meeko_command("t.pdb", "out", BOX, box_radius=12.5)
        self.assertIn("12.5", cmd)
        self.assertNotIn("8.0", cmd)


class TestFixCalciumCharges(unittest.TestCase):
    """Gasteiger handles metals poorly and may leave calcium at 0.000. A zero
    charge means the MIDAS calcium contributes nothing to the electrostatic map -
    the entire reason this campaign switched from Vina to AutoDock4."""

    def test_zero_charge_calcium_is_corrected_to_plus_two(self):
        atoms = [{"adtype": "Ca", "charge": "0.000", "chain": "B", "resseq": "501"}]
        fixed = fix_calcium_charges(atoms)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(atoms[0]["charge"], "+2.000")

    def test_nonzero_charge_calcium_is_left_alone(self):
        atoms = [{"adtype": "Ca", "charge": "+1.800", "chain": "B", "resseq": "502"}]
        fixed = fix_calcium_charges(atoms)
        self.assertEqual(fixed, [])
        self.assertEqual(atoms[0]["charge"], "+1.800")

    def test_non_calcium_atoms_are_ignored(self):
        atoms = [{"adtype": "OA", "charge": "0.000", "chain": "B", "resseq": "224"}]
        fixed = fix_calcium_charges(atoms)
        self.assertEqual(fixed, [])
        self.assertEqual(atoms[0]["charge"], "0.000")


class TestDetectsBreakage(unittest.TestCase):
    """compare_anchors has to report failure, not just success, or it is decoration."""

    def setUp(self):
        self.source = read_pdb_atoms(PDB)

    def _pdbqt_without_midas(self):
        return [a for a in self.source if not (a["chain"] == "B"
                                                and a["resseq"] == "501")]

    def test_reports_a_lost_calcium(self):
        rows = compare_anchors(self.source, self._pdbqt_without_midas())
        ca = next(r for r in rows if r["label"].startswith("Ca501"))
        self.assertTrue(ca["in_source"])
        self.assertFalse(ca["in_converted"])
        self.assertIsNone(ca["displacement"])

    def test_reports_a_displaced_calcium(self):
        atoms = [dict(a) for a in self.source]
        for a in atoms:
            if a["chain"] == "B" and a["resseq"] == "501":
                a["xyz"] = a["xyz"] + 0.5
        rows = compare_anchors(self.source, atoms)
        ca = next(r for r in rows if r["label"].startswith("Ca501"))
        self.assertGreater(ca["displacement"], TOLERANCE)


class TestVerifyReceptor(unittest.TestCase):
    """verify_receptor is the whole post-Meeko report, structured to be testable
    without running Meeko: it takes already-parsed atom lists."""

    def _synthetic_source_and_converted(self, acceptors, donors, calciums=6,
                                        ca_charge="+2.000", wrong_case=False):
        source = []
        converted = []
        # Both anchors, unmoved.
        for chain, resseq, resname, name in ANCHORS:
            atom = {"chain": chain, "resseq": resseq, "resname": resname,
                    "name": name, "xyz": __import__("numpy").array([0., 0., 0.])}
            source.append(dict(atom))
            conv = dict(atom)
            conv["adtype"] = "Ca" if resname == "CA" else "OA"
            conv["charge"] = ca_charge if resname == "CA" else "-0.500"
            converted.append(conv)
        # Remaining calciums (already counted one above if ANCHORS has one).
        existing_ca = sum(1 for a in converted if a.get("adtype") == "Ca")
        for i in range(calciums - existing_ca):
            adtype = "CA" if wrong_case else "Ca"
            converted.append({"chain": "A", "resseq": str(600 + i), "resname": "CA",
                              "name": "CA", "adtype": adtype, "charge": ca_charge,
                              "xyz": __import__("numpy").array([0., 0., 0.])})
        for i in range(acceptors):
            converted.append({"chain": "A", "resseq": str(700 + i), "resname": "SER",
                              "name": f"N{i}", "adtype": "NA", "charge": "-0.3",
                              "xyz": __import__("numpy").array([0., 0., 0.])})
        for i in range(donors):
            converted.append({"chain": "A", "resseq": str(800 + i), "resname": "SER",
                              "name": f"N{i}", "adtype": "N", "charge": "-0.3",
                              "xyz": __import__("numpy").array([0., 0., 0.])})
        return source, converted

    def test_fails_when_acceptors_are_the_majority(self):
        """The previous campaign's post-mortem: converting without hydrogens typed
        1,170 of 1,212 nitrogens as acceptors instead of donors, promoting a
        fourth-best pose to first. The healthy split was 101 NA / 1111 N."""
        source, converted = self._synthetic_source_and_converted(
            acceptors=1170, donors=42)
        report = verify_receptor(source, converted)
        self.assertGreater(report["acceptor_fraction"], 0.5)
        self.assertFalse(report["ok"])

    def test_passes_with_the_healthy_split(self):
        source, converted = self._synthetic_source_and_converted(
            acceptors=101, donors=1111)
        report = verify_receptor(source, converted)
        self.assertLess(report["acceptor_fraction"], 0.2)
        self.assertTrue(report["ok"])

    def test_fixes_zero_calcium_charge_and_still_reports_ok(self):
        source, converted = self._synthetic_source_and_converted(
            acceptors=101, donors=1111, ca_charge="0.000")
        report = verify_receptor(source, converted)
        self.assertEqual(len(report["calciums_fixed"]), 6)
        for ca in report["calciums"]:
            self.assertEqual(ca["charge"], "+2.000")
        self.assertTrue(report["ok"])

    def test_fails_when_a_calcium_is_missing(self):
        source, converted = self._synthetic_source_and_converted(
            acceptors=101, donors=1111, calciums=5)
        report = verify_receptor(source, converted)
        self.assertEqual(report["calcium_count"], 5)
        self.assertFalse(report["ok"])

    def test_fails_when_calcium_is_wrong_case(self):
        """The old hardcoded receptor-types list wrote 'CA' and autogrid4 is case
        sensitive, so calcium was never correctly declared. Verify catches it."""
        source, converted = self._synthetic_source_and_converted(
            acceptors=101, donors=1111, wrong_case=True)
        report = verify_receptor(source, converted)
        self.assertTrue(report["wrong_case_calciums"])
        self.assertFalse(report["ok"])


if __name__ == "__main__":
    unittest.main()
