import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import csv
from tempfile import TemporaryDirectory

from export_poses_v2 import (METAL_COLOURS, chimerax_script, contact_residues,
                             leads_from_csv, metal_atom)
from selectivity import label_variants

REPO = Path(__file__).resolve().parent.parent
AVB1 = str(REPO / "docking" / "receptor.pdb")
AVB6 = str(REPO / "docking" / "v2" / "isoforms" / "avb6" / "receptor.pdb")


class TestMetalColourIsKeyedToElement(unittest.TestCase):
    """alphaVbeta1's MIDAS is Ca and alphaVbeta6's is Mg, and the two views are
    read side by side. A colour picked per run would let one colour mean
    calcium in one figure and magnesium in the next."""

    def test_calcium_and_magnesium_differ(self):
        self.assertNotEqual(METAL_COLOURS["CA"], METAL_COLOURS["MG"])

    def test_calcium_keeps_v1s_goldenrod(self):
        # v1_leads.cxc drew Ca501 goldenrod; the v2 alphaVbeta1 view is the
        # same receptor and must not change colour between campaigns.
        self.assertEqual(METAL_COLOURS["CA"], "goldenrod")

    def test_the_colour_comes_from_the_structure_not_from_a_flag(self):
        self.assertEqual(metal_atom(AVB1, "B", "501")["colour"],
                         METAL_COLOURS["CA"])
        self.assertEqual(metal_atom(AVB6, "B", "2102")["colour"],
                         METAL_COLOURS["MG"])

    def test_an_absent_metal_raises_rather_than_defaulting(self):
        # Every distance in the view is drawn to this atom.
        with self.assertRaises(ValueError):
            metal_atom(AVB1, "B", "9999")


class TestMetalsAreNotStickResidues(unittest.TestCase):
    """The integrin interface carries three metal sites -- MIDAS, ADMIDAS and
    SyMBS. A neighbouring ion styled as a stick 'residue' hides that there is
    a second metal in reach, and a pose that reaches ADMIDAS instead of MIDAS
    is a different binding mode."""

    def test_a_neighbouring_ion_is_returned_separately(self):
        # 9CZD's ADMIDAS Ca sits 6.01 A from the MIDAS Mg, so a probe at the
        # MIDAS reaches it just past 6.
        midas = metal_atom(AVB6, "B", "2102")
        residues, metals = contact_residues(AVB6, midas["xyz"][None, :],
                                            radius=6.5)
        self.assertIn(("B", "2103"), metals)
        self.assertNotIn(("B", "2103"), residues)

    def test_amino_acids_stay_in_the_residue_list(self):
        midas = metal_atom(AVB1, "B", "501")
        residues, metals = contact_residues(AVB1, midas["xyz"][None, :],
                                            radius=6.0)
        self.assertTrue(residues)
        self.assertFalse(set(residues) & set(metals))


class TestLeadsComeFromTheFilterOutput(unittest.TestCase):
    """The labels are already in the file lead_filter.py wrote. Retyping them
    into the export command makes a second place for them to be wrong, and a
    mistyped label exports nothing -- the view is simply missing a molecule,
    with no error."""

    def _csv(self, tmp, rows, fields=("label", "all_stages")):
        path = Path(tmp) / "leads.csv"
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    def test_only_rows_passing_every_stage_are_returned(self):
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [{"label": "a", "all_stages": "True"},
                                   {"label": "b", "all_stages": "False"},
                                   {"label": "c", "all_stages": "True"}])
            self.assertEqual(leads_from_csv(path), ["a", "c"])

    def test_no_leads_is_an_empty_list_not_an_error(self):
        # TL-A-prime really did select nothing; that is the ablation result.
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [{"label": "a", "all_stages": "False"}])
            self.assertEqual(leads_from_csv(path), [])

    def test_the_wrong_csv_is_refused_rather_than_read_as_empty(self):
        # A funnel.csv has labels too, and would quietly yield nothing.
        with TemporaryDirectory() as tmp:
            path = self._csv(tmp, [{"label": "a", "geometry_pass": "True"}],
                             fields=("label", "geometry_pass"))
            with self.assertRaises(ValueError):
                leads_from_csv(path)


class TestReferenceSpelling(unittest.TestCase):
    """--controls names the compounds as CALIBRATION spells them (bare), and
    the pose files were written as docking/panel_pdbqt spells them
    (PANEL_-prefixed). Both ChimeraX views came out with every lead and not
    one control, reporting them as 'not docked against this receptor' when
    they were sitting right there."""

    def test_every_spelling_is_tried(self):
        from selectivity import label_variants
        self.assertIn("PANEL_PLN-1474", label_variants("PLN-1474"))
        self.assertIn("PLN-1474", label_variants("PANEL_PLN-1474"))

    def test_the_given_spelling_is_tried_first(self):
        self.assertEqual(label_variants("PLN-1474")[0], "PLN-1474")

    def test_a_plain_label_is_not_mangled(self):
        self.assertEqual(label_variants("gen_01883")[:2],
                         ("gen_01883", "gen_01883"))

    def test_the_exporter_finds_a_prefixed_pose_file(self):
        from export_poses_v2 import _load
        with TemporaryDirectory() as tmp:
            Path(tmp, "PANEL_PLN-1474_out.pdbqt").write_text("")
            # Reaches the file (and fails on its contents, not on the name).
            try:
                _load([tmp], "PLN-1474", None)
            except Exception as exc:                  # noqa: BLE001
                self.assertNotIn("not docked", str(exc))


class TestScript(unittest.TestCase):
    ENTRY = {"label": "gen_01883", "chain": "C", "pose": 1,
             "path": "gen_01883.pdb", "affinity": -7.95, "is_lead": True,
             "colour": "orange red", "metal_atom": "@O2",
             "metal_distance": 2.61, "donor_atom": "@N1",
             "donor_distance": 3.02}

    def _script(self, **kwargs):
        base = dict(entries=[self.ENTRY],
                    metal={"name": "CA", "element": "CA", "resname": "CA",
                           "colour": "goldenrod"},
                    midas=("B", "501"), contact=("B", "224"), context=[],
                    title="", neighbours={})
        base.update(kwargs)
        return chimerax_script(**base)

    def test_the_distance_spec_omits_the_chain_as_its_comment_says(self):
        # v1_leads.cxc: "chain is intentionally omitted from each ligand spec
        # so the atom name matches regardless of what chain ID the docking
        # output used." The comment travelled; for a while the code did not.
        script = self._script()
        self.assertIn("distance #2@O2 #1/B:501@CA", script)
        self.assertIn("intentionally omitted", script)
        self.assertNotIn("#2/C:1@O2", script)

    def test_both_distances_are_drawn_when_a_contact_is_registered(self):
        script = self._script()
        self.assertIn("#1/B:224@O color cyan", script)

    def test_no_donor_distance_is_invented_without_a_registered_contact(self):
        script = self._script(contact=None)
        self.assertNotIn("color cyan", script)
        self.assertIn("no pre-registered donor contact", script)

    def test_a_neighbouring_metal_is_named_in_the_legend(self):
        # Otherwise a goldenrod sphere in the alphaVbeta6 view reads as that
        # view's MIDAS, which is magnesium.
        script = self._script(metal={"name": "MG", "element": "MG",
                                     "resname": "MG", "colour": "medium sea green"},
                              midas=("B", "2102"), contact=None,
                              neighbours={("B", "2103"): "CA"})
        self.assertIn("NEIGHBOURING metal site", script)
        self.assertIn("medium sea green sphere (MG B/2102)", script)

    def test_a_non_carboxylate_contact_is_drawn_separately(self):
        """Compound 25 reaches the MIDAS with something that is not its
        carboxylate -- 7.4-9.8 A on the measured atom while PLIP still
        reports metal coordination. Drawing only the carboxylate distance
        puts a 9.8 A dash on a pose that is touching the metal."""
        entry = {**self.ENTRY, "metal_distance": 9.84, "near_atom": "@O5",
                 "near_distance": 2.31, "near_element": "O"}
        script = self._script(entries=[entry])
        self.assertIn("9.84 A carboxylate (FAIL", script)
        self.assertIn("2.31 A nearest O", script)
        self.assertIn("NOT a carboxylate", script)
        self.assertIn("distance #2@O5 #1/B:501@CA color orange", script)

    def test_nothing_extra_is_drawn_when_the_carboxylate_is_the_contact(self):
        # near_atom is None when the nearest coordinating atom IS the one the
        # filter measured; a second dash on the same pair would double it.
        script = self._script()
        self.assertNotIn("color orange", script)

    def test_the_pass_or_fail_of_each_drawn_distance_is_in_the_script(self):
        script = self._script()
        self.assertIn("2.61 A carboxylate (PASS, cutoff 3.2)", script)
        self.assertIn("3.02 A (PASS, cutoff 3.5)", script)

    def test_a_failing_pose_says_so(self):
        entry = {**self.ENTRY, "metal_distance": 8.37}
        script = self._script(entries=[entry])
        self.assertIn("8.37 A carboxylate (FAIL, cutoff 3.2)", script)


if __name__ == "__main__":
    unittest.main()
