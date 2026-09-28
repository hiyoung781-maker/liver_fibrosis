import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_maps import maps_cover_box, write_gpf

BOX = {"center": (10.0, 20.0, 30.0), "size": (22.0, 24.0, 20.0)}


class TestGridCoversSearchBox(unittest.TestCase):
    """AutoDock-GPU does not error when a ligand wanders outside the grid; it
    returns a plausible-looking but wrong energy. A mismatch between the map
    extent and the search box silently corrupts every downstream result."""

    def test_gpf_encodes_the_box_center(self):
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            text = Path(gpf).read_text()
            self.assertIn("gridcenter 10.000 20.000 30.000", text)

    def test_gpf_npts_covers_the_box_at_0375_spacing(self):
        """AutoDock4 default spacing is 0.375 A. npts must be even and must
        cover the box on every axis, or the grid falls short of the search box."""
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            npts = [int(v) for v in
                    [l for l in Path(gpf).read_text().splitlines()
                     if l.startswith("npts")][0].split()[1:4]]
            for n, extent in zip(npts, BOX["size"]):
                self.assertEqual(n % 2, 0)
                self.assertGreaterEqual(n * 0.375, extent)

    def test_detects_a_box_larger_than_the_maps(self):
        """This is the gate itself: maps_cover_box must fail a box that the
        grid cannot cover, and pass the box the grid was built for."""
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            bigger = {"center": BOX["center"], "size": (40.0, 40.0, 40.0)}
            self.assertFalse(maps_cover_box(gpf, bigger))
            self.assertTrue(maps_cover_box(gpf, BOX))


class TestCalciumIsTyped(unittest.TestCase):
    def test_gpf_requests_a_calcium_map(self):
        """The MIDAS Ca2+ is the single most important contact in this binding
        site. If CA is missing from receptor_types, no calcium interaction is
        computed at all -- half the reason for AD4 scoring over Vina scoring
        is its metal parameters."""
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            self.assertIn("CA", Path(gpf).read_text().split("receptor_types")[1]
                          .splitlines()[0].split())


if __name__ == "__main__":
    unittest.main()
