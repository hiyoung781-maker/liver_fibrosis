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


class TestReceptorPathIsAbsolute(unittest.TestCase):
    def test_receptor_line_is_absolute(self):
        """autogrid4 must be invoked with the maps directory as its working
        directory (gridfld/map/elecmap/dsolvmap are all resolved relative to
        it). If the receptor line instead carries the caller's relative path
        (e.g. "docking/v2/receptor.pdbqt"), autogrid4 fails with 'can't find
        or open receptor PDBQT file' -- an error that names a missing file,
        not a path-resolution problem, so it sends the reader looking in the
        wrong place. Writing an absolute path avoids that failure entirely."""
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            receptor_line = [l for l in Path(gpf).read_text().splitlines()
                              if l.startswith("receptor ")][0]
            receptor_path = receptor_line.split(maxsplit=1)[1]
            self.assertTrue(Path(receptor_path).is_absolute())

    def test_map_and_gridfld_filenames_stay_bare_basenames(self):
        """map/gridfld/elecmap/dsolvmap filenames must stay relative to the
        working directory, since that's how autogrid4 writes its outputs
        into the maps directory; absolutising them would scatter the maps
        instead of collecting them there."""
        with TemporaryDirectory() as tmp:
            gpf = write_gpf(BOX, "docking/receptor.pdbqt", Path(tmp) / "r.gpf")
            text = Path(gpf).read_text().splitlines()
            for prefix in ("gridfld ", "elecmap ", "dsolvmap "):
                line = [l for l in text if l.startswith(prefix)][0]
                fname = line.split(maxsplit=1)[1]
                self.assertFalse(Path(fname).is_absolute())
                self.assertEqual(fname, Path(fname).name)
            for line in [l for l in text if l.startswith("map ")]:
                fname = line.split(maxsplit=1)[1]
                self.assertFalse(Path(fname).is_absolute())
                self.assertEqual(fname, Path(fname).name)


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
