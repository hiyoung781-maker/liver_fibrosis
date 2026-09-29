import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prepare_maps import LIGAND_TYPES, maps_cover_box, receptor_types, write_gpf

BOX = {"center": (10.0, 20.0, 30.0), "size": (22.0, 24.0, 20.0)}

# Synthetic receptor fixture covering the two pairs the old hardcoded
# RECEPTOR_TYPES list conflated: non-polar H vs. donor HD, and non-acceptor
# S vs. acceptor SA. Also includes the real MIDAS calcium type "Ca" (not the
# "CA" the old constant used).
FIXTURE_PDBQT = """\
ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C
ATOM      2  N1  LIG A   1       1.000   0.000   0.000  1.00  0.00    +0.000 N
ATOM      3  H1  LIG A   1       2.000   0.000   0.000  1.00  0.00    +0.000 H
ATOM      4  H2  LIG A   1       3.000   0.000   0.000  1.00  0.00    +0.000 HD
ATOM      5  S1  LIG A   1       4.000   0.000   0.000  1.00  0.00    +0.000 S
ATOM      6  S2  LIG A   1       5.000   0.000   0.000  1.00  0.00    +0.000 SA
HETATM    7 CA   CA  A   2       6.000   0.000   0.000  1.00  0.00    +2.000 Ca
END
"""


def _write_fixture(path: Path) -> Path:
    path.write_text(FIXTURE_PDBQT)
    return path


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
        site. If Ca is missing from receptor_types, no calcium interaction is
        computed at all -- half the reason for AD4 scoring over Vina scoring
        is its metal parameters. Note the real PDBQT type is "Ca", not the
        old hardcoded "CA"."""
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp) / "receptor.pdbqt")
            gpf = write_gpf(BOX, str(fixture), Path(tmp) / "r.gpf")
            self.assertIn("Ca", Path(gpf).read_text().split("receptor_types")[1]
                          .splitlines()[0].split())


class TestReceptorTypesDerivedFromFile(unittest.TestCase):
    """The hardcoded RECEPTOR_TYPES list conflated non-polar H with donor HD,
    and non-acceptor S with acceptor SA -- both are real, distinct AD4 types
    that `obabel -xr -h` can put in the same receptor PDBQT. autogrid4 rejects
    a GPF whose receptor_types count doesn't match what's actually in the
    file, so the types must be read from the file, not guessed."""

    def test_distinct_types_from_fixture_including_h_hd_and_s_sa(self):
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp) / "receptor.pdbqt")
            types = receptor_types(fixture)
            self.assertEqual(types, sorted({"C", "N", "H", "HD", "S", "SA", "Ca"}))
            # both pairs the old hardcoded list conflated must be present
            self.assertIn("H", types)
            self.assertIn("HD", types)
            self.assertIn("S", types)
            self.assertIn("SA", types)

    def test_gpf_receptor_types_line_matches_receptor_types_function(self):
        """Prefer the real receptor.pdbqt if present (what the cluster
        actually has); fall back to the synthetic fixture otherwise."""
        real = Path("docking/receptor.pdbqt")
        with TemporaryDirectory() as tmp:
            receptor = real if real.exists() else _write_fixture(Path(tmp) / "receptor.pdbqt")
            gpf = write_gpf(BOX, str(receptor), Path(tmp) / "r.gpf")
            line = [l for l in Path(gpf).read_text().splitlines()
                    if l.startswith("receptor_types")][0]
            emitted = line.split()[1:]
            self.assertEqual(emitted, receptor_types(receptor))


class TestLigandTypesUnchanged(unittest.TestCase):
    """ligand_types is the deliberate, hardcoded vocabulary of the generated
    ligand library -- what maps get made FOR -- and must never be derived
    from the receptor. Calcium is receptor-only (MIDAS site), so it must not
    appear in ligand_types, and no per-receptor-type map line (e.g. for Ca)
    should ever be emitted -- autogrid4 makes one map per ligand type, with
    the receptor's contribution folded into each of those maps."""

    def test_ligand_types_is_the_deliberate_generated_molecule_list(self):
        self.assertEqual(
            LIGAND_TYPES,
            ["C", "A", "N", "NA", "OA", "S", "SA", "HD", "F", "Cl", "Br", "I"],
        )
        self.assertNotIn("Ca", LIGAND_TYPES)
        self.assertNotIn("CA", LIGAND_TYPES)

    def test_no_map_line_for_a_receptor_only_type(self):
        with TemporaryDirectory() as tmp:
            fixture = _write_fixture(Path(tmp) / "receptor.pdbqt")
            gpf = write_gpf(BOX, str(fixture), Path(tmp) / "r.gpf")
            map_lines = [l for l in Path(gpf).read_text().splitlines()
                         if l.startswith("map ")]
            self.assertEqual(len(map_lines), len(LIGAND_TYPES))
            for line in map_lines:
                fname = line.split(maxsplit=1)[1]
                self.assertNotIn(".Ca.", fname)
                self.assertNotIn(".CA.", fname)


if __name__ == "__main__":
    unittest.main()
