import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from dock_panel import prepare_panel_ligands

try:
    import meeko  # noqa: F401
    HAVE_MEEKO = True
except ImportError:
    HAVE_MEEKO = False


@unittest.skipUnless(HAVE_MEEKO, "meeko not installed")
class TestPanelMatchesTheLibrarysProtonation(unittest.TestCase):
    """prepare_panel_ligands exists so that "a difference in the docking result
    cannot come from a difference in ligand prep" -- its own docstring. The
    redocking gate adopted the anion (spec 7.4) and the generated library is
    now deprotonated, so that invariant REQUIRES the panel to be deprotonated
    too. It is not an option: the section 9.1 affinity filter is a PLN-1474
    relative comparison, and a reference in a different charge state from the
    candidates is not a comparison at all."""

    PANEL = "# one acid, one neutral\nCC(=O)O\tacid\nc1ccccc1\tneutral\n"

    def _run(self, tmp, **kwargs):
        smi = Path(tmp) / "panel.smi"
        smi.write_text(self.PANEL)
        out = Path(tmp) / "pdbqt"
        counts = prepare_panel_ligands(str(out), str(smi), **kwargs)
        texts = {p.stem: p.read_text() for p in out.glob("*.pdbqt")}
        return counts, texts

    def test_acids_are_deprotonated_by_default(self):
        with TemporaryDirectory() as tmp:
            _c, texts = self._run(tmp)
            smiles = next(l for l in texts["PANEL_acid"].splitlines()
                          if l.startswith("REMARK SMILES "))
            self.assertIn("[O-]", smiles)

    def test_a_molecule_without_an_acid_is_still_written(self):
        with TemporaryDirectory() as tmp:
            counts, texts = self._run(tmp)
            self.assertEqual(counts["written"], 2)
            self.assertEqual(counts["failures"], [])
            self.assertIn("PANEL_neutral", texts)

    def test_deprotonation_can_be_turned_off_explicitly(self):
        # The neutral arm also passed the gate; if the campaign ever adopts it
        # the panel has to follow, and the switch must be visible in a command.
        with TemporaryDirectory() as tmp:
            _c, texts = self._run(tmp, deprotonate=False)
            smiles = next(l for l in texts["PANEL_acid"].splitlines()
                          if l.startswith("REMARK SMILES "))
            self.assertNotIn("[O-]", smiles)


if __name__ == "__main__":
    unittest.main()
