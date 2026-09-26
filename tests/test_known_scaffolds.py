import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from known_scaffolds import is_scaffold_novel, known_scaffolds
from normalize import canonical_tautomer
from novelty import load_smi, murcko

PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"


class TestKnownScaffolds(unittest.TestCase):
    def test_reference_files_yield_106_scaffolds(self):
        self.assertEqual(len(known_scaffolds()), 106)

    def test_published_reference_compounds_are_all_covered(self):
        """The gate must never call a benchmark compound's scaffold novel.

        actives_extended alone misses PLN-1474, bexotegrast and A1AFA: it is
        defined as ChEMBL alphaVbeta1 actives <= 1 uM, and PLN-1474's structure
        came from AdisInsight rather than a ChEMBL activity record while A1AFA
        sits at pIC50 5.30, below the potency cut. PLN-1474 is the only clinical
        alphaVbeta1-selective compound in this project and its structure has been
        public since August 2023, so a generated molecule rebuilding its scaffold
        is not novel.
        """
        known = known_scaffolds()
        for path in ("data/benchmark_panel.smi", "data/similarity_refs.smi",
                     "data/actives_core.smi", "data/actives_extended.smi"):
            for smiles, label in load_smi(path):
                with self.subTest(path=path, label=label):
                    self.assertFalse(is_scaffold_novel(smiles, known))

    def test_empty_scaffold_is_not_in_the_known_set(self):
        """Review Focus 4: an acyclic molecule's Murcko scaffold is "".

        Every known active has rings, so "" must be absent - otherwise every
        acyclic generated molecule would be judged non-novel by accident.
        """
        self.assertNotIn("", known_scaffolds())

    def test_acyclic_molecule_reports_novel_and_is_flagged(self):
        known = known_scaffolds()
        self.assertEqual(murcko(Chem.MolFromSmiles("CCCCC(=O)O")), "")
        self.assertTrue(is_scaffold_novel("CCCCC(=O)O", known))


class TestTautomerDependence(unittest.TestCase):
    def test_murcko_differs_between_tautomers_without_normalization(self):
        """The bug: the same compound gives two different Murcko SMILES."""
        a = murcko(Chem.MolFromSmiles(Chem.MolToSmiles(
            Chem.MolFromSmiles(PLN_NONAROMATIC), isomericSmiles=False)))
        b = murcko(Chem.MolFromSmiles(Chem.MolToSmiles(
            Chem.MolFromSmiles(PLN_AROMATIC), isomericSmiles=False)))
        self.assertNotEqual(a, b)

    def test_murcko_agrees_after_normalization(self):
        a = murcko(canonical_tautomer(Chem.MolFromSmiles(PLN_NONAROMATIC)))
        b = murcko(canonical_tautomer(Chem.MolFromSmiles(PLN_AROMATIC)))
        self.assertEqual(a, b)

    def test_pln1474_is_not_novel_in_either_tautomer(self):
        """A known clinical compound must never pass the gate, either way written."""
        known = known_scaffolds()
        self.assertFalse(is_scaffold_novel(PLN_NONAROMATIC, known))
        self.assertFalse(is_scaffold_novel(PLN_AROMATIC, known))


class TestGateOnPriorSamples(unittest.TestCase):
    def test_861_percent_of_prior_samples_are_scaffold_novel(self):
        known = known_scaffolds()
        with open("logs/cmp.vigHoB/samples_prior.csv") as handle:
            smiles = [row["SMILES"] for row in csv.DictReader(handle)]
        verdicts = [is_scaffold_novel(s, known) for s in smiles]
        novel = sum(1 for v in verdicts if v)
        total = sum(1 for v in verdicts if v is not None)
        self.assertEqual(total, 488)
        self.assertEqual(novel, 420)
        self.assertAlmostEqual(novel / total, 0.861, places=2)


class TestReferenceSetsNormalizeCleanly(unittest.TestCase):
    def test_no_reference_molecule_hits_the_tautomer_fallback(self):
        """canonical_tautomer falls back to the un-canonicalized molecule when the
        enumerator raises. On the reference side that would silently weaken the
        gate, so it must never happen: all 424 reference molecules normalize.
        """
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        known_scaffolds()
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


class TestInvalidInput(unittest.TestCase):
    def test_unparseable_smiles_returns_none(self):
        """Review Focus 1: never crash the triage funnel on generator output."""
        known = known_scaffolds()
        for bad in ["", "not_a_smiles", "C((("]:
            with self.subTest(bad=bad):
                self.assertIsNone(is_scaffold_novel(bad, known))


if __name__ == "__main__":
    unittest.main()
