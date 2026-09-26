import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from normalize import (
    canonical_tautomer,
    canonical_tautomer_smiles,
    flatten,
)

# PLN-1474, written two ways: non-aromatic amidine tautomer and the aromatic
# pyridine tautomer stored in data/benchmark_panel.smi. Same formula C24H37N3O4.
PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"


class TestFlatten(unittest.TestCase):
    def test_removes_stereochemistry(self):
        mol = flatten(Chem.MolFromSmiles(PLN_NONAROMATIC))
        self.assertNotIn("@", Chem.MolToSmiles(mol))

    def test_returns_none_for_unparseable_input(self):
        self.assertIsNone(flatten(None))


class TestCanonicalTautomer(unittest.TestCase):
    def test_two_pln1474_tautomers_converge(self):
        a = canonical_tautomer_smiles(PLN_NONAROMATIC)
        b = canonical_tautomer_smiles(PLN_AROMATIC)
        self.assertEqual(a, b)

    def test_convergence_is_needed(self):
        """Without normalization the two forms differ - the bug this guards."""
        a = Chem.MolToSmiles(flatten(Chem.MolFromSmiles(PLN_NONAROMATIC)))
        b = Chem.MolToSmiles(flatten(Chem.MolFromSmiles(PLN_AROMATIC)))
        self.assertNotEqual(a, b)

    def test_invalid_smiles_returns_none_and_does_not_raise(self):
        """Review Focus 1: the generator emits unparseable SMILES."""
        for bad in ["", "not_a_smiles", "C(((", "[Xx]"]:
            with self.subTest(bad=bad):
                self.assertIsNone(canonical_tautomer_smiles(bad))

    def test_idempotent(self):
        once = canonical_tautomer_smiles(PLN_NONAROMATIC)
        twice = canonical_tautomer_smiles(once)
        self.assertEqual(once, twice)

    def test_failure_is_recorded_not_swallowed(self):
        """Review Focus 2: a normalization failure must leave a trace."""
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        # A valid molecule that the enumerator handles: no new failure.
        canonical_tautomer_smiles(PLN_AROMATIC)
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


if __name__ == "__main__":
    unittest.main()
