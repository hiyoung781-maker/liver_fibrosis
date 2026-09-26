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
# PLN-1474 with its pyridine nitrogen protonated - reachable because the
# generator's vocabulary includes [nH+].
PLN_PROTONATED = "CC1(C(=O)NC(CCCCCCCc2ccc3c([nH+]2)NCCC3)C(=O)O)CCOCC1"


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

    def test_protonated_and_neutral_pln1474_converge(self):
        """Review fix 9: charge must normalize the same way tautomers do."""
        a = canonical_tautomer_smiles(PLN_PROTONATED)
        b = canonical_tautomer_smiles(PLN_AROMATIC)
        self.assertEqual(a, b)

    def test_success_records_no_failure(self):
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        canonical_tautomer_smiles(PLN_AROMATIC)
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


class TestTautomerFailurePath(unittest.TestCase):
    """Review Focus 2: a normalization failure must leave a trace.

    RDKit's TautomerEnumerator does not raise on any of this project's 424
    reference molecules, so the only way to exercise the except branch is to
    substitute an enumerator that raises. Without this the branch is untested
    and deleting the append would not fail any test.
    """

    def setUp(self):
        import normalize

        self.normalize = normalize
        self.original = normalize._TAUTOMER
        self.before = list(normalize.TAUTOMER_FAILURES)

    def tearDown(self):
        self.normalize._TAUTOMER = self.original
        self.normalize.TAUTOMER_FAILURES[:] = self.before

    def _install_failing_enumerator(self):
        class Raising:
            def Canonicalize(self, mol):
                raise RuntimeError("enumerator failed")

        self.normalize._TAUTOMER = Raising()

    def test_failure_is_appended_to_the_failure_list(self):
        self._install_failing_enumerator()
        canonical_tautomer(Chem.MolFromSmiles("CC(=O)O"))
        self.assertEqual(len(self.normalize.TAUTOMER_FAILURES),
                         len(self.before) + 1)
        self.assertEqual(self.normalize.TAUTOMER_FAILURES[-1], "CC(=O)O")

    def test_failure_returns_the_flattened_molecule_not_none(self):
        """Deliberate: a dropped REFERENCE molecule would silently weaken the
        novelty gate, which is worse than comparing its un-canonicalized form.
        Task 3 asserts the reference sets produce zero failures, so this
        fallback never silently applies to the reference side."""
        self._install_failing_enumerator()
        result = canonical_tautomer(Chem.MolFromSmiles("CC(=O)O"))
        self.assertIsNotNone(result)
        self.assertEqual(Chem.MolToSmiles(result), "CC(=O)O")


if __name__ == "__main__":
    unittest.main()
