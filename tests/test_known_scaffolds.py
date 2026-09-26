import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from known_scaffolds import REFERENCE_FILES, is_scaffold_novel, known_scaffolds
from normalize import canonical_tautomer
from novelty import load_smi, murcko

PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"
# PLN-1474's pyridine nitrogen written protonated, as the de novo generator's
# vocabulary (which includes [nH+]) could plausibly emit it.
PLN_PROTONATED = "CC1(C(=O)NC(CCCCCCCc2ccc3c([nH+]2)NCCC3)C(=O)O)CCOCC1"

# Computed once for the whole module: known_scaffolds() reads and normalizes five
# files, and re-deriving it in nearly every test was costing ~860s across this
# file alone.
KNOWN = None


def setUpModule():
    global KNOWN
    KNOWN = known_scaffolds()


class TestKnownScaffolds(unittest.TestCase):
    def test_reference_files_yield_106_scaffolds(self):
        self.assertEqual(len(KNOWN), 106)

    def test_the_three_compounds_the_union_exists_for(self):
        """Direct regression guard for the five-file reference set.

        actives_extended.smi is "ChEMBL alphaVbeta1 actives <= 1 uM", which omits
        these three: PLN-1474's structure came from a drug-development database
        rather than a ChEMBL activity record, bexotegrast is a dual alphaVbeta6 /
        alphaVbeta1 compound, and A1AFA sits below the potency cut. Asserting both
        halves - absent from the single file, present in the union - is what fails
        if REFERENCE_FILES is ever narrowed back to actives_extended alone.
        """
        extended_only = known_scaffolds(("data/actives_extended.smi",))
        compounds = {
            "PLN-1474": "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1",
            "bexotegrast":
                "COCCN(CCCCc1ccc2c(n1)NCCC2)CCC(Nc1ncnc2ccccc12)C(=O)O",
            "A1AFA": "O=C(NC(Cc1ccccc1)C(=O)O)c1ccc(Cl)cc1Cl",
        }
        for label, smiles in compounds.items():
            with self.subTest(label=label):
                scaffold = murcko(canonical_tautomer(Chem.MolFromSmiles(smiles)))
                self.assertNotIn(scaffold, extended_only)
                self.assertIn(scaffold, KNOWN)
                self.assertFalse(is_scaffold_novel(smiles, KNOWN))

    def test_empty_scaffold_is_not_in_the_known_set(self):
        """Review Focus 4: an acyclic molecule's Murcko scaffold is "".

        Every known active has rings, so "" must be absent - otherwise every
        acyclic generated molecule would be judged non-novel by accident.
        """
        self.assertNotIn("", KNOWN)

    def test_acyclic_molecule_reports_novel_and_is_flagged(self):
        self.assertEqual(murcko(Chem.MolFromSmiles("CCCCC(=O)O")), "")
        self.assertTrue(is_scaffold_novel("CCCCC(=O)O", KNOWN))

    def test_known_scaffolds_file_matches_the_function(self):
        """Pin data/known_scaffolds.smi to what known_scaffolds() actually
        produces - §9.3 pre-registers this file by name, but nothing else checks
        that its contents still agree with the code that generates it."""
        with open("data/known_scaffolds.smi") as handle:
            from_file = {line.strip() for line in handle
                         if line.strip() and not line.startswith("#")}
        self.assertEqual(from_file, KNOWN)


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
        self.assertFalse(is_scaffold_novel(PLN_NONAROMATIC, KNOWN))
        self.assertFalse(is_scaffold_novel(PLN_AROMATIC, KNOWN))


class TestChargeDependence(unittest.TestCase):
    """Review fix 9: charge is the same reference-vs-measured asymmetry the
    tautomer work closed, left open. Reference .smi files are produced by
    standardize(), which applies Uncharger, so an un-normalized measured side
    can diverge on charge alone. The generator's vocabulary includes
    [N+] [N-] [O-] [S+] [n+], so this is reachable, unlike '.'."""

    def test_protonated_and_neutral_pln1474_converge(self):
        a = canonical_tautomer(Chem.MolFromSmiles(PLN_PROTONATED))
        b = canonical_tautomer(Chem.MolFromSmiles(PLN_AROMATIC))
        self.assertEqual(Chem.MolToSmiles(a), Chem.MolToSmiles(b))

    def test_protonated_pln1474_is_not_novel(self):
        self.assertFalse(is_scaffold_novel(PLN_PROTONATED, KNOWN))


class TestGateOnPriorSamples(unittest.TestCase):
    def test_861_percent_of_prior_samples_are_scaffold_novel(self):
        with open("logs/cmp.vigHoB/samples_prior.csv") as handle:
            smiles = [row["SMILES"] for row in csv.DictReader(handle)]
        verdicts = [is_scaffold_novel(s, KNOWN) for s in smiles]
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

        This test keeps its own explicit known_scaffolds() call, rather than the
        module-level KNOWN, so its before/after TAUTOMER_FAILURES delta stays
        meaningful regardless of what other tests already triggered.
        """
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        known_scaffolds()
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


class TestInvalidInput(unittest.TestCase):
    def test_unparseable_smiles_returns_none(self):
        """Review Focus 1: never crash the triage funnel on generator output."""
        for bad in ["", "not_a_smiles", "C((("]:
            with self.subTest(bad=bad):
                self.assertIsNone(is_scaffold_novel(bad, KNOWN))


if __name__ == "__main__":
    unittest.main()
