import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from build_library import build_library, read_sampling_csv, write_library
from novelty import load_smi

# The same compound written two ways. Chem.MolToSmiles canonicalizes both to the
# same string, so REINVENT's own unique_molecules would already collapse them.
SAME_MOLECULE = ("c1ccccc1C(=O)O", "OC(=O)c1ccccc1")

# PLN-1474 as a non-aromatic and an aromatic tautomer. These have DIFFERENT
# canonical SMILES and only converge after tautomer canonicalization, so they are
# the pair that the post-normalization dedupe exists to catch.
PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"


class TestReadSamplingCsv(unittest.TestCase):
    def _write(self, text):
        handle = Path(self.tmp.name) / "sampling.csv"
        handle.write_text(text)
        return str(handle)

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_reads_the_smiles_column(self):
        path = self._write("SMILES,NLL\nCCO,25.9\nCCC,31.4\n")
        self.assertEqual(read_sampling_csv(path), ["CCO", "CCC"])

    def test_skips_blank_cells(self):
        """REINVENT writes an empty SMILES cell when a sample fails to decode."""
        path = self._write("SMILES,NLL\nCCO,25.9\n,0.0\nCCC,31.4\n")
        self.assertEqual(read_sampling_csv(path), ["CCO", "CCC"])

    def test_missing_smiles_column_raises(self):
        """Fail loudly: a silent empty list would look like a successful run that
        sampled nothing, and the whole library would come out empty."""
        path = self._write("Smiles,NLL\nCCO,25.9\n")
        with self.assertRaises(KeyError):
            read_sampling_csv(path)


class TestBuildLibrary(unittest.TestCase):
    def test_drops_unparseable_smiles(self):
        result = build_library(["CCO", "not_a_smiles", "C((("])
        self.assertEqual(result["n_input"], 3)
        self.assertEqual(result["n_valid"], 1)
        self.assertEqual(len(result["rows"]), 1)

    def test_deduplicates_on_canonical_smiles(self):
        result = build_library(list(SAME_MOLECULE))
        self.assertEqual(result["n_valid"], 2)
        self.assertEqual(result["n_unique_canonical"], 1)
        self.assertEqual(len(result["rows"]), 1)

    def test_tautomer_duplicates_survive_canonical_dedupe_then_collapse(self):
        """The reason normalization runs at all, asserted in both directions.

        Two tautomers of PLN-1474 are two distinct canonical SMILES, so the
        canonical dedupe keeps both; they become one molecule only after
        tautomer canonicalization.
        """
        result = build_library([PLN_NONAROMATIC, PLN_AROMATIC])
        self.assertEqual(result["n_unique_canonical"], 2)
        self.assertEqual(result["n_unique_normalized"], 1)
        self.assertEqual(len(result["rows"]), 1)

    def test_rows_carry_normalized_and_raw_canonical_smiles(self):
        """Both structures are needed downstream: the normalized one is what we
        make claims about, the raw one is what the RL objective actually scored."""
        result = build_library([PLN_NONAROMATIC])
        normalized, raw = result["rows"][0]
        self.assertEqual(raw, Chem.MolToSmiles(
            Chem.MolFromSmiles(Chem.MolToSmiles(
                Chem.MolFromSmiles(PLN_NONAROMATIC), isomericSmiles=False))))
        self.assertNotEqual(normalized, raw)
        self.assertIsNotNone(Chem.MolFromSmiles(normalized))

    def test_reports_tautomer_failures(self):
        result = build_library(["CCO"])
        self.assertIn("n_tautomer_failures", result)
        self.assertEqual(result["n_tautomer_failures"], 0)

    def test_stereochemistry_is_dropped(self):
        """The de novo prior's vocabulary has no stereo tokens, so a stereocentre
        in sampled output is an artefact of the seed structure, not a claim."""
        result = build_library(["C[C@H](N)C(=O)O"])
        self.assertNotIn("@", result["rows"][0][0])

    def test_empty_input_yields_empty_library(self):
        result = build_library([])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["n_unique_normalized"], 0)


class TestWriteLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "library.smi")

    def test_round_trips_through_load_smi(self):
        result = build_library([PLN_AROMATIC, "CCO"])
        write_library(self.path, result, source="results/sample_agent.csv")
        rows = load_smi(self.path)
        self.assertEqual([s for s, _ in rows], [n for n, _ in result["rows"]])
        self.assertEqual([label for _, label in rows],
                         [raw for _, raw in result["rows"]])

    def test_header_records_provenance_and_counts(self):
        result = build_library([PLN_NONAROMATIC, PLN_AROMATIC])
        write_library(self.path, result, source="results/sample_agent.csv")
        header = [line for line in Path(self.path).read_text().splitlines()
                  if line.startswith("#")]
        joined = "\n".join(header)
        self.assertIn("results/sample_agent.csv", joined)
        self.assertIn("2", joined)   # n_input
        self.assertRegex(joined, r"column|field")  # explains the second column


class TestArmScopedOutputs(unittest.TestCase):
    """v1은 results/leads.csv처럼 아암 접미사 없는 경로에 썼다. 재실행 시
    이전 결과를 덮어써서 비교가 불가능해진다."""

    def test_output_path_includes_the_arm(self):
        from build_library import output_path
        self.assertEqual(output_path("TL-A-prime"), "data/v2_TL-A-prime/library.smi")
        self.assertEqual(output_path("TL-C"), "data/v2_TL-C/library.smi")

    def test_paths_carry_the_campaign_so_a_v4_arm_is_expressible(self):
        """The prefix used to be "v2_" in an f-string, which made TL-B
        (results/v4_TL-B) impossible to name here. It comes from
        make_rl_config.campaign_for now, the same source build_survivors uses,
        so the sample -> library -> survivors chain cannot disagree about where
        an arm's files live."""
        from build_library import input_path, output_path
        from build_survivors import library_path

        self.assertEqual(input_path("TL-B"), "results/v4_TL-B/sample.csv")
        self.assertEqual(output_path("TL-B"), "data/v4_TL-B/library.smi")
        for arm in ("TL-A-prime", "TL-B", "TL-C"):
            with self.subTest(arm=arm):
                self.assertEqual(output_path(arm), library_path(arm))

    def test_library_keeps_both_smiles_columns(self):
        """1열은 정규화 SMILES(모든 속성·신규성 주장의 대상), 2열은 정규화 전
        정준 SMILES(RL 목적함수가 실제로 채점한 구조). 토토머 선택이 디스크립터를
        움직이므로 둘의 불일치 크기를 측정 가능하게 남긴다."""
        from build_library import LIBRARY_COLUMNS
        self.assertEqual(LIBRARY_COLUMNS, ("smiles", "smiles_as_scored"))


if __name__ == "__main__":
    unittest.main()
