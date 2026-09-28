import json
import unittest

KNOWN = "data/known_scaffolds.smi"
BAND = "data/novelty_band.json"


def read_scaffolds(path):
    return {line.strip() for line in open(path)
            if line.strip() and not line.startswith("#")}


class TestKnownScaffoldsUnchanged(unittest.TestCase):
    """core에서 4종을 빼도 다섯 파일의 합집합은 변하면 안 된다.

    CWHM-12(αvβ1 IC50 1.8 nM)와 GLPG0187(1.3 nM)은 RGD-zwitterion이면서
    ≤1 µM이므로 규칙만으로 actives_core_B·extended에 들어간다. PLN-1474와
    bexotegrast는 benchmark_panel에 남는다. 합집합이 줄면 생성 분자가 공개
    화합물의 스캐폴드를 재현하고도 'novel'로 통과하는 구멍이 열린다.
    """

    def test_union_is_still_106_scaffolds(self):
        self.assertEqual(len(read_scaffolds(KNOWN)), 106)


class TestNoveltyBandRecomputed(unittest.TestCase):
    def test_core_band_reflects_25_molecules(self):
        """actives_core가 29→25로 줄었으니 band도 그 값을 따라가야 한다."""
        band = json.load(open(BAND))
        self.assertEqual(band["actives_core"]["n"], 25)

    def test_extended_band_is_unchanged(self):
        """extended는 이번 변경의 영향을 받지 않는다."""
        band = json.load(open(BAND))
        self.assertEqual(band["actives_extended"]["n"], 190)


if __name__ == "__main__":
    unittest.main()
