import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from tl_sweep_report import MIN_EPOCH, SCAFFOLD_FLOOR, select_epoch


def row(epoch, scaffolds, acid_pct=45.0, max_nn=0.9):
    return {"epoch": epoch, "unique_scaffolds": scaffolds,
            "acid_pct": acid_pct, "max_nn_tanimoto": max_nn}


class TestSelectionRule(unittest.TestCase):
    """spec §4.2: epoch 100-200 구간에서 고유 Murcko 스캐폴드 ≥ 700/1000을
    만족하는 가장 높은 epoch. 암기(max NN-Tanimoto)는 탈락 조건이 아니라
    보고 항목이다 - 하류의 Murcko 신규성 게이트가 이미 제거한다."""

    def test_picks_the_highest_qualifying_epoch(self):
        rows = [row(100, 820), row(120, 780), row(140, 760), row(160, 690)]
        self.assertEqual(select_epoch(rows), 140)

    def test_ignores_epochs_below_the_floor(self):
        """epoch 80이 더 좋아도 선택 구간 밖이다."""
        rows = [row(80, 900), row(100, 720), row(120, 690)]
        self.assertEqual(select_epoch(rows), 100)

    def test_returns_none_when_nothing_qualifies(self):
        """만족하는 체크포인트가 없으면 TL-A′는 아암에서 탈락한다."""
        rows = [row(100, 650), row(120, 600), row(140, 550)]
        self.assertIsNone(select_epoch(rows))

    def test_memorization_does_not_disqualify(self):
        """max NN-Tanimoto 1.000(학습 분자 축자 재현)이어도 탈락시키지 않는다.
        보고는 하되 선택은 스캐폴드 다양성으로 한다."""
        rows = [row(100, 720, max_nn=1.0), row(120, 650, max_nn=0.7)]
        self.assertEqual(select_epoch(rows), 100)

    def test_documented_thresholds(self):
        self.assertEqual(MIN_EPOCH, 100)
        self.assertEqual(SCAFFOLD_FLOOR, 700)


if __name__ == "__main__":
    unittest.main()
