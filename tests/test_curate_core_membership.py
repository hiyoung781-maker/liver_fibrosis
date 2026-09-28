import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

import curate_actives

CORE = "data/actives_core.smi"
PANEL = "data/benchmark_panel.smi"

RGD_TOOLS = {"bexotegrast", "CWHM-12", "GLPG0187", "PLN-1474"}


def read_smi(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        smiles, label = line.split("\t")
        rows.append((smiles, label))
    return rows


class TestCoreIsRuleDerived(unittest.TestCase):
    """core 멤버십이 하드코딩 주입 없이 규칙만으로 결정되는지.

    v1은 TOOL_COMPOUNDS 4종을 chemotype 규칙과 potency 규칙을 모두 우회해
    core에 주입했다. 그 결과 '왜 섞였는가'가 코드가 아니라 주석에만 있었다.
    """

    def test_core_has_no_rgd_tool_compounds(self):
        labels = {label for _, label in read_smi(CORE)}
        self.assertEqual(labels & RGD_TOOLS, set())

    def test_core_is_25_molecules(self):
        self.assertEqual(len(read_smi(CORE)), 25)

    def test_every_core_molecule_is_non_rgd(self):
        """curate_actives의 여섯 Arg-mimic SMARTS 중 어느 것도 걸리지 않아야 한다."""
        patterns = {name: Chem.MolFromSmarts(smarts)
                    for name, smarts in curate_actives.ARG_MIMIC_HEADS.items()}
        for name, pattern in patterns.items():
            self.assertIsNotNone(pattern, f"SMARTS가 파싱되지 않음: {name}")
        flagged = []
        for smiles, label in read_smi(CORE):
            mol = Chem.MolFromSmiles(smiles)
            hits = [n for n, p in patterns.items() if mol.HasSubstructMatch(p)]
            if hits:
                flagged.append((label, hits))
        self.assertEqual(flagged, [])

    def test_every_core_molecule_carries_a_carboxylic_acid(self):
        acid = Chem.MolFromSmarts(curate_actives.CARBOXYLIC_ACID)
        missing = [label for smiles, label in read_smi(CORE)
                   if not Chem.MolFromSmiles(smiles).HasSubstructMatch(acid)]
        self.assertEqual(missing, [])


class TestPanel(unittest.TestCase):
    """패널은 목표 프로파일 참조군 4종. pan-αv 화합물과 C8은 제외한다."""

    def test_panel_is_exactly_four_named_compounds(self):
        labels = {label for _, label in read_smi(PANEL)}
        self.assertEqual(labels,
                         {"A1AFA", "PLN-1474", "CHEMBL4649232", "bexotegrast"})

    def test_pan_av_compounds_are_absent(self):
        """CWHM-12는 αvβ8에 9배(0.2 vs 1.8 nM), αvβ3에 2.25배 더 강하다.
        GLPG0187은 완전 평탄한 pan-αv다. 목표 프로파일 참조군이 아니다."""
        labels = {label for _, label in read_smi(PANEL)}
        self.assertNotIn("CWHM-12", labels)
        self.assertNotIn("GLPG0187", labels)

    def test_c8_is_absent(self):
        labels = {label for _, label in read_smi(PANEL)}
        self.assertNotIn("C8", labels)


if __name__ == "__main__":
    unittest.main()
