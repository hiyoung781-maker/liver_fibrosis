import csv
import unittest

ANNOTATED = "data/actives_annotated.csv"


def load():
    with open(ANNOTATED) as handle:
        return {row["identifier"]: row for row in csv.DictReader(handle)}


class TestEndpointProvenance(unittest.TestCase):
    """v1 §3.4는 CHEMBL4649232를 'IC50 0.166 nM'으로 적었으나 실제로는 Ki다.
    출처도 1차 논문이 아니라 리뷰이며, 1차 출처는 따로 있다."""

    @classmethod
    def setUpClass(cls):
        cls.rows = load()

    def test_chembl4649232_endpoint_is_ki_not_ic50(self):
        row = self.rows["CHEMBL4649232"]
        self.assertEqual(row["endpoint_type"], "Ki")

    def test_chembl4649232_primary_source_is_the_chemmedchem_paper(self):
        """ChEMBL의 출처 문서 CHEMBL4602673은 Zheng & Leftheris 2020 리뷰이고,
        그 안의 compound 38이다. 1차 출처는 리뷰의 참고문헌 141이다."""
        row = self.rows["CHEMBL4649232"]
        self.assertIn("ChemMedChem", row["primary_source"])
        self.assertIn("2019", row["primary_source"])

    def test_compound_25_is_identified(self):
        """CHEMBL5532604 = Sabat compound 25. pIC50 9.5는 논문 Table 3의
        ELISA 값과 일치한다. 재발견하지 않도록 기록해 둔다."""
        row = self.rows["CHEMBL5532604"]
        self.assertEqual(row["paper_id"], "Sabat2024:25")
        self.assertEqual(row["pIC50"], "9.5")

    def test_every_row_declares_an_endpoint_type(self):
        blank = [ident for ident, row in self.rows.items()
                 if row["ic50_nM"] and not row["endpoint_type"]]
        self.assertEqual(blank, [])


if __name__ == "__main__":
    unittest.main()
