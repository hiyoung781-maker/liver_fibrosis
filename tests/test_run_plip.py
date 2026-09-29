import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_plip import INTERACTION_KINDS, parse_report

MINIMAL_XML = """<?xml version="1.0" ?>
<report><bindingsite id="1">
<identifiers><hetid>A1A</hetid><longname>A1A-CA</longname></identifiers>
<interactions>
  <metal_complexes><metal_complex id="1">
    <resnr>501</resnr><restype>CA</restype><reschain>B</reschain>
    <dist>2.68</dist><metal_type>Ca</metal_type>
  </metal_complex></metal_complexes>
  <hydrogen_bonds><hydrogen_bond id="1">
    <resnr>224</resnr><restype>ASN</restype><reschain>B</reschain>
    <dist_h-a>1.95</dist_h-a><dist_d-a>2.92</dist_d-a>
    <don_angle>158.4</don_angle><protisdon>False</protisdon>
  </hydrogen_bond></hydrogen_bonds>
  <pi_stacks><pi_stack id="1">
    <resnr>178</resnr><restype>TYR</restype><reschain>A</reschain>
    <centdist>6.21</centdist><angle>74.6</angle><offset>1.4</offset>
    <type>T</type>
  </pi_stack></pi_stacks>
</interactions></bindingsite></report>
"""

EMPTY_XML = """<?xml version="1.0" ?>
<report><bindingsite id="1"><interactions/></bindingsite></report>
"""

# A multi-site report like real PLIP output on 8W30: 13 sites, most of them
# calcium or glycan sites that must be skippable. Only the site whose
# longname carries the ligand's truncated hetid (A1A) should be selected.
MULTI_SITE_XML = """<?xml version="1.0" ?>
<report>
<bindingsite id="1">
<identifiers><hetid>CA</hetid><longname>CA</longname></identifiers>
<interactions><metal_complexes><metal_complex id="1">
  <resnr>50</resnr><restype>CA</restype><reschain>A</reschain>
  <dist>2.40</dist><metal_type>Ca</metal_type>
</metal_complex></metal_complexes></interactions>
</bindingsite>
<bindingsite id="2">
<identifiers><hetid>A1A</hetid><longname>A1A-CA</longname></identifiers>
<interactions>
  <hydrogen_bonds><hydrogen_bond id="1">
    <resnr>224</resnr><restype>ASN</restype><reschain>B</reschain>
    <dist_h-a>1.95</dist_h-a><dist_d-a>2.63</dist_d-a>
    <don_angle>156.61</don_angle><protisdon>False</protisdon>
  </hydrogen_bond></hydrogen_bonds>
</interactions>
</bindingsite>
</report>
"""


class TestParseReport(unittest.TestCase):
    def test_extracts_metal_complex_with_residue_and_distance(self):
        rec = parse_report(MINIMAL_XML)
        metal = rec["metal_complexes"][0]
        self.assertEqual((metal["reschain"], metal["resnr"]), ("B", 501))
        self.assertAlmostEqual(metal["dist"], 2.68)

    def test_extracts_hbond_with_its_donor_angle(self):
        """각도가 거리 기준과 PLIP을 가르는 지점이다. 반드시 보존한다."""
        rec = parse_report(MINIMAL_XML)
        hbond = rec["hbonds"][0]
        self.assertEqual((hbond["reschain"], hbond["resnr"]), ("B", 224))
        self.assertAlmostEqual(hbond["don_angle"], 158.4)

    def test_extracts_t_shaped_pi_stack(self):
        """v1의 거리 기준은 centroid 5.5 Å으로 8W30의 6.21 Å T-stack을
        기각했다. PLIP은 유형과 각도로 판정한다."""
        rec = parse_report(MINIMAL_XML)
        stack = rec["pi_stacking"][0]
        self.assertEqual(stack["type"], "T")
        self.assertAlmostEqual(stack["centdist"], 6.21)

    def test_every_kind_is_present_as_a_list(self):
        rec = parse_report(MINIMAL_XML)
        self.assertEqual(set(rec), set(INTERACTION_KINDS))
        for kind in INTERACTION_KINDS:
            self.assertIsInstance(rec[kind], list)


class TestNoInteractionsIsNotAnError(unittest.TestCase):
    """상호작용이 하나도 없는 포즈가 파서를 죽이면 배치 전체가 멈춘다."""

    def test_empty_report_returns_empty_lists_not_an_exception(self):
        rec = parse_report(EMPTY_XML)
        self.assertEqual(set(rec), set(INTERACTION_KINDS))
        self.assertTrue(all(rec[kind] == [] for kind in INTERACTION_KINDS))


class TestMultiSiteSelection(unittest.TestCase):
    """실제 8W30 리포트는 13개 결합 부위를 갖고 대부분은 칼슘/글리칸 부위다.
    사이트를 병합하지 않고 리간드 부위 하나만 골라야 한다."""

    def test_default_parse_merges_all_sites(self):
        """사이트를 지정하지 않으면 전체를 병합한다 (배치 기본 동작)."""
        rec = parse_report(MULTI_SITE_XML)
        self.assertEqual(len(rec["metal_complexes"]), 1)
        self.assertEqual(len(rec["hbonds"]), 1)

    def test_select_site_by_hetid_isolates_ligand_site(self):
        """hetid로 사이트를 고르면 칼슘 전용 사이트의 금속 착물이
        리간드 사이트의 상호작용과 섞이지 않는다."""
        rec = parse_report(MULTI_SITE_XML, hetid="A1A")
        self.assertEqual(rec["metal_complexes"], [])
        self.assertEqual(len(rec["hbonds"]), 1)
        self.assertAlmostEqual(rec["hbonds"][0]["dist_d-a"], 2.63)


if __name__ == "__main__":
    unittest.main()
