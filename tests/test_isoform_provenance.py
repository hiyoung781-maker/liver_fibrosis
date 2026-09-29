import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from prepare_isoform_receptor import StaleInputError, verify_ligand_provenance

A1AFA = "c1ccc(cc1)C[C@@H](C(=O)O)NC(=O)c2ccc(cc2Cl)Cl"
OBABEL_JUY = "O=C(CCCCc1ccc2c(n1)NCCC2)NCCC(NC(=O)c1nc2ccccc2s1)C(=O)O"
CCD_JUY = "c1ccc2c(c1)nc(s2)C(=O)N[C@@H](CCNC(=O)CCCCc3ccc4cccnc4n3)C(=O)O"


def make(tmp, manifest, sdf_smiles):
    d = Path(tmp)
    (d / "manifest.json").write_text(json.dumps(manifest))
    if sdf_smiles is not None:
        mol = Chem.MolFromSmiles(sdf_smiles)
        from rdkit.Chem import AllChem
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        writer = Chem.SDWriter(str(d / "ligand.sdf"))
        writer.write(mol)
        writer.close()
    return str(d)


class TestLigandProvenance(unittest.TestCase):
    """The ligand preparation failed on the cluster -- the old script was
    still checked out and did not know --ligand-ccd -- and the receptor step
    and the docking then ran happily on the Open Babel ligand.sdf left over
    from before. The whole re-run reproduced the invalid result to three
    decimal places, which is exactly how it looked like nothing had changed.

    A step that consumes a prepared file checks that the file was prepared
    the way the current pipeline prepares it."""

    def test_a_manifest_without_a_template_is_refused(self):
        # This is what the old, Open Babel path wrote.
        with TemporaryDirectory() as tmp:
            d = make(tmp, {"structure": "x.pdb", "ligand": ["JUY", "B", 712]},
                     CCD_JUY)
            with self.assertRaises(StaleInputError) as cm:
                verify_ligand_provenance(d)
            self.assertIn("template", str(cm.exception).lower())

    def test_an_sdf_that_disagrees_with_the_template_is_refused(self):
        # The exact failure: a stale Open Babel SDF beside a new manifest.
        with TemporaryDirectory() as tmp:
            d = make(tmp, {"ligand_template": CCD_JUY}, OBABEL_JUY)
            with self.assertRaises(StaleInputError) as cm:
                verify_ligand_provenance(d)
            self.assertIn("does not match", str(cm.exception))

    def test_a_matching_pair_passes(self):
        with TemporaryDirectory() as tmp:
            d = make(tmp, {"ligand_template": CCD_JUY}, CCD_JUY)
            self.assertTrue(verify_ligand_provenance(d))

    def test_charge_differences_do_not_count_as_a_mismatch(self):
        # The campaign deprotonates on purpose (spec 7.4's anion decision).
        with TemporaryDirectory() as tmp:
            d = make(tmp, {"ligand_template": A1AFA},
                     A1AFA.replace("C(=O)O", "C(=O)[O-]"))
            self.assertTrue(verify_ligand_provenance(d))

    def test_a_missing_sdf_is_refused(self):
        with TemporaryDirectory() as tmp:
            d = make(tmp, {"ligand_template": CCD_JUY}, None)
            with self.assertRaises(StaleInputError):
                verify_ligand_provenance(d)

    def test_a_missing_manifest_is_refused(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(StaleInputError):
                verify_ligand_provenance(tmp)


if __name__ == "__main__":
    unittest.main()
