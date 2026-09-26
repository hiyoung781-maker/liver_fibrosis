#!/usr/bin/env python3
"""Curate the integrin alphaVbeta1 actives sets for the REINVENT4 blueprint (section 3.1).

Builds four SMILES sets with deliberately separated roles:

The .smi files carry FLATTENED (non-isomeric) SMILES because that is what the
generator can represent: the de novo `reinvent.prior` vocabulary contains no
stereochemistry tokens at all (no @, @@, / or \\). Feeding isomeric SMILES to
transfer learning would let read_smiles_csv_file() drop every stereocentre-bearing
molecule silently - 27 of the core's 29. The isomeric form is kept in
actives_annotated.csv, which is the record of truth and the input for docking.

  actives_core.smi      TL-A input (non-RGD) + RL inception memory
  actives_core_B.smi    TL-B input: TL-A plus every confirmed RGD-zwitterion active
  actives_extended.smi  novelty NN-Tanimoto baseline ONLY (not TL, not QSAR)
  similarity_refs.smi   one molecule per TanimotoSimilarity endpoint
  benchmark_panel.smi   section 3.4 positive panel

Sources (all IDs verified against ChEMBL_37 at execution time):
  ChEMBL document CHEMBL5500400  Sabat et al. J Med Chem 2024, 67, 10306-10320
                                 doi 10.1021/acs.jmedchem.4c00743
  ChEMBL target   CHEMBL2111407  Integrin alpha-V/beta-1
  RCSB CCD        A1AFA          the 8W30 ligand TR01225179, with stereochemistry
  PubChem         tool compounds by name

Network responses are cached under data/.cache so the run is reproducible offline.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / ".cache"

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
SABAT_DOC = "CHEMBL5500400"
AVB1_TARGET = "CHEMBL2111407"
LIGAND_CCD = "A1AFA"

# All five are RGD mimetics.  They are in the core set for potency signal at the
# user's direction; the chemotype column keeps their contribution separable at D3.
# "C8" from the blueprint is deliberately absent: it is not a Sabat compound but an
# external comparator the paper cites, and the name resolves to nothing in PubChem.
# See the "Notes" section of the curation log.
TOOL_COMPOUNDS = ["bexotegrast", "CWHM-12", "GLPG0187", "PLN-1474"]

# Arg-mimetic basic heads.  A molecule carrying one binds in the zwitterionic RGD
# mode; the 8W30 ligand and the Sabat series carry none.
ARG_MIMIC_HEADS = {
    "guanidine/cyclic amidine": "[NX3][CX3](=[NX2])[NX3]",
    "tetrahydronaphthyridine": "[$(C1CCc2cccnc2N1),$(C1CCc2ccc[nX2]c2N1)]",
    "2-aminopyridine": "[nX2]1ccccc1[NX3;H1,H2]",
    "benzimidazol-2-amine": "[nX2]1c([NX3])[nX3]c2ccccc12",
    # exocyclic amine on an aromatic carbon flanked by two ring nitrogens: the
    # 2-aminoimidazole / 2-aminoimidazoline family, a classic integrin Arg mimic
    "2-aminoazole": "[NX3;H1,H2;!$(N=*)][c]([nX2,nX3])[nX2,nX3]",
    "aliphatic amine": (
        "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(Na);!$(N[SX4]);!$([N+](=O))]([CX4])[CX4,#1]"
    ),
}
CARBOXYLIC_ACID = "[CX3](=O)[OX2H1,OX1-]"


# --------------------------------------------------------------------------- io


def fetch(url: str, cache_key: str, tries: int = 4):
    """GET a JSON endpoint, caching the parsed response under data/.cache."""
    path = CACHE / f"{cache_key}.json"
    if path.exists():
        return json.loads(path.read_text())

    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.load(response)
            break
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3)

    path.write_text(json.dumps(payload))
    return payload


def fetch_text(url: str, cache_key: str, tries: int = 4) -> str:
    path = CACHE / cache_key
    if path.exists():
        return path.read_text()

    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = response.read().decode("utf-8", "replace")
            break
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(3)

    path.write_text(payload)
    return payload


def chembl_activities(target_id: str) -> list[dict]:
    activities, offset = [], 0
    while True:
        page = fetch(
            f"{CHEMBL}/activity.json?target_chembl_id={target_id}&limit=1000&offset={offset}",
            f"act_{target_id}_{offset}",
        )
        activities += page["activities"]
        offset += 1000
        if offset >= page["page_meta"]["total_count"]:
            return activities


def chembl_molecules(molecule_ids: list[str]) -> dict[str, dict]:
    molecules = {}
    for i in range(0, len(molecule_ids), 50):
        chunk = ",".join(molecule_ids[i : i + 50])
        page = fetch(
            f"{CHEMBL}/molecule.json?molecule_chembl_id__in={chunk}&limit=50", f"mol_{i:04d}"
        )
        for molecule in page["molecules"]:
            molecules[molecule["molecule_chembl_id"]] = molecule
    return molecules


def pubchem_smiles(name: str) -> str | None:
    quoted = urllib.parse.quote(name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quoted}"
        "/property/SMILES,ConnectivitySMILES/JSON"
    )
    try:
        payload = fetch(url, f"pubchem_{name.replace(' ', '_')}")
    except Exception:
        return None
    properties = payload["PropertyTable"]["Properties"][0]
    return properties.get("SMILES") or properties.get("ConnectivitySMILES")


def ccd_smiles(comp_id: str) -> str:
    """Isomeric SMILES for a PDB chemical component, straight from the RCSB CCD.

    ChEMBL's copy of this molecule (CHEMBL5556476) has lost its stereocentre, so the
    crystal-derived definition is the one to trust when isomeric_smiles = true.
    """
    text = fetch_text(
        f"https://files.rcsb.org/ligands/download/{comp_id}.cif", f"ccd_{comp_id}.cif"
    )
    for line in text.splitlines():
        if line.startswith(f"{comp_id} SMILES_CANONICAL") and "CACTVS" in line:
            return line.split('"')[1]
    raise RuntimeError(f"no SMILES_CANONICAL descriptor for {comp_id}")


# ------------------------------------------------------------------ chemistry


PRIOR = ROOT / "REINVENT4" / "priors" / "reinvent.prior"
_TOKEN_RE = re.compile(r"(\[[^\]]*\]|Br|Cl|.)")


def prior_vocabulary(path: Path = PRIOR) -> set[str] | None:
    """SMILES tokens the de novo prior can represent, read from its checkpoint.

    torch is not needed: a .prior is a zip and the token strings sit in the pickle.
    Returns None if the prior is not present, which turns the gate into a no-op.
    """
    if not path.exists():
        return None
    import io
    import pickletools

    with zipfile.ZipFile(path) as archive:
        entry = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        blob = archive.read(entry)
    listing = io.StringIO()
    try:
        pickletools.dis(blob, listing)
    except Exception:
        pass
    text = listing.getvalue()
    found = set(re.findall(r"BINUNICODE '([^']*)'", text))
    # The pickle also holds metadata keys and tensor names, so keep only strings
    # shaped like a SMILES token: a bracketed atom, a two-letter element, a %NN
    # ring closure, or a single SMILES character.  "^" and "$" are the model's
    # start/stop control tokens, not chemistry.
    smiles_token = re.compile(r"^(\[[^\]]*\]|Br|Cl|%\d{2}|[A-Za-z0-9#=+\-()/\\@.*])$")
    return {t for t in found if smiles_token.match(t) and t not in {"^", "$"}}


def flatten(smiles: str) -> str:
    """Canonical SMILES with stereochemistry removed - what the generator sees."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)


def out_of_vocabulary(smiles: str, vocab: set[str]) -> list[str]:
    return sorted(set(_TOKEN_RE.findall(smiles)) - vocab)


_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smiles: str) -> tuple[str, str] | None:
    """Strip salts, neutralise, return (isomeric canonical SMILES, InChIKey)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol)
    mol = _LARGEST.choose(mol)
    mol = _UNCHARGER.uncharge(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    remade = Chem.MolFromSmiles(canonical)
    if remade is None:
        return None
    return canonical, Chem.MolToInchiKey(remade)


# Tautomer canonicalization lives in scripts/normalize.py, NOT here. standardize()
# feeds data/actives_core.smi, which is the TL-A training input and the RL inception
# memory; changing it would invalidate the D3 result. Every Tanimoto and Murcko
# comparison imports canonical_tautomer from normalize instead.
from normalize import canonical_tautomer, canonical_tautomer_smiles  # noqa: E402


_FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=3, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)


def tanimoto(a: str, b: str) -> float:
    """ECFP-style count-fingerprint similarity, matching the RL component's defaults."""
    fps = [_FPGEN.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in (a, b)]
    return DataStructs.TanimotoSimilarity(*fps)


_HEAD_PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in ARG_MIMIC_HEADS.items()}
_ACID_PATTERN = Chem.MolFromSmarts(CARBOXYLIC_ACID)
for _name, _pattern in _HEAD_PATTERNS.items():
    assert _pattern is not None, f"bad SMARTS for {_name}"
assert _ACID_PATTERN is not None


_BASIC_N = Chem.MolFromSmarts(
    "[NX3;!$(N[#6,#16]=[O,S,N]);!$(N[SX4]);!$(N[CX3]=[CX3]);!$(Na);!$([N+](=O)[O-])]"
)


def describe(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    heads = [name for name, p in _HEAD_PATTERNS.items() if mol.HasSubstructMatch(p)]
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return {
        "has_carboxylic_acid": mol.HasSubstructMatch(_ACID_PATTERN),
        "chemotype": "RGD-zwitterion" if heads else "non-RGD",
        "arg_mimic_heads": "|".join(heads),
        "unflagged_basic_N": (not heads) and mol.HasSubstructMatch(_BASIC_N),
        "murcko_scaffold": scaffold,
        "MW": round(Descriptors.MolWt(mol), 1),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "cLogP": round(Crippen.MolLogP(mol), 2),
        "QED": round(QED.qed(mol), 3),
        "NumRotBond": Descriptors.NumRotatableBonds(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
    }


# ------------------------------------------------------------------- assembly


def best_avb1_potency(activities: list[dict]) -> dict[str, tuple[float, dict]]:
    """Most potent measured IC50/Ki/Kd per molecule, uncensored records only."""
    best: dict[str, tuple[float, dict]] = {}
    for activity in activities:
        if activity["standard_type"] not in ("IC50", "Ki", "Kd"):
            continue
        if activity.get("standard_units") != "nM":
            continue
        if activity.get("standard_relation") not in ("=", "<", "<="):
            continue
        value = activity.get("standard_value")
        if value is None:
            continue
        value = float(value)
        if value <= 0:
            continue
        key = activity["molecule_chembl_id"]
        if key not in best or value < best[key][0]:
            best[key] = (value, activity)
    return best


def main() -> int:
    DATA.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    log: list[str] = []

    def note(line: str = "") -> None:
        print(line)
        log.append(line)

    note("# Curation log - actives sets for integrin alphaVbeta1 (blueprint section 3.1)")
    note()
    note(f"Generated by `scripts/{Path(__file__).name}`.")
    note()

    status = fetch(f"{CHEMBL}/status.json", "status")
    note(f"ChEMBL release: {status['chembl_db_version']} ({status['chembl_release_date']})")
    note()

    activities = chembl_activities(AVB1_TARGET)
    molecule_ids = sorted({a["molecule_chembl_id"] for a in activities})
    molecules = chembl_molecules(molecule_ids)
    note("## Verified source identifiers")
    note()
    note(f"- target `{AVB1_TARGET}` Integrin alpha-V/beta-1: "
         f"{len(activities)} activities over {len(molecule_ids)} molecules")
    note(f"- document `{SABAT_DOC}` Sabat et al. J Med Chem 2024: "
         f"{len({a['molecule_chembl_id'] for a in activities if a['document_chembl_id'] == SABAT_DOC})}"
         " molecules")
    note(f"- CCD `{LIGAND_CCD}`: the 8W30 ligand TR01225179")
    note()

    potency = best_avb1_potency(activities)
    first_doc: dict[str, str] = {}
    for activity in activities:
        first_doc.setdefault(activity["molecule_chembl_id"], activity["document_chembl_id"])

    # ---- assemble every candidate, keyed by InChIKey
    records: dict[str, dict] = {}

    def add(smiles: str, source: str, identifier: str, potency_nm: float | None) -> str | None:
        result = standardize(smiles)
        if result is None:
            note(f"  ! unparsable SMILES from {source}/{identifier}")
            return None
        canonical, inchikey = result
        if inchikey in records:
            existing = records[inchikey]
            existing["source"] = f"{existing['source']}+{source}"
            if potency_nm is not None and (
                existing["ic50_nM"] is None or potency_nm < existing["ic50_nM"]
            ):
                existing["ic50_nM"] = potency_nm
                existing["pIC50"] = round(9 - np.log10(potency_nm), 2)
            return inchikey
        records[inchikey] = {
            "inchikey": inchikey,
            "smiles": canonical,
            "smiles_flat": flatten(canonical),
            "source": source,
            "identifier": identifier,
            "ic50_nM": potency_nm,
            "pIC50": round(9 - np.log10(potency_nm), 2) if potency_nm else None,
            "document": first_doc.get(identifier, ""),
            **describe(canonical),
        }
        return inchikey

    # the crystal ligand, with stereochemistry
    ligand_key = add(ccd_smiles(LIGAND_CCD), "RCSB-CCD", LIGAND_CCD, None)

    # every ChEMBL molecule measured against alphaVbeta1
    for molecule_id in molecule_ids:
        structures = molecules[molecule_id].get("molecule_structures") or {}
        smiles = structures.get("canonical_smiles")
        if not smiles:
            continue
        value = potency[molecule_id][0] if molecule_id in potency else None
        add(smiles, "ChEMBL", molecule_id, value)

    # named tool compounds
    tool_keys: dict[str, str] = {}  # name -> inchikey
    note("## Tool compounds")
    note()
    for name in TOOL_COMPOUNDS:
        smiles = pubchem_smiles(name)
        if not smiles:
            note(f"- {name}: NOT RESOLVED in PubChem - excluded, structure not public under this name")
            continue
        key = add(smiles, "PubChem", name, None)
        if key:
            tool_keys[name] = key
            records[key]["tool_name"] = name
            record = records[key]
            note(f"- {name}: {record['chemotype']}, MW {record['MW']}, "
                 f"TPSA {record['TPSA']}, QED {record['QED']}")
    note()

    # ---- the four sets
    actives_1um = {
        k for k, r in records.items() if r["ic50_nM"] is not None and r["ic50_nM"] <= 1000
    }
    # A ChEMBL molecule earns a place in the core only as a CONFIRMED non-RGD active:
    # an uncensored alphaVbeta1 IC50/Ki/Kd <= 1 uM.  The crystal ligand and the named
    # tool compounds are admitted on their literature record instead.  Every ChEMBL
    # molecule that fails the rule is listed under "Excluded from the core" below.
    core_keys = {k for k in actives_1um if records[k]["chemotype"] == "non-RGD"}
    core_keys |= {ligand_key} | set(tool_keys.values())
    core_keys = {k for k in core_keys if k}
    extended_keys = actives_1um

    excluded = []
    for key, record in records.items():
        if key in core_keys or record["chemotype"] != "non-RGD":
            continue
        if record["ic50_nM"] is None:
            reason = "no uncensored quantitative alphaVbeta1 IC50/Ki/Kd"
        else:
            reason = f"weakest-case alphaVbeta1 IC50 {record['ic50_nM']:.0f} nM > 1 uM"
        excluded.append((record["identifier"], record["document"], reason,
                         record["has_carboxylic_acid"]))

    # Each reference costs one endpoint, so spend them on molecules that are potent AND
    # mutually dissimilar - four congeners of one series would buy almost one signal.
    potent = sorted(
        (k for k in core_keys
         if records[k]["chemotype"] == "non-RGD" and records[k]["pIC50"] is not None),
        key=lambda k: -records[k]["pIC50"],
    )
    reference_keys = [ligand_key]                       # the crystal ligand anchors the set
    if potent:
        reference_keys.append(potent[0])                # the most potent non-RGD active
    for key in potent[1:]:
        if len(reference_keys) >= 5:
            break
        worst = max(tanimoto(records[key]["smiles"], records[c]["smiles"])
                    for c in reference_keys)
        if worst < 0.5:                                 # greedy MaxMin at a 0.5 ceiling
            reference_keys.append(key)
    pln = tool_keys.get("PLN-1474")                     # the drug-like RGD comparator
    if pln and pln not in reference_keys and len(reference_keys) < 6:
        reference_keys.append(pln)
    sabat = potent

    panel_keys = [ligand_key] + list(tool_keys.values())
    if sabat:
        panel_keys.append(sabat[0])  # most potent non-RGD Sabat compound = the cpd 25 class

    def write_set(filename: str, keys, header: list[str]) -> list[str]:
        # Deduplicate on the FLATTENED SMILES, not the InChIKey. Curation dedupes by
        # isomeric InChIKey, but the prior has no stereochemistry tokens, so distinct
        # stereoisomers collapse to one string the generator cannot tell apart -
        # 7 such pairs in core_B. Writing both lines would put the same molecule in
        # inception memory twice and overstate what the model actually sees. The
        # isomeric records stay complete in actives_annotated.csv.
        ordered, seen = [], set()
        for k in keys:
            if not k:
                continue
            flat = records[k]["smiles_flat"]
            if flat in seen:
                continue
            seen.add(flat)
            ordered.append(k)
        lines = [f"# {line}" for line in header]
        # TAB-separated, not space: REINVENT4's read_smiles_csv_file defaults to
        # delimiter="\t", so a space-separated "SMILES NAME" line is handed to the
        # tokenizer whole. Nothing errors - the name simply makes every molecule
        # fail the vocabulary check, and transfer learning would train on an empty
        # set while inception rejects all 29 actives outright.
        lines += [
            f"{records[k]['smiles_flat']}\t"
            f"{records[k].get('tool_name') or records[k]['identifier']}"
            for k in ordered
        ]
        (DATA / filename).write_text("\n".join(lines) + "\n")
        return ordered

    core_b_keys = core_keys | extended_keys

    core = write_set(
        "actives_core.smi",
        sorted(core_keys, key=lambda k: (records[k]["chemotype"], -(records[k]["pIC50"] or 0))),
        [
            "TL-A input + RL inception memory.",
            "Every non-RGD ChEMBL molecule with an uncensored alphaVbeta1 IC50/Ki/Kd <= 1 uM,",
            "plus the 8W30 crystal ligand and the named tool compounds.",
            "The tool compounds are RGD mimetics, included for potency signal at the user's",
            "direction; the chemotype column in actives_annotated.csv keeps them separable.",
        ],
    )
    core_b = write_set(
        "actives_core_B.smi",
        sorted(core_b_keys, key=lambda k: (records[k]["chemotype"], -(records[k]["pIC50"] or 0))),
        [
            "TL-B input: TL-A plus every confirmed RGD-zwitterion alphaVbeta1 active.",
            "The RGD mimetics carry the strongest binding signal but sit at the TPSA and QED",
            "the project is trying to escape, so this arm is run to be measured against TL-A",
            "at D3, not because it is expected to win.",
        ],
    )
    extended = write_set(
        "actives_extended.smi",
        sorted(extended_keys, key=lambda k: -(records[k]["pIC50"] or 0)),
        [
            "Novelty NN-Tanimoto baseline ONLY.",
            "Every ChEMBL alphaVbeta1 molecule with a measured IC50/Ki/Kd <= 1 uM.",
            "NOT transfer-learning input and NOT QSAR training data - see the curation log.",
        ],
    )
    references = write_set(
        "similarity_refs.smi",
        reference_keys,
        [
            "TanimotoSimilarity references: ONE MOLECULE PER ENDPOINT.",
            "REINVENT4 computes one score array per reference SMILES but",
            "compute_scores.py zips those arrays against the endpoint list, so every",
            "reference after the first in a single endpoint's params.smiles is silently",
            "discarded.  Declare each line below as its own endpoint.",
        ],
    )
    panel = write_set(
        "benchmark_panel.smi",
        panel_keys,
        ["Section 3.4 positive benchmark panel. Scored, never trained on."],
    )

    # ---- scaffold k-fold for diagnosing transfer learning (blueprint section 4)
    #
    # These folds diagnose TL; they do not define the model that generates. The
    # production prior trains on the FULL set (actives_core.smi / actives_core_B.smi)
    # with the epoch count the folds identify - withholding 6 of 29 molecules
    # permanently would discard 21% of the public non-RGD chemotype.
    #
    # Folds are scaffold-disjoint because a random split leaked: 50% of the core's
    # validation molecules and 65% of core_B's shared a Murcko scaffold with a
    # training molecule, so held-out NLL measured recall of a near-twin rather than
    # generalisation. They are k folds rather than one split because a single split
    # of 29 molecules over 19 scaffolds puts ~6 molecules in validation, where one
    # molecule moves the estimate by 17% and the number is dominated by which
    # scaffolds happened to land there. Report mean and range across folds.
    N_FOLDS = 5
    rng = np.random.default_rng(20260925)  # fixed seed: the folds must be reproducible
    (DATA / "folds").mkdir(exist_ok=True)

    for label, keys in (("core", core), ("core_B", core_b)):
        groups: dict[str, list[str]] = {}
        for key in keys:
            groups.setdefault(records[key]["murcko_scaffold"], []).append(key)

        scaffolds = sorted(groups)  # sort first so the shuffle is the only randomness
        scaffolds = [scaffolds[i] for i in rng.permutation(len(scaffolds))]
        # Largest group first into the currently smallest fold: standard greedy
        # partitioning, which keeps the folds comparable in size. The shuffle above
        # survives as the tie-break between equally sized groups.
        scaffolds.sort(key=lambda sc: -len(groups[sc]))

        folds: list[list[str]] = [[] for _ in range(N_FOLDS)]
        for scaffold in scaffolds:
            smallest = min(range(N_FOLDS), key=lambda i: len(folds[i]))
            folds[smallest] += groups[scaffold]

        placement = {}
        for i, fold in enumerate(folds):
            for key in fold:
                placement.setdefault(records[key]["murcko_scaffold"], set()).add(i)
        spanning = {sc for sc, seen in placement.items() if len(seen) > 1}
        assert not spanning, f"{label}: scaffold spans folds: {spanning}"

        note(f"### `actives_{label}.smi` — {N_FOLDS}-fold scaffold split")
        note()
        note("| fold | validation n | scaffolds | RGD-zwitterion | pIC50 range |")
        note("|---|---|---|---|---|")
        for i, fold in enumerate(folds, 1):
            train = [k for k in keys if k not in set(fold)]
            for part, subset in (("train", train), ("valid", fold)):
                write_set(f"folds/actives_{label}_fold{i}_{part}.smi", subset, [
                    f"TL diagnostic fold {i}/{N_FOLDS}, {part} ({len(subset)}/{len(keys)}), "
                    "scaffold-disjoint, seed 20260925.",
                    "For choosing num_epochs only. The production prior trains on the",
                    f"full actives_{label}.smi with the epoch count these folds identify.",
                ])
            pot = [records[k]["pIC50"] for k in fold if records[k]["pIC50"] is not None]
            rgd = sum(1 for k in fold if records[k]["chemotype"] == "RGD-zwitterion")
            span = f"{min(pot):.1f}–{max(pot):.1f}" if pot else "n/a"
            note(f"| {i} | {len(fold)} | {len({records[k]['murcko_scaffold'] for k in fold})} "
                 f"| {rgd} | {span} |")
        note()
        note(f"No Murcko scaffold spans two folds. Chemotype cannot be stratified at this "
             f"n: the core holds only 4 RGD-zwitterions in total, so they cannot be spread "
             f"evenly over {N_FOLDS} scaffold-disjoint folds.")
        note()

    # ---- novelty calibration band (blueprint sections 8.3 and 9)
    #
    # The old criterion, NN-Tanimoto < 0.4, was a convention borrowed from binary
    # ECFP4 and applied to a feature-based Morgan r=3 COUNT fingerprint, where the
    # same pair scores far higher. Measured against its own reference data it is
    # absurd: 83% of the core actives and 99% of the extended ones sit at >= 0.4
    # from an active with a DIFFERENT Murcko scaffold, so the rule would call almost
    # every published alphaVbeta1 series un-novel with respect to the others.
    #
    # The band below replaces it with something the data defines: how far apart are
    # two known actives that belong to different scaffolds? A generated molecule
    # below that band is at least as distant from the actives as one published series
    # is from another, which is what "a new chemotype" has to mean here.
    note("## Novelty calibration band")
    note()
    note("Nearest-neighbour Tanimoto from each active to an active with a *different*")
    note("Murcko scaffold, under the same fingerprint the scoring components use")
    note("(Morgan radius 3, feature invariants, counts).")
    note()
    note("| reference set | n | p25 | median | p75 | p90 |")
    note("|---|---|---|---|---|---|")
    band = {}
    for label, keys in (("actives_core", core), ("actives_extended", extended)):
        fps = [_FPGEN.GetCountFingerprint(Chem.MolFromSmiles(records[k]["smiles_flat"]))
               for k in keys]
        scaffolds = [records[k]["murcko_scaffold"] for k in keys]
        cross = []
        for i in range(len(keys)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            other = [sims[j] for j in range(len(keys)) if scaffolds[j] != scaffolds[i]]
            if other:
                cross.append(max(other))
        q = {f"p{p_}": round(float(np.percentile(cross, p_)), 3) for p_ in (25, 50, 75, 90)}
        band[label] = {"n": len(cross), **q}
        note(f"| `{label}` | {len(cross)} | {q['p25']:.3f} | {q['p50']:.3f} | "
             f"{q['p75']:.3f} | {q['p90']:.3f} |")
    note()
    note(f"**Novelty threshold = p25 of the `actives_extended` band = "
         f"{band['actives_extended']['p25']:.3f}.** A generated molecule below it is "
         "further from every known active than three quarters of those actives are from "
         "the nearest active of a different scaffold. Report the full distribution and "
         "each lead's percentile within this band, not only the pass/fail count - the "
         "band is fingerprint-specific and the percentile survives a change of "
         "fingerprint in a way a bare cut-off does not.")
    note()
    (DATA / "novelty_band.json").write_text(json.dumps(band, indent=2) + "\n")

    # ---- annotated table
    columns = [
        "inchikey", "smiles", "identifier", "source", "document", "ic50_nM", "pIC50",
        "smiles_flat", "chemotype", "arg_mimic_heads", "unflagged_basic_N",
        "has_carboxylic_acid",
        "MW", "TPSA", "cLogP",
        "QED", "NumRotBond", "HBD", "HBA", "murcko_scaffold",
    ]
    membership = {"core": set(core), "core_B": set(core_b), "extended": set(extended),
                  "similarity_ref": set(references), "benchmark": set(panel)}
    with (DATA / "actives_annotated.csv").open("w") as handle:
        handle.write(",".join(columns + list(membership)) + "\n")
        for key, record in sorted(records.items(), key=lambda kv: -(kv[1]["pIC50"] or 0)):
            row = [str(record.get(c, "")) for c in columns]
            row += ["1" if key in members else "0" for members in membership.values()]
            handle.write(",".join('"' + c.replace('"', '""') + '"' for c in row) + "\n")

    # ---- QC gate and summary
    note("## Notes")
    note()
    note("**C8 is unresolved and deliberately excluded.** The blueprint lists it as a tool")
    note("compound, but Sabat et al. cite it as a previously reported external inhibitor, not")
    note("as one of their own compounds, and the name resolves to nothing in PubChem. Table 4")
    note("of the paper gives C8 a cell-adhesion alphaVbeta1 pIC50 of 7.90 and describes a")
    note("phenylsulfonamidopyrrolidine. The closest candidate in ChEMBL is `CHEMBL3957812`")
    note("(pIC50 7.9, N-arylsulfonyl-L-proline scaffold, document `CHEMBL3862028`), but the")
    note("match rests on one coincident number, so C8 is left out rather than guessed.")
    note()
    note("**The paper's numbered series is fully covered.** Compound numbering runs to 25 plus")
    note("the external C8 (Table 4: *Cellular Selectivity Data for C8 and Benzimidazolone 25*),")
    note("and ChEMBL holds 26 molecules for the document. No PDF transcription is needed.")
    note()
    note("**alphaVbeta1 selectivity does not require leaving the RGD zwitterion.** Document")
    note("`CHEMBL3862028` (*Exploring N-Arylsulfonyl-L-proline Scaffold as a Platform for Potent")
    note("and Selective alphaVbeta1 Integrin Inhibitors*, ACS Med Chem Lett 2016) contributes 36")
    note("molecules, all of them selective and sub-nM yet all carrying a guanidine or")
    note("2-aminopyridine Arg mimic. They are correctly classified RGD-zwitterion and correctly")
    note("kept out of the core. Selectivity and chemotype are independent axes; it is")
    note("permeability, not selectivity, that the non-RGD core is chosen for.")
    note()

    note("## Excluded from the core")
    note()
    note("Non-RGD molecules in the alphaVbeta1 activity table that are not confirmed actives.")
    note("Kept in `actives_annotated.csv` with `core=0` so the decision stays auditable.")
    note()
    if excluded:
        note("| molecule | document | carboxylic acid | reason |")
        note("|---|---|---|---|")
        for identifier, document, reason, acid in sorted(excluded):
            note(f"| `{identifier}` | `{document}` | {'yes' if acid else 'NO'} | {reason} |")
    else:
        note("None.")
    note()
    note("`CHEMBL1629507` is a ChEMBL curation error: the paper is *Small molecule inhibitors")
    note("of hantavirus infection* (Bioorg Med Chem Lett 2010) and its assay description reads")
    note('"Inhibition of human integr**ase** alphaVbeta1 ... in african green monkey Vero E6')
    note('cells" - a hantavirus cell-infection assay, not an alphaVbeta1 binding assay, with no')
    note("quantitative value on any record.  Its four molecules are not alphaVbeta1 actives.")
    note()
    note("`CHEMBL5556476` is the stereochemistry-stripped copy of the 8W30 ligand (InChIKey")
    note("skeleton KDUPLHIWRBQFEC, no stereo layer) recorded at 5012 nM, against the crystal")
    note("ligand's defined L configuration.  The CCD definition of `A1AFA` is used instead.")
    note()
    note("## QC gate: prior vocabulary")
    note()
    vocab = prior_vocabulary()
    if vocab is None:
        note(f"SKIPPED - `{PRIOR.relative_to(ROOT)}` not present (check out REINVENT4 v4.5.11).")
    else:
        note(f"The de novo prior's alphabet ({len(vocab)} tokens) carries **no stereochemistry**:")
        note()
        note("```")
        note(" ".join(sorted(vocab)))
        note("```")
        note()
        failures = []
        for label, keys in (("core", core), ("core_B", core_b), ("extended", extended),
                            ("similarity_refs", references), ("benchmark", panel)):
            iso_bad = sum(1 for k in keys if out_of_vocabulary(records[k]["smiles"], vocab))
            flat_bad = [k for k in keys if out_of_vocabulary(records[k]["smiles_flat"], vocab)]
            failures += flat_bad
            note(f"- `{label}`: {len(keys) - iso_bad}/{len(keys)} would survive as isomeric, "
                 f"**{len(keys) - len(flat_bad)}/{len(keys)} as written (flattened)**")
        note()
        if failures:
            note(f"**{len(failures)} molecule(s) still carry out-of-vocabulary tokens:**")
            for key in failures:
                note(f"- `{records[key]['identifier']}` "
                     f"{out_of_vocabulary(records[key]['smiles_flat'], vocab)}")
        else:
            note("Every emitted SMILES is representable by the prior.")
    note()

    note("## QC gate: unflagged basic nitrogen in the core")
    note()
    note("The chemotype call drives TL-A, so any core molecule carrying a protonatable")
    note("nitrogen that no Arg-mimic pattern matched is listed here for manual review.")
    note()
    suspects = [k for k in core if records[k]["unflagged_basic_N"]]
    if suspects:
        note("| molecule | SMILES |")
        note("|---|---|")
        for key in suspects:
            label = records[key].get("tool_name") or records[key]["identifier"]
            note(f"| `{label}` | `{records[key]['smiles']}` |")
    else:
        note("None - every core molecule is either flagged RGD or carries no basic nitrogen.")
    note()

    note("## QC gate: carboxylic acid")
    note()
    offenders = [k for k in core if not records[k]["has_carboxylic_acid"]]
    if offenders:
        note(f"{len(offenders)} core molecule(s) carry no carboxylic acid - REPORTED, NOT DROPPED:")
        for key in offenders:
            note(f"- `{records[key]['identifier']}` {records[key]['smiles']}")
    else:
        note("All core molecules carry a carboxylic acid.")
    note()

    note("## Sets written")
    note()
    note("| file | n | role |")
    note("|---|---|---|")
    note(f"| `actives_core.smi` | {len(core)} | TL-A input + inception |")
    note(f"| `actives_core_B.smi` | {len(core_b)} | TL-B input |")
    note(f"| `actives_extended.smi` | {len(extended)} | novelty NN baseline only |")
    note(f"| `similarity_refs.smi` | {len(references)} | one molecule per endpoint |")
    note(f"| `benchmark_panel.smi` | {len(panel)} | section 3.4 positives |")
    note()

    def summarise(label: str, keys) -> None:
        rows = [records[k] for k in keys]
        if not rows:
            return
        scaffolds = {r["murcko_scaffold"] for r in rows}
        rgd = sum(1 for r in rows if r["chemotype"] == "RGD-zwitterion")
        med = lambda field: np.median([r[field] for r in rows])
        note(f"| {label} | {len(rows)} | {len(scaffolds)} | {rgd} ({100 * rgd / len(rows):.0f}%) "
             f"| {med('MW'):.0f} | {med('TPSA'):.0f} | {med('cLogP'):.2f} | {med('QED'):.2f} |")

    note("| set | n | Murcko scaffolds | RGD-zwitterion | MW | TPSA | cLogP | QED |")
    note("|---|---|---|---|---|---|---|---|")
    summarise("core (TL-A)", core)
    summarise("core_B (TL-B)", core_b)
    summarise("extended", extended)
    summarise("similarity_refs", references)
    note()

    (DATA / "CURATION_LOG.md").write_text("\n".join(log) + "\n")
    print(f"\nwrote {DATA}/CURATION_LOG.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
