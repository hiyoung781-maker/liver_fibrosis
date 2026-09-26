# Step 2 단일 stage RL — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 승인된 Step 2 재설계(단일 stage RL, 카르복실레이트 게이트, Murcko 신규성)를 실제 config·스크립트·데이터 산출물로 반영하고, spec이 주장하는 모든 수치를 회귀 테스트로 고정한다.

**Architecture:** 목적함수의 파이썬 참조 구현(`scripts/objective.py`)을 새로 만들어 REINVENT를 띄우지 않고도 임의의 SMILES 집합을 채점할 수 있게 하고, 그 위에 회귀 테스트를 얹는다. 타우토머 정규화와 Murcko 신규성은 `scripts/curate_actives.py`에 헬퍼로 추가하되 **TL 학습 파일은 바이트 단위로 불변**임을 테스트로 잠근다. TOML config는 마지막에 쓰고, REINVENT의 transform 이름 해석과 pydantic `extra="forbid"`를 통과하는지 검증한다.

**Tech Stack:** Python 3, RDKit 2026.03.4 (base conda 환경), numpy, `unittest`(stdlib — 이 저장소에 pytest가 없다), REINVENT4 v4.5.11 소스(`REINVENT4/`), tomllib(stdlib).

**Spec:** `docs/superpowers/specs/2026-09-26-step2-single-stage-rl-design.md`

## Global Constraints

- 작업 디렉터리: `/home/seyoung/academic_symposium/liver_fibrosis`. 모든 경로는 이 디렉터리 기준 상대경로.
- 테스트는 `python3 -m unittest discover -s tests -t . -v`로 실행한다. **pytest를 설치하지 말 것** — 저장소에 없고 필요하지도 않다.
- 지문은 항상 Morgan **radius 3, feature invariants, counts**: `rdFingerprintGenerator.GetMorganGenerator(radius=3, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())`, `GetCountFingerprint`.
- 카르복실레이트 SMARTS는 정확히 `[CX3](=O)[OX2H1,OX1-]`.
- 목적함수: `total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SAScore, 0.5)]) × alert_filter`, `geometric_mean = prod(max(scoreᵢ, 1e-8) ** (wᵢ / Σw))`.
- transform 값은 정확히: COOH `right_step(high=1)`, TPSA `double_sigmoid(low=40, high=115, coef_div=120, coef_si=20, coef_se=20)`, SAScore `reverse_sigmoid(low=6.0, high=8.0, k=0.5)`.
- `CustomAlerts` SMARTS 목록은 정확히: `["[NX3;H2][c]", "[*;r8]", "[*;r9]", "[*;r10]", "N=[N+]=[N-]", "C(=O)Cl", "[SH]", "[Nr0][Nr0]"]` + RDKit PAINS 카탈로그.
- RL 파라미터: `max_steps = 600`, `type = "dap"`, `sigma = 128`, `rate = 0.0001`, `batch_size = 128`, `diversity_filter = IdenticalMurckoScaffold(bucket_size = 25, minscore = 0.4)` 전 구간, `inception`은 `data/actives_core.smi` 시드.
- **`data/actives_core.smi`와 `data/actives_core_B.smi`는 바이트 단위로 변경 금지.** TL-A prior가 이 파일로 학습됐고 spec §4가 D3 결과를 불변으로 선언한다.
- **`torch`가 없다.** base conda 환경에 RDKit은 있지만 torch는 없어서 `import reinvent.scoring.transforms`가 실패한다. REINVENT의 transform 클래스를 테스트에서 쓰려면 `tests/test_config.py`의 `load_reinvent_transforms()`처럼 합성 패키지 이름으로 파일에서 직접 로드한다(검증 완료). REINVENT 자체를 실행하는 것은 이 계획의 범위가 아니다.
- **import 규약:** 새 모듈(`normalize.py`, `novelty.py`, `known_scaffolds.py`, `objective.py`)은 모두 `scripts/`에 놓고 **서로를 bare 이름으로 import**한다(`from normalize import ...`). `scripts/curate_actives.py`가 `python3 scripts/curate_actives.py`로 실행되어 `sys.path[0]`이 `scripts/`가 되기 때문이다. 테스트는 `sys.path`에 `scripts/`를 넣고 같은 방식으로 import하며, 데이터 경로는 저장소 루트 기준 상대경로이므로 `python3 -m unittest discover -s tests -t .`를 **저장소 루트에서** 실행해야 한다.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

## Review Focus

spec이 요구하지만 어떤 작업의 테스트도 직접 건드리지 않는, 사람을 물 가능성이 높은 입력 다섯 가지. 각 항목은 아래 해당 Task의 스텝으로 들어간다.

1. **파싱 불가 SMILES** — 생성기는 무효 SMILES를 낸다. 정규화·신규성 헬퍼가 예외를 던지면 triage 전체가 멈춘다. `None`을 돌려주고 호출자가 건너뛰어야 한다. → Task 1
2. **`TautomerEnumerator`가 실패하거나 분자를 바꾸지 못하는 경우** — 큰/이상한 분자에서 예외 또는 입력 그대로 반환이 가능하다. 이 경우 조용히 원본을 쓰면 기준 세트와 측정 세트의 정규화가 어긋난다(spec §2.7의 비대칭이 다시 생긴다). 실패를 삼키지 말고 원본 반환 + 카운터 기록. → Task 1
3. **카르복실레이트 SMARTS가 에스터/아마이드/테트라졸을 잡는지** — 잡으면 게이트가 열려 MIDAS anchor 없는 분자가 통과한다. → Task 4
4. **비고리 분자의 Murcko scaffold는 빈 문자열 `""`** — 106개 집합에 `""`가 없으므로 "신규"로 통과하고, diversity filter에서는 모든 비고리 분자가 같은 버킷을 공유한다. 명시적으로 처리하고 보고해야 한다. → Task 3
5. **REINVENT의 config 검증** — `get_transform`은 이름을 `lower().replace("_","")`로 해석하고(`right_step` → `rightstep` → `RightStep`), 섹션 파라미터는 pydantic `extra="forbid"`라서 오타 키가 하드 실패다. TOML을 쓰고 나서 실제로 파싱·검증해야 한다. → Task 5

---

### Task 1: 테스트 하네스 + 타우토머 정규화 헬퍼

**Files:**
- Create: `tests/__init__.py` (빈 파일)
- Create: `tests/test_normalize.py`
- Modify: `scripts/curate_actives.py` (`standardize()` 아래, 현재 226–239행 근처에 헬퍼 추가)
- Create: `scripts/normalize.py`

**Interfaces:**
- Consumes: 없음 (첫 작업)
- Produces:
  - `scripts/normalize.py`:
    - `FPGEN` — 위 Global Constraints의 Morgan 생성기 인스턴스
    - `flatten(mol: Chem.Mol) -> Chem.Mol | None` — 입체화학 제거 후 재파싱
    - `canonical_tautomer(mol: Chem.Mol) -> Chem.Mol | None` — flatten + 타우토머 정규화
    - `canonical_tautomer_smiles(smiles: str) -> str | None`
    - `TAUTOMER_FAILURES: list[str]` — 정규화가 예외로 실패한 SMILES 누적 목록
  - 이후 Task 2·3·4가 전부 `scripts/normalize.py`를 import 한다.

**왜 `standardize()`를 고치지 않는가 (spec §3.6에서 벗어나는 지점):** spec §3.6은 "`curate_actives.py`의 `standardize()`에 추가"라고 적었지만, `standardize()`의 출력이 `data/actives_core.smi`(TL-A 학습 입력 + RL inception 메모리)에 그대로 쓰인다(`curate_actives.py:487`, `:493`). 거기에 타우토머 정규화를 넣으면 학습 파일이 바뀔 수 있고 spec §4가 불변으로 선언한 D3 결과가 흔들린다. 요구사항의 실질은 "**기준 세트와 측정 세트에 동일한 정규화를 비교 직전에 적용**"이므로 비교 계층에 둔다. 학습 파일 불변은 Step 5의 테스트로 잠근다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/__init__.py`는 빈 파일로 만들고, `tests/test_normalize.py`:

```python
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
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `python3 -m unittest discover -s tests -t . -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.normalize'`

- [ ] **Step 3: 최소 구현을 쓴다**

`scripts/normalize.py`:

```python
"""Shared structure normalization for every fingerprint and scaffold comparison.

Reference sets and measured sets MUST pass through the same normalization before
any Tanimoto or Murcko comparison. Blueprint section 8.0 explains why: the same
molecule written as a different tautomer scores 0.458 against itself and yields a
different Murcko SMILES, which lets a known active pass as scaffold-novel.

This module deliberately does NOT touch curate_actives.standardize(), whose output
is written to data/actives_core.smi - the TL-A training input. See the plan.
"""

from __future__ import annotations

from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

__all__ = [
    "FPGEN",
    "TAUTOMER_FAILURES",
    "canonical_tautomer",
    "canonical_tautomer_smiles",
    "flatten",
]

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=3,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)

_TAUTOMER = rdMolStandardize.TautomerEnumerator()

# SMILES whose tautomer canonicalization raised. Callers report this count rather
# than letting a silent fallback desynchronize reference and measured sets.
TAUTOMER_FAILURES: list[str] = []


def flatten(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Drop stereochemistry - the de novo prior's vocabulary has no stereo tokens."""
    if mol is None:
        return None
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))


def canonical_tautomer(mol: Chem.Mol | None) -> Chem.Mol | None:
    """Flatten, then pick RDKit's canonical tautomer. None if either step fails."""
    flat = flatten(mol)
    if flat is None:
        return None
    try:
        with rdBase.BlockLogs():
            return _TAUTOMER.Canonicalize(flat)
    except Exception:
        TAUTOMER_FAILURES.append(Chem.MolToSmiles(flat))
        return flat


def canonical_tautomer_smiles(smiles: str | None) -> str | None:
    """Canonical SMILES of the canonical tautomer, or None for unparseable input."""
    if not smiles:
        return None
    mol = canonical_tautomer(Chem.MolFromSmiles(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
```

그리고 `scripts/curate_actives.py`의 `standardize()` 정의 바로 뒤(현재 239행 다음)에 다음 주석을 추가한다. **import는 넣지 않는다** — 이 파일은 `canonical_tautomer`를 직접 쓰지 않으므로 unused import가 된다(Task 2가 `cross_scaffold_band`만 import한다):

```python
# Tautomer canonicalization lives in scripts/normalize.py, NOT here. standardize()
# feeds data/actives_core.smi, which is the TL-A training input and the RL inception
# memory; changing it would invalidate the D3 result. Every Tanimoto and Murcko
# comparison goes through normalize.canonical_tautomer instead.
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

Run: `python3 -m unittest discover -s tests -t . -v`
Expected: PASS — 9 tests

`curate_actives.py`의 import가 깨지지 않는지도 확인한다:
Run: `python3 -c "import sys; sys.path.insert(0,'scripts'); import curate_actives; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: 커밋**

```bash
git add tests/__init__.py tests/test_normalize.py scripts/normalize.py scripts/curate_actives.py
git commit -m "$(cat <<'MSG'
feat: 공용 타우토머 정규화 헬퍼 추가

모든 지문/Murcko 비교 앞에 동일한 정규화를 적용하기 위한 scripts/normalize.py.
standardize()는 건드리지 않는다 - 그 출력이 data/actives_core.smi(TL-A 학습 입력)에
쓰이므로 D3 결과가 흔들린다. 정규화 실패는 삼키지 않고 TAUTOMER_FAILURES에 기록한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: novelty band 재계산 + `novelty_band.json` 재생성

**Files:**
- Create: `tests/test_novelty_band.py`
- Create: `scripts/novelty.py`
- Modify: `data/novelty_band.json`
- Modify: `scripts/curate_actives.py:627-641` (band 계산 루프를 `scripts/novelty.py` 호출로 교체)

**Interfaces:**
- Consumes: `scripts.normalize.canonical_tautomer`, `scripts.normalize.FPGEN`
- Produces:
  - `scripts/novelty.py`:
    - `load_smi(path: str) -> list[tuple[str, str]]` — `(smiles, label)` 목록, 주석/빈 줄 제외
    - `cross_scaffold_band(smiles_list: list[str], normalize: bool) -> dict[str, float]` — `{"n": int, "p25": float, "p50": float, "p75": float, "p90": float}`
    - `murcko(mol) -> str`
- 이후 Task 3이 `murcko`와 `load_smi`를 쓴다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_novelty_band.py`:

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from novelty import cross_scaffold_band, load_smi

CORE = "data/actives_core.smi"
EXTENDED = "data/actives_extended.smi"


class TestBandReproduction(unittest.TestCase):
    """Without normalization the band must reproduce the committed JSON exactly.

    That is the check that the reimplementation matches curate_actives.py:627-641.
    """

    def test_core_unnormalized_matches_committed_json(self):
        band = cross_scaffold_band([s for s, _ in load_smi(CORE)], normalize=False)
        self.assertEqual(band["n"], 29)
        self.assertAlmostEqual(band["p25"], 0.552, places=3)
        self.assertAlmostEqual(band["p50"], 0.676, places=3)

    def test_extended_unnormalized_matches_committed_json(self):
        band = cross_scaffold_band([s for s, _ in load_smi(EXTENDED)], normalize=False)
        self.assertEqual(band["n"], 190)
        self.assertAlmostEqual(band["p25"], 0.710, places=3)
        self.assertAlmostEqual(band["p50"], 0.775, places=3)


class TestBandAfterNormalization(unittest.TestCase):
    def test_core_is_unchanged(self):
        """0 of 29 core actives change tautomer, so the band must not move."""
        band = cross_scaffold_band([s for s, _ in load_smi(CORE)], normalize=True)
        self.assertAlmostEqual(band["p25"], 0.552, places=3)

    def test_extended_p25_drops_to_0680(self):
        """14 of 190 change tautomer; the threshold gets slightly stricter."""
        band = cross_scaffold_band([s for s, _ in load_smi(EXTENDED)], normalize=True)
        self.assertAlmostEqual(band["p25"], 0.680, places=3)
        self.assertLess(band["p25"], 0.710)


class TestLoadSmi(unittest.TestCase):
    def test_skips_comments_and_blank_lines(self):
        rows = load_smi("data/benchmark_panel.smi")
        self.assertEqual(len(rows), 6)
        self.assertIn("A1AFA", [label for _, label in rows])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `python3 -m unittest tests.test_novelty_band -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.novelty'`

- [ ] **Step 3: 최소 구현을 쓴다**

`scripts/novelty.py`:

```python
"""Scaffold novelty: the cross-scaffold distance band and the Murcko gate.

Blueprint section 8.3. The band is REPORTING CONTEXT, not a gate - a distance
cut-off passes 62% of molecules that sit on a published Murcko scaffold. The gate
is Murcko scaffold membership, which lives in scripts/known_scaffolds.py.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from normalize import FPGEN, canonical_tautomer

__all__ = ["cross_scaffold_band", "load_smi", "murcko"]


def load_smi(path: str) -> list[tuple[str, str]]:
    """Read a .smi file as (smiles, label) pairs, skipping comments and blanks."""
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            rows.append((fields[0], " ".join(fields[1:])))
    return rows


def murcko(mol: Chem.Mol) -> str:
    """Murcko scaffold SMILES. Acyclic molecules give "" - see Review Focus 4."""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def _prepare(smiles_list: list[str], normalize: bool) -> list[Chem.Mol]:
    out = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if normalize:
            mol = canonical_tautomer(mol)
        else:
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
        if mol is not None:
            out.append(mol)
    return out


def cross_scaffold_band(smiles_list: list[str], normalize: bool) -> dict:
    """How far apart are two known actives that belong to different scaffolds?

    For each molecule, the nearest neighbour among molecules with a DIFFERENT Murcko
    scaffold. Returns n and the 25/50/75/90th percentiles of that distribution.
    """
    mols = _prepare(smiles_list, normalize)
    fps = [FPGEN.GetCountFingerprint(mol) for mol in mols]
    scaffolds = [murcko(mol) for mol in mols]

    cross = []
    for i in range(len(mols)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        other = [sims[j] for j in range(len(mols)) if scaffolds[j] != scaffolds[i]]
        if other:
            cross.append(max(other))

    band = {"n": len(cross)}
    for percentile in (25, 50, 75, 90):
        band[f"p{percentile}"] = round(float(np.percentile(cross, percentile)), 3)
    return band
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

Run: `python3 -m unittest tests.test_novelty_band -v`
Expected: PASS — 5 tests. 특히 `test_extended_unnormalized_matches_committed_json`이 통과하면 재구현이 `curate_actives.py`와 일치한다는 뜻이다.

- [ ] **Step 5: `data/novelty_band.json`을 정규화 기준으로 재생성한다**

```bash
python3 - <<'PY'
import json, sys
sys.path.insert(0, "scripts")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from novelty import cross_scaffold_band, load_smi

band = {}
for label, path in (("actives_core", "data/actives_core.smi"),
                    ("actives_extended", "data/actives_extended.smi")):
    band[label] = cross_scaffold_band([s for s, _ in load_smi(path)], normalize=True)
band["_note"] = ("Tautomer-canonicalized (blueprint 8.0). REPORTING CONTEXT ONLY - "
                 "the novelty gate is Murcko scaffold membership, see "
                 "data/known_scaffolds.smi. Pre-normalization values were "
                 "core p25 0.552, extended p25 0.710.")
with open("data/novelty_band.json", "w") as handle:
    json.dump(band, handle, indent=2)
    handle.write("\n")
print(json.dumps(band, indent=2))
PY
```
Expected: `actives_core.p25 == 0.552`, `actives_extended.p25 == 0.68`

- [ ] **Step 6: `curate_actives.py`의 band 루프를 교체한다**

`scripts/curate_actives.py:627-641`의 `for label, keys in (("actives_core", core), ...)` 루프 본문에서 지문/scaffold/cross 계산을 지우고 `cross_scaffold_band`를 호출하게 바꾼다. `note(...)` 표 출력과 `band[label] = ...` 대입은 유지하고, 표 헤더에 정규화 사실을 적는다:

```python
    note("Nearest-neighbour Tanimoto from each active to an active with a *different*")
    note("Murcko scaffold, under the same fingerprint the scoring components use")
    note("(Morgan radius 3, feature invariants, counts), AFTER tautomer")
    note("canonicalization (blueprint 8.0).")
    note()
    note("| reference set | n | p25 | median | p75 | p90 |")
    note("|---|---|---|---|---|---|")
    band = {}
    for label, keys in (("actives_core", core), ("actives_extended", extended)):
        smiles_list = [records[k]["smiles_flat"] for k in keys]
        q = cross_scaffold_band(smiles_list, normalize=True)
        band[label] = q
        note(f"| `{label}` | {q['n']} | {q['p25']:.3f} | {q['p50']:.3f} | "
             f"{q['p75']:.3f} | {q['p90']:.3f} |")
```

파일 상단 import 블록(현재 44행 `from rdkit.Chem.Scaffolds import MurckoScaffold` 다음)에 추가한다:

```python
from novelty import cross_scaffold_band
```

**그리고 같은 함수의 threshold 안내문을 고친다(현재 `curate_actives.py:648-654`).** 그 `note(...)` 호출이 `data/CURATION_LOG.md`에 "**Novelty threshold = p25 of the `actives_extended` band**"를 써 넣는데, 이는 §8.3이 거리 기준을 게이트에서 철회하고 Murcko scaffold 소속으로 교체한 것과 정면으로 모순된다. 다음으로 교체한다:

```python
    note("**This band is REPORTING CONTEXT, not the novelty gate.** The gate is "
         "Murcko scaffold membership - see `data/known_scaffolds.smi` and blueprint "
         "section 8.3. A distance cut-off was withdrawn because it passes 62% of "
         "molecules whose Murcko scaffold is IDENTICAL to a published active's. "
         "Report each lead's nearest-neighbour Tanimoto and its percentile within "
         "this band alongside the binary scaffold verdict: the percentile survives "
         "a change of fingerprint in a way a bare cut-off does not.")
```

위 블록 바로 앞의 설명 주석(현재 620-623행, "The band below replaces it with something the data defines ... which is what \"a new chemotype\" has to mean here")도 같은 이유로 낡았다. 그 네 줄을 다음으로 교체한다:

```python
    # The band below is calibration context: how far apart are two known actives that
    # belong to different scaffolds? It is NOT the gate - see section 8.3. The gate is
    # binary Murcko scaffold membership, because a distance cut-off passes molecules
    # that sit on a published scaffold.
```

- [ ] **Step 7: 테스트를 다시 돌리고 커밋한다**

Run: `python3 -m unittest discover -s tests -t . -v`
Expected: PASS — 14 tests

```bash
git add tests/test_novelty_band.py scripts/novelty.py scripts/curate_actives.py data/novelty_band.json
git commit -m "$(cat <<'MSG'
feat: novelty band을 타우토머 정규화 후로 재계산

extended p25 0.710 -> 0.680 (core는 변경 분자 0개로 불변). 정규화 없이 돌리면
커밋된 JSON 값을 정확히 복원하는 테스트로 재구현의 일치를 잠갔다.
band는 게이트가 아니라 보고 맥락이다 - 게이트는 Task 3의 Murcko 집합.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: Murcko 신규성 게이트 + `data/known_scaffolds.smi`

**Files:**
- Create: `tests/test_known_scaffolds.py`
- Create: `scripts/known_scaffolds.py`
- Create: `data/known_scaffolds.smi`

**Interfaces:**
- Consumes: `scripts.normalize.canonical_tautomer`, `scripts.novelty.murcko`, `scripts.novelty.load_smi`
- Produces:
  - `scripts/known_scaffolds.py`:
    - `REFERENCE_FILES: tuple[str, ...]` — 기준 집합을 이루는 다섯 파일
    - `known_scaffolds(paths: Sequence[str] = REFERENCE_FILES) -> set[str]`
    - `is_scaffold_novel(smiles: str, known: set[str]) -> bool | None` — 파싱 실패 시 `None`
    - `write_known_scaffolds(out_path: str = "data/known_scaffolds.smi") -> int` — 쓴 scaffold 개수 반환
- 이후 Task 4가 `known_scaffolds`를 import 한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_known_scaffolds.py`:

```python
import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from known_scaffolds import is_scaffold_novel, known_scaffolds
from normalize import canonical_tautomer
from novelty import load_smi, murcko

PLN_NONAROMATIC = "O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O"
PLN_AROMATIC = "CC1(C(=O)NC(CCCCCCCc2ccc3c(n2)NCCC3)C(=O)O)CCOCC1"


class TestKnownScaffolds(unittest.TestCase):
    def test_reference_files_yield_106_scaffolds(self):
        self.assertEqual(len(known_scaffolds()), 106)

    def test_published_reference_compounds_are_all_covered(self):
        """The gate must never call a benchmark compound's scaffold novel.

        actives_extended alone misses PLN-1474, bexotegrast and A1AFA: it is
        defined as ChEMBL alphaVbeta1 actives <= 1 uM, and PLN-1474's structure
        came from AdisInsight rather than a ChEMBL activity record while A1AFA
        sits at pIC50 5.30, below the potency cut. PLN-1474 is the only clinical
        alphaVbeta1-selective compound in this project and its structure has been
        public since August 2023, so a generated molecule rebuilding its scaffold
        is not novel.
        """
        known = known_scaffolds()
        for path in ("data/benchmark_panel.smi", "data/similarity_refs.smi",
                     "data/actives_core.smi", "data/actives_extended.smi"):
            for smiles, label in load_smi(path):
                with self.subTest(path=path, label=label):
                    self.assertFalse(is_scaffold_novel(smiles, known))

    def test_empty_scaffold_is_not_in_the_known_set(self):
        """Review Focus 4: an acyclic molecule's Murcko scaffold is "".

        Every known active has rings, so "" must be absent - otherwise every
        acyclic generated molecule would be judged non-novel by accident.
        """
        self.assertNotIn("", known_scaffolds())

    def test_acyclic_molecule_reports_novel_and_is_flagged(self):
        known = known_scaffolds()
        self.assertEqual(murcko(Chem.MolFromSmiles("CCCCC(=O)O")), "")
        self.assertTrue(is_scaffold_novel("CCCCC(=O)O", known))


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
        known = known_scaffolds()
        self.assertFalse(is_scaffold_novel(PLN_NONAROMATIC, known))
        self.assertFalse(is_scaffold_novel(PLN_AROMATIC, known))


class TestGateOnPriorSamples(unittest.TestCase):
    def test_861_percent_of_prior_samples_are_scaffold_novel(self):
        known = known_scaffolds()
        with open("logs/cmp.vigHoB/samples_prior.csv") as handle:
            smiles = [row["SMILES"] for row in csv.DictReader(handle)]
        verdicts = [is_scaffold_novel(s, known) for s in smiles]
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
        """
        import normalize

        before = len(normalize.TAUTOMER_FAILURES)
        known_scaffolds()
        self.assertEqual(len(normalize.TAUTOMER_FAILURES), before)


class TestInvalidInput(unittest.TestCase):
    def test_unparseable_smiles_returns_none(self):
        """Review Focus 1: never crash the triage funnel on generator output."""
        known = known_scaffolds()
        for bad in ["", "not_a_smiles", "C((("]:
            with self.subTest(bad=bad):
                self.assertIsNone(is_scaffold_novel(bad, known))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `python3 -m unittest tests.test_known_scaffolds -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.known_scaffolds'`

- [ ] **Step 3: 최소 구현을 쓴다**

`scripts/known_scaffolds.py`:

```python
"""The novelty gate: is this Murcko scaffold already in the published record?

Blueprint section 8.3 and section 9.3. Binary, fingerprint-free, threshold-free.
The distance band it replaced passed 39 of 63 prior samples (62%) whose Murcko
scaffold was IDENTICAL to a published active's.

Both sides go through scripts.normalize.canonical_tautomer first: without it the
same compound yields two different Murcko SMILES (verified on PLN-1474).
"""

from __future__ import annotations

from typing import Sequence

from rdkit import Chem

from normalize import canonical_tautomer
from novelty import load_smi, murcko

__all__ = [
    "REFERENCE_FILES",
    "is_scaffold_novel",
    "known_scaffolds",
    "write_known_scaffolds",
]

# Every curated reference set, not just actives_extended. That file is defined as
# ChEMBL alphaVbeta1 actives <= 1 uM, which omits PLN-1474 (structure from
# AdisInsight, not a ChEMBL activity record), bexotegrast, and A1AFA (pIC50 5.30,
# below the cut) - so on its own the gate would call a molecule rebuilding
# PLN-1474's scaffold novel. The union is 106 scaffolds.
REFERENCE_FILES = (
    "data/actives_core.smi",
    "data/actives_core_B.smi",
    "data/actives_extended.smi",
    "data/benchmark_panel.smi",
    "data/similarity_refs.smi",
)
DEFAULT_OUTPUT = "data/known_scaffolds.smi"


def known_scaffolds(paths: Sequence[str] = REFERENCE_FILES) -> set[str]:
    """Murcko scaffolds of every published reference compound, tautomer-canonicalized."""
    scaffolds = set()
    for path in paths:
        for smiles, _ in load_smi(path):
            mol = canonical_tautomer(Chem.MolFromSmiles(smiles))
            if mol is None:
                continue
            scaffold = murcko(mol)
            if scaffold:  # "" means acyclic; no reference compound is acyclic
                scaffolds.add(scaffold)
    return scaffolds


def is_scaffold_novel(smiles: str, known: set[str]) -> bool | None:
    """True if this molecule's Murcko scaffold is absent from `known`.

    None for unparseable input, so the caller skips instead of crashing.
    An acyclic molecule has scaffold "" and is reported novel - correct, but
    the diversity filter buckets every acyclic molecule together, so the
    triage report states how many leads are acyclic.
    """
    mol = canonical_tautomer(Chem.MolFromSmiles(smiles)) if smiles else None
    if mol is None:
        return None
    return murcko(mol) not in known


def write_known_scaffolds(out_path: str = DEFAULT_OUTPUT) -> int:
    """Freeze the scaffold set to disk so triage does not recompute it."""
    scaffolds = sorted(known_scaffolds())
    lines = [
        "# Murcko scaffolds of every curated reference set - actives_core,",
        "# actives_core_B, actives_extended, benchmark_panel, similarity_refs -",
        "# tautomer-canonicalized per blueprint section 8.0. 106 scaffolds.",
        "# A generated molecule whose Murcko scaffold is absent here is",
        "# scaffold-novel (sections 8.3, 9.3). Regenerate with:",
        "#   python3 -c \"import sys; sys.path.insert(0,'scripts'); \\",
        "#     from known_scaffolds import write_known_scaffolds as w; w()\"",
    ]
    lines += scaffolds
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(scaffolds)
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

Run: `python3 -m unittest tests.test_known_scaffolds -v`
Expected: PASS — 10 tests

- [ ] **Step 5: `curate_actives.py`가 `_note`를 잃지 않게 한다**

Task 2의 Step 5 스니펫은 `data/novelty_band.json`에 "REPORTING CONTEXT ONLY" 주석을 넣었지만, `curate_actives.py`가 만드는 `band` dict에는 그 키가 없다. 따라서 **큐레이션 스크립트를 실제로 다시 돌리면 그 주석이 조용히 사라지고** JSON이 다시 게이트처럼 읽힌다. `curate_actives.py`의 `(DATA / "novelty_band.json").write_text(...)` 바로 앞에 다음을 추가한다:

```python
    band["_note"] = (
        "Tautomer-canonicalized (blueprint 8.0). REPORTING CONTEXT ONLY - the "
        "novelty gate is Murcko scaffold membership, see data/known_scaffolds.smi. "
        "Pre-normalization values were core p25 0.552, extended p25 0.710."
    )
```

확인: `python3 -c "import sys; sys.path.insert(0,'scripts'); import curate_actives; print('import ok')"`

- [ ] **Step 6: `data/known_scaffolds.smi`를 생성한다**

```bash
python3 -c "import sys; sys.path.insert(0,'scripts'); from known_scaffolds import write_known_scaffolds as w; print(w(), 'scaffolds')"
```
Expected: `106 scaffolds`

Run: `grep -vc '^#' data/known_scaffolds.smi`
Expected: `106`

- [ ] **Step 7: 커밋**

```bash
git add tests/test_known_scaffolds.py scripts/known_scaffolds.py data/known_scaffolds.smi scripts/curate_actives.py
git commit -m "$(cat <<'MSG'
feat: Murcko scaffold 신규성 게이트와 known_scaffolds.smi

큐레이션된 기준 세트 5개의 합집합에서 106개 scaffold를 고정 산출. actives_extended
단독(103개)은 PLN-1474/bexotegrast/A1AFA의 골격을 빠뜨려 그 골격을 재현한 분자를
"신규"로 통과시켰다. 이진 판정이라 지문 의존과 임계값이 없고 diversity filter와 같은
단위를 쓴다. prior 샘플 488개 중 420개(86.1%) 통과를 테스트로 고정했다. 비고리 분자의 빈 scaffold와 무효 SMILES도 명시 처리한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: 목적함수 참조 구현 + spec 수치 회귀 테스트

**Files:**
- Create: `tests/test_objective.py`
- Create: `scripts/objective.py`

**Interfaces:**
- Consumes: `scripts.novelty.load_smi`
- Produces:
  - `scripts/objective.py`:
    - `sigmoid(values, low, high, k) -> np.ndarray`
    - `reverse_sigmoid(values, low, high, k) -> np.ndarray`
    - `double_sigmoid(values, low, high, coef_div, coef_si, coef_se) -> np.ndarray`
    - `right_step(values, high) -> np.ndarray`
    - `geometric_mean(pairs: list[tuple[np.ndarray, float]]) -> np.ndarray`
    - `GATE_FLOOR: float` — 카르복실산 부재 시 총점 상한
    - `score(mols: list[Chem.Mol]) -> dict[str, np.ndarray]` — 키: `total`, `cooh`, `tpsa`, `sascore`, `alerts`
    - `score_smiles(smiles_list: list[str]) -> dict[str, np.ndarray]`
    - `CARBOXYLATE_SMARTS: str`, `ALERT_SMARTS: list[str]`, `TPSA_WINDOW: tuple`, `SASCORE_WINDOW: tuple` — Task 5의 TOML 테스트가 config 값과 대조하는 단일 진실 원천

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_objective.py`:

```python
import csv
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from novelty import load_smi
from objective import (
    GATE_FLOOR,
    double_sigmoid,
    geometric_mean,
    reverse_sigmoid,
    right_step,
    score_smiles,
)


def prior_smiles():
    with open("logs/cmp.vigHoB/samples_prior.csv") as handle:
        return [row["SMILES"] for row in csv.DictReader(handle)]


class TestTransformsMatchReinvent(unittest.TestCase):
    """Validate against real REINVENT output in logs/cmp.vigHoB/new3.csv.

    That file was scored with SlogP reverse_sigmoid(2, 5, k=0.4) and TPSA
    double_sigmoid(40, 120, 120, 20, 20). 19 rows are alert-filtered, and REINVENT
    overwrites every component with 0 for those, so they are excluded.
    """

    def setUp(self):
        with open("logs/cmp.vigHoB/new3.csv") as handle:
            self.rows = list(csv.DictReader(handle))

    def _cols(self, raw_col, score_col):
        raw, got = [], []
        for row in self.rows:
            if float(row["SlogP (raw)"]) == 0.0:  # alert-filtered rows
                continue
            raw.append(float(row[raw_col]))
            got.append(float(row[score_col]))
        return np.array(raw), np.array(got)

    def test_reverse_sigmoid_matches(self):
        raw, got = self._cols("SlogP (raw)", "SlogP")
        self.assertLess(np.abs(reverse_sigmoid(raw, 2.0, 5.0, 0.4) - got).max(), 1e-6)

    def test_double_sigmoid_matches(self):
        raw, got = self._cols("TPSA (raw)", "TPSA")
        computed = double_sigmoid(raw, 40.0, 120.0, 120.0, 20.0, 20.0)
        self.assertLess(np.abs(computed - got).max(), 1e-6)

    def test_geometric_mean_and_penalty_reproduce_total(self):
        rows = [r for r in self.rows if float(r["SlogP (raw)"]) != 0.0]
        col = lambda name: np.array([float(r[name]) for r in rows])
        scored = [
            (col("sim A1AFA"), 1.0),
            (col("sim CHEMBL4649232"), 1.0),
            (col("sim CHEMBL5532604"), 1.0),
            (col("TPSA"), 1.0),
            (col("SlogP"), 1.0),
            (col("QED"), 1.0),
        ]
        total = geometric_mean(scored) * col("carboxylic acid") * col("alerts")
        self.assertLess(np.abs(total - col("Score")).max(), 1e-6)


class TestRightStepGate(unittest.TestCase):
    def test_right_step_is_binary_at_one(self):
        got = right_step(np.array([0.0, 1.0, 2.0]), 1.0)
        np.testing.assert_array_equal(got, np.array([0.0, 1.0, 1.0]))

    def test_gate_floor_value(self):
        """Absence of carboxylate caps the total at 1e-8 ** (1.0 / 2.5)."""
        self.assertAlmostEqual(GATE_FLOOR, 6.3096e-04, places=7)

    def test_no_carboxylate_prior_sample_reaches_the_floor_at_most(self):
        result = score_smiles(prior_smiles())
        without = result["total"][result["cooh"] == 0.0]
        self.assertEqual(len(without), 219)
        self.assertLessEqual(without.max(), GATE_FLOOR + 1e-9)


class TestCarboxylateSmartsSpecificity(unittest.TestCase):
    """Review Focus 3: the gate must not open for a look-alike group."""

    def test_matches_acid_and_anion_only(self):
        cases = {
            "CC(=O)O": 1.0,        # carboxylic acid
            "CC(=O)[O-]": 1.0,     # carboxylate anion
            "CC(=O)OC": 0.0,       # methyl ester
            "CC(=O)N": 0.0,        # amide
            "CC=O": 0.0,           # aldehyde
            "c1nnn[nH]1": 0.0,     # tetrazole bioisostere
        }
        result = score_smiles(list(cases))
        for smiles, expected in zip(cases, result["cooh"]):
            with self.subTest(smiles=smiles):
                self.assertEqual(expected, cases[smiles])


class TestSpecNumbers(unittest.TestCase):
    """Pin every distribution the spec claims, so a drift in RDKit is loud."""

    def test_acid_subset_distribution(self):
        result = score_smiles(prior_smiles())
        keep = (result["cooh"] > 0) & (result["alerts"] > 0)
        acid = np.sort(result["total"][keep])
        self.assertEqual(len(acid), 254)
        self.assertAlmostEqual(float(np.median(acid)), 0.178, delta=0.01)
        self.assertAlmostEqual(float((acid > 0.95).mean()), 0.33, delta=0.02)
        self.assertAlmostEqual(float((acid > 0.5).mean()), 0.43, delta=0.02)

    def test_benchmark_panel_scores(self):
        rows = load_smi("data/benchmark_panel.smi")
        result = score_smiles([s for s, _ in rows])
        got = dict(zip([label for _, label in rows], result["total"]))
        expected = {
            "A1AFA": 1.000,
            "PLN-1474": 0.998,
            "bexotegrast": 0.878,
            "CHEMBL4649232": 0.060,
            "GLPG0187": 0.001,
            "CWHM-12": 0.001,
        }
        for label, value in expected.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(float(got[label]), value, delta=0.005)

    def test_sascore_window_does_not_punish_any_reference_molecule(self):
        """The rejected (3, 6) window gave actives_extended's worst molecule 0.093."""
        for path in ("data/actives_core.smi", "data/actives_extended.smi",
                     "data/benchmark_panel.smi"):
            with self.subTest(path=path):
                result = score_smiles([s for s, _ in load_smi(path)])
                self.assertAlmostEqual(float(result["sascore"].min()), 1.0, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `python3 -m unittest tests.test_objective -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.objective'`

- [ ] **Step 3: 최소 구현을 쓴다**

`scripts/objective.py`:

```python
"""Python reference implementation of the Step 2 RL objective (blueprint 5.1).

Lets us score any SMILES set offline, without launching REINVENT. The transforms
and the aggregator are ported from REINVENT4 v4.5.11:
  reinvent/scoring/transforms/sigmoid_functions.py
  reinvent/scoring/transforms/sigmoids.py, steps.py
  reinvent/scoring/aggregators/means.py
  reinvent/scoring/scorer.py:159-178

    total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SAScore, 0.5)]) * alert_filter

COOH is a GATE, not a penalty: REINVENT's MatchingSubstructure hardcodes
0.5 * (1.0 + match), and with that a molecule WITHOUT the carboxylate scored a flat
0.500 while molecules WITH it had a median of 0.203 - the agent's best move was to
drop the MIDAS anchor. GroupCount + right_step removes that inversion.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

__all__ = [
    "ALERT_SMARTS",
    "CARBOXYLATE_SMARTS",
    "GATE_FLOOR",
    "double_sigmoid",
    "geometric_mean",
    "reverse_sigmoid",
    "right_step",
    "score",
    "score_smiles",
    "sigmoid",
]

CARBOXYLATE_SMARTS = "[CX3](=O)[OX2H1,OX1-]"
ALERT_SMARTS = [
    "[NX3;H2][c]",  # primary aniline only - anilides/sulfonanilides/THN excluded
    "[*;r8]",
    "[*;r9]",
    "[*;r10]",
    "N=[N+]=[N-]",
    "C(=O)Cl",
    "[SH]",
    "[Nr0][Nr0]",
]

W_COOH, W_TPSA, W_SASCORE = 1.0, 1.0, 0.5
TPSA_WINDOW = (40.0, 115.0, 120.0, 20.0, 20.0)
SASCORE_WINDOW = (6.0, 8.0, 0.5)

# geometric_mean clamps a 0 component to 1e-8, so a molecule missing the carboxylate
# can reach at most this. Under DAP with sigma=128 that is ~128 nat below a perfect
# score - a gate in practice, though not literally zero. Triage re-applies a hard cut.
GATE_FLOOR = float(1e-8 ** (W_COOH / (W_COOH + W_TPSA + W_SASCORE)))

_CARBOXYLATE = Chem.MolFromSmarts(CARBOXYLATE_SMARTS)
_ALERTS = [Chem.MolFromSmarts(s) for s in ALERT_SMARTS]

_params = FilterCatalog.FilterCatalogParams()
_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_params)


def _stable_sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    h = k * x * np.log(10)
    positive = h >= 0
    y = np.zeros_like(x)
    y[positive] = 1.0 / (1.0 + np.exp(-h[positive]))
    y[~positive] = np.exp(h[~positive]) / (1.0 + np.exp(h[~positive]))
    return y.astype(np.float32)


def sigmoid(values, low: float, high: float, k: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    x = values - (high + low) / 2
    if high - low == 0:
        return (10.0 * k * x > 0).astype(np.float32)
    return _stable_sigmoid(x, 10.0 * k / (high - low))


def reverse_sigmoid(values, low: float, high: float, k: float) -> np.ndarray:
    return 1.0 - sigmoid(values, low, high, k)


def double_sigmoid(values, low, high, coef_div=100.0, coef_si=150.0,
                   coef_se=150.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    center = (high - low) / 2 + low
    out = np.zeros_like(x)
    left, right = x < center, x >= center
    if coef_div == 0:
        out[left] = (coef_si * (x[left] - low) > 0).astype(np.float32)
        out[right] = 1 - (coef_se * (x[right] - high) > 0).astype(np.float32)
    else:
        out[left] = _stable_sigmoid(x[left] - low, coef_si / coef_div)
        out[right] = 1 - _stable_sigmoid(x[right] - high, coef_se / coef_div)
    return out


def right_step(values, high: float) -> np.ndarray:
    return np.array([1.0 if v >= high else 0.0 for v in values], dtype=float)


def geometric_mean(pairs: list[tuple[np.ndarray, float]]) -> np.ndarray:
    scores = np.array([s for s, _ in pairs], dtype=float)
    weights = np.array([w for _, w in pairs], dtype=float)
    weights = np.broadcast_to(weights.reshape(-1, 1), scores.shape).astype(float)
    scores = np.maximum(scores, 1e-8)
    return np.prod(scores ** (weights / np.maximum(weights.sum(axis=0), 1e-8)), axis=0)


def score(mols: list[Chem.Mol]) -> dict[str, np.ndarray]:
    counts = np.array([len(m.GetSubstructMatches(_CARBOXYLATE)) for m in mols])
    cooh = right_step(counts, 1.0)
    tpsa = double_sigmoid(np.array([Descriptors.TPSA(m) for m in mols]), *TPSA_WINDOW)
    sa = reverse_sigmoid(
        np.array([sascorer.calculateScore(m) for m in mols]), *SASCORE_WINDOW
    )
    alerts = np.array([
        0.0 if (any(m.HasSubstructMatch(p) for p in _ALERTS) or _PAINS.HasMatch(m))
        else 1.0
        for m in mols
    ])
    total = geometric_mean(
        [(cooh, W_COOH), (tpsa, W_TPSA), (sa, W_SASCORE)]
    ) * alerts
    return {"total": total, "cooh": cooh, "tpsa": tpsa, "sascore": sa,
            "alerts": alerts}


def score_smiles(smiles_list: list[str]) -> dict[str, np.ndarray]:
    """Score valid molecules. Unparseable SMILES are dropped, as REINVENT does."""
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles_list) if m is not None]
    return score(mols)
```

- [ ] **Step 4: 테스트가 통과하는 것을 확인한다**

Run: `python3 -m unittest tests.test_objective -v`
Expected: PASS — 10 tests. `test_geometric_mean_and_penalty_reproduce_total`이 통과하면 집계 재구현이 실제 REINVENT 출력과 1e-6 이내로 일치한다.

- [ ] **Step 5: 커밋**

```bash
git add tests/test_objective.py scripts/objective.py
git commit -m "$(cat <<'MSG'
feat: 목적함수 파이썬 참조 구현과 spec 수치 회귀 테스트

REINVENT4 v4.5.11의 transform/aggregator를 이식해 REINVENT 없이 임의 SMILES를
채점한다. 실제 REINVENT 출력(logs/cmp.vigHoB/new3.csv)과 1e-6 이내 일치를 테스트로
확인한다. 게이트 바닥 6.31e-04, 산 부분집합 중앙값 0.178, 벤치마크 패널 6개 값,
SAScore 창이 참조 분자를 벌주지 않음, 카르복실레이트 SMARTS가 에스터/아마이드/
테트라졸을 잡지 않음을 모두 고정했다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: 단일 stage TOML config

**Files:**
- Create: `tests/test_config.py`
- Create: `configs/_rl_scoring.frag`
- Delete: `configs/_stage1_scoring.frag`, `configs/_stage2_scoring.frag`
- Modify: `reinvent4_avb1_scoring_config_sketch.toml` (전면 재작성)

**Interfaces:**
- Consumes: `objective.ALERT_SMARTS`, `objective.CARBOXYLATE_SMARTS` (테스트가 config 값과 대조), `REINVENT4/reinvent.scoring.transforms`
- Produces: 없음 (최종 산출물)

- [ ] **Step 1: 실패하는 테스트를 쓴다**

`tests/test_config.py`:

```python
import sys
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from objective import ALERT_SMARTS, CARBOXYLATE_SMARTS

SKETCH = "reinvent4_avb1_scoring_config_sketch.toml"


def load():
    with open(SKETCH, "rb") as handle:
        return tomllib.load(handle)


def endpoints(config):
    """Flatten every component endpoint into {name: (component_type, endpoint)}."""
    out = {}
    for block in config["stage"][0]["scoring"]["component"]:
        for component_type, body in block.items():
            for endpoint in body["endpoint"]:
                out[endpoint["name"]] = (component_type, endpoint)
    return out


class TestSingleStage(unittest.TestCase):
    def test_exactly_one_stage_of_600_steps(self):
        config = load()
        self.assertEqual(len(config["stage"]), 1)
        self.assertEqual(config["stage"][0]["max_steps"], 600)

    def test_common_parameters(self):
        config = load()
        self.assertEqual(config["learning_strategy"]["type"], "dap")
        self.assertEqual(config["learning_strategy"]["sigma"], 128)
        self.assertEqual(config["parameters"]["batch_size"], 128)
        self.assertEqual(config["diversity_filter"]["type"],
                         "IdenticalMurckoScaffold")
        self.assertEqual(config["diversity_filter"]["bucket_size"], 25)
        self.assertEqual(config["stage"][0]["scoring"]["type"], "geometric_mean")


class TestComponents(unittest.TestCase):
    def test_component_set_is_exactly_four(self):
        names = set(endpoints(load()))
        self.assertEqual(
            names,
            {"carboxylate MIDAS anchor", "TPSA", "SA score", "unwanted groups"},
        )

    def test_removed_components_are_absent(self):
        types = {t for t, _ in endpoints(load()).values()}
        for removed in ("TanimotoSimilarity", "MatchingSubstructure", "SlogP",
                        "Qed", "MolecularWeight", "NumRotBond", "HBondDonors"):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, types)

    def test_carboxylate_is_a_groupcount_gate(self):
        component_type, endpoint = endpoints(load())["carboxylate MIDAS anchor"]
        self.assertEqual(component_type, "GroupCount")
        self.assertEqual(endpoint["params"]["smarts"], [CARBOXYLATE_SMARTS])
        self.assertEqual(endpoint["transform"]["type"], "right_step")
        self.assertEqual(endpoint["transform"]["high"], 1)
        self.assertEqual(endpoint["weight"], 1.0)

    def test_tpsa_window(self):
        _, endpoint = endpoints(load())["TPSA"]
        transform = endpoint["transform"]
        self.assertEqual(transform["type"], "double_sigmoid")
        self.assertEqual(transform["low"], 40.0)
        self.assertEqual(transform["high"], 115.0)
        self.assertEqual(transform["coef_div"], 120.0)
        self.assertEqual(endpoint["weight"], 1.0)

    def test_sascore_guard_rail_window(self):
        _, endpoint = endpoints(load())["SA score"]
        transform = endpoint["transform"]
        self.assertEqual(transform["type"], "reverse_sigmoid")
        self.assertEqual(transform["low"], 6.0)
        self.assertEqual(transform["high"], 8.0)
        self.assertEqual(transform["k"], 0.5)
        self.assertEqual(endpoint["weight"], 0.5)

    def test_alerts_include_narrow_aniline_and_exclude_the_broad_one(self):
        _, endpoint = endpoints(load())["unwanted groups"]
        smarts = endpoint["params"]["smarts"]
        self.assertEqual(smarts, ALERT_SMARTS)
        self.assertNotIn("[NH2,NH][c]", smarts)


def load_reinvent_transforms():
    """Load REINVENT's transform registry WITHOUT importing the reinvent package.

    `import reinvent.scoring.transforms` pulls in torch, which is not installed in
    the base conda environment. The transform modules themselves need only numpy, so
    we load them under a synthetic package name; their relative `from .transform
    import Transform` then resolves inside it.
    """
    import importlib.util
    import types

    directory = (Path(__file__).resolve().parent.parent
                 / "REINVENT4" / "reinvent" / "scoring" / "transforms")
    package = types.ModuleType("rvt")
    package.__path__ = [str(directory)]
    sys.modules["rvt"] = package

    def load_module(name):
        spec = importlib.util.spec_from_file_location(
            f"rvt.{name}", directory / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"rvt.{name}"] = module
        spec.loader.exec_module(module)
        return module

    transform = load_module("transform")
    for name in ("sigmoid_functions", "sigmoids", "steps", "double_sigmoid"):
        load_module(name)
    return transform.get_transform


class TestReinventAcceptsTheNames(unittest.TestCase):
    """Review Focus 5: REINVENT resolves transform names and forbids extra keys.

    get_transform lowercases and strips underscores, so "right_step" -> "rightstep"
    -> RightStep. The parameter dataclasses reject unknown keys, so a typo in the
    TOML is a hard startup error rather than a silently ignored setting.
    """

    def test_transform_names_resolve(self):
        get_transform = load_reinvent_transforms()
        expected = {
            "right_step": "RightStep",
            "double_sigmoid": "DoubleSigmoid",
            "reverse_sigmoid": "ReverseSigmoid",
        }
        for name, class_name in expected.items():
            with self.subTest(name=name):
                cls, _ = get_transform(name)
                self.assertEqual(cls.__name__, class_name)

    def test_config_transform_blocks_instantiate(self):
        """Every transform table in the sketch must construct its REINVENT class."""
        get_transform = load_reinvent_transforms()
        for endpoint_name in ("carboxylate MIDAS anchor", "TPSA", "SA score"):
            with self.subTest(endpoint=endpoint_name):
                _, endpoint = endpoints(load())[endpoint_name]
                table = dict(endpoint["transform"])
                cls, param_cls = get_transform(table["type"])
                cls(param_cls(**table))

    def test_right_step_gate_is_binary(self):
        get_transform = load_reinvent_transforms()
        cls, param_cls = get_transform("right_step")
        transform = cls(param_cls(type="right_step", high=1))
        self.assertEqual(list(transform([0, 1, 2])), [0.0, 1.0, 1.0])


class TestFragments(unittest.TestCase):
    def test_stage_fragments_are_gone(self):
        self.assertFalse(Path("configs/_stage1_scoring.frag").exists())
        self.assertFalse(Path("configs/_stage2_scoring.frag").exists())

    def test_sketch_ends_with_the_fragment_verbatim(self):
        """The sketch is header + fragment, so the two cannot drift apart."""
        fragment = Path("configs/_rl_scoring.frag").read_text()
        sketch = Path(SKETCH).read_text()
        self.assertIn(fragment.strip(), sketch)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 테스트가 실패하는 것을 확인한다**

Run: `python3 -m unittest tests.test_config -v`
Expected: FAIL — 현재 sketch에 stage가 2개이고 `TanimotoSimilarity`가 남아 있으므로 여러 테스트가 실패한다.

- [ ] **Step 3: `configs/_rl_scoring.frag`를 쓴다**

```toml
# Step 2 single-stage RL scoring block (blueprint section 5.1).
#
#   total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SA, 0.5)]) * alert_filter
#
# The carboxylate is a GATE, not a MatchingSubstructure penalty: that component
# hardcodes 0.5 * (1.0 + match), and under it a molecule WITHOUT the carboxylate
# scored a flat 0.500 while molecules WITH it had a median of 0.203, because the
# acid adds ~37 TPSA and falls out of the window. The agent's best move was to drop
# the MIDAS anchor. GroupCount + right_step makes absence cost 6.31e-04 instead.
#
# TanimotoSimilarity is deliberately absent. The six references were chosen to be
# mutually DISSIMILAR (max pairwise Tanimoto 0.45), one endpoint each, under a
# geometric mean - which demands a molecule resemble six mutually unlike molecules
# at once. Prior-sample medians were 0.002-0.004 with 70% under 0.05. Potency comes
# from TL-A and inception; potency judgement comes from the section 8.5 docking
# geometry filter.

[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name = "carboxylate MIDAS anchor"
weight = 1.0
params.smarts = ["[CX3](=O)[OX2H1,OX1-]"]
transform.type = "right_step"
transform.high = 1

# Window anchored on the two orally advanced alphaV carboxylates: PLN-1474
# (TPSA 100.6, Phase 1 completed) and bexotegrast (112.5, Phase 2a). The cpd 25
# series cannot justify the window - its high TPSA (median 133.3) just restates
# that it was never proposed as an oral structure.
[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name = "TPSA"
weight = 1.0
transform.type = "double_sigmoid"
transform.low = 40.0
transform.high = 115.0
transform.coef_div = 120.0
transform.coef_si = 20.0
transform.coef_se = 20.0

# GUARD RAIL, not an optimization axis. Every reference molecule and every prior
# sample scores exactly 1.000 here, so this term contributes nothing at step 0; it
# only bites if 600 steps of RL drift past SA 6, the conventional hard-to-make line.
# The earlier (3, 6) window gave actives_extended's worst molecule (SA 5.10) a 0.093
# - punishing a published active.
[[stage.scoring.component]]
[stage.scoring.component.SAScore]
[[stage.scoring.component.SAScore.endpoint]]
name = "SA score"
weight = 0.5
transform.type = "reverse_sigmoid"
transform.low = 6.0
transform.high = 8.0
transform.k = 0.5

# The aniline pattern is [NX3;H2][c], primary aromatic amines only. The earlier
# [NH2,NH][c] zeroed 164 of 190 actives_extended (86%) and 5 of 6 benchmark
# positives, because an aryl-NH with one hydrogen matches - which catches
# tetrahydro-1,8-naphthyridine, the standard integrin Arg-mimic head.
[[stage.scoring.component]]
[stage.scoring.component.CustomAlerts]
[[stage.scoring.component.CustomAlerts.endpoint]]
name = "unwanted groups"
params.smarts = ["[NX3;H2][c]", "[*;r8]", "[*;r9]", "[*;r10]", "N=[N+]=[N-]", "C(=O)Cl", "[SH]", "[Nr0][Nr0]"]
```

- [ ] **Step 4: `reinvent4_avb1_scoring_config_sketch.toml`을 재작성한다**

기존 파일을 지우고, 헤더 + 공통 파라미터 + 위 fragment의 component 블록을 이어 붙여 하나의 완전한 config로 만든다. 헤더 주석:

```toml
# Step 2 single-stage RL, annotated. Blueprint section 5.
#
# Replaces the two-stage curriculum (Stage 1 focus 200 steps -> Stage 2 optimize
# 400 steps). Reasons, with measurements, in
# docs/superpowers/specs/2026-09-26-step2-single-stage-rl-design.md:
#   - TL-A, inception and Stage-1 similarity were three copies of "stay near the
#     actives"; only Stage 1 cost RL steps.
#   - Stage 1 raised similarity that Stage 2 then paid 400 steps to lower, which
#     contradicts the novelty hypothesis.
#   - The curriculum's premise (ill-conditioned objective at step 0) held only for
#     the similarity axis, and there it was logically unsatisfiable, not
#     mis-calibrated.
#
# PAINS is NOT expressible in this file: CustomAlerts takes SMARTS, and the RDKit
# PAINS catalog is a Python object. Apply it in the section 8.1 triage step, and in
# scripts/objective.py, which is the reference scorer these values are tested against
# (tests/test_config.py, tests/test_objective.py).
run_type = "staged_learning"
device = "cuda:0"

[parameters]
summary_csv_prefix = "results/rl"
use_checkpoint = false
purge_memories = false
prior_file = "priors/avb1_tl_a.prior"
agent_file = "priors/avb1_tl_a.prior"
batch_size = 128
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

# Applied for the whole run, unlike the prior study [113] which filtered only in its
# last stage to CONVERGE on enriched chemotypes. Our goal is the opposite direction.
[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4

[inception]
smiles_file = "data/actives_core.smi"
memory_size = 100
sample_size = 10

[[stage]]
chkpt_file = "results/rl_final.chkpt"
termination = "simple"
max_score = 1.0
min_steps = 600
max_steps = 600

[stage.scoring]
type = "geometric_mean"
```

그 뒤에 `configs/_rl_scoring.frag`의 내용을 그대로 이어 붙인다:

```bash
cat configs/_rl_scoring.frag >> reinvent4_avb1_scoring_config_sketch.toml
```

- [ ] **Step 5: 옛 fragment를 삭제한다**

이 두 파일은 **git에 추적되지 않은 상태**이므로 `git rm`은 `did not match any files`로 실패한다. 평문 `rm`을 쓴다:

```bash
rm configs/_stage1_scoring.frag configs/_stage2_scoring.frag
```

Step 6의 `test_stage_fragments_are_gone`은 `Path(...).exists()`를 보므로 이것으로 충족된다.

- [ ] **Step 6: 테스트가 통과하는 것을 확인한다**

Run: `python3 -m unittest tests.test_config -v`
Expected: PASS — 13 tests

전체 스위트도 확인한다:
Run: `python3 -m unittest discover -s tests -t . -v`
Expected: PASS — 47 tests

- [ ] **Step 7: 커밋**

`reinvent4_avb1_scoring_config_sketch.toml`도 추적되지 않은 파일이므로 diff에 새 파일로 나타난다 — 정상이다.

```bash
git add reinvent4_avb1_scoring_config_sketch.toml configs/_rl_scoring.frag tests/test_config.py
git commit -m "$(cat <<'MSG'
feat: sketch config를 단일 stage 600 steps로 재작성

GroupCount 카르복실레이트 게이트 + TPSA(40,115) + SAScore(6,8,k=0.5) +
CustomAlerts([NX3;H2][c] 포함). TanimotoSimilarity/MatchingSubstructure/SlogP/
QED/MW/RotBond/HBD 제거. _stage1_scoring.frag와 _stage2_scoring.frag를
_rl_scoring.frag 하나로 교체.

REINVENT의 get_transform 이름 해석(right_step -> RightStep)과 transform
파라미터 클래스가 실제로 이 값을 받는지 테스트로 확인한다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: 영문 블루프린트 동기화

**Files:**
- Modify: `avb1_10day_blueprint.md` (§1 표 3·4행, §3.4, §4 끝, §5 전면, §6, §7, §8, §9, §10, References)

**Interfaces:**
- Consumes: `korean_avb1_10day_blueprint.md`의 최종 내용 (커밋 `aabf0bd`)
- Produces: 없음 (최종 산출물)

- [ ] **Step 1: 두 파일의 구조 차이를 확인한다**

```bash
diff <(grep -c '' korean_avb1_10day_blueprint.md) <(grep -c '' avb1_10day_blueprint.md)
grep -n '^#\{2,3\} ' avb1_10day_blueprint.md
```
현재 영문판은 §5가 `### Stage 1 — "focus"` / `### Stage 2 — "optimize"` 구조이고 국문판은 §5.1–5.5 구조다.

- [ ] **Step 2: §5를 국문판 §5.1–5.5에 대응하는 영문 구조로 교체한다**

`### Stage 1 — "focus" (max_steps = 200)`부터 `### Stage 2` 절 끝까지(현재 233–256행)를 다음 소절로 교체한다. 국문판 해당 절의 **모든 표와 수치를 그대로** 옮기고, 산문만 영어로 쓴다:

- `### 5.1 The objective — three scored endpoints and one filter` — 국문 §5.1의 component 표, `GroupCount` TOML 블록, 집계식, MatchingSubstructure 역전 표(카르복실산 있음 0.203 / 없음 0.500)
- `### 5.2 What was removed from the objective` — 국문 §5.2의 제거 표 + `TanimotoSimilarity` zip 결함 인용 블록
- `### 5.3 Why a single stage and not a curriculum` — 3개 논거 + 보정 측정 표(총점 중앙값 0.011)
- `### 5.4 Component-by-component rationale` — TPSA 창 앵커 표, 창 후보 비교 표 + 계산 정정 주석, SlogP 제거, QED 제거, SAScore 창 표, aniline 패턴 표
- `### 5.5 Monitoring — diagnostics, not reward` — 진단 5개 + 개입 기준

- [ ] **Step 3: 나머지 절을 동기화한다**

| 위치 | 변경 |
|---|---|
| §1 표 3행 | `similarity + QED component` → docking 기하 필터/정성 분석으로 확인 |
| §1 표 4행 | "SlogP up to about 3"이 §5의 페널티 기준값 출처였고 **허용 하한을 페널티 상한으로 오독**한 것임을 명기 |
| §1 permeability 문단 | `Stage-2 RL` → `Step 2 RL (single stage)` |
| §3.4 | 음성 패널 폐기 + 비순환 비교 원칙 |
| §4 끝 | `Stage 2` → `Step 2 RL` |
| §6 | `Stage-2 final agent` → `the final RL agent`; 출력에 tautomer normalization 추가 |
| §7 | 음성 100개 삭제, component별 보고, aniline 패턴 선행 조건 |
| §8 | 항목 0(tautomer normalization) 신설, 8.1 카르복실레이트 하드 컷, 8.2 넓은 창, 8.3 Murcko 게이트, 8.7 |
| §9 | 9.3 Murcko, 9.4 패턴 기준, 9.5 비순환 축 |
| §10 | D2 음성 패널 삭제, D4/D5 단일 stage + 선행 작업 |
| References | `[113]` 추가, `[40]` 오인용 4곳 → `[113]` |

- [ ] **Step 4: 두 파일이 같은 사실을 말하는지 기계적으로 확인한다**

```bash
for token in '40, 115' '(6, 8' '6.31e-04' '0.178' '106' 'right_step' 'GroupCount' '\[NX3;H2\]\[c\]' '0.680' '\[113\]'; do
  k=$(grep -c "$token" korean_avb1_10day_blueprint.md)
  e=$(grep -c "$token" avb1_10day_blueprint.md)
  printf '%-22s ko=%-3s en=%-3s %s\n' "$token" "$k" "$e" "$([ "$e" -gt 0 ] && echo OK || echo MISSING)"
done
grep -n '\[40\]' avb1_10day_blueprint.md
```
Expected: 모든 토큰이 `en > 0`이고, `[40]`은 References의 REINVENT4 항목 한 줄에만 남는다.

- [ ] **Step 5: 커밋**

```bash
git add avb1_10day_blueprint.md
git commit -m "$(cat <<'MSG'
docs: 영문 블루프린트를 단일 stage 재설계에 동기화

korean_avb1_10day_blueprint.md(aabf0bd)와 같은 사실을 말하도록 5장을 5.1-5.5로
재작성하고 1, 3.4, 4, 6, 7, 8, 9, 10장과 References를 갱신했다.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
MSG
)"
```

---

## 실행 후 확인

```bash
python3 -m unittest discover -s tests -t . -v   # 47 tests, all pass
git log --oneline afa1099..HEAD                 # 6 commits
```
