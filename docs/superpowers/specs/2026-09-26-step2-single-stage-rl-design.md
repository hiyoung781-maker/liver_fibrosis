# Step 2 재설계: 2단계 커리큘럼 → 단일 stage RL

- 날짜: 2026-09-26
- 대상: `korean_avb1_10day_blueprint.md` §5 (Step 2), §7, §8, §9
- 상태: 승인됨. 구현 계획은 별도 문서.

## 1. 목표와 성공 기준

Step 2의 목표는 **TL-A focused prior가 이미 확보한 영역 안에서 물성을 다듬는 것**이며,
그 영역을 *찾는* 일은 Step 2의 일이 아니다. 후자는 Step 1(TL)과 inception이 이미 수행했다.

이 재설계가 성공했다고 말할 수 있는 조건:

1. RL 0 step에서 목적함수가 well-conditioned하다 — prior 샘플의 총점 중앙값이
   포화(≈1.0)도 아니고 소멸(≈0)도 아닌 중간 대역에 있다.
2. 어떤 scoring component도 참조 세트(`actives_core`, `actives_extended`,
   `similarity_refs`, `benchmark_panel`)를 구조적으로 0점 처리하지 않는다.
3. §9의 사전 등록 기준이 **false pass를 허용하지 않는다** — 특히 신규성 판정이
   표현(representation) 차이로 통과되지 않는다.
4. scored endpoint 수가 선행연구 수준(3개 내외)을 넘지 않는다.

## 2. 왜 바꾸는가 — 측정된 근거

### 2.1 "활성 물질 근처에 머물라"는 신호가 3중으로 중복된다

| 기전 | 비용 | 성격 |
|---|---|---|
| TL-A (prior를 활성 영역에 집중) | 이미 지불됨 (D3) | 수동적 |
| inception (활성 물질을 RL 메모리에 replay) | 무시 가능 | 수동적 |
| **Stage 1 유사도 (200 RL steps)** | **200 steps** | **능동적, 중복** |

Stage 1은 세 번째 사본이면서 유일하게 비싼 사본이다.

### 2.2 '당겼다 미는' 시퀀스는 신규성 가설과 상충한다

기존 §5는 스스로 이 모순을 기술하고 있었다 — "Stage 2에서 similarity가 **하락**한다,
그 하락이 바로 scaffold novelty가 나타나는 신호다." 즉 Stage 1이 200 steps를 들여
올린 값을 Stage 2가 400 steps를 들여 되돌린다. 신규성이 목표라면 애초에 올리지 않는 것이 맞다.

### 2.3 커리큘럼의 전제(초기 ill-conditioning)는 절반만 성립한다

커리큘럼을 정당화하는 일반 논거는 "기본 변환에서 prior 샘플 점수가 0에 가까워
gradient가 없다"는 것이다. 선행연구 [113]이 정확히 그 조건이다 —
가중치 균등(docking·QED·SAscore 각각 0.3) + REINVENT4 기본 변환 + 3 stages × 30 epochs.

> **인용 주의.** 블루프린트의 `[40]`은 REINVENT4 논문(J. Cheminformatics 2024)이며
> 이 선행연구가 아니다. 선행연구는 `[113]`으로 신규 등록했다 —
> Qie, Wang, Li et al., *Molecular Diversity* (2026), doi 10.1007/s11030-026-11625-z,
> "Integrating machine learning-based molecular design with experimental validation
> for the discovery of EGFR inhibitors in lung cancer".

우리는 지난 세 번의 반복에서 창(window)을 측정된 prior 분포에 맞게 보정했다.
그 보정 결과를 `logs/cmp.vigHoB/new3.csv`에서 재확인했다
(`molecules.smi` 493개 중 488개가 `samples_prior.csv`와 일치 — 사실상 TL-A prior 샘플 분포):

| component | p10 | 중앙값 | p90 | score < 0.05 |
|---|---|---|---|---|
| TPSA | 0.000 | **0.894** | 1.000 | 38% |
| SlogP | 0.008 | **0.943** | 1.000 | 16% |
| QED | 0.170 | **0.468** | 0.788 | 4% |
| carboxylic acid | 0.500 | 1.000 | 1.000 | 4% |
| alerts | 1.000 | 1.000 | 1.000 | 4% |
| sim A1AFA | 0.000 | **0.004** | 0.982 | 65% |
| sim CHEMBL4649232 | 0.000 | **0.002** | 0.885 | 70% |
| sim CHEMBL5532604 | 0.000 | **0.001** | 0.984 | 71% |
| **총점** (geometric_mean) | 0.000 | **0.011** | 0.332 | **70%** |

**물성축은 well-conditioned하다. similarity축은 전혀 아니다.**

그리고 similarity축의 ill-conditioning은 보정 실패가 아니라 **논리적 불가능**이다:
`similarity_refs.smi`의 6개는 "서로 유사하지 않도록"(최대 쌍별 Tanimoto 0.45)
*일부러* 고른 것인데, 이를 각각 독립 endpoint로 걸고 `geometric_mean`으로 묶으면
"서로 닮지 않은 6개와 동시에 닮아라"를 요구하게 된다. 가중치로 해결되지 않는다.

→ 결론: 처방은 커리큘럼이 아니라 **similarity를 목적함수에서 제거**하는 것이다.
그렇게 하면 `geometric_mean`을 유지해도 된다. AND가 문제였던 것은 달성 불가능한
endpoint가 있었기 때문이고, 그것을 빼면 곱셈 집계는 원래 의도(either/or 회피)대로 작동한다.

### 2.4 물성 endpoint가 5중으로 중복된다

QED의 8개 descriptor는 MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS다.
기존 Stage 2는 TPSA + SlogP + MW + RotBond + QED를 각각 독립 endpoint로 걸어
같은 물성을 5중 계상하고 있었다. `geometric_mean` 아래에서 각 endpoint는 곱셈 veto이므로,
달성 가능한 천장이 무너지고 reward hacking 압력만 커진다.

선행연구 [113]의 3×0.3(REINVENT4가 내부 정규화하므로 실효 1/3씩)은 취향이 아니라
조종 가능성(steerability)의 조건이다.

### 2.5 `SlogP` 기준값 "3"은 출처를 오독한 것이고, 방향이 반대다

세 파일이 서로 다르다:

| 출처 | 설정 |
|---|---|
| `korean_avb1_10day_blueprint.md` §5 | reverse_sigmoid, "약 3 초과 시 페널티" |
| `reinvent4_avb1_scoring_config_sketch.toml:139` | double_sigmoid, low=1.0 high=4.0 |
| 실제 실행된 `logs/cmp.vigHoB/new3.toml` | reverse_sigmoid, low=2.0 high=5.0, k=0.4 |

"3"은 마지막 것의 변곡점(low=2, high=5의 중간, score 0.5 지점)과 일치한다.

**출처는 찾았다 — 그리고 오독이다.** `korean_avb1_10day_blueprint.md` §1의 pocket 요구사항
표 4행("β1 hydrophobic subpocket")이 "소수성 부피 수용 가능 → SlogP 허용 범위 약 3까지"라고
기술한다. 그러나 이것은 접촉 거리(Pro186 3.16 Å, Tyr133 3.17 Å 등)에서 얻은 정성적
**허용 하한** — "적어도 3까지는 괜찮다" — 이며 "3을 넘으면 벌점"이라는 뜻이 아니다.
허용 범위의 하한을 페널티의 상한으로 바꿔 쓴 것이 오독의 내용이다. low/high 값 자체는
어느 문서에서도 유도되지 않는다.

더 중요한 것은 방향이다. 이 프로젝트는 **필수 카르복실레이트를 가진 음이온**을 만들고,
§9.2는 "예측 permeability에서 cpd 25를 능가"를 요구한다. 산성 화합물의 투과성 병목은
지질친화성 과다가 아니라 이온화된 카르복실레이트다. logP를 3 위에서 깎으면
개선하겠다고 선언한 축을 악화시킨다. 선행연구 [113]의 준거도 soft penalty가 아니라
hard filter (MW > 500 또는 logP > 5 제외)다.

### 2.6 `CustomAlerts`의 aniline 패턴이 참조 세트를 삭제한다

`[NH2,NH][c]`를 RDKit으로 전 데이터셋에 대해 측정 (매치 = 총점 0):

| 세트 | `[NH2,NH][c]` (기존) | `[NX3;H2][c]` (신규) |
|---|---|---|
| `actives_core` (29) | **9/29 (31%)** | 0/29 |
| `actives_extended` (190) | **164/190 (86%)** | 0/190 |
| `similarity_refs` (6) | **3/6** | 0/6 |
| `benchmark_panel` (6) | **5/6** | 0/6 |
| TL-A prior 샘플 (488) | **118/488 (24%)** | 12/488 (2.5%) |

benchmark_panel에서 0점이 되는 5개: **PLN-1474, bexotegrast, CWHM-12, GLPG0187,
CHEMBL4649232**. A1AFA만 살아남는다.

두 가지 귀결:

1. **테트라히드로-1,8-나프티리딘(THN)이 걸린다.** 아릴에 붙은 NH가 H 1개라서
   `[NH2,NH]`에 매치된다. THN은 인테그린 RGD 모방체의 표준 Arg-mimic head이므로,
   이 필터는 반응성 대사체를 막는 대신 **이 타깃 클래스의 핵심 파마코포어를 금지**하고 있었다.
2. **§7과 §9.5가 무효화될 상태였다.** 양성 패널 6개 중 5개가 0점이면
   "생성 집합이 음성 패널을 능가하고 양성 패널에 준할 것"은 생성 분자가 아무리 나빠도
   통과한다 — false pass다. `results/`가 비어 있어 발표될 수치가 오염된 것은 없다.

패턴은 단순화할 수 있다. 제안된
`[NX3;H2;!$(N[!#6]);!$(NC=O);!$(NS(=O)=O)][c]`와 `[NX3;H2][c]`는 모든 테스트에서
결과가 **완전히 동일**하다 — `H2`가 이미 N-acyl·N-sulfonyl·N-heteroatom을 배제하기
때문이다(그 경우 H가 1개 이하). 세 exclusion은 중복이므로 `[NX3;H2][c]`를 채택한다.

나머지 alert는 정상이다:

| SMARTS | core | extended | refs | bench | prior |
|---|---|---|---|---|---|
| `[*;r8]` / `[*;r9]` / `[*;r10]` | 0 | 0 | 0 | 0 | 0 |
| `N=[N+]=[N-]` | 0 | 0 | 0 | 0 | 1 |
| `C(=O)Cl` | 0 | 0 | 0 | 0 | 0 |
| `[SH]` | 0 | 0 | 0 | 0 | 2 |
| `[Nr0][Nr0]` | 0 | 1/190 | 0 | 0 | 16 |
| **합집합** | **0/29** | **1/190** | **0/6** | **0/6** | 19/488 (3.9%) |

단, 블루프린트는 "PAINS/reactive/aniline"이라고 기술하는데 실제 config에
**PAINS 패턴이 하나도 없다**. RDKit PAINS 카탈로그는 core 0/29, refs 0/6, bench 0/6,
prior 15/488 (3.1%)를 잡으므로 참조 세트에 비용이 없다. 추가하여 문서–config 간극을 닫는다.

### 2.7 타우토머 정규화 누락이 §9.3에 false-positive 경로를 만든다

PLN-1474를 두 가지로 쓸 수 있다 — 비방향족 아미딘 형태
(`O=C(O)[C@H](CCCCCCCC1=CC=C2C(N1)=NCCC2)NC(C3(C)CCOCC3)=O`)와
방향족 피리딘 형태(`data/benchmark_panel.smi`에 저장된 형태). 분자식은 둘 다 `C24H37N3O4`로 동일하다.

| | 비방향족 형태 | 방향족 형태 (패널) |
|---|---|---|
| 방향족 고리 수 | 0 | 1 |
| QED | 0.4332 | **0.4619** |
| TPSA | 100.0 | 100.6 |
| MolLogP | 3.71 | 3.71 |
| SAScore | 3.747 | 3.75 |
| `[NH2,NH][c]` alert | 미매치 | 매치 |
| **서로에 대한 NN-Tanimoto** | **0.458** | |

`rdMolStandardize.TautomerEnumerator().Canonicalize()` 적용 후 두 형태는 동일한
canonical SMILES로 수렴하고 Tanimoto 1.0, QED 0.4619가 된다.

`scripts/curate_actives.py:222-239`는 `Cleanup` + `LargestFragmentChooser` +
`Uncharger`만 수행하고 **타우토머 정규화를 하지 않는다**. 노출 규모를 측정했다
(타우토머 정규화 시 표현이 바뀌는 비율, 그리고 바뀐 분자가 *자기 자신의 다른 타우토머*에
대해 갖는 Tanimoto):

| 세트 | 표현 변경 | 자기-Tanimoto 중앙값 | **0.710 미만** |
|---|---|---|---|
| `actives_core` (29) | 0/29 | — | 0 |
| `benchmark_panel` (6) | 0/6 | — | 0 |
| `actives_extended` (190) | 14/190 (7.4%) | 0.736 | 4 |
| **TL-A prior 샘플 (488)** | **43/488 (8.8%)** | 0.621 (최소 0.379) | **34** |

§8.3의 신규성 기준은 NN-Tanimoto < 0.710이다. **생성 분자 488개 중 34개(7%)가
자기 자신의 다른 타우토머에 대해서도 "scaffold-novel" 판정을 받는다.** PLN-1474가
바로 그 예다 — 알려진 임상 화합물을 그대로 재현해도 0.458로 측정되어 신규 분자로 통과한다.

게다가 비대칭이다: ChEMBL actives는 일관된 방향족 형태로 들어오지만(core 0%, bench 0%)
생성 분자는 모델이 내보내는 형태 그대로다. 기준 쪽은 정규화돼 있고 측정 대상만 흔들린다.

aniline alert도 타우토머 의존적이므로(위 표), RL이 비방향족 형태로 드리프트해
필터를 회피하는 SMILES 수준 reward hacking이 가능하다. `[NX3;H2][c]`는 두 형태 모두
미매치라 이 경로가 닫히지만, 정규화를 상류에 두는 것이 근본 해결이다.

### 2.8 `QED`는 임상 화합물과 무작위 prior 샘플을 구별하지 못한다

PLN-1474(αvβ1 저해제 중 유일하게 임상에 착수)의 QED는 **0.4619**이고,
TL-A prior 샘플의 QED 중앙값은 **0.468**이다. 유일한 임상 진입 화합물이
평범한 prior 샘플과 구별되지 않는다 → 최적화 축으로 쓸 수 없다.

(0.433은 비방향족 타우토머로 계산한 값이며 §2.7의 artifact다.)

### 2.9 `SAScore`는 최적화 축이 아니다 — guard rail이다

sketch의 `reverse_sigmoid(low=3, high=6, k=0.5)` 적용 결과:

| 세트 | SA 중앙값 | SA p90 | SA 최대 | 변환 후 score 중앙값 | score < 0.05 |
|---|---|---|---|---|---|
| `actives_core` | 3.08 | 3.30 | 3.75 | 0.996 | 0% |
| `actives_extended` | 3.48 | 4.27 | 5.10 | 0.981 | 0% |
| `benchmark_panel` | 3.29 | 3.48 | 3.75 | 0.990 | 0% |
| TL-A prior (488) | 2.91 | 3.72 | 5.02 | **0.998** | **0%** |

분포 전체가 변환의 포화 영역에 있다. `geometric_mean`에서 이 항은 항상 ×1이며
0 step에서 gradient에 기여하지 않는다. 창을 prior 중앙값(2.91) 쪽으로 좁혀
살리는 것은 불가하다 — actives 자신이 3.08/3.48이므로 알려진 active를 벌주게 된다.

따라서 제거하지 않고 **guard rail로 유지하되 spec에 그렇게 명시**한다.
600 step RL은 prior가 한 번도 샘플링하지 않는 화학으로 드리프트할 수 있고,
그때만 작동하는 보험으로서는 정확한 설정이다. 이 문서는 SAScore가
"합성 가능성을 최적화한다"고 주장하지 않는다.

**정직한 요약: 0 step 기준 살아 있는 최적화 축은 카르복실레이트 anchor와 TPSA 둘이다.**
TL이 focusing을 이미 끝냈다는 전제를 받아들이면 이것이 일관된 귀결이다.

## 3. 설계

### 3.1 단일 stage RL

`max_steps = 600` (기존 200 + 400과 동일한 총 예산, 단계 경계 없음).

공통 설정은 기존 §5를 유지한다: `learning_strategy` type `"dap"`, `sigma = 128`,
`rate = 0.0001`, `batch_size = 128`, `diversity_filter` =
`IdenticalMurckoScaffold` (`bucket_size = 25`, `minscore = 0.4`, **전 구간 적용**),
`inception`은 active로 시드. 집계는 `geometric_mean`.

선행연구는 diversity filter를 Stage 3에만 적용했다(enriched chemotype으로의 수렴 유도).
우리 목표는 반대 방향(scaffold novelty)이므로 전 구간 적용을 유지한다.

### 3.2 Scoring function

| endpoint | 역할 | transform |
|---|---|---|
| `MatchingSubstructure` 카르복실레이트 | MIDAS anchor, 타협 불가 | — (존재 1.0 / 부재 0.5) |
| `TPSA` | 산성 리간드 투과성 축 (§9.2와 동일 축) | double_sigmoid, 측정된 prior 분포로 보정 |
| `SAScore` | **guard rail** (최적화 축 아님, §2.9) | reverse_sigmoid, 포화 영역 유지 |
| `CustomAlerts` (filter) | 구조적 liability | 아래 §3.3 |

TPSA 창은 `logs/cmp.vigHoB/new3.toml`에서 검증된 `low=40, high=120`
(prior raw 중앙값 102.4, score 중앙값 0.894)을 출발점으로 한다.

### 3.3 `CustomAlerts`

```
[NX3;H2][c]          # 1차 aniline만 (§2.6) — 아닐리드/설폰아닐리드/THN 불포함
[*;r8] [*;r9] [*;r10]
N=[N+]=[N-]
C(=O)Cl
[SH]
[Nr0][Nr0]
+ RDKit PAINS 카탈로그 (§2.6)
```

2-아미노피리딘/2-아미노피리미딘은 `[NX3;H2][c]`에 매치된다. 헤테로고리를 제외하려면
`[NX3;H2][c;!$(c~n);!$(c~o);!$(c~s)]`를 쓸 수 있으나, 차이는 prior 샘플 12개 vs 9개
(0.6%p)이고 참조 세트는 양쪽 모두 0이다. 더 엄격하고 단순한 `[NX3;H2][c]`를 채택한다.

### 3.4 목적함수에서 제거되는 것

| 제거 대상 | 사유 | 이동 위치 |
|---|---|---|
| Stage 1 전체 (200 steps) | §2.1, §2.2, §2.3 | 삭제 |
| `TanimotoSimilarity` (참조 6개) | §2.3 — 달성 불가능 | §3.5 진단 |
| `TanimotoSimilarity` anti-target 2개 | §8.4에 이미 존재 (중복) | §8.4 유지 |
| `SlogP` | §2.5 — 근거 없음, 방향 반대 | §8.2 triage 창 (logP ≤ 5) |
| `QED` | §2.8 — 구별력 없음 | §8.7 최종 선별의 보고 지표로만 (QED ≥ cpd 25) |
| `MolecularWeight`, `NumRotBond`, `HBD` | §2.4 — TPSA/QED와 중복 | §8.2 triage 창 |

`ChemProp2`는 §3.3에서 이미 no-go다. similarity까지 제거하면 **목적함수에 potency
항이 없다.** 이는 의도된 역할 분담이다 — potency 신호는 TL-A(카르복실산 7.6% → 51.1%)와
inception이 담당하고, potency 판정은 §8.5 docking 기하 필터가 사후에 수행한다.

### 3.5 모니터링 (보상이 아닌 진단, 100 step마다)

기존 §5의 모니터링 기대치("Stage 2에서 similarity 하락 = novelty 신호")는
similarity가 목적함수에서 빠지므로 폐기하고 다음으로 대체한다:

- `actives_extended` 대비 NN-Tanimoto 분포 (**타우토머 정규화 후**, §3.6)
- 카르복실산 통과율
- 고유 Murcko scaffold 수
- alert 매치율
- SAScore 분포 (guard rail이 실제로 작동하기 시작했는지)

중단 기준: NN-Tanimoto가 ~0.2 아래로 붕괴하고 **동시에** 카르복실산 통과율이
떨어지는 경우에만 개입한다. 둘 중 하나만으로는 개입하지 않는다.

### 3.6 타우토머 정규화 (§8 전처리 0단계, 신설)

**모든 지문 비교 앞에** `rdMolStandardize.TautomerEnumerator().Canonicalize()`를
의무 적용한다. 적용 대상:

1. 생성/prior 샘플 (§6 sampling 출력)
2. `data/actives_extended.smi`, `data/actives_core.smi`
3. `data/novelty_band.json` **재계산** — 0.710이 변할 수 있다 (extended 14개 영향)
4. anti-target counter-screen 참조 (§8.4)
5. `scripts/curate_actives.py`의 `standardize()` 함수에 추가

기준 세트와 측정 대상에 **동일한** 정규화를 적용하는 것이 요점이다 (§2.7의 비대칭).

## 4. 변경되지 않는 것

- §8 triage funnel의 순서와 내용 (§8.2의 창 목록, §8.3의 대역 기준 방식)
- §9 사전 등록 성공 기준 1–5와 반증 조항
- §12 "의도적으로 하지 않는 것" — RL 내부 docking 미사용 등
- TL-A prior 및 D3 결과 일체

## 5. 파일별 영향

| 파일 | 변경 |
|---|---|
| `korean_avb1_10day_blueprint.md` | **완료** — §5 재작성(§5.1–5.5), §1 표 3·4행 정정, §4 문구, §6, §7 선행조건, §8.0 신설 + §8.1–8.3, §9.3–9.5, §10 D4/D5 |
| `avb1_10day_blueprint.md` (영문) | 동일 내용 반영 — **후속 작업** |
| `reinvent4_avb1_scoring_config_sketch.toml` | 단일 stage로 재작성 — **후속 작업** |
| `configs/_stage1_scoring.frag` | 삭제 — **후속 작업** |
| `configs/_stage2_scoring.frag` | 단일 stage 반영 — **후속 작업** |
| `scripts/curate_actives.py` | `standardize()`에 타우토머 정규화 추가 — **후속 작업** |
| `data/novelty_band.json` | 재계산 — **후속 작업** |

## 6. 미해결 사항

- TPSA 창(`low=40, high=120`)은 `new3.toml`에서 검증됐으나 단일 stage 목적함수
  전체에 대해 다시 채점해 총점 중앙값이 §1의 조건 1(중간 대역)을 만족하는지 확인해야 한다.
- 타우토머 정규화 후 `novelty_band.json`의 0.710이 얼마로 이동하는지는 재계산 전까지 미지.
  §9.3의 수치는 재계산 결과로 갱신한다.
