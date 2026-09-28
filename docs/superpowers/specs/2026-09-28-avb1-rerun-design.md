# αvβ1 de novo 캠페인 재실행 (v2) — 설계 명세

**작성일:** 2026-09-28
**대상:** `avb1_10day_blueprint.md`의 워크플로우를 뼈대로 유지한 채, 조사 보고서(`reports/AvB1 리간드 데이터셋 타당성 검증.md`)의 검증 결과를 반영한 재실행
**선행 문서:** `avb1_10day_blueprint.md`, `avb1_target_decision_brief.md`, `reports/AvB1 리간드 데이터셋 타당성 검증.md`, `research_notes/AvB1 리간드 데이터셋 타당성 검증/`

---

## 0. 연구 목적 (재확인)

**non-RGD(Arg-mimic 염기성 머리 없음) 스캐폴드이면서, PLN-1474에 필적하는 αvβ1 affinity와 αvβ1 선택성을 가지고, 예측 독성이 PLN-1474보다 낮은 리드 화합물을 제안한다.**

v1과 달라진 점: **선택성이 사후 관측이 아니라 판정 기준에 포함된다.** 이에 따라 아이소폼 역도킹 단계가 새로 추가된다.

---

## 1. v1에서 바뀌는 것 — 요약

| # | 항목 | v1 | v2 |
|---|---|---|---|
| 1 | TL 학습셋 | 29 (non-RGD 25 + RGD tool 4) | **25, 순수 non-RGD** |
| 2 | TL 아암 | TL-A 단일 | **TL-A′ + TL-C 두 아암을 끝까지** |
| 3 | inception | `actives_core.smi` 시드 | **양 아암 모두 제거** |
| 4 | 벤치마크 패널 | 6종 | **4종** (CWHM-12·GLPG0187 제거) |
| 5 | 도킹 엔진 | Uni-Dock (Vina 점수) | **AutoDock-GPU (AD4 점수)** 전면 교체 |
| 6 | 게이트 판정 방식 | 원자 간 **거리**만 사용 | **PLIP 상호작용 검출**(각도 포함). 거리는 기록 전용 |
| 6b | 게이트 내용 | MIDAS + Asn224 고정 | **cpd 25 도킹 결과로 사전등록 규칙에 따라 결정** |
| 7 | affinity | "tie-break 전용"이라 선언 후 하드 필터 적용 (자기모순) | **하드 필터로 사전등록, 문구 정정** |
| 8 | 독성 | 가중 복합점수 순위 (게이트 아님) | **동등가중 복합점수, PLN-1474 기준 필터** |
| 9 | 선택성 | 판정 기준에서 제외 | **판정 기준 6번으로 포함, 역도킹 단계 신설** |
| 10 | 리드 선정 | 사후 affinity + 독성 순위 | **3단 필터 → 선택성 rank-sum 단일 랭킹** |

---

## 2. 성공 판정 기준 (샘플링 전 동결)

동일한 퍼널을 통과했을 때, **각 아암에 대해 독립적으로** 다음을 만족하는 분자가 최소 하나 존재하면 성공으로 본다.

1. **PLIP 상호작용 게이트** 통과 (§8에서 확정되는 게이트)
2. ADMET-AI 예측 Caco-2 투과성이 **PLN-1474보다 0.5 log 초과 우위**
3. **AutoDock-GPU affinity가 PLN-1474의 best passing pose보다 낮음** (= 더 강한 결합)
4. **독성 복합점수(§9.2 정의)가 PLN-1474보다 낮음**
5. Murcko 스캐폴드가 `data/known_scaffolds.smi`(106종)에 **없음** — 양측 모두 토토머 정규화 후 판정
6. `CustomAlerts` 무플래그
7. **선택성 rank-sum이 PLN-1474보다 높음** (§10). 단 §10.4의 보정 게이트를 통과하지 못하면 이 기준은 **평가 불가(unevaluable)** 로 기록하며, 실패로 처리하지도 성공으로 주장하지도 않는다.

기준 1–4·7은 §9.1의 필터·랭킹과 같은 양이며, 기준 5–6은 §6의 §8.2 생존자 정의에 이미 강제되어 있다(생성 라이브러리는 정의상 스캐폴드 신규·alert 무플래그다). 즉 성공 판정과 리드 선정은 **같은 수치를 쓴다.**

**반증 조항:** 1–6을 만족하는 분자가 없으면 그대로 보고한다. 카복실레이트–투과성–독성 긴장이 표적 내재적일 가능성의 증거이며, 그 자체가 결과다.

**비교 대상의 일관성:** affinity·투과성·독성·선택성 **네 축 모두 PLN-1474를 단일 비교 대상**으로 쓴다.

---

## 3. 데이터셋 변경

### 3.1 core 멤버십을 규칙으로만 결정

`scripts/curate_actives.py:64`의 `TOOL_COMPOUNDS` 강제 주입을 **제거**한다. 이 주입은 chemotype 규칙과 potency 규칙을 둘 다 우회했다. 제거 후 core는 한 문장으로 정의된다:

> core = non-RGD ∧ 검열되지 않은 αvβ1 IC50/Ki/Kd ≤ 1 µM (+ 8W30 결정 리간드 A1AFA)

| 파일 | v1 | v2 | 비고 |
|---|---|---|---|
| `data/actives_core.smi` | 29 | **25** | TL-A′ 입력 전용 (inception 폐지) |
| `data/benchmark_panel.smi` | 6 | **4** | A1AFA, PLN-1474, CHEMBL4649232, bexotegrast |
| `data/known_scaffolds.smi` | 106 | **106 (불변)** | 아래 근거 |
| `data/folds/` | 29분자 5-fold | 25분자 5-fold 재생성 | |
| `data/novelty_band.json` | core n=29 | core n=25 재계산 | |

**`known_scaffolds.smi`가 불변인 근거.** CWHM-12(`CHEMBL3319237`, αvβ1 IC50 1.8 nM)와 GLPG0187(αvβ1 1.3 nM)은 둘 다 RGD-zwitterion이면서 ≤1 µM이므로 **규칙만으로 `actives_core_B`·`actives_extended`에 자동 포함**된다. PLN-1474와 bexotegrast는 `benchmark_panel`(및 PLN-1474는 `similarity_refs`)에 남는다. 따라서 다섯 파일의 합집합은 변하지 않는다. `tests/test_known_scaffolds.py`가 수정 없이 통과해야 하며, 이것을 검증 장치로 쓴다.

### 3.2 패널에서 CWHM-12·GLPG0187을 제거하는 근거

ChEMBL 전수 조회 결과(2026-09-28):

| | αvβ1 | αvβ8 | αvβ3 | αvβ6 | αvβ5 |
|---|---|---|---|---|---|
| CWHM-12 | 1.8 nM | **0.2** | **0.8** | 1.5 | 61 |
| GLPG0187 | 1.3 nM | 1.2 | 3.7 | 1.4 | 2.0 |

**CWHM-12는 αvβ8에 9배, αvβ3에 2.25배 더 강하다.** 둘 다 αvβ1 선택적이 아니며 pan-αv다. 또한 두 αvβ1 수치 모두 1차 논문이 아니라 리뷰(`CHEMBL3351286`, Sheldrake & Patterson *J Med Chem* 2014)와 GSK αvβ6 논문(`CHEMBL4373707`, Barrett et al. 2019)의 "unknown origin" 제제에서 온다. 목표 프로파일 참조군으로 부적합하다.

**두 화합물은 폐기하지 않고 §10의 선택성 역도킹 보정 대조군으로 재배치한다.**

### 3.3 bexotegrast를 남기는 근거

목표 프로파일(αvβ1 선택적)은 아니지만, **§5.4의 TPSA 창 (40, 115)이 PLN-1474의 100.6과 bexotegrast의 112.5 위에 그어져 있다.** 제거하면 창의 유도 근거가 사라진다. 패널에 남기되 "dual αvβ6/αvβ1, 폐 적응증"으로 라벨을 정확히 단다.

### 3.4 C8 — 연구에서 제외

C8은 강한 αvβ1 선택성이나 최적 포켓 점유를 달성하지 못한 **비교 대상 화합물**이며, Sabat 논문에서 비-RGD 설계의 포켓 채우기·배치의 중요성을 강조하는 사례로 쓰인다. 패널·데이터셋 어디에도 넣지 않는다.

**다만 `CURATION_LOG.md`의 사실 오류는 정정한다.** 현재 로그는 "C8의 이름이 PubChem에서 아무것도 resolve하지 않는다"고 적었으나, Sabat 참고문헌 10 = **Reed NI et al., *Sci Transl Med* 2015;7(288):288ra79**가 C8의 출처다. (부수 효과: `CHEMBL4649232`의 assay가 `[3H]C8` 치환이라는 것은 αvβ1 툴 화합물을 트레이서로 쓴 결합 실험이라는 뜻으로, 그 Ki 값의 신뢰도를 높인다.)

### 3.5 CHEMBL4649232 — 패널 유지, 표기 정정

**유지 근거 (3중 문헌 확인):**
1. Zheng & Leftheris *J Med Chem* 2020, 63:5675–5696, p.58에서 **compound 38**로 등장하며, 인용 맥락이 *"the basic features of RGD mimics can be removed without sacrificing significant potency… 38 with pKi of 9.78 for αvβ1… will stimulate further interest in pursuing novel non-zwitterionic integrin modulators"* — 이 프로젝트의 가설 그 자체.
2. 1차 출처 = 리뷰의 참고문헌 141 = **Hatley RJD, Barrett TN, Slack RJ, et al., "The Design of Potent, Selective and Drug-like RGD αvβ1 Small-Molecule Inhibitors Derived from non-RGD α4β1 Antagonists", *ChemMedChem* 2019, 14(14), 1315–20.**
3. **Sabat 논문이 같은 논문을 참고문헌 16으로 독립 인용**하며, *"Inhibitors exhibiting limited affinity for αv integrins other than αvβ1—such as GSK-11 and C8—lack this same 3-aminopropionic framework"*라고 기술한다. 또한 *"our modeling of GSK-11 and 2 revealed significant overlap of the 2,6-dichlorobenzamide regions… suggesting that careful occupancy of this region can deliver preferential αvβ1 binding affinity."* CHEMBL4649232는 그 2,6-dichlorobenzamide를 가지며 3-aminopropionyl framework가 없다.

**표기 정정:** 엔드포인트는 **Ki(pKi 9.78)**이며 IC50이 아니다. 출처 문서는 `CHEMBL4602673`(리뷰), 1차 출처는 Hatley 2019. assay는 `CHEMBL4604552`([³H]C8 치환 결합). `actives_annotated.csv`에 `endpoint_type`·`primary_source` 컬럼을 추가한다.

**명시할 한계:** ⓐ 선택성은 **논문 서술 근거이며 큐레이션 데이터로 검증되지 않았다** — ChEMBL에 이 분자의 교차 아이소폼 기록이 0건이고, Hatley 2019 문서는 ChEMBL에 색인조차 없다(ChemMedChem 2019 vol.14 문서 0건 확인). ⓑ `CHEMBL4649232 == GSK-11` 여부는 미확인이다. **액션: Hatley 2019 원문 확보.**

### 3.6 compound 25 — 이미 저장소에 있음

**`CHEMBL5532604` = Sabat compound 25**로 확정. document `CHEMBL5500400`, **pIC50 9.5**(논문 Table 3 ELISA 값과 일치), TPSA 146.2, MW 481.8, non-RGD.

```
Cc1ccc(Cl)c(C(=O)N[C@@H](CNC(=O)Cn2c(=O)[nH]c3nc(C)c(F)cc32)C(=O)O)c1F
```

`actives_annotated.csv`에 `paper_id = "Sabat2024:25"`로 기록한다. **cpd 25는 `actives_core`의 구성원이므로 TL-A′의 학습 분자이며, 패널에 넣지 않는다.** §7의 관찰 대상으로만 쓴다.

### 3.7 기타 표기 정정

- `CHEMBL3319237`(CWHM-12)은 ChEMBL `pref_name`이 `None`이고 synonym 목록이 비어 있다. **추론 동정**임을 로그에 명시한다.
- `CHEMBL2381700`은 피리미딘에 붙은 3차 다이에틸아미노기를 가진다. SMARTS 기준으로는 non-RGD이나(지방족 아민 패턴이 `!$(Na)`로 방향족 결합 질소를 배제, 2-aminopyridine 패턴은 `[NX3;H1,H2]` 요구), **25개 중 유일하게 판정이 규칙에 의존하는 분자**다. 로그에 경계 사례로 기록하고 출처 document를 확인한다.
- `curate_actives.py:59` 주석 "All **five** are RGD mimetics" → 넷으로 정정.

### 3.8 철회된 지적

조사 노트 01의 "§3.1은 Arg-mimic 6분류를 말하는데 코드는 3개만 구현" 지적은 **틀렸다.** `ARG_MIMIC_HEADS`에는 guanidine/cyclic amidine, tetrahydronaphthyridine, 2-aminopyridine, benzimidazol-2-amine, 2-aminoazole, aliphatic amine **여섯 개가 모두 있다.** 문서와 코드는 일치한다.

---

## 4. TL 아암

| 아암 | prior | 학습 입력 | inception |
|---|---|---|---|
| **TL-A′** | `reinvent.prior` → TL → `focused_A_prime.prior` | `data/actives_core.smi` (25) | 없음 |
| **TL-C** | `reinvent.prior` (그대로) | 없음 | 없음 |

**단일 변수 절제:** 두 아암은 prior가 focused인지 stock인지만 다르다. 목적함수·diversity filter·스텝 수·시드·배치 순서 전부 동일하다.

### 4.1 inception 제거 근거 (임계값 없는 순서 논증)

1. `REINVENT4/reinvent/runmodes/RL/memories/inception.py:86-90` — 메모리는 점수 내림차순 정렬 후 `maxsize`로 잘린다. 클래스 docstring: *"Only the N most high scoring SMILES will be retained. This typically means that the original SMILES will be quickly replaced by newly created SMILES during optimization."*
2. 같은 파일 279행 — **매 스텝 배치 전체(128개)가 메모리에 추가**된다.
3. 같은 파일 41·57행 — `_smilies` 중복 제거 집합은 축출 시 갱신되지 않아, **한 번 밀려난 시드는 재진입이 영구 차단**된다.
4. blueprint §7.2 실측 — TPSA 133.3에서 TPSA 성분 0.001, 총점 0.060. `CURATION_LOG.md` 기준 core의 **TPSA 중앙값은 145**이고 TPSA 창은 (40, 115)이므로, 25개 중 A1AFA(TPSA 66.4, 총점 1.000)를 제외한 전부가 총점 0.06 이하다.
5. blueprint §6.2 실측 — 에이전트의 목적함수 총점 **중앙값 1.00**.

생성 분자가 시드보다 위에 정렬된다는 사실만으로 축출이 따라 나오므로 새 임계값이 필요 없다. blueprint §5가 요구했으나 한 번도 실행하지 않은 검증(*"Verify first that the seed molecules actually score well under the scoring function"*)의 답이 이것이다.

추가로 inception 손실은 저장된 **prior log-likelihood**를 쓰므로(175행), 같은 시드라도 TL-A′(prior_LL 높음)와 TL-C(prior_LL 낮음)에서 **부호가 반대인 신호**가 된다. 절제의 단일성을 위해서도 제거가 옳다.

### 4.2 TL-A′ epoch 선택 규칙 (사전등록)

200 epoch을 재사용하지 않는다. v1에서 200은 산 밀도·스캐폴드 수·QED로 골랐고, **암기(max NN-Tanimoto 1.000)는 고른 뒤 사후 기록**되었다. n이 25로 줄면 암기는 심해진다.

**스윕 범위: 0 → 200 epoch, 20 epoch 간격 체크포인트**, 각 지점에서 1,000개 샘플링. 선택 구간은 **epoch ≥ 100**으로 제한한다.

> epoch 100–200 구간의 체크포인트 중, **고유 Murcko 스캐폴드 ≥ 700/1000**을 만족하는 **가장 높은** epoch을 채택한다.
>
> 어떤 체크포인트도 만족하지 못하면 **TL-A′는 아암에서 탈락**시키고 그 사실을 기록한 뒤 TL-C 단일 아암으로 간다.

**암기는 탈락 조건이 아니라 정량 보고 항목이다.** 초안은 `max NN-Tanimoto < 1.000`을 선택 조건에 넣었으나, epoch ≥ 100 제약과 충돌할 수 있고 v1에서 이미 암기 꼬리가 **하류에서 제거**되는 것이 확인되었다(§8.3의 Murcko 신규성 게이트가 `known_scaffolds.smi` 소속 스캐폴드를 버린다). 따라서:

- 채택된 epoch에서 `actives_extended` 대비 **max NN-Tanimoto와 novelty band 초과 비율을 반드시 보고**한다 (v1: max 1.000, 4.9%가 기존 계열 유사도 대역 내, 1.2%가 p90 초과).
- **신규성 주장을 할 때 이 꼬리가 몇 개 제거되었는지 수치로 함께 밝힌다.** v1 §4의 표현("any novelty claim must state that this tail was removed first")을 그대로 계승한다.

### 4.2.1 로깅 요구사항

- **wandb에 epoch마다 train loss와 validation NLL을 모두 기록**한다.
- 프로덕션 스윕에서도 **`save_every_n_epochs = 20`**을 명시한다. v1 §4가 기록한 함정 — *"validation NLL is only computed on save epochs"* — 때문에, 이 값을 진단 실행에서만 1로 두면 프로덕션 스윕의 체크포인트에 validation NLL이 남지 않는다.
- `configs/tl.toml.in`의 `{SAVEFREQ}` 치환을 `run_d3.sh`에서 프로덕션 경로에도 적용하도록 수정한다.
- held-out NLL은 여전히 **게이트가 아니라 진단**이다(§4.3).

### 4.3 fold 불균형 (기록 사항)

core가 25로 줄면 스캐폴드는 19 → 약 15로 줄고, fold 1(9분자 azabenzimidazolone 단일 스캐폴드)이 전체의 36%를 차지한다. **held-out NLL은 게이트가 아니라 진단**이라는 §4의 기존 지위를 유지하며, epoch은 §4.2의 샘플링 메트릭으로 정한다.

---

## 5. RL (양 아암 동일)

기존 설정 유지: `learning_strategy = "dap"`, σ = 128, `rate = 1e-4`, `batch_size = 128`, `diversity_filter = IdenticalMurckoScaffold` (`bucket_size = 25`, `minscore = 0.4`, 전 구간 적용), `aggregation = geometric_mean`, **`max_steps = 1000`**.

**목적함수 (불변):**

| 컴포넌트 | 종류 | transform |
|---|---|---|
| `GroupCount` 카복실레이트 `[CX3](=O)[OX2H1,OX1-]` | scored, w=1.0 (게이트) | `right_step(high=1)` |
| `TPSA` | scored, w=1.0 | `double_sigmoid(low=40, high=115, coef_div=120, coef_si=coef_se=20)` |
| `SAScore` | scored, w=0.5 (가드레일) | `reverse_sigmoid(low=6.0, high=8.0, k=0.5)` |
| `CustomAlerts` (8 SMARTS) | filter | — |

**문서 모순 정정:** §5 본문의 "a single stage of **600 steps**"는 shared settings의 `max_steps = 1000` 및 §8.7.1의 "the 1000-step single-stage agent"와 충돌한다. 실행된 값은 1000이므로 **1000으로 통일**한다.

**모니터링(100스텝마다, 보상 아님):** NN-Tanimoto 분포(토토머 정규화 후), 카복실산 통과율, 고유 Murcko 스캐폴드 수, alert 매치율, SAScore 분포.

---

## 6. 샘플링과 §8.2 윈도

아암당 20,000개 샘플링 → 검증/중복제거 → 정준화 → **토토머 정규화** → 중복제거.

§8.2 속성 윈도 유지: TPSA는 RL과 동일 창, logP ≤ 5, MW 250–550, RotB.

`library.smi`는 기존대로 2열(1열 정규화 SMILES, 2열 정규화 전 정준 SMILES)을 유지한다.

---

## 7. 도킹 — AutoDock-GPU 전면 교체

### 7.1 엔진 교체의 함의

AutoDock-GPU는 **AutoDock4 점수함수 + 사전계산 그리드 맵**을 쓴다(Uni-Dock의 Vina 점수함수와 다름). 결과:

- **금속 항이 생긴다.** blueprint가 반복 인정한 최대 결함(*"Vina has no metal coordination term at all"*)이 부분적으로 해소된다. 다만 금속 항이 있다는 것과 정확하다는 것은 다르므로 **기하 필터는 유지**한다(v1과의 비교 가능성을 위해서도).
- **affinity 값이 Vina와 비교 불가**하다. PLN-1474 기준값을 새로 산출한다.
- **파서·그리드가 새로 필요**하다. `autogrid4`로 맵을 생성하고, 출력은 `.dlg`/`--xmloutput`이라 `REMARK VINA RESULT` 파서를 쓸 수 없다.

`scripts/dock_unidock.py`는 남기고 `scripts/dock_autodock_gpu.py`를 신설하되, **박스 정의는 기존 `box_from_ligand`를 그대로 임포트**한다(smina `--autobox_ligand` + `--autobox_add 6` 규칙 재현). 박스가 조용히 달라지는 것이 §8.5.1이 경고한 실패 모드다.

### 7.2 수용체 준비 (§8.5.2 교훈 계승)

- 8W30 chain A+B, 리간드·물·당쇄 제거, **Ca²⁺ 6개 유지**
- `obabel -xr -h` — 수소 포함. 수소 없이 변환하면 1,212개 질소 중 1,170개가 `NA`(H-결합 **수용체**)로 타이핑되어 단백질 전체의 H-결합 지형이 뒤집힌다.
- `prepare_receptor_pdbqt.py`의 **질소 donor/acceptor 비율 검사**를 통과해야 한다(수용체가 다수면 실패 판정).
- **`obabel -p 7.4` 금지** — 칼슘 6개를 전부 제거하고도 점수는 정상으로 보였다.

### 7.3 양성자화 상태 재결정 (사전등록)

§8.5.3이 중성 COOH를 택한 근거 중 하나는 *"Vina has no electrostatic term"*이었다. **AD4에는 정전기 항이 있으므로 이 근거가 소멸한다.**

> 대조군(A1AFA)을 **중성 COOH와 음이온 COO⁻ 두 형태로 모두** 도킹한다. 결정 접촉(Ca501 2.62 Å, Asn224 2.63 Å)을 재현하는 쪽을 채택한다. **둘 다 재현하면 음이온**을 택한다 — pH 7.4의 실제 존재 형태이고, 점수함수가 이제 전하를 본다.

### 7.4 재도킹 검증 게이트 (사전등록, 미통과 시 중단)

> AutoDock-GPU로 A1AFA를 8W30에 재도킹해 **상위 포즈**에서 다음을 모두 만족해야 한다:
> ⓐ 카복실레이트 O → Ca501 **≤ 3.2 Å**
> ⓑ 도너 → β1-Asn224 backbone O **≤ 3.5 Å**
> ⓒ 대칭보정 heavy-atom RMSD (`rdMolAlign.CalcRMS`) **< 2.0 Å**
> ⓓ **PLIP이 재도킹 포즈에서 Ca501 metal complex와 β1-Asn224 backbone 수소결합을 모두 보고**할 것 (§8.0의 검증을 결정 좌표가 아닌 재도킹 포즈에 적용한 것)
>
> 미통과 시 **AutoDock-GPU로 어떤 것도 순위 매기거나 거르지 않으며**, Uni-Dock 체제로 복귀하고 그 사실을 기록한다.

ⓐ·ⓑ는 거리로, ⓓ는 PLIP으로 같은 두 접촉을 확인한다. **둘이 어긋나면** — 예컨대 거리는 2.7 Å인데 PLIP이 수소결합으로 인정하지 않으면 — 그것이 §8.0이 지적한 각도 문제의 실물이므로, 어긋난 사례를 기록하고 PLIP 판정을 따른다.

RMSD는 반드시 `CalcRMS`로 계산한다. 인덱스 기반 나이브 RMSD는 v1에서 0.63 Å를 6.13 Å 실패로 보고했다.

### 7.5 1차 배치: cpd 25 + 패널 + 대조군

동일 배치·동일 수용체·동일 파라미터로 도킹한다: **compound 25(`CHEMBL5532604`), 패널 4종, 대조군 A1AFA.** PLN-1474의 새 affinity 기준값이 여기서 나온다.

### 7.6 측정 전량 보존

**거리 (판정에 쓰지 않음, 기록 전용).** `pose_geometry.py`를 확장해 포즈마다 다음을 모두 컬럼으로 기록한다:

`d_ca501`, `d_Asn224_bb`, `d_Leu225_bb`, `d_Leu225_sc`, `tyr178_nearest_ring_atom`, `tyr178_centroid`, `tyr178_ring_angle`, `d_Asp218`, `affinity`

**PLIP 상호작용 (판정 근거).** 포즈마다 PLIP이 보고한 상호작용을 잔기 단위로 기록한다:

`metal_complexes`(대상 이온·잔기·거리), `hbonds`(도너/억셉터 잔기·거리·각도), `hydrophobic_contacts`(잔기 목록), `pi_stacking`(잔기·유형 P/T·centroid 거리·각도·offset), `salt_bridges`, `water_bridges`, `halogen_bonds`

게이트는 §8.2의 사전등록 규칙 하나만 적용하되 나머지는 보고용으로 남는다. **포즈 파일과 PLIP 리포트를 아암별로 보존**한다 — v1은 42,419개 포즈를 남기지 않아 소급 적용이 불가능했다.

### 7.7 AutoDock-GPU 재현성 측정

같은 리간드를 다른 seed로 반복 도킹해 **엔진 자체의 affinity 편차**를 측정한다. 컷오프 근방 분자 수를 이 편차와 함께 보고한다(v1에서는 리드 20개 중 5개가 컷오프로부터 0.05 kcal/mol 이내였고, 이는 측정된 엔진 간 차이 0.16 kcal/mol보다 작았다).

---

## 8. 상호작용 게이트 판정 규칙 (도킹 **전** 확정)

### 8.0 PLIP을 게이트로, 거리는 기록으로

v1의 게이트는 원자 간 거리만 봤다. **거리 단독 기준은 이 프로젝트에서 이미 두 번 문제를 일으켰다.**

1. **Tyr178 기준이 결정 구조 자신을 기각했다.** "centroid ≤ 5.5 Å"으로 썼는데 8W30의 centroid는 **6.21 Å**이다. T-shaped 스택(고리 평면각 74.6°)은 두 고리를 수직으로 놓아 centroid를 벌리기 때문이다. v1은 "최근접 고리 원자 ≤ 4.5 Å"으로 땜질했다.
2. **수소결합 기준이 각도를 보지 않아 사실상 HBD 개수 필터로 작동했다.** `[#7,#8;!H0]` 원자가 Asn224 backbone O로부터 3.5 Å 이내면 통과인데, 측정된 통과율은 **HBD=1일 때 6.93% → 2일 때 37.4% → 3일 때 55.2%**다. 도너가 많으면 그중 하나는 우연히 거리 안에 들어온다.

**PLIP(Protein-Ligand Interaction Profiler)을 1차 상호작용 검출기로 채택한다.** 이유:

- **각도 조건을 포함한다** — donor–H···acceptor 각도를 보므로 기하학적으로 불가능한 "수소결합"을 거른다.
- **π-stacking을 parallel/T-shaped로 나눠 각각의 각도·offset 기준으로 판정한다** — 위 1번 문제가 애초에 발생하지 않는다.
- **metal complex를 명시적으로 검출한다** — MIDAS 접촉이 "거리 ≤ 3.2 Å"이 아니라 "Ca501과의 금속 배위"라는 화학적 서술이 된다.
- **β1-Leu225의 backbone 수소결합과 측쇄 소수성 접촉을 각각 분리해 준다** — §8.2의 판정 규칙이 필요로 하는 구분이다.
- 표준·인용 가능한 기준이라 포스터에서 방어하기 쉽다.

**치르는 비용 (명시):**

- **명시적 수소가 필요하다.** PDBQT는 비극성 수소가 병합되어 각도 계산이 불가능하다. 포즈를 수소 포함 PDB 복합체로 변환하는 단계를 추가한다(`scripts/poses_to_sdf.py`·`export_poses.py` 확장 → `scripts/poses_to_complex.py`).
- **v1의 통과율과 직접 비교할 수 없다.** PLIP 임계값은 PLIP의 것이다.
- **PLIP 기본값도 결국 외부 임계값이다.** "구조가 경계를 정하게 한다"는 v1의 원칙과는 방향이 다르나, 각도를 보는 표준 기준이라는 이점이 그 손실보다 크다고 판단한다.

**거리는 버리지 않는다.** §7.6의 모든 거리 컬럼을 그대로 기록하되 **판정에는 쓰지 않는다.**

#### PLIP 검증 게이트 (사전등록, 미통과 시 거리 기준으로 복귀)

> PLIP을 **8W30 결정 복합체**에 적용해 기록된 상호작용을 재현해야 한다:
> ⓐ **Ca501과의 metal complex**
> ⓑ **β1-Asn224 backbone O와의 수소결합**
> ⓒ **αv-Tyr178과의 π-stacking (T-shaped)**
> ⓓ **β1 소수성 포켓 접촉** (Tyr133 / Pro186 / Cys187 / Leu225 중 최소 하나)
>
> 재현하지 못하면 PLIP을 채택하지 않고 v1의 거리 기준으로 복귀하며, 그 사실을 기록한다.

**ⓒ가 핵심 확인 항목이다** — 거리 기준이 실패했던 바로 그 지점이므로, PLIP이 결정 구조의 T-stack을 잡지 못하면 도입 근거가 사라진다.

#### 실행 환경

PLIP은 **GPU 구현이 없는 CPU 도구**(OpenBabel 기반)다. K-BDS의 **`cpu64` 파티션에서 포즈 단위 병렬**로 실행한다. `8gpu`는 GPU 사용량과 무관하게 시간당 8 node-hour를 과금하므로(§8.5.1) 여기서 PLIP을 돌리면 할당량만 소모된다. 복합체당 1초 내외이므로 약 85,000 포즈(2아암)는 64코어에서 30분대다.

### 8.1 논문이 말하는 앵커

Sabat p.10307: *"When anchored by the **MIDAS and β1-Leu225 interactions**, a compact lipophilic group can be trajected into this hydrophobic pocket which likely contributes to 2's affinity for αvβ1."*

바로 앞: 3-aminopropionyl framework를 가진 화합물은 *"β1-Leu225 engagement would drive affinity for **multiple integrin isoforms**"*.

**읽어낼 구조:** Leu225 접촉은 선택성의 원천이 아니라 **공용 앵커**다(선택적 화합물과 pan 화합물이 모두 만든다). αvβ1 선호를 만드는 것은 그 앵커 위에서 **compact lipophilic group이 β1 소수성 포켓(Tyr133/Pro186/Cys187/Leu225 + αv-Tyr178)을 채우는지**다. GLPG0187은 MIDAS·Leu225를 만족시킨 상태에서 그 포켓을 못 채운다는 것이 논문의 모델링 결론이다.

**→ blueprint §8.4의 "Leu225가 선택성의 유일한 β3 차이 잔기"라는 해석을 이에 맞게 수정한다.**

### 8.2 판정 규칙 (사전등록)

기본 게이트는 **`Ca501 metal complex` AND `2차 극성 앵커`**이며, 2차 앵커의 정의만 cpd 25의 결과로 결정된다. 모든 판정은 **PLIP 상호작용 검출** 결과로 하며 거리는 쓰지 않는다.

> cpd 25의 포즈 중 **PLIP이 Ca501 metal complex를 보고하는 포즈 가운데 affinity 최상위 포즈**를 판정 대상으로 한다.
>
> 1. PLIP이 **β1-Asn224 backbone O와의 수소결합**을 보고 → **게이트 불변**: `Ca501 metal complex` AND `Asn224 backbone H-bond`
> 2. Asn224 수소결합은 없고 **β1-Leu225 backbone O와의 수소결합**을 보고 → **구간 완화**: `Ca501 metal complex` AND `β1 223–226 구간의 임의 backbone O와의 수소결합`
> 3. 둘 다 없음 → **`Ca501 metal complex` 단독**으로 축소하고 2차 접촉을 관측치로 강등
> 4. cpd 25가 metal complex를 보고하는 포즈를 하나도 내지 못함 → 게이트 불변, 사실을 기록
>
> **어떤 경우에도 Tyr178 π-stacking은 게이트가 되지 않는다** — 통과 집단에서 방향족 고리 보유의 농축이 1.03배(= 없음)로 측정되었고, compound 1의 벤질기에 특이적인 접촉이다. **Leu225 측쇄 소수성 접촉도 게이트가 되지 않는다** — v1 측정 64.3%로 변별력이 약하다. 둘 다 PLIP 출력에 그대로 기록된다.
>
> **모든 경우에 A1AFA가 채택된 게이트를 통과해야 한다.** 유일하게 결정학적으로 관측된 결합자를 거르는 게이트는 채택하지 않는다(§8.0의 PLIP 검증 게이트 ⓐ·ⓑ가 이를 보장한다).

### 8.3 구간 완화(2안)의 구조적 근거

인접 백본 카보닐 산소는 서로 3.80–3.85 Å 떨어져 있어, 한 잔기를 지목하면 이 계열이 실제로 쓰는 수용체를 배제한다. 조사한 5개 인테그린 복합체 중 Asn224 등가 접촉을 만드는 것은 tirofiban(β3-Asn215, 2.76 Å) 하나뿐이고, 나머지는 인접 카보닐을 쓴다 — 1L5G β3-Arg216 (3.46 Å), 2VDN β3-Arg216 (3.21 Å), 4UM9 β6-Ile219 (2.86 Å), **3VI4(α5β1) β1-Leu225 (3.74 Å)**. 3VI4는 우리와 **같은 β1 사슬**이라 가장 직접적인 근거다.

### 8.4 패널은 튜닝 대상이 아니라 보고 대상

채택된 게이트로 패널을 통과시키고 **알려진 활성체에 대한 위음성률**로 보고한다. v1에서는 GLPG0187(0/9)과 CHEMBL4649232(0/4)가 탈락했는데, **GLPG0187이 패널에서 빠진 v2에서는 CHEMBL4649232 하나가 유일한 위음성 후보**다 — 하필 목표 프로파일에 가장 잘 맞는 화합물이므로 §8.2 판정의 초점이 된다.

### 8.5 명시할 한계

cpd 25는 `actives_core`의 구성원이므로 **TL-A′의 학습 분자**다. 이 도킹은 독립 검증이 아니라 **시리즈 최적화 리드의 결합 양식 관찰**이다. 또한 논문의 cpd 25 결합 양식은 그 자체가 **모델**이며 결정학적으로 관측된 적이 없다(Figure 5B 캡션이 문자 그대로 *"Model of compound 25 bound to αvβ1"*). 우리 도킹 결과를 논문 모델과 비교하는 것은 **두 예측의 비교**다.

---

## 9. 리드 선정 — 3단 필터 + 선택성 단일 랭킹

### 9.1 필터 (모두 PLN-1474 기준, 사전등록)

| 순서 | 필터 | 기준 |
|---|---|---|
| 1 | 기하 게이트 | §8에서 확정된 게이트 |
| 2 | 투과성 | ADMET-AI Caco-2가 PLN-1474보다 **> 0.5 log** 우위 |
| 3 | affinity | AutoDock-GPU affinity가 PLN-1474의 **best passing pose** 값보다 **낮음**(더 음수 = 더 강함) |
| 4 | 독성 | 동등가중 복합점수(§9.2)가 PLN-1474의 같은 정의 값보다 **낮음** |

**스캐폴드 신규성과 alert는 여기 다시 나오지 않는다** — §6의 §8.2 생존자 정의에 이미 강제되어 있어, 이 필터에 들어오는 분자는 정의상 Murcko 신규이고 alert 무플래그다. v1도 같은 구조였다(§8.7.1: *"All 1,586 geometry survivors are Murcko-novel and alert-free by construction"*).

**affinity 문구 정정 (3곳).** §12 "No predicted-affinity claims" → "예측 affinity를 절대 친화도로 주장하지 않는다. affinity는 사전 정의된 참조 화합물(PLN-1474) 대비 **상대 하드 필터**로만 사용하며 순위 근거로 쓰지 않는다." §8.5b "score is used only to break ties" → "docking score는 §9의 PLN-1474 상대 하드 필터에 사용된다. 포즈 선택은 여전히 기하로만 한다." §8.7.1의 affinity 행을 "post hoc" → **"pre-registered"**.

**선행연구 인용을 정확히 한다.** [113](Qie/Wang/Li, *Molecular Diversity* 2026, doi 10.1007/s11030-026-11625-z)은 *"For reference, gefitinib yielded a docking score of −8.8 kcal/mol under identical parameters"*라고 적었으며 이는 **참조값 제시**다. 따라서 blueprint에는 "[113]이 참조 화합물의 도킹 점수를 비교 기준으로 제시한 선례를 따라, 본 연구는 이를 **하드 필터로 적용한다**"로 쓴다. "[113]대로 필터링했다"고 쓰지 않는다.

### 9.2 독성 복합점수 — 가중치 제거

v1의 DILI 2.0 / hERG·AMES 1.0 / CYP 0.5 가중치(분모 11.5)는 근거가 없고, CWHM-12(비경구 pan-αv 툴, TPSA 172.4)가 PLN-1474보다 "덜 독성"(0.130 vs 0.144)으로 나오는 위생 검사 실패를 냈다.

**v2 정의:** ADMET-AI 자체 성능이 가장 좋은 세 엔드포인트의 **동등 평균** — **SR-MMP (AUROC 0.925), NR-AhR (0.904), DILI (0.881)**.

PLN-1474의 문서화된 값으로 계산한 예상 기준값: (0.0208 + 0.0093 + 0.455) / 3 = **약 0.162**. 실제 실행에서 재계산한다.

**알려진 한계 (문서화, 은폐하지 않음):** 동등가중도 **동적 범위 지배 문제를 해결하지 못한다** — PLN-1474에서 DILI 0.455 대 SR-MMP 0.0208로, 평균은 여전히 DILI가 지배한다. 완화 수단은 엔드포인트별 순위 정규화 후 평균이지만 이는 변환을 하나 더 도입하는 것이므로, **v2는 동등 평균을 쓰고 이 한계를 명시**한다. hERG·AMES·CYP3A4/2C9/2D6는 필터에서 빠지되 **리드 카드에 병기**한다.

### 9.3 예상되는 생존 수 (v1 실측 기반 경고)

v1에서 세 필터를 다 걸었을 때: 1,586 → 389(투과성) → 216(affinity) → **2**(독성, PLN-1474 기준). 389개 중 독성까지 이긴 것은 11개였다.

원인도 측정되어 있다: *"A more permeable molecule is more lipophilic, and that raises the hERG, CYP and DILI predictions."* 이는 §9가 사전등록한 **카복실레이트–투과성 긴장**이 독성 축으로 번진 형태다.

**생존 수가 적으면 그 수 자체를 긴장의 측정값으로 보고한다.**

### 9.3.1 사후 완화 정책 — 허용하되 분리하고 기록한다

이 단계는 **REINVENT4가 생성한 것 중 그나마 나은 것을 추리는 과정**이므로, 기준이 지나치게 엄격해 리드가 거의 나오지 않으면 임계값을 사후에 완화할 수 있다. 단 다음 두 가지를 지킨다.

**(1) §2 성공 판정 기준은 사후에 움직이지 않는다.** 반증 조항이 여기 걸려 있어, 이 임계값을 사후에 바꾸면 사전등록 자체가 무의미해진다. §2는 **항상 사전등록 임계값 그대로 보고**한다.

**(2) §9의 리드 선정 임계값은 완화 가능하며, 모든 완화를 변경 대장에 기록한다.**

| 축 | 사전등록 임계값 | 완화 후 | 완화 전 생존 | 완화 후 생존 | 사유 |
|---|---|---|---|---|---|
| (실행 시 채움) | | | | | |

v1의 §8.7.1이 이미 "pre-registered / post hoc" 열을 갖고 있으므로 그 관행을 잇는다. 이렇게 하면 **"엄격한 기준으로는 N개, 완화하면 M개"를 둘 다 보고**하게 되어, 긴장의 크기가 오히려 정량화된다.

**포스터 서술 규칙:** 리드를 제시할 때 그 리드가 **사전등록 기준을 통과한 것인지 완화된 기준을 통과한 것인지 명시**한다. 두 집합을 섞어서 제시하지 않는다.

### 9.4 최종 랭킹

필터 4단을 통과한 분자를 **§10의 선택성 rank-sum**으로 정렬하고, Murcko 스캐폴드 중복 없이 상위부터 리드로 선정한다(목표 아암당 10–20개, 생존 수가 그보다 적으면 전부 보고).

---

## 10. 선택성 역도킹 (판정 기준 6번)

### 10.1 원칙

**서로 다른 수용체 간 절대 점수를 비교하지 않는다.** 대신 **같은 리간드에 대한 두 수용체의 점수 차이**를 쓴다:

```
Δ_s = affinity(subtype s) − affinity(αvβ1)
```

AD4/Vina류 점수는 리간드 크기·원자 수에 계통적으로 비례하므로, 동일 리간드의 차이값은 그 계통 오차를 약분한다. Δ가 클수록(양수일수록) off-target 결합이 나쁨 = 선택적.

**rank-sum 정의.** 검증을 통과한 아이소폼 `s`마다, 대상 집합(리드 후보 + 참조 화합물) 전체를 `Δ_s` **내림차순**으로 순위 매긴다(1위 = 가장 선택적). 각 분자의 선택성 rank-sum은 그 순위들의 합의 **역순 점수**로 정의하며, **값이 클수록 선택적**이다. 아이소폼별 점수 스케일 차이는 순위화 단계에서 흡수된다. 판정 기준 7번의 "PLN-1474보다 높음"은 이 rank-sum 기준이다.

### 10.2 대상 구조

| 아이소폼 | 구조 | 축 |
|---|---|---|
| αvβ3 | 1L5G | β 서브유닛 — 최우선 |
| α5β1 | 3VI4 | **α 서브유닛** (우리와 같은 β1) |
| αvβ6 | 4UM9 | bexotegrast·CWHM-12의 강한 표적 |
| αvβ8 | 6UJA 계열 | CWHM-12의 **최강** 표적 (0.2 nM) |

### 10.3 아이소폼별 재도킹 검증

각 구조의 자체 결정 리간드를 같은 프로토콜로 재도킹해 **PLIP이 MIDAS 금속 배위를 보고 + 대칭보정 RMSD < 2.0 Å**를 만족해야 한다. **재현하지 못하는 아이소폼은 분석에서 제외**하고 기록한다. αvβ1과 동일한 기준이다.

각 아이소폼의 결정 복합체에 대해서도 PLIP 프로파일을 만들어, 해당 구조의 문헌상 알려진 접촉이 재현되는지 확인한다(§8.0의 검증을 아이소폼별로 반복).

### 10.4 보정 게이트 (사전등록 — 선택성 축의 전제)

> **CWHM-12와 GLPG0187이 Δ 축에서 비선택적으로, PLN-1474와 CHEMBL4649232가 선택적으로** 나와야 한다. 재현하지 못하면 프로토콜에 변별력이 없는 것이므로, **생성 분자에 대한 어떤 선택성 진술도 하지 않으며 판정 기준 6번을 평가 불가로 기록한다.**

정답이 실험 측정값(CWHM-12: αvβ1 1.8 / αvβ3 0.8 / αvβ6 1.5 / αvβ8 0.2 nM; GLPG0187: 1.3 / 3.7 / 1.4 / 1.2 nM)이므로 무작위 디코이보다 강한 대조군이다.

### 10.5 대상 집합과 비용

§9.1의 필터 1–4를 통과한 분자 + 참조 화합물(PLN-1474, CHEMBL4649232, bexotegrast, A1AFA, CWHM-12, GLPG0187). 4 아이소폼 × 수백 리간드는 본 실행(7,763개 / 약 20분 / 8×A100)에 비해 미미하다.

### 10.6 기하 관측 (게이트 아님)

각 아이소폼에서 리드가 **유효한 결합 양식 포즈**를 만드는지 기록한다. 단 **MIDAS 카복실레이트 접촉은 class-invariant**이므로(측정: 1L5G 2.65, 2VDM 2.36, 2VDN 2.08, 3VI4 1.91, 4UM9 2.13, 8W30 2.62 Å — 전부 단좌) 이를 감점 기준으로 쓰면 전원 감점이 된다. 변별은 비-MIDAS 접촉에서 나와야 하며 그 영역은 증거가 약하므로, **보고 항목으로만** 둔다.

서열 정합 분석을 병행한다 — §8.4가 β1/β3에 대해 수행한 것(포켓 5잔기 중 Leu225만 다르고 β3는 Arg)을 β6·β8·α5로 확장한다.

### 10.7 명시할 한계

ⓐ αvβ1은 8W30 하나뿐이라 **교차 도킹 검증이 불가능**하다. ⓑ 강체 수용체라 유도 적합을 못 본다. ⓒ **이것은 선택성의 측정이 아니라 위험 평가**이며, 실제 선택성은 교차 아이소폼 IC50 측정으로만 판정된다.

---

## 11. 산출물 경로

v1 결과를 보존하고 아암별로 분기한다.

```
results/v1_TL-A/              # 기존 results/ 내용 이동
results/v2_TL-A-prime/
results/v2_TL-C/
results/v2_{arm}/plip.csv     # 포즈별 PLIP 상호작용 (판정 근거)
results/v2_{arm}/geometry.csv # 포즈별 거리 (기록 전용)
results/v2_threshold_ledger.md # §9.3.1 사후 완화 변경 대장
docking/v2/poses/{arm}/       # 포즈 파일 보존 (v1의 소급 불가 문제 방지)
docking/v2/plip_reports/{arm}/ # PLIP 원본 리포트
priors/focused_A_prime.prior
```

**신규 스크립트:** `scripts/dock_autodock_gpu.py`(박스는 기존 `box_from_ligand` 임포트), `scripts/poses_to_complex.py`(수소 포함 PDB 복합체 생성), `scripts/run_plip.py`(cpu64 병렬), `scripts/selectivity_rank.py`(§10의 Δ·rank-sum).

---

## 12. 블루프린트 수정 목록

| 위치 | 현재 | 수정 |
|---|---|---|
| §3.4 | CHEMBL4649232 "pIC50 9.78 (IC50 0.166 nM)" | **Ki 0.166 nM (pKi 9.78)**, assay `[3H]C8` 치환, 1차 출처 Hatley *ChemMedChem* 2019 |
| §0, §5.4, §7.5, decision brief §5/§7 | cpd 25 "MDCK < 0.1×10⁻⁶ cm/s" | **a–b 3.0 / b–a 3.2 ×10⁻⁶** (Table 3). `<0.1`은 Table 2 벤즈아마이드 시리즈 |
| decision brief 62·80·128행 | cpd 25 ">50-fold selective" (논문 인용처럼) | 논문은 배수를 인쇄하지 않음. Table 4 바닥값 유도: α4β1/α5β1/α8β1 ≥74×, **αvβ3 ≥23×**, αvβ6 ≥230×, αvβ8 ~220× |
| §4 arm 표 | TL-A "chemotype-pure" (같은 행에 14% RGD) | v2에서 TL-A′는 실제로 100% non-RGD |
| §7.5 | PLN-1474 "three αvβ6 entries (IC50 50 nM)" | **검열된 `<50 nM` IC50 1건 + 빈 kon/k_off 2건.** 결론(αvβ1 기록 없음)은 유지 |
| §1(5행), decision brief 63·98·202행 | "TR01225179"를 논문 명명법처럼 사용 | 논문은 이 코드를 쓰지 않음. **PDB 8W30 엔트리 타이틀에만** 존재하는 Takeda 사내 등록번호. 논문 호칭은 "acid **1**" |
| decision brief §6(98행) | "N-(2,4-dichlorobenzoyl)-**L**-phenylalanine = acid 1" | 구성은 맞음. 논문은 1을 **라세미**로 보고하며 "L"은 침전 모델에서 옴. pIC50 5.3은 50:50 혼합물 값 |
| (프로젝트 내 구전) | "αvβ1에 결합한 최초의 라세미 산성 화합물" | **논문에 없는 표현이며 논문은 어떤 우선권 주장도 하지 않는다.** 정확히는 "사내 compound collection ELISA 스크린 히트, 비-zwitterion이라는 이유로 우선순위, pIC50 5.3, 타 아이소폼 전부 <4.5" |
| §9.6 | 패널 기하 실패가 "four of the six are RGD zwitterions"라 예상됨 | v1 실패 2건 중 CHEMBL4649232는 프로젝트 자신이 **non-RGD**로 분류. v2에서는 유일한 위음성 후보 |
| `CURATION_LOG.md` | C8 "이름이 PubChem에서 resolve되지 않음" | C8의 출처는 **Reed NI et al., *Sci Transl Med* 2015;7(288):288ra79** (Sabat 참고문헌 10) |
| `CURATION_LOG.md` | `CHEMBL3319237` = CWHM-12 | ChEMBL `pref_name` None, synonym 없음 → **추론 동정**으로 표기 |
| §3.4 / §7.2 | 패널을 "reference distribution"이라 지칭 | n=4, 이종 assay. **명명된 랜드마크 집합**으로만 유효 |
| §9.6 / `logs/leads.txt` | "Caco-2 MAE 0.26–0.28", "실험실 간 불일치 0.57 log" | **미검증.** 번들된 ADMET-AI 성능 표에서 Chemprop-RDKit의 Caco-2 MAE를 확인해 인용하고, 0.57 log는 출처를 찾거나 삭제. 마진 0.5가 잡음 상한을 넘는다는 주장은 근거 확인 시에만 |
| §5 | "a single stage of 600 steps" | **1000 steps** (shared settings 및 §8.7.1과 일치) |
| §12 / §8.5b / §8.7.1 | affinity "순위에 쓰지 않음" ↔ 하드 필터 적용 | §9.1의 문구로 통일 |
| §8.4 | Leu225를 선택성의 원천으로 해석 | 논문상 Leu225는 **공용 앵커**이며, 선택성은 그 위에서 **β1 소수성 포켓 점유**로 결정됨 |
| §1(64행) | `CHEMBL5500400` 26분자, 커버리지 "complete" | **미검증.** 인용 전 재확인 |
| `curate_actives.py:59` | "All **five** are RGD mimetics" (넷 나열) | 넷으로 정정 |

---

## 13. 범위 밖 (의도적 제외)

- **음성 디코이 패널 없음.** 기하 게이트의 통과율에 분모가 없다는 한계를 명시하되, 이번 재실행에서는 만들지 않는다. §10의 보정 게이트가 실험 측정값 기반 대조군 역할을 부분적으로 대신한다.
- **RL 내부 도킹 없음.**
- **습식 실험 없음.** 산출물은 사전등록된 반증 가능 기준을 갖춘 in silico 리드 집합이다.
- **Mol2Mol/LibInvent/LinkInvent 없음.**

---

## 14. 미해결 항목 (인용 전 확인 필요)

1. **Hatley et al., *ChemMedChem* 2019, 14(14), 1315–20 원문 확보** — CHEMBL4649232 == GSK-11 여부, 측정된 교차 아이소폼 선택성, ADME.
2. **ADMET-AI 번들 성능 표** — Caco-2 MAE의 실제 값.
3. **"실험실 간 불일치 0.57 log"의 출처.**
4. **`CHEMBL5500400` 26분자 주장** 재확인.
5. **`CHEMBL2381700`의 출처 document와 계열** — 25개 중 유일한 chemotype 경계 사례.
6. **Bash 복구 후 RDKit으로 25개 non-RGD 전수 재확인** (현재는 파이프라인 QC 게이트 기록 + 수동 구조 검토로 확인됨).
