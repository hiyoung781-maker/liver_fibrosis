# §9.3.1 변경 대장 (v4) — §5.5 ADMET 창

§9.3.1은 리드 선정 임계값의 사후 완화를 허용하되 두 가지를 요구한다:
**§2 성공 판정 기준은 움직이지 않으며**, **모든 변경을 여기 기록한다.**
v2 변경분은 `results/v2_threshold_ledger.md`에 있다. 이 문서는 v4에서
`docs/superpowers/specs/2026-10-03-v4-rerun-design.md` §5.5가 도입한 ADMET
창의 변경 하나를 다룬다.

---

## 변경 1 — §9.2 논리곱을 DILI 단일 상한으로 (2026-10-04)

### 사전등록된 것

§5.5는 **v4 라이브러리 수치를 보기 전에** 다음을 문서에 고정했다:

| | 규칙 |
|---|---|
| Tier 1 (주) | `Caco2_Wang > -5.0` (BCS 고투과, Fa 80–100%) |
| Tier 2 (대안) | `Caco2_Wang > -5.7` (BCS 중간 하단, Fa 50–80%) |
| 독성 | §9.2 그대로 — SR-MMP·NR-AhR·DILI **세 엔드포인트 논리곱**, 승인 산성약 559개 60백분위 |

### 사전등록 규칙의 실측 결과 (13,767개)

```
§9.2 독성 통과            73  (0.53%)
tier1 ∩ 논리곱             2   <- 사전등록이 지시한 채택
tier2 ∩ 논리곱            43
```

**이 수치는 수정 후에도 계속 보고된다** — `admet_window.py`가 수정안을 쓸 때도
위 두 줄을 먼저 출력하고 산출물 헤더에 적는다. `lead_filter.py`의
`all_stages_preregistered`와 같은 규약이다.

### 변경한 것

**`Caco2_Wang > -5.7` ∧ `DILI < 0.75`.** SR-MMP와 NR-AhR은 이 단계에서
게이트가 아니라 **기록**이다.

```
Caco2 통과            11,836
DILI < 0.75 통과         459
∩ 채택                   334
    DILI 이미 통과 (< 0.669635)  160   -> 직행 리드 후보
    전환 구간 (0.6696–0.75)      174   -> 최적화 부모
§9.2 실패 엔드포인트 수별: 0개 43 / 1개 67 / 2개 119 / 3개 105
```

### 왜 사후 합리화가 아닌가

**"논리곱이 2개밖에 안 줘서"가 아니다.** 세 근거가 순서대로 있다.

**(1) 논리곱이 이 단계 자신의 설계와 모순된다.** `optimize_leads.py`
docstring: *"Parents are molecules that cleared the structure and affinity
stages, not the handful that also cleared toxicity ... toxicity is the axis
being repaired, so pre-filtering on it would discard exactly the molecules the
stage exists to improve."* 도킹 전에 세 엔드포인트를 모두 게이트하는 것이 바로
그 pre-filtering이다. 같은 docstring이 *"The reward function ... contains no
toxicity term and no permeability term ... Moving the threshold cannot repair
that. The generator has to be pointed at the axis."*라고도 적는다 — 승인
산성약 28.1% 대 캠페인 1.26%, 22배. v4 실측은 35.1% 대 0.53%, **66배**로 같은
진단이다.

**(2) 제약을 지는 것은 한 엔드포인트다.** 13,767개 실측:

| 엔드포인트 | 임계값 | 통과 | 라이브러리 중앙값 |
|---|---|---|---|
| **DILI** | < 0.669635 | **231 (1.7%)** | **0.9649** |
| NR-AhR | < 0.010712 | 987 (7.2%) | 0.1089 |
| SR-MMP | < 0.011112 | 2,853 (20.7%) | 0.0584 |

DILI만 제외하면 논리곱 통과가 73 → **807**이고, SR-MMP나 NR-AhR을 제외하면
105·101로 거의 변화가 없다. 77.3%가 셋 다 실패한다. 참조 화합물은 DILI가
깨끗하다 — PLN-1474 0.455, bexotegrast 0.330(둘 다 통과). 적응증이 간섬유증
이므로 표적 장기가 위험 장기라는 점에서 이 축의 비중이 특히 크다.

**(3) 0.75는 독립 데이터에서 유도했다 — 개수를 맞추려 고른 값이 아니다.**
v3의 유사체 1,990개(`results/v3_opt/analogues_admet.csv`)는 부모 DILI 위치별로
**무방향** 고유사도 유사체가 임계값을 넘는 비율을 준다:

| 부모 DILI | 짝수 | 전환 | 전환율 |
|---|---|---|---|
| 0.670–0.750 | 74 | 40 | **54.1%** |
| 0.750–0.850 | 272 | 23 | 8.5% |
| 0.850–0.920 | 434 | 10 | 2.3% |
| 0.920–1.010 | 825 | **0** | **0.0%** |

0.92 위 825짝에서 전환이 0이다. **0.75는 54% 구간의 상단**이고, 창의 역할은
부모를 탐색이 성공하는 자리에 놓는 것이다.

### 이 측정이 무엇을 말하지 않는가

v3의 mol2mol은 `run_type = "sampling"`(무방향, 스코어링 함수 없음)이었다.
생성 단계는 DILI를 전혀 개선하지 않았다 — 49.0% 개선 / 51.0% 악화, 델타 중앙값
+0.0004, 부모 중앙값 0.9026 → 유사체 0.9011. **따라서 위 전환율은 이 표본
예산에서의 수율이고, 화학적으로 가능한 것의 상한이 아니다.** 약한 구간도 표본을
늘리면 뚫린다(0.85–0.92에 부모당 ~44개, 0.92 위는 >800개). 다만 천장은 생성이
아니라 **ADMET 처리량**이다 — 13,767개에 49분이므로 334 부모 × 800 표본은
약 16시간이다.

**경사(RL)를 쓰지 않는 것은 의도된 선택이다.** `optimize_leads.py`:
*"Optimising a policy against ADMET-AI rewards whatever the model scores well,
and this repository has already recorded one such failure: ... the agent's best
move was to drop the MIDAS anchor."* v4가 reward hacking 때문에 돌고 있으므로
이 경고는 그대로 유효하다. 0.75 창은 경사 없이 같은 효과를 낸다.

### §2 성공 판정 기준

**움직이지 않았다.** 이 변경은 §5.5의 도킹 입력 선정에만 적용된다. 결합 양식
(MIDAS 산소 ∧ Asn224), 결합력 바(참조 화합물 같은 엔진 최고 포즈), 선택성
(평가 불가)은 그대로다.

### 서술 규칙

리드를 제시할 때 사전등록 기준 통과인지 수정 기준 통과인지 명시하고 두 집합을
섞지 않는다. **사전등록 5단 규칙으로는 Tier 1 ∩ 논리곱 = 2개**이고, 수정
기준으로 334개가 도킹에 들어간다. 최종 리드 카드에 어느 쪽인지 반드시 적는다.
