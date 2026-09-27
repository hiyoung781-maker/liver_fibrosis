# Target Decision Brief — 간섬유화 치료를 위한 Integrin αvβ1
### REINVENT4 기반 lead-proposal 프로젝트를 위한 근거 기반 타깃 우선순위 선정 및 논증 구조

**프로젝트:** 간섬유화(MASH 중심)의 직접 항섬유화 lead로서 신규 αvβ1 integrin 저해제의 in silico 설계.
**본 문서의 성격:** 7분 발표 및 A0 포스터를 위한 의사결정 기록(decision record) + 논증 골격. 모든 사실 주장에는 번호가 매겨진 참고문헌이 붙어 있다(목록은 문서 말미).

---

## 1. 설계 가설 (프로젝트를 한 문장으로)

> **선행 연구가 규명한 αvβ1 선택성 pharmacophore — MIDAS-metal carboxylate anchor, β1-Asn224 H-bond, αv-Tyr178 aromatic contact, β1 hydrophobic subpocket — 는 공개된 tool compound보다 passive permeability와 oral-likeness가 실질적으로 더 우수한 신규 scaffold로도 구현될 수 있다. 즉, tool-compound의 한계는 타깃 자체의 속성이 아니라 지금까지 탐색된 series의 속성이다.**

생성형 캠페인은 이 가설의 *검증 수단*이다. AI(REINVENT4)는 도구이지 주제가 아니다.

---

## 2. 문제 정의

- **간 관련 사망을 가장 강력하게 예측하는 것은 염증이나 지방증이 아니라 fibrosis stage다.** NAFLD/MASH에서 F4는 사망 또는 이식에 대해 HR ≈ 10.9 [52]; 스웨덴 biopsy 코호트에서는 NASH가 아니라 fibrosis stage가 사망률과 중증 간질환을 예측했다(F4에서 HR 최대 ≈ 105) [53]; 전향적 NASH CRN 코호트 역시 F3–F4가 간 합병증과 사망을 주도함을 확인했다 [55].
- **MASH는 흔한 질환이다** — 미국 성인의 약 6%(약 1,490만 명) [78].
- **승인된 두 약물은 대사적이며 간접적이다.** Resmetirom(THR-β agonist, 2024년 3월 비대상성이 아닌 F2–F3 MASH에 승인)은 fibrosis를 1단계 이상 개선한 비율이 placebo 14% 대비 24–26%에 그쳤다 [33, 36, 38]. Semaglutide(GLP-1 RA, 2025년 8월 동일 적응증 승인)는 22.4% 대비 36.8%에서 fibrosis를 개선했다 [78, 79, 80]. 두 약물 모두 간경변에는 승인되지 않았고, 섬유화 기전 자체를 직접 겨냥하지도 않는다.
- **직접 항섬유화 약물 개발은 무덤(graveyard)이다.** Selonsertib(ASK1)은 두 건의 Phase 3에서 실패했고 [1, 6], cenicriviroc(CCR2/5)은 Phase 3 AURORA에서 실패했으며 [2], emricasan(pan-caspase)도 실패했고 [5], simtuzumab(anti-LOXL2) 등도 같은 목록에 올랐다 [8]. NASH 프로그램의 약 70%가 Phase 2→3 전환에서, 약 42%가 Phase 3→승인 단계에서 실패한다 [7].
- **이 분야가 얻은 교훈:** human genetic support가 있는 타깃은 승인 가능성이 2배 이상 높다 [110, 107]. 그래서 유전학적 근거를 우선순위 매트릭스의 축으로 삼았고, αvβ1에 그 근거가 없다는 점은 숨기지 않고 약점으로 명시했다(7장).

**공백(Gap):** 섬유화를 일으키는 세포(hepatic stellate cell, HSC)나 그 matrix 산물에 직접 작용하는 승인 치료제는 없다. 직접 항섬유화제는 대사적 표준치료에 더해질 합리적인 add-on이다.

---

## 3. 우선순위 선정 프레임워크와 매트릭스

평가 기준(각 0–2점; 0점이 아니려면 근거가 필요):

| # | 기준 | 본 프로젝트에서 중요한 이유 |
|---|-----------|--------------------------------|
| C1 | Directness (HSC-intrinsic 또는 matrix-directed) | 사용자가 부과한 하드 필터; "직접 항섬유화제"의 정의 |
| C2 | Human validation (genetics 또는 clinical PoC) | 유전학적 근거는 성공 확률을 2배로 [110, 107] |
| C3 | 저분자 tractability + 알려진 chemotype | REINVENT4에는 prior와 scoring anchor가 필요 |
| C4 | 구조 데이터 | Docking 기반 선별에는 pocket이 필요 |
| C5 | 경쟁 whitespace | 제안에는 메워지지 않은 공백이 필요 |
| C6 | 7분 안에 전달되는 서사적 일관성 | 발표 형식상의 제약 |

| 타깃 | C1 | C2 | C3 | C4 | C5 | C6 | 판정 |
|--------|----|----|----|----|----|----|---------|
| **Integrin αvβ1** | ✓✓ HSC-intrinsic [10, 103] | ✓ in-vivo genetics + clinical class PoC [98, 103]; germline GWAS 없음 | ✓✓ 공개 SAR series + 임상 진입 화합물 [85, 111] | ✓✓ PDB 8W30 [112] | ✓✓ 간 프로그램이 방치됨 [86, 88] | ✓✓ | **추천** |
| HSD17B13 | ✗ 간세포 효소, 간접적 [30, 31] | ✓✓ LOF genetics [25, 27] | ✓✓ 저해제 + 구조 [21, 45] | ✓✓ | ✓ 임상은 RNAi 일변도 [57, 61] | ✓ | 탈락: directness 미충족; phenocopy 위험(촉매 저해 ≠ 단백질 소실) [16, 24]; 간경변에서 유해 신호 가능성 [28] |
| PNPLA3 | ✗ 대사적 | ✓✓ 가장 강력한 genetics [66] | ✗ gain-of-function neomorph; ASO/siRNA 전용 [67, 70, 71] | ✗ | ✓ | ✓ | 탈락: 저분자 전략 부재 |
| LOXL2/LOX | ✓✓ matrix-directed | ✗ simtuzumab이 간섬유화에서 실패 [8] | ✓ | ✓ | ✗ | ✗ | 탈락: 바로 이 적응증에서 타깃이 실패함 |
| ALK5 (TGFBR1) | ✓✓ | ✗ 전신적 TGF-β 차단의 독성 | ✓✓ | ✓✓ | ✗ | ✓ | 탈락: αvβ1이 해결하려는 반면교사 |
| MARC1 | ✗ 대사적 | ✓ genetics [75] | ✗ chemotype 없음 | ✗ | ✓✓ | ✗ | 탈락: 인간/마우스 간 기능 불일치 [74] |
| α8β1/α11β1 | ✓✓ HSC-specific [9] | ✗ | ✗ 저해제 없음 | ✗ | ✓✓ | ✓ | 탈락: 생성모델 prior 부재 |
| NOX1/4, TEAD/YAP | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | 탈락: 검증 근거가 얕고 간 PoC 없음 |
| THR-β / GLP-1 | ✗ | ✓✓ | ✓✓ | ✓✓ | ✗ 이미 승인됨 [33, 78] | ✓ | 탈락: whitespace 없음 |

이 매트릭스는 그대로 포스터에 실을 수 있으며, 2단계(근거 기반 우선순위 선정)가 실제로 작동한 필터였음을 보여준다 — directness 기준을 엄격히 적용해 우리의 첫 후보(HSD17B13)까지 탈락시켰다는 점에서 그렇다.

---

## 4. 왜 αvβ1인가: 근거의 사슬

1. **인과적이고 세포 내재적인 생물학.** PDGFRβ+ pericyte 계열 세포(HSC)에서 αv integrin을 결손시키면 마우스가 CCl₄ 유발 간섬유화로부터 보호되며, 저분자 αvβ1 저해제 CWHM-12는 섬유화를 감소시킨다. 이 경로는 여러 고형 장기에 공통된 핵심 경로다(Henderson, *Nat Med* 2013) [103, 14].
2. **인간 HSC에서의 기전.** αvβ1은 **인간 HSC에서 가장 풍부한 αv integrin**이며, 이를 차단하면 **SMAD 인산화나 TGF-β 활성화와 무관한 noncanonical 경로**를 통해 procollagen-1 생성이 억제된다 [10, 111]. 즉 αvβ1은 latent TGF-β 활성화 이상의 방식으로 섬유화를 조절한다(7장의 중복성 반론과 연결됨). RGD-integrin 차단은 HSC senescence도 유도한다(Kitsugi 2023, [111]에서 인용). 리뷰: [12, 9].
3. **간섬유화 모델에서의 저분자 입증.** Sabat 등(*J Med Chem* 2024)은 선택적 non-RGD αvβ1 저해제(azabenzimidazolone **25**: cellular αvβ1 pIC₅₀ 6.37; α4β1/α5β1/α8β1/αvβ3 및 αvβ6/αvβ8 대비 50배 이상 선택적)를 개발했고, **rat CDHFD 모델에서 용량 의존적 항섬유화 효능**을 보였다 [111].
4. **구조적 근거.** 인간 αvβ1 headpiece와 저해제 TR01225179의 공결정 구조(PDB 8W30) [112] — 6장에서 분석.
5. **약물 클래스의 임상적 검증.** PLN-1474(경구, αvβ1 선택적)는 Phase 1을 무사히 완료했고(지원자 84명, 용량제한독성 없음) 전임상 간섬유화 효능을 보였다 [85]. Bexotegrast(경구 αvβ6/αvβ1 이중 저해)는 IPF Phase 2a INTEGRIS에서 항섬유화 신호(FVC, 정량적 폐섬유화 영상, 바이오마커)와 양호한 내약성을 보였다 [98, 99, 101].
6. **공백(whitespace).** Novartis는 2023년 NASH 포트폴리오 *전체*를 정리하면서 PLN-1474를 중단했다(같은 시기에 tropifexor와 FGF21 프로그램도 정리됨) — 과학적 판단이 아니라 전략적 판단이었다 [86, 87, 90]. 그 이후 간 대상 αvβ1 프로그램이 진행 중이라는 보고는 없다 [88, 89].

---

## 5. 논증 구조 (선행 연구를 중심으로 논지를 짜는 방법)

**원칙: 선행 연구는 경쟁 상대가 아니라 문제 정의다.** Sabat 논문은 토대, 공백, 생성모델 prior, 구조, 그리고 벤치마크를 모두 제공한다.

**1단계 — 토대(선행 연구가 증명한 것; 재논증하지 말고 인용할 것).** 타깃 인과성 [103], 인간 HSC 기전 [10], in-vivo 저분자 효능 [111], 인체 안전성이 확인된 경구 화합물 [85], 섬유화에서의 클래스 PoC [98].

**2단계 — 공백(선행 연구가 명시적으로 제공하지 *못한* 것, 그들 자신의 수치로).**

| 화합물 | αvβ1 선택적? | Oral-like / drug-like? | 공개 SAR? | 상태 |
|----------|--------------------|-----------------------|-------------|--------|
| CWHM-12 (tool) | ✗ pan-αv [103, 111] | ✗ | ✓ | 문헌상의 tool compound |
| Sabat cpd 25 | ✓✓ (>50배) [111] | ✗ rat에서 **oral F = 1.3 %**(s.c. 59 %); series 전반의 MDCK permeability가 측정 하한(<0.1 × 10⁻⁶ cm/s); 효능에 **50 mg/kg BID s.c.** 필요 [111] | ✓ | 저자 스스로 **tool compound**로 규정 [111] |
| PLN-1474 | ✓ [85] | ✓ (경구, Ph1 무사 통과) [85] | ✗ 비공개 | **2023년 중단(전략적 사유)** [86, 87] |
| Bexotegrast | 부분적(αvβ6/αvβ1 이중) [98] | ✓ | 부분적 | 임상 진행 중 — 단, 간이 아니라 **폐** [98] |

**비어 있는 칸: 공개되어 있으면서 αvβ1 선택적이고 oral-like한 chemotype. 바로 그 칸이 이 프로젝트다.**

**3단계 — 설계 가설**(1장). 어떤 생성 작업보다 먼저 선언되며, 구조상 반증 가능하다.

**4단계 — 검증.** 공개 active에 대한 transfer learning → pocket pharmacophore + permeability 축 + selectivity counter-screen을 담은 multi-component score로 reinforcement learning → 8W30로의 docking 선별 → ADMET → novelty 분석(전체 프로토콜은 동반 blueprint 문서에).

**5단계 — 사전 등록된 성공 기준.** 생성된 lead는 다음을 만족해야 한다: (i) 8W30에서 TR01225179에 준하는 docking 성능을 보이며 MIDAS + Asn224 contact를 보존할 것; (ii) 예측 permeability/lipophilicity 축에서 cpd 25를 능가할 것; (iii) αvβ3/α5β1/αvβ6 counter-screen에서 깨끗할 것; (iv) 공개 series 대비 scaffold가 신규일 것(nearest-neighbor Tanimoto가 사전 설정한 상한 미만); (v) 동일한 funnel을 통과시킨 두 baseline — 알려진 active(양성 대조)와 무작위 drug-like ChEMBL 분자(음성 대조) — 를 능가할 것. **contact를 유지하면서 목표 축에서 tool compound를 능가하는 분자가 하나도 나오지 않는다면 가설은 반증된 것이며, 포스터는 그 결과를 그대로 보고한다.** 사전 등록된 반증 가능성이야말로 제안(proposal)을 튜토리얼과 구분 짓는 엄밀성의 신호다.

**6단계 — 심사위원이 지적하기 전에 먼저 밝히는 한계.** In silico는 검증이 아니다. 최종 산출물은 논거를 갖추고 벤치마킹된 제안과, 명확히 정의된 실험적 확인 경로다(competitive ELISA binding [111] 방식; 인간 HSC에서의 procollagen-1 assay [10] 방식).

---

## 6. 결합 포켓 특성 분석 (PDB 8W30, 본 연구에서 분석)

구조: 인간 αvβ1 headpiece와 **TR01225179**(= N-(2,4-dichlorobenzoyl)-L-phenylalanine, 논문의 screening hit "acid 1")의 공결정 [111, 112]. 아래 거리는 등록된 좌표에서 직접 측정했다(본 연구; 표는 `avb1_8w30_contacts.csv`, 그림은 `avb1_pocket_3d.png`, `avb1_pocket_contacts.png`).

**네 개의 anchor (리간드가 반드시 유지해야 할 요소):**

| Anchor | 측정된 기하 구조 | 비고 |
|--------|--------------------|------|
| **MIDAS metal coordination** | carboxylate OXT → Ca501 (β1): **2.62 Å**, monodentate | orthosteric RGD-integrin 결합에 carboxylate는 타협 불가; β1-Ser132/Ser134/Glu229가 측면에서 보조(3.0–3.3 Å) |
| **β1-Asn224 H-bond** | backbone O ← 리간드 amide NH: **2.63 Å** | 중심 amide가 H-bond donor |
| **αv-Tyr178 aromatic contact** | 최근접 ring 원자 **3.70 Å**; centroid–centroid 6.2 Å, ring-plane 각도 ~75°(T-shaped/offset) | 리간드의 phenylalanine ring과 상호작용 |
| **β1 hydrophobic subpocket** | dichlorophenyl ring이 Pro186 (3.16 Å), Tyr133 (3.17), Cys187 (3.88), Lys182 (3.82) 사이에 위치 | 형태로 결정되는 선택성 요소 |

**두 개의 growth vector (선행 연구가 겨냥했고, 생성 설계도 겨냥해야 할 방향):**

| Vector | 기하 구조 | 비고 |
|--------|----------|------|
| **αv-Asp218** | hit로부터 **7.0 Å** — 8W30에서는 *관여하지 않음* | cpd 25의 benzimidazolone NH는 Asp218에 수소를 공여하도록 설계되었다; 핵심 potency/selectivity vector [111] |
| **계면 water HOH2107** | hit로부터 4.7 Å; αv-Glu121과 β1-Ser177을 연결 | 후속 analog들이 이 물 분자를 겨냥해 설계되었다 [111] |

**설계상의 함의 (REINVENT4 프로토콜로 이어지는 다리):**
- 이 pocket은 **metalloenzyme과 유사**하다. 일반적인 docking score는 metal coordination을 모델링하지 못하므로, scoring이 carboxylate–MIDAS anchor를 보존하도록 해야 한다(substructure 제약 + docking 후 metal-distance 필터).
- **선택성은 β1-subpocket 문제**다(Asn224/Leu225 영역 + hydrophobic pocket의 형태). 따라서 αvβ3/α5β1/αvβ6에 대한 counter-screen은 사후 점검이 아니라 생성 목적함수 *안에* 들어가야 한다.
- **Permeability 한계는 구조적이다**: MIDAS carboxylate와 H-bonding amide가 TPSA를 높이고 logD를 낮춘다 — 이는 생성 score가 최적화해야 할 바로 그 축이자, cpd 25가 tool compound로 끝난 바로 그 이유다(oral F 1.3 %) [111]. 선택적 탐색 arm으로 acid bioisostere(tetrazole, acylsulfonamide)를 시험하며 potency 손실 위험을 감수할 수 있다 — 그 트레이드오프 자체가 곧 가설 검증이다.

---

## 7. 공격 표면 (예상 반론과 방어)

1. **"선행 연구가 이미 in-vivo 효능을 갖춘 선택적 αvβ1 저해제를 만들었는데, 무엇을 더하는가?"** 그 논문의 데이터가 답한다: cpd 25는 저자 스스로 규정한 *tool compound*로, oral F 1.3 %, 효능은 50 mg/kg s.c. BID에서만, permeability는 측정 하한이다 [111]. 이 프로젝트는 비어 있는 칸(공개 + 선택적 + oral-like)을 겨냥하며, cpd 25는 재현 대상이 아니라 넘어서야 할 벤치마크다.
2. **"human germline genetics가 없다 — 당신들 매트릭스 스스로 유전학이 성공률을 2배로 만든다고 했는데."** 가장 약한 기둥임을 인정한다 [110, 107]. 방어: ITGB1은 어디서나 필수적이므로 purifying selection이 흔한 LOF 변이를 제거한다 — 즉 인과적 타깃이라도 GWAS 침묵은 예상되는 결과다. 대신 검증은 세포 특이적 in-vivo genetics [103], 인간 HSC 약리학 [10], 임상 class PoC [98]에서 온다.
3. **"Novartis가 PLN-1474에서 손을 뗐다 — 뭔가 봤을 수도 있다."** 공개된 기록은 NASH 포트폴리오 전면 철수를 보여주며(관련 없는 여러 자산이 동시에 정리됨) [86, 87, 90], 데이터에 근거한 신호는 공개된 바 없다. 정직한 답: 비공개 데이터의 존재를 배제할 수는 없다 — 해결된 문제가 아니라 열린 질문으로 제시한다.
4. **"24종 integrin 전반에 걸친 선택성은 지뢰밭이다."** 맞는 지적이며, 그래서 counter-screen(αvβ3, α5β1, αvβ6, αvβ8)을 생성 scoring function에 내장한다. 이 pocket에서 50배 이상의 선택성은 실증적으로 달성 가능하다 [111]. 또한 αvβ6 선택적 항체 BG00011은 IPF에서 안전성 문제가 있었는데 [102], 이는 pan-αv가 아닌 β1 선택적 저해에 유리한 논거다.
5. **"TGF-β는 다른 경로(αvβ6/β8, thrombospondin-1, MMP, ROS)로도 활성화된다 — 차단은 부분적일 것이다."** TGF-β 축에 대해서는 일부 사실이다. 그러나 인간 HSC에서 αvβ1의 procollagen 조절은 부분적으로 TGF-β 활성화와 *무관*하다(noncanonical) [10, 111]. 전체 경로 차단이 아니라 HSC 축 제어로 프레이밍한다.
6. **"MIDAS carboxylate가 본질적으로 permeability의 상한을 만들 수 있다 — 가설이 틀릴 수도 있다."** 맞으며, 이는 사전 등록된 반증 조건으로 명시되어 있다(5장 5단계). 완화책: acidic bioisostere arm, integrin 문헌의 prodrug 선례, 그리고 pharmacophore를 유지하면서 permeability를 *어디까지* 밀어붙일 수 있는지를 규명한다는 fallback 주장 — 어느 쪽이든 정량적 답이 나온다.
7. **"만성적 TGF-β 조절은 장기적으로 안전하지 않을 수 있다(조직 복구, 종양 감시)."** Integrin에 국한된, 활성화 단계에서의 차단이야말로 전신적 TGF-β 차단 대비 제안된 안전성 기전이다 — 다만 이는 가설이며, 장기 안전성은 이 프로젝트만으로는 알 수 없다. 그 점을 명시한다.
8. **"생성 결과물은 약이 아니다."** 동의한다. 산출물은 벤치마킹되고 반증 가능한 lead *제안*과 명시적인 실험적 확인 경로이며, 이는 이 발표 형식이 요구하는 주장 수준에 해당한다.

---

## 8. 7분 스토리 아크

1. **(0:00–1:00) 사람을 죽이는 것은 섬유화다** — 예후 데이터 [52, 53, 55]; MASH의 규모 [78].
2. **(1:00–2:00) 두 건의 승인, 직접 항섬유화제는 0건** — resmetirom/semaglutide는 대사적이고 F2–F3에 한정되며 fibrosis 효과는 제한적 [33, 38, 78, 80]; 그리고 실패의 무덤 [1, 2, 5, 7, 8].
3. **(2:00–3:00) TGF-β 딜레마** — 마스터 조절자이지만 전신적으로는 약으로 만들 수 없다; 자연은 latent TGF-β를 integrin을 통해 *국소적으로* 활성화하며, 인간 HSC에서 그 스위치는 αvβ1이다 [10, 103].
4. **(3:00–4:00) 검증되었으나 버려진 타깃** — in-vivo genetics [103], 선택적 저해제의 효능 [111], 인체 안전성이 확인된 Ph1 화합물 [85], 섬유화에서의 class PoC [98] — 그리고 타깃을 공백으로 남긴 전략적 개발 중단 [86].
5. **(4:00–5:00) 선행 연구 자신의 수치로 본 진짜 공백** — 세 화합물 비교 표; cpd 25의 1.3 % 경구 생체이용률 [111]; 설계 가설.
6. **(5:00–6:15) 검증** — REINVENT4 funnel(TL → pharmacophore + selectivity + permeability scoring을 적용한 RL → 8W30 docking → ADMET → novelty), 벤치마크 패널, 사전 등록된 성공 기준 [40].
7. **(6:15–7:00) Lead 제안과 한계** — 레퍼런스 대비 3–5개 제안 lead; 반증 조건; 실험적 확인 경로.

---

## 9. References

1. Harrison SA et al. Selonsertib … STELLAR trials. *J Hepatol* 2020. doi:10.1016/j.jhep.2020.02.027
2. Anstee QM et al. Cenicriviroc … AURORA Phase III. *Clin Gastroenterol Hepatol* 2024. doi:10.1016/j.cgh.2023.04.003
5. Harrison SA et al. Emricasan in NASH F1–F3. *J Hepatol* 2020. doi:10.1016/j.jhep.2019.11.024
6. Rinella ME, Noureddin M. STELLAR 3 and 4: Lessons from the fall of Icarus. *J Hepatol* 2020. doi:10.1016/j.jhep.2020.04.034
7. Ratziu V, Friedman SL. Why Do So Many NASH Trials Fail? *Gastroenterology* 2023. doi:10.1053/j.gastro.2020.05.046
8. Drenth JPH, Schattenberg JM. The NASH drug development graveyard. *Expert Opin Investig Drugs* 2020. doi:10.1080/13543784.2020.1839888
9. Yokosaki Y, Nishimichi N. α8β1 and α11β1 on activated stellate cells. *Int J Mol Sci* 2021. doi:10.3390/ijms222312794
10. Han Z et al. Integrin αVβ1 regulates procollagen I … in human hepatic stellate cells. *Biochem J* 2021. doi:10.1042/bcj20200749
12. Rahman S et al. Integrins as a drug target in liver fibrosis. *Liver Int* 2022. doi:10.1111/liv.15157
14. Hinz B. It has to be the αv: myofibroblast integrins activate latent TGF-β1. *Nat Med* 2013. doi:10.1038/nm.3421
16. Ma Y et al. Antisense oligonucleotide targeting Hsd17b13 in a fibrosis mouse model. *J Lipid Res* 2024. doi:10.1016/j.jlr.2024.100514
21. Chen L et al. Highly potent, selective, liver-targeting HSD17B13 inhibitor. *J Med Chem* 2025. doi:10.1021/acs.jmedchem.5c00119
24. Ma Y et al. HSD17B13 deficiency does not protect mice from obesogenic diet injury. *Hepatology* 2021. doi:10.1002/hep.31517
25. Abul-Husn NS et al. A protein-truncating HSD17B13 variant and protection from chronic liver disease. *NEJM* 2018. doi:10.1056/nejmoa1712191
27. Stender S, Romeo S. HSD17B13 as a therapeutic target. *Liver Int* 2020. doi:10.1111/liv.14411
28. Gil-Gómez A et al. HSD17B13 rs72613567 and hepatic decompensation in cirrhosis. *Int J Mol Sci* 2022. doi:10.3390/ijms231911840
30. Vilar-Gomez E et al. HSD17B13 rs72613567 fibrosis protection mediation. *Clin Gastroenterol Hepatol* 2023. doi:10.1016/j.cgh.2022.11.002
31. Luukkonen PK et al. HSD17B13 variant, phospholipids and fibrosis in NAFLD. *JCI Insight* 2020. doi:10.1172/jci.insight.132158
33. FDA press release: Rezdiffra approval, March 14, 2024. fda.gov
36. Madrigal Pharmaceuticals press release, March 14, 2024.
38. Fierce Pharma: "FDA approves first MASH drug…", March 14, 2024.
40. Loeffler HH, He J et al. Reinvent 4: Modern AI-driven generative molecule design. *J Cheminform* 2024. doi:10.1186/s13321-024-00812-5
45. RCSB PDB 8G89 (HSD17B13–inhibitor complex).
52. Angulo P et al. Liver fibrosis, but no other histologic features, predicts long-term outcomes in NAFLD. *Gastroenterology* 2015. doi:10.1053/j.gastro.2015.04.043
53. Hagström H et al. Fibrosis stage but not NASH predicts mortality in NAFLD. *J Hepatol* 2017. doi:10.1016/j.jhep.2017.07.027
55. Sanyal AJ et al. Prospective study of outcomes in adults with NAFLD (NASH CRN). *N Engl J Med* 2021. (PMC8881985)
57. Sanyal AJ et al. Phase I RNAi (rapirosiran) targeting HSD17B13 in MASH. *J Hepatol* 2025. doi:10.1016/j.jhep.2025.05.031
61. Mak LY et al. Phase I/II ARO-HSD in NASH. *J Hepatol* 2022. doi:10.1016/j.jhep.2022.11.022
66. Lindén D, Tesz G, Loomba R. Targeting PNPLA3 to treat MASH. *Liver Int* 2024. doi:10.1111/liv.16186
67. Armisen J et al. AZD2693 (PNPLA3 ASO) Phase I. *J Hepatol* 2025. doi:10.1016/j.jhep.2024.12.046
70. Sherman DJ et al. PNPLA3-I148M is a neomorph. *Cell Rep* 2025. doi:10.1016/j.celrep.2025.116371
71. Wang Y et al. PNPLA3(148M) gain-of-function via ATGL inhibition. *J Hepatol* 2025. doi:10.1016/j.jhep.2024.10.048
73. Coyne ES et al. Loss of mARC1 reduces fibrosis in mouse CLD models. *Hepatol Commun* 2025. doi:10.1097/hc9.0000000000000637
74. Smagris E et al. Divergent role of MARC1 in human and mouse. *PLOS Genet* 2024. doi:10.1371/journal.pgen.1011179
75. Emdin CA et al. A missense variant in MARC1 and protection against liver disease. *PLOS Genet* 2020. doi:10.1371/journal.pgen.1008629
78. FDA: "FDA Approves Treatment for Serious Liver Disease Known as MASH" (Wegovy), August 15, 2025. fda.gov
79. Novo Nordisk press release: Wegovy approved for MASH, August 15, 2025.
80. Novo Nordisk company announcement (ESSENCE data), August 15, 2025.
85. Pliant Therapeutics press release: PLN-1474 Phase 1 completion and transfer to Novartis, March 16, 2021.
86. Pliant Therapeutics 8-K: Novartis termination of the collaboration (PLN-1474), February 17, 2023.
87. Fierce Biotech: "Novartis punts Pliant-partnered NASH prospect…", February 24, 2023.
88. AdisInsight drug profile: PLN-1474 (structure added Aug 2023; no development reported as of May 2025).
89. PatSnap Synapse drug profile: PLN-1474 (discontinued).
90. Endpoints News: "Novartis retreat from NASH takes down $80M alliance with Pliant", February 24, 2023.
98. Lancaster L et al. Bexotegrast in IPF: INTEGRIS-IPF trial. *Am J Respir Crit Care Med* 2024. doi:10.1164/rccm.202403-0636oc
99. Wuyts W et al. Long-term safety/antifibrotic activity of bexotegrast (ERS abstract). *Eur Respir J* 2023. doi:10.1183/13993003.congress-2023.oa1423
101. INTEGRIS-IPF full text, PMC11351797.
102. Wang F, Zhu M, Luo F. INTEGRIS-IPF: A New Hope for Tomorrow (editorial; BG00011 and GSK3008348 context). *Am J Respir Crit Care Med* 2024. doi:10.1164/rccm.202407-1295ed
103. Henderson NC et al. Targeting of αv integrin identifies a core molecular pathway that regulates fibrosis in several organs. *Nat Med* 2013. doi:10.1038/nm.3282
107. King EA, Davis JW, Degner JF. Are drug targets with genetic support twice as likely to be approved? *PLOS Genet* 2019. doi:10.1101/513945
110. Nelson MR et al. The support of human genetic evidence for approved drug indications. *Nat Genet* 2015. doi:10.1038/ng.3314
111. Sabat M et al. Design and Discovery of a Potent and Selective Inhibitor of Integrin αvβ1. *J Med Chem* 2024;67:10306–10320. doi:10.1021/acs.jmedchem.4c00743
112. RCSB PDB 8W30: crystal structure of the αvβ1 integrin headpiece with TR01225179. doi:10.2210/pdb8W30/pdb

*동반 파일: `avb1_10day_blueprint.md`(실행 계획 + REINVENT4 프로토콜), `reinvent4_avb1_scoring_config_sketch.toml`(주석 달린 config), `avb1_pocket_3d.png`, `avb1_pocket_contacts.png/.svg`, `avb1_8w30_contacts.csv`.*
