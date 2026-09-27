# 10일 Blueprint: REINVENT4를 이용한 Integrin αvβ1 저해제의 De Novo 설계

**프로젝트:** 근거 기반 타깃 우선순위 선정 → 간섬유화 치료를 위한 in silico 저해제 설계
**타깃:** integrin αvβ1 (hepatic stellate cell 내재적, 직접 항섬유화 타깃)
**기준 구조:** PDB 8W30 — TR01225179와 복합체를 이룬 αvβ1 headpiece [112]
**동반 파일:** `avb1_target_decision_brief.md`(논증 구조 + 공격 표면), `reinvent4_avb1_scoring_config_sketch.toml`(주석 달린, 문법 검증 완료 config), `avb1_pocket_3d.png`, `avb1_pocket_contacts.png/.svg`, `avb1_8w30_contacts.csv`
**큐레이션된 데이터(§3.1, 실행 완료):** `scripts/curate_actives.py` → `data/actives_core.smi`, `data/actives_core_B.smi`(+ `_train`/`_valid` 분할), `data/actives_extended.smi`, `data/similarity_refs.smi`, `data/benchmark_panel.smi`, `data/actives_annotated.csv`, `data/CURATION_LOG.md`
**계산 환경(D1, 실행 완료):** `TRANSFER.md`, `scripts/kbds_probe.sh`, `scripts/setup_kbds.sh`, `env/`, `slurm/` — REINVENT4 **v4.5.11** 고정, torch **2.5.1+cu118**, conda env **`reinvent`**

---

## 0. 설계 가설 (한 문장)

αvβ1 **결합** pharmacophore — MIDAS-metal을 배위하는 carboxylate, β1-Asn224에 대한 H-bond donor, αv-Tyr178에 맞서는 aromatic 요소 — 는 공개된 azabenzimidazolone series보다 **더 drug-like한 신규 scaffold**로도 구현될 수 있다. 그 series의 rat 경구 생체이용률 1.3%와 series 전반의 MDCK permeability < 0.1×10⁻⁶ cm/s는 **타깃 자체의 한계가 아니라 그 series에 국한된 한계**다 [111].

아래 모든 내용은 그 한 문장을 검증하기 위해 설계되었으며, 가설이 실패했을 때 그 사실을 정직하게 말할 수 있도록 구성되었다.

---

## 1. Pocket이 요구하는 것 — 구조를 scoring function의 언어로 번역하기

8W30 좌표 분석에서 도출(`avb1_8w30_contacts.csv` 참조):

| # | Pocket 특성 | 8W30에서의 기하 구조 | 생성/scoring에 대한 함의 |
|---|---|---|---|
| 1 | **Carboxylate → MIDAS Ca²⁺ (β1 Ca501)** | 2.62 Å, monodentate (두 번째 carboxylate O는 4.54 Å) | **타협 불가 anchor.** 알려진 모든 αvβ1/αv-class 저해제는 carboxylate다(RGD mimetic). → RL에서 `MatchingSubstructure`로 carboxylic acid 지정; docking 후 필터: carboxylate O → Ca501 ≤ 3.2 Å |
| 2 | **리간드 amide NH → β1-Asn224 backbone O** | 2.63 Å H-bond. **Asn224는 β3에 보존** — 결합 요소이며 선택성 요소가 아니다(§8.4) | 강력한 2차 anchor → docking 후 H-bond 필터(donor가 Asn224 O로부터 3.5 Å 이내); RL에서는 active와의 유사도를 통해 간접적으로 반영(모든 active가 이 contact를 형성) |
| 3 | **αv-Tyr178과의 aromatic contact** | 최근접 원자 3.70 Å, T-shaped/offset (centroid 6.21 Å, ring 각도 74.6°) | aromatic ring 1개 이상 필요; 목적함수로는 직접 강제하지 않고 §8.5 docking 기하 필터와 정성 분석에서 확인 |
| 4 | **β1 hydrophobic subpocket** (Pro186 3.16 Å, Tyr133 3.17 Å, Cys187 3.88 Å, Lys182 3.82 Å, Leu225 2.95 Å). **이 중 β3와 다른 것은 Leu225 하나**(β3는 Arg) — 선택성이 있다면 여기서 온다(§8.4) | hit의 dichlorophenyl이 여기에 위치 | 소수성 부피 수용 가능 → SlogP 약 3까지는 **확실히 허용된다**; potency가 이곳에서 결정됨. **주의:** 이 행이 §5의 `SlogP` 페널티 기준값 "약 3"의 출처였으나, 이는 접촉 거리에서 얻은 정성적 *허용 하한*("적어도 3까지는 괜찮다")이며 "3을 넘으면 벌점"을 뜻하지 않는다. 이 오독은 §5.4에서 정정하고 `SlogP`를 목적함수에서 제거했다 |
| 5 | **αv-Asp218 — 8W30 hit는 관여하지 않음** (7.0 Å) | 공개 series에서 cpd 25의 benzimidazolone NH가 관여 [111]. **닿는 방식이 문제다** — 염기성 Arg 모방체의 염다리는 pan-αv와 연관되고(GSK3008348의 1,8-나프티리딘 [114]), 중성 단일좌 H-bond는 그렇지 않다. αv 잔기이므로 αvβx 사이의 선택성은 이것으로 설명되지 않는다(§8.4) | Growth vector. substructure 필터로는 **의도적으로 인코딩하지 않음**(novelty를 과도하게 제약하므로); 상위 docking pose의 수동 검토에 사용 |
| 6 | **계면 water HOH2107** | 리간드로부터 4.68 Å; αv-Glu121(2.6 Å) / β1-Ser177(3.0 Å)을 연결 | water 매개 상호작용 또는 water 치환을 통한 이득 가능성; 사후 검토 용도로만 사용 |
| — | MIDAS-loop 잔기 Ser132/Ser134/Glu229 | carboxylate로부터 3.0–3.3 Å | metal site의 맥락 정보; 직접적 제약은 없음 |

**여기서 생기는 permeability 문제:** carboxylate는 pocket에 의해 고정되어 있으므로, permeability는 분자의 *나머지 부분*에서 되찾아야 한다 — TPSA 예산, lipophilic efficiency, rotatable bond, MW. 이것이 바로 Step 2 RL이 최적화하는 대상이며(5장 — 단일 stage에서 TPSA 축으로 대표시키고, 나머지 축은 §8.2 triage 창으로 넓게 본다), MIDAS-carboxylate와 permeability 사이의 긴장을 정직한 반증 위험으로 사전 등록해 둔 이유다(9장).

---

## 2. 모드 선택 — 어떤 REINVENT4 run type을 왜 쓰는가

REINVENT4 [39, 40]는 run type으로 `sampling`, `scoring`, `transfer_learning`, `staged_learning`, `enumeration`을, generator로 Reinvent(de novo), Mol2Mol, LibInvent, LinkInvent, Pepinvent를 제공한다.

| 모드 / generator | 역할 | 판정 |
|---|---|---|
| Reinvent de novo + `transfer_learning` | 큐레이션된 αvβ1 active로 prior를 집중시킴 | **주요 — Step 1** |
| Reinvent de novo + `staged_learning` | 다중 파라미터 목적함수에 대한 RL | **주요 — Step 2** |
| `sampling` | 최적 agent에서 최종 라이브러리 추출 | **주요 — Step 3** |
| `scoring` | 벤치마크 패널을 동일한 funnel에 통과시킴 | **주요 — Step 4** |
| Mol2Mol | cpd 25 주변의 근접 analog arm | **부차적 / 선택 사항** — 구조상 similarity에 제약되므로 scaffold novelty를 만들 수 없음; "cpd 25의 물성을 국소적으로 개선"하는 fallback으로만 유용 |
| LibInvent | **고정된** scaffold를 장식 | **기각** — 가설 자체가 scaffold novelty인데, azabenzimidazolone core를 고정하면 프로젝트가 벗어나려는 답을 미리 전제하게 됨 |
| LinkInvent | 두 fragment를 연결 | **기각** — 연결할 fragment 쌍이 없음; 단일 연속 pocket |
| Pepinvent | 펩타이드 | **기각** — 저분자 프로젝트 |
| `enumeration` | 반응 기반 enumeration | **기각** — 생성형이 아니며 novelty를 만들지 못함 |

**왜 de novo Reinvent만이 방어 가능한 주 전략인가:** 선행 연구가 남긴 공백은 *공개되어 있고, 선택적이며, oral-like한* αvβ1 chemotype의 부재다 — CWHM-12는 비선택적이고 [103], cpd 25는 선택적이지만 drug-like하지 않으며(oral F 1.3%) [111], PLN-1474는 drug-like하지만 비공개이고 개발이 중단되었으며 [88], bexotegrast는 폐질환을 위해 개발된 αvβ6/αvβ1 이중 저해제다 [98]. Scaffold novelty가 **곧** 산출물이며, 이를 만들어낼 수 있는 것은 de-novo generator와 scaffold diversity filter의 조합뿐이다.

---

## 3. 데이터 준비 (D1–D2)

### 3.1 Actives sets — 실행 완료, `data/CURATION_LOG.md` 참조

`scripts/curate_actives.py`가 ChEMBL_37을 대상으로 생성한다. 모든 출처 ID는 가정이 아니라 검증된 값이다.

**출처.** ChEMBL 문헌 `CHEMBL5500400` = Sabat 2024 [111](doi 10.1021/acs.jmedchem.4c00743), 26개 분자 — 논문의 화합물 번호는 25까지에 외부 비교군 C8을 더한 것으로 끝나므로 ChEMBL 커버리지는 **완전하며 PDF 전사가 필요 없다**. ChEMBL 타깃 `CHEMBL2111407` = Integrin αvβ1, 활성 413건 / 분자 253개. RCSB CCD `A1AFA` = 8W30 리간드 TR01225179, *N*-(2,4-dichlorobenzoyl)-L-phenylalanine — ChEMBL 사본(`CHEMBL5556476`)은 입체중심을 잃었으므로 CCD 정의를 사용한다. 툴 화합물은 PubChem.

**아래 모든 것을 좌우하는 화학형 구분.** 각 분자에 Arg 모방 염기성 헤드(guanidine/cyclic amidine, tetrahydronaphthyridine, 2-aminopyridine, benzimidazol-2-amine, 2-aminoazole, 지방족 아민) 보유 여부를 태깅한다. 보유한 분자는 양쪽성 이온 RGD 모드로 결합하며, 8W30 리간드와 Sabat series는 어느 것도 갖지 않는다.

| | n | Murcko scaffold | RGD 양쪽성 이온 | MW | TPSA | QED |
|---|---|---|---|---|---|---|
| ChEMBL αvβ1 actives ≤ 1 µM 전체 | 197 | 103 | 88% | 498 | 147 | 0.38 |
| **비-RGD만** | **24** | — | 0% | — | ~146 | — |
| 8W30 리간드 (A1AFA) | 1 | — | 0% | 338 | 66 | 0.88 |

**제약 조건: 이 화학형은 규모를 키울 수 없다.** 효력 컷을 풀어도 소용없다 — 비-RGD 분자는 ≤ 1 µM에서 35개, ≤ 10 µM에서 36개, 컷 없이도 43개이며, 그중 11개는 이후 RGD로 재분류되었고 4개는 큐레이션 오류로 판명되었다(아래). 비-RGD 화학형과 투과성 친화적 범위(TPSA ≤ 140, MW ≤ 550, QED ≥ 0.3)를 **동시에** 요구하면 공개 기록 전체에서 **n = 11**만 남는다. 이 거의 배타적인 관계가 §0이 겨냥한 공백이 실재한다는 가장 강력한 증거이며, TL 세트가 선택이 아니라 구조적으로 작을 수밖에 없는 이유다.

**ChEMBL 데이터 품질 문제 2건, 모두 조치함.** 문헌 `CHEMBL1629507`이 기여한 4개 분자의 어세이 설명은 *"Inhibition of human integr**ase** alphaVbeta1 … in african green monkey Vero E6 cells"* — 실제 논문은 *Small molecule inhibitors of hantavirus infection*이고 어떤 레코드도 정량값을 갖지 않는다. αvβ1 active가 아니다. 별도로 Sabat 분자 3개가 IC50 = 10000 nM 정확히(검열된 "> 10 µM")에 있으며 SAR의 음성 예시다. 모두 제외했고 큐레이션 로그에 기록되어 있다.

**분자 수가 둘인데 둘 다 맞다.** 큐레이션은 isomeric InChIKey로 중복을 제거하고, `.smi` 파일은 **canonical** SMILES로 한 번 더 제거한다 — prior가 입체이성질체를 구별하지 못하기 때문이다. core_B에서 7쌍이 이렇게 합쳐지므로 core_B는 **큐레이션 200분자 → 생성기가 구별 가능한 193개**, extended는 **197 → 190**이다. 아래 표의 수치는 `.smi` 기준이며, `actives_annotated.csv`에는 isomeric 레코드 200건이 모두 남아 있다.

**입체화학: `.smi` 파일은 canonical이며, 이건 사소한 문제가 아니다.** `REINVENT4/priors/reinvent.prior`에서 직접 읽어낸 de novo prior의 vocabulary는 33개 토큰이고 **입체화학이 전혀 없다** — `@`, `@@`, `/`, `\` 모두 부재:

```
# %10 ( ) - 0 1 2 3 4 5 6 7 8 9 = Br C Cl F N O S [N+] [N-] [O-] [S+] [n+] [nH] c n o s
```

REINVENT4는 vocabulary 밖 SMILES를 **조용히** 걸러내므로, isomeric 입력을 transfer learning에 넣었다면 아무 에러 없이 세트가 줄어든 채로 돌아갔을 것이다:

| 세트 | n | isomeric 통과 | 평탄화 통과 |
|---|---|---|---|
| core (TL-A) | 29 | **2** | **29** |
| core_B (TL-B) | 193 | 30 | 193 |
| extended | 190 | 30 | 190 |
| similarity_refs | 6 | 1 | 6 |
| 벤치마크 패널 | 6 | **0** | 6 |

따라서 모든 `.smi` 파일은 평탄화된 SMILES를 담으며, `curate_actives.py`가 prior의 실제 어휘를 게이트로 검사해 이 문제가 재발하지 않게 한다. 8W30 리간드의 L 배열을 포함한 isomeric 형태는 `actives_annotated.csv`의 `smiles` 열에 보존되며, docking 단계(§8.5)가 사용하는 것이 이쪽이다.

**생성된 세트**(표준화: RDKit cleanup → 최대 fragment → 중성화 → canonical isomeric SMILES → InChIKey 중복 제거 → `.smi`용 평탄화):

| 파일 | n | 역할 |
|---|---|---|
| `data/actives_core.smi` | **29** | **TL-A 입력 + inception.** 비-RGD 25 + RGD 툴 화합물 4 |
| `data/actives_core_B.smi` | **193** | **TL-B 입력.** TL-A + 확인된 RGD 양쪽성 이온 active 전체 |
| `data/actives_extended.smi` | 190 | **novelty NN-Tanimoto 기준선 전용** — TL 아님, QSAR 아님 |
| `data/similarity_refs.smi` | 6 | scoring endpoint 1개당 1분자 (§5 참조) |
| `data/benchmark_panel.smi` | 6 | §3.4 양성 |
| `data/folds/actives_core_fold{1..5}_{train,valid}.smi` | fold당 29 | TL-A **진단용** fold, 스캐폴드 분리 |
| `data/folds/actives_core_B_fold{1..5}_{train,valid}.smi` | fold당 193 | TL-B **진단용** fold, 스캐폴드 분리 |
| `data/actives_annotated.csv` | 254행 | 전체 후보: isomeric `smiles`와 `smiles_flat` 모두 + 멤버십 플래그, 효력, 화학형, 물성 |

RGD 툴 화합물 4개는 potency 신호를 위해 의도적으로 코어에 포함했다. `chemotype` 열이 D3에서 이들의 기여를 분리 확인할 수 있게 한다.

**QC 관문 3개(모두 통과).** 내보낸 모든 SMILES가 prior 어휘로 표현 가능하다. 모든 코어 분자가 carboxylic acid를 갖는다. Arg 모방 패턴이 놓친 양성자화 가능 질소를 가진 코어 분자는 없다 — 초기 SMARTS 누락으로 2-aminoimidazole RGD 모방체 11개가 비-RGD로 잘못 분류된 적이 있어 이 두 번째 관문을 둔다.

**C8은 의도적으로 제외했다.** Sabat 화합물이 아니라 논문이 인용한 외부 저해제이며, PubChem에서 이름으로 해석되지 않는다. 가장 가까운 ChEMBL 후보(`CHEMBL3957812`, pIC50 7.9, N-arylsulfonyl-L-proline scaffold)는 숫자 하나가 우연히 일치할 뿐이므로 추측해서 패널에 넣지 않고 로그에만 기록했다.

### 3.2 Counter-screen actives (선택성 deselection)
- Integrin **αvβ3**와 **α5β1**에 대한 ChEMBL active(IC50/Ki ≤ 1 µM)를 수집한다.
- **ChEMBL target ID는 ChEMBL_37 기준으로 검증 완료**(대괄호는 IC50 레코드 수): αvβ1 `CHEMBL2111407` [314], αvβ3 `CHEMBL1907598` [2158], α5β1 `CHEMBL2095226` [766], αvβ6 `CHEMBL2111416` [1361], αvβ8 `CHEMBL3430892` [276]. Sabat 문헌 하나만으로도 αvβ3/α5β1/αvβ6/αvβ8/αvβ5/α4β1 교차 IC50이 확보되므로, 주 series의 선택성 그림은 §3.1과 함께 딸려온다.
- §3.1 감사에서 나온 주의사항: **αvβ1 선택성은 RGD 양쪽성 이온을 포기해야 얻어지는 것이 아니다.** 문헌 `CHEMBL3862028`(*N-Arylsulfonyl-L-proline … Potent and Selective αvβ1 Inhibitors*, ACS Med Chem Lett 2016)이 기여한 36개 분자는 선택적이고 sub-nM이면서 전부 Arg 모방체를 갖는다. 선택성과 화학형은 독립적인 축이며, 비-RGD 코어를 고른 이유는 선택성이 아니라 **투과성**이다. 포스터에서 이 지점을 공격받을 것으로 예상할 것.
- ~~사후 활용: counter-screen active와 ECFP4 Tanimoto ≥ 0.5인 생성 분자에 플래그를 붙인다.~~ → **이 조항은 철회했다(2026-09-27).** 두 가지 이유다. 첫째, **이 세트는 산출된 적이 없다** — `data/`에 αvβ3/α5β1 active 파일이 없고 §10의 D2가 "모든 데이터 고정"으로 표시돼 있지만 이 항목은 빠졌다. 둘째, 위 주의사항이 스스로 적은 대로 유사도는 활성이 아니며, 선택성과 화학형은 독립적인 축이다. §8.4가 이를 구조적 관찰 3항목으로 대체하고 §9.4에서 선택성을 기준에서 제외한다.

### 3.3 Potency 모델 (`ChemProp2`) — **NO-GO, 큐레이션 시점에 확정**

100개 이상이라는 조건은 어떤 형태로도 방어 가능하게 충족되지 않는다:

- **어세이 이질성.** αvβ1 활성 413건이 **35종의 서로 다른 어세이**에 흩어져 있다 — 세포 접착, solid-phase receptor assay, ELISA-TMB, 형광편광 치환, HEK293 결합 — 그리고 서로 다른 양이온 조건에서 수행되었다. 인테그린 효력은 Mn²⁺/Mg²⁺ 활성화에 강하게 의존하므로, 통합된 pIC50 분산의 상당 부분은 화학이 아니라 어세이에서 온다.
- **충분히 큰 일관된 subset이 없다.** 최대 단일 어세이가 **n = 36**(`CHEMBL3863781`, "unknown origin" 세포 접착, 6.7 log 범위)이고 그다음이 30, 27, 26, 25다. 100에 근접하는 것이 없다.
- **분류 모델용 음성 클래스도 없다.** > 1 µM로 측정된 분자가 32개뿐이고 검열된(`>`) 관계를 가진 것은 10개다.

**이 판정은 계산 플랫폼과 무관하다.** K-BDS의 glibc 제약(§11)은 별개로 `chemprop` **패키지**를 설치 목록에서 제외시켰지만, 그것은 나중의 무관한 결정이다. ChemProp2가 배제되는 이유는 어세이 기록 자체이며 어느 머신에서든 동일하다 — 발표에서 공격받을 때 이 둘이 뒤섞이지 않도록 할 것.

**결과:** potency proxy는 `TanimotoSimilarity`이며, 따라서 §5의 endpoint 처리가 선택이 아니라 필수 요소가 된다. ChEMBL 수집의 실질적 이득은 QSAR 모델이 아니라 **novelty 기준선을 ~30개가 아닌 197개 active 위에서 계산한다는 점**이고, 이것이 §8.3의 scaffold novelty 주장을 실질적으로 강화한다.

### 3.4 벤치마크 패널 (D2에 고정)
- **양성:** `data/benchmark_panel.smi` — 8W30 리간드 A1AFA, 가장 강력한 비-RGD Sabat 화합물(cpd 25 계열), PLN-1474, bexotegrast, CWHM-12, GLPG0187. C8은 제외했다(§3.1 참조).
- **음성 패널은 폐기한다(2026-09-26).** 목적함수가 카르복실레이트를 게이트로 요구하므로(§5.1) 카르복실산이 없는 분자는 총점 상한이 6.31e-04다. 무작위 drug-like ChEMBL 분자는 대부분 카르복실산이 없으니 음성 패널은 그 바닥에 몰리고, "생성 집합이 음성 패널을 능가한다"는 **"생성 분자에 카르복실산이 있다"를 다시 재는 것**이 된다 — 목적함수가 강제하고 §5.5가 통과율로 이미 보고하는 사실이다. 정보량이 없으므로 만들지 않는다.
- 의미 있는 비교는 양성 패널이며, **순환하지 않는 축**에서만 한다: §8.5 docking 기하, §9.2 ADMET-AI permeability, Murcko 신규성(§8.3), alert/counter-screen. **총점으로 비교하지 않는다** — 총점은 우리가 설계한 목적함수의 값이므로 순환이다(§5.4).
- 이 패널은 생성 분자와 **완전히 동일한** scoring + triage funnel을 통과한다. 포스터의 모든 비교가 정직해지는 근거가 바로 이것이다.

---

## 4. Step 1 — Transfer learning: prior 집중시키기 (D3)

**근거:** 무작위 prior는 drug-like 공간 전체를 덮지만, αvβ1 저해제는 그중 아주 얇은 조각이다(필수 carboxylate + 특정 기하 구조). TL은 샘플링 밀도를 타깃 영역으로 이동시켜, RL이 carboxylate를 재발견하는 데 수백 step을 낭비하지 않게 한다.

**3개 arm, D3에서 논증이 아니라 데이터로 결정한다.** TL 1회는 GPU에서 수 분이므로 "29개로 충분한가"라는 질문은 관습이 아니라 실측으로 답한다. 참고로 REINVENT4 자체 레퍼런스 config(`configs/transfer_learning.toml`)는 100개를 쓰고 200 초과를 "large"로 부른다.

| arm | 입력 | n | RGD 양쪽성 이온 | 근거 |
|---|---|---|---|---|
| **TL-A** | `data/actives_core.smi` | 29 | 14% | 화학형 순도 우선; 가설 방향 |
| **TL-B** | `data/actives_core_B.smi` | 193 | 88% | 관습적 TL 규모이자 가장 강한 결합 신호. 다만 §0이 탈출하려는 공간으로 agent를 기울일 위험 |
| **TL-C** | 없음 — `reinvent.prior`에서 RL 직행 | — | — | TL 없는 기준선이자 §10의 전역 폴백 |

**레시피**(전체 블록은 `reinvent4_avb1_scoring_config_sketch.toml`에):
- `run_type = "transfer_learning"`, device CUDA
- Prior는 **REINVENT4 v4.5.11에 동봉**되어 있다 (`REINVENT4/priors/`, `sha512.sum` 포함 10개 파일, 전부 검증 완료) — Zenodo 다운로드 불필요. 입력 = `REINVENT4/priors/reinvent.prior`, 출력 = `focused_A.prior` / `focused_B.prior`
- `num_epochs = 10`, `batch_size = 32`, `num_refs = 0`
- **`isomeric_smiles` 키를 넣지 말 것.** v4.5.11의 TL config에는 이 필드가 없고 `GlobalConfig`가 pydantic `extra="forbid"`이므로 넣으면 시작 즉시 에러다. 넣을 이유도 없다 — prior가 입체화학을 표현하지 못하며(§3.1), 그래서 입력 `.smi`가 평탄화되어 있다.
- **`randomize_all_smiles = true`** — n = 29에서 이것은 선택이 아니다. 기본값(`randomize_smiles = true`, `randomize_all_smiles = false`)은 각 SMILES를 **파일을 읽을 때 한 번만** 무작위화하고 이후 모든 epoch에서 그 고정 문자열로 학습하므로, 증강이 전혀 없다. `randomize_all_smiles = true`로 두면 `Dataset._getitem_with_randomization`를 거쳐 접근할 때마다 새 무작위 SMILES를 생성한다 — 작은 transfer set에 대한 표준 처방이다(소스 주석: "if true much shallower minimum"). `reinvent/runmodes/TL/run_transfer_learning.py`와 `reinvent/models/reinvent/models/dataset.py`에서 확인했으며, 필드와 기본값은 v4.5.11과 `main`이 동일하다.
- `num_epochs`는 **추측하지 않는다** — 아래 진단 실행에서 결정된다.

**TL 실행에는 두 종류가 있고, 절대 섞으면 안 된다.**

| | 입력 | `validation_smiles_file` | 목적 |
|---|---|---|---|
| **진단**(arm당 5회) | `data/folds/actives_<set>_fold<i>_train.smi` | 대응하는 `_valid.smi` | 홀드아웃 NLL이 꺾이는 epoch 찾기 |
| **생산**(arm당 1회) | **전체** `data/actives_core.smi`(29) 또는 `actives_core_B.smi`(193) | **없음** | 실제로 생성할 prior. 진단에서 찾은 epoch 수로 학습 |

이전 초안은 전체 세트를 학습에 넣으면서 그중 일부를 validation으로 지정했다 — validation 분자가 학습 세트 안에 있어 홀드아웃 NLL이 암기된 데이터를 재는 꼴이 되고 아무것도 탐지하지 못한다. 두 실행을 분리할 것.

이 n에서 분자를 영구히 withhold하는 선택지는 없다: 29개 중 6개는 공개된 비-RGD 화학형 전체의 21%다. fold는 하이퍼파라미터를 고르기 위한 것이지 모델을 정의하는 것이 아니다.

**왜 단일 분할이 아니라 fold인가.** fold가 스캐폴드 분리인 이유는 무작위 분할이 누수됐기 때문이다 — TL-A는 valid의 50%, TL-B는 65%가 train과 Murcko scaffold를 공유했고, 이 데이터가 소수의 congeneric series 덩어리라서 그렇다. 그러면 홀드아웃 NLL이 일반화가 아니라 근접 유사체의 재현을 잰다. 그런데 스캐폴드 분할 **한 번**도 부족하다: scaffold 19개에 분자 29개면 valid가 ~6개인데, 분자 하나가 추정치를 17% 움직이고 결과는 어떤 scaffold가 거기 떨어졌는지에 좌우된다. **5개 fold의 평균과 범위로 보고하고, 단일 수치로는 절대 보고하지 않는다.**

TL-A의 fold는 구조상 크기가 고르지 않다 — fold 1은 9분자짜리 azabenzimidazolone 계열이며 scaffold 1개라 쪼갤 수 없다. 이 fold가 세트에서 가장 어려운 시험이고 가장 나쁜 점수가 예상된다. 결함이 아니라 정보다.

**한계는 숨기지 않고 명시한다.**
- **화학형은 층화할 수 없다.** core 전체에 RGD 양쪽성 이온이 4개뿐이라 스캐폴드 분리 fold 5개에 고르게 뿌릴 수 없다(0/1/1/0/2로 떨어진다). 효력은 퍼져 있다 — 모든 fold가 대략 pIC50 6–9를 포함한다.
- **TL-A와 TL-B의 홀드아웃 NLL을 비교하지 말 것.** 세트 크기와 scaffold 수가 달라 교란된다. arm 선택은 아래 D3 샘플링 진단으로 하며 화학형 이동이 결정적이다.
- **홀드아웃 NLL은 2차 진단이지 게이트가 아니다.** scaffold 일반화는 애초에 TL의 역할이 아니라는 반론이 타당하다 — TL은 알려진 active 근처로 확률 질량을 모으고, scaffold hopping은 RL의 diversity filter가 한다. 그 관점에서는 holdout-scaffold NLL이 멀쩡한 TL을 과소평가한다. 게이트는 샘플링 검사이고, NLL 수치는 `num_epochs`를 고르는 용도이자 "홀드아웃 active로 검증했는가"라는 질문에 답하는 용도다. 고정된 epoch 수가 아니라 홀드아웃 NLL이 상승하는 시점에 학습을 멈추기 위함이다. 이 n에서 과적합 위험은 실재하고 측정 가능하므로, 측정하지 않은 채로 두지 말 것.

### D3 결과 — 실행 완료, TL-A 채택

K-BDS가 아니라 로컬 TITAN Xp(sm_61)에서 실행했다. 두 환경은 동일하게 고정되어 있다(§11).

**arm 선택.** 각 prior에서 1,000개씩 샘플링, 손대지 않은 prior를 TL-C 기준선으로:

| | TL-C 기준선 | **TL-A (200 ep)** | TL-B (48 ep) |
|---|---|---|---|
| validity | 100.0 | 100.0 | 100.0 |
| 카르복실산 % | 7.6 | **51.1** | 62.2 |
| **Arg-모방 %** | 21.8 | **20.2 (−1.6 pp)** | **48.7 (+26.0 pp)** |
| 고유 scaffold / 1000 | 946 | 747 | 854 |
| TPSA 중앙값 | 71.7 | 103.7 | 107.0 |
| QED 중앙값 | 0.588 | 0.508 | 0.438 |

**TL-B는 사전 등록된 기준으로 탈락**이며, 독립 샘플링 3회에서 일관됐다. Arg-모방 드리프트가 학습량에 따라 커졌고(+17.5 → +26.0 pp), 88%가 RGD 양쪽성 이온인 세트에서는 필연적인 결과다. **TL-A가 이어받는다**: 카르복실산 밀도를 6.7배 올리면서 화학형을 건드리지 않았고, 이것이 §3.1에서 core를 비-RGD로 좁힌 이유 그 자체다.

**epoch은 fold 평균이 아니라 스윕에서 정했다.** fold 진단은 두 번 모두 천장에 부딪혔고(예산 20에서 10개 중 5개, 예산 50에서 10개 중 2개), 보고된 "최적 epoch"은 최소가 아니라 예산이었다. 생산 prior를 200 epoch까지 1회 학습하며 20마다 체크포인트를 남기고 각각 1,000개를 샘플링한 스윕 결과, 지표는 뒤집히지 않고 완만하게 거래된다:

| epoch | 0 | 40 | 80 | 120 | 160 | 200 |
|---|---|---|---|---|---|---|
| 카르복실산 % | 8.0 | 28.0 | 43.8 | 47.0 | 49.6 | 55.4 |
| Arg-모방 % | 21.8 | 21.6 | 18.5 | 21.3 | 22.7 | 18.5 |
| 고유 scaffold | 923 | 874 | 827 | 806 | 779 | 736 |
| QED 중앙값 | 0.599 | 0.518 | 0.519 | 0.491 | 0.513 | 0.482 |

어느 지점에서도 붕괴가 아니다 — 1,000개당 고유 scaffold 736개는 §4의 기준("actives 29개보다 훨씬 많을 것")을 크게 상회한다. 별도로 fold 하나를 200 epoch까지 돌린 프로브에서는 epoch 60에 검증 NLL 내부 최소가 나왔으나, fold 하나이고 예산 50에서의 5-fold 분산이 17–50이었으므로 이 수치만으로 하이퍼파라미터를 정할 수 없다. **생산은 200 epoch**이며, 근거는 샘플링 지표다.

**200 epoch의 정직한 대가: prior가 학습 세트의 일부를 암기한다.** `actives_extended`(190개) 기준으로 TL-A 샘플의 95.1%가 §8.3 신규성 임계 미만이지만, 4.9%가 알려진 계열 유사도 대역 안에 들어가고 1.2%는 그 p90을 넘으며, **최대 NN-Tanimoto가 1.000** — 학습 active를 그대로 재현한 샘플이 존재한다. 손대지 않은 prior의 최대값이 0.506이므로 이 꼬리는 전적으로 TL이 만든 것이다. §8.3이 하류에서 걸러내고 RL의 diversity filter가 벌점을 주지만, 신규성 주장을 할 때는 이 꼬리를 먼저 제거했다는 사실을 반드시 밝혀야 한다.

**TL은 TPSA를 올리고 QED를 낮춘다**(기준선 대비 +32.0, −0.080). 학습 actives가 TPSA ~145에 있으니 당연한 방향이며, 되돌리는 것은 TL이 아니라 Step 2 RL의 몫이다. TL이 drug-likeness를 개선했다고 말해서는 안 된다 — TL은 카르복실산 파마코포어를 심고, 물성은 RL이 만든다.

**D3 수용 기준**(각 focused prior에서 1,000개 샘플링):
- validity ≥ 95%
- carboxylic acid를 포함하는 비율이 random-prior baseline보다 뚜렷하게 높을 것
- active에 대한 평균 NN-Tanimoto가 random prior 대비 상승했을 것
- **붕괴하지 않았을 것:** 1,000개 샘플의 고유 Murcko scaffold 수가 active 개수보다 훨씬 많을 것(샘플이 active 자체와 거의 같다면 `num_epochs`를 ~5로 낮춘다)
- **화학형 이동(arm 선택 기준):** §3.1과 동일한 SMARTS로 채점한 Arg 모방 염기성 헤드 보유 비율, 그리고 TPSA 분포. TL-B는 둘 다 올릴 것으로 예상되며, 실제로 그렇다면 잘못된 영역을 prior에 주입한 것이므로 다른 수치와 무관하게 TL-A 또는 TL-C가 이어받는다.

---

## 5. Step 2 — Single-stage learning (RL): 단일 단계 구성 (D4–D5)

> **2026-09-26 재설계.** 기존 2단계 커리큘럼(Stage 1 focus 200 steps → Stage 2 optimize 400 steps)을 폐기하고 단일 stage 600 steps로 교체했다. 근거와 측정 데이터 전체는 `docs/superpowers/specs/2026-09-26-step2-single-stage-rl-design.md`에 있다.

**공통 설정:** `learning_strategy` type `"dap"`, `sigma = 128`, `rate = 0.0001`; `batch_size = 128`(GPU); `diversity_filter` = `IdenticalMurckoScaffold`, `bucket_size = 25`, `minscore = 0.4`(**전 구간 적용**); `inception`은 active로 시드("좋은 분자가 어떤 모습인지"에 대한 기억); 집계는 `geometric_mean`; `max_steps = 1000`.
* DAP (Differentiable Augmented Posterior): log-likelihood를 미분 가능한 형태로 매 스텝 다시 계산해서 역전파합니다. REINVENT 4의 기본 권장값이고, MAULI/MASCOF/SDAP 같은 다른 전략보다 수렴이 빠르고 안정적입니다. 특별한 이유 없으면 dap 유지가 맞습니다.
* sigma: 점수를 prior 확률 대비 얼마나 세게 밀어붙일지를 정하는 가중치입니다. score는 0~1로 정규화돼 있으므로, sigma=128이면 "score 1.0인 분자"는 log-likelihood 기준 prior보다 128 nat만큼 더 선호된다는 뜻입니다.
  - 낮으면(60~80): prior에 가깝게 머무름 → 합성 가능성/화학적 타당성은 좋지만 점수 개선이 느림
  - 높으면(150~256): 점수는 빨리 오르지만 exploitation이 심해져 한두 개 scaffold로 수렴(mode collapse)하거나 reward hacking(스코어 함수의 허점을 파고든 비현실적 구조)이 나옵니다
  - 128은 REINVENT 4의 표준 기본값이고, 다중 목적 함수를 섞어 쓸 때 무난한 출발점입니다
* bucket_size = 25: 동일 scaffold가 25개 모이면, 그 이후 같은 scaffold 분자는 score를 0으로 강제합니다. agent 입장에서는 "이 골격은 더 이상 보상이 없다" → 새 골격을 탐색하게 됩니다
* minscore = 0.4: score 0.4 이상인 분자만 버킷에 카운트됩니다. 쓰레기 분자까지 골격을 점유해서 유망한 scaffold를 조기에 막아버리는 걸 방지합니다.
* inception (active로 시드): Experience replay 버퍼입니다. 지금까지 나온 고득점 분자를 작은 메모리(보통 memory_size 100 내외)에 보관하고, 매 스텝 그중 일부(sample_size, 보통 10)를 뽑아 현재 배치의 loss에 섞어줍니다.
  - **seeding(smiles_file에 known active 투입)**의 의미: RL 초반에는 agent가 고득점 분자를 거의 못 만들어서 gradient 신호가 희박합니다. 알려진 활성 분자를 미리 버퍼에 넣어두면 "목표가 대략 이런 모습"이라는 positive example을 1스텝부터 학습하게 되어 수렴이 크게 빨라집니다
  - 주의점: 시드가 너무 강하면 agent가 시드 분자 주변만 맴돌아 신규성(novelty)이 떨어집니다. 결과 분자와 시드 간 Tanimoto 유사도를 꼭 확인하세요
  - 시드 분자가 실제로 scoring function에서 높은 점수를 받는지 먼저 검증해야 합니다. 시드가 저점수면 오히려 잘못된 신호를 줍니다

Diversity filter는 동일 Murcko scaffold당 몇 개의 분자까지 보상에 기여할 수 있는지를 제한하며, 이것이 scaffold hopping을 강제하는 기전이자 novelty 주장을 가능하게 하는 장치다. 선행연구 [113]은 diversity filter를 마지막 stage에만 걸어 enriched chemotype으로의 *수렴*을 유도했으나, 우리 목표는 반대 방향이므로 전 구간 적용을 유지한다.

### 5.1 목적함수 — scored endpoint 3개 + filter 1개

| 항목 | 종류 | 역할 | transform |
|---|---|---|---|
| `GroupCount` 카르복실레이트 | **scored** (w=1.0) — **게이트** | MIDAS anchor, 타협 불가 | `right_step(high=1)` → 보유 1.0 / 부재 0.0 |
| `TPSA` | **scored** (w=1.0) | 산성 리간드의 투과성 축 | `double_sigmoid(low=40, high=115, coef_div=120, coef_si=coef_se=20)` |
| `SAScore` | **scored** (w=0.5) | **guard rail — 최적화 축이 아니다**(§5.4) | `reverse_sigmoid(low=6.0, high=8.0, k=0.5)` |
| `CustomAlerts` | **filter**(0/1 마스크) | 구조적 liability | 아래 |

```toml
[[stage.scoring.component]]
[stage.scoring.component.GroupCount]
[[stage.scoring.component.GroupCount.endpoint]]
name = "carboxylate MIDAS anchor"
weight = 1.0
params.smarts = "[CX3](=O)[OX2H1,OX1-]"
transform.type = "right_step"
transform.high = 1
```

**집계 구조 — REINVENT4 소스에서 확인하고 실제 출력으로 검증했다.** `reinvent/scoring/scorer.py:159-178`은 scored component만 `aggregate`에 넣고 penalty component는 결과에 곱한다. 채택 구성은 penalty를 쓰지 않으므로

```
total = geometric_mean([(COOH, 1.0), (TPSA, 1.0), (SAScore, 0.5)]) × alert_filter
      = prod(max(scoreᵢ, 1e-8) ** (wᵢ / Σw)) × alert_filter          # aggregators/means.py
```

이 재구현을 `logs/cmp.vigHoB/new3.csv`의 실제 REINVENT 출력에 대조해 검증했다 — TPSA 2.2e-07, sigmoid 8.7e-08, **총점 1.9e-08**의 오차(filter에 걸린 19개 분자는 REINVENT가 모든 component를 0으로 덮으므로 제외).

**왜 `MatchingSubstructure`가 아니라 게이트인가 — 기존 구성은 카르복실레이트를 버리라고 가르쳤다.** `comp_matching_substructure.py:63`은 점수를 `0.5 * (1.0 + match)`로 **하드코딩**한다(부재 ×0.5, 조정 불가). 그 구성으로 prior 샘플을 채점하면:

| prior 샘플 | n | 총점 중앙값 | p90 | 최댓값 |
|---|---|---|---|---|
| 카르복실산 **있음** | 254 | **0.060** | 1.000 | 1.000 |
| 카르복실산 **없음** | 189 | **0.500** | 0.500 | 0.500 |

카르복실산이 없으면 0.500에 고정되는데 있으면 중앙값이 0.060이다 — 카르복실산이 TPSA를 약 37 올려 창 밖으로 밀어내기 때문이다. **agent의 기대보상 최적 전략이 MIDAS 카르복실레이트를 버리는 것**이었다. 이 프로젝트가 타협 불가라고 선언한 바로 그 구조다.

`CustomAlerts`로는 해결되지 않는다 — 그것은 "매치되면 0"이므로 **부재**를 표현할 수 없고, SMARTS에는 분자 수준 부정이 없다. 따라서 `GroupCount` + `right_step`을 scored endpoint로 쓴다. `geometric_mean`이 0을 1e-8로 clamp하므로 카르복실산 부재 시 총점 상한은 **6.31e-04**이고(prior 샘플 219개 전부가 그 아래) DAP σ=128 기준 만점과 **약 128 nat** 차이다 — 문자 그대로 0은 아니지만 실질적 게이트다. `w_COOH`를 올리면 바닥은 더 내려가지만(w=3 → 4.6e-06) TPSA 지수가 희석되어 gradient가 평탄해지므로 **w=1**로 둔다.

`CustomAlerts` 패턴:

```
[NX3;H2][c]            # 1차 aniline만 — 아닐리드/설폰아닐리드/THN은 제외(§5.4)
[*;r8] [*;r9] [*;r10]
N=[N+]=[N-]
C(=O)Cl
[SH]
[Nr0][Nr0]
```

PAINS는 **이 TOML로 표현할 수 없다** — `reinvent_plugins`에는 PAINS 컴포넌트가 없으므로 RL은 이 여덟 개 SMARTS만 적용한다. PAINS는 대신 `scripts/objective.py`의 오프라인 스코어러와 §8 triage에서 적용된다.

**endpoint를 3개로 제한하는 이유:** 선행연구 [113]은 docking score·QED·SAscore 3개 endpoint에 **각각 0.3의 균등 가중**을 주었다(REINVENT4가 내부적으로 정규화하므로 실효 가중은 1/3씩이다). 이것은 취향이 아니라 조종 가능성(steerability)의 조건이다. `geometric_mean` 아래에서 각 endpoint는 곱셈 veto이므로, endpoint를 늘리면 달성 가능한 천장이 무너지고 "우리 목적에 맞는 분자"가 아니라 "목적함수에만 최적화된 분자"가 나온다. 세밀한 filtering은 목적함수가 아니라 §8의 사후 triage와 정성 분석의 몫이다.

### 5.2 목적함수에서 제거한 것

| 제거 대상 | 사유 | 이동 위치 |
|---|---|---|
| **Stage 1 전체**(200 steps) | §5.3 | 삭제 |
| `TanimotoSimilarity` 참조 6개 | 구조적으로 달성 불가능(§5.3) | §5.5 진단 지표 |
| `TanimotoSimilarity` anti-target 2개 | §8.4에 이미 존재 — 중복 | §8.4 유지 |
| `SlogP` | 기준값 "3"에 근거 없음, 게다가 방향이 반대(§5.4) | §8.2 triage 창(logP ≤ 5) |
| `QED` | 임상 화합물과 무작위 prior 샘플을 구별 못 함(§5.4) | §8.7 보고 지표 |
| `MolecularWeight`, `NumRotBond`, `HBD` | QED/TPSA와 5중 중복(§5.1) | §8.2 triage 창 |

§3.3이 `ChemProp2`를 배제했고 여기서 similarity까지 제거하므로, **목적함수에 potency 항이 없다.** 이는 의도된 역할 분담이다 — potency 신호는 TL-A(카르복실산 7.6% → 51.1%, §4)와 inception이 담당하고, potency 판정은 §8.5 docking 기하 필터가 사후에 수행한다.

> **`TanimotoSimilarity`는 endpoint 1개당 참조 1개만 받는다 — 소스에서 확인하고 재현했다.** `comp_similarity.py`는 참조 SMILES 1개당 점수 배열 1개를 내보내지만(`scores.extend([... for fingerprint in fingerprints])`), `compute_scores.py`가 그 배열들을 **endpoint** 목록과 zip한다(v4.5.11은 `:107`, `main`은 `:175` — 두 버전 모두 동일한 결함). endpoint 1개에 참조 N개를 넣으면 zip이 잘라낸다 — 첫 참조만 채점되고 나머지는 **조용히 버려진다.** 참조 집합에 대해 집계하는 similarity 컴포넌트도 REINVENT4에는 없다(`reinvent/chemistry/similarity.py`의 max 집계 `calculate_tanimoto`는 Mol2Mol 샘플러 전용). **이것이 similarity를 목적함수에서 빼는 결정적 이유다** — 참조 6개를 각각 독립 endpoint로 선언하는 것이 유일한 방법이고, 그러면 §5.3의 불가능 조건에 걸린다.

### 5.3 왜 커리큘럼이 아니라 단일 stage인가

**(1) "활성 물질 근처에 머물라"는 신호가 3중으로 중복됐다.** TL-A가 prior를 활성 영역에 집중시키고(D3에서 측정), inception이 활성 물질을 RL 메모리에 replay하고, Stage 1 유사도가 200 스텝 동안 다시 활성 물질 쪽으로 당긴다. TL과 inception은 수동적이고 비용이 없다. Stage 1은 세 번째 사본이면서 유일하게 200 RL 스텝을 소모하는 사본이다.

**(2) '당겼다 미는' 시퀀스는 신규성 가설과 상충한다.** 기존 §5의 모니터링 문단이 스스로 이를 기술하고 있었다 — "Stage 2에서 similarity가 **하락**한다, 그 하락이 바로 scaffold novelty가 나타나는 신호다." Stage 1이 200 스텝을 들여 올린 값을 Stage 2가 400 스텝을 들여 되돌린다. 신규성이 목표라면 애초에 올리지 않는 것이 맞다.

**(3) 커리큘럼의 전제는 절반만 성립한다.** 커리큘럼을 정당화하는 논거는 초기 목적함수가 ill-conditioned하다는 것 — 기본 변환에서 prior 샘플 점수가 0에 가까워 학습할 gradient가 없다는 것이다. 선행연구 [113]이 정확히 그 조건이다(가중치 균등 + REINVENT4 기본 변환 + 3 stages × 30 epochs). 우리는 지난 세 번의 반복에서 모든 창을 측정된 prior 분포에 맞게 보정했다. 그 결과를 재확인했다(`logs/cmp.vigHoB/new3.csv` — `molecules.smi` 493개 중 488개가 `samples_prior.csv`와 일치하므로 사실상 TL-A prior 샘플 분포):

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
| **총점**(geometric_mean) | 0.000 | **0.011** | 0.332 | **70%** |

**물성축은 well-conditioned하다. similarity축은 전혀 아니다.** 그리고 similarity축의 ill-conditioning은 보정 실패가 아니라 **논리적 불가능**이다: `data/similarity_refs.smi`의 6개는 "서로 유사하지 않도록"(최대 쌍별 Tanimoto 0.45) 일부러 고른 것인데, 이를 각각 독립 endpoint로 걸고 `geometric_mean`으로 묶으면 "서로 닮지 않은 6개와 동시에 닮아라"를 요구하게 된다. 가중치로 해결되지 않는다.

따라서 처방은 커리큘럼이 아니라 **similarity를 목적함수에서 제거**하는 것이다. 그렇게 하면 `geometric_mean`을 유지해도 된다 — AND 문제가 발생한 것은 달성 불가능한 endpoint가 있었기 때문이고, 그것을 빼면 곱셈 집계는 원래 의도(similarity *또는* 물성만 만족시키는 회피 방지)대로 작동한다. TL이 focusing 작업을 이미 수행했으므로 커리큘럼의 전제 조건 자체가 사라졌다.

### 5.4 개별 component 판단 근거

**`TPSA` 창 = `(40, 115)`, 근거는 경구 임상 화합물 두 개다.** 창의 근거로 cpd 25 계열을 쓸 수 없다 — 그 계열의 TPSA가 높은 것은(중앙 133.3, 최대 170.0) **애초에 경구용으로 제안된 구조가 아니기 때문**이고, 따라서 "cpd 25가 창 밖에 있다"는 §0의 전제를 다시 말한 것에 불과하다(순환). 외부 증거가 되는 앵커는 경구로 실제 진전된 αv 카르복실산 두 개뿐이다:

| 화합물 | TPSA | 기록된 근거 |
|---|---|---|
| **PLN-1474** | **100.6** | 경구, αvβ1 선택적, Phase 1 무사 완료(84명, DLT 없음) [85] |
| **bexotegrast** | **112.5** | 경구 αvβ6/αvβ1 이중 저해, Phase 2a INTEGRIS [98, 99] |
| A1AFA | 66.4 | 8W30 결정 리간드 — 구조적 하한 |
| cpd 25 계열(Sabat 25개) | 중앙 133.3 / 최대 170.0 | 경구 F 1.3%, MDCK < 0.1×10⁻⁶ [111] — 경구용 제안 아님 |
| GLPG0187 / CWHM-12 | 158.7 / 172.4 | 비선택적 / 비경구 |

하한 40은 결정 리간드(66.4)보다 낮아 아무것도 벌주지 않는다. 상한 115는 **경구 임상 화합물 중 TPSA가 높은 쪽(bexotegrast 112.5)에 마진을 둔 값**이다. 후보 비교(카르복실산 게이트 적용 상태, "산 부분집합"은 카르복실산 보유 prior 샘플 254개):

| 창 | bexotegrast | PLN-1474 | A1AFA | 산 중앙값 | 산 > 0.5 | Sabat 중앙 |
|---|---|---|---|---|---|---|
| **(40, 115) — 채택** | 0.878 | 0.998 | 1.000 | **0.178** | 43% | **0.060** |
| (40, 120) | 0.978 | 1.000 | 1.000 | 0.372 | 48% | 0.130 |
| (45, 125) | 0.997 | 1.000 | 1.000 | 0.661 | 52% | 0.276 |
| (50, 130) | 1.000 | 1.000 | 0.999 | 0.910 | 59% | 0.546 |
| (60, 140) — 기각 | 1.000 | 1.000 | 0.968 | 0.976 | 63% | **0.968** |

`(40, 115)`에서 산 중앙값이 0.178로 낮은 것은 **의도된 방향**이다 — TL이 남긴 산은 TPSA가 높고(보유 샘플 p25 102 / 중앙 128 / p75 158) RL의 일이 그것을 끌어내리는 것이기 때문이다. 동시에 **33%가 0.95를 넘고 43%가 0.5를 넘으므로** 0 step에서 모방할 양성 신호가 충분하다 — 소멸도 포화도 아니다. 창을 넓히면 산 중앙값은 올라가지만 벗어나려는 Sabat 계열이 함께 보상받는다: `(50, 130)`에서 이미 0.546, `(60, 140)`에서 0.968이다. bexotegrast를 "허용하되 경계"(0.878)로 두는 것은 폐 대상 이중 저해제가 αvβ1 간 화합물의 최적점은 아니라는 점에서 정직한 처우다. Sabat 계열이 창 밖에 떨어지는 것은 근거가 아니라 결과다.

> **계산 정정(2026-09-27).** 이 표의 초기 버전은 부분집합 필터에 `총점 > 1e-3` 조건이 들어가 낮은 점수를 제외했고, 그 결과 산 중앙값을 (40,115)에서 0.647로 **과대 보고**했다. 위 수치가 정정된 값이다. 창 선택은 바뀌지 않는다 — 결정 근거는 경구 앵커 두 개와 Sabat 계열 배제이지 중앙값이 아니다.

**이것이 포스터 문구를 제약한다.** "생성 분자가 cpd 25보다 TPSA가 낮다"는 구성상 보장된 결과이므로 발견으로 제시할 수 없다. 투과성 주장은 §9.2(ADMET-AI, 독립 모델)와 §8.5(docking 기하)가 지탱해야 한다.

**같은 이유로 ADMET-AI를 RL 목적함수에 넣지 않는다.** 넣으면 §9.2도 순환이 된다. TPSA를 RL에, ADMET-AI를 사후 평가에 두는 분리가 투과성 주장을 비순환으로 유지하는 장치이며, 이는 §12의 "RL 내부 docking 미사용"과 같은 종류의 의도적 제약이다.

**`SlogP` 제거.** 세 파일이 서로 달랐다 — 이 문서는 "reverse_sigmoid, 약 3 초과 시 페널티", `reinvent4_avb1_scoring_config_sketch.toml`은 `double_sigmoid(low=1, high=4)`, 실제 실행된 `logs/cmp.vigHoB/new3.toml`은 `reverse_sigmoid(low=2, high=5, k=0.4)`. `reverse_sigmoid(2, 5, 0.4)`의 0.5 교차점은 실제로 **3.5**이고(logP 3.0에서 이미 0.823, 3.25에서 0.683), "약 3"은 페널티가 눈에 띄기 시작하는 지점을 가리킨 것이다. 출처는 §1 pocket 표 4행("소수성 부피 수용 가능 → SlogP 약 3까지")이지만, 그 행은 접촉 거리에서 얻은 정성적 **허용 하한**("적어도 3까지는 괜찮다")이며 "3을 넘으면 벌점"이 아니다 — 허용 범위의 하한을 페널티의 상한으로 바꿔 쓴 오독이다. low/high 값 자체는 어느 문서에서도 유도되지 않는다. 더 중요한 것은 방향이다 — 이 프로젝트는 필수 카르복실레이트를 가진 **음이온**을 만들고 §9.2는 "예측 permeability에서 cpd 25를 능가"를 요구한다. 산성 화합물의 투과성 병목은 지질친화성 과다가 아니라 이온화된 카르복실레이트이므로, logP를 3 위에서 깎으면 개선하겠다고 선언한 축을 악화시킨다. 선행연구 [113]의 준거도 soft penalty가 아니라 hard filter(MW > 500 또는 logP > 5 제외)다. → §8.2 triage 창으로 이동. **이 결정이 남긴 비용은 §6.3에서 측정했다** — logP 항이 없는 상태에서 agent의 cLogP 중앙값이 prior의 2.62에서 5.15로 드리프트해, `logP ≤ 5` 단독 통과율이 90.1% → 46.5%로 떨어졌다. 방향에 관한 위 논거는 유지되지만 드리프트의 크기는 위에서 예측하지 못한 것이다.

**`QED` 제거.** PLN-1474(αvβ1 저해제 중 유일하게 임상 착수)의 QED는 **0.4619**이고 TL-A prior 샘플의 QED 중앙값은 **0.468**이다. 유일한 임상 진입 화합물이 평범한 prior 샘플과 구별되지 않으므로 최적화 축으로 쓸 수 없다. (0.433은 비방향족 타우토머로 계산한 값이며 §8.0의 artifact다.) → §8.7 보고 지표로만 유지.

**`SAScore`는 guard rail이며, 창은 `(6, 8, k=0.5)`다.** 먼저 SA 원값 분포:

| 세트 | SA 중앙값 | SA p90 | SA 최대 |
|---|---|---|---|
| `actives_core` | 3.08 | 3.30 | 3.75 |
| `actives_extended` | 3.48 | 4.27 | **5.10** |
| `benchmark_panel` | 3.29 | 3.48 | 3.75 |
| TL-A prior (488) | 2.91 | 3.72 | **5.02** |

sketch의 `(3, 6, k=0.5)`는 **guard rail로 부적합하다.** 중앙값에서는 무해해 보이지만(모든 세트 0.98 이상) 꼬리에서 물어버린다 — `actives_extended`의 최악 분자(SA 5.10)가 **0.093**, prior의 최악 분자가 **0.121**을 받는다. 알려진 ChEMBL αvβ1 active를 벌주는 것이므로 §5.4의 aniline 필터와 같은 종류의 오류다. 창 후보별 곡선:

| 창 | SA=3.0 | SA=4.0 | SA=5.1 | SA=6.0 | SA=7.0 | SA=8.0 | 전 참조세트 최솟값 |
|---|---|---|---|---|---|---|---|
| `(3, 6, k=0.5)` — sketch | 0.997 | 0.872 | **0.091** | 0.003 | 0.000 | 0.000 | **0.093** |
| `(5, 8, k=0.4)` | 1.000 | 1.000 | 0.987 | 0.823 | 0.177 | 0.010 | 0.987 |
| `(6, 9, k=0.4)` | 1.000 | 1.000 | 0.999 | 0.990 | 0.823 | 0.177 | 0.999 |
| **`(6, 8, k=0.5)` — 채택** | 1.000 | 1.000 | **1.000** | 0.997 | 0.500 | 0.003 | **0.99998** |

`(6, 8, k=0.5)`를 채택한다. 참조 세트와 prior 샘플의 **최악 분자가 0.99998**을 받는다 — `reverse_sigmoid`는 로지스틱 곡선이라 1.0에 점근하되 도달하지 않으므로 정확히 1.000이 되지는 않는다. `actives_extended`의 최악 분자(SA 5.095)가 0.999983, prior의 최악 분자(SA 5.017)가 0.999989이고, `geometric_mean`의 0.2 지수를 거치면 총점을 **2.2e-06** 낮춘다 — 실질적으로 무기여다. 그리고 SA > 6(합성 난이도의 통상적 경계)부터 비로소 작동해 SA 7에서 0.5, SA 8에서 사실상 veto가 된다. `(6, 9, k=0.4)`는 SA 7에서도 0.823으로 너무 관용적이고, `(5, 8, k=0.4)`는 통상 경계보다 이른 SA 6에서 이미 물기 시작한다.

**이 문서는 SAScore가 합성 가능성을 최적화한다고 주장하지 않는다** — 0 step에서 0.99998 이상을 받아 실질적으로 무기여(총점을 2.2e-06 낮추는 수준)이며, 600 step RL이 prior가 한 번도 샘플링하지 않는 영역으로 드리프트할 때만 작동하는 보험이다.

**정직한 요약: 0 step 기준 살아 있는 최적화 축은 카르복실레이트 anchor와 TPSA 둘이다.** TL이 focusing을 이미 끝냈다는 전제를 받아들이면 이것이 일관된 귀결이다.

**aniline 패턴 교체 — `[NH2,NH][c]` → `[NX3;H2][c]`.** 기존 패턴을 RDKit으로 전 데이터셋에 대해 측정했다(매치 = `CustomAlerts`가 filter이므로 총점 0):

| 세트 | `[NH2,NH][c]`(기존) | `[NX3;H2][c]`(신규) |
|---|---|---|
| `actives_core` (29) | **9/29 (31%)** | 0/29 |
| `actives_extended` (190) | **164/190 (86%)** | 0/190 |
| `similarity_refs` (6) | **3/6** | 0/6 |
| `benchmark_panel` (6) | **5/6** | 0/6 |
| TL-A prior 샘플 (488) | **118/488 (24%)** | 12/488 (2.5%) |

`benchmark_panel`에서 0점이 되는 5개는 **PLN-1474, bexotegrast, CWHM-12, GLPG0187, CHEMBL4649232**이고 A1AFA만 살아남는다. 두 가지 귀결: **(a) 테트라히드로-1,8-나프티리딘(THN)이 걸린다** — 아릴에 붙은 NH가 H 1개라서 `[NH2,NH]`에 매치되며, THN은 인테그린 RGD 모방체의 표준 Arg-mimic head이므로 이 필터는 반응성 대사체를 막는 대신 이 타깃 클래스의 핵심 파마코포어를 금지하고 있었다. **(b) §7이 무효화될 상태였다** — 양성 패널 6개 중 5개가 **모든 component에서** 0점이 되면(filter는 전 항목을 덮는다) 기준 분포 자체가 소멸하고, 당시 §9.5 문구("생성 집합이 총점에서 음성 패널을 능가하고 양성 패널에 준할 것")는 생성 분자가 아무리 나빠도 통과한다. `results/`가 비어 있어 발표될 수치가 오염된 것은 없다. (§9.5는 이후 별개 사유 — 총점 비교의 순환성 — 로 비순환 축 비교로 개정했다.)

아닐린 관련 반응성 대사체 위험 자체는 유효하지만(해당 series의 이력 [111]), 필요한 것은 1차 방향족 아민에 한정된 패턴이다. `[NX3;H2;!$(N[!#6]);!$(NC=O);!$(NS(=O)=O)][c]`와 `[NX3;H2][c]`는 모든 테스트에서 결과가 **완전히 동일**하다 — `H2`가 이미 N-acyl·N-sulfonyl·N-heteroatom 치환을 배제하기 때문이다(그 경우 H가 1개 이하). 세 exclusion은 중복이므로 단순한 쪽을 채택한다. 2-아미노피리딘/2-아미노피리미딘은 `[NX3;H2][c]`에 매치되며, 헤테로고리를 제외하려면 `[NX3;H2][c;!$(c~n);!$(c~o);!$(c~s)]`를 쓸 수 있으나 차이는 prior 샘플 12개 vs 9개(0.6%p)이고 참조 세트는 양쪽 모두 0이므로 더 엄격하고 단순한 쪽을 쓴다.

나머지 alert는 정상이다 — 합집합으로 `actives_core` 0/29, `actives_extended` 1/190, `similarity_refs` 0/6, `benchmark_panel` 0/6, prior 19/488(3.9%). 이번에 바로잡은 간극: 이 문서는 "PAINS/reactive/aniline"이라고 기술해 왔는데, 실제로는 PAINS가 RL config의 패턴이 아니다 — `reinvent_plugins`에 PAINS 컴포넌트가 없어 애초에 TOML로 표현할 수 없고, 따라서 RL은 여덟 SMARTS만 적용하며 PAINS는 `scripts/objective.py`와 §8 triage에서만 적용된다. prior 샘플 측정치: 여덟 SMARTS 단독 31/488, `objective.py`의 전체 필터(여덟 SMARTS ∪ PAINS) 45/488, PAINS 단독 15/488 — `actives_core`(0/29), `similarity_refs`(0/6), `benchmark_panel`(0/6)에는 비용이 없지만 `actives_extended`(190개)에서는 3개가 걸린다 — `CHEMBL244434`, `CHEMBL244013`(둘 다 PAINS `mannich_A(296)`), `CHEMBL4756602`(`[Nr0][Nr0]`). 이 세 건이 spec의 success criterion 2와 §10 D4 체크포인트("참조 화합물 0건 확인")가 정직하게 보고해야 할 실제 비용이다.

### 5.5 모니터링 — 보상이 아닌 진단 (100 step마다)

similarity가 목적함수에서 빠지므로 기존 모니터링 기대치("Stage 2에서 similarity 하락 = novelty 신호")는 폐기하고 다음으로 대체한다:

- `actives_extended` 대비 NN-Tanimoto 분포 — **타우토머 정규화 후**(§8.0)
- 카르복실산 통과율
- 고유 Murcko scaffold 수
- alert 매치율
- SAScore 분포(guard rail이 실제로 작동하기 시작했는지)

이것들은 보상에 들어가지 않는다 — novelty 주장을 측정 가능하게 유지하면서 목적함수에서 값을 지불하지 않는 방식이다. **개입 기준:** NN-Tanimoto가 ~0.2 아래로 붕괴하고 **동시에** 카르복실산 통과율이 떨어지는 경우에만 개입한다. 둘 중 하나만으로는 개입하지 않으며, similarity 가중치를 올려 "고치려" 하지 말 것(그 component는 이제 존재하지 않는다).

---

## 6. Step 3 — Sampling (D6)

- RL 최종 agent에서 `run_type = "sampling"`(비교를 위해 D3에서 선택된 focused prior에서도 수행)
- SMILES 20,000–50,000개, 고유 분자만
- 출력: `library.smi` → 중복 제거 → canonicalize → **타우토머 정규화(§8.0)** → 물성/novelty 분포

### 6.1 실행 — 완료(2026-09-27)

두 arm 모두 20,000개를 요청했다. 20,000은 §6 범위의 하한이며, 타우토머 정규화가 분자당 약 0.14 s이므로 arm당 약 47분이고 arm이 둘이라는 비용에서 정했다.

| 산출물 | 내용 |
|---|---|
| `configs/sample_agent.toml` | `results/rl_final.chkpt`(1000-step 단일 stage agent), `num_smiles = 20000` |
| `configs/sample_prior.toml` | `priors/focused_A.prior`(TL-A) — **기준선 arm** |
| `data/library.smi` | agent arm, 19,485개. §9가 이름으로 사전 등록한 산출물이 이것이다 |
| `data/library_prior.smi` | prior arm, 18,346개 |
| `scripts/build_library.py` | CSV → `.smi`. 검증 → canonical 중복 제거 → 타우토머 정규화 → 재중복 제거 |
| `scripts/library_report.py` | 서술 분포만. **게이트 없음** — cut은 §8에 있다 |

`unique_molecules = true`가 이미 canonical 수준에서 중복을 제거하므로 `build_library.py`의 첫 중복 제거는 회계에 가깝고, 실제로 일하는 것은 정규화 **후** 두 번째 중복 제거다. 타우토머 두 개는 서로 다른 canonical SMILES이므로 그것 없이는 두 분자로 세어진다.

`library.smi`는 **두 열**을 담는다. 1열이 정규화된 SMILES(하류의 모든 물성·신규성 주장의 대상), 2열이 정규화 전 canonical SMILES(**RL 목적함수가 실제로 채점한 구조**)다. 타우토머 선택이 descriptor를 움직이므로 두 구조가 불일치할 수 있고, 그 크기를 측정 가능하게 남겨 두는 것이 두 열을 유지하는 이유다.

### 6.2 분포 — agent vs prior

| | agent(19,485) | prior(18,346) |
|---|---|---|
| MW 중앙 | 431.4 | 417.4 |
| **cLogP 중앙** | **5.15** | **2.62** |
| TPSA 중앙 | 74.3 | 102.4 |
| QED 중앙 | 0.44 | 0.50 |
| SAScore 중앙 | 2.73 | 2.85 |
| **카르복실산 %** | **96.7** | **51.5** |
| alert-free % | 96.9 | 91.7 |
| 목적함수 총점 중앙 | 1.00 | 0.00 |
| **Murcko-novel %** | **99.7** | 87.5 |
| 고유 scaffold | 17,268 | 11,875 |
| acyclic | 3 | 254 |
| NN-Tanimoto 중앙 / p90 | 0.34 / **0.39** | 0.35 / **0.58** |

prior의 Murcko-novel 87.5%는 §8.3이 488개 표본에서 측정한 86.1%를 18,346개에서 재현한 값이다.

NN-Tanimoto의 **중앙값은 두 arm이 같고 p90만 갈린다**(0.39 vs 0.58). RL이 actives 전체에서 멀어진 것이 아니라 §4가 기록한 TL-A의 암기 꼬리(최대 1.000)를 잘라낸 것이다. 신규성 개선은 분포의 이동이 아니라 꼬리의 제거로 서술해야 한다.

### 6.3 §8.2 사전 등록 창을 적용하면

§8.2의 창(`logP ≤ 5`, MW 250–550, TPSA 40–115)에 §8.1의 alert와 카르복실산 hard cut을 값싼 것부터 누적 적용한 결과다. **§8의 triage를 미리 실행한 것이 아니라, §6의 분포가 사전 등록된 창을 통과하는지를 확인한 것이다.**

| 누적 단계 | agent | prior |
|---|---|---|
| 전체 | 19,485 (100%) | 18,346 (100%) |
| + 카르복실산 | 18,845 (96.7%) | 9,454 (51.5%) |
| + alert-free | 18,273 (93.8%) | 8,920 (48.6%) |
| + **logP ≤ 5** | **8,430 (43.3%)** | 8,253 (45.0%) |
| + MW 250–550 | 8,162 (41.9%) | 6,634 (36.2%) |
| + TPSA 40–115 | 7,787 (40.0%) | 2,716 (14.8%) |
| + Murcko-novel | **7,767 (39.9%)** | **2,125 (11.6%)** |
| 생존 고유 scaffold | **6,849** | 1,446 |

단독 통과율로 보면 `logP ≤ 5`가 agent 46.5% / prior 90.1%, `TPSA 40–115`가 agent 95.5% / prior 55.7%다.

**RL이 한 일:** 생존율 11.6% → 39.9%, 생존 scaffold 1,446 → 6,849(4.7배). TPSA 창 통과가 55.7% → 95.5%, 카르복실산이 51.5% → 96.7%로 오른 것이 그 대부분이다. 목적함수가 겨냥한 두 축에서 정확히 이득이 났다.

**RL이 지불한 대가 — logP.** prior의 90.1%가 `logP ≤ 5`를 통과하는데 agent는 46.5%뿐이고, 누적 funnel에서 93.8% → 43.3%로 떨어지는 이 단계가 agent arm의 **유일한 지배적 손실**이다. 목적함수에 logP 항이 없으므로 agent가 TPSA 창(40–115) 안에 머무르면서 지질친화적 부피로 드리프트했다 — TPSA 중앙 74.3은 창 한가운데이지만 cLogP 중앙은 5.15다.

§5.4가 `SlogP`를 제거한 논거(필수 카르복실레이트를 가진 음이온의 투과성 병목은 지질친화성 과다가 아니라 이온화된 카르복실레이트이므로 logP를 3 위에서 깎으면 개선하겠다고 선언한 축을 악화시킨다)는 **방향에 관해서는 유지된다**. 그 논거가 예측하지 못한 것은 항을 완전히 제거했을 때 드리프트가 어디까지 가는지였고, 답은 중앙값 2.62 → 5.15다. 이것은 §5의 설계 결정이 남긴 측정된 비용이며, 사후에 숨기지 않고 여기에 기록한다.

**그럼에도 RL 재실행은 하지 않는다.** §8의 목표는 lead 10–20개이고 생존자가 7,767개, 생존 고유 scaffold가 6,849개다. logP 손실은 산출량을 위협하지 않는다. 대신 **포스터 문구를 제약한다**: "druggable"을 주장할 때 기준은 생존자 집합(logP 중앙 **3.97**, MW 중앙 379.4, TPSA 중앙 77.8)이어야 하고, library 전체(logP 중앙 5.15)를 근거로 삼으면 안 된다. §9.2의 permeability 비교도 생존자 집합에서 수행한다.

### 6.4 정규화가 실제로 한 일

타우토머 중복은 agent **0건**, prior **1건**이었고 정규화 실패는 양쪽 0건이다. 즉 **정규화는 중복 제거로는 아무 일도 하지 않았다.** 이것이 §8.0의 논거를 약화시키지는 않는다 — §8.0이 주장한 것은 중복 제거가 아니라 참조 세트와 측정 세트가 같은 정규화를 통과해야 한다는 비교 대칭성이고, PLN-1474의 두 타우토머가 서로 다른 Murcko SMILES를 낸다는 §8.3의 검증이 그 근거다. 다만 **"정규화가 중복을 걸러낸다"는 서술은 이 측정으로 지지되지 않으므로 쓰지 않는다.**

1열과 2열의 불일치: 19,485개 중 19,179개가 동일하고, 카르복실산 판정이 갈리는 것이 **4건**, alert 판정이 갈리는 것이 **31건**이다(prior는 9건 / 347건). 총점 변화량은 p90까지 0.000이므로 **백분위수만 보면 0으로 보이고**, 효과는 전적으로 꼬리에 있다. `library_report.py`가 백분위수와 함께 `n_shifted`와 `max_shift`를 보고하는 이유가 이것이다.

## 7. Step 4 — 벤치마크 패널 scoring (D6, sampling 이후)

- 고정된 벤치마크 패널에 **§5.1과 동일한 scoring function**을 적용해 `run_type = "scoring"` 실행
- **선행 조건:** `CustomAlerts`가 §5.4의 교체된 aniline 패턴(`[NX3;H2][c]`)을 쓰고 있어야 한다. 기존 `[NH2,NH][c]`로는 양성 패널 6개 중 5개가 0점이 되어 §9.5가 false pass한다 — 이 검증 없이 §7을 실행하지 말 것
- 출력: cpd 25, PLN-1474, bexotegrast, CWHM-12, GLPG0187에 대한 **component별** 점수(총점도 기록하되 비교 근거로 쓰지 않는다 — §3.4) — 포스터에 실릴 모든 비교의 배경이 되는 기준 분포

---

## 8. 사후 triage funnel (D7–D8)

순서가 중요하다 — 값싼 것부터 비싼 것 순으로:

0. **타우토머 정규화 — 모든 지문 비교의 선행 조건.** 모든 SMILES에 `rdMolStandardize.TautomerEnumerator().Canonicalize()`를 적용한다. **기준 세트와 측정 대상에 동일하게** 적용해야 한다.

   `scripts/curate_actives.py:222-239`의 `standardize()`는 `Cleanup` + `LargestFragmentChooser` + `Uncharger`만 수행하고 타우토머 정규화를 하지 않았다. 이것이 §8.3과 §9.3에 false-positive 경로를 만든다. 노출 규모를 측정했다(정규화 시 표현이 바뀌는 비율, 그리고 바뀐 분자가 *자기 자신의 다른 타우토머*에 대해 갖는 Tanimoto):

   | 세트 | 표현 변경 | 자기-Tanimoto 중앙값 | **0.680 미만** |
   |---|---|---|---|
   | `actives_core` (29) | 0/29 | — | 0 |
   | `benchmark_panel` (6) | 0/6 | — | 0 |
   | `actives_extended` (190) | 14/190 (7.4%) | 0.736 | 1 |
   | **TL-A prior 샘플 (488)** | **43/488 (8.8%)** | 0.621 (최소 0.379) | **29** |

   즉 **생성 분자 488개 중 29개(5.9%)가 자기 자신의 다른 타우토머에 대해서도 §8.3 기준으로 "scaffold-novel" 판정을 받는다.** PLN-1474가 그 예다 — 테트라히드로-1,8-나프티리딘을 비방향족 아미딘 형태로 쓰면 방향족 형태와 분자식은 같지만(`C24H37N3O4`) QED 0.4332 vs 0.4619, 방향족 고리 수 0 vs 1, 그리고 **서로에 대한 NN-Tanimoto가 0.458**이다. 알려진 임상 화합물을 그대로 재현해도 신규 분자로 통과한다는 뜻이다.

   비대칭이 문제의 핵심이다: ChEMBL actives는 일관된 방향족 형태로 들어오지만(core 0%, bench 0%) 생성 분자는 모델이 내보내는 형태 그대로다 — 기준 쪽은 정규화돼 있고 측정 대상만 흔들린다. 같은 이유로 aniline alert도 타우토머 의존적이므로(비방향족 형태는 `[NH2,NH][c]`에 미매치) RL이 표현을 바꿔 필터를 회피하는 SMILES 수준 reward hacking이 가능했다. §5.4의 `[NX3;H2][c]`는 두 형태 모두 미매치라 이 경로가 닫히지만, 정규화를 상류에 두는 것이 근본 해결이다.

   **적용 대상:** 생성/prior 샘플, `data/actives_extended.smi`, `data/actives_core.smi`, anti-target counter-screen 참조(§8.4), 그리고 `data/novelty_band.json` **재계산 — 실행 완료, 아래 §8.3 표**(extended p25 0.710 → 0.680).
1. **Validity / 중복 제거 / 카르복실레이트 재확인 / CustomAlerts 재확인**(§5.4의 교체된 패턴으로). 카르복실레이트는 RL에서 게이트지만 하한이 문자 그대로 0이 아니므로(6.31e-04) 여기서 **하드 컷**으로 다시 적용한다.
2. **물성 범위:** TPSA는 RL에서 사용한 것과 동일한 창을 쓰고, RL 목적함수에서 제거된 축은 여기서 **넓은 창으로** 본다 — logP ≤ 5(선행연구 [113]의 hard filter 준거), MW 250–550, RotB. 목적함수를 좁히고 triage를 넓히는 것이 이 재설계의 취지다(§5.1)
3. **Novelty — 빌려온 상수가 아니라 actives 자신으로 교정한다.** 학습에 쓴 29개가 아니라 **`data/actives_extended.smi`(190개 분자, Murcko scaffold 103개 — ChEMBL αvβ1 active ≤ 1 µM 전체)** 에 대한 nearest-neighbor Tanimoto를 Morgan radius 3, feature invariant, **count** 지문으로 잰다(§8.0의 타우토머 정규화를 거친 뒤에).

   기존 기준 `NN-Tanimoto < 0.4`는 **철회한다.** 이것은 이진 ECFP4용 관례인데 이 프로젝트는 이진 ECFP4를 쓰지 않으며, feature 기반 count 지문에서는 같은 쌍이 훨씬 높게 나온다. 자기 기준 데이터에 대보면 성립하지 않는다: **core actives의 83%, extended의 99%가 서로 다른 Murcko scaffold를 가진 active로부터 0.4 이상**에 있으므로, 이 규칙은 공개된 αvβ1 계열 거의 전부를 서로에 대해 "신규하지 않다"고 판정한다.

   대체 기준은 데이터가 정의하는 거리다 — 서로 다른 scaffold에 속한 두 known active가 얼마나 떨어져 있는가(`data/novelty_band.json`, `curate_actives.py`가 재계산):

   | 기준 집합 | 타우토머 | p25 | 중앙값 | p75 | p90 |
   |---|---|---|---|---|---|
   | `actives_core` (29, scaffold 19) | 현행 | 0.552 | 0.676 | 0.732 | 0.745 |
   | `actives_core` (29, scaffold 19) | **정규화** | 0.552 | 0.676 | 0.732 | 0.745 |
   | `actives_extended` (190, scaffold 103) | 현행 | 0.710 | 0.775 | 0.831 | 0.922 |
   | `actives_extended` (190, scaffold 103) | **정규화 — 채택** | **0.680** | 0.765 | 0.831 | 0.922 |

   §8.0의 타우토머 정규화를 적용해 재계산했다(`curate_actives.py:627-641`의 로직을 그대로 재현 — 정규화 없이 돌리면 현행 `data/novelty_band.json`의 0.552 / 0.710을 정확히 복원하므로 재현이 검증된다). `actives_core`는 타우토머 변경 분자가 0개이므로 불변이고, `actives_extended`는 14개가 바뀌어 p25가 **0.710 → 0.680**으로 내려간다. 기준이 약간 **엄격해지는** 방향이다.

   **그러나 거리 기준을 게이트로 쓰는 것도 철회한다(2026-09-26).** 측정해 보면 이 기준은 고장나 있다. Murcko scaffold가 공개 active 103개의 것과 **완전히 동일한** prior 샘플 63개에 `NN-Tanimoto < 0.680`을 적용하면 **39개(62%)가 "scaffold-novel"로 통과한다.** 공개된 골격 위에 올라간 분자를 신규 chemotype이라 부르는 셈이다. 거리는 연속량이고 골격 동일성은 이산량인데, 연속량의 절단값으로 이산적 질문("이 골격이 기존에 제안되었는가")에 답하려 한 것이 오류다.

   **채택 기준 — Murcko scaffold 신규성(이진).** 생성 분자의 Murcko scaffold가 `data/known_scaffolds.smi`의 **106개 집합에 없으면** scaffold-novel이다. 지문 의존이 없고 임계값이 없고 한 문장으로 기술된다. 그리고 목적함수의 diversity filter가 이미 `IdenticalMurckoScaffold`이므로, 보상 기전과 성공 기준이 **같은 단위**를 쓴다.

   **기준 집합은 `actives_extended` 단독이 아니라 큐레이션된 참조 파일 5개의 합집합이다**(`actives_core`, `actives_core_B`, `actives_extended`, `benchmark_panel`, `similarity_refs` → 106개, `scripts/known_scaffolds.py`가 산출). `actives_extended` 단독(103개)에는 **PLN-1474, bexotegrast, A1AFA의 scaffold가 없다** — 그 파일이 "ChEMBL αvβ1 active ≤ 1 µM"으로 정의되기 때문이다. PLN-1474 구조는 ChEMBL 활성 레코드가 아니라 AdisInsight 출처이고[88], A1AFA는 pIC50 5.30으로 효력 컷 아래다. 단독 집합을 쓰면 **PLN-1474의 골격을 재현한 생성 분자가 "scaffold-novel"로 통과한다** — 유일한 임상 αvβ1 선택적 화합물이고 2023년 8월부터 구조가 공개된 화합물인데도. 포스터가 비교 대상으로 내세우는 공개 화합물의 골격은 기준에 반드시 들어가야 한다. `tests/test_known_scaffolds.py`가 세 화합물 각각에 대해 "단독 집합에는 없고 합집합에는 있다"를 단언한다.

   - TL-A prior 샘플 488개 중 **420개(86.1%)** 통과 — 느슨한 필터다. 구속력은 §8.5 docking 기하와 §9.2 permeability가 갖는다.
   - **정확 구조 불일치만으로는 불충분하다.** TL-A 200 epoch는 암기한다(§4: 최대 NN-Tanimoto = 1.000, 학습 active를 그대로 재현한 샘플이 존재). 메틸 하나를 붙이면 "기존에 없던 구조"가 되므로, 이 기준으로는 "cpd 25에 메틸 붙인 것 아닌가"라는 공격을 막을 수 없다. Murcko는 막는다.
   - **NN-Tanimoto는 게이트가 아니라 보고 수치로 유지한다.** Murcko가 새로운 420개 중 **5개가 NN ≥ 0.680(최대 0.738)** 이므로 analog 논란이 가능하다. 각 lead의 NN 값과 **가장 가까운 known active의 이름**을 함께 적으면 임계값 없이 독자가 판단할 수 있다. 위 대역 표는 그 값을 읽는 맥락으로만 남긴다 — `data/novelty_band.json`은 게이트가 아니다.
   - **타우토머 정규화가 여기에 직접 걸린다(§8.0).** 검증했다: PLN-1474의 두 타우토머는 정규화 없이 **서로 다른 Murcko SMILES**를 낸다(`...C1=CC=C2CCCN=C2N1` vs `...c1ccc2c(n1)NCCC2`). 정규화 후 동일해진다. 정규화를 빼면 신규성 주장이 표현 차이로 뚫린다.
4. **선택성 — 주장하지 않고 관찰로 보고한다(2026-09-27 개정).** 기존의 "αvβ3/α5β1 active에 대한 Tanimoto ≥ 0.5 플래그" 조항은 **철회한다**: §3.2가 규정한 counter-screen 세트가 산출된 적이 없어 실행 불가이고, §3.2 자신이 "유사도 ≠ 활성"이라 적었을 만큼 기전이 약하다. 대신 §8.5의 통과 pose에서 다음 셋을 **측정해 보고한다**.

   (i) **β1-Leu225 측쇄 접촉** — 리간드 소수성 원자가 Leu225 측쇄 탄소(CB/CG/CD1/CD2)로부터 4.5 Å 이내인가. ITGB1(P05556, signal peptide 20잔기 제거)과 ITGB3(P05106)의 βI 도메인을 정렬하면 Sabat이 선택성의 원천으로 지목한 소수성 포켓 다섯 잔기 중 **Leu225만이 β3와 다르다**(β3는 **Arg** — 소수성 벽이 하전 측쇄로 바뀐다). Tyr133·Pro186·Cys187·Asn224·Asp226은 모두 보존이다. 이것이 GLPG0187·CWHM-12가 같은 3-aminopropionyl 골격으로 Leu225 **backbone**을 잡으면서도 pan인 이유를 설명한다 — 포켓은 Leu의 **측쇄**를 요구한다.

   (ii) **αv-Asp218 접촉의 성격** — Asp218 카르복실산소로부터 4.0 Å 이내에 (a) 중성 수소결합 공여체가 있는가, (b) 염기성 질소(Arg 모방 헤드)가 있는가. (b)는 문헌에서 pan-αv와 연관된다: αvβ6 임상 후보 GSK3008348의 1,8-나프티리딘이 Asp218과 **염다리**를 이루고 [114], Sabat의 THN 도입이 pan-αv를 낳았다 [111].

   (iii) **αv-Tyr178 π-스택** — 리간드 방향족 고리 중심이 Tyr178 고리 중심으로부터 5.5 Å 이내이며 고리 평면 각이 30° 이내(평행) 또는 60–90°(T-shaped)인가. 8W30 실측: 최근접 원자 3.70 Å, centroid 6.21 Å, 고리 각 74.6°.

   **이들은 선택성과 상관된 것으로 보고된 특징이며 선택성의 증거가 아니다.** αvβ1 선택성은 **β 서브유닛**(αvβ3/αvβ5/αvβ6/αvβ8 대비)과 **α 서브유닛**(α5β1/α4β1/α8β1 대비) 양쪽에서 결정되며 [114], 본 연구는 후자를 전혀 다루지 않는다 — 같은 리뷰가 α5의 Trp157·Gln221·Ser224를 후보로 들지만 분석하지 않았다. 그리고 선택성을 부여한다는 접촉은 **결정학적으로 관찰된 적이 없다**: 2020년까지 αvβ1 구조가 없었고 [114], 최초 구조인 8W30의 리간드는 pIC50 5.30이며 Asp218로부터 7.00 Å로 닿지 않는다. **실측 교차 IC50이 선택성 판정의 유일한 근거이며 후속 과제다.**
5. **Docking — affinity 순위가 아니라 기하학적 필터:**
   - Receptor: 8W30의 chain A+B; 리간드 TR01225179와 물 분자 제거(물 처리 결정을 기록할 것. HOH2107은 민감도 분석에서 유지 후보)
   - 도구: smina / AutoDock Vina; 리간드 centroid 기준 약 20 Å box
   - **알려진 한계:** Vina 계열 scoring function은 metal coordination을 모델링하지 못한다 — 가장 중요한 상호작용(carboxylate → MIDAS Ca²⁺)이 scorer에게는 보이지 않는다. 따라서:
     - **(a) 검증:** TR01225179를 redocking한다. 최상위 pose에서 carboxylate-O → Ca501 ≤ 3.2 Å를 요구한다(이상적으로는 결정 구조 pose 대비 heavy-atom RMSD < 2 Å도). **검증에 실패하면 어떤 것도 docking으로 순위 매기지 말 것** — similarity/QSAR 기반 triage로 후퇴하고 이를 포스터에 명시한다.
     - **(b) 사후 필터:** carboxylate O → Ca501 ≤ 3.2 Å **이면서** H-bond donor가 β1-Asn224 backbone O로부터 3.5 Å 이내에 있는 pose만 채택한다. Docking score는 동점 처리용으로만 사용한다.

   **검증 결과 — 실행 완료, 통과(2026-09-27).** conda `docking` 환경(smina 2020.12.10, openbabel)에서 8W30 chain A+B(리간드·물·글리칸 제거, **Ca²⁺ 6개 유지**)에 TR01225179를 되돌려 넣었다: `--autobox_ligand` + `--autobox_add 6 --exhaustiveness 16 --num_modes 20 --seed 42`.

   | pose | affinity | O→Ca501 | N→Asn224 O | RMSD |
   |---|---|---|---|---|
   | **1 (최상위)** | −6.91 | **2.72 Å** | **2.89 Å** | **0.63 Å** |
   | 2 | −6.83 | 2.31 | 3.27 | 1.75 |
   | 3 | −6.74 | 2.47 | 3.10 | 1.83 |
   | 결정 구조 | — | 2.62 | 2.63 | — |

   상위 3개 pose가 모두 RMSD 2 Å 이내이고 affinity 순위와 RMSD 순위가 일치한다. **금속 항이 없는데도 pose 생성은 성공했다** — Ca²⁺를 receptor에 유지하면 MIDAS가 좁고 극성인 오목부를 이루어 입체 배제만으로도 카르복실레이트의 자리가 결정된다. 금속 항 부재의 영향은 **affinity 순위**에 남으므로, score를 동점 처리용으로만 쓰는 설계가 정당화된다.

   > **RMSD 계산 주의.** 인덱스 순서를 가정한 naive RMSD는 6.13 Å을 낸다 — smina/obabel 출력의 원자 순서가 입력과 다르고 분자에 대칭(페닐·디클로로페닐)이 있기 때문이다. `rdMolAlign.CalcRMS`(대칭·원자매핑 고려, 좌표 정렬 없음)로 재야 0.63 Å이 나온다. **naive RMSD를 쓰면 통과한 검증을 실패로 오판한다.**

   **임계값을 그렇게 정한 이유.**

   *carboxylate-O → Ca501 ≤ 3.2 Å* — 첫째, **이 구조의 1차 배위권이 그 경계에서 끝난다.** Ca501을 배위하는 O/N을 거리순으로 보면 β1-Glu229 OE2 2.39 Å, β1-Ser132 OG 2.45 Å, β1-Ser134 OG 2.50 Å, **리간드 OXT 2.62 Å** — 그리고 **간극** — β1-Asp259 OD1 3.30 Å, β1-Asp130 OD2 3.85 Å. 2.62와 3.30 사이의 공백이 1차와 2차 배위권을 가르며 3.2 Å은 그 안에 있다. 즉 관례가 아니라 **이 구조 자신이 정의하는 경계**다. 둘째, Ca²⁺–O 배위의 통상 범위가 2.3–2.6 Å이므로 3.2 Å은 관대한 상한이며, 2.45 Å 해상도와 강체 receptor 가정에서 오는 좌표 오차를 흡수한다. 셋째, 결정값 2.62 Å에 0.58 Å 여유인데 redocking 최상위 pose가 2.72 Å(+0.10)이었으므로 과하지도 부족하지도 않다.

   **단일좌로 판정한다.** 리간드의 카르복실레이트 산소 둘 중 `OXT`만 2.62 Å이고 다른 하나(`O`)는 **4.54 Å**이다. 따라서 조건은 **"두 산소 중 하나라도 ≤ 3.2 Å"**이며, 양쪽을 요구하면 결정 구조 자신이 탈락한다. **Ca501을 지정하는 근거**는 Ca502가 리간드로부터 6.44 Å로 관여하지 않기 때문이다.

   *H-bond donor → β1-Asn224 backbone O ≤ 3.5 Å* — 실측값이 **2.63 Å**이다. 같은 산소의 다른 파트너와 비교하면 Leu225 N 2.25 Å(backbone 연쇄), 리간드 N 2.63 Å, Asn224 N 2.65 Å로 모두 2.2–2.7 Å 대역이다. 3.5 Å은 N···O 수소결합의 표준 관대 절단값이며 이 대역 위로 0.85 Å 여유를 둔다. Asn224는 chain B에만 있고 리간드도 chain B이므로 같은 사슬 내 상호작용이다.

   **이 접촉은 결합 요소이며 선택성 요소가 아니다.** Sabat 2024가 MIDAS와 Asn224 접촉이 *"observed αvβ1 affinity를 뒷받침하지만 isoform preference는 반드시 그렇지 않다"*고 명시하며, β1/β3 서열 정렬에서 **Asn224가 β3에 보존**되어 있어 그 구조적 이유가 확인된다(§8.4).

   *pose 생성 파라미터* — `--exhaustiveness 16 --num_modes 20`, `--seed 42`. 금속 항이 없으므로 올바른 기하가 최상위로 오르는 것을 기대할 수 없고 **pose를 많이 만들어 기하로 걸러내는 것**이 (b)의 설계다. redocking에서 상위 3개가 2 Å 이내였으므로 20 modes로 충분하다.

   *물 처리* — 기본 receptor는 물을 모두 제거한다. **HOH A2107**(리간드로부터 4.68 Å, 서브유닛 계면)만 유지한 receptor를 민감도 분석용으로 함께 만든다. Sabat 2024가 αv-Asp218 대신 **계면 결정수**를 engage하는 전략으로 선택성을 얻었다고 보고하기 때문이다. 두 receptor의 결과 차이를 보고한다.
6. **ADMET:** 생존 분자에 ADMET-AI(또는 동등 도구) 적용 — permeability proxy, 용해도, microsome 안정성, hERG, CYP. 의사결정의 기준선은 cpd 25의 *실측* 약점이다(MDCK < 0.1×10⁻⁶ cm/s, oral F 1.3% [111]). 예측치는 방향성 면에서 이 기준을 넘어야 한다.
7. **최종 선별:** 다음을 만족하는 lead 약 10–20개 — 기하학적 필터 통과, 예측 permeability가 cpd 25보다 우수, **Murcko scaffold가 §8.3의 106개 집합(`data/known_scaffolds.smi`)에 없음**, alert 없음, 카르복실레이트 보유. QED와 NN-Tanimoto(최근접 active 이름 포함)는 **게이트가 아니라 보고 지표**로 병기한다.

---

## 9. 사전 등록된 성공 기준 (D6 sampling 전에 고정)

**동일한** funnel을 통과시켰을 때, 다음을 만족하는 생성 분자가 최소 하나 있으면 제안은 성공이다:
1. 기하학적 docking 필터 통과(MIDAS + Asn224 contact 보존),
2. 예측 permeability 축에서 cpd 25를 능가,
3. **Murcko scaffold가 `data/known_scaffolds.smi`의 106개 집합에 없을 것**(§8.3 — 큐레이션된 참조 파일 5개의 합집합이며, `actives_extended` 단독 103개는 PLN-1474·bexotegrast·A1AFA를 빠뜨린다). 양쪽 모두 §8.0의 타우토머 정규화를 거친 뒤에 판정한다 — 정규화 없이는 같은 분자가 다른 Murcko SMILES를 낸다. NN-Tanimoto와 가장 가까운 known active는 **게이트가 아니라 보고 수치**로 병기한다,
4. CustomAlerts 플래그 없음(§5.4의 교체된 aniline 패턴 기준) — **counter-screen 조항은 철회했다**(§8.4): 세트가 산출된 적이 없고 기전이 약하다. **선택성은 이 기준에 포함되지 않는다**,
5. 그리고 **양성 패널(PLN-1474, bexotegrast, CWHM-12, GLPG0187, cpd 25)과 동일한 funnel·동일한 도구로 채점했을 때, 순환하지 않는 축에서 비교 가능할 것** — docking 기하 필터를 동등하게 통과하고, ADMET-AI permeability에서 cpd 25를 능가하며, Murcko scaffold가 신규일 것. **총점으로는 비교하지 않는다**(우리가 설계한 목적함수의 값이므로 순환 — §3.4, §5.4). 음성 패널 절은 폐기했다(§3.4).

**반증 조항:** 1–4를 만족하는 분자가 하나도 없으면, 그 사실을 그대로 보고한다. 이는 carboxylate–permeability 긴장이 **타깃 자체에 내재된 것**일 가능성을 시사하는 증거이며, 그 자체로 하나의 발견이자 이 프로젝트에 대한 가장 강한 공격에 대한 정직한 답이다.

---

## 10. 일자별 일정

| 일자 | 작업 | 체크포인트 / fallback |
|---|---|---|
| D1 | **완료.** K-BDS 실측으로 환경 확정(§11), `setup_kbds.sh` + `slurm/00_smoke.sbatch` 준비, prior는 v4.5.11 동봉분을 체크섬 검증. **§3.1 큐레이션도 완료** — `scripts/curate_actives.py`, 산출물은 `data/`, 근거는 `data/CURATION_LOG.md` | 세트 고정: core 29 / core_B 200 / extended 197 / refs 6 / panel 6 |
| D2 | Counter-screen set(**target ID 검증 완료, §3.2**) | 모든 데이터 고정. 음성 패널은 폐기(§3.4) |
| D3 | TL-A·TL-B 실행 + 3개 arm 전체 진단 | D3 기준으로 arm 1개 선택, **화학형 이동이 결정적**. TL-B가 Arg 모방체 비율이나 TPSA를 올리면 다른 수치와 무관하게 탈락 |
| D4 | **선행 작업:** 타우토머 정규화를 비교 계층(`scripts/normalize.py`)에 추가 → `novelty_band.json` 재계산(§8.0); `CustomAlerts` aniline 패턴 교체 검증(§5.4). **`curate_actives.standardize()`는 의도적으로 그대로 둔다** — 그 출력은 TL-A 학습 입력이자 RL inception seed인 `data/actives_core.smi`에 기록되므로, 여기에 타우토머 정규화를 추가하면 이미 완료된 D3 run이 무효화된다. 정규화는 비교 계층에만 존재한다(`scripts/normalize.py:8-9`, `scripts/curate_actives.py:243-246`). 그 후 RL 단일 stage 0–300 step | 참조 세트 0점 화합물이 **§5.4에 기록된 1건 외에 없음**을 확인한 뒤 착수 — RL이 적용하는 여덟 SMARTS는 `actives_extended`의 `CHEMBL4756602` 하나를 `[Nr0][Nr0]`(비고리 N–N, 하이드라진류)로 0점 처리한다. 이 패턴은 정당한 reactive alert이고 190개 중 1개는 aniline의 164/190 같은 파국이 아니므로 받아들이되 숨기지 않는다. PAINS는 RL에 없고 `objective.py`/§8 triage에서만 적용되므로 거기서는 3건이 된다(`CHEMBL244434`, `CHEMBL244013`이 `mannich_A(296)`). 100 step마다 §5.5 진단 기록 |
| D5 | RL 단일 stage 300–600 step | 최종 agent; 알려진 series로의 붕괴 감시(NN-Tanimoto가 0.6 초과로 상승 → DF `minscore` 0.4 → 0.5로 상향, 필요시 이전 체크포인트에서 재시작). **similarity 가중치 조정은 불가** — 그 component는 목적함수에 없다(§5.2) |
| D6 | Sampling 20k **(완료)**; 벤치마크 패널 scoring | `data/library.smi` 19,485개 + `data/library_prior.smi` 18,346개 (§6.1); 기준 점수 분포 |
| D7 | Docking 설정 + TR01225179 redocking 검증 | 검증된 프로토콜 **또는** similarity/QSAR 기반 triage로의 fallback 문서화 |
| D8 | Docking + 기하학적 필터 + ADMET + counter-screen | triage 표 |
| D9 | 벤치마크 비교, lead 선별, 결과 표 + 그림 | 상위 lead 10–20개 |
| D10 | 예비일: 포스터 조립(pocket 그림은 이미 완료), 7분 발표 리허설, 공격 표면 대비 | — |

**전역 fallback:**
- **당일 GPU 사용 불가:** `batch_size` 32–64, `max_steps` 절반, sampling 10k — 전체 과정은 여전히 완주 가능.
- **TL이 prior보다 못한 경우:** 그것이 arm TL-C이며 이미 D3 비교에 들어 있다 — prior를 그대로 가져가되 inception memory를 키우고 그 사실을 명시한다. (기존의 "active 30개 미만이면 TL 생략" 규칙은 대체되었다. 코어가 29개인 것은 공개된 비-RGD 화학형이 그만큼 작기 때문이고, 그것이 너무 작은지 여부는 이제 가정이 아니라 D3에서 측정한다.)
- **Docking 검증 실패:** similarity/QSAR 기반 triage만 수행; 한계를 명시적으로 기술.

---

## 11. 컴퓨팅 관련 메모 — K-BDS (KISTI), 실행 완료

KISTI K-BDS 클러스터에서 실행한다. 스택은 로그인 노드 실측(`scripts/kbds_probe.sh`)에서 정했으며 가정에 근거하지 않았다. 전체 근거는 `TRANSFER.md`에 있다.

| | |
|---|---|
| 로그인 노드 | `kbds.kisti.re.kr` (OTP) — **SSH/scp는 여기만 가능**; `kbds-dm.kisti.re.kr`은 FTP 전용(포트 21, 22는 닫힘) |
| OS, 스케줄러 | CentOS 7.9.2009 (**glibc 2.17**), Slurm 21.08.5, 배타적 노드, wall clock 120시간 |
| GPU | A100 40 GB, **드라이버 470.57.02** |
| 큐 | `1gpu`, `2gpu` (A100 40G); `4gpu`, `8gpu` (A100 80G); `debug-1gpu` (8시간 제한) |
| 환경 | conda env **`reinvent`**, Python 3.11 |

**측정 3건이 스택 전체를 결정했다:**

1. **glibc 2.17** — REINVENT4 `main`은 `torch==2.12.0`을 핀하는데 이 버전은 `manylinux_2_28` 휠만 있어 여기서 돌지 않는다. **REINVENT4를 `v4.5.11`로 고정**했다 — torch 핀(2.5.1)이 아직 glibc 2.17 휠을 갖는 마지막 릴리즈이자 prior를 레포에 동봉한 버전이다. 같은 제약으로 PyPI `rdkit`(역시 `manylinux_2_28`)도 배제되어 **rdkit은 conda-forge**에서 가져온다(`__glibc >=2.17`).
2. **드라이버 470.57.02** — CUDA 12는 드라이버 ≥ 525가 필요하므로 클러스터가 제공하는 `compilers/cuda/12.4`·`12.8` 모듈은 이 용도로 쓸 수 없다. CUDA 11.8은 ≥ 450이면 된다. v4.5.11 핀에서 바꾼 것은 **`torch==2.5.1+cu124` → `2.5.1+cu118`** 하나뿐이다 — 같은 torch, 다른 CUDA 빌드이며 `pyproject.toml` 주석이 예상하는 바로 그 교체다. **CUDA 모듈은 로드하지 않는다**: pip 휠이 자체 런타임을 포함하고, 툴킷을 로드하면 그것을 가려버린다.
3. **외부 인터넷이 열려 있다**(pypi·conda-forge·zenodo·github·rcsb 전부 200, 프록시 없음) — K-BDS에서 직접 설치하며 `conda-pack` 번들이 필요 없다. Apptainer와 Singularity는 로그인 노드에서 둘 다 깨져 있어(`libsubid.so.3` 부재) 컨테이너 경로는 쓸 수 없었다.

**선언된 의존성 3개를 의도적으로 제거했다.** 셋 다 스코어링 컴포넌트 하나씩만 지원하며, REINVENT4는 import에 실패한 컴포넌트를 건너뛰므로 나머지에 영향이 없다: `openEye-toolkits`(상용 라이선스, ROCS 전용), `chemprop`(§3.3이 ChemProp2를 no-go로 확정했고, scipy·scikit-learn을 끌어오는데 이들 최신 릴리즈는 `manylinux_2_28` 전용), `descriptastorus`(chemprop의 숨은 의존성으로 REINVENT4 코드가 import하지 않음). REINVENT4를 `pip install --no-deps`로 설치하는 이유이며, `setup_kbds.sh`는 마지막에 RL 목적함수가 쓰는 컴포넌트 9개가 전부 등록됐는지 검증한다.

**v4.5.11은 `torchvision`과 `scipy`를 선언하지 않았다** — 둘 다 `reinvent.Reinvent`에서 출발하는 경로(sampling report 경유)에서 모듈 레벨로 import되는데 `pyproject.toml`에 없다. `--no-deps`로 설치하므로 이건 설치 오류가 아니라 **첫 실행 시 크래시**가 된다. 그래서 `scripts/check_imports.py`가 모듈 레벨 import 그래프를 순회하고, `setup_kbds.sh`가 설치 **이전에** 이를 실행한다. torchvision은 `0.20.1+cu118`로 고정했고(0.20.1이 `torch==2.5.1`을 정확히 요구하며, 자체 CUDA 커널이 있어 같은 인덱스가 필요), scipy는 `manylinux_2_28`로 넘어간 1.17 미만으로 제한했다.

pip는 바이너리 전용(`--only-binary=:all:`)으로 설치한다. 이 옵션이 없으면, 최신 릴리즈가 `manylinux_2_28` 휠만 제공하는 패키지에서 pip가 조용히 sdist로 폴백하고, 빌드는 의존성 몇 단계 아래에서 엉뚱한 패키지를 탓하며 죽는다 — 첫 시도가 *"NumPy requires GCC >= 9.3"* 로 실패했는데 실제 원인은 pandas 2.3.3이 `manylinux_2_17` 휠을 없앤 것이었다. `env/requirements-pip.txt`의 모든 상한은 PyPI의 실제 cp311 휠 태그로 확인했다.

**명령**

- 이송: `TRANSFER.md` 참조 (45 GB 트리 중 2.3 GB만 — 나머지는 완료된 상류 작업)
- 로그인 노드에서 1회 빌드: `bash scripts/setup_kbds.sh`
- GPU 스모크 테스트: `mkdir -p logs && sbatch slurm/00_smoke.sbatch` — 실제 TL-A 분할로 TL 1 epoch + 200분자 샘플링을 돌리고 validity 80% 미만이면 실패 처리
- 실행: `reinvent -l run.log config.toml`
- 데이터 큐레이션: `python3 scripts/curate_actives.py` (RDKit만 필요. ChEMBL/PubChem/RCSB 응답은 `data/.cache`에 캐시되므로 오프라인 재현이 가능하고, 사용된 ChEMBL 릴리스가 로그에 기록된다).
- GPU: RL `batch_size = 128`; `1gpu`에 제출(A100 1장으로 충분하고 시간당 1 노드시간으로 가장 저렴). 실측 시점에 모든 GPU 파티션이 alloc 상태였으므로 대기를 예상할 것.

## 12. 이 프로토콜이 의도적으로 하지 않는 것

- **RL 내부 docking 미사용**(DockStream/Maize): 10일 프로젝트에는 너무 느리고, 어차피 metal coordination을 보지 못한다. Docking은 사후 기하학적 필터로만 쓴다.
- **Azabenzimidazolone core 기반 LibInvent 미사용:** 프로젝트가 벗어나려는 chemotype을 전제하게 되기 때문이다.
- **예측 affinity 주장 없음:** Vina score는 affinity가 아니며, 특히 metal site에서는 더욱 그렇다. 주장은 기하 구조 + 물성 + novelty의 틀로 제시한다.
- **Wet-lab 약속 없음:** 산출물은 사전 등록된 기준을 갖춘, 우선순위가 매겨지고 반증 가능한 in silico lead 집합이다.

---

## References

- [39] MolecularAI — REINVENT4 GitHub repository
- [40] REINVENT4, J. Cheminformatics (2024)
- [88] AdisInsight — PLN-1474 development status (structure public Aug 2023; no development as of May 2025)
- [98] Lancaster et al., INTEGRIS-IPF, Am. J. Respir. Crit. Care Med. (2024) — bexotegrast
- [103] Henderson et al., Nat. Med. (2013) — HSC αv deletion; CWHM-12
- [111] Sabat et al., J. Med. Chem. (2024) — Design and Discovery of a Potent and Selective Inhibitor of Integrin αvβ1
- [112] PDB 8W30 — αvβ1 headpiece + TR01225179
- [113] Qie, Wang, Li et al., *Molecular Diversity* (2026), doi 10.1007/s11030-026-11625-z — REINVENT4 stage-wise RL로 EGFR 저해제 설계 + 실험 검증. 이 프로젝트가 참조하는 **선행 RL 프로토콜**
- [114] Zheng Y, Leftheris K, *J. Med. Chem.* **2020**, 63, 5675–5696, doi 10.1021/acs.jmedchem.9b01869 — *Insights into Protein–Ligand Interactions in Integrin Complexes: Advances in Structure Determinations*. αv 서브유닛이 공유되므로 αvβx 사이의 선택성은 β 서브유닛이 결정한다는 원칙, 2020년까지 αvβ1 구조 부재, α5의 Trp157·Gln221·Ser224, GSK3008348의 Asp218 염다리, 2,6-디클로로페닐이 여러 인테그린 저해제의 범용 SDL 근접 모티프라는 사실의 출처.
