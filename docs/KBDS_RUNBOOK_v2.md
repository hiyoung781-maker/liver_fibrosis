# K-BDS 실행 런북 — v2 재실행

**대상 브랜치:** `avb1-rerun-v2`
**전제:** 로컬(TITAN Xp)에서 TL 스윕과 RL 두 아암, 샘플링까지 완료됨. K-BDS에서는 **도킹부터** 시작한다.
**상세 근거:** `results/v2_EXECUTION_CHECKLIST.md`(태스크별), `docs/superpowers/specs/2026-09-28-avb1-rerun-design.md`(설계)

---

## 0. 지금 적용하면 안 되는 것 — 먼저 읽을 것

**상호작용 게이트(`scripts/interaction_gate.py`)를 이번 실행에서 적용하지 말 것.**

Task 13의 대조군이 결함을 찾았다. PLIP은 배포된 8W30에서 리간드와 MIDAS 칼슘을 `A1A-CA`라는 **하나의 복합 사이트**로 묶으므로 hetid 선택이 2.62 Å 금속 배위를 본다. 그런데 `scripts/poses_to_complex.py`가 만든 복합체에서는 `LIG` 사이트의 `metal_complexes`가 **0건**이고 칼슘이 자기 사이트로 분리된다.

결과: 게이트 기준 (a) MIDAS 금속 배위가 **변환기로 만든 모든 복합체에서 조용히 실패**한다. 캠페인의 모든 포즈가 변환기를 거치므로 통과율 0%가 화학적으로 보이는 숫자로 보고된다.

**따라서 이번 실행의 범위는 "포즈와 PLIP 레코드를 만들어 저장하는 것"까지다.** 게이트는 저장된 레코드에 나중에 적용하므로, 계산을 먼저 돌려도 아무것도 낭비되지 않는다. 자세한 내용은 `results/v2_plip_validation.md`.

---

## 1. 도구 설치 (최초 1회)

로컬에는 `obabel`·`smina`·`plip`만 있고 **`autogrid4`와 `autodock_gpu`는 없다.** K-BDS에 둘 다 필요하다.

```bash
# PLIP (CPU 도구, GPU 구현 없음)
conda activate docking          # 또는 동등한 환경
pip install plip                # plip 3.0.1 + openbabel + lxml

# AutoDock4 suite (autogrid4) 와 AutoDock-GPU
# autogrid4: AutoDock4 배포판 또는 conda
# autodock_gpu: CUDA 컴파일 필요 (https://github.com/ccsb-scripps/AutoDock-GPU)
which autogrid4 autodock_gpu plip obabel   # 넷 다 나와야 진행 가능
```

**검증:** `python -m unittest tests.test_prepare_maps tests.test_dock_autodock_gpu tests.test_run_plip -v`
(`pytest`는 설치하지 않는다. 이 저장소의 테스트는 전부 `unittest`다.)

---

## 2. 수용체와 그리드 맵

```bash
python scripts/prepare_receptor_pdbqt.py --out docking/v2/receptor.pdbqt
python scripts/prepare_maps.py \
    --receptor docking/v2/receptor.pdbqt \
    --ligand-ref docking/ligand_ref.sdf \
    --out-dir docking/v2/maps
(cd docking/v2/maps && autogrid4 -p receptor.gpf -l receptor.glg)
```

**반드시 지킬 것**
- `obabel -xr -h` — 수소 포함. 수소 없이 변환하면 Open Babel이 단백질 질소 1,212개 중 1,170개를 H-결합 **수용체**로 타이핑해 단백질 전체의 수소결합 지형을 뒤집고, 4위 포즈를 1위로 올린다.
- **`obabel -p 7.4` 금지.** 칼슘 6개를 조용히 전부 제거하는데 affinity는 정상으로 보인다.
- `prepare_receptor_pdbqt.py`의 질소 donor/acceptor 비율 검사를 약화시키지 말 것.
- `prepare_maps.py`의 `maps_cover_box`가 통과해야 한다. AutoDock-GPU는 리간드가 그리드를 벗어나도 **오류를 내지 않고** 그럴듯한 값을 반환한다.

---

## 3. 재도킹 검증 게이트 — 통과 못 하면 중단

```bash
python scripts/validate_redock.py --protonation neutral --out results/v2_redock_neutral.json
python scripts/validate_redock.py --protonation anion   --out results/v2_redock_anion.json
```

**사전등록된 판정 (결과를 보고 바꾸지 말 것)**
- 통과 조건: 카복실레이트 O → Ca501 ≤ 3.2 Å, 도너 → β1-Asn224 backbone O ≤ 3.5 Å, **대칭보정** RMSD(`rdMolAlign.CalcRMS`) < 2.0 Å. 결정값은 2.62 / 2.63 Å.
- RMSD는 반드시 `CalcRMS`로. 인덱스 기반 좌표 뺄셈은 v1에서 0.63 Å 성공을 6.13 Å 실패로 보고했다(원자 순서가 보존되지 않고 분자에 대칭 페닐·다이클로로페닐 고리가 있다).
- **양성자화 결정:** 결정 접촉을 재현하는 쪽을 채택. 둘 다 재현하면 **음이온**을 택한다(AutoDock4에는 정전기 항이 있어 v1의 중성 선택 근거가 소멸).
- **미통과 시 AutoDock-GPU로 아무것도 거르지 말고** Uni-Dock 체제로 복귀, 그 사실을 기록.

### 엔진 재현성 측정 (사전등록)

```bash
python scripts/validate_redock.py --reproducibility \
    --ligands A1AFA PLN-1474 --seeds 1..10 \
    --out results/v2_engine_reproducibility.json
```

affinity가 tie-break가 아니라 **하드 필터**가 되었으므로 필요하다. v1에서는 리드 20개 중 5개가 컷오프로부터 0.05 kcal/mol 이내였고, 이는 프로젝트가 스스로 측정한 엔진 간 차이 0.16 kcal/mol보다 작았다. 리드 보고서는 나중에 "컷오프로부터 이 측정 잡음 이내에 몇 개가 있는가"를 반드시 밝혀야 한다.

---

## 4. 리간드 준비 → 도킹

**AutoDock-GPU는 SMILES가 아니라 PDBQT를 받는다.** `--ligands`는 준비된 PDBQT 경로 목록(한 줄에 하나)이다.

```bash
for ARM in TL-A-prime TL-C; do
  python scripts/prepare_ligands.py  --input data/v2_${ARM}/survivors.smi --out-dir docking/v2/ligands/${ARM}
  python scripts/ligands_to_pdbqt.py --in-dir docking/v2/ligands/${ARM}    --out-dir docking/v2/ligands_pdbqt/${ARM}
  ls docking/v2/ligands_pdbqt/${ARM}/*.pdbqt > docking/v2/ligand_index_${ARM}.txt

  python scripts/dock_autodock_gpu.py \
      --ligands docking/v2/ligand_index_${ARM}.txt \
      --maps    docking/v2/maps \
      --out     docking/v2/poses/${ARM} \
      --gpus 8
done
```

**포즈 파일을 반드시 보존할 것.** v1은 42,419개 포즈를 남기지 않아 어떤 기준도 소급 적용할 수 없었다.

**출력 형식 미검증 경고:** `parse_dlg`는 AutoDock4 계열의 `DOCKED:` 접두사와 `Estimated Free Energy of Binding` 필드를 가정한다. 바이너리가 로컬에 없어 확인하지 못했다. **대규모 실행 전에 실제 `.dlg` 하나를 열어 형식을 대조할 것.** 어긋나면 포즈 0개나 affinity 전부 `None`이 나오는데 파이프라인은 오류 없이 완주한다.

---

## 5. 복합체 변환 → PLIP (cpu64)

```bash
for ARM in TL-A-prime TL-C; do
  python scripts/poses_to_complex.py \
      --in  docking/v2/poses/${ARM} \
      --out docking/v2/complexes/${ARM}

  python scripts/run_plip.py \
      --in      docking/v2/complexes/${ARM} \
      --reports docking/v2/plip_reports/${ARM} \
      --out     results/v2_${ARM}/plip.csv \
      --jobs 64

  python scripts/pose_geometry.py \
      --poses    docking/v2/poses/${ARM} \
      --receptor docking/v2/receptor.pdbqt \
      --out      results/v2_${ARM}/geometry.csv
done
```

**PLIP은 `cpu64`에서 돌릴 것.** GPU 구현이 없다. `8gpu` 파티션은 GPU 사용 여부와 무관하게 시간당 8 node-hour를 과금하므로 거기서 돌리면 할당량만 태운다. 포즈당 약 1초이고 두 아암 합쳐 약 85,000 포즈이므로 64코어에서 30분대다.

주의: `plip`과 `obabel`이 기본 PATH에 없으면 **절대경로**를 쓸 것. PLIP은 XML을 `<stem>_report.xml`로 쓴다(`report.xml`이 아니다).

---

## 6. 여기서 멈춘다

`plip.csv`와 `geometry.csv`, 그리고 포즈·리포트 원본까지 만들어지면 이번 실행의 범위는 끝이다.

**아직 구현되지 않은 단계:** Task 14(거리 기록 전용화), 15(cpd 25 기반 게이트 확정), 16(게이트 적용), 17(ADMET·독성), 18(4단 필터·변경 대장), 19(아이소폼 역도킹·rank-sum), 20(리드 선정·성공 판정), 21(블루프린트 수정 21건).

**게이트 적용 전에 반드시 해결할 것:** §0의 금속 사이트 병합 결함.

---

## 로컬에서 이미 끝난 것 (다시 돌릴 필요 없음)

| 단계 | 결과 |
|---|---|
| TL-A′ 스윕 | `select_epoch` = **180**. `priors/focused_A_prime.prior`(gitignore, 로컬 전용) |
| TL-A′ vs TL-C 비교 | `results/v2_tl_vs_tlc.md` — 카복실산 8.9→54.1%, Arg-mimic 22.7→14.8%, **신규성 게이트 탈락 3.4→17.6%**, max NN 0.591→**1.000** |
| RL 두 아암 | 각 1,000스텝. 최종 점수 TL-A′ 0.91, TL-C 0.93 (총점은 **순환 지표**이므로 판정 근거로 쓰지 말 것) |
| 샘플링 | 아암당 약 19,800분자 |
| 후처리 | `data/v2_{arm}/library.smi`, `data/v2_{arm}/survivors.smi` |

`priors/`와 `*.chkpt`는 gitignore이므로 pull로 오지 않는다. K-BDS에서 RL을 다시 돌릴 필요는 없다 — 필요한 것은 `survivors.smi`뿐이고 그건 추적된다.
