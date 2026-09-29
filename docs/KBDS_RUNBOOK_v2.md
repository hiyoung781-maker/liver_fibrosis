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

### 먼저 확인할 것 — 기존 환경에 무엇이 있는지

로컬 머신에는 `docking` 환경의 `smina` 2020.12.10 과 `obabel` 3.2.1 만 있고, `unidock`·
`autodock_gpu`·`autogrid4` 는 **어느 conda 환경에도, 파일시스템 어디에도 없다**(확인함).
v1의 도킹은 K-BDS에서 돌았으므로(§8.5.1, §11) **K-BDS에는 `unidock` 환경이 있을 수 있다.**

```bash
conda env list
for t in unidock autodock_gpu autogrid4 smina obabel plip; do printf '%-14s ' "$t"; command -v $t || echo ABSENT; done
```

- **`unidock`이 있으면 폴백 경로가 즉시 확보된다.** §3의 사전등록 중단 조항 — AutoDock-GPU
  재도킹 검증 실패 시 Uni-Dock 체제로 복귀 — 을 바로 실행할 수 있다. `scripts/dock_unidock.py`가
  그대로 남아 있다.
- **그러나 Uni-Dock은 `autogrid4`/`autodock_gpu`를 대체하지 못한다.** Uni-Dock은 Vina 점수함수를
  쓰고 그리드 맵이 없다. v2가 AutoDock4로 옮기는 이유가 **AD4에는 금속 파라미터와 정전기 항이
  있다**는 것이고, v1의 최대 결함이 "Vina에는 금속 배위 항이 전혀 없어 MIDAS 카복실레이트가
  점수함수에 보이지 않는다"는 것이었다. 그건 Uni-Dock으로 얻을 수 없다.
- 따라서 `unidock`이 있어도 `autogrid4`와 `autodock_gpu`는 **따로 설치해야 한다.**

### AutoDock4 계열 설치 — 걸릴 함정 세 가지를 먼저 읽을 것

`unidock`·`unidocktools`·Meeko(`mk_prepare_ligand.py`)·Open Babel은 K-BDS에 이미 있다.
없는 것은 `autogrid4`, `autodock_gpu`, `plip`이다.

```bash
# PLIP — 쉽다
pip install plip

# autogrid4 — AutoDock4 배포판의 정적 바이너리. 컴파일 불필요.
#   https://autodock.scripps.edu/download-autodock4/  (Linux x86_64 tarball)
#   tarball 안에 autogrid4, autodock4, AD4_parameters.dat, AD4.1_bound.dat 가 있다.
#   conda 채널에 있으면 그쪽이 더 편하다: conda search -c bioconda autodock

# AutoDock-GPU — CUDA 컴파일
#   https://github.com/ccsb-scripps/AutoDock-GPU
#   export GPU_INCLUDE_PATH=$CUDA_HOME/include
#   export GPU_LIBRARY_PATH=$CUDA_HOME/lib64
#   make DEVICE=CUDA NUMWI=64
```

**함정 1 — 바이너리 이름이 `autodock_gpu`가 아니다.** AutoDock-GPU의 Makefile은 work-item
수를 이름에 붙여 `bin/autodock_gpu_64wi` 같은 파일을 만든다. `scripts/dock_autodock_gpu.py`의
`build_command`는 `autodock_gpu`를 호출하므로, 빌드 후 심볼릭 링크를 걸거나 PATH에 그 이름으로
노출해야 한다. 안 하면 `shutil.which("autodock_gpu")` 검사에서 "설치 안 됨"으로 걸린다.

```bash
ln -s "$(pwd)/bin/autodock_gpu_64wi" ~/bin/autodock_gpu   # 또는 동등한 위치
command -v autodock_gpu   # 나와야 한다
```

**함정 2 — autogrid4가 칼슘 원자 타입을 모를 수 있다.** `scripts/prepare_maps.py`는
`receptor_types`에 `CA`를 넣는다(MIDAS 칼슘은 이 결합 자리에서 가장 중요한 접촉이고, AD4로
옮기는 이유의 절반이 AD4에 금속 파라미터가 있다는 것이다). AD4 파라미터 파일에 `Ca`가 없으면
autogrid4가 unknown atom type으로 죽는다. **맵 생성 직후 반드시 확인할 것:**

```bash
(cd docking/v2/maps && autogrid4 -p receptor.gpf -l receptor.glg)
grep -i "error\|unknown\|WARNING" docking/v2/maps/receptor.glg | head
ls docking/v2/maps/*.CA.map      # 칼슘 맵이 실제로 생겼는지
```

칼슘 맵이 없으면 GPF에 `parameter_file AD4_parameters.dat` 를 추가하고 그 파일을 맵 디렉터리에
두거나, 파라미터 파일에 Ca 항목을 확인한다. **칼슘 맵 없이 진행하면 금속 상호작용이 아예
계산되지 않으므로, AD4로 바꾼 이유가 사라진다.**

**함정 3 — `.dlg` 출력 형식이 미검증이다.** `parse_dlg`는 `DOCKED:` 접두사와
`Estimated Free Energy of Binding` 필드를 가정한다. 로컬에 바이너리가 없어 확인하지 못했다.
**리간드 하나를 도킹해 실물을 열어보고 대조할 것. 이것을 건너뛰지 말 것.**

```bash
# 리간드 하나만 시험 도킹
autodock_gpu --ffile docking/v2/maps/receptor.maps.fld \
             --lfile <ligand>.pdbqt --resnam /tmp/probe --nrun 20 --seed 42
head -60 /tmp/probe.dlg                     # DOCKED: 접두사가 있는가
grep -c "Estimated Free Energy of Binding" /tmp/probe.dlg   # 20이어야 한다

# 파서가 실제로 그 파일을 읽는지
python - <<'EOF'
import sys; sys.path.insert(0, "scripts")
from dock_autodock_gpu import parse_dlg
poses = parse_dlg(open("/tmp/probe.dlg").read())
print(f"parsed {len(poses)} poses; affinities: {[p['affinity'] for p in poses[:3]]}")
assert len(poses) > 0, "파서가 형식을 못 읽는다 — 진행 전에 고칠 것"
assert poses[0]["affinity"] is not None, "에너지 필드를 못 찾는다"
EOF
```

형식이 다르면 `--xmloutput 1` 경로를 쓰거나 파서를 실제 형식에 맞춰야 한다. **형식이 어긋난
채로 대규모 실행하면 포즈 0개나 affinity 전부 `None`이 나오는데 파이프라인은 오류 없이
완주한다** — 이 프로젝트에서 세 번 일어난 실패 양식이다(5자 CCD 잔기명이 PDB 열을 넘겨 affinity를
전부 0으로 만든 것, 나이브 RMSD가 0.63 A 성공을 6.13 A 실패로 보고한 것, Open Babel 질소
타이핑이 4위 포즈를 1위로 올린 것). 공통점은 전부 그럴듯한 숫자를 냈다는 것이다.

**Uni-Dock은 지우지 말 것.** §3의 사전등록 중단 조항이 AutoDock-GPU 재도킹 검증 실패 시
Uni-Dock 복귀이고, `scripts/dock_unidock.py`가 그대로 있다.

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

### 도킹 전 단계의 아암 비교 — 이미 측정됨

| | TL-A′ | TL-C |
|---|---|---|
| 라이브러리 | 19,773 | 19,786 |
| 카복실레이트 | 19,550 (98.9%) | 19,493 (98.5%) |
| alert 무플래그 | 19,130 | 19,134 |
| **속성 윈도 통과 = survivors** | **4,513 (22.8%)** | **5,213 (26.3%)** |
| Arg-mimic (survivors) | 137 (3.0%) | 214 (4.1%) |
| 스캐폴드가 공개 106종에 속함 | 0 (0.0%) | 0 (0.0%) |
| 고유 스캐폴드 (survivors) | 4,159 | 4,906 |

**RL 이후 두 아암은 거의 구별되지 않는다.** prior 단계에서 카복실산 밀도가 54.1% 대 8.9%로
6.1배 벌어져 있었으나 RL 1,000스텝 뒤에는 98.9% 대 98.5%다 — 목적함수가 카복실레이트를 하드
게이트하므로 RL이 스스로 그 자리에 도달한다. Arg-mimic 편향도 prior의 −7.9 pp에서 −1.1 pp로
줄었다. 반대로 속성 윈도 통과율은 TL-C가 앞선다(26.3% 대 22.8%) — TL이 밀어올린 TPSA
(72.8 → 102.2)를 RL이 완전히 되돌리지 못했다.

**그래서 현재 판정은 TL-C를 주 아암으로 두고 TL-A′는 측정된 대조군으로 보고하는 것이다.**
다만 사전등록 규칙이 기하 통과율과 PLN-1474 대비 투과성까지 보기로 했으므로, **두 아암을
모두 도킹한다.** 여기서 확정하면 사후 변경이 된다. 추가 비용은 아암당 약 20분이다.

### 토토머 정규화 (질문이 나왔으므로 측정해 둠)

정규화로 표현이 바뀐 분자는 TL-A′ 275/19,773(1.4%), TL-C 358/19,786(1.8%)이고, 그중 Murcko
스캐폴드가 달라진 것이 258/336이다. **그중 신규성 판정이 뒤집힌 것은 0건**이다. 이번 라이브러리는
공개 106 스캐폴드에서 충분히 멀어 신규성 주장이 표현 선택에 의존하지 않는다 — 이것은 정규화가
낭비였다는 뜻이 아니라 신규성 주장의 독립적 근거다. 정규화는 유지한다(참조 측은 ChEMBL 유래로
일관된 방향족 형태인데 생성 측만 흔들리는 비대칭이 본래 이유이고, 0건은 이번 실행의 성질이지
보장이 아니다).

`priors/`와 `*.chkpt`는 gitignore이므로 pull로 오지 않는다. K-BDS에서 RL을 다시 돌릴 필요는 없다 — 필요한 것은 `survivors.smi`뿐이고 그건 추적된다.
