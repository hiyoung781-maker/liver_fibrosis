# K-BDS 실행 런북 — v2 재실행

**대상 브랜치:** `avb1-rerun-v2`
**전제:** 로컬(TITAN Xp)에서 TL 스윕과 RL 두 아암, 샘플링까지 완료됨. K-BDS에서는 **도킹부터** 시작한다.
**상세 근거:** `results/v2_EXECUTION_CHECKLIST.md`(태스크별), `docs/superpowers/specs/2026-09-28-avb1-rerun-design.md`(설계)

---

## 0. 먼저 읽을 것 — 해결된 결함과 실행 환경

### PLIP 금속 사이트 분리 결함 — **해결됨** (커밋 `aa1155a`)

Task 13의 대조군이 찾았던 결함이다. PLIP은 배포된 8W30에서 리간드와 MIDAS 칼슘을 `A1A-CA`라는 하나의 복합 사이트로 묶지만, `poses_to_complex.py`가 만든 복합체에서는 `LIG` 사이트의 `metal_complexes`가 0건이고 칼슘이 자기 사이트로 분리됐다. 게이트 기준 (a) MIDAS 금속 배위가 변환기로 만든 **모든** 복합체에서 조용히 실패하는 상태였다.

원인은 화학이 아니라 파일 형식이었다. PLIP은 입력 PDB에 두 잔기를 잇는 **`LINK` 레코드**가 있을 때만 리간드와 금속을 하나의 복합 리간드로 묶는다(`structure/preparation.py:129`에서 LINK을 파싱하고 `identify_kmers`가 그것만으로 클러스터링한다). 변환기가 ATOM/HETATM만 남겼기 때문에 LINK이 사라졌고, 다른 사이트에 속한 리간드 원자는 그 사이트 금속의 배위 후보가 될 수 없으므로 2.62 Å 접촉이 **두 사이트 어디에도** 보고되지 않았다.

`poses_to_complex.py`가 이제 MIDAS LINK을 쓴다. 실측 대조:

| | `LIG` 사이트 | Ca B501 사이트 |
|---|---|---|
| LINK 없음 | `metal_complexes` **0건** | coordination 3 (Ser132 2.45, Ser134 2.50, Glu229 2.39) — 리간드 접촉 없음 |
| LINK 있음 | `LIG-CA`, coordination **4**, 리간드 접촉 **2.62 Å** | (병합됨) |

2.62 Å와 coordination 4는 배포 구조 자체의 값이다. 트리거 창(`MIDAS_LINK_CUTOFF = 4.0 Å`)은 상호작용 기준이 아니다 — PLIP 자체 `METAL_DIST_MAX`가 3.0 Å이므로, LINK은 리간드 원자를 후보로 **보이게** 할 뿐 PLIP이 독립적으로 인정하지 않을 배위를 만들어낼 수 없다.

**따라서 상호작용 게이트를 이번 실행에서 적용해도 된다.** §0의 이전 판에 있던 "게이트를 적용하지 말 것" 경고는 철회한다.

### 환경 — 명령마다 두 conda 환경이 갈린다

이 저장소의 도킹 후처리는 RDKit(같은 프로세스)과 Open Babel·PLIP(하위 프로세스)을 함께 쓰는데, 로컬에서는 둘이 서로 다른 환경에 있다. K-BDS에서도 같으면 아래 형태로 실행한다.

```bash
PATH=$HOME/miniconda3/envs/docking/bin:$PATH \
~/miniconda3/envs/reinvent/bin/python scripts/<script>.py ...
```

`autodock_gpu`는 계산 노드가 제공하는 것보다 새 `libstdc++`에 링크되어 있어 GLIBCXX 오류로 죽는다. `conda activate` **뒤에** 다음을 내보낸다(`$CONDA_PREFIX`를 설정하는 것이 activate이므로 순서가 중요하다).

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

`slurm/v2_dock.sbatch`에는 들어가 있지만 **대화형 셸에는 안 걸린다.** 로그인/계산 노드에서 `dock_autodock_gpu.py`를 직접 돌릴 때는 매번 내보내야 한다. 빠뜨리면 `preflight()`가 배치를 계획하기 전에 잡아서 이 줄을 그대로 알려준다.

**그리고 도킹 셸에서 후처리를 돌리면 안 된다.** 이 변수는 `autodock_gpu`에만 필요한데, 같은 셸의 **Open Babel을 깨뜨린다**. `_openbabel.so`는 구 C++ ABI로 빌드되어 있고, conda의 lib 디렉터리를 앞에 붙이면 로더가 신 ABI로 빌드된 `libopenbabel.so`를 먼저 찾아 구 ABI의 `std::string` 심볼이 사라진다:

```
ImportError: .../openbabel/_openbabel.so: undefined symbol:
  _ZN9OpenBabel8OBPlugin7DisplayERSsPKcS3_
```

(망글링의 `RSs`가 구 ABI `std::string`이다.) 실제로 이 상태에서 `validate_redock.py`가 두 번 죽었다. 후처리는 그 변수 없이 돌린다:

```bash
env -u LD_LIBRARY_PATH python scripts/validate_redock.py ...
```

`validate_redock.py`는 이제 시작할 때 `obabel -V`와 `plip -h`를 실제로 실행해보고, 실패했는데 `LD_LIBRARY_PATH`가 설정돼 있으면 이 사실을 알려준다. 도킹(GPU)과 후처리(CPU)는 애초에 다른 파티션에서 도는 작업이므로 셸을 나누는 것이 맞다.

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
`receptor_types`를 receptor PDBQT 파일에서 실제로 등장하는 타입을 직접 읽어 채운다
(`receptor_types()` 함수, `prepare_maps.py`). MIDAS 칼슘은 이 결합 자리에서 가장 중요한
접촉이고, AD4로 옮기는 이유의 절반이 AD4에 금속 파라미터가 있다는 것이다. 주의: PDBQT 상의
칼슘 타입은 `CA`가 아니라 `Ca`이고, autogrid4는 맵을 리간드 타입(`ligand_types`)마다 하나씩
만들지 receptor 타입마다 만들지 않으므로 `*.Ca.map` 같은 파일은 애초에 생기지 않는다 -- 칼슘의
기여는 각 리간드 타입 맵에 이미 녹아 있다. AD4 파라미터 파일에 `Ca`가 없으면 autogrid4가 unknown
atom type으로 죽는다. **맵 생성 직후 반드시 확인할 것:**

```bash
(cd docking/v2/maps && autogrid4 -p receptor.gpf -l receptor.glg)
grep -i "error\|unknown\|WARNING" docking/v2/maps/receptor.glg | head
grep receptor_types docking/v2/maps/receptor.gpf   # Ca가 포함되어 있는지 확인
```

`receptor_types`에 `Ca`가 없거나 autogrid4가 unknown atom type 에러로 죽으면, GPF에
`parameter_file AD4_parameters.dat` 를 추가하고 그 파일을 맵 디렉터리에 두거나, 파라미터
파일에 Ca 항목을 확인한다. **`receptor_types`에 `Ca`가 없는 채로 진행하면 금속 상호작용이 아예
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

대조군 `CONTROL_crystal`은 **이미 일반 배치에 들어가 도킹됐다**(각 아암의 리간드 인덱스에 들어 있다). 게이트는 그 `.dlg`를 읽는다. 여기서 다시 도킹하면 캠페인이 실제로 쓴 실행이 아닌 다른 실행을 판정하게 되므로 그렇게 하지 않는다.

```bash
PATH=$HOME/miniconda3/envs/docking/bin:$PATH \
~/miniconda3/envs/reinvent/bin/python scripts/validate_redock.py \
    --dlg       docking/v2/poses/TL-A-prime/CONTROL_crystal.dlg \
    --receptor  docking/v2/receptor_h.pdb \
    --reference docking/ligand_ref.sdf \
    --plip-bin  $HOME/miniconda3/envs/docking/bin/plip \
    --workdir   docking/v2/redock_gate \
    --out       results/v2_redock_gate.md
```

마크다운 리포트와 같은 이름의 JSON이 함께 쓰인다. 통과하면 종료 코드 0, 미통과면 1이다.

**사전등록된 판정 (결과를 보고 바꾸지 말 것)**
- 통과 조건 네 가지: 카복실레이트 O → Ca501 ≤ 3.2 Å, 도너 → β1-Asn224 backbone O ≤ 3.5 Å, **대칭보정** RMSD(`rdMolAlign.CalcRMS`) < 2.0 Å, 그리고 PLIP이 재도킹 포즈에서 Ca501 금속 배위와 Asn224 수소결합을 **둘 다** 보고할 것. 결정값은 2.62 / 2.63 Å.
- 게이트는 **affinity 최저 포즈**에 적용된다. AutoDock-GPU는 DOCKED 블록을 에너지 순이 아니라 **run 순**으로 쓴다 — 대조군 `.dlg`는 −5.89, −6.23, −6.30, −6.30, −6.02로 시작하므로 첫 블록은 앞 다섯 중 최악이다. `best_pose()`가 명시적으로 정렬한다.
- RMSD는 반드시 `CalcRMS`로. 인덱스 기반 좌표 뺄셈은 v1에서 0.63 Å 성공을 6.13 Å 실패로 보고했다(원자 순서가 보존되지 않고 분자에 대칭 페닐·다이클로로페닐 고리가 있다).
- 거리와 PLIP이 **어긋나면** 리포트의 "Distance/PLIP disagreement" 절에 기록되고, spec §7.4대로 PLIP 판정을 따른다. 거리는 각도를 못 보고 PLIP은 보기 때문이다.
- **양성자화 결정:** 결정 접촉을 재현하는 쪽을 채택. 둘 다 재현하면 **음이온**을 택한다(AutoDock4에는 정전기 항이 있어 v1의 중성 선택 근거가 소멸). `--protonation` 은 어느 상태를 도킹했는지 리포트에 **기록**만 한다 — 상태 자체는 리간드 준비 단계에서 결정된다.
- **미통과 시 AutoDock-GPU로 아무것도 거르지 말고** Uni-Dock 체제로 복귀, 그 사실을 기록.

### 양성자화 두 상태 — 게이트는 두 번 돈다 (사전등록)

spec §7.4는 재도킹을 **중성과 음이온 두 상태로** 등록했고, 어느 쪽을 채택할지도 미리 정해뒀다: 결정 접촉을 재현하는 쪽, 둘 다 재현하면 **음이온**. 근거는 엔진 교체 자체다 — v1이 중성을 고른 건 Vina에 정전기 항이 없어 전하가 점수에 영향을 못 줬기 때문인데, AutoDock4에는 정전기 항이 있고 MIDAS 이온은 **Ca²⁺**다. 카복실레이트의 음전하가 이 결합 양식의 지배적 정전기 동인이므로, 중성 COOH로 도킹하면 그게 통째로 빠진다.

v2 배치는 **중성**으로 돌았다(`docking/ligand_ref.sdf`에 `M  CHG` 레코드가 없고, `prepare_ligands.py --control`의 기본값이 그 파일이다). 음이온 쪽은 아래로 만든다.

```bash
# 결정 좌표를 그대로 두고 형식전하만 바꾼다 (원자 변위 0.000000000 Å)
python scripts/deprotonate_acids.py

python scripts/prepare_ligands.py --control-only \
    --control docking/ligand_ref_anion.sdf \
    --out-dir docking/v2/ligands_anion

python scripts/ligands_to_pdbqt.py \
    --sdf-dir docking/v2/ligands_anion \
    --out-dir docking/v2/ligands_pdbqt_anion

ls docking/v2/ligands_pdbqt_anion/*.pdbqt > docking/v2/control_anion_index.txt

python scripts/dock_autodock_gpu.py \
    --ligands docking/v2/control_anion_index.txt \
    --maps docking/v2/maps --out docking/v2/poses/control_anion \
    --gpus 1 --seed 42

python scripts/validate_redock.py --protonation anion \
    --dlg       docking/v2/poses/control_anion/CONTROL_crystal.dlg \
    --receptor  docking/v2/receptor_h.pdb \
    --reference docking/v2/ligands_pdbqt_anion/CONTROL_crystal.pdbqt \
    --workdir   docking/v2/redock_gate_anion \
    --all-poses \
    --out       results/v2_redock_gate_anion.md
```

`--reference`는 **SDF가 아니라 준비된 대조군 PDBQT**를 준다. 도킹 포즈는 `PDBQT → Open Babel → SDF` 경로로 RDKit에 도착하고 Open Babel이 그 과정에서 양성자화를 다시 인식하는데, 참조를 SDF로 주면 두 쪽의 인식이 갈린다. 실측: 음이온 대조군의 참조 SDF는 `...C(=O)[O-]`로, 그 자신의 도킹 포즈는 `...C(=O)O`로 읽혔다. 부분구조 일치 0건인데 `CalcRMS`가 숫자를 돌려줬다 — 2.151 Å, AutoDock 자신의 reference RMSD는 같은 포즈에 1.46 Å.

`CalcRMS`는 대칭 등가인 모든 매핑 중 **최솟값**을 고르므로 올바른 대응의 RMSD를 **넘을 수 없다**. 넘었다면 올바른 대응을 못 찾은 것이다. 참조를 PDBQT로 주면 양쪽이 같은 변환기를 거쳐 인식이 같아지고, `.dlg` 포즈는 입력 PDBQT의 원자 순서를 그대로 갖고 있으므로 대응이 엔진 자신의 것이 된다.

이제 그래프가 다르면 `validate_redock.py`는 숫자를 내지 않고 두 SMILES를 찍으며 기준 (c)를 **실패로 기록**한다. 측정되지 않은 값이 사전등록 기준을 결정하는 일은 없다.

**이건 사후에 기준을 바꾸는 게 아니라 등록된 절차를 마저 실행하는 것이다.** 문턱 2.0 Å은 그대로다.

### 엔진 재현성 측정 (사전등록)

시드를 바꿔 대조군을 다시 도킹한 뒤, 시드별 `.dlg`를 한꺼번에 넘긴다.

```bash
# 대조군 한 줄만 뽑아 인덱스를 만든다 (--ligands 는 PDBQT 경로 목록이다)
grep CONTROL_crystal docking/v2/ligand_index_TL-A-prime.txt > docking/v2/control_index.txt

for S in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/dock_autodock_gpu.py \
      --ligands docking/v2/control_index.txt --maps docking/v2/maps \
      --out docking/v2/repro/seed_$S --gpus 1 --seed $S
done

PATH=$HOME/miniconda3/envs/docking/bin:$PATH \
~/miniconda3/envs/reinvent/bin/python scripts/validate_redock.py --reproducibility \
    --dlg docking/v2/repro/seed_*/CONTROL_crystal.dlg \
    --out results/v2_engine_reproducibility.md
```

affinity가 tie-break가 아니라 **하드 필터**가 되었으므로 필요하다. v1에서는 리드 20개 중 5개가 컷오프로부터 0.05 kcal/mol 이내였고, 이는 프로젝트가 스스로 측정한 엔진 간 차이 0.16 kcal/mol보다 작았다. 리드 보고서는 나중에 "컷오프로부터 이 측정 잡음 이내에 몇 개가 있는가"를 반드시 밝혀야 한다.

---

## 3b. 게이트 미통과 — Uni-Dock 복귀 (2026-09-29 결정)

AutoDock-GPU는 두 양성자화 상태 모두 기준 (c)에서 실패했다(RMSD 2.15 Å, 문턱 2.0 Å). (a)(b)(d)는 통과. 근거와 검증 경로는 `results/v2_redock_gate_decision.md`에 있다. spec §7.4의 사전등록대로 **AutoDock-GPU로 아무것도 순위 매기거나 거르지 않고** Uni-Dock 체제로 복귀한다.

이미 만든 AD4 포즈(TL-A′ 4,514, TL-C 5,212)와 PLIP 레코드는 **지우지 않는다.** 리드 선정에 쓰지 않을 뿐, 두 엔진의 기하 게이트 통과율 비교는 보고 가치가 있다.

### 3b.1 Uni-Dock에도 같은 게이트를 적용한다 — 건너뛰지 말 것

v1의 0.63 Å은 **smina**로 측정한 값이다. Uni-Dock v1.1.3은 `--scoring` 기본값이 `vina`로 점수함수는 같지만 구현과 탐색(GPU 병렬 Monte Carlo)이 다르다. smina 숫자를 근거로 Uni-Dock을 채택하는 것은, 이번에 AD4에서 저지를 뻔한 비약을 반대 방향으로 반복하는 것이다.

```bash
# 대조군 두 양성자화 상태를 Uni-Dock으로 재도킹
for STATE in neutral anion; do
  case $STATE in
    neutral) IDX=docking/v2/control_index.txt ;;
    anion)   IDX=docking/v2/control_anion_index.txt ;;
  esac
  python scripts/dock_unidock.py \
      --receptor        docking/v2/meeko_receptor.pdbqt \
      --ligand-index    $IDX \
      --out-dir         docking/v2/unidock/control_$STATE \
      --autobox-ligand  docking/ligand_ref.sdf \
      --gpus 1
done

# 같은 기준, 같은 문턱. --dlg 대신 --poses 로 Uni-Dock 출력을 읽는다.
for STATE in neutral anion; do
  case $STATE in
    neutral) REF=docking/v2/ligands_pdbqt/TL-A-prime/CONTROL_crystal.pdbqt ;;
    anion)   REF=docking/v2/ligands_pdbqt_anion/CONTROL_crystal.pdbqt ;;
  esac
  env -u LD_LIBRARY_PATH python scripts/validate_redock.py --protonation $STATE \
      --poses     docking/v2/unidock/control_$STATE/CONTROL_crystal_out.pdbqt \
      --receptor  docking/v2/receptor_h.pdb \
      --reference $REF \
      --workdir   docking/v2/unidock_gate_$STATE \
      --all-poses \
      --out       results/v2_redock_gate_unidock_$STATE.md
done
```

수용체·인덱스 경로는 실제 위치에 맞춘다. 대조군 인덱스가 없으면 아암 인덱스에서 뽑는다:

```bash
grep CONTROL_crystal docking/v2/ligand_index_TL-A-prime.txt > docking/v2/control_index.txt
```

**판정 분기 — 결과를 보기 전에 정해 둔다.**

- **통과** → 그 양성자화 상태로 3b.2를 진행한다. 둘 다 통과하면 **음이온**을 택한다 — spec §7.4의 원문 그대로다.

  (이 런북의 이전 판은 여기에 "중성"이라고 적었다. 근거로 든 *"§7.4가 음이온을 선호한 이유는 AutoDock4에 정전기 항이 있어서이고 Vina에는 없다"*가 정확하지 않았다. §7.4는 **두 가지** 이유를 적었고 — *"pH 7.4의 실제 존재 형태이고, 점수함수가 이제 전하를 본다"* — 앞의 것은 엔진과 무관하다. 게다가 Vina에 명시적 쿨롱 항이 없어도 두 상태의 점수는 실제로 갈린다: 카복실레이트가 되면 COOH 회전결합이 사라져 비틀림 수가 6에서 5로 줄고 수소결합 원자 유형이 바뀐다(중성 −6.718, 음이온 −6.913). 원문 규칙을 뒤집을 근거가 성립하지 않으므로 원문을 따른다.)
- **미통과** → **두 엔진 모두 사전등록 게이트에 실패했다는 뜻이다.** 그 자체가 이 캠페인의 주요 결과이며, 리드를 선정하지 않고 그 사실을 보고한다. 문턱을 옮기지 않는다.

### 3b.2 라이브러리를 음이온으로 다시 준비

**기존 PDBQT를 그대로 쓰면 안 된다.** 게이트가 채택한 상태는 음이온인데, 생성 리간드는 중성 카복실산으로 준비되어 있다(생존자 PDBQT의 `REMARK SMILES`가 `...C(=O)O...`). 대조군이 검증한 것과 다른 상태로 라이브러리를 도킹하면 프로토콜 검증이 그 도킹에 적용되지 않는다.

재임베딩이 아니다 — 형식전하 편집과 산성 수소 제거뿐이고 **중원자 좌표는 바뀌지 않는다**(테스트가 0 Å으로 고정). 분자 하나에 카복실산이 여럿이면 전부 탈양성자화한다(pH 7.4의 실제 상태). 카복실산이 없는 분자는 그대로 통과시키고 세어 둔다.

```bash
for ARM in TL-A-prime TL-C; do
  python scripts/deprotonate_acids.py \
      --sdf-dir docking/v2/ligands/${ARM} \
      --out-dir docking/v2/ligands_anion/${ARM}

  python scripts/ligands_to_pdbqt.py \
      --sdf-dir docking/v2/ligands_anion/${ARM} \
      --out-dir docking/v2/ligands_pdbqt_anion/${ARM}

  ls docking/v2/ligands_pdbqt_anion/${ARM}/*.pdbqt > docking/v2/ligand_index_anion_${ARM}.txt
done
```

`deprotonated + unchanged + failed = read`가 맞는지, `failed`가 0인지 확인한다. `ligands_to_pdbqt.py`가 인덱스(`.../ligands.txt`)를 직접 쓰므로 `ls > ...`는 필요 없다.

패널은 `dock_panel.py`가 SMILES에서 준비·도킹까지 한다. **기본값이 탈양성자화**로 바뀌었다 — 그 함수의 존재 이유가 *"도킹 결과 차이가 리간드 준비 차이에서 올 수 없게 한다"*이고, 라이브러리가 음이온이므로 패널도 음이온이어야 그 불변식이 성립한다. §9.1의 affinity 필터는 PLN-1474 **상대** 비교라서, 기준과 후보의 전하 상태가 다르면 비교 자체가 성립하지 않는다.

### 3b.3 두 아암 재도킹 + affinity 기준값 재산출

```bash
for ARM in TL-A-prime TL-C; do
  python scripts/dock_unidock.py \
      --receptor       docking/v2/meeko_receptor.pdbqt \
      --ligand-index   docking/v2/ligand_index_anion_${ARM}.txt \
      --out-dir        docking/v2/unidock/${ARM} \
      --autobox-ligand docking/ligand_ref.sdf \
      --gpus 8
done
```

**패널도 같은 실행에 포함해야 한다.** §9.1의 affinity 하드 필터는 PLN-1474 상대값이고, 그 기준값은 **같은 엔진·같은 박스**에서 나와야 한다. AD4로 얻은 어떤 값도 Vina 점수에 쓸 수 없다.

```bash
python scripts/dock_panel.py \
    --receptor        docking/v2/meeko_receptor.pdbqt \
    --autobox-ligand  docking/ligand_ref.sdf \
    --ligand-dir      docking/v2/panel_pdbqt_anion \
    --pose-dir        docking/v2/unidock/panel \
    --out-sdf         docking/v2/unidock/panel_poses.sdf \
    --out-csv         results/v2_panel_geometry.csv
```

`--no-deprotonate`는 캠페인이 중성 상태를 채택할 때만 쓴다(중성 아암도 0.66 Å으로 통과했으므로 가능한 선택이지만, 그 경우 라이브러리도 같이 되돌려야 한다).

이후 §5(복합체 변환 → PLIP)와 하류 단계는 입력 디렉터리만 `docking/v2/unidock/...`로 바꿔 그대로 진행한다.

### 3b.4 기록에 반드시 남길 것

기하 기준(§8.5b)이 **결정 포즈와 2.15 Å 대체 포즈를 구별하지 못했다.** 2.15 Å 포즈도 Ca501 2.51 Å, Asn224 2.81 Å로 기준 안에 있었고 PLIP도 두 상호작용을 모두 보고했다. 이 한계는 엔진을 바꿔도 사라지지 않으며, 기하 게이트 통과가 결정 결합 양식의 재현을 보장하지 않는다는 뜻이다. 리드 보고서에 명시한다.

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

**이 단계는 완료됐다.** 실제 바이너리(v1.6-20-gbe06a13)에서 8샤드로 돌아 `TL-A-prime` 4,514개, `TL-C` 5,212개의 `.dlg`가 나왔고 모든 샤드 로그가 `All jobs (N) ran without errors.`로 끝났다. `parse_dlg`의 형식(`DOCKED:` 접두사와 `Estimated Free Energy of Binding`)도 실제 출력에서 확인됐다.

`grep -i error` 로 로그를 검사하지 말 것 — 성공 메시지 안의 "without **errors**"에 걸린다. `grep "^Error:"` 를 쓴다.

---

## 5. 기하 필터 → PLIP (cpu64)

**§4의 AutoDock-GPU 경로는 폐기됐다(§3b).** 아래는 Uni-Dock 출력 기준이다.

### 5.1 포즈 수 검수 — 파일 개수를 세지 말 것

Uni-Dock은 `MODEL` 블록이 없는 파일을 쓰고도 exit 0으로 끝난다. 포즈 0개인 리간드는 **도킹 실패**이지 필터 탈락이 아니며, 둘을 섞으면 통과율의 분모가 틀어진다.

```bash
for ARM in TL-A-prime TL-C; do
  case $ARM in TL-A-prime) WANT=4514 ;; TL-C) WANT=5212 ;; esac
  python scripts/pose_census.py --pose-dir docking/v2/unidock/$ARM --expected $WANT \
      --out-csv results/v2_${ARM}/pose_census.csv \
      --out-json results/v2_${ARM}/pose_census.json
done
```

측정값(2026-09-29): TL-A′ 중앙값 3개·평균 4.10, TL-C 중앙값 6개·평균 6.62, 포즈 0개 리간드 없음. **두 아암의 포즈 수가 계통적으로 다르므로** 기하 게이트 통과율은 리간드 단위와 포즈 수 층화 두 가지로 보고한다 — 근거는 `results/v2_redock_gate_decision.md`.

### 5.2 기하 필터 (§8.5b)

```bash
for ARM in TL-A-prime TL-C; do
  python scripts/poses_to_sdf.py \
      --pose-dir docking/v2/unidock/${ARM} \
      --out-sdf  docking/v2/unidock/${ARM}_poses.sdf

  python scripts/pose_geometry.py \
      --poses    docking/v2/unidock/${ARM}_poses.sdf \
      --receptor docking/v2/receptor_h.pdb \
      --out-csv  results/v2_${ARM}/geometry.csv \
      --out-sdf  docking/v2/unidock/${ARM}_passing.sdf
done
```

`--receptor` 기본값은 v1의 `docking/receptor.pdb`다. **반드시 명시할 것.**

### 5.3 PLIP — 포즈 단위, cpu64

포즈마다 복합체를 만들고 PLIP을 돌려 한 줄씩 쓴다. 약 53,000 포즈.

```bash
# 로그인 노드에서, 저장소 안에서 제출한다 (가이드 p.19)
ssh bdata-login01
cd ~/liver_fibrosis && mkdir -p logs
sbatch slurm/v2_plip.sbatch

squeue -u $USER          # 진행 확인
```

**제출 위치가 중요하다.** 스크립트 경로는 제출한 디렉터리 기준이다 — 홈에서 제출하면 `Unable to open file slurm/v2_plip.sbatch`가 난다. sbatch가 없다는 뜻이 아니다.

**직접 실행하지 말 것.** 로그인 노드에서 `--jobs 64`로 맨손 실행했더니 모든 워커의 `plip`이 SIGINT로 죽었다(`exited -2`, numpy import 중 `KeyboardInterrupt`). 스케줄러 정책이지 도킹이나 화학의 실패가 아니다.

**파티션은 `cpu64-only`다.** 가이드 p.18의 큐 표에 CPU 큐는 `cpu32-only`(0.5 노드시간)와 `cpu64-only`(1 노드시간)뿐이고 **`cpu64`는 없다.** `8gpu`는 GPU를 안 써도 시간당 8 노드시간을 과금하므로 PLIP을 거기서 돌리면 할당량만 태운다. 모든 큐가 배타적 노드 정책이라 `--exclusive`는 불필요하다.

sbatch 스크립트가 `LD_LIBRARY_PATH`를 `unset`하고 `plip -h`·`obabel -V`를 실제로 실행해본 뒤 시작한다.

`plip`과 `obabel`이 기본 PATH에 없으면 `--plip-bin`에 절대경로를 준다. **`LD_LIBRARY_PATH`가 설정된 셸에서 돌리지 말 것** — §0의 Open Babel ABI 충돌.

CSV에는 §8.0의 네 기준(`metal_ca501`, `hbond_asn224`, `tyr178_contact`, `hydrophobic_pocket`)과 **상호작용 종류별 원시 개수**가 함께 들어간다. cpd 25 기반 게이트 결정이 아직 나지 않았고, **기록하지 않은 기준은 나중에 적용할 수 없기 때문이다** — v1이 포즈 42,419개를 버려 어떤 기준도 소급 적용하지 못한 것이 이 규칙의 출처다.

PLIP이 처리하지 못한 포즈는 `status` 열에 사유를 달고 **CSV에 남는다.** 빼버리면 하류의 모든 통과율이 조용히 분모를 잃는다.

## 6. 여기서 멈춘다

`plip.csv`와 `geometry.csv`, 그리고 포즈·리포트 원본까지 만들어지면 이번 실행의 범위는 끝이다.

**아직 구현되지 않은 단계:** Task 14(거리 기록 전용화), 15(cpd 25 기반 게이트 확정), 16(게이트 적용), 17(ADMET·독성), 18(4단 필터·변경 대장), 19(아이소폼 역도킹·rank-sum), 20(리드 선정·성공 판정), 21(블루프린트 수정 21건).

**해결됨:** §0의 금속 사이트 병합 결함(커밋 `aa1155a`)과 재도킹 게이트 CLI 미구현(커밋 `6c03fea`). 게이트 적용을 막는 것은 이제 없다.

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
