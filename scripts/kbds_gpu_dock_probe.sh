#!/bin/bash
# Read-only probe for GPU docking on K-BDS. Installs nothing; writes only its
# report file and a scratch dir it deletes on exit.
#
# Written against KBDS_연구환경사용자가이드 v12 (2025-12). Facts taken from it:
#   - cpu64-only : 5 nodes, 64 core, 1024 GB, charged 1 node-hour per wall hour
#   - 8gpu       : 3 nodes, 128 core, 896 GB, A100 80G x8, charged 8
#   - 1gpu       : 5 nodes,   8 core,  64 GB, A100 40G x1, charged 1
#   - debug-1gpu : 8-hour limit, "컴파일과 디버깅 용도" -> build here, charged 1
#   - wall clock : 5 days everywhere; every queue is exclusive-node
#   - /scratch/tools and /scratch/database hold pre-installed tools and DBs
#   - `conda pack` is documented for "외부 인터넷이 연결되지 않는 경우", so
#     internet on a compute node cannot be assumed - section 7 tests it
#
# RUN ON THE COMPUTE NODE. The nvcc run test and the OpenCL query need a real GPU:
#   [login]$ srun --jobid=<your running 8gpu job> --pty bash
#   [gpu-8-002]$ cd <repo> && bash scripts/kbds_gpu_dock_probe.sh
# or, cheaper and purpose-built for exactly this:
#   [login]$ srun -p debug-1gpu --gres=gpu:1 -t 1:00:00 --pty bash
#
# Goal: decide between Vina-GPU 2.0 (OpenCL) and Uni-Dock (CUDA). Both keep
# AutoDock Vina's scoring function, so the 0.63 A redocking result already recorded
# with smina still characterizes the method; this probe only looks for a toolchain.

set -u
OUT="${PWD}/kbds_gpu_dock_probe_report.txt"
exec > >(tee "$OUT") 2>&1

SCRATCH="${TMPDIR:-/tmp}/gpudock_probe_$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

hr() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((62 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "K-BDS GPU docking probe   $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "host=$(hostname)  user=$(id -un)  job=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-unknown}  pwd=$(pwd)"
echo "report -> $OUT"

hr "0. Am I on a compute node with a GPU?"
case "$(hostname)" in
  *login*) echo "  !! LOGIN NODE. Sections 4 and 2 will be meaningless. Rerun via srun." ;;
  *) echo "  compute node: $(hostname)" ;;
esac
if have nvidia-smi && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
             --format=csv 2>/dev/null
else
  echo "  no GPU visible"
fi

hr "1. Cores: is nproc physical or SMT?"
echo "nproc            : $(nproc)"
lscpu | grep -E '^Model name|^CPU\(s\):|^Thread\(s\) per core|^Core\(s\) per socket|^Socket\(s\)' || true
cat <<'NOTE'
  The guide's table: cpu64-only = 64 core, 8gpu node = 128 core. On an EPYC-7713
  (64 physical cores) an nproc of 128 means SMT, i.e. 64 physical. smina is
  compute-bound and gains little from SMT, so the CPU fallback costs
  1,180 CPU-hours / 64 = ~18.4 h, not the 9.2 h a 128 reading would suggest.
NOTE

hr "2. OpenCL -- decides whether Vina-GPU 2.0 is possible"
echo "-- ICD registry --"
ls -la /etc/OpenCL/vendors/ 2>/dev/null || echo "  ABSENT: no /etc/OpenCL/vendors"
echo "-- libOpenCL --"
ldconfig -p 2>/dev/null | grep -i opencl || echo "  ldconfig: none"
for p in /usr/lib64/libOpenCL.so{,.1} /usr/local/cuda/lib64/libOpenCL.so; do
  [ -e "$p" ] && echo "  found: $p"
done
echo "-- OpenCL headers (needed to compile) --"
find /apps /usr/include /usr/local -maxdepth 6 -name 'cl.h' -path '*CL*' \
     2>/dev/null | head -5 || true
have clinfo && { echo "-- clinfo --"; clinfo 2>&1 | head -15; } \
             || echo "  clinfo: not installed (absence is not evidence)"

hr "3. CUDA toolchain -- decides whether Uni-Dock is possible"
echo "Driver 470.57.02 is the CUDA 11.4 generation. CUDA 12 needs driver >= 525,"
echo "so cuda/12.4 and 12.8 are unusable here. cuda/11.4 + gcc/10.2.0 is the pair"
echo "(CUDA 11.4 supports gcc up to 10; gcc 4.8.5 is too old for modern C++)."
module purge >/dev/null 2>&1
for m in compilers/cuda/11.4 compilers/gcc/10.2.0 libraries/cmake/3.23.2; do
  if module load "$m" >/dev/null 2>&1; then echo "  loaded  $m"
  else echo "  FAILED  $m"; fi
done
module list 2>&1 | sed 's/^/  /'
echo "nvcc  : $(nvcc --version 2>/dev/null | tail -1 || echo ABSENT)"
echo "gcc   : $(gcc --version 2>/dev/null | head -1 || echo ABSENT)"
echo "g++   : $(g++ --version 2>/dev/null | head -1 || echo ABSENT)"
echo "cmake : $(cmake --version 2>/dev/null | head -1 || echo ABSENT)"
echo "make  : $(make --version 2>/dev/null | head -1 || echo ABSENT)"
echo "CUDA_HOME=${CUDA_HOME:-unset}"

hr "4. THE DECISIVE TEST: does nvcc compile AND a kernel run on this GPU?"
cat > "$SCRATCH/t.cu" <<'CUEOF'
#include <cstdio>
__global__ void k(float* o){ o[threadIdx.x] = threadIdx.x * 2.0f; }
int main(){
  int n=0;
  if(cudaGetDeviceCount(&n)!=cudaSuccess || n==0){ printf("NO DEVICE\n"); return 1; }
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  printf("  devices=%d  device0=%s  sm_%d%d  %.0f GB\n", n, p.name, p.major,
         p.minor, p.totalGlobalMem/1073741824.0);
  float *d, h[4]={-1,-1,-1,-1};
  if(cudaMalloc(&d, 4*sizeof(float))!=cudaSuccess){ printf("  MALLOC FAILED\n"); return 1; }
  k<<<1,4>>>(d);
  if(cudaDeviceSynchronize()!=cudaSuccess){ printf("  KERNEL LAUNCH FAILED\n"); return 1; }
  cudaMemcpy(h, d, 4*sizeof(float), cudaMemcpyDeviceToHost);
  printf("  kernel result = %.0f %.0f %.0f %.0f (expect 0 2 4 6)\n", h[0],h[1],h[2],h[3]);
  printf("  CUDA PATH: WORKS\n");
  return 0;
}
CUEOF
if have nvcc; then
  if nvcc -arch=sm_80 -o "$SCRATCH/t" "$SCRATCH/t.cu" 2>"$SCRATCH/nvcc.err"; then
    echo "  compiled for sm_80 (A100)"
    "$SCRATCH/t" || echo "  COMPILED BUT FAILED AT RUNTIME -- driver/toolkit mismatch."
  else
    echo "  nvcc COMPILE FAILED:"; head -20 "$SCRATCH/nvcc.err" | sed 's/^/    /'
  fi
else
  echo "  nvcc absent -- CUDA path closed"
fi

hr "5. Boost -- both Vina-GPU 2.0 and Uni-Dock link it"
[ -d /usr/include/boost ] && grep -m1 BOOST_LIB_VERSION /usr/include/boost/version.hpp
find /apps /scratch/tools -maxdepth 6 -path '*boost*' -name 'version.hpp' 2>/dev/null | head -3
conda list -n base 2>/dev/null | grep -iE '^boost' || echo "  no conda boost in base"
echo "  fallback: conda install -c conda-forge boost-cpp (user env, no admin)"

hr "6. Pre-installed tools and DBs (/scratch, per guide section 5)"
echo "-- /scratch/tools --"
ls /scratch/tools 2>/dev/null | head -40 || echo "  not readable"
echo "-- anything docking-ish already installed? --"
ls /scratch/tools 2>/dev/null | grep -iE 'dock|vina|smina|adfr|meeko|openbabel|obabel|rdkit' \
  || echo "  (none by name)"
find /scratch/tools -maxdepth 3 -type f \( -name 'vina*' -o -name '*smina*' \
     -o -name 'autodock*' -o -name 'autogrid*' \) 2>/dev/null | head -10
echo "-- /scratch/database (pdb, pubchem, moleculenet per the guide) --"
ls /scratch/database 2>/dev/null | head -20 || echo "  not readable"
for t in obabel smina vina idock unidock autogrid4 mk_prepare_ligand.py \
         prepare_receptor4.py; do
  printf '  %-24s %s\n' "$t" "$(command -v $t 2>/dev/null || echo '-')"
done
python -c "import rdkit; print('  rdkit  ', rdkit.__version__)" 2>/dev/null || echo "  rdkit: no"
python -c "import meeko; print('  meeko  ', meeko.__version__)" 2>/dev/null || echo "  meeko: no"
conda env list 2>/dev/null | sed 's/^/  /'

hr "7. Internet from THIS node -- decides where we can build"
echo "The guide documents `conda pack` for the no-internet case, so this matters:"
echo "if the compute node is offline, source must be fetched on the login node."
for host in github.com codeload.github.com conda.anaconda.org pypi.org; do
  if have curl; then
    code=$(curl -s -o /dev/null -m 8 -w '%{http_code}' "https://$host" 2>/dev/null)
    printf '  %-24s HTTP %s\n' "$host" "${code:-timeout}"
  else
    printf '  %-24s (no curl)\n' "$host"
  fi
done
echo "  proxy env: http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"

hr "8. Allocation budget -- node-hours, not wall hours"
GROUP=$(id -gn 2>/dev/null)
echo "group: $GROUP"
cat "/scratch/account/$GROUP" 2>/dev/null || echo "  /scratch/account/$GROUP not readable"
cat <<'NOTE'
  Charge rates from the guide: cpu64-only 1, 1gpu 1, debug-1gpu 1, 4gpu 4, 8gpu 8
  node-hours per WALL hour. So for the 7,767-ligand run:
    cpu64-only, measured    18.4 wall h x 1 = ~18 node-hours
    1gpu, if it takes 2 h    2.0 wall h x 1 = ~2  node-hours
    8gpu, if it takes 15 min  0.25 wall h x 8 = ~2 node-hours
  1gpu and 8gpu cost about the same in allocation because the charge scales with
  the GPU count; 8gpu only buys wall time, and only if the tool uses all 8 GPUs
  (one process per GPU). Build and smoke-test on debug-1gpu: same rate, 8 h cap.
NOTE

hr "9. Disk"
df -h "$HOME" /scratch 2>/dev/null
lfs quota -h -u "$(id -un)" "$HOME" 2>/dev/null || quota -s 2>/dev/null || true
echo "  budget: ~7,800 ligands x 20 poses. smina wrote ~45 KB per ligand's 20"
echo "  poses, so expect ~350 MB of poses, plus AD4-style grid maps if used."

hr "10. Verdict"
cat <<'NOTE'
  Read section 4 first, then 2, then 7.
    4 prints "CUDA PATH: WORKS"      -> Uni-Dock (CUDA, Vina scoring function).
    2 shows an ICD and libOpenCL     -> Vina-GPU 2.0 (OpenCL) is also open.
    7 shows HTTP 200 from github     -> build on the node; else fetch on login.
    neither 4 nor 2                  -> stay on CPU smina: 18.4 h, 18 node-hours.

  After a build succeeds, the FIRST thing to check is plumbing, not science:
  that Ca501 of chain B survives receptor conversion with a sane atom type. The
  section 8.5b geometry filter measures carboxylate-O to Ca501; if the calcium is
  dropped or mistyped, every pose fails for a reason that looks chemical. This
  project has already been bitten twice by that class of error - a 5-character
  residue name overflowing PDB columns and zeroing all affinities, and a naive
  index-wise RMSD reporting 6.13 A where the true value was 0.63 A.
NOTE
echo
echo "report written to: $OUT"
