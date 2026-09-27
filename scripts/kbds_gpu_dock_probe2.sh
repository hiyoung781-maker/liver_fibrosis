#!/bin/bash
# Stage-2 probe. Run on the GPU compute node, in the repo directory.
#   [gpu-8-002]$ bash scripts/kbds_gpu_dock_probe2.sh
#
# Stage 1 settled these, so they are not retested:
#   - gpu-8-002 driver is 550.54.14 (NOT the login node's 470.57.02), A100 80G x8,
#     compute capability 8.0 -> CUDA 12.4 is usable here after all.
#   - OpenCL is fully present: nvidia.icd, libOpenCL.so{,.1}, libnvidia-opencl.so.1,
#     headers under /apps/application/miniconda3/include/CL.
#   - 64 vCPU, 1 thread/core, CentOS 7.9, glibc 2.17, and NO tools installed at all.
#
# Stage 1 also produced one wrong verdict: it reported "nvcc absent -- CUDA path
# closed", but the real cause was module ordering. This module system rejects a
# library module unless a COMPILER module is loaded first ("COMPILER env vars were
# not properly defined"), and loading cuda before gcc left gcc at the system 4.8.5
# with CUDA_HOME unset. So the CUDA question is still open, and this retests it in
# the documented order: compiler, then toolkit, then libraries.
#
# Decides: OpenCL (Vina-GPU 2.0, keeps Vina's scoring function and therefore the
# recorded 0.63 A redocking result) vs CUDA (Uni-Dock). And whether anything can be
# fetched and built here at all.

set -u
OUT="${PWD}/kbds_gpu_dock_probe2_report.txt"
exec > >(tee "$OUT") 2>&1

SCRATCH="${TMPDIR:-/tmp}/gpudock2_$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

hr() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "K-BDS GPU docking probe, stage 2   $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "host=$(hostname)  partition=${SLURM_JOB_PARTITION:-?}  pwd=$(pwd)"

hr "1. Module order: compiler FIRST (stage 1's mistake)"
module purge >/dev/null 2>&1
echo "-- step 1: compilers/gcc/10.2.0 --"
module load compilers/gcc/10.2.0 2>&1 | sed 's/^/    /'
echo "   gcc now: $(gcc --version 2>/dev/null | head -1)"
echo "-- step 2: CUDA. Driver 550 supports 12.4, so try it first, then 11.8/11.4 --"
CUDA_MOD=""
for m in compilers/cuda/12.4 compilers/cuda/11.8 compilers/cuda/11.4; do
  module load "$m" 2>&1 | sed 's/^/    /'
  if have nvcc; then CUDA_MOD="$m"; echo "   OK -> $m gives nvcc"; break; fi
  echo "   $m: still no nvcc, unloading"
  module unload "$m" >/dev/null 2>&1
done
echo "-- step 3: cmake (needs the compiler env, which is why it failed before) --"
module load libraries/cmake/3.23.2 2>&1 | sed 's/^/    /'
echo
module list 2>&1 | sed 's/^/  /'
echo "  nvcc      : $(nvcc --version 2>/dev/null | tail -1 || echo ABSENT)"
echo "  gcc       : $(gcc --version 2>/dev/null | head -1 || echo ABSENT)"
echo "  g++       : $(g++ --version 2>/dev/null | head -1 || echo ABSENT)"
echo "  cmake     : $(cmake --version 2>/dev/null | head -1 || echo ABSENT)"
echo "  make      : $(make --version 2>/dev/null | head -1 || echo ABSENT)"
echo "  CUDA_HOME : ${CUDA_HOME:-unset}"
echo "  chosen    : ${CUDA_MOD:-NONE}"

hr "2. THE DECISIVE TEST: OpenCL compile AND run on an A100"
CLINC=""
for d in /apps/application/miniconda3/include "${CUDA_HOME:-/nonexistent}/include"; do
  [ -f "$d/CL/cl.h" ] && { CLINC="$d"; break; }
done
echo "  CL headers: ${CLINC:-NOT FOUND}"
cat > "$SCRATCH/cl.c" <<'CEOF'
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
int main(void){
  cl_uint np=0;
  if(clGetPlatformIDs(0,NULL,&np)!=CL_SUCCESS || np==0){ printf("  NO PLATFORM\n"); return 1; }
  cl_platform_id p[8]; clGetPlatformIDs(np>8?8:np,p,NULL);
  printf("  platforms=%u\n", np);
  for(cl_uint i=0;i<np && i<8;i++){
    char nm[256]="?", ver[256]="?"; cl_uint nd=0;
    clGetPlatformInfo(p[i],CL_PLATFORM_NAME,sizeof nm,nm,NULL);
    clGetPlatformInfo(p[i],CL_PLATFORM_VERSION,sizeof ver,ver,NULL);
    clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,0,NULL,&nd);
    printf("  [%u] %s | %s | GPU devices=%u\n", i, nm, ver, nd);
    if(nd){
      cl_device_id d[8]; clGetDeviceIDs(p[i],CL_DEVICE_TYPE_GPU,nd>8?8:nd,d,NULL);
      for(cl_uint j=0;j<nd && j<8;j++){
        char dn[256]="?"; cl_ulong mem=0; cl_uint cu=0;
        clGetDeviceInfo(d[j],CL_DEVICE_NAME,sizeof dn,dn,NULL);
        clGetDeviceInfo(d[j],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof mem,&mem,NULL);
        clGetDeviceInfo(d[j],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cu,&cu,NULL);
        printf("      dev%u %s  %.0f GB  %u CUs\n", j, dn, mem/1073741824.0, cu);
      }
      cl_context ctx=clCreateContext(NULL,1,d,NULL,NULL,NULL);
      if(ctx){ printf("  context created OK\n  OPENCL PATH: WORKS\n"); clReleaseContext(ctx); }
      else printf("  context creation FAILED\n");
      return 0;
    }
  }
  printf("  no GPU device on any platform\n"); return 1;
}
CEOF
if [ -n "$CLINC" ]; then
  if gcc -I"$CLINC" -o "$SCRATCH/clt" "$SCRATCH/cl.c" -lOpenCL 2>"$SCRATCH/cl.err"; then
    "$SCRATCH/clt" || echo "  compiled but failed at runtime"
  else
    echo "  COMPILE FAILED:"; head -15 "$SCRATCH/cl.err" | sed 's/^/    /'
  fi
else
  echo "  cannot test without CL/cl.h"
fi

hr "3. CUDA compile AND run (only if section 1 found nvcc)"
if have nvcc; then
  cat > "$SCRATCH/t.cu" <<'CUEOF'
#include <cstdio>
__global__ void k(float* o){ o[threadIdx.x]=threadIdx.x*2.0f; }
int main(){ int n=0;
  if(cudaGetDeviceCount(&n)!=cudaSuccess||n==0){printf("  NO DEVICE\n");return 1;}
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("  devices=%d device0=%s sm_%d%d\n",n,p.name,p.major,p.minor);
  float *d,h[4]={-1,-1,-1,-1}; cudaMalloc(&d,16); k<<<1,4>>>(d);
  if(cudaDeviceSynchronize()!=cudaSuccess){printf("  LAUNCH FAILED\n");return 1;}
  cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost);
  printf("  result=%.0f %.0f %.0f %.0f (expect 0 2 4 6)\n  CUDA PATH: WORKS\n",
         h[0],h[1],h[2],h[3]); return 0; }
CUEOF
  if nvcc -arch=sm_80 -o "$SCRATCH/t" "$SCRATCH/t.cu" 2>"$SCRATCH/nvcc.err"; then
    "$SCRATCH/t" || echo "  compiled but failed at runtime"
  else
    echo "  nvcc COMPILE FAILED:"; head -20 "$SCRATCH/nvcc.err" | sed 's/^/    /'
  fi
else
  echo "  skipped: no nvcc from any CUDA module"
fi

hr "4. Boost (both engines link it)"
for d in /apps/application/miniconda3/include /usr/include; do
  [ -f "$d/boost/version.hpp" ] && \
    echo "  $d/boost: $(grep -m1 BOOST_LIB_VERSION $d/boost/version.hpp | awk '{print $3}')"
done
ls /apps/application/miniconda3/lib/libboost_*.so 2>/dev/null | head -5 \
  || echo "  no prebuilt boost .so in that miniconda (headers may be header-only)"

hr "5. Internet from THIS node -- decides where we build"
for host in github.com codeload.github.com conda.anaconda.org repo.anaconda.com pypi.org; do
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "https://$host" 2>/dev/null)
  printf '  %-24s HTTP %s\n' "$host" "${code:-TIMEOUT}"
done
echo "  proxy: http_proxy=${http_proxy:-unset} https_proxy=${https_proxy:-unset}"
echo "  If these time out, source must be fetched on the login node; the guide's"
echo "  'conda pack' section exists for exactly that situation."

hr "6. Conda: can a user env be created here?"
echo "  conda    : $(command -v conda || echo ABSENT)"
echo "  CONDA_EXE: ${CONDA_EXE:-unset}"
conda env list 2>&1 | sed 's/^/    /' | head
echo "  .conda quota target: ${HOME}/.conda/envs"
df -h "$HOME" 2>/dev/null | sed 's/^/    /'

hr "7. Pre-installed tools and DBs (guide section 5)"
echo "-- /scratch/tools --"; ls /scratch/tools 2>/dev/null | head -40 || echo "  unreadable"
echo "-- docking-related by name --"
ls /scratch/tools 2>/dev/null | grep -iE 'dock|vina|smina|adfr|meeko|babel|rdkit' \
  || echo "  none"
find /scratch/tools -maxdepth 3 -type f \
     \( -name 'vina*' -o -name '*smina*' -o -name 'autodock*' -o -name 'autogrid*' \) \
     2>/dev/null | head -10
echo "-- /scratch/database --"; ls /scratch/database 2>/dev/null | head -20 || echo "  unreadable"

hr "8. Allocation budget"
GROUP=$(id -gn 2>/dev/null); echo "  group: $GROUP"
cat "/scratch/account/$GROUP" 2>/dev/null | sed 's/^/    /' \
  || echo "    /scratch/account/$GROUP not readable"
cat <<'NOTE'
  Charge per WALL hour (guide p.18): cpu64-only 1, 1gpu 1, debug-1gpu 1, 4gpu 4,
  8gpu 8. For the 7,767-ligand run:
     cpu64-only, measured 547 s/ligand / 64 cores = 18.4 h  -> ~18 node-hours
     1 GPU, if ~2 h                                          -> ~2  node-hours
     8 GPUs, if ~15 min                                      -> ~2  node-hours
  1gpu and 8gpu cost the same; 8gpu only buys wall time, and only if the tool is
  run as one process per GPU. Build and smoke-test on debug-1gpu (rate 1, 8 h cap,
  which the guide designates for compiling and debugging).
NOTE

hr "9. Verdict"
cat <<'NOTE'
  Section 2 prints "OPENCL PATH: WORKS"  -> Vina-GPU 2.0. Preferred: it keeps
      AutoDock Vina's scoring function, so the recorded 0.63 A redocking result
      still characterizes the method and no revalidation is owed.
  Section 3 prints "CUDA PATH: WORKS"    -> Uni-Dock is available as a fallback
      (also Vina scoring, CUDA instead of OpenCL).
  Neither                                -> CPU smina on cpu64-only: 18.4 h.

  Whichever is chosen, the first check after the build is PLUMBING, not science:
  that Ca501 of chain B survives receptor conversion with a sane atom type, and
  that its coordinates match docking/receptor.pdb. The section 8.5b filter measures
  carboxylate-O to Ca501; if the calcium is dropped or mistyped, every pose fails
  for a reason that looks chemical. Two errors of exactly this shape have already
  happened here: a 5-character residue name overflowing PDB columns and zeroing
  every affinity, and a naive index-wise RMSD reporting 6.13 A against a true 0.63.
NOTE
echo
echo "report -> $OUT"
