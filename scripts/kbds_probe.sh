#!/bin/bash
# Read-only probe of the K-BDS environment. Installs nothing, writes nothing
# outside the report file. Run on the login node (kbds.kisti.re.kr); no GPU needed.
#
#   bash kbds_probe.sh            # prints the report and saves kbds_probe_report.txt

set -u
OUT="${PWD}/kbds_probe_report.txt"
exec > >(tee "$OUT") 2>&1

hr() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "K-BDS probe  $(date -u '+%Y-%m-%d %H:%M UTC')  host=$(hostname)  user=$(id -un)"

hr "1. OS and glibc (decides the PyTorch route)"
cat /etc/redhat-release 2>/dev/null || cat /etc/os-release 2>/dev/null | head -2
echo "kernel : $(uname -r)"
echo -n "glibc  : "
if have getconf; then getconf GNU_LIBC_VERSION 2>/dev/null || ldd --version 2>&1 | head -1
else ldd --version 2>&1 | head -1; fi
echo "-> torch 2.12 (REINVENT4's pin) needs glibc >= 2.28; torch <= 2.6 works on 2.17"

hr "2. GPU and driver"
if have nvidia-smi; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | head -5
else echo "no nvidia-smi on this node (expected on a login node)"; fi

hr "3. CUDA / python / conda modules"
if have module || [ -n "${MODULESHOME:-}" ]; then
  module avail 2>&1 | tr -s ' ' '\n' | grep -iE 'cuda|cudnn|python|conda|anaconda|gcc|pytorch' | sort -u | head -40
else echo "no Environment Module system found"; fi

hr "4. conda"
if have conda; then
  echo "conda   : $(conda --version 2>&1)  at $(command -v conda)"
  echo "base    : $(conda info --base 2>/dev/null)"
  echo "envs dir: ${HOME}/.conda/envs"
  conda env list 2>&1 | head -10
  echo "--- channels in ~/.condarc ---"; cat "${HOME}/.condarc" 2>/dev/null || echo "(no ~/.condarc)"
  echo -n "conda-pack present: "; conda list -n base 2>/dev/null | grep -c conda-pack
else echo "conda NOT on PATH (try: module avail | grep -i conda)"; fi

hr "5. Container runtime (glibc-proof fallback)"
for c in apptainer singularity podman docker; do
  if have $c; then echo "$c : $($c --version 2>&1 | head -1)"; else echo "$c : not found"; fi
done

hr "6. Outbound internet (decides bundle-everything vs install-there)"
for url in https://pypi.org/simple/ https://conda.anaconda.org/conda-forge/noarch/repodata.json \
           https://zenodo.org https://github.com https://files.rcsb.org; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 12 "$url" 2>/dev/null || echo "FAIL")
  printf '  %-62s %s\n' "$url" "$code"
done
echo "  (200/301/302 = reachable; FAIL/000 = blocked)"
echo -n "  proxy vars: "; env | grep -iE '^(http|https|no)_proxy' || echo "(none set)"

hr "7. Storage and quota"
echo "HOME = $HOME"
df -h "$HOME" 2>/dev/null | tail -2
if have lfs; then echo "--- lustre quota ---"; lfs quota -h -u "$(id -un)" "$HOME" 2>&1 | head -6; fi
quota -s 2>/dev/null | head -5
for d in /scratch /scratch/database /scratch/tools; do
  [ -d "$d" ] && echo "$d : present" || echo "$d : absent"
done

hr "8. Slurm queues"
if have sinfo; then
  sinfo -o '%20P %5a %10l %6D %8t %N' 2>&1 | head -15
  echo "--- my running/pending jobs ---"; squeue -u "$(id -un)" 2>&1 | head -5
  echo "--- node-hour balance ---"
  for f in /scratch/account/*; do [ -r "$f" ] && echo "$f:" && head -5 "$f"; done 2>/dev/null
else echo "sinfo not found"; fi

hr "9. Login-node resources"
echo "cores : $(nproc 2>/dev/null)"
free -g 2>/dev/null | head -2

hr "DONE"
echo "Report saved to: $OUT"
