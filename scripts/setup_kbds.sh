#!/bin/bash
# Build the `reinvent` conda environment on K-BDS (KISTI). Run once, on the LOGIN
# node - it only downloads and installs, which is what the login node is for.
#
#   cd ~/liver_fibrosis && bash scripts/setup_kbds.sh
#
# Constraints this script is built around, all measured on bdata-login01:
#   CentOS 7.9, glibc 2.17   -> no manylinux_2_28 wheels (rules out PyPI rdkit and
#                               every torch newer than 2.6)
#   driver 470.57.02         -> CUDA 11.x only (CUDA 12 needs >= 525)
#   outbound internet open   -> install directly here; no conda-pack bundle needed
#
# No CUDA module is loaded on purpose: the pip torch wheel ships its own CUDA
# runtime and only needs libcuda.so from the driver. Loading compilers/cuda/12.x
# would put an incompatible toolkit ahead of it on the library path.

set -euo pipefail

ENV_NAME="${ENV_NAME:-reinvent}"
TORCH_VERSION="2.5.1+cu118"
# torchvision must match torch exactly (0.20.1 declares torch==2.5.1) and must come
# from the same CUDA index, since it ships its own CUDA kernels. REINVENT4 v4.5.11
# does not declare torchvision at all, but reinvent/runmodes/utils/image.py imports
# it at module level, so the CLI cannot start without it. Upstream main fixed this
# by adding `"torchvision",  # needs to match torch` to its dependencies.
TORCHVISION_VERSION="0.20.1+cu118"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "0. Preflight"
[ -d "$ROOT/REINVENT4" ] || die "REINVENT4/ not found under $ROOT - transfer incomplete?"
command -v conda >/dev/null || die "conda not on PATH"
echo "project   : $ROOT"
echo "conda     : $(conda --version) at $(command -v conda)"
echo "glibc     : $(getconf GNU_LIBC_VERSION 2>/dev/null || ldd --version | head -1)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
fi

# git describe when the history came along, otherwise the marker file written at
# transfer time - REINVENT4/.git is 1.5 GB and is usually left behind.
reinvent_ref="$(cd "$ROOT/REINVENT4" && git describe --tags 2>/dev/null \
  || head -1 "$ROOT/REINVENT4/.reinvent_version" 2>/dev/null \
  || echo unknown)"
echo "REINVENT4 : $reinvent_ref"
[ "$reinvent_ref" = "v4.5.11" ] || cat <<WARN

  WARNING: REINVENT4 is at '$reinvent_ref', not v4.5.11.
  v4.5.11 is the release this environment is pinned against: it is the newest one
  whose torch pin (2.5.1) still has a glibc 2.17 wheel, and it ships the priors in
  the repo. If the git history came along:  cd $ROOT/REINVENT4 && git checkout v4.5.11
  Otherwise re-transfer - the working tree itself is wrong.

WARN

say "0b. Does the install list cover everything REINVENT4 imports at startup?"
# v4.5.11 under-declares: torchvision and scipy are imported at module level but
# absent from its pyproject.toml. Since REINVENT4 installs with --no-deps, an
# undeclared import becomes a crash on first run. Catch it before the install, not
# after. Pure stdlib, so the base interpreter is enough.
python3 "$ROOT/scripts/check_imports.py" "$ROOT/REINVENT4" \
  || die "REINVENT4 imports a package this environment does not install (see above)"

say "1. Create the conda environment '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "already exists - reusing it (delete with: conda env remove -n $ENV_NAME)"
else
  conda env create -n "$ENV_NAME" -f "$ROOT/env/environment.yml"
fi

# conda activate is a shell function, so source the hook rather than relying on init
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "python    : $(python -V) at $(command -v python)"

say "2. Install torch $TORCH_VERSION and torchvision $TORCHVISION_VERSION"
python -m pip install --upgrade pip
python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "$TORCH_INDEX"

say "3. Install the remaining REINVENT4 dependencies"
# --only-binary=:all: is the point of this step, not a detail. Without it, a
# package whose newest release ships manylinux_2_28 wheels only makes pip fall back
# to the sdist, and the build fails several dependencies deep with a message about
# some other package ("NumPy requires GCC >= 9.3") instead of naming the culprit.
# With it, pip says exactly which package has no usable wheel.
python -m pip install --only-binary=:all: -r "$ROOT/env/requirements-pip.txt"

# molvs publishes no wheel; it is pure python and builds with setuptools alone.
python -m pip install -r "$ROOT/env/requirements-pip-sdist.txt"

say "4. Install REINVENT4 itself"
# --no-deps because three of v4.5.11's declared dependencies are deliberately not
# installed - openEye-toolkits, chemprop and descriptastorus. See the header of
# env/requirements-pip.txt for why each is absent and why nothing breaks.
python -m pip install --no-deps -e "$ROOT/REINVENT4"

say "5. Verify"
python - <<'PY'
import sys
import torch, torchvision, rdkit, numpy, pandas, scipy
print(f"  python  {sys.version.split()[0]}")
print(f"  torch   {torch.__version__}   CUDA build {torch.version.cuda}")
print(f"  torchvision {torchvision.__version__}   scipy {scipy.__version__}")
print(f"  rdkit   {rdkit.__version__}")
print(f"  numpy   {numpy.__version__}   pandas {pandas.__version__}")
avail = torch.cuda.is_available()
print(f"  torch.cuda.is_available() = {avail}")
if avail:
    print(f"  device  {torch.cuda.get_device_name(0)}  capability sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
else:
    print("  (no GPU visible here - expected on a login node; slurm/00_smoke.sbatch checks the real thing)")
import reinvent
print(f"  reinvent {getattr(reinvent, '__version__', '?')} imported OK")

# The three omitted dependencies must degrade to "component skipped", not a crash.
from reinvent.scoring.importer import get_registry
registry = get_registry()
print(f"  scoring components registered: {len(registry)}")
# Every component the blueprint's RL objective names, in registry form
# (importer.py keys on the class name lowercased).
needed = [
    "tanimotosimilarity", "matchingsubstructure", "customalerts",
    "tpsa", "slogp", "numrotbond", "molecularweight", "qed", "sascore",
]
missing = [n for n in needed if n not in registry]
if missing:
    raise SystemExit(f"FAIL: scoring components did not register: {missing}")
print(f"  all {len(needed)} components the RL objective needs are present")
PY

say "5b. Are the installed versions the ones we pinned?"
# A version can drift without anything failing: conda-forge rdkit pulls pandas,
# pillow and matplotlib-base as run dependencies, so if pip also owned them the two
# package managers would fight and leave the loser's dist-info behind. That is how
# this environment once ended up importing pandas 3.0.6 - which REINVENT4 forbids -
# while `conda list` still reported 2.3.2. Assert the versions, and assert that each
# shared package has exactly one dist-info.
python - <<'PY_CHECK'
import sys, pathlib, importlib.metadata as md

EXPECT = {                       # package -> (predicate, description)
    "torch":       (lambda v: v.startswith("2.5.1+cu118"), "2.5.1+cu118"),
    "torchvision": (lambda v: v.startswith("0.20.1+cu118"), "0.20.1+cu118"),
    "numpy":       (lambda v: v.split(".")[0] == "1", "1.x"),
    "pandas":      (lambda v: v.split(".")[0] == "2", "2.x"),
    "pillow":      (lambda v: v.split(".")[0] == "10", "10.x"),
    "scipy":       (lambda v: tuple(map(int, v.split(".")[:2])) < (1, 17), "< 1.17"),
}

bad = []
for name, (ok, want) in EXPECT.items():
    try:
        got = md.version(name)
    except md.PackageNotFoundError:
        bad.append(f"{name}: not installed (want {want})"); continue
    mark = "ok " if ok(got) else "BAD"
    print(f"  {mark} {name:12s} {got:16s} (want {want})")
    if not ok(got):
        bad.append(f"{name}: {got}, want {want}")

# conda and pip both writing the same package leaves two dist-info dirs behind
site = pathlib.Path(md.distribution("numpy").locate_file(""))
for name in ("pandas", "pillow", "matplotlib", "numpy", "scipy"):
    dists = sorted(d.name for d in site.glob(f"{name}-*.dist-info"))
    dists += sorted(d.name for d in site.glob(f"{name.capitalize()}-*.dist-info"))
    if len(set(dists)) > 1:
        bad.append(f"{name}: {len(dists)} dist-info dirs -> conda and pip both own it: {dists}")

if bad:
    print("\n  VERSION CHECK FAILED:")
    for b in bad: print(f"    - {b}")
    sys.exit(1)
print("  every pinned version matches and no package is owned twice")
PY_CHECK

say "6. Verify the priors"
( cd "$ROOT/REINVENT4/priors" && sha512sum -c sha512.sum ) \
  || die "prior checksum mismatch - re-checkout REINVENT4 v4.5.11"

say "7. The reinvent CLI"
# Never hide this failure's output: the CLI imports the whole run-mode chain
# (RL, TL, sampling, scoring), so a broken dependency surfaces here as a traceback
# and the traceback is the only thing that says which one.
if ! command -v reinvent >/dev/null; then
  die "no 'reinvent' on PATH - the editable install did not create the console script"
fi
echo "  binary: $(command -v reinvent)"
if reinvent --help >/tmp/reinvent_help.$$ 2>&1; then
  echo "  'reinvent' CLI works"
  rm -f /tmp/reinvent_help.$$
else
  echo
  echo "  --- reinvent --help failed, output follows ---"
  tail -40 /tmp/reinvent_help.$$
  rm -f /tmp/reinvent_help.$$
  die "'reinvent' CLI not callable (see the traceback above)"
fi

cat <<DONE

Setup complete.

  conda activate $ENV_NAME

Next: submit the smoke test, which is the first thing that actually touches a GPU.

  sbatch slurm/00_smoke.sbatch
  squeue -u \$(id -un)

DONE
