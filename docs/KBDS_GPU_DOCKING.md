# K-BDS GPU docking — environment and facts

Deliberately a document, not a script. Setting this up is two conda commands; an
earlier 138-line setup script wrapped them in diagnostics and `set -euo pipefail`,
which under conda's own `conda.sh` aborted with **no output at all**. The commands
below are meant to be pasted into a login-node shell.

## Getting the code onto K-BDS

The login node has outbound internet but **no GitHub SSH key**, so an `ssh://`
remote fails with `Permission denied (publickey)`. The repository is public, so
read-only HTTPS needs no credentials at all — no key to generate, no token:

```bash
cd /home01/<user>/liver_fibrosis
git remote set-url origin https://github.com/hiyoung781-maker/liver_fibrosis.git
git pull
```

Pushing from K-BDS is not set up and is not needed: code travels K-BDS-ward through
git, and results come back by ordinary file transfer. The pose SDF is ~350 MB and
does not belong in git anyway; only the geometry-filter summary CSV (a few MB) would.

Note the repository is public, so everything committed is world-readable.

## Workflow this fits

`~/kbds/kbds_gpu/*.sh` are **allocation holders**: each sbatch's an infinite
`while True: sleep(5)` loop to keep a node, and real work happens interactively via
`srun --jobid=<id> --pty bash`. So the deliverable for a docking run is a `.py` run
inside a held allocation, not an sbatch batch job.

## Install (LOGIN NODE — the compute node has no internet)

One command per line, no backslash continuations — a wrapped `conda install` lost
its second line on paste once, leaving an env with only Python in it and `unidock:
command not found` at the next step.

```bash
. /apps/application/miniconda3/etc/profile.d/conda.sh
conda create -y -n unidock python=3.11
conda install -y -n unidock -c conda-forge 'unidock=1.1.3=cuda118*' rdkit openbabel meeko numpy pandas
conda activate unidock
unidock --help | head -3
```

If the solver balks — `conda create` pulls `libgcc-ng 11.2.0` from `pkgs/main`
while `unidock` wants `>=12` — pin the channel instead of mixing:

```bash
conda env remove -y -n unidock
conda create -y -n unidock --override-channels -c conda-forge python=3.11 'unidock=1.1.3=cuda118*' rdkit openbabel meeko numpy pandas
```

**The env lands in `${HOME}/.conda/envs/<name>`** (K-BDS default), i.e.
`/home01/<user>/.conda/envs/unidock`. `/home01` is shared Lustre, which is precisely
why installing on the login node works: the internet-less compute node sees the same
env and needs only `conda activate unidock`.

**Two miniconda installs exist.** `~/kbds/kbds_gpu/*.sh` source
`/apps/application/miniconda3`, while the compute node's `CONDA_EXE` is
`/apps/application/miniconda3-new/bin/conda`. Use whichever lists the existing
`reinvent` env.

## Verify on the GPU node

```bash
srun --jobid=<your running 8gpu jobid> --pty bash
. /apps/application/miniconda3/etc/profile.d/conda.sh
conda activate unidock
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=<an idle index>
unidock --help | head -3
```

One thing is being tested: whether a `cuda118` build **initializes** on driver 550.
Installing and initializing are different — ADMET-AI locally returned
`torch.cuda.is_available() == True` and then died in `cuda_init`.

Pin `CUDA_VISIBLE_DEVICES`: the allocation is exclusive to the job, but that job may
be running something else. Uni-Dock needs one GPU.

## Why Uni-Dock, and why conda

**Uni-Dock implements AutoDock Vina's scoring function on CUDA.** The scoring
function being identical is what lets the recorded redocking result (0.63 Å RMSD,
smina) keep characterizing the method — blueprint §8.5(a)'s validation clause is
about whether docking can reproduce the crystal pose at all, not about a
reimplementation of the same function on different hardware. The search differs
(thread-parallel BFGS), so redocking the crystal ligand is still worth one run, as a
**plumbing check on the build and prep, not a revalidation.**

It matters less than it would otherwise because §8.5b selects on **geometry, not
affinity**: the filter measures distances, so it transfers across scoring functions
untouched.

**Conda, because source builds are impossible on the compute node.** Measured on
`gpu-8-002`:

| Fact | Consequence |
|---|---|
| No internet (github, conda, pypi all unreachable) | Cannot clone or download there |
| `module load` is a no-op; cmake fails on unset `env(COMPILER_VER)` | No toolchain via modules |
| gcc 4.8.5, cmake 2.8.12, no `nvcc` from any CUDA module | Cannot compile modern C++ |
| OpenCL present (`nvidia.icd`, `libOpenCL.so`), Boost complete | Vina-GPU 2.0 would be *possible* if it could be built |
| `/scratch/tools` has no docking software | Nothing pre-installed to reuse |

Conda packages ship their own CUDA runtime, so none of the above applies.

**Why the `cuda118` build.** Conda derives `__cuda` from the driver of the machine
running conda. The login node is 470.57.02 (CUDA 11.4); `gpu-8-002` is 550.54.14
(CUDA 12.x). A `cuda118` build satisfies both, so the env works wherever it runs and
no `CONDA_OVERRIDE_CUDA` gamble on minor-version compatibility is needed.
`unidock` also declares `__glibc >=2.17`, which CentOS 7.9 satisfies.

## Cost

Charge rates (guide p.18, per **wall** hour): `cpu64-only` 1, `1gpu` 1,
`debug-1gpu` 1, `4gpu` 4, `8gpu` 8.

| Route | Wall | Node-hours |
|---|---|---|
| `cpu64-only` smina, measured 547 s/ligand ÷ 64 cores | 18.4 h | ~18 |
| 1 GPU Uni-Dock, estimated | ~0.5–1 h | ~1 |

Run inside an allocation already held: it is charged whether or not anything uses
it, so the marginal cost is zero. `debug-1gpu` is contended in practice.

`/scratch/account/<group>` returned `49`; the guide calls it "자원 사용시간" without
saying whether that is used or remaining. Check the portal
(`cloud.kbds.re.kr` → 자원 이용현황) before planning around it. Note that an 8gpu
allocation burns 8 node-hours per wall hour — a 2-day hold is ~400 node-hours,
which dwarfs the docking run either way.

## The risk that actually matters

After the build works, check **plumbing, not science**: convert the receptor to
PDBQT and confirm **Ca501 of chain B** survives with its original coordinates.
§8.5b measures carboxylate-O → Ca501; if the calcium is dropped or mistyped, every
pose fails for a reason that looks chemical.

Two errors of exactly this shape have already occurred in this project: a
5-character residue name overflowing PDB columns and zeroing every affinity, and a
naive index-wise RMSD reporting 6.13 Å where `rdMolAlign.CalcRMS` gives 0.63 Å.

## What carries over from the CPU route

`scripts/build_survivors.py` (the 7,767-molecule input), `scripts/pose_geometry.py`
(the geometry filter — engine-independent, it measures distances) and
`scripts/prepare_ligands.py` (3D embedding) are all reused. Only the CPU sbatch
script was discarded.
