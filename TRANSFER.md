# Moving this project to K-BDS

Everything needed to run REINVENT4 on the KISTI K-BDS cluster lives in this
directory. Nothing has to be rebuilt by hand on the other side: transfer, run one
setup script, submit one smoke job.

The environment was designed against a live probe of `bdata-login01`
(`scripts/kbds_probe.sh`), not against assumptions. What that probe found, and what
each finding forced, is in **Why the versions are what they are** at the bottom.

---

## 1. What to copy

The project directory is 45 GB, but only ~775 MB of it belongs to this work. The
rest is the target-prioritisation stage that is already finished.

| | size | transfer? |
|---|---|---|
| `REINVENT4/` working tree, incl. 740 MB of priors | 747 MB | **yes** |
| `data/`, `scripts/`, `slurm/`, `env/`, `configs/` | 2.4 MB | **yes** |
| blueprints, decision brief, `8W30.cif`, pocket figures, project `.git` | 24 MB | **yes** |
| `REINVENT4/.git/` | 1.5 GB | no - see below |
| `opentarget/` | 38 GB | no - target prioritisation, done |
| `Geneformer/`, `geo_data/`, `msigdb/` | 4.6 GB | no - unrelated stages |
| `papers/` | 45 MB | no - curation is complete, the data is in `data/` |

**Transfers go to the login node, `kbds.kisti.re.kr`.** The guide names
`kbds-dm.kisti.re.kr` for data transfer, but that host serves **FTP on port 21
only** (FileZilla, Aspera) - its port 22 is closed. The login node is the only SSH
path, and the guide states it allows `ssh, scp, sftp, X11`.

One command, from the machine holding this directory:

```bash
rsync -avzP --partial \
  --exclude 'opentarget/' --exclude 'Geneformer/' --exclude 'geo_data/' \
  --exclude 'msigdb/' --exclude 'papers/' --exclude 'REINVENT4/.git/' \
  --exclude '__pycache__/' --exclude 'logs/' \
  /home/seyoung/academic_symposium/liver_fibrosis/ \
  <ID>@kbds.kisti.re.kr:~/liver_fibrosis/
```

rsync opens one SSH connection, so the OTP code and password are asked once. `-P`
plus `--partial` makes it resumable, which matters if the link drops - resuming
costs another OTP entry but no re-sent bytes. Keep it to a single pass; the login
node is shared.

`REINVENT4/.git` is excluded because it is twice the size of everything else and is
not needed to run anything. `setup_kbds.sh` still confirms the REINVENT4 version by
reading `REINVENT4/.reinvent_version`, a marker written here alongside the checkout.
Include the history (drop that one `--exclude`) only if you want to switch tags on
the cluster.

## 2. Build the environment (login node, once)

```bash
ssh <ID>@kbds.kisti.re.kr
cd ~/liver_fibrosis
bash scripts/setup_kbds.sh
```

Roughly 10 minutes, mostly the torch download. The script creates the conda
environment `reinvent`, verifies every version it installed, checks the ten prior
files against their shipped SHA-512 sums, and confirms the `reinvent` CLI runs.

It deliberately does **not** `module load` any CUDA. The pip torch wheel carries its
own CUDA 11.8 runtime and needs only `libcuda.so` from the driver; putting
`compilers/cuda/12.x` on the library path would shadow it with an incompatible
toolkit.

## 3. Smoke test (GPU, once)

```bash
mkdir -p logs
sbatch slurm/00_smoke.sbatch
squeue -u $(id -un)
```

This is the first thing that touches a GPU. It runs the real stack on the real
curated data - one epoch of transfer learning on the 23-molecule TL-A training
split, then sampling 200 molecules - and fails loudly if validity drops below 80%.
A failure here is a setup failure, never a science failure.

Read the result:

```bash
cat logs/smoke_<jobid>.out
```

`debug-1gpu` is capped at 8 hours and exists for exactly this. Note that every GPU
partition was fully allocated when the environment was probed, so expect to queue.

---

## Why the versions are what they are

Four measurements on `bdata-login01` decided the whole stack.

**glibc 2.17 (CentOS 7.9.2009).** REINVENT4's current `main` pins `torch==2.12.0`,
which ships `manylinux_2_28` wheels only - they need glibc 2.28 and cannot run here.
The newest release whose torch pin still has a glibc 2.17 wheel is **v4.5.11**
(`torch==2.5.1`). The same constraint rules out PyPI `rdkit`, whose wheels are also
`manylinux_2_28`; rdkit therefore comes from **conda-forge**, which declares
`__glibc >=2.17`.

**Driver 470.57.02.** CUDA 12 needs driver ≥ 525, so every `cu12x` build is out -
including the `compilers/cuda/12.4` and `12.8` modules the cluster offers. CUDA 11.8
needs only ≥ 450 and runs fine. So the single change to v4.5.11's pin is
`torch==2.5.1+cu124` → **`2.5.1+cu118`**: same torch, different CUDA build, which is
exactly the substitution its `pyproject.toml` comment anticipates.

**Outbound internet is open** (pypi, conda-forge, zenodo, github, rcsb all returned
200, no proxy). So packages install directly on K-BDS and no `conda-pack` bundle is
needed. This also means `scripts/curate_actives.py` can be re-run there if wanted,
though its results are already committed under `data/`.

**Apptainer and Singularity are both broken** on the login node (`libsubid.so.3`
missing), so a container - normally the clean answer to a glibc mismatch - was not
available.

Two further consequences worth knowing before reading the configs:

- **v4.5.11 ships the priors in the repo** (`REINVENT4/priors/`, ten files with a
  `sha512.sum`), so there is no Zenodo download step. Current `main` ships none.
- **v4.5.11's TL config has no `isomeric_smiles` field**, and its `GlobalConfig` uses
  pydantic `extra="forbid"` - an unknown key is a hard error, not a warning. This is
  moot anyway: the de novo prior's vocabulary contains no stereochemistry tokens at
  all, so every `.smi` file under `data/` carries flattened SMILES. The isomeric form
  is preserved in `data/actives_annotated.csv`, which is the record of truth and the
  input for the docking stage. See the prior-vocabulary gate in
  `data/CURATION_LOG.md`.

**Three declared dependencies are deliberately not installed.** Each backs exactly
one scoring component, and `reinvent/scoring/importer.py` skips a component it cannot
import, so nothing else is affected. This is why REINVENT4 is installed with
`pip install --no-deps`, and why `setup_kbds.sh` ends by asserting that all nine
components the RL objective names did register.

| dropped | why |
|---|---|
| `openEye-toolkits` | commercially licensed; backs ROCS only |
| `chemprop` | imported only by `comp_chemprop.py`; §3.3 rules ChemProp2 a no-go, and it drags in scipy and scikit-learn, whose current releases are `manylinux_2_28` only |
| `descriptastorus` | a hidden chemprop dependency; no REINVENT4 module imports it |

**v4.5.11 under-declares two dependencies.** `torchvision` and `scipy` are imported
at module level on the path from `reinvent.Reinvent` - `runmodes/utils/image.py` and
`runmodes/utils/plot.py` respectively, both pulled in by the sampling reports - yet
neither appears in its `pyproject.toml`. Upstream `main` fixed the torchvision half
(`"torchvision",  # needs to match torch`). Because REINVENT4 installs with
`--no-deps`, an undeclared import is a crash on first run rather than an install
error, so `scripts/check_imports.py` walks the module-level import graph and
`setup_kbds.sh` runs it *before* installing anything. Re-run it if the pinned
REINVENT4 version ever changes.

torchvision is pinned to **0.20.1+cu118**: 0.20.1 declares `torch==2.5.1` exactly,
and it needs the same CUDA index because it ships its own kernels. scipy is capped
below 1.17, which went `manylinux_2_28`.

**Why `--only-binary=:all:`.** When the newest release of a package publishes
`manylinux_2_28` wheels only, pip does not stop - it quietly falls back to the
sdist, and the build then fails several dependencies deep with a message about some
*other* package. The first attempt here died on `NumPy requires GCC >= 9.3` while
the real problem was that pandas 2.3.3 had dropped its `manylinux_2_17` wheel
(2.3.2 is the last that runs on glibc 2.17). Installing binary-only turns that into
a clear error naming the actual package. `molvs` is the one dependency that ships no
wheel at all; it is pure python and lives in `env/requirements-pip-sdist.txt`.
