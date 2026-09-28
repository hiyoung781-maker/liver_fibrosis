# avb1-rerun-v2 execution checklist

Runbook entries for long GPU jobs this plan requires but that cannot complete
inside an implementer's session. Each entry: what to run, what it must produce,
and the gate that decides the outcome.

## Task 5: TL-A′ checkpoint sweep and epoch selection

**Status as of this entry: not yet run.** `scripts/tl_sweep_report.py` and its
selection rule (`select_epoch`, `MIN_EPOCH=100`, `SCAFFOLD_FLOOR=700`) are
implemented and unit-tested (`tests/test_tl_sweep_report.py`, 5/5 passing) and
the aggregation CLI is smoke-tested end to end on 2 real (tiny) checkpoints —
see `results/v2_tl_sweep_smoke_fixture.csv` (NOT real sweep results, do not
treat as such). The actual 0→200 epoch, 11-checkpoint, 1000-sample-per-checkpoint
sweep requires hours of K-BDS GPU time and has not been run.

**What to run** (on a K-BDS GPU node, in the `reinvent` conda env):

```bash
bash scripts/kbds_submit_tl_sweep.sh
```

This backgrounds, in order:
1. `MODE=production-sweep bash scripts/run_d3.sh --arm A-prime` — renders
   `configs/tl.toml.in` with `num_epochs=200`, `save_every_n_epochs=20` for the
   TL-A/core arm (verified: `bash scripts/run_d3.sh --print-config --mode
   production-sweep` renders exactly `num_epochs = 200` /
   `save_every_n_epochs = 20` — this is the fix for the v1 bug where the
   default `diagnostic` mode set `save_every_n_epochs` equal to `num_epochs`,
   leaving checkpoints with no validation NLL). `--arm A-prime` restricts
   phases 1 and 2 to the core/TL-A arm only — no core_B diagnostic fold, no
   TL-B production train. This flag was added after review: `run_d3.sh`
   previously ran both arms unconditionally, which would have burned real GPU
   allocation on TL-B, an arm v2 discards outright (not just wasted
   wall-clock — the `8gpu` partition bills node-hours regardless of GPU count
   actually used).
2. `python scripts/tl_sweep_report.py --checkpoints priors/focused_A.prior
   --n-samples 1000 --final-epoch 200 --out results/v2_tl_sweep.csv`

**Artifacts it must produce:**
- `priors/focused_A.prior` + `priors/focused_A.prior.{20,40,...,180}.chkpt` —
  11 checkpoints including the final epoch (git-ignored, referenced by path)
- `logs/d3/<timestamp>/prod_core/tl.log` — must show validation NLL computed at
  every save epoch, not just the last
- wandb run `tl-a-prime-sweep` with **both** `train/loss` and `valid/nll` time
  series (check via `wandb_sync.py`'s local offline run or `wandb sync`)
- `results/v2_tl_sweep.csv` with columns `epoch, acid_pct, unique_scaffolds,
  max_nn_tanimoto, tpsa_median, qed_median` for epoch 0, 20, 40, ..., 200

**Gate (pre-registered, does not move after seeing results):**
`select_epoch(rows)` — highest epoch with epoch ≥ 100 AND unique Murcko
scaffolds ≥ 700 (of 1000 sampled). Memorization (`max_nn_tanimoto`) is reported
only, never a disqualifier — the downstream Murcko novelty gate
(`data/known_scaffolds.smi`) already removes memorized molecules.

- If `select_epoch` returns an epoch: copy that checkpoint to
  `priors/focused_A_prime.prior` and record the choice.
- If `select_epoch` returns `None`: TL-A′ is **dropped as an arm**. Record this
  in `results/v2_tl_decision.md` and proceed with TL-C alone in Task 6.

Also confirm the wandb NLL-series check above before trusting the CSV — if
`valid/nll` is missing, the run silently fell back to the diagnostic
checkpoint cadence and must be re-run, not patched after the fact.
