# Methods and Experiment Protocol

This document lists every method that produced a result file under
`history/experiments/`, and the protocol under which it was run.

Source of method definitions: `../compositions/base.py`.
Source of protocol metadata: the JSON result records themselves.

---

## Shared experiment settings

| Setting | Value |
|---|---|
| Datasets | `buchwald_sub4`, `suzuki` |
| Seeds | `100, 200, …, 2000` (20 seeds) |
| Query budget | 40 steps (only test-pool queries count) |
| Selector | `argmax` (all methods below) |
| Acquisition (default) | Expected Improvement (EI), `xi=0.01` |
| Backend | BoTorch / GPyTorch |

### Prior protocols used in these results

Two prior protocols appear in the archived results. They **must not be compared
or aggregated** with each other, or with the current competition fixed-prior code.

#### 1. Seeded subsample (`prior_protocol="seeded_subsample"`)

- Explicit field in every `*_seeded.json` record.
- `n_initial = 5`
- For each `(dataset, seed)` pair, 5 rows are drawn from the train set using a
  local `RandomState(seed)`. The five indices are stored as `initial_indices`.
- Example record (lgbo_mean_shift, buchwald_sub4, seed=100):
  ```
  prior_protocol = "seeded_subsample"
  n_initial      = 5
  initial_indices = [33, 34, 31, 5, 1]
  status         = "ok"
  ```
- All 7 seeded files: 40/40 records `status="ok"`.

#### 2. Legacy full-prior (unlabeled)

- Files `H1_*`–`H4_*`.
- No `prior_protocol` or `n_initial` field in the JSON.
- Provenance from `roadmap.md` / `ledger.jsonl`: complete train set as prior,
  deterministic seeds (no seed variance under pure GP).
- Roadmap marks H1–H5 conclusions as **legacy**.

#### 3. Competition fixed-prior (current code, no results yet)

- Full `train.csv` prior: Buchwald 35 rows, Suzuki 29 rows.
- Buchwald encoder: 32-dim `options.json` decision schema.
- Intended output names: `*_fixed_prior.json`.
- **No fixed-prior result file existed when this archive was created.**

---

## Methods with complete seeded results (`n_initial=5`)

All 7 methods below have 40 records (20 seeds × 2 datasets),
`prior_protocol="seeded_subsample"`, `n_initial=5`, `status="ok"`.

| Method | Surrogate | Acquisition | Selector | LLM strategy | Key params | Result file |
|---|---|---|---|---|---|---|
| `gpbo_ei` | `botorch_matern` | EI | argmax | none | `use_llm=False`, `xi=0.01` | `gpbo_ei_seeded.json` |
| `gpbo_manifold` | `botorch_manifold` | EI | argmax | none | `use_llm=False`, `xi=0.01`, `evolve_interval=5` | `gpbo_manifold_seeded.json` |
| `gpbo_alas` | `botorch_alas` | EI | argmax | none | `use_llm=False`, `xi=0.01`, `mode=alas`, `init_alpha=1.5` | `gpbo_alas_seeded.json` |
| `gpbo_dkl` | `botorch_dkl` | EI | argmax | none | `use_llm=False`, `xi=0.01`, `hidden_dim=16`, `n_layers=2` | `gpbo_dkl_seeded.json` |
| `gpbo_cake` | `botorch_cake` | EI | argmax | none | `use_llm=False`, `xi=0.01`, `evolve_interval=5`, `population_size=6` | `gpbo_cake_seeded.json` |
| `lgbo_mean_shift` | `botorch_matern` | EI | argmax | `lgbo_mean_shift` | `use_llm=True`, `xi=0.01` | `lgbo_mean_shift_seeded.json` |
| `lgbo_cake` | `botorch_cake` | EI | argmax | `lgbo_mean_shift` | `use_llm=True`, `xi=0.01`, `evolve_interval=5`, `population_size=6` | `lgbo_cake_seeded.json` |

### Method short descriptions

- **gpbo_ei** — pure GP + EI baseline, no LLM.
- **gpbo_manifold** — Kernel Manifold evolution of the kernel structure every 5 steps; no LLM.
- **gpbo_alas** — ALAS learnable α-stable kernel (`init_alpha=1.5`); no LLM.
- **gpbo_dkl** — Deep Kernel Learning (`hidden_dim=16`, 2 layers); no LLM.
- **gpbo_cake** — CAKE kernel population (size 6) with structural evolution every 5 steps; LLM is used only for kernel evolution, not for candidate selection (`llm_strategy=none`, `use_llm=False` at the selection level).
- **lgbo_mean_shift** — standard Matérn GP + EI, plus LGBO point-mode mean-shift driven by the LLM (`use_llm=True`).
- **lgbo_cake** — CAKE kernel evolution combined with LGBO mean-shift (both LLM entry points on).

### Seeded per-dataset mean metrics

`t95` is lower-is-better; `best_found` and `AUC` are higher-is-better.

| Method | Dataset | best_found | t95 | AUC |
|---|---|---:|---:|---:|
| gpbo_ei | buchwald_sub4 | 82.5933 | 27.60 | 76.8078 |
| gpbo_ei | suzuki | 92.3165 | 28.55 | 85.3356 |
| gpbo_manifold | buchwald_sub4 | 84.4554 | 21.55 | 79.0192 |
| gpbo_manifold | suzuki | 97.2660 | 20.45 | 87.7596 |
| gpbo_alas | buchwald_sub4 | 82.5849 | 25.45 | 76.7341 |
| gpbo_alas | suzuki | 88.8565 | 28.45 | 83.3371 |
| gpbo_dkl | buchwald_sub4 | 84.7376 | 20.80 | 79.3892 |
| gpbo_dkl | suzuki | 92.9570 | 24.80 | 86.8569 |
| gpbo_cake | buchwald_sub4 | 85.2091 | 20.85 | 78.3566 |
| gpbo_cake | suzuki | 95.3215 | 21.70 | 87.3488 |
| lgbo_mean_shift | buchwald_sub4 | 82.5947 | 25.05 | 76.8134 |
| lgbo_mean_shift | suzuki | 93.7885 | 23.50 | 86.6481 |
| lgbo_cake | buchwald_sub4 | 84.9019 | 20.95 | 78.1670 |
| lgbo_cake | suzuki | 95.6410 | 20.20 | 88.0761 |

---

## Legacy full-prior methods (H1–H4)

These files predate protocol metadata. They use the complete train set as prior
(not `n_initial=5`). Seeds and datasets are the same 20 × 2 coverage, but
deterministic pure-GP runs produced no meaningful seed variance. Roadmap marks
their conclusions as legacy.

| File | Composition | Hypothesis | Notes |
|---|---|---|---|
| `H1_gpbo_manifold.json` | `gpbo_manifold` | H1 | Kernel Manifold |
| `H2_gpbo_alas.json` | `gpbo_alas` | H2 | ALAS |
| `H3_gpbo_dkl.json` | `gpbo_dkl` | H3 | DKL |
| `H4_gpbo_cake.json` | `gpbo_cake` | H4 | CAKE base |
| `H4_gpbo_cake_fixed.json` | `gpbo_cake` | H4 variant | Same metrics as base CAKE |
| `H4_gpbo_cake_prompt_fixed.json` | `gpbo_cake` | H4 variant | Same metrics as base CAKE |
| `H4_gpbo_cake_ensemble.json` | `gpbo_cake_ensemble` | H4 variant | Ensemble CAKE |

Legacy per-dataset means are listed in `EXPERIMENT_MANIFEST.md`.

---

## Methods defined but not present in this archive

These compositions exist in `compositions/base.py` but have no completed result
file under either protocol in this archive:

| Method | Surrogate | Acquisition | LLM strategy | Notes |
|---|---|---|---|---|
| `gpbo_ucb` | `botorch_matern` | UCB (`kappa=2.576`) | none | smoke-tier only |
| `lmabo_adaptive` | `botorch_matern` | EI | `lmabo_adaptive_acq` | not run to full |
| `bora_adaptive` | `botorch_matern` | EI | `bora_adaptive` | not run to full |
| `llm_in_loop` | `botorch_matern` | EI | `llm_in_loop_pick` | not run to full |
| `lgbo_softmax` | `botorch_matern` | EI | `lgbo_mean_shift` | selector=`softmax_explore` |
| `lmabo_ucb` | `botorch_matern` | UCB | `lmabo_adaptive_acq` | not run to full |

---

## What a single result record looks like (seeded)

```json
{
  "composition": "lgbo_mean_shift",
  "dataset": "buchwald_sub4",
  "seed": 100,
  "prior_protocol": "seeded_subsample",
  "n_initial": 5,
  "initial_indices": [33, 34, 31, 5, 1],
  "status": "ok",
  "elapsed_s": <float>,
  "metrics": {
    "best_found": 80.68492948,
    "initial_round_found_best": 61.04819615,
    "t95": 41,
    "AUC_best_so_far": 77.9820809855
  }
}
```

Legacy records omit `prior_protocol`, `n_initial`, `initial_indices`, and
`status`, and may carry extra fields such as `analysis`, `kernel_history`,
and `final_kernel`.

---

## How to re-read these results

1. Confirm the protocol of the file you are looking at (`seeded_subsample` vs
   unlabeled legacy vs future `fixed_train_prior`).
2. Do not mix protocols in a leaderboard or composite score.
3. Use `EXPERIMENT_MANIFEST.md` for the full checksum table.
4. Use `ARCHITECTURE.md` for the code path that produced these results.
5. The authoritative raw files remain at `../history/experiments/`; the
   copies under `snapshots/` are integrity backups only.
