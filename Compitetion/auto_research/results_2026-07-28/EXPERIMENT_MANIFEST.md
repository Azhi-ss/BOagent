# Auto-Research Experiment Manifest

Generated from the files present on 2026-07-28T21:32:45+08:00 at source commit `313c86cca68fd62adeea73734285aabaa12db2a5` plus uncommitted working-tree results.

All raw JSON files listed below were left unchanged. Metrics are arithmetic means over the 20 records for each dataset. `t95` is lower-is-better; yield and AUC metrics are higher-is-better.

## Protocol Warning

Do not compare or aggregate rows across protocol families:

- `legacy_full_prior`: inferred from `roadmap.md` and `history/ledger.jsonl`; raw JSON is unlabeled.
- `seeded_subsample`: explicit in JSON, with `n_initial=5` and five `initial_indices`.
- `fixed_train_prior`: current runtime protocol; no result files existed at snapshot time.

## Seeded-Subsample Results (`n_initial=5`)

Every file has 40/40 `status="ok"` records: 20 seeds (`100..2000`) on each of
`buchwald_sub4` and `suzuki`.

**Protocol fields present in every record:**

| Field | Value |
|---|---|
| `prior_protocol` | `"seeded_subsample"` |
| `n_initial` | **`5`** |
| `initial_indices` | 5 train-row indices drawn by `RandomState(seed)` |
| `status` | `"ok"` |

Method component tables (surrogate / acquisition / selector / LLM) are in
`METHODS.md`.

| File | Dataset | best_found | t95 | AUC | SHA-256 prefix |
|---|---|---:|---:|---:|---|
| `gpbo_ei_seeded.json` | buchwald_sub4 | 82.5933 | 27.60 | 76.8078 | `572bc2837830` |
| `gpbo_ei_seeded.json` | suzuki | 92.3165 | 28.55 | 85.3356 | `572bc2837830` |
| `gpbo_manifold_seeded.json` | buchwald_sub4 | 84.4554 | 21.55 | 79.0192 | `0e4cbffd7dd9` |
| `gpbo_manifold_seeded.json` | suzuki | 97.2660 | 20.45 | 87.7596 | `0e4cbffd7dd9` |
| `gpbo_alas_seeded.json` | buchwald_sub4 | 82.5849 | 25.45 | 76.7341 | `11176dd994e8` |
| `gpbo_alas_seeded.json` | suzuki | 88.8565 | 28.45 | 83.3371 | `11176dd994e8` |
| `gpbo_dkl_seeded.json` | buchwald_sub4 | 84.7376 | 20.80 | 79.3892 | `f329c08e7223` |
| `gpbo_dkl_seeded.json` | suzuki | 92.9570 | 24.80 | 86.8569 | `f329c08e7223` |
| `gpbo_cake_seeded.json` | buchwald_sub4 | 85.2091 | 20.85 | 78.3566 | `c04815a54a58` |
| `gpbo_cake_seeded.json` | suzuki | 95.3215 | 21.70 | 87.3488 | `c04815a54a58` |
| `lgbo_mean_shift_seeded.json` | buchwald_sub4 | 82.5947 | 25.05 | 76.8134 | `ed56095d2eee` |
| `lgbo_mean_shift_seeded.json` | suzuki | 93.7885 | 23.50 | 86.6481 | `ed56095d2eee` |
| `lgbo_cake_seeded.json` | buchwald_sub4 | 84.9019 | 20.95 | 78.1670 | `e0692dd4027d` |
| `lgbo_cake_seeded.json` | suzuki | 95.6410 | 20.20 | 88.0761 | `e0692dd4027d` |

## Legacy Full-Prior Results

Each file has 40 records covering both datasets and all 20 seeds. These files predate protocol metadata and generally omit `status`. The roadmap marks their conclusions as legacy because the deterministic full-prior runs produced no meaningful seed variance.

| File | Dataset | best_found | t95 | AUC | SHA-256 prefix |
|---|---|---:|---:|---:|---|
| `H1_gpbo_manifold.json` | buchwald_sub4 | 85.8947 | 8.00 | 82.9645 | `521348509bfd` |
| `H1_gpbo_manifold.json` | suzuki | 96.7400 | 4.00 | 94.8443 | `521348509bfd` |
| `H2_gpbo_alas.json` | buchwald_sub4 | 79.7664 | 41.00 | 73.1950 | `69b257300026` |
| `H2_gpbo_alas.json` | suzuki | 96.2000 | 17.00 | 92.0567 | `69b257300026` |
| `H3_gpbo_dkl.json` | buchwald_sub4 | 84.8955 | 2.00 | 83.6013 | `723539b85da2` |
| `H3_gpbo_dkl.json` | suzuki | 96.9700 | 10.00 | 93.3042 | `723539b85da2` |
| `H4_gpbo_cake.json` | buchwald_sub4 | 83.0965 | 9.00 | 81.6450 | `2fc18f64f928` |
| `H4_gpbo_cake.json` | suzuki | 98.6900 | 6.00 | 93.8610 | `2fc18f64f928` |
| `H4_gpbo_cake_fixed.json` | buchwald_sub4 | 83.0965 | 9.00 | 81.6450 | `fc08d0a8316e` |
| `H4_gpbo_cake_fixed.json` | suzuki | 98.6900 | 6.00 | 93.8610 | `fc08d0a8316e` |
| `H4_gpbo_cake_prompt_fixed.json` | buchwald_sub4 | 83.0965 | 9.00 | 81.6450 | `d03023774c28` |
| `H4_gpbo_cake_prompt_fixed.json` | suzuki | 98.6900 | 6.00 | 93.8610 | `d03023774c28` |
| `H4_gpbo_cake_ensemble.json` | buchwald_sub4 | 85.8947 | 8.00 | 81.1760 | `46182a9cd835` |
| `H4_gpbo_cake_ensemble.json` | suzuki | 98.6900 | 24.00 | 92.9765 | `46182a9cd835` |

The authoritative interpretation and hypothesis verdicts remain in `../roadmap.md` and `ledger.jsonl`; this manifest records the artifacts rather than rejudging them.

## Earlier Tier Artifacts

| File | Records | Coverage | Protocol metadata | SHA-256 prefix |
|---|---:|---|---|---|
| `full_20260725_combined.json` | 40 | `gpbo_ei`, both datasets, 20 seeds | unlabeled | `0fabedb0eace` |
| `full_buchwald.json` | 20 | `gpbo_ei`, Buchwald, 20 seeds | unlabeled | `e61881062487` |
| `full_suzuki.json` | 20 | `gpbo_ei`, Suzuki, 20 seeds | unlabeled | `5da3fb784131` |
| `confirm_20260725_061438.json` | 10 | `gpbo_ei`, both datasets, 5 seeds | unlabeled | `80342b5403e0` |
| `smoke_gpbo_20260725_055641.json` | 12 | `gpbo_ei` + `gpbo_ucb`, both datasets, 3 seeds | unlabeled | `05d99a4f8bf2` |
| `ledger.jsonl` | append-only | research events and conclusions | n/a | `689fe71abb08` |
| `champion.json` | object | last persisted champion | n/a | `9e8fe781395f` |
| `../baselines/seed_baseline.json` | object | baseline reference | n/a | `d34984d2347` |

## Full Experiment Checksums

```text
11176dd994e89997fa287fa43220482401e87dbb4b453f6b6964730ca779c96b  gpbo_alas_seeded.json
c04815a54a58179e1e115e3c752966ff95193350535be258323a8a1e4cff1b2d  gpbo_cake_seeded.json
f329c08e7223ff2560e591ae554e8dc0ca35a931eba69177ac0078d30a3edf48  gpbo_dkl_seeded.json
572bc283783077a276af832013cbc8768615965bb8d4676565a8877410199b24  gpbo_ei_seeded.json
0e4cbffd7dd916de37d70ac927c889c887f7dd3c32e0fb3229d75e9937b1cb7f  gpbo_manifold_seeded.json
521348509bfdaad9bab53de7f01f7aa9c45de5ed63468c2684caab629f083391  H1_gpbo_manifold.json
69b25730002692c24676dad4573384102fa88025b4405a76123d6049ab3684f1  H2_gpbo_alas.json
723539b85da2a24312de3463a5e512dba2519d41c65d1b0f667a831690442970  H3_gpbo_dkl.json
46182a9cd83535754affbda1f77539a25e80e848cd92666e199aec2192b734c  H4_gpbo_cake_ensemble.json
fc08d0a8316e5e496475ce8768f741117014dc6b070a113d31efc67791c56c44  H4_gpbo_cake_fixed.json
2fc18f64f928908043b521569bcd2914619e3ef32b58a68abb6a6cbdc9b47fa1  H4_gpbo_cake.json
d03023774c287bd5ce72c1d68e8f4238a9c6b1defeda1734880d64f137cffba1  H4_gpbo_cake_prompt_fixed.json
e0692dd4027d43ad4994073ca83038a4d0693515903871ce8181a2337a10a353  lgbo_cake_seeded.json
ed56095d2eee9a32a770dca0eba4e658c72944d3c475f7736b471a454e6288df  lgbo_mean_shift_seeded.json
```

## Git and Snapshot State

At manifest generation time:

- `lgbo_mean_shift_seeded.json` was tracked but modified: committed `HEAD` held only 8 Buchwald records; the working file held the complete 40 records.
- `lgbo_cake_seeded.json` was untracked and held 40 complete records.
- Both vulnerable files were copied byte-for-byte to `snapshots/2026-07-28-fixed-prior-transition/` with checksums.
- The other 12 experiment JSON files were already tracked by Git.
- No `*_fixed_prior.json` result existed, and the PID in `rerun_llm.pid` was stale.

These new manifest/snapshot files are themselves uncommitted until included in the task's Phase 3.4 commit.
