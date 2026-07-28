# Auto-Research Result Snapshot

Captured: 2026-07-28T21:32:45+08:00
Source commit: `313c86cca68fd62adeea73734285aabaa12db2a5`

This snapshot preserves the two completed seeded-subsample LLM result files that were at risk of being lost during the transition to the competition fixed-prior protocol. The files are byte-for-byte copies; the source experiment JSON files were not modified.

| Snapshot | Records | Coverage | Protocol | Source state | SHA-256 |
|---|---:|---|---|---|---|
| `lgbo_mean_shift_seeded_40.json` | 40 | 2 datasets x 20 seeds | `seeded_subsample`, `n_initial=5` | Source tracked but expanded from 8 committed records to 40 uncommitted records | `ed56095d2eee9a32a770dca0eba4e658c72944d3c475f7736b471a454e6288df` |
| `lgbo_cake_seeded_40.json` | 40 | 2 datasets x 20 seeds | `seeded_subsample`, `n_initial=5` | Source untracked | `e0692dd4027d43ad4994073ca83038a4d0693515903871ce8181a2337a10a353` |

All 80 records have `status="ok"`. Seeds are `100, 200, ..., 2000` for both `buchwald_sub4` and `suzuki`.

These are historical five-point-prior results. They are not competition fixed-prior results and must not be aggregated with `prior_protocol="fixed_train_prior"` outputs.

No fixed-prior result file existed when this snapshot was taken. `rerun_llm.pid` contained PID `108965`, but that process was no longer running.
