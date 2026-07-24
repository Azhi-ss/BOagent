---
name: dataset-alignment
description: Standardizes raw scientific datasets (Perovskite, Reaction Optimization, Battery, etc.) into the unified BOagent dataset schema. Use when migrating, formatting, or creating new dataset benchmarks for bo-core.
---

# Dataset Alignment & Standardization Specification

This skill enforces a **Unified Schema** across all experimental and benchmark datasets in BOagent. Every dataset must be organized as a self-contained directory under `datasets/<category>/<dataset_name>/`.

## Standard Directory Structure

```
datasets/<category>/<dataset_name>/
├── searchspace.csv      # Complete candidate space (all rows, includes target column)
├── train.csv            # Labeled prior/initial training split (includes target column)
├── test.csv             # Labeled test/groundtruth split (includes target column)
├── test_features.csv    # Unobserved candidate pool features (EXCLUDES target column)
├── options.json         # Dictionary mapping feature names to candidate lists or range bounds
└── README.md            # Metadata, domain description, feature definitions, and target units
```

## Standardized Dataset Categories

| Category | Description & Datasets | Target Column |
| :--- | :--- | :--- |
| `perovskite/` | Perovskite solar cell formulation & physics (`band_alignment`, `defects_doping`) | `eta` (PCE %) |
| `chemical_reactions/` | 2026 Material Science Reaction Optimization (`buchwald_sub4`, `suzuki`) | `Yield` (%) |
| `battery/` | Battery electrolyte & cathode recipes (`battery_bo_searchspace`) | Specific Capacity |

## File Specifications & Contracts

1. **`searchspace.csv`**:
   - Contains all $N$ candidate rows.
   - Columns: `[Feature_1, Feature_2, ..., Target_Col]`.
   - All feature values must be clean strings or numeric floats without NaN values in required features.

2. **`train.csv` & `test.csv`**:
   - Pre-split deterministic subsets of `searchspace.csv`.
   - `train.csv` contains $N_{train}$ rows (e.g. 10 for perovskite, 35 for Buchwald, 29 for Suzuki).
   - `test.csv` contains $N_{test} = N - N_{train}$ rows.

3. **`test_features.csv`**:
   - Matches `test.csv` line for line, but **omits the target column** to prevent data leakage during LLM exploration.

4. **`options.json`**:
   - JSON key-value pairs where keys are feature names and values are sorted arrays of valid unique categorical options or discrete levels.

5. **`README.md`**:
   - Markdown documenting: Domain context, target column units, feature definitions, row counts, and source literature DOI references.

## Execution Workflow: Converting a Raw Dataset

When standardizing a new raw dataset (e.g. `.xlsx` or raw `.csv`):

1. **Create Target Directory**:
   `mkdir -p datasets/<category>/<dataset_name>`

2. **Generate Standard CSVs & `options.json`**:
   Extract feature columns and target column, shuffle/split with local `np.random.RandomState(seed=42)`:
   - Export `searchspace.csv`, `train.csv`, `test.csv`, and `test_features.csv`.
   - Export `options.json` using unique values per feature column.

3. **Update Data Loader (`data_loader.py`)**:
   - Add loader function `load_<dataset_name>_data()` in `packages/bo-core/bo_core/benchmark/data_loader.py`.
   - Register dataset key in `DATA_LOADERS` dict.

4. **Run Verification**:
   - Execute `python -m pytest packages/bo-core/tests/test_dataset_loaders.py` to confirm clean loading.
