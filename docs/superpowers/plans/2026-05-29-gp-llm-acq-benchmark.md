# GP+LLM Acquisition & Benchmark 对齐 PVK-LLM 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 GP+LLM 两阶段 acquisition function 和 benchmark 基础设施，对齐 PVK-LLM benchmark 脚本的行为。

**Architecture:** 新增 `gp_llm_acq.py`（独立模块，不修改 PVK-LLM 核心库）和 `benchmark/` 包（data_loader、runner、cli），在 `pvk_llm_bo_runtime.py` 和 `api.py` 中添加注入点和 API endpoint。

**Tech Stack:** Python 3.10+, sklearn GaussianProcessRegressor, pandas, DeepSeekClient (OpenAI 兼容), FastAPI, argparse

**Spec:** `docs/superpowers/specs/2026-05-29-gp-llm-acq-benchmark-design.md`

---

## 文件结构

```
BOagent/backend/
├── gp_llm_acq.py                # [NEW] GP+LLM 两阶段 ACQ
├── benchmark/
│   ├── __init__.py               # [NEW] 空 init
│   ├── data_loader.py            # [NEW] 数据加载 + train/test split
│   ├── runner.py                 # [NEW] Benchmark 执行引擎
│   └── cli.py                    # [NEW] CLI 入口
├── pvk_llm_bo_runtime.py         # [MODIFY] acq_class 注入参数
├── api.py                        # [MODIFY] POST /api/v1/benchmark
└── tests/
    ├── test_gp_llm_acq.py        # [NEW] GPLLM_ACQ 单元测试
    ├── test_benchmark_data_loader.py  # [NEW] 数据加载测试
    └── test_benchmark_runner.py       # [NEW] Runner 测试
```

---

### Task 1: 添加 sklearn 依赖

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: 在 requirements.txt 中添加 scikit-learn**

```diff
+ scikit-learn
```

Run: `cd backend && grep scikit-learn requirements.txt`
Expected: `scikit-learn`

- [ ] **Step 2: 安装依赖**

Run: `cd backend && pip install scikit-learn`
Expected: Successfully installed scikit-learn

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add scikit-learn dependency for GP+LLM ACQ"
```

---

### Task 2: 实现 gp_llm_acq.py — GP+LLM 两阶段 ACQ 核心

**Files:**
- Create: `backend/gp_llm_acq.py`

- [ ] **Step 1: 创建 gp_llm_acq.py**

```python
from __future__ import annotations

import os
import time
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from llm_client import DeepSeekClient, LlmCallResult


class GPLLM_ACQ:
    """GP pre-filtering + LLM selection two-stage acquisition function.

    Aligns with PVK-LLM benchmark CustomLLM_ACQ logic:
        1. Train sklearn GP on observed (config, score) pairs
        2. UCB-score all unobserved points in the full dataset
        3. Send top-k candidates to LLM for materials-science reasoning
        4. LLM returns refined n_candidates selections
        5. Fall back to GP UCB ranking if LLM fails
    """

    def __init__(
        self,
        task_context: dict[str, Any],
        n_candidates: int,
        n_templates: int,
        lower_is_better: bool,
        chat_engine: str,
        top_k: int = 20,
        alpha: float = 0.1,
    ) -> None:
        self.task_context = task_context
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.lower_is_better = lower_is_better
        self.chat_engine = chat_engine
        self.top_k = top_k
        self.alpha = alpha
        self.feature_cols: list[str] = list(task_context["feature_cols"])
        self.target_col: str = str(task_context["target_col"])
        self.df: pd.DataFrame = task_context["df"]
        self.hyperparameter_constraints: dict = task_context.get(
            "hyperparameter_constraints", {}
        )
        self._llm_client: DeepSeekClient | None = None

    def _get_llm_client(self) -> DeepSeekClient:
        if self._llm_client is None:
            self._llm_client = DeepSeekClient.from_env()
            if self.chat_engine:
                self._llm_client.model = self.chat_engine
        return self._llm_client

    # ------------------------------------------------------------------
    # Public interface (matches PVK-LLM LLM_ACQ signature)
    # ------------------------------------------------------------------

    def get_candidate_points(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[pd.DataFrame, float, float]:
        """Generate candidate points via GP pre-filter + LLM refinement.

        Returns:
            (candidate_points_df, cost, time_taken)
        """
        start_time = time.time()
        alpha = alpha if alpha is not None else self.alpha

        # 1. Build GP training data from observed points
        X_train, y_train = self._build_training_data(observed_configs, observed_fvals)

        # 2. Find unobserved points in full dataset
        unobserved = self._find_unobserved(observed_configs)

        # 3. Handle edge case: no unobserved points left
        if not unobserved:
            return self._fallback_random_candidates(observed_configs), 0.0, time.time() - start_time

        # 4. Train GP and compute UCB scores
        try:
            gp_predictions = self._gp_ucb_predict(X_train, y_train, unobserved, alpha)
        except Exception:
            # GP training failed (e.g. too few points) — random sample
            return self._fallback_random_from(unobserved), 0.0, time.time() - start_time

        # 5. Take top-k by UCB
        gp_predictions.sort(key=lambda x: x[3], reverse=True)
        top_formulas = gp_predictions[: self.top_k]

        # 6. LLM batch evaluation
        observed_data = self._build_observed_data(observed_configs, observed_fvals)

        try:
            prompt = self._build_batch_evaluation_prompt(top_formulas, observed_data)
            response_text = self._call_llm(prompt)
            selected_indices = self._parse_llm_response(response_text, len(top_formulas))
            selected = [top_formulas[i] for i in selected_indices[: self.n_candidates]]
        except Exception:
            # LLM failed — fall back to GP UCB top-n_candidates
            top_formulas.sort(key=lambda x: x[3], reverse=True)
            selected = top_formulas[: self.n_candidates]

        # 7. Build result DataFrame
        candidate_points = pd.DataFrame()
        for formula_dict, _idx, _actual, _ucb, _mean, _std in selected:
            candidate_points = pd.concat(
                [candidate_points, pd.DataFrame([formula_dict])],
                ignore_index=True,
            )

        end_time = time.time()
        return candidate_points, 0.0, end_time - start_time

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_train = []
        y_train = []
        for i, (_, obs_config) in enumerate(observed_configs.iterrows()):
            features = [obs_config[col] for col in self.feature_cols]
            X_train.append(features)
            y_train.append(observed_fvals.iloc[i]["score"])
        return np.array(X_train), np.array(y_train)

    def _find_unobserved(
        self, observed_configs: pd.DataFrame
    ) -> list[tuple[dict[str, float], int, float]]:
        unobserved: list[tuple[dict[str, float], int, float]] = []
        for i, row in self.df.iterrows():
            formula = row[self.feature_cols].values.astype(float)
            is_observed = False
            for _, obs in observed_configs.iterrows():
                obs_values = np.array(
                    [float(obs[col]) for col in self.feature_cols]
                )
                if np.allclose(formula, obs_values, rtol=1e-5):
                    is_observed = True
                    break
            if not is_observed:
                formula_dict = {
                    col: float(formula[j]) for j, col in enumerate(self.feature_cols)
                }
                unobserved.append(
                    (formula_dict, i, float(row[self.target_col]))
                )
        return unobserved

    def _gp_ucb_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        unobserved: list[tuple[dict[str, float], int, float]],
        alpha: float,
    ) -> list[tuple[dict[str, float], int, float, float, float, float]]:
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            [1.0] * len(self.feature_cols), (1e-2, 1e2)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True,
        )
        gp.fit(X_train_scaled, y_train)

        predictions: list[
            tuple[dict[str, float], int, float, float, float, float]
        ] = []
        for formula_dict, idx, actual_val in unobserved:
            features = np.array(
                [[formula_dict[col] for col in self.feature_cols]]
            )
            features_scaled = scaler_X.transform(features)
            pred_mean, pred_std = gp.predict(features_scaled, return_std=True)
            ucb_score = float(pred_mean[0] + alpha * pred_std[0])
            predictions.append(
                (
                    formula_dict,
                    idx,
                    actual_val,
                    ucb_score,
                    float(pred_mean[0]),
                    float(pred_std[0]),
                )
            )
        return predictions

    def _build_observed_data(
        self,
        observed_configs: pd.DataFrame,
        observed_fvals: pd.DataFrame,
    ) -> list[tuple[dict[str, float], float]]:
        observed_data: list[tuple[dict[str, float], float]] = []
        for i, (_, obs_config) in enumerate(observed_configs.iterrows()):
            obs_values = {
                col: float(obs_config[col]) for col in self.feature_cols
            }
            obs_val = float(observed_fvals.iloc[i]["score"])
            observed_data.append((obs_values, obs_val))
        return observed_data

    def _build_batch_evaluation_prompt(
        self,
        top_formulas: list[
            tuple[dict[str, float], int, float, float, float, float]
        ],
        observed_data: list[tuple[dict[str, float], float]],
    ) -> str:
        """Build materials-science domain prompt for LLM batch evaluation."""
        model_name = self.task_context.get("model", "perovskite")
        target_name = self.target_col

        prompt = f"""
You are a professional materials scientist specializing in perovskite solar cell optimization. I will provide you with multiple candidate formulations, some observed formulation data, and the prediction results from a Gaussian process model. Please select the formulation from these that is most likely to produce high power conversion efficiency ({target_name}).

Known Information:

The efficiency ({target_name}) of perovskite solar cells is closely related to the material characteristics.

Key parameters include:
"""
        for col in self.feature_cols:
            prompt += f"\n{col}: Feature parameter for {model_name} optimization"

        prompt += "\n\nObserved formulations and their efficiencies:"

        for i, (obs_values, obs_eta) in enumerate(observed_data):
            prompt += f"\n formulation{i + 1}: "
            for col in self.feature_cols:
                prompt += f"{col}={obs_values[col]:.4f}, "
            prompt += f"{target_name}={obs_eta:.4f}"

        prompt += "\n\nCandidate formulations:"

        for i, (
            formula_dict,
            _idx,
            _actual,
            _ucb,
            pred_mean,
            pred_std,
        ) in enumerate(top_formulas):
            prompt += f"\n\nCandidate formulation {i + 1}:\n"
            for col in self.feature_cols:
                prompt += f"{col}={formula_dict[col]:.4f}, "
            prompt += f"\nGP Prediction: mean={pred_mean:.4f}, std={pred_std:.4f}"

        prompt += f"""

Based on materials science principles, the observed data, and the GP model predictions, please select the {self.n_candidates} formulations from the candidates above that are most likely to produce high {target_name}. Considerations should include:
1. Similarity to known high-efficiency formulations
2. The likely efficiency of charge separation and transport
3. The potential for recombination losses
4. Novelty or innovation compared to observed formulations

Please respond in the following format:
Analysis: [Your detailed analysis]
Selected Formulations: [List of formulation numbers, e.g., 1, 3, 5, 7, 9]
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via DeepSeekClient for batch evaluation."""
        client = self._get_llm_client()

        if not client.is_configured():
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not configured; cannot call LLM for batch evaluation"
            )

        result: LlmCallResult = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional materials scientist specializing in "
                        "perovskite solar cell optimization. Your task is to select "
                        "formulations from the candidates that are most likely to "
                        "produce high power conversion efficiency."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            extra_body={"thinking": {"type": "disabled"}},
        )

        if result.status != "success":
            raise RuntimeError(
                f"LLM call failed: {result.error or result.status}"
            )

        return result.content

    def _parse_llm_response(
        self, response: str, num_formulas: int
    ) -> list[int]:
        """Parse LLM response to extract selected formulation indices (0-indexed)."""
        # Try "Selected Formulations:" pattern first
        patterns = [
            r"Selected\s+Formulations?\s*:\s*(.+)",
            r"Selected\s+formulations?\s*:\s*(.+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                numbers_text = match.group(1).strip()
                indices = [
                    int(x.strip()) - 1
                    for x in re.split(r"[,\s]+", numbers_text)
                    if x.strip().isdigit()
                ]
                if indices:
                    return [i for i in indices if 0 <= i < num_formulas]

        # Fallback: find all numbers in response
        all_numbers = re.findall(r"\b(\d+)\b", response)
        indices = [int(n) - 1 for n in all_numbers if 1 <= int(n) <= num_formulas]
        if indices:
            seen: set[int] = set()
            unique = [i for i in indices if not (i in seen or seen.add(i))]
            return unique

        # Last resort: first n_candidates
        return list(range(min(self.n_candidates, num_formulas)))

    def _fallback_random_candidates(
        self, observed_configs: pd.DataFrame
    ) -> pd.DataFrame:
        """Return random candidates from observed when no unobserved points remain."""
        n = min(self.n_candidates, len(observed_configs))
        sampled = observed_configs.sample(n=n, replace=False)
        result = pd.DataFrame()
        for _, row in sampled.iterrows():
            config = {col: float(row[col]) for col in self.feature_cols}
            result = pd.concat(
                [result, pd.DataFrame([config])], ignore_index=True
            )
        return result

    def _fallback_random_from(
        self,
        unobserved: list[tuple[dict[str, float], int, float]],
    ) -> tuple[pd.DataFrame, float, float]:
        """Return random unobserved candidates when GP training fails."""
        n = min(self.n_candidates, len(unobserved))
        rng = np.random.RandomState(42)
        indices = rng.choice(len(unobserved), size=n, replace=False)
        result = pd.DataFrame()
        for idx in indices:
            result = pd.concat(
                [result, pd.DataFrame([unobserved[idx][0]])],
                ignore_index=True,
            )
        return result, 0.0, 0.0
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd backend && python -c "from gp_llm_acq import GPLLM_ACQ; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/gp_llm_acq.py
git commit -m "feat: add GP+LLM two-stage acquisition function (GPLLM_ACQ)"
```

---

### Task 3: 实现 benchmark/data_loader.py — 数据加载

**Files:**
- Create: `backend/benchmark/__init__.py`
- Create: `backend/benchmark/data_loader.py`

- [ ] **Step 1: 创建 benchmark/__init__.py**

```python
"""BOagent benchmark infrastructure — GP+LLM BO evaluation pipeline."""
```

- [ ] **Step 2: 创建 benchmark/data_loader.py**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Default data paths (relative to PVK-LLM project root)
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "PVK-LLM" / "custom_perovskite_dataset"

BAND_ALIGNMENT_FEATURES = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
DEFECTS_DOPING_FEATURES = [
    "Nt_PVK/ETL", "Nt_HTL/PVK", "Na_PVK", "Nd_PVK",
    "Na_HTL", "Nd_HTL", "Na_ETL", "Nd_ETL",
]
TARGET_COL = "eta"


def load_band_alignment_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load band alignment dataset and split into train/test.

    Returns:
        dict with keys: train_x, train_y, test_x, test_y,
                        feature_cols, target_col, df
    """
    if file_path is None:
        file_path = DEFAULT_DATA_ROOT / "bandAlignment.xlsx"
    df = pd.read_excel(Path(file_path))
    return _split_data(df, BAND_ALIGNMENT_FEATURES, TARGET_COL, n_train, seed)


def load_defects_doping_data(
    file_path: str | Path | None = None,
    n_train: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Load defects & doping dataset and split into train/test.

    Returns:
        dict with keys: train_x, train_y, test_x, test_y,
                        feature_cols, target_col, df
    """
    if file_path is None:
        file_path = DEFAULT_DATA_ROOT / "defectsAndDoping.xlsx"
    df = pd.read_excel(Path(file_path))
    return _split_data(df, DEFECTS_DOPING_FEATURES, TARGET_COL, n_train, seed)


def _split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_train: int,
    seed: int,
) -> dict[str, Any]:
    """Split DataFrame into train/test sets."""
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset. Available: {df.columns.tolist()}")

    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return {
        "train_x": train_df[feature_cols].values,
        "train_y": train_df[target_col].values,
        "test_x": test_df[feature_cols].values,
        "test_y": test_df[target_col].values,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "df": df,
    }


def build_task_context(
    task_id: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Build task_context dict from loaded data, matching PVK-LLM format.

    Args:
        task_id: "band_alignment" or "defects_doping"
        data: dict from load_*_data()
    """
    feature_cols = data["feature_cols"]
    df = data["df"]

    hyperparameter_constraints: dict[str, list[Any]] = {}
    for col in feature_cols:
        col_data = pd.to_numeric(df[col], errors="coerce").dropna()
        hyperparameter_constraints[col] = [
            "float",
            "linear",
            [float(col_data.min()), float(col_data.max())],
        ]

    return {
        "model": task_id,
        "task": "regression",
        "metric": "neg_mean_squared_error",
        "num_classes": 1,
        "n_classes": 1,
        "lower_is_better": False,
        "num_samples": int(len(df)),
        "tot_feats": len(feature_cols),
        "cat_feats": 0,
        "num_feats": len(feature_cols),
        "feature_cols": feature_cols,
        "target_col": data["target_col"],
        "hyperparameter_constraints": hyperparameter_constraints,
        "df": df,
    }


DATA_LOADERS = {
    "band_alignment": load_band_alignment_data,
    "defects_doping": load_defects_doping_data,
}
```

- [ ] **Step 3: 验证导入**

Run: `cd backend && python -c "from benchmark.data_loader import DATA_LOADERS, build_task_context; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/benchmark/__init__.py backend/benchmark/data_loader.py
git commit -m "feat: add benchmark data loader with train/test split"
```

---

### Task 4: 实现 benchmark/runner.py — Benchmark 执行引擎

**Files:**
- Create: `backend/benchmark/runner.py`

- [ ] **Step 1: 创建 benchmark/runner.py**

```python
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.data_loader import DATA_LOADERS, build_task_context
from gp_llm_acq import GPLLM_ACQ


class BenchmarkRunner:
    """Execute a single-seed PVKBO benchmark run with GP+LLM ACQ."""

    def __init__(
        self,
        task_id: str,
        n_initial: int = 5,
        n_trials: int = 20,
        seed: int = 42,
        sm_mode: str = "discriminative",
        chat_engine: str = "deepseek-v4-flash",
        n_candidates: int = 10,
        n_templates: int = 2,
        n_gens: int = 5,
        alpha: float = 0.1,
        top_k: int = 20,
        output_dir: str | Path = "results",
        data_path: str | Path | None = None,
    ) -> None:
        self.task_id = task_id
        self.n_initial = n_initial
        self.n_trials = n_trials
        self.seed = seed
        self.sm_mode = sm_mode
        self.chat_engine = chat_engine
        self.n_candidates = n_candidates
        self.n_templates = n_templates
        self.n_gens = n_gens
        self.alpha = alpha
        self.top_k = top_k
        self.output_dir = Path(output_dir)
        self.data_path = Path(data_path) if data_path else None

        if task_id not in DATA_LOADERS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(DATA_LOADERS)}"
            )
        if sm_mode not in ("discriminative", "generative"):
            raise ValueError(
                f"Unknown sm_mode: {sm_mode}. Use 'discriminative' or 'generative'"
            )

    def run(self) -> dict[str, Any]:
        """Execute the benchmark and return results dict."""
        # 1. Load data
        loader = DATA_LOADERS[self.task_id]
        data = loader(
            file_path=self.data_path, n_train=self.n_initial, seed=self.seed
        )
        task_context = build_task_context(self.task_id, data)

        # 2. Initialize PVKBO components
        # Add PVK-LLM to path if needed
        pvk_root = _resolve_pvk_root()
        if pvk_root and str(pvk_root) not in sys.path:
            sys.path.insert(0, str(pvk_root))

        from pvk_bo.pvk_bo import PVKBO

        top_pct = 0.25 if self.sm_mode == "generative" else None

        # Build init_f from train set
        def init_f(n_samples: int) -> list[dict[str, float]]:
            rng = np.random.RandomState(self.seed)
            indices = rng.choice(
                len(data["train_x"]), min(n_samples, len(data["train_x"])),
                replace=False,
            )
            configs: list[dict[str, float]] = []
            for idx in indices:
                config = {}
                for j, col in enumerate(data["feature_cols"]):
                    config[col] = float(data["train_x"][idx, j])
                configs.append(config)
            return configs

        # Build bbox_eval_f
        def bbox_eval_f(
            candidate_config: dict[str, Any],
        ) -> tuple[dict[str, float], dict[str, float]]:
            config = {
                col: float(candidate_config[col])
                for col in data["feature_cols"]
            }
            X = np.array(
                [[config[col] for col in data["feature_cols"]]]
            )
            # Search full dataset for exact or nearest match
            all_X = data["df"][data["feature_cols"]].values.astype(float)
            all_y = data["df"][data["target_col"]].values.astype(float)
            distances = np.sqrt(np.sum((all_X - X) ** 2, axis=1))
            nearest_idx = int(np.argmin(distances))
            score = float(all_y[nearest_idx])

            # Compute generalization score on test set
            test_X = data["test_x"]
            test_y = data["test_y"]
            test_distances = np.sqrt(np.sum((test_X - X) ** 2, axis=1))
            test_nearest_idx = int(np.argmin(test_distances))
            gen_score = float(test_y[test_nearest_idx])

            return config, {
                "score": score,
                "generalization_score": gen_score,
            }

        # Instantiate PVKBO with GPLLM_ACQ
        pvkbo = PVKBO(
            task_context=task_context,
            sm_mode=self.sm_mode,
            n_candidates=self.n_candidates,
            n_templates=self.n_templates,
            n_gens=self.n_gens,
            alpha=self.alpha,
            n_initial_samples=self.n_initial,
            n_trials=self.n_trials,
            init_f=init_f,
            bbox_eval_f=bbox_eval_f,
            chat_engine=self.chat_engine,
            top_pct=top_pct,
        )

        # Replace acq_func with GPLLM_ACQ
        pvkbo.acq_func = GPLLM_ACQ(
            task_context=task_context,
            n_candidates=self.n_candidates,
            n_templates=self.n_templates,
            lower_is_better=task_context["lower_is_better"],
            chat_engine=self.chat_engine,
            top_k=self.top_k,
            alpha=self.alpha,
        )

        # 3. Run optimization
        configs, fvals = pvkbo.optimize(test_metric="generalization_score")

        # 4. Find best
        best_idx = fvals["score"].idxmax()
        best_config = configs.iloc[best_idx].to_dict()
        best_score = float(fvals.iloc[best_idx]["score"])
        best_gen_score = float(fvals.iloc[best_idx]["generalization_score"])

        # 5. Build search history
        search_history = pd.concat([configs, fvals], axis=1)

        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "configs": configs,
            "fvals": fvals,
            "search_history": search_history,
            "best_config": best_config,
            "best_score": best_score,
            "best_generalization_score": best_gen_score,
            "llm_query_cost": pvkbo.llm_query_cost,
            "llm_query_time": pvkbo.llm_query_time,
        }

    def save_results(self, result: dict[str, Any]) -> None:
        """Persist benchmark results to output_dir."""
        save_dir = (
            self.output_dir
            / f"results_{self.sm_mode}"
            / self.task_id
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # CSV: search history
        result["search_history"].to_csv(
            save_dir / f"{self.seed}.csv", index=False
        )

        # JSON: search info (cost/time metadata)
        search_info = {
            "llm_query_cost_breakdown": result["llm_query_cost"],
            "llm_query_time_breakdown": result["llm_query_time"],
            "llm_query_cost": sum(result["llm_query_cost"]),
            "llm_query_time": sum(result["llm_query_time"]),
        }
        with open(save_dir / f"{self.seed}_search_info.json", "w") as f:
            json.dump(search_info, f, indent=2)

        # JSON: summary
        summary = {
            "task_id": result["task_id"],
            "seed": result["seed"],
            "best_config": result["best_config"],
            "best_score": result["best_score"],
            "best_generalization_score": result["best_generalization_score"],
            "n_trials": self.n_trials,
            "n_initial": self.n_initial,
            "sm_mode": self.sm_mode,
            "chat_engine": self.chat_engine,
            "convergence_curve": result["fvals"]["score"].tolist(),
            "generalization_curve": result["fvals"]["generalization_score"].tolist(),
        }
        with open(save_dir / f"{self.seed}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


def run_multi_seed(
    task_id: str,
    seeds: list[int],
    output_dir: str | Path = "results",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run benchmark across multiple seeds sequentially.

    Returns:
        List of result dicts, one per seed.
    """
    results: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n{'=' * 80}")
        print(f"Running {task_id} benchmark — seed {seed}")
        print(f"{'=' * 80}")
        runner = BenchmarkRunner(
            task_id=task_id,
            seed=seed,
            output_dir=output_dir,
            **kwargs,
        )
        result = runner.run()
        runner.save_results(result)
        results.append(result)
        print(
            f"Seed {seed}: best_score={result['best_score']:.4f}, "
            f"best_gen={result['best_generalization_score']:.4f}"
        )
    return results


def _resolve_pvk_root() -> Path | None:
    """Resolve PVK-LLM project root from env or default location."""
    env_root = os.environ.get("PVK_LLM_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    # Default: sibling to BOagent
    default = Path(__file__).resolve().parent.parent.parent.parent / "PVK-LLM"
    if default.exists():
        return default
    return None
```

- [ ] **Step 2: 验证导入**

Run: `cd backend && python -c "from benchmark.runner import BenchmarkRunner, run_multi_seed; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/benchmark/runner.py
git commit -m "feat: add benchmark runner with GP+LLM ACQ integration"
```

---

### Task 5: 实现 benchmark/cli.py — CLI 入口

**Files:**
- Create: `backend/benchmark/cli.py`

- [ ] **Step 1: 创建 benchmark/cli.py**

```python
"""CLI entry point for PVKBO benchmark.

Usage:
    cd backend
    python -m benchmark.cli \\
        --task band_alignment \\
        --engine deepseek-v4-flash \\
        --sm_mode discriminative \\
        --n_trials 20 \\
        --n_initial 5 \\
        --seed 42 \\
        --output_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure backend/ is on sys.path when invoked as python -m benchmark.cli
_backend_dir = Path(__file__).resolve().parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PVKBO Benchmark — GP+LLM acquisition function evaluation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="band_alignment",
        choices=["band_alignment", "defects_doping"],
        help="Task to benchmark (default: band_alignment)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="deepseek-v4-flash",
        help="LLM chat engine model name (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--sm_mode",
        type=str,
        default="discriminative",
        choices=["discriminative", "generative"],
        help="Surrogate model mode (default: discriminative)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of BO trials (default: 20)",
    )
    parser.add_argument(
        "--n_initial",
        type=int,
        default=5,
        help="Number of initial samples (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed run (e.g. '42,123,456')",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=10,
        help="Number of candidate points per trial (default: 10)",
    )
    parser.add_argument(
        "--n_templates",
        type=int,
        default=2,
        help="Number of LLM prompt templates (default: 2)",
    )
    parser.add_argument(
        "--n_gens",
        type=int,
        default=5,
        help="Number of LLM generations (default: 5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="UCB exploration parameter (default: 0.1)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k GP candidates to send to LLM (default: 20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to Excel data file (default: PVK-LLM/custom_perovskite_dataset/)",
    )

    args = parser.parse_args()

    from benchmark.runner import BenchmarkRunner, run_multi_seed

    common_kwargs = {
        "task_id": args.task,
        "n_initial": args.n_initial,
        "n_trials": args.n_trials,
        "sm_mode": args.sm_mode,
        "chat_engine": args.engine,
        "n_candidates": args.n_candidates,
        "n_templates": args.n_templates,
        "n_gens": args.n_gens,
        "alpha": args.alpha,
        "top_k": args.top_k,
        "output_dir": args.output_dir,
        "data_path": args.data_path,
    }

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        results = run_multi_seed(seeds=seeds, **common_kwargs)
        print(f"\nMulti-seed complete. {len(results)} runs finished.")
        for r in results:
            print(
                f"  seed={r['seed']}: best={r['best_score']:.4f}, "
                f"gen={r['best_generalization_score']:.4f}"
            )
    else:
        runner = BenchmarkRunner(seed=args.seed, **common_kwargs)
        result = runner.run()
        runner.save_results(result)
        print(f"\nBenchmark complete.")
        print(f"  seed={result['seed']}")
        print(f"  best_score={result['best_score']:.4f}")
        print(f"  best_generalization_score={result['best_generalization_score']:.4f}")
        print(f"  results saved to: {runner.output_dir / f'results_{args.sm_mode}' / args.task}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 CLI help**

Run: `cd backend && python -m benchmark.cli --help`
Expected: argparse help output with all options

- [ ] **Step 3: Commit**

```bash
git add backend/benchmark/cli.py
git commit -m "feat: add benchmark CLI entry point"
```

---

### Task 6: 修改 pvk_llm_bo_runtime.py — 支持 ACQ 注入

**Files:**
- Modify: `backend/pvk_llm_bo_runtime.py`

- [ ] **Step 1: 为 RealPvkBoRuntime.__init__ 添加 acq_class 参数**

In `pvk_llm_bo_runtime.py`, modify `RealPvkBoRuntime.__init__`:

```diff
     def __init__(
         self,
         pvk_reference_root: str | Path = DEFAULT_PVK_REFERENCE_ROOT,
         data_root: str | Path = DEFAULT_REAL_DATA_ROOT,
         env_path: str | Path = DEFAULT_ENV_PATH,
         pvk_bo_class: type | None = None,
+        acq_class: type | None = None,
     ) -> None:
         self.pvk_reference_root = Path(pvk_reference_root)
         self.data_root = _normalize_data_root(Path(data_root))
         self.env_path = Path(env_path)
         self._pvk_bo_class = pvk_bo_class
+        self._acq_class = acq_class
         self._sessions: dict[str, dict[str, Any]] = {}
```

- [ ] **Step 2: 在 create_session 中使用注入的 acq_class**

After PVKBO instantiation in `create_session`, add ACQ replacement logic:

```diff
         pvkbo = pvk_bo_class(
             task_context=task_context,
             sm_mode=request.sm_mode,
             ...
         )
+
+        if self._acq_class is not None:
+            pvkbo.acq_func = self._acq_class(
+                task_context=task_context,
+                n_candidates=request.n_candidates,
+                n_templates=request.n_templates,
+                lower_is_better=task_context["lower_is_better"],
+                chat_engine=chat_engine,
+            )

         init_cost, init_time = pvkbo._initialize()
```

- [ ] **Step 3: 验证现有测试仍然通过**

Run: `cd backend && python -m pytest tests/test_pvk_llm_bo_runtime.py -v`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/pvk_llm_bo_runtime.py
git commit -m "feat: add acq_class injection support to RealPvkBoRuntime"
```

---

### Task 7: 修改 api.py — 添加 Benchmark API endpoint

**Files:**
- Modify: `backend/api.py`

- [ ] **Step 1: 在 api.py 顶部添加 benchmark 导入**

```diff
+ from benchmark.runner import BenchmarkRunner
```

- [ ] **Step 2: 添加 Benchmark 请求体模型**

在 `api.py` 中，在 `class AgentChatBody` 之后添加：

```python
class CreateBenchmarkBody(BaseModel):
    task_id: str = Field(default="band_alignment", pattern="^(band_alignment|defects_doping)$")
    n_initial: int = Field(default=5, ge=1, le=50)
    n_trials: int = Field(default=20, ge=1, le=200)
    seed: int = Field(default=42, ge=0)
    seeds: list[int] | None = None
    sm_mode: str = Field(default="discriminative", pattern="^(discriminative|generative)$")
    n_candidates: int = Field(default=10, ge=1, le=50)
    n_templates: int = Field(default=2, ge=1, le=10)
    n_gens: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.1, ge=-1.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=100)
    output_dir: str = Field(default="results")
```

- [ ] **Step 3: 添加 POST /api/v1/benchmark endpoint**

```python
@app.post(
    "/api/v1/benchmark",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_benchmark_run(body: CreateBenchmarkBody) -> dict[str, Any]:
    """Submit a benchmark run. Returns immediately with run metadata;
    execution happens synchronously (for now)."""
    from benchmark.runner import run_multi_seed

    emit_backend_log(
        "benchmark.request",
        f"收到 benchmark 请求: {body.task_id}",
        detail={"task_id": body.task_id, "seed": body.seed, "seeds": body.seeds},
    )

    try:
        common_kwargs = {
            "task_id": body.task_id,
            "n_initial": body.n_initial,
            "n_trials": body.n_trials,
            "sm_mode": body.sm_mode,
            "chat_engine": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "n_candidates": body.n_candidates,
            "n_templates": body.n_templates,
            "n_gens": body.n_gens,
            "alpha": body.alpha,
            "top_k": body.top_k,
            "output_dir": body.output_dir,
        }

        if body.seeds:
            results = run_multi_seed(seeds=body.seeds, **common_kwargs)
        else:
            runner = BenchmarkRunner(seed=body.seed, **common_kwargs)
            result = runner.run()
            runner.save_results(result)
            results = [result]

        emit_backend_log(
            "benchmark.complete",
            f"Benchmark 完成: {body.task_id}",
            detail={
                "task_id": body.task_id,
                "runs": len(results),
                "best_scores": [r["best_score"] for r in results],
            },
        )

        return success(
            {
                "task_id": body.task_id,
                "runs": len(results),
                "results": [
                    {
                        "seed": r["seed"],
                        "best_score": r["best_score"],
                        "best_generalization_score": r["best_generalization_score"],
                    }
                    for r in results
                ],
                "output_dir": str(
                    Path(body.output_dir)
                    / f"results_{body.sm_mode}"
                    / body.task_id
                ),
            }
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data file not found: {exc}",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except Exception as exc:
        emit_backend_log(
            "benchmark.error",
            f"Benchmark 失败: {exc}",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Benchmark run failed: {exc}",
        ) from None
```

- [ ] **Step 4: 验证 API 可启动**

Run: `cd backend && timeout 5 python -m uvicorn api:app --port 8000 2>&1 || true`
Expected: No import errors, server starts

- [ ] **Step 5: Commit**

```bash
git add backend/api.py
git commit -m "feat: add POST /api/v1/benchmark endpoint"
```

---

### Task 8: 编写测试 — test_gp_llm_acq.py

**Files:**
- Create: `backend/tests/test_gp_llm_acq.py`

- [ ] **Step 1: 创建 test_gp_llm_acq.py**

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gp_llm_acq import GPLLM_ACQ


def _band_alignment_task_context() -> dict:
    """Minimal task_context for band_alignment testing."""
    df = pd.DataFrame(
        [
            {"CHI_PVK": 3.8, "Eg_HTL": 2.1, "CHI_HTL": 2.4, "Eg_ETL": 3.0, "CHI_ETL": 4.1, "eta": 22.4},
            {"CHI_PVK": 3.9, "Eg_HTL": 2.0, "CHI_HTL": 2.5, "Eg_ETL": 3.1, "CHI_ETL": 4.0, "eta": 23.2},
            {"CHI_PVK": 4.0, "Eg_HTL": 2.2, "CHI_HTL": 2.6, "Eg_ETL": 3.2, "CHI_ETL": 4.2, "eta": 21.9},
            {"CHI_PVK": 4.1, "Eg_HTL": 2.3, "CHI_HTL": 2.3, "Eg_ETL": 3.3, "CHI_ETL": 4.3, "eta": 24.1},
            {"CHI_PVK": 3.7, "Eg_HTL": 1.9, "CHI_HTL": 2.7, "Eg_ETL": 2.9, "CHI_ETL": 3.9, "eta": 20.8},
            {"CHI_PVK": 4.2, "Eg_HTL": 2.4, "CHI_HTL": 2.8, "Eg_ETL": 3.4, "CHI_ETL": 4.4, "eta": 25.0},
            {"CHI_PVK": 3.6, "Eg_HTL": 1.8, "CHI_HTL": 2.2, "Eg_ETL": 2.8, "CHI_ETL": 3.8, "eta": 19.5},
        ]
    )
    feature_cols = ["CHI_PVK", "Eg_HTL", "CHI_HTL", "Eg_ETL", "CHI_ETL"]
    constraints = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        constraints[col] = ["float", "linear", [float(values.min()), float(values.max())]]

    return {
        "model": "band_alignment",
        "task": "regression",
        "metric": "neg_mean_squared_error",
        "num_classes": 1,
        "n_classes": 1,
        "lower_is_better": False,
        "num_samples": len(df),
        "tot_feats": len(feature_cols),
        "cat_feats": 0,
        "num_feats": len(feature_cols),
        "feature_cols": feature_cols,
        "target_col": "eta",
        "hyperparameter_constraints": constraints,
        "df": df,
    }


def _observed_data(task_context: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """First 2 rows as observed."""
    df = task_context["df"]
    configs = df.iloc[:2][task_context["feature_cols"]].reset_index(drop=True)
    fvals = pd.DataFrame({"score": df.iloc[:2]["eta"].values})
    return configs, fvals


class TestGPLLM_ACQ:
    def test_init_stores_attributes(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=10,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        assert acq.n_candidates == 10
        assert acq.feature_cols == ctx["feature_cols"]
        assert acq.target_col == "eta"
        assert acq.top_k == 20
        assert acq.alpha == 0.1

    def test_find_unobserved_filters_observed_points(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=10,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        configs, fvals = _observed_data(ctx)

        unobserved = acq._find_unobserved(configs)
        # 7 total rows, 2 observed → 5 unobserved
        assert len(unobserved) == 5

    def test_get_candidate_points_returns_dataframe(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=3,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
            top_k=5,
        )
        configs, fvals = _observed_data(ctx)

        # Mock LLM to return a valid selection
        with patch.object(acq, "_call_llm", return_value="Analysis: test\nSelected Formulations: 1, 2, 3"):
            candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)

        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) <= 3
        for col in ctx["feature_cols"]:
            assert col in candidates.columns
        assert isinstance(cost, float)
        assert isinstance(elapsed, float)

    def test_fallback_when_llm_fails(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=3,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
            top_k=5,
        )
        configs, fvals = _observed_data(ctx)

        # Mock LLM to raise
        with patch.object(acq, "_call_llm", side_effect=RuntimeError("API error")):
            candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)

        # Should fall back to GP UCB
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 3

    def test_all_points_observed_returns_random_candidates(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=2,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        # All rows as observed
        df = ctx["df"]
        configs = df[ctx["feature_cols"]].copy()
        fvals = pd.DataFrame({"score": df["eta"].values})

        candidates, cost, elapsed = acq.get_candidate_points(configs, fvals)
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 2

    def test_build_batch_evaluation_prompt_contains_features(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        configs, fvals = _observed_data(ctx)
        unobserved = acq._find_unobserved(configs)
        X_train, y_train = acq._build_training_data(configs, fvals)
        predictions = acq._gp_ucb_predict(X_train, y_train, unobserved, 0.1)
        top = predictions[:3]
        observed_data = acq._build_observed_data(configs, fvals)

        prompt = acq._build_batch_evaluation_prompt(top, observed_data)

        assert "CHI_PVK" in prompt
        assert "Eg_HTL" in prompt
        assert "eta" in prompt
        assert "Selected Formulations" in prompt
        assert "GP Prediction" in prompt

    def test_parse_llm_response_standard_format(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        response = "Analysis: These look promising.\nSelected Formulations: 1, 3, 5, 7, 9"
        indices = acq._parse_llm_response(response, 20)
        assert indices == [0, 2, 4, 6, 8]

    def test_parse_llm_response_fallback_to_numbers(self):
        ctx = _band_alignment_task_context()
        acq = GPLLM_ACQ(
            task_context=ctx,
            n_candidates=5,
            n_templates=2,
            lower_is_better=False,
            chat_engine="deepseek-v4-flash",
        )
        response = "I recommend formulations 2, 4, and 6."
        indices = acq._parse_llm_response(response, 20)
        assert 1 in indices  # formulation 2 → index 1
        assert 3 in indices  # formulation 4 → index 3
        assert 5 in indices  # formulation 6 → index 5
```

- [ ] **Step 2: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/test_gp_llm_acq.py -v`
Expected: All tests PASS (8 tests)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_gp_llm_acq.py
git commit -m "test: add GPLLM_ACQ unit tests"
```

---

### Task 9: 编写测试 — test_benchmark_data_loader.py

**Files:**
- Create: `backend/tests/test_benchmark_data_loader.py`

- [ ] **Step 1: 创建 test_benchmark_data_loader.py**

```python
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.data_loader import (
    BAND_ALIGNMENT_FEATURES,
    DEFECTS_DOPING_FEATURES,
    DATA_LOADERS,
    TARGET_COL,
    build_task_context,
)


def _create_test_excel(feature_cols: list[str]) -> Path:
    """Create a minimal test Excel file with random data."""
    rng = np.random.RandomState(42)
    n_rows = 30
    data = {}
    for col in feature_cols:
        data[col] = rng.uniform(1.0, 5.0, n_rows)
    data[TARGET_COL] = rng.uniform(15.0, 25.0, n_rows)
    df = pd.DataFrame(data)

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    df.to_excel(tmp.name, index=False)
    return Path(tmp.name)


class TestBandAlignmentDataLoader:
    def test_load_returns_expected_structure(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            assert "train_x" in data
            assert "train_y" in data
            assert "test_x" in data
            assert "test_y" in data
            assert data["feature_cols"] == BAND_ALIGNMENT_FEATURES
            assert data["target_col"] == TARGET_COL
            assert isinstance(data["df"], pd.DataFrame)
            assert data["train_x"].shape[0] == 10
            assert data["test_x"].shape[0] == 20
            assert data["train_x"].shape[1] == len(BAND_ALIGNMENT_FEATURES)
        finally:
            path.unlink(missing_ok=True)

    def test_different_seeds_produce_different_splits(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data1 = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            data2 = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=99)
            # Train sets should differ
            assert not np.array_equal(data1["train_x"], data2["train_x"])
        finally:
            path.unlink(missing_ok=True)

    def test_build_task_context_for_band_alignment(self):
        path = _create_test_excel(BAND_ALIGNMENT_FEATURES)
        try:
            data = DATA_LOADERS["band_alignment"](file_path=path, n_train=10, seed=42)
            ctx = build_task_context("band_alignment", data)
            assert ctx["model"] == "band_alignment"
            assert ctx["lower_is_better"] is False
            assert ctx["feature_cols"] == BAND_ALIGNMENT_FEATURES
            assert ctx["target_col"] == "eta"
            assert "hyperparameter_constraints" in ctx
            for col in BAND_ALIGNMENT_FEATURES:
                assert col in ctx["hyperparameter_constraints"]
                constraint = ctx["hyperparameter_constraints"][col]
                assert constraint[0] == "float"
                assert constraint[1] == "linear"
                assert len(constraint[2]) == 2
        finally:
            path.unlink(missing_ok=True)


class TestDefectsDopingDataLoader:
    def test_load_returns_expected_structure(self):
        path = _create_test_excel(DEFECTS_DOPING_FEATURES)
        try:
            data = DATA_LOADERS["defects_doping"](file_path=path, n_train=10, seed=42)
            assert data["train_x"].shape[0] == 10
            assert data["test_x"].shape[0] == 20
            assert data["train_x"].shape[1] == len(DEFECTS_DOPING_FEATURES)
            assert data["feature_cols"] == DEFECTS_DOPING_FEATURES
        finally:
            path.unlink(missing_ok=True)
```

- [ ] **Step 2: 运行测试**

Run: `cd backend && python -m pytest tests/test_benchmark_data_loader.py -v`
Expected: All tests PASS (4 tests)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_benchmark_data_loader.py
git commit -m "test: add benchmark data loader tests"
```

---

### Task 10: 编写测试 — test_benchmark_runner.py

**Files:**
- Create: `backend/tests/test_benchmark_runner.py`

- [ ] **Step 1: 创建 test_benchmark_runner.py**

```python
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from benchmark.data_loader import BAND_ALIGNMENT_FEATURES, TARGET_COL
from benchmark.runner import BenchmarkRunner


def _create_test_excel() -> Path:
    """Create a minimal test Excel file with 30 rows of band_alignment data."""
    rng = np.random.RandomState(42)
    n_rows = 30
    data = {}
    for col in BAND_ALIGNMENT_FEATURES:
        data[col] = rng.uniform(1.0, 5.0, n_rows)
    data[TARGET_COL] = rng.uniform(15.0, 25.0, n_rows)
    df = pd.DataFrame(data)

    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    df.to_excel(tmp.name, index=False)
    return Path(tmp.name)


class MockPVKBO:
    """Minimal mock of PVKBO that records calls and returns fake data."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observed_configs = pd.DataFrame()
        self.observed_fvals = pd.DataFrame()
        self.llm_query_cost: list[float] = []
        self.llm_query_time: list[float] = []
        self.acq_func = None

    def optimize(self, test_metric="generalization_score"):
        # Return fake optimization trace
        feature_cols = self.kwargs["task_context"]["feature_cols"]
        configs = pd.DataFrame(
            [{col: 4.0 for col in feature_cols}] * 5
        )
        fvals = pd.DataFrame(
            {
                "score": [22.0, 22.5, 23.0, 23.5, 24.0],
                "generalization_score": [21.0, 21.5, 22.0, 22.5, 23.0],
            }
        )
        self.llm_query_cost = [0.01] * 5
        self.llm_query_time = [0.5] * 5
        self.observed_configs = configs
        self.observed_fvals = fvals
        return configs, fvals


class TestBenchmarkRunner:
    def test_init_validates_task_id(self):
        with pytest.raises(ValueError, match="Unknown task_id"):
            BenchmarkRunner(task_id="invalid_task")

    def test_init_validates_sm_mode(self):
        with pytest.raises(ValueError, match="Unknown sm_mode"):
            BenchmarkRunner(task_id="band_alignment", sm_mode="invalid")

    def test_run_produces_result_dict(self, monkeypatch):
        path = _create_test_excel()
        tmp_dir = Path(tempfile.mkdtemp())

        try:
            # Mock the PVKBO import
            monkeypatch.setattr(
                "benchmark.runner._resolve_pvk_root",
                lambda: Path("/fake/pvk"),
            )

            with patch(
                "benchmark.runner.GPLLM_ACQ",
                autospec=True,
            ) as mock_acq_class:
                mock_acq = MagicMock()
                mock_acq.get_candidate_points.return_value = (
                    pd.DataFrame([{col: 4.0 for col in BAND_ALIGNMENT_FEATURES}]),
                    0.0,
                    0.1,
                )
                mock_acq_class.return_value = mock_acq

                # Inject mock PVKBO module
                import sys
                sys.modules["pvk_bo"] = MagicMock()
                sys.modules["pvk_bo.pvk_bo"] = MagicMock()
                sys.modules["pvk_bo.pvk_bo"].PVKBO = MockPVKBO

                runner = BenchmarkRunner(
                    task_id="band_alignment",
                    n_initial=5,
                    n_trials=3,
                    seed=42,
                    output_dir=tmp_dir,
                    data_path=path,
                )

                result = runner.run()

                assert result["task_id"] == "band_alignment"
                assert result["seed"] == 42
                assert "best_score" in result
                assert "best_generalization_score" in result
                assert isinstance(result["search_history"], pd.DataFrame)
                assert isinstance(result["best_config"], dict)
        finally:
            path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            # Clean up mock modules
            for mod in ["pvk_bo.pvk_bo", "pvk_bo"]:
                sys.modules.pop(mod, None)

    def test_save_results_creates_output_files(self):
        path = _create_test_excel()
        tmp_dir = Path(tempfile.mkdtemp())

        try:
            runner = BenchmarkRunner(
                task_id="band_alignment",
                n_initial=5,
                n_trials=3,
                seed=42,
                output_dir=tmp_dir,
                data_path=path,
            )

            result = {
                "task_id": "band_alignment",
                "seed": 42,
                "search_history": pd.DataFrame({"score": [22.0, 23.0]}),
                "best_config": {"CHI_PVK": 4.0},
                "best_score": 23.0,
                "best_generalization_score": 22.0,
                "llm_query_cost": [0.01, 0.01],
                "llm_query_time": [0.5, 0.5],
            }

            runner.save_results(result)

            save_dir = tmp_dir / "results_discriminative" / "band_alignment"
            assert save_dir.exists()
            assert (save_dir / "42.csv").exists()
            assert (save_dir / "42_search_info.json").exists()
            assert (save_dir / "42_summary.json").exists()

            # Verify summary content
            with open(save_dir / "42_summary.json") as f:
                summary = json.load(f)
            assert summary["best_score"] == 23.0
            assert summary["n_trials"] == 3
            assert summary["convergence_curve"] == [22.0, 23.0]
        finally:
            path.unlink(missing_ok=True)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 2: 运行测试**

Run: `cd backend && python -m pytest tests/test_benchmark_runner.py -v`
Expected: All tests PASS (3 tests)

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_benchmark_runner.py
git commit -m "test: add benchmark runner tests"
```

---

### Task 11: 运行全部测试验证

**Files:**
- No files created or modified.

- [ ] **Step 1: 运行全部后端测试**

Run: `cd backend && python -m pytest -v`
Expected: All tests PASS (including all existing + new tests)

- [ ] **Step 2: 确认测试数量**

Run: `cd backend && python -m pytest --co -q | tail -1`
Expected: Shows total test count including new tests

---

## Self-Review Checklist

1. **Spec coverage**: All spec requirements mapped to tasks:
   - GP+LLM ACQ → Task 2 ✓
   - Data loader → Task 3 ✓
   - Benchmark runner → Task 4 ✓
   - CLI → Task 5 ✓
   - ACQ injection → Task 6 ✓
   - API endpoint → Task 7 ✓
   - Tests → Tasks 8-11 ✓

2. **Placeholder scan**: No TBD/TODO/incomplete sections ✓

3. **Type consistency**: GPLLM_ACQ.get_candidate_points signature matches across gp_llm_acq.py definition, runner.py usage, and test mocks ✓
