# BOagent Architectural Analysis Report

## 1. Analysis Range
This analysis covers the core BOagent codebase, focusing on:
- **Project Structure & Layering**: Frontend (React/TS) and Backend (FastAPI).
- **Optimization Engine**: `backend/optimization/` including Bayesian Optimization logic and LLM-informed refinement.
- **LLM Integration**: `KnowledgeEngine` and compatibility patches for legacy LLM tools.
- **Data Layer**: Excel-based data loading and splitting in `backend/benchmark/data_loader.py`.
- **API Boundary**: REST and SSE interfaces in `backend/api.py`.

## 2. Key Facts and Design Patterns

### 2.1 Layered Architecture
- **Frontend (React/TS)**: Located in `frontend/`, providing visualization for benchmarks and an "Operational Mode" for human-in-the-loop experiments.
- **API Layer (FastAPI)**: `backend/api.py` acts as the orchestrator, exposing endpoints for benchmark execution, comparison streaming (SSE), and operational suggestions.
- **Optimization Engine**: A modular core consisting of:
    - `BayesianOptimizer`: Traditional GP-based surrogate modeling.
    - `KnowledgeEngine`: LLM-based reasoning that injects materials science domain knowledge into the acquisition process.
    - `SearchSpace`: Abstracted handling of discrete and continuous parameter spaces.
- **Benchmark Layer**: `backend/benchmark/` provides tools to compare traditional BO against LLM-enhanced BO across multiple seeds.

### 2.2 Design Patterns
- **Strategy Pattern**: Acquisition functions (EI, UCB, PI) are swappable strategies within the optimizer.
- **Refinement Pattern**: The `KnowledgeEngine` acts as a refinement layer on top of GP suggestions, filtering candidates based on physical rules (CBO/VBO offsets, trap densities).
- **Dynamic Scientific Memory (DSM)**: A cumulative insight pattern where the LLM summarizes lessons from historical trials, stored in `VectorMemory` for persistent context.
- **Compatibility Layer**: `pvk_llm_compat.py` uses monkey-patching to bridge modern SDKs (pandas, langchain, OpenAI) with legacy code requirements.

## 3. Optimization Engine & LLM Interaction
The "Secret Sauce" of BOagent is the hybrid acquisition function:
1. **Surrogate Phase**: `BayesianOptimizer` uses a Matern-kernel Gaussian Process to score the entire search space.
2. **Knowledge Phase**: `KnowledgeEngine` builds a domain-specific prompt including:
    - **Physical Context**: Hardcoded semiconductor rules (e.g., VBO range [1.7, 2.0] eV).
    - **Observed History**: Recent high-performing formulations.
    - **GP Suggestions**: Top-K candidates from the surrogate phase.
3. **Refinement Phase**: The LLM selects the final candidates, balancing physical robustness with exploration.

## 4. Data Layer & Benchmarking
- **Source**: Primary data resides in `PVK-LLM/custom_perovskite_dataset` Excel workbooks (`bandAlignment.xlsx`, `defectsAndDoping.xlsx`).
- **Loader**: `data_loader.py` provides deterministic splitting (train/test) based on random seeds, ensuring reproducibility.
- **Feature Sets**:
    - **Band Alignment**: `CHI_PVK`, `Eg_HTL`, `CHI_HTL`, `Eg_ETL`, `CHI_ETL`.
    - **Defects & Doping**: Trap densities and doping concentrations (Na/Nd).

## 5. Guidelines for Future Developers

### 5.1 Extending the Optimization Logic
- **New Search Spaces**: Subclass `SearchSpace` in `space.py`. Ensure `get_unobserved` handles vectorized distance checks for discrete pools.
- **New Physical Rules**: Update `KnowledgeEngine.build_prompt` to include new semiconductor physics constraints when new feature columns are added.

### 5.2 API & Frontend Integration
- **Streaming**: For long-running benchmarks, use the SSE pattern established in `compare_benchmark_stream`.
- **Validation**: Use Pydantic models in `api.py` to enforce strict parameter boundaries for alpha, kappa, and n_trials.

## 6. Potential Pitfalls and Architectural Red Lines

### 6.1 Architectural Red Lines
- **No Direct LLM Scoring**: Never let the LLM score the entire search space; it must only refine a Top-K pool selected by the GP to maintain computational efficiency and statistical grounding.
- **Memory Integrity**: The `VectorMemory` must persist across iterations. Avoid resetting it during a single benchmark run unless explicitly requested.
- **Compatibility Patches**: Do NOT remove the monkey-patches in `pvk_llm_compat.py` without verifying legacy PVKBO code compatibility.

### 6.2 Potential Pitfalls
- **Token Truncation**: Materials science prompts are long. Ensure `max_tokens` is sufficient (min 512 for DeepSeek) and "thinking" mode is disabled where appropriate to stay within latency limits.
- **RNG Leaks**: Always use local `np.random.RandomState` within optimization loops to avoid global state issues in parallel benchmark runs.
- **Path Traversal**: Always validate `output_dir` in API calls to prevent writing results outside the designated results folder.

## 7. Recommended AGENTS.md / Skill Inputs
Add the following to `AGENTS.md` to guide future agents:
```markdown
## Optimization Engine Rules
- Prioritize `GaussianProcessRegressor` for surrogate modeling.
- LLM calls in `KnowledgeEngine` must use the `deepseek-v4-flash` or similar high-context models.
- Domain rules for Band Alignment (CBO/VBO) and Defect/Doping are hardcoded in `KnowledgeEngine.build_prompt` and must be maintained.
- Data loading always goes through `backend/benchmark/data_loader.py` to ensure train/test split consistency.
```

### Skill Input for `bo-code-expert`
- **Context**: Materials science Bayesian Optimization, FastAPI backend, React frontend.
- **Key Files**: `backend/optimization/optimizer.py`, `backend/optimization/knowledge.py`, `backend/api.py`.
- **Target Models**: DeepSeek-v4-flash, GPT-4o.
