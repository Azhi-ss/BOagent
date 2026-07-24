# BOagent AI Developer Guide (CLAUDE.md)

Welcome to the **BOagent** codebase! This guide serves as your primary context document to help you understand the architecture, domain concepts, local workflows, and development guidelines. Always consult this document before starting any implementation, refactoring, or testing task.

---

## 1. Project Overview & Tech Stack

BOagent is an LLM-driven Bayesian Optimization (BO) workflow orchestrator and scientific dashboard designed for perovskite solar cell material formulation optimization. It leverages high-context domain reasoning to augment traditional Gaussian Process (GP) statistical modeling.

### Core Tech Stack
*   **Backend**: FastAPI, Python 3.10+, Uvicorn, Scikit-learn (Gaussian Process Regressor), NumPy, Pandas, SciPy, pytest.
*   **Frontend**: React 19, Vite 6, TypeScript 6, Recharts 3.8, Tailwind CSS 4.x.
*   **AI/LLM**: DeepSeek API (`deepseek-v4-flash` / `deepseek-v4-pro`), Doubao Embedding API (Volcengine/Ark) utilizing the default `doubao-embedding-vision-250615` model (configured via `DOUBAO_EMBEDDING_MODEL`) for semantic memory retrieval.
*   **Dataset Integration**: Uses unified dataset schema located in the local `datasets/` directory (e.g., `datasets/perovskite`, `datasets/battery`). No `.env` configuration is required.

---

## 2. Codebase Structure & Responsibilities

```
BOagent/
├── apps/
│   ├── api/                       # FastAPI Backend Service
│   │   ├── api.py                 # FastAPI server, SSE stream controller, rest endpoints
│   │   ├── conftest.py            # pytest path configuration
│   │   ├── pyproject.toml         # Python dependencies (incl. bo-core workspace link)
│   │   └── tests/                 # API automated tests
│   └── web/                       # React Frontend Application
│       ├── src/
│       │   ├── App.tsx            # Root view container & mode manager
│       │   ├── BenchMode.tsx      # Benchmark execution & dual-curve visualization panel
│       │   ├── OperationalMode.tsx # Human-in-the-loop experimental suggestion interface
│       │   ├── types.ts           # Global TypeScript declarations
│       │   ├── components/        # UI elements (ConvergenceChart, LandscapeCanvas, etc.)
│       │   └── lib/api.ts         # Typed fetch/SSE client configurations
│       ├── tests/                 # Playwright E2E functional test cases
│       ├── package.json           # Frontend npm dependencies
│       └── vite.config.ts         # Vite compile settings (with Tailwind 4.x integration)
├── packages/
│   └── bo-core/                   # Algorithm core package (pip install -e)
│       ├── bo_core/
│       │   ├── optimization/      # Core Bayesian Optimization Engine
│       │   │   ├── optimizer.py   # BayesianOptimizer orchestrator, GP training, acquisition scoring
│       │   │   ├── knowledge.py   # KnowledgeEngine, semiconductor physics rules & prompting
│       │   │   ├── memory.py      # VectorMemory, Doubao Ark embedding and numpy retrieval
│       │   │   └── space.py       # SearchSpace definitions (Continuous & Discrete)
│       │   ├── benchmark/         # Performance Evaluation Engine
│       │   │   ├── data_loader.py # Dataset loading & deterministic seed-based splitting
│       │   │   ├── runner.py      # Single seed benchmark coordinator
│       │   │   ├── comparison.py  # Multi-seed parallel benchmark comparison
│       │   │   └── bo_step.py     # Step-by-step benchmark runner bridging BO components
│       │   ├── llm_client.py      # Unified DeepSeek API client wrapper
│       │   └── pvk_llm_compat.py  # Legacy patches bridging pandas, langchain, OpenAI
│       ├── tests/                 # Core algorithm automated tests
│       ├── benchmark_agent_team.py # Multi-agent benchmark coordinator
│       ├── run_prompt_ablation.py # A/B/C prompt variant benchmark experiment
│       └── pyproject.toml         # bo-core package definition & dependencies
├── scripts/                       # Orchestration & Evaluation Scripts
│   ├── smart_gemini.sh           # Gemini CLI Smart Routing runner with fallback
│   ├── run_parallel_subagents.sh # Multi-agent concurrent analysis coordinator
│   └── grade_skills.py           # Custom AI skill card grading evaluation runner
├── evals/                         # AI Skill Evaluation Suites
│   ├── evals.json                # JSON-configured test cases and assertions
│   └── grading.json              # Output evaluation execution results
└── CLAUDE.md                      # This guide
```

---

## 3. Core Physics-Informed Domain Rules

The optimizer is not a purely statistical black-box model. It incorporates explicit device physics heuristics during the suggestion phase.

### Semiconductor Physics Formulas
*   **Conduction Band Offset (CBO)**: $CBO = \chi_{PVK} - \chi_{ETL}$. Ideal range: $[-0.1, 0.3]$ eV.
    *   *Constraint*: A negative CBO (cliff) results in high $V_{oc}$ loss due to interface recombination. A large positive CBO (spike) blocks electron extraction, hurting $J_{sc}$.
*   **Valence Band Offset (VBO)**: $VBO = (\chi_{HTL} + E_{g,HTL}) - \chi_{PVK}$. Ideal range: $[1.7, 2.0]$ eV.
    *   *Constraint*: VBO below $1.7$ eV causes $V_{oc}$ loss; above $2.0$ eV blocks hole extraction.
*   **Electron Blocking**: HTL LUMO ($\chi_{HTL}$) must be higher than PVK LUMO ($\chi_{PVK}$) by at least $0.5$ eV to effectively suppress electron flow to the anode.
*   **Recombination Trap Density ($Nt$)**: Trap density at interfaces ($N_{t,\text{PVK/ETL}}$ or $N_{t,\text{HTL/PVK}}$) must be minimized. Logarithmic reduction in $Nt$ yields linear gains in open-circuit voltage ($V_{oc}$).
*   **Doping Concentration ($Na, Nd$) & Built-in Potential ($V_{bi}$)**: Active acceptor doping in HTL ($Na_{HTL}$) and donor doping in ETL ($Nd_{ETL}$) improves charge separation built-in potential ($V_{bi}$).
    *   *Constraint*: Active doping must be strictly capped below $10^{19} \text{ cm}^{-3}$ to prevent tunneling-assisted recombination and interface leakages.
*   **Shunt Resistance (Rsh)**: Minimize counter-doping concentration in layer interfaces (e.g., $Nd_{HTL}$, $Na_{ETL}$) to avoid parasitic shunt paths that reduce Fill Factor (FF).

### Hybrid LLM-GP Selection Pipeline
The optimizer combines GP surrogate scores with LLM viability judgments via a hybrid scoring formula:
1.  GP produces a Top-K candidate pool (default K=20) ranked by acquisition score.
2.  Each candidate is queried via a "Yes/No" viability prompt; the log-probability of "Yes" is extracted.
3.  **Hybrid Score**: `GP_Score + (γ × std(GP_Scores)) × log_prob(Yes)`, where `γ` (default 0.1) controls LLM influence.
4.  The LLM can alternatively generate a Python `score_candidate(c: dict) -> float` heuristic function, executed in a sandboxed `exec()` to rank large pools (10k+ points) before GP scoring.

---

## 4. Key Developer Workflows & Commands

### 4.1 Local Development Commands

#### Backend Service (FastAPI on Port 8000)
> [!NOTE]
> The backend server must be started inside the `apps/api/` directory to resolve relative imports correctly.
```bash
cd apps/api
uv run uvicorn api:app --reload --port 8000
```

#### Frontend Web Application (Vite on Port 5173)
```bash
cd apps/web
npm install
npm run dev
```

### 4.2 Automated Testing Commands

#### API Backend Unit/Integration Tests
```bash
cd apps/api
uv run pytest
```

#### Algorithm Core Unit Tests (`bo-core`)
*Note: We enforce TDD for algorithm logic. You MUST use `--cov` to ensure your new logic is covered. Minimum coverage requirement is 80% overall, and ≥90% for `bo_core/optimization/`.*
```bash
cd packages/bo-core
uv run pytest --cov=bo_core --cov-report=term-missing
```

#### ML Data & Algorithm Inspection Tools
*   **Linting & Static Type Checks**: `uv run ruff check .` and `uv run mypy packages/bo-core/bo_core`
*   **Data Validation (`pandera`)**: Schema verification for search spaces & dataset DataFrames.
*   **Property-Based Testing (`hypothesis`)**: Boundary and matrix stability verification for GP operations.
*   **ML Health & Data Drift (`evidently`)**: Model drift detection and automated data quality checks for surrogate models.

#### Frontend Playwright E2E Tests
*Note: Make sure the backend server is running on port 8000 before executing E2E tests.*
```bash
cd apps/web
npm run test:e2e             # Headless E2E (Playwright)
npm run test:e2e:chromium    # Run E2E tests exclusively on Chromium
npx playwright test --ui     # Interactive UI Mode
```


---

## 5. Architectural Red Lines & Constraints

> [!WARNING]
> **No Direct LLM Scoring**: Never let the LLM score or select candidate formulations from the raw search space. The search space must always be pre-filtered using the Gaussian Process surrogate model to produce a Top-K pool (typically top 20). The LLM's role is strictly to refine these top candidates using physical reasoning.
> Failure to follow this rule will blow the token budget and ruin convergence guarantees.

> [!IMPORTANT]
> **Do Not Refactor `pvk_llm_compat.py`**: This file contains critical monkey-patches bridging legacy pandas, langchain, and OpenAI schemas used by the original `PVK-LLM` package. It fixes:
> - **Pandas**: Restores legacy integer-positional indexing (`Series.__getitem__` falling back to `iloc` if KeyErrors occur).
> - **Langchain**: Exposes `FewShotPromptTemplate` and `PromptTemplate` at the top level namespace if they were moved to `langchain_core`.
> - **OpenAI/DeepSeek**: Intercepts `AsyncCompletions.create` for models starting with `deepseek`, forcing `n=1`, `max_tokens >= 512`, and setting thinking disabled to prevent warnings.
> Removing or modifying these patches without exhaustive testing will break the core `PVKBO` integration.

> [!CAUTION]
> **Local RNG State**: Always use locally seeded instances of `np.random.RandomState` in data loaders and optimization loops. Avoid using the global `np.random` to prevent random state leakage across parallel benchmark threads.

> [!CAUTION]
> **Sensitive Files Protection**: Do not arbitrarily modify `**/.env*` files or `**/*_results.json` files without explicit user confirmation.

---

## 6. Existing Capabilities & Reuse Guide

Before writing new utility helpers or components, check if they already exist:
*   **API SSE client**: `apps/web/src/lib/api.ts` contains `streamComparison` which implements POST-based SSE stream chunk decoding. SSE event types: `meta`, `seed_start`, `step_start`, `aggregate`, `done`. Use this for any long-running API streaming.
*   **Tailwind 4.x Theme & Fonts**: All custom colors are defined in `apps/web/src/index.css` via the `@theme` block. Key palettes: `--color-graphite-*` (slate/dark), `--color-signal-*` (emerald/LLMBO), `--color-amber-*` (Traditional BO). Typography: `Space Grotesk` (display), `Plus Jakarta Sans` (body), `JetBrains Mono` (code). Custom animations: `pulse-ring` (live status), `value-flash` (data updates). **Avoid hardcoded hex values** — use CSS variables.
*   **Physics Formulas Prompt Builder**: `packages/bo-core/bo_core/optimization/knowledge.py` maps task feature columns to formulas and hints. If you add new parameters, add them to `build_prompt` mapping instead of copying the builder logic.
*   **Insight Persistence & RAG**: `packages/bo-core/bo_core/optimization/memory.py` handles writing insights to `insights.jsonl` and calculating embeddings using Doubao API. If the embedding client is missing keys or unreachable, it falls back to recency-based retrieval (returning the last `top_k` insights) instead of raising an error.
*   **Flat API Response Helpers**: Use `success(data)` and `error_response(msg, code)` in `api.py` for consistent response shapes. Avoid deeply nested response structures.
*   **Declarative Color Maps**: Frontend components use lookup maps (e.g., `typeColorMap` in `LandscapeCanvas.tsx`) instead of nested ternaries for conditional styling.

---

## 7. Change Verification Checklist

Before marking a task as complete, execute this checklist:
- [ ] **Backend Test Verification**: Run `uv run pytest` inside `apps/api/` to verify all unit/integration tests pass.
- [ ] **Frontend E2E Verification**: Run `npm run test:e2e:chromium` inside `apps/web/` to confirm React renders. Ensure Playwright locators use `:visible` filters (e.g., `page.locator("button:has-text('Run'):visible")`) to avoid ambiguous locator failures due to co-rendering of modes via display toggles.
- [ ] **E2E Timeout Settings**: Ensure testing specs override defaults using `test.setTimeout(180000)` or `240000` for benchmark specs to prevent timeouts during long-running LLM analysis.
- [ ] **Typescript Compiler Build**: Ensure `npm run build` runs clean without type compiler warnings.
- [ ] **Security Boundaries**: Validate that `output_dir` prevents path traversal (`..` and absolute paths validation). No API secrets should be hardcoded (always load via environment variables).
- [ ] **Scientific Memory Fallback**: Confirm that VectorMemory retrieves insights via recency-based fallback safely without throwing exceptions when `ARK_API_KEY` is missing or unreachable.
- [ ] **UI Animation Restraints**: Ensure Recharts Line elements in `ConvergenceChart.tsx` set `isAnimationActive={false}` to avoid bouncing glitches during SSE streaming.
- [ ] **Local RNG Isolation**: Verify no global `np.random` usage in optimization or benchmark paths; all randomness via locally seeded `RandomState`.

---

## 8. Authoritative Specifications & Reference Knowledge (`docs/`)

The `docs/` directory serves as the repository's single source of truth for external API contracts, hardware adaptation guidelines, real-time framework specifications, and defensive programming rules.

Before implementing features or modifying core modules, agents **MUST** consult the corresponding specification documents under `docs/`:

*   **API & Framework Specifications (`docs/hardware/`, `docs/api_specs/`, etc.)**: Real-time Context7-retrieved API contracts, constructor signatures, parameter tables, and version-locked library usage rules.
*   **Hardware Adaptation & Performance Guides**: Workload mapping across compute units (CPU, GPU, NPU), thread allocation, vectorization flags, and hardware-specific runtime configurations.
*   **Defensive Rules & Failure Modes**: Documented edge cases, numerical stability guards, precision requirements, and thread-safety invariants.

> [!NOTE]
> **Extensibility Protocol**: When fetching or adding new API specs, hardware benchmarks, or domain research, save the normalized Markdown files under `docs/<category>/` and register them in the category index (`docs/<category>/README.md`). Reference these documents in task planning artifacts before writing code.
