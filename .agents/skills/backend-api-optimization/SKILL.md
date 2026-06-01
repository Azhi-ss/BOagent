---
name: backend-api-optimization
description: Guidelines and constraints for modifying the FastAPI api.py server, the BayesianOptimizer core logic, the physics-informed KnowledgeEngine prompting, or the VectorMemory embeddings/RAG layers. Use this skill whenever the user asks to modify python backend code under backend/optimization/ or backend/api.py.
---

# Backend API & Optimization Skill (backend-api-optimization)

Use this skill when implementing new API endpoints, adjusting GP/Acquisition calculations, updating semiconductor physics rules, or editing vector database/insight RAG logic.

---

## 1. Trigger Scopes & Scenarios
- Modifying FastAPI routes or middleware in `/home/dministrator/project/BOagent/backend/api.py`.
- Adjusting Gaussian Process, acquisition functions, or search spaces in `/home/dministrator/project/BOagent/backend/optimization/optimizer.py` and `space.py`.
- Editing physics heuristics, prompts, or parsers in `/home/dministrator/project/BOagent/backend/optimization/knowledge.py`.
- Refactoring embedding models, vector storage, or recency fallback logic in `/home/dministrator/project/BOagent/backend/optimization/memory.py`.

---

## 2. Key Files to Read First
- `/home/dministrator/project/BOagent/backend/api.py`: FastAPI endpoints and SSE stream setups.
- `/home/dministrator/project/BOagent/backend/optimization/optimizer.py`: GP fitting and scoring loops.
- `/home/dministrator/project/BOagent/backend/optimization/knowledge.py`: Physics prompt building and text parsing.
- `/home/dministrator/project/BOagent/backend/optimization/memory.py`: VectorMemory embedding retrieval.
- `/home/dministrator/project/BOagent/backend/pvk_llm_compat.py`: Legacy patches bridging pandas/langchain/OpenAI. Do NOT refactor or delete these patches!

---

## 3. Recommended Workflow & Architectural Rules

### 3.1 FastAPI Endpoints & Contract Guidelines
*   **Rest Endpoints**: Use Pydantic schemas (v2) to validate incoming request bodies (e.g., `CreateBenchmarkBody`, `CompareBenchmarkBody`, `OperationalSuggestBody`).
*   **SSE Streams**:
    *   **GET Stream (`/api/v1/logs/stream`)**: Formatted as `data: {json}\n\n` with UTF-8 encoding.
    *   **POST Stream (`/api/v1/benchmark/compare/stream`)**: Streams real-time aggregate events using `StreamingResponse` wrapping an `asyncio.Queue` consumer. SSE event types: `meta`, `seed_start`, `step_start`, `aggregate`, `done`. Heavy calculations must execute in a thread/executor (`loop.run_in_executor(None, produce)`) to prevent blocking the event loop.
*   **Input Validation**: Prevent path traversal by sanitizing file output paths:
    ```python
    output_path = Path(body.output_dir)
    if output_path.is_absolute() or ".." in str(output_path):
        raise HTTPException(status_code=400, detail="output_dir must be a relative path without '..' traversal.")
    ```

### 3.2 Bayesian Optimization & Gaussian Process
*   **Kernel Configuration**: `scikit-learn` `GaussianProcessRegressor` utilizes a `Matern` kernel with standard configuration:
    ```python
    kernel = C(1.0, (1e-3, 1e3)) * Matern([1.0] * len(self.space.feature_cols), (1e-2, 1e2), nu=2.5)
    ```
*   **Acquisition Strategies**: Supports `ei`, `ucb`, and `pi`. Scaling parameter `kappa` (default `2.576`) controls exploration-exploitation balance.
*   **Hybrid LLM-GP Scoring**: Top-K candidates from GP are queried via a "Yes/No" viability prompt. The log-probability of "Yes" is extracted and combined: `Hybrid = GP_Score + (γ × std(GP_Scores)) × log_prob(Yes)` where `γ` defaults to `0.1`.
*   **RNG Isolation**: Always use locally seeded instances of `np.random.RandomState` (e.g., `rng = np.random.RandomState()`) in data loaders and optimization loops. Avoid using the global `np.random` to prevent random state leakage across parallel benchmark threads.

### 3.3 Physics-Informed KnowledgeEngine
*   **Physics Offsets Constraints**:
    *   **Conduction Band Offset (CBO)**: $CBO = \chi_{PVK} - \chi_{ETL}$. Ideal range: $[-0.1, 0.3]$ eV. Negative CBO (cliff) causes $V_{oc}$ loss; large positive CBO (spike) blocks electron extraction.
    *   **Valence Band Offset (VBO)**: $VBO = (\chi_{HTL} + E_{g,HTL}) - \chi_{PVK}$. Ideal range: $[1.7, 2.0]$ eV. VBO below 1.7 eV causes $V_{oc}$ loss; above 2.0 eV blocks hole extraction.
    *   **Electron Blocking**: HTL LUMO ($\chi_{HTL}$) must be higher than PVK LUMO ($\chi_{PVK}$) by at least $0.5$ eV.
    *   **Trap Density ($Nt$)**: Recombination trap density at interfaces ($N_{t,\text{PVK/ETL}}$ or $N_{t,\text{HTL/PVK}}$) must be minimized. Logarithmic reduction in $Nt$ yields linear gains in open-circuit voltage ($V_{oc}$).
    *   **Doping Concentration ($Na, Nd$)**: Capped below $10^{19} \text{ cm}^{-3}$ to prevent tunneling-assisted recombination.
*   **Parsing Safety**: The parser searches for a `Selected Formulations:` header using regular expressions. Ensure prompts instruct the LLM to follow this exact output structure.

### 3.4 Dynamic Scientific Memory (DSM)
*   **Insight Persistence**: Summarize insights using LLM only when a new best score is found **AND** at least 10 observations exist. Store them in `VectorMemory` at `insights.jsonl`.
*   **Embeddings**: Uses Doubao Ark embedding endpoint model `doubao-embedding-large-text-250515` (or as configured in `DOUBAO_EMBEDDING_MODEL`).
*   **Fallback Logic**: If `ARK_API_KEY` or base URLs are missing/unreachable, the vector memory MUST fallback to returning the top `iteration` items (recency-based retrieval) instead of throwing an error.

---

## 4. Key Verification Commands
Activate the environment and run:
```bash
cd /home/dministrator/project/BOagent/backend
source .venv/bin/activate
pytest tests/test_api.py
pytest tests/test_unification.py
```

---

## 5. Prohibited Actions (Red Lines)
> [!WARNING]
> **No Direct LLM Scoring**: Never let the LLM score or select candidate formulations from the raw search space. The search space must always be pre-filtered using the Gaussian Process surrogate model first to produce a Top-K pool (typically top 20). The LLM's role is strictly to refine these top candidates using physical reasoning. Even when using a generated heuristic function, the primary workflow must prioritize the GP surrogate model as the first-stage filter to maintain convergence guarantees.
> Failure to follow this rule will blow the token budget and ruin convergence guarantees.
