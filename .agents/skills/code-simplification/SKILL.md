---
name: code-simplification
description: Simplifies code for clarity. Use when refactoring code for clarity without changing behavior. Use when code works but is harder to read, maintain, or extend than it should be. Use when reviewing code that has accumulated unnecessary complexity.
---

# Code Simplification Skill (code-simplification)

Use this skill when refactoring complex modules, eliminating duplicate calculation helpers, cleaning up nested states, or simplifying verbose control flows without altering behavior.

---

## 1. Trigger Scopes & Scenarios
- Refactoring backend endpoint controllers or optimization algorithms in `backend/`.
- Cleaning up React component hooks, state variables, or JSX layouts in `frontend/`.
- Eliminating redundant logic, deeply nested conditions, or long functions (>50 lines).

---

## 2. Key Files to Read First
- `/home/dministrator/project/BOagent/backend/pvk_llm_compat.py`: Core legacy compatibility patches. Do NOT simplify or touch this file!
- `/home/dministrator/project/BOagent/backend/optimization/optimizer.py`: High-context calculations. Do not overcomplicate or introduce global randomness.

---

## 3. Recommended Workflow & Architectural Rules

### 3.1 Keep Code Simplicity (Karpathy Guidelines)
*   **Minimal Code**: Write the minimum amount of code that solves the specific problem. No speculative features or abstractions for single-use code.
*   **Surgical Changes**: Touch only what you must. Avoid refactoring unrelated code, comments, or formatting adjacent to your changes.
*   **Preserve Behavior**: Refactoring must never alter inputs, outputs, side effects, or error cases. Maintain all functional constraints.

### 3.2 Backend Simplification Heuristics
*   **Early Returns & Guard Clauses**: Replace deep nesting of conditions with early returns.
*   **Avoid Nested Ternaries**: Replace mental-stack-heavy nested ternaries with clean lookup maps or standard `if-else` blocks.
*   **Task/Worker Safety**: Delegate heavy calculations in stream endpoints to thread executors instead of blocking the FastAPI event loop.

### 3.3 Frontend Simplification Heuristics
*   **State Isolation**: Keep benchmark configuration/state separate from operational states. Toggling pages via `display: none` is preferred to merge-state syncing.
*   **Declarative Map Rendering**: Replace complex for-loops with standard array method expressions (`.map`, `.filter`).

---

## 4. Key Verification Commands
```bash
# Verify backend unit/integration tests
cd /home/dministrator/project/BOagent/backend
python -m pytest

# Verify frontend compilation
cd /home/dministrator/project/BOagent/frontend
npm run build
```

---

## 5. Existing Capabilities & Reuse Guide
*   **CSS Theme Variables**: Consume theme variables (`var(--color-signal-500)`) in CSS and inline React style attributes. Avoid hardcoded hex colors.
*   **Physics Formulation Builder**: Re-use `build_prompt` in `knowledge.py` for compiling physics rules instead of writing inline prompts.

---

## 6. Prohibited Actions (Red Lines)
> [!WARNING]
> **No Legacy Patch Deletions**: Do not delete or refactor any code inside `backend/pvk_llm_compat.py`.
> **No Global Randomness**: Do not use global `np.random` state; always instantiate local `np.random.RandomState` generators.
> **No Heavy State Managers**: Do not add Redux, MobX, or Zustand to the React frontend.
