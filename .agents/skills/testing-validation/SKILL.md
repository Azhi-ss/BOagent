---
name: testing-validation
description: Rules and protocols for running backend unit/integration tests (pytest) and frontend E2E tests (Playwright) in BOagent. Use this skill whenever the user asks to run test suites, check code coverage, add new test cases, mock API calls, or verify modifications to the codebase.
---

# Testing & Quality Verification Skill (testing-validation)

Use this skill when running tests, writing assertions, or checking the correctness of changes made to backend or frontend components.

---

## 1. Trigger Scopes & Scenarios
- Verifying code logic changes before submitting PRs.
- Adding tests for new endpoints, optimization options, or components.
- Troubleshooting test failures, timeouts, or flakey specs.

---

## 2. Key Directories & Configurations
- **Backend Tests**: `/home/dministrator/project/BOagent/backend/tests/` (pytest)
- **Frontend E2E Tests**: `/home/dministrator/project/BOagent/frontend/tests/e2e/` (Playwright)
- **Playwright Config**: `/home/dministrator/project/BOagent/frontend/playwright.config.ts`

---

## 3. Recommended Workflow & Test Suite Execution

### 3.1 Backend Test Framework (Pytest)
To run unit and integration tests, activate the backend virtual environment:
```bash
cd /home/dministrator/project/BOagent/backend
source .venv/bin/activate
# Run all backend tests
pytest tests/
# Run specific test file
pytest tests/test_api.py -v
```

### 3.2 Mocking Structures & Guidelines
- **Dummy API Keys**: Set a dummy key (e.g. `DEEPSEEK_API_KEY=sk-test`) during test execution. The central API client will skip actual remote calls, triggering the optimizer to run purely locally.
- **Mocking PVK-LLM**: Tests like `test_benchmark_runner.py` mock the `PVKBO` object via `sys.modules` injection:
  ```python
  from unittest.mock import MagicMock
  import sys
  # Mock legacy PVK-LLM imports to prevent runtime import issues
  mock_pvk = MagicMock()
  sys.modules['pvk_llm'] = mock_pvk
  ```
- **SSE Stream Mocking**: Mock the `BOStepEngine` or iterate directly on `ComparisonRunner.events()` to isolate network delays when testing parallel multi-seed event queues in `test_parallel_benchmark.py`.

### 3.3 Frontend E2E Tests (Playwright)
Make sure the FastAPI server is running on `http://localhost:8000` before triggering E2E:
```bash
cd /home/dministrator/project/BOagent/frontend
# Run headless E2E tests
npm run test:e2e
# Run E2E tests selectively on Chromium (ignores Webkit library dependency errors)
npm run test:e2e:chromium
# Run with Playwright interactive UI
npx playwright test --ui
```

### 3.4 Handling UI Selectors and E2E Timeouts
- **Ambiguous Locators**: Benchmark mode and Operational mode pages render concurrently (using display toggles). Always use `:visible` filters or unique `data-testid` properties (e.g., `page.locator("button:has-text('Run'):visible")`) to avoid hitting ambiguous locator errors.
- **Handling Latency (Timeouts)**: LLM analysis queries can take 20s to 40s. Playwright specs override standard 30s timeouts: use `test.setTimeout(180000)` for typical specs or `test.setTimeout(240000)` for benchmark specs to accommodate API latency.

---

## 4. Pre-Merge Verification Checklist
Before finishing a feature task, run the following verification steps:
1.  **Backend Pytest**: `pytest tests/` must return all passes.
2.  **SSE Streaming Integrity**: Run `npx playwright test tests/e2e/benchmark.spec.ts` (or with `--project=chromium`) to verify data events bind correctly to Recharts.
3.  **No Unhandled Promise Rejections**: Inspect browser console output during test runs to confirm SSE connections abort cleanly upon page redirection/close.
4.  **Security Boundaries**: Verify that `output_dir` changes do not allow path traversals (checked via `test_api.py`).
