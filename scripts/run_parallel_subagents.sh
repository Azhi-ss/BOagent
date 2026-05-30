#!/bin/bash
set -e

mkdir -p backend/tmp_report

echo "Starting subagent 1 (Backend, Physics, RAG Memory)..."
./scripts/smart_gemini.sh "Analyze the backend folder of BOagent repository. Specially check backend/api.py, backend/pvk_llm_compat.py, and all files under backend/optimization/. Gather facts about:
1. Directory structure of the backend and core data flow.
2. Backend API routes and endpoints in api.py, including parameter structures/validation, error handling, SSE stream processing (such as comparison streams and live logs).
3. The semiconductor physics domain rules, including conduction band offset (CBO), valence band offset (VBO), electron blocking, recombination trap density, active doping concentration constraints.
4. RAG logic: VectorMemory, Doubao Ark embedding endpoint model, fallback mechanism when ARK keys are missing.
Check actual files, look up definitions and variables. Do not make up facts. Format the findings in Markdown and write them to backend/tmp_report/subagent_backend.md. When done, output a summary and finish." > backend/tmp_report/subagent_backend.log 2>&1 &
PID1=$!

echo "Starting subagent 2 (DevOps, Build & Testing)..."
./scripts/smart_gemini.sh "Analyze the BOagent repository for local development, build configuration, and testing. Specifically check:
1. Dependency management in backend/requirements.txt, frontend/package.json.
2. Startup commands and ports for backend (FastAPI) and frontend (Vite/React).
3. Testing framework and structure: backend tests in backend/tests/ (pytest configuration, mock structures such as MagicMock, sys.modules injects, and SSE stream mocks), frontend E2E tests in frontend/tests/e2e/ (Playwright configuration, selector handling, timeout adjustments, etc.).
4. Troubleshooting common failures (e.g. ModuleNotFoundError, CORS issues, missing dataset files, API route validation).
Verify facts by reading actual files. Write the findings in Markdown to backend/tmp_report/subagent_testing_devops.md. When done, output a summary and finish." > backend/tmp_report/subagent_testing_devops.log 2>&1 &
PID2=$!

echo "Starting subagent 3 (Frontend UI & Interaction)..."
./scripts/smart_gemini.sh "Analyze the frontend of BOagent repository. Specifically check:
1. Directory structure of the frontend (under frontend/src/).
2. Root layout, mode switcher, and state management in App.tsx, BenchMode.tsx, and OperationalMode.tsx.
3. Custom components in components/ (e.g. ConvergenceChart.tsx, LandscapeCanvas.tsx, MetricReadout.tsx, AcquisitionConfig.tsx).
4. Recharts integration rules, including responsiveness, and disabling animation inside ConvergenceChart.
5. Tailwind CSS 4.x configuration and rules (Vite integration, CSS-first config, @theme rules and variables, custom colors, glassmorphism, and avoiding hex colors).
Verify facts by reading actual files. Write the findings in Markdown to backend/tmp_report/subagent_frontend.md. When done, output a summary and finish." > backend/tmp_report/subagent_frontend.log 2>&1 &
PID3=$!

echo "Starting subagent 4 (Code Simplification & Refactoring rules)..."
./scripts/smart_gemini.sh "Analyze the BOagent repository for code simplicity, refactoring rules, and software design quality. Specifically check:
1. Identifying code components that tend to accumulate complexity or over-engineering (e.g., nesting in API responses, excessive abstractions, or unnecessary helper methods).
2. Rules for minimal changes: keeping code clean, matching existing style, and surgical edits.
3. Code patterns to avoid (e.g., adding unused packages, redundant state managers like Redux/MobX, or duplicate calculation helpers).
Verify facts by reading actual files. Write the findings in Markdown to backend/tmp_report/subagent_simplification.md. When done, output a summary and finish." > backend/tmp_report/subagent_simplification.log 2>&1 &
PID4=$!

echo "Subagents running with PIDs: $PID1, $PID2, $PID3, $PID4"
echo "Waiting for all subagents to complete..."
wait $PID1 $PID2 $PID3 $PID4
echo "All subagents have completed execution."
