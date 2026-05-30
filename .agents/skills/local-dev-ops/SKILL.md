---
name: local-dev-ops
description: Rules and instructions for local environment setup, dependency management, starting backend/frontend servers, compiling builds, and troubleshooting environment configurations in BOagent. Use this skill whenever the user asks to start/restart development servers, install packages, configure environment variables (.env files), or troubleshoot connection, CORS, and build issues.
---

# Local Development & Ops Skill (local-dev-ops)

Use this skill when setting up, running, building, or troubleshooting the BOagent application services.

---

## 1. Trigger Scopes & Scenarios
- Setting up the project on a new local machine or environment.
- Starting, stopping, or restarting the backend FastAPI or frontend Vite services.
- Configuring `.env` files or managing packages (`requirements.txt`, `package.json`).
- Resolving CORS errors, port clashes, and module resolution issues.

---

## 2. Key Directories & Locations
- **Project Root**: `/home/dministrator/project/BOagent`
- **Backend Root**: `/home/dministrator/project/BOagent/backend`
- **Frontend Root**: `/home/dministrator/project/BOagent/frontend`
- **Dataset Sibling Directory**: `/home/dministrator/project/PVK-LLM` (must be cloned as a sibling to `BOagent` for benchmark loaders).

---

## 3. Recommended Workflow & Service Activation

### 3.1 Backend Execution (FastAPI)
FastAPI must run from the `/home/dministrator/project/BOagent/backend` folder to ensure python resolutions for the `optimization` and `benchmark` directories match:
```bash
cd /home/dministrator/project/BOagent/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn api:app --reload --port 8000
```
By default, the backend maps to `http://localhost:8000`.

### 3.2 Frontend Execution (Vite)
Vite runs from `/home/dministrator/project/BOagent/frontend`:
```bash
cd /home/dministrator/project/BOagent/frontend
npm install
npm run dev
```
By default, the Vite dev server maps to `http://localhost:5173`.

---

## 4. Environment Variables (.env)

### Backend `.env` (`backend/.env`)
Create backend `.env` containing keys for deepseek and embedding:
```bash
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_FLASH_MODEL=deepseek-v4-flash
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_REASONING_EFFORT=high

ARK_API_KEY=ark-your-key
ARK_API_BASE=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_EMBEDDING_MODEL=doubao-embedding-large-text-250515

PVK_LLM_ROOT=../PVK-LLM
PVK_DATA_ROOT=../PVK-LLM/custom_perovskite_dataset
```

### Frontend `.env` (`frontend/.env`)
Ensure API URL maps correctly:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

---

## 5. Troubleshooting & Pitfalls

### 5.1 Python ModuleNotFoundErrors
- **Symptom**: `ModuleNotFoundError: No module named 'optimization'`
- **Fix**: The uvicorn worker must start inside `backend/` using `python -m uvicorn api:app`. Running it from the project root will fail due to `sys.path` defaults.

### 5.2 CORS Policies Blocking Frontend
- **Symptom**: Browser Console logs CORS errors on API calls.
- **Fix**: FastAPI's `CORSMiddleware` in `backend/api.py` restricts origin access. If your Vite runs on a port other than `5173` (e.g. `5174` or `5175`), update `api.py` middleware `allow_origins` to allow that port.

### 5.3 Missing Dataset Files (503 / 404 Errors)
- **Symptom**: Endpoint `/api/v1/tasks` or `/api/v1/benchmark` throws 503/404 indicating "Data file not found".
- **Fix**: Ensure the dataset path matches `PVK_DATA_ROOT`. Verify if `/home/dministrator/project/PVK-LLM/custom_perovskite_dataset` contains the target Excel books (e.g., `bandAlignment.xlsx`).

---

## 6. Verification Commands
```bash
# Verify backend status
curl -s http://localhost:8000/api/v1/health
# Verify tasks data integration
curl -s http://localhost:8000/api/v1/tasks
```
