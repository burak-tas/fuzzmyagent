# fuzzmyagent.ai

Property-based fuzz testing web app for agent-to-agent endpoints.

## What fuzzmyagent does

• 🔗 Systematically tests A2A endpoints instead of one-off prompting  
• 📂 Import a CSV testcase suite to automate prompt testing at scale  
• 🤖 Run massive prompt batches against your agent endpoints  
• 🧪 Surface edge cases, weird behaviors, and silent failures  
• 📊 Export Test Reports & A2A Responses — structured outputs for debugging, audits, and regression testing  
• ⚡ Lightweight — minimal setup, no heavy frameworks  

Advanced capabilities:

• 🧠 LLM-Based Fuzzing Testcase Generator — automatically creates diverse, adversarial prompts  
• 🔐 Optional fuzzing mode using your own OpenAI key for deeper stress testing  

Instead of manually poking your agent and hoping for the best, you can now:

➡️ Reproduce failures  
➡️ Share testcase libraries  
➡️ Benchmark agent robustness  
➡️ Catch breaking changes before users do  

Built for developers working on AI agents, A2A protocols, and LLM apps who want their systems to survive contact with reality — not just demos.

## Project Structure

- `backend/` FastAPI API, runner, rules engine, SQLite storage
- `frontend/` React + Vite UI

## Run Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Wizard (Release 1)

Use the new step flow under:

- `http://localhost:5173/wizard/start`

Steps:

1. Start (endpoint)
2. Discover (agent card)
3. Configure (fuzzer + OpenAI)
4. Generate (OpenAI testcase generation)
5. Review (editable testcase table)
6. Run (live progress via websocket)
7. Report (summary + details + export)

## A2A Endpoint Contract

Backend sends:

```json
{ "input": "...", "meta": { "run_id": "...", "case_index": 0 } }
```

Expected response JSON (either is fine):

```json
{ "output": "..." }
```

or

```json
{ "message": "..." }
```

Optional for tool policies:

```json
{ "output": "...", "tool_calls": [{ "name": "tool_name", "args": {} }] }
```

## Supported Rule Types

- `json_parseable`
- `max_length` (`chars`)
- `forbidden_substrings` (`values`)
- `regex_must_match` (`pattern`)
- `tool_calls_allowlist` (`allow`)

## Wizard API Endpoints

- `POST /api/discovery`
- `POST /api/openai/test`
- `POST /api/testcases/generate`
- `GET /api/runs/{run_id}/testcases`
- `PATCH /api/testcases/{testcase_id}`
- `DELETE /api/testcases/{testcase_id}`
- `POST /api/runs/{run_id}/start`
- `POST /api/wizard/runs/{run_id}/stop`
- `GET /api/runs/{run_id}/report`
- `GET /api/wizard/runs/{run_id}`

## Notes

- Failure shrinking is intentionally lightweight in this MVP.
- CORS is open for local development.
- Storage is local SQLite (generated on startup, do not commit DB files).

## SQLite / DB

You should not commit the SQLite database file. The backend creates it automatically on startup.
