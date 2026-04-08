# Dev checklist for autoresearch-meta

This file is for people changing code in this repo.

Goal:
- make a change
- run a few cheap checks locally
- catch breakage before trying real workers or real `train.py`

---

## What simulation is for

Use simulation as a **regression / smoke test** for the meta-agent itself.

It helps answer:
- did I break the scheduler?
- did I break the API?
- did I break worker registration/auth?
- did I break config pull / tick / result / sync flow?
- does the server still run end-to-end?

It does **not** fully prove:
- your real project's `train.py` patching works
- your real training loop calls `report()` correctly
- a long multi-hour research run behaves perfectly
- LLM-generated hypotheses are fully wired into live orchestration

Think of simulation like this:

```json
{
  "purpose": "fast repo-level safety check",
  "catches": "runtime/protocol/integration breakage",
  "does_not_fully_prove": "real-world ML training integration"
}
```

---

## Before you change anything

### 1. Make sure Python dependencies are installed

If you are using the repo virtualenv:

```bash
cd /path/to/autoresearch-meta
python -m venv .venv
source .venv/bin/activate
pip install -r meta_server/requirements.txt
pip install -r worker/requirements.txt
pip install aiohttp
```

Why `aiohttp`?
- the HTTP simulation mode uses it
- local in-memory simulation does not need it, but server simulation does

---

## After making code changes

Run these checks in order.

---

## Check 1 — fast local simulation

This is the cheapest first check.

```bash
cd /path/to/autoresearch-meta
.venv/bin/python simulate.py --workers 10 --rounds 4 --trace-file traces/local_trace.jsonl
```

What this checks:
- scheduler logic
- warmup behavior
- early stopping behavior
- extend/stop actions
- in-memory run lifecycle

What you should expect:
- first rounds may show no kills because of warmup
- later rounds should show some kills if warmup threshold is crossed
- command should finish cleanly
- trace file should be created

Artifacts:
- `traces/local_trace.jsonl`

Good sign:
- simulation completes
- no traceback
- trace file contains `local_tick`, `local_round_summary`, `local_simulation_complete`

---

## Check 2 — full HTTP/server simulation

This checks the actual server endpoints and worker-agent protocol.

### Step A — start the server locally

```bash
cd /path/to/autoresearch-meta
META_ENROLL_TOKEN=test-invite .venv/bin/python -m uvicorn meta_server.main:app --host 127.0.0.1 --port 8000
```

Leave that terminal running.

### Step B — in another terminal, run the HTTP simulator

```bash
cd /path/to/autoresearch-meta
.venv/bin/python simulate.py \
  --against-server \
  --workers 3 \
  --rounds 2 \
  --meta-url http://127.0.0.1:8000 \
  --enroll-token test-invite \
  --trace-file traces/http_trace.jsonl
```

What this checks:
- `/register`
- invite-token auth
- per-worker token issuance
- `/next_config/{worker_id}`
- `/tick`
- `/result`
- `/sync/{worker_id}`
- basic DB writes and queue flow

Artifacts:
- `traces/http_trace.jsonl`

Good sign:
- simulator completes without traceback
- server stays up
- health endpoint responds
- leaderboard responds
- trace file contains register/tick/result/sync events

---

## Check 3 — inspect traces

### Local trace example

Look at the first few lines:

```bash
head -n 20 traces/local_trace.jsonl
```

You should see events like:
- `local_run_started`
- `local_tick`
- `local_run_completed`
- `local_round_summary`

### HTTP trace example

```bash
head -n 40 traces/http_trace.jsonl
```

You should see events like:
- `http_register_request`
- `http_register_response`
- `http_next_config_request`
- `http_next_config_response`
- `http_tick_request`
- `http_tick_response`
- `http_result_request`
- `http_result_response`
- `http_sync_response`

That means you can inspect:
- what config was assigned
- what was sent to `/tick`
- what action came back
- what result was submitted
- what `program.md` workers received

---

## Check 4 — verify server state manually

While the server is running, test these:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/leaderboard
curl http://127.0.0.1:8000/program.md
curl http://127.0.0.1:8000/pipeline/status
```

What to look for:
- `/health` returns JSON and no error
- `/leaderboard` returns valid JSON list
- `/program.md` returns text
- `/pipeline/status` returns JSON

---

## What to do if simulation fails

Use the failure to narrow down the issue.

### If local simulation fails
Likely areas:
- `meta_server/scheduler.py`
- `simulate.py`
- core in-memory run lifecycle

### If HTTP simulation fails but local simulation passes
Likely areas:
- `meta_server/api.py`
- auth changes
- response schema mismatches
- DB serialization / deserialization
- server startup / FastAPI / Pydantic issues

### If `/leaderboard` or `/sync` breaks
Likely cause:
- JSON fields are being returned as strings instead of dicts
- Pydantic schema mismatch

### If `/tick` fails unexpectedly
Likely cause:
- run-id lifecycle issue
- auth issue
- async logic inside request handlers

### If `population_id` or `hypothesis_statement` is missing from traces
Likely cause:
- `meta_server/runtime.py` RuntimeState failed to initialize
- Worker not being assigned to a population (check `population_manager.active_populations`)

---

## Checking population-aware orchestration

Inspect the trace to confirm each worker got a population assignment and hypothesis statement:

```bash
grep '"http_next_config_response"' traces/http_trace.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    r = e.get('response', {})
    print(r.get('population_id'), '|', r.get('population_strategy'), '|', r.get('hypothesis_statement','')[:50])
"
```

Expected output (one line per worker round):
```
pop_a8365b | investigate | Depth > 10 improves val_bpb
pop_fea45f | investigate | Learning rate × batch size interact
...
```

Check that sync returns population-specific `program_md`:
```bash
grep '"http_sync_response"' traces/http_trace.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    r = e.get('response', {})
    prog = r.get('program_md','')[:60]
    print(r.get('population_id'), '|', prog)
"
```

Expected output:
```
pop_a8365b | # pop_a8365b — INVESTIGATE
```

If `population_id` is `null`, the runtime state was not initialized. Check the server startup logs for:
```
Spawned pop_XXXXX (investigate, 3 workers) for "Depth > 10 improves val_bpb"
```

---

## Recommended minimum release gate

Before you consider a change "safe enough" for teammates, do all of these:

```json
{
  "must_pass": [
    "local simulation completes",
    "http simulation completes",
    "trace files are written",
    "/health works",
    "/leaderboard works",
    "population_id present in next_config traces",
    "hypothesis_statement present in next_config traces",
    "sync returns population-specific program_md",
    "no traceback in server logs"
  ]
}
```

---

## When to test with a real worker

Only after the simulation checks pass.

Then do one real integration test with:
- one real worker
- one real `train.py`
- real `setup_worker.py`
- real `worker/run.py --max-runs 1`

That validates the part simulation cannot fully prove: patching and training-loop integration.

---

## Good development workflow

```json
{
  "step_1": "make code change",
  "step_2": "run local simulation",
  "step_3": "run http simulation",
  "step_4": "inspect traces (population fields, program_md)",
  "step_5": "check health + leaderboard",
  "step_6": "only then test with a real worker"
}
```

---

## Useful commands cheat sheet

### Fast local sim

```bash
.venv/bin/python simulate.py --workers 10 --rounds 4 --trace-file traces/local_trace.jsonl
```

### Start local server

```bash
META_ENROLL_TOKEN=test-invite .venv/bin/python -m uvicorn meta_server.main:app --host 127.0.0.1 --port 8000
```

### HTTP sim

```bash
.venv/bin/python simulate.py \
  --against-server \
  --workers 3 \
  --rounds 2 \
  --meta-url http://127.0.0.1:8000 \
  --enroll-token test-invite \
  --trace-file traces/http_trace.jsonl
```

### Inspect traces

```bash
head -n 40 traces/http_trace.jsonl
head -n 20 traces/local_trace.jsonl
```

### Check server endpoints

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/leaderboard
curl http://127.0.0.1:8000/program.md
curl http://127.0.0.1:8000/meta_log
