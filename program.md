# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**Worker objective: contribute to the swarm's lowest val_bpb.**
You are one worker among many coordinated by the Meta-PI agent. You do not choose global strategy alone; you execute assigned experiments accurately and report results reliably.

**Time budget** is fixed by the agent per run (typically around 5 minutes).
Do not optimize for longer training time; focus on executing the assigned config faithfully.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but avoid dramatic memory blowups and repeated OOM patterns.

**Simplicity criterion** still applies: cleaner changes with similar or better val_bpb are preferred over brittle complexity.

**First run behavior**: baseline should be established during worker setup. After setup, follow agent-assigned runs.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

Primary logging is server-side via `/result` submissions.

Worker-local notes are optional (for debugging only). If you keep local notes, include:
1. `exp_id`
2. `val_bpb`
3. `peak_vram_mb` (or GB)
4. `status` (`ok` or `crash`)
5. short note on failure mode (if any)

Do not rely on local logs as source of truth; Meta-PI server history is authoritative.

## Worker run loop (Meta-PI coordinated)

LOOP UNTIL STOPPED:

1. Sync for latest instructions (`program.md`) and metadata.
2. Pull next assigned config from Meta-PI (`/next_config/{worker_id}`).
3. Patch `train.py` with assigned `config_delta` (and assigned budget).
4. Run training and capture outputs.
5. Submit results to Meta-PI (`/result`) with `exp_id`, `val_bpb`, and crash info when relevant.
6. Sync again for possible updated instructions.
7. Repeat.

**Important coordination rule**: do not independently keep/discard global strategy. Always report outcomes; Meta-PI decides population-level direction and future assignments.

**Timeout**: If a run exceeds safe timeout (about 10 minutes wall clock), abort and report as failure.

**Crash handling**:
- If crash is due to assigned config being too aggressive (e.g. OOM), report it and continue to next assignment.
- If crash is due to local environment or obvious worker bug, fix worker-side issue and continue.

**Continuity rule**:
- Base `program.md` is the immutable charter.
- Meta-PI may update the mutable section over time.
- Workers should check for updates every run and apply them when changed.

<!-- BAD_PI_MUTABLE_START -->

## Meta-PI live update (auto-generated)

No live update yet.
Use this base template for setup and initial run behavior until the first agent update arrives.

<!-- BAD_PI_MUTABLE_END -->
