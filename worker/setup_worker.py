#!/usr/bin/env python3
"""
worker/setup_worker.py

Run ONCE before starting the main worker loop.
  1. Runs unmodified train.py to record your baseline val_bpb
  2. Registers with the meta-agent
  3. Writes .worker_config.json for run.py to read

Usage:
    python worker/setup_worker.py \
        --worker-id alice-h100 \
        --gpu-type H100 \
        --train-py /path/to/autoresearch/train.py \
        --meta-url http://<meta-agent-host>:8000
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from worker.client import register, health


def _check_report_instrumentation(train_py: Path):
    """
    Inspect train.py for report() calls.
    Warn clearly if missing — early stopping won't work without it.
    """
    src = train_py.read_text(errors="ignore")
    has_import = "from worker.report import" in src or "import report" in src
    has_call   = "report(" in src

    has_budget_key = re.search(r"^\s*TOTAL_WALL_CLOCK_TIME\s*=", src, re.MULTILINE) is not None

    expected_patch_keys = [
        "LR", "BATCH_SIZE", "HIDDEN_SIZE", "N_LAYERS", "WEIGHT_DECAY", "OPTIMIZER",
    ]
    missing_patch_keys = [
        k for k in expected_patch_keys
        if re.search(rf"^\s*{re.escape(k)}\s*=", src, re.MULTILINE) is None
    ]

    if has_import and has_call:
        print("  train.py: report() instrumentation found.")
    else:
        print()
        print("  " + "!" * 60)
        if not has_call:
            print("  WARNING: train.py does not call report(metric, progress)")
            print()
            print("  Early stopping disabled — no intermediate signal")
            print()
            print("  To enable early stopping, add to your train.py:")
            print()
            print("    from worker.report import report")
            print()
            print("    # Inside your training loop, at each eval step:")
            print("    report(val_bpb, progress)")
            print("    # where progress = elapsed_seconds / total_budget_seconds")
            print("    #       OR      = current_step / total_steps")
            print()
            print("  Without report(), the meta-agent receives no signal")
            print("  and cannot kill bad runs early or extend good ones.")
        elif not has_import:
            print("  NOTE: report() call found but import may be missing.")
            print("  Make sure train.py has: from worker.report import report")
        print("  " + "!" * 60)
        print()

    if has_budget_key:
        print("  train.py: TOTAL_WALL_CLOCK_TIME found.")
    else:
        print()
        print("  " + "!" * 60)
        print("  WARNING: train.py is missing TOTAL_WALL_CLOCK_TIME at top-level.")
        print()
        print("  Add this near your other hyperparameter constants:")
        print()
        print("    TOTAL_WALL_CLOCK_TIME = 300")
        print()
        print("  The agent patches this value per run to control budget.")
        print("  " + "!" * 60)
        print()

    if missing_patch_keys:
        print("  NOTE: Some recommended top-level tunable keys were not found:")
        print("    " + ", ".join(missing_patch_keys))
        print("  This is okay if your problem uses a different search space,")
        print("  but keep tunables as top-level KEY = value assignments.")
    else:
        print("  train.py: recommended top-level tunable keys found.")


def run_baseline(train_py: Path) -> float:
    """Run train.py unmodified, parse val_bpb from stdout."""
    print(f"Running baseline ({train_py})…")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(train_py)],
        capture_output=True, text=True, cwd=train_py.parent
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr

    # Try to parse "val_bpb: 1.2345" or "val bpb 1.2345"
    m = re.search(r"val[_\s]bpb[:\s=]+([0-9.]+)", output, re.IGNORECASE)
    if m:
        bpb = float(m.group(1))
        print(f"  Baseline val_bpb = {bpb:.4f}  (in {elapsed:.0f}s)")
        return bpb
    print("  WARNING: could not parse val_bpb from output. Using placeholder 2.0")
    print("  --- stdout ---")
    print(output[-2000:])
    return 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id",  required=True, help="Unique name, e.g. 'alice-h100-0'")
    p.add_argument("--gpu-type",   required=True, help="GPU model, e.g. 'H100'")
    p.add_argument("--train-py",   required=True, help="Path to autoresearch train.py")
    p.add_argument("--meta-url",   default="http://localhost:8000")
    p.add_argument("--contact",    default=None,  help="Optional email/discord")
    p.add_argument("--enroll-token", default=None,
                   help="Shared invite token if the server requires one")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Skip baseline run (use if you already know your val_bpb)")
    p.add_argument("--baseline-bpb", type=float, default=None)
    args = p.parse_args()

    os.environ["META_AGENT_URL"] = args.meta_url
    train_py = Path(args.train_py).resolve()
    if not train_py.exists():
        print(f"ERROR: {train_py} not found"); sys.exit(1)

    # Check meta-agent is reachable
    print(f"Checking meta-agent at {args.meta_url}…")
    try:
        h = health()
        print(f"  Connected. Experiments so far: {h['experiments']}")
    except Exception as e:
        print(f"  ERROR: cannot reach meta-agent: {e}"); sys.exit(1)

    # Baseline
    if args.skip_baseline and args.baseline_bpb:
        baseline_bpb = args.baseline_bpb
    else:
        baseline_bpb = run_baseline(train_py)

    # Register
    print(f"Registering worker '{args.worker_id}'…")
    resp = register(
        args.worker_id,
        args.gpu_type,
        baseline_bpb,
        args.contact,
        enroll_token=args.enroll_token,
    )
    print(f"  {resp['message']}")
    print("\n--- program.md (current) ---")
    print(resp["current_program_md"][:1000])
    print("---\n")

    # ── Check train.py for report() calls ────────────────────────────────
    _check_report_instrumentation(train_py)

    # Save config for run.py
    cfg = {
        "worker_id":   args.worker_id,
        "gpu_type":    args.gpu_type,
        "baseline_bpb": baseline_bpb,
        "train_py":    str(train_py),
        "meta_url":    args.meta_url,
        "contact":     args.contact,
        "worker_token": resp["worker_token"],
    }
    cfg_path = Path(__file__).parent / ".worker_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Config saved to {cfg_path}")
    print("\nSetup complete! Now run:  python worker/run.py")


if __name__ == "__main__":
    main()
