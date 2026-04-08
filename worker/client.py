"""
worker/client.py
HTTP client — all calls the worker makes to the meta-agent.
"""
from __future__ import annotations
import os
import time
import requests
from typing import Optional

META_URL = os.environ.get("META_AGENT_URL", "http://localhost:8000")
TIMEOUT  = 15  # seconds


def _url(path: str) -> str:
    return f"{META_URL.rstrip('/')}/{path.lstrip('/')}"


def _auth_headers(worker_token: Optional[str]) -> dict:
    return {"X-Worker-Token": worker_token} if worker_token else {}


def register(
    worker_id: str,
    gpu_type: str,
    baseline_bpb: float,
    contact: Optional[str] = None,
    enroll_token: Optional[str] = None,
) -> dict:
    r = requests.post(_url("/register"), json={
        "worker_id": worker_id,
        "gpu_type": gpu_type,
        "baseline_bpb": baseline_bpb,
        "contact": contact,
        "enroll_token": enroll_token,
    }, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_next_config(worker_id: str, worker_token: Optional[str], retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(
                _url(f"/next_config/{worker_id}"),
                headers=_auth_headers(worker_token),
                timeout=TIMEOUT,
            )
            if r.status_code == 503:
                print(f"  Queue empty, retrying in 15s… ({attempt+1}/{retries})")
                time.sleep(15)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Network error ({e}), retrying in 10s…")
            time.sleep(10)
    raise RuntimeError("Could not get next config after retries")


def submit_result(
    worker_id: str,
    worker_token: Optional[str],
    exp_id: str,
    config: dict,
    config_delta: dict,
    val_bpb: Optional[float],
    delta_bpb: Optional[float],
    duration_seconds: float,
    error: Optional[str] = None,
) -> dict:
    payload = {
        "worker_id": worker_id,
        "exp_id": exp_id,
        "config": config,
        "config_delta": config_delta,
        "val_bpb": val_bpb,
        "delta_bpb": delta_bpb,
        "duration_seconds": duration_seconds,
        "error": error,
    }
    r = requests.post(_url("/result"), json=payload, headers=_auth_headers(worker_token), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def sync(worker_id: str, worker_token: Optional[str]) -> dict:
    r = requests.get(_url(f"/sync/{worker_id}"), headers=_auth_headers(worker_token), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_spec_payload() -> Optional[dict]:
    """
    Get speculative program payload when server confidence gates allow deployment.
    Returns None when no deployable spec exists.
    """
    r = requests.get(_url("/pipeline/spec_payload"), timeout=TIMEOUT)
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


def get_flush_token() -> Optional[str]:
    r = requests.get(_url("/pipeline/flush_token"), timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("flush_token")


def health() -> dict:
    r = requests.get(_url("/health"), timeout=5)
    r.raise_for_status()
    return r.json()
