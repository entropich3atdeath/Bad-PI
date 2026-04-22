#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh


st.set_page_config(page_title="Bad PI Dashboard", page_icon="📈", layout="wide")


def _get_json(base_url: str, path: str, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(f"{base_url}{path}", headers=headers or {})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8")
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))


def _safe_fetch(base_url: str) -> tuple[dict[str, Any], str | None]:
    try:
        health = _get_json(base_url, "/health")
        runs_stats = _get_json(base_url, "/runs/stats")
        leaderboard = _get_json(base_url, "/leaderboard")
        populations = _get_json(base_url, "/populations")
        theory = _get_json(base_url, "/theory_graph")
        theory_human = _get_json(base_url, "/theory_graph/human?include_graph=true")
        shadow = _get_json(base_url, "/shadow/scorecards")
        return {
            "health": health,
            "runs_stats": runs_stats,
            "leaderboard": leaderboard if isinstance(leaderboard, list) else leaderboard.get("leaderboard", []),
            "populations": populations.get("populations", []) if isinstance(populations, dict) else [],
            "theory": theory,
            "theory_human": theory_human,
            "shadow_scorecards": shadow.get("scorecards", []) if isinstance(shadow, dict) else [],
            "fetched_at": time.time(),
        }, None
    except urllib.error.HTTPError as e:
        return {}, f"HTTP error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return {}, f"Connection error: {e.reason}"
    except Exception as e:
        return {}, f"Unexpected error: {type(e).__name__}: {e}"


def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "probe_worker_id" not in st.session_state:
        st.session_state.probe_worker_id = ""
    if "probe_worker_token" not in st.session_state:
        st.session_state.probe_worker_token = ""


def _record_snapshot(payload: dict[str, Any]):
    theory_nodes = payload.get("theory", {}).get("nodes", [])
    row = {
        "ts": payload.get("fetched_at", time.time()),
        "experiments": payload.get("health", {}).get("experiments", 0),
        "best_delta": payload.get("runs_stats", {}).get("best_delta", None),
        "kill_rate": payload.get("runs_stats", {}).get("kill_rate", None),
        "active_workers": payload.get("health", {}).get("active_workers", 0),
        "posteriors": {
            str(n.get("statement", n.get("id", "unknown"))): float(n.get("posterior", 0.0))
            for n in theory_nodes
        },
    }

    # Record every fetch so the dashboard can show trajectory over time,
    # even when experiment count is temporarily flat.
    st.session_state.history.append(row)

    if len(st.session_state.history) > 2000:
        st.session_state.history = st.session_state.history[-2000:]


def _build_posterior_history_points() -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for snap in st.session_state.history:
        ts = snap.get("ts")
        ts_label = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
        experiments = snap.get("experiments")
        for statement, posterior in (snap.get("posteriors") or {}).items():
            points.append(
                {
                    "ts": ts,
                    "ts_label": ts_label,
                    "experiments": experiments,
                    "statement": statement,
                    "posterior": posterior,
                }
            )
    return points


def _status_counts(nodes: list[dict[str, Any]]) -> dict[str, int]:
    out = {"supported": 0, "active": 0, "refuted": 0, "other": 0}
    for n in nodes:
        s = str(n.get("status", "other")).lower()
        if s in out:
            out[s] += 1
        else:
            out["other"] += 1
    return out


def _sample_population_programs(base_url: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Local debug helper: sample one worker per active population and fetch /sync
    with worker token auth, so we can preview population-specific program_md blocks.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        state_path = root / "meta_server" / "runtime_state.json"
        db_path = root / "meta_server" / "experiments.db"

        if not state_path.exists() or not db_path.exists():
            return [], "runtime_state.json or experiments.db not found"

        state = json.loads(state_path.read_text(encoding="utf-8"))
        worker_pop = state.get("population_manager", {}).get("worker_populations", {})
        if not worker_pop:
            return [], "no worker-population assignments yet"

        by_pop: dict[str, str] = {}
        for wid, pid in worker_pop.items():
            if pid not in by_pop:
                by_pop[pid] = wid

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT worker_id, auth_token FROM workers").fetchall()
        con.close()
        tokens = {r["worker_id"]: r["auth_token"] for r in rows}

        previews: list[dict[str, Any]] = []
        for pid, wid in sorted(by_pop.items()):
            token = tokens.get(wid)
            if not token:
                continue
            sync = _get_json(base_url, f"/sync/{wid}", headers={"X-Worker-Token": token})
            text = str(sync.get("program_md") or "")

            mutable = ""
            if "<!-- BAD_PI_MUTABLE_START -->" in text and "<!-- BAD_PI_MUTABLE_END -->" in text:
                mutable = text.split("<!-- BAD_PI_MUTABLE_START -->", 1)[1].split("<!-- BAD_PI_MUTABLE_END -->", 1)[0].strip()

            previews.append(
                {
                    "population_id": sync.get("population_id") or pid,
                    "worker_id": wid,
                    "strategy": sync.get("population_strategy"),
                    "hypothesis": sync.get("hypothesis_statement"),
                    "program_digest": sync.get("program_digest"),
                    "sync_link": f"{base_url}/sync/{wid}",
                    "curl": f"curl -s -H 'X-Worker-Token: {token}' '{base_url}/sync/{wid}' | jq -r .program_md",
                    "program_md_full": text,
                    "mutable_preview": "\n".join(mutable.splitlines()[:12]) if mutable else "",
                }
            )

        return previews, None
    except Exception as e:
        return [], f"population preview unavailable: {type(e).__name__}: {e}"


_init_state()

st.sidebar.title("Bad PI Dashboard")
default_base_url = os.environ.get("BAD_PI_META_URL", "http://localhost:8000")
base_url = st.sidebar.text_input("Meta server URL", value=default_base_url).rstrip("/")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False, key="auto_refresh_enabled")
refresh_sec = st.sidebar.slider("Refresh interval (seconds)", min_value=2, max_value=60, value=8, step=1)
if st.sidebar.button("Refresh now"):
    st.rerun()

if auto_refresh:
    st_autorefresh(interval=refresh_sec * 1000, key="dashboard_autorefresh")

payload, err = _safe_fetch(base_url)

st.title("📈 Bad PI live dashboard")
st.caption("Health, leaderboard, belief dynamics, and worker-facing program view")

if err:
    st.error(err)
    st.stop()

_record_snapshot(payload)

health = payload["health"]
runs_stats = payload["runs_stats"]
leaderboard = payload["leaderboard"]
populations = payload.get("populations", [])
nodes = payload.get("theory", {}).get("nodes", [])
shadow_scorecards = payload.get("shadow_scorecards", [])
status = _status_counts(nodes)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Experiments", health.get("experiments", 0))
k2.metric("Active workers", health.get("active_workers", 0))
k3.metric("Queue depth", health.get("queue_depth", 0))
k4.metric("Best delta", runs_stats.get("best_delta", "n/a"))
k5.metric("Kill rate %", runs_stats.get("kill_rate", "n/a"))
k6.metric("Hypotheses", len(nodes))

left, right = st.columns([2, 1])

with left:
    st.subheader("Belief posterior trajectories")
    points = _build_posterior_history_points()
    chart_mode = st.radio(
        "View",
        ["Trajectory (line)", "Current posterior (bar)", "Change since first snapshot (bar)"],
        horizontal=True,
    )

    if chart_mode == "Trajectory (line)":
        if len(points) >= 2:
            fig = px.line(
                points,
                x="ts",
                y="posterior",
                color="statement",
                markers=True,
                hover_data=["ts_label", "experiments"],
                title="Posterior by hypothesis over time",
            )
            fig.update_layout(height=380, yaxis=dict(range=[0, 1]), xaxis_title="time")
            st.plotly_chart(fig, config={"responsive": True})
        else:
            st.info("Collecting trajectory points. Wait for at least two refreshes.")
    elif chart_mode == "Current posterior (bar)":
        curr = [
            {
                "statement": n.get("statement", n.get("id", "unknown")),
                "posterior": float(n.get("posterior", 0.0)),
            }
            for n in nodes
        ]
        if curr:
            fig = px.bar(curr, x="statement", y="posterior", title="Current posterior by hypothesis")
            fig.update_layout(height=380, yaxis=dict(range=[0, 1]), xaxis_title="hypothesis")
            st.plotly_chart(fig, config={"responsive": True})
        else:
            st.info("No hypothesis rows available.")
    else:
        if st.session_state.history:
            first = st.session_state.history[0].get("posteriors", {})
            last = st.session_state.history[-1].get("posteriors", {})
            rows = []
            for k, v in last.items():
                rows.append({
                    "statement": k,
                    "delta_posterior": float(v) - float(first.get(k, v)),
                })
            fig = px.bar(rows, x="statement", y="delta_posterior", title="Posterior change since first dashboard snapshot")
            fig.update_layout(height=380, xaxis_title="hypothesis")
            st.plotly_chart(fig, config={"responsive": True})
        else:
            st.info("No history yet.")

with right:
    st.subheader("Current hypothesis statuses")
    pie = go.Figure(
        data=[
            go.Pie(
                labels=list(status.keys()),
                values=list(status.values()),
                hole=0.45,
            )
        ]
    )
    pie.update_layout(height=380)
    st.plotly_chart(pie, config={"responsive": True})

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("Leaderboard (top workers)")
    st.dataframe(leaderboard[:25], width="stretch", hide_index=True)

with c2:
    st.subheader("Current posterior snapshot")
    snapshot_rows = [
        {
            "statement": n.get("statement"),
            "status": n.get("status"),
            "posterior": n.get("posterior"),
            "n_experiments": n.get("n_experiments"),
            "effect_mu": n.get("effect_mu"),
            "effect_sem": n.get("effect_sem"),
        }
        for n in nodes
    ]
    snapshot_rows = sorted(snapshot_rows, key=lambda r: r.get("posterior") or 0, reverse=True)
    st.dataframe(snapshot_rows, width="stretch", hide_index=True)

st.subheader("Population allocation (preference view)")
if populations:
    st.dataframe(populations, width="stretch", hide_index=True)
else:
    st.info("No active population rows yet.")

st.subheader("Shadow hypothesis scorecards")
if shadow_scorecards:
    st.dataframe(shadow_scorecards, width="stretch", hide_index=True)
else:
    st.info("No shadow hypotheses yet. Add one via /shadow/hypotheses.")

st.subheader("Theory graph summary")
derived = payload.get("theory_human", {}).get("derived_layer", {})
summary_text = derived.get("summary_text") or "(no summary)"
st.text(summary_text)

st.subheader("View worker-facing program.md")
with st.expander("Create/use probe worker and fetch /sync program_md", expanded=False):
    enroll_token = st.text_input("Enroll token (only needed if your server requires it)", value="", type="password")
    probe_col1, probe_col2 = st.columns(2)

    with probe_col1:
        if st.button("Register new probe worker"):
            wid = f"dash_probe_{random.randint(1000,9999)}"
            try:
                reg = _post_json(
                    base_url,
                    "/register",
                    {
                        "worker_id": wid,
                        "gpu_type": "DASHBOARD",
                        "baseline_bpb": 0.1,
                        "contact": None,
                        "enroll_token": enroll_token or None,
                    },
                )
                st.session_state.probe_worker_id = wid
                st.session_state.probe_worker_token = reg.get("worker_token", "")
                st.success(f"Registered probe worker: {wid}")
            except Exception as e:
                st.error(f"Failed to register probe worker: {e}")

    with probe_col2:
        st.session_state.probe_worker_id = st.text_input("Worker ID", value=st.session_state.probe_worker_id)
        st.session_state.probe_worker_token = st.text_input(
            "Worker token",
            value=st.session_state.probe_worker_token,
            type="password",
        )

    if st.button("Fetch /sync program_md"):
        try:
            headers = {}
            if st.session_state.probe_worker_token:
                headers["X-Worker-Token"] = st.session_state.probe_worker_token
            sync = _get_json(base_url, f"/sync/{st.session_state.probe_worker_id}", headers=headers)
            st.json(
                {
                    "experiment_count": sync.get("experiment_count"),
                    "population_id": sync.get("population_id"),
                    "population_strategy": sync.get("population_strategy"),
                    "hypothesis_id": sync.get("hypothesis_id"),
                    "program_digest": sync.get("program_digest"),
                }
            )
            st.code(sync.get("program_md", ""), language="markdown")
        except Exception as e:
            st.error(f"Failed to fetch /sync: {e}")

st.subheader("Population-specific program.md previews")
previews, perr = _sample_population_programs(base_url)
if perr:
    st.info(perr)
else:
    st.caption("One sampled worker per active population")
    for p in previews:
        with st.expander(
            f"{p['population_id']} · {p.get('strategy', 'unknown')} · {p.get('hypothesis', '')}",
            expanded=False,
        ):
            st.write({
                "population_id": p.get("population_id"),
                "worker_id": p.get("worker_id"),
                "strategy": p.get("strategy"),
                "hypothesis": p.get("hypothesis"),
                "program_digest": p.get("program_digest"),
            })
            st.markdown(f"Sync endpoint: {p['sync_link']}")
            st.code(p.get("curl", ""), language="bash")
            st.code(p.get("mutable_preview", ""), language="markdown")
            st.markdown("Full worker-facing program.md")
            st.code(p.get("program_md_full", ""), language="markdown")

st.caption(f"Server: {base_url} · snapshots tracked this session: {len(st.session_state.history)}")
