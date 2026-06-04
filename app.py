"""
app.py — AI Legal Review System
================================
Streamlit UI simulating a government-approved legal document review platform.

Run:
    python -m streamlit run app.py

Features:
  - Dashboard with system status & difficulty selector
  - Clause Review Panel (text, type, jurisdiction, risk)
  - Agent Action Display (approve / flag / redline / escalate / clarify)
  - Metrics Panel (reward, F1, precision, recall)
  - Step Simulation (next step / full run)
  - Document Storage (upload + in-memory store)
  - Encryption Simulation (SHA-256 hash + base64)
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import sys
import types
import time
from datetime import datetime
from typing import Optional

import numpy as np
import streamlit as st

# ── Mock heavy deps if not installed (for demo without gymnasium/faker) ──────
def _mock_if_missing():
    for mod, sub in [("gymnasium","gymnasium.spaces"),("faker",None)]:
        if mod not in sys.modules:
            m = types.ModuleType(mod)
            if mod == "gymnasium":
                sp = types.ModuleType("gymnasium.spaces")
                class _D:
                    def __init__(self,n): self.n=n
                    def contains(self,x): return 0<=int(x)<self.n
                    def sample(self): return random.randint(0,self.n-1)
                class _B:
                    def __init__(self,lo,hi,shape,dtype): self.shape=shape;self.dtype=dtype
                class _E:
                    metadata={}
                    action_space=None; observation_space=None
                    def reset(self,**kw): pass
                    def step(self,a): pass
                    def close(self): pass
                sp.Discrete=_D; sp.Box=_B
                m.Env=_E; m.spaces=sp
                sys.modules["gymnasium"]=m; sys.modules["gymnasium.spaces"]=sp
            elif mod == "faker":
                class _F:
                    @staticmethod
                    def seed(s): random.seed(s)
                    def bs(self):
                        return random.choice([
                            "synergize scalable markets","leverage agile frameworks",
                            "matrix B2B deliverables","orchestrate cross-platform content",
                            "disintermediate granular paradigms","productize end-to-end channels",
                        ])
                m.Faker=_F
                sys.modules["faker"]=m

_mock_if_missing()
sys.path.insert(0,".")

from env.legal_review_env import (
    LegalReviewEnv, Action, ACTION_LABELS, RiskLevel,
    CLAUSE_TYPES, DIFFICULTY_PRESETS,
)
from grader import grade_episode, grade_all_tasks
from tasks import make_env


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Review System",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — government-grade: white/navy/gray ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    color: #1a1a2e;
}
.main { background: #f5f6f8; }

/* Header band */
.gov-header {
    background: #0a2a5e;
    color: white;
    padding: 14px 24px;
    display: flex; align-items: center; gap: 16px;
    border-bottom: 3px solid #c8a951;
    margin-bottom: 0;
    font-family: 'Source Serif 4', serif;
}
.gov-header .seal { font-size: 32px; }
.gov-header h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: 0.02em; }
.gov-header p  { margin: 0; font-size: 11px; color: #a8bcd8; letter-spacing: 0.06em; }

/* Section labels */
.section-label {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    color: #5a6a8a; text-transform: uppercase;
    border-bottom: 1px solid #d8dde8;
    padding-bottom: 4px; margin-bottom: 12px;
}

/* Metric tiles */
.metric-tile {
    background: white; border: 1px solid #d8dde8;
    border-left: 3px solid #0a2a5e;
    border-radius: 3px; padding: 12px 16px;
    margin-bottom: 8px;
}
.metric-tile .label { font-size: 11px; color: #5a6a8a; font-weight: 500; }
.metric-tile .value { font-size: 22px; font-weight: 600; color: #0a2a5e; font-family: 'IBM Plex Mono',monospace; }

/* Clause card */
.clause-card {
    background: white; border: 1px solid #d8dde8;
    border-top: 3px solid #0a2a5e;
    border-radius: 3px; padding: 16px 20px;
    margin-bottom: 12px;
}
.clause-text {
    font-family: 'Source Serif 4', serif;
    font-size: 13px; line-height: 1.7;
    color: #2a3a5e; padding: 12px;
    background: #f9fafb; border-left: 3px solid #c8a951;
    border-radius: 0 3px 3px 0;
}

/* Risk badges */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 2px; font-size: 11px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-none     { background:#eaf3e8; color:#1a5c1a; border:1px solid #a8d8a8; }
.badge-low      { background:#e8f0fa; color:#1a3a6e; border:1px solid #a8c0e8; }
.badge-medium   { background:#fff3e0; color:#8a4a00; border:1px solid #f0c060; }
.badge-high     { background:#fce8e8; color:#8a1a1a; border:1px solid #e8a0a0; }
.badge-critical { background:#4a0000; color:#ffffff; border:1px solid #8a1a1a; }

/* Action badge */
.action-approve   { background:#eaf3e8; color:#1a5c1a; }
.action-flag      { background:#fce8e8; color:#8a1a1a; }
.action-redline   { background:#fff3e0; color:#8a4a00; }
.action-clarify   { background:#e8f0fa; color:#1a3a6e; }
.action-escalate  { background:#f0e8fa; color:#4a1a8a; }

/* Encrypted text */
.encrypted {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    background: #1a1a2e; color: #00ff88;
    padding: 10px 14px; border-radius: 3px;
    word-break: break-all; line-height: 1.6;
    border: 1px solid #2a4a2e;
}

/* Log output */
.log-box {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    background: #0a1a0a; color: #88cc88;
    padding: 12px 16px; border-radius: 3px;
    max-height: 300px; overflow-y: auto;
    line-height: 1.6; border: 1px solid #1a3a1a;
}
.log-start  { color: #60a0ff; }
.log-step   { color: #88cc88; }
.log-end    { color: #ffcc00; }
.log-err    { color: #ff6060; }

/* Status dot */
.status-online  { color: #1a8a1a; font-weight:600; }
.status-offline { color: #8a1a1a; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0a2a5e !important; }
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label { color: #a8bcd8 !important; }
section[data-testid="stSidebar"] hr { border-color: #1a4a8e !important; }

/* Buttons */
.stButton > button {
    background: #0a2a5e !important; color: white !important;
    border: none !important; border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 18px !important;
    letter-spacing: 0.03em !important;
}
.stButton > button:hover { background: #1a4a8e !important; }

/* Table */
.gov-table { width:100%; border-collapse:collapse; font-size:12px; }
.gov-table th { background:#0a2a5e; color:white; padding:8px 12px; text-align:left; font-weight:500; }
.gov-table td { padding:7px 12px; border-bottom:1px solid #d8dde8; }
.gov-table tr:nth-child(even) td { background:#f5f6f8; }
</style>
""", unsafe_allow_html=True)


# ── Government header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="gov-header">
  <div class="seal">⚖</div>
  <div>
    <h1>AI Legal Review System</h1>
    <p>OFFICIAL USE ONLY &nbsp;·&nbsp; DOCUMENT CLASSIFICATION: RESTRICTED &nbsp;·&nbsp; VERSION 1.0.0</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "env":              None,
        "difficulty":       "easy",
        "obs":              None,
        "done":             False,
        "step_num":         0,
        "log_lines":        [],
        "all_results":      {},
        "documents":        [],       # uploaded doc store
        "current_clause":   None,
        "last_action":      None,
        "last_reward":      None,
        "metrics_snapshot": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_badge(risk_name: str) -> str:
    cls = {"NONE":"none","LOW":"low","MEDIUM":"medium","HIGH":"high","CRITICAL":"critical"}.get(risk_name,"low")
    return f'<span class="badge badge-{cls}">{risk_name}</span>'

def _action_badge(action_name: str) -> str:
    cls = {
        "approve":"approve","flag_clause":"flag","redline":"redline",
        "request_clarify":"clarify","escalate_counsel":"escalate",
    }.get(action_name,"approve")
    return f'<span class="badge action-{cls}">{action_name.upper().replace("_"," ")}</span>'

def _encrypt_text(text: str) -> dict:
    sha = hashlib.sha256(text.encode()).hexdigest()
    b64 = base64.b64encode(text.encode()).decode()
    return {"sha256": sha, "base64": b64}

def _agent_action(clause_ctx: dict) -> int:
    _HIGH = {"indemnity","limitation_of_liability","termination","ip_ownership"}
    if clause_ctx.get("time_remaining", 999) < 10:
        return int(Action.ESCALATE_COUNSEL)
    if clause_ctx.get("clause_type","") in _HIGH:
        return int(Action.FLAG_CLAUSE)
    if clause_ctx.get("has_nested_ref", False):
        return int(Action.REQUEST_CLARIFY)
    if clause_ctx.get("clause_type","") in {"payment_terms","data_protection"} and clause_ctx.get("prior_redlines",0) > 1:
        return int(Action.REDLINE)
    return int(Action.APPROVE)

def _start_env(difficulty: str):
    env = make_env(difficulty)
    obs = env.reset()
    st.session_state.env              = env
    st.session_state.difficulty       = difficulty
    st.session_state.obs              = obs
    st.session_state.done             = False
    st.session_state.step_num         = 0
    st.session_state.log_lines        = []
    st.session_state.last_action      = None
    st.session_state.last_reward      = None
    st.session_state.metrics_snapshot = None
    snap = env.state()
    st.session_state.current_clause   = snap.get("current_clause")
    st.session_state.log_lines.append(
        f'<span class="log-start">[START] Task: {difficulty}</span>'
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖ Control Panel")
    st.markdown("---")

    diff = st.selectbox(
        "Task Difficulty",
        ["easy", "medium", "hard"],
        index=["easy","medium","hard"].index(st.session_state.difficulty),
        help="easy=NDA(10) | medium=SaaS(50) | hard=M&A(120)",
    )

    if st.button("▶  Initialise Environment"):
        _start_env(diff)
        st.success("Environment ready.")

    st.markdown("---")
    st.markdown("**System Status**")
    env_status = "● ONLINE" if st.session_state.env else "○ NOT INITIALISED"
    cls = "status-online" if st.session_state.env else "status-offline"
    st.markdown(f'<span class="{cls}">{env_status}</span>', unsafe_allow_html=True)

    if st.session_state.env:
        preset = DIFFICULTY_PRESETS[st.session_state.difficulty]
        st.markdown(f"**Contract:** {preset['contract_type']}")
        st.markdown(f"**Clauses:** {preset['n_clauses']}")
        tb = preset.get("time_budget")
        st.markdown(f"**Time budget:** {'∞' if tb is None else tb}")
        st.markdown(f"**Step:** {st.session_state.step_num}")
        st.markdown(f"**Done:** {'✓' if st.session_state.done else '—'}")

    st.markdown("---")
    st.markdown("**Reference**")
    st.markdown("""
    `0` APPROVE  
    `1` FLAG  
    `2` REDLINE  
    `3` CLARIFY  
    `4` ESCALATE
    """)


# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")


# ──────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN: Clause Review + Agent + Metrics
# ──────────────────────────────────────────────────────────────────────────────
with col_left:

    # ── Dashboard summary strip ───────────────────────────────────────────────
    st.markdown('<div class="section-label">Dashboard</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    env = st.session_state.env
    total_reward = 0.0
    reviewed = 0
    remaining = 0
    if env:
        info = env._get_info()
        total_reward = info["total_reward"]
        reviewed     = info["reviewed"]
        remaining    = info["remaining"]
    d1.metric("Reviewed",    reviewed)
    d2.metric("Remaining",   remaining)
    d3.metric("Total Reward", f"{total_reward:.3f}")
    d4.metric("Status", "DONE" if st.session_state.done else "ACTIVE")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Clause Review Panel ───────────────────────────────────────────────────
    st.markdown('<div class="section-label">Clause Review Panel</div>', unsafe_allow_html=True)
    clause = st.session_state.current_clause

    if clause:
        meta_cols = st.columns(4)
        meta_cols[0].markdown(f"**Clause ID**<br>`{clause['clause_id']}`", unsafe_allow_html=True)
        meta_cols[1].markdown(f"**Type**<br>`{clause['clause_type']}`",    unsafe_allow_html=True)
        meta_cols[2].markdown(f"**Jurisdiction**<br>`{clause['jurisdiction']}`", unsafe_allow_html=True)
        meta_cols[3].markdown(
            f"**Risk**<br>{_risk_badge(clause.get('true_risk','—'))}",
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="clause-card">
          <div class="clause-text">{clause.get('text','—')}</div>
          <div style="margin-top:8px; font-size:11px; color:#5a6a8a;">
            Nested ref: {'YES' if clause.get('has_nested_ref') else 'no'} &nbsp;|&nbsp;
            Prior redlines: {clause.get('prior_redlines',0)} &nbsp;|&nbsp;
            Liability clause: {'YES' if clause.get('contains_liability') else 'no'}
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.done:
        st.info("Episode complete. Reinitialise to start a new review.")
    else:
        st.info("Initialise an environment from the sidebar to begin.")

    # ── Agent Action ──────────────────────────────────────────────────────────
    if st.session_state.last_action is not None:
        st.markdown('<div class="section-label">Last Agent Decision</div>', unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        ac1.markdown(
            f"**Action:** {_action_badge(ACTION_LABELS[st.session_state.last_action])}",
            unsafe_allow_html=True,
        )
        ac2.markdown(f"**Reward:** `{st.session_state.last_reward:+.4f}`")

    # ── Metrics Panel ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Performance Metrics</div>', unsafe_allow_html=True)
    snap_m = st.session_state.metrics_snapshot
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("F1 Score",  f"{snap_m['f1']:.3f}"        if snap_m else "—")
    mc2.metric("Precision", f"{snap_m['precision']:.3f}" if snap_m else "—")
    mc3.metric("Recall",    f"{snap_m['recall']:.3f}"    if snap_m else "—")
    mc4.metric("Score",
        f"{grade_episode(env):.3f}" if (env and env._cursor > 0) else "—"
    )

    # ── Step controls ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Simulation Controls</div>', unsafe_allow_html=True)
    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        if st.button("▶  Next Step"):
            if env and not st.session_state.done:
                s   = env.state()
                ctx = s.get("current_clause") or {}
                ctx["contract_type"]  = s.get("contract_type","")
                ctx["time_remaining"] = s.get("time_remaining",999)
                action = _agent_action(ctx)
                obs, reward, done, info = env.step(action)
                st.session_state.step_num    += 1
                st.session_state.obs          = obs
                st.session_state.done         = done
                st.session_state.last_action  = action
                st.session_state.last_reward  = reward
                st.session_state.metrics_snapshot = env.episode_metrics()
                new_snap = env.state()
                st.session_state.current_clause = new_snap.get("current_clause")
                st.session_state.log_lines.append(
                    f'<span class="log-step">[STEP]  step={st.session_state.step_num}, reward={reward:.4f}</span>'
                )
                if done:
                    score = grade_episode(env)
                    st.session_state.log_lines.append(
                        f'<span class="log-end">[END]   Task: {st.session_state.difficulty}, Score: {score:.4f}</span>'
                    )
            else:
                st.warning("Initialise environment first or episode is complete.")

    with btn2:
        if st.button("⏩  Run Full Simulation"):
            if env and not st.session_state.done:
                progress_bar = st.progress(0)
                n_total = env.preset["n_clauses"]
                while not st.session_state.done:
                    s   = env.state()
                    ctx = s.get("current_clause") or {}
                    ctx["contract_type"]  = s.get("contract_type","")
                    ctx["time_remaining"] = s.get("time_remaining",999)
                    action = _agent_action(ctx)
                    obs, reward, done, info = env.step(action)
                    st.session_state.step_num   += 1
                    st.session_state.done        = done
                    st.session_state.last_action = action
                    st.session_state.last_reward = reward
                    st.session_state.log_lines.append(
                        f'<span class="log-step">[STEP]  step={st.session_state.step_num}, reward={reward:.4f}</span>'
                    )
                    progress_bar.progress(min(st.session_state.step_num / n_total, 1.0))
                    if done:
                        break
                score = grade_episode(env)
                st.session_state.metrics_snapshot = env.episode_metrics()
                new_snap = env.state()
                st.session_state.current_clause = new_snap.get("current_clause")
                st.session_state.log_lines.append(
                    f'<span class="log-end">[END]   Task: {st.session_state.difficulty}, Score: {score:.4f}</span>'
                )
                progress_bar.progress(1.0)
                st.success(f"Simulation complete. Final score: {score:.4f}")
            else:
                st.warning("Initialise environment first or episode is complete.")

    with btn3:
        if st.button("↺  Reset"):
            if diff:
                _start_env(diff)
                st.rerun()

    # ── Decision log table ────────────────────────────────────────────────────
    if env and env._decisions:
        st.markdown('<div class="section-label">Decision Log</div>', unsafe_allow_html=True)
        rows = env._decisions[-10:]  # last 10
        rows_html = "".join(
            f"<tr><td>{d['clause_id']}</td><td>{d['clause_type']}</td>"
            f"<td>{d['jurisdiction']}</td><td>{d['action'].upper()}</td>"
            f"<td>{d['true_risk']}</td><td>{d['reward']:+.3f}</td></tr>"
            for d in reversed(rows)
        )
        st.markdown(f"""
        <table class="gov-table">
          <thead><tr>
            <th>ID</th><th>Type</th><th>Jurisdiction</th>
            <th>Action</th><th>True Risk</th><th>Reward</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Log + Document Storage + Encryption
# ──────────────────────────────────────────────────────────────────────────────
with col_right:

    # ── Execution log ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Execution Log</div>', unsafe_allow_html=True)
    log_html = "<br>".join(st.session_state.log_lines) if st.session_state.log_lines else "No log entries yet."
    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Document storage ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Document Storage</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload contract document (.txt / .pdf / any text)",
        type=["txt","pdf","md","csv","json"],
        key="doc_upload",
    )
    manual_text = st.text_area(
        "Or paste document text directly:",
        height=90,
        placeholder="Paste contract text here…",
        key="manual_text",
    )

    if st.button("▲  Store Document"):
        content = ""
        filename = "manual_input.txt"
        if uploaded:
            content  = uploaded.read().decode("utf-8", errors="replace")
            filename = uploaded.name
        elif manual_text.strip():
            content = manual_text.strip()
        if content:
            enc = _encrypt_text(content)
            entry = {
                "id":        len(st.session_state.documents) + 1,
                "filename":  filename,
                "stored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "size":      f"{len(content)} chars",
                "sha256":    enc["sha256"][:16] + "…",
                "content":   content,
                "enc":       enc,
            }
            st.session_state.documents.append(entry)
            st.success(f"Document '{filename}' stored (ID {entry['id']}).")
        else:
            st.warning("No content to store.")

    if st.session_state.documents:
        st.markdown(f"**Stored documents: {len(st.session_state.documents)}**")
        rows_html = "".join(
            f"<tr><td>{d['id']}</td><td>{d['filename']}</td>"
            f"<td>{d['stored_at']}</td><td>{d['size']}</td>"
            f"<td><code style='font-size:10px'>{d['sha256']}</code></td></tr>"
            for d in st.session_state.documents
        )
        st.markdown(f"""
        <table class="gov-table">
          <thead><tr><th>#</th><th>Filename</th><th>Stored At</th><th>Size</th><th>SHA-256 (prefix)</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Encryption simulation ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">Encryption Simulation</div>', unsafe_allow_html=True)

    enc_input = st.text_area(
        "Text to encrypt:",
        value="This agreement is governed by the laws of New York.",
        height=70,
        key="enc_input",
    )

    if st.button("🔒  Encrypt"):
        if enc_input.strip():
            enc = _encrypt_text(enc_input.strip())
            st.markdown("**Original text:**")
            st.code(enc_input.strip(), language=None)

            st.markdown("**SHA-256 hash:**")
            st.markdown(f'<div class="encrypted">{enc["sha256"]}</div>', unsafe_allow_html=True)

            st.markdown("**Base64 encoding:**")
            st.markdown(f'<div class="encrypted">{enc["base64"]}</div>', unsafe_allow_html=True)

            # If a stored document selected, show its encryption too
            if st.session_state.documents:
                last = st.session_state.documents[-1]
                with st.expander(f"View encryption for stored doc: {last['filename']}"):
                    st.markdown("**SHA-256:**")
                    st.markdown(f'<div class="encrypted">{last["enc"]["sha256"]}</div>', unsafe_allow_html=True)
                    st.markdown("**Base64 (first 200 chars):**")
                    b64_preview = last["enc"]["base64"][:200] + ("…" if len(last["enc"]["base64"]) > 200 else "")
                    st.markdown(f'<div class="encrypted">{b64_preview}</div>', unsafe_allow_html=True)

    # ── All-tasks score summary ───────────────────────────────────────────────
    if st.session_state.all_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Multi-Task Score Summary</div>', unsafe_allow_html=True)
        agg = grade_all_tasks(st.session_state.all_results)
        score_html = "".join(
            f"<tr><td>{t.upper()}</td><td>{s:.4f}</td></tr>"
            for t, s in agg["per_task"].items()
        )
        score_html += f"<tr><td><strong>OVERALL</strong></td><td><strong>{agg['final_score']:.4f}</strong></td></tr>"
        st.markdown(f"""
        <table class="gov-table">
          <thead><tr><th>Task</th><th>Score</th></tr></thead>
          <tbody>{score_html}</tbody>
        </table>
        <p style="font-size:11px;color:#5a6a8a;margin-top:4px;">
          Passed: {'✓ YES' if agg['passed'] else '✗ NO'} (threshold 0.40)
        </p>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #d8dde8; padding-top:10px; font-size:10px; color:#8a9ab8; text-align:center; letter-spacing:0.04em;">
  AI LEGAL REVIEW SYSTEM &nbsp;·&nbsp; META PYTORCH OPENENV HACKATHON &nbsp;·&nbsp; v1.0.0 &nbsp;·&nbsp;
  FOR DEMONSTRATION PURPOSES ONLY — NOT LEGAL ADVICE
</div>
""", unsafe_allow_html=True)
