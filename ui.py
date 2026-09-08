from __future__ import annotations
import re
import sys
import time
from pathlib import Path
from typing import Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import api  # noqa: E402  (same-process backend — api.ensure_resources / api.answer)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_NAME = "Verus"
APP_SUBTITLE = "Legal research assistant"
APP_SCOPE = "Pakistan cyber-crime & criminal law"
CSS_PATH = ROOT / "assets" / "styles.css"

# Curated entry points that exercise the statutes + case law + KG paths.
DEFAULT_SUGGESTIONS = [
    "What is the punishment for qatl-i-amd under the Pakistan Penal Code?",
    "Can post-arrest bail be granted in a PECA cyber-crime case?",
    "What due-diligence duties does the AML Act place on banks?",
]

MAX_PROVISIONS_SHOWN = 8
MAX_CASE_LAW_SHOWN = 6

# --------------------------------------------------------------------------- #
# Page + style setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title=f"{APP_NAME} · Legal Research",
    page_icon=":material/gavel:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        css = ""
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


_inject_css()


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "backend" not in st.session_state:
        st.session_state["backend"] = None  # {"ok": bool, "detail": str} once ready


_init_state()


# --------------------------------------------------------------------------- #
# Backend access (warmed once per process)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _warm_backend() -> dict:
    """Load every heavy singleton once and report readiness."""
    try:
        api.ensure_resources()
        return {"ok": True, "detail": ""}
    except Exception as exc:  # noqa: BLE001 — surface as a clean status, not a crash
        print(f"[ui] warm-up failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}


def _run_backend(question: str, history: list) -> dict:
    """Call the RAG pipeline; never raises — returns a result/error payload."""
    t0 = time.perf_counter()
    try:
        result = api.answer(
            question,
            use_llm=True,
            multi_hop=True,
            history=history or None,
        )
        return {"ok": True, "result": result, "latency": time.perf_counter() - t0}
    except Exception as exc:  # noqa: BLE001
        print(f"[ui] backend error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency": time.perf_counter() - t0,
        }


# --------------------------------------------------------------------------- #
# Small presentational blocks
# --------------------------------------------------------------------------- #
def _render_topbar(backend: Optional[dict]) -> None:
    if backend is None:
        label, cls = "Preparing", "prep"
    elif backend.get("ok"):
        label, cls = "Ready", ""
    else:
        label, cls = "Degraded", "down"
    st.markdown(
        '<div class="uia-topbar">'
        '<div class="uia-brandwrap">'
        f'<span class="uia-brand-mark">{APP_NAME[0]}</span>'
        '<div class="uia-brandtxt">'
        f'<span class="uia-brand-name">{APP_NAME}</span>'
        f'<span class="uia-brand-sub">{APP_SUBTITLE} · {APP_SCOPE}</span>'
        "</div>"
        "</div>"
        '<div class="uia-topmeta">'
        f'<span class="uia-status"><span class="uia-dot {cls}"></span>{label}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _engine_status_items():
    """Read-only engine status rows for the sidebar (presentation only).

    Reads the already-warmed api module state; never triggers a load.
    Returns None while the engine is still preparing.
    """
    backend = st.session_state.get("backend")
    if backend is None:
        return None
    try:
        stt = dict(getattr(api, "_status", {}) or {})
        llm_ok = bool(getattr(api, "llm_client", None) and api.llm_client.llm_reachable())
        return [
            ("Retrieval models", stt.get("models") == "up"),
            ("Vector store · Qdrant", stt.get("store") == "up"),
            ("Knowledge graph · Neo4j", stt.get("kg") == "up"),
            ("Language model", llm_ok),
        ]
    except Exception:
        return [("Engine", bool(backend and backend.get("ok")))]


def _render_welcome() -> None:
    st.markdown(
        '<div class="uia-welcome">'
        f'<div class="uia-eyebrow">{APP_NAME} · LEGAL RESEARCH</div>'
        '<div class="uia-welcome-h">How can I help you today?</div>'
        '<div class="uia-welcome-p">Ask a question about Pakistani cyber-crime and '
        "criminal law. Every answer is grounded in the statutes and case law it "
        "cites, so you can verify each point.</div>"
        '<div class="uia-pills">'
        '<span class="uia-pill">Statutes & sections</span>'
        '<span class="uia-pill">Case law & bail</span>'
        '<span class="uia-pill">Cross-act references</span>'
        '<span class="uia-pill">Cited answers</span>'
        "</div>"
        '<div class="uia-examples">'
        '<div class="uia-examples-h">Try an example</div>'
        '<div class="uia-examples-p">Ask one of these to see the research flow.</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        for s in DEFAULT_SUGGESTIONS:
            if st.button(s, type="secondary", use_container_width=True):
                # Defer processing to the next run so the welcome state is
                # replaced by the conversation before the question is asked.
                st.session_state["_pending_query"] = s
                st.rerun()


def _render_unavailable(backend: dict) -> None:
    st.markdown(
        '<div class="uia-welcome"><div class="uia-welcome-h">The engine is '
        "not available</div>"
        '<div class="uia-welcome-p">The research services could not be started. '
        "Check that Qdrant / Neo4j are reachable and the model cache is intact, "
        "then restart the app.</div></div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Startup detail: {backend.get('detail', 'unknown')}")


# --------------------------------------------------------------------------- #
# Chat message helpers
# --------------------------------------------------------------------------- #
_TIMESTAMP_RE = re.compile(r"_\d{6,}$")


def _friendly_doc_name(doc: Optional[str]) -> str:
    """Turn raw doc keys ("Criminal_Misc..._2025_638764333946605900.pdf") into
    readable labels ("Criminal_Misc..._2025")."""
    name = (doc or "").strip()
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = _TIMESTAMP_RE.sub("", name).strip(" _")
    return name or "Document"


def _is_judgment(e: dict) -> bool:
    return (e.get("doc_type") or "").strip().lower() in ("judgment", "judgement")


def _prov_bullet(i: int, e: dict) -> str:
    doc = _friendly_doc_name(e.get("doc"))
    dtype = (e.get("doc_type") or "").strip().lower()
    graph_tag = " · via graph" if e.get("from_graph") else ""

    section = e.get("section_no")
    if section is None:
        return f"{i}. **{doc}** `{dtype or 'provision'}`{graph_tag}"

    title = (e.get("section_title") or "").strip()
    title_part = f", *{title}*" if title else ""
    return f"{i}. **{doc}** — §{section}{title_part} `{dtype or 'provision'}`{graph_tag}"


def _graph_provision_bullet(j: dict) -> Optional[str]:
    """A section reached through the knowledge graph but without full text."""
    act, section = j.get("act"), j.get("section_no")
    if act and section is not None:
        return f"- **{_friendly_doc_name(act)}** — §{section} `via knowledge graph`"
    return None


def _case_law_bullet(j: dict) -> str:
    case = j.get("case_no") or _friendly_doc_name(j.get("act")) or "Judgment"
    bits = [f"**{case}**"]
    court = j.get("court")
    if court:
        bits.append(court)
    line = " · ".join(bits)
    if j.get("outcome"):
        line += f" — {j.get('outcome')}"
    return f"- {line} `case law · graph`"


def _render_sources(evidence: list, case_law: list) -> None:
    statutory = [e for e in evidence if not _is_judgment(e)]
    judgments = [e for e in evidence if _is_judgment(e)]
    graph_only = [
        b for j in case_law if not j.get("case_no")
        for b in [_graph_provision_bullet(j)] if b
    ]
    cases = [j for j in case_law if j.get("case_no")]

    lines: list[str] = []

    provisions = [_prov_bullet(i, e) for i, e in enumerate(statutory[:MAX_PROVISIONS_SHOWN], 1)]
    if provisions or graph_only:
        lines.append("**Provisions**")
        lines.extend(provisions)
        lines.extend(graph_only[:MAX_CASE_LAW_SHOWN])

    case_lines: list[str] = []
    # Judgment chunks retrieved from the vector store.
    for j in judgments[:MAX_CASE_LAW_SHOWN]:
        case_lines.append(f"- **{_friendly_doc_name(j.get('doc'))}** `judgment`")
    # Case law discovered through the knowledge graph (CITES).
    for j in cases[:MAX_CASE_LAW_SHOWN]:
        case_lines.append(_case_law_bullet(j))
    if case_lines:
        if lines:
            lines.append("")
        lines.append("**Case law**")
        lines.extend(case_lines)

    if lines:
        st.markdown("\n".join(lines))


def _caption(res: dict, latency: float) -> str:
    mode = "Deep research" if res.get("mode") == "multi_hop" else "Quick search"
    hops = res.get("hops")
    parts = [mode]
    if hops:
        parts.append(f"{hops} hop{'s' if hops != 1 else ''}")
    parts.append(f"{latency:.1f}s")
    return " · ".join(parts)


def _no_answer_note(llm_used: bool) -> str:
    if llm_used:
        return (
            "I wasn't able to draft a full answer from the material I retrieved. "
            "The sources below are what the engine found — try rephrasing or "
            "asking a more specific question."
        )
    return (
        "I couldn't generate a written answer because the language model is "
        "unreachable right now. Here are the most relevant provisions and case "
        "law retrieved for your question."
    )


def _assistant_message(payload: dict) -> dict:
    base = {
        "role": "assistant",
        "content": None,
        "error": False,
        "sources": [],
        "case_law": [],
        "caption": None,
        "latency": round(payload.get("latency", 0.0), 1),
        "ts": time.strftime("%H:%M"),
    }
    if not payload.get("ok"):
        return {**base, "error": True}

    res = payload["result"]
    answer = (res.get("answer") or "").strip()
    evidence = res.get("evidence") or []
    case_law = res.get("kg_evidence") or []
    llm_used = bool(res.get("llm_used"))
    caption = _caption(res, payload.get("latency", 0.0))

    if answer and not answer.startswith("(LLM synthesis failed"):
        return {
            **base,
            "content": answer,
            "llm_used": llm_used,
            "sources": evidence,
            "case_law": case_law,
            "caption": caption,
        }
    if evidence or case_law:
        return {
            **base,
            "content": _no_answer_note(llm_used),
            "llm_used": llm_used,
            "sources": evidence,
            "case_law": case_law,
            "caption": caption,
        }
    # Nothing usable at all.
    return {**base, "error": True}


def _render_assistant(msg: dict, show_sources: bool) -> None:
    ts = msg.get("ts") or ""
    if not msg.get("error") and (msg.get("content") or msg.get("sources") or msg.get("case_law")):
        st.markdown(
            f'<div class="uia-msg-tag"><span class="uia-msg-tag-dot"></span>'
            f"Verus{(' · ' + ts) if ts else ''}</div>",
            unsafe_allow_html=True,
        )
    if msg.get("content"):
        st.markdown(msg["content"])
    if msg.get("error"):
        st.markdown("Sorry, I couldn't process that request. Please try again.")

    sources, case_law = msg.get("sources") or [], msg.get("case_law") or []
    if show_sources and not msg.get("error") and (sources or case_law):
        with st.expander(f"Sources · {len(sources) + len(case_law)}", expanded=False):
            _render_sources(sources, case_law)
    if msg.get("caption") and not msg.get("error"):
        st.caption(msg["caption"])


# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
def _submit(text: str, show_sources: bool) -> None:
    """Run the backend for an already-appended user turn; render + store reply."""
    messages = st.session_state["messages"]

    # Prior turns (excluding the just-added user message) -> context.
    history = [
        {"role": m["role"], "content": m.get("content") or ""}
        for m in messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            payload = _run_backend(text, history)
        msg = _assistant_message(payload)
        _render_assistant(msg, show_sources)

    messages.append(msg)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="uia-side-brand">'
        f'<span class="uia-brand-mark">{APP_NAME[0]}</span>'
        '<div class="uia-brandtxt">'
        f'<div class="uia-brand-name">{APP_NAME}</div>'
        f'<div class="uia-brand-sub">{APP_SUBTITLE}</div>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="uia-side-label">Session</div>', unsafe_allow_html=True)
    if st.button(
        "New chat",
        type="primary",
        use_container_width=True,
        help="Clear this conversation and start a fresh one.",
    ):
        st.session_state["messages"] = []
        st.session_state.pop("_pending_query", None)
        st.rerun()

    _msgs = st.session_state.get("messages", [])
    _turns = len([m for m in _msgs if m.get("role") == "user"])
    if _msgs:
        st.caption(f"{_turns} question{'s' if _turns != 1 else ''} in this session")
    else:
        st.caption("Start a new research session")

    st.divider()

    st.markdown('<div class="uia-side-label">Citations</div>', unsafe_allow_html=True)
    st.checkbox(
        "Show sources",
        value=True,
        key="show_sources",
        help="List the provisions and case law behind each answer.",
    )
    st.caption("Open any answer's sources to verify every cited provision.")

    st.divider()

    st.markdown('<div class="uia-side-label">Engine</div>', unsafe_allow_html=True)
    _eng = _engine_status_items()
    if _eng is None:
        st.caption("Preparing the research engine…")
    else:
        for _label, _ok in _eng:
            _dot_cls = "" if _ok else "off"
            st.markdown(
                f'<div class="uia-eng-row"><span class="uia-eng-dot {_dot_cls}"></span>'
                f"<span>{_label}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.caption("Advisory research only — not legal advice.")

# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
backend = st.session_state["backend"]

# First visit for this session: warm the engine behind a clean splash.
if backend is None:
    _render_topbar(None)
    st.markdown(
        '<div class="uia-splash"><div class="uia-welcome-h">Welcome to '
        f"{APP_NAME}</div>"
        '<div class="uia-welcome-p">Preparing the legal research engine… '
        "the first load can take a minute or two.</div></div>",
        unsafe_allow_html=True,
    )
    with st.spinner("Loading models and connecting to the knowledge graph…"):
        backend = _warm_backend()
    st.session_state["backend"] = backend
    st.rerun()

_render_topbar(backend)

if not backend.get("ok"):
    _render_unavailable(backend)
    st.stop()

messages = st.session_state["messages"]
show_sources = bool(st.session_state.get("show_sources", True))

# Persistent input — declared before rendering so we know, in this same run,
# whether a new question is incoming (Streamlit pins it at the bottom anyway).
prompt = st.chat_input("Ask a legal question…", key="chat_input")
suggested = st.session_state.pop("_pending_query", None)
incoming = (prompt or suggested or "").strip()

# Append the user turn up front so the conversation (not the welcome state)
# is what gets rendered while the assistant is still working on the answer.
if incoming:
    messages.append({"role": "user", "content": incoming, "ts": time.strftime("%H:%M")})

if messages:
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant(msg, show_sources)
else:
    _render_welcome()

if incoming:
    _submit(incoming, show_sources)
