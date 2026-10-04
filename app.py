# app.py
import asyncio
import traceback
from pathlib import Path

import streamlit as st
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Market Research & Blog Writer", layout="wide")

APP_NAME = "Crew AI"
BLOG_PATH = Path("blog-posts") / "new_post.md"

PROVIDERS = {
    "Google Gemini": {
        "prefix": "gemini",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "state_key": "gemini_api_key",
        "input_key": "gemini_key_input",
        "label": "Google Gemini API Key",
        "placeholder": "Enter Google Gemini API Key",
    },
    "OpenAI": {
        "prefix": "openai",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "state_key": "openai_api_key",
        "input_key": "openai_key_input",
        "label": "OpenAI API Key",
        "placeholder": "Enter OpenAI API Key",
    },
}

CSS = """
<style>
:root {
    --bg-main: #F1F5F9;
    --bg-sidebar: #FFFFFF;
    --bg-card: #FFFFFF;
    --bg-results: #F0F4F8;
    --primary: #4F46E5;
    --primary-hover: #4338CA;
    --ai-accent: #8B5CF6;
    --brown: #8B5E3C;
    --brown-hover: #7A5034;
    --text-dark: #111827;
    --text-secondary: #475569;
    --text-muted: #94A3B8;
    --border: #CBD5E1;
    --border-light: #E2E8F0;
    --success: #16A34A;
    --danger: #DC2626;
    --danger-hover: #B91C1C;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --radius-sm: 6px;
    --radius: 8px;
    --radius-lg: 12px;
    --transition: 150ms ease;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-dark);
}

.stApp, h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText,
div[data-testid="stMarkdownContainer"] {
    color: var(--text-dark) !important;
}

h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

h2 {
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: -0.01em;
}

h3 {
    font-size: 1.125rem;
    font-weight: 600;
}

h4 {
    font-size: 1rem;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-light);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-dark) !important;
}

.sidebar-brand {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}

.sidebar-subtitle {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin-bottom: 1.5rem;
}

.sidebar-section {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
    color: var(--text-dark) !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: var(--bg-main);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 0.625rem 1rem;
    margin-bottom: 0.5rem;
    transition: all var(--transition);
    cursor: pointer;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    border-color: var(--primary);
    background: #EEF2FF;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div {
    background: #EEF2FF;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] input:checked + div span {
    color: var(--primary) !important;
}

[data-testid="stSidebar"] .stSelectbox label {
    font-weight: 500;
    color: var(--text-secondary) !important;
    font-size: 0.875rem;
}

.api-key-card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    margin-top: 1rem;
    box-shadow: var(--shadow-sm);
}

.api-key-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.api-key-card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-dark);
}

.api-key-status {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.625rem;
    border-radius: 999px;
}

.api-key-status.configured {
    background: #ECFDF5;
    color: var(--success);
}

.api-key-status.not-configured {
    background: #FEF2F2;
    color: var(--danger);
}

.api-key-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

.api-key-display {
    background: var(--bg-main);
    border: 1px solid var(--border-light);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.8125rem;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    word-break: break-all;
}

.stButton > button[kind="primary"],
.stButton > button:not([kind]):not([data-testid="baseButton-secondary"]),
.stDownloadButton > button {
    background-color: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all var(--transition) !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button[kind="primary"]:hover,
.stButton > button:not([kind]):not([data-testid="baseButton-secondary"]):hover,
.stDownloadButton > button:hover {
    background-color: var(--primary-hover) !important;
    color: #FFFFFF !important;
    box-shadow: var(--shadow) !important;
    transform: translateY(-1px);
}

.stButton > button[kind="primary"]:active,
.stButton > button:not([kind]):not([data-testid="baseButton-secondary"]):active,
.stDownloadButton > button:active {
    transform: translateY(0);
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button:focus,
.stButton > button:focus-visible,
.stDownloadButton > button:focus,
.stDownloadButton > button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.3) !important;
}

.stButton > button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.625rem 1.25rem !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: all var(--transition) !important;
}

.stButton > button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {
    background-color: var(--bg-main) !important;
    color: var(--text-dark) !important;
    border-color: var(--border) !important;
}

div[data-testid="stSidebar"] .stButton > button:has(span:contains("Save")),
div[data-testid="stSidebar"] .stButton > button:has(span:contains("Add")) {
    background-color: var(--primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    width: 100%;
    transition: all var(--transition) !important;
}

div[data-testid="stSidebar"] .stButton > button:has(span:contains("Save")):hover,
div[data-testid="stSidebar"] .stButton > button:has(span:contains("Add")):hover {
    background-color: var(--primary-hover) !important;
    box-shadow: var(--shadow) !important;
    transform: translateY(-1px);
}

.stTextInput > div > div > input,
.stTextArea > div > textarea,
.stSelectbox > div > div[data-baseweb="select"],
.stNumberInput > div > div > input {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-dark) !important;
    font-size: 0.875rem !important;
    transition: all var(--transition) !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus,
.stSelectbox > div > div[data-baseweb="select"]:focus-within,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important;
    outline: none !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > textarea::placeholder {
    color: var(--text-muted) !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stNumberInput label {
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
}

.stTextArea > div > textarea {
    min-height: 120px !important;
    resize: vertical !important;
    padding: 0.75rem 1rem !important;
    line-height: 1.6 !important;
}

.stSelectbox > div > div[data-baseweb="select"] > div {
    background-color: var(--bg-card) !important;
}

.stSelectbox [data-baseweb="popover"] ul li {
    color: var(--text-dark) !important;
}

.stSelectbox [data-baseweb="popover"] ul li:hover,
.stSelectbox [data-baseweb="popover"] ul li[aria-selected="true"] {
    background-color: #EEF2FF !important;
    color: var(--primary) !important;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow var(--transition), transform var(--transition);
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.card h4 {
    margin: 0 0 0.5rem 0;
    color: var(--text-dark);
}

.card p {
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.6;
}

div[data-testid="stContainer"][border="true"],
div.stContainer > div[style*="border"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--bg-card) !important;
    box-shadow: var(--shadow-sm) !important;
}

.badge {
    display: inline-flex;
    align-items: center;
    background: #EEF2FF;
    color: var(--primary);
    border: 1px solid var(--border-light);
    border-radius: 999px;
    padding: 0.25rem 0.875rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.badge-ai {
    background: #F5F3FF;
    color: var(--ai-accent);
    border-color: var(--border-light);
}

.workflow-container {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding: 0.5rem 0;
}

.workflow-step {
    flex: 0 0 auto;
    min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    text-align: center;
    color: var(--text-dark);
    font-weight: 600;
    font-size: 0.875rem;
    box-shadow: var(--shadow-sm);
    transition: all var(--transition);
    position: relative;
}

.workflow-step:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
}

.workflow-step.step-1 {
    border-left: 4px solid var(--primary);
    background: #EEF2FF;
}

.workflow-step.step-2 {
    border-left: 4px solid var(--ai-accent);
    background: #F5F3FF;
}

.workflow-step.step-3 {
    border-left: 4px solid #3B82F6;
    background: #EFF6FF;
}

.workflow-step.step-4 {
    border-left: 4px solid #A855F7;
    background: #FAF5FF;
}

.workflow-step.step-5 {
    border-left: 4px solid var(--primary);
    background: #EEF2FF;
}

.workflow-step.step-6 {
    border-left: 4px solid var(--success);
    background: #ECFDF5;
}

.workflow-arrow {
    flex: 0 0 auto;
    color: var(--text-muted);
    font-size: 1.25rem;
    line-height: 1;
    padding: 0 0.25rem;
}

.status-ok {
    color: var(--success);
    font-weight: 600;
}

.status-no {
    color: var(--danger);
    font-weight: 600;
}

.stAlert {
    border-radius: var(--radius) !important;
    border: none !important;
    box-shadow: var(--shadow) !important;
}

.stAlert[data-testid="stSuccess"] {
    background: #ECFDF5 !important;
    color: var(--success) !important;
}

.stAlert[data-testid="stError"] {
    background: #FEF2F2 !important;
    color: var(--danger) !important;
}

.stAlert[data-testid="stWarning"] {
    background: #FFFAEB !important;
    color: #B45309 !important;
}

.stAlert[data-testid="stInfo"] {
    background: #EEF2FF !important;
    color: var(--primary) !important;
}

.stStatusWidget {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow) !important;
}

.stStatusWidget [data-testid="stStatusWidgetContent"] {
    color: var(--text-dark) !important;
}

hr,
.stDivider {
    border-color: var(--border-light) !important;
    margin: 1.5rem 0 !important;
}

.stCaption,
.caption,
small {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
}

.subtitle {
    color: var(--text-secondary) !important;
    font-size: 1.125rem !important;
    margin-top: -0.5rem !important;
    font-weight: 400;
    line-height: 1.5;
}

.streamlit-expanderHeader {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-dark) !important;
    font-weight: 500 !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--primary) !important;
}

.streamlit-expanderContent {
    border: 1px solid var(--border-light) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    background: var(--bg-main) !important;
}

code,
.stCodeBlock,
pre {
    background-color: var(--bg-main) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius) !important;
    color: var(--text-dark) !important;
}

.stDownloadButton > button {
    background-color: #8B4513 !important;
    color: #FFFFFF !important;
}

.stDownloadButton > button:hover {
    background-color: #FF6B00 !important;
    opacity: 1 !important;
}

::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted);
}

@media (max-width: 768px) {
    .workflow-step {
        min-width: 140px;
        padding: 0.75rem 1rem;
        font-size: 0.8125rem;
    }
}

.stApp div[class*="st-key-del_"] .stButton > button,
.stApp div[class*="st-key-save_"] .stButton > button,
.stApp div[class*="st-key-start_research"] .stButton > button {
    background-color: #8B5E3C !important;
    border: none !important;
    color: #FFFFFF !important;
}

.stApp div[class*="st-key-del_"] .stButton > button:hover,
.stApp div[class*="st-key-save_"] .stButton > button:hover,
.stApp div[class*="st-key-start_research"] .stButton > button:hover {
    background-color: #7A5034 !important;
    color: #FFFFFF !important;
}

.stApp div[class*="st-key-del_"] .stButton > button p,
.stApp div[class*="st-key-del_"] .stButton > button span,
.stApp div[class*="st-key-save_"] .stButton > button p,
.stApp div[class*="st-key-save_"] .stButton > button span,
.stApp div[class*="st-key-start_research"] .stButton > button p,
.stApp div[class*="st-key-start_research"] .stButton > button span {
    color: #FFFFFF !important;
}

.stApp div[class*="st-key-research_summary_box"],
.stApp div[class*="st-key-blog_post_box"] {
    background-color: #FBFBF9 !important;
}
</style>
"""


# ----------------------------------------------------------------------------
# Session state & helpers
# ----------------------------------------------------------------------------
def initialize_session_state():
    defaults = {
        "gemini_api_key": "",
        "openai_api_key": "",
        "selected_provider": "Google Gemini",
        "selected_model": PROVIDERS["Google Gemini"]["models"][0],
        "research_result": "",
        "blog_result": "",
        "save_error": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def mask_api_key(key: str) -> str:
    if not key:
        return ""

    prefix = "sk-" if key.startswith("sk-") else key[:4]
    suffix = key[-4:] if len(key) > 8 else ""

    return f"{prefix}{'*' * 12}{suffix}"


def save_api_key(provider: str):
    cfg = PROVIDERS[provider]
    value = st.session_state.get(cfg["input_key"], "").strip()

    if not value:
        st.session_state["key_msg"] = (
            "warning",
            "Please enter an API key first.",
        )
        return

    st.session_state[cfg["state_key"]] = value
    st.session_state[cfg["input_key"]] = ""
    st.session_state["key_msg"] = (
        "success",
        f"{provider} API key saved.",
    )


def delete_api_key(provider: str):
    cfg = PROVIDERS[provider]

    st.session_state[cfg["state_key"]] = ""
    st.session_state[cfg["input_key"]] = ""

    st.session_state["key_msg"] = (
        "info",
        f"{provider} API key deleted.",
    )


def current_api_key() -> str:
    return st.session_state.get(
        PROVIDERS[st.session_state.selected_provider]["state_key"],
        "",
    )


# ----------------------------------------------------------------------------
# UI sections
# ----------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"<div class='sidebar-brand'>{APP_NAME}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='sidebar-subtitle'>Multi-Agent AI Platform</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='sidebar-section'>Provider</div>",
            unsafe_allow_html=True,
        )

        provider = st.radio(
            "Provider",
            list(PROVIDERS.keys()),
            key="selected_provider",
            label_visibility="collapsed",
        )

        cfg = PROVIDERS[provider]

        st.markdown(
            "<div class='sidebar-section'>Model</div>",
            unsafe_allow_html=True,
        )

        model = st.selectbox(
            "Model",
            cfg["models"],
            key=f"model_select_{provider}",
            label_visibility="collapsed",
        )

        st.session_state.selected_model = model

        st.markdown(
            "<div class='sidebar-section'>API Key</div>",
            unsafe_allow_html=True,
        )

        saved_key = st.session_state[cfg["state_key"]]

        if saved_key:
            st.markdown(
                f"""
                <div class="api-key-card">
                    <div class="api-key-card-header">
                        <span class="api-key-card-title">{cfg['label']}</span>
                        <span class="api-key-status configured">
                            <span class="api-key-status-dot"></span>Configured
                        </span>
                    </div>
                    <div class="api-key-display">{mask_api_key(saved_key)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.button(
                "Delete API Key",
                key=f"del_{provider}",
                on_click=delete_api_key,
                args=(provider,),
            )

        else:
            st.text_input(
                cfg["placeholder"],
                type="password",
                key=cfg["input_key"],
                placeholder=cfg["placeholder"],
                label_visibility="collapsed",
            )

            st.markdown(
                """
                <div class="api-key-card" style="margin-top: 0.5rem;">
                    <span class="api-key-status not-configured">
                        <span class="api-key-status-dot"></span>API Key not configured
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.button(
                "Add / Save API Key",
                key=f"save_{provider}",
                on_click=save_api_key,
                args=(provider,),
            )

        msg = st.session_state.pop("key_msg", None)

        if msg:
            getattr(st, msg[0])(msg[1])

        st.divider()

        st.markdown(f"**Current Provider:** {provider}")
        st.markdown(f"**Current Model:** {model}")

        st.caption(
            "Keys live only in this browser session's memory and are never written to disk."
        )


def render_header():
    st.markdown(
        f"<span class='badge'>{APP_NAME}</span>",
        unsafe_allow_html=True,
    )

    st.title("AI Market Research & Blog Writer")

    st.markdown(
        "<p class='subtitle'>Multi-Agent AI Research & Content Generation Platform</p>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="card">
            <h4>About {APP_NAME}</h4>
            <p>
            {APP_NAME} is a multi-agent AI application built with CrewAI. Enter any AI-related topic and
            a <b>Market Research Analyst</b> agent searches the web for the latest developments, then a
            <b>Content Writer</b> agent turns that research into a polished 4-paragraph blog post in
            Markdown. Choose Google Gemini or OpenAI as the engine, use your own API key, and download
            the finished post as <code>new_post.md</code>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow():
    st.markdown("### Multi-Agent Workflow")

    steps = [
        "User Query",
        "Market Research Analyst",
        "Web Search",
        "AI Trend Analysis",
        "Content Writer",
        "Final Blog Post",
    ]

    step_html = '<div class="workflow-container">'

    for i, step in enumerate(steps):
        step_num = i + 1

        step_html += (
            f'<div class="workflow-step step-{step_num}">{step}</div>'
        )

        if i < len(steps) - 1:
            step_html += '<div class="workflow-arrow">→</div>'

    step_html += "</div>"

    st.markdown(step_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# CrewAI building blocks
# ----------------------------------------------------------------------------
def get_selected_llm() -> LLM:
    provider = st.session_state.selected_provider
    cfg = PROVIDERS[provider]
    model = st.session_state.selected_model
    api_key = st.session_state[cfg["state_key"]]

    if provider == "Google Gemini":
        return LLM(
            model=f"gemini/{model}",
            api_key=api_key,
        )

    if provider == "OpenAI":
        return LLM(
            model=f"openai/{model}",
            api_key=api_key,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def create_web_search_tool():
    """
    Direct DuckDuckGo/DDGS web search.

    Unlike the previous implementation, search errors are raised instead
    of being returned as normal text. This prevents CrewAI from treating
    a search error as valid research content.
    """

    @tool("Web Search Tool")
    def web_search_tool(query: str) -> str:
        """Search the web for the latest AI industry news and information."""

        try:
            results = DDGS().text(
                query,
                max_results=5,
            )

            if not results:
                raise RuntimeError(
                    "No web search results were returned."
                )

            formatted_results = []

            for result in results:
                title = result.get("title", "")
                body = result.get("body", "")
                href = result.get("href", "")

                formatted_results.append(
                    f"Title: {title}\n"
                    f"Summary: {body}\n"
                    f"URL: {href}"
                )

            return "\n\n".join(formatted_results)

        except Exception as exc:
            raise RuntimeError(
                f"Web search is currently unavailable: {exc}"
            ) from exc

    return web_search_tool


def create_agents(llm: LLM, search_tool):
    researcher = Agent(
        role="Market Research Analyst",
        goal="Provide up-to-date market analysis of the AI industry",
        backstory="An expert analyst with a keen eye for market trends.",
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    writer = Agent(
        role="Content Writer",
        goal="Craft engaging blog posts about the AI industry",
        backstory="A skilled writer with a passion for technology.",
        tools=[],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    return researcher, writer


def create_tasks(researcher: Agent, writer: Agent, query: str):
    research_task = Task(
        description=(
            "Search the web for the latest AI trends and provide a summarized report. "
            f"Focus on this topic: {query}"
        ),
        expected_output=(
            "A summary of the top 3 trending developments in AI "
            "with insights on their impact."
        ),
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Write an engaging blog post about the AI industry "
            "based on the research analyst's summary."
        ),
        expected_output=(
            "A well-structured, 4-paragraph blog post in markdown "
            "format with simple, engaging content."
        ),
        agent=writer,
        context=[research_task],
    )

    return research_task, write_task


def create_crew(agents, tasks) -> Crew:
    return Crew(
        agents=list(agents),
        tasks=list(tasks),
        process=Process.sequential,
        verbose=False,
        planning=False,
    )


def run_crew(query: str):
    """
    Build everything from the sidebar selection and run the crew.

    Search errors and LLM errors are allowed to propagate to the main
    error handler instead of being converted into fake research output.
    """

    llm = get_selected_llm()

    search_tool = create_web_search_tool()

    researcher, writer = create_agents(
        llm,
        search_tool,
    )

    research_task, write_task = create_tasks(
        researcher,
        writer,
        query,
    )

    crew = create_crew(
        (researcher, writer),
        (research_task, write_task),
    )

    result = asyncio.run(
        crew.kickoff_async()
    )

    research_text = (
        research_task.output.raw
        if research_task.output
        else ""
    )

    blog_text = (
        getattr(result, "raw", None)
        or str(result)
    )

    return research_text, blog_text


def save_blog(content: str) -> Path:
    BLOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned = content.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    BLOG_PATH.write_text(
        cleaned,
        encoding="utf-8",
    )

    return BLOG_PATH


def render_results():
    if (
        not st.session_state.research_result
        and not st.session_state.blog_result
    ):
        return

    st.markdown("### Research Summary")

    with st.container(
        border=True,
        key="research_summary_box",
    ):
        st.markdown(
            st.session_state.research_result
            or "_No research output._"
        )

    st.markdown("### AI Generated Blog Post")

    with st.container(
        border=True,
        key="blog_post_box",
    ):
        st.markdown(
            st.session_state.blog_result
            or "_No blog output._"
        )

    if st.session_state.save_error:
        st.warning(
            "The blog could not be saved to disk, but you can still download it below."
        )

        with st.expander("Technical details"):
            st.code(
                st.session_state.save_error
            )

    else:
        st.caption(
            f"Saved to `{BLOG_PATH.as_posix()}`"
        )

    st.download_button(
        "Download Markdown",
        data=st.session_state.blog_result,
        file_name="new_post.md",
        mime="text/markdown",
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    initialize_session_state()

    st.markdown(
        CSS,
        unsafe_allow_html=True,
    )

    render_sidebar()
    render_header()
    render_workflow()

    st.markdown(
        "### What would you like to research?"
    )

    with st.container(border=True):
        query = st.text_area(
            "Research topic",
            placeholder="Example: Latest trends in Generative AI",
            height=130,
            label_visibility="collapsed",
        )

        start = st.button(
            "Start AI Research",
            key="start_research",
        )

    if start:
        provider = st.session_state.selected_provider
        model = st.session_state.selected_model

        if not provider or not model:
            st.error(
                "Please select a provider and a model."
            )

        elif not current_api_key():
            st.error(
                "Please add your API key from the sidebar before starting the research."
            )

        elif not query.strip():
            st.warning(
                "Please enter a research topic."
            )

        else:
            st.session_state.research_result = ""
            st.session_state.blog_result = ""
            st.session_state.save_error = ""

            try:
                with st.status(
                    "AI agents are working...",
                    expanded=True,
                ) as status:

                    st.write(
                        "**Market Research Analyst** — Status: Searching the web..."
                    )

                    st.write(
                        "**Content Writer** — Status: Waiting for research..."
                    )

                    research_text, blog_text = run_crew(
                        query.strip()
                    )

                    if not research_text.strip():
                        raise RuntimeError(
                            "Research completed without producing a valid research summary."
                        )

                    if not blog_text.strip():
                        raise RuntimeError(
                            "Blog generation completed without producing a valid blog post."
                        )

                    st.write(
                        "Research completed."
                    )

                    st.write(
                        "Blog generation completed."
                    )

                    try:
                        save_blog(blog_text)

                    except Exception:
                        st.session_state.save_error = (
                            traceback.format_exc()
                        )

                    status.update(
                        label="All agents finished",
                        state="complete",
                        expanded=False,
                    )

                st.session_state.research_result = research_text
                st.session_state.blog_result = blog_text

            except Exception as exc:
                error_text = str(exc).lower()

                if (
                    "429" in error_text
                    or "resource_exhausted" in error_text
                    or "quota" in error_text
                ):
                    st.error(
                        "Gemini API rate limit or quota has been reached. "
                        "Please wait and try again, or switch to OpenAI from the sidebar."
                    )

                elif (
                    "web search" in error_text
                    or "dns" in error_text
                    or "connection" in error_text
                    or "network" in error_text
                ):
                    st.error(
                        "Web search is currently unavailable due to a network or DNS issue. "
                        "Please try again later."
                    )

                else:
                    st.error(
                        "Something went wrong while running the AI agents."
                    )

                with st.expander("Technical details"):
                    st.code(
                        traceback.format_exc()
                    )

    render_results()


main()
