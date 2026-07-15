import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import io
from pathlib import Path

from data_loader import load_data, detect_columns, get_primary_cols, build_data_summary
from analysis import (
    compute_time_trend, compute_distribution, compute_top_n,
    compute_statistics, compute_forecast, build_analysis_summary,
)
from charts import (
    plot_time_trend, plot_count_bar, plot_distribution,
    plot_numeric_distributions, plot_top_n, plot_correlation,
    plot_boxplots, plot_forecast, build_custom_chart, CHART_TYPES,
)
from ai_engine import (
    stream_ai, generate_auto_insights, parse_chart_request,
    GROQ_MODELS, OPENROUTER_MODELS,
)
from cleaning import (
    get_missing_report,
    fill_missing,
    remove_outliers,
    remove_duplicates,
    standardize_text,
    convert_column_type,
    get_cleaning_summary,
)

# ── v2: RAG Deep Chat + Visual Explorer ──
try:
    from rag_chat import df_to_chunks, build_faiss_index, rag_answer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Assyrian-AI · Data Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #0F172A; color: #F1F5F9; }
    section[data-testid="stSidebar"] { background: #1E293B; }
    [data-testid="metric-container"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 16px !important;
    }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.5rem !important; }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 0.8rem !important; }
    .chat-user {
        background: #1E40AF; border-radius: 10px 10px 2px 10px;
        padding: 10px 14px; margin: 6px 0; color: #F1F5F9;
        max-width: 80%; margin-left: auto;
    }
    .chat-ai {
        background: #1E293B; border-radius: 10px 10px 10px 2px;
        padding: 10px 14px; margin: 6px 0; color: #F1F5F9;
        max-width: 92%; border: 1px solid #334155; line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { background: #1E293B; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; }
    .stTabs [aria-selected="true"] { color: #F1F5F9 !important; }
    .insight-box {
        background: #1E293B; border-left: 3px solid #2563EB;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0;
        color: #F1F5F9; font-size: 0.92rem; line-height: 1.7;
        white-space: pre-wrap;
    }
    .brand-header {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0 6px 0;
    }
    .brand-name {
        font-size: 1.15rem; font-weight: 700;
        color: #F1F5F9; letter-spacing: 0.3px;
    }
    .brand-sub {
        font-size: 0.75rem; color: #64748B;
    }
    .provider-badge {
        display: inline-block; background: #1E293B;
        border: 1px solid #334155; border-radius: 20px;
        padding: 2px 10px; font-size: 0.75rem; color: #94A3B8;
        margin-left: 8px;
    }
    [data-testid="stFileUploader"] {
        background: #1E293B; border: 1px dashed #334155;
        border-radius: 10px; padding: 20px;
    }
    .stButton > button {
        background: #2563EB; color: white;
        border: none; border-radius: 6px; font-weight: 600;
    }
    .stButton > button:hover { background: #1D4ED8; }
    #MainMenu, header, footer { visibility: hidden; }
    .rag-source {
        background: #0F172A; border: 1px solid #334155;
        border-radius: 6px; padding: 10px 14px; margin: 4px 0;
        font-size: 0.78rem; color: #64748B; font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in {
    "df": None, "col_info": {}, "primary": {},
    "data_summary": "", "analysis_summary": "",
    "chat_history": [], "display_history": [],
    "auto_insights": "", "api_key_set": False,
    "custom_figs": {}, "freq": "M",
    "provider": "groq", "model": None,
    "original_df": None, "cleaning_log": [],
    # v2 RAG state
    "rag_index": None, "rag_chunks": None, "rag_model": None,
    "rag_history": [], "rag_indexed": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _download_btn(fig, filename, key):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#0F172A")
    buf.seek(0)
    st.download_button("⬇️ Download PNG", data=buf,
                       file_name=f"{filename}.png", mime="image/png",
                       key=f"dl_{key}")


def chart_builder(df, tab_key, default_x=None, default_y=None):
    st.markdown("---")
    st.markdown("### 🛠️ Custom Chart Builder")

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in all_cols if c not in num_cols]

    b_tab, ai_tab = st.tabs(["🎛️ Dropdowns", "💬 Describe it"])

    with b_tab:
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_type = st.selectbox("Chart type", CHART_TYPES, key=f"ct_{tab_key}")
        with c2:
            x_opts = ["(none)"] + all_cols
            x_idx = x_opts.index(default_x) if default_x in x_opts else 0
            x_col = st.selectbox("X axis / Group by", x_opts, index=x_idx, key=f"xc_{tab_key}")
            x_col = None if x_col == "(none)" else x_col
        with c3:
            y_opts = ["(none)"] + num_cols
            y_idx = y_opts.index(default_y) if default_y in y_opts else 0
            y_col = st.selectbox("Y axis / Value", y_opts, index=y_idx, key=f"yc_{tab_key}")
            y_col = None if y_col == "(none)" else y_col

        c4, c5, c6 = st.columns(3)
        with c4:
            color_col = st.selectbox("Color by", ["(none)"] + cat_cols, key=f"cc_{tab_key}")
            color_col = None if color_col == "(none)" else color_col
        with c5:
            top_n = st.slider("Top N", 5, 50, 15, key=f"tn_{tab_key}")
        with c6:
            chart_title = st.text_input("Title (optional)", key=f"ti_{tab_key}")

        if st.button("📊 Generate", key=f"gen_{tab_key}", use_container_width=True):
            fig = build_custom_chart(df, chart_type, x_col, y_col, color_col, top_n, chart_title)
            if fig:
                st.session_state.custom_figs[tab_key] = fig

        if tab_key in st.session_state.custom_figs:
            st.pyplot(st.session_state.custom_figs[tab_key])
            _download_btn(st.session_state.custom_figs[tab_key], f"chart_{tab_key}", tab_key)

    with ai_tab:
        st.caption("Describe the chart in plain English — no API key needed.")
        prompt = st.text_input("prompt", label_visibility="collapsed",
                               placeholder='e.g. "bar chart of salary by department, top 10"',
                               key=f"aip_{tab_key}")
        if st.button("🤖 Build from description", key=f"aib_{tab_key}", use_container_width=True):
            if prompt.strip():
                ct, xc, yc, tn = parse_chart_request(prompt, df)
                st.caption(f"→ **{ct}** · X=`{xc}` · Y=`{yc}` · Top {tn}")
                fig = build_custom_chart(df, ct, xc, yc, top_n=tn, title=prompt)
                ai_key = f"{tab_key}_ai"
                if fig:
                    st.session_state.custom_figs[ai_key] = fig

                if ai_key in st.session_state.custom_figs:
                    st.pyplot(st.session_state.custom_figs[ai_key])
                    _download_btn(st.session_state.custom_figs[ai_key], f"chart_{ai_key}", ai_key)
            else:
                st.warning("Type a description first.")


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Logo + branding ──
    logo_path = Path("logo.jpeg")
    if not logo_path.exists():
        logo_path = Path("logo.jpg")
    if not logo_path.exists():
        logo_path = Path("logo.png")

    if logo_path.exists():
        col_logo, col_brand = st.columns([1, 2.5])
        with col_logo:
            st.image(str(logo_path), width=60)
        with col_brand:
            st.markdown("""
            <div class="brand-name">Assyrian-AI</div>
            <div class="brand-sub">AI Data Analyst</div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:1.2rem; font-weight:700; color:#F1F5F9;">🧠 Assyrian-AI</div>
        <div style="font-size:0.75rem; color:#64748B; margin-bottom:4px;">AI Data Analyst</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── AI Provider ──
    st.markdown("### 🤖 AI Provider")
    provider = st.radio(
        "Provider", ["Groq (Free)", "OpenRouter (Free)"],
        horizontal=True, label_visibility="collapsed",
    )
    provider_key = "groq" if "Groq" in provider else "openrouter"
    st.session_state.provider = provider_key

    if provider_key == "groq":
        model_list = GROQ_MODELS
        key_label = "Groq API Key"
        key_help = "Free at console.groq.com → No credit card needed"
        key_env = "GROQ_API_KEY"
        key_link = "https://console.groq.com"
    else:
        model_list = OPENROUTER_MODELS
        key_label = "OpenRouter API Key"
        key_help = "Free at openrouter.ai → Free models available"
        key_env = "OPENROUTER_API_KEY"
        key_link = "https://openrouter.ai/keys"

    selected_model = st.selectbox("Model", model_list, key="model_select")
    st.session_state.model = selected_model

    api_key_input = st.text_input(
        key_label, type="password",
        placeholder="Paste your key here",
        help=key_help,
    )
    if api_key_input:
        os.environ[key_env] = api_key_input
        st.session_state.api_key_set = True
        st.success("Key saved ✓")

    st.markdown(f"<a href='{key_link}' target='_blank' style='font-size:0.78rem; color:#2563EB;'>🔗 Get a free key →</a>",
               unsafe_allow_html=True)

    st.markdown("---")

    # ── File upload ──
    st.markdown("### 📁 Upload Data")
    uploaded = st.file_uploader("CSV, Excel or JSON",
                                type=["csv", "xlsx", "xls", "json"])

    if uploaded:
        with st.spinner("Loading..."):
            df = load_data(uploaded)
            col_info = detect_columns(df)
            primary = get_primary_cols(col_info)
            data_sum = build_data_summary(df, col_info)
            anal_sum = build_analysis_summary(df, col_info)

            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.cleaning_log = []
            st.session_state.col_info = col_info
            st.session_state.primary = primary
            st.session_state.data_summary = data_sum
            st.session_state.analysis_summary = anal_sum
            st.session_state.chat_history = []
            st.session_state.display_history = []
            st.session_state.auto_insights = ""
            st.session_state.custom_figs = {}
            # Reset RAG state on new upload
            st.session_state.rag_index = None
            st.session_state.rag_chunks = None
            st.session_state.rag_model = None
            st.session_state.rag_indexed = False
            st.session_state.rag_history = []

        st.success(f"✓ {len(df):,} rows × {len(df.columns)} cols")

        n_changes = len(st.session_state.cleaning_log)
        _buf = io.StringIO()
        st.session_state.df.to_csv(_buf, index=False)
        st.download_button(
            f"⬇️ Download data{' (cleaned)' if n_changes else ''} (CSV)",
            data=_buf.getvalue(),
            file_name="dataset_cleaned.csv" if n_changes else "dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("🔍 Detected column roles"):
            for role in ("date", "value", "category", "id"):
                cols = col_info.get(role, [])
                if cols:
                    st.markdown(f"**{role.title()}:** {', '.join(cols[:4])}")

        with st.expander("⚙️ Override column roles"):
            all_c = ["(auto)"] + df.columns.tolist()
            overrides = {}
            for role, label in [("date","Date/Time"), ("value","Primary Value"),
                                ("category","Category/Group"), ("id","ID column")]:
                current = primary.get(role)
                idx = all_c.index(current) if current in all_c else 0
                chosen = st.selectbox(label, all_c, index=idx, key=f"ov_{role}")
                if chosen != "(auto)":
                    overrides[role] = chosen
            if st.button("Apply overrides") and overrides:
                primary.update(overrides)
                st.session_state.primary = primary
                st.session_state.analysis_summary = build_analysis_summary(df, col_info)
                st.success("Applied ✓")

    st.markdown("---")
    st.markdown("### ⏱ Time grouping")
    freq = st.radio("Group by", ["D","W","M","Q","Y"], index=2, horizontal=True,
                    format_func=lambda x: {"D":"Day","W":"Week","M":"Month","Q":"Quarter","Y":"Year"}[x])
    st.session_state.freq = freq

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.8;'>
    Built by <b style='color:#94A3B8;'>Assyrian-AI</b><br>
    github.com/Assyrian-AI<br>
    Powered by Groq & OpenRouter<br>
    v2.0 — RAG Deep Chat + Visual Explorer
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────
if st.session_state.df is None:
    logo_path = Path("logo.jpeg")
    if not logo_path.exists(): logo_path = Path("logo.jpg")
    if not logo_path.exists(): logo_path = Path("logo.png")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if logo_path.exists():
            st.image(str(logo_path), width=120)

    st.markdown("""
    <div style='text-align:center; padding: 10px 20px 60px 20px;'>
        <h1 style='color:#F1F5F9; font-size:2.2rem; margin-bottom:6px;'>Assyrian-AI Data Analyst v2</h1>
        <p style='color:#94A3B8; font-size:1.05rem; max-width:560px; margin:0 auto 10px;'>
            Upload <em>any</em> CSV, Excel, or JSON — sales, HR, finance, healthcare,
            logistics, surveys, sports, or anything else.<br>
            Get instant charts, statistics, AI-powered insights, RAG deep chat over your actual rows, and a drag-and-drop visual explorer.
        </p>
        <div style='color:#64748B; font-size:0.85rem; line-height:2.2; margin-top:18px;'>
            ① Get a free API key from <b style='color:#94A3B8'>Groq</b> or <b style='color:#94A3B8'>OpenRouter</b>
            &nbsp;→&nbsp;
            ② Upload your file
            &nbsp;→&nbsp;
            ③ Explore, clean, chat & ask deep questions
        </div>
        <div style='margin-top:28px; display:flex; justify-content:center; gap:16px; flex-wrap:wrap;'>
            <a href='https://console.groq.com' target='_blank'
               style='background:#1E293B; border:1px solid #334155; border-radius:8px;
                      padding:8px 18px; color:#F1F5F9; text-decoration:none; font-size:0.88rem;'>
                🟢 Get Groq key (free)
            </a>
            <a href='https://openrouter.ai/keys' target='_blank'
               style='background:#1E293B; border:1px solid #334155; border-radius:8px;
                      padding:8px 18px; color:#F1F5F9; text-decoration:none; font-size:0.88rem;'>
                🔵 Get OpenRouter key (free)
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# DATA LOADED — shared state
# ─────────────────────────────────────────────
df = st.session_state.df
col_info = st.session_state.col_info
primary = st.session_state.primary
freq = st.session_state.freq

trends = compute_time_trend(df, col_info, freq=freq)
val_label = (primary.get("value") or "Value")

# ── Header bar with logo ──
logo_path = Path("logo.jpeg")
if not logo_path.exists(): logo_path = Path("logo.jpg")
if not logo_path.exists(): logo_path = Path("logo.png")

hc1, hc2 = st.columns([0.06, 0.94])
with hc1:
    if logo_path.exists():
        st.image(str(logo_path), width=46)
with hc2:
    st.markdown(
        f"<span style='font-size:1.1rem; font-weight:700; color:#F1F5F9;'>Assyrian-AI</span>"
        f"<span class='provider-badge'>{st.session_state.provider.upper()} · {st.session_state.model or ''}</span>",
        unsafe_allow_html=True,
    )

# ── KPIs ──
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Rows", f"{len(df):,}")
with k2:
    st.metric("Columns", f"{len(df.columns)}")
with k3:
    val_col = primary.get("value")
    if val_col and val_col in df.columns:
        st.metric(f"Total {val_col[:14]}", f"{df[val_col].sum():,.0f}")
    else:
        st.metric("Numeric Cols", len(col_info.get("numeric", [])))
with k4:
    cat_col = primary.get("category")
    if cat_col and cat_col in df.columns:
        st.metric(f"Unique {cat_col[:12]}", f"{df[cat_col].nunique():,}")
    else:
        st.metric("Categories", len(col_info.get("categorical", [])))
with k5:
    null_pct = df.isnull().sum().sum() / df.size * 100
    st.metric("Missing Data", f"{null_pct:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🤖 AI Chat", "📈 Trends", "📊 Distribution", "🏆 Top N",
    "🔬 Statistics & Forecast", "🧹 Clean Data",
    "💬 Deep Chat (RAG)", "🔍 Visual Explorer",
])

# ── TAB 1: AI CHAT ───────────────────────────
with tab1:
    col_chat, col_right = st.columns([3, 2])

    with col_chat:
        st.markdown("### Ask anything about your data")

        if not st.session_state.auto_insights and st.session_state.api_key_set:
            with st.spinner("Generating insights..."):
                try:
                    st.session_state.auto_insights = generate_auto_insights(
                        st.session_state.data_summary,
                        st.session_state.analysis_summary,
                        provider=st.session_state.provider,
                        model=st.session_state.model,
                    )
                except Exception as e:
                    st.session_state.auto_insights = f"Error: {e}"

        for msg in st.session_state.display_history:
            css = "chat-user" if msg["role"] == "user" else "chat-ai"
            icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>',
                       unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "", height=80, label_visibility="collapsed",
                placeholder="e.g. What trends do you see? Which group performs best?",
            )
            send = st.form_submit_button("Send →", use_container_width=True)

        if send and user_input.strip():
            if not st.session_state.api_key_set:
                provider_name = "Groq (console.groq.com)" if st.session_state.provider == "groq" else "OpenRouter (openrouter.ai)"
                st.error(f"Enter your free {provider_name} API key in the sidebar to use AI Chat.")
            else:
                st.session_state.display_history.append({"role": "user", "content": user_input})
                with st.spinner("Thinking..."):
                    try:
                        chunks = list(stream_ai(
                            st.session_state.data_summary,
                            st.session_state.analysis_summary,
                            user_input,
                            history=st.session_state.chat_history.copy(),
                            provider=st.session_state.provider,
                            model=st.session_state.model,
                        ))
                        resp = "".join(chunks)
                        st.session_state.display_history.append({"role": "assistant", "content": resp})
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.session_state.chat_history.append({"role": "assistant", "content": resp})
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.session_state.display_history = []
            st.rerun()

    with col_right:
        st.markdown("### 🔍 Auto Insights")
        if st.session_state.auto_insights:
            st.markdown(f'<div class="insight-box">{st.session_state.auto_insights}</div>',
                       unsafe_allow_html=True)
        elif not st.session_state.api_key_set:
            pname = "Groq" if st.session_state.provider == "groq" else "OpenRouter"
            st.info(f"Add your free {pname} API key in the sidebar to generate insights.")
        else:
            st.info("Insights will appear here.")

        st.markdown("#### 💡 Quick questions")
        quick_qs = [
            "What are the main patterns in this data?",
            "Which group or category has the highest values?",
            "Are there any anomalies or outliers?",
            "What is the overall trend over time?",
            "What are your top 3 recommendations?",
        ]
        if primary.get("category"):
            quick_qs.insert(1, f"Break down the data by {primary['category']}.")
        if primary.get("value"):
            quick_qs.insert(2, f"What drives changes in {primary['value']}?")

        for q in quick_qs[:6]:
            if st.button(q, key=f"q_{q[:25]}", use_container_width=True):
                if st.session_state.api_key_set:
                    st.session_state.display_history.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        try:
                            chunks = list(stream_ai(
                                st.session_state.data_summary,
                                st.session_state.analysis_summary,
                                q,
                                history=st.session_state.chat_history.copy(),
                                provider=st.session_state.provider,
                                model=st.session_state.model,
                            ))
                            resp = "".join(chunks)
                            st.session_state.display_history.append({"role": "assistant", "content": resp})
                            st.session_state.chat_history.append({"role": "user", "content": q})
                            st.session_state.chat_history.append({"role": "assistant", "content": resp})
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("Add your API key first.")


# ── TAB 2: TRENDS ────────────────────────────
with tab2:
    date_col = primary.get("date")
    st.markdown(f"### 📈 {val_label} Over Time")

    if date_col:
        st.caption(f"Date column: `{date_col}` · Grouped by **{freq}**")
        if not trends.empty:
            if "Total" in trends.columns:
                fig = plot_time_trend(trends, val_label)
                if fig: st.pyplot(fig)
                fig2 = plot_count_bar(trends)
                if fig2: st.pyplot(fig2)
                with st.expander("View data table"):
                    disp = trends.copy()
                    disp["Period"] = disp["Period"].dt.strftime("%Y-%m-%d")
                    st.dataframe(disp, use_container_width=True)
        else:
            st.info("Could not compute trends. Check column roles in the sidebar.")
    else:
        st.info("No date/time column detected. Check column roles in the sidebar ⚙️")

    chart_builder(df, "trends", default_x=date_col, default_y=primary.get("value"))


# ── TAB 3: DISTRIBUTION ──────────────────────
with tab3:
    cat_col = primary.get("category")
    st.markdown("### 📊 Distribution & Groups")

    if cat_col:
        available_cats = col_info.get("categorical", [])
        chosen_cat = st.selectbox("Group by column", available_cats,
                                  index=available_cats.index(cat_col) if cat_col in available_cats else 0) \
                     if len(available_cats) > 1 else cat_col

        dist_df = compute_distribution(df, col_info, group_col=chosen_cat)
        if not dist_df.empty:
            fig = plot_distribution(dist_df, chosen_cat, val_label)
            if fig: st.pyplot(fig)
            with st.expander("View distribution table"):
                st.dataframe(dist_df, use_container_width=True)
    else:
        st.info("No categorical column detected for grouping.")

    num_cols = col_info.get("numeric", [])
    if num_cols:
        st.markdown("#### Numeric distributions")
        fig_hist = plot_numeric_distributions(df, num_cols)
        if fig_hist: st.pyplot(fig_hist)

    chart_builder(df, "distribution", default_x=cat_col, default_y=primary.get("value"))


# ── TAB 4: TOP N ─────────────────────────────
with tab4:
    st.markdown("### 🏆 Top N Analysis")

    rank_cols = col_info.get("categorical", []) + col_info.get("id", [])
    val_cols = col_info.get("numeric", [])

    if rank_cols and val_cols:
        c1, c2, c3 = st.columns(3)
        with c1:
            rank_col = st.selectbox("Rank by item", rank_cols)
        with c2:
            metric_col = st.selectbox("Metric", val_cols,
                                      index=val_cols.index(primary.get("value"))
                                      if primary.get("value") in val_cols else 0)
        with c3:
            n_show = st.slider("How many", 5, 50, 15)

        top_df = compute_top_n(df, col_info, rank_col=rank_col, metric_col=metric_col, top_n=n_show)
        if not top_df.empty:
            fig = plot_top_n(top_df, rank_col, "Total")
            if fig: st.pyplot(fig)
            with st.expander("View table"):
                st.dataframe(top_df, use_container_width=True)
    else:
        st.info("Need at least one categorical and one numeric column for Top N.")

    chart_builder(df, "topn",
                 default_x=primary.get("category") or primary.get("id"),
                 default_y=primary.get("value"))


# ── TAB 5: STATISTICS & FORECAST ─────────────
with tab5:
    st.markdown("### 🔬 Statistics")

    stats = compute_statistics(df, col_info)

    if "describe" in stats:
        st.markdown("#### Descriptive statistics")
        st.dataframe(stats["describe"], use_container_width=True)

    if "correlation" in stats:
        st.markdown("#### Correlation matrix")
        fig = plot_correlation(stats["correlation"])
        if fig: st.pyplot(fig)

    num_cols = col_info.get("numeric", [])
    if num_cols:
        st.markdown("#### Box plots")
        fig2 = plot_boxplots(df, num_cols)
        if fig2: st.pyplot(fig2)

    if "outliers" in stats and not stats["outliers"].empty:
        with st.expander(f"⚠️ Outliers — {len(stats['outliers'])} rows (z-score > 3)"):
            st.dataframe(stats["outliers"], use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Forecast")

    if primary.get("date") and primary.get("value"):
        fcast_df = compute_forecast(df, col_info, freq=freq)
        if not fcast_df.empty:
            fig3 = plot_forecast(fcast_df, val_label)
            if fig3: st.pyplot(fig3)
            future = fcast_df[fcast_df["Type"] == "Forecast"]
            if not future.empty:
                fd = future[["Period","Total"]].copy()
                fd["Period"] = fd["Period"].dt.strftime("%Y-%m-%d")
                fd["Total"] = fd["Total"].round(2)
                st.dataframe(fd, use_container_width=True)
            st.caption("⚠️ Linear trend model — for production use consider Prophet or ARIMA.")
        else:
            st.info("Need at least 4 time periods to generate a forecast.")
    else:
        st.info("Forecast requires a date column and a numeric value column.")

    chart_builder(df, "stats", default_x=primary.get("date"), default_y=primary.get("value"))


# ── TAB 6: CLEAN DATA ────────────────────────
with tab6:
    st.markdown("### 🧹 Clean Data")
    st.caption("Every action below modifies your working dataset immediately — all other tabs update too. Use **Reset** to start over from the original upload.")

    working_df = st.session_state.df

    # ── Status row ──
    summary = get_cleaning_summary(st.session_state.original_df, working_df)
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Rows now", f"{summary['cleaned_rows']:,}", f"{-summary['rows_removed']:,}" if summary['rows_removed'] else None)
    with s2: st.metric("Original rows", f"{summary['original_rows']:,}")
    with s3: st.metric("Missing cells now", f"{summary['cleaned_nulls']:,}", f"-{summary['nulls_fixed']:,}" if summary['nulls_fixed'] else None)
    with s4: st.metric("Actions applied", len(st.session_state.cleaning_log))

    rcol1, rcol2 = st.columns([4, 1])
    with rcol2:
        if st.button("↺ Reset to original", use_container_width=True):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.cleaning_log = []
            new_col_info = detect_columns(st.session_state.df)
            st.session_state.col_info = new_col_info
            st.session_state.primary = get_primary_cols(new_col_info)
            st.session_state.data_summary = build_data_summary(st.session_state.df, new_col_info)
            st.session_state.analysis_summary = build_analysis_summary(st.session_state.df, new_col_info)
            st.rerun()

    if st.session_state.cleaning_log:
        with st.expander(f"📜 Cleaning log ({len(st.session_state.cleaning_log)} actions)", expanded=False):
            for i, action in enumerate(st.session_state.cleaning_log, 1):
                st.markdown(f"<span style='color:#94A3B8; font-size:0.85rem;'>{i}. {action}</span>", unsafe_allow_html=True)

    st.markdown("---")

    clean_tab1, clean_tab2, clean_tab3, clean_tab4 = st.tabs([
        "🩹 Missing Values", "🧮 Outliers & Duplicates", "🔤 Text Cleanup", "🔁 Fix Column Types"
    ])

    # ── MISSING VALUES ──
    with clean_tab1:
        missing_report = get_missing_report(working_df)
        if missing_report.empty:
            st.success("No missing values in this dataset. ✓")
        else:
            st.dataframe(missing_report, use_container_width=True)
            st.markdown("#### Fill or drop missing values")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                target_col = st.selectbox("Column", missing_report["Column"].tolist(), key="mv_col")
            is_numeric = pd.api.types.is_numeric_dtype(working_df[target_col])
            options = (["mean", "median", "zero", "custom", "ffill", "bfill", "drop_rows"] if is_numeric
                      else ["mode", "custom", "ffill", "bfill", "drop_rows"])
            with mc2:
                strategy = st.selectbox("Strategy", options, key="mv_strategy")
            with mc3:
                custom_val = st.text_input("Custom value (if selected)", key="mv_custom") if strategy == "custom" else None

            if st.button("Apply fill", key="mv_apply", use_container_width=True):
                new_df = fill_missing(working_df, target_col, strategy, custom_val)
                n_before = working_df[target_col].isnull().sum()
                n_after = new_df[target_col].isnull().sum() if target_col in new_df.columns else 0
                rows_diff = len(working_df) - len(new_df)
                st.session_state.df = new_df
                if strategy == "drop_rows":
                    st.session_state.cleaning_log.append(f"Dropped {rows_diff} rows with missing `{target_col}`")
                else:
                    st.session_state.cleaning_log.append(f"Filled {n_before - n_after} missing values in `{target_col}` using **{strategy}**")
                new_col_info = detect_columns(new_df)
                st.session_state.col_info = new_col_info
                st.session_state.primary = get_primary_cols(new_col_info)
                st.session_state.data_summary = build_data_summary(new_df, new_col_info)
                st.session_state.analysis_summary = build_analysis_summary(new_df, new_col_info)
                st.rerun()

    # ── OUTLIERS & DUPLICATES ──
    with clean_tab2:
        st.markdown("#### Remove outliers")
        num_cols_now = working_df.select_dtypes(include="number").columns.tolist()
        if num_cols_now:
            oc1, oc2 = st.columns(2)
            with oc1:
                outlier_cols = st.multiselect("Columns to check", num_cols_now, default=num_cols_now[:min(3,len(num_cols_now))], key="out_cols")
            with oc2:
                outlier_method = st.selectbox("Method", ["IQR", "Z-score"], key="out_method")
            if st.button("Remove outlier rows", key="out_apply", use_container_width=True) and outlier_cols:
                new_df, n_removed = remove_outliers(working_df, outlier_cols, outlier_method)
                st.session_state.df = new_df
                st.session_state.cleaning_log.append(f"Removed {n_removed} outlier rows from {', '.join(f'`{c}`' for c in outlier_cols)} ({outlier_method})")
                st.rerun()
        else:
            st.info("No numeric columns to check for outliers.")

        st.markdown("---")
        st.markdown("#### Remove duplicate rows")
        dup_count = working_df.duplicated().sum()
        st.caption(f"Currently **{dup_count:,}** exact duplicate rows detected.")
        dup_subset = st.multiselect("Check duplicates based on (leave empty = all columns)", working_df.columns.tolist(), key="dup_subset")
        if st.button("Remove duplicates", key="dup_apply", use_container_width=True):
            new_df, n_removed = remove_duplicates(working_df, dup_subset if dup_subset else None)
            st.session_state.df = new_df
            st.session_state.cleaning_log.append(f"Removed {n_removed} duplicate rows" + (f" (based on {', '.join(dup_subset)})" if dup_subset else ""))
            st.rerun()

    # ── TEXT CLEANUP ──
    with clean_tab3:
        st.markdown("#### Standardize text columns")
        text_cols = working_df.select_dtypes(include="object").columns.tolist()
        if not text_cols:
            st.info("No text/categorical columns detected.")
        else:
            tc1, tc2 = st.columns(2)
            with tc1:
                sel_text_cols = st.multiselect("Columns", text_cols, default=text_cols[:min(3,len(text_cols))], key="txt_cols")
            with tc2:
                ops = st.multiselect("Operations", ["strip", "lower", "upper", "title"], default=["strip"], key="txt_ops")
            if st.button("Apply text cleanup", key="txt_apply", use_container_width=True) and sel_text_cols and ops:
                new_df = standardize_text(working_df, sel_text_cols, ops)
                st.session_state.df = new_df
                st.session_state.cleaning_log.append(f"Standardized text in {', '.join(f'`{c}`' for c in sel_text_cols)} ({', '.join(ops)})")
                st.rerun()

    # ── FIX COLUMN TYPES ──
    with clean_tab4:
        st.markdown("#### Convert a column's data type")
        ftc1, ftc2 = st.columns(2)
        with ftc1:
            type_col = st.selectbox("Column", working_df.columns.tolist(), key="type_col")
            st.caption(f"Current dtype: `{working_df[type_col].dtype}`")
        with ftc2:
            target_type = st.selectbox("Convert to", ["numeric", "datetime", "text"], key="type_target")
        if st.button("Convert column", key="type_apply", use_container_width=True):
            new_df, err = convert_column_type(working_df, type_col, target_type)
            if err:
                st.error(f"Conversion failed: {err}")
            else:
                n_failed = new_df[type_col].isnull().sum() - working_df[type_col].isnull().sum()
                st.session_state.df = new_df
                new_col_info = detect_columns(new_df)
                st.session_state.col_info = new_col_info
                st.session_state.primary = get_primary_cols(new_col_info)
                st.session_state.data_summary = build_data_summary(new_df, new_col_info)
                st.session_state.analysis_summary = build_analysis_summary(new_df, new_col_info)
                msg = f"Converted `{type_col}` to **{target_type}**"
                if n_failed > 0:
                    msg += f" ({n_failed} values became null and couldn't convert)"
                st.session_state.cleaning_log.append(msg)
                st.rerun()

    # ── EXPORT ──
    st.markdown("---")
    st.markdown("### ⬇️ Export Cleaned Dataset")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        csv_buf = io.StringIO()
        st.session_state.df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download as CSV", data=csv_buf.getvalue(),
                           file_name="cleaned_data.csv", mime="text/csv", use_container_width=True)
    with ec2:
        xlsx_buf = io.BytesIO()
        st.session_state.df.to_excel(xlsx_buf, index=False, engine="openpyxl")
        xlsx_buf.seek(0)
        st.download_button("⬇️ Download as Excel", data=xlsx_buf,
                           file_name="cleaned_data.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with ec3:
        json_buf = st.session_state.df.to_json(orient="records", date_format="iso")
        st.download_button("⬇️ Download as JSON", data=json_buf,
                           file_name="cleaned_data.json", mime="application/json", use_container_width=True)

    with st.expander("👀 Preview current dataset", expanded=False):
        st.dataframe(st.session_state.df, use_container_width=True)


# ── TAB 7: DEEP CHAT (RAG) ───────────────────
with tab7:
    st.markdown("### 💬 Deep Chat — Grounded in Your Actual Rows")
    st.caption(
        "Unlike AI Chat (which uses a data summary), Deep Chat retrieves the most relevant "
        "rows from your file using FAISS vector search, then answers based on real data. "
        "Requires an OpenRouter API key."
    )

    if not RAG_AVAILABLE:
        st.error(
            "RAG dependencies not installed. Run:\n```\npip install faiss-cpu sentence-transformers\n```"
        )
    else:
        col_idx, col_info_rag = st.columns([3, 1])

        with col_idx:
            if not st.session_state.rag_indexed:
                st.info(
                    f"Click **Build Index** to prepare {len(df):,} rows for deep search. "
                    "Takes ~10–30 seconds depending on file size."
                )

            if st.button("🔨 Build Index", disabled=st.session_state.rag_indexed):
                with st.spinner("Chunking rows and building FAISS index (first time only)..."):
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_st = SentenceTransformer("all-MiniLM-L6-v2")
                        chunks = df_to_chunks(df, chunk_size=10)
                        index, _ = build_faiss_index(chunks, model_st)
                        st.session_state.rag_model = model_st
                        st.session_state.rag_chunks = chunks
                        st.session_state.rag_index = index
                        st.session_state.rag_indexed = True
                        st.success(f"✓ Indexed {len(chunks)} chunks across {len(df):,} rows.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")

        with col_info_rag:
            if st.session_state.rag_indexed:
                st.success(f"✓ Index ready\n{len(st.session_state.rag_chunks)} chunks")
                if st.button("🔄 Rebuild"):
                    st.session_state.rag_indexed = False
                    st.session_state.rag_history = []
                    st.rerun()

        if st.session_state.rag_indexed:
            st.markdown("---")

            for msg in st.session_state.rag_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="chat-ai">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                    if msg.get("sources"):
                        with st.expander(f"📎 {len(msg['sources'])} retrieved chunks"):
                            for s in msg["sources"]:
                                st.markdown(
                                    f'<div class="rag-source">Rows {s["start_row"]+1}–{s["end_row"]+1} '
                                    f'· score: {s["score"]:.4f}<br><pre>{s["text"][:400]}...</pre></div>',
                                    unsafe_allow_html=True
                                )

            with st.form("rag_form", clear_on_submit=True):
                rag_input = st.text_area(
                    "", height=80, label_visibility="collapsed",
                    placeholder="e.g. Which rows have the highest values? What's unusual about row 12?",
                )
                rag_send = st.form_submit_button("Ask →", use_container_width=True)

            if rag_send and rag_input.strip():
                if not os.environ.get("OPENROUTER_API_KEY"):
                    st.error("Paste your OpenRouter API key in the sidebar to use Deep Chat.")
                else:
                    st.session_state.rag_history.append({"role": "user", "content": rag_input})
                    with st.spinner("Retrieving relevant rows and generating answer..."):
                        try:
                            result = rag_answer(
                                question=rag_input,
                                index=st.session_state.rag_index,
                                chunks=st.session_state.rag_chunks,
                                model=st.session_state.rag_model,
                                top_k=5,
                            )
                            st.session_state.rag_history.append({
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["sources"],
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"RAG error: {e}")

            if st.session_state.rag_history and st.button("Clear Deep Chat"):
                st.session_state.rag_history = []
                st.rerun()

            st.markdown("#### 💡 Try these")
            rag_quick = [
                "Which rows have the highest values?",
                "Are there any unusual or suspicious rows?",
                "What patterns do you see in the first 50 rows?",
                "Which entries are missing data?",
            ]
            for q in rag_quick:
                if st.button(q, key=f"rq_{q[:20]}", use_container_width=True):
                    if os.environ.get("OPENROUTER_API_KEY"):
                        st.session_state.rag_history.append({"role": "user", "content": q})
                        with st.spinner("Retrieving..."):
                            try:
                                result = rag_answer(
                                    question=q,
                                    index=st.session_state.rag_index,
                                    chunks=st.session_state.rag_chunks,
                                    model=st.session_state.rag_model,
                                    top_k=5,
                                )
                                st.session_state.rag_history.append({
                                    "role": "assistant",
                                    "content": result["answer"],
                                    "sources": result["sources"],
                                })
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                    else:
                        st.warning("Add your OpenRouter key first.")


# ── TAB 8: VISUAL EXPLORER (PyGWalker) ───────
with tab8:
    st.markdown("### 🔍 Visual Explorer")
    st.caption(
        "Drag and drop columns to build your own charts — no code needed. "
        "Powered by PyGWalker (Tableau-style exploration)."
    )
    try:
        import pygwalker as pyg
        pyg_html = pyg.to_html(df)
        st.components.v1.html(pyg_html, height=700, scrolling=True)
    except ImportError:
        st.error(
            "PyGWalker not installed. Run:\n```\npip install pygwalker\n```\nthen restart the app."
        )
    except Exception as e:
        st.error(f"Visual Explorer error: {e}")