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
            x_idx  = x_opts.index(default_x) if default_x in x_opts else 0
            x_col  = st.selectbox("X axis / Group by", x_opts, index=x_idx, key=f"xc_{tab_key}")
            x_col  = None if x_col == "(none)" else x_col
        with c3:
            y_opts = ["(none)"] + num_cols
            y_idx  = y_opts.index(default_y) if default_y in y_opts else 0
            y_col  = st.selectbox("Y axis / Value", y_opts, index=y_idx, key=f"yc_{tab_key}")
            y_col  = None if y_col == "(none)" else y_col

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
        key_label  = "Groq API Key"
        key_help   = "Free at console.groq.com → No credit card needed"
        key_env    = "GROQ_API_KEY"
        key_link   = "https://console.groq.com"
    else:
        model_list = OPENROUTER_MODELS
        key_label  = "OpenRouter API Key"
        key_help   = "Free at openrouter.ai → Free models available"
        key_env    = "OPENROUTER_API_KEY"
        key_link   = "https://openrouter.ai/keys"

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
            df       = load_data(uploaded)
            col_info = detect_columns(df)
            primary  = get_primary_cols(col_info)
            data_sum = build_data_summary(df, col_info)
            anal_sum = build_analysis_summary(df, col_info)

            st.session_state.df               = df
            st.session_state.col_info         = col_info
            st.session_state.primary          = primary
            st.session_state.data_summary     = data_sum
            st.session_state.analysis_summary = anal_sum
            st.session_state.chat_history     = []
            st.session_state.display_history  = []
            st.session_state.auto_insights    = ""
            st.session_state.custom_figs      = {}

        st.success(f"✓ {len(df):,} rows × {len(df.columns)} cols")

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
                idx     = all_c.index(current) if current in all_c else 0
                chosen  = st.selectbox(label, all_c, index=idx, key=f"ov_{role}")
                if chosen != "(auto)":
                    overrides[role] = chosen
            if st.button("Apply overrides") and overrides:
                primary.update(overrides)
                st.session_state.primary          = primary
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
    v2.0
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────
if st.session_state.df is None:
    # Show logo centered on landing too
    logo_path = Path("logo.jpeg")
    if not logo_path.exists(): logo_path = Path("logo.jpg")
    if not logo_path.exists(): logo_path = Path("logo.png")

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if logo_path.exists():
            st.image(str(logo_path), width=120)

    st.markdown("""
    <div style='text-align:center; padding: 10px 20px 60px 20px;'>
        <h1 style='color:#F1F5F9; font-size:2.2rem; margin-bottom:6px;'>Assyrian-AI Data Analyst</h1>
        <p style='color:#94A3B8; font-size:1.05rem; max-width:560px; margin:0 auto 10px;'>
            Upload <em>any</em> CSV, Excel, or JSON — sales, HR, finance, healthcare,
            logistics, surveys, sports, or anything else.<br>
            Get instant charts, statistics, and AI-powered insights.
        </p>
        <div style='color:#64748B; font-size:0.85rem; line-height:2.2; margin-top:18px;'>
            ① Get a free API key from <b style='color:#94A3B8'>Groq</b> or <b style='color:#94A3B8'>OpenRouter</b>
            &nbsp;→&nbsp;
            ② Upload your file
            &nbsp;→&nbsp;
            ③ Explore & ask questions
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
df       = st.session_state.df
col_info = st.session_state.col_info
primary  = st.session_state.primary
freq     = st.session_state.freq
trends   = compute_time_trend(df, col_info, freq=freq)
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 AI Chat", "📈 Trends", "📊 Distribution", "🏆 Top N", "🔬 Statistics & Forecast",
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
            css  = "chat-user" if msg["role"] == "user" else "chat-ai"
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
    val_cols  = col_info.get("numeric", [])

    if rank_cols and val_cols:
        c1, c2, c3 = st.columns(3)
        with c1:
            rank_col = st.selectbox("Rank by item", rank_cols)
        with c2:
            metric_col = st.selectbox("Metric",  val_cols,
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
                fd["Total"]  = fd["Total"].round(2)
                st.dataframe(fd, use_container_width=True)
            st.caption("⚠️ Linear trend model — for production use consider Prophet or ARIMA.")
        else:
            st.info("Need at least 4 time periods to generate a forecast.")
    else:
        st.info("Forecast requires a date column and a numeric value column.")

    chart_builder(df, "stats", default_x=primary.get("date"), default_y=primary.get("value"))
