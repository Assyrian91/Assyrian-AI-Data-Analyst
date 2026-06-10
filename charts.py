import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── Shared style tokens ──────────────────────
BG      = "#0F172A"
CARD    = "#1E293B"
TEXT    = "#F1F5F9"
MUTED   = "#94A3B8"
GRID    = "#334155"
PALETTE = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED","#0891B2","#DB2777","#EA580C"]
ACCENT  = PALETTE[0]


def _fig(w=11, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_color(CARD)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")
    return fig, ax


def _title(ax, t):
    ax.set_title(t, color=TEXT, fontsize=12, pad=10)


def _fmt_axis(ax, axis="y"):
    fmt = mtick.FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 1 else f"{v:.2f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ─────────────────────────────────────────────
# TAB 1 — TIME TREND
# ─────────────────────────────────────────────

def plot_time_trend(trends: pd.DataFrame, value_label: str = "Total"):
    if trends.empty or "Total" not in trends.columns:
        return None
    fig, ax = _fig(11, 4)
    x, y = trends["Period"], trends["Total"]

    ax.fill_between(x, y, alpha=0.12, color=ACCENT)
    ax.plot(x, y, color=ACCENT, linewidth=2.5, marker="o", markersize=4, label=value_label)

    if len(y) >= 3:
        ma = y.rolling(3, min_periods=1).mean()
        ax.plot(x, ma, color="#F59E0B", linewidth=1.5, linestyle="--", label="3-period avg")

    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8, framealpha=0.6)
    _fmt_axis(ax, "y")
    _title(ax, f"{value_label} Over Time")
    fig.tight_layout()
    return fig


def plot_count_bar(trends: pd.DataFrame):
    if trends.empty or "Count" not in trends.columns:
        return None
    fig, ax = _fig(11, 3)
    ax.bar(trends["Period"], trends["Count"], color=PALETTE[1], alpha=0.8, width=20)
    _fmt_axis(ax, "y")
    _title(ax, "Record Count Over Time")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# TAB 2 — DISTRIBUTION
# ─────────────────────────────────────────────

def plot_distribution(dist: pd.DataFrame, cat_col: str, value_label: str = "Total"):
    if dist.empty or cat_col not in dist.columns:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.patch.set_facecolor(BG)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(dist))]

    # Donut
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    col_data = dist["Total"] if "Total" in dist.columns else dist["Count"]
    wedges, texts, autotexts = ax1.pie(
        col_data.head(8), labels=dist[cat_col].head(8).astype(str),
        autopct="%1.0f%%", colors=colors[:8], startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
    )
    for t in texts:   t.set_color(TEXT); t.set_fontsize(8)
    for at in autotexts: at.set_color(BG); at.set_fontsize(7); at.set_fontweight("bold")
    ax1.set_title("Share by Group", color=TEXT, fontsize=11)

    # Horizontal bar
    ax2 = axes[1]
    ax2.set_facecolor(CARD)
    plot_data = dist.head(12).sort_values("Count", ascending=True)
    ax2.barh(plot_data[cat_col].astype(str), plot_data["Count"],
             color=colors[:len(plot_data)], alpha=0.85)
    ax2.set_title("Count by Group", color=TEXT, fontsize=11)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for sp in ax2.spines.values(): sp.set_color(CARD)
    ax2.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, num_cols: list):
    """Small multiples histogram grid for numeric columns."""
    cols = num_cols[:8]
    if not cols:
        return None
    n   = len(cols)
    ncols_grid = min(4, n)
    nrows_grid = (n + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid,
                              figsize=(ncols_grid * 3.5, nrows_grid * 2.8),
                              facecolor=BG)
    axes = np.array(axes).flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.set_facecolor(CARD)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values(): sp.set_color(CARD)
        data = df[col].dropna()
        ax.hist(data, bins=25, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor=BG)
        ax.set_title(col[:20], color=TEXT, fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.4, linestyle="--")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# TAB 3 — TOP N
# ─────────────────────────────────────────────

def plot_top_n(top_df: pd.DataFrame, label_col: str, value_col: str = "Total"):
    if top_df.empty or label_col not in top_df.columns:
        return None
    fig, ax = _fig(11, max(4, len(top_df) * 0.4))
    df_s = top_df.sort_values(value_col, ascending=True)
    labels = [str(v)[:35] for v in df_s[label_col]]
    vals   = df_s[value_col].values
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    ax.barh(labels, vals, color=colors, alpha=0.85)
    ax.set_title(f"Top Items by {value_col}", color=TEXT, fontsize=12, pad=10)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
    ax.grid(axis="y", visible=False)
    for sp in ax.spines.values(): sp.set_color(CARD)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# TAB 4 — STATISTICS
# ─────────────────────────────────────────────

def plot_correlation(corr: pd.DataFrame):
    if corr is None or corr.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(5, len(corr)), max(4, len(corr) * 0.8)),
                            facecolor=BG)
    ax.set_facecolor(CARD)
    im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8, color=MUTED)

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            val = corr.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.5 else TEXT)

    _title(ax, "Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_boxplots(df: pd.DataFrame, num_cols: list):
    cols = num_cols[:8]
    if not cols:
        return None
    fig, ax = _fig(11, 4)
    data = [df[c].dropna().values for c in cols]
    bp   = ax.boxplot(data, patch_artist=True, notch=False,
                      medianprops=dict(color=TEXT, linewidth=2),
                      whiskerprops=dict(color=MUTED),
                      capprops=dict(color=MUTED),
                      flierprops=dict(marker="o", color=MUTED, markersize=3, alpha=0.4))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(cols) + 1))
    ax.set_xticklabels([c[:15] for c in cols], rotation=20, ha="right", fontsize=8)
    _title(ax, "Distribution of Numeric Columns")
    _fmt_axis(ax, "y")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# TAB 5 — FORECAST
# ─────────────────────────────────────────────

def plot_forecast(fcast: pd.DataFrame, value_label: str = "Total"):
    if fcast.empty or "Total" not in fcast.columns:
        return None
    fig, ax = _fig(11, 4)

    hist   = fcast[fcast["Type"] == "Historical"]
    future = fcast[fcast["Type"] == "Forecast"]

    ax.plot(hist["Period"], hist["Total"], color=ACCENT, linewidth=2.5,
            marker="o", markersize=3, label="Historical")
    ax.fill_between(hist["Period"], hist["Total"], alpha=0.1, color=ACCENT)

    if not future.empty:
        connect_x = [hist["Period"].iloc[-1], future["Period"].iloc[0]]
        connect_y = [hist["Total"].iloc[-1],  future["Total"].iloc[0]]
        ax.plot(connect_x, connect_y, color="#F59E0B", linewidth=2, linestyle="--")
        ax.plot(future["Period"], future["Total"], color="#F59E0B",
                linewidth=2.5, linestyle="--", marker="s", markersize=5, label="Forecast")
        ax.fill_between(future["Period"],
                        future["Total"] * 0.85, future["Total"] * 1.15,
                        alpha=0.1, color="#F59E0B", label="±15% band")

    ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8, framealpha=0.6)
    _fmt_axis(ax, "y")
    _title(ax, f"{value_label} — Forecast")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# CUSTOM CHART BUILDER
# ─────────────────────────────────────────────

CHART_TYPES = [
    "Bar chart", "Horizontal bar", "Line chart", "Area chart",
    "Scatter plot", "Histogram", "Pie chart", "Box plot",
]


def build_custom_chart(df: pd.DataFrame, chart_type: str,
                        x_col, y_col, color_col=None,
                        top_n: int = 20, title: str = ""):
    plt.close("all")
    fig, ax = _fig()

    try:
        if x_col and y_col and x_col != y_col:
            if pd.api.types.is_numeric_dtype(df[x_col]):
                plot_df = df[[x_col, y_col]].dropna()
            else:
                plot_df = (
                    df.groupby(x_col)[y_col].sum()
                    .reset_index()
                    .sort_values(y_col, ascending=False)
                    .head(top_n)
                )
        elif x_col:
            plot_df       = df[x_col].value_counts().head(top_n).reset_index()
            plot_df.columns = [x_col, "Count"]
            y_col         = "Count"
        else:
            return None

        n_items = len(plot_df)
        colors  = [PALETTE[i % len(PALETTE)] for i in range(n_items)]

        if chart_type == "Bar chart":
            ax.bar(plot_df[x_col].astype(str), plot_df[y_col], color=colors, alpha=0.85)
            ax.set_xticklabels(plot_df[x_col].astype(str), rotation=35, ha="right", fontsize=8)

        elif chart_type == "Horizontal bar":
            ax.barh(plot_df[x_col].astype(str), plot_df[y_col], color=colors, alpha=0.85)
            ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
            ax.grid(axis="y", visible=False)

        elif chart_type == "Line chart":
            ax.plot(range(n_items), plot_df[y_col],
                    color=ACCENT, linewidth=2.5, marker="o", markersize=4)
            ax.fill_between(range(n_items), plot_df[y_col], alpha=0.1, color=ACCENT)
            ax.set_xticks(range(n_items))
            ax.set_xticklabels(plot_df[x_col].astype(str), rotation=35, ha="right", fontsize=8)

        elif chart_type == "Area chart":
            ax.fill_between(range(n_items), plot_df[y_col], alpha=0.4, color=ACCENT)
            ax.plot(range(n_items), plot_df[y_col], color=ACCENT, linewidth=2)
            ax.set_xticks(range(n_items))
            ax.set_xticklabels(plot_df[x_col].astype(str), rotation=35, ha="right", fontsize=8)

        elif chart_type == "Scatter plot":
            raw    = df[[x_col, y_col]].dropna()
            c_vals = None
            if color_col and color_col in df.columns:
                cats   = df[color_col].astype("category").cat.codes
                c_vals = [PALETTE[i % len(PALETTE)] for i in cats[raw.index]]
            ax.scatter(raw[x_col], raw[y_col],
                       c=c_vals if c_vals else ACCENT,
                       alpha=0.5, s=15, edgecolors="none")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif chart_type == "Histogram":
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                return None
            ax.hist(df[x_col].dropna(), bins=30, color=ACCENT, alpha=0.85, edgecolor=BG)
            ax.set_xlabel(x_col)

        elif chart_type == "Pie chart":
            small = plot_df.head(8)
            wedges, texts, autotexts = ax.pie(
                small[y_col], labels=small[x_col].astype(str),
                autopct="%1.0f%%", colors=PALETTE[:len(small)],
                startangle=140, wedgeprops=dict(edgecolor=BG, linewidth=1.5),
                pctdistance=0.75,
            )
            for t in texts:   t.set_color(TEXT); t.set_fontsize(8)
            for at in autotexts: at.set_color(BG); at.set_fontsize(7)
            ax.set_facecolor(BG)

        elif chart_type == "Box plot":
            num_cols = [y_col] if y_col else df.select_dtypes("number").columns[:5].tolist()
            data = [df[c].dropna().values for c in num_cols]
            bp   = ax.boxplot(data, patch_artist=True,
                              medianprops=dict(color=TEXT, linewidth=2))
            for patch, c in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_xticks(range(1, len(num_cols) + 1))
            ax.set_xticklabels(num_cols, rotation=20, ha="right", fontsize=8)

        t = title or f"{chart_type} — {y_col} by {x_col}"
        ax.set_title(t, color=TEXT, fontsize=11, pad=10)
        fig.tight_layout()
        return fig

    except Exception as e:
        plt.close("all")
        import streamlit as st
        st.error(f"Chart error: {e}")
        return None
