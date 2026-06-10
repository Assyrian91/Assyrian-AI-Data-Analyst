import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────
# TAB 1 — OVERVIEW  (replaces "Revenue")
# Trend of the primary numeric value over time (any domain)
# ─────────────────────────────────────────────

def compute_time_trend(df: pd.DataFrame, col_info: dict, freq: str = "M") -> pd.DataFrame:
    """
    Aggregate primary value column by time period.
    Works for any dataset that has a date + numeric column.
    freq: 'D' daily, 'W' weekly, 'ME' monthly, 'QE' quarterly, 'YE' yearly
    """
    from data_loader import get_primary_cols
    primary  = get_primary_cols(col_info)
    date_col = primary["date"]
    val_col  = primary["value"]

    if not date_col:
        return pd.DataFrame()

    df2 = df.copy()
    df2["_date"] = pd.to_datetime(df2[date_col], errors="coerce")
    df2 = df2.dropna(subset=["_date"])

    freq_map = {"D": "D", "W": "W", "M": "ME", "Q": "QE", "Y": "YE"}
    resample_freq = freq_map.get(freq, "ME")

    indexed   = df2.set_index("_date")
    resampled = indexed.resample(resample_freq)

    result = resampled.size().reset_index()
    result.columns = ["Period", "Count"]

    if val_col and val_col in indexed.columns:
        totals         = resampled[val_col].sum().reset_index()
        averages       = resampled[val_col].mean().reset_index()
        totals.columns   = ["Period", "Total"]
        averages.columns = ["Period", "Average"]
        result = result.merge(totals,   on="Period", how="left")
        result = result.merge(averages, on="Period", how="left")

    return result.sort_values("Period")


# ─────────────────────────────────────────────
# TAB 2 — DISTRIBUTION  (replaces "Customers")
# Segment / group breakdown of any categorical column
# ─────────────────────────────────────────────

def compute_distribution(df: pd.DataFrame, col_info: dict, group_col: str = None) -> pd.DataFrame:
    """
    Group by a categorical column, aggregate numeric value.
    If group_col is None, uses the primary category column.
    """
    from data_loader import get_primary_cols
    primary  = get_primary_cols(col_info)
    cat_col  = group_col or primary["category"]
    val_col  = primary["value"]

    if not cat_col or cat_col not in df.columns:
        return pd.DataFrame()

    agg = {"Count": (cat_col, "count")}
    if val_col and val_col in df.columns:
        agg["Total"]   = (val_col, "sum")
        agg["Average"] = (val_col, "mean")

    result = (
        df.groupby(cat_col)
        .agg(**agg)
        .reset_index()
        .sort_values("Count", ascending=False)
    )
    result["Share_%"] = (result["Count"] / result["Count"].sum() * 100).round(1)
    return result


# ─────────────────────────────────────────────
# TAB 3 — TOP N  (replaces "Products")
# Rank any column by any metric
# ─────────────────────────────────────────────

def compute_top_n(df: pd.DataFrame, col_info: dict,
                  rank_col: str = None, metric_col: str = None,
                  top_n: int = 15, ascending: bool = False) -> pd.DataFrame:
    """
    Rank items in rank_col by aggregated metric_col.
    Falls back to primary category / value if not specified.
    """
    from data_loader import get_primary_cols
    primary   = get_primary_cols(col_info)
    rank_col  = rank_col  or primary["category"] or primary["id"]
    metric_col = metric_col or primary["value"]

    if not rank_col or rank_col not in df.columns:
        return pd.DataFrame()

    if metric_col and metric_col in df.columns:
        result = (
            df.groupby(rank_col)[metric_col]
            .sum()
            .reset_index()
            .rename(columns={metric_col: "Total"})
            .sort_values("Total", ascending=ascending)
            .head(top_n)
        )
        result["Rank"] = range(1, len(result) + 1)
    else:
        result = (
            df[rank_col].value_counts()
            .head(top_n)
            .reset_index()
        )
        result.columns = [rank_col, "Total"]
        result["Rank"] = range(1, len(result) + 1)

    if "Total" in result.columns:
        result["Total"] = result["Total"].round(2)
    return result


# ─────────────────────────────────────────────
# TAB 4 — STATISTICS  (replaces "Forecast")
# Descriptive stats, correlation, outliers
# ─────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame, col_info: dict) -> dict:
    """
    Returns a dict with:
    - 'describe': extended describe for all numeric cols
    - 'correlation': correlation matrix
    - 'outliers': rows with z-score > 3 on any numeric col
    - 'skew': skewness per numeric col
    """
    num_cols = col_info.get("numeric", [])
    result   = {}

    if num_cols:
        desc = df[num_cols].describe().T
        desc["skew"]     = df[num_cols].skew()
        desc["kurtosis"] = df[num_cols].kurtosis()
        result["describe"] = desc.round(3)

        if len(num_cols) >= 2:
            result["correlation"] = df[num_cols].corr().round(3)

        # Outliers via z-score
        try:
            z     = np.abs(scipy_stats.zscore(df[num_cols].dropna()))
            mask  = (z > 3).any(axis=1)
            outlier_idx = df[num_cols].dropna().index[mask]
            result["outliers"] = df.loc[outlier_idx].head(50).reset_index(drop=True)
        except Exception:
            result["outliers"] = pd.DataFrame()

    return result


# ─────────────────────────────────────────────
# FORECAST  (generic — any numeric time series)
# ─────────────────────────────────────────────

def compute_forecast(df: pd.DataFrame, col_info: dict,
                     periods: int = 6, freq: str = "M") -> pd.DataFrame:
    """
    Simple linear-trend forecast on primary value column.
    Returns combined historical + forecast DataFrame.
    """
    trends = compute_time_trend(df, col_info, freq=freq)
    if trends.empty or "Total" not in trends.columns or len(trends) < 4:
        return pd.DataFrame()

    ts = trends.set_index("Period")["Total"].dropna()
    n  = len(ts)
    x  = np.arange(n)

    try:
        log_y   = np.log1p(np.clip(ts.values, 0, None))
        coeffs  = np.polyfit(x, log_y, 1)
        poly    = np.poly1d(coeffs)

        freq_alias = {"M": "MS", "Q": "QS", "Y": "YS", "W": "W", "D": "D"}
        f_alias = freq_alias.get(freq, "MS")
        future_dates = pd.date_range(
            ts.index[-1] + pd.tseries.frequencies.to_offset(f_alias),
            periods=periods, freq=f_alias
        )
        future_vals = np.expm1(poly(np.arange(n, n + periods)))

        hist  = pd.DataFrame({"Period": ts.index, "Total": ts.values, "Type": "Historical"})
        fcast = pd.DataFrame({"Period": future_dates, "Total": future_vals, "Type": "Forecast"})
        return pd.concat([hist, fcast], ignore_index=True)
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# AI CONTEXT SUMMARY
# ─────────────────────────────────────────────

def build_analysis_summary(df: pd.DataFrame, col_info: dict) -> str:
    from data_loader import get_primary_cols
    primary = get_primary_cols(col_info)
    lines   = []

    trends = compute_time_trend(df, col_info)
    if not trends.empty and "Total" in trends.columns:
        lines.append(f"Total across all periods: {trends['Total'].sum():,.2f}")
        best = trends.loc[trends['Total'].idxmax(), 'Period']
        lines.append(f"Peak period: {best} ({trends['Total'].max():,.2f})")
        mom = trends['Total'].pct_change().mean() * 100
        lines.append(f"Avg period-over-period change: {mom:.1f}%")

    dist = compute_distribution(df, col_info)
    if not dist.empty:
        cat_col = primary["category"]
        top3 = dist.head(3)[cat_col].tolist() if cat_col in dist.columns else []
        lines.append(f"\nTop 3 groups in '{cat_col}': {top3}")
        lines.append(f"Total unique groups: {len(dist)}")

    stats = compute_statistics(df, col_info)
    if "describe" in stats:
        desc = stats["describe"]
        lines.append(f"\nNumeric columns: {list(desc.index)}")
        for col in list(desc.index)[:3]:
            row = desc.loc[col]
            lines.append(f"  {col}: mean={row['mean']:.2f}, std={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}")

    if "outliers" in stats and not stats["outliers"].empty:
        lines.append(f"\nOutliers detected (z>3): {len(stats['outliers'])} rows")

    fcast = compute_forecast(df, col_info)
    if not fcast.empty:
        future = fcast[fcast["Type"] == "Forecast"]
        if not future.empty:
            lines.append(f"\nForecast avg (next periods): {future['Total'].mean():,.2f}")

    return "\n".join(lines)
