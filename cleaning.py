import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing value report."""
    rows = []
    for col in df.columns:
        n_null = df[col].isnull().sum()
        if n_null > 0:
            rows.append({
                "Column": col,
                "Missing": int(n_null),
                "Missing %": round(n_null / len(df) * 100, 1),
                "Dtype": str(df[col].dtype),
            })
    return pd.DataFrame(rows).sort_values("Missing", ascending=False) if rows else pd.DataFrame()


def fill_missing(df: pd.DataFrame, col: str, strategy: str, custom_value=None) -> pd.DataFrame:
    """
    Fill missing values in a single column.
    strategy: 'mean', 'median', 'mode', 'zero', 'custom', 'ffill', 'bfill', 'drop_rows'
    """
    df = df.copy()
    if strategy == "drop_rows":
        return df.dropna(subset=[col])

    if pd.api.types.is_numeric_dtype(df[col]):
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "custom" and custom_value is not None:
            df[col] = df[col].fillna(float(custom_value))
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
    else:
        if strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "custom" and custom_value is not None:
            df[col] = df[col].fillna(str(custom_value))
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()

    return df


def remove_outliers(df: pd.DataFrame, cols: List[str], method: str = "IQR") -> Tuple[pd.DataFrame, int]:
    """Remove rows flagged as outliers in any of the given numeric columns."""
    df = df.copy()
    mask = pd.Series(False, index=df.index)

    for col in cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        if method == "IQR":
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            col_mask = (df[col] < lo) | (df[col] > hi)
        else:  # Z-score
            z = (df[col] - data.mean()) / data.std()
            col_mask = z.abs() > 3
        mask |= col_mask.fillna(False)

    n_removed = int(mask.sum())
    return df[~mask], n_removed


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows, optionally based on a subset of columns."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=subset if subset else None)
    return cleaned, before - len(cleaned)


def standardize_text(df: pd.DataFrame, cols: List[str], operations: List[str]) -> pd.DataFrame:
    """
    Standardize text columns.
    operations: list containing any of 'strip', 'lower', 'upper', 'title'
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        if "strip" in operations:
            s = s.str.strip()
        if "lower" in operations:
            s = s.str.lower()
        elif "upper" in operations:
            s = s.str.upper()
        elif "title" in operations:
            s = s.str.title()
        df[col] = s.where(df[col].notna(), df[col])  # preserve original NaNs
    return df


def convert_column_type(df: pd.DataFrame, col: str, target_type: str) -> Tuple[pd.DataFrame, str]:
    """
    Convert a column to a target type.
    target_type: 'numeric', 'datetime', 'text'
    Returns (df, error_message_or_empty_string)
    """
    df = df.copy()
    try:
        if target_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif target_type == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif target_type == "text":
            df[col] = df[col].astype(str)
        return df, ""
    except Exception as e:
        return df, str(e)


def get_cleaning_summary(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
    """Compare original vs cleaned dataset."""
    return {
        "original_rows": len(original_df),
        "cleaned_rows": len(cleaned_df),
        "rows_removed": len(original_df) - len(cleaned_df),
        "original_nulls": int(original_df.isnull().sum().sum()),
        "cleaned_nulls": int(cleaned_df.isnull().sum().sum()),
        "nulls_fixed": int(original_df.isnull().sum().sum()) - int(cleaned_df.isnull().sum().sum()),
    }