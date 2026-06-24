import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from difflib import get_close_matches


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


# ─── NEW FUNCTIONS ────────────────────────────────────────────────────────────


def standardize_values(df: pd.DataFrame, col: str, mapping: Dict) -> pd.DataFrame:
    """
    Map inconsistent values to standard values using a dictionary.
    
    Example:
        mapping = {
            "US": "United States",
            "U.S.A": "United States",
            "USA": "United States",
            "U.S.": "United States",
            "UK": "United Kingdom",
            "U.K.": "United Kingdom",
        }
    
    Args:
        df: DataFrame
        col: Column name to standardize
        mapping: Dictionary of {old_value: new_value}
    
    Returns:
        DataFrame with standardized values
    """
    df = df.copy()
    
    # Create reverse lookup for case-insensitive matching
    mapping_lower = {str(k).lower().strip(): v for k, v in mapping.items()}
    
    def apply_mapping(val):
        if pd.isna(val):
            return val
        val_str = str(val).strip()
        # Try exact match first
        if val_str in mapping:
            return mapping[val_str]
        # Try case-insensitive match
        if val_str.lower() in mapping_lower:
            return mapping_lower[val_str.lower()]
        return val
    
    df[col] = df[col].apply(apply_mapping)
    return df


def normalize_dates(df: pd.DataFrame, col: str, output_format: str = "%Y-%m-%d") -> Tuple[pd.DataFrame, str]:
    """
    Parse and normalize dates to a consistent format.
    
    Handles mixed formats like:
        - "01/15/2024"
        - "Jan 15, 2024"
        - "15-01-2024"
        - "2024-01-15"
        - "01-15-2024"
    
    Args:
        df: DataFrame
        col: Column name with date values
        output_format: strftime format string (default: "%Y-%m-%d")
    
    Returns:
        Tuple of (DataFrame, error_message_or_empty_string)
    """
    df = df.copy()
    try:
        # Parse dates with flexible format detection
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        
        # Count how many successfully parsed
        n_parsed = parsed.notna().sum()
        n_total = df[col].notna().sum()
        
        if n_total > 0 and n_parsed == 0:
            return df, f"Could not parse any dates in column '{col}'"
        
        # Format to consistent output
        df[col] = parsed.dt.strftime(output_format)
        
        # Replace "NaT" strings back with actual NaN
        df[col] = df[col].replace("NaT", np.nan)
        
        return df, ""
    except Exception as e:
        return df, str(e)


def fuzzy_standardize(df: pd.DataFrame, col: str, valid_values: List[str], threshold: int = 80) -> Tuple[pd.DataFrame, Dict]:
    """
    Match messy text values to closest valid value using fuzzy matching.
    
    Fixes typos like:
        - "Californa" → "California"
        - "Untied States" → "United States"
        - "New Yrok" → "New York"
    
    Args:
        df: DataFrame
        col: Column name to clean
        valid_values: List of correct/standard values
        threshold: Minimum similarity score (0-100) to accept a match
    
    Returns:
        Tuple of (DataFrame, changes_made_dict)
    """
    df = df.copy()
    changes = {}
    valid_lower = [str(v).lower().strip() for v in valid_values]
    
    unique_vals = df[col].dropna().unique()
    
    for val in unique_vals:
        val_str = str(val).strip()
        val_lower = val_str.lower()
        
        # Skip if already a valid value (exact or case-insensitive)
        if val_str in valid_values or val_lower in valid_lower:
            continue
        
        # Use difflib for fuzzy matching (no extra dependency)
        matches = get_close_matches(val_lower, valid_lower, n=1, cutoff=threshold / 100)
        
        if matches:
            # Find the original-case version of the matched value
            match_idx = valid_lower.index(matches[0])
            corrected = valid_values[match_idx]
            
            # Apply the fix
            df[col] = df[col].replace(val, corrected)
            changes[val_str] = corrected
    
    return df, changes


def suggest_cleaning(df: pd.DataFrame) -> Dict:
    """
    Automatically detect potential data quality issues in the dataset.
    
    Checks for:
        - Inconsistent text values (e.g., US vs U.S.A)
        - Potential date columns not typed as datetime
        - Columns with many unique values that might be IDs
        - Numeric columns stored as text
        - High missing value columns
        - Duplicate rows
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary of {column_name: [list of issues found]}
    """
    suggestions = {}
    
    for col in df.columns:
        issues = []
        non_null = df[col].dropna()
        
        if len(non_null) == 0:
            continue
        
        # ── Check for inconsistent text values ──
        if non_null.dtype == 'object':
            unique_vals = non_null.astype(str).str.strip().unique()
            unique_lower = [v.lower() for v in unique_vals]
            
            # Find groups of similar values
            seen = set()
            for i, val in enumerate(unique_lower):
                if val in seen:
                    continue
                similar = [unique_vals[j] for j in range(len(unique_lower))
                           if val in unique_lower[j] or unique_lower[j] in val
                           and unique_vals[j] != unique_vals[i]]
                if similar:
                    issues.append(f"Similar values found: '{unique_vals[i]}' ↔ {similar}")
                    for s in similar:
                        seen.add(s.lower())
            
            # Check for mixed case inconsistencies
            if len(unique_lower) != len(set(unique_lower)):
                issues.append("Mixed case values detected (e.g., 'Apple' vs 'apple')")
        
        # ── Check for potential date columns ──
        if non_null.dtype == 'object':
            sample = non_null.head(100)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True)
                parse_rate = parsed.notna().sum() / len(sample)
                if parse_rate > 0.7:
                    issues.append(f"Looks like dates ({parse_rate*100:.0f}% parsed) — consider converting to datetime")
            except Exception:
                pass
        
        # ── Check for numeric stored as text ──
        if non_null.dtype == 'object':
            sample = non_null.head(100)
            try:
                numeric_converted = pd.to_numeric(sample, errors="coerce")
                numeric_rate = numeric_converted.notna().sum() / len(sample)
                if numeric_rate > 0.8:
                    issues.append(f"Looks numeric ({numeric_rate*100:.0f}% converted) — consider converting to numeric type")
            except Exception:
                pass
        
        # ── Check for high cardinality (possible ID column) ──
        if non_null.nunique() == len(non_null) and len(non_null) > 10:
            issues.append(f"All {len(non_null)} values are unique — might be an ID column")
        
        # ── Check for high missing percentage ──
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 50:
            issues.append(f"High missing rate: {missing_pct:.1f}%")
        elif missing_pct > 20:
            issues.append(f"Moderate missing rate: {missing_pct:.1f}%")
        
        if issues:
            suggestions[col] = issues
    
    # ── Check for duplicate rows ──
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        suggestions["__row_duplicates__"] = [
            f"Found {n_dupes} duplicate rows ({n_dupes/len(df)*100:.1f}%)"
        ]
    
    return suggestions


def auto_clean(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Automatically clean common data issues without user input.
    
    Performs:
        - Strip whitespace from all text columns
        - Normalize dates if detected
        - Convert numeric-looking text to numbers
        - Remove completely empty rows
        - Optionally: remove outliers
    
    Args:
        df: DataFrame to clean
        aggressive: If True, also removes outliers and fills remaining nulls
    
    Returns:
        Tuple of (cleaned DataFrame, log of actions taken)
    """
    df = df.copy()
    log = {"actions": [], "columns_affected": []}
    
    # Remove completely empty rows
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        log["actions"].append(f"Removed {before - len(df)} fully empty rows")
    
    # Strip whitespace from all text columns
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            before_vals = df[col].tolist()
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
            if df[col].tolist() != before_vals:
                text_cols.append(col)
    if text_cols:
        log["actions"].append(f"Stripped whitespace from: {text_cols}")
        log["columns_affected"].extend(text_cols)
    
    # Detect and normalize date columns
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True)
                if parsed.notna().sum() / len(sample) > 0.8:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[col] = df[col].dt.strftime("%Y-%m-%d")
                    df[col] = df[col].replace("NaT", np.nan)
                    log["actions"].append(f"Auto-converted '{col}' to date format (YYYY-MM-DD)")
                    log["columns_affected"].append(col)
            except Exception:
                pass
    
    # Convert numeric-looking text columns
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            try:
                numeric_converted = pd.to_numeric(sample, errors="coerce")
                if numeric_converted.notna().sum() / len(sample) > 0.9:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    log["actions"].append(f"Auto-converted '{col}' to numeric")
                    log["columns_affected"].append(col)
            except Exception:
                pass
    
    # Aggressive mode: remove outliers and fill nulls
    if aggressive:
        # Fill remaining nulls
        for col in df.columns:
            n_null = df[col].isnull().sum()
            if n_null == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
                log["actions"].append(f"Filled '{col}' nulls with median ({n_null} values)")
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    log["actions"].append(f"Filled '{col}' nulls with mode ({n_null} values)")
        
        # Remove outliers from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            before = len(df)
            mask = pd.Series(False, index=df.index)
            for col in numeric_cols:
                data = df[col].dropna()
                if len(data) < 4:
                    continue
                q1, q3 = data.quantile(0.25), data.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                col_mask = (df[col] < lo) | (df[col] > hi)
                mask |= col_mask.fillna(False)
            n_outliers = int(mask.sum())
            if n_outliers > 0:
                df = df[~mask]
                log["actions"].append(f"Removed {n_outliers} outlier rows (IQR method)")
    
    log["total_actions"] = len(log["actions"])
    return df, log