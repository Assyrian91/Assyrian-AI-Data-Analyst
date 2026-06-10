import io
from pathlib import Path
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────

def _looks_like_csv(raw_bytes: bytes) -> bool:
    try:
        sample = raw_bytes[:1024].decode(errors="ignore")
    except Exception:
        return False
    return "," in sample and "\n" in sample


def load_data(file_or_path) -> pd.DataFrame:
    if isinstance(file_or_path, (str, Path)):
        p = Path(file_or_path)
        s = p.suffix.lower()
        if s == ".csv":
            return pd.read_csv(p)
        if s in {".xls", ".xlsx"}:
            return pd.read_excel(p)
        if s == ".json":
            return pd.read_json(p)
        return pd.read_csv(p)

    name   = getattr(file_or_path, "name", None)
    suffix = Path(name).suffix.lower() if name else None
    raw    = file_or_path.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    bio = io.BytesIO(raw)

    if suffix == ".csv" or (suffix is None and _looks_like_csv(raw)):
        bio.seek(0); return pd.read_csv(bio)
    if suffix in {".xls", ".xlsx"}:
        bio.seek(0); return pd.read_excel(bio)
    if suffix == ".json":
        bio.seek(0); return pd.read_json(bio)
    bio.seek(0)
    try:
        return pd.read_csv(bio)
    except Exception:
        bio.seek(0); return pd.read_json(bio)


# ─────────────────────────────────────────────
# SMART COLUMN DETECTION  (domain-agnostic)
# ─────────────────────────────────────────────

# Hint keywords for each role — ordered by priority
_ROLE_HINTS = {
    "date": [
        "date", "time", "timestamp", "datetime", "created", "updated",
        "period", "month", "year", "day", "week", "at", "on",
    ],
    "value": [
        "amount", "revenue", "sales", "total", "price", "cost", "fee",
        "salary", "income", "spend", "budget", "profit", "loss",
        "score", "rate", "value", "sum", "qty", "quantity", "units",
        "count", "number", "num", "vol", "volume",
    ],
    "category": [
        "type", "category", "group", "segment", "class", "kind",
        "status", "stage", "label", "tag", "department", "region",
        "country", "city", "state", "product", "item", "name",
        "gender", "industry", "channel", "source", "medium",
    ],
    "id": [
        "id", "key", "code", "no", "number", "ref", "uuid",
        "customer", "user", "employee", "account", "order", "invoice",
        "client", "person", "contact",
    ],
    "text": [
        "description", "note", "comment", "detail", "summary",
        "title", "subject", "message", "feedback", "review",
    ],
}


def _is_date_col(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(30)
    if len(sample) == 0:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        return parsed.notna().mean() >= 0.7
    except Exception:
        return False


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Returns a dict with keys: date, value, category, id, text.
    Each maps to a list of matching column names, ranked by confidence.
    Also returns 'numeric' and 'categorical' for convenience.
    """
    result = {role: [] for role in _ROLE_HINTS}
    result["numeric"]     = df.select_dtypes(include="number").columns.tolist()
    result["categorical"] = []

    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        series    = df[col]

        # Date
        if _is_date_col(series):
            result["date"].append(col)
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            # check hints for value vs id
            matched = False
            for hint in _ROLE_HINTS["value"]:
                if hint in col_lower:
                    result["value"].insert(0, col)
                    matched = True
                    break
            if not matched:
                for hint in _ROLE_HINTS["id"]:
                    if hint in col_lower:
                        result["id"].append(col)
                        matched = True
                        break
            if not matched:
                result["value"].append(col)
            continue

        # String-based roles
        nunique = series.nunique(dropna=True)
        total   = len(series.dropna())

        for role in ("id", "category", "text"):
            for hint in _ROLE_HINTS[role]:
                if hint in col_lower:
                    result[role].append(col)
                    break
            else:
                continue
            break
        else:
            # fallback: low cardinality → category, high → text/id
            if nunique <= max(50, total * 0.05):
                result["category"].append(col)
            elif nunique == total:
                result["id"].append(col)
            else:
                result["text"].append(col)

    # Build categorical convenience list
    result["categorical"] = [
        c for c in df.columns
        if c not in result["numeric"] and c not in result["date"]
        and df[c].nunique(dropna=True) <= max(50, len(df) * 0.05)
    ]

    # Deduplicate each list while preserving order
    for k in result:
        seen = set()
        deduped = []
        for v in result[k]:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        result[k] = deduped

    return result


def get_primary_cols(col_info: dict) -> dict:
    """Convenience: pick the single best column for each role."""
    return {
        "date":     col_info["date"][0]     if col_info["date"]     else None,
        "value":    col_info["value"][0]    if col_info["value"]    else None,
        "category": col_info["category"][0] if col_info["category"] else None,
        "id":       col_info["id"][0]       if col_info["id"]       else None,
        "text":     col_info["text"][0]     if col_info["text"]     else None,
    }


# ─────────────────────────────────────────────
# SUMMARY FOR AI CONTEXT
# ─────────────────────────────────────────────

def build_data_summary(df: pd.DataFrame, col_info: dict) -> str:
    primary = get_primary_cols(col_info)
    lines   = []

    lines.append(f"Shape: {len(df):,} rows × {len(df.columns)} columns")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Detected roles: {primary}")

    # Numeric stats
    num_cols = col_info["numeric"]
    if num_cols:
        lines.append("\nNumeric summary:")
        lines.append(df[num_cols].describe().round(2).to_string())

    # Date range
    if primary["date"]:
        try:
            dates = pd.to_datetime(df[primary["date"]], errors="coerce").dropna()
            lines.append(f"\nDate range: {dates.min().date()} → {dates.max().date()}")
        except Exception:
            pass

    # Categorical samples
    for col in col_info["categorical"][:3]:
        top = df[col].value_counts().head(5)
        lines.append(f"\nTop values in '{col}': {top.to_dict()}")

    # Nulls
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        lines.append(f"\nMissing values: {nulls.to_dict()}")

    return "\n".join(lines)
