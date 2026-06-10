import os

# ─────────────────────────────────────────────
# PROVIDER SELECTION
# Supports: Groq (free) and OpenRouter (free models)
# ─────────────────────────────────────────────

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen-2.5-72b-instruct:free",
]

SYSTEM_PROMPT = """You are an expert data analyst. You work with any kind of dataset —
sales, HR, finance, healthcare, logistics, surveys, sports, scientific data, or anything else.

You receive a structured summary of a dataset and answer questions about it.

Rules:
- Always be specific: reference actual column names, numbers, and values from the data.
- Structure your responses with short headers when covering multiple points.
- If asked for recommendations, be direct and actionable.
- Infer the domain from the data — never assume retail unless it clearly is retail.
- Keep responses under 400 words unless the user asks for a deep dive.
- If a question can't be answered from the data, say so clearly.
"""


def _groq_stream(messages, model):
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY not set. Get a free key at console.groq.com")
    client = Groq(api_key=key)
    with client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _openrouter_stream(messages, model):
    from openai import OpenAI
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set. Get a free key at openrouter.ai")
    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/Assyrian-AI",
            "X-Title": "Assyrian-AI Data Analyst",
        },
    )
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _build_messages(data_summary, analysis_summary, user_question, history):
    context = (
        f"=== DATASET ===\n{data_summary}\n\n"
        f"=== ANALYSIS ===\n{analysis_summary}"
    ).strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not history:
        messages.append({
            "role": "user",
            "content": f"{context}\n\n---\n\nFirst question: {user_question}",
        })
    else:
        if history and "DATASET" not in history[0]["content"]:
            history[0]["content"] = f"{context}\n\n---\n\n{history[0]['content']}"
        messages += history
        messages.append({"role": "user", "content": user_question})

    return messages


def stream_ai(data_summary, analysis_summary, user_question,
              history=None, provider="groq", model=None):
    """
    Unified streaming function.
    provider: "groq" or "openrouter"
    model: None = use first default for provider
    """
    messages = _build_messages(data_summary, analysis_summary, user_question, history or [])

    if provider == "groq":
        m = model or GROQ_MODELS[0]
        yield from _groq_stream(messages, m)
    elif provider == "openrouter":
        m = model or OPENROUTER_MODELS[0]
        yield from _openrouter_stream(messages, m)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_auto_insights(data_summary, analysis_summary, provider="groq", model=None):
    question = (
        "Analyse this dataset and give me:\n"
        "1. What kind of data this appears to be (domain / business context)\n"
        "2. 3 key findings with specific numbers\n"
        "3. The single biggest anomaly or red flag\n"
        "4. 2 actionable recommendations\n"
        "Be specific. Use actual values from the data."
    )
    return "".join(stream_ai(data_summary, analysis_summary, question,
                              provider=provider, model=model))


def parse_chart_request(prompt: str, df) -> tuple:
    """
    Parse a natural-language chart request without needing any API key.
    Returns (chart_type, x_col, y_col, top_n).
    """
    import re
    p          = prompt.lower()
    cols_lower = {c.lower().replace("_", " "): c for c in df.columns}
    for c in df.columns:
        cols_lower[c.lower()] = c

    chart_type = "Bar chart"
    if any(w in p for w in ["line", "trend", "over time", "monthly", "daily", "weekly"]):
        chart_type = "Line chart"
    elif any(w in p for w in ["scatter", " vs ", "versus", "compare"]):
        chart_type = "Scatter plot"
    elif any(w in p for w in ["histogram", "distribution", "frequency"]):
        chart_type = "Histogram"
    elif any(w in p for w in ["pie", "share", "proportion", "breakdown"]):
        chart_type = "Pie chart"
    elif any(w in p for w in ["horizontal", "hbar"]):
        chart_type = "Horizontal bar"
    elif "area" in p:
        chart_type = "Area chart"
    elif any(w in p for w in ["box", "outlier", "quartile"]):
        chart_type = "Box plot"

    found_cols = []
    for lc, orig in cols_lower.items():
        if lc in p and orig not in found_cols:
            found_cols.append(orig)

    x_col = found_cols[0] if found_cols else None
    y_col = found_cols[1] if len(found_cols) > 1 else None

    if not x_col:
        import pandas as pd
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols
                    and df[c].nunique() < min(50, len(df) * 0.1)]
        for kw in ["by ", "per ", "for each ", "grouped by "]:
            if kw in p:
                after = p.split(kw)[-1]
                for lc, orig in cols_lower.items():
                    if lc in after:
                        x_col = orig
                        break
        if not x_col and cat_cols:
            x_col = cat_cols[0]
        if not y_col and num_cols:
            y_col = num_cols[0]

    tn_match = re.search(r"top\s*(\d+)", p)
    top_n    = int(tn_match.group(1)) if tn_match else 15

    return chart_type, x_col, y_col, top_n
