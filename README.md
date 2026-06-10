# 🧠 Assyrian-AI · Data Analyst

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Free AI](https://img.shields.io/badge/AI-Groq%20%7C%20OpenRouter-purple)

> A general-purpose AI-powered data analysis app. Upload **any** CSV, Excel, or JSON file and get instant charts, statistics, and AI-powered insights — completely free.

---

## ✨ Features

- 📁 **Any dataset** — retail, HR, finance, healthcare, logistics, surveys, sports, or anything else
- 🤖 **AI Chat** — ask questions about your data in plain English
- 📈 **Trends** — automatic time-series analysis
- 📊 **Distribution** — group breakdowns and histograms
- 🏆 **Top N** — rank any column by any metric
- 🔬 **Statistics** — correlation matrix, box plots, outlier detection
- 🔮 **Forecast** — 6-period trend forecast
- 🛠️ **Custom Chart Builder** — 8 chart types via dropdowns or natural language
- 🆓 **100% Free AI** — powered by Groq & OpenRouter (no credit card needed)

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Assyrian-AI/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free API key

| Provider | Link | Notes |
|----------|------|-------|
| **Groq** | [console.groq.com](https://console.groq.com) | No credit card · Very fast |
| **OpenRouter** | [openrouter.ai/keys](https://openrouter.ai/keys) | Free models available |

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`, paste your API key in the sidebar, and upload your file.

---

## 📁 Project Structure

```
ai-data-analyst/
├── app.py              # Main Streamlit app
├── data_loader.py      # Smart file loading + column detection
├── analysis.py         # Trends, distribution, statistics, forecast
├── charts.py           # All matplotlib visualizations
├── ai_engine.py        # Groq + OpenRouter AI integration
├── requirements.txt    # Dependencies
├── logo.jpeg           # Your logo (place in root folder)
└── README.md
```

---

## 🖥️ Windows Quick Run

```cmd
cd "C:\path\to\your\project"
streamlit run app.py
```

---

## 🤖 Supported AI Models

**Groq (Free)**
- `llama-3.3-70b-versatile` ← recommended
- `llama-3.1-8b-instant` (fastest)
- `mixtral-8x7b-32768`
- `gemma2-9b-it`

**OpenRouter (Free models)**
- `meta-llama/llama-3.3-70b-instruct:free`
- `mistralai/mistral-7b-instruct:free`
- `google/gemma-2-9b-it:free`
- `qwen/qwen-2.5-72b-instruct:free`

---

## 📊 Tested Datasets

| Domain | Example columns |
|--------|----------------|
| Retail | Date, Product, Quantity, Price, Country |
| HR | HireDate, Department, Salary, Performance |
| Finance | Date, Stock, Open, Close, Volume |
| Healthcare | Date, Diagnosis, Cost, Region |
| Logistics | ShipDate, Origin, Destination, Weight |

---

## 📝 License

MIT — free to use, modify, and distribute.

---

<div align="center">
Built by <b>Assyrian-AI</b> · Powered by Groq & OpenRouter
</div>