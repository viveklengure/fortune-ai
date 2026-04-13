# Fortune AI — Financial Intelligence Platform

AI-powered financial analytics covering 19 Fortune 500 technology companies. Combines real market data, vector search, and Claude to deliver RAG-powered Q&A, anomaly detection, 2-year forecasting, peer comparison, and portfolio analysis — all in a 9-page Streamlit app.

---

## Quick Start

```bash
git clone https://github.com/viveklengure/fortune-ai.git
cd fortune-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
python run.py          # Option 3 → full setup, Option 4 → launch app
```

**API keys needed:**
- `ANTHROPIC_API_KEY` — [console.anthropic.com](https://console.anthropic.com)
- `FMP_API_KEY` — [financialmodelingprep.com/developer/docs](https://financialmodelingprep.com/developer/docs) (free tier)

---

## Architecture

```
FMP API → SQLite → sentence-transformers → ChromaDB
                ↓
         LangChain RAG → Claude API → Streamlit (9 pages)
```

## Pages

| # | Page | What it does |
|---|------|-------------|
| 1 | Dashboard | KPI cards, top-10 revenue chart, market cap scatter |
| 2 | RAG Chat | Natural language Q&A over financial data |
| 3 | Anomaly Monitor | AI-narrated anomaly detection with severity tiers |
| 4 | Company Explorer | Per-company financials, charts, YoY trend analysis |
| 5 | Daily Digest | One-click AI portfolio executive summary |
| 6 | Forecasting | Linear regression + Claude analyst outlook |
| 7 | Peer Comparison | Side-by-side metrics + AI comparison |
| 8 | Portfolio Trends | Sparkline grid, top/bottom performers |
| 9 | Sector Analysis | Revenue and margin aggregates by sector |

---

## Docs

- [`PLATFORM_OVERVIEW.md`](PLATFORM_OVERVIEW.md) — full feature scope, data layer, AI stack, anomaly criteria
- [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) — step-by-step technical walkthrough of every module

---

## Tech Stack

Python · Claude Sonnet · LangChain · ChromaDB · sentence-transformers · Streamlit · Plotly · SQLite · NumPy · Financial Modeling Prep API
