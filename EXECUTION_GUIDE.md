# Fortune AI — Step-by-Step Execution Guide

A complete walkthrough of how the Fortune AI platform works end-to-end, from data ingestion to the live Streamlit app.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Step 1 — Data Ingestion (`src/ingest.py`)](#4-step-1--data-ingestion)
5. [Step 2 — Embedding Generation (`src/embed.py`)](#5-step-2--embedding-generation)
6. [Step 3 — RAG Pipeline (`src/rag.py`)](#6-step-3--rag-pipeline)
7. [Step 4 — Anomaly Detection Agent (`src/agent.py`)](#7-step-4--anomaly-detection-agent)
8. [Step 5 — Executive Digest (`src/digest.py`)](#8-step-5--executive-digest)
9. [Step 6 — Forecasting (`src/forecast.py`)](#9-step-6--forecasting)
10. [Step 7 — Trends Analysis (`src/trends.py`)](#10-step-7--trends-analysis)
11. [Step 8 — Streamlit App (`app.py`)](#11-step-8--streamlit-app)
12. [Running the Project](#12-running-the-project)
13. [API Keys & Rate Limits](#13-api-keys--rate-limits)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Architecture Overview

```
Financial Modeling Prep API
        │
        ▼
  src/ingest.py  ──────────────────────────►  db/financial.db (SQLite)
  (profile, income                              ├── companies table
   statements, metrics)                         ├── financials table
                                                └── metrics table
        │
        ▼
  src/embed.py  ───────────────────────────►  db/chroma/ (ChromaDB)
  (sentence-transformers                        └── fortune500_financials
   all-MiniLM-L6-v2)                                collection (19 vectors)
        │
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
   src/rag.py    src/agent.py   src/digest.py  src/forecast.py src/trends.py
   LangChain +   Rule-based      Claude API     numpy polyfit   YoY calc +
   ChromaDB +    anomaly scan    portfolio      + Claude API    sector agg
   Claude API    + Claude API    summary
        │              │              │              │              │
        └──────────────┴──────────────┴──────────────┴──────────────┘
                                      │
                                      ▼
                                   app.py
                              Streamlit (9 pages)
```

**Data flow summary:**
1. FMP API → SQLite (structured financial data)
2. SQLite → sentence-transformers → ChromaDB (semantic search index)
3. User query → ChromaDB retrieval → Claude → Answer
4. SQLite → rule checks → Claude narration → Anomaly report
5. SQLite → numpy regression → Statistical forecast
6. SQLite → Claude → AI analyst narrative

---

## 2. Project Structure

```
fortune-ai/
├── data/
│   ├── raw/                    # Reserved for raw API response caching
│   └── processed/              # Reserved for cleaned data exports
├── db/
│   ├── financial.db            # SQLite database (auto-created on first ingest)
│   └── chroma/                 # ChromaDB vector store (auto-created on embed)
├── src/
│   ├── __init__.py
│   ├── ingest.py               # FMP API → SQLite
│   ├── embed.py                # SQLite → ChromaDB
│   ├── rag.py                  # RAG Q&A pipeline
│   ├── agent.py                # Anomaly detection
│   ├── digest.py               # AI executive digest
│   ├── forecast.py             # Statistical + AI forecasting
│   └── trends.py               # YoY trends + sector aggregates
├── app.py                      # Streamlit 9-page application
├── run.py                      # CLI entry point
├── requirements.txt
├── .env                        # Your API keys (not committed)
├── .env.example                # Template
├── EXECUTION_GUIDE.md          # This document
└── README.md
```

---

## 3. Environment Setup

### Prerequisites
- Python 3.10+
- A free [Financial Modeling Prep](https://financialmodelingprep.com/developer/docs) API key
- An [Anthropic](https://console.anthropic.com/) API key

### Installation

```bash
# 1. Navigate to the project
cd fortune-ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
```

### Configure `.env`

```env
ANTHROPIC_API_KEY=sk-ant-...
FMP_API_KEY=your_fmp_key_here
```

---

## 4. Step 1 — Data Ingestion

**File:** `src/ingest.py`  
**Purpose:** Pull real financial data from Financial Modeling Prep API and store it in SQLite.

### What it does

1. Loops over 19 hardcoded Fortune 500 tech company tickers
2. For each ticker, makes two FMP API calls:
   - `/stable/profile` — current price, market cap, sector, 52-week range
   - `/stable/income-statement` — last 4 years of revenue, net income, gross profit, operating income
3. Calculates derived metrics: YoY growth rates, gross/operating/net margins
4. Writes everything to three SQLite tables

### SQLite Schema

```sql
-- Company profile (one row per ticker)
CREATE TABLE companies (
    ticker        TEXT PRIMARY KEY,
    name          TEXT,
    sector        TEXT,
    market_cap    REAL,
    pe_ratio      REAL,
    eps           REAL,
    week52_high   REAL,
    week52_low    REAL,
    current_price REAL,
    updated_at    TEXT
);

-- Annual income statement data (up to 4 rows per ticker)
CREATE TABLE financials (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker           TEXT,
    year             INTEGER,
    revenue          REAL,
    net_income       REAL,
    gross_profit     REAL,
    operating_income REAL,
    UNIQUE(ticker, year)
);

-- Most recent year derived metrics (one row per ticker)
CREATE TABLE metrics (
    ticker                TEXT PRIMARY KEY,
    revenue_growth_yoy    REAL,
    net_income_growth_yoy REAL,
    gross_margin          REAL,
    operating_margin      REAL,
    net_margin            REAL
);
```

### Run standalone

```bash
python -c "from src.ingest import run_ingestion; run_ingestion()"
```

### Expected output

```
Ingesting AAPL... done
Ingesting MSFT... done
...
Ingested 19 of 19 companies successfully
```

### Notes
- FMP free tier: 250 requests/day. Each company uses 2 requests = 38 total for 19 companies.
- Income statements may be unavailable for some tickers if the daily limit is hit. Re-run the next day for remaining tickers.
- All values stored in **raw dollars** (not billions). Divide by `1_000_000_000` for display.

---

## 5. Step 2 — Embedding Generation

**File:** `src/embed.py`  
**Purpose:** Convert financial data into vector embeddings and store in ChromaDB for semantic search.

### What it does

1. Reads all 19 companies from SQLite (joining companies + metrics + financials)
2. For each company, builds a rich text document containing:
   - Latest year summary (revenue, margins, market cap)
   - YoY growth rates
   - Full year-by-year historical breakdown (all 4 years)
   - A narrative summary paragraph
3. Encodes the document using `sentence-transformers` (`all-MiniLM-L6-v2` — a 384-dimension model, ~90MB)
4. Stores the embedding + document + metadata in ChromaDB collection `fortune500_financials`

### Sample embedded document

```
Apple Inc. (AAPL) Financial Summary:
Latest (2025): Revenue $416.2B, Net Income $112.0B, Gross Margin 46.9%,
Operating Margin 32.0%, Net Margin 26.9%, Market Cap $3.8T, PE Ratio N/A.
YoY Revenue Growth: +6.4%, YoY Net Income Growth: +19.5%.
Annual History:
  2022: Revenue $394.3B, Net Income $99.8B, Gross Profit $170.8B, Operating Income $119.4B
  2023: Revenue $383.3B, Net Income $97.0B, Gross Profit $169.1B, Operating Income $114.3B
  2024: Revenue $391.0B, Net Income $93.7B, Gross Profit $180.7B, Operating Income $123.2B
  2025: Revenue $416.2B, Net Income $112.0B, Gross Profit $195.2B, Operating Income $133.1B
Analysis: Apple Inc. shows positive revenue momentum...
```

### ChromaDB collection metadata per document

```python
{
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "market_cap": 3809702080166.0,
    "revenue": 416161000000.0,
    "sector": "Technology"
}
```

### Run standalone

```bash
python -c "from src.embed import run_embedding; run_embedding()"
```

### Expected output

```
Embedding AAPL... done
Embedding MSFT... done
...
Embedded 19 companies into ChromaDB
```

> Note: ChromaDB telemetry warnings (`capture() takes 1 positional argument`) are harmless and can be ignored.

---

## 6. Step 3 — RAG Pipeline

**File:** `src/rag.py`  
**Purpose:** Answer natural language questions about company financials using Retrieval-Augmented Generation.

### How RAG works here

```
User question
     │
     ▼
sentence-transformers encodes question → 384-dim query vector
     │
     ▼
ChromaDB cosine similarity search → top 5 most relevant company documents
     │
     ▼
Documents injected into Claude's context window as "Context:"
     │
     ▼
Claude answers using only the retrieved context
     │
     ▼
Returns: { "answer": str, "sources": [tickers] }
```

### System prompt

> "You are Fortune AI, a financial intelligence assistant. You have access to detailed financial data for Fortune 500 technology companies. Answer questions accurately using the provided context. Always cite which companies you are referencing. Format numbers clearly: use $B for billions, $M for millions, % for percentages. If you don't have data for something, say so clearly."

### Example queries

- *"Which company had the highest revenue growth?"*
- *"How has Amazon's net income changed over 4 years?"*
- *"Compare Microsoft and Google's profit margins"*
- *"Which companies have negative net income?"*

### Run standalone

```bash
python -c "
from src.rag import ask
result = ask('Which company had the highest revenue in 2025?')
print(result['answer'])
print('Sources:', result['sources'])
"
```

---

## 7. Step 4 — Anomaly Detection Agent

**File:** `src/agent.py`  
**Purpose:** Automatically scan all companies for financial red flags and generate AI narratives explaining each anomaly.

### Detection rules

| Anomaly Type | Condition | Severity |
|---|---|---|
| Revenue Decline | YoY revenue growth < -5% | High if < -15%, else Medium |
| Net Income Drop | YoY net income growth < -20% | High |
| Valuation Outlier | PE ratio > 100 or PE ratio < 0 | Medium |
| Margin Compression | Gross margin < 20% | Medium |

### Execution flow

```
For each company in SQLite:
  1. Pull metrics (revenue_growth_yoy, net_income_growth_yoy, pe_ratio, gross_margin)
  2. Apply 4 rule checks
  3. For each triggered rule → call Claude API with anomaly details
  4. Claude writes a 2-3 sentence factual narrative
  5. Append to results list
```

### Claude system prompt

> "You are a financial analyst. Write a brief, factual 2-3 sentence narrative explaining this financial anomaly. Be specific about the numbers. Do not speculate beyond what the data shows."

### Return structure

```python
[
    {
        "ticker": "SNOW",
        "company_name": "Snowflake Inc.",
        "anomaly_type": "Margin Compression",
        "severity": "medium",
        "metric_value": "Gross margin: 67.1%",
        "narrative": "Snowflake Inc. reported a gross margin of 67.1%..."
    }
]
```

### Run standalone

```bash
python -c "
from src.agent import detect_anomalies
results = detect_anomalies()
for a in results:
    print(f'[{a[\"severity\"].upper()}] {a[\"company_name\"]} - {a[\"anomaly_type\"]}')
"
```

---

## 8. Step 5 — Executive Digest

**File:** `src/digest.py`  
**Purpose:** Generate a single AI-written executive summary of the entire 19-company portfolio.

### Execution flow

1. Load all 19 companies' latest financials from SQLite
2. Call `detect_anomalies()` to get current anomaly list
3. Build a structured prompt with all portfolio data + anomalies
4. Send to Claude with executive digest system prompt
5. Return formatted markdown string

### Output sections

```markdown
## Portfolio Overview
## Top Performers (top 3 by revenue growth)
## Anomalies & Watch List
## Market Highlights
```

### Run standalone

```bash
python -c "from src.digest import generate_digest; print(generate_digest())"
```

---

## 9. Step 6 — Forecasting

**File:** `src/forecast.py`  
**Purpose:** Provide two independent 2-year forward projections — one statistical, one AI-generated.

### Function 1: `statistical_forecast(ticker, years_ahead=2)`

**Method:** Ordinary Least Squares linear regression using `numpy.polyfit(degree=1)`

**How it works:**

```
Historical data (4 years) → numpy.polyfit → slope + intercept
                                            → extrapolate 2 years forward
                                            → calculate R² for each metric
```

**R² interpretation:**
- R² = 1.0 → perfect linear trend, projection is highly reliable
- R² = 0.7–0.9 → moderate trend, reasonable projection
- R² < 0.5 → volatile or non-linear history, treat forecast with caution

**Why linear regression?**
- Simple and interpretable — no black box
- Works well for companies with steady growth (AAPL, MSFT)
- R² score gives an honest confidence signal
- Limitation: assumes the future continues the past trend linearly; does not account for market shocks, product cycles, or macro events

**Return structure:**

```python
{
    "ticker": "AAPL",
    "last_actual_year": 2025,
    "historical": {
        "years": [2022, 2023, 2024, 2025],
        "revenue": [394.3B, 383.3B, 391.0B, 416.2B],
        "net_income": [...],
        "gross_profit": [...]
    },
    "forecast": {
        "years": [2026, 2027],
        "revenue": [432.1B, 448.5B],
        "net_income": [...],
        "gross_profit": [...]
    },
    "confidence": {
        "revenue_r2": 0.891,
        "net_income_r2": 0.743,
        "gross_profit_r2": 0.952
    }
}
```

### Function 2: `ai_forecast_narrative(ticker)`

**Method:** Claude Sonnet (`claude-sonnet-4-20250514`) with a detailed financial prompt

**What Claude receives:**
- Company name, sector, market cap, current price
- Full 4-year income statement history
- YoY growth rates and margin percentages

**What Claude produces:**

```markdown
## Revenue Outlook
## Profitability Trajectory
## Key Risks
## Key Opportunities
## Analyst Verdict
```

**Difference from statistical model:**
- The statistical model extrapolates the mathematical trend
- Claude reasons qualitatively — it considers margin trajectories, growth deceleration patterns, and sector context
- Both are complementary: the stats model gives a number, Claude explains the story

### Run standalone

```bash
python src/forecast.py
```

---

## 10. Step 7 — Trends Analysis

**File:** `src/trends.py`  
**Purpose:** Calculate year-over-year trends for individual companies and aggregate them at portfolio and sector level.

### Functions

**`get_yoy_trends(ticker)`**

For each financial metric, computes:
- Absolute values per year
- YoY dollar change
- YoY percentage change

First year always has `None` for YoY fields (no prior year to compare).

**`get_portfolio_trends()`**

Runs `get_yoy_trends()` for all 19 companies. Cached in Streamlit with 3600s TTL since it loads all companies.

**`get_sector_aggregates()`**

Groups companies by sector (from `companies` table), then sums revenue and net income across all companies per year per sector. Returns one entry per sector with year-by-year totals.

### Run standalone

```bash
python src/trends.py
```

---

## 11. Step 8 — Streamlit App

**File:** `app.py`  
**Command:** `streamlit run app.py`

### Page Reference

| Page | Key Feature | Data Source |
|---|---|---|
| 1 — Dashboard | KPI cards, top-10 bar, market cap scatter | SQLite |
| 2 — RAG Chat | Natural language Q&A, chat history | ChromaDB + Claude |
| 3 — Anomaly Monitor | On-demand anomaly scan with AI narratives | SQLite + Claude |
| 4 — Company Explorer | Per-company profile + 4-year charts + dual-axis trend charts | SQLite |
| 5 — Daily Digest | One-click AI portfolio summary | SQLite + Claude |
| 6 — Forecasting | Statistical model charts + AI outlook side by side | SQLite + numpy + Claude |
| 7 — Peer Comparison | Side-by-side metrics, trend charts, AI comparison | SQLite + Claude |
| 8 — Portfolio Trends | Sparkline grid, top/bottom performers, stacked bar chart | SQLite |
| 9 — Sector Analysis | Sector cards, revenue trends, margin comparison, company breakdown | SQLite |

### Performance optimizations

- `@st.cache_data(ttl=300)` on `load_companies()` and `load_financials()` — avoids repeat SQLite queries
- `@st.cache_data(ttl=3600)` on `get_portfolio_trends()` and `get_sector_aggregates()` — expensive full-portfolio loads cached for 1 hour
- AI narrative results cached in `st.session_state` keyed by ticker — regeneration is opt-in, not automatic
- SQLite and ChromaDB only queried when the user navigates to the relevant page

---

## 12. Running the Project

### Full setup (first time)

```bash
cd fortune-ai
source venv/bin/activate
python run.py
```

Select:
- **Option 3** — Full setup (ingest + embed) — ~2 minutes
- **Option 4** — Start Streamlit app

### CLI menu

```
=== Fortune AI === Financial Intelligence Platform

  1. Ingest financial data
  2. Generate embeddings
  3. Full setup (ingest + embed)
  4. Start Streamlit app
  5. Test forecasting module
  6. Test trends module
  7. Exit
```

### Start app directly

```bash
streamlit run app.py
```

App runs at: `http://localhost:8501`

### Run individual modules

```bash
# Ingest only
python -c "from src.ingest import run_ingestion; run_ingestion()"

# Embed only
python -c "from src.embed import run_embedding; run_embedding()"

# Test RAG
python src/rag.py

# Test anomaly detection
python src/agent.py

# Test forecasting
python src/forecast.py

# Test trends
python src/trends.py
```

---

## 13. API Keys & Rate Limits

| Service | Free Tier Limits | Used For |
|---|---|---|
| Financial Modeling Prep | 250 requests/day | 2 requests per company × 19 = 38 total |
| Anthropic Claude | Pay-per-use | RAG chat, anomaly narratives, digest, forecasting, peer comparison |

### FMP rate limit strategy

- 0.5s delay between each company during ingestion
- If 402 error appears mid-ingestion, the daily limit is hit — wait 24 hours and re-run Option 1
- Already-ingested companies will be updated (INSERT OR REPLACE) and not double-counted

### Anthropic cost estimate (approximate)

| Feature | Tokens per call (est.) | Cost (claude-sonnet) |
|---|---|---|
| RAG Chat answer | ~2,000 | ~$0.003 |
| Anomaly narrative | ~500 | ~$0.001 |
| Executive Digest | ~3,000 | ~$0.005 |
| AI Forecast Narrative | ~2,500 | ~$0.004 |
| Peer Comparison | ~2,000 | ~$0.003 |

---

## 14. Troubleshooting

### Site can't be reached / port already in use

```bash
lsof -ti:8501 | xargs kill -9
streamlit run app.py
```

### FMP returns 402 Payment Required

Daily API limit hit. Wait until midnight UTC and re-run ingestion.

### FMP returns 403 Forbidden

Wrong base URL. The free tier uses `/stable/`, not `/api/v3/`. Check `FMP_BASE` in `ingest.py`.

### yfinance 429 Too Many Requests

Yahoo Finance rate-limits aggressively. This project uses FMP instead — do not revert to yfinance.

### ChromaDB telemetry warnings

```
Failed to send telemetry event: capture() takes 1 positional argument but 3 were given
```

Harmless. ChromaDB version mismatch in telemetry library. Does not affect functionality.

### "No data found" message in Streamlit

Run Option 3 from `run.py` first. The SQLite database must exist before the app can display data.

### Markdown italics in AI output

Claude sometimes uses underscores in metric names (e.g. `revenue_growth_yoy`). These are escaped automatically before rendering in the Forecasting page. If you see this on other pages, the same `re.sub(r'(?<!\s)_(?!\s)', r'\\_', text)` fix applies.

---

*Built with Python 3.11 · Claude Sonnet · LangChain · ChromaDB · Streamlit · Plotly · Financial Modeling Prep API*
