# Fortune AI — Platform Overview

> An end-to-end AI-powered financial intelligence platform covering 19 Fortune 500 technology companies. Combines real financial data ingestion, vector search, large language model reasoning, statistical modeling, and interactive visualization in a single Streamlit application.

---

## What This Platform Does

Fortune AI pulls live financial data from the Financial Modeling Prep API, stores it in a local SQLite database, indexes it in a ChromaDB vector store, and exposes nine AI-powered features through a web interface. No email, no scheduler — everything runs on demand through the browser.

---

## Companies Covered (19 Fortune 500 Tech)

| Ticker | Company | Ticker | Company |
|--------|---------|--------|---------|
| AAPL | Apple | NOW | ServiceNow |
| MSFT | Microsoft | WDAY | Workday |
| AMZN | Amazon | PLTR | Palantir |
| GOOGL | Alphabet (Google) | ADBE | Adobe |
| META | Meta | INTU | Intuit |
| ORCL | Oracle | QCOM | Qualcomm |
| CRM | Salesforce | AVGO | Broadcom |
| NVDA | Nvidia | IBM | IBM |
| NFLX | Netflix | SAP | SAP |
| SNOW | Snowflake | | |

---

## Data Layer

### What Data Is Collected

For each company, the platform ingests:

**Company Profile**
- Current stock price
- Market capitalization
- PE ratio and EPS
- 52-week high and low
- Sector classification

**Annual Income Statements (last 4 years)**
- Total revenue
- Net income
- Gross profit
- Operating income

**Derived Metrics (calculated, not pulled)**
- Revenue growth YoY (%)
- Net income growth YoY (%)
- Gross margin (%)
- Operating margin (%)
- Net margin (%)

### Where Data Is Stored

| Store | Purpose | Technology |
|-------|---------|------------|
| SQLite (`db/financial.db`) | Structured financial data, all queries, charts | SQLite via Python `sqlite3` |
| ChromaDB (`db/chroma/`) | Vector embeddings for semantic search (RAG) | ChromaDB local persistent client |

### Data Source

**Financial Modeling Prep (FMP) API** — free tier, `stable` endpoint  
- `/stable/profile` — company profile  
- `/stable/income-statement` — annual financials  
- 38 total API calls for 19 companies (2 per company)  
- Free tier: 250 requests/day

---

## AI & ML Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Encode financial documents into 384-dim vectors |
| Vector search | ChromaDB | Retrieve top-5 relevant company documents for RAG |
| LLM orchestration | LangChain (`langchain-anthropic`, `langchain-chroma`) | Chain retrieval + generation for RAG pipeline |
| Language model | Claude Sonnet (`claude-sonnet-4-20250514`) | All AI text generation |
| Statistical model | `numpy.polyfit` (degree=1) | Linear regression for 2-year revenue/income forecasting |

---

## Features Built

### 1. Financial Data Ingestion (`src/ingest.py`)
- Pulls profile + 4 years of income statements from FMP API for all 19 companies
- Calculates and stores YoY growth rates and profit margins
- Error handling: logs failures and continues, reports success count at end
- Rate-limiting: 0.5s delay between companies to respect API limits

### 2. Vector Embedding Pipeline (`src/embed.py`)
- Reads all financial data from SQLite
- Builds rich text documents per company: includes latest summary, all 4 years of historical data, and a narrative paragraph
- Encodes with `all-MiniLM-L6-v2` (~90MB model, runs locally, no API key needed)
- Stores 19 vectors in ChromaDB collection `fortune500_financials`

### 3. RAG Chat Interface (`src/rag.py` + Page 2)
- User asks a natural language question
- Question is encoded into a vector and matched against ChromaDB (top-5 cosine similarity)
- Top-5 company documents are injected into Claude's context
- Claude answers using only the retrieved context, cites company tickers
- Full chat history maintained in session state
- Returns: answer text + source tickers

### 4. Anomaly Detection Agent (`src/agent.py` + Page 3)
- Scans all 19 companies against 4 financial rules:
  - Revenue decline: YoY growth < -5%
  - Net income drop: YoY growth < -20%
  - Valuation outlier: PE ratio > 100 or < 0
  - Margin compression: Gross margin < 20%
- For each triggered rule, calls Claude to write a 2–3 sentence analyst narrative
- Returns severity (high/medium/low), metric value, and AI explanation per anomaly
- Displayed as expandable cards with color-coded severity badges

### 5. AI Executive Digest (`src/digest.py` + Page 5)
- Aggregates all 19 companies' latest financials
- Calls `detect_anomalies()` for current flags
- Sends full portfolio context to Claude
- Claude produces a structured executive summary:
  - Portfolio Overview
  - Top Performers (top 3 by revenue growth)
  - Anomalies & Watch List
  - Market Highlights
- Under 500 words, specific numbers, professional tone

### 6. Statistical Forecasting (`src/forecast.py` + Page 6, left column)
- Pulls last 4 years of actuals per ticker from SQLite
- Fits a linear regression (`numpy.polyfit`, degree=1) separately for:
  - Revenue
  - Net income
  - Gross profit
- Projects 2 years beyond the last actual data year
- Calculates R² for each metric as a confidence indicator
- Visualized as historical solid line + dashed forecast line + shaded forecast zone

### 7. AI Analyst Narrative (`src/forecast.py` + Page 6, right column)
- Completely separate from the statistical model
- Sends full 4-year income statement history + margins to Claude
- Claude writes a forward-looking 2-year analyst report:
  - Revenue Outlook
  - Profitability Trajectory
  - Key Risks
  - Key Opportunities
  - Analyst Verdict
- Results cached in session state per ticker — regeneration is opt-in

### 8. YoY Trend Analysis (`src/trends.py` + Page 4 enhancement)
- Calculates year-over-year dollar change and percentage change for:
  - Revenue, net income, gross profit, operating income
- Displayed as dual-axis charts (bar = absolute value, line = YoY %)
- Color-coded bars: green if YoY positive, red if negative
- Summary table with color-coded YoY% cells

### 9. Portfolio Trends (`src/trends.py` + Page 8)
- Loads YoY trends for all 19 companies
- Sparkline grid (4 per row): each card shows company name, latest revenue, revenue YoY%, net income YoY%, and a tiny revenue trend line
- Top 5 and Bottom 5 performers by revenue growth
- Portfolio stacked/grouped bar chart: all companies' revenue by year, toggle between stacked and grouped view

### 10. Sector Aggregates (`src/trends.py` + Page 9)
- Groups all companies by sector
- Sums revenue and net income per sector per year
- Sector summary cards: company count, total revenue, total net income
- Sector revenue trend chart (one line per sector)
- Margin comparison: grouped bar chart of average gross margin % and net margin % per sector
- Company breakdown: select a sector, view all companies ranked by revenue

### 11. Peer Comparison (Page 7)
- Select any two companies from dropdowns
- Side-by-side metric comparison with green/red highlights (better value wins)
  - Revenue, net income, market cap, gross margin, net margin, operating margin, revenue growth YoY
- Revenue trend comparison chart (both companies on one chart)
- Net income comparison chart
- Margins comparison: grouped bar chart (gross, operating, net margin)
- AI comparison: Claude writes a structured head-to-head analysis
  - Head-to-Head Summary
  - Strengths of Company A
  - Strengths of Company B
  - Which to Watch and Why

### 12. Dashboard (Page 1)
- 4 KPI metric cards: Total companies tracked, latest data date, anomalies detected, companies with positive revenue growth
- Top 10 companies by revenue: horizontal bar chart (Plotly)
- Market cap vs revenue scatter chart with ticker labels, colored by sector, sized by market cap

### 13. Company Explorer (Page 4)
- Dropdown to select any of the 19 companies
- Full profile: price, market cap, PE ratio, EPS, all three margins, 52-week range
- Revenue and net income line charts (4 years)
- Trend analysis section with 3 dual-axis charts + YoY summary table

---

## Application Structure

### Entry Points

| Command | What it does |
|---------|-------------|
| `python run.py` | Interactive CLI menu |
| `streamlit run app.py` | Launch web app directly |
| `python src/<module>.py` | Test any module standalone |

### CLI Menu Options

```
1. Ingest financial data       → runs src/ingest.py
2. Generate embeddings         → runs src/embed.py
3. Full setup (ingest + embed) → runs both in sequence
4. Start Streamlit app         → launches app.py
5. Test forecasting module     → runs statistical_forecast("AAPL")
6. Test trends module          → runs all three trend functions
7. Exit
```

### Streamlit Pages (9 total)

```
1. Dashboard          — KPI overview + charts
2. RAG Chat           — Natural language Q&A
3. Anomaly Monitor    — On-demand anomaly scan
4. Company Explorer   — Per-company deep dive + trends
5. Daily Digest       — AI portfolio summary
6. Forecasting        — Statistical model + AI outlook
7. Peer Comparison    — Side-by-side company analysis
8. Portfolio Trends   — Sparklines + performers + stacked chart
9. Sector Analysis    — Sector-level aggregates + breakdown
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data source | Financial Modeling Prep (FMP) API |
| Structured storage | SQLite |
| Vector storage | ChromaDB (local, no API key) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM orchestration | LangChain (`langchain-anthropic`, `langchain-chroma`) |
| Language model | Anthropic Claude Sonnet (`claude-sonnet-4-20250514`) |
| Statistical modeling | NumPy (`polyfit`) |
| Web framework | Streamlit |
| Charting | Plotly |
| Data manipulation | Pandas, NumPy |
| Environment config | python-dotenv |
| HTTP client | Requests |

---

## What This Demonstrates

1. **End-to-end RAG architecture** — data ingestion → vector indexing → semantic retrieval → LLM generation, built from scratch without a managed service
2. **Agentic LLM patterns** — rule-based detection feeding into LLM narration, combining deterministic logic with generative AI
3. **Hybrid forecasting** — statistical model (interpretable, with confidence scoring) paired with LLM qualitative reasoning — complementary, not competing approaches
4. **Production-aware design** — caching strategies, rate limit handling, graceful error handling, modular file structure, secrets management
5. **Full-stack AI application** — clean separation between data layer, AI/ML layer, and presentation layer across 2,700+ lines of Python

---

*Fortune AI — Built with Python · Claude API · LangChain · ChromaDB · Streamlit*
