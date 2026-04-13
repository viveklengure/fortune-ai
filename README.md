# Fortune AI — Financial Intelligence Platform

An AI-powered financial intelligence platform covering Fortune 500 technology companies. Pulls real financial data, stores it locally, and provides RAG-powered Q&A, anomaly detection, and AI-generated executive digests — all inside a Streamlit web app.

## Architecture

```
yfinance → SQLite → sentence-transformers → ChromaDB
                ↓
         LangChain RAG → Claude API → Streamlit (5 pages)
```

## Setup

```bash
# 1. Clone / navigate to the project
cd fortune-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# Edit .env and add your Anthropic API key
```

## Usage

```bash
python run.py
```

Select **Option 3** (Full setup) on first run to ingest financial data and generate embeddings. This takes ~2–3 minutes.

Then select **Option 4** to launch the Streamlit app in your browser.

## Features

| Page | Description |
|------|-------------|
| Dashboard | KPI cards, top-10 revenue bar chart, market cap vs revenue scatter |
| RAG Chat | Ask natural-language questions about any company's financials |
| Anomaly Monitor | AI-narrated detection of revenue declines, margin compression, valuation outliers |
| Company Explorer | Per-company financial profile with 4-year revenue & net income charts |
| Daily Digest | One-click AI executive summary of the full portfolio |

## Companies Covered

AAPL, MSFT, AMZN, GOOGL, META, ORCL, CRM, NVDA, NFLX, SNOW, IBM, SAP, NOW, WDAY, PLTR, ADBE, INTU, QCOM, AVGO

## Interview Talking Points

- **End-to-end RAG pipeline**: Combines yfinance data ingestion, SQLite storage, ChromaDB vector search, LangChain orchestration, and Claude — demonstrating how to build a production-grade retrieval-augmented generation system from scratch.
- **Agentic anomaly detection**: The agent module autonomously identifies financial red flags across 19 companies and calls Claude to generate analyst-quality narratives, showcasing tool-augmented LLM reasoning.
- **Full-stack AI application**: Streamlit + Plotly frontend with modular Python backend illustrates how to ship an interactive AI product with clean separation of concerns (ingest, embed, RAG, agent, UI layers).
