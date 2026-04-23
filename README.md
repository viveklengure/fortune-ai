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
FMP API → SQLite → SentenceTransformer embeddings → ChromaDB
                ↓
  LangChain ConversationalRetrievalChain + ConversationBufferMemory
                ↓
         Claude Sonnet → Streamlit (9 pages)
```

## RAG Pipeline

The Q&A page uses a proper conversational RAG setup:

- **Retrieval** — ChromaDB semantic search (top-5 company docs) via LangChain's native Chroma vectorstore retriever
- **Memory** — `ConversationBufferMemory` keeps full conversation history so follow-up questions work naturally (e.g. *"How do their margins compare?"* after asking about revenue)
- **Chain** — `ConversationalRetrievalChain` wires retriever → prompt → Claude in one declarative pipeline, replacing manual prompt assembly
- **Reset** — "Clear conversation" button wipes memory and starts a fresh session

## Pages

| # | Page | What it does |
|---|------|-------------|
| 1 | Dashboard | KPI cards, top-10 revenue chart, market cap scatter |
| 2 | RAG Chat | Conversational Q&A with memory over financial data |
| 3 | Anomaly Monitor | AI-narrated anomaly detection with severity tiers |
| 4 | Company Explorer | Per-company financials, charts, YoY trend analysis |
| 5 | Daily Digest | One-click AI portfolio executive summary |
| 6 | Forecasting | Linear regression + Claude analyst outlook |
| 7 | Peer Comparison | Side-by-side metrics + AI comparison |
| 8 | Portfolio Trends | Sparkline grid, top/bottom performers |
| 9 | Sector Analysis | Revenue and margin aggregates by sector |
| 10 | Company Report | AI analyst report: template-driven, grounded in historical data |

## Company Intelligence Report

Page 10 implements a template-driven AI analytics workflow:

1. **Data context** — historical financials (4 years) + current metrics pulled from SQLite for the selected company
2. **Template injection** — a structured output template is passed to Claude alongside the data, defining exactly which metrics to produce and in what format
3. **Grounded generation** — Claude fills in the template using only the real numbers, producing a 5-section report:
   - **Snapshot** — executive summary (revenue, net income, net margin, market cap)
   - **Historical Performance** — year-by-year table with YoY growth calculations
   - **Trend Analysis** — narrative on revenue momentum and margin direction
   - **Signal** — momentum, margin health, and valuation verdicts
   - **Analyst Commentary** — executive-friendly paragraph on risks and tailwinds
4. **Download** — report exportable as `.txt`

This pattern — historical context + structured template + current data — is the architecture behind automated analyst reports at firms like Bloomberg and JPMorgan.

---

## Project Structure

```
╔══════════════════════════════════════════════════════════════════╗
║                     🏦  fortune-ai/                              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  📱 app.py                 Streamlit UI — 10-page application    ║
║  🚀 run.py                 CLI entry point (setup + launch)      ║
║  📋 requirements.txt       Python dependencies                   ║
║                                                                  ║
╠══════════════╦═══════════════════════════════════════════════════╣
║  📂 src/     ║   AI & Data Pipeline Modules                      ║
╠══════════════╬═══════════════════════════════════════════════════╣
║              ║                                                   ║
║              ║  🌐 ingest.py      FMP API → SQLite ingestion     ║
║              ║  🧠 embed.py       Build docs + ChromaDB vectors  ║
║              ║  💬 rag.py         ConversationalRetrievalChain   ║
║              ║  📊 report.py      Template-driven AI reports     ║
║              ║  🔍 agent.py       Anomaly detection engine       ║
║              ║  📈 forecast.py    Linear regression + narrative  ║
║              ║  📰 digest.py      Daily portfolio summary        ║
║              ║  📉 trends.py      Sparklines + sector rollups    ║
║              ║                                                   ║
╠══════════════╩═══════════════════════════════════════════════════╣
║  📂 db/                                                          ║
║  ├── 🗄️  financial.db      SQLite — companies, financials,       ║
║  │                          metrics (19 Fortune 500 companies)   ║
║  └── 🔮 chroma/            ChromaDB vector store                ║
║                             (384-dim embeddings, 19 documents)   ║
╠══════════════════════════════════════════════════════════════════╣
║  📂 data/                                                        ║
║  ├── raw/                  Raw API responses                     ║
║  └── processed/            Cleaned intermediate data            ║
╠══════════════════════════════════════════════════════════════════╣
║  📄 .env.example           API key template                      ║
║  📄 PLATFORM_OVERVIEW.md   Full feature + AI stack docs          ║
║  📄 EXECUTION_GUIDE.md     Step-by-step technical walkthrough    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Docs

- [`PLATFORM_OVERVIEW.md`](PLATFORM_OVERVIEW.md) — full feature scope, data layer, AI stack, anomaly criteria
- [`EXECUTION_GUIDE.md`](EXECUTION_GUIDE.md) — step-by-step technical walkthrough of every module

---

## Tech Stack

Python · Claude Sonnet · LangChain · ChromaDB · sentence-transformers · Streamlit · Plotly · SQLite · NumPy · Financial Modeling Prep API
