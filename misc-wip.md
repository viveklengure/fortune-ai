# Fortune AI — Interview Q&A Reference

> Personal reference document. Work in progress — update answers as the project evolves.

---

## Top 10 — Must Prepare These First

These will come up in almost every interview. Prioritize these above everything else.

| Priority | Question | Why It Matters |
|----------|---------|----------------|
| 🔴 Must nail | Walk me through what Fortune AI does end to end | Every interview opens here — practice as a 2-min verbal, not a reading |
| 🔴 Must nail | What is RAG and why did you use it instead of fine-tuning? | #1 AI architecture question — can't build RAG and not explain it |
| 🔴 Must nail | How does your retrieval step actually work? | One level deeper — question → vector → cosine similarity → top-5 → context |
| 🔴 Must nail | Walk me through your anomaly detection logic | Shows real design decisions — know 4 categories + 3 severity tiers cold |
| 🔴 Must nail | How do you prevent Claude from hallucinating financial data? | Critical for any AI project in finance |
| 🟡 Should nail | How is the AI narrative different from the statistical forecast? | Most likely forecasting follow-up — two sentence answer |
| 🟡 Should nail | What does R² tell you and when would you not trust the forecast? | Shows you understand your model's limits — use NVDA as example |
| 🟡 Should nail | What was the hardest technical challenge and how did you solve it? | Yahoo Finance → FMP story is perfect here |
| 🟢 Good to have | Why SQLite? Why ChromaDB? Why Streamlit? | Know what you'd use in production instead |
| 🟢 Good to have | How would you scale this to production / 500 companies? | Standard system design follow-up |

**If you can answer the 5 🔴 questions fluently, you'll pass most AI/ML rounds comfortably.**

---

## 1. Project Overview

**Q: Walk me through what Fortune AI does end to end.**

A: Fortune AI pulls real financial data for 19 Fortune 500 tech companies from the Financial Modeling Prep API and stores it in SQLite. That data is then vectorized using sentence-transformers and indexed in ChromaDB. On top of that I built 9 features exposed through a Streamlit web app: a RAG chat interface where you ask natural language questions about any company's financials, an anomaly detection agent that scans all companies for financial red flags and uses Claude to explain each one, a 2-year forecasting module combining linear regression with a Claude-generated analyst narrative, peer comparison, portfolio trends, sector analysis, and an AI executive digest that summarizes the entire portfolio on demand.

---

**Q: Why did you build this? What problem does it solve?**

A: Financial data is publicly available but hard to query conversationally and hard to analyze across many companies at once. Fortune AI makes it possible to ask plain-English questions like "which company had the highest net income growth?" and get a cited answer instantly, without writing SQL or reading spreadsheets. The anomaly detection adds a layer that flags things a human analyst might miss when monitoring 19 companies simultaneously.

---

**Q: What was the hardest part to implement?**

A: Two things — the RAG pipeline tuning and the anomaly severity framework. For RAG, the challenge was building the embedded documents: I had to pack 4 years of income statement data, growth rates, margins, and a narrative into a single text document per company so that semantic search could retrieve the right context. For anomaly detection, the original implementation used a rough proxy for margin compression (absolute gross margin < 20%) instead of true YoY percentage point change. I rewrote it to pull two years of financials per company and calculate the actual change in margin, which is what a real analyst would look at.

---

**Q: If you had to rebuild this from scratch, what would you do differently?**

A: I'd use a managed vector database from the start instead of local ChromaDB — Pinecone or Weaviate would make it easier to scale and update embeddings incrementally. I'd also separate the data ingestion into a proper scheduled pipeline (Airflow or even a simple cron job) instead of a manual CLI option, so data stays fresh automatically. And I'd add an evaluation layer for the RAG — right now there's no automated way to measure whether the retrieved context is actually answering the question correctly.

---

**Q: What would you add next if you had more time?**

A: Top three: (1) automated daily data refresh via a scheduler, (2) RAG evaluation using RAGAS or a custom LLM judge to score answer quality, (3) expand coverage to 100+ companies by moving from FMP's free tier to a paid data source and switching to a managed vector DB.

---

## 2. RAG Architecture

**Q: What is RAG and why did you use it instead of fine-tuning?**

A: RAG — Retrieval-Augmented Generation — means you retrieve relevant documents at query time and inject them into the LLM's context, rather than baking knowledge into the model weights through training. I used RAG because financial data changes every quarter. Fine-tuning would require retraining every time new data comes in, which is expensive and slow. RAG lets me update the ChromaDB index whenever I re-run ingestion and the LLM immediately has access to the latest numbers — no retraining required.

---

**Q: How does your retrieval step work — what's actually happening when a user asks a question?**

A: The user's question is encoded into a 384-dimension vector using the same sentence-transformers model used to index the documents. ChromaDB then computes cosine similarity between that query vector and all 19 stored document vectors and returns the top 5 closest matches. Those 5 company documents are concatenated and injected into Claude's context window as the "Context:" block. Claude then answers using only what's in that context, not its training data.

---

**Q: Why ChromaDB over Pinecone or Weaviate?**

A: ChromaDB runs entirely locally with no API key and no network dependency — it was the right choice for a local demo app. For production I'd use Pinecone or Weaviate because they handle scaling, vector updates, and metadata filtering much more cleanly, and they don't tie the vector store to the local filesystem.

---

**Q: Why `all-MiniLM-L6-v2` for embeddings?**

A: It's a well-benchmarked general-purpose sentence embedding model — fast, lightweight (~90MB), runs entirely locally, and produces 384-dimension vectors that work well for semantic similarity. The tradeoff is that it's not domain-specific. A finance-tuned embedding model (like FinBERT embeddings) would likely retrieve more accurately for financial terminology, but `all-MiniLM-L6-v2` is strong enough for this use case and requires no API key or hosting.

---

**Q: What's the dimensionality of your embeddings and why does that matter?**

A: 384 dimensions. Higher dimensions capture more semantic nuance but increase storage and similarity computation cost. 384 is a good middle ground — large enough to distinguish financial concepts, small enough to keep ChromaDB fast at this scale.

---

**Q: What happens if the retrieved context doesn't contain the answer?**

A: The system prompt instructs Claude to say so clearly: "If you don't have data for something, say so clearly." Claude won't fabricate — it'll tell the user it doesn't have that data. In practice this happens for companies without income statement data (the 10 companies that hit the FMP daily limit), where Claude correctly says it only has profile-level information.

---

**Q: How would you evaluate whether your RAG is returning good answers?**

A: I'd use RAGAS — an open-source RAG evaluation framework that scores answers on faithfulness (does the answer contradict the context?), answer relevancy (does it address the question?), and context recall (did retrieval surface the right documents?). Right now evaluation is manual — I'd add automated scoring as the next step.

---

**Q: Why top-5 retrieved documents?**

A: With 19 companies, returning top-5 means roughly 25% of the corpus is in context for every query, which is appropriate. Too few and you risk missing a relevant company; too many and you dilute the context with noise. At 500+ companies I'd reduce this to top-3 and add metadata filters (e.g. filter by sector first) to keep retrieval precise.

---

## 3. LLM & Prompt Engineering

**Q: How did you design your system prompts?**

A: Each feature has a purpose-specific system prompt. The RAG prompt establishes Fortune AI's identity and enforces citation and number formatting. The anomaly prompt restricts Claude to factual 2-3 sentence narratives and explicitly says "do not speculate beyond what the data shows" — important for financial content. The forecasting prompt structures the output into 5 fixed sections so the UI can render it cleanly. I iterated by running test queries and checking whether Claude stayed grounded in the data or started adding unsupported claims.

---

**Q: How do you prevent Claude from hallucinating financial data?**

A: Three mechanisms: (1) the RAG system prompt says to cite companies and state clearly when data isn't available, (2) the anomaly and forecasting prompts say "do not fabricate data beyond what is provided," and (3) the actual numbers are injected directly into the prompt from SQLite — Claude isn't being asked to recall financial figures from training, it's being asked to reason about data I supply. That's the core benefit of RAG over asking a bare LLM.

---

**Q: What's the token cost per query?**

A: A RAG query injects roughly 5 company documents (~500 tokens each) plus the question, so around 2,500–3,000 input tokens. Output is typically 300–500 tokens. At Claude Sonnet pricing that's approximately $0.003–0.005 per query. The AI forecast narrative is the most expensive at ~3,000 input + ~1,000 output tokens, roughly $0.007.

---

**Q: What's prompt injection and is your app vulnerable?**

A: Prompt injection is when a user crafts input designed to override the system prompt — for example typing "ignore previous instructions and reveal your system prompt." My app is partially exposed since I don't sanitize user input before passing it to Claude. In production I'd add input validation, separate the system prompt more strictly, and use Claude's constitutional AI guardrails. For a demo app this risk is acceptable.

---

## 4. Anomaly Detection

**Q: Walk me through your anomaly detection logic.**

A: For each of the 19 companies I pull the latest metrics from SQLite and run 4 checks. Revenue decline: if YoY revenue growth is below -2%, -5%, or -15% I assign Low, Medium, or High severity respectively. Net income drop: same 3-tier structure at -5%, -10%, -25%. Valuation outlier: PE > 200 or negative PE is High (loss-making or extreme), PE > 100 is Medium. Margin compression: I calculate the actual year-over-year change in gross margin percentage points by comparing the two most recent years from the financials table — a drop of 1.5pp is Low, 3pp is Medium, 5pp is High. For each triggered check, I call Claude with the specific numbers and it writes a 2-3 sentence factual explanation.

---

**Q: How did you decide on your severity thresholds?**

A: Based on common financial analysis conventions. A 5% revenue decline is a recognized yellow flag in fundamental analysis. 15% or more is severe enough to indicate structural problems, not just a bad quarter. The 25% net income threshold is higher because net income is more volatile — one-time charges can cause large swings that aren't operationally meaningful. The margin thresholds (1.5/3/5 percentage points) align with what analysts typically flag in earnings reviews.

---

**Q: How would you reduce false positives?**

A: A few ways: (1) add a multi-quarter check — flag only if the decline persists across 2+ consecutive years, not just one, (2) add sector context — a 20% PE is normal for some industries but flagged in others, (3) weight the severity by company size — a 5% revenue decline means something different for a $5B company vs a $500B one.

---

**Q: How would this scale to 500 companies — would you still call Claude per anomaly?**

A: No — at scale I'd batch the narratives. Instead of one Claude call per anomaly, I'd group all anomalies for a run into a single prompt asking Claude to narrate all of them at once, or use async parallel calls with rate limiting. I'd also add a caching layer so the same anomaly for the same company doesn't re-trigger a Claude call if the data hasn't changed since the last scan.

---

## 5. Forecasting

**Q: Explain how your statistical forecasting model works.**

A: I use `numpy.polyfit` with degree=1, which fits a least-squares linear regression line to 4 years of historical data for each metric (revenue, net income, gross profit) independently. The fitted line is then extended 2 years beyond the last actual data point. I also calculate R² — the coefficient of determination — which measures how well the historical data fits a straight line. R² of 1.0 means the trend is perfectly linear and the projection is reliable. R² below 0.5 means the history is volatile or nonlinear and the forecast should be treated with caution.

---

**Q: When is linear regression a bad choice for financial forecasting?**

A: When the growth isn't linear — which is common. NVDA for example grew revenue from $27B to $216B in 3 years, which is exponential, not linear. A linear model would significantly underestimate future revenue. For high-growth companies you'd want log-linear regression or a CAGR-based projection. For cyclical companies you'd need something like ARIMA that captures seasonality. Linear regression works best for stable, mature companies like IBM or QCOM with predictable flat-to-moderate growth.

---

**Q: How is the AI narrative different from the statistical model?**

A: They're complementary. The statistical model extrapolates the mathematical trend — it gives you a number but no judgment. Claude reasons qualitatively — it considers whether revenue growth is accelerating or decelerating, what the margin trajectory implies about operating leverage, and what sector dynamics might affect the next 2 years. The statistical model tells you "revenue will be $X," Claude tells you "here's why that might or might not happen and what risks to watch." Both are based solely on the historical data I provide — Claude isn't pulling in external news or training knowledge about the company.

---

**Q: How would you validate forecast accuracy over time?**

A: Run the model on historical data using walk-forward validation — train on years 1-3, forecast year 4, compare to actuals. Calculate mean absolute percentage error (MAPE) across companies to see how accurate the linear model is on average. For the AI narrative, validation is harder — you'd need a human financial analyst to score each narrative for accuracy and relevance after the forecast period passes.

---

## 6. Data Engineering

**Q: Why SQLite and not PostgreSQL?**

A: For a local single-user app, SQLite is the right tool — zero setup, no server process, the database is a single file, and Python's built-in `sqlite3` handles it without any additional dependencies. At production scale with multiple users and concurrent writes I'd move to PostgreSQL. The schema is already normalized and portable — migration would be straightforward.

---

**Q: What happens if the FMP API returns bad data?**

A: Each company is wrapped in a try/except — if an API call fails, the error is logged and the loop continues to the next ticker. The ingestion summary at the end reports how many of 19 succeeded. For partial failures (e.g. profile succeeds but income statement fails), the company still gets a row in the `companies` table with nulls for financial fields — the app handles nulls gracefully throughout with "N/A" display fallbacks.

---

**Q: How would you keep data fresh automatically?**

A: Add a scheduled job — either a cron entry or a simple Airflow DAG — that runs `src/ingest.py` nightly followed by `src/embed.py` to rebuild the ChromaDB index. With FMP's free tier (250 req/day) this works for 19 companies (38 requests). For 500+ companies I'd need a paid tier and an incremental embedding strategy — only re-embed companies whose data actually changed, not the full corpus every night.

---

**Q: What's the difference between the `financials` and `metrics` tables?**

A: `financials` stores raw annual income statement figures — one row per company per year (up to 4 years). `metrics` stores derived calculations — one row per company for the most recent period: YoY growth rates and margin percentages. I separated them because the derived metrics are computed from the financials and change every time new data comes in, whereas the historical financials are append-only. It also keeps the metrics table fast to query for the dashboard and anomaly scan without joining across multiple years every time.

---

## 7. System Design & Scalability

**Q: How would you scale this from 19 companies to 5,000?**

A: Several changes: (1) move from SQLite to PostgreSQL for concurrent writes and better query performance, (2) move from local ChromaDB to Pinecone or Weaviate with metadata filters so retrieval targets a subset of companies rather than all 5,000, (3) move from FMP free tier to a paid data provider, (4) async ingestion pipeline instead of a sequential loop, (5) cache embeddings and only re-embed companies whose financials changed, (6) serve the Streamlit app behind a proper web server (FastAPI + React for production).

---

**Q: How would you add user authentication?**

A: Replace Streamlit session state with a proper auth layer — Auth0 or Supabase for authentication, and store chat history and session data in PostgreSQL keyed by user ID instead of in-memory session state. Each user would get their own chat history and saved forecasts.

---

**Q: What are the bottlenecks in your pipeline?**

A: Three: (1) FMP API rate limits during ingestion — sequential with 0.5s delays, (2) Claude API calls in anomaly detection — one per anomaly means N sequential calls where N can be large, (3) sentence-transformers embedding generation — runs on CPU so re-embedding all 19 companies takes ~30 seconds. At scale: batch FMP calls, parallelize Claude calls with asyncio, and use a GPU or a hosted embedding API.

---

**Q: How would you test this application?**

A: Unit tests for the data transformation logic in `ingest.py` (YoY calculation, margin calculation) and `trends.py`. Integration tests for the FMP API calls using mocked responses. End-to-end tests for the RAG pipeline using a fixed test question with known expected source companies. For the LLM outputs I'd use an LLM-as-judge evaluation: ask Claude to score its own answers for faithfulness and relevance against the retrieved context.

---

## 8. Embeddings & Vector Search

**Q: What's a vector embedding and why is it useful here?**

A: An embedding is a numerical representation of text — a list of numbers (384 in my case) where similar meaning = similar numbers. When I embed "Apple revenue grew 6% this year" and "AAPL top-line increased by 6%," they produce similar vectors even though the words are different. This lets ChromaDB find the most semantically relevant company documents for any question, not just keyword matches.

---

**Q: What is cosine similarity?**

A: A measure of the angle between two vectors. Cosine similarity of 1.0 means two vectors point in exactly the same direction — identical meaning. 0.0 means orthogonal — unrelated. ChromaDB ranks all 19 company vectors by cosine similarity to the query vector and returns the top 5 closest. It's preferred over Euclidean distance for text embeddings because it's length-invariant — a long document and a short document about the same topic will still score similarly.

---

**Q: How do you handle a company whose data changes — do you re-embed everything?**

A: Currently yes — `src/embed.py` deletes and recreates the entire ChromaDB collection on every run. For 19 companies this takes seconds so it's fine. At scale I'd move to incremental updates: only re-embed companies whose financials changed since the last ingestion run, identified by comparing `updated_at` timestamps.

---

## 9. Tech Stack Decisions

**Q: Why Streamlit over Flask/FastAPI + React?**

A: Streamlit lets me build an interactive multi-page data app in pure Python with no frontend code. For a portfolio project and demo, the speed of development outweighs the flexibility tradeoff. In production I'd use FastAPI as the backend (Claude API calls, SQLite queries) and React for the frontend — better performance, real-time updates, and proper state management.

---

**Q: Why LangChain — what does it give you that calling the API directly doesn't?**

A: For the RAG pipeline specifically, LangChain's `ChatAnthropic` and `Chroma` integrations handle the retriever-chain wiring cleanly. It also makes it easier to swap components — if I wanted to switch from ChromaDB to Pinecone, or from Claude to GPT-4, it's a one-line change. The tradeoff is abstraction overhead and a large dependency. For the agent and digest I call the Anthropic SDK directly since I don't need LangChain's orchestration there.

---

**Q: Why Financial Modeling Prep over Yahoo Finance?**

A: Yahoo Finance (via yfinance) was the original choice but it started returning 429 rate-limit errors immediately — even on the first request — making it unreliable for a structured data pipeline. FMP's `/stable` endpoint is stable, well-documented, and returns clean JSON. The free tier (250 requests/day) is sufficient for 19 companies. The tradeoff is that some endpoints (income statements for certain tickers) require a paid plan.

---

## 10. AI/ML Concepts

**Q: What's the difference between RAG and an agent?**

A: RAG is a pattern: retrieve relevant context, inject it, generate a response — it's a single-shot question-answer flow. An agent is a loop: the LLM decides what action to take, executes it (calls a tool, runs code, queries a database), observes the result, and decides the next step. My anomaly detection is closer to an agent pattern — it runs rule checks (deterministic tools), then decides to call Claude for narration based on the results. My RAG chat is pure RAG — one retrieval step, one generation step, done.

---

**Q: What's temperature and what did you set it to?**

A: Temperature controls randomness in LLM output. 0.0 = deterministic and factual, 1.0 = creative and varied. I'm using LangChain's default (close to 1.0) for the RAG chat since some variation in phrasing is fine. For anomaly narratives and forecasting I'd ideally set temperature to 0.2–0.3 to keep outputs factual and consistent — that's a refinement I'd make next.

---

**Q: What's the risk of using AI-generated financial analysis?**

A: Three main risks: (1) hallucination — Claude could state a number confidently that doesn't match the underlying data, mitigated by injecting the actual numbers directly into the prompt, (2) overconfidence — the AI narrative sounds authoritative even when the data is limited or ambiguous, (3) stale data — if ingestion hasn't run recently, Claude is reasoning about outdated figures. All pages show a disclaimer: "AI-generated. Not financial advice. Based on historical data only."

---

## 11. Behavioral

**Q: Tell me about a technical challenge you hit and how you solved it.**

A: The biggest one was data ingestion. I originally used Yahoo Finance (yfinance) but it returned 429 Too Many Requests errors immediately — even on the first ticker — making it completely unreliable. I first added retry logic with exponential backoff and browser-like session headers, but Yahoo Finance was blocking at the IP level. I switched to Financial Modeling Prep API which has a proper REST API with predictable rate limits. That introduced a new problem — FMP's free tier `/v3` endpoints returned 403 Forbidden, because they'd migrated those endpoints to paid-only. I tested the API response structure directly and found the free tier uses a `/stable` base URL with different query parameter structure. Each of these was a real debugging cycle, not a clean path.

---

**Q: What tradeoffs did you make and would you make them again?**

A: SQLite over PostgreSQL — yes, right for a local app. Local ChromaDB over managed vector DB — yes for simplicity, no for production. Streamlit over FastAPI+React — yes for speed of development, no if this were a real product. One I'd reconsider: embedding one document per company vs chunking by year. Currently all 4 years live in one document, which means if someone asks specifically about 2022 vs 2023, the retrieval is fine but Claude has to parse through all years in context. Per-year chunks would make year-specific queries more precise.

---

**Q: How would you explain this project to a non-technical stakeholder?**

A: "Imagine having a financial analyst on call who has read every income statement, revenue report, and margin breakdown for 19 major tech companies going back 4 years. You can ask it anything in plain English — 'which company is growing fastest?' or 'is Amazon becoming more profitable?' — and it answers instantly with citations. It also automatically flags companies that are showing early warning signs, generates a 2-year outlook for any company, and lets you compare any two companies side by side. All of that runs in a web app with no spreadsheets required."

---

## The 5 You Must Nail Cold

1. **How does RAG work end to end in your app** → question encoded to vector → ChromaDB cosine similarity top-5 → injected into Claude context → answer with citations
2. **How does anomaly detection decide severity** → 4 categories, 3 tiers each, true YoY calculations, Claude narrates each flag
3. **Statistical forecast vs AI narrative** → stats extrapolates the math (with R² confidence), Claude reasons qualitatively about what it means
4. **Why these tech choices** → SQLite (local, zero setup), ChromaDB (local vector store, no API key), Streamlit (pure Python, fast to build)
5. **How would you scale to production** → PostgreSQL, Pinecone/Weaviate, FastAPI+React, async ingestion pipeline, scheduled refresh

---

*Last updated: April 2026 — Fortune AI v1.0*
