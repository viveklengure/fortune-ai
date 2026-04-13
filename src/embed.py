"""
Fortune AI - Embedding Generation
Reads financial data from SQLite and stores vector embeddings in ChromaDB.
"""

import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"
CHROMA_PATH = BASE_DIR / "db" / "chroma"
COLLECTION_NAME = "fortune500_financials"


def fmt_billions(value: float | None) -> str:
    if value is None:
        return "N/A"
    b = value / 1e9
    return f"${b:.1f}B"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def build_document(row: dict, financials: list[dict]) -> str:
    name = row["name"]
    ticker = row["ticker"]
    market_cap = fmt_billions(row.get("market_cap"))
    pe = row.get("pe_ratio")
    pe_str = f"{pe:.1f}" if pe else "N/A"

    rev_growth = fmt_pct(row.get("revenue_growth_yoy"))
    ni_growth = fmt_pct(row.get("net_income_growth_yoy"))
    gross_margin = fmt_pct(row.get("gross_margin"))
    op_margin = fmt_pct(row.get("operating_margin"))
    net_margin = fmt_pct(row.get("net_margin"))

    # Most recent year
    latest = financials[0] if financials else {}
    revenue = fmt_billions(latest.get("revenue"))
    net_income = fmt_billions(latest.get("net_income"))

    # Multi-year historical breakdown (sorted oldest → newest)
    historical = sorted(financials, key=lambda x: x.get("year", 0))
    yearly_lines = []
    for f in historical:
        year = f.get("year", "N/A")
        r = fmt_billions(f.get("revenue"))
        ni = fmt_billions(f.get("net_income"))
        gp = fmt_billions(f.get("gross_profit"))
        oi = fmt_billions(f.get("operating_income"))
        yearly_lines.append(
            f"  {year}: Revenue {r}, Net Income {ni}, Gross Profit {gp}, Operating Income {oi}"
        )
    historical_text = "\n".join(yearly_lines) if yearly_lines else "  No historical data available."

    # Narrative
    trend = "positive" if (row.get("revenue_growth_yoy") or 0) > 0 else "declining"
    margin_health = "healthy" if (row.get("gross_margin") or 0) > 40 else "moderate"
    narrative = (
        f"{name} shows {trend} revenue momentum with YoY growth of {rev_growth}. "
        f"The company maintains {margin_health} margins with a gross margin of {gross_margin} "
        f"and an operating margin of {op_margin}. "
        f"Net income growth stands at {ni_growth}, reflecting "
        f"{'strong' if (row.get('net_income_growth_yoy') or 0) > 0 else 'pressured'} bottom-line performance. "
        f"With a market cap of {market_cap} and PE ratio of {pe_str}, the valuation reflects "
        f"{'premium' if (pe or 0) > 30 else 'value'} pricing in the market."
    )

    doc = (
        f"{name} ({ticker}) Financial Summary:\n"
        f"Latest ({latest.get('year', 'N/A')}): Revenue {revenue}, Net Income {net_income}, "
        f"Gross Margin {gross_margin}, Operating Margin {op_margin}, Net Margin {net_margin}, "
        f"Market Cap {market_cap}, PE Ratio {pe_str}.\n"
        f"YoY Revenue Growth: {rev_growth}, YoY Net Income Growth: {ni_growth}.\n"
        f"Annual History:\n{historical_text}\n"
        f"Analysis: {narrative}"
    )
    return doc


def run_embedding() -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    companies = conn.execute("""
        SELECT c.*, m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
    """).fetchall()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Reset collection to avoid stale data
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    for company in companies:
        row = dict(company)
        ticker = row["ticker"]
        print(f"Embedding {ticker}...", end=" ", flush=True)

        financials = conn.execute(
            "SELECT * FROM financials WHERE ticker = ? ORDER BY year DESC",
            (ticker,)
        ).fetchall()
        financials = [dict(f) for f in financials]

        doc = build_document(row, financials)
        embedding = model.encode(doc).tolist()

        collection.add(
            ids=[ticker],
            embeddings=[embedding],
            documents=[doc],
            metadatas=[{
                "ticker": ticker,
                "company_name": row.get("name", ""),
                "market_cap": float(row.get("market_cap") or 0),
                "revenue": float(financials[0].get("revenue") or 0) if financials else 0,
                "sector": row.get("sector", ""),
            }]
        )
        print("done")

    conn.close()
    print(f"\nEmbedded {len(companies)} companies into ChromaDB")


if __name__ == "__main__":
    run_embedding()
