"""
Fortune AI - Trends Analysis Module
YoY trends, portfolio trends, and sector aggregates.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"


def get_yoy_trends(ticker: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT year, revenue, net_income, gross_profit, operating_income "
        "FROM financials WHERE ticker = ? ORDER BY year ASC",
        (ticker,)
    ).fetchall()
    conn.close()

    years = [r["year"] for r in rows]
    fields = ["revenue", "net_income", "gross_profit", "operating_income"]
    result = {"ticker": ticker, "years": years}

    for field in fields:
        values = [r[field] for r in rows]
        yoy_change = [None]
        yoy_pct = [None]
        for i in range(1, len(values)):
            prev = values[i - 1]
            curr = values[i]
            if prev is not None and curr is not None and prev != 0:
                yoy_change.append(curr - prev)
                yoy_pct.append((curr - prev) / abs(prev) * 100)
            else:
                yoy_change.append(None)
                yoy_pct.append(None)
        result[field] = {
            "values": values,
            "yoy_change": yoy_change,
            "yoy_pct": yoy_pct,
        }

    return result


def get_portfolio_trends() -> list:
    conn = sqlite3.connect(DB_PATH)
    tickers = [r[0] for r in conn.execute("SELECT ticker FROM companies ORDER BY ticker").fetchall()]
    conn.close()
    return [get_yoy_trends(t) for t in tickers]


def get_sector_aggregates() -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    companies = conn.execute("SELECT ticker, sector FROM companies").fetchall()
    ticker_to_sector = {r["ticker"]: (r["sector"] or "Other") for r in companies}

    financials = conn.execute(
        "SELECT ticker, year, revenue, net_income FROM financials ORDER BY year ASC"
    ).fetchall()
    conn.close()

    # Group by sector → year
    sector_data: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"revenue": 0, "net_income": 0}))
    sector_tickers: dict[str, set] = defaultdict(set)

    for f in financials:
        sector = ticker_to_sector.get(f["ticker"], "Other")
        sector_tickers[sector].add(f["ticker"])
        year = f["year"]
        sector_data[sector][year]["revenue"] += f["revenue"] or 0
        sector_data[sector][year]["net_income"] += f["net_income"] or 0

    result = {}
    for sector, year_map in sector_data.items():
        years = sorted(year_map.keys())
        result[sector] = {
            "companies": sorted(sector_tickers[sector]),
            "years": years,
            "total_revenue": [year_map[y]["revenue"] for y in years],
            "total_net_income": [year_map[y]["net_income"] for y in years],
        }

    return result


if __name__ == "__main__":
    print("=== YoY Trends: AAPL ===")
    t = get_yoy_trends("AAPL")
    for year, rev, pct in zip(t["years"], t["revenue"]["values"], t["revenue"]["yoy_pct"]):
        pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
        print(f"  {year}: Revenue ${rev/1e9:.1f}B  YoY {pct_str}")

    print("\n=== Portfolio Trends (summary) ===")
    portfolio = get_portfolio_trends()
    for p in portfolio:
        if p["revenue"]["values"]:
            latest = p["revenue"]["values"][-1]
            pct = p["revenue"]["yoy_pct"][-1]
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A"
            print(f"  {p['ticker']:5}  Revenue ${latest/1e9:.1f}B  YoY {pct_str}")

    print("\n=== Sector Aggregates ===")
    sectors = get_sector_aggregates()
    for sector, data in sectors.items():
        latest_rev = data["total_revenue"][-1] if data["total_revenue"] else 0
        print(f"  {sector}: {len(data['companies'])} companies, latest revenue ${latest_rev/1e9:.1f}B")
