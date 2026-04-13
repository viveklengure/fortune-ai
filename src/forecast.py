"""
Fortune AI - Forecasting Module
Statistical linear regression forecast + AI narrative via Claude.
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"

SYSTEM_PROMPT = (
    "You are a senior financial analyst. You are given historical financial data for a public "
    "technology company. Write a forward-looking 2-year outlook. Structure your response with "
    "these exact sections:\n"
    "## Revenue Outlook\n"
    "## Profitability Trajectory\n"
    "## Key Risks\n"
    "## Key Opportunities\n"
    "## Analyst Verdict\n"
    "Be specific with numbers. Reference actual historical figures. "
    "Do not fabricate data beyond what is provided. Professional tone."
)


def _r_squared(y_actual: list[float], y_pred: list[float]) -> float:
    y = np.array(y_actual, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def statistical_forecast(ticker: str, years_ahead: int = 2) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT year, revenue, net_income, gross_profit FROM financials "
        "WHERE ticker = ? ORDER BY year ASC",
        (ticker,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"error": f"No financial data found for {ticker}"}

    years = [r["year"] for r in rows]
    revenue = [r["revenue"] or 0 for r in rows]
    net_income = [r["net_income"] or 0 for r in rows]
    gross_profit = [r["gross_profit"] or 0 for r in rows]

    last_year = years[-1]
    forecast_years = list(range(last_year + 1, last_year + years_ahead + 1))
    all_years = years + forecast_years

    def project(values: list[float]) -> tuple[list[float], list[float], float]:
        x = np.array(years, dtype=float)
        y = np.array(values, dtype=float)
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        fitted = poly(x).tolist()
        projected = poly(np.array(forecast_years, dtype=float)).tolist()
        r2 = _r_squared(y, fitted)
        return projected, fitted, r2

    rev_proj, rev_fit, rev_r2 = project(revenue)
    ni_proj, ni_fit, ni_r2 = project(net_income)
    gp_proj, gp_fit, gp_r2 = project(gross_profit)

    return {
        "ticker": ticker,
        "last_actual_year": last_year,
        "historical": {
            "years": years,
            "revenue": revenue,
            "net_income": net_income,
            "gross_profit": gross_profit,
        },
        "forecast": {
            "years": forecast_years,
            "revenue": rev_proj,
            "net_income": ni_proj,
            "gross_profit": gp_proj,
        },
        "confidence": {
            "revenue_r2": round(rev_r2, 4),
            "net_income_r2": round(ni_r2, 4),
            "gross_profit_r2": round(gp_r2, 4),
        },
    }


def ai_forecast_narrative(ticker: str) -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    company = conn.execute(
        "SELECT * FROM companies WHERE ticker = ?", (ticker,)
    ).fetchone()
    financials = conn.execute(
        "SELECT * FROM financials WHERE ticker = ? ORDER BY year ASC", (ticker,)
    ).fetchall()
    metrics = conn.execute(
        "SELECT * FROM metrics WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn.close()

    if not company or not financials:
        return f"Insufficient data to generate outlook for {ticker}."

    name = company["name"]

    fin_lines = []
    for f in financials:
        rev = f"${f['revenue']/1e9:.1f}B" if f["revenue"] else "N/A"
        ni = f"${f['net_income']/1e9:.1f}B" if f["net_income"] else "N/A"
        gp = f"${f['gross_profit']/1e9:.1f}B" if f["gross_profit"] else "N/A"
        oi = f"${f['operating_income']/1e9:.1f}B" if f["operating_income"] else "N/A"
        fin_lines.append(f"  {f['year']}: Revenue {rev}, Net Income {ni}, Gross Profit {gp}, Operating Income {oi}")

    def fmt(v, pct=False):
        if v is None:
            return "N/A"
        return f"{v:.1f}%" if pct else f"${v/1e9:.1f}B"

    prompt = (
        f"Company: {name} ({ticker})\n"
        f"Sector: {company['sector']}\n"
        f"Market Cap: {fmt(company['market_cap'])}\n"
        f"Current Price: ${company['current_price']}\n\n"
        f"Historical Financials:\n" + "\n".join(fin_lines) + "\n\n"
        f"Key Metrics (most recent year):\n"
        f"  Revenue Growth YoY: {fmt(metrics['revenue_growth_yoy'] if metrics else None, pct=True)}\n"
        f"  Net Income Growth YoY: {fmt(metrics['net_income_growth_yoy'] if metrics else None, pct=True)}\n"
        f"  Gross Margin: {fmt(metrics['gross_margin'] if metrics else None, pct=True)}\n"
        f"  Operating Margin: {fmt(metrics['operating_margin'] if metrics else None, pct=True)}\n"
        f"  Net Margin: {fmt(metrics['net_margin'] if metrics else None, pct=True)}\n\n"
        f"Based on the above, write a 2-year forward-looking financial outlook."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    print("=== Statistical Forecast: AAPL ===")
    result = statistical_forecast("AAPL")
    print(f"Historical years: {result['historical']['years']}")
    print(f"Forecast years:   {result['forecast']['years']}")
    print(f"Revenue forecast: {[f'${v/1e9:.1f}B' for v in result['forecast']['revenue']]}")
    print(f"Confidence R²:    {result['confidence']}")

    print("\n=== AI Narrative: AAPL ===")
    narrative = ai_forecast_narrative("AAPL")
    print(narrative)
