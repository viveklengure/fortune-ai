"""
Fortune AI - Anomaly Detection Agent
Scans financial metrics and uses Claude to narrate anomalies.

Severity framework:
  High   — significant deterioration requiring immediate attention
  Medium — notable concern worth monitoring
  Low    — borderline signal, watch but not alarming
"""

import os
import sqlite3
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"

SYSTEM_PROMPT = (
    "You are a financial analyst. Write a brief, factual 2-3 sentence narrative explaining "
    "this financial anomaly. Be specific about the numbers. "
    "Do not speculate beyond what the data shows."
)

# ── Severity thresholds ────────────────────────────────────────────────────────
# Revenue Decline
REV_HIGH   = -15.0   # YoY growth < -15%  → High
REV_MEDIUM = -5.0    # YoY growth < -5%   → Medium
REV_LOW    = -2.0    # YoY growth < -2%   → Low (slowing but not declining sharply)

# Net Income Drop
NI_HIGH    = -25.0   # YoY growth < -25%  → High
NI_MEDIUM  = -10.0   # YoY growth < -10%  → Medium
NI_LOW     = -5.0    # YoY growth < -5%   → Low

# Valuation Outlier (PE ratio)
PE_HIGH    = 200     # PE > 200 or PE < 0 → High (extreme or negative)
PE_MEDIUM  = 100     # PE > 100           → Medium (elevated but seen in growth stocks)

# Margin Compression (true YoY drop in gross margin, in percentage points)
MARGIN_HIGH   = -5.0  # Gross margin dropped > 5pp YoY  → High
MARGIN_MEDIUM = -3.0  # Gross margin dropped > 3pp YoY  → Medium
MARGIN_LOW    = -1.5  # Gross margin dropped > 1.5pp YoY → Low


def _get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _narrate(client: anthropic.Anthropic, ticker: str, company: str, anomaly_type: str, detail: str) -> str:
    prompt = (
        f"Company: {company} ({ticker})\n"
        f"Anomaly: {anomaly_type}\n"
        f"Details: {detail}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def detect_anomalies() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Pull current + prior year gross margin for true YoY margin compression
    rows = conn.execute("""
        SELECT c.ticker, c.name, c.pe_ratio,
               m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
    """).fetchall()

    # For margin compression: need 2 most recent years per ticker
    margin_by_ticker = {}
    fin_rows = conn.execute("""
        SELECT ticker, year, revenue, gross_profit
        FROM financials
        ORDER BY ticker, year DESC
    """).fetchall()
    conn.close()

    # Build per-ticker list of (year, gross_margin%) sorted newest first
    from collections import defaultdict
    yearly = defaultdict(list)
    for f in fin_rows:
        if f["revenue"] and f["gross_profit"] and f["revenue"] != 0:
            gm = f["gross_profit"] / f["revenue"] * 100
            yearly[f["ticker"]].append((f["year"], gm))

    for ticker, years in yearly.items():
        if len(years) >= 2:
            # years[0] = most recent, years[1] = prior year
            margin_by_ticker[ticker] = years[0][1] - years[1][1]  # pp change

    client = _get_client()
    anomalies = []

    for row in rows:
        ticker    = row["ticker"]
        name      = row["name"]
        rev_growth = row["revenue_growth_yoy"]
        ni_growth  = row["net_income_growth_yoy"]
        pe         = row["pe_ratio"]
        checks = []

        # ── 1. Revenue Decline ─────────────────────────────────────────────
        if rev_growth is not None:
            if rev_growth < REV_HIGH:
                checks.append({
                    "anomaly_type": "Revenue Decline",
                    "severity": "high",
                    "metric_value": f"{rev_growth:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue fell {rev_growth:.1f}% YoY, well below the -15% high-severity threshold.",
                })
            elif rev_growth < REV_MEDIUM:
                checks.append({
                    "anomaly_type": "Revenue Decline",
                    "severity": "medium",
                    "metric_value": f"{rev_growth:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue declined {rev_growth:.1f}% YoY, below the -5% concern threshold.",
                })
            elif rev_growth < REV_LOW:
                checks.append({
                    "anomaly_type": "Revenue Slowdown",
                    "severity": "low",
                    "metric_value": f"{rev_growth:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue growth slowed to {rev_growth:.1f}% YoY, approaching flat growth.",
                })

        # ── 2. Net Income Drop ─────────────────────────────────────────────
        if ni_growth is not None:
            if ni_growth < NI_HIGH:
                checks.append({
                    "anomaly_type": "Net Income Drop",
                    "severity": "high",
                    "metric_value": f"{ni_growth:.1f}% YoY net income growth",
                    "detail": f"{name} net income fell {ni_growth:.1f}% YoY, exceeding the -25% high-severity threshold.",
                })
            elif ni_growth < NI_MEDIUM:
                checks.append({
                    "anomaly_type": "Net Income Drop",
                    "severity": "medium",
                    "metric_value": f"{ni_growth:.1f}% YoY net income growth",
                    "detail": f"{name} net income declined {ni_growth:.1f}% YoY, a notable profitability deterioration.",
                })
            elif ni_growth < NI_LOW:
                checks.append({
                    "anomaly_type": "Net Income Pressure",
                    "severity": "low",
                    "metric_value": f"{ni_growth:.1f}% YoY net income growth",
                    "detail": f"{name} net income dipped {ni_growth:.1f}% YoY, a mild but emerging profitability concern.",
                })

        # ── 3. Valuation Outlier ───────────────────────────────────────────
        if pe is not None:
            if pe < 0:
                checks.append({
                    "anomaly_type": "Valuation Outlier",
                    "severity": "high",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} has a negative PE ratio of {pe:.1f}, indicating the company is currently loss-making.",
                })
            elif pe > PE_HIGH:
                checks.append({
                    "anomaly_type": "Valuation Outlier",
                    "severity": "high",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} has an extremely elevated PE ratio of {pe:.1f}, far above the 200x threshold.",
                })
            elif pe > PE_MEDIUM:
                checks.append({
                    "anomaly_type": "Valuation Outlier",
                    "severity": "medium",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} has a high PE ratio of {pe:.1f}, above the 100x elevated valuation threshold.",
                })

        # ── 4. Margin Compression (true YoY pp change) ────────────────────
        margin_change = margin_by_ticker.get(ticker)
        if margin_change is not None:
            if margin_change < MARGIN_HIGH:
                checks.append({
                    "anomaly_type": "Margin Compression",
                    "severity": "high",
                    "metric_value": f"Gross margin dropped {margin_change:.1f}pp YoY",
                    "detail": f"{name} gross margin fell {margin_change:.1f} percentage points YoY, a severe compression exceeding the -5pp threshold.",
                })
            elif margin_change < MARGIN_MEDIUM:
                checks.append({
                    "anomaly_type": "Margin Compression",
                    "severity": "medium",
                    "metric_value": f"Gross margin dropped {margin_change:.1f}pp YoY",
                    "detail": f"{name} gross margin declined {margin_change:.1f} percentage points YoY, above the -3pp concern threshold.",
                })
            elif margin_change < MARGIN_LOW:
                checks.append({
                    "anomaly_type": "Margin Compression",
                    "severity": "low",
                    "metric_value": f"Gross margin dropped {margin_change:.1f}pp YoY",
                    "detail": f"{name} gross margin slipped {margin_change:.1f} percentage points YoY, an early-stage compression signal.",
                })

        for check in checks:
            narrative = _narrate(client, ticker, name, check["anomaly_type"], check["detail"])
            anomalies.append({
                "ticker": ticker,
                "company_name": name,
                "anomaly_type": check["anomaly_type"],
                "severity": check["severity"],
                "metric_value": check["metric_value"],
                "narrative": narrative,
            })

    return anomalies


if __name__ == "__main__":
    results = detect_anomalies()
    if not results:
        print("No anomalies detected.")
    for a in results:
        print(f"[{a['severity'].upper()}] {a['company_name']} - {a['anomaly_type']}")
        print(f"  Metric: {a['metric_value']}")
        print(f"  {a['narrative']}\n")
