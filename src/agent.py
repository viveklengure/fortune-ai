"""
Fortune AI - Anomaly Detection Agent
Scans financial metrics and uses Claude to narrate anomalies.
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

    rows = conn.execute("""
        SELECT c.ticker, c.name, c.pe_ratio,
               m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
    """).fetchall()

    conn.close()

    client = _get_client()
    anomalies = []

    for row in rows:
        ticker = row["ticker"]
        name = row["name"]
        rev_growth = row["revenue_growth_yoy"]
        ni_growth = row["net_income_growth_yoy"]
        pe = row["pe_ratio"]
        gross_margin = row["gross_margin"]

        checks = []

        if rev_growth is not None and rev_growth < -5:
            checks.append({
                "anomaly_type": "Revenue Decline",
                "severity": "high" if rev_growth < -15 else "medium",
                "metric_value": f"{rev_growth:.1f}% YoY revenue growth",
                "detail": f"{name} has a revenue YoY growth of {rev_growth:.1f}%, which is below the -5% threshold.",
            })

        if ni_growth is not None and ni_growth < -20:
            checks.append({
                "anomaly_type": "Net Income Drop",
                "severity": "high",
                "metric_value": f"{ni_growth:.1f}% YoY net income growth",
                "detail": f"{name} experienced a net income decline of {ni_growth:.1f}% year-over-year.",
            })

        if pe is not None and (pe > 100 or pe < 0):
            label = "extremely high" if pe > 100 else "negative"
            checks.append({
                "anomaly_type": "Valuation Outlier",
                "severity": "medium",
                "metric_value": f"PE ratio: {pe:.1f}",
                "detail": f"{name} has a {label} PE ratio of {pe:.1f}, which is outside normal valuation ranges.",
            })

        # Margin compression: we flag if gross margin is below 20% as a proxy
        # (full YoY margin compression requires two years of margin data — approximated here)
        if gross_margin is not None and gross_margin < 20:
            checks.append({
                "anomaly_type": "Margin Compression",
                "severity": "medium",
                "metric_value": f"Gross margin: {gross_margin:.1f}%",
                "detail": f"{name} has a gross margin of {gross_margin:.1f}%, indicating potential margin pressure.",
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
    for a in results:
        print(f"[{a['severity'].upper()}] {a['company_name']} - {a['anomaly_type']}")
        print(f"  Metric: {a['metric_value']}")
        print(f"  {a['narrative']}\n")
