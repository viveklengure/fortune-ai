"""
Fortune AI - Executive Digest Generator
Produces an AI-generated portfolio summary using Claude.
"""

import os
import sqlite3
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.agent import detect_anomalies

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"

SYSTEM_PROMPT = (
    "You are Fortune AI. Write a concise executive digest of the Fortune 500 tech portfolio. "
    "Use these sections:\n"
    "## Portfolio Overview\n"
    "## Top Performers (top 3 by revenue growth)\n"
    "## Anomalies & Watch List\n"
    "## Market Highlights\n"
    "Keep it under 500 words. Use specific numbers. Professional tone."
)


def generate_digest() -> str:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    companies = conn.execute("""
        SELECT c.ticker, c.name, c.market_cap, c.pe_ratio, c.current_price,
               m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin,
               f.revenue, f.net_income
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
        LEFT JOIN (
            SELECT ticker, revenue, net_income
            FROM financials
            WHERE year = (SELECT MAX(year) FROM financials f2 WHERE f2.ticker = financials.ticker)
        ) f ON c.ticker = f.ticker
    """).fetchall()
    conn.close()

    # Build portfolio context
    lines = []
    for row in companies:
        rev = row["revenue"]
        rev_str = f"${rev/1e9:.1f}B" if rev else "N/A"
        rev_growth = row["revenue_growth_yoy"]
        rev_g_str = f"{rev_growth:.1f}%" if rev_growth is not None else "N/A"
        net_margin = row["net_margin"]
        nm_str = f"{net_margin:.1f}%" if net_margin is not None else "N/A"
        mcap = row["market_cap"]
        mcap_str = f"${mcap/1e9:.0f}B" if mcap else "N/A"
        lines.append(
            f"{row['name']} ({row['ticker']}): Revenue {rev_str}, "
            f"YoY Growth {rev_g_str}, Net Margin {nm_str}, Market Cap {mcap_str}"
        )

    portfolio_data = "\n".join(lines)

    anomalies = detect_anomalies()
    anomaly_lines = []
    for a in anomalies:
        anomaly_lines.append(f"- {a['company_name']} ({a['ticker']}): {a['anomaly_type']} [{a['severity']}] — {a['metric_value']}")
    anomaly_text = "\n".join(anomaly_lines) if anomaly_lines else "No anomalies detected."

    user_content = (
        f"Portfolio Data:\n{portfolio_data}\n\n"
        f"Current Anomalies:\n{anomaly_text}"
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text


if __name__ == "__main__":
    digest = generate_digest()
    print(digest)
