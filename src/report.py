"""
Fortune AI - AI-Powered Company Intelligence Report
Pulls historical + current data from SQLite, injects a structured template,
and uses Claude to generate a formatted analyst report grounded in real numbers.
"""

import os
import sqlite3
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"

# ── Output template ────────────────────────────────────────────────────────────
# This template is injected into the prompt so Claude knows exactly what format
# and which metrics to produce — the core of the template-driven AI workflow.

REPORT_TEMPLATE = """
FORTUNE AI — COMPANY INTELLIGENCE REPORT
==========================================
Company: {name} ({ticker})          Report Date: {report_date}

SECTION 1 — SNAPSHOT
---------------------
[2-3 sentence executive summary covering current revenue, net income,
net margin, and market cap. Written for a non-technical stakeholder.]

SECTION 2 — HISTORICAL PERFORMANCE
------------------------------------
Year | Revenue  | Net Income | Net Margin | YoY Rev Growth | YoY NI Growth
-----|----------|------------|------------|----------------|---------------
[Fill one row per year from oldest to newest using the data provided.
Use $B for billions, % for percentages. Calculate YoY growth between years.]

SECTION 3 — TREND ANALYSIS
----------------------------
[3-4 sentences. Is revenue accelerating or decelerating? Are margins
expanding or compressing? What does the net income trend reveal about
operational leverage or cost pressure?]

SECTION 4 — SIGNAL
-------------------
Momentum     : [ACCELERATING / STABLE / DECELERATING]
Margin Health: [EXPANDING / STABLE / COMPRESSING]
Valuation    : [PREMIUM / FAIR / VALUE]

[One sentence explaining each signal verdict.]

SECTION 5 — ANALYST COMMENTARY
--------------------------------
[One full paragraph written for an executive audience. Synthesise the
historical context with the current state. Highlight what to watch going
forward — risks, tailwinds, or inflection points visible in the data.]
==========================================
"""

SYSTEM_PROMPT = (
    "You are Fortune AI, a senior financial analyst. "
    "You will be given real financial data for a company and a report template. "
    "Your job is to produce the report by filling in every section of the template "
    "using only the data provided — do not invent numbers. "
    "Format all numbers exactly as specified: $B for billions, % for percentages. "
    "Keep each section concise and professional. "
    "Output only the completed report, nothing else."
)


def _load_company_data(ticker: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    company = conn.execute(
        """
        SELECT c.*, m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
        WHERE c.ticker = ?
        """,
        (ticker,),
    ).fetchone()

    if not company:
        conn.close()
        return None

    financials = conn.execute(
        "SELECT * FROM financials WHERE ticker = ? ORDER BY year ASC",
        (ticker,),
    ).fetchall()

    conn.close()
    return {
        "company": dict(company),
        "financials": [dict(f) for f in financials],
    }


def _format_data_context(data: dict) -> str:
    c = data["company"]
    fins = data["financials"]

    def b(v):
        return f"${v/1e9:.2f}B" if v else "N/A"

    def pct(v):
        return f"{v:.1f}%" if v is not None else "N/A"

    lines = [
        f"Company: {c['name']} ({c['ticker']})",
        f"Sector: {c.get('sector', 'N/A')}",
        f"Market Cap: {b(c.get('market_cap'))}",
        f"PE Ratio: {c.get('pe_ratio', 'N/A')}",
        f"Current Price: ${c.get('current_price', 'N/A')}",
        f"Latest YoY Revenue Growth: {pct(c.get('revenue_growth_yoy'))}",
        f"Latest YoY Net Income Growth: {pct(c.get('net_income_growth_yoy'))}",
        f"Latest Net Margin: {pct(c.get('net_margin'))}",
        "",
        "Annual Financials (oldest to newest):",
    ]

    for f in fins:
        lines.append(
            f"  {f['year']}: Revenue={b(f.get('revenue'))}, "
            f"Net Income={b(f.get('net_income'))}, "
            f"Gross Profit={b(f.get('gross_profit'))}, "
            f"Operating Income={b(f.get('operating_income'))}"
        )

    return "\n".join(lines)


def generate_report(ticker: str) -> dict:
    """
    Generate a structured AI analyst report for a given ticker.
    Returns: {"report": str, "company_name": str} or {"error": str}
    """
    data = _load_company_data(ticker.upper())
    if not data:
        return {"error": f"No data found for ticker {ticker}"}

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        max_tokens=2048,
    )

    data_context = _format_data_context(data)
    filled_template = REPORT_TEMPLATE.format(
        name=data["company"]["name"],
        ticker=ticker.upper(),
        report_date=date.today().strftime("%B %d, %Y"),
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("{system}"),
        HumanMessagePromptTemplate.from_template(
            "Here is the financial data:\n\n{data}\n\n"
            "Here is the report template to fill in:\n\n{template}\n\n"
            "Produce the completed report."
        ),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "system": SYSTEM_PROMPT,
        "data": data_context,
        "template": filled_template,
    })

    return {
        "report": response.content,
        "company_name": data["company"]["name"],
    }


if __name__ == "__main__":
    result = generate_report("AAPL")
    print(result.get("report") or result.get("error"))
