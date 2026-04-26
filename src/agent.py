"""
Fortune AI - Anomaly Detection
Two modes:
  detect_anomalies()          — fast, rule-based, no extra API calls
  detect_anomalies_agentic()  — Claude-driven tool loop, deeper but more expensive
"""

import os
import json
import sqlite3
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"

SYSTEM_PROMPT = """You are a senior financial analyst investigating a portfolio of tech companies for anomalies.

You have tools to query financial data. Use them to:
1. Get the full company list first
2. Investigate each company's metrics and financials
3. Compare against peers when something looks unusual
4. Flag genuine anomalies with a severity (high/medium/low) and a factual narrative

Severity guide:
- high: significant deterioration requiring immediate attention (e.g. revenue down >15%, net income down >25%, negative PE, gross margin collapsed >5pp)
- medium: notable concern worth monitoring (e.g. revenue down >5%, net income down >10%, PE >100, margin down >3pp)
- low: early signal, borderline but worth watching

Be thorough but do not flag every company. Only flag what genuinely stands out.
When done investigating, call finish_investigation with no arguments."""

# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "get_company_list",
        "description": "Returns all tracked companies with their ticker, name, sector, and latest key metrics (revenue growth YoY, net income growth YoY, gross margin, PE ratio).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_financials",
        "description": "Returns annual income statement history for a company (revenue, net income, gross profit, operating income per year).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_metrics",
        "description": "Returns the latest computed metrics for a company: YoY growth rates and margin percentages.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "flag_anomaly",
        "description": "Flag a company as having a financial anomaly. Call this once per anomaly found.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":       {"type": "string", "description": "Stock ticker symbol"},
                "company_name": {"type": "string", "description": "Full company name"},
                "anomaly_type": {"type": "string", "description": "Short label, e.g. 'Revenue Decline', 'Margin Compression', 'Valuation Outlier'"},
                "severity":     {"type": "string", "enum": ["high", "medium", "low"]},
                "metric_value": {"type": "string", "description": "The specific metric that triggered this, e.g. '-18.2% YoY revenue growth'"},
                "narrative":    {"type": "string", "description": "2-3 sentence factual explanation grounded in the data"},
            },
            "required": ["ticker", "company_name", "anomaly_type", "severity", "metric_value", "narrative"],
        },
    },
    {
        "name": "finish_investigation",
        "description": "Call this when you have finished investigating all companies and flagged all anomalies.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ── Tool implementations ───────────────────────────────────────────────────────

def _get_company_list() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.ticker, c.name, c.sector, c.pe_ratio, c.market_cap,
               m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.operating_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_financials(ticker: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT year, revenue, net_income, gross_profit, operating_income "
        "FROM financials WHERE ticker = ? ORDER BY year DESC",
        (ticker,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_metrics(ticker: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM metrics WHERE ticker = ?", (ticker,)
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def _get_peers(ticker: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sector_row = conn.execute(
        "SELECT sector FROM companies WHERE ticker = ?", (ticker,)
    ).fetchone()
    if not sector_row:
        conn.close()
        return []
    sector = sector_row["sector"]
    rows = conn.execute("""
        SELECT c.ticker, c.name, m.revenue_growth_yoy, m.net_income_growth_yoy,
               m.gross_margin, m.net_margin
        FROM companies c
        LEFT JOIN metrics m ON c.ticker = m.ticker
        WHERE c.sector = ? AND c.ticker != ?
    """, (sector, ticker)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _dispatch_tool(name: str, inputs: dict, flagged: list) -> str:
    if name == "get_company_list":
        result = _get_company_list()
    elif name == "get_financials":
        result = _get_financials(inputs["ticker"])
    elif name == "get_metrics":
        result = _get_metrics(inputs["ticker"])
    elif name == "get_peers":
        result = _get_peers(inputs["ticker"])
    elif name == "flag_anomaly":
        flagged.append({
            "ticker":       inputs["ticker"],
            "company_name": inputs["company_name"],
            "anomaly_type": inputs["anomaly_type"],
            "severity":     inputs["severity"],
            "metric_value": inputs["metric_value"],
            "narrative":    inputs["narrative"],
        })
        result = {"status": "flagged", "ticker": inputs["ticker"]}
    elif name == "finish_investigation":
        result = {"status": "done"}
    else:
        result = {"error": f"Unknown tool: {name}"}

    return json.dumps(result, default=str)


# ── Rule-based (standard) scan ────────────────────────────────────────────────

REV_HIGH, REV_MEDIUM, REV_LOW     = -15.0, -5.0, -2.0
NI_HIGH,  NI_MEDIUM,  NI_LOW      = -25.0, -10.0, -5.0
PE_HIGH,  PE_MEDIUM               = 200,   100
MARGIN_HIGH, MARGIN_MEDIUM, MARGIN_LOW = -5.0, -3.0, -1.5

_NARRATE_PROMPT = (
    "You are a financial analyst. Write a brief, factual 2-3 sentence narrative explaining "
    "this financial anomaly. Be specific about the numbers. "
    "Do not speculate beyond what the data shows."
)


def _narrate(client: anthropic.Anthropic, ticker: str, company: str, anomaly_type: str, detail: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=_NARRATE_PROMPT,
        messages=[{"role": "user", "content": f"Company: {company} ({ticker})\nAnomaly: {anomaly_type}\nDetails: {detail}"}],
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

    from collections import defaultdict
    fin_rows = conn.execute("""
        SELECT ticker, year, revenue, gross_profit
        FROM financials ORDER BY ticker, year DESC
    """).fetchall()
    conn.close()

    yearly = defaultdict(list)
    for f in fin_rows:
        if f["revenue"] and f["gross_profit"] and f["revenue"] != 0:
            yearly[f["ticker"]].append(f["gross_profit"] / f["revenue"] * 100)

    margin_change = {t: yrs[0] - yrs[1] for t, yrs in yearly.items() if len(yrs) >= 2}

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    anomalies = []

    for row in rows:
        ticker, name = row["ticker"], row["name"]
        rev_g, ni_g, pe = row["revenue_growth_yoy"], row["net_income_growth_yoy"], row["pe_ratio"]
        checks = []

        if rev_g is not None:
            if rev_g < REV_HIGH:
                checks.append({"anomaly_type": "Revenue Decline", "severity": "high",
                    "metric_value": f"{rev_g:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue fell {rev_g:.1f}% YoY, well below the -15% threshold."})
            elif rev_g < REV_MEDIUM:
                checks.append({"anomaly_type": "Revenue Decline", "severity": "medium",
                    "metric_value": f"{rev_g:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue declined {rev_g:.1f}% YoY."})
            elif rev_g < REV_LOW:
                checks.append({"anomaly_type": "Revenue Slowdown", "severity": "low",
                    "metric_value": f"{rev_g:.1f}% YoY revenue growth",
                    "detail": f"{name} revenue growth slowed to {rev_g:.1f}% YoY."})

        if ni_g is not None:
            if ni_g < NI_HIGH:
                checks.append({"anomaly_type": "Net Income Drop", "severity": "high",
                    "metric_value": f"{ni_g:.1f}% YoY net income growth",
                    "detail": f"{name} net income fell {ni_g:.1f}% YoY."})
            elif ni_g < NI_MEDIUM:
                checks.append({"anomaly_type": "Net Income Drop", "severity": "medium",
                    "metric_value": f"{ni_g:.1f}% YoY net income growth",
                    "detail": f"{name} net income declined {ni_g:.1f}% YoY."})
            elif ni_g < NI_LOW:
                checks.append({"anomaly_type": "Net Income Pressure", "severity": "low",
                    "metric_value": f"{ni_g:.1f}% YoY net income growth",
                    "detail": f"{name} net income dipped {ni_g:.1f}% YoY."})

        if pe is not None:
            if pe < 0:
                checks.append({"anomaly_type": "Valuation Outlier", "severity": "high",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} has a negative PE of {pe:.1f}, indicating losses."})
            elif pe > PE_HIGH:
                checks.append({"anomaly_type": "Valuation Outlier", "severity": "high",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} PE of {pe:.1f} is extremely elevated."})
            elif pe > PE_MEDIUM:
                checks.append({"anomaly_type": "Valuation Outlier", "severity": "medium",
                    "metric_value": f"PE ratio: {pe:.1f}",
                    "detail": f"{name} PE of {pe:.1f} is above the 100x threshold."})

        mc = margin_change.get(ticker)
        if mc is not None:
            if mc < MARGIN_HIGH:
                checks.append({"anomaly_type": "Margin Compression", "severity": "high",
                    "metric_value": f"Gross margin dropped {mc:.1f}pp YoY",
                    "detail": f"{name} gross margin fell {mc:.1f}pp YoY."})
            elif mc < MARGIN_MEDIUM:
                checks.append({"anomaly_type": "Margin Compression", "severity": "medium",
                    "metric_value": f"Gross margin dropped {mc:.1f}pp YoY",
                    "detail": f"{name} gross margin declined {mc:.1f}pp YoY."})
            elif mc < MARGIN_LOW:
                checks.append({"anomaly_type": "Margin Compression", "severity": "low",
                    "metric_value": f"Gross margin dropped {mc:.1f}pp YoY",
                    "detail": f"{name} gross margin slipped {mc:.1f}pp YoY."})

        for check in checks:
            narrative = _narrate(client, ticker, name, check["anomaly_type"], check["detail"])
            anomalies.append({"ticker": ticker, "company_name": name, **check, "narrative": narrative})

    return anomalies


# ── Agentic loop ───────────────────────────────────────────────────────────────

def detect_anomalies_agentic() -> list[dict]:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    flagged: list[dict] = []
    messages = [
        {
            "role": "user",
            "content": "Investigate the portfolio for financial anomalies. Use your tools to analyse each company thoroughly, then flag what you find.",
        }
    ]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        # Check stop conditions
        if response.stop_reason == "end_turn":
            break

        if response.stop_reason != "tool_use":
            break

        # Process all tool calls in this turn
        tool_results = []
        finished = False
        for block in response.content:
            if block.type != "tool_use":
                continue

            result_str = _dispatch_tool(block.name, block.input, flagged)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_str,
            })

            if block.name == "finish_investigation":
                finished = True

        messages.append({"role": "user", "content": tool_results})

        if finished:
            break

    return flagged


if __name__ == "__main__":
    import sys
    fn = detect_anomalies_agentic if "--agentic" in sys.argv else detect_anomalies
    results = fn()
    if not results:
        print("No anomalies detected.")
    for a in results:
        print(f"[{a['severity'].upper()}] {a['company_name']} - {a['anomaly_type']}")
        print(f"  Metric: {a['metric_value']}")
        print(f"  {a['narrative']}\n")
