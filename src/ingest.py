"""
Fortune AI - Financial Data Ingestion
Pulls real financial data from Financial Modeling Prep (FMP) and stores in SQLite.
"""

import os
import sqlite3
import logging
import time
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "db" / "financial.db"
FMP_BASE = "https://financialmodelingprep.com/stable"

COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "NVDA": "Nvidia",
    "NFLX": "Netflix",
    "SNOW": "Snowflake",
    "IBM": "IBM",
    "SAP": "SAP",
    "NOW": "ServiceNow",
    "WDAY": "Workday",
    "PLTR": "Palantir",
    "ADBE": "Adobe",
    "INTU": "Intuit",
    "QCOM": "Qualcomm",
    "AVGO": "Broadcom",
}


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS companies (
            ticker        TEXT PRIMARY KEY,
            name          TEXT,
            sector        TEXT,
            market_cap    REAL,
            pe_ratio      REAL,
            eps           REAL,
            week52_high   REAL,
            week52_low    REAL,
            current_price REAL,
            updated_at    TEXT
        );

        CREATE TABLE IF NOT EXISTS financials (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT,
            year             INTEGER,
            revenue          REAL,
            net_income       REAL,
            gross_profit     REAL,
            operating_income REAL,
            UNIQUE(ticker, year)
        );

        CREATE TABLE IF NOT EXISTS metrics (
            ticker                TEXT PRIMARY KEY,
            revenue_growth_yoy    REAL,
            net_income_growth_yoy REAL,
            gross_margin          REAL,
            operating_margin      REAL,
            net_margin            REAL
        );
    """)
    conn.commit()


def safe_float(value) -> float | None:
    try:
        v = float(value)
        return None if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return None


def fmp_get(endpoint: str, params: dict = {}) -> list | dict | None:
    api_key = os.environ.get("FMP_API_KEY", "")
    url = f"{FMP_BASE}{endpoint}"
    try:
        resp = requests.get(url, params={"apikey": api_key, **params}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "Error Message" in data:
            logger.error(f"FMP error for {endpoint}: {data['Error Message']}")
            return None
        return data
    except Exception as e:
        logger.error(f"FMP request failed {endpoint}: {e}")
        return None


def ingest_ticker(ticker: str, name: str, conn: sqlite3.Connection) -> bool:
    print(f"Ingesting {ticker}...", end=" ", flush=True)
    try:
        # ── Company profile ──────────────────────────────────────────────────
        profile_data = fmp_get("/profile", {"symbol": ticker})
        if not profile_data:
            raise ValueError("No profile data returned")
        profile = profile_data[0]

        market_cap    = safe_float(profile.get("marketCap"))
        current_price = safe_float(profile.get("price"))
        sector        = profile.get("sector") or "Technology"
        name          = profile.get("companyName") or name

        # 52-week range comes as "low-high" string
        week52_high = week52_low = pe_ratio = eps = None
        range_str = profile.get("range", "")
        if range_str and "-" in range_str:
            parts = range_str.split("-")
            try:
                week52_low  = float(parts[0])
                week52_high = float(parts[1])
            except ValueError:
                pass

        conn.execute("""
            INSERT OR REPLACE INTO companies
            (ticker, name, sector, market_cap, pe_ratio, eps, week52_high, week52_low, current_price, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, name, sector, market_cap, pe_ratio, eps,
              week52_high, week52_low, current_price, datetime.utcnow().isoformat()))

        # ── Income statements (last 4 annual) ───────────────────────────────
        income_data = fmp_get("/income-statement", {"symbol": ticker, "limit": 4, "period": "annual"})
        annual_data = {}
        if income_data:
            for stmt in income_data:
                year = int(stmt.get("calendarYear") or stmt["date"][:4])
                revenue          = safe_float(stmt.get("revenue"))
                net_income       = safe_float(stmt.get("netIncome"))
                gross_profit     = safe_float(stmt.get("grossProfit"))
                operating_income = safe_float(stmt.get("operatingIncome"))
                annual_data[year] = (revenue, net_income, gross_profit, operating_income)
                conn.execute("""
                    INSERT OR REPLACE INTO financials
                    (ticker, year, revenue, net_income, gross_profit, operating_income)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ticker, year, revenue, net_income, gross_profit, operating_income))

        # ── YoY growth + margins ─────────────────────────────────────────────
        revenue_growth = net_income_growth = None
        gross_margin = operating_margin = net_margin = None

        years = sorted(annual_data.keys(), reverse=True)
        if len(years) >= 2:
            r0, ni0, gp0, oi0 = annual_data[years[0]]
            r1, ni1, _,   _   = annual_data[years[1]]
            if r0 and r1 and r1 != 0:
                revenue_growth = (r0 - r1) / abs(r1) * 100
            if ni0 and ni1 and ni1 != 0:
                net_income_growth = (ni0 - ni1) / abs(ni1) * 100
            if r0 and gp0:
                gross_margin = gp0 / r0 * 100
            if r0 and oi0:
                operating_margin = oi0 / r0 * 100
            if r0 and ni0:
                net_margin = ni0 / r0 * 100

        conn.execute("""
            INSERT OR REPLACE INTO metrics
            (ticker, revenue_growth_yoy, net_income_growth_yoy, gross_margin, operating_margin, net_margin)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, revenue_growth, net_income_growth, gross_margin, operating_margin, net_margin))

        conn.commit()
        print("done")
        return True

    except Exception as e:
        print(f"FAILED ({e})")
        logger.error(f"Failed to ingest {ticker}: {e}")
        return False


def run_ingestion() -> None:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key or api_key == "your_fmp_api_key_here":
        print("ERROR: FMP_API_KEY not set in .env file.")
        print("Get a free key at: https://financialmodelingprep.com/developer/docs")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    success = 0
    total = len(COMPANIES)
    for ticker, name in COMPANIES.items():
        if ingest_ticker(ticker, name, conn):
            success += 1
        time.sleep(0.5)  # gentle rate limiting

    conn.close()
    print(f"\nIngested {success} of {total} companies successfully")


if __name__ == "__main__":
    run_ingestion()
