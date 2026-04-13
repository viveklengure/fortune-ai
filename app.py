"""
Fortune AI - Streamlit Web Application
9-page financial intelligence platform.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import anthropic
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "financial.db"

st.set_page_config(
    page_title="Fortune AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ────────────────────────────────────────────────────────────────────

def db_exists() -> bool:
    return DB_PATH.exists()


def no_data_msg():
    st.warning("⚠️ No data found. Run **Option 3** from `run.py` to set up your data first.")


def fmt_b(v) -> str:
    if v is None:
        return "N/A"
    return f"${v/1e9:.1f}B"


def fmt_pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1f}%"


@st.cache_data(ttl=300)
def load_companies() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT c.ticker, c.name, c.sector, c.market_cap, c.pe_ratio,
               c.eps, c.week52_high, c.week52_low, c.current_price, c.updated_at,
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
    """, conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_financials(ticker: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM financials WHERE ticker = ? ORDER BY year ASC",
        conn, params=(ticker,)
    )
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_portfolio_trends():
    from src.trends import get_portfolio_trends
    return get_portfolio_trends()


@st.cache_data(ttl=3600)
def load_sector_aggregates():
    from src.trends import get_sector_aggregates
    return get_sector_aggregates()


# ── sidebar navigation ─────────────────────────────────────────────────────────

st.sidebar.title("Fortune AI")
st.sidebar.markdown("*Financial Intelligence Platform*")

# Support navigation from Portfolio Trends card clicks
if "nav_page" in st.session_state:
    default_page = st.session_state.pop("nav_page")
else:
    default_page = "Dashboard"

pages = [
    "Dashboard",
    "RAG Chat",
    "Anomaly Monitor",
    "Company Explorer",
    "Daily Digest",
    "Forecasting",
    "Peer Comparison",
    "Portfolio Trends",
    "Sector Analysis",
]

page = st.sidebar.radio("Navigate", pages, index=pages.index(default_page) if default_page in pages else 0)
st.sidebar.markdown("---")
st.sidebar.caption("Data powered by FMP · AI by Claude")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("Fortune AI — Financial Intelligence Platform")

    if not db_exists():
        no_data_msg()
        st.stop()

    df = load_companies()
    if df.empty:
        no_data_msg()
        st.stop()

    total = len(df)
    latest_date = df["updated_at"].dropna().max()
    latest_date_str = latest_date[:10] if latest_date else "N/A"
    positive_growth = int((df["revenue_growth_yoy"].dropna() > 0).sum())
    anomaly_count = int(
        (df["revenue_growth_yoy"].dropna() < -5).sum()
        + (df["net_income_growth_yoy"].dropna() < -20).sum()
        + (df["pe_ratio"].dropna().apply(lambda x: x > 100 or x < 0)).sum()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Companies Tracked", total)
    col2.metric("Latest Data Date", latest_date_str)
    col3.metric("Anomalies Detected", anomaly_count)
    col4.metric("Positive Revenue Growth", positive_growth)

    st.markdown("---")

    top10 = df.dropna(subset=["revenue"]).nlargest(10, "revenue").copy()
    top10["revenue_b"] = top10["revenue"] / 1e9
    fig_bar = px.bar(
        top10.sort_values("revenue_b"),
        x="revenue_b", y="name", orientation="h",
        title="Top 10 Companies by Revenue ($B)",
        labels={"revenue_b": "Revenue ($B)", "name": "Company"},
        color="revenue_b", color_continuous_scale="Blues",
    )
    fig_bar.update_layout(coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    scatter_df = df.dropna(subset=["revenue", "market_cap"]).copy()
    scatter_df["revenue_b"] = scatter_df["revenue"] / 1e9
    scatter_df["market_cap_b"] = scatter_df["market_cap"] / 1e9
    fig_scatter = px.scatter(
        scatter_df, x="revenue_b", y="market_cap_b", text="ticker",
        title="Market Cap vs Revenue",
        labels={"revenue_b": "Revenue ($B)", "market_cap_b": "Market Cap ($B)"},
        color="sector", size="market_cap_b", size_max=60,
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RAG CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RAG Chat":
    st.title("Ask Fortune AI")

    if not db_exists():
        no_data_msg()
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                st.caption(f"Sources: {', '.join(msg['sources'])}")

    question = st.chat_input("Ask about Fortune 500 financials...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.rag import ask
                    result = ask(question)
                    answer = result["answer"]
                    sources = result["sources"]
                except Exception as e:
                    answer = f"Error: {e}"
                    sources = []
            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "sources": sources,
        })


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Monitor":
    st.title("Anomaly Detection")

    if not db_exists():
        no_data_msg()
        st.stop()

    severity_badge = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🟢 Low"}

    if st.button("Run Anomaly Scan", type="primary"):
        with st.spinner("Scanning portfolio for anomalies..."):
            try:
                from src.agent import detect_anomalies
                anomalies = detect_anomalies()
            except Exception as e:
                st.error(f"Error running scan: {e}")
                anomalies = []

        if not anomalies:
            st.success("No anomalies detected.")
        else:
            st.warning(f"Found {len(anomalies)} anomaly/anomalies.")
            for a in anomalies:
                badge = severity_badge.get(a["severity"], a["severity"])
                with st.expander(f"{a['company_name']} ({a['ticker']}) — {a['anomaly_type']}  {badge}"):
                    col1, col2 = st.columns([1, 2])
                    col1.markdown(f"**Severity:** {badge}")
                    col1.markdown(f"**Metric:** {a['metric_value']}")
                    col2.markdown(f"**Analysis:**\n{a['narrative']}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — COMPANY EXPLORER (enhanced)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Company Explorer":
    st.title("Company Explorer")

    if not db_exists():
        no_data_msg()
        st.stop()

    df = load_companies()
    if df.empty:
        no_data_msg()
        st.stop()

    # Support navigation from Portfolio Trends
    default_ticker = st.session_state.get("selected_ticker", df["ticker"].iloc[0])
    company_options = dict(zip(df["name"] + " (" + df["ticker"] + ")", df["ticker"]))
    default_label = next((k for k, v in company_options.items() if v == default_ticker), list(company_options.keys())[0])
    selected_label = st.selectbox("Select a Company", list(company_options.keys()), index=list(company_options.keys()).index(default_label))
    selected_ticker = company_options[selected_label]

    row = df[df["ticker"] == selected_ticker].iloc[0]
    st.markdown(f"### {row['name']} ({row['ticker']})")
    st.markdown(f"**Sector:** {row.get('sector', 'N/A')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${row['current_price']:.2f}" if pd.notna(row['current_price']) else "N/A")
    col2.metric("Market Cap", fmt_b(row['market_cap']) if pd.notna(row.get('market_cap')) else "N/A")
    col3.metric("PE Ratio", f"{row['pe_ratio']:.1f}" if pd.notna(row.get('pe_ratio')) else "N/A")
    col4.metric("EPS", f"${row['eps']:.2f}" if pd.notna(row.get('eps')) else "N/A")

    col5, col6, col7 = st.columns(3)
    col5.metric("Gross Margin", fmt_pct(row['gross_margin']) if pd.notna(row.get('gross_margin')) else "N/A")
    col6.metric("Operating Margin", fmt_pct(row['operating_margin']) if pd.notna(row.get('operating_margin')) else "N/A")
    col7.metric("Net Margin", fmt_pct(row['net_margin']) if pd.notna(row.get('net_margin')) else "N/A")

    if pd.notna(row.get('week52_low')) and pd.notna(row.get('week52_high')):
        st.markdown(f"**52-Week Range:** ${row['week52_low']:.2f} – ${row['week52_high']:.2f}")

    fin_df = load_financials(selected_ticker)

    if not fin_df.empty:
        st.markdown("---")
        col_left, col_right = st.columns(2)
        with col_left:
            fig_rev = px.line(fin_df, x="year", y="revenue",
                title=f"{row['name']} — Revenue (Last {len(fin_df)} Years)",
                labels={"revenue": "Revenue ($)", "year": "Year"}, markers=True)
            fig_rev.update_traces(line_color="#1f77b4")
            fig_rev.update_yaxes(tickformat="$.2s")
            st.plotly_chart(fig_rev, use_container_width=True)
        with col_right:
            fig_ni = px.line(fin_df, x="year", y="net_income",
                title=f"{row['name']} — Net Income (Last {len(fin_df)} Years)",
                labels={"net_income": "Net Income ($)", "year": "Year"}, markers=True)
            fig_ni.update_traces(line_color="#2ca02c")
            fig_ni.update_yaxes(tickformat="$.2s")
            st.plotly_chart(fig_ni, use_container_width=True)

        # ── Trend Analysis subsection ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📈 Trend Analysis")

        from src.trends import get_yoy_trends
        trends = get_yoy_trends(selected_ticker)
        years = trends["years"]

        def dual_axis_chart(title: str, field: str, color: str):
            values_b = [v / 1e9 if v is not None else None for v in trends[field]["values"]]
            yoy_pcts = trends[field]["yoy_pct"]
            bar_colors = []
            for p in yoy_pcts:
                if p is None:
                    bar_colors.append("gray")
                elif p >= 0:
                    bar_colors.append("green")
                else:
                    bar_colors.append("red")

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(
                x=years, y=values_b, name="Value ($B)",
                marker_color=bar_colors, opacity=0.7,
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                x=years, y=yoy_pcts, name="YoY %",
                mode="lines+markers", line=dict(color=color, width=2),
            ), secondary_y=True)
            fig.update_layout(title=f"{row['name']} — {title} Trend (with YoY %)", height=320)
            fig.update_yaxes(title_text="Value ($B)", secondary_y=False, tickformat="$.1f")
            fig.update_yaxes(title_text="YoY %", secondary_y=True, ticksuffix="%")
            return fig

        st.plotly_chart(dual_axis_chart("Revenue", "revenue", "#1f77b4"), use_container_width=True)
        st.plotly_chart(dual_axis_chart("Net Income", "net_income", "#2ca02c"), use_container_width=True)
        st.plotly_chart(dual_axis_chart("Gross Profit", "gross_profit", "#ff7f0e"), use_container_width=True)

        # Summary table
        table_rows = []
        for i, yr in enumerate(years):
            def cell(field, i=i):
                v = trends[field]["values"][i]
                p = trends[field]["yoy_pct"][i]
                return fmt_b(v), (f"{p:+.1f}%" if p is not None else "—")

            r_val, r_pct = cell("revenue")
            ni_val, ni_pct = cell("net_income")
            gp_val, gp_pct = cell("gross_profit")
            table_rows.append({
                "Year": yr,
                "Revenue": r_val, "Rev YoY%": r_pct,
                "Net Income": ni_val, "NI YoY%": ni_pct,
                "Gross Profit": gp_val, "GP YoY%": gp_pct,
            })

        tbl_df = pd.DataFrame(table_rows)

        def color_pct(val):
            if isinstance(val, str) and val.startswith("+"):
                return "color: green"
            elif isinstance(val, str) and val.startswith("-"):
                return "color: red"
            return ""

        st.dataframe(
            tbl_df.style.applymap(color_pct, subset=["Rev YoY%", "NI YoY%", "GP YoY%"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No historical financials available for this company.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DAILY DIGEST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Daily Digest":
    st.title("Executive Digest")
    st.markdown("*AI-generated portfolio summary*")

    if not db_exists():
        no_data_msg()
        st.stop()

    if "digest_text" not in st.session_state:
        st.session_state.digest_text = None
    if "digest_time" not in st.session_state:
        st.session_state.digest_time = None

    btn_label = "Regenerate" if st.session_state.digest_text else "Generate Digest"
    if st.button(btn_label, type="primary"):
        with st.spinner("Generating digest..."):
            try:
                from src.digest import generate_digest
                st.session_state.digest_text = generate_digest()
                st.session_state.digest_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                st.error(f"Error generating digest: {e}")

    if st.session_state.digest_text:
        st.markdown(st.session_state.digest_text)
        st.caption(f"Generated at {st.session_state.digest_time}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Forecasting":
    st.title("📊 Forecasting & Outlook")
    st.markdown("*2-Year statistical projection + AI analyst narrative*")

    if not db_exists():
        no_data_msg()
        st.stop()

    df = load_companies()
    company_options = dict(zip(df["name"] + " (" + df["ticker"] + ")", df["ticker"]))
    selected_label = st.selectbox("Select a Company", list(company_options.keys()))
    selected_ticker = company_options[selected_label]

    col_left, col_right = st.columns(2)

    # ── LEFT: Statistical Forecast ────────────────────────────────────────────
    with col_left:
        st.markdown("#### Statistical Forecast")
        st.info(
            "**Model:** Linear Regression (`numpy.polyfit`, degree=1)\n\n"
            "A separate least-squares regression line is fit to each metric "
            "(revenue, net income, gross profit) using the last 4 years of actuals. "
            "The line is then extrapolated 2 years forward. "
            "**R²** measures how well the historical data fits a straight line — "
            "higher R² means the trend is more consistent and the projection is more reliable. "
            "Best suited for companies with steady, predictable growth; less reliable for volatile or high-growth companies.",
            icon="📐",
        )
        stat_key = f"stat_forecast_{selected_ticker}"

        if st.button("Run Statistical Model", key="btn_stat", disabled=st.session_state.get("stat_loading", False)):
            st.session_state["stat_loading"] = True
            with st.spinner("Running forecast model..."):
                from src.forecast import statistical_forecast
                st.session_state[stat_key] = statistical_forecast(selected_ticker)
            st.session_state["stat_loading"] = False

        fc = st.session_state.get(stat_key)
        if fc and "error" not in fc:
            hist = fc["historical"]
            proj = fc["forecast"]
            conf = fc["confidence"]

            def forecast_chart(title: str, hist_vals: list, proj_vals: list,
                                color: str, light_color: str):
                fig = go.Figure()
                all_years = hist["years"] + proj["years"]
                all_vals_b = [v / 1e9 for v in hist_vals] + [v / 1e9 for v in proj_vals]
                split = len(hist["years"])

                fig.add_trace(go.Scatter(
                    x=hist["years"], y=[v / 1e9 for v in hist_vals],
                    name="Historical", mode="lines+markers",
                    line=dict(color=color, width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=proj["years"], y=[v / 1e9 for v in proj_vals],
                    name="Forecast", mode="lines+markers",
                    line=dict(color=light_color, width=2, dash="dash"),
                ))
                # Shaded forecast zone
                fig.add_vrect(
                    x0=hist["years"][-1], x1=proj["years"][-1],
                    fillcolor=light_color, opacity=0.1, line_width=0,
                    annotation_text="Forecast", annotation_position="top left",
                )
                fig.update_layout(title=title, height=280, yaxis_tickformat="$.1f",
                                  yaxis_title="$B", showlegend=True)
                return fig

            st.plotly_chart(forecast_chart(
                "Revenue Forecast ($B)", hist["revenue"], proj["revenue"], "#1f77b4", "#aec7e8"
            ), use_container_width=True)
            st.plotly_chart(forecast_chart(
                "Net Income Forecast ($B)", hist["net_income"], proj["net_income"], "#2ca02c", "#98df8a"
            ), use_container_width=True)
            st.plotly_chart(forecast_chart(
                "Gross Profit Forecast ($B)", hist["gross_profit"], proj["gross_profit"], "#ff7f0e", "#ffbb78"
            ), use_container_width=True)

            # Confidence indicators
            st.markdown("**Confidence Indicators** *(R² — closer to 1.0 = more reliable)*")
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue R²", f"{conf['revenue_r2']:.3f}")
            c2.metric("Net Income R²", f"{conf['net_income_r2']:.3f}")
            c3.metric("Gross Profit R²", f"{conf['gross_profit_r2']:.3f}")

            # Data table
            rows = []
            for yr, rv, ni, gp in zip(hist["years"], hist["revenue"], hist["net_income"], hist["gross_profit"]):
                rows.append({"Year": yr, "Revenue": fmt_b(rv), "Net Income": fmt_b(ni), "Gross Profit": fmt_b(gp), "Type": "Historical"})
            for yr, rv, ni, gp in zip(proj["years"], proj["revenue"], proj["net_income"], proj["gross_profit"]):
                rows.append({"Year": yr, "Revenue": fmt_b(rv), "Net Income": fmt_b(ni), "Gross Profit": fmt_b(gp), "Type": "🔮 Forecast"})
            tbl = pd.DataFrame(rows)
            st.dataframe(tbl.style.apply(
                lambda row: ["font-style: italic" if row["Type"] == "🔮 Forecast" else "" for _ in row], axis=1
            ), use_container_width=True, hide_index=True)

    # ── RIGHT: AI Narrative ───────────────────────────────────────────────────
    with col_right:
        st.markdown("#### AI Analyst Outlook")
        st.info(
            "**Model:** Claude Sonnet (claude-sonnet-4-20250514) via Anthropic API\n\n"
            "Claude is given the company's 4-year income statement history and key margins, "
            "then asked to write a structured forward-looking analyst report. "
            "It reasons over revenue trajectory, profitability trends, and known industry dynamics "
            "to produce the outlook — it does **not** run a statistical model itself.",
            icon="🤖",
        )
        ai_key = f"ai_narrative_{selected_ticker}"

        if st.button("Generate AI Outlook ↗", key="btn_ai", disabled=st.session_state.get("ai_loading", False)):
            st.session_state["ai_loading"] = True
            with st.spinner("Asking Claude to analyze trends..."):
                from src.forecast import ai_forecast_narrative
                raw = ai_forecast_narrative(selected_ticker)
                # Escape bare underscores inside words to prevent markdown italics
                import re
                clean = re.sub(r'(?<!\s)_(?!\s)', r'\_', raw)
                st.session_state[ai_key] = {
                    "text": clean,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            st.session_state["ai_loading"] = False

        ai_result = st.session_state.get(ai_key)
        if ai_result:
            st.markdown(ai_result["text"])
            st.caption(f"Generated at {ai_result['time']}")
            st.warning("⚠️ AI-generated outlook. Not financial advice. Based on historical data only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PEER COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Peer Comparison":
    st.title("🔍 Peer Comparison")
    st.markdown("*Side-by-side financial comparison of any two companies*")

    if not db_exists():
        no_data_msg()
        st.stop()

    df = load_companies()
    tickers = df["ticker"].tolist()
    names = df["name"].tolist()
    options = [f"{n} ({t})" for n, t in zip(names, tickers)]
    ticker_map = dict(zip(options, tickers))

    col_a, col_b = st.columns(2)
    default_a = next((o for o in options if "Apple" in o), options[0])
    default_b = next((o for o in options if "Microsoft" in o), options[1])
    sel_a = col_a.selectbox("Company A", options, index=options.index(default_a))
    sel_b = col_b.selectbox("Company B", options, index=options.index(default_b))

    if st.button("Compare", type="primary"):
        ta, tb = ticker_map[sel_a], ticker_map[sel_b]

        if ta == tb:
            st.warning("Please select two different companies.")
            st.stop()

        ra = df[df["ticker"] == ta].iloc[0]
        rb = df[df["ticker"] == tb].iloc[0]
        fin_a = load_financials(ta)
        fin_b = load_financials(tb)

        # ── Section 1: Snapshot ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Snapshot Comparison")

        def compare_metric(label, va, vb, fmt_fn, higher_better=True):
            fa, fb = fmt_fn(va), fmt_fn(vb)
            ca = cb = ""
            if va is not None and vb is not None and not (pd.isna(va) or pd.isna(vb)):
                if higher_better:
                    ca = "🟢" if va >= vb else "🔴"
                    cb = "🟢" if vb >= va else "🔴"
                else:
                    ca = "🟢" if va <= vb else "🔴"
                    cb = "🟢" if vb <= va else "🔴"
            return label, f"{ca} {fa}", f"{cb} {fb}"

        metrics_compare = [
            compare_metric("Revenue", ra.get("revenue"), rb.get("revenue"), fmt_b),
            compare_metric("Net Income", ra.get("net_income"), rb.get("net_income"), fmt_b),
            compare_metric("Market Cap", ra.get("market_cap"), rb.get("market_cap"), fmt_b),
            compare_metric("Gross Margin %", ra.get("gross_margin"), rb.get("gross_margin"), fmt_pct),
            compare_metric("Net Margin %", ra.get("net_margin"), rb.get("net_margin"), fmt_pct),
            compare_metric("Operating Margin %", ra.get("operating_margin"), rb.get("operating_margin"), fmt_pct),
            compare_metric("Revenue Growth YoY", ra.get("revenue_growth_yoy"), rb.get("revenue_growth_yoy"), fmt_pct),
        ]

        hdr_col1, hdr_col2, hdr_col3 = st.columns([2, 1, 1])
        hdr_col1.markdown(f"**Metric**")
        hdr_col2.markdown(f"**{ta}**")
        hdr_col3.markdown(f"**{tb}**")
        for label, va_str, vb_str in metrics_compare:
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(label)
            c2.markdown(va_str)
            c3.markdown(vb_str)

        # ── Section 2: Revenue Trend ──────────────────────────────────────
        if not fin_a.empty and not fin_b.empty:
            st.markdown("---")
            st.markdown("### Revenue Trend Comparison")
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(x=fin_a["year"], y=fin_a["revenue"]/1e9,
                mode="lines+markers", name=ta, line=dict(color="#1f77b4")))
            fig_rev.add_trace(go.Scatter(x=fin_b["year"], y=fin_b["revenue"]/1e9,
                mode="lines+markers", name=tb, line=dict(color="#ff7f0e")))
            fig_rev.update_layout(yaxis_title="Revenue ($B)", height=350)
            st.plotly_chart(fig_rev, use_container_width=True)

            st.markdown("### Profitability Comparison")
            fig_ni = go.Figure()
            fig_ni.add_trace(go.Scatter(x=fin_a["year"], y=fin_a["net_income"]/1e9,
                mode="lines+markers", name=ta, line=dict(color="#1f77b4")))
            fig_ni.add_trace(go.Scatter(x=fin_b["year"], y=fin_b["net_income"]/1e9,
                mode="lines+markers", name=tb, line=dict(color="#ff7f0e")))
            fig_ni.update_layout(yaxis_title="Net Income ($B)", height=350)
            st.plotly_chart(fig_ni, use_container_width=True)

        # ── Section 3: Margins ────────────────────────────────────────────
        st.markdown("### Margins Comparison")
        margin_cols = ["gross_margin", "operating_margin", "net_margin"]
        margin_labels = ["Gross Margin %", "Operating Margin %", "Net Margin %"]
        fig_margins = go.Figure()
        fig_margins.add_trace(go.Bar(name=ta, x=margin_labels,
            y=[ra.get(c) for c in margin_cols], marker_color="#1f77b4"))
        fig_margins.add_trace(go.Bar(name=tb, x=margin_labels,
            y=[rb.get(c) for c in margin_cols], marker_color="#ff7f0e"))
        fig_margins.update_layout(barmode="group", yaxis_ticksuffix="%", height=350)
        st.plotly_chart(fig_margins, use_container_width=True)

        # ── Section 4: AI Comparison ──────────────────────────────────────
        st.markdown("---")
        st.markdown("### AI Comparison Summary")
        ai_cmp_key = f"ai_compare_{ta}_{tb}"

        if st.button("Generate AI Comparison", disabled=st.session_state.get("ai_cmp_loading", False)):
            st.session_state["ai_cmp_loading"] = True
            with st.spinner("Asking Claude to compare..."):
                def company_block(row, fin):
                    lines = [f"{row['name']} ({row['ticker']}):",
                             f"  Revenue: {fmt_b(row.get('revenue'))}, Net Income: {fmt_b(row.get('net_income'))}",
                             f"  Market Cap: {fmt_b(row.get('market_cap'))}",
                             f"  Gross Margin: {fmt_pct(row.get('gross_margin'))}, Net Margin: {fmt_pct(row.get('net_margin'))}",
                             f"  Revenue Growth YoY: {fmt_pct(row.get('revenue_growth_yoy'))}"]
                    if not fin.empty:
                        for _, f in fin.iterrows():
                            lines.append(f"  {int(f['year'])}: Revenue {fmt_b(f['revenue'])}, Net Income {fmt_b(f['net_income'])}")
                    return "\n".join(lines)

                prompt = company_block(ra, fin_a) + "\n\n" + company_block(rb, fin_b)
                system = (
                    f"You are a financial analyst. Compare these two companies across revenue growth, "
                    f"profitability, margins, and market position. Write a structured comparison with: "
                    f"## Head-to-Head Summary, ## Strengths of {ta}, ## Strengths of {tb}, "
                    f"## Which to Watch and Why. Be specific with numbers. Under 400 words."
                )
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=768,
                    system=system, messages=[{"role": "user", "content": prompt}],
                )
                st.session_state[ai_cmp_key] = resp.content[0].text
            st.session_state["ai_cmp_loading"] = False

        if ai_cmp_key in st.session_state:
            st.markdown(st.session_state[ai_cmp_key])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PORTFOLIO TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Portfolio Trends":
    st.title("📉 Portfolio Trends")
    st.markdown("*Revenue and profit trends across all tracked companies*")

    if not db_exists():
        no_data_msg()
        st.stop()

    with st.spinner("Loading portfolio trends..."):
        portfolio = load_portfolio_trends()
        df = load_companies()

    # Filter to companies with financial data
    portfolio = [p for p in portfolio if p["revenue"]["values"]]

    # ── Section 1: Sparklines Grid ────────────────────────────────────────────
    st.markdown("### Company Sparklines")
    cols = st.columns(4)
    for i, p in enumerate(portfolio):
        ticker = p["ticker"]
        company_row = df[df["ticker"] == ticker]
        name = company_row["name"].values[0] if not company_row.empty else ticker
        rev_vals = p["revenue"]["values"]
        latest_rev = rev_vals[-1] if rev_vals else None
        rev_pct = p["revenue"]["yoy_pct"][-1] if p["revenue"]["yoy_pct"] else None
        ni_pct = p["net_income"]["yoy_pct"][-1] if p["net_income"]["yoy_pct"] else None

        rev_arrow = ("▲" if (rev_pct or 0) >= 0 else "▼")
        rev_color = "green" if (rev_pct or 0) >= 0 else "red"
        ni_arrow = ("▲" if (ni_pct or 0) >= 0 else "▼")
        ni_color = "green" if (ni_pct or 0) >= 0 else "red"

        with cols[i % 4]:
            # Sparkline
            spark = go.Figure()
            spark.add_trace(go.Scatter(
                x=p["years"], y=[v / 1e9 if v else 0 for v in rev_vals],
                mode="lines", line=dict(color="#1f77b4", width=1.5),
            ))
            spark.update_layout(
                height=60, margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            with st.container(border=True):
                st.markdown(f"**{ticker}** — {name[:20]}")
                st.markdown(f"Rev: {fmt_b(latest_rev)}")
                st.markdown(
                    f"<span style='color:{rev_color}'>{rev_arrow} {fmt_pct(rev_pct)}</span> Rev  "
                    f"<span style='color:{ni_color}'>{ni_arrow} {fmt_pct(ni_pct)}</span> NI",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(spark, use_container_width=True, config={"displayModeBar": False})
                if st.button("Explore →", key=f"explore_{ticker}"):
                    st.session_state["nav_page"] = "Company Explorer"
                    st.session_state["selected_ticker"] = ticker
                    st.rerun()

    # ── Section 2: Top & Bottom Performers ───────────────────────────────────
    st.markdown("---")
    st.markdown("### Top & Bottom Performers")
    perf_df = df.dropna(subset=["revenue_growth_yoy"]).copy()
    perf_df = perf_df.sort_values("revenue_growth_yoy", ascending=False)

    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("#### 🏆 Top 5 by Revenue Growth")
        top5 = perf_df.head(5)[["name", "ticker", "revenue_growth_yoy"]].reset_index(drop=True)
        top5.index += 1
        top5.columns = ["Company", "Ticker", "Revenue Growth YoY%"]
        st.dataframe(top5.style.applymap(
            lambda v: "color: green" if isinstance(v, float) and v > 0 else "",
            subset=["Revenue Growth YoY%"]
        ), use_container_width=True)

    with col_bot:
        st.markdown("#### 📉 Bottom 5 by Revenue Growth")
        bot5 = perf_df.tail(5).sort_values("revenue_growth_yoy")[["name", "ticker", "revenue_growth_yoy"]].reset_index(drop=True)
        bot5.index += 1
        bot5.columns = ["Company", "Ticker", "Revenue Growth YoY%"]
        st.dataframe(bot5.style.applymap(
            lambda v: "color: red" if isinstance(v, float) and v < 0 else "",
            subset=["Revenue Growth YoY%"]
        ), use_container_width=True)

    # ── Section 3: Stacked/Grouped Bar Chart ─────────────────────────────────
    st.markdown("---")
    st.markdown("### Portfolio Revenue Chart")
    chart_mode = st.radio("Chart style", ["Stacked", "Grouped"], horizontal=True)
    barmode = "stack" if chart_mode == "Stacked" else "group"

    all_fin = pd.read_sql_query(
        "SELECT ticker, year, revenue FROM financials ORDER BY year", sqlite3.connect(DB_PATH)
    )
    all_fin["revenue_b"] = all_fin["revenue"] / 1e9
    fig_port = px.bar(
        all_fin, x="year", y="revenue_b", color="ticker", barmode=barmode,
        title="Portfolio Revenue by Year ($B)",
        labels={"revenue_b": "Revenue ($B)", "year": "Year", "ticker": "Company"},
    )
    fig_port.update_layout(height=450)
    st.plotly_chart(fig_port, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — SECTOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sector Analysis":
    st.title("🏭 Sector Analysis")
    st.markdown("*Revenue and profitability aggregated by sector*")

    if not db_exists():
        no_data_msg()
        st.stop()

    sectors = load_sector_aggregates()
    df = load_companies()

    # ── Section 1: Sector Summary Cards ──────────────────────────────────────
    st.markdown("### Sector Overview")
    sector_list = list(sectors.keys())
    cols = st.columns(min(len(sector_list), 4))
    for i, (sector, data) in enumerate(sectors.items()):
        latest_rev = data["total_revenue"][-1] if data["total_revenue"] else 0
        latest_ni = data["total_net_income"][-1] if data["total_net_income"] else 0
        with cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"**{sector}**")
                st.markdown(f"{len(data['companies'])} companies")
                st.markdown(f"Revenue: {fmt_b(latest_rev)}")
                st.markdown(f"Net Income: {fmt_b(latest_ni)}")

    # ── Section 2: Sector Revenue Trend ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Sector Revenue Trend")
    fig_sec = go.Figure()
    for sector, data in sectors.items():
        fig_sec.add_trace(go.Scatter(
            x=data["years"],
            y=[v / 1e9 for v in data["total_revenue"]],
            name=f"{sector} ({len(data['companies'])})",
            mode="lines+markers",
        ))
    fig_sec.update_layout(yaxis_title="Total Revenue ($B)", height=400)
    st.plotly_chart(fig_sec, use_container_width=True)

    # ── Section 3: Margins by Sector ─────────────────────────────────────────
    st.markdown("### Sector Profitability (Avg Margins)")
    sector_margins = []
    for sector in sector_list:
        tickers = sectors[sector]["companies"]
        sec_df = df[df["ticker"].isin(tickers)]
        avg_gm = sec_df["gross_margin"].mean()
        avg_nm = sec_df["net_margin"].mean()
        sector_margins.append({"Sector": sector, "Gross Margin %": avg_gm, "Net Margin %": avg_nm})

    sm_df = pd.DataFrame(sector_margins).dropna()
    fig_margins = go.Figure()
    fig_margins.add_trace(go.Bar(name="Gross Margin %", x=sm_df["Sector"], y=sm_df["Gross Margin %"], marker_color="#1f77b4"))
    fig_margins.add_trace(go.Bar(name="Net Margin %", x=sm_df["Sector"], y=sm_df["Net Margin %"], marker_color="#2ca02c"))
    fig_margins.update_layout(barmode="group", yaxis_ticksuffix="%", height=380)
    st.plotly_chart(fig_margins, use_container_width=True)

    # ── Section 4: Company breakdown ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Company Breakdown by Sector")
    selected_sector = st.selectbox("Select Sector", sector_list)

    sec_tickers = sectors[selected_sector]["companies"]
    sec_df = df[df["ticker"].isin(sec_tickers)].copy()

    # Join latest revenue from financials
    conn = sqlite3.connect(DB_PATH)
    latest_fin = pd.read_sql_query("""
        SELECT ticker, revenue, net_income, gross_profit
        FROM financials
        WHERE year = (SELECT MAX(year) FROM financials f2 WHERE f2.ticker = financials.ticker)
    """, conn)
    conn.close()

    sec_detail = sec_df.merge(latest_fin, on="ticker", how="left")
    sec_detail["Revenue ($B)"] = sec_detail["revenue_y"].apply(fmt_b) if "revenue_y" in sec_detail else sec_detail.get("revenue", pd.Series()).apply(fmt_b)
    if "revenue_y" in sec_detail.columns:
        sec_detail["_rev_sort"] = sec_detail["revenue_y"]
    else:
        sec_detail["_rev_sort"] = 0

    tbl = sec_detail[["ticker", "name", "gross_margin"]].copy()
    tbl.columns = ["Ticker", "Company", "Gross Margin %"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Bar chart
    plot_df = sec_detail.dropna(subset=["_rev_sort"]).sort_values("_rev_sort", ascending=False)
    if not plot_df.empty:
        fig_sec_bar = px.bar(
            plot_df, x="name", y="_rev_sort",
            title=f"{selected_sector} — Companies by Revenue",
            labels={"_rev_sort": "Revenue ($)", "name": "Company"},
            color="_rev_sort", color_continuous_scale="Blues",
        )
        fig_sec_bar.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30, height=380)
        fig_sec_bar.update_yaxes(tickformat="$.2s")
        st.plotly_chart(fig_sec_bar, use_container_width=True)
