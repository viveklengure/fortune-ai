"""
Fortune AI - Entry Point CLI
"""

import subprocess
import sys


BANNER = """
╔═══════════════════════════════════════════╗
║         === Fortune AI ===                ║
║   Financial Intelligence Platform        ║
╚═══════════════════════════════════════════╝
"""

MENU = """
  1. Ingest financial data
  2. Generate embeddings
  3. Full setup (ingest + embed)
  4. Start Streamlit app
  5. Test forecasting module
  6. Test trends module
  7. Exit

Select an option: """


def ingest():
    print("\n[1/1] Ingesting financial data from yfinance...")
    from src.ingest import run_ingestion
    run_ingestion()


def embed():
    print("\n[1/1] Generating embeddings into ChromaDB...")
    from src.embed import run_embedding
    run_embedding()


def full_setup():
    print("\n[1/2] Ingesting financial data...")
    from src.ingest import run_ingestion
    run_ingestion()
    print("\n[2/2] Generating embeddings...")
    from src.embed import run_embedding
    run_embedding()
    print("\n✅ Full setup complete. Run option 4 to start the app.")


def start_app():
    print("\nStarting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    print(BANNER)
    while True:
        try:
            choice = input(MENU).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice == "1":
            ingest()
        elif choice == "2":
            embed()
        elif choice == "3":
            full_setup()
        elif choice == "4":
            start_app()
        elif choice == "5":
            print("\nTesting forecasting module on AAPL...")
            from src.forecast import statistical_forecast, ai_forecast_narrative
            fc = statistical_forecast("AAPL")
            print(f"Forecast years: {fc.get('forecast', {}).get('years')}")
            print(f"Revenue forecast: {[f'${v/1e9:.1f}B' for v in fc.get('forecast', {}).get('revenue', [])]}")
            print(f"Confidence: {fc.get('confidence')}")
        elif choice == "6":
            print("\nTesting trends module...")
            from src.trends import get_yoy_trends, get_portfolio_trends, get_sector_aggregates
            t = get_yoy_trends("AAPL")
            print(f"AAPL years: {t['years']}")
            print(f"AAPL revenue YoY%: {[f'{v:.1f}%' if v else 'N/A' for v in t['revenue']['yoy_pct']]}")
            sectors = get_sector_aggregates()
            for s, d in sectors.items():
                print(f"  {s}: {len(d['companies'])} companies")
        elif choice == "7":
            print("Goodbye.")
            break
        else:
            print("Invalid option. Please choose 1–5.")


if __name__ == "__main__":
    main()
