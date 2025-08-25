import yfinance as yf
import pandas as pd


def fetch_stock_metrics(symbol):
    """
    Fetches stock metrics such as projected price growth, dividend growth,
    and PP&E growth for the past 2 years.
    Returns a dictionary with the results or None on error.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        if hist.empty or "Close" not in hist:
            return None

        # Price Growth (simple projected estimate based on 2y)
        price_growth = ((hist["Close"][-1] - hist["Close"][0]) / hist["Close"][0]) * 100

        # Dividend Growth
        dividends = ticker.dividends
        if not dividends.empty:
            dividends = dividends.resample("Y").sum()
            if len(dividends) >= 2:
                dividend_growth = ((dividends[-1] - dividends[-2]) / dividends[-2]) * 100
            else:
                dividend_growth = 0.0
        else:
            dividend_growth = 0.0

        # PP&E Growth
        financials = ticker.balance_sheet
        if "Property Plant Equipment" in financials:
            ppe = financials.loc["Property Plant Equipment"]
            if len(ppe) >= 2:
                ppe_growth = ((ppe.iloc[0] - ppe.iloc[1]) / abs(ppe.iloc[1])) * 100
            else:
                ppe_growth = 0.0
        else:
            ppe_growth = 0.0

        return {
            "projected_price": round(price_growth, 2),
            "dividend_growth": round(dividend_growth, 2),
            "ppe_growth": round(ppe_growth, 2)
        }

    except Exception as e:
        print(f"Error fetching stock metrics for {symbol}: {e}")
        return None
