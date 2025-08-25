import os
import sys
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime

# Append parent directory so we can import feature_engineering
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stock_ai.feature_engineering import add_features, encode_sector_column, get_sector_mapping

MODEL_PATH = "stock_ai/lightgbm_model.pkl"
FEATURE_NAMES_PATH = "stock_ai/feature_names.pkl"
SECTOR_MAPPING_PATH = "stock_ai/sector_mapping.joblib"

PREDICTIONS_FILE = "stonk_download/predictions.csv"   # ✅ store results here

REQUIRED_RAW_COLS = ["open", "high", "low", "close", "volume"]

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV columns to: open, high, low, close, adj_close, volume."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # drop ticker symbol

    df.columns = [str(c).strip().lower() for c in df.columns]

    col_map = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
    }
    df.rename(columns=col_map, inplace=True)

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df

def fetch_stock_data(symbol, period="6mo"):
    """Fetch OHLCV data from Yahoo Finance and normalize column names."""
    try:
        df = yf.download(symbol, period=period, auto_adjust=False)

        if df.empty:
            return None

        df = normalize_ohlcv_columns(df)
        return df.reset_index()

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        return None

def prepare_features(df, symbol):
    """Add engineered features and encode sectors."""
    sector_map = joblib.load(SECTOR_MAPPING_PATH)
    df["symbol"] = symbol
    df = encode_sector_column(df, sector_map=sector_map)
    df = add_features(df, sector_map=sector_map)
    return df

def predict_for_symbol(symbol):
    model = joblib.load(MODEL_PATH)
    selected_features = joblib.load(FEATURE_NAMES_PATH)

    raw_df = fetch_stock_data(symbol)
    if raw_df is None:
        return None

    raw_df["symbol"] = symbol
    feat_df = prepare_features(raw_df, symbol)

    # Keep only last row for prediction
    latest = feat_df.dropna(subset=selected_features).tail(1)
    if latest.empty:
        print(f"⚠️ Not enough data to compute features for {symbol}")
        return None

    latest = latest[selected_features]

    proba = model.predict_proba(latest)[:, 1][0]
    return proba

from datetime import datetime, timezone

def save_prediction(symbol, proba, source="manual"):
    """Save or update prediction so only the latest per symbol is kept."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")  # ✅ modern UTC

    # Try loading existing CSV
    if os.path.exists(PREDICTIONS_FILE):
        try:
            df = pd.read_csv(PREDICTIONS_FILE)
            # Ensure only expected columns
            expected_cols = ["timestamp", "symbol", "probability", "source"]
            if not all(col in df.columns for col in expected_cols):
                print("⚠️ predictions.csv has wrong columns, recreating...")
                df = pd.DataFrame(columns=expected_cols)
        except Exception as e:
            print(f"⚠️ Could not read predictions.csv ({e}), recreating file.")
            df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])
    else:
        df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])

    # Drop any old row for this symbol
    df = df[df["symbol"] != symbol]

    # Append new row
    new_row = pd.DataFrame([{
        "timestamp": ts,
        "symbol": symbol,
        "probability": round(float(proba), 6),  # ✅ clean float
        "source": source
    }])
    df = pd.concat([df, new_row], ignore_index=True)

    # Save back
    df.to_csv(PREDICTIONS_FILE, index=False)
    print(f"💾 Saved prediction for {symbol} to {PREDICTIONS_FILE}")

if __name__ == "__main__":
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    print(f"🔍 Fetching data for {symbol}...")
    score = predict_for_symbol(symbol)
    if score is not None:
        print(f"✅ Prediction probability for {symbol}: {score:.4f}")
        save_prediction(symbol, score, source="manual")   # ✅ overwrite old entry
    else:
        print(f"❌ Could not fetch or process stock data.")

    print("\n📊 Scanning all S&P 500 stocks...")
    try:
        sp500_df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = sp500_df["Symbol"].tolist()
    except Exception as e:
        print(f"❌ Could not fetch S&P 500 list: {e}")
        tickers = []

    results = []
    for t in tickers:
        proba = predict_for_symbol(t)
        if proba is not None:
            results.append((t, proba))
            save_prediction(t, proba, source="sp500_scan")   # ✅ overwrite old entry

    if results:
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        print("\n🏆 Top 10 predictions in S&P 500:")
        for sym, sc in ranked[:10]:
            print(f"{sym}: {sc:.4f}")