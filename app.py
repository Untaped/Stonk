from flask import Flask, request, render_template
from datetime import datetime
from db import Stock, SessionLocal
from db import PriceHistory
from collections import defaultdict
from datetime import timedelta
import yfinance as yf
from flask import jsonify
import pandas as pd
from patterns import detect_patterns
from pattern_utils import detect_patterns
from sqlalchemy import Column, String, Float, Date
from db import Base
import os
import csv
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Paths
MODEL_PATH = "model.pkl"
SNAPSHOT_DIR = "Stonk Download/snapshots"
PREDICTIONS_CSV = "predictions.csv"


# 🔮 Prediction Route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    symbol = None
    error = None

    if request.method == "POST":
        symbol = request.form["symbol"].upper()
        today = datetime.now().snapshots_master.csvftime("%Y-%m-%d")
        file_path = os.path.join(SNAPSHOT_DIR, f"price_history_{today}.csv")

        if not os.path.exists(file_path):
            error = f"Snapshot for {today} not found."
            return render_template("predict.html", prediction=None, error=error)

        df = pd.read_csv(file_path)
        df.columns = [col.lower() for col in df.columns]
        df = df[df["symbol"] == symbol]

        if df.empty:
            error = f"No data found for symbol '{symbol}'."
            return render_template("predict.html", prediction=None, error=error)

        try:
            df, _ = add_features(df)
        except Exception as e:
            error = f"Error in feature engineering: {e}"
            return render_template("predict.html", prediction=None, error=error)

        features = ['return_1d', 'ma_5', 'ma_10', 'volatility_5d']
        df = df.dropna(subset=features)

        if df.empty:
            error = "Not enough data to compute prediction."
            return render_template("predict.html", prediction=None, error=error)

        X_latest = df[features].iloc[[-1]]

        try:
            model = joblib.load(MODEL_PATH)
            pred_up = int(model.predict(X_latest)[0])
            pred_return = float(model.predict(X_latest)[0])  # If same model; change if separate
        except Exception as e:
            error = f"Prediction error: {e}"
            return render_template("predict.html", prediction=None, error=error)

        prediction = {
            "predicted_up": pred_up,
            "predicted_return": pred_return
        }

    return render_template("predict.html", prediction=prediction, symbol=symbol, error=error)


def load_predictions():
    """Read latest predictions from predictions.csv"""
    if not os.path.exists(PREDICTIONS_CSV):
        return []
    try:
        df = pd.read_csv(PREDICTIONS_CSV)
        df = df.sort_values("timestamp", ascending=False)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ Error reading predictions.csv: {e}")
        return []

from datetime import datetime, timezone

def predict_next_day(symbol, source="next_day"):
    try:
        # --- Get recent history ---
        df = yf.Ticker(symbol).history(period="6mo")
        if df.empty or len(df) < 10:
            print(f"⚠️ Not enough data for {symbol}")
            return None

        # --- Load model + features ---
        model = joblib.load(MODEL_PATH)
        selected_features = joblib.load("stock_ai/feature_names.pkl")
        sector_map = joblib.load("stock_ai/sector_mapping.joblib")

        # --- Prepare features (same as predict_single_stock) ---
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df["symbol"] = symbol

        from stock_ai.feature_engineering import add_features, encode_sector_column
        df = encode_sector_column(df, sector_map=sector_map)
        df = add_features(df, sector_map=sector_map)

        latest = df.dropna(subset=selected_features).tail(1)
        if latest.empty:
            print(f"⚠️ Could not build features for {symbol}")
            return None

        X_latest = latest[selected_features]

        # --- Predict probability ---
        proba = model.predict_proba(X_latest)[:, 1][0]

        # --- Save to CSV (aligns with new format) ---
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if os.path.exists(PREDICTIONS_CSV):
            try:
                preds_df = pd.read_csv(PREDICTIONS_CSV)
            except Exception as e:
                print(f"⚠️ Error reading predictions.csv: {e}, recreating file.")
                preds_df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])
        else:
            preds_df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])

        # Drop old entry for this symbol
        preds_df = preds_df[preds_df["symbol"] != symbol]

        # Add new entry
        new_row = pd.DataFrame([{
            "timestamp": ts,
            "symbol": symbol,
            "probability": round(float(proba), 6),
            "source": source
        }])
        preds_df = pd.concat([preds_df, new_row], ignore_index=True)
        preds_df.to_csv(PREDICTIONS_CSV, index=False)

        # --- Return result for API/HTML ---
        return {
            "symbol": symbol,
            "timestamp": ts,
            "probability": round(float(proba), 4),
            "source": source
        }

    except Exception as e:
        print(f"Prediction error for {symbol}: {e}")
        return None


def get_ohlc_data(symbol, days=30):
    session = SessionLocal()
    start_date = datetime.now().date() - timedelta(days=days)
    results = session.query(Stock).filter(
        Stock.symbol == symbol,
        Stock.date >= start_date
    ).order_by(Stock.date).all()
    
    ohlc = []
    for row in results:
        ohlc.append({
            "x": row.date.strftime("%Y-%m-%d"),
            "o": row.open,
            "h": row.high,
            "l": row.low,
            "c": row.close,
        })
    session.close()
    return ohlc

ticker = yf.Ticker("AAPL")
df = ticker.history(period="1mo")  # daily OHLC

df_patterns = detect_patterns(df)

print(df_patterns[['Open', 'High', 'Low', 'Close', 'Hammer', 'Doji']])

def is_hammer(row):
    body = abs(row['Close'] - row['Open'])
    lower_shadow = row['Open'] - row['Low'] if row['Close'] > row['Open'] else row['Close'] - row['Low']
    upper_shadow = row['High'] - max(row['Close'], row['Open'])
    return (
        body <= (row['High'] - row['Low']) * 0.3 and
        lower_shadow >= 2 * body and
        upper_shadow <= body * 0.3
    )

def is_doji(row, threshold=0.1):
    body = abs(row['Close'] - row['Open'])
    return body <= (row['High'] - row['Low']) * threshold

def detect_patterns(df):
    df['Hammer'] = df.apply(is_hammer, axis=1)
    df['Doji'] = df.apply(is_doji, axis=1)
    return df


def get_price_history_db(symbol, timeframe='2Y'):
    today = datetime.now().date()
    delta_days = {
        '1M': 30,
        '6M': 180,
        '1Y': 365,
        '2Y': 730
    }.get(timeframe, 730)

    start_date = today - timedelta(days=delta_days)

    try:
        df = yf.Ticker(symbol).history(start=start_date, end=today, interval="1d")
        if df.empty:
            return {
                "dates": [],
                "prices": [],
                "hammer": [],
                "doji": [],
                "year_growths": {},
                "projected_growth": None
            }
        
        # ✅ Detect candlestick patterns BEFORE returning
        df = detect_patterns(df)  # Adds Hammer/Doji columns
        
        df = df.reset_index()
        df['Date'] = df['Date'].dt.date

        # ✅ Prepare year growths as before
        df['Year'] = df['Date'].apply(lambda d: d.year)
        year_data = df.groupby('Year')['Close'].apply(list).to_dict()

        year_growths = {}
        sorted_years = sorted(year_data.keys())
        for i in range(1, len(sorted_years)):
            y_prev, y_curr = sorted_years[i - 1], sorted_years[i]
            if year_data[y_prev] and year_data[y_curr]:
                start_price = year_data[y_prev][0]
                end_price = year_data[y_curr][-1]
                if start_price and end_price:
                    growth = ((end_price - start_price) / start_price) * 100
                    year_growths[y_curr] = round(growth, 2)

        recent_growths = list(year_growths.values())[-2:]
        projected_growth = round(sum(recent_growths) / len(recent_growths), 2) if recent_growths else None

        # ✅ Return ALL needed lists with equal lengths
        return {
            "dates": df['Date'].astype(str).tolist(),
            "prices": df['Close'].tolist(),
            "hammer": df['Hammer'].tolist() if 'Hammer' in df.columns else [False] * len(df),
            "doji": df['Doji'].tolist() if 'Doji' in df.columns else [False] * len(df),
            "year_growths": year_growths,
            "projected_growth": projected_growth
        }

    except Exception as e:
        print(f"Error getting price history for {symbol}: {e}")
        return {
            "dates": [],
            "prices": [],
            "hammer": [],
            "doji": [],
            "year_growths": {},
            "projected_growth": None
        }

app = Flask(__name__)

# Load S&P 500 symbols (still useful for filtering)
import pandas as pd
def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        symbols = df['Symbol'].tolist()
        return symbols
    except Exception as e:
        print("Error fetching S&P 500 symbols:", e)
        return []

# Get fundamentals from the database
def get_fundamentals(symbol, date=None):
    session = SessionLocal()
    if not date:
        date = datetime.now().date()

    stock = session.query(Stock).filter(Stock.symbol == symbol).order_by(Stock.date.desc()).first()

    session.close()

    if not stock:
        raise ValueError(f"No data found for {symbol} on {date}")

    return {
        'shortName': stock.name,
        'symbol': stock.symbol,
        'price': stock.price,
        'marketCap': stock.market_cap,
        'averageVolume': stock.average_volume,
        'revenueGrowth': stock.revenue_growth,
        'earningsGrowth': stock.earnings_growth,
        'netIncomeToCommon': stock.net_income,
        'returnOnEquity': stock.roe,
        'debtToEquity': stock.debt_to_equity,
        'currentRatio': stock.current_ratio,
        'freeCashflow': stock.free_cashflow,
        'forwardPE': stock.forward_pe,
        'pegRatio': stock.peg_ratio,
        'ipoDate': None  # IPO date not stored yet in db
    }

# Placeholder for price history (optional future use)
def get_price_history(symbol):
    return {
        "dates": [],
        "prices": []
    }

def get_projected_growth_metrics(symbol):
    import pandas as pd
    ticker = yf.Ticker(symbol)

    # --- Dividends ---
    dividends = ticker.dividends
    if dividends.empty:
        dividend_growth = None
    else:
        dividends_by_year = dividends.groupby(dividends.index.year).sum()
        dividend_growth = calc_avg_growth(dividends_by_year)

    # --- PP&E ---
    try:
        bs = ticker.balance_sheet
        ppe = bs.loc['Property Plant Equipment']
        ppe = ppe[::-1]  # reverse to chronological order
        ppe.index = pd.to_datetime(ppe.index).year
        ppe_growth = calc_avg_growth(ppe)
    except Exception as e:
        print(f"Error fetching PP&E: {e}")
        ppe_growth = None

    return {
        "dividend_growth": dividend_growth,
        "ppe_growth": ppe_growth
    }

def calc_avg_growth(series):
    sorted_years = sorted(series.index)
    growths = []

    for i in range(1, len(sorted_years)):
        prev = series[sorted_years[i - 1]]
        curr = series[sorted_years[i]]
        if prev and curr and prev != 0:
            growths.append(((curr - prev) / prev) * 100)

    if len(growths) >= 2:
        return round(sum(growths[-2:]) / 2, 2)
    elif growths:
        return round(growths[-1], 2)
    else:
        return None


# Scoring algorithm remains unchanged
def evaluate_stock_criteria(f):
    score = 0
    total_criteria = 12
    reasons = []

    def safe_gt(val, threshold, label):
        if val is not None and val > threshold:
            reasons.append(label)
            return True
        return False

    def safe_lt(val, threshold, label):
        if val is not None and val < threshold:
            reasons.append(label)
            return True
        return False

    if safe_gt(f.get('marketCap'), 2_000_000_000, "Market cap > $2B"): score += 1
    if safe_gt(f.get('averageVolume'), 500_000, "High average volume"): score += 1
    if safe_gt(f.get('revenueGrowth'), 0.1, "Revenue growth > 10%"): score += 1.5
    if safe_gt(f.get('earningsGrowth'), 0.12, "Earnings growth > 12%"): score += 1.5
    if safe_gt(f.get('netIncomeToCommon'), 0, "Positive net income"): score += 1
    if safe_gt(f.get('returnOnEquity'), 0.15, "ROE > 15%"): score += 1.5
    if safe_lt(f.get('debtToEquity'), 0.8, "Low debt-to-equity (< 0.8)"): score += 1
    if safe_gt(f.get('currentRatio'), 2.0, "Healthy current ratio (> 2.0)"): score += 0.5
    if safe_gt(f.get('freeCashflow'), 0, "Positive free cash flow"): score += 1.5
    if safe_lt(f.get('forwardPE'), 20.0, "Reasonable Forward PE < 20"): score += 1
    if safe_lt(f.get('pegRatio'), 1.5, "PEG ratio < 1.5"): score += 1.5

    pct = (score / total_criteria) * 100

    if pct >= 60:
        recommendation = f"BUY: Strong candidate ({pct:.0f}% of criteria met)"
    elif pct >= 40:
        recommendation = f"WATCHLIST: Meets {pct:.0f}% of criteria"
    else:
        recommendation = f"DO NOT BUY: Only {pct:.0f}% of criteria met"

    return score, recommendation, reasons

# Routes

def build_candlestick_data(df):
    return [
        {
            "x": row.name.strftime('%Y-%m-%d'),
            "o": row['Open'],
            "h": row['High'],
            "l": row['Low'],
            "c": row['Close']
        }
        for _, row in df.iterrows()
    ]

@app.route('/sp500')
def sp500_list():
    session = SessionLocal()
    today = datetime.now().date()
    symbols = get_sp500_symbols()

    stocks = session.query(Stock).filter(
        Stock.date == today,
        Stock.symbol.in_(symbols),
        Stock.score >= 10
    ).order_by(Stock.symbol).all()

    session.close()

    return render_template("sp500.html", stocks=[{
        'symbol': s.symbol,
        'shortName': s.name,
        'recommendation': s.recommendation,
        'score': s.score
    } for s in stocks])

@app.route('/database')
def view_from_database():
    session = SessionLocal()
    today = datetime.now().date()

    stocks = session.query(Stock).filter(Stock.date == today, Stock.score >= 8).order_by(Stock.symbol).all()
    session.close()

    return render_template("sp500.html", stocks=[{
        'symbol': s.symbol,
        'shortName': s.name,
        'recommendation': s.recommendation,
        'score': s.score
    } for s in stocks])

@app.route('/history')
def show_history():
    session = SessionLocal()
    data = session.query(Stock).order_by(Stock.date.desc(), Stock.symbol).limit(100).all()
    session.close()
    return render_template("history.html", stocks=data)

@app.route('/', methods=['GET', 'POST'])
def index():
    # ✅ Always define defaults so they're accessible
    timeframe = "2Y"  
    symbol = ""
    ohlc_data = None
    stock_data = None
    recommendation = None
    price_history = None
    error = None
    growth_summary = None
    pattern_results = None
    top_reasons = []
    closes, dates, hammer, doji = [], [], [], []
    prediction_result = None

    if request.method == 'POST':
        # ✅ Extract values from POST safely
        symbol = request.form.get('symbol', '').upper().strip()
        timeframe = request.form.get('timeframe', '2Y')  # override default only if POSTed

        try:
            # ✅ Always define timeframe_map INSIDE this block
            timeframe_map = {"1M": 30, "6M": 180, "1Y": 365, "2Y": 730}
            days = timeframe_map.get(timeframe, 30)

            # ✅ Now safe to call DB for OHLC
            ohlc_data = get_ohlc_data(symbol, days=days)

            # ✅ Fundamentals + Recommendation
            fundamentals = get_fundamentals(symbol)
            stock_data = fundamentals
            score, recommendation_text, top_reasons = evaluate_stock_criteria(fundamentals)
            recommendation = recommendation_text

            # ✅ Price history normalized
            price_history = get_price_history_db(symbol, timeframe)
            preds = load_predictions()
            ai_pred = next((p for p in preds if p["symbol"] == symbol), None)
            closes = price_history.get("prices", [])
            dates = price_history.get("dates", [])
            hammer = price_history.get("hammer", [])
            doji = price_history.get("doji", [])
            print("Lengths:", len(closes), len(dates), len(hammer), len(doji))

            # ✅ Growth metrics
            growth_metrics = get_projected_growth_metrics(symbol)
            growth_summary = {
                "projected_price": price_history.get("projected_growth", 0),
                "dividend_growth": growth_metrics.get("dividend_growth", 0),
                "ppe_growth": growth_metrics.get("ppe_growth", 0)
            }

            # ✅ Pattern detection only if valid symbol
            if symbol:
                df = yf.Ticker(symbol).history(period="1mo")
                df = detect_patterns(df)
                pattern_results = df[df[['Hammer', 'Doji']].any(axis=1)][['Hammer', 'Doji']]
                hammer = df['Hammer'].tolist() if 'Hammer' in df.columns else []
                doji = df['Doji'].tolist() if 'Doji' in df.columns else []

        except Exception as e:
            error = f"Error retrieving data: {e}"
            closes, dates, hammer, doji = [], [], [], []

    # ✅ Always render safely, even on GET
    return render_template(
        'index.html',
        symbol=symbol,
        stock_data=stock_data,
        recommendation=recommendation,
        price_history=price_history,
        growth_summary=growth_summary,
        pattern_results=pattern_results,
        error=error,
        timeframe=timeframe,
        top_reasons=top_reasons,
        ohlc_data=ohlc_data,
        closes=closes,
        dates=dates,
        hammer=hammer,
        doji=doji,
        prediction_result=prediction_result
    )

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    results = []
    total_initial = 0
    total_final = 0

    if request.method == 'POST':
        for i in range(5):
            symbol = request.form.get(f'symbol{i}', '').strip().upper()
            amount = request.form.get(f'amount{i}', '').strip()
            if not symbol or not amount:
                continue

            try:
                amount = float(amount)
                data = get_price_history_db(symbol)
                prices = data.get('prices', [])
                if len(prices) < 2:
                    continue

                start_price = prices[0]
                end_price = prices[-1]
                shares = amount / start_price
                final_value = shares * end_price

                results.append({
                    'symbol': symbol,
                    'start_price': round(start_price, 2),
                    'end_price': round(end_price, 2),
                    'amount': amount,
                    'final_value': round(final_value, 2),
                    'gain_pct': round(((final_value - amount) / amount) * 100, 2)
                })

                total_initial += amount
                total_final += final_value

            except Exception as e:
                print(f"Error processing {symbol}: {e}")

    return render_template(
        "portfolio.html",
        results=results,
        total_initial=round(total_initial, 2),
        total_final=round(total_final, 2)
    )

@app.route('/api/price_history/<symbol>')
def api_price_history(symbol):
    data = get_price_history_db(symbol)
    return jsonify({"prices": data.get("prices", [])})

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])

    symbols = get_sp500_symbols()
    suggestions = [s for s in symbols if query in s.upper()]
    return jsonify(suggestions[:10])

from pattern_utils import detect_patterns  # your pattern module

if __name__ == '__main__':
    app.run(debug=True)