from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
from datetime import datetime
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from flask import send_from_directory
from datetime import timedelta
import requests
from io import StringIO
from datetime import datetime

load_dotenv()

app = Flask(__name__)

# Configuration
PREDICTIONS_CSV = "stonk_download/predictions.csv"
FUNDAMENTALS_CSV = "stonk_download/fundamentals.csv"
PREDICTIONS_5PCT_CSV = "stonk_download/predictions_5pct.csv"
NASDAQ_PREDICTIONS_CSV = "stonk_download/nasdaq_predictions.csv"

def get_last_update_time():
    path = "stonk_download/predictions.csv"
    if os.path.exists(path):
        # Get file modification time
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    return "No Data"

MARKET_DATA_DIR = os.getenv('DATA_DIR', "market_data")
MARKET_FUNDAMENTALS_CSV = os.path.join(MARKET_DATA_DIR, "market_sp500_fundamentals.csv")
MARKET_PRICE_HISTORY_CSV = os.path.join(MARKET_DATA_DIR, "market_sp500_price_history.csv")

# Use relative paths consistently to avoid Mac absolute path errors
MARKET_PREDICTIONS_CSV = "stonk_download/market_predictions_multi_threshold.csv"
MARKET_PREDICTIONS_SCAN_CSV = "stonk_download/market_predictions.csv"

MARKET_PREDICTIONS_5PCT_CSV = os.getenv('MARKET_PREDICTIONS_5PCT_CSV', "stonk_download/market_predictions_5pct.csv")
MARKET_MODEL_PATHS = {
    '1pct': 'stock_ai/market_lightgbm_model_1pct.pkl',
    '3pct': 'stock_ai/market_lightgbm_model_3pct.pkl',
    '5pct': 'stock_ai/market_lightgbm_model_5pct.pkl',
    '10pct': 'stock_ai/market_lightgbm_model_10pct.pkl'
}
MARKET_FEATURE_PATHS = {
    '1pct': 'stock_ai/market_feature_names_1pct.pkl',
    '3pct': 'stock_ai/market_feature_names_3pct.pkl',
    '5pct': 'stock_ai/market_feature_names_5pct.pkl',
    '10pct': 'stock_ai/fmarket_eature_names_10pct.pkl'
}
MARKET_MODEL_5PCT_PATH = "stock_ai/market_lightgbm_model_5pct.pkl"
MARKET_FEATURE_NAMES_5PCT_PATH = "stock_ai/market_feature_names_5pct.pkl"
MARKET_PREDICTIONS_5PCT_CSV = "stonk_download/market_predictions_5pct.csv"

PMARKET_REDICTIONS_SCAN_CSV = "stonk_download/market_predictions.csv"

# --- Helper Functions ---

def load_predictions(file_path):
    """Read predictions from CSV"""
    if not os.path.exists(file_path):
        return []
    try:
        df = pd.read_csv(file_path)
        # Ensure standard@a column names
        df.columns = [c.lower() for c in df.columns]
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return []

def get_sp500_symbols():
    """Scrape S&P 500 symbols from Wikipedia with a User-Agent"""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Check for errors
        table = pd.read_html(StringIO(response.text))
        df = table[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def evaluate_stock_criteria(fundamentals):
    """
    Evaluates stock fundamentals and returns a score, recommendation, and list of reasons.
    """
    score = 0
    reasons = []
    
    rev_growth = fundamentals.get('revenueGrowth')
    if rev_growth and rev_growth > 0:
        score += 20
        reasons.append("Positive Revenue Growth")
        
    # Example logic: Add points for reasonable PE ratio
    pe = fundamentals.get('forwardPE')
    if pe and 0 < pe < 25:
        score += 20
        reasons.append("Healthy Forward PE")

    # Determine recommendation based on score
    if score >= 38:
        recommendation = "BUY"
    else:
        recommendation = "DISREGARD"
        reasons.append("Weak Fundamentals")

    return score, recommendation, reasons

def get_latest_prediction(symbol):
    """Find specific symbol in S&P 500 OR NASDAQ"""
    symbol_clean = symbol.strip().upper()
    
    # 1. Search S&P 500
    sp500_preds = load_predictions(PREDICTIONS_CSV)
    for row in sp500_preds:
        if row.get('symbol', '').strip().upper() == symbol_clean:
            return row
            
    # 2. Search NASDAQ
    nasdaq_preds = load_predictions(NASDAQ_PREDICTIONS_CSV)
    for row in nasdaq_preds:
        if row.get('symbol', '').strip().upper() == symbol_clean:
            return row
            
    return None

def get_fundamentals(symbol):
    """Get basic info (Sector, Market Cap) if file exists"""
    try:
        if os.path.exists(FUNDAMENTALS_CSV):
            df = pd.read_csv(FUNDAMENTALS_CSV)
            row = df[df['symbol'] == symbol]
            if not row.empty:
                data = row.iloc[0].to_dict()
                return {
                    'shortName': data.get('name', symbol),
                    'sector': data.get('sector', 'Unknown'),
                    'marketCap': data.get('market_cap', 0),
                    'price': data.get('price', 0)
                }
    except Exception:
        pass
    # Return dummy data if file missing or error
    return {'shortName': symbol, 'sector': 'Unknown', 'marketCap': 0, 'price': 0}

# --- Routes ---

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'mode': 'lightweight_viewer'})

@app.route('/')
def index(): # Changed to GET-only for cleaner initial load, form submits to same URL
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    symbol = request.form.get('symbol', '').upper().strip()
    prediction = None
    stock_data = None
    error = None

    if symbol:
        # Now searches BOTH databases
        raw_data = get_latest_prediction(symbol)
        
        if raw_data:
            prediction = {
                'symbol': raw_data.get('symbol'),
                'recommendation': raw_data.get('recommendation', 'HOLD'),
                'probability': raw_data.get('probability', 0), 
                '3pct': raw_data.get('prob_3pct', 0),
                '5pct': raw_data.get('prob_5pct', 0),
                '10pct': raw_data.get('prob_10pct', 0),
                '1mo': raw_data.get('prob_1mo', 0),
            }
            stock_data = get_fundamentals(symbol)
        else:
            error = f"No prediction found for {symbol} in S&P 500 or NASDAQ databases."

    #return render_template(
        #'index.html',
        #symbol=symbol,
        #prediction=prediction,
        #stock_data=stock_data,
        #error=error,
        #ohlc_data=[], closes=[], dates=[]
    #)

    # We pass empty lists for charts to disable them without breaking HTML
    return render_template(
        'index.html',
        symbol=symbol,
        prediction=prediction,
        stock_data=stock_data,
        error=error,
        ohlc_data=[],
        closes=[],
        dates=[],
        growth_summary=None,
        price_history=None
    )

@app.route('/api/stocks')
def api_stocks():
    preds = load_predictions(PREDICTIONS_CSV)
    # This sends raw data that your Android app can "consume"
    return jsonify(preds)

@app.route('/manifest.json')
def serve_manifest():
    return send_from_directory('static', 'manifest.json', mimetype='application/manifest+json')

@app.route('/sw.js')
def serve_sw():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

#@app.route('/login', methods=['GET', 'POST'])
#def login():
    #error = None
    #if request.method == 'POST':
        # Check if the code matches
        #if request.form.get('code') == ACCESS_CODE:
            #session.permanent = True
            #session['sp500_unlocked'] = True
            #return redirect(url_for('sp500_list'))
        #else:
            #error = "Invalid access code. Please try again."
            # We don't redirect here; we just fall through to render_template
            # This ensures the 'error' variable actually makes it to the page
            #flash(error, "danger")
    
    #return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('sp500_unlocked', None)
    flash("You have been logged out safely.", "info") # Add this
    return redirect(url_for('index'))

@app.route('/sp500')
def sp500_list():
    # --- SECURITY CHECK ---
    # if not session.get('sp500_unlocked'):
    #    return redirect(url_for('login'))
    # ----------------------

    """Show top predictions from CSV"""
    preds = load_predictions(PREDICTIONS_CSV)
    
    formatted_stocks = []
    for p in preds:
        # Read 'combined_score' (which is what predict_single_stock.py saves)
        # Fallback to 'probability' if combined_score is missing
        raw_score = p.get('combined_score', p.get('probability', 0))
        try:
            prob = float(raw_score)
        except:
            prob = 0.0

        consistency = float(p.get('consistency', 0))
        
        if prob >= 0.35 and consistency >= 0.85:
            recommendation = "BUY" # Strong Buy
        elif prob >= 0.35:
            recommendation = "CONSIDER"
        else:
            recommendation = "DISREGARD"

        formatted_stocks.append({
            'symbol': p.get('symbol'),
            'score_percent': f"{prob*100:.1f}%",
            'probability': prob, 
            'recommendation': recommendation, 
            'currentPrice': float(p.get('price', 0)), 
            'shortName': p.get('name', p.get('symbol')), 
            'sector': p.get('sector', 'Unknown')
        })

    # Sort by probability descending
    try:
        formatted_stocks.sort(key=lambda x: x['probability'], reverse=True)
    except:
        pass

    return render_template("SP500.html", stocks=formatted_stocks[:100], last_updated="Static Data")

@app.route('/nasdaq')
def nasdaq_list():
    """Show S&P 500 stocks using predictions from CSV data - FULLY OFFLINE"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stocks = []
    
    try:
        # Check if file exists
        if not os.path.exists(MARKET_PREDICTIONS_CSV):
             return render_template("nasdaq.html", stocks=[], 
                                 error=f"File not found: {MARKET_PREDICTIONS_CSV}. Run market_predict_single_stock.py --scan first.",
                                 last_updated=current_time)

        df_latest = pd.read_csv(MARKET_PREDICTIONS_CSV)
        
        if df_latest.empty:
            return render_template("nasdaq.html", stocks=[], 
                                 error="No predictions found in CSV.",
                                 last_updated=current_time)
        
        # Get the most recent prediction for each symbol
        if 'timestamp' in df_latest.columns:
            df_latest = df_latest.sort_values('timestamp').groupby('symbol').tail(1)
        
        print(f"Found {len(df_latest)} unique symbols with predictions")
        
        for _, row in df_latest.iterrows():
            try:
                symbol = row['symbol'].upper()
                
                # Handle different column names (combined_score vs probability) ---
                raw_score = row.get('combined_score', row.get('probability', 0))
                probability = float(raw_score)
                # ------------------------------------------------------------------------

                # Create recommendation based on probability
                # Use consistency if available, otherwise default to high prob
                consistency = float(row.get('consistency', 1.0))

                if probability >= 0.80 and consistency >= 0.85:
                    recommendation = 'BUY'
                elif probability >= 0.75:
                    recommendation = 'CONSIDER' # Matches SP500 logic
                else:
                    recommendation = 'DISREGARD'
                
                # Try to get fundamentals data from CSV if available
                # (You can use your existing helper or row data if the CSV has it)
                stock_name = row.get('name', symbol) 
                sector = row.get('sector', 'Unknown')
                price = float(row.get('price', 0))

                stocks.append({
                    'symbol': symbol,
                    'shortName': stock_name,
                    'sector': sector,
                    'currentPrice': price,
                    'probability': probability,
                    'recommendation': recommendation,
                    'score_percent': f"{probability * 100:.1f}%",
                    'score': f"{probability:.3f}",
                    'timestamp': row.get('timestamp', '')
                })
                
            except Exception as e:
                print(f"Error processing {row.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by probability (highest first)
        stocks.sort(key=lambda x: x['probability'], reverse=True)
        
        return render_template("nasdaq.html", stocks=stocks, last_updated=current_time)

    except Exception as e:
        print(f"Error in nasdaq_list: {e}")
        return render_template("nasdaq.html",stocks= [], error=str(e), last_updated=current_time)
    
@app.route('/autocomplete')
def autocomplete():
    """Autocomplete merging both S&P 500 and NASDAQ"""
    query = request.args.get('q', '').upper()
    
    # Load both lists
    sp500 = load_predictions(PREDICTIONS_CSV)
    nasdaq = load_predictions(NASDAQ_PREDICTIONS_CSV)
    
    # Create a set of all symbols to avoid duplicates
    all_symbols = set()
    for p in sp500:
        if p.get('symbol'): all_symbols.add(p['symbol'])
    for p in nasdaq:
        if p.get('symbol'): all_symbols.add(p['symbol'])
        
    # Filter matches
    matches = [s for s in all_symbols if query in s]
    matches.sort()
    
    return jsonify(matches[:10])

if __name__ == '__main__':
    app.run(debug=True)