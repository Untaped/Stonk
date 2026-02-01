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

# --- CONFIGURATION ---
# 1. Secret Key is REQUIRED for sessions. 
app.secret_key = os.environ.get('SECRET_KEY', 'dev_default_key_change_this_in_prod')

# 2. Set your Access Code
ACCESS_CODE = os.environ.get('SP500_ACCESS_CODE', '12345')


# Configuration
PREDICTIONS_CSV = "stonk_download/predictions.csv"
PREDICTIONS_5PCT_CSV = "stonk_download/predictions_5pct.csv"
NASDAQ_PREDICTIONS_CSV = "stonk_download/nasdaq_predictions.csv"

def get_last_update_time():
    path = "stonk_download/predictions.csv"
    if os.path.exists(path):
        # Get file modification time
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
    return "No Data"

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
    if score >= 40:
        recommendation = "BUY"
    elif score >= 20:
        recommendation = "HOLD"
    else:
        recommendation = "SELL"
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
                '1pct': raw_data.get('prob_1pct', 0),   
                '3pct': raw_data.get('prob_3pct', 0),
                '5pct': raw_data.get('prob_5pct', 0),
                '10pct': raw_data.get('prob_10pct', 0),
                '1mo': raw_data.get('prob_1mo', 0),
            }
            stock_data = get_fundamentals(symbol)
        else:
            error = f"No prediction found for {symbol} in S&P 500 or NASDAQ databases."

    return render_template(
        'index.html',
        symbol=symbol,
        prediction=prediction,
        stock_data=stock_data,
        error=error,
        ohlc_data=[], closes=[], dates=[]
    )

    # We pass empty lists for charts to disable them without breaking HTML
    return render_template(
        'index.html',
        symbol=symbol,
        prediction=prediction,
        stock_data=stock_data,
        error=error,
        # EMPTY CHART DATA
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
    error = None
    if request.method == 'POST':
        # Check if the code matches
        if request.form.get('code') == ACCESS_CODE:
            session.permanent = True
            session['sp500_unlocked'] = True
            return redirect(url_for('sp500_list'))
        else:
            error = "Invalid access code. Please try again."
            # We don't redirect here; we just fall through to render_template
            # This ensures the 'error' variable actually makes it to the page
            flash(error, "danger")
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('sp500_unlocked', None)
    flash("You have been logged out safely.", "info") # Add this
    return redirect(url_for('index'))

@app.route('/sp500')
def sp500_list():
    # --- SECURITY CHECK ---
    #if not session.get('sp500_unlocked'):
        #return redirect(url_for('login'))
    # ----------------------

    """Show top predictions from CSV"""
    preds = load_predictions(PREDICTIONS_CSV)
    
    try:
        preds.sort(key=lambda x: float(x.get('probability', 0)), reverse=True)
    except:
        pass

    formatted_stocks = []
    for p in preds:
        prob = float(p.get('probability', 0))
        formatted_stocks.append({
            'symbol': p.get('symbol'),
            'score_percent': f"{prob*100:.1f}%",
            'probability': prob,  # <--- ADD THIS LINE
            'recommendation': p.get('recommendation', 'N/A'),
            'currentPrice': p.get('price', 'N/A'),
            'shortName': p.get('name', p.get('symbol')), # Ensure name exists for table
            'sector': p.get('sector', 'Unknown')        # Ensure sector exists for table
        })

    return render_template("SP500.html", stocks=formatted_stocks[:100], last_updated="Static Data")

@app.route('/nasdaq')
def nasdaq_list():
    preds = load_predictions(NASDAQ_PREDICTIONS_CSV)
    
    formatted_stocks = []
    for p in preds:
        # 1. Normalize 'probability' (Handle different column names)
        # Some CSVs use 'combined_score', others 'probability'
        raw_score = p.get('combined_score', p.get('probability', 0))
        try:
            prob = float(raw_score)
        except:
            prob = 0.0

        # 2. Create the display variables
        p['probability'] = prob
        p['score_percent'] = f"{prob*100:.1f}%"
        
        # 3. Ensure other required fields exist
        if 'shortName' not in p:
            p['shortName'] = p.get('name', p.get('symbol', 'N/A'))
            
        if 'currentPrice' not in p:
            p['currentPrice'] = p.get('price', 0)
            
        if 'sector' not in p:
            p['sector'] = "Unknown"

        formatted_stocks.append(p)

    # Sort by high confidence
    try:
        formatted_stocks.sort(key=lambda x: float(x.get('probability', 0)), reverse=True)
    except:
        pass

    return render_template("NASDAQ.html", stocks=formatted_stocks[:100], last_updated="Live")
# --- PHONE ROUTES ---

@app.route('/indexphone', methods=['GET', 'POST'])
def index_phone():
    """Phone version of the Dashboard"""
    symbol = ""
    prediction = None
    stock_data = None
    error = None

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').upper().strip()
        
        # Look up prediction (Same logic as desktop)
        raw_data = get_latest_prediction(symbol)
        
        if raw_data:
            prediction = {
                'symbol': raw_data.get('symbol'),
                'recommendation': raw_data.get('recommendation', 'HOLD'),
                'probability': raw_data.get('probability', 0),
                '1pct': raw_data.get('prob_1pct', 0),   
                '3pct': raw_data.get('prob_3pct', 0),
                '5pct': raw_data.get('prob_5pct', 0),
                '10pct': raw_data.get('prob_10pct', 0),
                '1mo': raw_data.get('prob_1mo', 0),
            }
            stock_data = get_fundamentals(symbol)
        else:
            error = f"No prediction found for {symbol}."

    return render_template(
        'indexphone.html',
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

@app.route('/sp500phone')
def sp500_list_phone():
    # --- SECURITY CHECK ---
    # This ensures they must be logged in to view the phone page
    #if not session.get('sp500_unlocked'):
        #return redirect(url_for('login'))
    # ----------------------

    """Show top predictions from CSV for Phone"""
    preds = load_predictions(PREDICTIONS_CSV)
    
    try:
        preds.sort(key=lambda x: float(x.get('probability', 0)), reverse=True)
    except:
        pass

    formatted_stocks = []
    for p in preds:
        prob = float(p.get('probability', 0))
        formatted_stocks.append({
            'symbol': p.get('symbol'),
            'score_percent': f"{prob*100:.1f}%",
            'probability': prob,
            'recommendation': p.get('recommendation', 'N/A'),
            'currentPrice': p.get('price', 'N/A'),
            'shortName': p.get('name', p.get('symbol')),
            'sector': p.get('sector', 'Unknown')
        })

    return render_template("SP500phone.html", stocks=formatted_stocks, last_updated="Static Data")

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