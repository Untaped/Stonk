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

load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
# 1. Secret Key is REQUIRED for sessions. 
# On Render, set this as an environment variable 'SECRET_KEY'
app.secret_key = os.environ.get('SECRET_KEY', 'dev_default_key_change_this_in_prod')

# 2. Set your Access Code
# On Render, set 'SP500_ACCESS_CODE' in your Environment Variables
ACCESS_CODE = os.environ.get('SP500_ACCESS_CODE', '12345') # Default is 12345


# Configuration
# We only care about the predictions file now.
# This file is small and contains the results of your local training.
PREDICTIONS_CSV = "stonk_download/predictions.csv"
PREDICTIONS_5PCT_CSV = "stonk_download/predictions_5pct.csv"

# --- Helper Functions ---

def load_predictions(file_path):
    """Read predictions from CSV"""
    if not os.path.exists(file_path):
        return []
    try:
        df = pd.read_csv(file_path)
        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return []

# Add to app.py if you strictly need to support the old collector script
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
    
    # Example logic: Add points for positive growth
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

def get_latest_prediction(symbol, file_path=PREDICTIONS_CSV):
    """Find specific symbol in the CSV"""
    preds = load_predictions(file_path)
    symbol_clean = symbol.strip().upper()
    
    # Search for the symbol
    for row in preds:
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

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main dashboard - VIEWER ONLY"""
    symbol = ""
    prediction = None
    stock_data = None
    error = None

    if request.method == 'POST':
        symbol = request.form.get('symbol', '').upper().strip()
        
        # 1. Look up prediction in the CSV
        raw_data = get_latest_prediction(symbol)
        
        if raw_data:
            # Format data for the HTML template
            prediction = {
                'symbol': raw_data.get('symbol'),
                'recommendation': raw_data.get('recommendation', 'HOLD'),
                'probability': raw_data.get('probability', 0),
                # Map CSV columns to template variables
                '1pct': raw_data.get('prob_1pct', 0),   
                '3pct': raw_data.get('prob_3pct', 0),
                '5pct': raw_data.get('prob_5pct', 0),
                '10pct': raw_data.get('prob_10pct', 0),
                '1mo': raw_data.get('prob_1mo', 0),
            }
            
            # 2. Get basic info (Optional)
            stock_data = get_fundamentals(symbol)
        else:
            error = f"No prediction found for {symbol}. (This is a lightweight viewer: data must be uploaded first)."

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('code') == ACCESS_CODE:
            session.permanent = True
            session['sp500_unlocked'] = True
            
            # Check if there is a saved destination in the session
            next_page = session.pop('next_page', None) # .pop retrieves and deletes it
            
            if next_page == 'phone':
                return redirect(url_for('sp500_list_phone'))
            
            # Default to desktop if no specific destination was set
            return redirect(url_for('sp500_list'))
        else:
            error = "Invalid access code. Please try again."
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
    if not session.get('sp500_unlocked'):
        # Clear the next_page just in case, so they go to desktop default
        session.pop('next_page', None) 
        return redirect(url_for('login'))
    # ----------------------

    # ... (Rest of your existing sp500_list code stays the same) ...
    """Show top predictions from CSV"""
    preds = load_predictions(PREDICTIONS_CSV)
    # ... etc ...
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

    return render_template("SP500.html", stocks=formatted_stocks, last_updated="Static Data")

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
    if not session.get('sp500_unlocked'):
        # Save 'phone' as the intended destination before kicking them to login
        session['next_page'] = 'phone'
        return redirect(url_for('login'))
    # ----------------------

    # ... (Rest of your existing sp500_list_phone code stays the same) ...
    """Show top predictions from CSV for Phone"""
    preds = load_predictions(PREDICTIONS_CSV)
    # ... etc ...
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
    """Simple autocomplete based on what is in the CSV"""
    query = request.args.get('q', '').upper()
    preds = load_predictions(PREDICTIONS_CSV)
    symbols = [p.get('symbol') for p in preds if p.get('symbol') and query in p.get('symbol')]
    return jsonify(symbols[:10])

# Dummy routes to prevent 404s if links exist
@app.route('/portfolio')
def portfolio():
    return render_template("portfolio.html", results=[], error="Portfolio disabled in lightweight mode")

@app.route('/predict_5pct', methods=['GET', 'POST'])
def predict_5pct():
    # Redirects to main page logic or simplistic 5% viewer
    return index() 

if __name__ == '__main__':
    app.run(debug=True)