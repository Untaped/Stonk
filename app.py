from flask import Flask, request, render_template, jsonify
from datetime import datetime
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from flask import send_from_directory

# Load environment variables
load_dotenv()

app = Flask(__name__)

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

@app.route('/sp500')
def sp500_list():
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

    return render_template("sp500.html", stocks=formatted_stocks, last_updated="Static Data")

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
