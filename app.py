from flask import Flask, request, render_template, jsonify, session, redirect, url_for, flash
import logging

# The IP log is in the file server_access.log :D:D:D:D:D:D

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
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask import abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


#Logging IPsssssss
def get_location(ip):
    response = requests.get(f"https://ipinfo.io/{ip}/json")
    return response.json()

load_dotenv()

app = Flask(__name__)
# ... your existing ProxyFix and Limiter setup ...

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=2, x_proto=1, x_host=1, x_prefix=1
)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://" # Uses local memory, perfect for lightweight apps
)

# --- NEW LOGGING SETUP ---
# 1. Configure where the logs are saved
logging.basicConfig(
    filename='server_access.log', 
    level=logging.INFO,  # <--- Ensure there is a comma here
    format='%(asctime)s - IP: %(message)s' # <--- And here if you add more lines
)

# A secret key is required for secure cookies (sessions)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'replace-this-with-a-random-string-in-production')

# Use SQLite locally, but allow Render to inject a PostgreSQL database URL
# Important: Render uses ephemeral disks, so SQLite will wipe on every deploy. 
# You will need a free PostgreSQL database for production.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///litlguy_users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Tells Flask where to redirect unauthorized users
login_manager.login_message = "Please log in to access this feature."

# --- NEW: Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    
    # THIS IS THE CRITICAL LINE THAT IS MISSING OR INDENTED WRONG:
    saved_stocks = db.relationship('SavedStock', backref='owner', lazy=True)

class SavedStock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    saved_price = db.Column(db.Float, nullable=True)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

@app.before_request
def block_bad_requests():
    # 1. Block malicious paths instantly
    bad_patterns = ['.php', '/wp-', '/.env', '/cgi-bin', '.git']
    if any(pattern in request.path for pattern in bad_patterns):
        abort(403) 

    # 2. Get the REAL IP (Check Cloudflare first, then fallback to ProxyFix)
    client_ip = request.headers.get('CF-Connecting-IP', request.remote_addr)
    path = request.path
    
    # 3. Log legitimate traffic
    if not path.startswith('/static/'):
        logging.info(f"{client_ip} accessed {path}")
# 2. Automatically log every visitor's IP and the page they visited
@app.before_request
def log_request_info():
    # Because you are using ProxyFix, request.remote_addr is the real IP
    client_ip = request.remote_addr
    path = request.path
    
    # Ignore logging for static files to keep your log clean
    if not path.startswith('/static/'):
        logging.info(f"{client_ip} accessed {path}")

# --- NEW: Authentication Routes ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists. Please choose another or log in.")
            return redirect(url_for('register'))
            
        # Hash password and save user
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password_hash=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! Please log in.")
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index')) # Redirect to home page
        else:
            flash("Invalid username or password.")
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
@app.route('/premium-feature')
@login_required
def premium_feature():
    # If a user is not logged in, Flask automatically redirects them to the /login page
    return render_template('premium.html', username=current_user.username)

# Configuration
PREDICTIONS_CSV = "stonk_download/predictions.csv"
FUNDAMENTALS_CSV = "stonk_download/fundamentals.csv"
PREDICTIONS_5PCT_CSV = "stonk_download/predictions_5pct.csv"
NASDAQ_PREDICTIONS_CSV = "stonk_download/nasdaq_predictions.csv"

@app.route('/toggle_stock', methods=['POST'])
@login_required
def toggle_stock():
    data = request.get_json()
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400

    existing_stock = SavedStock.query.filter_by(user_id=current_user.id, symbol=symbol).first()
    
    if existing_stock:
        db.session.delete(existing_stock)
        db.session.commit()
        return jsonify({'status': 'removed', 'symbol': symbol})
    else:
        # --- NEW: Robust Price Finder ---
        current_price = 0.0
        
        # 1. Try getting the price from the prediction files first
        raw_data = get_latest_prediction(symbol)
        if raw_data:
            # Check for both possible column names
            if raw_data.get('current_price'):
                current_price = float(raw_data.get('current_price', 0.0))
            elif raw_data.get('price'):
                current_price = float(raw_data.get('price', 0.0))
        
        # 2. If it's still 0.0, fallback to fundamentals
        if current_price == 0.0:
            fund_data = get_fundamentals(symbol)
            current_price = float(fund_data.get('price', 0.0))
        
        # Save the stock WITH the located price
        new_stock = SavedStock(symbol=symbol, user_id=current_user.id, saved_price=current_price)
        db.session.add(new_stock)
        db.session.commit()
        
        return jsonify({'status': 'added', 'symbol': symbol, 'saved_price': current_price})

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
    '3pct': 'stock_ai/market_lightgbm_model_3pct.pkl',
    '5pct': 'stock_ai/market_lightgbm_model_5pct.pkl',
    '10pct': 'stock_ai/market_lightgbm_model_10pct.pkl',
    '1mo': 'stock_ai/market_lightgbm_model_1mo.pkl',
    '30_3pct': 'stock_ai/market_lightgbm_model_30_3pct.pkl',
    '30_5pct': 'stock_ai/market_lightgbm_model_30_5pct.pkl',
    '30_10pct': 'stock_ai/market_lightgbm_model_30_10pct.pkl',
    '30_1mo': 'stock_ai/market_lightgbm_model_30_1mo.pkl'
}
MARKET_FEATURE_PATHS = {
    '3pct': 'stock_ai/market_feature_names_3pct.pkl',
    '5pct': 'stock_ai/market_feature_names_5pct.pkl',
    '10pct': 'stock_ai/market_eature_names_10pct.pkl',
    '1mo': 'stock_ai/market_eature_names_1mo.pkl',
    '30_3pct': 'stock_ai/market_feature_names_30_3pct.pkl',
    '30_5pct': 'stock_ai/market_feature_names_30_5pct.pkl',
    '30_10pct': 'stock_ai/market_eature_names_30_10pct.pkl',
    '30_1mo': 'stock_ai/market_eature_names_30_1mo.pkl'
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
def index():
    saved_stocks_data = []
    
    if current_user.is_authenticated:
        for stock in current_user.saved_stocks:
            # Find current price robustly
            current_price = 0.0
            raw_data = get_latest_prediction(stock.symbol)
            
            # Check for both possible column names in your CSVs
            if raw_data:
                current_price = float(raw_data.get('current_price', raw_data.get('price', 0.0)))
            
            # If the CSV didn't have it, fallback to the fundamentals file
            if current_price == 0.0:
                fund_data = get_fundamentals(stock.symbol)
                current_price = float(fund_data.get('price', 0.0))

            saved_stocks_data.append({
                'symbol': stock.symbol, 
                'price': current_price,
                'saved_price': stock.saved_price or 0.00
            })
            
    return render_template('index.html', saved_stocks=saved_stocks_data)

@app.route('/', methods=['POST'])
def index_post():
    symbol = request.form.get('symbol', '').upper().strip()
    prediction = None
    stock_data = None
    error = None

    saved_stocks = []
    if current_user.is_authenticated:
        for stock in current_user.saved_stocks:
            # Find current price robustly
            current_price = 0.0
            raw_data = get_latest_prediction(stock.symbol)
            if raw_data:
                current_price = float(raw_data.get('current_price', raw_data.get('price', 0.0)))
            
            if current_price == 0.0:
                fund_data = get_fundamentals(stock.symbol)
                current_price = float(fund_data.get('price', 0.0))

            saved_stocks.append({
                'symbol': stock.symbol, 
                'price': current_price,
                'saved_price': stock.saved_price or 0.00
            })
    # ---------------------------------------------------------

    if symbol:
        raw_data = get_latest_prediction(symbol)
        
        if raw_data:
            # Extract 5d
            p3 = float(raw_data.get('prob_3pct', 0))
            p3 = float(raw_data.get('prob_3pct', 0))
            p5 = float(raw_data.get('prob_5pct', 0))
            p10 = float(raw_data.get('prob_10pct', 0))
            p15 = float(raw_data.get('prob_15pct', 0))

            # 2. Extract 30-Day probabilities
            p30_3 = float(raw_data.get('prob_30_3pct', 0))
            p30_5 = float(raw_data.get('prob_30_5pct', 0))
            p30_10 = float(raw_data.get('prob_30_10pct', 0))
            p30_15 = float(raw_data.get('prob_30_1mo', 0))
            
            rec = raw_data.get('recommendation', 'HOLD')
            is_linear_5d = (p3 > p5) and (p5 > p10) and (p10 > p15)
            score_5d = float(raw_data.get('score_5d', 0))
            if score_5d >= 0.35:
                rec_5d = "BUY" if is_linear_5d else "CONSIDER"
            else:
                rec_5d = "DISREGARD"

            is_linear_30d = (p30_3 > p30_5) and (p30_5 > p30_10) and (p30_10 > p30_15)
            score_30d = float(raw_data.get('score_30d', 0))
            if score_30d >= 0.35:
                rec_30d = "BUY" if is_linear_30d else "CONSIDER"
            else:
                rec_30d = "DISREGARD"

            prediction = {
                'symbol': raw_data.get('symbol'),
                'recommendation': rec,
                'probability': raw_data.get('probability', 0), 
                '3pct': p3, '5pct': p5, '10pct': p10, '1mo': p15,
                '30_3pct': p30_3, '30_5pct': p30_5, '30_10pct': p30_10, '30_1mo': p30_15
            }
            stock_data = get_fundamentals(symbol)
        else:
            error = f"No prediction found for {symbol} in S&P 500 databases."

    # Make sure saved_stocks is passed here!
    return render_template('index.html', symbol=symbol, prediction=prediction, stock_data=stock_data, error=error, saved_stocks=saved_stocks)

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
    preds = load_predictions(PREDICTIONS_CSV)
    
    formatted_stocks_5d = []
    formatted_stocks_30d = []
    
    for p in preds:
        # 5-day probabilities
        p3 = float(p.get('prob_3pct', 0))
        p5 = float(p.get('prob_5pct', 0))
        p10 = float(p.get('prob_10pct', 0))
        p15 = float(p.get('prob_1mo', 0))
        
        # 30-day probabilities
        p30_3 = float(p.get('prob_30_3pct', 0))
        p30_5 = float(p.get('prob_30_5pct', 0))
        p30_10 = float(p.get('prob_30_10pct', 0))
        p30_15 = float(p.get('prob_30_1mo', 0))

        # Because `combined_score` mixed everything in your predict file, 
        # let's approximate the isolated 5d vs 30d scores using your weights.
        prob_5d = (p3*0.05 + p5*0.15 + p10*0.4 + p15*0.4)
        prob_30d = (p30_3*0.05 + p30_5*0.15 + p30_10*0.4 + p30_15*0.4)

        # 5-Day Rec
        is_linear_5d = (p3 > p5) and (p5 > p10) and (p10 > p15)
        if prob_5d >= 0.35 and is_linear_5d: rec_5d = "BUY"
        elif prob_5d >= 0.30: rec_5d = "CONSIDER"
        else: rec_5d = "DISREGARD"

        # 30-Day Rec
        is_linear_30d = (p30_3 > p30_5) and (p30_5 > p30_10) and (p30_10 > p30_15)
        if prob_30d >= 0.35 and is_linear_30d: rec_30d = "BUY"
        elif prob_30d >= 0.30: rec_30d = "CONSIDER"
        else: rec_30d = "DISREGARD"

        base_info = {
            'symbol': p.get('symbol'),
            'currentPrice': float(p.get('price', 0)), 
            'shortName': p.get('name', p.get('symbol')), 
            'sector': p.get('sector', 'Unknown')
        }

        formatted_stocks_5d.append({**base_info, 'probability': prob_5d, 'score_percent': f"{prob_5d*100:.1f}%", 'recommendation': rec_5d})
        formatted_stocks_30d.append({**base_info, 'probability': prob_30d, 'score_percent': f"{prob_30d*100:.1f}%", 'recommendation': rec_30d})

    # Sort both lists by highest probability
    formatted_stocks_5d.sort(key=lambda x: x['probability'], reverse=True)
    formatted_stocks_30d.sort(key=lambda x: x['probability'], reverse=True)
    # Calculate actual statistics for the 5-Day timeframe
    stats_5d = {
        'buy': sum(1 for stock in formatted_stocks_5d if stock['recommendation'] == 'BUY'),
        'consider': sum(1 for stock in formatted_stocks_5d if stock['recommendation'] == 'CONSIDER'),
        'disregard': sum(1 for stock in formatted_stocks_5d if stock['recommendation'] == 'DISREGARD')
    }
    
    # Calculate actual statistics for the 30-Day timeframe
    stats_30d = {
        'buy': sum(1 for stock in formatted_stocks_30d if stock['recommendation'] == 'BUY'),
        'consider': sum(1 for stock in formatted_stocks_30d if stock['recommendation'] == 'CONSIDER'),
        'disregard': sum(1 for stock in formatted_stocks_30d if stock['recommendation'] == 'DISREGARD')
    }

    user_watchlist = []
    if current_user.is_authenticated:
        user_watchlist = [w.symbol for w in SavedStock.query.filter_by(user_id=current_user.id).all()]

    return render_template(
        "SP500.html",
        stocks_5d=formatted_stocks_5d[:100], 
        stocks_30d=formatted_stocks_30d[:100], 
        last_updated="Static Data",
        stats_5d=stats_5d,
        stats_30d=stats_30d,
        user_watchlist=user_watchlist
    )

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

                # Get individual probabilities for logic
                p3 = float(row.get('prob_3pct', 0))
                p5 = float(row.get('prob_5pct', 0))
                p10 = float(row.get('prob_10pct', 0))
                p15 = float(row.get('prob_1mo', 0))

                # --- NEW LOGIC ---
                # Check linearity: 3 > 5 > 10 > 15
                is_linear = (p3 > p5) and (p5 > p10) and (p10 > p15)

                if probability >= 0.35 and is_linear:
                    recommendation = 'BUY'
                elif probability >= 0.35:
                    recommendation = 'CONSIDER' # Non-linear confidence
                else:
                    recommendation = 'DISREGARD'
                # -----------------
                
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
        
        # ADD THIS BLOCK: Get user's watchlist if logged in
        user_watchlist = []
        if current_user.is_authenticated:
            user_watchlist = [w.symbol for w in SavedStock.query.filter_by(user_id=current_user.id).all()]
        
        return render_template("nasdaq.html", stocks=stocks, last_updated=current_time, user_watchlist=user_watchlist)

    except Exception as e:
        print(f"Error in nasdaq_list: {e}")
        return render_template("nasdaq.html", stocks=[], error=str(e), last_updated=current_time)

@app.route('/about')
def about_page():
    return render_template('about.html')

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