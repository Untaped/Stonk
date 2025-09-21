from flask import Flask, request, render_template, jsonify
from datetime import datetime, timedelta, timezone
import pandas as pd
import os
import csv
import joblib
from collections import defaultdict
import json
import sys

app = Flask(__name__)

# Configuration
DATA_DIR = "data"
FUNDAMENTALS_CSV = os.path.join(DATA_DIR, "sp500_fundamentals.csv")
PRICE_HISTORY_CSV = os.path.join(DATA_DIR, "sp500_price_history.csv")
PREDICTIONS_CSV = "stonk_download/predictions.csv"
MODEL_PATH = "model.pkl"

# Cache for loaded data
_fundamentals_cache = None
_price_history_cache = None
_cache_timestamp = None

# Add the path for stock_ai imports
sys.path.append('.')

# Try to import the feature engineering functions
try:
    from stock_ai.feature_engineering import add_features, encode_sector_column
    FEATURE_ENGINEERING_AVAILABLE = True
    print("✅ Feature engineering functions imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import feature engineering: {e}")
    FEATURE_ENGINEERING_AVAILABLE = False

def load_csv_data():
    """Load and cache CSV data with timestamp checking"""
    global _fundamentals_cache, _price_history_cache, _cache_timestamp
    
    current_time = datetime.now().timestamp()
    
    # Refresh cache every 5 minutes or if not loaded
    if (_cache_timestamp is None or 
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return render_template("sp500.html", stocks=[], 
                             error=f"Error loading predictions: {e}",
                             last_updated=current_time)

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    """Portfolio backtesting using CSV data"""
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
                data = get_price_history_csv(symbol)
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Make predictions using CSV data with growth chart and OHLC data"""
    prediction = None
    symbol = None
    error = None
    growth_summary = None
    ohlc_data = None

    if request.method == 'POST':
        symbol = request.form['symbol'].upper().strip()
        
        if not symbol:
            error = "Please enter a stock symbol"
        else:
            try:
                print(f"Attempting prediction for symbol: {symbol}")
                
                # Generate AI prediction
                prediction = predict_next_day(symbol)
                
                if prediction is None:
                    error = f"Could not generate prediction for {symbol}. This could be due to insufficient data or the symbol not being available in our dataset."
                elif not isinstance(prediction, dict):
                    error = f"Invalid prediction format returned for {symbol}"
                elif 'probability' not in prediction:
                    error = f"Prediction missing probability data for {symbol}"
                elif prediction.get('probability') is None:
                    error = f"Prediction returned null probability for {symbol}"
                else:
                    # Validate probability is a valid number
                    try:
                        prob_value = float(prediction['probability'])
                        if prob_value < 0 or prob_value > 1:
                            error = f"Invalid probability value: {prob_value}"
                        else:
                            print(f"Successfully generated prediction for {symbol}: {prob_value}")
                            
                            # Get additional chart data if prediction successful
                            try:
                                # Get growth metrics for the chart
                                growth_metrics = get_projected_growth_metrics(symbol)
                                if growth_metrics:
                                    growth_summary = {
                                        "projected_price": max(
                                            (growth_metrics.get("revenue_growth", 0) or 0) * 100,
                                            (growth_metrics.get("earnings_growth", 0) or 0) * 100,
                                            5  # Minimum 5% projection
                                        ),
                                        "dividend_growth": (growth_metrics.get("dividend_growth", 0) or 0) * 100,
                                        "revenue_growth": (growth_metrics.get("revenue_growth", 0) or 0),
                                        "earnings_growth": (growth_metrics.get("earnings_growth", 0) or 0)
                                    }
                                
                                # Get OHLC data for candlestick chart
                                ohlc_data = get_ohlc_data_csv(symbol, days=60)  # 2 months of data
                                
                            except Exception as chart_error:
                                print(f"Warning: Could not load chart data for {symbol}: {chart_error}")
                                # Don't set error here - prediction still valid
                                
                    except (ValueError, TypeError):
                        error = f"Invalid probability format for {symbol}: {prediction.get('probability')}"
                        
            except Exception as e:
                error = f"Prediction error for {symbol}: {str(e)}"
                print(f"Exception in predict route: {e}")
                import traceback
                traceback.print_exc()

    return render_template("index.html", 
                         prediction=prediction, 
                         symbol=symbol, 
                         error=error,
                         growth_summary=growth_summary,
                         ohlc_data=ohlc_data)

# API Routes - All using CSV data only
# Add these API routes to support the growth chart functionality

@app.route('/api/fundamentals/<symbol>')
def api_fundamentals(symbol):
    """API endpoint to get fundamental data from CSV"""
    try:
        fundamentals = get_fundamentals(symbol.upper())
        return jsonify(fundamentals)
    except Exception as e:
        return jsonify({
            'error': f'Could not fetch fundamentals for {symbol}',
            'details': str(e)
        }), 404

@app.route('/api/price_history/<symbol>')
def api_price_history(symbol):
    """API endpoint for price history from CSV"""
    try:
        data = get_price_history_csv(symbol.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({
            'error': f'Could not fetch price history for {symbol}',
            'details': str(e),
            'dates': [],
            'prices': []
        }), 404

@app.route('/api/growth_metrics/<symbol>')
def api_growth_metrics(symbol):
    """API endpoint for growth metrics used in charts"""
    try:
        symbol = symbol.upper()
        fundamentals = get_fundamentals(symbol)
        
        # Calculate growth metrics for chart
        growth_data = {
            'symbol': symbol,
            'revenueGrowth': fundamentals.get('revenueGrowth', 0),
            'earningsGrowth': fundamentals.get('earningsGrowth', 0), 
            'dividendYield': fundamentals.get('dividendYield', 0),
            'projectedPrice': max(
                (fundamentals.get('revenueGrowth', 0) or 0) * 100,
                (fundamentals.get('earningsGrowth', 0) or 0) * 100,
                5  # Minimum 5% projection
            )
        }
        
        return jsonify(growth_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Could not fetch growth metrics for {symbol}',
            'details': str(e)
        }), 404

# Update the existing get_projected_growth_metrics function for better integration
def get_projected_growth_metrics(symbol):
    """Get growth metrics from fundamentals CSV with better error handling"""
    try:
        fundamentals = get_fundamentals(symbol)
        return {
            "dividend_growth": fundamentals.get('dividendYield', 0),
            "revenue_growth": fundamentals.get('revenueGrowth', 0),  
            "earnings_growth": fundamentals.get('earningsGrowth', 0),
            "projected_price": max(
                (fundamentals.get('revenueGrowth', 0) or 0) * 100,
                (fundamentals.get('earningsGrowth', 0) or 0) * 100,
                5  # Minimum projection
            )
        }
    except Exception as e:
        print(f"Error getting growth metrics for {symbol}: {e}")
        return {
            "dividend_growth": 0,
            "revenue_growth": 0,  
            "earnings_growth": 0,
            "projected_price": 0
        }

@app.route('/api/detailed_analysis/<symbol>')
def api_detailed_analysis(symbol):
    """API endpoint for comprehensive stock analysis using CSV data"""
    try:
        symbol = symbol.upper()
        
        # Get fundamentals
        fundamentals = get_fundamentals(symbol)
        
        # Get price history
        price_history = get_price_history_csv(symbol)
        
        # Get prediction if available
        predictions = load_predictions()
        prediction = next((p for p in predictions if p["symbol"] == symbol), None)
        
        # Evaluate stock criteria
        score, recommendation, reasons = evaluate_stock_criteria(fundamentals)
        
        # Create comprehensive analysis
        analysis = {
            'symbol': symbol,
            'fundamentals': fundamentals,
            'price_history': price_history,
            'prediction': prediction,
            'evaluation': {
                'score': score,
                'recommendation': recommendation,
                'reasons': reasons
            },
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({
            'error': f'Could not perform detailed analysis for {symbol}',
            'details': str(e)
        }), 404

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Autocomplete for stock symbols from CSV"""
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])

    symbols = get_sp500_symbols()
    suggestions = [s for s in symbols if query in s.upper()]
    return jsonify(suggestions[:10])

@app.route('/test-csv')
def test_csv():
    """Test route to check CSV data availability"""
    try:
        fundamentals_df, price_history_df = load_csv_data()
        
        result = {
            "status": "success",
            "fundamentals_loaded": not fundamentals_df.empty,
            "fundamentals_count": len(fundamentals_df) if not fundamentals_df.empty else 0,
            "price_history_loaded": not price_history_df.empty,
            "price_history_count": len(price_history_df) if not price_history_df.empty else 0,
            "sample_symbols": []
        }
        
        if not fundamentals_df.empty:
            result["sample_symbols"] = fundamentals_df['symbol'].head(10).tolist()
            result["fundamentals_columns"] = fundamentals_df.columns.tolist()
        
        if not price_history_df.empty:
            result["price_history_columns"] = price_history_df.columns.tolist()
            result["price_sample_symbols"] = price_history_df['symbol'].head(10).tolist()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "traceback": str(e)
        }), 500

@app.route('/data-status')
def data_status():
    """Show status of CSV data files"""
    fundamentals_df, price_history_df = load_csv_data()
    
    status = {
        "fundamentals": {
            "exists": os.path.exists(FUNDAMENTALS_CSV),
            "records": len(fundamentals_df) if not fundamentals_df.empty else 0,
            "last_modified": None
        },
        "price_history": {
            "exists": os.path.exists(PRICE_HISTORY_CSV),
            "records": len(price_history_df) if not price_history_df.empty else 0,
            "last_modified": None
        },
        "predictions": {
            "exists": os.path.exists(PREDICTIONS_CSV),
            "records": 0,
            "last_modified": None
        }
    }
    
    # Get file modification times
    files_to_check = [
        ("fundamentals", FUNDAMENTALS_CSV), 
        ("price_history", PRICE_HISTORY_CSV),
        ("predictions", PREDICTIONS_CSV)
    ]
    
    for file_key, file_path in files_to_check:
        if os.path.exists(file_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            status[file_key]["last_modified"] = mod_time.strftime("%Y-%m-%d %H:%M:%S")
            
            if file_key == "predictions":
                try:
                    pred_df = pd.read_csv(file_path)
                    status[file_key]["records"] = len(pred_df)
                except:
                    pass
    
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting FULLY SELF-CONTAINED Flask app...")
    print("📋 No external dependencies - all data from CSV files")
    print(f"📁 Looking for data in: {DATA_DIR}")
    print(f"📊 Fundamentals file: {FUNDAMENTALS_CSV}")
    print(f"📈 Price history file: {PRICE_HISTORY_CSV}")
    print(f"🤖 Predictions file: {PREDICTIONS_CSV}")
    
    # Load data on startup to check availability
    load_csv_data()
    
    app.run(debug=True)time - _cache_timestamp > 300 or
        _fundamentals_cache is None or 
        _price_history_cache is None):
        
        print("🔄 Loading CSV data...")
        
        # Load fundamentals
        if os.path.exists(FUNDAMENTALS_CSV):
            _fundamentals_cache = pd.read_csv(FUNDAMENTALS_CSV)
            print(f"✅ Loaded {len(_fundamentals_cache)} fundamental records")
        else:
            _fundamentals_cache = pd.DataFrame()
            print("⚠️ Fundamentals CSV not found")
        
        # Load price history
        if os.path.exists(PRICE_HISTORY_CSV):
            _price_history_cache = pd.read_csv(PRICE_HISTORY_CSV)
            
            # Handle datetime conversion more robustly
            try:
                # First, try standard conversion
                _price_history_cache['date'] = pd.to_datetime(_price_history_cache['date'], utc=True)
                # Convert to timezone-naive
                _price_history_cache['date'] = _price_history_cache['date'].dt.tz_localize(None)
            except Exception as e1:
                print(f"⚠️ First datetime conversion failed: {e1}")
                try:
                    # Fallback: try without UTC
                    _price_history_cache['date'] = pd.to_datetime(_price_history_cache['date'], errors='coerce')
                except Exception as e2:
                    print(f"⚠️ Second datetime conversion failed: {e2}")
                    # Last resort: keep as string and handle in individual functions
                    print("⚠️ Keeping dates as strings - will handle conversion per function")
            
            print(f"✅ Loaded {len(_price_history_cache)} price history records")
        else:
            _price_history_cache = pd.DataFrame()
            print("⚠️ Price history CSV not found")
        
        _cache_timestamp = current_time
    
    return _fundamentals_cache, _price_history_cache

def get_sp500_symbols():
    """Get S&P 500 symbols from fundamentals CSV"""
    fundamentals_df, _ = load_csv_data()
    if not fundamentals_df.empty:
        return sorted(fundamentals_df['symbol'].unique().tolist())
    
    # Fallback to hardcoded list if CSV not available
    return [
        'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'GOOG', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK.B',
        'JPM', 'WMT', 'ORCL', 'V', 'LLY', 'MA', 'NFLX', 'XOM', 'COST', 'JNJ',
        'HD', 'ABBV', 'PG', 'BAC', 'CVX', 'KO', 'AMD', 'TMUS', 'GE', 'UNH'
    ]

def safe_numeric(value):
    """Convert to float, return None if not possible"""
    try:
        if value is None or value == '' or value == 'N/A' or pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def get_fundamentals(symbol, date=None):
    """Get fundamental data from CSV instead of database"""
    fundamentals_df, _ = load_csv_data()
    
    if fundamentals_df.empty:
        raise ValueError(f"No fundamental data available")
    
    symbol_data = fundamentals_df[fundamentals_df['symbol'] == symbol]
    
    if symbol_data.empty:
        raise ValueError(f"No data found for {symbol}")
    
    latest_record = symbol_data.iloc[-1]
    
    return {
        'shortName': latest_record.get('name', symbol),
        'symbol': latest_record.get('symbol', symbol),
        'price': safe_numeric(latest_record.get('price')),
        'marketCap': safe_numeric(latest_record.get('market_cap')),
        'averageVolume': safe_numeric(latest_record.get('average_volume')),
        'revenueGrowth': safe_numeric(latest_record.get('revenue_growth')),
        'earningsGrowth': safe_numeric(latest_record.get('earnings_growth')),
        'netIncomeToCommon': safe_numeric(latest_record.get('free_cashflow')),
        'returnOnEquity': safe_numeric(latest_record.get('return_on_equity')),
        'debtToEquity': safe_numeric(latest_record.get('debt_to_equity')),
        'currentRatio': safe_numeric(latest_record.get('current_ratio')),
        'freeCashflow': safe_numeric(latest_record.get('free_cashflow')),
        'forwardPE': safe_numeric(latest_record.get('forward_pe')),
        'pegRatio': safe_numeric(latest_record.get('peg_ratio')),
        'beta': safe_numeric(latest_record.get('beta')),
        'dividendYield': safe_numeric(latest_record.get('dividend_yield')),
        'sector': latest_record.get('sector'),
        'industry': latest_record.get('industry'),
        'ipoDate': None
    }

def get_price_history_csv(symbol, timeframe='2Y'):
    """Get price history from CSV with timeframe filtering - Fixed for chart compatibility"""
    _, price_history_df = load_csv_data()
    
    if price_history_df.empty:
        print(f"No price history data loaded from CSV")
        return {
            "dates": [],
            "prices": [],
            "volumes": [],
            "hammer": [],
            "doji": [],
            "year_growths": {},
            "projected_growth": None
        }
    
    # Filter by symbol
    symbol_data = price_history_df[price_history_df['symbol'] == symbol].copy()
    print(f"Found {len(symbol_data)} records for {symbol}")
    
    if symbol_data.empty:
        print(f"No data found for symbol {symbol}")
        return {
            "dates": [],
            "prices": [],
            "volumes": [],
            "hammer": [],
            "doji": [],
            "year_growths": {},
            "projected_growth": None
        }
    
    # Apply timeframe filter
    today = datetime.now()
    timeframe_days = {
        '1M': 30,
        '6M': 180, 
        '1Y': 365,
        '2Y': 730
    }.get(timeframe, 730)

    start_date = today - timedelta(days=timeframe_days)
    
    # Ensure dates are datetime objects
    if symbol_data['date'].dtype == 'object':
        try:
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
        except Exception as e:
            print(f"Error converting dates: {e}")
            return {
                "dates": [],
                "prices": [],
                "volumes": [],
                "hammer": [],
                "doji": [],
                "year_growths": {},
                "projected_growth": None
            }
    
    # Filter by date
    symbol_data = symbol_data[symbol_data['date'] >= start_date]
    print(f"After date filtering: {len(symbol_data)} records")

    if symbol_data.empty:
        print(f"No data after date filtering for {symbol}")
        return {
            "dates": [],
            "prices": [],
            "volumes": [],
            "hammer": [],
            "doji": [],
            "year_growths": {},
            "projected_growth": None
        }
    
    # Sort by date
    symbol_data = symbol_data.sort_values('date')
    
    # Clean and validate all data
    clean_dates = []
    clean_prices = []
    clean_volumes = []
    hammer_flags = []
    doji_flags = []
    
    for _, row in symbol_data.iterrows():
        try:
            # Validate price
            close_price = float(row['close'])
            open_price = float(row.get('open', close_price))
            high_price = float(row.get('high', close_price))
            low_price = float(row.get('low', close_price))
            volume = float(row.get('volume', 0))
            
            # Skip invalid data
            if (pd.isna(close_price) or close_price <= 0 or 
                pd.isna(open_price) or pd.isna(high_price) or pd.isna(low_price)):
                continue
            
            # Format date properly
            if hasattr(row['date'], 'strftime'):
                date_str = row['date'].strftime('%Y-%m-%d')
            else:
                date_str = str(row['date'])[:10]
            
            # Calculate candlestick patterns
            body = abs(close_price - open_price)
            total_range = high_price - low_price
            
            is_hammer = False
            is_doji = False
            
            if total_range > 0:
                lower_shadow = min(open_price, close_price) - low_price
                upper_shadow = high_price - max(open_price, close_price)
                
                # Hammer detection
                is_hammer = (
                    body <= total_range * 0.3 and
                    lower_shadow >= 2 * body and
                    upper_shadow <= body * 0.3
                )
                
                # Doji detection
                is_doji = body <= total_range * 0.1
            
            # Add clean data
            clean_dates.append(date_str)
            clean_prices.append(close_price)
            clean_volumes.append(volume)
            hammer_flags.append(is_hammer)
            doji_flags.append(is_doji)
            
        except (ValueError, TypeError, AttributeError) as e:
            print(f"Skipping invalid row for {symbol}: {e}")
            continue
    
    print(f"Final clean data for {symbol}: {len(clean_prices)} price points")
    if clean_prices:
        print(f"Price range: ${min(clean_prices):.2f} to ${max(clean_prices):.2f}")
    
    # Calculate simple year-over-year growth
    year_growths = {}
    projected_growth = None
    
    if len(clean_prices) >= 2:
        if len(clean_prices) > 250:  # More than a year of data
            year_ago_price = clean_prices[-250] if len(clean_prices) >= 250 else clean_prices[0]
            current_price = clean_prices[-1]
            if year_ago_price > 0:
                growth = ((current_price - year_ago_price) / year_ago_price) * 100
                year_growths[2024] = round(growth, 2)
                projected_growth = round(growth, 2)
    
    return {
        "dates": clean_dates,
        "prices": clean_prices,
        "volumes": clean_volumes,
        "hammer": hammer_flags,
        "doji": doji_flags,
        "year_growths": year_growths,
        "projected_growth": projected_growth
    }

def get_ohlc_data_csv(symbol, days=30):
    """Get OHLC data from CSV for candlestick charts"""
    _, price_history_df = load_csv_data()
    
    if price_history_df.empty:
        return []
    
    # Filter by symbol first
    symbol_data = price_history_df[price_history_df['symbol'] == symbol].copy()
    
    if symbol_data.empty:
        return []
    
    # Filter by date range - handle different date formats
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Check if dates are strings and need conversion
    if symbol_data['date'].dtype == 'object':
        try:
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
        except:
            print("⚠️ Could not convert date strings to datetime")
            return []
    
    # Ensure timezone compatibility
    if hasattr(symbol_data['date'].iloc[0], 'tz') and symbol_data['date'].iloc[0].tz is not None:
        # DataFrame has timezone-aware dates
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
    else:
        # DataFrame has timezone-naive dates
        start_date = start_date.replace(tzinfo=None)
    
    # Apply date filter to the already filtered symbol_data
    symbol_data = symbol_data[
        symbol_data['date'] >= start_date
    ].copy()
    
    if symbol_data.empty:
        return []
    
    # Sort by date
    symbol_data = symbol_data.sort_values('date')
    
    ohlc = []
    for _, row in symbol_data.iterrows():
        # Handle potential timezone issues in date formatting
        date_str = row['date'].strftime("%Y-%m-%d") if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        
        ohlc.append({
            "x": date_str,
            "o": float(row['open']),
            "h": float(row['high']),
            "l": float(row['low']),
            "c": float(row['close']),
        })
    
    return ohlc

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

def predict_next_day(symbol, source="next_day"):
    """Predict using CSV data with proper feature engineering - FULLY OFFLINE"""
    try:
        print(f"Starting offline prediction for {symbol}")
        
        # Check if feature engineering is available
        if not FEATURE_ENGINEERING_AVAILABLE:
            print("❌ Feature engineering functions not available")
            return None
        
        # Get recent price data from CSV
        price_data = get_price_history_csv(symbol, timeframe='1Y')  # Use more data for better features
        
        if not price_data['dates'] or len(price_data['dates']) < 50:  # Need more data for technical indicators
            print(f"❌ Not enough data for {symbol}: {len(price_data.get('dates', []))} data points (need at least 50)")
            return None

        print(f"✅ Found {len(price_data['dates'])} price data points")

        # Create DataFrame with OHLCV data (required for proper feature engineering)
        ohlc_data = get_ohlc_data_csv(symbol, days=365)  # Get a full year of OHLC data
        
        if not ohlc_data or len(ohlc_data) < 50:
            print(f"❌ Not enough OHLC data for {symbol}: {len(ohlc_data)} data points")
            return None
            
        # Convert OHLC data to DataFrame format expected by feature engineering
        df = pd.DataFrame(ohlc_data)
        df.rename(columns={'x': 'date', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        
        # Add volume data if available
        if len(price_data['volumes']) == len(df):
            df['volume'] = price_data['volumes']
        else:
            # Use a default volume if not available
            df['volume'] = df['close'] * 1000000  # Rough approximation
            
        df['symbol'] = symbol
        
        print(f"DataFrame shape after OHLC setup: {df.shape}")

        # Check model files
        model_files = {
            "lightgbm_model": "stock_ai/lightgbm_model.pkl",
            "feature_names": "stock_ai/feature_names.pkl", 
            "sector_mapping": "stock_ai/sector_mapping.joblib"
        }
        
        missing_files = [name for name, path in model_files.items() if not os.path.exists(path)]
        if missing_files:
            print(f"❌ Missing model files: {missing_files}")
            return None

        # Load model and features
        model = joblib.load(model_files["lightgbm_model"])
        selected_features = joblib.load(model_files["feature_names"])
        sector_map = joblib.load(model_files["sector_mapping"])
        
        print(f"Model loaded, expecting {len(selected_features)} features")

        # Apply sector encoding first
        try:
            df = encode_sector_column(df, sector_map=sector_map)
            print("✅ Sector encoding completed")
        except Exception as e:
            print(f"⚠️ Sector encoding failed: {e}")
            df['sector_encoded'] = 0

        # Apply the same feature engineering as in training
        try:
            df = add_features(df, sector_map=sector_map)
            print("✅ Feature engineering completed")
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            return None
        
        print(f"Available columns after feature engineering: {len(df.columns)} columns")

        # Check which features we have
        available_features = [col for col in selected_features if col in df.columns]
        missing_features = [col for col in selected_features if col not in df.columns]
        
        print(f"✅ Available features: {len(available_features)}/{len(selected_features)}")
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
        
        if len(available_features) < len(selected_features) * 0.8:  # Need at least 80% of features
            print(f"❌ Too many missing features ({len(missing_features)} missing)")
            return None

        # Get latest row with required features
        latest = df.dropna(subset=available_features).tail(1)
        if latest.empty:
            print(f"❌ No complete rows found after dropping NaN values")
            return None

        print(f"✅ Using latest row for prediction")
        
        X_latest = latest[available_features]
        
        # Handle missing features by using 0 (the model should handle this)
        if len(available_features) < len(selected_features):
            # Create a full feature vector with missing features set to 0
            full_features = pd.DataFrame(0, index=X_latest.index, columns=selected_features)
            full_features[available_features] = X_latest[available_features]
            X_latest = full_features

        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_latest)[:, 1][0]
        else:
            pred = model.predict(X_latest)[0]
            proba = max(0, min(1, (pred + 1) / 2))

        # Save to predictions CSV
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        if os.path.exists(PREDICTIONS_CSV):
            try:
                preds_df = pd.read_csv(PREDICTIONS_CSV)
            except Exception as e:
                print(f"⚠️ Error reading predictions CSV: {e}, creating new file.")
                preds_df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(PREDICTIONS_CSV), exist_ok=True)
            preds_df = pd.DataFrame(columns=["timestamp", "symbol", "probability", "source"])

        # Remove old prediction for this symbol
        preds_df = preds_df[preds_df["symbol"] != symbol]

        # Add new prediction
        new_row = pd.DataFrame([{
            "timestamp": ts,
            "symbol": symbol,
            "probability": round(float(proba), 6),
            "source": source
        }])
        preds_df = pd.concat([preds_df, new_row], ignore_index=True)
        preds_df.to_csv(PREDICTIONS_CSV, index=False)

        result = {
            "symbol": symbol,
            "timestamp": ts,
            "probability": round(float(proba), 4),
            "source": source
        }
        
        print(f"✅ Offline prediction completed: {symbol} = {proba:.4f}")
        return result

    except Exception as e:
        print(f"❌ Prediction error for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_stock_criteria(f):
    """Stock evaluation scoring system"""
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

def get_projected_growth_metrics(symbol):
    """Get growth metrics from fundamentals CSV"""
    try:
        fundamentals = get_fundamentals(symbol)
        return {
            "dividend_growth": fundamentals.get('dividendYield', 0),
            "revenue_growth": fundamentals.get('revenueGrowth', 0),
            "earnings_growth": fundamentals.get('earningsGrowth', 0)
        }
    except:
        return {
            "dividend_growth": 0,
            "revenue_growth": 0,  
            "earnings_growth": 0
        }

# Template filters
@app.template_filter('safe_float')
def safe_float_filter(value, default=0):
    """Safely convert value to float, return default if conversion fails"""
    try:
        if value is None or value == '' or value == 'N/A':
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)

@app.template_filter('safe_percent')
def safe_percent_filter(value, default=0):
    """Safely convert value to percentage"""
    try:
        if value is None or value == '' or value == 'N/A':
            return float(default)
        return float(value) * 100
    except (ValueError, TypeError):
        return float(default)

@app.template_filter('safe_billions')
def safe_billions_filter(value, default=0):
    """Safely convert value to billions"""
    try:
        if value is None or value == '' or value == 'N/A':
            return float(default)
        return float(value) / 1e9
    except (ValueError, TypeError):
        return float(default)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    # Initialize defaults
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
        symbol = request.form.get('symbol', '').upper().strip()
        timeframe = request.form.get('timeframe', '2Y')

        try:
            # Get OHLC data from CSV
            timeframe_map = {"1M": 30, "6M": 180, "1Y": 365, "2Y": 730}
            days = timeframe_map.get(timeframe, 30)
            ohlc_data = get_ohlc_data_csv(symbol, days=days)

            # Get fundamentals from CSV
            fundamentals = get_fundamentals(symbol)
            stock_data = fundamentals
            score, recommendation_text, top_reasons = evaluate_stock_criteria(fundamentals)
            recommendation = recommendation_text

            # Get price history from CSV
            price_history = get_price_history_csv(symbol, timeframe)
            closes = price_history.get("prices", [])
            dates = price_history.get("dates", [])
            hammer = price_history.get("hammer", [])
            doji = price_history.get("doji", [])

            # Growth metrics
            growth_metrics = get_projected_growth_metrics(symbol)
            growth_summary = {
                "projected_price": price_history.get("projected_growth", 0),
                "dividend_growth": growth_metrics.get("dividend_growth", 0),
                "revenue_growth": growth_metrics.get("revenue_growth", 0),
                "earnings_growth": growth_metrics.get("earnings_growth", 0)
            }

            # Get AI prediction
            preds = load_predictions()
            prediction_result = next((p for p in preds if p["symbol"] == symbol), None)

        except Exception as e:
            error = f"Error retrieving data: {e}"
            closes, dates, hammer, doji = [], [], [], []

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

@app.route('/sp500')
def sp500_list():
    """Show S&P 500 stocks using predictions from CSV data - FULLY OFFLINE"""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load predictions from CSV
        df_latest = pd.read_csv(PREDICTIONS_CSV)
        
        if df_latest.empty:
            return render_template("sp500.html", stocks=[], 
                                 error="No predictions found in CSV. Run update_sp500_data.py first.",
                                 last_updated=current_time)
        
        # Get the most recent prediction for each symbol
        df_latest = df_latest.sort_values('timestamp').groupby('symbol').tail(1)
        print(f"Found {len(df_latest)} unique symbols with predictions")
        
        stocks = []
        
        for _, row in df_latest.iterrows():
            try:
                symbol = row['symbol'].upper()
                probability = float(row['probability'])
                
                # Create recommendation based on probability
                if probability > 0.7:
                    recommendation = 'BUY'
                elif probability > 0.5:
                    recommendation = 'HOLD'
                else:
                    recommendation = 'SELL'
                
                # Try to get fundamentals data from CSV if available
                try:
                    fundamentals = get_fundamentals(symbol)
                    stock_name = fundamentals.get('shortName', symbol)
                    sector = fundamentals.get('sector', 'Unknown')
                    price = fundamentals.get('price', 0)
                except:
                    stock_name = symbol
                    sector = 'Unknown'
                    price = 0
                
                stocks.append({
                    'symbol': symbol,
                    'shortName': stock_name,
                    'sector': sector,
                    'currentPrice': price,
                    'probability': probability,
                    'recommendation': recommendation,
                    'score_percent': f"{probability * 100:.1f}%",
                    'score': f"{probability:.3f}",
                    'timestamp': row['timestamp']
                })
                
            except Exception as e:
                print(f"Error processing {row.get('symbol', 'unknown')}: {e}")
                continue
        
        # Sort by probability (highest first)
        stocks.sort(key=lambda x: x['probability'], reverse=True)
        
        print(f"Successfully processed {len(stocks)} stocks from CSV data")
        
        return render_template("sp500.html", stocks=stocks, last_updated=current_time)
        
    except Exception as e:
        print(f"Error in sp500_list: {e}")
        import traceback
        traceback.print_exc()
        
        current_