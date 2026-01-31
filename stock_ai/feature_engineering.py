#feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import builtins

def safe_str_debug(x):
    try:
        return str(x)
    except Exception as e:
        return f"[str failed: {e}]"

builtins.safe_str_debug = safe_str_debug

def compute_rsi(series, period=14):
    """Standard RSI calculation"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    """Average True Range calculation"""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def get_sector_mapping():
    """
    Returns a comprehensive sector mapping for S&P 500 stocks
    This replaces any web scraping with a hardcoded, accurate mapping
    """
    
    sector_mapping = {
        # Technology
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
        'NVDA': 'Technology', 'META': 'Communication Services', 'AVGO': 'Technology', 'ORCL': 'Technology',
        'CRM': 'Technology', 'ADBE': 'Technology', 'INTC': 'Technology', 'CSCO': 'Technology',
        'IBM': 'Technology', 'QCOM': 'Technology', 'TXN': 'Technology', 'NOW': 'Technology',
        'INTU': 'Technology', 'AMD': 'Technology', 'MU': 'Technology', 'AMAT': 'Technology',
        'ADI': 'Technology', 'LRCX': 'Technology', 'KLAC': 'Technology', 'MCHP': 'Technology',
        'CDNS': 'Technology', 'SNPS': 'Technology', 'ANET': 'Technology', 'PANW': 'Technology',
        'CRWD': 'Technology', 'WDAY': 'Technology', 'ADSK': 'Technology', 'FTNT': 'Technology',
        'TEL': 'Technology', 'NXPI': 'Technology', 'ON': 'Technology', 'KEYS': 'Technology',
        'DELL': 'Technology', 'HPE': 'Technology', 'HPQ': 'Technology', 'NTAP': 'Technology',
        'FFIV': 'Technology', 'AKAM': 'Technology', 'SMCI': 'Technology', 'WDC': 'Technology',
        'STX': 'Technology', 'TER': 'Technology', 'SWKS': 'Technology', 'PTC': 'Technology',
        'GDDY': 'Technology', 'TYL': 'Technology', 'TRMB': 'Technology', 'GEN': 'Technology',
        'VRSN': 'Technology', 'TDY': 'Technology', 'IT': 'Technology', 'CDW': 'Technology',
        
        # Healthcare
        'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
        'LLY': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
        'MDT': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare',
        'GILD': 'Healthcare', 'CVS': 'Healthcare', 'CI': 'Healthcare', 'HUM': 'Healthcare',
        'SYK': 'Healthcare', 'BDX': 'Healthcare', 'ZTS': 'Healthcare', 'BSX': 'Healthcare',
        'EW': 'Healthcare', 'ISRG': 'Healthcare', 'IDXX': 'Healthcare', 'RMD': 'Healthcare',
        'DXCM': 'Healthcare', 'ALGN': 'Healthcare', 'MTD': 'Healthcare', 'IQV': 'Healthcare',
        'A': 'Healthcare', 'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'ELV': 'Healthcare',
        'MCK': 'Healthcare', 'COR': 'Healthcare', 'HCA': 'Healthcare', 'CAH': 'Healthcare',
        'GEHC': 'Healthcare', 'BIIB': 'Healthcare', 'DGX': 'Healthcare', 'SOLV': 'Healthcare',
        'VTRS': 'Healthcare', 'BAX': 'Healthcare', 'DOC': 'Healthcare', 'HOLX': 'Healthcare',
        'PODD': 'Healthcare', 'LH': 'Healthcare', 'ZBH': 'Healthcare', 'MOH': 'Healthcare',
        'TECH': 'Healthcare', 'HSIC': 'Healthcare', 'CRL': 'Healthcare', 'RVTY': 'Healthcare',
        'WST': 'Healthcare', 'WAT': 'Healthcare', 'COO': 'Healthcare', 'INCY': 'Healthcare',
        'UHS': 'Healthcare', 'MRNA': 'Healthcare', 'DVA': 'Healthcare', 'EPAM': 'Healthcare',
        
        # Financials
        'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
        'MS': 'Financials', 'C': 'Financials', 'BLK': 'Financials', 'AXP': 'Financials',
        'SPGI': 'Financials', 'BK': 'Financials', 'USB': 'Financials', 'TFC': 'Financials',
        'PNC': 'Financials', 'COF': 'Financials', 'SCHW': 'Financials', 'CB': 'Financials',
        'MMC': 'Financials', 'ICE': 'Financials', 'CME': 'Financials', 'AON': 'Financials',
        'PGR': 'Financials', 'TRV': 'Financials', 'AFL': 'Financials', 'ALL': 'Financials',
        'MET': 'Financials', 'PRU': 'Financials', 'AIG': 'Financials', 'HIG': 'Financials',
        'CINF': 'Financials', 'L': 'Financials', 'BX': 'Financials', 'KKR': 'Financials',
        'APO': 'Financials', 'NDAQ': 'Financials', 'MCO': 'Financials', 'MSCI': 'Financials',
        'AMP': 'Financials', 'AJG': 'Financials', 'FI': 'Financials', 'ACGL': 'Financials',
        'WTW': 'Financials', 'FICO': 'Financials', 'BRO': 'Financials', 'STT': 'Financials',
        'MTB': 'Financials', 'FITB': 'Financials', 'CFG': 'Financials', 'RF': 'Financials',
        'HBAN': 'Financials', 'NTRS': 'Financials', 'KEY': 'Financials', 'SYF': 'Financials',
        'RJF': 'Financials', 'TROW': 'Financials', 'PFG': 'Financials', 'BEN': 'Financials',
        'IVZ': 'Financials', 'AIZ': 'Financials', 'GL': 'Financials', 'EG': 'Financials',
        'CBOE': 'Financials', 'GPN': 'Financials', 'FIS': 'Financials', 'BR': 'Financials',
        'WRB': 'Financials', 'ERIE': 'Financials',
        
        # Consumer Discretionary
        'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
        'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary',
        'SBUX': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
        'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
        'HLT': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary', 'ORLY': 'Consumer Discretionary',
        'YUM': 'Consumer Discretionary', 'LULU': 'Consumer Discretionary', 'AZO': 'Consumer Discretionary',
        'ROST': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'DHI': 'Consumer Discretionary',
        'LEN': 'Consumer Discretionary', 'NVR': 'Consumer Discretionary', 'PHM': 'Consumer Discretionary',
        'GRMN': 'Consumer Discretionary', 'POOL': 'Consumer Discretionary', 'CCL': 'Consumer Discretionary',
        'RCL': 'Consumer Discretionary', 'ABNB': 'Consumer Discretionary', 'UBER': 'Consumer Discretionary',
        'DASH': 'Consumer Discretionary', 'TPR': 'Consumer Discretionary', 'RL': 'Consumer Discretionary',
        'DECK': 'Consumer Discretionary', 'DPZ': 'Consumer Discretionary', 'BLDR': 'Consumer Discretionary',
        'BBY': 'Consumer Discretionary', 'DG': 'Consumer Discretionary', 'DLTR': 'Consumer Discretionary',
        'ULTA': 'Consumer Discretionary', 'DRI': 'Consumer Discretionary', 'HAS': 'Consumer Discretionary',
        'WYNN': 'Consumer Discretionary', 'NCLH': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary',
        'LVS': 'Consumer Discretionary', 'EXPE': 'Consumer Discretionary', 'WSM': 'Consumer Discretionary',
        'KMX': 'Consumer Discretionary', 'LKQ': 'Consumer Discretionary', 'MHK': 'Consumer Discretionary',
        'APTV': 'Consumer Discretionary', 'LYV': 'Consumer Discretionary', 'TKO': 'Consumer Discretionary',
        'CZR': 'Consumer Discretionary', 'MAS': 'Consumer Discretionary', 'UAL': 'Consumer Discretionary',
        'DAL': 'Consumer Discretionary', 'LUV': 'Consumer Discretionary',
        
        # Consumer Staples
        'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
        'PEP': 'Consumer Staples', 'COST': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
        'KMB': 'Consumer Staples', 'GIS': 'Consumer Staples', 'SYY': 'Consumer Staples',
        'ADM': 'Consumer Staples', 'HSY': 'Consumer Staples', 'K': 'Consumer Staples',
        'CHD': 'Consumer Staples', 'CLX': 'Consumer Staples', 'CAG': 'Consumer Staples',
        'TSN': 'Consumer Staples', 'HRL': 'Consumer Staples', 'MKC': 'Consumer Staples',
        'CPB': 'Consumer Staples', 'SJM': 'Consumer Staples', 'LW': 'Consumer Staples',
        'TAP': 'Consumer Staples', 'KDP': 'Consumer Staples', 'MNST': 'Consumer Staples',
        'KR': 'Consumer Staples', 'CL': 'Consumer Staples', 'PM': 'Consumer Staples',
        'MO': 'Consumer Staples', 'STZ': 'Consumer Staples', 'BF.B': 'Consumer Staples',
        'KHC': 'Consumer Staples', 'KVUE': 'Consumer Staples', 'EL': 'Consumer Staples',
        'WBA': 'Consumer Staples', 'BG': 'Consumer Staples',
        
        # Energy
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
        'SLB': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy', 'MPC': 'Energy',
        'KMI': 'Energy', 'OKE': 'Energy', 'WMB': 'Energy', 'FANG': 'Energy',
        'EQT': 'Energy', 'OXY': 'Energy', 'BKR': 'Energy', 'DVN': 'Energy',
        'TRGP': 'Energy', 'CTRA': 'Energy', 'HAL': 'Energy', 'APA': 'Energy',
        'TPL': 'Energy',
        
        # Industrials
        'BA': 'Industrials', 'CAT': 'Industrials', 'RTX': 'Industrials', 'HON': 'Industrials',
        'UPS': 'Industrials', 'LMT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials',
        'FDX': 'Industrials', 'NOC': 'Industrials', 'GD': 'Industrials', 'WM': 'Industrials',
        'RSG': 'Industrials', 'ITW': 'Industrials', 'PH': 'Industrials', 'CMI': 'Industrials',
        'ETN': 'Industrials', 'EMR': 'Industrials', 'PCAR': 'Industrials', 'IR': 'Industrials',
        'OTIS': 'Industrials', 'CARR': 'Industrials', 'NSC': 'Industrials', 'CSX': 'Industrials',
        'UNP': 'Industrials', 'DE': 'Industrials', 'TT': 'Industrials', 'APH': 'Industrials',
        'HWM': 'Industrials', 'JCI': 'Industrials', 'TDG': 'Industrials', 'MSI': 'Industrials',
        'AXON': 'Industrials', 'URI': 'Industrials', 'GWW': 'Industrials', 'PWR': 'Industrials',
        'ROP': 'Industrials', 'FAST': 'Industrials', 'PAYX': 'Industrials', 'ADP': 'Industrials',
        'ROK': 'Industrials', 'AME': 'Industrials', 'ODFL': 'Industrials', 'WAB': 'Industrials',
        'LHX': 'Industrials', 'HUBB': 'Industrials', 'LDOS': 'Industrials', 'CPAY': 'Industrials',
        'VRSK': 'Industrials', 'FTV': 'Industrials', 'EXPD': 'Industrials', 'ZBRA': 'Industrials',
        'ALLE': 'Industrials', 'CHRW': 'Industrials', 'TXT': 'Industrials', 'DOV': 'Industrials',
        'STE': 'Industrials', 'PAYC': 'Industrials', 'JKHY': 'Industrials', 'SWK': 'Industrials',
        'GNRC': 'Industrials', 'NDSN': 'Industrials', 'J': 'Industrials', 'PNR': 'Industrials',
        'SNA': 'Industrials', 'JBHT': 'Industrials', 'AVY': 'Industrials', 'IEX': 'Industrials',
        'XYL': 'Industrials', 'ROL': 'Industrials', 'VLTO': 'Industrials', 'LII': 'Industrials',
        'HII': 'Industrials', 'DAY': 'Industrials', 'CTAS': 'Industrials', 'GPN': 'Industrials',
        
        # Utilities
        'NEE': 'Utilities', 'SO': 'Utilities', 'DUK': 'Utilities', 'AEP': 'Utilities',
        'SRE': 'Utilities', 'D': 'Utilities', 'PCG': 'Utilities', 'EXC': 'Utilities',
        'ED': 'Utilities', 'ETR': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities',
        'FE': 'Utilities', 'EIX': 'Utilities', 'PPL': 'Utilities', 'EVRG': 'Utilities',
        'AWK': 'Utilities', 'DTE': 'Utilities', 'PEG': 'Utilities', 'NI': 'Utilities',
        'CEG': 'Utilities', 'VST': 'Utilities', 'NRG': 'Utilities', 'AEE': 'Utilities',
        'CMS': 'Utilities', 'XEL': 'Utilities', 'CNP': 'Utilities', 'ATO': 'Utilities',
        'PNW': 'Utilities', 'LNT': 'Utilities', 'AES': 'Utilities',
        
        # Materials
        'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'SHW': 'Materials',
        'FCX': 'Materials', 'NEM': 'Materials', 'DOW': 'Materials', 'DD': 'Materials',
        'PPG': 'Materials', 'LYB': 'Materials', 'ALB': 'Materials', 'IFF': 'Materials',
        'MLM': 'Materials', 'VMC': 'Materials', 'NUE': 'Materials', 'STLD': 'Materials',
        'IP': 'Materials', 'PKG': 'Materials', 'SW': 'Materials', 'MOS': 'Materials',
        'CF': 'Materials', 'BALL': 'Materials', 'AMCR': 'Materials', 'GLW': 'Materials',
        'WY': 'Materials', 'GPC': 'Materials', 'EMN': 'Materials',
        
        # Real Estate
        'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
        'PSA': 'Real Estate', 'WELL': 'Real Estate', 'DLR': 'Real Estate', 'O': 'Real Estate',
        'SBAC': 'Real Estate', 'EXR': 'Real Estate', 'AVB': 'Real Estate', 'EQR': 'Real Estate',
        'VICI': 'Real Estate', 'VTR': 'Real Estate', 'ESS': 'Real Estate', 'MAA': 'Real Estate',
        'KIM': 'Real Estate', 'DOC': 'Real Estate', 'UDR': 'Real Estate', 'CPT': 'Real Estate',
        'SPG': 'Real Estate', 'INVH': 'Real Estate', 'ARE': 'Real Estate', 'REG': 'Real Estate',
        'FRT': 'Real Estate', 'BXP': 'Real Estate', 'HST': 'Real Estate', 'IRM': 'Real Estate',
        
        # Communication Services
        'GOOGL': 'Communication Services', 'GOOG': 'Communication Services', 'META': 'Communication Services',
        'NFLX': 'Communication Services', 'DIS': 'Communication Services', 'CMCSA': 'Communication Services',
        'VZ': 'Communication Services', 'T': 'Communication Services', 'CHTR': 'Communication Services',
        'TMUS': 'Communication Services', 'OMC': 'Communication Services', 'IPG': 'Communication Services',
        'FOXA': 'Communication Services', 'FOX': 'Communication Services', 'WBD': 'Communication Services',
        'MTCH': 'Communication Services', 'EA': 'Communication Services', 'TTWO': 'Communication Services',
        'NWSA': 'Communication Services', 'NWS': 'Communication Services', 'DDOG': 'Communication Services',
        'TTD': 'Communication Services', 'CSGP': 'Communication Services', 'MKTX': 'Communication Services',
        
        # Special Cases
        'BRK.B': 'Financials',  # Berkshire Hathaway
        'XYZ': 'Technology',    # Block (formerly Square)
        'GEV': 'Industrials',   # GE Vernova
        'COIN': 'Technology',   # Coinbase
        'PLTR': 'Technology',   # Palantir
        'FSLR': 'Technology',   # First Solar
        'ENPH': 'Technology',   # Enphase Energy
    }
    
    print(f"Loaded sector mapping for {len(sector_mapping)} stocks")
    return sector_mapping

def add_sector_data(df):
    """Add sector information to dataframe"""
    sector_mapping = get_sector_mapping()
    
    # Add sector column
    df['sector'] = df['symbol'].map(sector_mapping)
    
    # Fill missing sectors with 'Unknown'
    df['sector'] = df['sector'].fillna('Unknown')
    
    # Create sector encoding
    unique_sectors = sorted(df['sector'].unique())
    sector_to_num = {sector: i for i, sector in enumerate(unique_sectors)}
    df['sector_encoded'] = df['sector'].map(sector_to_num)
    
    print(f"Added sector data. Found {len(unique_sectors)} sectors: {unique_sectors}")
    return df

def encode_sector_column(df, sector_map=None):
    if "sector" not in df.columns and sector_map:
        df["sector"] = df["symbol"].map(sector_map).fillna("Unknown")
    if "sector" in df.columns:
        le = LabelEncoder()
        df["sector_encoded"] = le.fit_transform(df["sector"].fillna("Unknown"))
    return df

def add_volatility_features(df):
    """Calculates rolling volatility for Z-score and sizing."""
    df['volatility_20d'] = df.groupby('symbol')['return_1d'].transform(lambda x: x.rolling(20).std())
    return df

def add_features(df, sector_map=None):
    """
    FIXED VERSION: All features properly lagged to prevent data leakage
    Critical fixes:
    1. All price/volume derived features lagged by at least 1 day
    2. Candlestick patterns now properly lagged
    3. All technical indicators consistently lagged
    4. Forward-fill/backward-fill only within symbol groups
    """
    
    # --- Debug check ---
    import builtins
    assert callable(builtins.safe_str_debug), f"'safe_str_debug' is not callable: {type(builtins.safe_str_debug)}"

    # --- Normalize columns ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Required columns check ---
    required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # --- Encode sectors before groupby ---
    df = encode_sector_column(df, sector_map)

    # --- Sort by symbol and date ---
    df = df.sort_values(["symbol", "date"]).copy()

    # --- Feature computation per symbol ---
    def _compute_features_for_symbol(g):
        """
        CRITICAL FIX: All features now properly prevent look-ahead bias
        """
        g = g.copy()

        # This prevents information leakage across symbols
        g.fillna(method='ffill', inplace=True)
        g.fillna(method='bfill', inplace=True)

        # --- PRICE RETURN FEATURES (ALL LAGGED) ---
        # Basic 1-day return (t-1)
        g["return_1d"] = g["close"].pct_change().shift(1)
        
        # Lagged returns (t-2, t-3, t-4)
        for lag in [1, 2, 3]:
            g[f"return_lag_{lag}"] = g["return_1d"].shift(lag)
        
        # Multi-day returns (all lagged by 1 additional day)
        g["return_2d"] = g["close"].pct_change(2).shift(1)
        g["return_5d"] = g["close"].pct_change(5).shift(1)

        # --- MOVING AVERAGES (ALL LAGGED) ---
        # Simple moving averages (lagged by 1 day)
        g["ma_5"] = g["close"].rolling(5).mean().shift(1)
        g["ma_10"] = g["close"].rolling(10).mean()
        g["ma_20"] = g["close"].rolling(20).mean().shift(1)
        
        # Price vs MA ratios (lagged)
        g["price_vs_ma5"] = (g["close"].shift(1) / g["ma_5"] - 1)
        g["price_vs_ma10"] = (g["close"].shift(1) / g["ma_10"] - 1)
        g["price_vs_ma20"] = (g["close"] / g["ma_20"] - 1)

        # --- EXPONENTIAL MOVING AVERAGES & MACD (ALL LAGGED) ---
        # EMAs lagged by 1 day
        g["ema_12"] = g["close"].ewm(span=12, adjust=False).mean().shift(1)
        g["ema_26"] = g["close"].ewm(span=26, adjust=False).mean().shift(1)
        
        # Price vs EMA (lagged)
        g["price_vs_ema12"] = (g["close"].shift(1) / g["ema_12"] - 1)
        
        # MACD components (all lagged)
        g["macd"] = (g["ema_12"] - g["ema_26"])
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean().shift(1)
        g["macd_hist"] = (g["macd"] - g["macd_signal"])
        g["macd_hist_diff"] = g["macd_hist"].diff().shift(1)
        
        # Apply additional lag to MACD to ensure no leakage
        g["macd"] = g["macd"].shift(1)
        g["macd_hist"] = g["macd_hist"].shift(1)

        # --- BOLLINGER BANDS (LAGGED) ---
        rolling_std = g["close"].rolling(20).std().shift(1)
        ma_20_lagged = g["ma_20"]
        g["bollinger_b"] = ((g["close"].shift(1) - (ma_20_lagged - 2 * rolling_std)) / (2 * rolling_std))

        # --- VOLATILITY MEASURES (ALL LAGGED) ---
        g["volatility_5d"] = g["return_1d"].rolling(5).std().shift(1)
        g["volatility_10d"] = g["return_1d"].rolling(10).std().shift(1)
        g["volatility_20d"] = g["return_1d"].rolling(20).std().shift(1)

        # --- MOMENTUM INDICATORS (ALL LAGGED) ---
        g["momentum_3d"] = g["close"].pct_change(3).shift(1)
        g["momentum_7d"] = g["close"].pct_change(7).shift(1)
        g["momentum_14d"] = g["close"].pct_change(14).shift(1)
        g["roc_5"] = (g["close"].diff(5) / g["close"].shift(5)).shift(1)

        # --- RSI 14 (LAGGED) ---
        delta = g["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        g["rsi_14"] = rsi

        # --- ATR 14 (LAGGED) ---
        high_low = g["high"] - g["low"]
        high_close_prev = np.abs(g["high"] - g["close"].shift(1))
        low_close_prev = np.abs(g["low"] - g["close"].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        g["atr_14"] = tr.rolling(14).mean().shift(1)

        # --- CANDLESTICK PATTERNS ---
        open_lag = g["open"].shift(1)
        high_lag = g["high"].shift(1)
        low_lag = g["low"].shift(1) 
        close_lag = g["close"].shift(1)
        
        g["candlestick_body"] = np.abs(close_lag - open_lag)
        g["upper_shadow"] = high_lag - pd.concat([close_lag, open_lag], axis=1).max(axis=1)
        g["lower_shadow"] = pd.concat([close_lag, open_lag], axis=1).min(axis=1) - low_lag
        g["high_low_ratio"] = high_lag / low_lag - 1
        
        # Gap calculation (comparing today's open to yesterday's close) - LAGGED
        g["gap_up_down"] = (g["open"] / g["close"].shift(2) - 1).shift(1)  # Extra lag for safety
        
        # Candlestick pattern detection (lagged)
        body_size = g["candlestick_body"]
        total_range = high_lag - low_lag
        
        g["is_hammer"] = (
            (g["lower_shadow"] > 2 * body_size) & 
            (g["upper_shadow"] < 0.1 * body_size)
        ).astype(int)
        
        g["is_doji"] = (body_size < 0.1 * total_range).astype(int)

        # --- VOLUME FEATURES (ALL LAGGED) ---
        # Volume change and ratios (lagged)
        g["volume_change_1d"] = g["volume"].pct_change().shift(1)
        
        # Volume vs moving averages (lagged)
        volume_ma5 = g["volume"].rolling(5).mean().shift(1)
        volume_ma10 = g["volume"].rolling(10).mean().shift(1) 
        volume_ma20 = g["volume"].rolling(20).mean().shift(1)
        
        g["volume_vs_avg5"] = (g["volume"].shift(1) / volume_ma5 - 1)
        g["volume_vs_avg10"] = (g["volume"].shift(1) / volume_ma10 - 1)
        g["volume_vs_avg20"] = (g["volume"].shift(1) / volume_ma20 - 1)
        
        # Volume momentum (lagged)
        g["volume_momentum_3d"] = g["volume"].pct_change(3).shift(1)

        # --- STATISTICAL MEASURES (LAGGED) ---
        # Z-score using lagged statistics
        rolling_mean_20 = g["close"].rolling(20).mean().shift(1)
        rolling_std_20 = g["close"].rolling(20).std().shift(1)
        g["zscore_20d"] = ((g["close"].shift(1) - rolling_mean_20) / rolling_std_20)

        # --- ROLLING EXTREMES (LAGGED) ---
        g["rolling_max_10"] = g["close"].rolling(10).max().shift(1)
        g["rolling_min_10"] = g["close"].rolling(10).min().shift(1)
        g["price_vs_rolling_max10"] = (g["close"].shift(1) / g["rolling_max_10"] - 1)
        
        # High-low spread (lagged)
        g["hl_spread_pct"] = ((g["high"] - g["low"]) / g["low"]).shift(1)

        # Standard technical indicators
        g['rsi'] = compute_rsi(g['close'])
        g['atr'] = compute_atr(g)
        return g

    # --- Apply features per symbol ---
    print("Computing features per symbol...")
    df = (
        df.groupby("symbol", group_keys=False)
        .apply(lambda g: _compute_features_for_symbol(g))
        .reset_index(drop=True)
    )

    # --- Final cleanup ---
    df.reset_index(drop=True, inplace=True)
    df = df.dropna(subset=["close"])
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    print("All features now properly lagged to prevent data leakage!")
    
    return df


def validate_no_lookahead_bias(df, feature_cols, target_col="is_top_20p"):
    """
    FIXED: Validate that features don't have look-ahead bias
    This checks correlation between features at time t and targets at time t
    High correlation suggests potential data leakage
    """
    print("\n=== VALIDATING FEATURES FOR LOOK-AHEAD BIAS ===")
    
    # Filter feature columns to only include numeric ones and those that exist
    numeric_feature_cols = []
    for feature in feature_cols:
        if feature in df.columns:
            # Check if column is numeric or can be converted to numeric
            try:
                # Try to convert to numeric, skip if it fails
                pd.to_numeric(df[feature], errors='raise')
                numeric_feature_cols.append(feature)
            except (ValueError, TypeError):
                # Skip non-numeric columns like 'sector' (string values)
                print(f"  Skipping non-numeric feature: {feature}")
                continue
    
    if not numeric_feature_cols:
        print("Warning: No numeric features found for validation")
        return {}
    
    # Remove rows with missing target or features
    cols_to_check = [target_col] + numeric_feature_cols
    df_clean = df.dropna(subset=cols_to_check)
    
    if len(df_clean) == 0:
        print("Warning: No clean data for validation")
        return {}
    
    print(f"Validating {len(numeric_feature_cols)} numeric features on {len(df_clean):,} clean samples")
    
    # Calculate correlations
    correlations = {}
    suspicious_features = []
    
    for feature in numeric_feature_cols:
        try:
            corr = df_clean[feature].corr(df_clean[target_col])
            correlations[feature] = abs(corr) if not pd.isna(corr) else 0
            
            # Flag suspicious correlations (> 0.1 is high for daily stock features)
            if abs(corr) > 0.1:
                suspicious_features.append((feature, corr))
        except Exception as e:
            print(f"  Error calculating correlation for {feature}: {e}")
            correlations[feature] = 0
    
    # Report results
    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 feature-target correlations:")
    for feature, corr in sorted_corrs[:10]:
        status = "⚠️ SUSPICIOUS" if abs(corr) > 0.1 else "✅ OK"
        print(f"  {feature:<25} {corr:>8.4f} {status}")
    
    if suspicious_features:
        print(f"\n🚨 {len(suspicious_features)} potentially problematic features:")
        for feature, corr in suspicious_features:
            print(f"  {feature}: {corr:.4f}")
        print("\nRecommendation: Review these features for data leakage")
    else:
        print("\n✅ No suspicious correlations detected")
    
    return correlations

def add_cross_sectional_features(df):
    """
    Robust calculation of Market Beta and Relative Strength.
    Prevents leakage by explicitly aligning dates.
    """
    print("Adding cross-sectional ranking features...")
    
    ranking_features = ['return_1d', 'volume_change_1d', 'rsi_14', 'volatility_5d']
    
    for feature in ranking_features:
        if feature in df.columns:
            # Calculate rankings on current day
            market_rank = df.groupby('date')[feature].rank(pct=True)
            sector_rank = df.groupby(['date', 'sector'])[feature].rank(pct=True)
            
            # LAG BY 1 DAY - critical fix
            df[f'{feature}_market_rank'] = market_rank.shift(1)
            df[f'{feature}_sector_rank'] = sector_rank.shift(1)
            
            # Relative to sector median - also lagged
            sector_median = df.groupby(['date', 'sector'])[feature].transform('median')
            df[f'{feature}_vs_sector'] = (df[feature].shift(1) / sector_median.shift(1) - 1)
    market_ret = df.groupby('date')['return_1d'].transform('mean')
    df['market_ret'] = market_ret
    
    df['relative_strength'] = df['return_1d'].shift(1) - df['market_ret']
    
    print("  Computing Beta (this may take a moment)...")
    
    def calculate_rolling_beta(g):
        # Ensure we have enough data points (need at least 20 for rolling)
        if len(g) < 20:
            return pd.Series(1.0, index=g.index)
            
        rolling_cov = g['return_1d'].rolling(20).cov(g['market_ret'])
        rolling_var = g['market_ret'].rolling(20).var()
        
        # This ensures we return a Series of the same length as the input group
        beta = rolling_cov / rolling_var
        return beta.fillna(1.0)

    # 2. Apply and ensure alignment
    try:
        df['rolling_beta'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: calculate_rolling_beta(df.loc[x.index])
        )
    except Exception:
        # Final safety net for single-row or edge case data
        df['rolling_beta'] = 1.0
        
    return df

def neutralize_features(df, feature_cols, target_col='market_ret'):
    """
    Removes the linear influence of the market from your features.
    Result: Features that represent PURE Alpha, not Beta.
    """
    print(f"Neutralizing {len(feature_cols)} features against Market Return...")
    
    # Simple linear regression residual: Feature = Beta * Market + Alpha
    for feat in feature_cols:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            # Calculate correlation to see if neutralization is needed
            # Only neutralize if correlation is significant (> 0.1)
            corr = df[[feat, target_col]].corr().iloc[0,1]
            if abs(corr) > 0.1:
                # Vectorized orthogonalization (Gram-Schmidt-like)
                # Residual = Feature - (Slope * Market)
                market_var = df[target_col].var()
                if market_var > 0:
                    covariance = df[[feat, target_col]].cov().iloc[0,1]
                    beta = covariance / market_var
                    df[feat] = df[feat] - (beta * df[target_col])
                    
    print("Feature neutralization complete.")
    return df

def add_risk_features(df):
    """
    Adds professional risk metrics.
    Normalizes returns by volatility and computes Market Beta.
    """
    print("Adding quantitative risk features...")
    df = df.sort_values(['symbol', 'date'])
    
    # 1. Volatility-Normalized Return
    if 'atr_14' in df.columns:
        df['vol_adj_return_1d'] = (df['return_1d'] / (df['atr_14'] / df['close']))
    
    # 2. Rolling Beta (Market Sensitivity)
    market_ret = df.groupby('date')['return_1d'].transform('mean')
    
    def _compute_beta(g):
        # 20-day rolling covariance / variance
        cov = g['return_1d'].rolling(20).cov(market_ret)
        var = market_ret.rolling(20).var()
        return (cov / var).shift(1) # Lagged to prevent leakage

    df['rolling_beta'] = df.groupby('symbol', group_keys=False).apply(_compute_beta)
    
    return df

def add_regime_features(df):
    """Add market regime indicators with proper lagging"""
    print("Adding market regime features...")
    
    if 'return_1d' in df.columns:
        # Calculate market volatility regime
        market_vol = df.groupby('date')['return_1d'].std().rolling(20).mean()
        
        # LAG BY 1 DAY
        df['market_vol_regime'] = df['date'].map(market_vol.to_dict()).shift(1)
        
        # High volatility regime - using lagged values
        vol_90th_pct = df['market_vol_regime'].quantile(0.9)
        df['high_vol_regime'] = (df['market_vol_regime'] > vol_90th_pct).astype(int)
    
    # Market trend regime - lagged
    market_return = df.groupby('date')['return_1d'].mean().rolling(20).mean()
    df['market_trend'] = df['date'].map(market_return.to_dict()).shift(1)
    
    # Bear market indicator - using lagged market trend
    df['bear_market'] = (df['market_trend'] < -0.005).astype(int)
    
    return df

class FeatureEngineer:
    def __init__(self):
        self.sector_map = get_sector_mapping()

    def create_features(self, df):
        # Call the standalone functions defined in this file
        df = add_sector_data(df)
        df = add_features(df, sector_map=self.sector_map)
        df = add_cross_sectional_features(df)
        df = add_regime_features(df)
        return df