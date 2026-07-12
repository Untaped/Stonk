# feature_engineering.py
import os

import joblib
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Low-level technical indicators
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI using exponential weighted smoothing (alpha = 1/period).

    FIX: the previous implementation used .rolling(period).mean() for both
    gain and loss, which is a simple arithmetic mean — not the Wilder average
    used by every charting platform.  This diverges meaningfully near RSI
    extremes (overbought / oversold thresholds), so the model was trained on
    RSI levels that don't correspond to the industry-standard signal.

    Wilder's method: seed the first average as the plain mean of the first
    `period` observations, then apply an EWM with alpha=1/period from there.
    pandas .ewm(alpha=1/period, adjust=False) replicates this exactly because
    adjust=False uses the recursive formula:
        avg_t = alpha * obs_t + (1 - alpha) * avg_{t-1}
    which is Wilder's smoothed average.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.

    FIX: use pandas .max(axis=1) instead of np.max() so NaNs are handled
    correctly and the result is always a properly-indexed Series.
    """
    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Sector mapping
# ══════════════════════════════════════════════════════════════════════════════

def get_sector_mapping() -> dict:
    """
    Return a symbol → sector mapping.

    FIX: the CSV written by daily_data_collector.py is 'sp500_fundamentals.csv',
    not 'nasdaq_fundamentals.csv'.  Try that file first, then fall back to the
    hardcoded list.
    """
    csv_path = "data/sp500_fundamentals.csv"          # ← was 'nasdaq_fundamentals.csv'

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "symbol" in df.columns and "sector" in df.columns:
                df["symbol"] = df["symbol"].astype(str).str.replace(".", "-", regex=False)
                mapping = dict(zip(df["symbol"], df["sector"]))
                print(f"✅ Loaded dynamic sector mapping for {len(mapping)} stocks.")
                return mapping
        except Exception as e:
            print(f"⚠️ Error reading sector CSV: {e}")

    print("⚠️ Sector CSV not found or invalid. Using fallback hardcoded list.")
    sector_mapping = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
        "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
        "ADBE": "Technology", "INTC": "Technology", "CSCO": "Technology",
        "IBM":  "Technology", "QCOM": "Technology", "TXN": "Technology",
        "NOW":  "Technology", "INTU": "Technology", "AMD":  "Technology",
        "MU":   "Technology", "AMAT": "Technology", "ADI":  "Technology",
        "LRCX": "Technology", "KLAC": "Technology", "MCHP": "Technology",
        "CDNS": "Technology", "SNPS": "Technology", "ANET": "Technology",
        "PANW": "Technology", "CRWD": "Technology", "WDAY": "Technology",
        "ADSK": "Technology", "FTNT": "Technology", "TEL":  "Technology",
        "NXPI": "Technology", "ON":   "Technology", "KEYS": "Technology",
        "DELL": "Technology", "HPE":  "Technology", "HPQ":  "Technology",
        "NTAP": "Technology", "FFIV": "Technology", "AKAM": "Technology",
        "SMCI": "Technology", "WDC":  "Technology", "STX":  "Technology",
        "TER":  "Technology", "SWKS": "Technology", "PTC":  "Technology",
        "GDDY": "Technology", "TYL":  "Technology", "TRMB": "Technology",
        "GEN":  "Technology", "VRSN": "Technology", "TDY":  "Technology",
        "IT":   "Technology", "CDW":  "Technology",
        "XYZ":  "Technology", "COIN": "Technology", "PLTR": "Technology",
        "FSLR": "Technology", "ENPH": "Technology",

        # Communication Services
        "GOOGL": "Communication Services", "GOOG": "Communication Services",
        "META":  "Communication Services", "NFLX": "Communication Services",
        "DIS":   "Communication Services", "CMCSA":"Communication Services",
        "VZ":    "Communication Services", "T":    "Communication Services",
        "CHTR":  "Communication Services", "TMUS": "Communication Services",
        "OMC":   "Communication Services", "IPG":  "Communication Services",
        "FOXA":  "Communication Services", "FOX":  "Communication Services",
        "WBD":   "Communication Services", "MTCH": "Communication Services",
        "EA":    "Communication Services", "TTWO": "Communication Services",
        "NWSA":  "Communication Services", "NWS":  "Communication Services",
        "DDOG":  "Communication Services", "TTD":  "Communication Services",
        "CSGP":  "Communication Services", "MKTX": "Communication Services",

        # Healthcare
        "UNH": "Healthcare", "JNJ":  "Healthcare", "PFE":  "Healthcare",
        "ABBV":"Healthcare", "LLY":  "Healthcare", "MRK":  "Healthcare",
        "TMO": "Healthcare", "ABT":  "Healthcare", "MDT":  "Healthcare",
        "DHR": "Healthcare", "BMY":  "Healthcare", "AMGN": "Healthcare",
        "GILD":"Healthcare", "CVS":  "Healthcare", "CI":   "Healthcare",
        "HUM": "Healthcare", "SYK":  "Healthcare", "BDX":  "Healthcare",
        "ZTS": "Healthcare", "BSX":  "Healthcare", "EW":   "Healthcare",
        "ISRG":"Healthcare", "IDXX": "Healthcare", "RMD":  "Healthcare",
        "DXCM":"Healthcare", "ALGN": "Healthcare", "MTD":  "Healthcare",
        "IQV": "Healthcare", "A":    "Healthcare", "VRTX": "Healthcare",
        "REGN":"Healthcare", "ELV":  "Healthcare", "MCK":  "Healthcare",
        "COR": "Healthcare", "HCA":  "Healthcare", "CAH":  "Healthcare",
        "GEHC":"Healthcare", "BIIB": "Healthcare", "DGX":  "Healthcare",
        "SOLV":"Healthcare", "VTRS": "Healthcare", "BAX":  "Healthcare",
        "HOLX":"Healthcare", "PODD": "Healthcare", "LH":   "Healthcare",
        "ZBH": "Healthcare", "MOH":  "Healthcare", "TECH": "Healthcare",
        "HSIC":"Healthcare", "CRL":  "Healthcare", "RVTY": "Healthcare",
        "WST": "Healthcare", "WAT":  "Healthcare", "COO":  "Healthcare",
        "INCY":"Healthcare", "UHS":  "Healthcare", "MRNA": "Healthcare",
        "DVA": "Healthcare", "EPAM": "Healthcare",

        # Financials
        "JPM": "Financials", "BAC":  "Financials", "WFC":  "Financials",
        "GS":  "Financials", "MS":   "Financials", "C":    "Financials",
        "BLK": "Financials", "AXP":  "Financials", "SPGI": "Financials",
        "BK":  "Financials", "USB":  "Financials", "TFC":  "Financials",
        "PNC": "Financials", "COF":  "Financials", "SCHW": "Financials",
        "CB":  "Financials", "MMC":  "Financials", "ICE":  "Financials",
        "CME": "Financials", "AON":  "Financials", "PGR":  "Financials",
        "TRV": "Financials", "AFL":  "Financials", "ALL":  "Financials",
        "MET": "Financials", "PRU":  "Financials", "AIG":  "Financials",
        "HIG": "Financials", "CINF": "Financials", "L":    "Financials",
        "BX":  "Financials", "KKR":  "Financials", "APO":  "Financials",
        "NDAQ":"Financials", "MCO":  "Financials", "MSCI": "Financials",
        "AMP": "Financials", "AJG":  "Financials", "FI":   "Financials",
        "ACGL":"Financials", "WTW":  "Financials", "FICO": "Financials",
        "BRO": "Financials", "STT":  "Financials", "MTB":  "Financials",
        "FITB":"Financials", "CFG":  "Financials", "RF":   "Financials",
        "HBAN":"Financials", "NTRS": "Financials", "KEY":  "Financials",
        "SYF": "Financials", "RJF":  "Financials", "TROW": "Financials",
        "PFG": "Financials", "BEN":  "Financials", "IVZ":  "Financials",
        "AIZ": "Financials", "GL":   "Financials", "EG":   "Financials",
        "CBOE":"Financials", "GPN":  "Financials", "FIS":  "Financials",
        "BR":  "Financials", "WRB":  "Financials", "ERIE": "Financials",
        "BRK-B":"Financials",                          # yfinance normalised form

        # Consumer Discretionary
        "AMZN":"Consumer Discretionary", "TSLA":"Consumer Discretionary",
        "HD":  "Consumer Discretionary", "MCD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
        "SBUX":"Consumer Discretionary", "TJX": "Consumer Discretionary",
        "BKNG":"Consumer Discretionary", "GM":  "Consumer Discretionary",
        "F":   "Consumer Discretionary", "MAR": "Consumer Discretionary",
        "HLT": "Consumer Discretionary", "CMG": "Consumer Discretionary",
        "ORLY":"Consumer Discretionary", "YUM": "Consumer Discretionary",
        "LULU":"Consumer Discretionary", "AZO": "Consumer Discretionary",
        "ROST":"Consumer Discretionary", "TGT": "Consumer Discretionary",
        "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
        "NVR": "Consumer Discretionary", "PHM": "Consumer Discretionary",
        "GRMN":"Consumer Discretionary", "POOL":"Consumer Discretionary",
        "CCL": "Consumer Discretionary", "RCL": "Consumer Discretionary",
        "ABNB":"Consumer Discretionary", "UBER":"Consumer Discretionary",
        "DASH":"Consumer Discretionary", "TPR": "Consumer Discretionary",
        "RL":  "Consumer Discretionary", "DECK":"Consumer Discretionary",
        "DPZ": "Consumer Discretionary", "BLDR":"Consumer Discretionary",
        "BBY": "Consumer Discretionary", "DG":  "Consumer Discretionary",
        "DLTR":"Consumer Discretionary", "ULTA":"Consumer Discretionary",
        "DRI": "Consumer Discretionary", "HAS": "Consumer Discretionary",
        "WYNN":"Consumer Discretionary", "NCLH":"Consumer Discretionary",
        "MGM": "Consumer Discretionary", "LVS": "Consumer Discretionary",
        "EXPE":"Consumer Discretionary", "WSM": "Consumer Discretionary",
        "KMX": "Consumer Discretionary", "LKQ": "Consumer Discretionary",
        "MHK": "Consumer Discretionary", "APTV":"Consumer Discretionary",
        "LYV": "Consumer Discretionary", "TKO": "Consumer Discretionary",
        "CZR": "Consumer Discretionary", "MAS": "Consumer Discretionary",
        "UAL": "Consumer Discretionary", "DAL": "Consumer Discretionary",
        "LUV": "Consumer Discretionary",

        # Consumer Staples
        "WMT": "Consumer Staples", "PG":   "Consumer Staples",
        "KO":  "Consumer Staples", "PEP":  "Consumer Staples",
        "COST":"Consumer Staples", "MDLZ": "Consumer Staples",
        "KMB": "Consumer Staples", "GIS":  "Consumer Staples",
        "SYY": "Consumer Staples", "ADM":  "Consumer Staples",
        "HSY": "Consumer Staples", "K":    "Consumer Staples",
        "CHD": "Consumer Staples", "CLX":  "Consumer Staples",
        "CAG": "Consumer Staples", "TSN":  "Consumer Staples",
        "HRL": "Consumer Staples", "MKC":  "Consumer Staples",
        "CPB": "Consumer Staples", "SJM":  "Consumer Staples",
        "LW":  "Consumer Staples", "TAP":  "Consumer Staples",
        "KDP": "Consumer Staples", "MNST": "Consumer Staples",
        "KR":  "Consumer Staples", "CL":   "Consumer Staples",
        "PM":  "Consumer Staples", "MO":   "Consumer Staples",
        "STZ": "Consumer Staples", "BF-B": "Consumer Staples",
        "KHC": "Consumer Staples", "KVUE": "Consumer Staples",
        "EL":  "Consumer Staples", "WBA":  "Consumer Staples",
        "BG":  "Consumer Staples",

        # Energy
        "XOM": "Energy", "CVX":  "Energy", "COP":  "Energy", "EOG":  "Energy",
        "SLB": "Energy", "PSX":  "Energy", "VLO":  "Energy", "MPC":  "Energy",
        "KMI": "Energy", "OKE":  "Energy", "WMB":  "Energy", "FANG": "Energy",
        "EQT": "Energy", "OXY":  "Energy", "BKR":  "Energy", "DVN":  "Energy",
        "TRGP":"Energy", "CTRA": "Energy", "HAL":  "Energy", "APA":  "Energy",
        "TPL": "Energy",

        # Industrials
        "BA":  "Industrials", "CAT":  "Industrials", "RTX":  "Industrials",
        "HON": "Industrials", "UPS":  "Industrials", "LMT":  "Industrials",
        "GE":  "Industrials", "MMM":  "Industrials", "FDX":  "Industrials",
        "NOC": "Industrials", "GD":   "Industrials", "WM":   "Industrials",
        "RSG": "Industrials", "ITW":  "Industrials", "PH":   "Industrials",
        "CMI": "Industrials", "ETN":  "Industrials", "EMR":  "Industrials",
        "PCAR":"Industrials", "IR":   "Industrials", "OTIS": "Industrials",
        "CARR":"Industrials", "NSC":  "Industrials", "CSX":  "Industrials",
        "UNP": "Industrials", "DE":   "Industrials", "TT":   "Industrials",
        "APH": "Industrials", "HWM":  "Industrials", "JCI":  "Industrials",
        "TDG": "Industrials", "MSI":  "Industrials", "AXON": "Industrials",
        "URI": "Industrials", "GWW":  "Industrials", "PWR":  "Industrials",
        "ROP": "Industrials", "FAST": "Industrials", "PAYX": "Industrials",
        "ADP": "Industrials", "ROK":  "Industrials", "AME":  "Industrials",
        "ODFL":"Industrials", "WAB":  "Industrials", "LHX":  "Industrials",
        "HUBB":"Industrials", "LDOS": "Industrials", "CPAY": "Industrials",
        "VRSK":"Industrials", "FTV":  "Industrials", "EXPD": "Industrials",
        "ZBRA":"Industrials", "ALLE": "Industrials", "CHRW": "Industrials",
        "TXT": "Industrials", "DOV":  "Industrials", "STE":  "Industrials",
        "PAYC":"Industrials", "JKHY": "Industrials", "SWK":  "Industrials",
        "GNRC":"Industrials", "NDSN": "Industrials", "J":    "Industrials",
        "PNR": "Industrials", "SNA":  "Industrials", "JBHT": "Industrials",
        "AVY": "Industrials", "IEX":  "Industrials", "XYL":  "Industrials",
        "ROL": "Industrials", "VLTO": "Industrials", "LII":  "Industrials",
        "HII": "Industrials", "DAY":  "Industrials", "CTAS": "Industrials",
        "GEV": "Industrials",

        # Utilities
        "NEE": "Utilities", "SO":   "Utilities", "DUK":  "Utilities",
        "AEP": "Utilities", "SRE":  "Utilities", "D":    "Utilities",
        "PCG": "Utilities", "EXC":  "Utilities", "ED":   "Utilities",
        "ETR": "Utilities", "WEC":  "Utilities", "ES":   "Utilities",
        "FE":  "Utilities", "EIX":  "Utilities", "PPL":  "Utilities",
        "EVRG":"Utilities", "AWK":  "Utilities", "DTE":  "Utilities",
        "PEG": "Utilities", "NI":   "Utilities", "CEG":  "Utilities",
        "VST": "Utilities", "NRG":  "Utilities", "AEE":  "Utilities",
        "CMS": "Utilities", "XEL":  "Utilities", "CNP":  "Utilities",
        "ATO": "Utilities", "PNW":  "Utilities", "LNT":  "Utilities",
        "AES": "Utilities",

        # Materials
        "LIN": "Materials", "APD":  "Materials", "ECL":  "Materials",
        "SHW": "Materials", "FCX":  "Materials", "NEM":  "Materials",
        "DOW": "Materials", "DD":   "Materials", "PPG":  "Materials",
        "LYB": "Materials", "ALB":  "Materials", "IFF":  "Materials",
        "MLM": "Materials", "VMC":  "Materials", "NUE":  "Materials",
        "STLD":"Materials", "IP":   "Materials", "PKG":  "Materials",
        "SW":  "Materials", "MOS":  "Materials", "CF":   "Materials",
        "BALL":"Materials", "AMCR": "Materials", "GLW":  "Materials",
        "WY":  "Materials", "GPC":  "Materials", "EMN":  "Materials",

        # Real Estate
        "AMT":  "Real Estate", "PLD":  "Real Estate", "CCI":  "Real Estate",
        "EQIX": "Real Estate", "PSA":  "Real Estate", "WELL": "Real Estate",
        "DLR":  "Real Estate", "O":    "Real Estate", "SBAC": "Real Estate",
        "EXR":  "Real Estate", "AVB":  "Real Estate", "EQR":  "Real Estate",
        "VICI": "Real Estate", "VTR":  "Real Estate", "ESS":  "Real Estate",
        "MAA":  "Real Estate", "KIM":  "Real Estate", "DOC":  "Real Estate",
        "UDR":  "Real Estate", "CPT":  "Real Estate", "SPG":  "Real Estate",
        "INVH": "Real Estate", "ARE":  "Real Estate", "REG":  "Real Estate",
        "FRT":  "Real Estate", "BXP":  "Real Estate", "HST":  "Real Estate",
        "IRM":  "Real Estate",
    }

    print(f"Loaded sector mapping for {len(sector_mapping)} stocks")
    return sector_mapping


# ══════════════════════════════════════════════════════════════════════════════
# Sector helpers
# ══════════════════════════════════════════════════════════════════════════════

def add_sector_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector column + ordinal encoding to dataframe."""
    sector_mapping = get_sector_mapping()
    df = df.copy()
    df["sector"] = df["symbol"].map(sector_mapping).fillna("Unknown")

    unique_sectors = sorted(df["sector"].unique())
    sector_to_num  = {s: i for i, s in enumerate(unique_sectors)}
    df["sector_encoded"] = df["sector"].map(sector_to_num)

    print(f"Added sector data. Found {len(unique_sectors)} sectors: {unique_sectors}")
    return df


# ── Persisted, fixed sector → int encoding (Issue #5) ──────────────────────────
# The 11 standard GICS sectors, alphabetically, plus an explicit "Unknown"
# bucket for anything that doesn't map cleanly (missing sector data, a
# not-yet-classified new listing, etc). Fixed order = fixed integer codes,
# forever — this is the whole point: a symbol's sector_encoded value must be
# the same number whether it's 2019 training data or today's live scan.
GICS_SECTORS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Healthcare",
    "Industrials",
    "Materials",
    "Real Estate",
    "Technology",
    "Utilities",
    "Unknown",
]
SECTOR_ENCODING_PATH = "stock_ai/sector_encoding.joblib"


def get_sector_encoding() -> dict:
    """
    Return the persisted {sector_name: int} mapping used for the
    'sector_encoded' feature, creating and saving it on first use.

    This is deliberately NOT a LabelEncoder().fit_transform() on whatever
    sectors happen to be present in a given call — see encode_sector_column()
    for why that was unsafe. It's fixed once against the full canonical GICS
    list (so it never needs to change shape) and reused verbatim by every
    caller from then on, training and inference alike.
    """
    if os.path.exists(SECTOR_ENCODING_PATH):
        try:
            return joblib.load(SECTOR_ENCODING_PATH)
        except Exception as e:
            print(f"⚠️ Could not load {SECTOR_ENCODING_PATH}: {e}; rebuilding.")

    mapping = {name: i for i, name in enumerate(GICS_SECTORS)}
    try:
        out_dir = os.path.dirname(SECTOR_ENCODING_PATH)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(mapping, SECTOR_ENCODING_PATH)
        print(f"✅ Created sector encoding at {SECTOR_ENCODING_PATH}: {mapping}")
    except Exception as e:
        print(f"⚠️ Could not persist {SECTOR_ENCODING_PATH}: {e} (using in-memory mapping)")
    return mapping


def encode_sector_column(df: pd.DataFrame, sector_map: dict = None) -> pd.DataFrame:
    """
    Ensure 'sector' and 'sector_encoded' columns exist.

    FIX (Issue #5): sector_encoded used to come from a fresh
    LabelEncoder().fit_transform(df["sector"]) on *every call*, both at
    training and separately at each inference call. That happens to line up
    only because training and inference currently both see all 11 GICS
    sectors on a given day, so alphabetical ordering is stable — but nothing
    enforced that. If one sector's data failed to fetch on a given day,
    every sector's integer code would shift silently (no error raised),
    quietly corrupting this feature for the whole run. It was also fed to
    LightGBM as a plain numeric column, imposing a fake ordering (Real
    Estate "between" Materials and Technology) the trees had to work around.

    Now sector_encoded comes from get_sector_encoding() — a mapping fixed
    once over the full canonical GICS sector list, persisted to disk, and
    reused verbatim by every caller (training and inference alike) forever
    after. Pair with categorical_feature=['sector_encoded'] at fit time
    (see train_model.py) so LightGBM treats it as unordered.
    """
    if "sector" not in df.columns:
        if sector_map:
            df["sector"] = df["symbol"].map(sector_map).fillna("Unknown")
        else:
            df["sector"] = "Unknown"
    else:
        df["sector"] = df["sector"].fillna("Unknown")

    encoding = get_sector_encoding()
    unknown_code = encoding.get("Unknown", max(encoding.values(), default=0) + 1)

    df["sector_encoded"] = df["sector"].map(encoding)
    if df["sector_encoded"].isna().any():
        newly_seen = sorted(set(df.loc[df["sector_encoded"].isna(), "sector"]))
        print(
            f"⚠️ Sector(s) not in the persisted encoding, mapping to 'Unknown' "
            f"(code {unknown_code}): {newly_seen}"
        )
        df["sector_encoded"] = df["sector_encoded"].fillna(unknown_code)
    df["sector_encoded"] = df["sector_encoded"].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Per-symbol feature block
# ══════════════════════════════════════════════════════════════════════════════

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rolling 20-day return volatility per symbol."""
    df["volatility_20d"] = df.groupby("symbol")["return_1d"].transform(
        lambda x: x.rolling(20).std()
    )
    return df


def add_features(df: pd.DataFrame, sector_map: dict = None) -> pd.DataFrame:
    """
    Main per-symbol feature engineering.

    Order of operations
    -------------------
    1. Normalise column names.
    2. Encode sector once (prevents double-encoding by FeatureEngineer).
    3. Sort by [symbol, date].
    4. Compute technicals per symbol group.
    5. Compute market-relative features across the full frame.
    """

    # ── Normalise columns ────────────────────────────────────────────────────
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ── Encode sectors (only if not already encoded) ─────────────────────────
    if "sector_encoded" not in df.columns:
        df = encode_sector_column(df, sector_map)

    # ── Sort ─────────────────────────────────────────────────────────────────
    df = df.sort_values(["symbol", "date"]).copy()

    # ── Per-symbol technicals ─────────────────────────────────────────────────
    def _compute_features_for_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        # Forward-fill only: carry the last known OHLCV value forward across
        # missing sessions (e.g. trading halts, stale data feeds).
        # bfill is intentionally absent — filling backward uses a *future*
        # observation to patch a gap, which is lookahead bias.  Any NaNs that
        # remain after ffill (leading rows with no prior data) are left as NaN
        # and handled by column-median imputation in clean_feature_matrix.
        g.ffill(inplace=True)

        # 1. Returns
        g["return_1d"] = g["close"].pct_change()
        g["return_5d_lag1"] = g["close"].pct_change(5).shift(1)
        g["return_21d_lag1"]  = g["close"].pct_change(21).shift(1)
        g["return_63d_lag1"]  = g["close"].pct_change(63).shift(1)
        g["return_126d_lag1"] = g["close"].pct_change(126).shift(1)
        g["return_lag_1"] = g["return_1d"].shift(1)
        g["return_lag_2"] = g["return_1d"].shift(2)
        g["return_lag_3"] = g["return_1d"].shift(3)

        # 2. Moving averages
        g["ma_5"]  = g["close"].rolling(5).mean()
        g["ma_10"] = g["close"].rolling(10).mean()
        g["ma_20"] = g["close"].rolling(20).mean()
        g["price_vs_ma5"]  = g["close"] / g["ma_5"]  - 1
        g["price_vs_ma10"] = g["close"] / g["ma_10"] - 1
        g["price_vs_ma20"] = g["close"] / g["ma_20"] - 1

        # 3. MACD & momentum
        g["ema_12"]      = g["close"].ewm(span=12, adjust=False).mean()
        g["ema_26"]      = g["close"].ewm(span=26, adjust=False).mean()
        g["macd"]        = g["ema_12"] - g["ema_26"]
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"]   = g["macd"] - g["macd_signal"]
        g["momentum_3d"] = g["close"].pct_change(3)
        g["momentum_7d"] = g["close"].pct_change(7)
        g["roc_5"]       = g["close"].diff(5) / g["close"].shift(5)

        # 4. Bollinger %B
        rolling_std    = g["close"].rolling(20).std()
        lower_band     = g["ma_20"] - 2 * rolling_std
        band_width     = 4 * rolling_std   # distance from lower to upper band
        g["bollinger_b"] = (g["close"] - lower_band) / band_width.replace(0, np.nan)

        # 5. RSI & ATR
        g["rsi"]          = compute_rsi(g["close"])
        g["atr_14"]       = compute_atr(g)
        g["atr_pct_lag5"] = (g["atr_14"] / g["close"]).shift(5)
        # NOTE: atr_pct_vs_sector is cross-sectional — computed after the
        # groupby completes (see below), not here inside the per-symbol closure.

        # 6. Candlestick patterns
        body         = (g["close"] - g["open"]).abs()
        upper_shadow = g["high"]  - g[["close", "open"]].max(axis=1)
        lower_shadow = g[["close", "open"]].min(axis=1) - g["low"]
        total_range  = (g["high"] - g["low"]).replace(0, np.nan)

        g["is_hammer"] = (
            (lower_shadow > 2 * body) &
            (upper_shadow < 0.1 * body.replace(0, np.nan))
        ).astype(int)
        g["is_doji"] = (body < 0.1 * total_range).astype(int)

        # 7. Volume
        g["volume_change_1d"] = g["volume"].pct_change()
        g["volume_ma5"]       = g["volume"].rolling(5).mean()
        g["volume_vs_avg5"]   = g["volume"] / g["volume_ma5"].replace(0, np.nan) - 1
        # Relative volume vs 20-day average (stronger squeeze/breakout signal)
        g["volume_ma20"]      = g["volume"].rolling(20).mean()
        g["volume_vs_avg20"]  = g["volume"] / g["volume_ma20"].replace(0, np.nan) - 1

        # 8. Z-score
        rolling_mean_20 = g["close"].rolling(20).mean()
        rolling_std_20  = g["close"].rolling(20).std().replace(0, np.nan)
        g["zscore_20d"] = (g["close"] - rolling_mean_20) / rolling_std_20

        # 9. Rolling extremes — lagged by 1 so today's close is not in the window
        #    FIX: guard against zero denominator before dividing
        rolling_max_10 = g["close"].rolling(10).max().shift(1)
        g["rolling_max_10"]         = rolling_max_10
        g["price_vs_rolling_max10"] = (
            g["close"] / rolling_max_10.replace(0, np.nan) - 1
        )

        # 10. 52-week high proximity  (strictly lagged — uses close up to prior day)
        #     pct_from_52w_high = 0 means AT the high; negative means below it.
        #     Breakouts above zero (closing above prior-year max) are a documented
        #     momentum signal.
        rolling_52w_high = g["close"].shift(1).rolling(252, min_periods=20).max()
        g["rolling_52w_high"]     = rolling_52w_high
        g["pct_from_52w_high"]    = (
            g["close"] / rolling_52w_high.replace(0, np.nan) - 1
        )
        # Binary flag: today's close exceeds the lagged 52-week high (breakout)
        g["near_52w_high"] = (g["pct_from_52w_high"] >= -0.03).astype(int)
        g["at_52w_breakout"] = (g["pct_from_52w_high"] > 0.0).astype(int)

        return g

    print("Computing features per symbol...")
    # FIX: pandas 3.0 made DataFrameGroupBy.apply() always exclude the
    # grouping column from what's passed to (and returned by) the applied
    # function — the "operated on the grouping columns" deprecation from
    # 2.2 is the only behavior in 3.0. df.groupby("symbol").apply(...) would
    # then silently drop 'symbol' from the result on a pandas>=3.0 install,
    # breaking every `df["symbol"] == ...` filter and `groupby("date")` call
    # downstream that still needs it. Direct iteration is unaffected by that
    # change and yields the identical per-symbol frames .apply() used to, so
    # this is a version-safe swap, not a behavior change — same default sort
    # order (sort=True) as the .apply() call it replaces.
    df = pd.concat(
        [_compute_features_for_symbol(g) for _, g in df.groupby("symbol")],
        ignore_index=True,
    )

    df = df.dropna(subset=["close"])

    # ── Cross-sectional ATR feature (needs all symbols present in df) ─────────
    # This MUST live here, after the per-symbol groupby, so that atr_pct_lag5
    # exists for every symbol before we compute the sector median.
    if "atr_pct_lag5" in df.columns and "sector" in df.columns:
        sector_median_atr = df.groupby(["date", "sector"])["atr_pct_lag5"].transform("median")
        df["atr_pct_vs_sector"] = df["atr_pct_lag5"] - sector_median_atr

    # ── Fundamental / sentiment features (populated from stocks table columns) ─
    # These columns arrive pre-joined from the stocks table.  We transform them
    # here so the model sees normalised, model-ready signals.

    # Days-to-earnings: encode proximity as a categorical bucket and a numeric
    # signal clamped to [-7, 30].  Values < 0 mean post-earnings (days since).
    if "days_to_earnings" in df.columns:
        dte = pd.to_numeric(df["days_to_earnings"], errors="coerce")
        df["days_to_earnings_clamp"] = dte.clip(-7, 30)
        # Earnings week flag: within 5 calendar days before or 2 days after
        df["earnings_week"] = (
            dte.between(-2, 5, inclusive="both")
        ).astype(int)
        # Pre-earnings drift window (days 6–14 before)
        df["pre_earnings_drift"] = (
            dte.between(6, 14, inclusive="both")
        ).astype(int)

    # Short interest ratio (days-to-cover).  High short interest + price momentum
    # = squeeze setup.  Cap at 30 to mute outliers.
    if "short_ratio" in df.columns:
        sr = pd.to_numeric(df["short_ratio"], errors="coerce")
        df["short_ratio_clamp"] = sr.clip(0, 30)
        # High-short flag: top ~15 % of S&P 500 by days-to-cover
        df["high_short_interest"] = (sr >= 5).astype(int)

    # FIX (minor issue: "no size factor"): market_cap was collected daily by
    # daily_data_collector.get_fundamentals() (info.get("marketCap")) but
    # never made it into a model feature — size/liquidity is a well-
    # documented cross-sectional return factor and was simply absent.
    # Log-transformed because market cap spans several orders of magnitude
    # across the S&P 500; log keeps splits meaningful across that range and
    # keeps the training-median fallback used at inference for missing
    # features (see predict_single_stock.py) a sensible "medium-size company"
    # value rather than a distorted mean of a heavily right-skewed column.
    if "market_cap" in df.columns:
        mcap = pd.to_numeric(df["market_cap"], errors="coerce")
        df["log_market_cap"] = np.log(mcap.where(mcap > 0))

        # FIX (fundamentals gap): free_cashflow / net_income are collected
        # daily by get_fundamentals() (info.get("freeCashflow") /
        # "netIncomeToCommon") but were never referenced anywhere in this
        # file. Both are raw dollar figures spanning the same orders of
        # magnitude market_cap does above — feeding them in directly would
        # just be a second, noisier size proxy (same reasoning as
        # log_market_cap just above), so they're normalized into FCF yield
        # and earnings yield instead. The raw dollar columns themselves
        # (market_cap, free_cashflow, net_income) are excluded from the
        # model in train_model.py's _get_feature_columns() — they exist in
        # this dataframe only as inputs to the transforms below.
        mcap_positive = mcap.where(mcap > 0)
        if "free_cashflow" in df.columns:
            fcf = pd.to_numeric(df["free_cashflow"], errors="coerce")
            df["fcf_yield"] = (fcf / mcap_positive).clip(-1, 1)
        if "net_income" in df.columns:
            net_inc = pd.to_numeric(df["net_income"], errors="coerce")
            df["earnings_yield"] = (net_inc / mcap_positive).clip(-1, 1)

    # FIX (fundamentals gap): peg_ratio / earnings_growth / current_ratio are
    # collected daily by get_fundamentals() (info.get("pegRatio") /
    # "earningsGrowth" / "currentRatio") but were never referenced anywhere
    # in this file. Unlike free_cashflow/net_income above, all three are
    # already scale-free (a multiple, a growth rate, a ratio) so they're
    # used directly — just coerced to numeric and lightly clipped the same
    # way short_ratio_clamp is above. PEG in particular blows up toward
    # +/-inf as growth approaches zero, so the clip bound matters more here
    # than for the other two; adjust these bounds if they clip real signal
    # rather than just data glitches once you can see the distribution.
    if "peg_ratio" in df.columns:
        df["peg_ratio"] = pd.to_numeric(df["peg_ratio"], errors="coerce").clip(-15, 15)
    if "earnings_growth" in df.columns:
        df["earnings_growth"] = pd.to_numeric(df["earnings_growth"], errors="coerce").clip(-3, 5)
    if "current_ratio" in df.columns:
        df["current_ratio"] = pd.to_numeric(df["current_ratio"], errors="coerce").clip(0, 10)

    # NOTE: an "IV rank" feature (iv_rank / high_iv_rank) used to live here,
    # gated on `if "iv_rank" in df.columns`. daily_data_collector.get_fundamentals()
    # never actually populated that column (yfinance's `ticker.info` doesn't
    # expose options-implied-volatility rank — that requires pulling and
    # processing the options chain separately), so the column never existed
    # and this branch never ran. Removed rather than left as dead code that
    # silently muddies feature-importance audits. Re-add once a real IV-rank
    # source is wired into daily_data_collector.py.


    # ── Market-relative features (after per-symbol features exist) ───────────
    print("Computing market-relative features...")

    if "rsi" in df.columns:
        df["rs_relative"] = (
            df["rsi"] - df.groupby("date")["rsi"].transform("mean")
        )
    else:
        print("⚠️ Warning: 'rsi' column missing, skipping rs_relative.")

    if "volume_vs_avg5" in df.columns:
        df["volume_rel_strength"] = (
            df["volume_vs_avg5"]
            - df.groupby("date")["volume_vs_avg5"].transform("mean")
        )
    else:
        print("⚠️ Warning: 'volume_vs_avg5' column missing, skipping volume_rel_strength.")

    if "volume_vs_avg20" in df.columns:
        df["volume_rel_strength_20d"] = (
            df["volume_vs_avg20"]
            - df.groupby("date")["volume_vs_avg20"].transform("mean")
        )

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Cross-sectional features
# ══════════════════════════════════════════════════════════════════════════════

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features based on all stocks on a given date.

    FIX: relative_strength now uses a *leave-one-out* market return so a
    single stock's own return does not contaminate its own RS signal.
    For large universes (500 stocks) the bias is negligible, but this is
    the statistically correct formulation.
    """
    n_stocks = df.groupby("date")["return_1d"].transform("count")
    date_sum = df.groupby("date")["return_1d"].transform("sum")

    # Exclude self: (sum − self) / (n − 1)
    market_ret_excl_self = (date_sum - df["return_1d"]) / (n_stocks - 1).replace(0, np.nan)

    # For the shared 'market_ret' column (used for neutralisation) we keep the
    # simple mean — it's the canonical market return signal, not a feature per se.
    df["market_ret"] = df.groupby("date")["return_1d"].transform("mean")

    # Relative strength uses the leave-one-out version to avoid self-inclusion
    df["relative_strength"] = df["return_1d"] - market_ret_excl_self

    # Percentile ranks within each date cross-section
    for feature in ["return_1d", "rsi", "volatility_20d"]:
        if feature in df.columns:
            df[f"{feature}_rank"] = df.groupby("date")[feature].rank(pct=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Sector context features  (sector momentum · within-sector rank · sector RS)
# ══════════════════════════════════════════════════════════════════════════════

def add_sector_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three complementary sector-aware signal groups:

    1. Sector momentum  — how well is the sector doing?
       ``sector_ret_1d``        : equal-weight mean 1-day return of all stocks
                                  in the same sector on the same date.
       ``sector_ret_5d``        : same, but using the 5-day lagged return
                                  (``return_5d_lag1``) so there is zero
                                  look-ahead bias.
       ``sector_ret_21d``       : 21-day version.
       ``sector_momentum_trend``: 1 if sector_ret_5d > sector_ret_21d
                                  (short-term sector momentum expanding),
                                  else 0.

    2. Within-sector rank  — where does this stock stand among peers?
       ``sector_return_rank``   : percentile rank of ``return_1d`` within
                                  [date × sector] cross-section (0 = worst,
                                  1 = best).
       ``sector_rsi_rank``      : same for RSI.
       ``sector_momentum_rank`` : same for ``momentum_7d``.
       ``sector_volume_rank``   : same for ``volume_vs_avg20``.

    3. Sector-relative strength  — stock alpha vs its own sector:
       ``sector_relative_ret``  : stock's ``return_1d`` minus
                                  sector mean ``return_1d``
                                  (leave-one-out so self is excluded).
       ``sector_rs_5d``         : 5-day version using ``return_5d_lag1``.

    All features are strictly backward-looking so there is no look-ahead bias.

    Prerequisites
    -------------
    ``add_features()`` must have been called first so that ``return_1d``,
    ``return_5d_lag1``, ``return_21d_lag1``, ``rsi``, ``momentum_7d``,
    and ``volume_vs_avg20`` exist in ``df``.
    """
    required = {"symbol", "date", "sector", "return_1d"}
    missing  = required - set(df.columns)
    if missing:
        print(f"⚠️  add_sector_context_features: missing columns {missing}, skipping.")
        return df

    df = df.copy()
    df = df.sort_values(["date", "sector", "symbol"])

    # ── 1. Sector momentum ────────────────────────────────────────────────────
    # 1-day: mean of all stocks in the sector on that date
    sector_ret_1d = (
        df.groupby(["date", "sector"])["return_1d"].transform("mean")
    )
    df["sector_ret_1d"] = sector_ret_1d

    # 5-day and 21-day: use already-lagged columns to stay bias-free
    for col, out in [
        ("return_5d_lag1",  "sector_ret_5d"),
        ("return_21d_lag1", "sector_ret_21d"),
    ]:
        if col in df.columns:
            df[out] = df.groupby(["date", "sector"])[col].transform("mean")

    if "sector_ret_5d" in df.columns and "sector_ret_21d" in df.columns:
        df["sector_momentum_trend"] = (
            (df["sector_ret_5d"] > df["sector_ret_21d"]).astype(int)
        )

    # ── 2. Within-sector percentile ranks ─────────────────────────────────────
    rank_signals = {
        "return_1d":       "sector_return_rank",
        "rsi":             "sector_rsi_rank",
        "momentum_7d":     "sector_momentum_rank",
        "volume_vs_avg20": "sector_volume_rank",
    }
    for src_col, out_col in rank_signals.items():
        if src_col in df.columns:
            df[out_col] = (
                df.groupby(["date", "sector"])[src_col]
                .rank(pct=True, na_option="keep")
            )

    # ── 3. Sector-relative strength (leave-one-out) ───────────────────────────
    # Avoids self-inclusion: (sector_sum − self) / (n − 1)
    n_in_sector = df.groupby(["date", "sector"])["return_1d"].transform("count")
    sector_sum  = df.groupby(["date", "sector"])["return_1d"].transform("sum")
    sector_mean_excl_self = (
        (sector_sum - df["return_1d"]) / (n_in_sector - 1).replace(0, np.nan)
    )
    df["sector_relative_ret"] = df["return_1d"] - sector_mean_excl_self

    if "return_5d_lag1" in df.columns:
        n5   = df.groupby(["date", "sector"])["return_5d_lag1"].transform("count")
        sum5 = df.groupby(["date", "sector"])["return_5d_lag1"].transform("sum")
        sector_mean_5d_excl = (
            (sum5 - df["return_5d_lag1"]) / (n5 - 1).replace(0, np.nan)
        )
        df["sector_rs_5d"] = df["return_5d_lag1"] - sector_mean_5d_excl

    new_cols = [c for c in df.columns if c.startswith("sector_")]
    print(f"✅ Sector context features added: {new_cols}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Regime features
# ══════════════════════════════════════════════════════════════════════════════

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime indicators (bull / bear / volatile) with strict lagging."""
    print("Adding market regime features...")

    if "return_1d" not in df.columns:
        return df

    daily_market_ret = df.groupby("date")["return_1d"].mean()

    synthetic_index = (1 + daily_market_ret).cumprod() * 100

    market_sma_50  = synthetic_index.rolling(window=50,  min_periods=20).mean()
    market_sma_200 = synthetic_index.rolling(window=200, min_periods=50).mean()
    market_vol     = daily_market_ret.rolling(window=20, min_periods=10).std()
    vol_90th_pct   = market_vol.expanding(min_periods=252).quantile(0.9)

    # Strictly lagged by 1 day to prevent look-ahead bias
    is_bull     = ((synthetic_index > market_sma_200) & (market_sma_50 > market_sma_200)).astype(int).shift(1)
    is_bear     = ((synthetic_index < market_sma_200) & (market_sma_50 < market_sma_200)).astype(int).shift(1)
    is_volatile = (market_vol > vol_90th_pct).astype(int).shift(1)

    # FIX (idea #1 — "regime features computed as continuous, then thrown
    # away"): neutralize_features() strips the same-day market-return
    # correlation out of nearly every other feature (market_ret is excluded
    # from the final matrix), so these regime signals were the model's ONLY
    # non-neutralized window onto "what's the tape doing" — and that window
    # was three binary flags built from synthetic_index / market_sma_200 /
    # market_vol, with the underlying continuous values computed and then
    # discarded. The lagging discipline here is already correct (same
    # .shift(1) as the flags above), so keeping a couple of the continuous
    # versions too costs nothing new to get right — it's genuinely just
    # "don't throw this away," not new computation:
    #   market_trend_strength — signed distance from the 200-day trend
    #     (0 = right at trend; the bull/bear flags are this same quantity's
    #     sign, at a fixed 0 threshold, with no magnitude).
    #   market_realized_vol   — the rolling 20-day return stdev itself
    #     (is_volatile is this same quantity's position relative to its
    #     90th percentile, collapsed to 0/1, with no magnitude).
    # Left as NaN (not .fillna(0)) for the early rows before either rolling
    # window has enough history — consistent with how every other
    # continuous feature in add_features() is handled; clean_feature_matrix
    # median-fills these later, same as it does for everything else.
    trend_strength = (synthetic_index / market_sma_200 - 1).shift(1)
    realized_vol   = market_vol.shift(1)

    df["market_bull_regime"]     = df["date"].map(is_bull.to_dict()).fillna(0)
    df["market_bear_regime"]     = df["date"].map(is_bear.to_dict()).fillna(0)
    df["market_volatile_regime"] = df["date"].map(is_volatile.to_dict()).fillna(0)
    df["market_trend_strength"]  = df["date"].map(trend_strength.to_dict())
    df["market_realized_vol"]    = df["date"].map(realized_vol.to_dict())

    # Remove any stale regime columns from previous versions
    df = df.drop(
        columns=["market_vol_regime", "market_trend", "bear_market", "high_vol_regime"],
        errors="ignore",
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Feature neutralisation
# ══════════════════════════════════════════════════════════════════════════════

def neutralize_features(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "market_ret",
) -> pd.DataFrame:
    """
    Remove the linear influence of market return from each feature.
    Only neutralises features whose |correlation| with the target exceeds 0.1.
    """
    print(f"Neutralizing {len(feature_cols)} features against {target_col!r}...")

    for feat in feature_cols:
        if feat not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        corr = df[[feat, target_col]].corr().iloc[0, 1]
        if pd.isna(corr) or abs(corr) <= 0.1:
            continue
        market_var = df[target_col].var()
        if market_var <= 0:
            continue
        covariance = df[[feat, target_col]].cov().iloc[0, 1]
        beta       = covariance / market_var
        df[feat]   = df[feat] - beta * df[target_col]

    print("Feature neutralization complete.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Risk features
# ══════════════════════════════════════════════════════════════════════════════

def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds professional risk metrics: volatility-normalised return and
    rolling market beta.

    FIX: rolling beta now aligns per-symbol returns against a pre-computed
    daily market return Series (indexed by date) rather than against the
    full-frame positional Series, avoiding silent index misalignment.
    """
    print("Adding quantitative risk features...")
    df = df.sort_values(["symbol", "date"]).copy()

    # Volatility-normalised return
    if "atr_14" in df.columns:
        atr_pct = (df["atr_14"] / df["close"].replace(0, np.nan))
        df["vol_adj_return_1d"] = df["return_1d"] / atr_pct.replace(0, np.nan)

    # Daily market return indexed by date — computed once, then joined into groups
    daily_market_ret: pd.Series = df.groupby("date")["return_1d"].mean()

    def _compute_beta(g: pd.DataFrame) -> pd.Series:
        """
        20-day rolling beta for a single symbol.

        FIX: align the market return Series to this group's dates explicitly
        so that positional/index misalignment cannot occur.
        """
        mkt = g["date"].map(daily_market_ret)   # align market → group dates
        cov = g["return_1d"].rolling(20).cov(mkt)
        var = mkt.rolling(20).var()
        return (cov / var.replace(0, np.nan)).shift(1)   # lag to prevent leakage

    df["rolling_beta"] = (
        df.groupby("symbol", group_keys=False)
        .apply(_compute_beta)
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )

    return df

def label_triple_barrier_multiclass(
    df: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    look_ahead: int,
    col_name: str = "barrier_label",
) -> pd.DataFrame:
    """
    Assign a 3-class directional label to every row.
 
    Classes
    -------
    2  "TP"      – take-profit barrier hit first (or tied on the same candle)
    1  "EXPIRE"  – neither barrier hit within the window
    0  "SL"      – stop-loss barrier hit first
 
    The labeller walks each symbol's forward candles one day at a time so it
    can correctly determine *which* barrier was touched first when both are
    breached within the same window.  Daily-resolution OHLCV means we use
    the heuristic: if both H ≥ upper AND L ≤ lower on the same candle we
    call it a tie and assign TP (conservative: the intraday high is assumed
    to have been reached before the intraday low, which is the standard
    Lopez-de-Prado convention for bull signals).
 
    Parameters
    ----------
    df          : Must be sorted by [symbol, date].  Must contain
                  high, low, close columns.
    take_profit : Fractional profit target, e.g. 0.03 for 3 %.
    stop_loss   : Fractional stop-loss level,  e.g. 0.02 for 2 %.
    look_ahead  : Maximum number of forward candles to check.
    col_name    : Name of the output column.
 
    Returns
    -------
    df with a new integer column `col_name` ∈ {0, 1, 2}.
    The last `look_ahead` rows per symbol are NaN (incomplete window).
    """
    import numpy as np
    import pandas as pd
 
    df = df.sort_values(["symbol", "date"]).copy()
    labels = np.full(len(df), np.nan)

    # Precompute a map from DataFrame index value → positional offset in `labels`
    # so we never do an O(N) scan per row when writing results back.
    index_to_pos = {idx_val: pos for pos, idx_val in enumerate(df.index)}

    # Build a fast numpy view per symbol so we avoid repeated .loc calls
    for sym, grp in df.groupby("symbol", sort=False):
        idx   = grp.index.to_numpy()
        highs = grp["high"].to_numpy()
        lows  = grp["low"].to_numpy()
        closes = grp["close"].to_numpy()
        n     = len(idx)

        for i in range(n - look_ahead):
            entry   = closes[i]
            upper   = entry * (1.0 + take_profit)
            lower   = entry * (1.0 - stop_loss)
            outcome = 1  # default: expire

            for j in range(i + 1, min(i + look_ahead + 1, n)):
                hit_tp = highs[j] >= upper
                hit_sl = lows[j]  <= lower

                if hit_tp and hit_sl:
                    # Both on the same candle → TP wins (bull-bias convention)
                    outcome = 2
                    break
                elif hit_tp:
                    outcome = 2
                    break
                elif hit_sl:
                    outcome = 0
                    break

            labels[index_to_pos[idx[i]]] = outcome
 
    df[col_name] = labels
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Relative / excess-return triple-barrier (idea #2)
# ══════════════════════════════════════════════════════════════════════════════

def label_triple_barrier_relative(
    df: pd.DataFrame,
    take_profit: float,
    stop_loss: float,
    look_ahead: int,
    col_name: str = "barrier_label_rel",
) -> pd.DataFrame:
    """
    Relative/excess-return triple-barrier labelling: TP/SL fire when a
    stock's cumulative return SINCE ENTRY, minus the broad market's own
    concurrent cumulative return since entry, crosses +take_profit or
    -stop_loss — "did this stock beat (or lag) the market by X%", not "did
    this stock move X% in absolute terms" (that's label_triple_barrier_multiclass).

    Why this exists alongside the absolute labeller, not instead of it
    -------------------------------------------------------------------
    The absolute triple-barrier correctly matches how algo_index_manager
    actually exits positions (flat +25%/-7% moves off entry price) — that
    one is not being replaced. But it also means the model's only target
    never isolates stock-picking skill from broad market direction: during
    a broad rally almost every stock's absolute barriers look easy to
    clear; during a selloff a genuinely strong relative performer can still
    get labelled SL/EXPIRE purely because the tide went out. This is a
    genuinely different, complementary label meant to train an ADDITIONAL
    model alongside the absolute ones (see
    ImprovedModelTrainer.create_relative_target_variable), not replace them.

    No new lookahead: both the stock's path and the market's path are
    already-resolved history by the time you're labelling training data —
    comparing two already-known future paths to each other introduces
    nothing beyond what labelling from future price data at all already
    requires.

    Close-to-close only, deliberately
    ----------------------------------
    Unlike label_triple_barrier_multiclass, this does NOT use intraday
    high/low for tie-breaking. There's no principled way to compare one
    stock's intraday high against a "market's intraday high" aggregated
    across hundreds of different stocks' highs on the same day — the
    aggregate doesn't correspond to anything a real trade could touch. This
    trades away the absolute labeller's same-day tie-break precision for a
    comparison that's actually well-defined: the market side is a real,
    computable cross-sectional average close-to-close return.

    Parameters mirror label_triple_barrier_multiclass — take_profit /
    stop_loss / look_ahead are now excess-return thresholds, not absolute
    price-move thresholds.

    Returns
    -------
    df with a new integer column `col_name` ∈ {0, 1, 2} (SL / EXPIRE / TP,
    same convention as label_triple_barrier_multiclass). The last
    `look_ahead` rows per symbol are NaN (incomplete window).
    """
    df = df.sort_values(["symbol", "date"]).copy()

    # Market's own daily return: cross-sectional mean of every stock's own
    # close-to-close return on each date — the same construction
    # add_regime_features() uses for its synthetic_index, computed
    # independently here since this runs at target-creation time, before
    # add_features_to_data() (where add_regime_features lives) has run.
    daily_close_ret  = df.groupby("symbol")["close"].pct_change()
    ret_by_date      = pd.DataFrame({"date": df["date"], "ret": daily_close_ret})
    market_ret_by_date = ret_by_date.groupby("date")["ret"].mean().sort_index()
    # A cumulative index lets "market's return from day i to day j" be read
    # off as market_cum[j] / market_cum[i] - 1 for ANY i, j — needed since
    # the day-by-day walk below re-anchors to a new entry date at every i.
    market_cum_by_date = (1.0 + market_ret_by_date.fillna(0.0)).cumprod()
    date_to_cum = market_cum_by_date.to_dict()

    labels = np.full(len(df), np.nan)
    index_to_pos = {idx_val: pos for pos, idx_val in enumerate(df.index)}

    for sym, grp in df.groupby("symbol", sort=False):
        idx    = grp.index.to_numpy()
        closes = grp["close"].to_numpy()
        dates  = grp["date"].to_numpy()
        n      = len(idx)

        for i in range(n - look_ahead):
            entry_price = closes[i]
            entry_mkt   = date_to_cum.get(dates[i])
            if entry_price <= 0 or not entry_mkt or entry_mkt <= 0:
                continue
            outcome = 1  # default: expire

            for j in range(i + 1, min(i + look_ahead + 1, n)):
                mkt_j = date_to_cum.get(dates[j])
                if not mkt_j or mkt_j <= 0:
                    continue
                stock_cumret  = closes[j] / entry_price - 1.0
                market_cumret = mkt_j / entry_mkt - 1.0
                excess        = stock_cumret - market_cumret

                if excess >= take_profit:
                    outcome = 2
                    break
                elif excess <= -stop_loss:
                    outcome = 0
                    break

            labels[index_to_pos[idx[i]]] = outcome

    df[col_name] = labels
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Look-ahead bias validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_no_lookahead_bias(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "is_top_20p",
) -> dict:
    """
    Check for suspiciously high feature–target correlations that may indicate
    data leakage.  Returns a dict of {feature: |correlation|}.
    """
    print("\n=== VALIDATING FEATURES FOR LOOK-AHEAD BIAS ===")

    numeric_feature_cols = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        try:
            pd.to_numeric(df[feature], errors="raise")
            numeric_feature_cols.append(feature)
        except (ValueError, TypeError):
            print(f"  Skipping non-numeric feature: {feature}")

    if not numeric_feature_cols:
        print("Warning: No numeric features found for validation")
        return {}

    cols_to_check = [target_col] + numeric_feature_cols
    df_clean = df.dropna(subset=cols_to_check)

    if len(df_clean) == 0:
        print("Warning: No clean data for validation")
        return {}

    print(f"Validating {len(numeric_feature_cols)} numeric features on {len(df_clean):,} clean samples")

    correlations       = {}
    suspicious_features = []

    for feature in numeric_feature_cols:
        try:
            corr = df_clean[feature].corr(df_clean[target_col])
            correlations[feature] = abs(corr) if not pd.isna(corr) else 0
            if abs(corr) > 0.1:
                suspicious_features.append((feature, corr))
        except Exception as e:
            print(f"  Error calculating correlation for {feature}: {e}")
            correlations[feature] = 0

    sorted_corrs = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    print("Top 10 feature-target correlations:")
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


# ══════════════════════════════════════════════════════════════════════════════
# Convenience class
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Thin wrapper that holds a cached sector map and orchestrates the full
    feature-engineering pipeline.

    FIX: no longer calls add_sector_data() (which re-runs get_sector_mapping()
    and creates a *different* ordinal encoding) before add_features().
    Instead, encode_sector_column() is called once inside add_features(),
    which is the single source of truth for the sector encoding.
    """

    def __init__(self):
        self.sector_map = get_sector_mapping()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_features(df, sector_map=self.sector_map)
        df = add_cross_sectional_features(df)
        df = add_sector_context_features(df)   # sector momentum + rank + RS
        df = add_regime_features(df)
        return df