# patterns.py

import pandas as pd

def detect_patterns(df):
    df['Hammer'] = (df['Close'] > df['Open']) & ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))
    df['Doji'] = abs(df['Close'] - df['Open']) <= 0.1 * (df['High'] - df['Low'])
    return df
