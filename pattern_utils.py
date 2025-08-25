def detect_patterns(df):
    """
    Add simple candlestick pattern detections.
    Currently supports Hammer and Doji. Extendable.
    """
    df = df.copy()

    # Hammer pattern logic
    df['Hammer'] = (
        (df['Close'] > df['Open']) &
        ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))
    )

    # Doji pattern logic
    df['Doji'] = abs(df['Close'] - df['Open']) <= 0.1 * (df['High'] - df['Low'])

    return df