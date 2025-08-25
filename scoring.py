def score_stock(growth, dividend_yield, pe_ratio):
    """Simple scoring based on weights."""
    score = (
        growth * 0.5 +
        dividend_yield * 0.3 +
        ((100 / pe_ratio) * 0.2 if pe_ratio else 0)
    )

    return round(score, 2)