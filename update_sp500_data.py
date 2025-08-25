import yfinance as yf
import pandas as pd
from datetime import datetime
from db import SessionLocal, Stock, PriceHistory  # Adjust these imports to match your schema

def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table['Symbol'].tolist()

def update_data():
    session = SessionLocal()
    symbols = get_sp500_symbols()
    today = datetime.now().date()

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d")

            if history.empty:
                continue

            close_price = history['Close'][-1]

            # Extract extra metrics
            dividend_yield = info.get("dividendYield", 0.0)
            pe_ratio = info.get("trailingPE", 0.0)
            market_cap = info.get("marketCap", 0)
            beta = info.get("beta", 0.0)
            name = info.get("shortName", symbol)

            # Update stock metadata
            stock = session.query(Stock).filter(Stock.symbol == symbol).first()
            if not stock:
                stock = Stock(symbol=symbol)
            stock.name = name
            stock.dividend_yield = dividend_yield
            stock.pe_ratio = pe_ratio
            stock.market_cap = market_cap
            stock.beta = beta
            session.merge(stock)

            # Save today's price
            price_entry = PriceHistory(
                symbol=symbol,
                date=today,
                close=close_price
            )
            session.merge(price_entry)
            session.commit()

            print(f"✅ {symbol}: ${close_price:.2f} | Div: {dividend_yield} | PE: {pe_ratio}")
        except Exception as e:
            print(f"❌ Error with {symbol}: {e}")
            continue

    session.close()

if __name__ == "__main__":
    update_data()
