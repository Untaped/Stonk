from sqlalchemy import Column, Integer, String, Float, Date, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime


Base = declarative_base()

class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    date = Column(Date, default=datetime.date.today)
    price = Column(Float)
    market_cap = Column(Float)
    average_volume = Column(Float)
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    net_income = Column(Float)
    roe = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    free_cashflow = Column(Float)
    forward_pe = Column(Float)
    peg_ratio = Column(Float)
    score = Column(Float)
    recommendation = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    hammer = Column(Boolean, default=False)
    doji = Column(Boolean, default=False)


class PriceHistory(Base):
    __tablename__ = "price_history"
    __table_args__ = {'extend_existing': True}  # 👈 Add this line

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    close_price = Column(Float)  # if you’re still using this field
    hammer = Column(Boolean, default=False)
    doji = Column(Boolean, default=False)

# Database setup (only run once to create tables)
DATABASE_URL = "sqlite:///stocks.db"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)