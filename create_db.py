from sqlalchemy import create_engine, Column, Integer, String, Float, Date, Boolean, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Stock(Base):
    __tablename__ = 'stocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    name = Column(String)
    date = Column(Date)
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
    volume = Column(Integer)   # << Make sure this is here
    hammer = Column(Boolean)
    doji = Column(Boolean)
    notes = Column(Text, nullable=True)

engine = create_engine('sqlite:///stocks.db')
Base.metadata.create_all(engine)

print("Database and table created!")