import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# NIFTY 50 Tickers on Yahoo Finance (approximated main liquid stocks for the app)
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "TATASTEEL.NS", "TATAMOTORS.NS", "ASIANPAINT.NS", "ULTRACEMCO.NS", "HCLTECH.NS",
    "M&M.NS", "WIPRO.NS", "POWERGRID.NS", "NTPC.NS", "BAJAJFINSV.NS",
    "KOTAKBANK.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "ONGC.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "BRITANNIA.NS",
    "NESTLEIND.NS", "GRASIM.NS", "HINDALCO.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "CIPLA.NS", "TECHM.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS", "TATACOMM.NS",
    "TATACONSUM.NS", "UPL.NS", "BPCL.NS", "SHREECEM.NS", "BAJAJ-AUTO.NS"
]

def fetch_data(tickers, start_date, end_date=None):
    """
    Fetches historical daily adjusted close prices for the requested tickers.
    """
    if not tickers:
        return pd.DataFrame()
    
    if end_date is None:
        end_date = datetime.date.today()
        
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    else:
        data = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    
    # If a single ticker is passed, yfinance might return a Series. Convert to DataFrame.
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
    
    # Forward fill missing data, then drop any rows that are still purely NaN
    data = data.ffill().dropna()
    return data

def calculate_returns(price_data):
    """
    Calculates daily percentage returns for the historical price data.
    """
    return price_data.pct_change().dropna()

def calculate_portfolio_returns(returns, weights):
    """
    Calculates the historical returns of the portfolio given an array/list of weights.
    """
    # Ensure weights sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Weighted sum of returns across all assets for each day
    portfolio_daily_returns = returns.dot(weights)
    return portfolio_daily_returns
