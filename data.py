import yfinance as yf
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(tickers, start_date, end_date):
    if isinstance(tickers, str):
        tickers = [t.strip().upper() for t in tickers.split(',')]
    
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    data.dropna(how='all', inplace=True)
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    
    returns = data.pct_change().dropna()
    return data, returns

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark(start_date, end_date, ticker="^NSEI"):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)['Close']
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    returns = data.pct_change().dropna()
    return returns
