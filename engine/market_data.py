"""
Market data fetching.

Pulls 1-year weekly closing prices for the four equity indices + gold ETF,
converts to weekly returns, and appends a synthetic Bank FD 'category' with
a fixed annual return and zero volatility.
"""

import yfinance as yf
import streamlit as st

TICKERS = {
    "largecap": "^CNX100",
    "midcap": "NIFTYMIDCAP150.NS",
    "flexicap": "^CRSLDX",
    "smallcap": "NIFTYSMLCAP250.NS",
    "gold": "GOLDBEES.NS",
}

FD_ANNUAL_RETURN = 0.065


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data():
    """
    Returns:
        returns (pd.DataFrame): weekly % returns for all 6 categories
                                 (largecap, midcap, flexicap, smallcap, gold, fd)
        volatility (pd.Series): annualized volatility per category
    """
    data = yf.download(
        list(TICKERS.values()), period="1y", interval="1wk", auto_adjust=False
    )["Close"]
    data = data.rename(columns={v: k for k, v in TICKERS.items()})

    returns = data.pct_change(fill_method=None).dropna()

    fd_weekly_return = (1 + FD_ANNUAL_RETURN) ** (1 / 52) - 1
    returns["fd"] = fd_weekly_return

    volatility = returns.std() * (52 ** 0.5)

    return returns, volatility
