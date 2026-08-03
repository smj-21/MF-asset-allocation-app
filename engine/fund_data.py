"""
Fund data pipeline:
  1. Curated 45-fund list (5 categories x ~10 funds), manually sourced from
     Sharpely (AUM + expense ratio as reference fields).
  2. Match each fund to its real mfapi.in scheme code, preferring the
     Direct Growth plan variant via fuzzy string matching.
  3. Fetch NAV history per scheme and compute 1Y/3Y return, volatility,
     and Sharpe ratio.

The full pipeline is expensive (network calls per fund), so the entry
point `fetch_all_fund_data` is cached by Streamlit.
"""

import time

import numpy as np
import pandas as pd
import requests
import streamlit as st

HEADERS = {"User-Agent": "Mozilla/5.0"}
CATEGORIES = ["largecap", "midcap", "flexicap", "smallcap", "gold"]

# ============================================================
# Curated fund data (45 funds across 5 categories)
# ============================================================
CURATED_DATA = {
    "largecap": [
        {"fund_name": "ICICI Prudential Large Cap Fund", "aum_cr": 79420.70, "return_1y": -0.21, "return_3y": 12.83, "return_5y": 13.79, "expense_ratio": 1.02},
        {"fund_name": "SBI Large Cap Fund", "aum_cr": 55064.00, "return_1y": 2.23, "return_3y": 10.26, "return_5y": 11.57, "expense_ratio": 0.85},
        {"fund_name": "Nippon India Large Cap Fund", "aum_cr": 53227.00, "return_1y": 0.86, "return_3y": 12.97, "return_5y": 15.85, "expense_ratio": 0.87},
        {"fund_name": "HDFC Large Cap Fund", "aum_cr": 39023.70, "return_1y": 1.01, "return_3y": 10.95, "return_5y": 13.39, "expense_ratio": 1.02},
        {"fund_name": "Mirae Asset Large Cap Fund", "aum_cr": 38379.30, "return_1y": 1.21, "return_3y": 10.26, "return_5y": 10.42, "expense_ratio": 0.75},
        {"fund_name": "Axis Large Cap Fund", "aum_cr": 30912.60, "return_1y": 1.13, "return_3y": 9.97, "return_5y": 8.36, "expense_ratio": 1.01},
        {"fund_name": "Aditya Birla Sun Life Large Cap Fund", "aum_cr": 29029.30, "return_1y": 0.15, "return_3y": 11.17, "return_5y": 11.57, "expense_ratio": 1.05},
        {"fund_name": "UTI Nifty 50 Index Fund", "aum_cr": 28685.10, "return_1y": -1.24, "return_3y": 8.40, "return_5y": 10.09, "expense_ratio": 0.25},
        {"fund_name": "HDFC Nifty 50 Index Fund", "aum_cr": 23703.00, "return_1y": -1.32, "return_3y": 8.34, "return_5y": 10.04, "expense_ratio": 0.30},
        {"fund_name": "ICICI Prudential Nifty 50 Index Fund", "aum_cr": 16839.30, "return_1y": -1.28, "return_3y": 8.35, "return_5y": 10.05, "expense_ratio": 0.25},
    ],
    "midcap": [
        {"fund_name": "HDFC Mid Cap Fund", "aum_cr": 100858.00, "return_1y": 9.34, "return_3y": 19.53, "return_5y": 20.47, "expense_ratio": 0.75},
        {"fund_name": "Kotak Midcap Fund", "aum_cr": 67611.00, "return_1y": 8.28, "return_3y": 19.86, "return_5y": 17.91, "expense_ratio": 0.51},
        {"fund_name": "Nippon India Growth Mid Cap Fund", "aum_cr": 49169.10, "return_1y": 10.56, "return_3y": 21.40, "return_5y": 19.80, "expense_ratio": 0.80},
        {"fund_name": "Motilal Oswal Midcap Fund", "aum_cr": 37473.90, "return_1y": -0.86, "return_3y": 20.33, "return_5y": 22.91, "expense_ratio": 0.95},
        {"fund_name": "Axis Midcap Fund", "aum_cr": 33803.00, "return_1y": 8.20, "return_3y": 17.73, "return_5y": 14.94, "expense_ratio": 0.82},
        {"fund_name": "SBI Midcap Fund", "aum_cr": 24126.70, "return_1y": 5.89, "return_3y": 12.86, "return_5y": 15.07, "expense_ratio": 1.01},
        {"fund_name": "DSP Midcap Fund", "aum_cr": 20170.30, "return_1y": 6.36, "return_3y": 16.52, "return_5y": 12.91, "expense_ratio": 0.81},
        {"fund_name": "Mirae Asset Midcap Fund", "aum_cr": 19542.60, "return_1y": 13.00, "return_3y": 18.35, "return_5y": 16.87, "expense_ratio": 0.90},
        {"fund_name": "Edelweiss Mid Cap Fund", "aum_cr": 17748.30, "return_1y": 10.57, "return_3y": 23.15, "return_5y": 19.61, "expense_ratio": 0.71},
        {"fund_name": "HSBC Midcap Fund", "aum_cr": 15351.90, "return_1y": 19.08, "return_3y": 24.48, "return_5y": 19.14, "expense_ratio": 1.22},
    ],
    "smallcap": [
        {"fund_name": "Nippon India Small Cap Fund", "aum_cr": 78407.00, "return_1y": 6.99, "return_3y": 16.74, "return_5y": 19.38, "expense_ratio": 0.70},
        {"fund_name": "HDFC Small Cap Fund", "aum_cr": 40416.80, "return_1y": 0.02, "return_3y": 12.64, "return_5y": 15.75, "expense_ratio": 0.76},
        {"fund_name": "SBI Small Cap Fund", "aum_cr": 40156.70, "return_1y": 3.70, "return_3y": 11.89, "return_5y": 14.20, "expense_ratio": 0.74},
        {"fund_name": "Quant Small Cap Fund", "aum_cr": 33739.00, "return_1y": 10.80, "return_3y": 18.52, "return_5y": 18.61, "expense_ratio": 0.95},
        {"fund_name": "Axis Small Cap Fund", "aum_cr": 29393.80, "return_1y": 7.81, "return_3y": 16.50, "return_5y": 16.80, "expense_ratio": 0.77},
        {"fund_name": "Bandhan Small Cap Fund", "aum_cr": 28466.20, "return_1y": 9.37, "return_3y": 27.03, "return_5y": 19.80, "expense_ratio": 0.58},
        {"fund_name": "DSP Small Cap Fund", "aum_cr": 19634.90, "return_1y": 11.39, "return_3y": 17.29, "return_5y": 17.55, "expense_ratio": 0.82},
        {"fund_name": "Kotak-Small Cap Fund", "aum_cr": 18690.80, "return_1y": 3.25, "return_3y": 13.07, "return_5y": 13.84, "expense_ratio": 0.63},
        {"fund_name": "HSBC Small Cap Fund", "aum_cr": 17830.20, "return_1y": 6.33, "return_3y": 15.29, "return_5y": 17.57, "expense_ratio": 0.77},
        {"fund_name": "Franklin India Small Cap Fund", "aum_cr": 14336.40, "return_1y": 3.00, "return_3y": 14.92, "return_5y": 17.02, "expense_ratio": 1.04},
    ],
    "flexicap": [
        {"fund_name": "Parag Parikh Flexi Cap Fund", "aum_cr": 143388.00, "return_1y": -0.96, "return_3y": 14.04, "return_5y": 13.40, "expense_ratio": 0.70},
        {"fund_name": "HDFC Flexi Cap Fund", "aum_cr": 106496.00, "return_1y": 5.02, "return_3y": 17.25, "return_5y": 18.76, "expense_ratio": 0.74},
        {"fund_name": "Kotak Flexicap Fund", "aum_cr": 55850.30, "return_1y": 1.45, "return_3y": 13.29, "return_5y": 12.45, "expense_ratio": 0.66},
        {"fund_name": "Aditya Birla Sun Life Flexi Cap Fund", "aum_cr": 26726.70, "return_1y": 10.29, "return_3y": 16.30, "return_5y": 13.44, "expense_ratio": 0.90},
        {"fund_name": "UTI - Flexi Cap Fund", "aum_cr": 22881.60, "return_1y": 1.10, "return_3y": 9.72, "return_5y": 7.03, "expense_ratio": 1.05},
        {"fund_name": "SBI Flexicap Fund", "aum_cr": 22685.10, "return_1y": 1.09, "return_3y": 8.94, "return_5y": 9.80, "expense_ratio": 1.27},
        {"fund_name": "ICICI Prudential Flexicap Fund", "aum_cr": 22506.90, "return_1y": 10.19, "return_3y": 17.56, "return_5y": 16.57, "expense_ratio": 0.90},
        {"fund_name": "Franklin India Flexi Cap Fund", "aum_cr": 19274.10, "return_1y": -1.05, "return_3y": 13.32, "return_5y": 14.09, "expense_ratio": 0.98},
        {"fund_name": "Axis Flexi Cap Fund", "aum_cr": 13327.80, "return_1y": 3.85, "return_3y": 13.78, "return_5y": 10.70, "expense_ratio": 0.92},
        {"fund_name": "Canara Robeco Flexicap Fund", "aum_cr": 13327.70, "return_1y": 1.87, "return_3y": 12.69, "return_5y": 11.91, "expense_ratio": 0.64},
    ],
    "gold": [
        {"fund_name": "Nippon India ETF Gold BeES", "aum_cr": 58000.00, "return_1y": None, "return_3y": None, "return_5y": None, "expense_ratio": None},
        {"fund_name": "ICICI Prudential Gold ETF", "aum_cr": 25900.00, "return_1y": None, "return_3y": None, "return_5y": 25.34, "expense_ratio": 0.50},
        {"fund_name": "HDFC Gold ETF", "aum_cr": 24534.20, "return_1y": None, "return_3y": None, "return_5y": None, "expense_ratio": 0.59},
        {"fund_name": "SBI Gold ETF", "aum_cr": 23579.50, "return_1y": None, "return_3y": None, "return_5y": None, "expense_ratio": 0.65},
        {"fund_name": "Axis Gold ETF", "aum_cr": 6500.00, "return_1y": None, "return_3y": None, "return_5y": None, "expense_ratio": 0.50},
    ],
}


def build_curated_df():
    rows = []
    for category, funds in CURATED_DATA.items():
        for fund in funds:
            fund_copy = fund.copy()
            fund_copy["category"] = category
            rows.append(fund_copy)
    return pd.DataFrame(rows)


# ============================================================
# mfapi.in scheme matching
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_scheme_list_with_retry(max_retries=3, wait_seconds=5):
    url = "https://api.mfapi.in/mf"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            return pd.DataFrame(response.json())
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
    raise Exception("Failed to fetch scheme master list after retries.")


def find_scheme_code(fund_name, scheme_list):
    """Fuzzy-match a curated fund name to its mfapi.in scheme code,
    preferring the Direct Growth plan and excluding IDCW/dividend variants."""
    matches = scheme_list[scheme_list["schemeName"].str.contains(fund_name, case=False, na=False)]
    if matches.empty:
        return None, None

    direct_growth = matches[
        matches["schemeName"].str.contains("direct", case=False, na=False)
        & matches["schemeName"].str.contains("growth", case=False, na=False)
        & ~matches["schemeName"].str.contains("idcw|dividend", case=False, na=False, regex=True)
    ]
    if not direct_growth.empty:
        return direct_growth.iloc[0]["schemeCode"], direct_growth.iloc[0]["schemeName"]
    return matches.iloc[0]["schemeCode"], matches.iloc[0]["schemeName"]


# ============================================================
# NAV history + per-fund metrics
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nav_with_retry(scheme_code, max_retries=3, wait_seconds=5):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            nav_data = response.json()
            nav_df = pd.DataFrame(nav_data["data"])
            nav_df["date"] = pd.to_datetime(nav_df["date"], format="%d-%m-%Y")
            nav_df["nav"] = nav_df["nav"].astype(float)
            nav_df = nav_df.sort_values("date").reset_index(drop=True)
            time.sleep(0.5)  # gentle rate-limiting; only runs on a real fetch, not a cache hit
            return nav_df
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(wait_seconds)
    return None


def compute_fund_metrics(nav_df, risk_free_rate=0.065):
    nav_df = nav_df[nav_df["nav"] > 0].copy()
    nav_df = nav_df.set_index("date")

    latest_date = nav_df.index.max()
    latest_nav = nav_df["nav"].iloc[-1]

    one_yr_ago = latest_date - pd.DateOffset(years=1)
    three_yr_ago = latest_date - pd.DateOffset(years=3)

    nav_1yr = nav_df.loc[:one_yr_ago]["nav"].iloc[-1]
    nav_3yr = nav_df.loc[:three_yr_ago]["nav"].iloc[-1]

    return_1y = (latest_nav / nav_1yr - 1) * 100
    return_3y = ((latest_nav / nav_3yr) ** (1 / 3) - 1) * 100

    daily_returns = nav_df["nav"].pct_change().dropna()
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    volatility_val = daily_returns.std() * (252 ** 0.5) * 100

    sharpe = (return_1y - risk_free_rate * 100) / volatility_val

    return {
        "return_1y_calc": round(return_1y, 2),
        "return_3y_calc": round(return_3y, 2),
        "volatility": round(volatility_val, 2),
        "sharpe_ratio": round(sharpe, 2),
    }


# ============================================================
# Full pipeline entry point (cached)
# ============================================================
def fetch_all_fund_data(_progress_callback=None):
    """
    Runs the full fund data pipeline: curated list -> scheme matching ->
    NAV fetch -> metrics -> merged final_df.

    This function itself is NOT cached (it drives UI progress updates via
    _progress_callback, which Streamlit's cache can't safely replay). The
    expensive network calls underneath (fetch_scheme_list_with_retry,
    fetch_nav_with_retry) ARE individually cached, so repeat calls across
    reruns/users are still fast -- this just re-loops over already-cached
    results almost instantly.

    _progress_callback(done, total) is called after each fund if provided.
    """
    curated_df = build_curated_df()

    scheme_list = fetch_scheme_list_with_retry()
    curated_df["scheme_code"], curated_df["matched_name"] = zip(
        *curated_df["fund_name"].apply(lambda name: find_scheme_code(name, scheme_list))
    )

    total = len(curated_df)
    all_metrics = []
    for i, (idx, row) in enumerate(curated_df.iterrows(), start=1):
        scheme_code = int(row["scheme_code"])
        nav_df = fetch_nav_with_retry(scheme_code)
        if nav_df is not None:
            metrics = compute_fund_metrics(nav_df)
            metrics["scheme_code"] = scheme_code
            all_metrics.append(metrics)
        if _progress_callback is not None:
            _progress_callback(i, total)

    metrics_df = pd.DataFrame(all_metrics)
    final_df = curated_df.merge(metrics_df, on="scheme_code")

    return final_df
