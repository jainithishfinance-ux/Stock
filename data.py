"""
data.py — Real-Time Stock Data Fetcher
StockVision Pro | Knowledge Institute of Technology
Fetches live & historical data from Yahoo Finance (yfinance)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

# ─────────────────────────────────────────────
# STOCK UNIVERSE
# ─────────────────────────────────────────────
INDIAN_STOCKS = {
    "TCS":       {"name": "Tata Consultancy Services", "sector": "IT"},
    "INFY":      {"name": "Infosys Ltd",                "sector": "IT"},
    "WIPRO":     {"name": "Wipro Ltd",                  "sector": "IT"},
    "HCLTECH":   {"name": "HCL Technologies",           "sector": "IT"},
    "TECHM":     {"name": "Tech Mahindra",              "sector": "IT"},
    "RELIANCE":  {"name": "Reliance Industries",        "sector": "Energy"},
    "ONGC":      {"name": "ONGC Ltd",                   "sector": "Energy"},
    "NTPC":      {"name": "NTPC Ltd",                   "sector": "Energy"},
    "HDFCBANK":  {"name": "HDFC Bank",                  "sector": "Finance"},
    "ICICIBANK": {"name": "ICICI Bank",                 "sector": "Finance"},
    "SBIN":      {"name": "State Bank of India",        "sector": "Finance"},
    "BAJFINANCE":{"name": "Bajaj Finance",              "sector": "Finance"},
    "HINDUNILVR":{"name": "Hindustan Unilever",         "sector": "FMCG"},
    "ITC":       {"name": "ITC Ltd",                    "sector": "FMCG"},
    "SUNPHARMA": {"name": "Sun Pharmaceutical",         "sector": "Pharma"},
}

US_STOCKS = {
    "AAPL":  {"name": "Apple Inc",          "sector": "Tech"},
    "MSFT":  {"name": "Microsoft Corp",     "sector": "Tech"},
    "GOOGL": {"name": "Alphabet Inc",       "sector": "Tech"},
    "AMZN":  {"name": "Amazon.com Inc",     "sector": "Tech"},
    "TSLA":  {"name": "Tesla Inc",          "sector": "EV"},
    "NVDA":  {"name": "NVIDIA Corp",        "sector": "Semiconductors"},
    "META":  {"name": "Meta Platforms",     "sector": "Tech"},
    "NFLX":  {"name": "Netflix Inc",        "sector": "Tech"},
}

def _nse_ticker(sym):
    """Append .NS for Indian stocks."""
    return sym + ".NS"

def get_live_quote(symbol, is_indian=True):
    """Fetch live quote for a single stock."""
    try:
        ticker_sym = _nse_ticker(symbol) if is_indian else symbol
        ticker = yf.Ticker(ticker_sym)
        info = ticker.fast_info
        hist = ticker.history(period="2d", interval="1d")

        if hist.empty:
            return None

        current = float(hist["Close"].iloc[-1])
        prev    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
        change  = current - prev
        pct     = (change / prev * 100) if prev else 0

        return {
            "sym":     symbol,
            "name":    (INDIAN_STOCKS if is_indian else US_STOCKS).get(symbol, {}).get("name", symbol),
            "sector":  (INDIAN_STOCKS if is_indian else US_STOCKS).get(symbol, {}).get("sector", ""),
            "price":   round(current, 2),
            "change":  round(change, 2),
            "pct":     round(pct, 2),
            "open":    round(float(hist["Open"].iloc[-1]), 2),
            "high":    round(float(hist["High"].iloc[-1]), 2),
            "low":     round(float(hist["Low"].iloc[-1]), 2),
            "volume":  int(hist["Volume"].iloc[-1]),
            "prev_close": round(prev, 2),
            "market_cap": getattr(info, "market_cap", 0) or 0,
            "pe":      round(getattr(info, "price_to_book", 0) or 0, 2),
            "currency": "INR" if is_indian else "USD",
        }
    except Exception as e:
        print(f"[data] Error fetching {symbol}: {e}")
        return None


def get_all_live_quotes():
    """Fetch live quotes for all stocks in the universe."""
    results = []

    for sym in INDIAN_STOCKS:
        q = get_live_quote(sym, is_indian=True)
        if q:
            results.append(q)

    for sym in US_STOCKS:
        q = get_live_quote(sym, is_indian=False)
        if q:
            results.append(q)

    return results


def get_historical_data(symbol, period="1y", interval="1d", is_indian=True):
    """
    Fetch OHLCV historical data + computed technical indicators.

    Returns a pandas DataFrame with columns:
    Open, High, Low, Close, Volume,
    MA_20, MA_50, RSI, MACD, Signal_Line,
    BB_Upper, BB_Lower, Daily_Return, Volatility
    """
    ticker_sym = _nse_ticker(symbol) if is_indian else symbol
    try:
        df = yf.download(ticker_sym, period=period, interval=interval, progress=False)
        if df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)

        # ── Moving Averages ──────────────────────────────────────────────
        df["MA_20"]  = df["Close"].rolling(window=20).mean()
        df["MA_50"]  = df["Close"].rolling(window=50).mean()
        df["MA_200"] = df["Close"].rolling(window=200).mean()

        # ── RSI (14-period) ──────────────────────────────────────────────
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ── MACD (12, 26, 9) ────────────────────────────────────────────
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["Signal_Line"]

        # ── Bollinger Bands (20-period, 2σ) ─────────────────────────────
        rolling_std  = df["Close"].rolling(window=20).std()
        df["BB_Mid"]   = df["MA_20"]
        df["BB_Upper"] = df["MA_20"] + 2 * rolling_std
        df["BB_Lower"] = df["MA_20"] - 2 * rolling_std

        # ── Returns & Volatility ─────────────────────────────────────────
        df["Daily_Return"] = df["Close"].pct_change()
        df["Volatility"]   = df["Daily_Return"].rolling(20).std() * np.sqrt(252)

        # ── Binary Target (1 = next day up, 0 = down) ───────────────────
        df["Target_Binary"]     = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df["Target_Continuous"] = df["Close"].shift(-1)

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    except Exception as e:
        print(f"[data] Historical error {symbol}: {e}")
        return None


def get_sector_performance():
    """Return % daily change per sector (Indian markets)."""
    sector_map = {}
    for sym, meta in INDIAN_STOCKS.items():
        sector = meta["sector"]
        q = get_live_quote(sym, is_indian=True)
        if q:
            sector_map.setdefault(sector, []).append(q["pct"])

    return {sector: round(np.mean(vals), 2) for sector, vals in sector_map.items()}


def get_market_news():
    """
    Simple market news via yfinance news API.
    Falls back to curated static headlines if API unavailable.
    """
    news_items = []
    try:
        ticker = yf.Ticker("^NSEI")   # Nifty 50
        raw_news = ticker.news or []
        for item in raw_news[:10]:
            news_items.append({
                "title":     item.get("title", ""),
                "source":    item.get("publisher", "Yahoo Finance"),
                "link":      item.get("link", "#"),
                "time":      datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%b %d, %H:%M"),
                "sentiment": "neutral",   # sentiment filled by model.py
            })
    except Exception:
        pass

    # Fallback / augment with static curated news
    if len(news_items) < 5:
        news_items += [
            {"title": "RBI keeps repo rate unchanged at 6.5% amid inflation concerns",
             "source": "Economic Times", "link": "#", "time": "Today",
             "sentiment": "neutral"},
            {"title": "TCS Q4 results beat estimates; revenue up 8.2% YoY",
             "source": "Moneycontrol",  "link": "#", "time": "Today",
             "sentiment": "positive"},
            {"title": "Reliance Industries to invest ₹75,000 Cr in green energy",
             "source": "Business Standard", "link": "#", "time": "Today",
             "sentiment": "positive"},
            {"title": "HDFC Bank NPA rises slightly; management guides caution",
             "source": "Mint", "link": "#", "time": "Today",
             "sentiment": "negative"},
            {"title": "Wipro wins $300M AI transformation deal from European bank",
             "source": "Financial Express", "link": "#", "time": "Today",
             "sentiment": "positive"},
        ]

    return news_items[:10]


def prepare_features(df):
    """
    Extract feature matrix X and targets y from a historical DataFrame.
    Returns (X, y_binary, y_continuous, feature_names)
    """
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_20", "MA_50", "RSI",
        "MACD", "Signal_Line", "MACD_Hist",
        "BB_Upper", "BB_Lower",
        "Daily_Return", "Volatility",
    ]
    df_clean = df.dropna(subset=feature_cols + ["Target_Binary", "Target_Continuous"])
    X          = df_clean[feature_cols].values
    y_binary   = df_clean["Target_Binary"].values
    y_cont     = df_clean["Target_Continuous"].values
    dates      = df_clean.index.tolist()
    return X, y_binary, y_cont, feature_cols, dates
