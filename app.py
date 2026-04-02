"""
app.py — Flask API Backend
StockVision Pro | Knowledge Institute of Technology
Connects React/HTML frontend ↔ Python ML engine
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import traceback
import threading
import time
import os

from data import (
    get_all_live_quotes,
    get_historical_data,
    get_sector_performance,
    get_market_news,
    prepare_features,
    INDIAN_STOCKS,
    US_STOCKS,
)
from model import (
    run_prediction,
    get_all_model_accuracy,
    simple_sentiment,
)

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")   # Allow all origins for local dev / Firebase hosting

# ── In-memory cache (refreshed every 60 s) ───────────────────────────────────
_cache = {
    "quotes":  {"data": [], "ts": 0},
    "sector":  {"data": {}, "ts": 0},
    "news":    {"data": [], "ts": 0},
}
CACHE_TTL = 60   # seconds

def _is_stale(key):
    return time.time() - _cache[key]["ts"] > CACHE_TTL

def _refresh_quotes():
    data = get_all_live_quotes()
    _cache["quotes"] = {"data": data, "ts": time.time()}
    return data

def _refresh_sector():
    data = get_sector_performance()
    _cache["sector"] = {"data": data, "ts": time.time()}
    return data

def _refresh_news():
    items = get_market_news()
    # Enrich with sentiment
    for item in items:
        if item.get("sentiment", "neutral") == "neutral":
            item["sentiment"] = simple_sentiment(item.get("title", ""))
    _cache["news"] = {"data": items, "ts": time.time()}
    return items

# Background refresh thread
def _background_refresh():
    while True:
        try:
            _refresh_quotes()
            _refresh_sector()
        except Exception as e:
            print(f"[bg] refresh error: {e}")
        time.sleep(60)

# Background refresh thread (skip in serverless environments like Vercel)
if not os.environ.get("VERCEL"):
    threading.Thread(target=_background_refresh, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ok(data, **kwargs):
    return jsonify({"status": "ok", "data": data, **kwargs})

def _err(msg, code=400):
    return jsonify({"status": "error", "message": msg}), code

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "app": "StockVision Pro API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /api/stocks",
            "GET  /api/stocks/<symbol>",
            "GET  /api/history/<symbol>",
            "POST /api/predict",
            "GET  /api/sector",
            "GET  /api/news",
            "GET  /api/models/accuracy",
            "GET  /api/heatmap",
            "POST /api/screener",
            "GET  /api/compare?a=SYM&b=SYM",
        ]
    })


# ── 1. Live Quotes ─────────────────────────────────────────────────────────
@app.route("/api/stocks")
def get_stocks():
    """Return live quotes for all stocks."""
    try:
        if _is_stale("quotes") or not _cache["quotes"]["data"]:
            data = _refresh_quotes()
        else:
            data = _cache["quotes"]["data"]
        return _ok(data)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


@app.route("/api/stocks/<symbol>")
def get_stock(symbol):
    """Return live quote for a single stock."""
    try:
        sym = symbol.upper()
        is_indian = sym in INDIAN_STOCKS
        if not is_indian and sym not in US_STOCKS:
            return _err(f"Unknown symbol: {sym}", 404)

        from data import get_live_quote
        q = get_live_quote(sym, is_indian=is_indian)
        if not q:
            return _err(f"Could not fetch data for {sym}", 500)
        return _ok(q)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 2. Historical Data ─────────────────────────────────────────────────────
@app.route("/api/history/<symbol>")
def get_history(symbol):
    """
    Return OHLCV + technical indicators for <symbol>.
    Query params:
      period   = 1mo | 3mo | 6mo | 1y | 2y | 5y  (default: 1y)
      interval = 1d | 1wk | 1mo                   (default: 1d)
    """
    try:
        sym      = symbol.upper()
        period   = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        is_indian = sym in INDIAN_STOCKS

        df = get_historical_data(sym, period=period, interval=interval,
                                 is_indian=is_indian)
        if df is None or df.empty:
            return _err(f"No historical data for {sym}", 404)

        result = {
            "symbol":  sym,
            "period":  period,
            "interval": interval,
            "count":   len(df),
            "dates":   [str(d)[:10] for d in df.index],
            "open":    df["Open"].round(2).tolist(),
            "high":    df["High"].round(2).tolist(),
            "low":     df["Low"].round(2).tolist(),
            "close":   df["Close"].round(2).tolist(),
            "volume":  df["Volume"].astype(int).tolist(),
            "ma20":    df["MA_20"].round(2).tolist(),
            "ma50":    df["MA_50"].round(2).tolist(),
            "rsi":     df["RSI"].round(2).tolist(),
            "macd":    df["MACD"].round(4).tolist(),
            "macd_signal": df["Signal_Line"].round(4).tolist(),
            "bb_upper": df["BB_Upper"].round(2).tolist(),
            "bb_lower": df["BB_Lower"].round(2).tolist(),
            "volatility": df["Volatility"].round(4).tolist(),
        }
        return _ok(result)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 3. ML Prediction ──────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run ML prediction for a symbol with a chosen model.

    Body (JSON):
    {
      "symbol": "TCS",
      "model":  "lstm" | "xgb" | "rf" | "lr" | "svm",
      "period": "1y"   (optional, default 1y)
    }
    """
    try:
        body      = request.get_json(force=True) or {}
        symbol    = body.get("symbol", "TCS").upper()
        model_key = body.get("model", "rf").lower()
        period    = body.get("period", "1y")

        is_indian = symbol in INDIAN_STOCKS

        df = get_historical_data(symbol, period=period, is_indian=is_indian)
        if df is None or df.empty:
            return _err(f"No data available for {symbol}", 404)

        X, y_binary, y_cont, feature_names, dates = prepare_features(df)
        if len(X) < 100:
            return _err("Insufficient historical data for training (< 100 rows)", 400)

        current_price = float(df["Close"].iloc[-1])
        result = run_prediction(
            symbol, model_key, X, y_binary, y_cont,
            feature_names, dates, current_price
        )
        return _ok(result)

    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 4. Sector Performance ─────────────────────────────────────────────────
@app.route("/api/sector")
def sector():
    try:
        if _is_stale("sector"):
            data = _refresh_sector()
        else:
            data = _cache["sector"]["data"]
        return _ok(data)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 5. Market News + Sentiment ────────────────────────────────────────────
@app.route("/api/news")
def news():
    try:
        if _is_stale("news"):
            data = _refresh_news()
        else:
            data = _cache["news"]["data"]
        return _ok(data)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 6. Model Accuracy Benchmarks ──────────────────────────────────────────
@app.route("/api/models/accuracy")
def model_accuracy():
    return _ok(get_all_model_accuracy())


# ── 7. Heatmap Data ────────────────────────────────────────────────────────
@app.route("/api/heatmap")
def heatmap():
    """Return % change for every stock (for heatmap rendering)."""
    try:
        if _is_stale("quotes") or not _cache["quotes"]["data"]:
            data = _refresh_quotes()
        else:
            data = _cache["quotes"]["data"]

        hm = [{"sym": q["sym"], "name": q["name"], "pct": q["pct"],
                "price": q["price"], "sector": q["sector"]}
              for q in data]
        return _ok(hm)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ── 8. Stock Screener ─────────────────────────────────────────────────────
@app.route("/api/screener", methods=["POST"])
def screener():
    """
    Filter stocks by criteria.

    Body (JSON):
    {
      "min_pe":    0,
      "max_pe":    50,
      "min_change": -5,
      "sector":   "IT" | "" (empty = all),
      "signal":   "buy" | "sell" | "hold" | "" (empty = all)
    }
    """
    try:
        body      = request.get_json(force=True) or {}
        min_pe    = float(body.get("min_pe", 0))
        max_pe    = float(body.get("max_pe", 999))
        min_chg   = float(body.get("min_change", -100))
        sector    = body.get("sector", "").lower()
        signal    = body.get("signal", "").lower()

        if _is_stale("quotes") or not _cache["quotes"]["data"]:
            data = _refresh_quotes()
        else:
            data = _cache["quotes"]["data"]

        results = []
        for q in data:
            pe  = q.get("pe", 0) or 0
            pct = q.get("pct", 0) or 0
            sec = q.get("sector", "").lower()

            if pe < min_pe or pe > max_pe:         continue
            if pct < min_chg:                      continue
            if sector and sector not in sec:        continue

            # Derive simple signal from RSI/MA logic (fast heuristic)
            sig = _derive_signal(q)
            if signal and signal != sig:           continue

            results.append({**q, "signal": sig})

        return _ok(results)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


def _derive_signal(quote):
    """Simple buy/sell/hold heuristic based on daily change."""
    pct = quote.get("pct", 0)
    if pct > 1.5:   return "buy"
    if pct < -1.5:  return "sell"
    return "hold"


# ── 9. Stock Comparison ────────────────────────────────────────────────────
@app.route("/api/compare")
def compare():
    """
    Compare two stocks side by side.
    Query params: a=SYM&b=SYM&period=1y
    """
    try:
        sym_a  = request.args.get("a", "TCS").upper()
        sym_b  = request.args.get("b", "INFY").upper()
        period = request.args.get("period", "6mo")

        results = {}
        for sym in [sym_a, sym_b]:
            is_indian = sym in INDIAN_STOCKS
            df = get_historical_data(sym, period=period, is_indian=is_indian)
            if df is not None and not df.empty:
                results[sym] = {
                    "dates":  [str(d)[:10] for d in df.index],
                    "close":  df["Close"].round(2).tolist(),
                    "ma20":   df["MA_20"].round(2).tolist(),
                    "rsi":    df["RSI"].round(2).tolist(),
                    "normalized": (df["Close"] / float(df["Close"].iloc[0]) * 100).round(2).tolist(),
                }
            else:
                results[sym] = {}

        return _ok(results)
    except Exception as e:
        traceback.print_exc()
        return _err(str(e), 500)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  StockVision Pro — Flask API")
    print("  http://localhost:5000")
    print("=" * 60)
    # Initial warm-up (non-blocking)
    threading.Thread(target=_refresh_quotes, daemon=True).start()
    threading.Thread(target=_refresh_news,   daemon=True).start()
    app.run(debug=True, port=5000, threaded=True)
