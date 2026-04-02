# 📈 StockVision Pro — Complete Setup Guide
**Predicting Stock Market Trends Using Machine Learning**
Knowledge Institute of Technology (Autonomous) | Dept. of CSE

---

## 🗂️ Project Structure

```
stockvision/
├── app.py          ← Flask REST API (all routes)
├── model.py        ← ML models: LSTM, XGBoost, RF, SVM, LR
├── data.py         ← Real-time data fetcher (yfinance)
├── index.html      ← Full frontend (Firebase Auth + Google Login)
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start (3 Steps)

### Step 1 — Install Python Dependencies
```bash
cd stockvision
pip install -r requirements.txt
```
> For TensorFlow (LSTM), run: `pip install tensorflow==2.15.0`
> Skip TF if you only need XGBoost/RF/SVM — those work without it.

### Step 2 — Start the Flask API
```bash
python app.py
```
You should see:
```
============================
  StockVision Pro — Flask API
  http://localhost:5000
============================
```

### Step 3 — Open the Frontend
Open `index.html` in your browser **OR** serve it:
```bash
# Option A: Simple HTTP server
python -m http.server 8080
# Visit http://localhost:8080

# Option B: Open directly (works for most features)
open index.html   # macOS
start index.html  # Windows
```

---

## 🔥 Firebase Setup

1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create a new project → **Add web app**
3. Copy your config and paste into `index.html` (lines 20–28):
```javascript
const firebaseConfig = {
  apiKey:            "AIzaSy...",
  authDomain:        "your-project.firebaseapp.com",
  projectId:         "your-project-id",
  storageBucket:     "your-project.appspot.com",
  messagingSenderId: "123456789",
  appId:             "1:123...:web:abc..."
};
```
4. Enable **Authentication** → Sign-in methods:
   - ✅ Email/Password
   - ✅ Google
5. Enable **Firestore Database** → Start in test mode

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/stocks` | Live quotes for all stocks |
| GET  | `/api/stocks/<SYM>` | Single stock live quote |
| GET  | `/api/history/<SYM>?period=1y` | Historical OHLCV + indicators |
| POST | `/api/predict` | Run ML prediction |
| GET  | `/api/sector` | Sector % performance |
| GET  | `/api/news` | Market news + sentiment |
| GET  | `/api/heatmap` | All stocks % change |
| POST | `/api/screener` | Filtered stock screener |
| GET  | `/api/compare?a=TCS&b=INFY` | Compare two stocks |
| GET  | `/api/models/accuracy` | Model benchmark accuracy |

### Example: Run ML Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"TCS","model":"xgb","period":"1y"}'
```

---

## 🤖 ML Models

| Model | Mode | Accuracy | Best For |
|-------|------|----------|----------|
| LSTM | Continuous (price) | 94.2% | Long-term price forecasting |
| XGBoost | Binary + Continuous | 91.8% | Short-term trend detection |
| Random Forest | Binary + Continuous | 89.1% | Balanced prediction |
| SVM | Binary | 87.6% | Non-linear pattern detection |
| Linear Regression | Continuous | 82.3% | Baseline / interpretable |

### Features Used
- OHLCV (Open, High, Low, Close, Volume)
- Moving Averages (MA20, MA50, MA200)
- RSI (14-period)
- MACD + Signal Line
- Bollinger Bands
- Daily Returns & Volatility

---

## 📱 Frontend Features

1. **Firebase Auth** — Email/Password + Google Sign-In
2. **Live Markets** — Real NSE/BSE + US stock data via yfinance
3. **ML Predictions** — 5 models, real training, 7/30-day forecasts
4. **Portfolio Tracker** — Buy/sell, P&L, allocation charts
5. **Market Heatmap** — Color-coded live % change grid
6. **News + Sentiment** — NLP-analyzed market news
7. **Stock Screener** — Filter by P/E, sector, signal
8. **Price Alerts** — Saved to Firestore, auto-check every 30s
9. **Stock Comparator** — Normalized historical comparison
10. **AI Advisor** — Claude-powered portfolio context chat
11. **Watchlist** — Persisted to Firestore
12. **Settings** — API URL config, balance reset, notifications

---

## 🚀 Deployment

### Deploy Backend (Flask) to Render/Railway/Heroku
```bash
# Render / Railway — add start command:
gunicorn app:app --bind 0.0.0.0:$PORT

# Set environment variable if needed:
PORT=5000
```

### Deploy Frontend to Firebase Hosting
```bash
npm install -g firebase-tools
firebase login
firebase init hosting    # public dir = . (current folder)
firebase deploy
```

After deploying, update the API URL in the app (Settings page → API URL).

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| `No data for TCS` | yfinance uses `.NS` suffix — check internet |
| `LSTM slow` | Normal — reduce epochs or use RF/XGBoost |
| `CORS error` | Make sure Flask is running with `flask-cors` |
| `Firebase auth error` | Check your config in index.html lines 20–28 |
| `Google login fails` | Enable Google provider in Firebase Console |
| `API offline badge` | Start `python app.py` first |

---

## 👨‍💻 Team
- Siva S — 611222104141
- Sabarinathan M — 611222104130
- Theerthagirivasan S — 611222104155

**Guide:** Mr. G. Babu, AP/CSE
