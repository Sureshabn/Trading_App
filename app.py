# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Live Stock Analysis", layout="wide")

# ---- Fetch Historical Data ----
@st.cache_data(ttl=300)
def fetch_historical_df(symbol, start_date, end_date):
    df = yf.download(symbol + ".NS", start=start_date, end=end_date, interval="1d")
    if df.empty:
        return None
        
    df = df.reset_index()
    
    # --- ROBUST FIX APPLIED HERE ---
    # 1. Define the columns we want to keep (using 'Adj Close' for the final 'close')
    required_cols = ["Date", "Open", "High", "Low", "Adj Close", "Volume"]
    
    # 2. Filter the DataFrame to ensure only these columns remain
    # This prevents the issue of having both 'Close' and 'Adj Close' simultaneously
    if all(col in df.columns for col in required_cols):
        df = df[required_cols]
    else:
        # Fallback in case of missing columns, though unlikely for standard stocks
        return None 

    # 3. Rename the columns to your standard format
    df.rename(columns={"Date":"date", "Open":"open", "High":"high", "Low":"low",
                       "Adj Close":"close", "Volume":"volume"}, inplace=True)
    # --- END FIX ---
    
    df["openinterest"] = 0
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # This loop is now safe because 'close' is guaranteed to be a single Series
    numeric_cols = ['open','high','low','close','volume','openinterest']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df[['date','open','high','low','close','volume','openinterest']]

# ---- Technical Indicators ----
def add_technical_indicators(df):
    close_series = df['close']
    df['fast_ma'] = close_series.rolling(20).mean()
    df['slow_ma'] = close_series.rolling(50).mean()
    df['rsi'] = ta.momentum.RSIIndicator(close_series, window=14).rsi()
    macd_indicator = ta.trend.MACD(close_series)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    bollinger = ta.volatility.BollingerBands(close_series)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    return df

# ---- Fundamentals ----
@st.cache_data(ttl=3600)
def fetch_fundamentals(symbol):
    info = yf.Ticker(symbol + ".NS").info
    # Handle cases where some fundamentals might be missing
    return {
        'P/E Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'ROE (%)': info.get('returnOnEquity'),
        'Debt-to-Equity': info.get('debtToEquity'),
        'Dividend Yield': info.get('dividendYield')
    }

# ---- Streamlit UI ----
st.title("📈 Live Stock Analysis & Recommendation")

# Default symbol for the Indian market
symbol = st.text_input("Enter Stock Symbol:", "TCS").upper()
start_date = st.date_input("Start Date", datetime(2022,1,1))
end_date = datetime.today() - timedelta(days=1)

if st.button("Run Analysis"):
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_historical_df(symbol, start_date, end_date)
    
    if df is None or df.empty:
        st.error(f"No historical data available for {symbol}. Please check the symbol and dates.")
    else:
        df = add_technical_indicators(df)
        fundamentals = fetch_fundamentals(symbol)
        latest = df.iloc[-1]

        # ---- Recommendation Logic ----
        score = 0
        score += 2 if latest['fast_ma'] > latest['slow_ma'] else -2
        score += 1 if latest['rsi'] < 30 else (-1 if latest['rsi'] > 70 else 0)
        score += 1 if latest['macd'] > latest['macd_signal'] else -1
        score += 1 if latest['close'] < latest['bb_low'] else (-1 if latest['close'] > latest['bb_high'] else 0)
        
        # Check if P/E Ratio is valid before scoring
        pe_ratio = fundamentals.get('P/E Ratio')
        if pe_ratio is not None:
             score += 1 if pe_ratio < 20 else (-1 if pe_ratio > 25 else 0)

        recommendation = "STRONG BUY" if score >= 4 else "BUY" if score >= 2 else "HOLD" if score > -2 else "SELL" if score > -4 else "STRONG SELL"

        # ---- Display Data ----
        st.subheader(f"📊 Latest Data for {symbol} ({latest['date'].strftime('%Y-%m-%d')})")
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
        
        st.subheader("🏦 Fundamental Metrics")
        # Ensure only valid fundamentals are displayed
        valid_fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
        st.table(pd.DataFrame([valid_fundamentals]))
        
        st.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")
