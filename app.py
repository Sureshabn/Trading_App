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

    # Flatten MultiIndex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Normalize column names
    df.rename(columns=lambda x: x.lower().replace(' ', ''), inplace=True)

    # Standard column mapping
    col_map = {
        'date': 'date',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'adjclose': 'close',  # adjusted close if available
        'close': 'close',
        'volume': 'volume'
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Ensure 'close' exists
    if 'close' not in df.columns:
        df['close'] = pd.NA

    # Add openinterest
    df['openinterest'] = 0

    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['date'] = pd.NaT

    # Safe numeric conversion
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'openinterest']
    for col in numeric_cols:
        if col in df.columns and isinstance(df[col], pd.Series):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.NA

    return df[['date', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]

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
    return {
        'P/E Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'ROE (%)': info.get('returnOnEquity'),
        'Debt-to-Equity': info.get('debtToEquity'),
        'Dividend Yield': info.get('dividendYield')
    }

# ---- Streamlit UI ----
st.title("📈 Live Stock Analysis & Recommendation")

symbol = st.text_input("Enter Stock Symbol:", "TCS").upper()
start_date = st.date_input("Start Date", datetime(2022,1,1))
end_date = datetime.today() - timedelta(days=1)

if st.button("Run Analysis"):
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_historical_df(symbol, start_date, end_date)
    
    if df is None or df.empty:
        st.error(f"No historical data available for {symbol}.")
    else:
        df = add_technical_indicators(df)
        fundamentals = fetch_fundamentals(symbol)
        
        # Defensive: check if latest row exists
        if not df.empty:
            latest = df.iloc[-1]

            # ---- Recommendation Logic ----
            score = 0
            score += 2 if latest.get('fast_ma', 0) > latest.get('slow_ma', 0) else -2
            score += 1 if latest.get('rsi', 50) < 30 else (-1 if latest.get('rsi',50) > 70 else 0)
            score += 1 if latest.get('macd',0) > latest.get('macd_signal',0) else -1
            score += 1 if latest.get('close',0) < latest.get('bb_low',0) else (-1 if latest.get('close',0) > latest.get('bb_high',0) else 0)
            
            pe_ratio = fundamentals.get('P/E Ratio')
            if pe_ratio is not None:
                 score += 1 if pe_ratio < 20 else (-1 if pe_ratio > 25 else 0)

            recommendation = "STRONG BUY" if score >= 4 else "BUY" if score >= 2 else "HOLD" if score > -2 else "SELL" if score > -4 else "STRONG SELL"

            # ---- Display Data ----
            st.subheader(f"📊 Latest Data for {symbol} ({latest['date'].strftime('%Y-%m-%d')})")
            st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
            
            st.subheader("🏦 Fundamental Metrics")
            valid_fundamentals = {k: v for k, v in fundamentals.items() if v is not None}
            st.table(pd.DataFrame([valid_fundamentals]))
            
            st.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")
        else:
            st.warning("No valid rows found in historical data.")
