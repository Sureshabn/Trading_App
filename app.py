# app.py
import streamlit as st
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import ta
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Live Stock Analysis", layout="wide")

# ---- Alpha Vantage API Key ----
API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your key

# ---- Fetch Historical Data ----
@st.cache_data(ttl=300)
def fetch_historical_df(symbol, start_date, end_date):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        # Fetch daily adjusted data
        df, meta = ts.get_daily_adjusted(symbol=symbol + ".NS", outputsize='full')
        df = df.reset_index().rename(columns={
            'date': 'date',
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '6. volume': 'volume'
        })
        df['date'] = pd.to_datetime(df['date'])
        # Filter by date range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        if df.empty:
            return None
        df['openinterest'] = 0
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'openinterest']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[['date','open','high','low','close','volume','openinterest']]
    except Exception as e:
        st.error(f"Error fetching data from Alpha Vantage: {e}")
        return None

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

# ---- Streamlit UI ----
st.title("📈 Live Stock Analysis & Recommendation (Alpha Vantage)")

symbol = st.text_input("Enter NSE Stock Symbol:", "TCS").upper()
start_date = st.date_input("Start Date", datetime(2022,1,1))
end_date = datetime.today() - timedelta(days=1)

start_date_dt = datetime.combine(start_date, datetime.min.time())
end_date_dt = datetime.combine(end_date, datetime.min.time())

if st.button("Run Analysis"):
    with st.spinner(f"Fetching data for {symbol} from Alpha Vantage..."):
        df = fetch_historical_df(symbol, start_date_dt, end_date_dt)
    
    if df is None or df.empty:
        st.error(f"No valid historical data available for {symbol}.")
    else:
        df = add_technical_indicators(df)
        latest = df.iloc[-1]

        # ---- Recommendation Logic ----
        score = 0
        score += 2 if latest.get('fast_ma',0) > latest.get('slow_ma',0) else -2
        score += 1 if latest.get('rsi',50) < 30 else (-1 if latest.get('rsi',50) > 70 else 0)
        score += 1 if latest.get('macd',0) > latest.get('macd_signal',0) else -1
        score += 1 if latest.get('close',0) < latest.get('bb_low',0) else (-1 if latest.get('close',0) > latest.get('bb_high',0) else 0)
        recommendation = "STRONG BUY" if score >= 4 else "BUY" if score >= 2 else "HOLD" if score > -2 else "SELL" if score > -4 else "STRONG SELL"

        # ---- Display Data ----
        latest_date = latest.get('date')
        date_str = latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else "N/A"
        st.subheader(f"📊 Latest Data for {symbol} ({date_str})")
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
        st.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")
