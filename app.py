# app.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Zerodha Live Stock Analysis", layout="wide")
st.title("📈 Zerodha Live Stock Analysis & Recommendation")

# ---- Zerodha API credentials from Streamlit Secrets ----
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

kite = KiteConnect(api_key=API_KEY)

# ---- Session State for Access Token ----
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "token_date" not in st.session_state:
    st.session_state["token_date"] = None

# ---- Check if token exists and is valid today ----
if st.session_state["access_token"] and st.session_state["token_date"] == str(date.today()):
    kite.set_access_token(st.session_state["access_token"])
    st.success("✅ Using saved access token for today!")

# ---- Login Section ----
if not st.session_state["access_token"]:
    st.subheader("🔑 Zerodha Login")
    login_url = kite.login_url()
    st.markdown(f"[Click here to login to Zerodha]({login_url})")
    request_token = st.text_input("Paste Request Token after login:")

    if request_token:
        try:
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            st.session_state["access_token"] = data["access_token"]
            st.session_state["token_date"] = str(date.today())
            kite.set_access_token(st.session_state["access_token"])
            st.success("✅ Access token generated and saved for today!")
        except Exception as e:
            st.error(f"⚠️ Error generating session: {e}")

# ---- Stock Analysis Section ----
if st.session_state["access_token"]:
    st.subheader("📊 Stock Analysis")

    symbol = st.text_input("Enter NSE Stock Symbol:", "TCS").upper()
    refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)

    if st.button("Start Live Analysis"):

        instruments = kite.instruments("NSE")
        df_instruments = pd.DataFrame(instruments)
        row = df_instruments[df_instruments["tradingsymbol"] == symbol]

        if row.empty:
            st.error(f"❌ Symbol {symbol} not found on NSE")
        else:
            token = int(row.iloc[0]["instrument_token"])

            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            rec_placeholder = st.empty()

            while True:
                try:
                    # ---- Fetch last 5 days 5-min candles ----
                    start_time = datetime.now() - timedelta(days=5)
                    end_time = datetime.now()
                    hist = kite.historical_data(token, start_time, end_time, interval="5minute")
                    df = pd.DataFrame(hist)
                    if df.empty:
                        st.warning("⚠️ No data available")
                        time.sleep(refresh_interval)
                        continue

                    # Ensure numeric columns
                    for col in ["open","high","low","close","volume"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.sort_values("date", ascending=False)

                    # ---- Technical Indicators ----
                    df["fast_ma"] = df["close"].rolling(20).mean()
                    df["slow_ma"] = df["close"].rolling(50).mean()
                    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
                    macd = ta.trend.MACD(df["close"])
                    df["macd"] = macd.macd()
                    df["macd_signal"] = macd.macd_signal()
                    boll = ta.volatility.BollingerBands(df["close"])
                    df["bb_high"] = boll.bollinger_hband()
                    df["bb_low"] = boll.bollinger_lband()

                    # Latest valid row
                    df_valid = df.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low"])
                    if df_valid.empty:
                        latest = df.iloc[0]
                    else:
                        latest = df_valid.iloc[0]

                    # ---- Recommendation ----
                    score = 0
                    if not pd.isna(latest["fast_ma"]) and not pd.isna(latest["slow_ma"]):
                        score += 2 if latest["fast_ma"] > latest["slow_ma"] else -2
                    if not pd.isna(latest["rsi"]):
                        score += 1 if latest["rsi"] < 30 else (-1 if latest["rsi"] > 70 else 0)
                    if not pd.isna(latest["macd"]) and not pd.isna(latest["macd_signal"]):
                        score += 1 if latest["macd"] > latest["macd_signal"] else -1
                    if not pd.isna(latest["close"]) and not pd.isna(latest["bb_high"]) and not pd.isna(latest["bb_low"]):
                        score += 1 if latest["close"] < latest["bb_low"] else (-1 if latest["close"] > latest["bb_high"] else 0)

                    recommendation = (
                        "STRONG BUY" if score >= 4 else
                        "BUY" if score >= 2 else
                        "HOLD" if score > -2 else
                        "SELL" if score > -4 else
                        "STRONG SELL"
                    )

                    # ---- Candlestick Chart ----
                    df_plot = df.sort_values("date")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=df_plot['date'],
                        open=df_plot['open'],
                        high=df_plot['high'],
                        low=df_plot['low'],
                        close=df_plot['close'],
                        name='Price'
                    ))
                    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['fast_ma'], line=dict(color='blue', width=1), name='Fast MA'))
                    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['slow_ma'], line=dict(color='orange', width=1), name='Slow MA'))
                    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'))
                    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'))
                    fig.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=600)
                    chart_placeholder.plotly_chart(fig, use_container_width=True)

                    # ---- Table & Recommendation ----
                    table_placeholder.dataframe(df.head(50), use_container_width=True)
                    rec_placeholder.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")

                except Exception as e:
                    st.error(f"Error fetching data: {e}")

                time.sleep(refresh_interval)
