# app.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
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
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

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
    start_date = st.date_input("Start Date", datetime(2022,1,1))
    end_date = st.date_input("End Date", datetime.today())
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)

    try:
        # Fetch NSE instruments
        instruments = kite.instruments("NSE")
        df_instruments = pd.DataFrame(instruments)
        row = df_instruments[df_instruments["tradingsymbol"] == symbol]

        if row.empty:
            st.error(f"❌ Symbol {symbol} not found on NSE")
        else:
            token = int(row.iloc[0]["instrument_token"])

            # ---- Historical Analysis ----
            if st.button("Run Historical Analysis"):
                hist = kite.historical_data(token, start_date, end_date, interval="day")
                df_hist = pd.DataFrame(hist)
                if df_hist.empty:
                    st.warning("⚠️ No historical data available")
                else:
                    for col in ["open","high","low","close","volume"]:
                        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
                    df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
                    df_hist = df_hist.sort_values("date")

                    # Technical indicators
                    df_hist["fast_ma"] = df_hist["close"].rolling(20).mean()
                    df_hist["slow_ma"] = df_hist["close"].rolling(50).mean()
                    df_hist["rsi"] = ta.momentum.RSIIndicator(df_hist["close"], window=14).rsi()
                    macd = ta.trend.MACD(df_hist["close"])
                    df_hist["macd"] = macd.macd()
                    df_hist["macd_signal"] = macd.macd_signal()
                    boll = ta.volatility.BollingerBands(df_hist["close"])
                    df_hist["bb_high"] = boll.bollinger_hband()
                    df_hist["bb_low"] = boll.bollinger_lband()

                    # Buy/Sell signals
                    df_hist["signal"] = 0
                    for i, row_hist in df_hist.iterrows():
                        score = 0
                        if not pd.isna(row_hist["fast_ma"]) and not pd.isna(row_hist["slow_ma"]):
                            score += 2 if row_hist["fast_ma"] > row_hist["slow_ma"] else -2
                        if not pd.isna(row_hist["rsi"]):
                            score += 1 if row_hist["rsi"] < 30 else (-1 if row_hist["rsi"] > 70 else 0)
                        if not pd.isna(row_hist["macd"]) and not pd.isna(row_hist["macd_signal"]):
                            score += 1 if row_hist["macd"] > row_hist["macd_signal"] else -1
                        if not pd.isna(row_hist["close"]) and not pd.isna(row_hist["bb_high"]) and not pd.isna(row_hist["bb_low"]):
                            score += 1 if row_hist["close"] < row_hist["bb_low"] else (-1 if row_hist["close"] > row_hist["bb_high"] else 0)
                        if score >= 3:
                            df_hist.at[i, "signal"] = 1
                        elif score <= -3:
                            df_hist.at[i, "signal"] = -1

                    # Historical Chart
                    st.subheader("📈 Historical Candlestick Chart")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Candlestick(
                        x=df_hist['date'], open=df_hist['open'], high=df_hist['high'],
                        low=df_hist['low'], close=df_hist['close'], name='Price'
                    ))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['fast_ma'], line=dict(color='blue', width=1), name='Fast MA'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['slow_ma'], line=dict(color='orange', width=1), name='Slow MA'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'))
                    
                    # Buy/Sell triangles
                    buy_hist = df_hist[df_hist["signal"] == 1]
                    sell_hist = df_hist[df_hist["signal"] == -1]
                    fig_hist.add_trace(go.Scatter(
                        x=buy_hist['date'], y=buy_hist['close'],
                        mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Buy Signal'
                    ))
                    fig_hist.add_trace(go.Scatter(
                        x=sell_hist['date'], y=sell_hist['close'],
                        mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Sell Signal'
                    ))

                    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True, height=600)
                    st.plotly_chart(fig_hist, use_container_width=True)

            # ---- Live Analysis Toggle ----
            st.subheader("🚦 Live Analysis")
            if not st.session_state["live_running"]:
                if st.button("Start Live Analysis"):
                    st.session_state["live_running"] = True
            else:
                if st.button("Stop Live Analysis"):
                    st.session_state["live_running"] = False

            chart_placeholder = st.empty()
            table_placeholder = st.empty()
            rec_placeholder = st.empty()

            while st.session_state["live_running"]:
                try:
                    intraday_start = datetime.now() - timedelta(days=5)
                    hist = kite.historical_data(token, intraday_start, datetime.now(), interval="5minute")
                    df_live = pd.DataFrame(hist)
                    if df_live.empty:
                        st.warning("⚠️ No live data available")
                        time.sleep(refresh_interval)
                        continue

                    for col in ["open","high","low","close","volume"]:
                        df_live[col] = pd.to_numeric(df_live[col], errors='coerce')
                    df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
                    df_live = df_live.sort_values("date")

                    # Technical indicators
                    df_live["fast_ma"] = df_live["close"].rolling(20).mean()
                    df_live["slow_ma"] = df_live["close"].rolling(50).mean()
                    df_live["rsi"] = ta.momentum.RSIIndicator(df_live["close"], window=14).rsi()
                    macd = ta.trend.MACD(df_live["close"])
                    df_live["macd"] = macd.macd()
                    df_live["macd_signal"] = macd.macd_signal()
                    boll = ta.volatility.BollingerBands(df_live["close"])
                    df_live["bb_high"] = boll.bollinger_hband()
                    df_live["bb_low"] = boll.bollinger_lband()

                    # Buy/Sell signals
                    df_live["signal"] = 0
                    for i, row_live in df_live.iterrows():
                        score = 0
                        if not pd.isna(row_live["fast_ma"]) and not pd.isna(row_live["slow_ma"]):
                            score += 2 if row_live["fast_ma"] > row_live["slow_ma"] else -2
                        if not pd.isna(row_live["rsi"]):
                            score += 1 if row_live["rsi"] < 30 else (-1 if row_live["rsi"] > 70 else 0)
                        if not pd.isna(row_live["macd"]) and not pd.isna(row_live["macd_signal"]):
                            score += 1 if row_live["macd"] > row_live["macd_signal"] else -1
                        if not pd.isna(row_live["close"]) and not pd.isna(row_live["bb_high"]) and not pd.isna(row_live["bb_low"]):
                            score += 1 if row_live["close"] < row_live["bb_low"] else (-1 if row_live["close"] > row_live["bb_high"] else 0)
                        if score >= 3:
                            df_live.at[i, "signal"] = 1
                        elif score <= -3:
                            df_live.at[i, "signal"] = -1

                    # Latest valid row for recommendation
                    df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low"])
                    latest = df_valid.iloc[0] if not df_valid.empty else df_live.iloc[0]

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

                    # Live Plot
                    df_plot_live = df_live.sort_values("date")
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Candlestick(
                        x=df_plot_live['date'], open=df_plot_live['open'], high=df_plot_live['high'],
                        low=df_plot_live['low'], close=df_plot_live['close'], name='Price'
                    ))
                    fig_live.add_trace(go.Scatter(x=df_plot_live['date'], y=df_plot_live['fast_ma'], line=dict(color='blue', width=1), name='Fast MA'))
                    fig_live.add_trace(go.Scatter(x=df_plot_live['date'], y=df_plot_live['slow_ma'], line=dict(color='orange', width=1), name='Slow MA'))
                    fig_live.add_trace(go.Scatter(x=df_plot_live['date'], y=df_plot_live['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'))
                    fig_live.add_trace(go.Scatter(x=df_plot_live['date'], y=df_plot_live['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'))

                    buy_signals = df_plot_live[df_plot_live["signal"] == 1]
                    sell_signals = df_plot_live[df_plot_live["signal"] == -1]
                    fig_live.add_trace(go.Scatter(
                        x=buy_signals['date'], y=buy_signals['close'],
                        mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Buy Signal'
                    ))
                    fig_live.add_trace(go.Scatter(
                        x=sell_signals['date'], y=sell_signals['close'],
                        mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Sell Signal'
                    ))

                    fig_live.update_layout(xaxis_rangeslider_visible=False, height=600)
                    chart_placeholder.plotly_chart(fig_live, use_container_width=True)
                    table_placeholder.dataframe(df_live.tail(50), use_container_width=True)
                    rec_placeholder.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")

                    time.sleep(refresh_interval)
                except Exception as e:
                    st.error(f"Error fetching live data: {e}")
                    time.sleep(refresh_interval)

    except Exception as e:
        st.error(f"Error initializing analysis: {e}")
