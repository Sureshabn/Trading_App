# app.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    # ---- Historical Analysis ----
    if st.button("Run Historical Analysis"):
        try:
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE")
            else:
                token = int(row.iloc[0]["instrument_token"])
                hist = kite.historical_data(token, start_date, end_date, interval="day")
                df_hist = pd.DataFrame(hist)
                if df_hist.empty:
                    st.warning("⚠️ No historical data available")
                else:
                    # Numeric conversion
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

                    # Historical Candlestick Chart
                    st.subheader("📈 Historical Candlestick Chart")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Candlestick(
                        x=df_hist['date'],
                        open=df_hist['open'],
                        high=df_hist['high'],
                        low=df_hist['low'],
                        close=df_hist['close'],
                        name='Price'
                    ))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['fast_ma'], line=dict(color='blue', width=1), name='Fast MA'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['slow_ma'], line=dict(color='orange', width=1), name='Slow MA'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'))
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'))
                    fig_hist.update_layout(xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=True, height=600)
                    st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error fetching historical data: {e}")

    # ---- Live Analysis Control ----
    if "live_running" not in st.session_state:
        st.session_state["live_running"] = False

    if st.button("Start Live Analysis"):
        st.session_state["live_running"] = True

    if st.button("Stop Live Analysis"):
        st.session_state["live_running"] = False

    # ---- Live Analysis Loop ----
    if st.session_state["live_running"]:
        try:
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
                last_updated_placeholder = st.empty()
                last_update = datetime.now() - timedelta(seconds=refresh_interval)

                while st.session_state["live_running"]:
                    if (datetime.now() - last_update).total_seconds() >= refresh_interval:
                        last_update = datetime.now()
                        try:
                            intraday_start = datetime.now() - timedelta(days=5)
                            hist = kite.historical_data(token, intraday_start, datetime.now(), interval="5minute")
                            df_live = pd.DataFrame(hist)
                            if df_live.empty:
                                st.warning("⚠️ No live data available")
                                time.sleep(1)
                                continue

                            for col in ["open","high","low","close","volume"]:
                                df_live[col] = pd.to_numeric(df_live[col], errors='coerce')
                            df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
                            df_live = df_live.sort_values("date", ascending=True)

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

                            # Latest row
                            df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low"])
                            latest = df_valid.iloc[-1] if not df_valid.empty else df_live.iloc[-1]

                            # Recommendation
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

                            # Last updated timestamp
                            last_updated_placeholder.markdown(f"⏱ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                            # Plotly Live Chart
                            fig_live = go.Figure()
                            fig_live.add_trace(go.Candlestick(
                                x=df_live['date'],
                                open=df_live['open'],
                                high=df_live['high'],
                                low=df_live['low'],
                                close=df_live['close'],
                                name='Price'
                            ))
                            fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name='Fast MA'))
                            fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name='Slow MA'))
                            fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'))
                            fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'))
                            chart_placeholder.plotly_chart(fig_live, use_container_width=True)
                            table_placeholder.dataframe(df_live.tail(50).sort_values("date", ascending=False), use_container_width=True)
                            rec_placeholder.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")

                        except Exception as e:
                            st.error(f"Error fetching live data: {e}")
                    time.sleep(1)
        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
