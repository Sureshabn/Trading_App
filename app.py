# app.py
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta

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

    if st.button("Run Analysis"):
        try:
            # Fetch NSE instruments
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE")
            else:
                token = int(row.iloc[0]["instrument_token"])
                hist = kite.historical_data(
                    token,
                    start_date,
                    end_date,
                    interval="day"
                )
                df = pd.DataFrame(hist)

                if df.empty:
                    st.warning("⚠️ No historical data available for this range.")
                else:
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

                    latest = df.iloc[-1]

                    # ---- Recommendation Logic ----
                    score = 0
                    score += 2 if latest["fast_ma"] > latest["slow_ma"] else -2
                    score += 1 if latest["rsi"] < 30 else (-1 if latest["rsi"] > 70 else 0)
                    score += 1 if latest["macd"] > latest["macd_signal"] else -1
                    score += 1 if latest["close"] < latest["bb_low"] else (-1 if latest["close"] > latest["bb_high"] else 0)

                    recommendation = (
                        "STRONG BUY" if score >= 4 else
                        "BUY" if score >= 2 else
                        "HOLD" if score > -2 else
                        "SELL" if score > -4 else
                        "STRONG SELL"
                    )

                    # ---- Display Data ----
                    st.subheader(f"Latest Data for {symbol} ({latest['date'].strftime('%Y-%m-%d')})")
                    st.dataframe(df.sort_values("date", ascending=False).tail(50), use_container_width=True)

                    st.subheader(f"💡 Recommendation: {recommendation} (Score: {score})")

        except Exception as e:
            st.error(f"Error fetching data: {e}")
