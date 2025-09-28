# app.py - IMPROVED VERSION
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
st.title("📈 Zerodha Live Stock Analysis & Recommendation (Pro Edition)")

# ---- Zerodha API credentials from Streamlit Secrets ----
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

kite = KiteConnect(api_key=API_KEY)

# ---- Session State ----
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "token_date" not in st.session_state:
    st.session_state["token_date"] = None
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

# ---- Check token validity ----
if st.session_state["access_token"] and st.session_state["token_date"] == str(date.today()):
    kite.set_access_token(st.session_state["access_token"])
    st.success("✅ Using saved access token for today!")

# ---- Zerodha Login (Unchanged) ----
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

    # ---- Indicator Calculation Function (DRY Principle) ----
    def calculate_indicators(df):
        # 1. Switched to Exponential Moving Averages (EMA)
        df["fast_ma"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["slow_ma"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        
        # 2. RSI (Standard)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        
        # 3. MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff() # Histogram for stronger momentum signals
        
        # 4. Bollinger Bands (Standard)
        boll = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()
        return df

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
                    for col in ["open","high","low","close","volume"]:
                        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
                    df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
                    df_hist = df_hist.sort_values("date")

                    df_hist = calculate_indicators(df_hist)

                    # ---- Plotly Chart with MACD + RSI ----
                    st.subheader("📈 Historical Candlestick with EMAs & Oscillators")
                    fig_hist = make_subplots(
                        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI")
                    )

                    # Price and Overlays
                    fig_hist.add_trace(go.Candlestick(x=df_hist['date'], open=df_hist['open'], high=df_hist['high'], low=df_hist['low'], close=df_hist['close'], name='Price'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['fast_ma'], line=dict(color='blue', width=1), name='Fast EMA (20)'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['slow_ma'], line=dict(color='orange', width=1), name='Slow EMA (50)'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'), row=1, col=1)

                    # MACD and RSI
                    fig_hist.add_trace(go.Bar(x=df_hist['date'], y=df_hist['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1) # Added MACD Histogram
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['macd'], line=dict(color='purple', width=1), name='MACD'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['macd_signal'], line=dict(color='pink', width=1, dash='dot'), name='MACD Signal'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                    
                    # RSI Overbought/Oversold lines
                    fig_hist.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_hist.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig_hist.update_layout(xaxis_rangeslider_visible=True, height=700)
                    st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching historical data: {e}")

    # ---- Live Analysis ----
    if st.button("Start Live Analysis"):
        st.session_state["live_running"] = True

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

                        df_live = calculate_indicators(df_live)

                        # Latest valid row
                        df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low"])
                        # Use the latest fully calculated bar for decision making
                        latest = df_valid.iloc[-1] if not df_valid.empty else df_live.iloc[-1] 

                        # --- PROFESSIONAL RECOMMENDATION LOGIC (Weighted Scoring) ---
                        score = 0
                        
                        # 1. TREND (MAs) - Highest Weight (+/- 4)
                        if not pd.isna(latest["fast_ma"]) and not pd.isna(latest["slow_ma"]):
                            # Bullish trend confirmation
                            if latest["fast_ma"] > latest["slow_ma"]:
                                score += 4
                            # Bearish trend confirmation
                            elif latest["fast_ma"] < latest["slow_ma"]:
                                score -= 4
                        
                        # 2. MOMENTUM (MACD) - Medium Weight (+/- 2)
                        if not pd.isna(latest["macd"]) and not pd.isna(latest["macd_signal"]):
                            # Bullish momentum acceleration (Crossover AND positive histogram)
                            if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
                                score += 2
                            # Bearish momentum acceleration (Crossover AND negative histogram)
                            elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
                                score -= 2

                        # 3. REVERSION/EXTREMES (RSI/BB) - Lowest Weight (+/- 1)
                        if not pd.isna(latest["rsi"]):
                            # Oversold is bullish signal
                            if latest["rsi"] < 30 and latest["rsi"] < latest.shift(1)["rsi"]: # Check if it is moving up from oversold
                                score += 1
                            # Overbought is bearish signal
                            elif latest["rsi"] > 70 and latest["rsi"] > latest.shift(1)["rsi"]: # Check if it is moving down from overbought
                                score -= 1
                        
                        if not pd.isna(latest["close"]) and not pd.isna(latest["bb_high"]) and not pd.isna(latest["bb_low"]):
                            # Price below BB Low (Potential short-term bounce/Cover short)
                            if latest["close"] < latest["bb_low"]:
                                score += 1
                            # Price above BB High (Potential pullback/Take profit)
                            elif latest["close"] > latest["bb_high"]:
                                score -= 1

                        recommendation = (
                            "STRONG BUY" if score >= 6 else    # Max possible score: 4 (MA) + 2 (MACD) + 1 (RSI) + 1 (BB) = 8
                            "BUY" if score >= 3 else           # A trend or strong momentum signal
                            "HOLD/NEUTRAL" if score > -3 else  # Close to 0 score
                            "SELL" if score > -6 else          # A downward trend or strong bearish momentum signal
                            "STRONG SELL"                      # Max negative score: -8
                        )

                        # Live Chart with MACD+RSI
                        fig_live = make_subplots(
                            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                            row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI")
                        )

                        # Price and Overlays
                        fig_live.add_trace(go.Candlestick(x=df_live['date'], open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'], name='Price'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name='Fast EMA (20)'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name='Slow EMA (50)'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'), row=1, col=1)
                        
                        # MACD and RSI
                        fig_live.add_trace(go.Bar(x=df_live['date'], y=df_live['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1) # Added MACD Histogram
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['macd'], line=dict(color='purple', width=1), name='MACD'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['macd_signal'], line=dict(color='pink', width=1, dash='dot'), name='MACD Signal'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                        fig_live.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig_live.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig_live.update_layout(xaxis_rangeslider_visible=False, height=700)
                        chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                        table_placeholder.dataframe(df_live.tail(10), use_container_width=True) # Show only the latest 10 rows
                        rec_placeholder.subheader(f"💡 Recommendation: **{recommendation}** (Score: {score}) at Close: {latest['close']:.2f}")

                    except Exception as e:
                        st.error(f"Error fetching live data: {e}")

                    time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
