# app.py - PRO VERSION 2 (with Risk Management & Flexibility)
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
st.title("📈 Zerodha Live Stock Analysis & Recommendation (Pro V2)")

# ---- Zerodha API credentials from Streamlit Secrets ----
# Assuming API_KEY and API_SECRET are securely handled in st.secrets
# NOTE: You must have these variables defined in your Streamlit secrets file (secrets.toml)
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
except KeyError:
    st.error("⚠️ Error: API_KEY or API_SECRET not found in Streamlit secrets.")
    st.stop()

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
    try:
        kite.set_access_token(st.session_state["access_token"])
        st.success("✅ Using saved access token for today!")
    except Exception as e:
        st.error(f"Error setting access token: {e}")
        st.session_state["access_token"] = None # Invalidate token on error


# ---- Zerodha Login ----
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
            # Rerun app to continue to the analysis section
            st.rerun() 
        except Exception as e:
            st.error(f"⚠️ Error generating session: {e}")

# ----------------------------------------------------------------------
# ---- Stock Analysis Section ----
# ----------------------------------------------------------------------
if st.session_state["access_token"]:
    st.subheader("📊 Stock Analysis")

    # --- 1. Flexibility and Parameterization (ATR, R:R, Indicator Windows) ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        symbol = st.text_input("NSE Symbol:", "TCS").upper()
    with col2:
        # FIX: Explicitly using 'value=' to fix TypeError
        fast_ema_w = st.number_input("Fast EMA W.", min_value=5, value=20) 
    with col3:
        # FIX: Explicitly using 'value='
        slow_ema_w = st.number_input("Slow EMA W.", min_value=10, value=50)
    with col4:
        # FIX: Explicitly using 'value='
        rsi_w = st.number_input("RSI/ATR W.", min_value=5, value=14)
    with col5:
        # FIX: Explicitly using 'value='
        risk_rr = st.number_input("R:R Ratio (1:X)", min_value=1.0, step=0.5, value=2.0)

    start_date = st.date_input("Historical Start Date", datetime(2022,1,1))
    end_date = st.date_input("Historical End Date", datetime.today())
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Intraday Bar Interval", ["5minute", "15minute", "30minute"])


    # ---- Indicator Calculation Function (DRY Principle) ----
    @st.cache_data(ttl=600) # Cache data for 10 minutes
    def calculate_indicators(df, fast_w, slow_w, rsi_w):
        # 1. Exponential Moving Averages (EMA)
        df["fast_ma"] = ta.trend.EMAIndicator(df["close"], window=fast_w).ema_indicator()
        df["slow_ma"] = ta.trend.EMAIndicator(df["close"], window=slow_w).ema_indicator()
        
        # 2. RSI (Standard)
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_w).rsi()
        
        # 3. MACD
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff() 
        
        # 4. Bollinger Bands (Standard)
        boll = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()

        # 5. Average True Range (ATR) - For Risk Management
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=rsi_w).average_true_range()

        return df

    # ---- Historical Analysis ----
    if st.button("Run Historical Analysis (Daily Timeframe)"):
        try:
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE")
            else:
                token = int(row.iloc[0]["instrument_token"])
                # The historical API only supports daily data for long ranges
                hist = kite.historical_data(token, start_date, end_date, interval="day")
                df_hist = pd.DataFrame(hist)

                if df_hist.empty:
                    st.warning("⚠️ No historical data available")
                else:
                    for col in ["open","high","low","close","volume"]:
                        df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce')
                    df_hist['date'] = pd.to_datetime(df_hist['date'], errors='coerce')
                    df_hist = df_hist.sort_values("date")

                    df_hist = calculate_indicators(df_hist, fast_ema_w, slow_ema_w, rsi_w)

                    st.subheader(f"📈 Historical ({symbol}) Candlestick with EMAs & Oscillators")
                    
                    fig_hist = make_subplots(
                        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI")
                    )

                    # Price and Overlays
                    fig_hist.add_trace(go.Candlestick(x=df_hist['date'], open=df_hist['open'], high=df_hist['high'], low=df_hist['low'], close=df_hist['close'], name='Price'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'), row=1, col=1)

                    # MACD and RSI
                    fig_hist.add_trace(go.Bar(x=df_hist['date'], y=df_hist['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['macd'], line=dict(color='purple', width=1), name='MACD'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['macd_signal'], line=dict(color='pink', width=1, dash='dot'), name='MACD Signal'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                    
                    fig_hist.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_hist.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig_hist.update_layout(xaxis_rangeslider_visible=True, height=700)
                    st.plotly_chart(fig_hist, use_container_width=True)

        except Exception as e:
            st.error(f"Error fetching historical data: {e}")


    # ---- Live Analysis ----
    if st.button("Start Intraday Bar-Close Analysis"):
        st.session_state["live_running"] = True

    if st.session_state["live_running"]:
        try:
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE")
                st.session_state["live_running"] = False
                st.stop()
            else:
                token = int(row.iloc[0]["instrument_token"])
                chart_placeholder = st.empty()
                table_placeholder = st.empty()
                rec_placeholder = st.empty()
                targets_placeholder = st.empty() 

                while st.session_state["live_running"]:
                    try:
                        # Fetch enough data for indicators to stabilize
                        intraday_start = datetime.now() - timedelta(days=5) 
                        hist = kite.historical_data(token, intraday_start, datetime.now(), interval=live_interval)
                        df_live = pd.DataFrame(hist)
                        
                        if df_live.empty:
                            st.warning("⚠️ No live data available")
                            time.sleep(refresh_interval)
                            continue

                        # Data cleanup
                        for col in ["open","high","low","close","volume"]:
                            df_live[col] = pd.to_numeric(df_live[col], errors='coerce')
                        df_live['date'] = pd.to_datetime(df_live['date'], errors='coerce')
                        df_live = df_live.sort_values("date")

                        df_live = calculate_indicators(df_live, fast_ema_w, slow_ema_w, rsi_w)

                        # Latest valid row (must have all indicators calculated)
                        df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr"])
                        
                        if df_valid.empty:
                            st.warning("⚠️ Not enough data yet to calculate indicators.")
                            time.sleep(refresh_interval)
                            continue

                        latest = df_valid.iloc[-1]
                        prev = df_valid.iloc[-2] if len(df_valid) >= 2 else None


                        # --- PROFESSIONAL RECOMMENDATION LOGIC (Weighted Scoring) ---
                        score = 0
                        is_bullish_trend = False
                        is_bearish_trend = False
                        
                        # 1. TREND (MAs) - Highest Weight (+/- 4)
                        if latest["fast_ma"] > latest["slow_ma"]:
                            score += 4
                            is_bullish_trend = True
                        elif latest["fast_ma"] < latest["slow_ma"]:
                            score -= 4
                            is_bearish_trend = True
                        
                        # 2. MOMENTUM (MACD) - Medium Weight (+/- 2)
                        # Check for crossover AND positive/negative histogram
                        if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
                            score += 2
                        elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
                            score -= 2

                        # 3. REVERSION/EXTREMES (RSI/BB) - Lowest Weight (+/- 1), **only confirms trend-following pullback**
                        if prev is not None:
                            # Bullish pullback confirmation (Trend is UP, Price/RSI is LOW and reversing)
                            if is_bullish_trend:
                                # RSI is rising from the lower half (pullback entry)
                                if latest["rsi"] < 50 and latest["rsi"] > prev["rsi"]:
                                    score += 1
                                # Price bounced off BB Low
                                if latest["close"] < latest["bb_low"] and latest["close"] > prev["close"]:
                                    score += 1

                            # Bearish pullback confirmation (Trend is DOWN, Price/RSI is HIGH and reversing)
                            elif is_bearish_trend:
                                # RSI is falling from the upper half (pullback entry)
                                if latest["rsi"] > 50 and latest["rsi"] < prev["rsi"]:
                                    score -= 1
                                # Price rejected by BB High
                                if latest["close"] > latest["bb_high"] and latest["close"] < prev["close"]:
                                    score -= 1
                        
                        
                        # --- RISK MANAGEMENT CALCULATIONS (ATR Based) ---
                        stop_loss, take_profit = None, None
                        risk_multiple = 2.0 # Standard Stop-Loss distance in ATRs
                        
                        if not pd.isna(latest["atr"]):
                            # Suggestion based on a strong buy signal
                            if score >= 4: 
                                stop_loss_price = latest["close"] - (latest["atr"] * risk_multiple)
                                take_profit_price = latest["close"] + (latest["atr"] * risk_multiple * risk_rr)
                                stop_loss = f"{stop_loss_price:.2f}"
                                take_profit = f"{take_profit_price:.2f}"
                            # Suggestion based on a strong sell signal
                            elif score <= -4: 
                                stop_loss_price = latest["close"] + (latest["atr"] * risk_multiple)
                                take_profit_price = latest["close"] - (latest["atr"] * risk_multiple * risk_rr)
                                stop_loss = f"{stop_loss_price:.2f}"
                                take_profit = f"{take_profit_price:.2f}"


                        recommendation = (
                            "STRONG BUY" if score >= 6 else    
                            "BUY" if score >= 3 else          
                            "HOLD/NEUTRAL" if score > -3 else 
                            "SELL" if score > -6 else         
                            "STRONG SELL"                     
                        )

                        # Live Chart (Plotly)
                        fig_live = make_subplots(
                            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                            row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI")
                        )

                        # Price and Overlays
                        fig_live.add_trace(go.Candlestick(x=df_live['date'], open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'], name='Price'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'), row=1, col=1)
                        
                        # MACD and RSI
                        fig_live.add_trace(go.Bar(x=df_live['date'], y=df_live['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['macd'], line=dict(color='purple', width=1), name='MACD'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['macd_signal'], line=dict(color='pink', width=1, dash='dot'), name='MACD Signal'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                        fig_live.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig_live.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig_live.update_layout(xaxis_rangeslider_visible=False, height=700)
                        chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                        table_placeholder.dataframe(df_live.tail(10), use_container_width=True)
                        rec_placeholder.subheader(f"💡 Recommendation: **{recommendation}** (Score: {score})")
                        
                        # Display Stop-Loss and Take-Profit
                        if stop_loss and take_profit:
                            targets_placeholder.markdown(f"""
                                **Risk Management (ATR $\times$ {risk_multiple} | R:R 1:{risk_rr})** * **Entry Price:** {latest['close']:.2f}
                                * **Suggested Stop-Loss:** **{stop_loss}**
                                * **Suggested Take-Profit:** **{take_profit}**
                            """)
                        else:
                            targets_placeholder.markdown("⚠️ **Risk Management:** No strong signal (Score $\in [-3, 3]$) or ATR unavailable.")


                    except Exception as e:
                        st.error(f"Error during live data analysis loop: {e}")
                        # If a runtime error occurs, stop the loop
                        st.session_state["live_running"] = False
                        st.rerun()

                    time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
            st.session_state["live_running"] = False
