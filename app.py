# app.py - PRO VERSION 3 (Final with Advanced Risk Management & Logic)
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import numpy as np

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
st.title("📈 Zerodha Stock Analysis & Risk Manager (Pro V3)")

# ---- Zerodha API credentials from Streamlit Secrets ----
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
except KeyError:
    st.error("⚠️ Error: API_KEY or API_SECRET not found in Streamlit secrets.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)

# ---- Session State (Unchanged) ----
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None
if "token_date" not in st.session_state:
    st.session_state["token_date"] = None
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

# ---- Check token validity & Login Logic (Omitted for brevity, assumed functional) ----
if st.session_state["access_token"] and st.session_state["token_date"] == str(date.today()):
    try:
        kite.set_access_token(st.session_state["access_token"])
        st.success("✅ Using saved access token for today!")
    except Exception as e:
        st.error(f"Error setting access token: {e}")
        st.session_state["access_token"] = None 

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
            st.rerun() 
        except Exception as e:
            st.error(f"⚠️ Error generating session: {e}")

# ----------------------------------------------------------------------
# ---- Stock Analysis Section ----
# ----------------------------------------------------------------------
if st.session_state["access_token"]:
    st.subheader("📊 Stock Analysis")

    # --- 1. Flexibility and Parameterization & Position Sizing ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        symbol = st.text_input("NSE Symbol:", "TCS").upper()
    with col2:
        fast_ema_w = st.number_input("Fast EMA W.", min_value=5, value=20) 
    with col3:
        slow_ema_w = st.number_input("Slow EMA W.", min_value=10, value=50)
    with col4:
        rsi_w = st.number_input("RSI/ATR W.", min_value=5, value=14)
    with col5:
        risk_rr = st.number_input("R:R Ratio (1:X)", min_value=1.0, step=0.5, value=2.0)

    colA, colB, colC = st.columns(3)
    with colA:
        account_size = st.number_input("Account Size (₹)", min_value=1000, value=100000, step=1000)
    with colB:
        risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with colC:
        atr_stop_mult = st.number_input("ATR Stop Multiplier", min_value=1.0, value=2.0, step=0.5)

    start_date = st.date_input("Historical Start Date", datetime(2022,1,1))
    end_date = st.date_input("Historical End Date", datetime.today())
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Intraday Bar Interval", ["5minute", "15minute", "30minute"])


    # ---- Indicator Calculation Function ----
    @st.cache_data(ttl=600)
    def calculate_indicators(df, fast_w, slow_w, rsi_w):
        # Data cleaning and type conversion
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values("date")
        
        # EMAs
        df["fast_ma"] = ta.trend.EMAIndicator(df["close"], window=fast_w).ema_indicator()
        df["slow_ma"] = ta.trend.EMAIndicator(df["close"], window=slow_w).ema_indicator()
        
        # MA Slope (Trend Strength) - Calculated as the change over the last 3 periods
        df["fast_ma_slope"] = (df["fast_ma"] - df["fast_ma"].shift(3)) / 3
        
        # RSI & MACD
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_w).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff() 
        
        # Bollinger Bands & ATR
        boll = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=rsi_w).average_true_range()

        return df

    # ---- Historical Analysis Button (omitted plot generation for brevity, assumed functional) ----
    if st.button("Run Historical Analysis (Daily Timeframe)"):
        # ... (Historical data fetch, cleanup, and plot logic - similar to V2)
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
                    df_hist = calculate_indicators(df_hist, fast_ema_w, slow_ema_w, rsi_w)
                    
                    st.subheader(f"📈 Historical ({symbol}) Candlestick with EMAs & Oscillators")
                    # Simplified Plotly Chart Generation (Similar to V2)
                    fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI"))
                    fig_hist.add_trace(go.Candlestick(x=df_hist['date'], open=df_hist['open'], high=df_hist['high'], low=df_hist['low'], close=df_hist['close'], name='Price'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_high'], line=dict(color='green', width=1, dash='dot'), name='BB High'), row=1, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['bb_low'], line=dict(color='red', width=1, dash='dot'), name='BB Low'), row=1, col=1)
                    fig_hist.add_trace(go.Bar(x=df_hist['date'], y=df_hist['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                    fig_hist.add_trace(go.Scatter(x=df_hist['date'], y=df_hist['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
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
                risk_placeholder = st.empty() # New placeholder for risk info

                while st.session_state["live_running"]:
                    try:
                        intraday_start = datetime.now() - timedelta(days=5) 
                        hist = kite.historical_data(token, intraday_start, datetime.now(), interval=live_interval)
                        df_live = pd.DataFrame(hist)
                        
                        if df_live.empty:
                            st.warning("⚠️ No data available")
                            time.sleep(refresh_interval)
                            continue

                        df_live = calculate_indicators(df_live, fast_ema_w, slow_ema_w, rsi_w)

                        df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope"])
                        
                        if df_valid.empty:
                            st.warning("⚠️ Not enough data yet to calculate all indicators.")
                            time.sleep(refresh_interval)
                            continue

                        latest = df_valid.iloc[-1]
                        prev = df_valid.iloc[-2] if len(df_valid) >= 2 else None


                        # --- PROFESSIONAL RECOMMENDATION LOGIC (Weighted Scoring) ---
                        score = 0
                        is_bullish_trend = False
                        is_bearish_trend = False
                        
                        # 1. TREND (MAs) - Highest Weight (+/- 4) + Slope (+/- 2)
                        if latest["fast_ma"] > latest["slow_ma"]:
                            score += 4
                            is_bullish_trend = True
                            if latest["fast_ma_slope"] > 0.001: # Check for positive slope
                                score += 2 
                        elif latest["fast_ma"] < latest["slow_ma"]:
                            score -= 4
                            is_bearish_trend = True
                            if latest["fast_ma_slope"] < -0.001: # Check for negative slope
                                score -= 2

                        # 2. MOMENTUM (MACD) - Medium Weight (+/- 2)
                        if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
                            score += 2
                        elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
                            score -= 2

                        # 3. REVERSION/EXTREMES (RSI/BB) - Lowest Weight (+/- 1), only confirms trend-following pullback
                        if prev is not None:
                            # Bullish pullback confirmation
                            if is_bullish_trend:
                                if latest["rsi"] < 50 and latest["rsi"] > prev["rsi"]:
                                    score += 1 # RSI rising from mid-range
                                if latest["close"] < latest["bb_low"]:
                                    score += 1 # Price touched BB Low (potential bounce)

                            # Bearish pullback confirmation
                            elif is_bearish_trend:
                                if latest["rsi"] > 50 and latest["rsi"] < prev["rsi"]:
                                    score -= 1 # RSI falling from mid-range
                                if latest["close"] > latest["bb_high"]:
                                    score -= 1 # Price touched BB High (potential rejection)
                        
                        
                        # --- RISK MANAGEMENT CALCULATIONS ---
                        stop_loss, take_profit = None, None
                        suggested_quantity = 0

                        if not pd.isna(latest["atr"]):
                            risk_per_trade = account_size * (risk_percent / 100)
                            stop_distance = latest["atr"] * atr_stop_mult
                            
                            # Position Sizing: Max quantity based on 1% risk
                            if stop_distance > 0:
                                suggested_quantity = int(risk_per_trade / stop_distance)
                                if suggested_quantity < 1: suggested_quantity = 1

                            # Suggestion based on a strong buy signal (Score >= 6)
                            if score >= 6: 
                                stop_loss_price = latest["close"] - stop_distance
                                take_profit_price = latest["close"] + (stop_distance * risk_rr)
                                stop_loss = f"{stop_loss_price:.2f}"
                                take_profit = f"{take_profit_price:.2f}"
                            # Suggestion based on a strong sell signal (Score <= -6)
                            elif score <= -6: 
                                stop_loss_price = latest["close"] + stop_distance
                                take_profit_price = latest["close"] - (stop_distance * risk_rr)
                                stop_loss = f"{stop_loss_price:.2f}"
                                take_profit = f"{take_profit_price:.2f}"

                        
                        recommendation = (
                            "STRONG BUY" if score >= 8 else # Max score now 4 (MA) + 2 (Slope) + 2 (MACD) + 2 (Reversion) = 10
                            "BUY" if score >= 4 else          
                            "HOLD/NEUTRAL" if score > -4 else 
                            "SELL" if score > -8 else         
                            "STRONG SELL"                     
                        )

                        # Live Chart (Visualizing SL/TP)
                        fig_live = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI"))
                        fig_live.add_trace(go.Candlestick(x=df_live['date'], open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'], name='Price'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                        
                        # Visualize SL/TP only if a strong signal is present
                        if stop_loss and take_profit:
                            # Stop-Loss line
                            fig_live.add_hline(y=float(stop_loss), line_dash="dash", line_color="red", row=1, col=1, annotation_text="SL")
                            # Take-Profit line
                            fig_live.add_hline(y=float(take_profit), line_dash="dash", line_color="green", row=1, col=1, annotation_text="TP")

                        fig_live.add_trace(go.Bar(x=df_live['date'], y=df_live['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                        fig_live.update_layout(xaxis_rangeslider_visible=False, height=700)
                        chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                        table_placeholder.dataframe(df_live.tail(10), use_container_width=True)
                        rec_placeholder.subheader(f"💡 **{symbol}** Recommendation: **{recommendation}** (Score: {score})")
                        
                        # Display Risk Management and Position Sizing
                        targets_placeholder.markdown(f"""
                            **Trade Plan (R:R 1:{risk_rr} | Stop: {atr_stop_mult}x ATR)**
                            * **Entry Price:** **{latest['close']:.2f}**
                            * **Suggested Stop-Loss:** **{stop_loss if stop_loss else 'N/A'}**
                            * **Suggested Take-Profit:** **{take_profit if take_profit else 'N/A'}**
                        """)
                        
                        if stop_loss and take_profit:
                            risk_placeholder.markdown(f"""
                                **Position Sizing (1% Risk):**
                                * **Risk Amount:** ₹ {risk_per_trade:.2f}
                                * **Stop Distance:** ₹ {stop_distance:.2f}
                                * **Max Quantity:** **{suggested_quantity}** shares
                            """)
                        else:
                            risk_placeholder.markdown("⚠️ **Position Sizing:** Requires a STRONG BUY/SELL signal to calculate quantity.")


                    except Exception as e:
                        st.error(f"Error during live data analysis loop: {e}")
                        st.session_state["live_running"] = False
                        st.rerun()

                    time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
            st.session_state["live_running"] = False
