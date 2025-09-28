# app.py - PRO VERSION 9 (Final with HTF Filter, Momentum Decay, and Volatility Cap)
import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import numpy as np

# --- 1. SETUP & AUTHENTICATION (NO CHANGE) ---
st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
st.title("📈 Zerodha Stock Analysis & Risk Manager (Pro V9 - FULLY SAFEGURADED)")

try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
except KeyError:
    st.error("⚠️ Error: API_KEY or API_SECRET not found in Streamlit secrets.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)

# Session State Initialization (Simplified)
if "access_token" not in st.session_state: st.session_state["access_token"] = None
if "token_date" not in st.session_state: st.session_state["token_date"] = None
if "live_running" not in st.session_state: st.session_state["live_running"] = False

# [ ... Login and token check logic as before ... ]
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


# --- 2. PARAMETERS (UPDATED) ---
if st.session_state["access_token"]:
    st.subheader("📊 Stock Analysis")

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
    
    colD, colE, colF = st.columns(3)
    with colD:
        volume_conf_mult = st.number_input("Volume Confirmation Multiplier (x ATR Vol)", min_value=1.0, value=1.5, step=0.1)
    with colE:
        swing_lookback = st.number_input("Swing Lookback Periods (for S/R)", min_value=5, value=10)
    with colF:
        # NEW: Volatility Cap
        max_stop_perc = st.number_input("Max Stop Distance (% of Price)", min_value=0.5, value=2.0, max_value=5.0, step=0.1)


    start_date = st.date_input("Historical Start Date", datetime(2022,1,1))
    end_date = st.date_input("Historical End Date", datetime.today())
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Intraday Bar Interval", ["5minute", "15minute", "30minute"])

    # --- 3. HELPER FUNCTIONS (UPDATED) ---
    
    @st.cache_data(ttl=600) 
    def calculate_indicators(df, fast_w, slow_w, rsi_w):
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values("date")
        
        # EMAs and Slope
        df["fast_ma"] = ta.trend.EMAIndicator(df["close"], window=fast_w).ema_indicator()
        df["slow_ma"] = ta.trend.EMAIndicator(df["close"], window=slow_w).ema_indicator()
        df["fast_ma_slope"] = (df["fast_ma"] - df["fast_ma"].shift(3)) / 3
        
        # RSI, MACD, BB, ATR
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_w).rsi()
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff() 
        boll = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=rsi_w).average_true_range()

        # Volume Average (ATR Volume)
        df["atr_volume"] = df["volume"].rolling(window=rsi_w).mean()

        return df

    @st.cache_data(ttl=86400) # Daily data can be cached for 24 hours
    def fetch_and_calculate_daily_trend(token):
        """Fetches daily data to determine the High-Timeframe trend."""
        try:
            # Need a start date far enough back to calculate 50-day EMA (e.g., 200 days)
            daily_start = datetime.now() - timedelta(days=200) 
            daily_hist = kite.historical_data(token, daily_start, datetime.now(), interval="day")
            df_daily = pd.DataFrame(daily_hist)
            if df_daily.empty:
                return "UNKNOWN"
            
            df_daily = df_daily.sort_values("date")
            df_daily["ema20"] = ta.trend.EMAIndicator(df_daily["close"], window=20).ema_indicator()
            df_daily["ema50"] = ta.trend.EMAIndicator(df_daily["close"], window=50).ema_indicator()
            
            latest_daily = df_daily.iloc[-1]
            
            if latest_daily["ema20"] > latest_daily["ema50"]:
                return "BULLISH"
            elif latest_daily["ema20"] < latest_daily["ema50"]:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            st.error(f"Error fetching daily trend data: {e}")
            return "ERROR"


    # ---- Historical Analysis Button (Plotting Logic - NO CHANGE) ----
    if st.button("Run Historical Analysis (Daily Timeframe)"):
        # [ ... Historical Analysis Logic as in V8 ... ]
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


    # --- 4. LIVE ANALYSIS LOOP (UPDATED) ---
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
                
                # Fetch High-Timeframe Trend once before the loop
                daily_trend = fetch_and_calculate_daily_trend(token)
                st.info(f"🗺️ **High-Timeframe (Daily) Trend:** {daily_trend}")

                # Placeholders
                chart_placeholder = st.empty()
                table_placeholder = st.empty()
                rec_placeholder = st.empty()
                targets_placeholder = st.empty() 
                risk_placeholder = st.empty() 
                score_rules_placeholder = st.empty()
                score_card_placeholder = st.empty()

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

                        # Drop NaN, ensuring we have enough data for RSI lookback (at least 4 bars)
                        df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])
                        
                        if len(df_valid) < 4:
                            st.warning("⚠️ Not enough data yet to calculate all indicators and momentum decay.")
                            time.sleep(refresh_interval)
                            continue

                        latest = df_valid.iloc[-1]
                        prev = df_valid.iloc[-2]
                        # For RSI decay: 3 bars ago (4th last row)
                        rsi_3_ago = df_valid.iloc[-4]['rsi'] if len(df_valid) >= 4 else latest['rsi']


                        # --- CORE SCORING LOGIC (As before) ---
                        score = 0
                        is_bullish_trend = False
                        is_bearish_trend = False
                        trend_score = 0
                        momentum_score = 0
                        reversion_score = 0
                        
                        # ... (Score calculation logic remains the same)
                        
                        # A. TREND (MAs)
                        if latest["fast_ma"] > latest["slow_ma"]:
                            score += 4; trend_score += 4; is_bullish_trend = True
                            if latest["fast_ma_slope"] > 0.001: score += 2; trend_score += 2
                        elif latest["fast_ma"] < latest["slow_ma"]:
                            score -= 4; trend_score -= 4; is_bearish_trend = True
                            if latest["fast_ma_slope"] < -0.001: score -= 2; trend_score -= 2

                        # B. MOMENTUM (MACD)
                        if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
                            score += 2; momentum_score += 2
                        elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
                            score -= 2; momentum_score -= 2
                        else:
                            if latest["macd"] > latest["macd_signal"]: momentum_score += 1 
                            elif latest["macd"] < latest["macd_signal"]: momentum_score -= 1

                        # C. REVERSION/EXTREMES (RSI/BB)
                        if is_bullish_trend:
                            if latest["rsi"] < 50 and latest["rsi"] > prev["rsi"]: score += 1; reversion_score += 1
                            if latest["close"] < latest["bb_low"] and latest["close"] > prev["close"]: score += 1; reversion_score += 1
                        elif is_bearish_trend:
                            if latest["rsi"] > 50 and latest["rsi"] < prev["rsi"]: score -= 1; reversion_score -= 1
                            if latest["close"] > latest["bb_high"] and latest["close"] < prev["close"]: score -= 1; reversion_score -= 1


                        # --- SAFEGURADS AND CONFIRMATION LOGIC (UPDATED) ---
                        
                        # 1. Price Action (Swing High/Low)
                        df_lookback = df_valid.iloc[-swing_lookback-1:-1]
                        swing_high = df_lookback["high"].max() if not df_lookback.empty else latest["high"]
                        swing_low = df_lookback["low"].min() if not df_lookback.empty else latest["low"]
                        
                        is_breakout_confirmed = False
                        if is_bullish_trend and latest["close"] > swing_high:
                            is_breakout_confirmed = True
                        elif is_bearish_trend and latest["close"] < swing_low:
                            is_breakout_confirmed = True
                            
                        # 2. Volume Confirmation
                        is_volume_confirmed = False
                        volume_ratio = latest["volume"] / latest["atr_volume"] if latest["atr_volume"] > 0 else 0.0
                        if volume_ratio >= volume_conf_mult:
                            is_volume_confirmed = True
                            
                        # 3. Momentum Decay Check (NEW)
                        is_momentum_decay = False
                        if score >= 6 and latest['rsi'] < rsi_3_ago: # Bullish signal but RSI is decreasing
                            is_momentum_decay = True
                        elif score <= -6 and latest['rsi'] > rsi_3_ago: # Bearish signal but RSI is increasing
                            is_momentum_decay = True
                            
                        # 4. Volatility Cap Check (NEW)
                        is_volatility_safe = True
                        max_allowed_stop = latest['close'] * (max_stop_perc / 100)
                        stop_distance = latest["atr"] * atr_stop_mult
                        
                        if stop_distance > max_allowed_stop:
                            is_volatility_safe = False
                            
                        # --- FINAL RECOMMENDATION (COMPREHENSIVE) ---
                        
                        recommendation = ""
                        is_actionable = False
                        
                        # Check for STRONG Signals
                        if score >= 8:
                            recommendation = "STRONG BUY"
                        elif score <= -8:
                            recommendation = "STRONG SELL"
                        elif score >= 4:
                            recommendation = "BUY"
                        elif score <= -4:
                            recommendation = "SELL"
                        else:
                            recommendation = "HOLD/NEUTRAL"
                            
                        
                        # Apply All Safeguards to Strong/Actionable Signals (Score >= 6 or <= -6)
                        if abs(score) >= 6:
                            direction = 1 if score > 0 else -1
                            
                            # 1. Daily Trend Filter
                            if (direction == 1 and daily_trend == "BEARISH") or (direction == -1 and daily_trend == "BULLISH"):
                                recommendation = f"{recommendation} (Against Daily Trend: {daily_trend})"
                            
                            # 2. Momentum Decay Filter
                            elif is_momentum_decay:
                                recommendation = f"{recommendation} (Momentum Fading)"

                            # 3. Volatility Filter
                            elif not is_volatility_safe:
                                recommendation = f"{recommendation} (RISK HIGH: ATR Wide)"

                            # 4. Volume Filter
                            elif not is_volume_confirmed:
                                recommendation = f"{recommendation} (Low Volume)"
                                
                            # 5. Price Action Filter
                            elif not is_breakout_confirmed:
                                recommendation = f"{recommendation} (No Breakout)"
                                
                            else:
                                # All checks passed
                                is_actionable = True


                        # --- RISK MANAGEMENT CALCULATIONS (UPDATED) ---
                        stop_loss, take_profit = 'N/A', 'N/A'
                        suggested_quantity = 0
                        risk_per_trade = account_size * (risk_percent / 100)
                        
                        if is_actionable and not pd.isna(latest["atr"]):
                            if score > 0: 
                                stop_loss_price = latest["close"] - stop_distance
                                take_profit_price = latest["close"] + (stop_distance * risk_rr)
                            else: 
                                stop_loss_price = latest["close"] + stop_distance
                                take_profit_price = latest["close"] - (stop_distance * risk_rr)
                                
                            stop_loss = f"{stop_loss_price:.2f}"
                            take_profit = f"{take_profit_price:.2f}"
                            
                            if stop_distance > 0:
                                suggested_quantity = int(risk_per_trade / stop_distance)
                                if suggested_quantity < 1: suggested_quantity = 1


                        # --- DATA OUTPUT ---
                        
                        # Static Score Rule Table (Skipped for brevity, remains the same)
                        # Dynamic Score Card (Updated with Volatility and Momentum Decay flags)
                        score_data = pd.DataFrame({
                            "Component": ["Trend (EMA)", "Slope (Strength)", "MACD (Momentum)", "RSI/BB (Reversion)", "Volume Ratio", "RSI Decay (3-bar)", "Volatility Cap"],
                            "Current Value": [
                                f"{latest['fast_ma']:.2f} / {latest['slow_ma']:.2f}",
                                f"{latest['fast_ma_slope']:.4f}",
                                f"{latest['macd_hist']:.4f}",
                                f"{latest['rsi']:.2f}",
                                f"{volume_ratio:.2f}x",
                                f"{latest['rsi']:.2f} vs {rsi_3_ago:.2f}",
                                f"Max: {max_allowed_stop:.2f}"
                            ],
                            "Safeguard Status": [
                                "", # Not a safeguard
                                "", # Not a safeguard
                                "", # Not a safeguard
                                "", # Not a safeguard
                                "✅" if is_volume_confirmed else "❌",
                                "❌" if is_momentum_decay else "✅",
                                "❌" if not is_volatility_safe else "✅"
                            ]
                        })
                        score_card_placeholder.dataframe(score_data.set_index("Component"), use_container_width=True)

                        # Charting (No change)
                        # ...
                        
                        chart_placeholder.subheader(f"📊 Live Chart for {symbol} ({live_interval} bars)")
                        # [ ... Charting code for fig_live ... ]
                        
                        fig_live = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI"))
                        
                        fig_live.add_trace(go.Candlestick(x=df_live['date'], open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'], name='Price'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                        
                        if is_actionable:
                            fig_live.add_hline(y=float(stop_loss), line_dash="dash", line_color="red", row=1, col=1, annotation_text="SL")
                            fig_live.add_hline(y=float(take_profit), line_dash="dash", line_color="green", row=1, col=1, annotation_text="TP")
                        
                        fig_live.add_hline(y=swing_high, line_dash="dash", line_color="purple", row=1, col=1, annotation_text="Swing High")
                        fig_live.add_hline(y=swing_low, line_dash="dash", line_color="brown", row=1, col=1, annotation_text="Swing Low")

                        fig_live.add_trace(go.Bar(x=df_live['date'], y=df_live['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                        fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                        fig_live.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                        fig_live.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                        fig_live.update_layout(xaxis_rangeslider_visible=False, height=700)
                        chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                        
                        table_placeholder.dataframe(df_live.tail(10), use_container_width=True)
                        rec_placeholder.subheader(f"💡 **{symbol}** Recommendation: **{recommendation}** (Total Score: {score})")
                        
                        # --- SCORE BREAKDOWN & RISK OUTPUT ---
                        targets_placeholder.markdown(f"""
                            **Score Breakdown:** * **Trend (MA/Slope):** **{trend_score}** / $\pm 6$
                            * **Momentum (MACD):** **{momentum_score}** / $\pm 2$
                            * **Reversion (RSI/BB):** **{reversion_score}** / $\pm 2$
                        """)
                        
                        risk_placeholder.markdown(f"""
                            ---
                            **Trade Plan (R:R 1:{risk_rr} | Stop: {atr_stop_mult}x ATR)**
                            * **Current Close Price:** **{latest['close']:.2f}**
                            * **High-Timeframe Trend (Daily 20/50 EMA):** **{daily_trend}**
                            * **Suggested Stop-Loss:** **{stop_loss}**
                            * **Suggested Take-Profit:** **{take_profit}**
                        """)
                        
                        if is_actionable:
                            risk_placeholder.markdown(f"""
                                **Position Sizing (1% Risk):**
                                * **Risk Amount:** ₹ {risk_per_trade:.2f}
                                * **Stop Distance:** ₹ {stop_distance:.2f} (Max Allowed: ₹ {max_allowed_stop:.2f})
                                * **Max Quantity:** **{suggested_quantity}** shares
                            """)
                        else:
                            risk_placeholder.markdown("⚠️ **Actionable Trade:** Requires a high score ($|\text{Score}| \ge 6$) **AND** confirmation from all safeguards to calculate actionable targets.")


                    except Exception as e:
                        st.error(f"Error during live data analysis loop: {e}")
                        st.session_state["live_running"] = False
                        # st.rerun() # Commented out to prevent aggressive rerunning on error

                    time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
            st.session_state["live_running"] = False
