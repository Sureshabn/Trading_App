import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import numpy as np
import pytz

# Set the Indian Standard Time (IST) timezone
IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
# Updated Title to reflect the latest version with MTFA and Adaptive Risk
st.title("📈 Zerodha Stock Analysis & Risk Manager (Pro V12 - Historical & MTFA)")

# ---- Zerodha API credentials from Streamlit Secrets ----
try:
    API_KEY = st.secrets["API_KEY"]
    API_SECRET = st.secrets["API_SECRET"]
except KeyError:
    st.error("⚠️ Error: API_KEY or API_SECRET not found in Streamlit secrets.")
    st.stop()

kite = KiteConnect(api_key=API_KEY)

# ---- Session State ----
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


# ----------------------------------------------------------------------
# ---- Configuration Parameters (Common to both sections) ----
# ----------------------------------------------------------------------
if st.session_state["access_token"]:
    st.subheader("⚙️ Configuration")

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

    colA, colB, colC, colD = st.columns(4)
    with colA:
        account_size = st.number_input("Account Size (₹)", min_value=1000, value=100000, step=1000)
    with colB:
        risk_percent = st.number_input("Max Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with colC:
        atr_stop_mult = st.number_input("ATR Stop Multiplier", min_value=1.0, value=2.0, step=0.5)
    with colD:
        volume_conf_mult = st.number_input("Volume Conf. Multiplier (e.g., 1.5x ATR Vol)", min_value=1.0, value=1.5, step=0.1)
    
    swing_lookback = st.number_input("Swing Lookback Periods (for S/R)", min_value=5, value=10)
    
    macro_risk_override = st.checkbox(
        "Apply **Macro/Global Risk Override**",
        value=False,
        help="Downgrade signals due to poor Nifty/Global setup or upcoming key data (e.g., Fed/RBI)."
    )
    company_news_override = st.checkbox(
        f"Apply **{symbol} News Override**",
        value=False,
        help=f"Downgrade signals for {symbol} due to company-specific negative news or pending major announcements (e.g., results)."
    )

    # ---- Indicator Calculation Function (Updated for Daily Trend) ----
    @st.cache_data(ttl=600) 
    def calculate_indicators(df, fast_w, slow_w, rsi_w):
        # Ensure correct data types and sort
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df = df.sort_values("date")
        
        # EMAs and Slope
        df["fast_ma"] = ta.trend.EMAIndicator(df["close"], window=fast_w).ema_indicator()
        df["slow_ma"] = ta.trend.EMAIndicator(df["close"], window=slow_w).ema_indicator()
        df["fast_ma_slope"] = (df["fast_ma"] - df["fast_ma"].shift(3)) / 3
        
        # RSI, MACD, BB, ATR
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff() 
        df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=rsi_w).rsi()
        boll = ta.volatility.BollingerBands(df["close"])
        df["bb_high"] = boll.bollinger_hband()
        df["bb_low"] = boll.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=rsi_w).average_true_range()

        # Volume Average (ATR Volume)
        df["atr_volume"] = df["volume"].rolling(window=rsi_w).mean()

        return df

    # ---- Charting Function (Reused for both Historical and Live) ----
    def plot_data(df, fast_w, slow_w, swing_high=None, swing_low=None, ltp=None, sl=None, tp=None, interval_label=""):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7,0.3], subplot_titles=(f"{symbol} Price Action ({interval_label})", "MACD & RSI"))
        
        # CANDLESTICK AND EMAS
        fig.add_trace(go.Candlestick(x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_w})'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_w})'), row=1, col=1)
        
        # Add Live LTP marker and line
        if ltp is not None:
            # Get latest date/time for plotting LTP
            latest_time = df['date'].iloc[-1].tz_convert(IST) if df['date'].iloc[-1].tzinfo is not None else df['date'].iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[latest_time], 
                y=[ltp], 
                mode='markers', 
                name='LTP (Live)', 
                marker=dict(size=10, color='lime', symbol='diamond')), 
                row=1, col=1)
            
            fig.add_hline(
                y=ltp, 
                line_dash="solid", 
                line_color="lime", 
                line_width=1, 
                row=1, col=1, 
                annotation_text=f"LTP: {ltp:.2f}", 
                annotation_position="top left"
            )

        # Visualize SL/TP and S/R
        if sl: fig.add_hline(y=float(sl), line_dash="dash", line_color="red", row=1, col=1, annotation_text="SL")
        if tp: fig.add_hline(y=float(tp), line_dash="dash", line_color="green", row=1, col=1, annotation_text="TP")
        if swing_high: fig.add_hline(y=swing_high, line_dash="dash", line_color="purple", row=1, col=1, annotation_text=f"Resistance ({swing_lookback})")
        if swing_low: fig.add_hline(y=swing_low, line_dash="dash", line_color="brown", row=1, col=1, annotation_text=f"Support ({swing_lookback})")

        # MACD AND RSI
        fig.add_trace(go.Bar(x=df['date'], y=df['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(xaxis_rangeslider_visible=False, height=700)
        return fig

    # --- Analysis Logic Function (for both Historical and Live) ---
    def run_analysis_logic(df_valid, df_daily, ltp, symbol, fast_ema_w, slow_ema_w, rsi_w, swing_lookback, volume_conf_mult, risk_rr, atr_stop_mult, account_size, risk_percent, macro_risk_override, company_news_override, interval_label):
        
        latest = df_valid.iloc[-1]
        prev = df_valid.iloc[-2] if len(df_valid) >= 2 else None
        current_price_for_analysis = ltp if ltp is not None else latest["close"] 
        entry_price = current_price_for_analysis # Use this for SL/TP calculation

        
        # --- MTFA: Daily Trend ---
        if df_daily.empty or len(df_daily) < slow_ema_w:
            daily_trend = "UNKNOWN"
        else:
            latest_daily = df_daily.iloc[-1]
            if latest_daily["fast_ma"] > latest_daily["slow_ma"]:
                daily_trend = "BULLISH"
            elif latest_daily["fast_ma"] < latest_daily["slow_ma"]:
                daily_trend = "BEARISH"
            else:
                daily_trend = "NEUTRAL"

        # --- CORE SCORING LOGIC (Intraday) ---
        score = 0
        is_bullish_trend = False
        is_bearish_trend = False
        trend_score = 0
        momentum_score = 0
        reversion_score = 0
        flag_ma_cross_up, flag_ma_cross_down, flag_slope_pos, flag_slope_neg, flag_macd_bull, flag_macd_bear, flag_rsi_bull, flag_rsi_bear = [False] * 8
        
        # A. TREND (MAs)
        if latest["fast_ma"] > latest["slow_ma"]:
            score += 4; trend_score += 4; is_bullish_trend = True; flag_ma_cross_up = True
            if latest["fast_ma_slope"] > 0.001: 
                score += 2; trend_score += 2; flag_slope_pos = True
        elif latest["fast_ma"] < latest["slow_ma"]:
            score -= 4; trend_score -= 4; is_bearish_trend = True; flag_ma_cross_down = True
            if latest["fast_ma_slope"] < -0.001: 
                score -= 2; trend_score -= 2; flag_slope_neg = True

        # B. MOMENTUM (MACD)
        if latest["macd"] > latest["macd_signal"] and latest["macd_hist"] > 0:
            score += 2; momentum_score += 2; flag_macd_bull = True
        elif latest["macd"] < latest["macd_signal"] and latest["macd_hist"] < 0:
            score -= 2; momentum_score -= 2; flag_macd_bear = True
        else:
            if latest["macd"] > latest["macd_signal"]: momentum_score += 1 
            elif latest["macd"] < latest["macd_signal"]: momentum_score -= 1


        # C. REVERSION/EXTREMES (RSI/BB)
        if prev is not None:
            if is_bullish_trend:
                if latest["rsi"] < 50 and latest["rsi"] > prev["rsi"]:
                    score += 1; reversion_score += 1; flag_rsi_bull = True
                if latest["close"] < latest["bb_low"] and latest["close"] > prev["close"]:
                    score += 1; reversion_score += 1
            elif is_bearish_trend:
                if latest["rsi"] > 50 and latest["rsi"] < prev["rsi"]:
                    score -= 1; reversion_score -= 1; flag_rsi_bear = True
                if latest["close"] > latest["bb_high"] and latest["close"] < prev["close"]:
                    score -= 1; reversion_score -= 1
                    
        # --- ADVANCED SAFEGURADS & CONFIRMATION LOGIC ---
        df_lookback = df_valid.iloc[-(swing_lookback + 1):-1]
        swing_high = df_lookback["high"].max() if not df_lookback.empty else latest["high"]
        swing_low = df_lookback["low"].min() if not df_lookback.empty else latest["low"]
        
        is_volume_confirmed = False
        volume_ratio = 0.0
        if latest["atr_volume"] > 0:
            volume_ratio = latest["volume"] / latest["atr_volume"]
            if volume_ratio >= volume_conf_mult:
                is_volume_confirmed = True

        is_breakout_confirmed = False
        if is_bullish_trend and current_price_for_analysis > swing_high:
            is_breakout_confirmed = True
        elif is_bearish_trend and current_price_for_analysis < swing_low:
            is_breakout_confirmed = True
            
        # --- LOW VOLATILITY FILTER ---
        atr_series = df_valid["atr"].tail(50).sort_values()
        low_vol_threshold = atr_series.iloc[int(len(atr_series) * 0.20)] if len(atr_series) >= 5 else 0
        is_low_volatility = (latest["atr"] <= low_vol_threshold) if low_vol_threshold > 0 else False
        
        
        # --- FINAL RECOMMENDATION AND ADAPTIVE SCORING ---
        recommendation = (
            "STRONG BUY" if score >= 8 else 
            "BUY" if score >= 4 else  
            "HOLD/NEUTRAL" if score > -4 else 
            "SELL" if score > -8 else  
            "STRONG SELL"            
        )

        # A. MTFA Trend Alignment Filter
        is_trend_aligned = (daily_trend == "BULLISH" and is_bullish_trend) or \
                            (daily_trend == "BEARISH" and is_bearish_trend)
        
        if not is_trend_aligned and recommendation in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
            recommendation += " (Counter-Trend/Unconfirmed)"
            # Set score to 0 to prevent position sizing/trade recommendation
            effective_score = 0 
        else:
            effective_score = score
            
        # B. Low Volatility Filter
        if is_low_volatility and recommendation.startswith(("STRONG BUY", "BUY", "STRONG SELL", "SELL")):
                recommendation = "HOLD (Low Volatility Filter)"
                effective_score = 0

        # C. Macro/Fundamental Override
        is_overridden = False
        if (macro_risk_override or company_news_override) and recommendation.startswith(("STRONG BUY", "BUY", "STRONG SELL", "SELL", "HOLD (Low Volatility")):
            recommendation = "HOLD/NEUTRAL (Macro/News Override)"
            is_overridden = True
            effective_score = 0
        
        # D. Final Technical Confirmation Check
        final_actionable = False
        if not is_overridden and not is_low_volatility and is_trend_aligned:
            
            # Use effective_score for final check
            if effective_score >= 6: # Buy signal
                if not is_volume_confirmed or not is_breakout_confirmed:
                    recommendation = recommendation.split(" ")[0] + " (Wait for V/B)"
                else:
                    final_actionable = True
            
            elif effective_score <= -6: # Sell signal
                if not is_volume_confirmed or not is_breakout_confirmed:
                    recommendation = recommendation.split(" ")[0] + " (Wait for V/B)"
                else:
                    final_actionable = True


        # --- RISK MANAGEMENT CALCULATIONS (Adaptive Risk) ---
        stop_loss, take_profit = None, None
        suggested_quantity = 0
        
        effective_risk_percent = risk_percent
        # Reduce risk by half for non-aligned or unconfirmed high-score trades
        if not is_trend_aligned or not (is_volume_confirmed and is_breakout_confirmed) and effective_score != 0:
            effective_risk_percent = risk_percent / 2
        
        risk_per_trade = account_size * (effective_risk_percent / 100)
        stop_distance = latest["atr"] * atr_stop_mult
        
        if final_actionable and not pd.isna(latest["atr"]) and stop_distance > 0:
            suggested_quantity = int(risk_per_trade / stop_distance)
            if suggested_quantity < 1: suggested_quantity = 1

            if score >= 6: # Buy signal (score used here for direction, final_actionable ensures quality)
                stop_loss_price = entry_price - stop_distance
                take_profit_price = entry_price + (stop_distance * risk_rr)
                stop_loss = f"{stop_loss_price:.2f}"
                take_profit = f"{take_profit_price:.2f}"
            elif score <= -6: # Sell signal
                stop_loss_price = entry_price + stop_distance
                take_profit_price = entry_price - (stop_distance * risk_rr)
                stop_loss = f"{stop_loss_price:.2f}"
                take_profit = f"{take_profit_price:.2f}"
        
        
        # --- CONSOLE OUTPUTS (Simplified for Historical/One-time analysis) ---

        st.markdown("---")
        st.subheader(f"💡 **{symbol}** Recommendation: **{recommendation}** (Total Score: {score})")
        st.caption(f"Calculations based on **{interval_label}** candle closed at: **{latest['date'].tz_convert(IST).strftime('%Y-%m-%d %H:%M:%S') if latest['date'].tzinfo is not None else latest['date'].strftime('%Y-%m-%d %H:%M:%S')}** (Price: ₹{current_price_for_analysis:.2f})")

        st.markdown("### 🚦 Analysis Filters")
        st.markdown(f"""
            * **Macro Trend (Daily EMA):** **{daily_trend}** ({'✅ Aligned' if is_trend_aligned else '❌ Counter-Trend'}).
            * **Low Volatility Filter (ATR $\le$ {low_vol_threshold:.4f}):** **{'🔴 ACTIVE' if is_low_volatility else '✅ OK'}**.
            * **Macro/News Override:** **{'🔴 ACTIVE' if is_overridden else '✅ INACTIVE'}**.
        """)

        st.markdown("### 📊 Trade Plan Summary")
        if final_actionable:
            st.success(f"✅ **ACTIONABLE TRADE:** High-Conviction setup with **{effective_risk_percent:.2f}%** Risk.")
            st.markdown(f"""
                * **Risk Per Trade:** ₹ {risk_per_trade:.2f}
                * **Stop Distance:** ₹ {stop_distance:.2f} (ATR Multiplier: {atr_stop_mult}x)
                * **Max Quantity:** **{suggested_quantity}** shares
                * **Stop-Loss (SL):** **{stop_loss}**
                * **Take-Profit (TP):** **{take_profit}** (R:R 1:{risk_rr})
            """)
        else:
            st.warning("Trade is currently **NON-ACTIONABLE**. Check filters and confirmation status.")
            st.markdown(f"""
                **Confirmation Status:**
                * **Volume $\ge {volume_conf_mult:.1f}$x:** **{'✅' if is_volume_confirmed else '❌'}** (Ratio: {volume_ratio:.2f}x)
                * **Breakout (vs S/R):** **{'✅' if is_breakout_confirmed else '❌'}** (Swing High: {swing_high:.2f}, Swing Low: {swing_low:.2f})
            """)
            st.markdown(f"**Potential SL/TP:** SL: **{stop_loss if stop_loss else 'N/A'}** | TP: **{take_profit if take_profit else 'N/A'}**")
        
        st.markdown("---")
        return swing_high, swing_low, stop_loss, take_profit, entry_price


    # ----------------------------------------------------------------------
    # ---- Historical Data Analysis Section (Simple Daily & Advanced Options) ----
    # ----------------------------------------------------------------------
    st.header("🕰️ Historical Data Analysis")

    # --- Simple Daily Historical Analysis (Button as requested in image) ---
    st.subheader("1. Simple Daily Chart (No MTFA/Risk Logic)")
    daily_start_date = st.date_input("Daily Chart Start Date", datetime.now() - timedelta(days=60)) # Default to 60 days
    daily_end_date = st.date_input("Daily Chart End Date", datetime.today())

    # BUTTON 1: "Run Historical Analysis (Daily Timeframe)"
    if st.button("Run Historical Analysis (Daily Timeframe)"): 
        try:
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE")
            else:
                token = int(row.iloc[0]["instrument_token"])
                hist = kite.historical_data(token, daily_start_date, daily_end_date, interval="day")
                df_hist_daily_simple = pd.DataFrame(hist)

                if df_hist_daily_simple.empty:
                    st.warning("⚠️ No historical data available for the selected range.")
                else:
                    df_hist_daily_simple = calculate_indicators(df_hist_daily_simple, fast_ema_w, slow_ema_w, rsi_w)
                    df_hist_daily_simple_valid = df_hist_daily_simple.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])

                    st.subheader(f"📈 Historical ({symbol}) Daily Candlestick with EMAs & Oscillators")
                    # Plot using the generic plot_data function
                    fig_simple_daily = plot_data(df_hist_daily_simple_valid, fast_ema_w, slow_ema_w, interval_label="Daily")
                    st.plotly_chart(fig_simple_daily, use_container_width=True)
                    st.dataframe(df_hist_daily_simple_valid.tail(10))

        except Exception as e:
            st.error(f"Error fetching simple daily historical data: {e}")

    st.markdown("---")

    # --- Advanced Historical Analysis with MTFA Logic (Necessary for testing all features) ---
    st.subheader("2. Advanced Historical Analysis (with MTFA & Filters)")
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        h_start_date = st.date_input("Analysis Start Date", datetime.now() - timedelta(days=10))
    with h_col2:
        h_end_date = st.date_input("Analysis End Date", datetime.today())
    with h_col3:
        h_interval = st.selectbox("Analysis Bar Interval", ["60minute", "30minute", "15minute", "5minute", "3minute", "day"]) # Added 'day' here too

    if st.button("Run Advanced Historical Analysis & Chart"):
        
        # --- Data Fetching ---
        try:
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]
            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE.")
                st.stop()
            token = int(row.iloc[0]["instrument_token"])
            
            # 1. Fetch Main Historical Data for selected interval
            hist_data = kite.historical_data(token, h_start_date, h_end_date, interval=h_interval)
            df_hist = pd.DataFrame(hist_data)
            
            # Ensure enough bars for calculations
            if df_hist.empty or len(df_hist) < slow_ema_w * 2:
                 st.error(f"⚠️ Not enough {h_interval} data for robust indicator calculation. Need at least {slow_ema_w * 2} bars.")
                 st.stop()

            df_hist = calculate_indicators(df_hist, fast_ema_w, slow_ema_w, rsi_w)
            df_hist_valid = df_hist.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])

            # 2. Fetch Daily Data for MTFA (Contextual) - Increased lookback to 200 days
            macro_start_h = datetime.combine(h_end_date - timedelta(days=200), datetime.min.time()) 
            hist_daily_h = kite.historical_data(token, macro_start_h, h_end_date, interval="day")
            df_daily_h = pd.DataFrame(hist_daily_h)
            
            if df_daily_h.empty or len(df_daily_h) < slow_ema_w:
                st.warning("⚠️ Not enough daily data for robust macro trend calculation in historical analysis. Daily filter will be UNKNOWN.")
            df_daily_h = calculate_indicators(df_daily_h, fast_ema_w, slow_ema_w, rsi_w)


            # --- Analysis Execution ---
            st.markdown("---")
            st.header(f"Results for **{symbol}** (Latest Bar: {h_interval})")
            
            # Use the last close price of the historical data as the 'LTP' for analysis
            historical_ltp_for_analysis = df_hist_valid.iloc[-1]['close']

            swing_high, swing_low, stop_loss, take_profit, _ = run_analysis_logic(
                df_hist_valid, df_daily_h, historical_ltp_for_analysis, symbol, fast_ema_w, slow_ema_w, rsi_w, swing_lookback, volume_conf_mult, risk_rr, atr_stop_mult, account_size, risk_percent, macro_risk_override, company_news_override, h_interval
            )
            
            # --- Charting ---
            fig_hist = plot_data(df_hist_valid, fast_ema_w, slow_ema_w, swing_high=swing_high, swing_low=swing_low, ltp=historical_ltp_for_analysis, sl=stop_loss, tp=take_profit, interval_label=h_interval)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.subheader("Raw Data (Tail)")
            st.dataframe(df_hist_valid.tail(10))


        except Exception as e:
            st.error(f"Error fetching historical data: {e}")

    st.markdown("---")
    # ----------------------------------------------------------------------
    # ---- Live Analysis Section (Kept as before) ----
    # ----------------------------------------------------------------------
    
    st.header("⚡ Live Bar-Close Analysis (Auto-Refresh)")
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Live Bar Interval", ["5minute", "15minute", "30minute"])


    # BUTTON 2: "Start Intraday Bar-Close Analysis"
    if st.button("Start Intraday Bar-Close Analysis"): 
        st.session_state["live_running"] = True
    
    # Add a stop button to the live analysis section
    if st.session_state["live_running"]:
        if st.button("Stop Live Intraday Analysis"):
            st.session_state["live_running"] = False
            st.success("Live analysis stopped.")
            st.rerun() 
            
    if st.session_state["live_running"]:
        
        # Initialize placeholders once
        st.markdown("---")
        st.subheader(f"🔴 Live Analysis Running for {symbol} ({live_interval})")
        
        chart_placeholder = st.empty()
        table_placeholder = st.empty()
        debug_placeholder = st.sidebar.empty()

        try:
            # --- Common Setup ---
            instruments = kite.instruments("NSE")
            df_instruments = pd.DataFrame(instruments)
            row = df_instruments[df_instruments["tradingsymbol"] == symbol]

            if row.empty:
                st.error(f"❌ Symbol {symbol} not found on NSE. Stopping live analysis.")
                st.session_state["live_running"] = False
                st.stop()
            
            token = int(row.iloc[0]["instrument_token"])

            while st.session_state["live_running"]:
                try:
                    # --- 1. MTFA: Fetch Daily Data for Macro Trend ---
                    macro_start = datetime.combine(date.today() - timedelta(days=200), datetime.min.time()) # Increased to 200 days
                    hist_daily = kite.historical_data(token, macro_start, date.today(), interval="day")
                    df_daily = pd.DataFrame(hist_daily)
                    
                    if df_daily.empty or len(df_daily) < slow_ema_w:
                        st.warning("⚠️ Not enough daily data for robust macro trend calculation. Daily filter will be UNKNOWN.")
                    df_daily = calculate_indicators(df_daily, fast_ema_w, slow_ema_w, rsi_w)


                    # --- 2. Fetch Intraday Data for Live Signal ---
                    intraday_start = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())
                    end_time = datetime.now() 

                    hist = kite.historical_data(token, intraday_start, end_time, interval=live_interval)
                    df_live = pd.DataFrame(hist)
                    
                    # Fetch Live Quote (LTP)
                    ltp = None
                    quotes = kite.quote([f'NSE:{symbol}'])
                    if f'NSE:{symbol}' in quotes:
                        ltp = quotes[f'NSE:{symbol}']['last_price']
                        entry_price = ltp
                    else:
                        entry_price = df_live.iloc[-1]['close'] # Use last closed price if LTP fails


                    if df_live.empty:
                        time.sleep(refresh_interval)
                        continue

                    df_live = calculate_indicators(df_live, fast_ema_w, slow_ema_w, rsi_w)
                    df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])
                    
                    if len(df_valid) < 2:
                        time.sleep(refresh_interval)
                        continue

                    
                    # --- RUN ANALYSIS LOGIC ---
                    swing_high, swing_low, stop_loss, take_profit, current_entry = run_analysis_logic(
                        df_valid, df_daily, ltp, symbol, fast_ema_w, slow_ema_w, rsi_w, swing_lookback, volume_conf_mult, risk_rr, atr_stop_mult, account_size, risk_percent, macro_risk_override, company_news_override, live_interval
                    )
                    
                    # --- CHARTING ---
                    fig_live = plot_data(df_valid, fast_ema_w, slow_ema_w, swing_high=swing_high, swing_low=swing_low, ltp=ltp, sl=stop_loss, tp=take_profit, interval_label=live_interval)
                    chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                    # Display the tail of the data
                    st.markdown("### Raw Data (Tail)")
                    table_placeholder.dataframe(df_valid.tail(10), use_container_width=True)

                except Exception as e:
                    st.error(f"Error during live data analysis loop: {e}. Stopping live analysis.")
                    st.session_state["live_running"] = False
                    st.exception(e)
                    
                    if st.button("Resume Live Analysis"):
                        st.session_state["live_running"] = True
                        st.rerun()

                time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
            st.session_state["live_running"] = False
