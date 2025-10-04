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
st.title("📈 Zerodha Stock Analysis & Risk Manager (Pro V10 - Advanced MTFA & Structure)")

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
        risk_percent = st.number_input("Max Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    with colC:
        atr_stop_mult = st.number_input("ATR Stop Multiplier", min_value=1.0, value=2.0, step=0.5)
    
    colD, colE, colF, colG = st.columns(4)
    with colD:
        # Volume Confirmation Threshold
        volume_conf_mult = st.number_input("Volume Conf. Multiplier (e.g., 1.5x ATR Vol)", min_value=1.0, value=1.5, step=0.1)
    with colE:
        # Swing High/Low Lookback
        swing_lookback = st.number_input("Swing Lookback Periods (for S/R)", min_value=5, value=10)
    
    # --- MACRO/FUNDAMENTAL SAFEGURAD INPUTS (Kept from previous version) ---
    with colF:
        macro_risk_override = st.checkbox(
            "Apply **Macro/Global Risk Override**",
            value=False,
            help="Downgrade signals due to poor Nifty/Global setup or upcoming key data (e.g., Fed/RBI)."
        )
    with colG:
        company_news_override = st.checkbox(
            f"Apply **{symbol} News Override**",
            value=False,
            help=f"Downgrade signals for {symbol} due to company-specific negative news or pending major announcements (e.g., results)."
        )
    # --- END SAFEGURAD INPUTS ---


    start_date = st.date_input("Historical Start Date", datetime(2022,1,1))
    end_date = st.date_input("Historical End Date", datetime.today())
    # Recommended minimum is 30, increased to 60 for stability
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Intraday Bar Interval", ["5minute", "15minute", "30minute"])


    # ---- Indicator Calculation Function (Updated for Daily Trend) ----
    # Cache based on symbol, fast_w, slow_w, rsi_w, and the interval (which will change for Daily vs Intraday)
    @st.cache_data(ttl=600) 
    def calculate_indicators(df, fast_w, slow_w, rsi_w):
        # Ensure correct data types
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

    # ---- Live Analysis ----
    if st.button("Start Intraday Bar-Close Analysis"):
        st.session_state["live_running"] = True

    if st.session_state["live_running"]:
        
        # Initialize placeholders once
        chart_placeholder = st.empty()
        table_placeholder = st.empty()
        rec_placeholder = st.empty()
        targets_placeholder = st.empty() 
        risk_placeholder = st.empty() 
        score_rules_placeholder = st.empty()
        score_card_placeholder = st.empty()
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
                    # We need enough history for Daily EMAs (e.g., 50 days)
                    macro_start = datetime.combine(date.today() - timedelta(days=90), datetime.min.time()) 
                    hist_daily = kite.historical_data(token, macro_start, date.today(), interval="day")
                    df_daily = pd.DataFrame(hist_daily)
                    
                    if df_daily.empty or len(df_daily) < slow_ema_w:
                        st.warning("⚠️ Not enough daily data for robust macro trend calculation. Skipping MTFA filter.")
                        daily_trend = "UNKNOWN"
                    else:
                        df_daily = calculate_indicators(df_daily, fast_ema_w, slow_ema_w, rsi_w)
                        latest_daily = df_daily.iloc[-1]
                        if latest_daily["fast_ma"] > latest_daily["slow_ma"]:
                            daily_trend = "BULLISH"
                        elif latest_daily["fast_ma"] < latest_daily["slow_ma"]:
                            daily_trend = "BEARISH"
                        else:
                            daily_trend = "NEUTRAL"


                    # --- 2. Fetch Intraday Data for Live Signal ---
                    # Go back 5 days for robust intraday indicator calculation
                    intraday_start = datetime.combine(date.today() - timedelta(days=5), datetime.min.time())
                    end_time = datetime.now() 

                    debug_placeholder.caption(f"Fetching {live_interval} data for Token: {token}\nDaily Trend: {daily_trend}")

                    hist = kite.historical_data(token, intraday_start, end_time, interval=live_interval)
                    df_live = pd.DataFrame(hist)
                    
                    # Fetch Live Quote (LTP)
                    ltp = None
                    quotes = kite.quote([f'NSE:{symbol}'])
                    if f'NSE:{symbol}' in quotes:
                        ltp = quotes[f'NSE:{symbol}']['last_price']

                    if df_live.empty:
                        st.warning(f"⚠️ No closed {live_interval} data available for {symbol}. Retrying in {refresh_interval}s...")
                        time.sleep(refresh_interval)
                        continue

                    df_live = calculate_indicators(df_live, fast_ema_w, slow_ema_w, rsi_w)

                    # Filter for rows where all essential indicators are calculated
                    df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])
                    
                    if len(df_valid) < 2:
                        st.warning("⚠️ Not enough history available to calculate all indicators. Waiting for more closed bars.")
                        time.sleep(refresh_interval)
                        continue

                    # Use the latest **complete** candle for all calculations
                    latest = df_valid.iloc[-1]
                    prev = df_valid.iloc[-2] 
                    latest_candle_time = latest['date'].tz_convert(IST).strftime('%Y-%m-%d %H:%M:%S') if latest['date'].tzinfo is not None else latest['date'].strftime('%Y-%m-%d %H:%M:%S')
                    current_price_for_analysis = ltp if ltp is not None else latest["close"] 


                    # --- 3. CORE SCORING LOGIC (Intraday) ---
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
                                
                    # --- 4. ADVANCED SAFEGURADS & CONFIRMATION LOGIC ---
                    
                    df_lookback = df_valid.iloc[-swing_lookback-1:-1]
                    swing_high = df_lookback["high"].max() if not df_lookback.empty else latest["high"]
                    swing_low = df_lookback["low"].min() if not df_lookback.empty else latest["low"]
                    
                    is_volume_confirmed = False
                    volume_ratio = 0.0
                    if latest["atr_volume"] > 0:
                        volume_ratio = latest["volume"] / latest["atr_volume"]
                        if volume_ratio >= volume_conf_mult:
                            is_volume_confirmed = True

                    is_breakout_confirmed = False
                    # Check for price action breakout using the most recent price
                    if is_bullish_trend and current_price_for_analysis > swing_high:
                        is_breakout_confirmed = True
                    elif is_bearish_trend and current_price_for_analysis < swing_low:
                        is_breakout_confirmed = True
                        
                    # --- NEW: LOW VOLATILITY FILTER ---
                    # Check if current ATR is in the bottom 20% of the last 50 ATR readings
                    atr_series = df_valid["atr"].tail(50).sort_values()
                    low_vol_threshold = atr_series.iloc[int(len(atr_series) * 0.20)] if len(atr_series) >= 5 else 0
                    is_low_volatility = (latest["atr"] <= low_vol_threshold) if low_vol_threshold > 0 else False
                    
                    
                    # --- 5. FINAL RECOMMENDATION AND ADAPTIVE SCORING ---
                    
                    # Base Recommendation
                    recommendation = (
                        "STRONG BUY" if score >= 8 else 
                        "BUY" if score >= 4 else  
                        "HOLD/NEUTRAL" if score > -4 else 
                        "SELL" if score > -8 else  
                        "STRONG SELL"            
                    )

                    # --- Apply Filters & Adaptive Adjustments ---
                    
                    # A. MTFA Trend Alignment Filter
                    is_trend_aligned = (daily_trend == "BULLISH" and is_bullish_trend) or \
                                       (daily_trend == "BEARISH" and is_bearish_trend)
                    
                    if not is_trend_aligned and recommendation in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
                        recommendation += " (Counter-Trend/Unconfirmed)"
                        # Reduce score for counter-trend signals to reflect lower conviction
                        score = 0 # Effectively neutralizes the trade
                        
                    # B. Low Volatility Filter (Highest Priority Technical Filter)
                    if is_low_volatility and recommendation in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
                         recommendation = "HOLD (Low Volatility Filter)"
                         score = 0 # Neutralize all scores in choppy markets


                    # C. Macro/Fundamental Override (Highest Priority Overall)
                    is_overridden = False
                    if (macro_risk_override or company_news_override) and recommendation.startswith(("STRONG BUY", "BUY", "STRONG SELL", "SELL", "HOLD (Low Volatility")):
                        recommendation = "HOLD/NEUTRAL (Macro/News Override)"
                        is_overridden = True
                    
                    # D. Final Technical Confirmation Check (Only if not overridden or filtered)
                    final_actionable = False
                    if not is_overridden and not is_low_volatility and is_trend_aligned:
                        if recommendation in ["STRONG BUY", "BUY"] and score >= 6:
                            if not is_volume_confirmed or not is_breakout_confirmed:
                                recommendation = recommendation.split(" ")[0] + " (Wait for V/B)" # Simplified confirmation message
                            else:
                                final_actionable = True
                        
                        elif recommendation in ["STRONG SELL", "SELL"] and score <= -6:
                            if not is_volume_confirmed or not is_breakout_confirmed:
                                recommendation = recommendation.split(" ")[0] + " (Wait for V/B)"
                            else:
                                final_actionable = True


                    # --- 6. RISK MANAGEMENT CALCULATIONS (Adaptive Risk) ---
                    stop_loss, take_profit = None, None
                    suggested_quantity = 0
                    
                    # Adaptive Risk Logic
                    effective_risk_percent = risk_percent
                    if not is_trend_aligned:
                         # Halve risk for counter-trend or unaligned signals
                        effective_risk_percent = risk_percent / 2
                    
                    risk_per_trade = account_size * (effective_risk_percent / 100)
                    stop_distance = latest["atr"] * atr_stop_mult
                    entry_price = current_price_for_analysis 
                    
                    # Only calculate SL/TP if the trade is actionable AND not filtered/overridden
                    if final_actionable and not pd.isna(latest["atr"]) and stop_distance > 0:
                        suggested_quantity = int(risk_per_trade / stop_distance)
                        if suggested_quantity < 1: suggested_quantity = 1

                        if score >= 6: # Buy signal
                            stop_loss_price = entry_price - stop_distance
                            take_profit_price = entry_price + (stop_distance * risk_rr)
                            stop_loss = f"{stop_loss_price:.2f}"
                            take_profit = f"{take_profit_price:.2f}"
                        elif score <= -6: # Sell signal
                            stop_loss_price = entry_price + stop_distance
                            take_profit_price = entry_price - (stop_distance * risk_rr)
                            stop_loss = f"{stop_loss_price:.2f}"
                            take_profit = f"{take_profit_price:.2f}"

                    
                    # --- CHART AND DATA OUTPUT ---
                    
                    with score_rules_placeholder.container():
                        st.markdown("### 🚦 Score Matrix Rules & Thresholds")
                        st.table(pd.DataFrame({
                            "Component": ["EMA Cross", "EMA Slope", "MACD Histogram", "RSI/BB Pullback"],
                            "Points (+/-)": [4, 2, 2, 1],
                            "Total Max Points": [6, 6, 2, 2]
                        }))
                        st.markdown(f"""
                            **FILTERS APPLIED:** * **Macro Trend:** {daily_trend} (Intraday signal must align or score is neutralized).
                            * **Volatility:** {'🔴 Low Volatility Filter Active' if is_low_volatility else '✅ Volatility OK'} (Threshold: ATR $\le$ {low_vol_threshold:.4f}).
                            * **Override:** {'🔴 ACTIVE' if is_overridden else '✅ INACTIVE'}.
                        """)
                    
                    # Dynamic Score Card 
                    score_card_placeholder.dataframe(pd.DataFrame({
                        "Component": ["Daily Trend", "Intraday Trend (MA)", "EMA Slope (Strength)", "MACD Hist (Momentum)", "RSI/BB (Reversion)", "Volume Ratio", "Breakout Conf."],
                        "Current Value": [
                            daily_trend,
                            f"{latest['fast_ma']:.2f} / {latest['slow_ma']:.2f}",
                            f"{latest['fast_ma_slope']:.4f}",
                            f"{latest['macd_hist']:.4f}",
                            f"{latest['rsi']:.2f}",
                            f"{volume_ratio:.2f}x (Req: {volume_conf_mult:.1f}x)",
                            f"Price {current_price_for_analysis:.2f} vs S/R {swing_high:.2f}/{swing_low:.2f}"
                        ],
                        "Signal Status": [
                            "✅" if is_trend_aligned or daily_trend == "NEUTRAL" else "❌ (Counter)",
                            "✅" if flag_ma_cross_up else ("❌" if flag_ma_cross_down else "-"),
                            "✅" if flag_slope_pos else ("❌" if flag_slope_neg else "-"),
                            "✅" if flag_macd_bull else ("❌" if flag_macd_bear else "-"),
                            "✅" if reversion_score != 0 else "-",
                            "✅" if is_volume_confirmed else "❌",
                            "✅" if is_breakout_confirmed else "❌"
                        ],
                        "Points": [
                            "N/A",
                            f"{4 if flag_ma_cross_up else (-4 if flag_ma_cross_down else 0)}",
                            f"{2 if flag_slope_pos else (-2 if flag_slope_neg else 0)}",
                            f"{2 if flag_macd_bull else (-2 if flag_macd_bear else 0)}",
                            f"{1 if reversion_score > 0 else (-1 if reversion_score < 0 else 0)}",
                            "N/A",
                            "N/A"
                        ]
                    }).set_index("Component"), use_container_width=True)

                    
                    # Live Chart (Visualizing SL/TP and S/R) - Charting logic remains same
                    fig_live = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7,0.3], subplot_titles=("Price", "MACD & RSI"))
                    
                    # CANDLESTICK AND EMAS
                    fig_live.add_trace(go.Candlestick(x=df_live['date'], open=df_live['open'], high=df_live['high'], low=df_live['low'], close=df_live['close'], name='Price'), row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['fast_ma'], line=dict(color='blue', width=1), name=f'Fast EMA ({fast_ema_w})'), row=1, col=1)
                    fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['slow_ma'], line=dict(color='orange', width=1), name=f'Slow EMA ({slow_ema_w})'), row=1, col=1)
                    
                    # Add Live LTP marker and line
                    if ltp is not None:
                        fig_live.add_trace(go.Scatter(
                            x=[datetime.now(IST)], 
                            y=[ltp], 
                            mode='markers', 
                            name='LTP (Live)', 
                            marker=dict(size=10, color='lime', symbol='diamond')), 
                            row=1, col=1)
                        
                        fig_live.add_hline(
                            y=ltp, 
                            line_dash="solid", 
                            line_color="lime", 
                            line_width=1, 
                            row=1, col=1, 
                            annotation_text=f"LTP: {ltp:.2f}", 
                            annotation_position="top left"
                        )

                    # Visualize SL/TP and S/R
                    if stop_loss and take_profit:
                        fig_live.add_hline(y=float(stop_loss), line_dash="dash", line_color="red", row=1, col=1, annotation_text="SL")
                        fig_live.add_hline(y=float(take_profit), line_dash="dash", line_color="green", row=1, col=1, annotation_text="TP")
                    
                    # Visualize Swing High/Low
                    fig_live.add_hline(y=swing_high, line_dash="dash", line_color="purple", row=1, col=1, annotation_text=f"Resistance ({swing_lookback})")
                    fig_live.add_hline(y=swing_low, line_dash="dash", line_color="brown", row=1, col=1, annotation_text=f"Support ({swing_lookback})")


                    # MACD AND RSI
                    fig_live.add_trace(go.Bar(x=df_live['date'], y=df_live['macd_hist'], name='MACD Hist', marker_color='grey'), row=2, col=1)
                    fig_live.add_trace(go.Scatter(x=df_live['date'], y=df_live['rsi'], line=dict(color='brown', width=1), name='RSI'), row=2, col=1)
                    fig_live.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_live.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                    fig_live.update_layout(xaxis_rangeslider_visible=False, height=700)
                    chart_placeholder.plotly_chart(fig_live, use_container_width=True)

                    # Display the tail of the data
                    table_placeholder.dataframe(df_live.tail(10), use_container_width=True)
                    
                    # Display Recommendation and Candle Time
                    rec_placeholder.subheader(f"💡 **{symbol}** Recommendation: **{recommendation}** (Total Score: {score})")
                    rec_placeholder.caption(f"Calculations based on **{live_interval}** candle closed at: **{latest_candle_time}** (LTP: ₹{entry_price:.2f})")
                    
                    # --- SCORE BREAKDOWN & CONFLICT ANALYSIS ---
                    targets_placeholder.markdown(f"""
                        ---
                        **Intraday Score Breakdown:** * **Trend (MA/Slope):** **{trend_score}** / $\pm 6$
                        * **Momentum (MACD):** **{momentum_score}** / $\pm 2$
                        * **Reversion (RSI/BB):** **{reversion_score}** / $\pm 2$
                    """)
                    
                    if not final_actionable:
                        if "Override" in recommendation:
                            targets_placeholder.error("🚨 **TRADE CANCELLED:** Macro or Company-Specific Risk Override is **ACTIVE**. Technical signals are ignored.")
                        elif "Low Volatility" in recommendation:
                            targets_placeholder.error("🚨 **TRADE CANCELLED:** Low Volatility filter active. Market is too choppy/ranging. Await expansion.")
                        elif "Counter-Trend" in recommendation:
                            targets_placeholder.warning(f"⚠️ **LOW CONVICTION:** Intraday signal is **Counter-Trend** to the **Daily ({daily_trend})** trend. Risk is reduced.")
                        else:
                            targets_placeholder.warning("Market is balanced or awaiting confirmation.")
                        
                        targets_price = current_price_for_analysis 
                        targets_placeholder.markdown(f"""
                            **Entry Triggers (Current Price/LTP: {targets_price:.2f}):**
                            * **BULLISH:** Price above **{swing_high:.2f}** AND Score $\ge 6$ AND Volume $\ge {volume_conf_mult:.1f}$x.
                            * **BEARISH:** Price below **{swing_low:.2f}** AND Score $\le -6$ AND Volume $\ge {volume_conf_mult:.1f}$x.
                        """)
                    
                    # Display Risk Management and Position Sizing
                    risk_placeholder.markdown(f"""
                        ---
                        **Trade Plan (R:R 1:{risk_rr} | Stop: {atr_stop_mult}x ATR)**
                        * **Entry Price (LTP):** **{entry_price:.2f}** (ATR: ₹{latest['atr']:.2f})
                        * **Suggested Stop-Loss:** **{stop_loss if stop_loss else 'N/A'}**
                        * **Suggested Take-Profit:** **{take_profit if take_profit else 'N/A'}**
                    """)
                    
                    if final_actionable and stop_loss and take_profit:
                        risk_placeholder.success(f"✅ **ACTIONABLE TRADE:** Targets calculated with **{effective_risk_percent:.2f}%** Risk.")
                        risk_placeholder.markdown(f"""
                            **Position Sizing (Effective Risk {effective_risk_percent:.2f}%):**
                            * **Risk Amount:** ₹ {risk_per_trade:.2f}
                            * **Stop Distance:** ₹ {stop_distance:.2f}
                            * **Max Quantity:** **{suggested_quantity}** shares
                        """)
                    else:
                        risk_placeholder.markdown(f"⚠️ **Risk:** **{effective_risk_percent:.2f}%** (Risk is adjusted to be lower if not trend aligned).")
                        risk_placeholder.warning("Trade is currently **NON-ACTIONABLE**. Await full confirmation from Volume, Breakout, and Trend Alignment.")


                except Exception as e:
                    st.error(f"Error during live data analysis loop: {e}. Stopping live analysis.")
                    st.session_state["live_running"] = False
                    st.exception(e)

                time.sleep(refresh_interval)

        except Exception as e:
            st.error(f"Error initializing live analysis: {e}")
            st.session_state["live_running"] = False
