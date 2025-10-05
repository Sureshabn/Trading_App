import streamlit as st
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker
import pandas as pd
import ta
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import numpy as np
import pytz
import threading # For running the Ticker in the background

# Set the Indian Standard Time (IST) timezone
IST = pytz.timezone('Asia/Kolkata')

st.set_page_config(page_title="Zerodha Stock Analysis", layout="wide")
st.title("📈 Zerodha Stock Analysis & Risk Manager (Pro V9 - Enhanced Live Safety)")

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
# NEW STATE FOR MTFA
if "long_term_trend" not in st.session_state: st.session_state["long_term_trend"] = "UNKNOWN"
# NEW STATES FOR LIVE TICKER STREAMING
if "latest_ticks" not in st.session_state: st.session_state["latest_ticks"] = {}
if "ticker_running" not in st.session_state: st.session_state["ticker_running"] = False
if "kws_instance" not in st.session_state: st.session_state["kws_instance"] = None
if "instrument_token" not in st.session_state: st.session_state["instrument_token"] = None
if "tradingsymbol" not in st.session_state: st.session_state["tradingsymbol"] = "TCS"
# NEW STATE FOR THREAD SAFETY (FIX)
if "state_lock" not in st.session_state: st.session_state["state_lock"] = threading.Lock() 


# ----------------------------------------------------------------------
## 🔴 Kite Ticker Callbacks and Thread Management
# ----------------------------------------------------------------------

def on_ticks(ws, ticks):
    """Callback when ticks are received."""
    # This function runs in the background thread. Access to Streamlit elements is NOT allowed.
    # FIX: Acquire lock for thread safety
    if st.session_state.get("state_lock"):
        with st.session_state["state_lock"]:
            for tick in ticks:
                instrument_token = tick.get('instrument_token')
                if instrument_token:
                    # Safely store the latest tick data in the session state
                    st.session_state["latest_ticks"][instrument_token] = tick

def on_connect(ws, response):
    """Callback on successful connect. Subscribe to tokens here."""
    token = st.session_state.get("instrument_token")
    if token:
        # Subscribe to the token saved in session state
        ws.subscribe([token])
        # Use full mode for comprehensive data (LTP, depth, etc.)
        ws.set_mode(ws.MODE_FULL, [token]) 
        
def on_close(ws, code, reason):
    """Callback when connection is closed."""
    # This updates the shared state, which the main Streamlit thread reads
    st.session_state["ticker_running"] = False
    
def on_error(ws, code, reason):
    """Callback on connection error."""
    # print(f"Ticker error {code}: {reason}") # Print to console for background thread debug
    pass

def on_reconnect(ws, attempt, delay):
    """Callback on reconnection attempt."""
    # print(f"Reconnecting... Attempt {attempt} in {delay}s") # Print to console
    pass

def initialize_ticker_thread(api_key, access_token):
    """Initializes and runs KiteTicker in a separate thread."""
    
    # Do not start if already running
    if st.session_state["kws_instance"] and st.session_state["ticker_running"]:
        return True

    try:
        kws = KiteTicker(api_key, access_token)
        
        # Assign callbacks
        kws.on_ticks = on_ticks
        kws.on_connect = on_connect
        kws.on_close = on_close
        kws.on_error = on_error
        kws.on_reconnect = on_reconnect

        # Store the instance in session state
        st.session_state["kws_instance"] = kws
        st.session_state["ticker_running"] = True
        
        # Connect in threaded mode so it doesn't block the Streamlit app
        kws.connect(threaded=True)
        return True

    except Exception as e:
        st.error(f"Error initializing KiteTicker: {e}")
        st.session_state["ticker_running"] = False
        st.session_state["kws_instance"] = None
        return False
        
def stop_ticker_stream():
    """Stops the KiteTicker thread gracefully."""
    if st.session_state["kws_instance"]:
        st.session_state["kws_instance"].close()
        # Clean up session state
        st.session_state["kws_instance"] = None
        st.session_state["ticker_running"] = False
        st.session_state["latest_ticks"] = {}
        # st.info("Kite Ticker connection closed.") # Moved info message to the UI

# ----------------------------------------------------------------------
## 🔑 Zerodha Login
# ----------------------------------------------------------------------

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
## ⚙️ Analysis Parameters
# ----------------------------------------------------------------------
if st.session_state["access_token"]:
    st.subheader("⚙️ Analysis Parameters")

    # --- 1. Flexibility and Parameterization & Position Sizing ---
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # --- Instrument Selection and Token Lookup ---
    with col1:
        current_symbol = st.session_state.get("tradingsymbol", "TCS")
        new_symbol = st.text_input("NSE Symbol:", current_symbol).upper()
        
        # Check if the symbol changed or if the token is not yet set
        if new_symbol != current_symbol or st.session_state["instrument_token"] is None:
            # Only perform instrument lookup if we have an API connection
            try:
                instruments = kite.instruments("NSE")
                df_instruments = pd.DataFrame(instruments)
                row = df_instruments[df_instruments["tradingsymbol"] == new_symbol]
                
                if row.empty:
                    st.error(f"❌ Symbol {new_symbol} not found on NSE.")
                    st.session_state["instrument_token"] = None
                else:
                    token = int(row.iloc[0]["instrument_token"])

                    # FIX: Stop the ticker stream if the instrument changes while running
                    if st.session_state["ticker_running"] and token != st.session_state["instrument_token"]:
                        stop_ticker_stream()
                        st.warning(f"Symbol changed from {current_symbol} to {new_symbol}. Ticker stream reset.")
                        
                    st.session_state["instrument_token"] = token
                    st.session_state["tradingsymbol"] = new_symbol
            except Exception as e:
                st.warning("⚠️ Could not fetch instrument list. Check API limits or network.")
                st.session_state["instrument_token"] = None

    # Use session state values for the rest of the code
    symbol = st.session_state["tradingsymbol"]
    token = st.session_state["instrument_token"]

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
    
    colD, colE, colF, colG = st.columns(4)
    with colD:
        # Volume Confirmation Threshold
        volume_conf_mult = st.number_input("Volume Conf. Multiplier (e.g., 1.5x ATR Vol)", min_value=1.0, value=1.5, step=0.1)
    with colE:
        # Swing High/Low Lookback
        swing_lookback = st.number_input("Swing Lookback Periods (for S/R)", min_value=5, value=10)
    
    # --- NEW SAFEGURAD INPUTS ---
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
    # --- END NEW SAFEGURAD INPUTS ---

    start_date = st.date_input("Historical Start Date", datetime(2022,1,1))
    end_date = st.date_input("Historical End Date", datetime.today())
    # Recommended minimum is 30, increased to 60 for stability
    refresh_interval = st.slider("Live refresh interval (seconds)", 30, 300, 60)
    live_interval = st.selectbox("Intraday Bar Interval", ["5minute", "15minute", "30minute"])


    # ---- Indicator Calculation Function ----
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

    # ======================================================================
    # ---- MTFA (Multi-Timeframe Analysis) Function ----
    # ======================================================================
    @st.cache_data(ttl=3600) # Cache for 1 hour as daily trend doesn't change quickly
    # FIX: Prepend with an underscore to prevent Streamlit from hashing the KiteConnect object
    def get_long_term_trend(_kite_api, token, fast_w, slow_w):
        """Fetches Daily data and determines the long-term trend based on EMA cross."""
        try:
            if not token:
                return "UNKNOWN"
            # Fetch data for the last 150 days (more than enough for 50/20 EMA)
            start_date = date.today() - timedelta(days=150)
            end_date = date.today() 
            
            # Use the corrected argument name here
            hist_daily = _kite_api.historical_data(token, start_date, end_date, interval="day")
            df_daily = pd.DataFrame(hist_daily)
            
            if df_daily.empty:
                return "UNKNOWN"

            # Calculate only the required EMAs
            df_daily["fast_ma"] = ta.trend.EMAIndicator(df_daily["close"], window=fast_w).ema_indicator()
            df_daily["slow_ma"] = ta.trend.EMAIndicator(df_daily["close"], window=slow_w).ema_indicator()
            
            # Get the latest complete daily candle
            latest_daily = df_daily.dropna(subset=["fast_ma", "slow_ma"]).iloc[-1]
            
            if latest_daily["fast_ma"] > latest_daily["slow_ma"]:
                return "BULLISH" # Fast EMA > Slow EMA on Daily
            elif latest_daily["fast_ma"] < latest_daily["slow_ma"]:
                return "BEARISH" # Fast EMA < Slow EMA on Daily
            else:
                return "NEUTRAL"
                
        except Exception as e:
            # st.error(f"Error fetching daily trend: {e}") # Suppress error in the background
            return "UNKNOWN"
    # ======================================================================

    # ----------------- UI Tabs -----------------
    # SPLIT THE UI INTO TWO TABS
    tab1, tab2 = st.tabs(["📊 Strategy & Live Analysis (Polling)", "🔴 Live Ticker Stream (WebSocket)"])

    # ----------------------------------------------------------------------
    ## 📊 Tab 1: Polling Analysis
    # ----------------------------------------------------------------------
    with tab1:
        st.subheader("📊 Strategy & Bar-Close Analysis")
        if token is None:
            st.warning(f"Please enter a valid symbol (e.g., TCS) to run the analysis.")
        
        # ---- Historical Analysis Button (Daily) ----
        if st.button("Run Historical Analysis (Daily Timeframe)"):
            if token is None:
                st.error(f"❌ Cannot run historical analysis: Token for {symbol} is invalid.")
            else:
                try:
                    # Fetching historical data
                    hist = kite.historical_data(token, start_date, end_date, interval="day")
                    df_hist = pd.DataFrame(hist)

                    if df_hist.empty:
                        st.warning("⚠️ No historical data available")
                    else:
                        df_hist = calculate_indicators(df_hist, fast_ema_w, slow_ema_w, rsi_w)
                        
                        st.subheader(f"📈 Historical ({symbol}) Candlestick with EMAs & Oscillators")
                        
                        # Charting logic 
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


        # ---- Live Analysis (Polling) ----
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

            if token is None:
                st.error(f"❌ Token for {symbol} not found. Stopping live analysis.")
                st.session_state["live_running"] = False
                st.stop()
            else:
                try:
                    # --- NEW: Get Long-Term Trend before the loop ---
                    # NOTE: The function call uses 'kite', but the cached function definition uses '_kite_api' 
                    st.session_state["long_term_trend"] = get_long_term_trend(kite, token, fast_ema_w, slow_ema_w)
                    # --- END NEW: Get Long-Term Trend ---

                    while st.session_state["live_running"]:
                        try:
                            # --- FIX: Set start time to 3 days ago for robust fetching ---
                            today = date.today()
                            # Go back 3 days to safely capture the last trading day's close for indicators.
                            intraday_start = datetime.combine(today - timedelta(days=3), datetime.min.time())
                            end_time = datetime.now() # Fetch up to the current moment

                            
                            # Display debug info in the sidebar - CRITICAL FOR DIAGNOSIS
                            debug_placeholder.caption(f"Fetching {live_interval} data for Token: {token}\nFrom: {intraday_start.strftime('%Y-%m-%d %H:%M:%S')}\nTo: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

                            # Fetch historical data (only closed bars)
                            hist = kite.historical_data(token, intraday_start, end_time, interval=live_interval)
                            df_live = pd.DataFrame(hist)
                            
                            # --- NEW: Fetch Live Quote (LTP) ---
                            ltp = None
                            quotes = kite.quote([f'NSE:{symbol}'])
                            if f'NSE:{symbol}' in quotes:
                                ltp = quotes[f'NSE:{symbol}']['last_price']
                            # --- END NEW LTP FETCH ---

                            if df_live.empty:
                                st.warning(f"⚠️ No closed {live_interval} data available for {symbol}. Market might not be open yet, or API is delayed. Retrying in {refresh_interval}s...")
                                time.sleep(refresh_interval)
                                continue

                            df_live = calculate_indicators(df_live, fast_ema_w, slow_ema_w, rsi_w)

                            # Filter for rows where all essential indicators are calculated (i.e., not NaN)
                            df_valid = df_live.dropna(subset=["fast_ma","slow_ma","rsi","macd","macd_signal","bb_high","bb_low", "atr", "fast_ma_slope", "atr_volume"])
                            
                            if df_valid.empty:
                                st.warning("⚠️ Not enough history available to calculate all indicators. Waiting for more closed bars.")
                                time.sleep(refresh_interval)
                                continue

                            # Use the latest **complete** candle for all calculations
                            latest = df_valid.iloc[-1]
                            prev = df_valid.iloc[-2] if len(df_valid) >= 2 else None
                            latest_candle_time = latest['date'].tz_convert(IST).strftime('%Y-%m-%d %H:%M:%S') if latest['date'].tzinfo is not None else latest['date'].strftime('%Y-%m-%d %H:%M:%S')
                            

                            # --- CORE SCORING LOGIC (Using 'latest' closed bar) ---
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
                                        score -= 1; reversion_score += -1
                                        
                            # --- SAFEGURADS AND CONFIRMATION LOGIC (Technical) ---
                            
                            df_lookback = df_valid.iloc[-swing_lookback-1:-1] # Look at the bars preceding the current bar
                            swing_high = df_lookback["high"].max() if not df_lookback.empty else latest["high"]
                            swing_low = df_lookback["low"].min() if not df_lookback.empty else latest["low"]
                            
                            is_volume_confirmed = False
                            volume_ratio = 0.0
                            if latest["atr_volume"] > 0:
                                volume_ratio = latest["volume"] / latest["atr_volume"]
                                if volume_ratio >= volume_conf_mult:
                                    is_volume_confirmed = True

                            is_breakout_confirmed = False
                            # Use LTP for the breakout check as it's the most current price
                            current_price_for_breakout = ltp if ltp is not None else latest["close"] 
                            
                            if is_bullish_trend and current_price_for_breakout > swing_high:
                                is_breakout_confirmed = True
                            elif is_bearish_trend and current_price_for_breakout < swing_low:
                                is_breakout_confirmed = True
                            
                            
                            # --- FINAL RECOMMENDATION ---
                            
                            recommendation = (
                                "STRONG BUY" if score >= 8 else 
                                "BUY" if score >= 4 else  
                                "HOLD/NEUTRAL" if score > -4 else 
                                "SELL" if score > -8 else  
                                "STRONG SELL"            
                            )

                            # *** NEW: MTFA (Multi-Timeframe) Override Check ***
                            is_mtfa_conflict = False
                            if st.session_state["long_term_trend"] == "BEARISH" and score > 0:
                                # Downgrade bullish signals when daily trend is down (penalty: 5 points)
                                score = max(0, score - 5) 
                                if score == 0:
                                    recommendation = "HOLD/NEUTRAL (MTFA Conflict)"
                                    is_mtfa_conflict = True
                                    
                            elif st.session_state["long_term_trend"] == "BULLISH" and score < 0:
                                # Downgrade bearish signals when daily trend is up (penalty: 5 points)
                                score = min(0, score + 5) 
                                if score == 0:
                                    recommendation = "HOLD/NEUTRAL (MTFA Conflict)"
                                    is_mtfa_conflict = True
                            # *** END NEW MTFA ***
                            
                            # *** NEW: MACRO/FUNDAMENTAL OVERRIDE SAFEGURAD ***
                            is_overridden = False
                            if (macro_risk_override or company_news_override) and recommendation in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
                                # Downgrade any actionable trade to HOLD
                                recommendation = "HOLD/NEUTRAL (Macro/News Override)"
                                is_overridden = True
                            # *** END NEW OVERRIDE ***
                            
                            # Apply Technical Safeguard Filtering to the strongest signals (Original Logic)
                            if not is_overridden and not is_mtfa_conflict: # Only apply technical filtering if not already overridden
                                if recommendation in ["STRONG BUY", "BUY"] and score >= 6:
                                    if not is_volume_confirmed and not is_breakout_confirmed:
                                        recommendation = "BUY (Wait for Volume & Breakout)"
                                    elif not is_volume_confirmed:
                                        recommendation = "BUY (Low Volume Confirmation)"
                                    elif not is_breakout_confirmed:
                                        recommendation = "BUY (No Breakout Yet)"
                                
                                if recommendation in ["STRONG SELL", "SELL"] and score <= -6:
                                    if not is_volume_confirmed and not is_breakout_confirmed:
                                        recommendation = "SELL (Wait for Volume & Breakout)"
                                    elif not is_volume_confirmed:
                                        recommendation = "SELL (Low Volume Confirmation)"
                                    elif not is_breakout_confirmed:
                                        recommendation = "SELL (No Breakout Yet)"


                            # --- RISK MANAGEMENT CALCULATIONS ---
                            stop_loss, take_profit = None, None
                            suggested_quantity = 0
                            risk_per_trade = account_size * (risk_percent / 100)
                            stop_distance = latest["atr"] * atr_stop_mult
                            entry_price = current_price_for_breakout # Use the most recent price for position sizing
                            
                            if not pd.isna(latest["atr"]) and stop_distance > 0:
                                suggested_quantity = int(risk_per_trade / stop_distance)
                                if suggested_quantity < 1: suggested_quantity = 1

                                # MODIFIED CONDITION to INCLUDE OVERRIDE CHECK AND MTFA CHECK
                                if not is_overridden and not is_mtfa_conflict and score >= 6 and is_breakout_confirmed and is_volume_confirmed: 
                                    stop_loss_price = entry_price - stop_distance
                                    take_profit_price = entry_price + (stop_distance * risk_rr)
                                    stop_loss = f"{stop_loss_price:.2f}"
                                    take_profit = f"{take_profit_price:.2f}"
                                # MODIFIED CONDITION to INCLUDE OVERRIDE CHECK AND MTFA CHECK
                                elif not is_overridden and not is_mtfa_conflict and score <= -6 and is_breakout_confirmed and is_volume_confirmed: 
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
                                    **DECISION THRESHOLDS (Total Max: $\pm 10$):**
                                    * **STRONG BUY/SELL:** $|Score| \ge 8$
                                    * **BUY/SELL:** $|Score| \ge 4$
                                    * **HOLD/NEUTRAL:** $|Score| < 4$
                                """)
                            
                            # Dynamic Score Card 
                            score_card_placeholder.dataframe(pd.DataFrame({
                                "Component": ["Daily Trend (MTFA Filter)", "Fast EMA (Trend)", "EMA Slope (Strength)", "MACD Hist (Momentum)", "RSI/BB (Reversion)", "Volume Ratio"],
                                "Current Value": [
                                    st.session_state["long_term_trend"], # NEW MTFA Row
                                    f"{latest['fast_ma']:.2f} / {latest['slow_ma']:.2f}",
                                    f"{latest['fast_ma_slope']:.4f}",
                                    f"{latest['macd_hist']:.4f}",
                                    f"{latest['rsi']:.2f} (BB: {latest['bb_low']:.2f}/{latest['bb_high']:.2f})",
                                    f"{volume_ratio:.2f}x (ATR Vol: {latest['atr_volume']:.0f})"
                                ],
                                "Bullish Met": [
                                    "✅" if st.session_state["long_term_trend"] == "BULLISH" else "❌", # NEW MTFA Logic
                                    "✅" if flag_ma_cross_up else "❌",
                                    "✅" if flag_slope_pos else "❌",
                                    "✅" if flag_macd_bull else "❌",
                                    "✅" if flag_rsi_bull or (is_bullish_trend and latest['close'] < latest['bb_low']) else "❌",
                                    "✅" if is_volume_confirmed else "❌"
                                ],
                                "Bearish Met": [
                                    "✅" if st.session_state["long_term_trend"] == "BEARISH" else "❌", # NEW MTFA Logic
                                    "✅" if flag_ma_cross_down else "❌",
                                    "✅" if flag_slope_neg else "❌",
                                    "✅" if flag_macd_bear else "❌",
                                    "✅" if flag_rsi_bear or (is_bearish_trend and latest['close'] > latest['bb_high']) else "❌",
                                    "✅" if is_volume_confirmed else "❌"
                                ],
                                "Points": [
                                    f"Penalty: -5 (if conflict)", # NEW MTFA Logic
                                    f"{4 if flag_ma_cross_up else (-4 if flag_ma_cross_down else 0)}",
                                    f"{2 if flag_slope_pos else (-2 if flag_slope_neg else 0)}",
                                    f"{2 if flag_macd_bull else (-2 if flag_macd_bear else 0)}",
                                    f"{1 if reversion_score > 0 else (-1 if reversion_score < 0 else 0)}",
                                    "N/A"
                                ]
                            }).set_index("Component"), use_container_width=True)

                            # Live Chart (Visualizing SL/TP and S/R)
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
                            fig_live.add_hline(y=swing_high, line_dash="dash", line_color="purple", row=1, col=1, annotation_text="Swing High")
                            fig_live.add_hline(y=swing_low, line_dash="dash", line_color="brown", row=1, col=1, annotation_text="Swing Low")


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
                            # ADDED: This caption is crucial for diagnosing the delay issue.
                            rec_placeholder.caption(f"Calculations based on **{live_interval}** candle closed at: **{latest_candle_time}** (LTP: ₹{entry_price:.2f})")
                            
                            # --- SCORE BREAKDOWN & CONFLICT ANALYSIS ---
                            targets_placeholder.markdown(f"""
                                ---
                                **Score Breakdown:** * **Daily Trend:** **{st.session_state["long_term_trend"]}**
                                * **Trend (MA/Slope):** **{trend_score}** / $\pm 6$
                                * **Momentum (MACD):** **{momentum_score}** / $\pm 2$
                                * **Reversion (RSI/BB):** **{reversion_score}** / $\pm 2$
                            """)
                            
                            if recommendation == "HOLD/NEUTRAL" or "Wait for" in recommendation or "Override" in recommendation:
                                if "Override" in recommendation:
                                    targets_placeholder.error("🚨 **TRADE CANCELLED:** Macro/News or **MTFA Conflict** is **ACTIVE**. Technical signals are ignored.")
                                else:
                                    targets_placeholder.warning("Market is balanced or awaiting confirmation. Avoid entry.")
                                
                                conflict_col1, conflict_col2 = targets_placeholder.columns(2)
                                
                                if is_mtfa_conflict:
                                    conflict_col1.markdown(f"🚫 **MAJOR CONFLICT:** Intraday signal conflicts with **Daily Trend ({st.session_state['long_term_trend']})**. Signal neutralized.")
                                elif abs(trend_score) > 0 and np.sign(trend_score) != np.sign(momentum_score) and abs(momentum_score) > 0:
                                    conflict_col1.markdown("🚫 **CONFLICT:** Trend direction is opposed by Momentum. Wait for alignment.")
                                elif abs(trend_score) < 4 and abs(momentum_score) < 2:
                                    conflict_col1.markdown("📉 **WEAKNESS:** All components are low-scoring. Range-bound market.")
                                
                                targets_price = current_price_for_breakout 
                                conflict_col2.markdown(f"""
                                    **Entry Triggers (Current Price/LTP: {targets_price:.2f}):**
                                    * **BULLISH:** Close above **{swing_high:.2f}** AND Score $\ge 6$ AND Volume $\ge {volume_conf_mult:.1f}$x.
                                    * **BEARISH:** Close below **{swing_low:.2f}** AND Score $\le -6$ AND Volume $\ge {volume_conf_mult:.1f}$x.
                                """)
                            
                            # Display Risk Management and Position Sizing
                            risk_placeholder.markdown(f"""
                                ---
                                **Trade Plan (R:R 1:{risk_rr} | Stop: {atr_stop_mult}x ATR)**
                                * **Entry Price (LTP):** **{entry_price:.2f}**
                                * **Swing High/Low ({swing_lookback} bars):** **{swing_high:.2f}** / **{swing_low:.2f}**
                                * **Suggested Stop-Loss:** **{stop_loss if stop_loss else 'N/A'}**
                                * **Suggested Take-Profit:** **{take_profit if take_profit else 'N/A'}**
                            """)
                            
                            if stop_loss and take_profit and not is_overridden and not is_mtfa_conflict:
                                risk_placeholder.markdown(f"""
                                    **Position Sizing (Risk {risk_percent}%):**
                                    * **Risk Amount:** ₹ {risk_per_trade:.2f}
                                    * **Stop Distance:** ₹ {stop_distance:.2f}
                                    * **Max Quantity:** **{suggested_quantity}** shares
                                """)
                            elif is_overridden or is_mtfa_conflict:
                                risk_placeholder.markdown("⚠️ **Trade Cancelled:** Risk Management Skipped due to **Macro/News Override** or **MTFA Conflict**.")
                            else:
                                risk_placeholder.markdown("⚠️ **Actionable Trade:** Requires a high score ($|\text{Score}| \ge 6$), **Volume Confirmation**, and a **Price Breakout** to calculate actionable targets.")


                        except Exception as e:
                            st.error(f"Error during live data analysis loop: {e}. Stopping live analysis.")
                            st.session_state["live_running"] = False
                            st.exception(e) # Display full traceback for better debugging

                        time.sleep(refresh_interval)

                except Exception as e:
                    st.error(f"Error initializing live analysis: {e}")
                    st.session_state["live_running"] = False


    # ----------------------------------------------------------------------
    ## 🔴 Tab 2: WebSocket Streaming
    # ----------------------------------------------------------------------
    with tab2:
        st.header(f"🔴 Real-time Tick Data for {symbol}")
        
        if token is None:
            st.warning(f"Please enter a valid symbol (e.g., TCS) in the 'Analysis Parameters' section.")
        elif st.session_state["access_token"] and token:
            
            col_start, col_stop, col_spacer = st.columns([1, 1, 4])

            # 1. Start/Stop Button to manage the Ticker thread
            with col_start:
                if not st.session_state["ticker_running"]:
                    if st.button("Start Ticker Stream"):
                        if initialize_ticker_thread(API_KEY, st.session_state["access_token"]):
                            st.rerun() # Rerun to refresh the UI and start the display loop
                        
            with col_stop:
                if st.session_state["ticker_running"]:
                    if st.button("Stop Ticker Stream"):
                        stop_ticker_stream()
                        st.rerun()

            if st.session_state["ticker_running"]:
                st.success("✅ Kite Ticker connected and streaming in a background thread.")
            else:
                st.info("Click 'Start Ticker Stream' to begin receiving live ticks (LTP, Market Depth).")


            # 2. Display the live data
            if st.session_state["ticker_running"]:
                
                live_placeholder = st.empty()
                
                # Streamlit UI loop to read data from the background thread's shared memory
                while st.session_state["ticker_running"]:
                    
                    # FIX: Acquire lock for thread safety when reading the shared state
                    latest_tick = None
                    if st.session_state.get("state_lock"):
                        with st.session_state["state_lock"]:
                            # Check for the specific instrument's tick
                            latest_tick = st.session_state["latest_ticks"].get(token)
                    
                    if latest_tick:
                        
                        # Formatting the tick data for display
                        ltp = latest_tick.get("last_price", 0)
                        volume = latest_tick.get("volume", 0)
                        timestamp_utc = latest_tick.get("exchange_timestamp", 0)
                        
                        # Convert to IST for display
                        timestamp = datetime.fromtimestamp(timestamp_utc).astimezone(IST).strftime("%H:%M:%S.%f")[:-3] if timestamp_utc else "N/A"
                        
                        display_data = {
                            "Trading Symbol": symbol,
                            "Instrument Token": token,
                            "Last Traded Price (LTP)": f'₹ {ltp:.2f}',
                            "Last Traded Quantity (LTQ)": latest_tick.get("last_quantity", 0),
                            "Volume Traded": f'{volume:,}',
                            "Timestamp": timestamp,
                            "Change (%)": f'{latest_tick.get("change", 0):.2f}%',
                            "Open": latest_tick.get("ohlc", {}).get("open", 0),
                            "High": latest_tick.get("ohlc", {}).get("high", 0),
                            "Low": latest_tick.get("ohlc", {}).get("low", 0),
                            "Close": latest_tick.get("ohlc", {}).get("close", 0)
                        }
                        
                        # Extract Market Depth
                        depth_table = []
                        bids = latest_tick.get("depth", {}).get("buy", [])
                        asks = latest_tick.get("depth", {}).get("sell", [])
                        
                        # Ensure both are the same length (max 5)
                        max_depth = 5
                        for i in range(max_depth):
                            bid = bids[i] if i < len(bids) else {"quantity": 0, "price": 0, "orders": 0}
                            ask = asks[i] if i < len(asks) else {"quantity": 0, "price": 0, "orders": 0}
                            
                            depth_table.append({
                                "Bid QTY": bid["quantity"],
                                "Bid PRICE": bid["price"],
                                "Ask PRICE": ask["price"],
                                "Ask QTY": ask["quantity"],
                            })

                        # Update the live placeholder
                        with live_placeholder.container():
                            st.subheader("Current Live Quote")
                            st.table(pd.DataFrame([display_data]).T.rename(columns={0: "Value"}))
                            
                            st.subheader("Market Depth (5 Levels)")
                            st.dataframe(pd.DataFrame(depth_table), hide_index=True, use_container_width=True)

                    else:
                        live_placeholder.info("Waiting for first ticks or connection to be established...")

                    # Poll the shared state every 0.2 seconds (Streamlit UI refresh rate)
                    time.sleep(0.2) 
            
            elif st.session_state["kws_instance"] is not None and not st.session_state["ticker_running"]:
                st.error("Kite Ticker connection closed or failed.")

        else:
            st.warning("Please log in and ensure a valid symbol is selected in the 'Analysis Parameters' section.")
