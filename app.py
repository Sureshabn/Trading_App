import streamlit as st
from kiteconnect import KiteConnect
import pandas as pd
from datetime import datetime, timedelta

# ---- Read secrets safely ----
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]

# ---- Create KiteConnect instance ----
kite = KiteConnect(api_key=API_KEY)

# ---- Store token in session_state ----
if "access_token" not in st.session_state:
    st.session_state["access_token"] = None

st.title("📈 Zerodha Live Stock Analysis")

# ---- Step 1: Login URL ----
st.subheader("🔑 Zerodha Login")
login_url = kite.login_url()
st.markdown(f"[Click here to login to Zerodha]({login_url})")

# ---- Step 2: Request Token Input ----
request_token = st.text_input("Paste Request Token after login:")

if request_token:
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        st.session_state["access_token"] = data["access_token"]
        st.success("✅ Access token generated successfully!")
        kite.set_access_token(st.session_state["access_token"])
    except Exception as e:
        st.error(f"Error generating session: {e}")

# ---- Step 3: Fetch Historical Data ----
if st.session_state["access_token"]:
    symbol = st.text_input("Enter NSE Stock Symbol:", "TCS").upper()
    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    end_date = datetime.today()

    if st.button("Fetch Data"):
        try:
            # Zerodha requires instrument_token instead of symbol directly
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
                    st.warning("No historical data available for this range.")
                else:
                    st.dataframe(df.tail(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error fetching historical data: {e}")
