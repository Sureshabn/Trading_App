# app.py
import streamlit as st
import pandas as pd
from kiteconnect import KiteConnect
from datetime import datetime, timedelta

st.set_page_config(page_title="Zerodha Stock Data", layout="wide")

# ---- Load API credentials from Streamlit Secrets ----
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]
ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

# ---- Initialize KiteConnect ----
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

# ---- Function to fetch historical data ----
@st.cache_data(ttl=300)
def fetch_historical_df(symbol, start_date, end_date, interval="day"):
    try:
        instrument = kite.ltp(f"NSE:{symbol}")
        token = instrument[f"NSE:{symbol}"]["instrument_token"]

        data = kite.historical_data(token, start_date, end_date, interval)
        if not data:
            return None

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)

        return df[["date", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None


# ---- Streamlit UI ----
st.title("📈 Zerodha NSE Stock Data Extractor")

symbol = st.text_input("Enter NSE Stock Symbol:", "TCS").upper()
start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.date_input("End Date", datetime.today())

if st.button("Fetch Data"):
    with st.spinner(f"Fetching data for {symbol}..."):
        df = fetch_historical_df(symbol, datetime.combine(start_date, datetime.min.time()), datetime.combine(end_date, datetime.min.time()))

    if df is None or df.empty:
        st.error(f"No historical data available for {symbol}. Try another symbol.")
    else:
        st.success(f"✅ Data fetched for {symbol}")
        st.dataframe(df, use_container_width=True)

        st.line_chart(df.set_index("date")[["Close"]])
