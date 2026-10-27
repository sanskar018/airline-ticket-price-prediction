import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Airline Ticket Price Prediction", page_icon="✈️", layout="wide")
st.title("✈️ Airline Ticket Price Prediction")
st.write("Estimate a ticket price using historical flight data. This app does not show live airline booking prices.")
st.sidebar.header("1. Load your dataset")
uploaded_file = st.sidebar.file_uploader("Upload the flight-price CSV file", type=["csv"])

def find_column(df, names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None

def convert_duration(value):
    if pd.isna(value): return np.nan
    text = str(value).strip().lower()
    if text.isdigit(): return float(text)
    hours = minutes = 0
    for part in text.split():
        try:
            if "h" in part: hours = int(part.replace("h", ""))
            elif "m" in part: minutes = int(part.replace("m", ""))
        except ValueError:
            pass
    return hours * 60 + minutes

def convert_stops(value):
    text = str(value).strip().lower().replace(" ", "_")
    mapping = {"zero": 0, "non-stop": 0, "nonstop": 0, "0": 0,
               "one": 1, "1": 1, "two": 2, "two_or_more": 2, "2": 2, "3": 2}
    return mapping.get(text, np.nan)

if uploaded_file is None:
    st.info("Upload your CSV file in the left sidebar to start.")
    st.markdown("**Expected columns:** `duration`, `days_left`, `stops`, `price`, `source_city`, and `destination_city`.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
    cols = {
        "duration": find_column(raw, ["duration"]),
        "days_left": find_column(raw, ["days_left"]),
        "stops": find_column(raw, ["stops", "number_of_stops"]),
        "price": find_column(raw, ["price", "ticket_price"]),
        "source_city": find_column(raw, ["source_city", "source", "from_city"]),
        "destination_city": find_column(raw, ["destination_city", "destination", "to_city"])
    }
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing) + ". Please use a dataset containing source and destination cities.")
    data = pd.DataFrame({
        "duration_minutes": raw[cols["duration"]].apply(convert_duration),
        "days_left": pd.to_numeric(raw[cols["days_left"]], errors="coerce"),
        "stops_number": raw[cols["stops"]].apply(convert_stops),
        "price": pd.to_numeric(raw[cols["price"]], errors="coerce"),
        "source_city": raw[cols["source_city"]].astype("string").str.strip(),
        "destination_city": raw[cols["destination_city"]].astype("string").str.strip()
    })
    data = data.dropna(subset=list(data.columns))
    data = data[(data["duration_minutes"] >= 0) & (data["days_left"] >= 0)]
except Exception as e:
    st.error(f"Could not prepare the dataset: {e}")
    st.stop()

if len(data) < 10:
    st.error("The CSV has too few valid rows to train a useful model. Please upload a larger dataset.")
    st.stop()

# One-hot encode city names so the model can learn from source/destination.
features = ["duration_minutes", "days_left", "stops_number", "source_city", "destination_city"]
X = pd.get_dummies(data[features], columns=["source_city", "destination_city"])
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.success(f"Dataset loaded: {len(data):,} usable rows")
c1, c2, c3 = st.columns(3)
c1.metric("Average ticket price", f"₹{data['price'].mean():,.0f}")
c2.metric("Mean Absolute Error", f"₹{mean_absolute_error(y_test, predictions):,.0f}")
c3.metric("R² score", f"{r2_score(y_test, predictions):.3f}")
with st.expander("Preview dataset"):
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("2. Estimate a ticket price")
left, right = st.columns(2)
source_cities = sorted(data["source_city"].astype(str).unique().tolist())
destination_cities = sorted(data["destination_city"].astype(str).unique().tolist())
with left:
    source_city = st.selectbox("Source city (From)", source_cities)
    destination_city = st.selectbox("Destination city (To)", destination_cities)
    duration = st.number_input("Flight duration (minutes)", min_value=1, max_value=2000, value=120, step=10)
    days_left = st.number_input("Days left before departure", min_value=0, max_value=365, value=20, step=1)
    stops_label = st.selectbox("Number of stops", ["Nonstop (0)", "One stop (1)", "Two or more (2)"])
    stops_number = {"Nonstop (0)": 0, "One stop (1)": 1, "Two or more (2)": 2}[stops_label]
with right:
    st.markdown("#### Your flight details")
    st.write(f"**From:** {source_city}")
    st.write(f"**To:** {destination_city}")
    st.write(f"**Duration:** {duration} minutes")
    st.write(f"**Days left:** {days_left}")
    st.write(f"**Stops:** {stops_label}")
    st.caption("Prediction is based on patterns in your uploaded historical dataset.")

if st.button("Predict ticket price", type="primary"):
    new_row = pd.DataFrame([{
        "duration_minutes": duration, "days_left": days_left, "stops_number": stops_number,
        "source_city": source_city, "destination_city": destination_city
    }])
    new_X = pd.get_dummies(new_row, columns=["source_city", "destination_city"])
    new_X = new_X.reindex(columns=X.columns, fill_value=0)
    price = max(0, float(model.predict(new_X)[0]))
    st.markdown("### Estimated ticket price")
    st.success(f"₹{price:,.2f}")
    st.caption("This is an estimate, not a guaranteed or live fare.")

st.subheader("3. Explore the dataset")
chart1, chart2 = st.columns(2)
with chart1:
    st.write("**Ticket-price distribution**")
    st.bar_chart(data["price"].clip(upper=data["price"].quantile(0.98)).value_counts().sort_index().head(40))
with chart2:
    st.write("**Average price by days left**")
    st.line_chart(data.groupby("days_left")["price"].mean().sort_index())
st.warning("This model includes source city and destination city, duration, days left, and stops. It does not include airline, travel class, or departure time, and it cannot provide live booking prices.")
