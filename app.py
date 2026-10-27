import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Airline Ticket Price Prediction", page_icon="✈️", layout="wide")

st.title("✈️ Airline Ticket Price Prediction")
st.write(
    "This beginner-friendly app estimates a ticket price using a historical flight-price dataset. "
    "It does not show live airline booking prices."
)

st.sidebar.header("1. Load your dataset")
uploaded_file = st.sidebar.file_uploader("Upload the flight-price CSV file", type=["csv"])

def convert_duration(value):
    """Convert values such as '2h 50m' or '2h' into minutes."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    hours = 0
    minutes = 0
    for part in text.split():
        if "h" in part:
            try:
                hours = int(part.replace("h", ""))
            except ValueError:
                pass
        elif "m" in part:
            try:
                minutes = int(part.replace("m", ""))
            except ValueError:
                pass
    # Also accept a duration already stored as a number of minutes.
    if text.isdigit():
        return float(text)
    return hours * 60 + minutes

def convert_stops(value):
    """Convert common stop labels to numbers."""
    text = str(value).strip().lower().replace(" ", "_")
    mapping = {
        "zero": 0, "non-stop": 0, "nonstop": 0, "0": 0,
        "one": 1, "1": 1,
        "two": 2, "two_or_more": 2, "2": 2, "3": 2
    }
    return mapping.get(text, np.nan)

def prepare_data(raw_df):
    required = ["duration", "days_left", "stops", "price"]
    missing = [col for col in required if col not in raw_df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing these required columns: "
            + ", ".join(missing)
            + ". Please use the flight-price dataset from the project."
        )

    data = raw_df.copy()
    data["duration_minutes"] = data["duration"].apply(convert_duration)
    data["stops_number"] = data["stops"].apply(convert_stops)
    for col in ["duration_minutes", "days_left", "stops_number", "price"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["duration_minutes", "days_left", "stops_number", "price"])
    data = data[(data["duration_minutes"] >= 0) & (data["days_left"] >= 0)]
    return data

if uploaded_file is None:
    st.info("Upload your CSV file in the left sidebar to start.")
    st.markdown(
        """
        **Expected columns:** `duration`, `days_left`, `stops`, and `price`.

        **Dataset:** The common Kaggle flight-price dataset has these columns. The app uses three simple inputs
        (duration, days left, and stops) to keep the model easy to explain in a viva.
        """
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    data = prepare_data(raw_df)
except Exception as error:
    st.error(f"Could not prepare the dataset: {error}")
    st.stop()

if len(data) < 10:
    st.error("The CSV has too few valid rows to train a useful model. Please upload a larger dataset.")
    st.stop()

X = data[["duration_minutes", "days_left", "stops_number"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

st.success(f"Dataset loaded: {len(data):,} usable rows")
col1, col2, col3 = st.columns(3)
col1.metric("Average ticket price", f"₹{data['price'].mean():,.0f}")
col2.metric("Mean Absolute Error", f"₹{mean_absolute_error(y_test, predictions):,.0f}")
col3.metric("R² score", f"{r2_score(y_test, predictions):.3f}")

with st.expander("Preview dataset"):
    st.dataframe(data.head(10), use_container_width=True)

st.subheader("2. Estimate a ticket price")
left, right = st.columns(2)

with left:
    duration = st.number_input(
        "Flight duration (minutes)", min_value=1, max_value=2000, value=120, step=10
    )
    days_left = st.number_input(
        "Days left before departure", min_value=0, max_value=365, value=15, step=1
    )
    stops_label = st.selectbox(
        "Number of stops", ["Nonstop (0)", "One stop (1)", "Two or more (2)"]
    )
    stops_number = {"Nonstop (0)": 0, "One stop (1)": 1, "Two or more (2)": 2}[stops_label]

with right:
    st.markdown("#### Your flight details")
    st.write(f"**Duration:** {duration} minutes")
    st.write(f"**Days left:** {days_left}")
    st.write(f"**Stops:** {stops_label}")
    st.caption("The prediction is based on patterns in your uploaded historical dataset.")

if st.button("Predict ticket price", type="primary"):
    input_data = pd.DataFrame({
        "duration_minutes": [duration],
        "days_left": [days_left],
        "stops_number": [stops_number]
    })
    predicted_price = float(model.predict(input_data)[0])
    predicted_price = max(0, predicted_price)
    st.markdown("### Estimated ticket price")
    st.success(f"₹{predicted_price:,.2f}")
    st.caption("This is an estimate, not a guaranteed or live fare.")

st.subheader("3. Explore the dataset")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.write("**Ticket-price distribution**")
    st.bar_chart(data["price"].clip(upper=data["price"].quantile(0.98)).value_counts().sort_index().head(40))
with chart_col2:
    st.write("**Average price by days left**")
    average_by_days = data.groupby("days_left", as_index=True)["price"].mean().sort_index()
    st.line_chart(average_by_days)

st.warning(
    "Model limitation: this simple version uses only duration, days left, and stops. "
    "It does not include airline, source/destination, travel class, or departure time, so accuracy may be limited. "
    "It predicts from historical data and cannot provide live booking prices."
)
