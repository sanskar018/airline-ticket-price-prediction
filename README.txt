# Airline Ticket Price Prediction — Streamlit App

## Files
- `app.py` — Streamlit web app
- `requirements.txt` — required Python packages

## Dataset
Use the flight-price CSV used in the notebook. The app expects these columns:
- `duration`
- `days_left`
- `stops`
- `price`

The common Kaggle flight-price dataset from the project has these columns.

## Run on your computer
1. Install Python.
2. Extract this ZIP folder.
3. Open Command Prompt/Terminal in this folder.
4. Install packages:

   `pip install -r requirements.txt`

5. Start the app:

   `streamlit run app.py`

6. Open the local URL shown in the terminal (usually `http://localhost:8501`).
7. Upload your flight-price CSV using the sidebar.

## Important
This model is a simple multiple linear regression model using duration, days left, and number of stops. It uses historical data and does not return live fares. The app retrains the model from the CSV each time the app reruns.
