# Airline Ticket Price Prediction — Streamlit App
# ✈️ Airline Ticket Price Prediction

A beginner-friendly data science project that estimates airline ticket prices
from a historical flight-price dataset.

## Features
- Data cleaning and descriptive statistics
- Data visualization (histogram, box plot, bar, pie, scatter, line charts)
- Probability and random sampling
- Central Limit Theorem demonstration
- T-test: nonstop vs connecting flight prices
- Multiple linear regression using duration, days left, and number of stops
- Interactive Streamlit web app for price prediction

## Files
- `Airline_Ticket_Price_Prediction_Streamlit.ipynb`: Google Colab notebook
- `app.py`: Streamlit web app
- `requirements.txt`: Python dependencies

## Dataset
[Indian Flight Price Prediction (Kaggle)](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).
Download `Clean_Dataset.csv` and upload it in the app's sidebar.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Limitations
The model uses only duration, days left, and stops. It is trained on
historical data and does not provide live booking prices.

