AIRLINE TICKET PRICE PREDICTION - STREAMLIT

1. Extract this ZIP file.
2. Put your flight dataset CSV in a place you can find.
3. Open the extracted folder in File Explorer.
4. Click the address bar, type cmd, and press Enter.
5. Run: python -m pip install -r requirements.txt
6. Run: python -m streamlit run app.py
7. In the browser, upload your flight-price CSV.

Required CSV columns (column names are case-insensitive):
- duration
- days_left
- stops (or number_of_stops)
- price (or ticket_price)
- source_city (or source / from_city)
- destination_city (or destination / to_city)

Duration can be a number of minutes or text such as 2h 50m.
This app estimates prices from historical data and does not show live fares.
