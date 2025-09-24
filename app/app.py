import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')

import os
import re
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
st.title(("Forex Currency Trend Predictor"))

options = {
    'AUSTRALIAN DOLLAR': 'AUSTRALIA - AUSTRALIAN DOLLAR/US$',
    'EURO': 'EURO AREA - EURO/US$',
    'NEW ZEALAND DOLLAR': 'NEW ZEALAND - NEW ZELAND DOLLAR/US$',
    'GREAT BRITAIN POUNDS': 'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
    'BRAZILIAN REAL': 'BRAZIL - REAL/US$',
    'CANADIAN DOLLAR': 'CANADA - CANADIAN DOLLAR/US$',
    'CHINESE YUAN': 'CHINA - YUAN/US$',
    'HONG KONG DOLLAR': 'HONG KONG - HONG KONG DOLLAR/US$',
    'INDIAN RUPEE': 'INDIA - INDIAN RUPEE/US$',
    'KOREAN WON': 'KOREA - WON/US$',
    'MEXICAN PESO': 'MEXICO - MEXICAN PESO/US$',
    'SOUTH AFRICAN RAND$': 'SOUTH AFRICA - RAND/US$',
    'SINGAPORE DOLLAR': 'SINGAPORE - SINGAPORE DOLLAR/US$',
    'DANISH KRONE': 'DENMARK - DANISH KRONE/US$',
    'JAPANESE YEN': 'JAPAN - YEN/US$',
    'MALAYSIAN RINGGIT': 'MALAYSIA - RINGGIT/US$',
    'NORWEGIAN KRONE': 'NORWAY - NORWEGIAN KRONE/US$',
    'SWEDEN KRONA': 'SWEDEN - KRONA/US$',
    'SRILANKAN RUPEE': 'SRI LANKA - SRI LANKAN RUPEE/US$',
    'SWISS FRANC': 'SWITZERLAND - FRANC/US$',
    'NEW TAIWAN DOLLAR': 'TAIWAN - NEW TAIWAN DOLLAR/US$',
    'THAI BAHT': 'THAILAND - BAHT/US$'
}

def get_data (filename):
    """ Function to call data from a named data folder and apply preprocessing"""
    filepath = f"app/data/{filename}"
        
    if os.path.exists(filepath):
        
        data = pd.read_csv(filepath)
        
        print(data.head())
        data.dropna()
        data['Time Serie'] = pd.to_datetime(data['Time Serie'], format='%d-%m-%Y')
        
        data = data.drop("Unnamed: 0", axis = 1) # remove index column 
        data = data.drop("Unnamed: 24", axis = 1) # remove final column with 0 distinct values
        data = data.replace("ND", np.nan) # replace NDs with NaN values
        data = data.dropna() 

        reformat = data.drop("Time Serie", axis = 1)
        time_series = data[["Time Serie"]].iloc[1:]
        reformat_ = reformat.iloc[1:].astype(float) # reformat currencies as floats

        reformat_.insert(0, "Time Serie", time_series.values) 
        data_cleaned = reformat_.copy()
        
        print(data_cleaned.head())
        return data_cleaned
        
    else:
        print("Error: Data not found")
        return None
        
    
def make_forecast(forecast_length, currency, data):
    
    with st.spinner(f'Processing {currency}...'):
        
        filename = str(re.sub(r'[^\w\-]', '_', currency))
        print(f'Filename/Currency: {filename}')
        
        lstm_path = f"app/models/{filename}.h5"
        other_path = f"app/models/{filename}.pkl"
        
        currency_column = options[currency]
        
        if os.path.exists(lstm_path):
            
            model = load_model(lstm_path, compile = False)
            data = data[currency_column].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            lookback = 30
            
            def create_lstm_sequences(data, lookback):
                """Create sequences for LSTM training"""
                X, y = [], []
                for i in range(lookback, len(data)):
                    X.append(data[i-lookback:i, 0])
                    y.append(data[i, 0])  # Next day
                return np.array(X), np.array(y)
            
            X, y = create_lstm_sequences(scaled_data, lookback)
            
            # Train/test split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Reshape for LSTM [samples, timesteps, features]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            
            model.compile(optimizer='adam', loss='mse', metrics = ['mae'])
            
            history = model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=25, 
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Make predictions
            steps = forecast_length
            
            def forecast_lstm(model, last_sequence, steps, scaler):
                forecast = []
                current_sequence = last_sequence.copy()

                for _ in range(steps):
                    # Predict next value
                    pred_scaled = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
                    
                    # Store inverse transformed prediction
                    pred = scaler.inverse_transform(pred_scaled)[0, 0]
                    forecast.append(pred)
                    
                    # Update the sequence: drop first value, append prediction
                    current_sequence = np.append(current_sequence[1:], pred_scaled)

                return forecast
            
            last_sequence = X_test[-1].flatten()
            forecast = forecast_lstm(model, last_sequence, steps, scaler)
        
            return forecast
        
        elif os.path.exists(other_path):
            with open (other_path, 'rb') as f:
                model = pickle.load(f)
                pd.read_pickle(other_path)
                if str('arima') in model:
                    forecast_result = model.get_forecast(steps=forecast_length)
                    forecast = forecast_result.predicted_mean
                    return forecast

                else: 
                    future = model.make_future_dataframe(periods=forecast_length, freq='Y')
                    forecast_result = model.predict(future)
                    forecast = forecast_result['yhat'][-forecast_length:].values
                    return forecast
        
        else:
            print('Error: Model not available')
            
            return None
    


with st.form(key='user_form'):
    
    selected_option = st.selectbox('Choose a currency:', options)
    forecast_length = st.number_input(
    "Enter number of forecast periods",  # Label displayed to the user
    min_value=1,         # Minimum value allowed
    max_value=20,      # Maximum value allowed
    value=3,            # Default value
    step=1              # Increment step
)
    submit_button = st.form_submit_button(label='Generate Predictions')

if submit_button:
    data_cleaned = get_data("Foreign_Exchange_Rates.xls")
    forecast_result = make_forecast(forecast_length, selected_option, data_cleaned)
    future_dates = pd.date_range(start="2019-12-31", end="2039-12-31", freq="YE")
    future_dates = future_dates[:forecast_length]
    
    final_forecast = pd.DataFrame({
    "Year": future_dates,
    "Forecast": forecast_result
    })
    
    st.subheader(f"Yearly forecast for {selected_option}:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(final_forecast)
    
    with col2:
        st.line_chart(final_forecast.set_index("Year"))
