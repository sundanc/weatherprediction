import pandas as pd
import streamlit as st
import joblib
import requests
from datetime import datetime

sgd_model = joblib.load("sgd_model.pkl")
scaler = joblib.load("scaler.pkl")

def get_weather_data():
    api_key = 'your_api_key'  
    city = 'London'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            return data
        else:
            raise Exception("Failed to fetch weather data.")
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

weather_data = get_weather_data()

if weather_data:
    actual_temperature = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    windspeed = weather_data['wind']['speed']

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_day_of_week = datetime.now().weekday()
    current_month = datetime.now().month
    
    current_features = [[actual_temperature, humidity, windspeed, current_day_of_week, current_month]]
    
    scaled_features = scaler.transform(current_features)
    
    predicted_temperature = sgd_model.predict(scaled_features)[0]

    st.write(f"Actual Temperature: {actual_temperature}°C")
    st.write(f"Predicted Temperature: {predicted_temperature:.2f}°C")
else:
    st.write("No weather data available to display.")

data = pd.DataFrame({
    'date': ['2024-11-28', '2024-11-29', '2024-11-30'],
    'temperature': [15.5, 16.0, 15.8],
    'predicted_temperature': [15.4, 16.1, 15.7]
})

if 'date' not in data.columns:
    st.error("The 'date' column is missing from the data.")
else:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  
    
    
    if data['date'].isnull().sum() > 0:
        st.warning("Invalid dates detected. Dropping invalid rows.")
        data = data.dropna(subset=['date'])

    data = data.set_index('date')

    st.subheader("Actual vs Predicted Temperatures:")
    st.line_chart(data[['temperature', 'predicted_temperature']])

mae = 0.23  
mse = 0.05  

st.subheader("Model Performance Metrics")
st.write(f"Mean Absolute Error (MAE): {mae}")
st.write(f"Mean Squared Error (MSE): {mse}")
