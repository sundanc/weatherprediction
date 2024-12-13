# Weather Prediction App

This project implements a weather prediction system that predicts the temperature based on real-time weather data, including features like humidity, wind speed, and day-related features (day of the week, month). It uses a pre-trained machine learning model (SGD Regressor) to predict the temperature and compares the predicted temperature with the actual temperature fetched from a public weather API (OpenWeatherMap).

## Features

- Fetch real-time weather data for a specific city.
- Predict the temperature using a pre-trained machine learning model.

![image](https://github.com/user-attachments/assets/4ad38c0a-042c-4c06-91c1-f4206803f6d5)


- Display actual vs predicted temperatures using a line chart.
- Show model performance metrics (Mean Absolute Error, Mean Squared Error).

![image](https://github.com/user-attachments/assets/802afd49-d338-4f9d-b4d9-4af6713a29a2)

  
- Interactive user interface using Streamlit.

## Requirements

- Python 3.8+
- `pandas`
- `streamlit`
- `requests`
- `scikit-learn`
- `joblib`

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
