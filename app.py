import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import date

st.title("Clothing Demand Forecasting App")
st.write("This version lets you manually input expected values and select a forecast start date to simulate future demand.")

# Load model
@st.cache_resource
def load_model():
    with open("sarimax_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Forecast Settings
st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=30, value=14)
start_date = st.sidebar.date_input("Forecast Start Date", value=date.today())

# User inputs for exogenous variables
st.sidebar.header("Expected Future Values")
discount = st.sidebar.number_input("Expected Discount (%)", min_value=0, max_value=100, value=10)
price = st.sidebar.number_input("Expected Price", min_value=0.0, value=50.0)
competitor_price = st.sidebar.number_input("Expected Competitor Price", min_value=0.0, value=55.0)

# Create synthetic exogenous dataframe
future_exog = pd.DataFrame({
    "Discount": [discount] * forecast_days,
    "Price": [price] * forecast_days,
    "Competitor Pricing": [competitor_price] * forecast_days
})

# Set forecast date index based on user input
future_index = pd.date_range(start=pd.to_datetime(start_date), periods=forecast_days, freq='D')
future_exog.index = future_index

# Forecast using SARIMAX
forecast = model.get_forecast(steps=forecast_days, exog=future_exog)
pred_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plotting
st.subheader("Forecast Results")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pred_mean.index, pred_mean, label="Forecast", color="green")
ax.fill_between(pred_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="lightgreen", alpha=0.3)
ax.set_title("Future Forecast of Clothing Demand")
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.success("Forecast completed successfully.")