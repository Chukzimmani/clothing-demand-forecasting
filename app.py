
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

st.title("Clothing Demand Forecasting App")
st.write("This app uses a pre-trained SARIMAX model to forecast clothing demand based on discount, price, and competitor pricing.")

# Load model
@st.cache_resource
def load_model():
    with open("sarimax_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Load exogenous data
@st.cache_data
def load_data():
    df = pd.read_csv("retail_store_inventory_preprocessed.csv", parse_dates=["Date"])
    df = df[df["Category"] == 3]
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    grouped = df.groupby(df.index).agg({
        "Units Sold": "sum",
        "Discount": "mean",
        "Price": "mean",
        "Competitor Pricing": "mean"
    })
    return grouped

model = load_model()
data = load_data()

# Date input for forecasting
st.sidebar.header("Forecast Controls")
start_date = st.sidebar.date_input("Start Date", value=data.index[-30].date())
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=30, value=14)

# Subset exogenous variables for prediction
exog = data[["Discount", "Price", "Competitor Pricing"]]
y = data["Units Sold"]

# Validate range
if start_date not in data.index:
    st.warning("Selected start date is not in the dataset.")
else:
    start_idx = data.index.get_loc(pd.to_datetime(start_date))
    if start_idx + forecast_horizon > len(data):
        st.warning("Forecast horizon exceeds available data.")
    else:
        exog_future = exog.iloc[start_idx:start_idx+forecast_horizon]
        forecast = model.get_forecast(steps=forecast_horizon, exog=exog_future)
        pred_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Plot
        st.subheader("Forecast Results")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y.index, y, label="Historical Sales")
        ax.plot(pred_mean.index, pred_mean, label="Forecast", color="green")
        ax.fill_between(pred_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="lightgreen", alpha=0.3)
        ax.set_title("Forecasted Clothing Demand")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.success("Forecast completed successfully.")
