# Clothing Demand Forecasting with SARIMAX

This project demonstrates a demand forecasting model for the **Clothing** category using a SARIMAX model and integrates the results into an interactive **Streamlit** web app.

## 🔍 Project Overview
- **Objective:** Forecast clothing product demand using time series data and external factors such as Discount, Price, and Competitor Pricing.
- **Model:** SARIMAX (Seasonal ARIMA with Exogenous Variables)
- **Tools:** Python, Pandas, Statsmodels, Streamlit

## 📁 Files Included
- `app.py` — Streamlit app script for forecasting
- `sarimax_model.pkl` — Pre-trained SARIMAX model
- `retail_store_inventory_preprocessed.csv` — Cleaned dataset
- `requirements.txt` — Required Python packages
- `create_sarimax_model.ipynb` — Notebook to train and save the SARIMAX model
- `generate_streamlit_app.ipynb` — Notebook to generate the Streamlit script
- `sarimax_modeling_clothing.ipynb` — Full modeling workflow
- `presentation.pptx` — Final presentation with insights and app screenshots

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/clothing-demand-forecasting.git
   cd clothing-demand-forecasting
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📊 Model Summary
- Trained on historical clothing data
- Uses price, discount, and competitor pricing as exogenous variables
- RMSE: 108.87

## 🎯 Use Cases
- Inventory planning
- Pricing strategy optimization
- Promotional impact analysis

## 📌 Author
Chukwudi Ekweani  
MSDA Final Project | Nexford University