# DeepModels-TimeForecaster-Streamlit-App


# Time-Series Analysis Dashboard

## Overview
An advanced **Streamlit-based dashboard** for stock market and Financial data time-series analysis and prediction, combining traditional time series models with deep learning approaches.  

## Features

### ✔ Time-Series Data Information(Financial Data In Here)
* Real time stock data visualization and analysis
* Interactive price charts and technical indicators (OHLC) 

### ✔ Deep Learning Models
* **Multiple Architecture**
  * Temporal Convolutional Networks (TCN)
  * Transformer models
  * Block RNN (LSTM And variant)
  * N-BEATS
  * 
### ✔ Traditional Time Series Models
* **ARIMA (AutoRegressive Integrated Moving Average)**
  * Customizable parameters (p, d, q)
  * Interactive model training and evaluation
  * Prediction visualization
* **SARIMAX (Seasonal ARIMA with Exogenous Variables)**
  * Support for external features  
  * Advanced forecasting capabilities
  * Performance metrics (MSE, RMSE, MAE)


 
 

### ✔ Interactive Interface
* Clean and modern design with Plotly visualizations
* Real-time model training and prediction 

## How to Use 

1. **Clone the Repository**
```bash
git clone  
cd  
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Application**
```bash
streamlit run app.py
```

## Project Structure
```
├── app.py                  # Main application file
├── pages/
│   ├── stock_info.py      # stock information display
│   ├── traditional_models.py # ARIMA and SARIMAX models
│   └── deep_models.py     # deep learning models implementations (with darts)
└── utilities/
    └── data_processor.py  # data handling and preprocessing
```
 

 
 
