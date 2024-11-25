import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from utilities.data_processor import DataProcessor

class TraditionalModelsPage:
    def __init__(self):
        self.data_processor = DataProcessor()
    
    def train_arima(self, y_train, order=(1,1,1)):
        with st.spinner("Training ARIMA model..."):
            model = ARIMA(y_train, order=order)
            return model.fit()
    
    def train_sarimax(self, y_train, X_train, order=(1,1,1), seasonal_order=(1,1,1,12)):
        with st.spinner("Training SARIMAX model..."):
            model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
            return model.fit(disp=False)
    
    def plot_predictions(self, y_train, y_test, predictions, title):
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_train.index,
            y=y_train.values,
            name="Training Data",
            line=dict(color="blue")
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=y_test.values,
            name="Actual Test Data",
            line=dict(color="orange")
        ))
        
        fig.add_trace(go.Scatter(
            x=y_test.index,
            y=predictions,
            name="Predictions",
            line=dict(color="green")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            height=500
        )
        
        return fig
    
    def show_page(self):
        st.title("Traditional Time Series Models")
        
        #load and prepare data (with data prepr)
        with st.spinner("Preparing data..."):
            df = self.data_processor.load_data()
            df = self.data_processor.prepare_features(df)
            X_train, X_test, y_train, y_test = self.data_processor.prepare_data_for_traditional_models(df)
        
        #model selection
        model_type = st.radio("Select Model", ["ARIMA", "SARIMAX"])
        
        if model_type == "ARIMA":
            #parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("AR Order (p)", 0, 5, 1)
            with col2:
                d = st.number_input("Difference Order (d)", 0, 2, 1)
            with col3:
                q = st.number_input("MA Order (q)", 0, 5, 1)
            
            if st.button("Train ARIMA Model"):
                model = self.train_arima(y_train, order=(p,d,q))
                predictions = model.forecast(steps=len(y_test))
                
                #show predictions plot
                fig = self.plot_predictions(y_train, 
                                            y_test,
                                            predictions,
                                            "ARIMA Model Predictions")
                st.plotly_chart(fig, use_container_width=True)
                
                #show metrics
                mse = ((predictions - y_test) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(predictions - y_test).mean()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MSE", f"{mse:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("MAE", f"{mae:.2f}")
        #sarimax       
        else:  
            #set parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("AR Order (p)", 0, 5, 1)
            with col2:
                d = st.number_input("Difference Order (d)", 0, 2, 1)
            with col3:
                q = st.number_input("MA Order (q)", 0, 5, 1)
            
            if st.button("Train SARIMAX Model"):
                model = self.train_sarimax(y_train, X_train)
                predictions = model.forecast(steps=len(y_test), exog=X_test)
                
                #predictions plot
                fig = self.plot_predictions(y_train,
                                            y_test, 
                                            predictions, 
                                            "SARIMAX Model Predictions")
                st.plotly_chart(fig, use_container_width=True)
                
                