import streamlit as st
import plotly.graph_objects as go
from darts.models import (
    TCNModel,
    TransformerModel,
    BlockRNNModel
)
from darts.metrics import mape, rmse
import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Tuple
from utilities.data_processor import DataProcessor
from darts import concatenate

class DeepModelsPage:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {
            'TCN': TCNModel, 
            'BlockRNN': BlockRNNModel,
            'Transformer': TransformerModel
        }
        self.model_configs = {
            'TCN': {
                'input_chunk_length': 6,
                'output_chunk_length': 1,
                'num_layers': 3,
                'num_filters': 32,
                'batch_size': 64,
                'n_epochs': 5
            }, 
            'BlockRNN': {
                'input_chunk_length': 6,
                'output_chunk_length': 1,
                'batch_size': 64,
                'n_epochs': 5,
                'hidden_dim': 64,
                'model': 'LSTM'
            },
            'Transformer': {
                'input_chunk_length': 6,
                'output_chunk_length': 1,
                'batch_size': 64,
                'n_epochs': 10,
                'd_model': 64,
                'nhead': 4,
                'num_encoder_layers': 3
            }
        }

    def show_page(self):
        st.title("Deep Learning Models")
        
        #load and prepare data (with selected features from data preprocessor)
        df = self.data_processor.load_data()
        df = self.data_processor.prepare_features(df)
        
        #deep model selection
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model = st.selectbox(
                "Select Model",
                options=list(self.models.keys())
            )
            
            train_button = st.button("Train Model", type="primary")
        
        #model configuration (hyper params)
        with col2:
            st.subheader("Model Configuration")
            config = self.display_model_config(selected_model)
        
        #training model section
        if train_button:
            self.handle_training(df, selected_model, config)

    def display_model_config(self, model_name: str) -> Dict[str, Any]: 
        config = self.model_configs[model_name].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            config['input_chunk_length'] = st.number_input(
                "Input Chunk Length",
                min_value=1,
                value=config['input_chunk_length']
            )
            config['batch_size'] = st.number_input(
                "Batch Size",
                min_value=1,
                value=config['batch_size']
            )
            config['n_epochs'] = st.number_input(
                "Number of Epochs",
                min_value=1,
                value=config['n_epochs']
            )
        
        with col2:
            if model_name == 'TCN':
                config['num_filters'] = st.number_input(
                    "Number of Filters",
                    min_value=1,
                    value=config['num_filters']
                )
                config['num_layers'] = st.number_input(
                    "Number of Layers",
                    min_value=1,
                    value=config['num_layers']
                )
            
            elif model_name == 'Transformer':
                config['d_model'] = st.number_input(
                    "D Model",
                    min_value=1,
                    value=config['d_model']
                )
                config['nhead'] = st.number_input(
                    "Number of Heads",
                    min_value=1,
                    value=config['nhead']
                )
            
            elif model_name == 'BlockRNN':
                config['hidden_dim'] = st.number_input(
                    "Hidden Dimension",
                    min_value=1,
                    value=config['hidden_dim']
                )
                config['model'] = st.selectbox(
                    "RNN Type",
                    options=['LSTM', 'GRU', 'RNN'],
                    index=0
                )
        
        return config

 
 
    def handle_training(self, df: pd.DataFrame, model_name: str, config: Dict[str, Any]): 
        feature_cols = ['Open_lag_1',
                      'High_lag_1',
                      'Low_lag_1',
                      'Price_lag_5',
                      'Price_sma_9',
                      'Price_sma_13']
        
        with st.spinner("Preparing data..."):
            #get splited data and scalers
            (train_features, test_features, 
             train_target, test_target,
             scaler_features, scaler_target) = self.data_processor.prepare_data_for_deep_models(
             df, feature_cols
            )

        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        #init selected model
        model = self.models[model_name](**config)
        
        #training with progress updates
        start_time = time.time()
        for epoch in range(config['n_epochs']):
            status_text.text(f"Training epoch {epoch + 1}/{config['n_epochs']}")
            model.fit(
                train_target,
                past_covariates=train_features,
                verbose=False
            )
            progress = (epoch + 1) / config['n_epochs']
            progress_bar.progress(progress)
        
        training_time = time.time() - start_time
        
        # make predictions with one step forecasting
        with st.spinner("Generating predictions..."):
            predictions = []
            historical_target = train_target
            
            for i in range(len(test_target)):
                #one step prediction
                pred = model.predict(
                    n=1,
                    series=historical_target,
                    past_covariates=train_features.append(test_features[:i]) if i > 0 else train_features
                )
                predictions.append(pred)
                
                #update historical data
                historical_target = historical_target.append(test_target[i:i+1])
            
            #all predictions
            all_predictions = concatenate(predictions, axis=0)
            
            #inverse predictions and actual values
            predictions_original = scaler_target.inverse_transform(all_predictions)
            test_original = scaler_target.inverse_transform(test_target)
            
         
            mape_score = mape(test_original, predictions_original)
            rmse_score = rmse(test_original, predictions_original)
            
             
            self.display_results(
                predictions_original,
                test_original,
                mape_score,
                rmse_score,
                training_time
            )

    def  display_results(
        self,
        predictions,
        actual,
        mape_score ,
        rmse_score ,
        training_time 
    ):
     
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAPE", f"{mape_score:.2f}%")
        col2.metric("RMSE", f"{rmse_score:.2f}")
        col3.metric("Training Time", f"{training_time:.2f}s")
        
         
        st.subheader("Predictions vs Actual Values")
        fig = go.Figure()
        
         
        fig.add_trace(go.Scatter(
            x=actual.time_index,
            y=actual.values().flatten(),
            name="Actual",
            line=dict(color='blue')
        ))
        
        #predictions
        fig.add_trace(go.Scatter(
            x=predictions.time_index,
            y=predictions.values().flatten(),
            name="Predicted",
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Model Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)