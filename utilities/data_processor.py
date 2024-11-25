import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

class DataProcessor:
    @staticmethod
    def load_data(file_path="XAU-USD-2018-2024.csv"):
        START_DATE = '2022-01-01'
        END_DATE = '2024-11-15'
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[(df["Date"]>=START_DATE) & (df["Date"]<=END_DATE) ]

        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df.drop(['Vol.'], axis=1, inplace=True)
        
        #convert string values to float
        for column in df.columns:
            df[column] = df[column].str.replace(',', '')
            df[column] = df[column].str.replace('%', '')
            df[column] = df[column].astype(float)
        
        return df
    
    @staticmethod
    def prepare_features(df):
        #lagged features
        base_columns = ["Price", "High", "Low", "Open"]
        for col in base_columns:
            for lag in [1, 5]:
                df[f"{col}_lag_{lag}"] = df[col].shift(-lag)
        
        #technical indicators
        for col in ["Price"]:
            for sma in [9, 13]:
                df[f"{col}_sma_{sma}"] = df[col].rolling(window=sma).mean()
        
        df.dropna(inplace=True)
        return df
    
    @staticmethod
    def prepare_data_for_traditional_models(df, test_size=30):
        features = [
            'Open_lag_1',
            'High_lag_1',
            'Low_lag_1',
            'Price_lag_5',
            'Price_sma_9',
            'Price_sma_13',
        ]
        
        X = df[features]
        y = df['Price']
        
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def prepare_data_for_deep_models(df, feature_cols, target_col='Price', test_size=30, timesteps=6):
        df = df.asfreq('B')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        series_features = TimeSeries.from_dataframe(df, value_cols=feature_cols)
        series_target = TimeSeries.from_dataframe(df, value_cols=target_col)
        
        scaler_features = Scaler()
        scaler_target = Scaler()
        
        series_features_scaled = scaler_features.fit_transform(series_features)
        series_target_scaled = scaler_target.fit_transform(series_target)
        
        train_target = series_target_scaled[:-test_size]
        test_target = series_target_scaled[-test_size:]
        train_features = series_features_scaled[:-test_size]
        test_features = series_features_scaled[-test_size:]
        
        return (train_features, test_features, train_target, test_target,
                scaler_features, scaler_target)