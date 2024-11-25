import streamlit as st
import plotly.graph_objects as go
from utilities.data_processor import DataProcessor

class StockInfoPage:
    def __init__(self):
        self.data_processor = DataProcessor()
        
    def show_page(self):
        st.title("Stock Price Analysis")
        
        with st.spinner("Loading data..."):
            df = self.data_processor.load_data()
        
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", df.index.min())
        with col2:
            end_date = st.date_input("End Date", df.index.max())
        
        # date range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        df_filtered = df[mask]
        
        #ohlc chart
        fig = go.Figure(data=[go.Candlestick(
            x=df_filtered.index,
            open=df_filtered['Open'],
            high=df_filtered['High'],
            low=df_filtered['Low'],
            close=df_filtered['Price']
        )])
        
        fig.update_layout(
            title="Stock OHLC Chart",
            yaxis_title="Price",
            xaxis_title="Date",
            template="plotly_dark",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # show statistics
        st.subheader("Price Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${df_filtered['Price'].iloc[-1]:,.2f}")
        with col2:
            st.metric("Highest Price", f"${df_filtered['High'].max():,.2f}")
        with col3:
            st.metric("Lowest Price", f"${df_filtered['Low'].min():,.2f}")
        with col4:
            price_change = df_filtered['Price'].iloc[-1] - df_filtered['Price'].iloc[0]
            st.metric("Price Change", f"${price_change:,.2f}")