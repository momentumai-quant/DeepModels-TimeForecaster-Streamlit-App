import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_option_menu import option_menu
from pages.stock_info import StockInfoPage
from pages.traditional_models import TraditionalModelsPage
from pages.deep_models import DeepModelsPage

class StockAnalysisApp:
    def __init__(self):
        self.setup_page()
        
    def setup_page(self):
        st.set_page_config(
            page_title="Stock Analysis Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    def hide_sidebar(self):
        st.markdown("""
            <style>
                [data-testid="stSidebar"] {visibility: hidden;}
                [data-testid="stSidebarCollapsedControl"] {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)
        
    def run(self):
        selected_page = option_menu(
            menu_title=None,
            options=["Stock Info", "Traditional Models", "Deep Learning Models"],
            icons=["info-circle", "graph-up", "cpu"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        
        if selected_page == "Stock Info":
            StockInfoPage().show_page()
        elif selected_page == "Traditional Models":
            TraditionalModelsPage().show_page()
        else:
            DeepModelsPage().show_page()
        self.hide_sidebar() 
 
app = StockAnalysisApp()
app.run()