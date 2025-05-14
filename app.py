import streamlit as st
import pandas as pd

import input_data as data
import data_visualization as dvz
import data_cleansing as dcsg
import data_preprocessing as dppc
import data_modeling as dm
import competition_page as comp

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
footer > div:first-of-type {visibility: hidden;}

/* Sidebar styling */
div[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}

/* Navigation items */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    padding: 10px 20px;
}

/* Title text styling */
.title-text {
    font-size: 1.5em;
    color: #ffbb56 !important;
    margin: 20px 0;
    font-weight: bold;
    text-align: center;
}

/* Custom button style */
.stButton > button {
    background-color: #0068c9 !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 1.2em !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background-color: #0052a3 !important;
    transform: scale(1.05) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* Copyright footer */
.copyright {
    position: fixed;
    bottom: 10px;
    left: 50%;
    transform: translateX(-50%);
    font-family: Arial, sans-serif;
    color: #ffffff;
    font-size: 14px;
    text-align: center;
    opacity: 0.8;
    z-index: 999;
}
</style>
"""

st.markdown(hide_st_style, unsafe_allow_html=True)

def start_page():
    st.markdown("""
        <div class="start-container">
            <div class="vertical-center">
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("big_data.png", use_container_width=True)
        st.markdown('<div class="title-text">Machine Learning Toolkit</div>', unsafe_allow_html=True)
        if st.button("START YOUR JOURNEY", 
                   use_container_width=True,
                   type="primary",
                   key="start_button"):
            st.session_state.page = "main"
            st.rerun()
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def main_page():
    # Sidebar navigation with logo
    with st.sidebar:
        st.image("big_data.png", use_container_width=True)
        selection = st.radio("Navigation Menu", 
                           ["🚀 Data", "📊 Visualization", "🧹 Cleansing", 
                            "⚙️ Preprocessing", "🤖 Modeling", "🏆 Competition"], 
                           key="navbar")

    # Main content area
    if selection == "🚀 Data":
        data.data_page()
    elif selection == "📊 Visualization":
        dvz.data_visualization_page()
    elif selection == "🧹 Cleansing":
        dcsg.data_cleansing_page()
    elif selection == "⚙️ Preprocessing":
        dppc.data_preprocessing_page()
    elif selection == "🤖 Modeling":
        dm.modeling_page()
    elif selection == "🏆 Competition":
        comp.competition()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "start"
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

# Handle page navigation
if st.session_state.page == "start":
    start_page()
else:
    main_page()

# Copyright footer
st.markdown("""
    <div class="copyright">
        © 2024 MBC Big Data
    </div>
""", unsafe_allow_html=True)
