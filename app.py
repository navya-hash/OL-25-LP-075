


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

st.set_page_config(
    page_title="Project Overview",
    page_icon=Image.open("pages/images/app.png")  # icon for home page
)

def home():
    st.title("Problem Statement")
    st.markdown("""
    As a Machine Learning Engineer at NeuronInsights Analytics, you’ve been contracted...
    """)
    
    st.header("Dataset Overview")
    st.markdown("""
    Dataset Source: [*Mental Health in Tech Survey*](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)  
    * Collected by OSMI (Open Sourcing Mental Illness)
    """)
    
    df = pd.read_csv('survey.csv')
    st.header("Dataset")
    st.dataframe(df)
    st.header("Initial 5 rows")
    st.code(df.head())
    st.subheader("Shape")
    st.code(df.shape)
    st.divider()
    st.subheader("Null counts and data types")
    data_null, dtypes = st.columns(2)

    with data_null:
        st.code(df.isna().sum())
    with dtypes:
        st.code(df.dtypes)

# Run the home page
home()
