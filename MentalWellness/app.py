

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# def home():


#     st.title("Problem Statement")
#     st.markdown(""" As a Machine Learning Engineer at NeuronInsights Analytics, you’ve been contracted by a coalition of
# leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
# disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
# proactively identify and support at-risk employees. Your role is to analyze survey data from over 1,500 tech
# professionals, covering workplace policies, personal mental health history, openness to seeking help, and
# perceived employer support.""")
    
#     st.header("Datset Overview")
#     st.markdown("""Dataset Source:[*Mental Health in Tech Survey*](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
#             * Collected by OSMI (Open Sourcing Mental Illness)""")
    
#     df=pd.read_csv('survey.csv')
#     st.header("Dataset")
#     st.dataframe(df)
#     st.header("Initial 5 rows")
#     st.code(df.head())
#     st.subheader("Shape")
#     st.code(df.shape)
#     st.divider()
#     st.subheader("Null counts and data types")
#     data_null,dtypes=st.columns(2)

#     with data_null:
#         st.code(df.isna().sum())
#     with dtypes:
#         st.code(df.dtypes)

# app1=st.navigation(
#     [
#         st.Page(home,title="Project Overview"),
#         st.Page("pages/EDA.py",title="Data exploration and analysis"),
#         st.Page("pages/supervised.py",title="Age prediction and Treatment classification"),
#         st.Page("pages/unsupervised.py",title="Distinct personas")
#     ]
# )

# app1.run()





import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

st.set_page_config(
    page_title="Project Overview",
    page_icon=Image.open("MentalWellnesspages/images/app.png")  # icon for home page
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
