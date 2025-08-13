import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

st.set_page_config(
    page_title="Unsupervised",

    page_icon=Image.open("pages/images/unsupervised.png")  
)

st.markdown(""" The task is to segment persons into different personas
## Algorithms
* **K-Means clustering**
* **Agglomerative Clustering**    
* **DBScan** 
                       
             """)

st.divider()

st.markdown(""" 
* Score using K MEANS : 0.5704114437103271
* Score using Agglomerative Clustering : 0.5546134114265442
* Score using DBSCAN: 0.4086267650127411
""")

st.header("Visualizations")
st.image('pages/images/cluster.png')

st.divider()
st.header("User Personas")
cluster_df=joblib.load("cluster.joblib")
st.dataframe(cluster_df)

