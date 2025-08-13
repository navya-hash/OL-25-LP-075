import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image


st.set_page_config(
    page_title="EDA",

    page_icon=Image.open("pages/images/eda.png")  
)



df=pd.read_csv('survey.csv')
st.header("Datset Overview")

st.dataframe(df.head())
st.subheader("Null counts and data types")
data_null,dtypes=st.columns(2)

with data_null:
    st.code(df.isna().sum())
with dtypes:
    st.code(df.dtypes)

st.subheader("Dataset after removing invalid Data")
df_clean=joblib.load('df_clean.joblib')
st.dataframe(df_clean)

st.divider()

st.header("Univariate feature analysis")
st.image("pages/images/uni1.png")
cols=st.columns(2)
with cols[0]:
   
   gender_treatment_percent=df_clean.groupby('Gender')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by gender:")
   st.write(gender_treatment_percent)
   
with cols[1]:
   
   age_treatment_percent=df_clean.groupby('Age')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by age:")
   st.write(age_treatment_percent)

st.divider()
cols=st.columns(2)
st.image("pages/images/uni2.png")

with cols[1]:
   
   company_size_treatment_percent=df_clean.groupby('no_employees')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by company size:")
   st.write(company_size_treatment_percent)
   
with cols[0]:
   
   work_treatment_percent=df_clean.groupby('work_interfere')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by work interference:")
   st.write(work_treatment_percent)

st.divider()
cols=st.columns(2)
st.image("pages/images/uni3.png")


with cols[0]:
   
   fam_his_treatment_percent=df_clean.groupby('family_history')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Family history:")
   st.write(fam_his_treatment_percent)

with cols[1]:
   
   benefits_treatment_percent=df_clean.groupby('benefits')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Benefits:")
   st.write(benefits_treatment_percent)

st.divider()
cols=st.columns(2)
st.image("pages/images/uni4.png")


with cols[0]:
   
   care_options_treatment_percent=df_clean.groupby('care_options')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Care options:")
   st.write(care_options_treatment_percent)

with cols[1]:
   
   anonymity_treatment_percent=df_clean.groupby('anonymity')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Anonymity:")
   st.write(anonymity_treatment_percent)


st.divider()
cols=st.columns(2)
st.image("pages/images/uni5.png")


with cols[0]:
   
   mental_health_consequence_treatment_percent=df_clean.groupby('mental_health_consequence')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Mental health consequence:")
   st.write(mental_health_consequence_treatment_percent)

with cols[1]:
   
   coworkers_treatment_percent=df_clean.groupby('coworkers')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by coworkers:")
   st.write(coworkers_treatment_percent)



st.divider()
cols=st.columns(2)
st.image("pages/images/uni6.png")


with cols[0]:
   
   supervisor_treatment_percent=df_clean.groupby('supervisor')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by supervisor:")
   st.write(supervisor_treatment_percent)

with cols[1]:
   
   mental_health_interview_treatment_percent=df_clean.groupby('mental_health_interview')['treatment'].value_counts(normalize=True).loc[:, 'Yes'] * 100
   st.markdown("### Treatment rates by Mental health interview:")
   st.write(mental_health_interview_treatment_percent)





st.divider()
st.header("Multivariate feature analysis")
st.image("pages/images/heatmap.png")
