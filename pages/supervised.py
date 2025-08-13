import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image

st.set_page_config(
    page_title="Supervised",

    page_icon=Image.open("pages/images/supervised.png")  
)

st.markdown(""" The task is to predict whether a person is likely to seek mental health treatment (treatment column: yes/no).
 
**ALGORITHMS**
* **Logistic Regression**
* **Random Forest Classifier**    
* **XGBoost** 
* **SVM**                         
             """)

st.divider()
st.header("Logistic Regression")
st.markdown("""The tuned hyperparameters are as follows:
 * *C*: 0.1
 * *penalty* :l2  
 * *solver*  :lbfgs     

 Classification report:
            
""")
st.code("""     precision    recall  f1-score   support

          No       0.73      0.73      0.73       123
         Yes       0.71      0.71      0.71       114

    accuracy                           0.72       237
   macro avg       0.72      0.72      0.72       237
weighted avg       0.72      0.72      0.72       237

""")


st.divider()
st.header("Random Forest Classifier")
st.markdown("""The tuned hyperparameters are as follows:
 * *max_depth*: 3
 * *min_samples_leaf* :3 
 * *min_samples_split* :2
 * *n_estimators* :150               

 Classification report:
            
""")
st.code("""    precision    recall  f1-score   support

          No       0.76      0.67      0.71       123
         Yes       0.68      0.77      0.72       114

    accuracy                           0.72       237
   macro avg       0.72      0.72      0.72       237
weighted avg       0.72      0.72      0.72       237

""")


st.divider()
st.header("XGBoost Classifier")
st.markdown("""The tuned hyperparameters are as follows:
 * *colsample_bytree*: 1.0
 * *learning_rate* :0.01
 * *max_depth* :5
 * *n_estimators* :200 
 * *subsample* :0.8                       

 Classification report:
            
""")
st.code("""   
               precision    recall  f1-score   support

           0       0.74      0.72      0.73       123
           1       0.71      0.73      0.72       114

    accuracy                           0.73       237
   macro avg       0.73      0.73      0.73       237
weighted avg       0.73      0.73      0.73       237


""")


st.divider()
st.header("SVM")
st.markdown("""The tuned hyperparameters are as follows:
 * *C*: 1
 * *gamma* :scale
 * *kernel* :rbf
                    

 Classification report:
            
""")
st.code("""   
                 precision    recall  f1-score   support

           0       0.73      0.75      0.74       123
           1       0.72      0.70      0.71       114

    accuracy                           0.73       237
   macro avg       0.73      0.72      0.73       237
weighted avg       0.73      0.73      0.73       237


""")


st.divider()
st.markdown(""" Next the task is to predict the respondent's age.

**ALGORITHMS**
* **Linear Regression**
* **Decision Tree Regressor** 
* **Random Forest Regressor**    
* **XGBoost Regressor** 
* **SVM**                         
             """)

st.header("Linear regression")
st.markdown("""
* MAE: 4.957805907172996
* RMSE: 6.211803111037516
* R2 Score -0.03646033086543965
            """)

st.header("Decision Tree Regressor")
st.markdown("""
* MAE: 4.8354430379746836
* RMSE: 6.017555189953393
* R2 Score 0.027347997869086593
            """)

st.header("Random Forest  Regressor")
st.markdown("""
* MAE: 4.751054852320675
* RMSE: 5.942054937278612
* R2 Score 0.05160196296533637

            """)

st.header("XGBoost Regressor")
st.markdown("""
* MAE: 4.708860874176025
* RMSE: 5.878516674041748
* R2 Score 0.07177597284317017

            """)

st.header("XGBoost Regressor Log Transformed target")
st.markdown("""
* MAE: 4.675891399383545
* RMSE: 5.899425506591797
* R2 Score 0.06516128778457642
            """)
st.divider()
st.header("Regression Model Comparison")

data = {
    "Model": [
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
        "XGBoost regressor",
        "XGBoost regressor (Log-Transformed) target",
        
    ],
    "MAE": [ 4.95,4.83,4.75,4.70,4.67],
    "RMSE": [6.21,6.01,5.94,5.87,5.89],
    "R² Score": [-0.03,0.02,0.05,0.07,0.06]
}

df = pd.DataFrame(data)

# Display table
st.table(df)









#Form
import streamlit as st
from datetime import datetime

st.title("Mental Health Survey Form")
st.write("Enter the details and know whether a person would seek treatment.")
classi_model=joblib.load("logistic_model.joblib")
with st.form(key="classify"):
    # Timestamp
    timestamp = st.date_input("Timestamp", value=datetime.now())

    # Age
    age = st.number_input("Age",step=1)

    # Gender
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not specified"])

    # Country
    country = st.text_input("Country")

    # State
    state = st.text_input("State")

    # Self-employed
    self_employed = st.radio("Are you self-employed?", ["Yes", "No"])

    # Family history
    family_history = st.radio("Do you have a family history of mental illness?", ["Yes", "No"])

   
    # Work interference
    work_interfere = st.selectbox(
        "If you have a mental health condition, does it interfere with your work?",
        ["Never", "Rarely", "Sometimes", "Often"]
    )

    # Number of employees
    no_employees = st.number_input(
        "How many employees does your company have?",step=1
    )

    # Remote work
    remote_work = st.radio("Do you work remotely at least 50% of the time?", ["Yes", "No"])

    # Tech company
    tech_company = st.radio("Is your employer primarily a tech company?", ["Yes", "No"])

    # Benefits
    benefits = st.selectbox("Does your employer provide mental health benefits?",
                            ["Yes", "No", "Don't know"])

    # Care options
    care_options = st.selectbox("Do you know the options for mental health care your employer provides?",
                                ["Yes", "No", "Not sure"])

    # Wellness program
    wellness_program = st.selectbox("Has your employer discussed mental health in a wellness program?",
                                    ["Yes", "No", "Don't know"])

    # Seek help
    seek_help = st.selectbox("Does your employer provide resources to seek help?",
                             ["Yes", "No", "Don't know"])

    # Anonymity
    anonymity = st.selectbox("Is your anonymity protected if you use treatment resources?",
                             ["Yes", "No", "Don't know"])

    # Leave
    leave = st.selectbox("How easy is it to take medical leave for mental health?",
                         ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])

    # Mental health consequence
    mental_health_consequence = st.selectbox(
        "Would discussing a mental health issue with employer have negative consequences?",
        ["Yes", "No", "Maybe"]
    )

    # Physical health consequence
    phys_health_consequence = st.selectbox(
        "Would discussing a physical health issue with employer have negative consequences?",
        ["Yes", "No", "Maybe"]
    )

    # Coworkers
    coworkers = st.selectbox(
        "Would you discuss a mental health issue with coworkers?",
        ["Yes", "No", "Some of them"]
    )

    # Supervisor
    supervisor = st.selectbox(
        "Would you discuss a mental health issue with your direct supervisor(s)?",
        ["Yes", "No", "Some of them"]
    )

    # Mental health interview
    mental_health_interview = st.selectbox(
        "Would you bring up a mental health issue in an interview?",
        ["Yes", "No", "Maybe"]
    )

    # Physical health interview
    phys_health_interview = st.selectbox(
        "Would you bring up a physical health issue in an interview?",
        ["Yes", "No", "Maybe"]
    )

    # Mental vs Physical
    mental_vs_physical = st.selectbox(
        "Does your employer take mental health as seriously as physical health?",
        ["Yes", "No", "Don't know"]
    )

    # Observed consequence
    obs_consequence = st.selectbox(
        "Have you heard/observed negative consequences for coworkers with mental health conditions?",
        ["Yes", "No"]
    )

    comments = st.text_area("Any additional notes or comments?")
    submit_button1 = st.form_submit_button(label="Submit")





if submit_button1:
   input_df = pd.DataFrame([{
        "Timestamp": timestamp,
        "Age": age,
        "Gender": gender,
        "Country": country,
        "state": state,
        "self_employed": self_employed,
        "family_history": family_history,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence,
        "comments": comments
    }])

    # Predict
   prediction = classi_model.predict(input_df)[0]

   if(prediction=='Yes'):
    st.markdown("### Person is likely to seek treatment")
   else:
    st.markdown("### Person is not likely to seek treatment")



#Age prediction
st.divider()
st.write("Enter the details and know the age of the person")

regress_model=joblib.load("regression_model.joblib")

with st.form(key="regress"):
    # Timestamp
    timestamp = st.date_input("Timestamp", value=datetime.now())



    # Gender
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Not specified"])

    # Country
    country = st.text_input("Country")

    # State
    state = st.text_input("State")

    # Self-employed
    self_employed = st.radio("Are you self-employed?", ["Yes", "No"])

    # Family history
    family_history = st.radio("Do you have a family history of mental illness?", ["Yes", "No"])

    #treatment
    treatment = st.radio("Have you sought treatment for a mental health condition?", ["Yes", "No"])
    # Work interference
    work_interfere = st.selectbox(
        "If you have a mental health condition, does it interfere with your work?",
        ["Never", "Rarely", "Sometimes", "Often"]
    )

    # Number of employees
    no_employees = st.number_input(
        "How many employees does your company have?",step=1
    )

    # Remote work
    remote_work = st.radio("Do you work remotely at least 50% of the time?", ["Yes", "No"])

    # Tech company
    tech_company = st.radio("Is your employer primarily a tech company?", ["Yes", "No"])

    # Benefits
    benefits = st.selectbox("Does your employer provide mental health benefits?",
                            ["Yes", "No", "Don't know"])

    # Care options
    care_options = st.selectbox("Do you know the options for mental health care your employer provides?",
                                ["Yes", "No", "Not sure"])

    # Wellness program
    wellness_program = st.selectbox("Has your employer discussed mental health in a wellness program?",
                                    ["Yes", "No", "Don't know"])

    # Seek help
    seek_help = st.selectbox("Does your employer provide resources to seek help?",
                             ["Yes", "No", "Don't know"])

    # Anonymity
    anonymity = st.selectbox("Is your anonymity protected if you use treatment resources?",
                             ["Yes", "No", "Don't know"])

    # Leave
    leave = st.selectbox("How easy is it to take medical leave for mental health?",
                         ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"])

    # Mental health consequence
    mental_health_consequence = st.selectbox(
        "Would discussing a mental health issue with employer have negative consequences?",
        ["Yes", "No", "Maybe"]
    )

    # Physical health consequence
    phys_health_consequence = st.selectbox(
        "Would discussing a physical health issue with employer have negative consequences?",
        ["Yes", "No", "Maybe"]
    )

    # Coworkers
    coworkers = st.selectbox(
        "Would you discuss a mental health issue with coworkers?",
        ["Yes", "No", "Some of them"]
    )

    # Supervisor
    supervisor = st.selectbox(
        "Would you discuss a mental health issue with your direct supervisor(s)?",
        ["Yes", "No", "Some of them"]
    )

    # Mental health interview
    mental_health_interview = st.selectbox(
        "Would you bring up a mental health issue in an interview?",
        ["Yes", "No", "Maybe"]
    )

    # Physical health interview
    phys_health_interview = st.selectbox(
        "Would you bring up a physical health issue in an interview?",
        ["Yes", "No", "Maybe"]
    )

    # Mental vs Physical
    mental_vs_physical = st.selectbox(
        "Does your employer take mental health as seriously as physical health?",
        ["Yes", "No", "Don't know"]
    )

    # Observed consequence
    obs_consequence = st.selectbox(
        "Have you heard/observed negative consequences for coworkers with mental health conditions?",
        ["Yes", "No"]
    )

    comments = st.text_area("Any additional notes or comments?")

    submit_button = st.form_submit_button(label="Submit")





if submit_button:
   input_df2 = pd.DataFrame([{
        "Timestamp": timestamp,
        
        "Gender": gender,
        "Country": country,
        "state": state,
        "self_employed": self_employed,
        "family_history": family_history,
        "treatment": treatment,
        "work_interfere": work_interfere,
        "no_employees": no_employees,
        "remote_work": remote_work,
        "tech_company": tech_company,
        "benefits": benefits,
        "care_options": care_options,
        "wellness_program": wellness_program,
        "seek_help": seek_help,
        "anonymity": anonymity,
        "leave": leave,
        "mental_health_consequence": mental_health_consequence,
        "phys_health_consequence": phys_health_consequence,
        "coworkers": coworkers,
        "supervisor": supervisor,
        "mental_health_interview": mental_health_interview,
        "phys_health_interview": phys_health_interview,
        "mental_vs_physical": mental_vs_physical,
        "obs_consequence": obs_consequence,
        "comments": comments
    }])
   
   Age_prediction=regress_model.predict(input_df2)[0]

   Age_prediction=int(Age_prediction)

   st.markdown(f"### Age: {Age_prediction}")


   

   

        
        
    





