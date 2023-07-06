import os
import pandas as pd
import streamlit as st

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("https://www.pngitem.com/pimgs/m/76-761296_machine-learning-model-icon-hd-png-download.png")
    st.title("AUTOMATED ML APP")
    st.info("This application will allow you to build an automated Machine Learning Pipeline using Streamlit, Pandas Profiling, and PyCaret.")

    choice = st.radio("NAVIGATION ↪", ["Upload", "Profiling", "Modelling", "Download"])

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling!")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "Modelling":
    st.title("Machine Learning")
    target = st.selectbox("Choose your Target column", df.columns)
    if st.button("Train Model"):
    
        categorical_cols = df.select_dtypes(include=['object']).columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
        setup(df_encoded, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings.")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')
        
if choice == "Download":
    st.title("Model Download")
    st.info("Click to download your trained model.")
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
