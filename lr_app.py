import streamlit as st 
import numpy as np 
import pandas as pd 
import sklearn 
import pickle 

model = pickle.load(open('linear_regression_model.pkl','rb'))
st.title("Sales Prediction through advertising expenditure")
tv = st.text_input('Enter Tv sales.........')
radio = st.text_input('Enter Radio sales...')
newspaper = st.text_input('Enter newspaper sales....')

if st.button("predict"):
    features = np.array([[tv,radio,newspaper]], dtype=np.float64)
    results = model.predict(features).reshape(1,-1)
    st.write("Predicted sales::::",results[0])


