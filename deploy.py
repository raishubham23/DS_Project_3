# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:39:07 2023

@author: saura
"""

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained Random Forest model
model = joblib.load("random_forest_model.pkl")  # Make sure to use the correct path to your .pkl file

# Create a Streamlit web app
st.title("Bankruptcy Prediction App")


# Create input fields for user input
st.header("Input Features")

feature1 = st.selectbox("Industrial Risk", [0, 0.5, 1])
feature2 = st.selectbox("Management Risk", [0, 0.5, 1])
feature3 = st.selectbox("Financial Flexibility", [0, 0.5, 1])
feature4 = st.selectbox("Credibility", [0, 0.5, 1])
feature5 = st.selectbox("Competitiveness", [0, 0.5, 1])
feature6 = st.selectbox("Operating Risk", [0, 0.5, 1])

# Create a button to make predictions
if st.button("Predict"):
    # Create an array with the user's input
    user_input = np.array([feature1, feature2, feature3, feature4, feature5, feature6]).reshape(1, -1)

    # Use the model to make predictions
    prediction = model.predict(user_input)
    probabilities = model.predict_proba(user_input)

    # Convert the prediction to 'Non-Bankruptcy' (1) or 'Bankruptcy' (0)
    if prediction[0] == 1:
        result = "Non-Bankruptcy"
    else:
        result = "Bankruptcy"

    # Display the prediction to the user
    st.header("Prediction")
    st.write(f"Prediction : {result}")

    # Create a pie chart to show the probabilities with legends
    st.header("Probability Chart")
    labels = ['Bankruptcy', 'Non-Bankruptcy']
    sizes = [probabilities[0][0], probabilities[0][1]]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')

    # Add legends to the chart
    ax1.legend(labels, title="Legend", loc="upper right")

    st.pyplot(fig1)



