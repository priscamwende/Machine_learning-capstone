# Import libraries
import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open('Customer_churn_model.pkl', 'rb'))

# Load scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# App title
st.title("Customer Churn Prediction App")

st.write("Enter customer information below:")

# User inputs
credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products")
estimated_salary = st.number_input("Estimated Salary")

# Prediction button
if st.button("Predict Churn"):

    # Convert input into array
    features = np.array([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        estimated_salary
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    # Probability
    probability = model.predict_proba(scaled_features)[0][1]

    # Output
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")

    st.write(f"Churn Probability: {probability:.2f}")