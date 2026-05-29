# Import libraries
import streamlit as st
import pickle
import numpy as np


# LOAD MODEL + SCALER

model = pickle.load(open("Customer_churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# APP TITLE

st.title("Customer Churn Prediction App")
st.write("Enter customer information below:")


# NUMERICAL INPUTS

credit_score = st.number_input("Credit Score")
tenure = st.number_input("Tenure")
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products")
estimated_salary = st.number_input("Estimated Salary")
age = st.number_input("Age")
satisfaction = st.number_input("Satisfaction Score")
points = st.number_input("Point Earned")


# BINARY / CATEGORICAL INPUTS

is_active = st.selectbox("Is Active Member", [0, 1])
has_card = st.selectbox("Has Credit Card", [0, 1])

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
card_type = st.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM"])


# ENCODING

gender_male = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

card_gold = 1 if card_type == "GOLD" else 0
card_platinum = 1 if card_type == "PLATINUM" else 0
card_silver = 1 if card_type == "SILVER" else 0


# PREDICTION

if st.button("Predict Churn"):

    # MUST match training feature order (16 features)
    features = np.array([[
        credit_score,
        tenure,
        balance,
        num_products,
        is_active,
        estimated_salary,
        age,
        has_card,
        satisfaction,
        points,
        geo_germany,
        geo_spain,
        gender_male,
        card_gold,
        card_platinum,
        card_silver
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)

    # Probability
    probability = model.predict_proba(scaled_features)[0][1]

    # Output
    if prediction[0] == 1:
        st.error("⚠ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"Churn Probability: {probability:.2f}")