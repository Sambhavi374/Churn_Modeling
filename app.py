import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model


# load the model
model = load_model('model.h5')

# load the encoder and scaler
with open('geo_encoder.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)     


## streamlit app

st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on their information.")
st.write("Please enter the customer information below:")

# input fields
geography = st.selectbox("Geography", geo_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100)
balance = st.number_input("Balance", min_value=0.0, max_value=100000.0)
credit_score = st.number_input("Credit Score", min_value=0)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=1000000.0, value=50000.0)
tenure = st.slider("Tenure", min_value=0, max_value=50, value=5)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
 
})

# encode the geography

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

# Combine ohe colums with the rest of the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


# Scale the data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display the prediction
st.write(f"Prediction Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
    prediction_label = "Churn"
else:
    st.write('The customer is not likely to churn.')
    prediction_label = "Not Churn"

