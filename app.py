import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained model
with open("LinearRegression.pkl", "rb") as file:
    model = pickle.load(file)

# Define the feature columns (based on your one-hot encoding in cleaning.ipynb)
feature_columns = [
    'math score', 'reading score', 'writing score',
    'gender_female', 'gender_male',
    'test preparation course_completed', 'test preparation course_none',
    'race/ethnicity_group A', 'race/ethnicity_group B', 'race/ethnicity_group C',
    'race/ethnicity_group D', 'race/ethnicity_group E',
    'parental level of education_associate\'s degree',
    'parental level of education_bachelor\'s degree',
    'parental level of education_high school',
    'parental level of education_master\'s degree',
    'parental level of education_some college',
    'parental level of education_some high school',
    'lunch_free/reduced', 'lunch_standard'
]

# Function to preprocess user input into model-ready format
def preprocess_input(data):
    input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    
    input_df['math score'] = data['math_score']
    input_df['reading score'] = data['reading_score']
    input_df['writing score'] = data['writing_score']
    
    input_df[f"gender_{data['gender'].lower()}"] = 1
    input_df[f"test preparation course_{data['test_preparation_course'].lower()}"] = 1
    input_df[f"race/ethnicity_group {data['race_ethnicity'].split()[1]}"] = 1
    input_df[f"parental level of education_{data['parental_education'].lower()}"] = 1
    input_df[f"lunch_{data['lunch'].lower()}"] = 1
    
    return input_df

st.title("Student Performance Prediction")
st.write("Enter the student details below to predict their Mean Score.")

with st.form(key="prediction_form"):
    math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

    gender = st.selectbox("Gender", ["Female", "Male"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
    parental_education = st.selectbox(
        "Parental Level of Education",
        ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
    )
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])

    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    input_data = {
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_education": parental_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course
    }
    
    processed_input = preprocess_input(input_data)
    
    prediction = model.predict(processed_input)[0]
    
    st.success(f"Predicted Score: **{prediction:.2f}**")
    st.write("Note: Mean Score is calculated as (math score + reading score + writing score / 3).")

st.write("### How to Use")
st.write("1. Enter the student's scores and select their demographic details.")
st.write("2. Click 'Predict' to see the estimated Mean Score.")
st.write("3. The model is trained on the StudentsPerformance dataset using Linear Regression.")