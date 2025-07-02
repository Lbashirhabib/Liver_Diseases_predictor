import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model
@st.cache_resource
def load_model():
    with open('liver_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# App title
st.title('Liver Disease Prediction System')

# Sidebar with information
st.sidebar.header('About')
st.sidebar.info(
    """
    This app predicts the stage of liver disease based on clinical parameters.
    """
)

# Input form
st.header('Patient Information')

# Create input fields for all your model features
# (Replace these with the actual features your model uses)
age = st.number_input('Age', min_value=0, max_value=120, value=45)
gender = st.selectbox('Gender', ['Male', 'Female'])
total_bilirubin = st.number_input('Total Bilirubin (mg/dL)', min_value=0.0, value=0.7)
direct_bilirubin = st.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, value=0.1)
alkaline_phosphotase = st.number_input('Alkaline Phosphotase (IU/L)', min_value=0, value=187)
alamine_aminotransferase = st.number_input('Alamine Aminotransferase (IU/L)', min_value=0, value=16)
aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase (IU/L)', min_value=0, value=18)
total_proteins = st.number_input('Total Proteins (g/dL)', min_value=0.0, value=6.8)
albumin = st.number_input('Albumin (g/dL)', min_value=0.0, value=3.3)
albumin_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, value=0.9)

# Convert gender to numerical value
gender_num = 1 if gender == 'Male' else 0

# Create feature dictionary
features = {
    'Age': age,
    'Gender': gender_num,
    'Total_Bilirubin': total_bilirubin,
    'Direct_Bilirubin': direct_bilirubin,
    'Alkaline_Phosphotase': alkaline_phosphotase,
    'Alamine_Aminotransferase': alamine_aminotransferase,
    'Aspartate_Aminotransferase': aspartate_aminotransferase,
    'Total_Protiens': total_proteins,
    'Albumin': albumin,
    'Albumin_and_Globulin_Ratio': albumin_globulin_ratio
}

# Convert to DataFrame
input_df = pd.DataFrame([features])

# Prediction button
if st.button('Predict Liver Disease Stage'):
    try:
        # Preprocess the input data
        scaler = StandardScaler()
        X = scaler.fit_transform(input_df)
        
        # Make prediction
        prediction = model.predict(X)
        
        # Display prediction
        st.subheader('Prediction Result')
        
        # Map stage numbers to meaningful descriptions
        stage_descriptions = {
            1: 'Stage 1: Mild or no liver disease',
            2: 'Stage 2: Moderate liver disease',
            3: 'Stage 3: Severe liver disease',
            4: 'Stage 4: Very severe or end-stage liver disease'
        }
        
        predicted_stage = prediction[0]
        st.success(f'Predicted Liver Disease Stage: {predicted_stage} - {stage_descriptions.get(predicted_stage, "Unknown stage")}')
        
        # Show probability distribution if it's a classifier with probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            prob_df = pd.DataFrame({
                'Stage': [1, 2, 3, 4],
                'Probability': probabilities
            })
            st.bar_chart(prob_df.set_index('Stage'))
        
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

