import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model and preprocessing objects
model = pickle.load(open('parkinson_model.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Set the title of the app
st.title("Parkinson's Disease Prediction App")
st.subheader("Enter the following features to predict Parkinson's disease")

# Create a sidebar for better organization
st.sidebar.header("Input Features")
st.sidebar.markdown("Please enter the values for the following features:")

# Create input fields for the 22 features in the sidebar
features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
            'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
            'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE']

input_data = []
for feature in features:
    # Set the number_input to accept up to 5 decimal places
    input_value = st.sidebar.number_input(
        feature, value=0.0, format="%.5f", step=0.00001)
    input_data.append(input_value)

# Create a button to trigger prediction
if st.sidebar.button('Predict'):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data], columns=features)

    # Apply MinMaxScaler
    scaled_data = scaler.transform(input_df)

    # Apply PCA transformation
    pca_data = pca.transform(scaled_data)

    # Make prediction
    prediction = model.predict(pca_data)

    # Display prediction results in the main area
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.success('The person does not have Parkinson\'s disease')
    else:
        st.error('The person has Parkinson\'s disease')

# Add some additional information or instructions at the bottom
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This application uses a machine learning model to predict whether an individual has Parkinson's disease based on various vocal feature inputs. 
Please ensure that all values are entered accurately for reliable predictions.
""")
