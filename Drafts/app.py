import pickle
import streamlit as st

# Load the trained model from file
saved_model = pickle.load(open('parkinson_model.sav', 'rb'))

# Set up the title and description
st.title("Parkinson's Disease Prediction")
st.write("""
This app predicts whether a person has Parkinson's disease based on input attributes. 
Please provide the following values:
""")

# Create columns for input fields
col1, col2, col3, col4, col5 = st.columns(5)

# Input fields for user data
with col1:
    MDVP_Fo = st.number_input(
        'Average vocal fundamental frequency (Hz)', min_value=80, max_value=270, step=1)
    MDVP_Fhi = st.number_input(
        'Maximum vocal fundamental frequency (Hz)', min_value=100, max_value=600, step=1)
    MDVP_Flo = st.number_input(
        'Minimum vocal fundamental frequency (Hz)', min_value=60, max_value=240, step=1)
    MDVP_Jitter = st.number_input(
        'Jitter (percentage)', min_value=0.0, max_value=0.03, step=0.001)
    MDVP_RAP = st.number_input(
        'RAP (Relative Amplitude Perturbation)', min_value=0.0, max_value=0.02, step=0.001)

with col2:
    MDVP_PPQ = st.number_input(
        'PPQ (Period Perturbation Quotient)', min_value=0.0, max_value=0.02, step=0.001)
    Jitter_DDP = st.number_input(
        'DDP (Degree of Perturbation)', min_value=0.0, max_value=0.06, step=0.001)
    MDVP_Shimmer = st.number_input(
        'Shimmer (dB)', min_value=0.01, max_value=0.12, step=0.001)
    MDVP_Shimmer_dB = st.number_input(
        'Shimmer in dB', min_value=0.09, max_value=1.3, step=0.01)
    Shimmer_APQ3 = st.number_input(
        'Shimmer_APQ3', min_value=0.0, max_value=0.06, step=0.001)

with col3:
    Shimmer_APQ5 = st.number_input(
        'Shimmer_APQ5', min_value=0.01, max_value=0.08, step=0.001)
    MDVP_APQ = st.number_input(
        'APQ (Amplitude Perturbation Quotient)', min_value=0.01, max_value=0.14, step=0.001)
    Shimmer_DDA = st.number_input(
        'Shimmer_DDA', min_value=0.01, max_value=0.17, step=0.001)
    NHR = st.number_input('NHR (Noise-to-Harmonics Ratio)',
                          min_value=0.0, max_value=0.31, step=0.01)
    HNR = st.number_input('HNR (Harmonics-to-Noise Ratio)',
                          min_value=8, max_value=35, step=1)

with col4:
    RPDE = st.number_input('RPDE (Recurrence Period Density Entropy)',
                           min_value=0.25, max_value=0.70, step=0.01)
    DFA = st.number_input('DFA (Detrended Fluctuation Analysis)',
                          min_value=0.55, max_value=0.85, step=0.01)
    spread1 = st.number_input(
        'Spread 1', min_value=-7.9, max_value=-2.4, step=0.1)
    spread2 = st.number_input(
        'Spread 2', min_value=0.01, max_value=0.45, step=0.01)
    D2 = st.number_input('D2 (Dynamical Complexity)',
                         min_value=1.4, max_value=3.7, step=0.1)

with col5:
    PPE = st.number_input('PPE (Pitch Period Entropy)',
                          min_value=0.04, max_value=0.55, step=0.01)

# Placeholder for the result
parkinson_status = ''

# Make predictions
if st.button('Predict Parkinson Status'):
    # Prepare the input data in the correct format
    input_features = [
        MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
        MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
        NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
    ]

    input_features = [input_features]  # 2D array for prediction
    prediction = saved_model.predict(input_features)

    # Interpret prediction
    if prediction[0] == 1:
        parkinson_status = 'This person has Parkinson\'s disease.'
    else:
        parkinson_status = "This person is healthy and doesn't have the disease."

st.success(parkinson_status)
