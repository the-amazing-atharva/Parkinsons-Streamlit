import pickle
import streamlit as st
saved_model = pickle.load(open('parkinson_model.sav', 'rb'))
st.title("Parkinson's Disease Prediction ")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    MDVP_Fo = st.number_input(
        'Average vocal fundamental frequency', min_value=80, max_value=270)
    MDVP_Fhi = st.number_input(
        'Maximum vocal fundamental frequency', min_value=100, max_value=600)
    MDVP_Flo = st.number_input(
        'Minimum vocal fundamental frequency', min_value=60, max_value=240)
    MDVP_Jitter = st.number_input(
        'Variation in fundamental frequency-Jitter', min_value=0.0, max_value=0.03)
    MDVP_RAP = st.number_input(
        'Variation in fundamental frequency-RAP', min_value=0.0, max_value=0.02)
with col2:
    MDVP_PPQ = st.number_input(
        'Variation in fundamental frequency-PPQ', min_value=0.0, max_value=0.02)
    Jitter_DDP = st.number_input(
        'Variation in fundamental frequency-JitterDDp', min_value=0.0, max_value=0.06)
    MDVP_Shimmer = st.number_input(
        'Amplitude of NHR, HNR', min_value=0.01, max_value=0.12)
    MDVP_Shimmer_dB = st.number_input(
        'Amplitude of NHR, HNR (in dB)', min_value=0.09, max_value=1.3)
    Shimmer_APQ3 = st.number_input(
        'Shimmer_APQ3', min_value=0.0, max_value=0.06)
with col3:
    Shimmer_APQ5 = st.number_input(
        'Shimmer_APQ5', min_value=0.01, max_value=0.08)
    MDVP_APQ = st.number_input(
        'Variation in fundamental frequency-APQ', min_value=0.01, max_value=0.14)
    Shimmer_DDA = st.number_input(
        'Shimmer_DDA', min_value=0.01, max_value=0.17)
    NHR = st.number_input('NHR', min_value=0.0, max_value=0.31)
    HNR = st.number_input('HNR', min_value=8, max_value=35)
with col4:
    RPDE = st.number_input('RPDE', min_value=0.25, max_value=0.70)
    DFA = st.number_input('DFA', min_value=0.55, max_value=0.85)
    spread1 = st.number_input('Spread_1', min_value=-7.9, max_value=-2.4)
    spread2 = st.number_input('Spread_2', min_value=0.01, max_value=0.45)
    D2 = st.number_input('D2', min_value=1.4, max_value=3.7)
with col5:
    PPE = st.number_input('PPE', min_value=0.04, max_value=0.55)

parkison_status = ''
if st.button('Parkison Predict Status'):
    input_features = [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
                      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA,
                      NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
    input_features = [input_features]
    prediction = saved_model.predict(input_features)
    if (prediction[0] == 1):
        parkison_status = 'This person has a Parkison Disease'
    else:
        parkison_status = "This person is healthy and doesn't have the disease"
st.success(parkison_status)
