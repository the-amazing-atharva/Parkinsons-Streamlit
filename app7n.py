from PIL import Image
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Parkinson Prediction", "FAQs"])

# Home Page
if page == "Home":
    st.title("Welcome to Parkinson's Disease Prediction App")

    # Load and display an image
    image = Image.open('parkinson2.jpg')
    st.image(image, caption="Parkinson's Disease Awareness",
             use_column_width=True)

    # Add a detailed paragraph description
    st.write("""
This web application utilizes a comprehensive dataset to predict the likelihood of Parkinson's Disease (PD) based on various vocal measurements. Parkinson's Disease is a significant health concern globally, with approximately 1 million people affected in the United States alone. 
             
        In India, the scenario is equally alarming, as studies indicate that the prevalence of Parkinson's is rising, with estimates suggesting that around 1.5 million people may be living with the disease in the country. Early diagnosis is crucial for improving treatment outcomes, as timely intervention can help manage symptoms more effectively.

### Key Statistics:
- In India, about 60,000 new cases of Parkinson's Disease are diagnosed each year.
- The disease manifests through symptoms such as tremors, stiffness, and balance difficulties.
- Research indicates that awareness and education about PD are vital for early detection and management.

Navigate to the prediction page using the sidebar for more information.

### Check Out the Dataset
To access the official dataset on Parkinson's disease, visit [Oxford's Repository](https://archive.ics.uci.edu/dataset/174/parkinsons).
""")

    # Video Introduction
    st.subheader("For More Info on Parkinsons")
    st.video("https://www.youtube.com/watch?v=u_tozEV7f4k")


# FAQs Page
if page == "FAQs":
    st.subheader("Frequently Asked Questions")

    # Create a collapsible FAQ section using expander
    with st.expander("What is Parkinson's Disease?", expanded=False):
        st.write("""
        Parkinson's Disease is a neurodegenerative disorder that primarily affects movement. 
        It can lead to tremors, stiffness, and difficulty with balance and coordination.
        """)

    with st.expander("How can I use this app?", expanded=False):
        st.write("""
        You can easily navigate to the prediction page using the sidebar. 
        Follow the on-screen instructions to input your vocal measurements and receive predictions.
        """)

    with st.expander("What are the symptoms of Parkinson's Disease?", expanded=False):
        st.write("""
        Common symptoms include:
        - Tremors or shaking
        - Stiffness in limbs
        - Slowness of movement
        - Balance problems
        - Changes in speech or writing
        """)

    with st.expander("Is there a cure for Parkinson's Disease?", expanded=False):
        st.write("""
        Currently, there is no cure for Parkinson's Disease, but treatments are available to manage symptoms. 
        Early diagnosis and intervention can significantly improve quality of life.
        """)

    with st.expander("Where can I find more resources?", expanded=False):
        st.write("""
        For more information, consider visiting:
        - [Parkinson's Foundation](https://www.parkinson.org)
        - [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease)
        - [National Institute of Neurological Disorders and Stroke](https://www.ninds.nih.gov/health-information/patient-caregiver-education/patient-resources)
        """)


# Parkinson Prediction Page
if page == "Parkinson Prediction":
    st.title("Parkinson's Disease Prediction App")

    # Load the saved model and preprocessing objects
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    st.subheader("Enter the following features to predict Parkinson's disease")

    # Create 4 columns for input fields
    columns = st.columns(4)

    # Create a list of features
    features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
                'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                'NHR', 'HNR', 'RPDE', 'DFA',
                'spread1', 'spread2', 'D2', 'PPE']

    input_data = []

    # Distribute input fields across 4 columns
    for i, feature in enumerate(features):
        with columns[i % 4]:  # Distribute features across columns
            input_value = st.number_input(
                feature, value=0.0, format="%.5f", step=0.00001)
            input_data.append(input_value)

    # Create a button to trigger prediction
    if st.button('Predict'):
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

    # Add additional information or instructions at the bottom
    st.markdown("---")
    st.markdown("### About This App")
    st.markdown("""
    This application uses a machine learning model to predict whether an individual has Parkinson's disease based on various vocal feature inputs. 
    Please ensure that all values are entered accurately for reliable predictions.
    """)
