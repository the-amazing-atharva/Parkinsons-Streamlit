import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# Load the saved model and preprocessing objects


@st.cache_resource
def load_model():
    model = pickle.load(open('parkinson_model.pkl', 'rb'))
    pca = pickle.load(open('pca.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, pca, scaler


model, pca, scaler = load_model()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Home", "Parkinson Prediction", "FAQs"])

# Home Page
if page == "Home":
    st.title("Welcome to Parkinson's Disease Prediction App")

    # Load and display an image
    image = Image.open('parkinson2.jpg')
    st.image(image, caption="Parkinson's Disease Awareness",
             use_column_width=True)

    # Add a detailed paragraph description
    st.write("""
    Parkinson's disease is a progressive movement disorder of the nervous system. It causes nerve cells (neurons) in parts of the brain to weaken, become damaged, and die, leading to symptoms that include problems with movement, tremor, stiffness, and impaired balance. As symptoms progress, people with Parkinson’s disease (PD) may have difficulty walking, talking, or completing other simple tasks. 
             
    This web application utilizes a comprehensive dataset to predict the likelihood of Parkinson's Disease (PD) based on various vocal measurements. Parkinson's Disease is a significant health concern globally, with approximately 1 million people affected in the United States alone.
             
            In India, the scenario is equally alarming, as studies indicate that the prevalence of Parkinson's is rising, with estimates suggesting that around 1.5 million people may be living with the disease in the country. Early diagnosis is crucial for improving treatment outcomes, as timely intervention can help manage symptoms more effectively.
    """)

    # Key Statistics in an expander
    with st.expander("Key Statistics"):
        st.write("""
        - In India, about 60,000 new cases of Parkinson's Disease are diagnosed each year.
        - The disease manifests through symptoms such as tremors, stiffness, and balance difficulties.
        - Research indicates that awareness and education about PD are vital for early detection and management.
        """)

    # Video Introduction
    st.subheader("For More Info on Parkinson's")
    st.video("https://www.youtube.com/watch?v=u_tozEV7f4k")

    # Dataset link
    st.markdown("### Check Out the Official Dataset")
    st.markdown(
        "[Oxford's Repository](https://archive.ics.uci.edu/dataset/174/parkinsons)")

# Parkinson Prediction Page
elif page == "Parkinson Prediction":
    st.title("Parkinson's Disease Prediction")

    st.write("""
    Enter the vocal measurements below to predict the likelihood of Parkinson's disease.
    Please ensure all values are entered accurately for reliable predictions.
    """)

    # Create tabs for input methods
    input_method = st.radio("Choose input method:", [
                            "Manual Input", "CSV Upload"])

    if input_method == "Manual Input":
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
            with columns[i % 4]:
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
            probability = model.predict_proba(pca_data)[0][1]

            # Display prediction results
            st.subheader("Prediction Result")
            if prediction[0] == 0:
                st.success(
                    f'The person does not have Parkinson\'s disease ')
            else:
                st.error(
                    f'The person has Parkinson\'s disease ')

            # Visualize feature importance
            feature_importance = abs(pca.components_[0])
            fig = go.Figure(data=[go.Bar(x=features, y=feature_importance)])
            fig.update_layout(title="Feature Importance",
                              xaxis_title="Features", yaxis_title="Importance")
            st.plotly_chart(fig)

    else:  # CSV Upload
        st.write("Upload a CSV file with the required features.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.write(input_df)

            if st.button('Predict'):
                # Apply MinMaxScaler
                scaled_data = scaler.transform(input_df)

                # Apply PCA transformation
                pca_data = pca.transform(scaled_data)

                # Make prediction
                predictions = model.predict(pca_data)
                probabilities = model.predict_proba(pca_data)[:, 1]

                # Add predictions to the dataframe
                input_df['Prediction'] = predictions
                input_df['Probability'] = probabilities

                st.subheader("Prediction Results")
                st.write(input_df)

                # Download results
                csv = input_df.to_csv(index=False)
                st.download_button(
                    label="Download results as CSV",
                    data=csv,
                    file_name="parkinson_predictions.csv",
                    mime="text/csv",
                )

# # Statistics Page
# elif page == "Statistics":
#     st.title("Parkinson's Disease Statistics")

#     # Sample data (replace with real data)
#     years = [2015, 2016, 2017, 2018, 2019, 2020]
#     cases = [1200000, 1250000, 1300000, 1350000, 1400000, 1450000]

#     # Line chart
#     fig_line = go.Figure(data=go.Scatter(
#         x=years, y=cases, mode='lines+markers'))
#     fig_line.update_layout(title='Parkinson\'s Disease Cases in India Over Time',
#                            xaxis_title='Year',
#                            yaxis_title='Number of Cases')
#     st.plotly_chart(fig_line)

#     # Age distribution (sample data)
#     age_groups = ['30-40', '41-50', '51-60', '61-70', '71+']
#     distribution = [5, 15, 30, 35, 15]

#     # Pie chart
#     fig_pie = go.Figure(data=[go.Pie(labels=age_groups, values=distribution)])
#     fig_pie.update_layout(
#         title='Age Distribution of Parkinson\'s Disease Patients')
#     st.plotly_chart(fig_pie)


# FAQs Page
elif page == "FAQs":
    st.title("Frequently Asked Questions")

    faqs = [
        ("What is Parkinson's Disease?",
         "Parkinson's Disease is a neurodegenerative disorder that primarily affects movement. It can lead to tremors, stiffness, and difficulty with balance and coordination."),
        ("How can I use this app?",
         "Navigate to the prediction page using the sidebar. Follow the on-screen instructions to input your vocal measurements and receive predictions."),
        ("What are the symptoms of Parkinson's Disease?",
         "Common symptoms include: tremors or shaking, stiffness in limbs, slowness of movement, balance problems, and changes in speech or writing."),
        ("Is there a cure for Parkinson's Disease?",
         "Currently, there is no cure for Parkinson's Disease, but treatments are available to manage symptoms. Early diagnosis and intervention can significantly improve quality of life."),
        ("Where can I find more resources?",
         "For more information, consider visiting: - [Parkinson's Foundation](https://www.parkinson.org), - [Mayo Clinic](https://www.mayoclinic.org/diseases-conditions/parkinsons-disease),  - [National Institute of Neurological Disorders and Stroke](https://www.ninds.nih.gov/health-information/patient-caregiver-education/patient-resources)")
    ]

    for question, answer in faqs:
        with st.expander(question):
            st.write(answer)

         # Additional statistics
    st.subheader("Key Facts")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estimated cases in India (2020)", "1.5 million")
        st.metric("Annual new cases", "60,000+")
    with col2:
        st.metric("Male to Female ratio", "2.66 : 1")
        st.metric("Average age of onset", "57.73 years")

# Add a footer
st.markdown("---")
st.markdown("© 2024 Parkinson's Disease Prediction App. All rights reserved.")
