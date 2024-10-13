# Parkinson's Disease Prediction App 🧠 🔍

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)<br><br>

Parkinson's disease (PD) is a neurodegenerative disorder that affects movement control. This project leverages **machine learning** techniques to predict the likelihood of an individual having Parkinson's disease based on their medical features. The model is trained on a dataset containing medical records, and a **Streamlit app** provides a user-friendly interface to interact with the prediction model. 🧬✨

## Deployed Web Application 🌐

🔗 [Parkinson's Disease Prediction](https://parkinsons-prediction-app-the-amazing-atharva.streamlit.app/)

### About This App 🖥️

This application uses a **machine learning model** to predict whether an individual has Parkinson's disease based on various vocal feature inputs. 🎤🔊 The app allows users to interactively input medical data and receive a prediction.

### About This Repository 📂

This repository contains:
- A trained **machine learning model** for predicting Parkinson's disease.
- A **Streamlit app** that provides an interactive interface for predictions.

---

## Project Structure 🛠️

The project is organized into different components, including model training, data processing, and the Streamlit web app. Here’s the directory structure:


```
├── model_files/           # Folder containing model files and other relevant files
│   ├── parkinson_model.pkl  # Saved trained model in pickle format
│   ├── parkinson_model.sav  # Another format of the trained model
│   ├── pca.pkl             # Principal Component Analysis (PCA) model
│   ├── scaler.pkl         # StandardScaler model used during training
|
├── data/                  # Folder for dataset and related files
│   └── parkinsons.data     # Dataset used for model training
|
├── requirements.txt       # Python dependencies required to run the project
└── oldrequirements1.txt   # Old version of requirements file (if needed)
|
├── app8.py                # Streamlit app for interactive prediction
├── dv_cp_.py              # Helper functions and data preprocessing code
|
| etc...
```
## Machine Learning Models Trained & Evaluated 🧑‍💻
The following Machine Learning models were trained and evaluated: <br>
1️⃣ Logistic Regression <br>
2️⃣ Random Forest Classifier <br>
3️⃣ Decision Tree Classifier <br>
4️⃣ Support Vector Machine Classifier <br>
5️⃣ Naive Bayes Classifier <br>
6️⃣ K Nearest Neighbor Classifier <br>

## Parkinson's Disease Data Set Description 📊

<table>
  <tr>
    <th><b>Data Set Characteristics</b></th>
    <th><b>Multivariate</b></th>
  </tr>
  <tr>
    <td>Number of Instances</td> 
    <td>197</td>
  </tr>
  <tr>
    <td>Area</td>
    <td>Life</td>
  </tr>
  <tr>
    <td>Attribute Characteristics</td>
    <td>Real</td>
  </tr>
  <tr>
    <td>Number of Attributes</td>
    <td>23</td>
  </tr>
  <tr>
    <td>Date Donated</td>
    <td>2008-06-26</td>
  </tr>
  <tr>
    <td>Associated Task</td>
    <td>Classification</td>
  </tr>
  <tr>
    <td>Missing Values?</td>
    <td>N/A</td>
  </tr>
</table>

<h1>Attribute Information</h1>

## Medical Attribute Information 📋
<table>
  <tr>
    <th><b>Attribute</b></th>
    <th><b>Meaning</b></th>
  </tr>
  <tr>
    <td>name</td> 
    <td>ASCII subject name and recording number</td>
  </tr>
  <tr>
    <td>MDVP:Fo(Hz)</td> 
    <td>Average vocal fundamental frequency</td>
  </tr>
  <tr>
    <td>MDVP:Fhi(Hz)</td> 
    <td>Maximum vocal fundamental frequency</td>
  </tr>
  <tr>
    <td>MDVP:Flo(Hz)</td>
    <td>Minimum vocal fundamental frequency</td>
  </tr>
  <tr>
    <td>MDVP:Jitter(%)</td>
    <td>Measure of variation in fundamental frequency</td>
  <tr>
    <td>MDVP:Jitter(Abs)</td>
    <td>Measure of variation in fundamental frequency</td>
  <tr>
    <td>MDVP:RAP</td>
    <td>Measure of variation in fundamental frequency</td>
  <tr>
    <td>MDVP:PPQ</td>
    <td>Measure of variation in fundamental frequency</td>
  <tr>
    <td>Jitter:DDP</td> 
    <td>Measure of variation in fundamental frequency</td>
  </tr>
  <tr>
    <td>MDVP:Shimmer</td>
    <td>Measure of variation in amplitude</td>
  <tr>
    <td>MDVP:Shimmer(dB)</td>
    <td>Measure of variation in amplitude</td>
  <tr>
    <td>Shimmer:APQ3</td>
    <td>Measure of variation in amplitude</td>
  <tr>
    <td>Shimmer:APQ5</td>
    <td>Measure of variation in amplitude</td>
  <tr>
    <td>MDVP:APQ</td>
    <td>Measure of variation in amplitude</td>
  <tr>
    <td>Shimmer:DDA</td>
    <td>Measure of variation in amplitude</td>
  </tr>
  <tr>
    <td>NHR</td>
    <td>Measure of ratio of noise to tonal components in the voice</td>
  <tr>
    <td>HNR</td> 
    <td>Measure of ratio of noise to tonal components in the voice</td>
  </tr>
  <tr>
    <td>status(Target variable)</td> 
    <td>Health status of the subject (one) - Parkinson's, (zero) - healthy</td>
  </tr>
  <tr>
    <td>RPDE</td>
    <td>Non-linear dynamical complexity measure</td>
  <tr>
    <td>D2</td> 
    <td>Non-linear dynamical complexity measure</td>
  </tr>
  <tr>
    <td>DFA</td> 
    <td>Signal fractal scaling exponent</td>
  </tr>
  <tr>
    <td>spread1</td>
    <td>Non-linear measure of fundamental frequency variation</td>
  <tr>
    <td>spread2</td>
    <td>Non-linear measure of fundamental frequency variation</td>
  <tr>
    <td>PPE</td>
    <td>Non-linear measure of fundamental frequency variation</td>
  </tr>
</table>


## Installation 🛠️

### 1. Clone the repository

```bash
git clone https://github.com/your-username/parkinson-disease-prediction.git
cd parkinson-disease-prediction
```

### 2. Set up a virtual environment (optional but recommended)

#### Using `venv`:

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Model Training 🔬

The model has been trained using a dataset of medical features of individuals, which can be found in the file `parkinsons.data`. The training process involves the following steps:

1. **Preprocessing**: Data cleaning, handling missing values, and scaling features.
2. **Model Selection**: An appropriate machine learning model is trained on the data.
3. **PCA Transformation**: Principal Component Analysis (PCA) is applied to reduce dimensionality.
4. **Model Saving**: The final trained model and other relevant artifacts like scaler and PCA are saved as pickle files (`.pkl`, `.sav`).

You can use these pre-trained models to predict Parkinson's disease on new data by using the Streamlit app.

## Using the Streamlit App 🚀

To interact with the trained model and make predictions, we have built a simple Streamlit web app.

### Running the app

1. Navigate to the project directory.
2. Run the following command:

   ```bash
   streamlit run app8.py
   ```

3. A web browser will automatically open the app, or you can access it at `http://localhost:8501` in your browser.

### Features of the Streamlit App

- Input medical features related to the patient.
- Predict if the individual is likely to have Parkinson's disease.
- Visualize the prediction result with features importance.

## Files 📁

- `parkinson_model.pkl`: The trained machine learning model saved using `pickle`.
- `parkinson_model.sav`: An alternate format of the trained model.
- `pca.pkl`: Principal Component Analysis model for dimensionality reduction.
- `scaler.pkl`: Scaler model used to normalize input data before feeding it into the model.

## Contributing 🤝

We welcome contributions to improve the project! If you'd like to contribute, feel free to:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request.

## Acknowledgements 🙏

- Dataset: [Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Libraries used: `scikit-learn`, `streamlit`, `pandas`, and `matplotlib`.
- Special thanks to **Streamlit Cloud**

## Group Details 👩‍💻👨‍💻

This project was developed by a group of 4 students from **VIT Pune**, under the **CSAI-B** branch.

| **Roll Number** | **Official Name**              |
|-----------------|--------------------------------|
| 33              | Shrey Santosh Rupnavar         |
| 37              | Salitri Atharva Akhil          |
| 60              | Tanishq Sudhir Thuse           |
| 61              | Tripti Prakash Mirani          |


#### If you have any questions or suggestions, feel free to open an issue or reach out directly! 😄👋



