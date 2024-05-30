import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
import joblib
import io

# Load the model and scaler
RF = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_fraud(transaction):
    """
    Predict if a given credit card transaction is fraudulent.
    
    Parameters:
    transaction (list or numpy array): A list or array containing the transaction features excluding 'Time' and 'Class'.
    
    Returns:
    int: 1 if the transaction is fraudulent, 0 otherwise.
    """
    # Ensure the input is a numpy array
    if not isinstance(transaction, np.ndarray):
        transaction = np.array(transaction)
    
    # Reshape the transaction data for scaling and normalization
    transaction = transaction.reshape(1, -1)
    
    # Standardize the transaction features
    transaction = scaler.transform(transaction)
    
    # Normalize the transaction features
    transaction = normalize(transaction, norm="l1")
    
    # Predict using the trained model
    prediction = RF.predict(transaction)
    
    return prediction[0]

st.title("Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())
    
    # Visualize the data
    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)
    
    # Check the correlation between the columns
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

    # Standardizing Features by removing the Mean and Scaling to Unit Variance
    df.iloc[:, 1:30] = scaler.transform(df.iloc[:, 1:30])
    
    # Input for prediction
    st.subheader("Predict Fraud for a New Transaction")
    input_data = []
    for i in range(1, 30):
        value = st.number_input(f"Feature {i}", value=0.0)
        input_data.append(value)
    
    if st.button("Predict"):
        prediction = predict_fraud(input_data)
        st.write("Fraudulent" if prediction == 1 else "Not Fraudulent")
