import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

# Streamlit app
st.title('Credit Card Fraud Detection')

# Upload CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the dataframe
    st.write("First few rows of the dataset:")
    st.write(df.head())

    # Display dataset shape
    st.write("Shape of the dataset:")
    st.write(df.shape)

    # Display dataset description
    st.write("Dataset description:")
    st.write(df.describe())

    # Visualize the data
    st.write("Class distribution:")
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=df, ax=ax)
    st.pyplot(fig)

    # Check the correlation between the columns
    st.write("Correlation matrix:")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)
    
    # Display value counts of the 'Class' column
    st.write("Class value counts:")
    st.write(df['Class'].value_counts())
    
    # Standardizing Features by removing the Mean and Scaling to Unit Variance
    df.iloc[:, 1:30] = StandardScaler().fit_transform(df.iloc[:, 1:30])
    data_matrix = df.values

    # We're excluding the Time Variable from the Dataset
    X = data_matrix[:, 1:30]
    y = data_matrix[:, 30]
    X = normalize(X, norm="l1")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Training the RandomForest Classifier
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)

    # Evaluation
    y_pred = RF.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    st.write("Classification Report:")
    st.write(pd.DataFrame(report).transpose())

    # Prediction input form
    st.write("### Make a Prediction")
    input_data = []
    for i in range(1, 30):
        input_val = st.number_input(f"V{i}", value=0.0)
        input_data.append(input_val)
    
    # Normalize and reshape the input data
    input_data = np.array(input_data).reshape(1, -1)
    input_data = normalize(input_data, norm="l1")
    
    if st.button("Predict"):
        prediction = RF.predict(input_data)
        prediction_proba = RF.predict_proba(input_data)
        st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
        st.write(f"Prediction Probability: {prediction_proba[0]}")
else:
    st.write("Please upload a CSV file to continue.")
