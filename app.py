import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("Credit Card Fraud Detection")

@st.cache
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache
def preprocess_data(df):
    # Standardizing Features by removing the Mean and Scaling to Unit Variance
    df.iloc[:, 1:30] = StandardScaler().fit_transform(df.iloc[:, 1:30])
    data_matrix = df.values

    # Excluding the Time Variable from the Dataset
    X = data_matrix[:, 1:30]
    y = data_matrix[:, 30]

    X = normalize(X, norm="l1")
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

@st.cache
def train_model(X_train, y_train):
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    return RF

def main():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)

        st.write("### Data Preview")
        st.write(df.head())

        # Visualize the data
        st.write("### Class Distribution")
        sns.countplot(x='Class', data=df)
        st.pyplot()

        # Check the correlation between the columns
        st.write("### Correlation Matrix")
        corr = df.corr()
        sns.heatmap(corr, annot=True)
        st.pyplot()

        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_model(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = accuracy_score(y_test, y_pred)

        st.write("### Model Evaluation")
        st.write("#### Classification Report")
        st.json(report)
        st.write(f"#### Accuracy: {accuracy:.2f}")

        st.write("### Make a Prediction")
        input_data = {}
        for column in df.columns[1:30]:
            input_data[column] = st.number_input(f"Input {column}", value=0.0)

        input_df = pd.DataFrame([input_data])
        input_df = StandardScaler().fit_transform(input_df)
        input_df = normalize(input_df, norm="l1")

        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")

if __name__ == "__main__":
    main()
