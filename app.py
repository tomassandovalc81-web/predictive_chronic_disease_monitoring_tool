import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import plotly.express as px

# Set page config
st.set_page_config(page_title="Chronic Disease Risk Predictor", layout="wide")

# Title and description
st.title("🏥 Predictive Chronic Disease Monitoring Tool")
st.markdown("Predict chronic disease risk based on patient health metrics using Machine Learning")

# Sidebar for navigation
page = st.sidebar.radio("Navigation", ["Home", "Prediction", "Model Evaluation", "Data Explorer"])

# Load and prepare data
@st.cache_resource
def load_data():
    data = pd.DataFrame({
        'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 52, 29, 55, 30, 65, 28, 58, 42, 38, 51, 47],
        'BMI': [22, 30, 28, 35, 20, 31, 33, 29, 27, 34, 24, 36, 25, 32, 23, 37, 26, 29, 30, 31],
        'Glucose': [90, 150, 130, 180, 85, 160, 170, 140, 120, 175, 95, 185, 100, 190, 92, 188, 145, 125, 165, 155],
        'Exercise_Hours': [7, 3, 5, 2, 8, 4, 1, 3, 6, 2, 7, 1, 6, 0, 8, 1, 4, 5, 3, 2],
        'Family_History': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        'Blood_Pressure': [120, 140, 130, 150, 115, 145, 160, 135, 125, 155, 118, 165, 122, 170, 116, 175, 148, 128, 158, 152],
        'Cholesterol': [180, 220, 200, 240, 175, 230, 250, 210, 190, 245, 185, 255, 195, 260, 182, 258, 215, 198, 235, 225],
        'Risk': [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    })
    return data

@st.cache_resource
def train_models(X_train, y_train, X_test, y_test):
    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    # Decision Tree with GridSearch
    dt_model = DecisionTreeClassifier(random_state=42)
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_leaf': [1, 5, 10],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return lr_model, grid_search.best_estimator_

# Load data
data = load_data()
X = data[['Age', 'BMI', 'Glucose', 'Exercise_Hours', 'Family_History', 'Blood_Pressure', 'Cholesterol']]
y = data['Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train models
lr_model, dt_model = train_models(X_train, y_train, X_test, y_test)

if page == "Home":
    st.markdown("""
    ### About This Tool
    
    This application uses Machine Learning to predict the risk of chronic disease based on health metrics. 
    Two models are trained on patient data:
    
    **Models Used:**
    - 🔹 **Logistic Regression**: A statistical model for binary classification
    - 🌳 **Decision Tree**: A tree-based model that makes interpretable predictions
    
    **Features Analyzed:**
    - Age, BMI, Glucose levels, Exercise hours
    - Family history, Blood pressure, Cholesterol
    
    ### How to Use
    1. Go to the **Prediction** page to enter patient data
    2. View **Model Evaluation** to see how well each model performs
    3. Explore the **Data Explorer** to understand the dataset
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    with col3:
        st.metric("Total Features", X.shape[1])

elif page == "Prediction":
    st.header("Patient Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age (years)", 18, 85, 50)
        bmi = st.slider("BMI", 15, 45, 25)
        glucose = st.slider("Glucose (mg/dL)", 70, 200, 100)
        
    with col2:
        exercise_hours = st.slider("Exercise Hours (per week)", 0, 10, 5)
        family_history = st.selectbox("Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        blood_pressure = st.slider("Blood Pressure (systolic)", 100, 180, 120)
    
    cholesterol = st.slider("Cholesterol (mg/dL)", 150, 300, 200)
    
    if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
        patient_data = [[age, bmi, glucose, exercise_hours, family_history, blood_pressure, cholesterol]]
        
        # Logistic Regression Prediction
        lr_prediction = lr_model.predict(patient_data)[0]
        lr_probability = lr_model.predict_proba(patient_data)[0][1]
        
        # Decision Tree Prediction
        dt_prediction = dt_model.predict(patient_data)[0]
        dt_probability = dt_model.predict_proba(patient_data)[0][1]
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Logistic Regression")
            if lr_prediction == 1:
                st.error(f"⚠️ Higher Risk Predicted")
            else:
                st.success(f"✅ Lower Risk Predicted")
            st.metric("Risk Probability", f"{lr_probability:.1%}")
        
        with col2:
            st.subheader("Decision Tree")
            if dt_prediction == 1:
                st.error(f"⚠️ Higher Risk Predicted")
            else:
                st.success(f"✅ Lower Risk Predicted")
            st.metric("Risk Probability", f"{dt_probability:.1%}")
        
        st.divider()
        st.info("""
        **Disclaimer**: These predictions are based on machine learning models trained on sample data. 
        Always consult with healthcare professionals for medical advice.
        """)

elif page == "Model Evaluation":
    st.header("Model Performance Metrics")
    
    # Decision Tree Evaluation
    dt_predictions = dt_model.predict(X_test)
    dt_probability = dt_model.predict_proba(X_test)[:, 1]
    
    dt_accuracy = accuracy_score(y_test, dt_predictions)
    dt_precision = precision_score(y_test, dt_predictions)
    dt_recall = recall_score(y_test, dt_predictions)
    dt_f1 = f1_score(y_test, dt_predictions)
    dt_roc_auc = roc_auc_score(y_test, dt_probability)
    
    # Logistic Regression Evaluation
    lr_predictions = lr_model.predict(X_test)
    lr_probability = lr_model.predict_proba(X_test)[:, 1]
    
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_precision = precision_score(y_test, lr_predictions)
    lr_recall = recall_score(y_test, lr_predictions)
    lr_f1 = f1_score(y_test, lr_predictions)
    lr_roc_auc = roc_auc_score(y_test, lr_probability)
    
    # Metrics Comparison
    st.subheader("Performance Metrics Comparison")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc],
        'Decision Tree': [dt_accuracy, dt_precision, dt_recall, dt_f1, dt_roc_auc]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Tree - Confusion Matrix")
        cm_dt = confusion_matrix(y_test, dt_predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm_dt, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_dt[i, j], ha='center', va='center', color='white', fontsize=16, weight='bold')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Logistic Regression - Confusion Matrix")
        cm_lr = confusion_matrix(y_test, lr_predictions)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm_lr, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm_lr[i, j], ha='center', va='center', color='white', fontsize=16, weight='bold')
        st.pyplot(fig)
    
    st.subheader("Decision Tree - Feature Importance")
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#1f77b4')
    ax.set_xlabel('Importance Score')
    st.pyplot(fig)

elif page == "Data Explorer":
    st.header("Dataset Exploration")
    
    st.subheader("Dataset Overview")
    st.dataframe(data, use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)
    
    st.subheader("Risk Distribution")
    risk_counts = data['Risk'].value_counts()
    fig = px.pie(values=risk_counts.values, names=['Lower Risk', 'Higher Risk'], title="Distribution of Risk Levels")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Glucose vs BMI (Colored by Risk)")
    fig = px.scatter(data, x='Glucose', y='BMI', color='Risk', 
                     color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                     labels={'Risk': 'Risk Level'},
                     title="Glucose vs BMI")
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(data, x='Age', nbins=10, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("BMI Distribution")
        fig = px.histogram(data, x='BMI', nbins=10, title="BMI Distribution")
        st.plotly_chart(fig, use_container_width=True)