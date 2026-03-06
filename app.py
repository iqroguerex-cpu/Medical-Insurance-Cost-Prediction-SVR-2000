import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Medical Insurance Cost Prediction")

st.markdown("""
This application predicts **medical insurance charges** using a machine learning model.

The model analyzes several factors that affect insurance costs, including:

- Age
- Body Mass Index (BMI)
- Number of children
- Smoking status

Using these inputs, the system estimates the **expected insurance cost**.
""")

st.divider()

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("insurance_data_2000.csv")

# ---------------------------------------------------
# LOAD TRAINED MODEL + SCALERS
# ---------------------------------------------------

model = joblib.load("svr_model.pkl")
sc_X = joblib.load("scaler_X.pkl")
sc_y = joblib.load("scaler_y.pkl")

# ---------------------------------------------------
# MODEL PERFORMANCE (NO RETRAINING)
# ---------------------------------------------------

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1,1)

# Scale features
X_scaled = sc_X.transform(X)

# Predict
y_pred_scaled = model.predict(X_scaled).reshape(-1,1)

# Convert back to original scale
y_pred = sc_y.inverse_transform(y_pred_scaled)
y_actual = y

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)
rmse = mse ** 0.5

# ---------------------------------------------------
# SIDEBAR USER INPUT
# ---------------------------------------------------

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 65, 30)
bmi = st.sidebar.slider("BMI", 18.0, 45.0, 25.0)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.selectbox("Smoker", ["No","Yes"])

smoker_value = 1 if smoker == "Yes" else 0

input_data = [[age, bmi, children, smoker_value]]

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

scaled_input = sc_X.transform(input_data)

prediction_scaled = model.predict(scaled_input).reshape(-1,1)

prediction = sc_y.inverse_transform(prediction_scaled)

# ---------------------------------------------------
# DISPLAY PREDICTION
# ---------------------------------------------------

st.header("Predicted Insurance Cost")

st.metric(
    label="Estimated Medical Insurance Charge",
    value=f"${prediction[0][0]:,.2f}"
)

st.divider()

st.header("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:,.2f}")
col2.metric("MSE", f"{mse:,.2f}")
col3.metric("RMSE", f"{rmse:,.2f}")
col4.metric("R² Score", f"{r2:.3f}")

st.info("""
These metrics measure how accurate the machine learning model is.

• **MAE** → Average prediction error  
• **MSE** → Squared error used in training  
• **RMSE** → Error in the same unit as cost  
• **R² Score** → How well the model explains the data
""")

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.header("Dataset Overview")

st.dataframe(df)

st.write("Dataset Statistics")

st.write(df.describe())

st.divider()

# ---------------------------------------------------
# VISUALIZATIONS
# ---------------------------------------------------

st.header("Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["Age"], df["Charges"])
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Insurance Charges")
    ax1.set_title("Age vs Charges")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["BMI"], df["Charges"])
    ax2.set_xlabel("BMI")
    ax2.set_ylabel("Insurance Charges")
    ax2.set_title("BMI vs Charges")
    st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.hist(df["Charges"], bins=20)
ax3.set_title("Distribution of Insurance Charges")
st.pyplot(fig3)

st.divider()

# ---------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------

st.header("Feature Importance")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_scaled = sc_X.transform(X)

result = permutation_importance(
    model,
    X_scaled,
    y,
    n_repeats=10,
    random_state=42
)

features = ["Age","BMI","Children","Smoker"]

importance = pd.Series(result.importances_mean, index=features)

st.bar_chart(importance)

st.divider()

# ---------------------------------------------------
# HOW THE MODEL WORKS
# ---------------------------------------------------

st.header("How This Machine Learning Model Works")

st.markdown("""
The prediction system uses **Support Vector Regression (SVR)**.

The algorithm learns patterns from historical data and predicts insurance charges.

### Important Parameters

**C**  
Controls how strictly the model tries to avoid prediction errors.

**Gamma**  
Controls how much influence each data point has.

**Epsilon**  
Defines a margin where small prediction errors are ignored.
""")

st.divider()

# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

st.header("About This Project")

st.markdown("""
This project demonstrates a **complete machine learning workflow**:

1. Data preprocessing  
2. Feature scaling  
3. Hyperparameter tuning  
4. Model training  
5. Model deployment using Streamlit  

Technologies used:

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
""")

st.caption("Created by Chinmay V Chatradamath 🚀")
