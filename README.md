# 💰 Medical Insurance Cost Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?logo=streamlit)](https://hswu28n9tcpj6ujldhbqnr.streamlit.app/)

---

# 📊 Project Overview

This project predicts **medical insurance costs** using a **machine learning regression model**.

The application is built using **Support Vector Regression (SVR)** and deployed as an **interactive Streamlit dashboard** where users can input personal information and instantly receive a predicted insurance cost.

The project demonstrates a **complete machine learning workflow**, including:

* Data preprocessing
* Feature scaling
* Hyperparameter tuning
* Model training
* Model evaluation
* Interactive visualization
* Streamlit deployment

---

# 🚀 Features

✔ Predict medical insurance cost instantly
✔ Interactive Streamlit dashboard
✔ Hyperparameter tuning using **GridSearchCV**
✔ Model performance metrics
✔ Multiple data visualizations
✔ Feature importance analysis
✔ User-friendly explanations for non-technical users

---

# 🧠 Machine Learning Model

The model used in this project is **Support Vector Regression (SVR)** with an **RBF kernel**.

SVR works by learning relationships between input features and the target variable to make predictions.

### Input Features

The model uses the following patient attributes:

* **Age**
* **BMI (Body Mass Index)**
* **Number of Children**
* **Smoking Status**

### Target Variable

* **Insurance Charges**

---

# ⚙️ Hyperparameter Tuning

The model parameters were optimized using **GridSearchCV**, which tests multiple parameter combinations and selects the best performing model.

### Parameters Tuned

**C**

Controls how strictly the model tries to avoid prediction errors.

* Small C → simpler model
* Large C → fits training data closely

**Gamma**

Controls the influence of individual training samples.

* Small gamma → smoother predictions
* Large gamma → more complex model

**Epsilon**

Defines a margin where small prediction errors are ignored.

---

# 📈 Model Evaluation Metrics

The model performance is evaluated using:

### Mean Absolute Error (MAE)

Average difference between predicted and actual values.

### Mean Squared Error (MSE)

Average squared prediction error.

### Root Mean Squared Error (RMSE)

Square root of MSE, easier to interpret.

### R² Score

Measures how well the model explains the data.

```
1 → Perfect predictions
0 → Model predicts average
Negative → Poor model
```

---

# 📊 Visualizations Included

The Streamlit dashboard includes several charts to help understand the dataset and model:

* Age vs Insurance Charges
* BMI vs Insurance Charges
* Distribution of Insurance Costs
* Actual vs Predicted comparison
* Feature Importance chart

---

# 🖥️ Streamlit Dashboard

Users can interactively enter:

* Age
* BMI
* Number of Children
* Smoking Status

The application then predicts the **estimated insurance cost instantly**.

---

# 📂 Project Structure

```
insurance-cost-prediction
│
├── app.py
├── train_model.py
├── insurance_data_2000.csv
├── svr_model.pkl
├── scaler_X.pkl
├── scaler_y.pkl
├── requirements.txt
└── README.md
```

---

# ▶️ Run the Project Locally

### Clone the repository

```
git clone https://github.com/yourusername/insurance-cost-prediction.git
cd insurance-cost-prediction
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run the Streamlit app

```
streamlit run app.py
```

---

# 🛠 Technologies Used

* Python
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

# 👨‍💻 Author

**Chinmay V Chatradamath**

---

⭐ If you found this project useful, consider giving the repository a **star**.
