# 💰 Personal Expense Pattern Analyzer

A Machine Learning–based Streamlit web application that analyzes personal expense data by classifying transactions into categories, predicting monthly spending using regression models, and detecting abnormal or suspicious transactions through anomaly detection techniques.

---

## 📌 Project Overview

Managing personal finances effectively requires understanding spending behavior.  
This project uses Machine Learning algorithms to automatically analyze expense data and provide meaningful insights through an interactive web interface.

The system performs:
- Expense category classification
- Monthly expense prediction
- Anomaly detection for unusual transactions
- Data visualization for better understanding of spending patterns

---

## 🚀 Features

- 🔍 **Expense Category Classification** using Random Forest Classifier  
- 📈 **Monthly Expense Prediction** using Regression Models  
- 🚨 **Anomaly Detection** using Isolation Forest  
- 📊 **Interactive Visualizations** with charts and graphs  
- 🌐 **Streamlit Web Application** with a clean and user-friendly interface  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Web Framework:** Streamlit  
- **Machine Learning Models:**  
  - Random Forest Classifier  
  - Random Forest Regressor  
  - Isolation Forest  

---

## 📂 Project Structure
```
Personal-Expense-Pattern-Analyzer/
│
├── app.py                          # Streamlit web application
├── expense_data.csv                # Expense dataset
├── expense_classifier_model.pkl    # Trained classification model
├── monthly_expense_regressor.pkl   # Trained regression model
├── anomaly_detector.pkl            # Trained anomaly detection model
├── scaler.pkl                      # Feature scaling object
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```


---

## ⚙️ How to Run the Project

## 1️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 3️⃣ Open in Browser

After running the Streamlit command, the application will automatically open in your default web browser.  
If it does not open automatically, manually visit:

```
http://localhost:8501/
```
