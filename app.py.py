# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 2. PAGE CONFIG (FIRST STREAMLIT COMMAND)
# ===============================
st.set_page_config(
    page_title="Personal Expense Pattern Analyzer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")

# ===============================
# 3. LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("expense_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    return df

df = load_data()

# ===============================
# 4. LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    regressor = pickle.load(open("monthly_expense_regressor.pkl", "rb"))
    anomaly_model = pickle.load(open("anomaly_detector.pkl", "rb"))
    return regressor, anomaly_model

regressor, anomaly_model = load_models()

# ===============================
# 5. RULE-BASED CATEGORY FUNCTION (FINAL & CORRECT)
# ===============================
def predict_category(amount, payment, food_limit, shopping_limit):
    payment = payment.lower()

    if payment in ["fuel", "petrol", "diesel"]:
        return "Transport"
    elif payment in ["electricity", "rent", "wifi"]:
        return "Bills"
    elif amount >= shopping_limit:
        return "Shopping"
    elif amount <= food_limit:
        return "Food"
    else:
        return "Others"

# ===============================
# 6. SIDEBAR (THIS IS HOW SIDEBAR IS ADDED)
# ===============================
st.sidebar.title("💰 Expense Analyzer")
st.sidebar.write("Analyze & predict personal expenses")

# ---- Sidebar Navigation ----
menu = st.sidebar.radio(
    "📌 Navigation",
    [
        "Dashboard",
        "Expense Category Prediction",
        "Monthly Expense Prediction",
        "Anomaly Detection",
        "Visual Analytics"
    ]
)

# ---- Sidebar Rule Settings ----
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Category Rules")

food_limit = st.sidebar.slider(
    "🍔 Food max amount (₹)",
    min_value=50,
    max_value=300,
    value=100
)

shopping_limit = st.sidebar.slider(
    "🛍️ Shopping min amount (₹)",
    min_value=1000,
    max_value=5000,
    value=2000
)

# ===============================
# 7. DASHBOARD
# ===============================
if menu == "Dashboard":
    st.title("📊 Expense Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", len(df))
    col2.metric("Total Expense", f"₹ {df['Amount'].sum():,.0f}")
    col3.metric("Average Expense", f"₹ {df['Amount'].mean():.2f}")

    st.subheader("Recent Transactions")
    st.dataframe(df.tail(10), use_container_width=True)

# ===============================
# 8. EXPENSE CATEGORY PREDICTION
# ===============================
elif menu == "Expense Category Prediction":
    st.title("🔍 Expense Category Prediction")
    st.caption("Rule-based classification for better accuracy and explainability.")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "💵 Expense Amount (₹)",
            min_value=1.0,
            step=1.0
        )

    with col2:
        payment_mode = st.selectbox(
            "💳 Payment Mode",
            ["Cash", "UPI", "Card", "Fuel", "Electricity", "Rent", "Other"]
        )

    if st.button("✅ Predict Category"):
        category = predict_category(
            amount,
            payment_mode,
            food_limit,
            shopping_limit
        )

        st.success(f"Predicted Expense Category: **{category}**")

        st.info(
            f"""
            🔎 **Rules Used**
            - Food ≤ ₹{food_limit}
            - Shopping ≥ ₹{shopping_limit}
            - Fuel → Transport
            - Electricity / Rent → Bills
            """
        )

# ===============================
# 9. MONTHLY EXPENSE PREDICTION
# ===============================
elif menu == "Monthly Expense Prediction":
    st.title("📈 Monthly Expense Prediction")
    st.caption("Machine Learning based expense forecasting.")

    month = st.selectbox("Select Month", list(range(1, 13)))
    predicted_amount = regressor.predict([[month]])[0]

    st.metric(
        label=f"Predicted Expense for Month {month}",
        value=f"₹ {predicted_amount:,.2f}"
    )

# ===============================
# 10. ANOMALY DETECTION
# ===============================
elif menu == "Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.caption("Red points indicate unusual expenses.")

    df["Anomaly"] = anomaly_model.predict(df[["Amount"]])

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x=df.index,
        y=df["Amount"],
        hue=df["Anomaly"],
        palette={1: "green", -1: "red"},
        ax=ax
    )

    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Unusual Expense Detection")
    ax.legend(labels=["Normal", "Anomaly"])

    st.pyplot(fig)

# ===============================
# 11. VISUAL ANALYTICS
# ===============================
else:
    st.title("📊 Visual Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        df["Category"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=ax1
        )
        ax1.set_title("Expense Distribution by Category")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Payment_Mode", data=df, ax=ax2)
        ax2.set_title("Payment Mode Usage")
        st.pyplot(fig2)

# ===============================
# 12. FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "👩‍💻 **Developed by Nidhi Ghatole**  \n"
    "📘 Personal Expense Pattern Analyzer | Hybrid ML Project"
)