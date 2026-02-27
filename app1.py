import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Personal Expense Pattern Analyzer",
    page_icon="💰",
    layout="wide"
)

# ================= LOAD MODELS =================
clf = pickle.load(open("expense_classifier_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
reg_model = pickle.load(open("monthly_expense_regressor.pkl", "rb"))
iso = pickle.load(open("anomaly_detector.pkl", "rb"))

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size:40px;
    font-weight:700;
    color:#2c3e50;
}
.sub-title {
    font-size:18px;
    color:#555;
}
.card {
    background-color:#f9f9f9;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<p class="main-title">💰 Personal Expense Pattern Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">'
    'An intelligent Machine Learning system to classify expenses, '
    'predict monthly spending, and detect abnormal transactions.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# ================= LOAD DATA =================
df = pd.read_csv("expense_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.dayofweek
df['Is_Weekend'] = (df['Weekday'] >= 5).astype(int)

# ================= SIDEBAR =================
st.sidebar.title("📊 Navigation Panel")
menu = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Expense Category Prediction",
        "Monthly Expense Forecast",
        "Anomaly Detection",
        "Visual Analytics"
    ]
)

# ================= DASHBOARD =================
if menu == "Dashboard":
    st.subheader("📌 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Total Expense", f"₹ {df['Amount'].sum():,.0f}")
    col3.metric("Avg Transaction", f"₹ {df['Amount'].mean():.2f}")

    st.markdown("### 📄 Dataset Preview")
    st.dataframe(df.head(15), use_container_width=True)

# ================= CATEGORY PREDICTION =================
elif menu == "Expense Category Prediction":
    st.subheader("🔍 Predict Expense Category")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        amount = st.number_input("💵 Expense Amount", min_value=1.0)
        month = st.selectbox("📅 Month", list(range(1,13)))
        day = st.number_input("📆 Day", min_value=1, max_value=31)
        weekday = st.selectbox("🗓️ Weekday (0=Mon, 6=Sun)", list(range(7)))
        is_weekend = 1 if weekday >= 5 else 0
        payment = st.selectbox("💳 Payment Mode (Encoded)", [0, 1, 2])

        if st.button("🎯 Predict Category"):
            input_data = np.array(
                [[amount, month, day, weekday, is_weekend, payment]]
            )
            input_scaled = scaler.transform(input_data)
            prediction = clf.predict(input_scaled)

            st.success(f"✅ Predicted Category Code: **{prediction[0]}**")

        st.markdown('</div>', unsafe_allow_html=True)

# ================= MONTHLY FORECAST =================
elif menu == "Monthly Expense Forecast":
    st.subheader("📈 Monthly Expense Forecast")

    month_input = st.selectbox("Select Month", list(range(1,13)))
    prediction = reg_model.predict([[month_input]])

    st.metric(
        label=f"Predicted Expense for Month {month_input}",
        value=f"₹ {prediction[0]:,.2f}"
    )

# ================= ANOMALY DETECTION =================
elif menu == "Anomaly Detection":
    st.subheader("🚨 Abnormal Transaction Detection")

    df['Anomaly'] = iso.predict(df[['Amount']])

    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(
        x=df.index,
        y=df['Amount'],
        hue=df['Anomaly'],
        palette={1:"green", -1:"red"},
        ax=ax
    )
    ax.set_title("Anomaly Detection using Isolation Forest")
    ax.set_xlabel("Transaction Index")
    ax.set_ylabel("Amount")

    st.pyplot(fig)

# ================= VISUAL ANALYTICS =================
else:
    st.subheader("📊 Expense Analytics")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        df['Category'].value_counts().plot(
            kind='pie',
            autopct='%1.1f%%',
            ax=ax1
        )
        ax1.set_title("Expense Distribution by Category")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Payment_Mode', data=df, ax=ax2)
        ax2.set_title("Payment Mode Usage")
        st.pyplot(fig2)

    st.markdown("### 🔎 Feature Importance")
    importances = clf.feature_importances_
    features = ['Amount','Month','Day','Weekday','Is_Weekend','Payment_Encoded']

    fig3, ax3 = plt.subplots()
    ax3.barh(features, importances)
    ax3.set_title("Feature Importance - Expense Classification")
    st.pyplot(fig3)

# ================= FOOTER =================
st.divider()
st.markdown(
    "💡 **Developed using Machine Learning & Streamlit** | "
    "Expense Classification • Prediction • Anomaly Detection"
)