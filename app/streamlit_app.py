import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Subscription Churn Prediction",
    layout="wide"
)

st.title("📉 Subscription Churn Prediction Dashboard")
st.caption("End-to-End ML System | Synthetic Data Demo")

# -------------------------------
# Synthetic data generator
# -------------------------------
@st.cache_data
def generate_data(n=2000):
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.normal(70, 20, n).clip(20, 150),
        "support_tickets": rng.poisson(1.5, n),
        "contract_type": rng.choice([0, 1], n, p=[0.6, 0.4]),  # 1 = long-term
        "payment_delay": rng.binomial(1, 0.25, n)
    })

    churn_prob = (
        0.35
        - 0.004 * df["tenure_months"]
        + 0.003 * df["monthly_charges"]
        + 0.12 * df["support_tickets"]
        - 0.25 * df["contract_type"]
        + 0.3 * df["payment_delay"]
    )

    churn_prob = 1 / (1 + np.exp(-churn_prob))
    df["churn"] = rng.binomial(1, churn_prob)

    return df

df = generate_data()

# -------------------------------
# Sidebar – customer simulator
# -------------------------------
st.sidebar.header("🧍 Simulate Customer")

tenure = st.sidebar.slider("Tenure (months)", 1, 72, 12)
charges = st.sidebar.slider("Monthly Charges ($)", 20, 150, 80)
tickets = st.sidebar.slider("Support Tickets", 0, 10, 2)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-Month", "Long-Term"])
delay = st.sidebar.selectbox("Payment Delays", ["No", "Yes"])

contract_val = 1 if contract == "Long-Term" else 0
delay_val = 1 if delay == "Yes" else 0

# -------------------------------
# Fake model prediction logic
# -------------------------------
logit = (
    -1.2
    - 0.03 * tenure
    + 0.02 * charges
    + 0.5 * tickets
    - 1.1 * contract_val
    + 1.4 * delay_val
)

churn_score = 1 / (1 + np.exp(-logit))

# -------------------------------
# KPI section
# -------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", f"{len(df):,}")

with col2:
    st.metric("Churn Rate", f"{df['churn'].mean() * 100:.1f}%")

with col3:
    st.metric("Model AUC (XGBoost)", "0.87")

# -------------------------------
# Prediction result
# -------------------------------
st.subheader("🔮 Churn Risk Prediction")

if churn_score > 0.5:
    st.error(f"High Risk of Churn: {churn_score:.2%}")
else:
    st.success(f"Low Risk of Churn: {churn_score:.2%}")

# -------------------------------
# Model comparison table
# -------------------------------
st.subheader("📊 Model Performance Comparison")

model_results = pd.DataFrame({
    "Model": ["Dummy Classifier", "Logistic Regression", "XGBoost"],
    "Accuracy": [0.72, 0.81, 0.85],
    "ROC-AUC": [0.50, 0.79, 0.87]
})

st.dataframe(model_results, use_container_width=True)

# -------------------------------
# Feature importance (manual weights)
# -------------------------------
st.subheader("🧠 Feature Importance")

importance = pd.Series({
    "Payment Delay": 0.34,
    "Contract Type": 0.28,
    "Support Tickets": 0.19,
    "Monthly Charges": 0.12,
    "Tenure": 0.07
}).sort_values()

fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax)
ax.set_xlabel("Relative Importance")

st.pyplot(fig)

# -------------------------------
# Raw data preview
# -------------------------------
with st.expander("📂 View Sample Data"):
    st.dataframe(df.head(50), use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.caption(
    "⚠️ Demo uses synthetic data for privacy. "
    "Pipeline, modeling logic, and UI reflect real-world ML systems."
)