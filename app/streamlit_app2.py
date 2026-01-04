import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Subscription Churn Intelligence",
    layout="wide"
)

# ----------------------------
# Load Data & Model
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/valid_with_predictions.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_churn.pkl")

df = load_data()
model = load_model()

# ----------------------------
# Helper Functions
# ----------------------------
def retention_simulation(df, target_pct, success_rate):
    top_users = df.sort_values(
        "churn_probability", ascending=False
    ).head(int(len(df) * target_pct))
    return top_users["revenue_at_risk"].sum() * success_rate


def monotonic_analysis(df, feature, bins=10):
    temp = df[[feature, "churn_probability"]].copy()
    temp["bin"] = pd.qcut(temp[feature], bins, duplicates="drop")
    return temp.groupby("bin")["churn_probability"].mean()


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "Customer Risk Explorer",
        "Feature Insights",
        "Retention Simulator"
    ]
)

# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    st.title("📊 Executive Churn Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Avg Churn Probability",
        f"{df['churn_probability'].mean():.2%}"
    )

    col2.metric(
        "Total Revenue at Risk",
        f"${df['revenue_at_risk'].sum():,.0f}"
    )

    top_10_risk = (
        df.sort_values("revenue_at_risk", ascending=False)
          .head(int(len(df) * 0.1))["revenue_at_risk"].sum()
    )

    col3.metric(
        "Top 10% Revenue Risk",
        f"${top_10_risk:,.0f}"
    )

    st.subheader("Revenue at Risk Distribution")

    fig, ax = plt.subplots()
    ax.hist(df["revenue_at_risk"], bins=40)
    ax.set_xlabel("Revenue at Risk")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

# ============================================================
# PAGE 2 — CUSTOMER RISK EXPLORER
# ============================================================
elif page == "Customer Risk Explorer":
    st.title("🔍 Customer Risk Explorer")

    customer_id = st.number_input(
        "Select Customer Index",
        min_value=int(df.index.min()),
        max_value=int(df.index.max()),
        value=int(df.index[0])
    )

    customer = df.loc[customer_id]

    st.metric(
        "Churn Probability",
        f"{customer['churn_probability']:.2%}"
    )

    st.metric(
        "Revenue at Risk",
        f"${customer['revenue_at_risk']:,.2f}"
    )

    st.subheader("Customer Snapshot")
    st.dataframe(
        customer.drop(["revenue_at_risk"]).to_frame("Value")
    )

# ============================================================
# PAGE 3 — FEATURE INSIGHTS (NO SHAP)
# ============================================================
elif page == "Feature Insights":
    st.title("📈 Feature Insights")

    st.subheader("Top Drivers (XGBoost Gain Importance)")

    importances = model.get_booster().get_score(
        importance_type="gain"
    )

    imp_df = (
        pd.DataFrame(importances.items(),
                     columns=["Feature", "Gain"])
          .sort_values("Gain", ascending=False)
          .head(15)
    )

    fig, ax = plt.subplots()
    ax.barh(
        imp_df["Feature"][::-1],
        imp_df["Gain"][::-1]
    )
    ax.set_xlabel("Gain")
    st.pyplot(fig)

    st.subheader("Feature vs Churn Trend")

    feature = st.selectbox(
        "Select feature",
        ["MonthlyCharges", "tenure", "avg_weekly_sessions"]
    )

    trend = monotonic_analysis(df, feature)

    fig, ax = plt.subplots()
    ax.plot(trend.values, marker="o")
    ax.set_ylabel("Avg Churn Probability")
    ax.set_xlabel("Feature Bin (Low → High)")
    st.pyplot(fig)

# ============================================================
# PAGE 4 — RETENTION SIMULATOR
# ============================================================
elif page == "Retention Simulator":
    st.title("🧮 Retention What-If Simulator")

    col1, col2 = st.columns(2)

    target_pct = col1.slider(
        "Target % of Highest-Risk Customers",
        0.05, 0.50, 0.10, 0.05
    )

    success_rate = col2.slider(
        "Retention Success Rate",
        0.10, 0.80, 0.30, 0.05
    )

    saved_revenue = retention_simulation(
        df, target_pct, success_rate
    )

    st.metric(
        "Estimated Revenue Saved",
        f"${saved_revenue:,.0f}"
    )

    st.caption(
        "Simulation assumes targeted intervention "
        "on highest churn-risk customers."
    )