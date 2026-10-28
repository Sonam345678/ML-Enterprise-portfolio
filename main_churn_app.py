import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Enterprise Churn & Revenue Risk Engine", layout="wide"
)


# Self-contained Data Generator
def create_churn_dataset(n_samples=1500, random_state=42):
    np.random.seed(random_state)
    account_ids = [f"ACC-{1000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 60, size=n_samples)
    monthly_recurring_revenue = np.round(
        np.random.uniform(500, 15000, size=n_samples), 2
    )
    contract_type = np.random.choice(
        ["Month-to-Month", "One-Year", "Two-Year"],
        size=n_samples,
        p=[0.5, 0.3, 0.2],
    )
    support_tickets_30d = np.random.poisson(lam=2.5, size=n_samples)
    login_frequency_drop = np.random.uniform(-0.8, 0.4, size=n_samples)
    nps_score = np.random.randint(1, 11, size=n_samples)
    feature_utilization_score = np.random.uniform(0.1, 1.0, size=n_samples)

    logit = (
        -1.2
        - (tenure_months * 0.04)
        + (support_tickets_30d * 0.45)
        - (login_frequency_drop * 1.8)
        - (nps_score * 0.25)
        - (feature_utilization_score * 2.1)
        + np.where(contract_type == "Month-to-Month", 0.8, -0.5)
    )

    churn_prob = 1 / (1 + np.exp(-logit))
    churned = (churn_prob > np.percentile(churn_prob, 75)).astype(int)

    return pd.DataFrame({
        "Account_ID": account_ids,
        "Tenure_Months": tenure_months,
        "MRR_USD": monthly_recurring_revenue,
        "Contract_Type": contract_type,
        "Support_Tickets_30D": support_tickets_30d,
        "Login_Drop_Ratio": login_frequency_drop,
        "NPS_Score": nps_score,
        "Feature_Utilization": feature_utilization_score,
        "Churn": churned,
    })


@st.cache_data
def load_data():
    return create_churn_dataset()


@st.cache_resource
def train_model(df):
    X = df[
        [
            "Tenure_Months",
            "MRR_USD",
            "Support_Tickets_30D",
            "Login_Drop_Ratio",
            "NPS_Score",
            "Feature_Utilization",
            "Contract_Type",
        ]
    ].copy()
    X = pd.get_dummies(X, columns=["Contract_Type"], drop_first=True)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
    }

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)

    return model, X_train, X_test, y_test, y_prob, metrics, shap_vals


st.title("📊 Enterprise Customer Churn & Revenue-at-Risk Engine")
st.markdown(
    "**Master's Data Science Showcase:** XGBoost Predictive Modeling, SHAP Explainable AI, and Financial Risk Optimization."
)

df = load_data()
model, X_train, X_test, y_test, y_prob, metrics, shap_vals = train_model(df)

col1, col2, col3, col4 = st.columns(4)
total_mrr = df["MRR_USD"].sum()
test_df = X_test.copy()
test_df["Churn_Prob"] = y_prob
test_df["Revenue_at_Risk"] = test_df["MRR_USD"] * test_df["Churn_Prob"]
total_risk = test_df["Revenue_at_Risk"].sum()

col1.metric("Model ROC-AUC Score", f"{metrics['ROC_AUC']:.2%}")
col2.metric("Total Tracked ARR", f"${total_mrr * 12:,.0f}")
col3.metric("Projected Annual Risk", f"${total_risk * 12:,.0f}")
col4.metric(
    "High Risk Accounts", f"{(test_df['Churn_Prob'] > 0.6).sum()} / {len(test_df)}"
)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Financial Risk Analysis",
        "🔍 Explainable AI (SHAP)",
        "🎛️ Executive What-If Simulator",
    ]
)

with tab1:
    st.subheader("Top High-Risk Enterprise Accounts")
    risk_summary = test_df.sort_values(
        by="Revenue_at_Risk", ascending=False
    ).head(10)

    formatted_summary = pd.DataFrame({
        "MRR ($)": risk_summary["MRR_USD"],
        "Tenure (Months)": risk_summary["Tenure_Months"],
        "NPS Score": risk_summary["NPS_Score"],
        "Churn Probability": risk_summary["Churn_Prob"].map("{:.2%}".format),
        "Monthly Risk ($)": risk_summary["Revenue_at_Risk"].map(
            "${:,.2f}".format
        ),
    })

    st.dataframe(formatted_summary, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 3))
    sns.scatterplot(
        data=test_df,
        x="MRR_USD",
        y="Churn_Prob",
        size="Revenue_at_Risk",
        hue="Churn_Prob",
        palette="Reds",
        sizes=(20, 200),
        ax=ax,
    )
    ax.set_title("Customer Risk Distribution: MRR vs Churn Probability")
    ax.set_ylabel("Churn Probability")
    ax.set_xlabel("Monthly Recurring Revenue ($)")
    st.pyplot(fig)

with tab2:
    st.subheader("Model Decision Interpretation via SHAP")
    st.write(
        "SHAP values measure feature contribution towards increasing or decreasing customer churn risk."
    )

    fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_vals, X_test, max_display=7, show=False)
    st.pyplot(fig_shap)

with tab3:
    st.subheader("Interactive Client Intervention Simulator")
    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        sim_mrr = st.number_input("Account MRR ($)", 1000, 20000, 8500)
        sim_tenure = st.slider("Account Tenure (Months)", 1, 60, 12)
        sim_tickets = st.slider("Support Tickets (30 Days)", 0, 10, 5)
        sim_nps = st.slider("NPS Score", 1, 10, 4)

    with col_sim2:
        sim_login_drop = st.slider("Login Frequency Drop (%)", -1.0, 0.5, -0.4)
        sim_feature_util = st.slider("Feature Utilization Score", 0.0, 1.0, 0.3)
        sim_contract = st.selectbox(
            "Contract Type", ["Month-to-Month", "One-Year", "Two-Year"]
        )

    contract_one_year = 1 if sim_contract == "One-Year" else 0
    contract_two_year = 1 if sim_contract == "Two-Year" else 0

    input_data = pd.DataFrame(
        [[
            sim_tenure,
            sim_mrr,
            sim_tickets,
            sim_login_drop,
            sim_nps,
            sim_feature_util,
            contract_one_year,
            contract_two_year,
        ]],
        columns=X_train.columns,
    )

    raw_prob = model.predict_proba(input_data)[0][1]

    st.markdown("### Simulation Result")
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Predicted Churn Probability", f"{raw_prob:.2%}")
    res_col2.metric("Monthly Risk Value", f"${(raw_prob * sim_mrr):,.2f}")

    if raw_prob > 0.5:
        st.error(
            "⚠️ **High Risk Account!** Recommended Action: Assign Dedicated Success Manager and Offer 1-Year Contract Lock-in."
        )
    else:
        st.success("✅ **Healthy Account.** Low Risk of immediate churn.")
