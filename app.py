import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Executive Retention Dashboard", layout="wide")

st.title("📊 Executive Customer Retention & Revenue Risk Dashboard")
st.markdown("Real-time predictive analytics tool for executive decision-making.")

@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']].dropna()
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

df = load_data()

total_customers = len(df)
churned_customers = len(df[df['survived'] == 0])
retention_rate = (df['survived'] == 1).mean() * 100
total_revenue = df['fare'].sum() * 100
at_risk_revenue = df[df['survived'] == 0]['fare'].sum() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Active Accounts", f"{total_customers:,}")
col2.metric("Customer Retention Rate", f"{retention_rate:.1f}%")
col3.metric("Total Contract Revenue", f"${total_revenue:,.2f}")
col4.metric("Revenue At Risk", f"${at_risk_revenue:,.2f}")

st.divider()

st.subheader("🔮 Account Retention Risk Predictor")

c1, c2, c3 = st.columns(3)
with c1:
    pclass = st.selectbox("Customer Tier (1 = VIP, 3 = Basic):", [1, 2, 3])
    sex = st.selectbox("Gender / Demographics:", ["Female", "Male"])
with c2:
    age = st.slider("Account Age / Tenure:", 1, 80, 30)
    fare = st.number_input("Monthly Revenue / Fee ($):", 10.0, 500.0, 50.0)
with c3:
    embarked = st.selectbox("Region Code:", ["Region S", "Region C", "Region Q"])

X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

sex_val = 1 if sex == "Male" else 0
emb_val = 0 if embarked == "Region S" else (1 if embarked == "Region C" else 2)

input_data = pd.DataFrame([[pclass, sex_val, age, fare, emb_val]], 
                          columns=['pclass', 'sex', 'age', 'fare', 'embarked'])

retention_prob = model.predict_proba(input_data)[0][1] * 100
churn_risk = 100 - retention_prob

st.write("---")
st.write(f"### Predicted Account Churn Risk: **{churn_risk:.1f}%**")

if churn_risk > 50:
    st.error("⚠️ **HIGH RISK ACCOUNT DETECTED**")
    st.markdown("**Executive Action Required:** Schedule priority account call and review service tiers.")
else:
    st.success("✅ **HEALTHY / LOW RISK ACCOUNT**")
    st.markdown("**Growth Strategy:** Prime target for account expansion and upsell.")
