import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

def generate_hr_data(n_samples=1000):
    experience = np.random.randint(0, 15, size=n_samples)
    education_level = np.random.choice([1, 2, 3], size=n_samples, p=[0.4, 0.4, 0.2])
    interview_score = np.random.randint(50, 100, size=n_samples)
    gender = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    score = (experience * 2) + (education_level * 10) + (interview_score * 0.5)
    biased_score = score + (gender * 15)
    hired = (biased_score > np.percentile(biased_score, 60)).astype(int)
    
    return pd.DataFrame({
        'Experience_Years': experience,
        'Education_Level': education_level,
        'Interview_Score': interview_score,
        'Gender': gender,
        'Hired': hired
    })

def calculate_fairness_metrics(y_true, y_pred, protected_attr):
    df_eval = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': protected_attr})
    rate_group_0 = df_eval[df_eval['group'] == 0]['y_pred'].mean()
    rate_group_1 = df_eval[df_eval['group'] == 1]['y_pred'].mean()
    disparate_impact = rate_group_0 / rate_group_1 if rate_group_1 > 0 else 0
    return disparate_impact, rate_group_0, rate_group_1

def compute_sample_weights(df, protected_col, target_col):
    n = len(df)
    weights = np.ones(n)
    for group in df[protected_col].unique():
        for outcome in df[target_col].unique():
            p_group = len(df[df[protected_col] == group]) / n
            p_outcome = len(df[df[target_col] == outcome]) / n
            expected_count = n * p_group * p_outcome
            actual_count = len(df[(df[protected_col] == group) & (df[target_col] == outcome)])
            weight = expected_count / actual_count if actual_count > 0 else 1.0
            mask = (df[protected_col] == group) & (df[target_col] == outcome)
            weights[mask] = weight
    return weights

def main():
    st.set_page_config(page_title="AI HR Bias Audit", layout="wide")
    st.title("Auditing and Mitigating Algorithmic Bias in HR Screening Systems")
    st.markdown("Research Methodology Showcase: Evaluating Fairness vs. Accuracy Trade-offs")

    df = generate_hr_data(1200)
    
    X = df[['Experience_Years', 'Education_Level', 'Interview_Score', 'Gender']]
    y = df['Hired']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Baseline Model
    clf_base = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)
    dir_base, rate0_base, rate1_base = calculate_fairness_metrics(y_test, y_pred_base, X_test['Gender'])
    
    # Debiased Model
    sample_weights = compute_sample_weights(pd.concat([X_train, y_train], axis=1), 'Gender', 'Hired')
    clf_debiased = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_debiased.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_debiased = clf_debiased.predict(X_test)
    acc_deb = accuracy_score(y_test, y_pred_debiased)
    dir_deb, rate0_deb, rate1_deb = calculate_fairness_metrics(y_test, y_pred_debiased, X_test['Gender'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔴 Baseline Model (Biased)")
        st.metric("Accuracy", f"{acc_base:.2%}")
        st.metric("Disparate Impact Ratio", f"{dir_base:.4f}")
        st.write(f"- Selection Rate (Group 0 / Female): **{rate0_base:.2%}**")
        st.write(f"- Selection Rate (Group 1 / Male): **{rate1_base:.2%}**")

    with col2:
        st.markdown("### 🟢 Debiased Model (Fair)")
        st.metric("Accuracy", f"{acc_deb:.2%}", delta=f"{(acc_deb - acc_base):.2%}")
        st.metric("Disparate Impact Ratio", f"{dir_deb:.4f}")
        st.write(f"- Selection Rate (Group 0 / Female): **{rate0_deb:.2%}**")
        st.write(f"- Selection Rate (Group 1 / Male): **{rate1_deb:.2%}**")

    st.subheader("Selection Rate Comparison")
    
    # Compact figure size with explicit width control in Streamlit
    fig, ax = plt.subplots(figsize=(6, 2.8))
    categories = ['Base Female', 'Base Male', 'Fair Female', 'Fair Male']
    rates = [rate0_base, rate1_base, rate0_deb, rate1_deb]
    
    sns.barplot(x=categories, y=rates, palette=['#ff9999', '#66b3ff', '#99ff99', '#339966'], ax=ax)
    ax.set_ylabel("Selection Rate", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='both', labelsize=8)
    ax.axhline(0.80, color='red', linestyle='--', label='80% Target')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    
    # Graph size restrict karne ke liye columns use kiye hain
    graph_col, _ = st.columns([2, 1])
    with graph_col:
        st.pyplot(fig, use_container_width=True)

if __name__ == "__main__":
    main()
