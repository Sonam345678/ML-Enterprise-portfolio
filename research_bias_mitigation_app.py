import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import streamlit as st

st.set_page_config(
    page_title="Research Methodology: Algorithmic Bias Mitigation",
    layout="wide",
)


# 1. Dataset Generation with Synthetic Gender Bias
@st.cache_data
def generate_biased_hr_data(n_samples=2000, seed=42):
    np.random.seed(seed)
    experience = np.random.randint(1, 15, size=n_samples)
    education_score = np.random.randint(1, 5, size=n_samples)
    interview_score = np.random.randint(40, 100, size=n_samples)

    # 0: Group A (Protected/Female), 1: Group B (Unprotected/Male)
    gender = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])

    # Merit score
    true_score = (
        (experience * 2.5) + (education_score * 5) + (interview_score * 0.4)
    )

    # Biased ground truth labels (historical hiring decision bias)
    biased_score = true_score + (gender * 12)  # Privilege bump for group 1
    hired = (biased_score > np.percentile(biased_score, 65)).astype(int)

    return pd.DataFrame({
        "Experience_Years": experience,
        "Education_Level": education_score,
        "Interview_Score": interview_score,
        "Gender_Group": gender,  # Protected Attribute
        "Hired": hired,
    })


# 2. Bias Calculation Metrics
def compute_fairness_metrics(df, pred_col, group_col):
    group_0 = df[df[group_col] == 0]
    group_1 = df[df[group_col] == 1]

    selection_rate_0 = group_0[pred_col].mean()
    selection_rate_1 = group_1[pred_col].mean()

    disparate_impact = (
        (selection_rate_0 / selection_rate_1) if selection_rate_1 > 0 else 0
    )

    return selection_rate_0, selection_rate_1, disparate_impact


# UI & Research Presentation
st.title("🔬 Research Methodology Showcase")
st.subheader(
    "Auditing and Mitigating Algorithmic Bias in HR Screening Systems"
)
st.markdown(
    "**M.Sc. Data Science Research Paper Project:** Evaluating Fair Reweighing Techniques to Eliminate Demographic Disparity in Machine Learning Models."
)

df = generate_biased_hr_data()

# Data Split
X = df[
    ["Experience_Years", "Education_Level", "Interview_Score", "Gender_Group"]
]
y = df["Hired"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Sidebar Intervention Controls
st.sidebar.header("🎛️ Research Parameters")
mitigation_active = st.sidebar.checkbox("Enable Fair Reweighing Mitigation", False)
threshold_adj = st.sidebar.slider(
    "Protected Group Decision Threshold", 0.3, 0.7, 0.5
)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)

if mitigation_active:
    # Pre-processing: Calculate Instance Weights to Counter Bias
    # Weight = P(Y) * P(A) / P(Y, A)
    weights = np.ones(len(y_train))
    for g in [0, 1]:
        for h in [0, 1]:
            mask = (X_train["Gender_Group"] == g) & (y_train == h)
            p_g = (X_train["Gender_Group"] == g).mean()
            p_h = (y_train == h).mean()
            p_gh = mask.mean()
            if p_gh > 0:
                weights[mask] = (p_g * p_h) / p_gh

    model.fit(X_train, y_train, sample_weight=weights)
else:
    model.fit(X_train, y_train)

# Predictions & Metrics
probs = model.predict_proba(X_test)[:, 1]

# Custom threshold for protected group if mitigation active
test_res = X_test.copy()
test_res["True_Hired"] = y_test
test_res["Prob"] = probs

if mitigation_active:
    test_res["Pred_Hired"] = np.where(
        test_res["Gender_Group"] == 0,
        (test_res["Prob"] >= threshold_adj).astype(int),
        (test_res["Prob"] >= 0.5).astype(int),
    )
else:
    test_res["Pred_Hired"] = (test_res["Prob"] >= 0.5).astype(int)

sr_0, sr_1, di_ratio = compute_fairness_metrics(
    test_res, "Pred_Hired", "Gender_Group"
)
acc = accuracy_score(y_test, test_res["Pred_Hired"])

# Executive Research Dashboard Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Model Accuracy", f"{acc:.2%}")
col2.metric("Group A Selection Rate", f"{sr_0:.2%}")
col3.metric("Group B Selection Rate", f"{sr_1:.2%}")
col4.metric(
    "Disparate Impact Ratio",
    f"{di_ratio:.3f}",
    delta="Fair (>= 0.80)" if di_ratio >= 0.8 else "Biased (< 0.80)",
)

st.markdown("---")

tab1, tab2 = st.tabs(
    ["📊 Fairness Audit & Disparity Visuals", "📜 Academic Methodology Summary"]
)

with tab1:
    st.subheader("Selection Rate & Bias Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(
        x=["Group A (Protected)", "Group B (Unprotected)"],
        y=[sr_0, sr_1],
        palette=["#e74c3c" if di_ratio < 0.8 else "#2ecc71", "#3498db"],
        ax=ax[0],
    )
    ax[0].set_ylabel("Selection Rate (Hired %)")
    ax[0].set_title("Hiring Rate Parity Comparison")
    ax[0].axhline(
        sr_1 * 0.8, color="red", linestyle="--", label="80% Fairness Threshold"
    )
    ax[0].legend()

    sns.kdeplot(
        data=test_res,
        x="Prob",
        hue="Gender_Group",
        common_norm=False,
        fill=True,
        ax=ax[1],
    )
    ax[1].set_title("Prediction Probability Density by Group")
    ax[1].set_xlabel("Model Predicted Hiring Probability")

    st.pyplot(fig)

    if di_ratio < 0.8:
        st.warning(
            "⚠️ **80% Rule Violation Detected:** Disparate Impact Ratio is below 0.80. The baseline model shows statistically significant bias against Group A."
        )
    else:
        st.success(
            "✅ **Algorithmic Fairness Achieved:** Disparate Impact Ratio meets academic fairness standards (≥ 0.80)."
        )

with tab2:
    st.markdown("""
    ### Research Paper Methodology Breakdown
    * **Objective:** Mitigate historical bias present in training labels without destroying overall accuracy.
    * **80% Rule Metric:** Disparate Impact Ratio $DI = \\frac{P(\\hat{Y}=1 | A=0)}{P(\\hat{Y}=1 | A=1)}$ must be $\\ge 0.80$.
    * **Technique:** Pre-processing Reweighing assigns inverse probability weights to over-represented/under-represented intersections of protected attributes and outcomes.
    """)
