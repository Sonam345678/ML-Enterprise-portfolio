import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Sales & Profit Analysis Executive Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling to match corporate UI theme
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 1. High-Precision Synthetic Dataset Generator
@st.cache_data
def generate_sales_data():
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    n_records = 4782

    order_dates = np.random.choice(dates, size=n_records)
    regions = ["North", "South", "East", "West"]
    categories = [
        "Electronics",
        "Furniture",
        "Clothing",
        "Office Supplies",
        "Others",
    ]
    salespersons = [
        "Alex Reed",
        "Sarah Cole",
        "Michael Chang",
        "Priya Sharma",
        "David Miller",
    ]

    assigned_region = np.random.choice(regions, size=n_records, p=[0.3, 0.25, 0.25, 0.2])
    assigned_category = np.random.choice(
        categories, size=n_records, p=[0.30, 0.24, 0.20, 0.15, 0.11]
    )
    assigned_salesperson = np.random.choice(salespersons, size=n_records)

    sales_amount = np.random.uniform(50, 1200, size=n_records)
    profit_margin_pct = np.random.uniform(0.12, 0.38, size=n_records)
    profit_amount = sales_amount * profit_margin_pct

    df = pd.DataFrame({
        "Order_Date": order_dates,
        "Year": pd.to_datetime(order_dates).year,
        "Month": pd.to_datetime(order_dates).strftime("%b"),
        "Month_Num": pd.to_datetime(order_dates).month,
        "Region": assigned_region,
        "Category": assigned_category,
        "Salesperson": assigned_salesperson,
        "Sales": sales_amount,
        "Profit": profit_amount,
    })
    return df.sort_values(by="Order_Date").reset_index(drop=True)


df = generate_sales_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 FILTERS")
selected_year = st.sidebar.selectbox(
    "Year", options=[2024], index=0
)

months_list = ["All"] + list(
    df.sort_values("Month_Num")["Month"].unique()
)
selected_month = st.sidebar.selectbox("Month", options=months_list, index=0)

regions_list = ["All"] + list(df["Region"].unique())
selected_region = st.sidebar.selectbox("Region", options=regions_list, index=0)

categories_list = ["All"] + list(df["Category"].unique())
selected_category = st.sidebar.selectbox(
    "Category", options=categories_list, index=0
)

salespersons_list = ["All"] + list(df["Salesperson"].unique())
selected_salesperson = st.sidebar.selectbox(
    "Sales Person", options=salespersons_list, index=0
)

# Filtering Logic
filtered_df = df.copy()
if selected_month != "All":
    filtered_df = filtered_df[filtered_df["Month"] == selected_month]
if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]
if selected_salesperson != "All":
    filtered_df = filtered_df[
        filtered_df["Salesperson"] == selected_salesperson
    ]

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align: center; color: #1b4332;'>📗 SALES & PROFIT ANALYSIS DASHBOARD</h1>",
    unsafe_allow_html=True,
)
st.markdown("<p style='text-align: center; color: #6c757d;'>Executive Reporting Period: Jan - Dec 2024</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- TOP METRICS CARDS ----------------
col1, col2, col3, col4 = st.columns(4)

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
profit_margin = (
    (total_profit / total_sales * 100) if total_sales > 0 else 0
)
total_orders = len(filtered_df)

col1.metric("TOTAL SALES", f"${total_sales:,.0f}", delta="18.6% vs Last Year")
col2.metric("TOTAL PROFIT", f"${total_profit:,.0f}", delta="22.4% vs Last Year")
col3.metric("PROFIT MARGIN", f"{profit_margin:.2f}%", delta="3.1% vs Last Year")
col4.metric("TOTAL ORDERS", f"{total_orders:,}", delta="15.2% vs Last Year")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- ROW 1 VISUALS ----------------
r1_col1, r1_col2 = st.columns([2, 1])

with r1_col1:
    st.subheader("SALES vs PROFIT (Monthly Trend)")
    monthly_trend = (
        filtered_df.groupby(["Month_Num", "Month"])[["Sales", "Profit"]]
        .sum()
        .reset_index()
        .sort_values("Month_Num")
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=monthly_trend["Month"],
            y=monthly_trend["Sales"],
            mode="lines+markers",
            name="Sales",
            line=dict(color="#2d6a4f", width=3),
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=monthly_trend["Month"],
            y=monthly_trend["Profit"],
            mode="lines+markers",
            name="Profit",
            line=dict(color="#1b4332", width=3, dash="dot"),
        )
    )
    fig_trend.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with r1_col2:
    st.subheader("PROFIT BY CATEGORY")
    cat_profit = (
        filtered_df.groupby("Category")["Profit"].sum().reset_index()
    )
    fig_donut = px.pie(
        cat_profit,
        values="Profit",
        names="Category",
        hole=0.5,
        color_discrete_sequence=[
            "#2d6a4f",
            "#40916c",
            "#52b788",
            "#74c69d",
            "#95d5b2",
        ],
    )
    fig_donut.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ---------------- ROW 2 VISUALS ----------------
r2_col1, r2_col2 = st.columns([1, 1])

with r2_col1:
    st.subheader("SALES BY CATEGORY")
    cat_sales = (
        filtered_df.groupby("Category")["Sales"]
        .sum()
        .reset_index()
        .sort_values(by="Sales", ascending=True)
    )
    fig_bar = px.bar(
        cat_sales,
        x="Sales",
        y="Category",
        orientation="h",
        color_discrete_sequence=["#2d6a4f"],
    )
    fig_bar.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with r2_col2:
    st.subheader("SALES & PROFIT BY REGION")
    region_summary = (
        filtered_df.groupby("Region")
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
        )
        .reset_index()
    )
    region_summary["Profit_Margin_%"] = (
        region_summary["Total_Profit"] / region_summary["Total_Sales"] * 100
    ).round(2)

    # Format table for display
    region_summary_disp = region_summary.copy()
    region_summary_disp["Total_Sales"] = region_summary_disp[
        "Total_Sales"
    ].apply(lambda x: f"${x:,.0f}")
    region_summary_disp["Total_Profit"] = region_summary_disp[
        "Total_Profit"
    ].apply(lambda x: f"${x:,.0f}")
    region_summary_disp["Profit_Margin_%"] = region_summary_disp[
        "Profit_Margin_%"
    ].apply(lambda x: f"{x:.2f}%")

    st.dataframe(region_summary_disp, use_container_width=True, hide_index=True)
