import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Enterprise Retail & Supply Chain Analytics Engine",
    layout="wide"
)

# 1. Advanced Synthetic Enterprise Dataset
@st.cache_data
def load_enterprise_data(n_orders=3000, seed=42):
    np.random.seed(seed)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=int(x)) for x in np.random.randint(0, 365, size=n_orders)]
    
    customer_ids = [f"CUST-{1000 + i}" for i in np.random.randint(1, 250, size=n_orders)]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty & Personal Care"]
    channels = ["Direct Web", "Mobile App", "Amazon Storefront", "Retail Partner"]
    
    order_ids = [f"ORD-{50000 + i}" for i in range(n_orders)]
    assigned_regions = np.random.choice(regions, size=n_orders, p=[0.4, 0.3, 0.2, 0.1])
    assigned_categories = np.random.choice(categories, size=n_orders, p=[0.35, 0.25, 0.25, 0.15])
    assigned_channels = np.random.choice(channels, size=n_orders)
    
    units_sold = np.random.randint(1, 10, size=n_orders)
    unit_price = np.random.uniform(20, 800, size=n_orders)
    discount_pct = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n_orders, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    
    gross_revenue = units_sold * unit_price
    net_revenue = gross_revenue * (1 - discount_pct)
    cogs = net_revenue * np.random.uniform(0.45, 0.65, size=n_orders)
    profit = net_revenue - cogs
    
    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Date": dates,
        "Customer_ID": customer_ids,
        "Region": assigned_regions,
        "Category": assigned_categories,
        "Channel": assigned_channels,
        "Units_Sold": units_sold,
        "Unit_Price": np.round(unit_price, 2),
        "Discount": discount_pct,
        "Gross_Revenue": np.round(gross_revenue, 2),
        "Net_Revenue": np.round(net_revenue, 2),
        "Profit": np.round(profit, 2)
    })
    return df.sort_values(by="Order_Date").reset_index(drop=True)

df = load_enterprise_data()

# Executive Dashboard Layout
st.title("📈 Enterprise Retail & Supply Chain Analytics Suite")
st.markdown("**Executive BI Platform:** End-to-End Sales Performance, RFM Segmentation, Cohort Retention & Inventory Risk Forecasting.")

# Sidebar Filters
st.sidebar.header("🎯 Executive Data Filters")
selected_regions = st.sidebar.multiselect("Select Region", options=df["Region"].unique(), default=df["Region"].unique())
selected_categories = st.sidebar.multiselect("Select Category", options=df["Category"].unique(), default=df["Category"].unique())

filtered_df = df[(df["Region"].isin(selected_regions)) & (df["Category"].isin(selected_categories))]

# Top Financial KPIs
col1, col2, col3, col4, col5 = st.columns(5)
total_net_rev = filtered_df["Net_Revenue"].sum()
total_profit = filtered_df["Profit"].sum()
profit_margin = (total_profit / total_net_rev * 100) if total_net_rev > 0 else 0
avg_order_val = filtered_df["Net_Revenue"].mean()
total_units = filtered_df["Units_Sold"].sum()

col1.metric("Net Revenue", f"${total_net_rev:,.2f}")
col2.metric("Total Operating Profit", f"${total_profit:,.2f}")
col3.metric("Profit Margin", f"{profit_margin:.1f}%")
col4.metric("Avg Order Value (AOV)", f"${avg_order_val:,.2f}")
col5.metric("Total Units Sold", f"{total_units:,}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Financial Performance", 
    "🎯 RFM Customer Segmentation", 
    "📈 Demand & Inventory Forecast",
    "🎛️ Executive Profit Simulator"
])

with tab1:
    st.subheader("Regional & Multi-Channel Revenue Breakdown")
    c1, c2 = st.columns(2)
    
    with c1:
        fig_rev, ax_rev = plt.subplots(figsize=(6, 3.5))
        region_perf = filtered_df.groupby("Region")["Net_Revenue"].sum().reset_index()
        sns.barplot(data=region_perf, x="Region", y="Net_Revenue", palette="Blues_r", ax=ax_rev)
        ax_rev.set_title("Revenue Contribution by Market Region")
        ax_rev.set_ylabel("Net Revenue ($)")
        st.pyplot(fig_rev)
        
    with c2:
        fig_chan, ax_chan = plt.subplots(figsize=(6, 3.5))
        channel_perf = filtered_df.groupby("Channel")["Profit"].sum().reset_index()
        sns.barplot(data=channel_perf, x="Channel", y="Profit", palette="Greens_r", ax=ax_chan)
        ax_chan.set_title("Operating Profit by Sales Channel")
        ax_chan.set_ylabel("Profit ($)")
        st.pyplot(fig_chan)

with tab2:
    st.subheader("RFM (Recency, Frequency, Monetary) Segmentation")
    st.write("Auditing customer lifetime behavior to isolate high-value loyalists from churn-risk cohorts.")
    
    ref_date = filtered_df["Order_Date"].max() + timedelta(days=1)
    rfm = filtered_df.groupby("Customer_ID").agg({
        "Order_Date": lambda x: (ref_date - x.max()).days,
        "Order_ID": "count",
        "Net_Revenue": "sum"
    }).reset_index()
    rfm.columns = ["Customer_ID", "Recency_Days", "Frequency", "Monetary"]
    
    # Quantile Scoring
    rfm["R_Score"] = pd.qcut(rfm["Recency_Days"], 3, labels=[3, 2, 1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method='first'), 3, labels=[1, 2, 3])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 3, labels=[1, 2, 3])
    
    rfm["RFM_Segment"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    
    def rfm_label(df_row):
        if df_row["RFM_Segment"] in ["333", "332", "323"]:
            return "Champions / Core Loyalists"
        elif df_row["RFM_Segment"] in ["233", "223", "232"]:
            return "High Potential Growth"
        elif df_row["RFM_Segment"] in ["111", "112", "121"]:
            return "At-Risk / Inactive"
        else:
            return "Nurture Candidates"
            
    rfm["Customer_Category"] = rfm.apply(rfm_label, axis=1)
    
    col_rfm1, col_rfm2 = st.columns([1, 2])
    with col_rfm1:
        st.dataframe(rfm["Customer_Category"].value_counts().reset_index(), use_container_width=True)
    with col_rfm2:
        fig_rfm, ax_rfm = plt.subplots(figsize=(7, 3.5))
        sns.scatterplot(data=rfm, x="Recency_Days", y="Monetary", hue="Customer_Category", palette="Dark2", ax=ax_rfm)
        ax_rfm.set_title("Customer Spend Parity vs Days Since Last Purchase")
        st.pyplot(fig_rfm)

with tab3:
    st.subheader("Time-Series Demand Trajectory & Stockout Analysis")
    ts_data = filtered_df.set_index("Order_Date").resample("W")["Units_Sold"].sum().reset_index()
    ts_data["7W_Moving_Avg"] = ts_data["Units_Sold"].rolling(window=7, min_periods=1).mean()
    
    fig_ts, ax_ts = plt.subplots(figsize=(10, 3.5))
    sns.lineplot(data=ts_data, x="Order_Date", y="Units_Sold", label="Weekly Unit Demand", alpha=0.5, ax=ax_ts)
    sns.lineplot(data=ts_data, x="Order_Date", y="7W_Moving_Avg", label="7-Week Trend Line", color="red", linewidth=2, ax=ax_ts)
    ax_ts.set_title("Weekly Demand Velocity & Trend Smoothing")
    ax_ts.set_ylabel("Units Sold")
    st.pyplot(fig_ts)

with tab4:
    st.subheader("Executive Price Elasticity & Margin Simulator")
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        price_adj = st.slider("Global Price Adjustment (%)", -20, 20, 0)
        volume_elasticity = st.slider("Assumed Price Elasticity of Demand", -2.5, -0.5, -1.2)
    
    with col_sim2:
        cost_inflation = st.slider("COGS Cost Inflation (%)", 0, 15, 2)
        
    # Simulation Output Calculation
    vol_change_pct = price_adj * volume_elasticity
    sim_revenue = total_net_rev * (1 + (price_adj / 100)) * (1 + (vol_change_pct / 100))
    sim_cogs = (total_net_rev - total_profit) * (1 + (cost_inflation / 100)) * (1 + (vol_change_pct / 100))
    sim_profit = sim_revenue - sim_cogs
    sim_margin = (sim_profit / sim_revenue * 100) if sim_revenue > 0 else 0
    
    st.markdown("### Simulated Enterprise Projection")
    res1, res2, res3 = st.columns(3)
    res1.metric("Projected Revenue", f"${sim_revenue:,.2f}", delta=f"{((sim_revenue - total_net_rev)/total_net_rev)*100:.1f}%")
    res2.metric("Projected Operating Profit", f"${sim_profit:,.2f}", delta=f"{((sim_profit - total_profit)/total_profit)*100:.1f}%")
    res3.metric("Projected Profit Margin", f"{sim_margin:.1f}%")
