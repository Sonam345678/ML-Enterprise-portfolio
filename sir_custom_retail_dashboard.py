import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Fast Page Configuration
st.set_page_config(
    page_title="Retail Sales & Profit Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Custom CSS for Smooth UX
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stMetric {
        background: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# 1. Ultra-Fast In-Memory Data Pipeline with Caching
@st.cache_data(ttl=3600, show_spinner="Loading Analytics Engine...")
def load_enterprise_data(n_rows=3000):
    np.random.seed(42)
    start_date = pd.Timestamp("2024-04-01")
    end_date = pd.Timestamp("2026-03-31")
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    order_dates = np.random.choice(dates, size=n_rows)
    
    outlets = [
        {"code": "OUT-101", "name": "Bhayandar West Store", "state": "Maharashtra"},
        {"code": "OUT-102", "name": "Andheri Central Hub", "state": "Maharashtra"},
        {"code": "OUT-103", "name": "Ahmedabad Ring Rd", "state": "Gujarat"},
        {"code": "OUT-104", "name": "Bengaluru Indiranagar", "state": "Karnataka"},
        {"code": "OUT-105", "name": "Connaught Place Outlet", "state": "Delhi-NCR"}
    ]
    
    products = [
        {"code": "PRD-501", "name": "Smart LED TV 43''", "base_price": 24000, "landing": 16000},
        {"code": "PRD-502", "name": "Air Conditioner 1.5T", "base_price": 35000, "landing": 24000},
        {"code": "PRD-503", "name": "Double Door Refrigerator", "base_price": 28000, "landing": 19000},
        {"code": "PRD-504", "name": "Microwave Oven 25L", "base_price": 9500, "landing": 6200}
    ]
    
    records = []
    for d in order_dates:
        outlet = outlets[np.random.randint(0, len(outlets))]
        product = products[np.random.randint(0, len(products))]
        qty = np.random.randint(1, 8)
        selling_price = product["base_price"] * np.random.uniform(0.95, 1.05)
        landing_price = product["landing"]
        freight_pct = np.random.choice([0.02, 0.03, 0.04])
        
        sale_val = selling_price * qty
        cost_val = landing_price * qty
        freight_val = sale_val * freight_pct
        tot_cost = cost_val + freight_val
        gross_prof = sale_val - tot_cost
        
        records.append({
            "State": outlet["state"],
            "Outlet code": outlet["code"],
            "Outlet name": outlet["name"],
            "Date": d,
            "Product code": product["code"],
            "Product name": product["name"],
            "Selling Price": round(selling_price, 2),
            "Landing Price": round(landing_price, 2),
            "Freight percent": freight_pct * 100,
            "Sale Qty": qty,
            "Sale Value": round(sale_val, 2),
            "Cost value": round(cost_val, 2),
            "Freight Value": round(freight_val, 2),
            "Total cost": round(tot_cost, 2),
            "Gross profit": round(gross_prof, 2)
        })
        
    df = pd.DataFrame(records)
    
    # Financial Year April-March Conversion Logic
    def get_fy(dt):
        return f"{dt.year}-{str(dt.year+1)[-2:]}" if dt.month >= 4 else f"{dt.year-1}-{str(dt.year)[-2:]}"
        
    df["Financial_Year"] = df["Date"].apply(get_fy)
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Fin_Month_Order"] = df["Date"].dt.month.apply(lambda m: m - 3 if m >= 4 else m + 9)
    
    return df

df = load_enterprise_data()

# Header Layout
st.title("🏬 Retail Outlet Sales & Profit Analytics Engine")
st.markdown("---")

# Sidebar Dynamic Controls
st.sidebar.header("🎯 Executive Control Panel")
selected_fy = st.sidebar.selectbox("Select Financial Year", sorted(df["Financial_Year"].unique(), reverse=True))

fy_df = df[df["Financial_Year"] == selected_fy]
sorted_months = fy_df.groupby(["Fin_Month_Order", "Month_Name"]).size().reset_index().sort_values("Fin_Month_Order")["Month_Name"].tolist()

selected_month = st.sidebar.selectbox("Filter Outlet Breakdown by Month", ["All Months"] + sorted_months)

# Main KPI Row
kpi_df = fy_df if selected_month == "All Months" else fy_df[fy_df["Month_Name"] == selected_month]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"₹{kpi_df['Sale Value'].sum():,.0f}")
c2.metric("Total Cost", f"₹{kpi_df['Total cost'].sum():,.0f}")
c3.metric("Gross Profit", f"₹{kpi_df['Gross profit'].sum():,.0f}")
margin = (kpi_df['Gross profit'].sum() / kpi_df['Sale Value'].sum() * 100) if kpi_df['Sale Value'].sum() > 0 else 0
c4.metric("Profit Margin %", f"{margin:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# Visual 1: Financial Year Monthly Trend
st.subheader(f"📈 Financial Year {selected_fy} Month-Wise Profit (Apr - Mar)")
m_trend = fy_df.groupby(["Fin_Month_Order", "Month_Name"]).agg({"Gross profit": "sum"}).reset_index().sort_values("Fin_Month_Order")

fig_m = px.bar(m_trend, x="Month_Name", y="Gross profit", color_discrete_sequence=["#2d6a4f"], text_auto=".2s")
fig_m.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=20))
st.plotly_chart(fig_m, use_container_width=True)

# Visual 2: Outlet-Wise Profit
st.subheader(f"🏬 Outlet-Wise Profit Distribution ({selected_month})")
out_df = fy_df.groupby("Outlet name").agg({"Gross profit": "sum", "Sale Value": "sum"}).reset_index().sort_values("Gross profit", ascending=True)

fig_out = px.bar(out_df, x="Gross profit", y="Outlet name", orientation="h", color="Gross profit", color_continuous_scale="Viridis")
fig_out.update_layout(height=320)
st.plotly_chart(fig_out, use_container_width=True)
