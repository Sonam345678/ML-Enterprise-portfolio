import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Retail Sales & Profit Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)

# Optimized Cached Dataset Generator
@st.cache_data(ttl=3600, show_spinner="Loading Executive Dashboard...")
def generate_sir_dataset(n_rows=4000):
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
        {"code": "OUT-105", "name": "Connaught Place Outlet", "state": "Delhi-NCR"},
        {"code": "OUT-106", "name": "Chennai T-Nagar", "state": "Tamil Nadu"}
    ]
    
    products = [
        {"code": "PRD-501", "name": "Smart LED TV 43''", "base_price": 24000, "landing": 16000},
        {"code": "PRD-502", "name": "Air Conditioner 1.5T", "base_price": 35000, "landing": 24000},
        {"code": "PRD-503", "name": "Double Door Refrigerator", "base_price": 28000, "landing": 19000},
        {"code": "PRD-504", "name": "Microwave Oven 25L", "base_price": 9500, "landing": 6200},
        {"code": "PRD-505", "name": "Front Load Washer", "base_price": 31000, "landing": 21000}
    ]
    
    records = []
    for d in order_dates:
        outlet = outlets[np.random.randint(0, len(outlets))]
        product = products[np.random.randint(0, len(products))]
        
        qty = np.random.randint(1, 10)
        selling_price = product["base_price"] * np.random.uniform(0.92, 1.05)
        landing_price = product["landing"]
        freight_pct = np.random.choice([0.02, 0.03, 0.04, 0.05])
        
        sale_value = selling_price * qty
        cost_value = landing_price * qty
        freight_value = sale_value * freight_pct
        total_cost = cost_value + freight_value
        gross_profit = sale_value - total_cost
        
        records.append({
            "State": outlet["state"],
            "Outlet code": outlet["code"],
            "Outlet name": outlet["name"],
            "Date": pd.to_datetime(d),
            "Product code": product["code"],
            "Product name": product["name"],
            "Selling Price": np.round(selling_price, 2),
            "Landing Price": np.round(landing_price, 2),
            "Freight percent": freight_pct * 100,
            "Sale Qty": qty,
            "Sale Value": np.round(sale_value, 2),
            "Cost value": np.round(cost_value, 2),
            "Freight Value": np.round(freight_value, 2),
            "Total cost": np.round(total_cost, 2),
            "Gross profit": np.round(gross_profit, 2)
        })
        
    df = pd.DataFrame(records)
    
    # Financial Year Assignment Logic
    def assign_fy(dt):
        year = dt.year
        month = dt.month
        return f"{year}-{str(year+1)[-2:]}" if month >= 4 else f"{year-1}-{str(year)[-2:]}"
            
    df["Financial_Year"] = df["Date"].apply(assign_fy)
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Month_Num"] = df["Date"].dt.month
    
    def assign_fin_month_order(m):
        return m - 3 if m >= 4 else m + 9
        
    df["Fin_Month_Order"] = df["Month_Num"].apply(assign_fin_month_order)
    return df.sort_values(by="Date").reset_index(drop=True)

df = generate_sir_dataset()

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align: center; color: #1b4332;'>🏬 Retail Outlet Sales & Profit Analytics Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d;'>Executive Financial Reporting System</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🎯 Dynamic Control Panel")

fy_options = sorted(df["Financial_Year"].unique(), reverse=True)
selected_fy = st.sidebar.selectbox("Select Financial Year", options=fy_options, index=0)

fy_filtered_df = df[df["Financial_Year"] == selected_fy]

fin_months_sorted = (
    fy_filtered_df.groupby(["Fin_Month_Order", "Month_Name"])
    .size()
    .reset_index()
    .sort_values("Fin_Month_Order")["Month_Name"]
    .tolist()
)

selected_month = st.sidebar.selectbox("Select Month for Outlet Breakdown", options=["All Months"] + fin_months_sorted, index=0)

states_list = ["All States"] + list(df["State"].unique())
selected_state = st.sidebar.selectbox("Filter by State", options=states_list, index=0)

active_df = fy_filtered_df.copy()
if selected_state != "All States":
    active_df = active_df[active_df["State"] == selected_state]

# ---------------- TOP SUMMARY KPIS ----------------
kpi_df = active_df.copy()
if selected_month != "All Months":
    kpi_df = kpi_df[kpi_df["Month_Name"] == selected_month]

c1, c2, c3, c4, c5 = st.columns(5)
tot_sales = kpi_df["Sale Value"].sum()
tot_cost = kpi_df["Total cost"].sum()
tot_profit = kpi_df["Gross profit"].sum()
profit_margin = (tot_profit / tot_sales * 100) if tot_sales > 0 else 0
tot_qty = kpi_df["Sale Qty"].sum()

c1.metric("Total Sale Value", f"₹{tot_sales:,.0f}")
c2.metric("Total Cost", f"₹{tot_cost:,.0f}")
c3.metric("Gross Profit", f"₹{tot_profit:,.0f}")
c4.metric("Profit Margin %", f"{profit_margin:.2f}%")
c5.metric("Total Sale Qty", f"{tot_qty:,}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- ROW 1: MONTH-WISE PROFIT WITH LINE CHART ----------------
st.subheader(f"📈 1. Month-Wise Gross Profit (Financial Year: {selected_fy})")
st.caption("Chronological April to March sequence displaying 12 financial monthly data points.")

m_trend = (
    active_df.groupby(["Fin_Month_Order", "Month_Name"])
    .agg({"Gross profit": "sum", "Sale Value": "sum"})
    .reset_index()
    .sort_values("Fin_Month_Order")
)

fig_m = go.Figure()
fig_m.add_trace(go.Bar(
    x=m_trend["Month_Name"],
    y=m_trend["Gross profit"],
    name="Gross Profit (₹)",
    marker_color="#2d6a4f",
    text=m_trend["Gross profit"].apply(lambda x: f"₹{x/1000:,.0f}K"),
    textposition="outside"
))

fig_m.add_trace(go.Scatter(
    x=m_trend["Month_Name"],
    y=m_trend["Sale Value"],
    name="Sale Value (₹)",
    yaxis="y2",
    line=dict(color="#d90429", width=3, dash="dot")
))

fig_m.update_layout(
    height=380,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(title="Gross Profit (₹)"),
    yaxis2=dict(title="Total Sale Value (₹)", overlaying="y", side="right")
)
st.plotly_chart(fig_m, use_container_width=True)

# ---------------- ROW 2: OUTLET-WISE PROFIT + TOP OUTLET TABLE ----------------
st.markdown("---")
st.subheader(f"🏬 2. Outlet-Wise Profit Breakdown ({selected_month})")

outlet_df = active_df.copy()
if selected_month != "All Months":
    outlet_df = outlet_df[outlet_df["Month_Name"] == selected_month]

r2_c1, r2_c2 = st.columns([3, 2])

with r2_c1:
    out_summary = (
        outlet_df.groupby(["Outlet code", "Outlet name"])
        .agg({"Gross profit": "sum", "Sale Value": "sum"})
        .reset_index()
        .sort_values("Gross profit", ascending=True)
    )
    
    fig_out = px.bar(
        out_summary,
        x="Gross profit",
        y="Outlet name",
        orientation="h",
        color="Gross profit",
        color_continuous_scale="Greens",
        text_auto=".2s",
        title=f"Gross Profit by Outlet ({selected_month})"
    )
    fig_out.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig_out, use_container_width=True)

with r2_c2:
    st.markdown("#### Top Outlet Performance Summary")
    out_table = out_summary.sort_values("Gross profit", ascending=False).copy()
    out_table["Margin_%"] = (out_table["Gross profit"] / out_table["Sale Value"] * 100).round(2)
    
    disp_table = pd.DataFrame({
        "Outlet": out_table["Outlet name"],
        "Sale Value": out_table["Sale Value"].apply(lambda x: f"₹{x:,.0f}"),
        "Gross Profit": out_table["Gross profit"].apply(lambda x: f"₹{x:,.0f}"),
        "Margin %": out_table["Margin_%"].apply(lambda x: f"{x:.1f}%")
    })
    st.dataframe(disp_table, use_container_width=True, hide_index=True)

# ---------------- ROW 3: DETAILED TRANSACTION AUDIT DATA TABLE ----------------
st.markdown("---")
with st.expander("🔍 View Raw Financial Data Audit Log"):
    st.dataframe(
        outlet_df[[
            "State", "Outlet code", "Outlet name", "Date", "Product code", "Product name",
            "Selling Price", "Landing Price", "Freight percent", "Sale Qty",
            "Sale Value", "Cost value", "Freight Value", "Total cost", "Gross profit"
        ]],
        use_container_width=True
    )
