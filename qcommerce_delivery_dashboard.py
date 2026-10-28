import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Instamart Real-Time Q-Commerce Delivery Engine", layout="wide"
)


# 1. Dynamic Dataset Generator with Time-Series Dynamics
@st.cache_data
def generate_qcommerce_data(n_orders=1200):
    np.random.seed(42)
    order_ids = [f"ORD-{10000 + i}" for i in range(n_orders)]

    zones = ["Andheri West", "Bhayandar East", "Bandra West", "Powai", "Thane"]
    categories = [
        "Fresh Vegetables & Fruits",
        "Dairy & Breakfast",
        "Munchies & Snacks",
        "Cold Drinks & Juices",
        "Instant Food",
    ]

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_orders, freq="5min")
    assigned_zones = np.random.choice(zones, size=n_orders)
    assigned_categories = np.random.choice(
        categories, size=n_orders, p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )

    order_values = np.round(np.random.uniform(150, 2500, size=n_orders), 2)
    distances_km = np.round(np.random.uniform(0.5, 6.5, size=n_orders), 1)

    # Calculate simulated delivery time based on distance & congestion
    base_time = 8 + (distances_km * 3.5)
    rider_delay = np.random.exponential(scale=3, size=n_orders)
    actual_delivery_time = np.round(base_time + rider_delay, 1)

    is_delayed = (actual_delivery_time > 15).astype(int)

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Timestamp": dates,
        "Hour": dates.hour,
        "Zone": assigned_zones,
        "Category": assigned_categories,
        "Order_Value_INR": order_values,
        "Distance_KM": distances_km,
        "Delivery_Time_Mins": actual_delivery_time,
        "Is_Delayed": is_delayed,
    })
    return df


st.title("⚡ Instamart Real-Time Q-Commerce Delivery & Operations Engine")
st.markdown(
    "**Live Operations Analytics:** Monitoring 10-Minute Delivery SLA, Peak Surge Volumes, and Dynamic Rider Allocations."
)

df = generate_qcommerce_data()

# Live Refresh Sidebar Option
st.sidebar.header("⏱️ Live Time Controls")
auto_refresh = st.sidebar.checkbox("Enable Real-Time Auto Refresh (5s)", False)
selected_zone = st.sidebar.multiselect(
    "Select Target Zones", options=df["Zone"].unique(), default=df["Zone"].unique()
)
selected_hour = st.sidebar.slider("Filter by Hour of Day", 0, 23, (8, 22))

# Filter Data based on UI input
filtered_df = df[
    (df["Zone"].isin(selected_zone))
    & (df["Hour"] >= selected_hour[0])
    & (df["Hour"] <= selected_hour[1])
]

# Top Key Performance Indicators
col1, col2, col3, col4 = st.columns(4)
total_orders = len(filtered_df)
avg_delivery = filtered_df["Delivery_Time_Mins"].mean()
sla_breach_pct = (filtered_df["Is_Delayed"].mean()) * 100
total_gmv = filtered_df["Order_Value_INR"].sum()

col1.metric("Total Live Orders", f"{total_orders:,}")
col2.metric("Avg Delivery Speed", f"{avg_delivery:.1f} Mins")
col3.metric(
    "15-Min SLA Breach Rate",
    f"{sla_breach_pct:.1f}%",
    delta="-Good" if sla_breach_pct < 15 else "+High Delay",
)
col4.metric("Total GMV Generated", f"₹{total_gmv:,.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    [
        "📈 Time-Series Order Dynamics",
        "🗺️ Zone Performance & SLA Breakdown",
        "🎛️ Dynamic Delivery Time Simulator",
    ]
)

with tab1:
    st.subheader("Hourly Order Volume & Delivery Speed Trajectory")
    fig, ax1 = plt.subplots(figsize=(10, 3.5))

    hourly_stats = (
        filtered_df.groupby("Hour")
        .agg(
            Order_Count=("Order_ID", "count"),
            Avg_Speed=("Delivery_Time_Mins", "mean"),
        )
        .reset_index()
    )

    sns.barplot(
        data=hourly_stats,
        x="Hour",
        y="Order_Count",
        ax=ax1,
        color="#3498db",
        alpha=0.6,
    )
    ax1.set_ylabel("Order Volume", color="#3498db")

    ax2 = ax1.twinx()
    sns.lineplot(
        data=hourly_stats,
        x="Hour",
        y="Avg_Speed",
        ax=ax2,
        color="#e74c3c",
        marker="o",
        linewidth=2.5,
    )
    ax2.set_ylabel("Avg Delivery Time (Mins)", color="#e74c3c")
    ax1.set_title("Peak Demand vs Average Delivery Latency")

    st.pyplot(fig)

with tab2:
    st.subheader("High-Density Delivery Zone Metrics")
    col_z1, col_z2 = st.columns(2)

    with col_z1:
        fig_zone, ax_zone = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(
            data=filtered_df,
            x="Zone",
            y="Delivery_Time_Mins",
            palette="Set2",
            ax=ax_zone,
        )
        ax_zone.axhline(
            15, color="red", linestyle="--", label="SLA Limit (15 Mins)"
        )
        ax_zone.set_title("Delivery Time Distribution Across Zones")
        ax_zone.tick_params(axis="x", rotation=30)
        ax_zone.legend()
        st.pyplot(fig_zone)

    with col_z2:
        st.write("#### Delayed Orders Audit Table")
        delayed_table = filtered_df[filtered_df["Is_Delayed"] == 1][
            [
                "Order_ID",
                "Zone",
                "Category",
                "Distance_KM",
                "Delivery_Time_Mins",
                "Order_Value_INR",
            ]
        ].head(8)
        st.dataframe(delayed_table, use_container_width=True)

with tab3:
    st.subheader("Interactive Dispatch & Weather Delay Simulator")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        sim_dist = st.slider("Target Distance (KM)", 0.5, 10.0, 2.5)
        sim_weather = st.selectbox(
            "Weather Condition",
            ["Clear", "Light Rain", "Heavy Rain (Monsoon Surge)"],
        )
        sim_traffic = st.selectbox(
            "Traffic Level", ["Low", "Moderate", "Peak Heavy Traffic"]
        )

    with col_s2:
        sim_riders = st.slider("Available Delivery Fleet Ratio (%)", 20, 100, 85)

    # Simulation Logic
    weather_factor = (
        1.0
        if sim_weather == "Clear"
        else (1.3 if sim_weather == "Light Rain" else 1.7)
    )
    traffic_factor = (
        1.0
        if sim_traffic == "Low"
        else (1.25 if sim_traffic == "Moderate" else 1.6)
    )
    fleet_factor = 1.0 + ((100 - sim_riders) / 100) * 0.8

    predicted_time = (
        (8 + (sim_dist * 3.0)) * weather_factor * traffic_factor * fleet_factor
    )

    st.markdown("### Estimated Delivery Window Result")
    res_c1, res_c2 = st.columns(2)
    res_c1.metric("Predicted Delivery Time", f"{predicted_time:.1f} Mins")

    if predicted_time <= 15:
        res_c2.metric("SLA Status", "On-Time Guaranteed ✅")
        st.success("⚡ Order safely within the 15-minute quick-commerce SLA!")
    else:
        res_c2.metric("SLA Status", "Delayed Breach ⚠️")
        st.error(
            "🚨 Delivery latency expected to breach 15 mins! Triggering Surge Fee & Allocation Buffer."
        )

# Handle Auto-refresh simulation loop
if auto_refresh:
    time.sleep(5)
    st.rerun()
