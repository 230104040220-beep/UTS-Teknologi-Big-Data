import streamlit as st
import pandas as pd
import plotly.express as px

from pyspark.sql import SparkSession

from sklearn.linear_model import LinearRegression
import numpy as np

# ==========================================
# SPARK SESSION
# ==========================================
spark = SparkSession.builder \
    .appName("DashboardEnergy") \
    .getOrCreate()

# ==========================================
# ABSOLUTE PATH
# ==========================================
base_path = "/home/mudah/bigdata-project/uts-tbg-NIMANDA/output"

# ==========================================
# LOAD PARQUET
# ==========================================
energy_total = spark.read.parquet(
    f"{base_path}/energy_total"
).toPandas()

energy_time = spark.read.parquet(
    f"{base_path}/energy_time"
).toPandas()

ml_data = spark.read.parquet(
    f"{base_path}/ml_energy"
).toPandas()

# ==========================================
# TITLE
# ==========================================
st.title("Smart Energy Consumption Analytics")

# ==========================================
# SIDEBAR
# ==========================================
sector_list = energy_time["sector"].unique()

selected_sector = st.sidebar.selectbox(
    "Pilih Sektor",
    sector_list
)

filtered_data = energy_time[
    energy_time["sector"] == selected_sector
]

# ==========================================
# KPI
# ==========================================
total_usage = filtered_data["avg_power_usage"].sum()

st.metric(
    label="Total Konsumsi Energi",
    value=f"{total_usage:.2f} kWh"
)

# ==========================================
# GRAFIK
# ==========================================
fig = px.line(
    filtered_data,
    x="minute_group",
    y="avg_power_usage",
    title=f"Tren Konsumsi {selected_sector}",
    markers=True
)

st.plotly_chart(fig)

# ==========================================
# MACHINE LEARNING
# ==========================================
X = ml_data[["hour"]]
y = ml_data["power_usage"]

model = LinearRegression()
model.fit(X, y)

future_hour = np.array([[20]])

prediction = model.predict(future_hour)

st.subheader("Prediksi Konsumsi Energi")

st.write(
    f"Prediksi konsumsi jam 20:00 = {prediction[0]:.2f} kWh"
)

spark.stop()
