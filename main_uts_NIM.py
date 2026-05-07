from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import random
from datetime import datetime, timedelta

# ==========================================
# MEMBUAT SPARK SESSION
# ==========================================
spark = SparkSession.builder \
    .appName("SmartEnergyAnalytics") \
    .getOrCreate()

# ==========================================
# GENERATE DUMMY DATA
# ==========================================
sectors = ["Industrial_A", "Industrial_B", "Residential_C"]

start_time = datetime.now()

data = []

for i in range(150):
    timestamp = start_time + timedelta(minutes=i)

    for sector in sectors:
        power_usage = random.randint(100, 1000)

        data.append((
            timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            sector,
            power_usage
        ))

# ==========================================
# CREATE DATAFRAME
# ==========================================
columns = ["timestamp", "sector", "power_usage"]

df = spark.createDataFrame(data, columns)

# ==========================================
# CONVERT TIMESTAMP
# ==========================================
df = df.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"))
)

# ==========================================
# TOTAL KONSUMSI ENERGI PER SEKTOR
# ==========================================
energy_total = df.groupBy("sector").agg(
    sum("power_usage").alias("total_power_usage")
)

# ==========================================
# AGREGASI KONSUMSI TIAP 10 MENIT
# ==========================================
energy_time = df.withColumn(
    "minute_group",
    floor(minute(col("timestamp")) / 10) * 10
)

energy_time = energy_time.groupBy(
    hour(col("timestamp")).alias("hour"),
    col("minute_group"),
    "sector"
).agg(
    avg("power_usage").alias("avg_power_usage")
).orderBy("hour", "minute_group")

# ==========================================
# DATASET MACHINE LEARNING
# ==========================================
ml_data = df.withColumn(
    "hour",
    hour(col("timestamp"))
).select(
    "hour",
    "power_usage"
)

# ==========================================
# ABSOLUTE PATH
# ==========================================
base_path = "/home/mudah/bigdata-project/uts-tbg-NIMANDA/output"

# ==========================================
# SIMPAN KE FORMAT PARQUET
# ==========================================
energy_total.write.mode("overwrite").parquet(
    f"{base_path}/energy_total"
)

energy_time.write.mode("overwrite").parquet(
    f"{base_path}/energy_time"
)

ml_data.write.mode("overwrite").parquet(
    f"{base_path}/ml_energy"
)

# ==========================================
# TAMPILKAN HASIL
# ==========================================
print("=== TOTAL ENERGY ===")
energy_total.show()

print("=== ENERGY TIME ===")
energy_time.show()

print("=== ML DATA ===")
ml_data.show()

print("PARQUET BERHASIL DISIMPAN!")

# ==========================================
# STOP SPARK
# ==========================================
spark.stop()
