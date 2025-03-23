import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load process log data
df = pd.read_csv("process_logs.csv")

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Selecting relevant features for anomaly detection
features = ["CPU %", "Memory (MB)"]

# Fill missing values with 0
df[features] = df[features].fillna(0)

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)  # 5% anomalies
df['Anomaly'] = model.fit_predict(df[features])

# Mark anomalies (-1 means anomaly, 1 means normal)
df['Anomaly'] = df['Anomaly'].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# Visualize Anomalies
plt.figure(figsize=(10, 5))

# Normal data points
plt.scatter(df['Timestamp'], df["CPU %"], label="Normal", color='blue')

# Anomalous data points
anomalies = df[df['Anomaly'] == "Anomaly"]
plt.scatter(anomalies['Timestamp'], anomalies["CPU %"], label="Anomalies", color='red')

plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Anomaly Detection")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Save detected anomalies
anomalies.to_csv("anomalies_detected.csv", index=False)
print("Anomaly detection completed! Check anomalies_detected.csv for details.")
