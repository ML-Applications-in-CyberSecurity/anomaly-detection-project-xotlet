import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Path to the anomaly log
anomaly_log_path = os.path.join(os.path.dirname(__file__), 'anomalies_log.csv')
# Path to the normal log (user should create this if they want to compare)
normal_log_path = os.path.join(os.path.dirname(__file__), 'normal_log.csv')

# Read anomalies
anomalies = pd.read_csv(anomaly_log_path)

# Try to read normal data if available
if os.path.exists(normal_log_path):
    normal = pd.read_csv(normal_log_path)
    has_normal = True
else:
    normal = None
    has_normal = False

plt.figure(figsize=(8, 6))

# Plot normal points if available
if has_normal:
    sns.scatterplot(
        data=normal,
        x='packet_size',
        y='duration_ms',
        color='gray',
        label='Normal',
        s=40,
        alpha=0.5
    )

# Plot anomalies
sns.scatterplot(
    data=anomalies,
    x='packet_size',
    y='duration_ms',
    hue='confidence_score',
    palette='coolwarm',
    style=None,
    s=80,
    legend='brief',
    label='Anomaly'
)

plt.title('Packet Size vs Duration: Normal vs Anomaly')
plt.xlabel('Packet Size')
plt.ylabel('Duration (ms)')
plt.legend(title='Legend')
plt.tight_layout()
plt.show()

# To create 'normal_log.csv', log normal points from your client or server with the same columns as anomalies_log.csv (except llm_explanation). 